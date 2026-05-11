import numpy as np
import torch
import wandb

from src.algos.agent_speq import SPEQAgent
from src.algos.core import test_agent


class SOPEAgent(SPEQAgent):
    """
    SOPE: SPEQ with policy-loss early stopping on a held-out validation slice.

    At each stabilization phase a fraction `val_pct` of the online buffer is
    sampled (with a symmetric count from the offline buffer), removed from the
    training buffers, and used as a fixed validation set. Q-network updates
    proceed until policy-loss on this validation set stops improving for
    `val_patience` updates. The validation transitions are restored to the
    buffers when the phase ends.
    """

    def __init__(
        self,
        env_name: str,
        obs_dim: int,
        act_dim: int,
        act_limit: float,
        device: torch.device,
        hidden_sizes: tuple = (256, 256),
        replay_size: int = int(1e6),
        batch_size: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        polyak: float = 0.995,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: str = 'mbpo',
        start_steps: int = 5000,
        utd_ratio: int = 1,
        num_Q: int = 2,
        policy_update_delay: int = 20,
        target_drop_rate: float = 0.0,
        layer_norm: bool = True,
        o2o: bool = True,
        offline_epochs: int = 75000,
        trigger_interval: int = 10000,
        val_check_interval: int = 1000,
        val_patience: int = 10000,
        val_pct: float = 0.1,
    ):
        super().__init__(
            env_name, obs_dim, act_dim, act_limit, device,
            hidden_sizes, replay_size, batch_size, lr, gamma, polyak,
            alpha, auto_alpha, target_entropy, start_steps, utd_ratio,
            num_Q, policy_update_delay, target_drop_rate, layer_norm,
            o2o=True, offline_epochs=offline_epochs, trigger_interval=trigger_interval
        )

        self.val_check_interval = val_check_interval
        self.val_patience = val_patience
        self.val_pct = val_pct

        self.val_batches_online = None
        self.val_batches_offline = None

        self._removed_online_data = None
        self._removed_offline_data = None

        self._stopping_epochs_history = []

    def _sample_fixed_validation_batches(self):
        online_batches = []
        offline_batches = []

        n_val_transitions = max(self.batch_size, int(self.replay_buffer.size * self.val_pct))

        if self.replay_buffer.size >= self.batch_size:
            n_online_samples = min(n_val_transitions, self.replay_buffer.size - self.batch_size)
            if n_online_samples >= self.batch_size:
                all_online_indices = np.random.choice(
                    self.replay_buffer.size,
                    size=n_online_samples,
                    replace=False
                )

                self._removed_online_data = {
                    'obs1': self.replay_buffer.obs1_buf[all_online_indices].copy(),
                    'obs2': self.replay_buffer.obs2_buf[all_online_indices].copy(),
                    'acts': self.replay_buffer.acts_buf[all_online_indices].copy(),
                    'rews': self.replay_buffer.rews_buf[all_online_indices].copy(),
                    'done': self.replay_buffer.done_buf[all_online_indices].copy(),
                }

                n_complete_batches = n_online_samples // self.batch_size
                for i in range(n_complete_batches):
                    start_idx = i * self.batch_size
                    end_idx = start_idx + self.batch_size
                    batch_indices = all_online_indices[start_idx:end_idx]

                    online_batches.append({
                        'obs': torch.FloatTensor(self.replay_buffer.obs1_buf[batch_indices].copy()).to(self.device),
                        'obs_next': torch.FloatTensor(self.replay_buffer.obs2_buf[batch_indices].copy()).to(self.device),
                        'acts': torch.FloatTensor(self.replay_buffer.acts_buf[batch_indices].copy()).to(self.device),
                        'rews': torch.FloatTensor(self.replay_buffer.rews_buf[batch_indices].copy()).unsqueeze(1).to(self.device),
                        'done': torch.FloatTensor(self.replay_buffer.done_buf[batch_indices].copy()).unsqueeze(1).to(self.device),
                    })

                self.replay_buffer.remove_indices(all_online_indices)

        if self.replay_buffer_offline.size >= self.batch_size:
            n_offline_samples = min(n_val_transitions, self.replay_buffer_offline.size - self.batch_size)
            if n_offline_samples >= self.batch_size:
                all_offline_indices = np.random.choice(
                    self.replay_buffer_offline.size,
                    size=n_offline_samples,
                    replace=False
                )

                self._removed_offline_data = {
                    'obs1': self.replay_buffer_offline.obs1_buf[all_offline_indices].copy(),
                    'obs2': self.replay_buffer_offline.obs2_buf[all_offline_indices].copy(),
                    'acts': self.replay_buffer_offline.acts_buf[all_offline_indices].copy(),
                    'rews': self.replay_buffer_offline.rews_buf[all_offline_indices].copy(),
                    'done': self.replay_buffer_offline.done_buf[all_offline_indices].copy(),
                }

                n_complete_batches = n_offline_samples // self.batch_size
                for i in range(n_complete_batches):
                    start_idx = i * self.batch_size
                    end_idx = start_idx + self.batch_size
                    batch_indices = all_offline_indices[start_idx:end_idx]

                    offline_batches.append({
                        'obs': torch.FloatTensor(self.replay_buffer_offline.obs1_buf[batch_indices].copy()).to(self.device),
                        'obs_next': torch.FloatTensor(self.replay_buffer_offline.obs2_buf[batch_indices].copy()).to(self.device),
                        'acts': torch.FloatTensor(self.replay_buffer_offline.acts_buf[batch_indices].copy()).to(self.device),
                        'rews': torch.FloatTensor(self.replay_buffer_offline.rews_buf[batch_indices].copy()).unsqueeze(1).to(self.device),
                        'done': torch.FloatTensor(self.replay_buffer_offline.done_buf[batch_indices].copy()).unsqueeze(1).to(self.device),
                    })

                self.replay_buffer_offline.remove_indices(all_offline_indices)

        self.val_batches_online = online_batches
        self.val_batches_offline = offline_batches

        total_online = len(online_batches) * self.batch_size
        total_offline = len(offline_batches) * self.batch_size

        print(f"  Validation set ({self.val_pct*100:.1f}%): {len(online_batches)} online + {len(offline_batches)} offline batches")
        print(f"  Removed {total_online} online + {total_offline} offline transitions from training")
        print(f"  Remaining: online={self.replay_buffer.size}, offline={self.replay_buffer_offline.size}")

    def _restore_validation_data(self):
        if self._removed_online_data is not None:
            n_restored = len(self._removed_online_data['obs1'])
            for i in range(n_restored):
                self.replay_buffer.store(
                    self._removed_online_data['obs1'][i],
                    self._removed_online_data['acts'][i],
                    self._removed_online_data['rews'][i],
                    self._removed_online_data['obs2'][i],
                    self._removed_online_data['done'][i]
                )
            print(f"  Restored {n_restored} online transitions")

        if self._removed_offline_data is not None:
            n_restored = len(self._removed_offline_data['obs1'])
            for i in range(n_restored):
                self.replay_buffer_offline.store(
                    self._removed_offline_data['obs1'][i],
                    self._removed_offline_data['acts'][i],
                    self._removed_offline_data['rews'][i],
                    self._removed_offline_data['obs2'][i],
                    self._removed_offline_data['done'][i]
                )
            print(f"  Restored {n_restored} offline transitions")

        print(f"  Final buffer sizes: online={self.replay_buffer.size}, offline={self.replay_buffer_offline.size}")

        self.val_batches_online = None
        self.val_batches_offline = None
        self._removed_online_data = None
        self._removed_offline_data = None

    def _evaluate_policy_loss_on_fixed_batches(self) -> float:
        if not self.val_batches_online and not self.val_batches_offline:
            return 0.0

        total_loss = 0.0
        n_batches = 0

        all_batches = self.val_batches_online + self.val_batches_offline

        for batch in all_batches:
            with torch.no_grad():
                action, _, _, log_prob, _, _ = self.policy_net.forward(batch['obs'])
                q_values = [q_net(torch.cat([batch['obs'], action], 1)) for q_net in self.q_net_list]
                q_mean = torch.mean(torch.cat(q_values, dim=1), dim=1, keepdim=True)
                policy_loss = (self.alpha * log_prob - q_mean).mean()
                total_loss += policy_loss.item()
                n_batches += 1

        return total_loss / max(1, n_batches)

    def finetune_offline(self, epochs: int = None, test_env=None, current_env_step: int = None) -> int:
        epochs = epochs or self.offline_epochs

        self._sample_fixed_validation_batches()

        initial_loss = self._evaluate_policy_loss_on_fixed_batches()
        print(f"  Initial val policy_loss: {initial_loss:.6f}")

        best_loss = initial_loss
        steps_without_improvement = 0
        epochs_performed = 0

        for e in range(epochs):
            epochs_performed = e + 1

            if e > 0 and e % self.val_check_interval == 0:
                val_loss = self._evaluate_policy_loss_on_fixed_batches()

                improved = val_loss < best_loss * 0.999
                status = "improved" if improved else "no improvement"
                print(f"  Epoch {e}: val policy_loss={val_loss:.6f} (best={best_loss:.6f}, {status}, patience={steps_without_improvement}/{self.val_patience})")

                if improved:
                    best_loss = val_loss
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += self.val_check_interval

                if steps_without_improvement >= self.val_patience:
                    print(f"  Early stop @ epoch {e} (no improvement for {self.val_patience} steps)")
                    break

            if self.replay_buffer_offline.size > 0:
                obs, obs_next, acts, rews, done = self.sample_data_mix(self.batch_size)
            else:
                obs, obs_next, acts, rews, done = self.sample_data(self.batch_size)

            y_q = self.get_sac_q_target(obs_next, rews, done)
            q_preds = [q_net(torch.cat([obs, acts], 1)) for q_net in self.q_net_list]
            q_cat = torch.cat(q_preds, dim=1)
            y_q_expanded = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            q_loss = self.expectile_loss(q_cat - y_q_expanded).mean() * self.num_Q

            for q_opt in self.q_optimizer_list:
                q_opt.zero_grad()
            q_loss.backward()
            for q_opt in self.q_optimizer_list:
                q_opt.step()

            self.update_target_networks()

            if (e + 1) % 5000 == 0:
                current_val = self._evaluate_policy_loss_on_fixed_batches()
                print(f"  Offline epoch {e+1}/{epochs}, val policy_loss={current_val:.6f}")

                if test_env is not None and current_env_step is not None:
                    test_rw = test_agent(self, test_env, 1000, None)
                    wandb.log({"OfflineEvalReward": np.mean(test_rw)}, step=current_env_step)

        final_loss = self._evaluate_policy_loss_on_fixed_batches()
        print(f"  Final val policy_loss: {final_loss:.6f} (started at {initial_loss:.6f})")

        self._stopping_epochs_history.append(epochs_performed)

        if current_env_step is not None:
            wandb.log({"OfflineEpochs": epochs_performed}, step=current_env_step)

        self._restore_validation_data()

        return epochs_performed
