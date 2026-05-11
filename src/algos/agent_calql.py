import numpy as np
import torch
import torch.optim as optim
import wandb

from src.algos.agent_base import BaseAgent
from src.algos.core import test_agent


class CalQLReplayBuffer:
    # Replay buffer that also stores the Monte-Carlo return needed by Cal-QL's calibration term.

    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs1_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.mc_returns_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
    
    def store(self, obs, act, rew, next_obs, done, mc_return=0.0):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.mc_returns_buf[self.ptr] = mc_return
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample_batch(self, batch_size: int) -> dict:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs], obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs], rews=self.rews_buf[idxs],
            done=self.done_buf[idxs], mc_returns=self.mc_returns_buf[idxs]
        )


class CalQLAgent(BaseAgent):

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
        policy_update_delay: int = 1,
        target_drop_rate: float = 0.0,
        layer_norm: bool = True,
        o2o: bool = False,
        cql_alpha: float = 5.0,
        cql_n_actions: int = 10,
        cql_temp: float = 1.0,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = 0.8,
        mixing_ratio: float = 0.5,
    ):
        super().__init__(
            env_name, obs_dim, act_dim, act_limit, device,
            hidden_sizes, replay_size, batch_size, lr, gamma, polyak,
            alpha, auto_alpha, target_entropy, start_steps, utd_ratio,
            num_Q, policy_update_delay, target_drop_rate, layer_norm, o2o
        )
        
        self.cql_alpha = cql_alpha
        self.cql_n_actions = cql_n_actions
        self.cql_temp = cql_temp
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.mixing_ratio = mixing_ratio

        # Offline buffer must carry MC returns for Cal-QL calibration.
        self.replay_buffer_offline = CalQLReplayBuffer(obs_dim, act_dim, replay_size)

        if cql_lagrange:
            self.log_cql_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.cql_alpha_optimizer = optim.Adam([self.log_cql_alpha], lr=lr)

    def store_data_offline(self, o, a, r, o2, d, mc_return=0.0):
        self.replay_buffer_offline.store(o, a, r, o2, d, mc_return)

    def sample_data_mix(self, batch_size: int):
        offline_size = int(batch_size * self.mixing_ratio)
        online_size = batch_size - offline_size
        
        if self.replay_buffer.size < online_size:
            online_size = max(1, self.replay_buffer.size)
            offline_size = batch_size - online_size
        
        batch_offline = self.replay_buffer_offline.sample_batch(offline_size)
        batch_online = self.replay_buffer.sample_batch(online_size)
        
        online_mc = np.zeros(online_size, dtype=np.float32)
        
        batch = {
            'obs1': np.concatenate([batch_offline['obs1'], batch_online['obs1']]),
            'obs2': np.concatenate([batch_offline['obs2'], batch_online['obs2']]),
            'acts': np.concatenate([batch_offline['acts'], batch_online['acts']]),
            'rews': np.concatenate([batch_offline['rews'], batch_online['rews']]),
            'done': np.concatenate([batch_offline['done'], batch_online['done']]),
            'mc_returns': np.concatenate([batch_offline['mc_returns'], online_mc]),
        }
        
        obs = torch.FloatTensor(batch['obs1']).to(self.device)
        obs_next = torch.FloatTensor(batch['obs2']).to(self.device)
        acts = torch.FloatTensor(batch['acts']).to(self.device)
        rews = torch.FloatTensor(batch['rews']).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(batch['done']).unsqueeze(1).to(self.device)
        mc_returns = torch.FloatTensor(batch['mc_returns']).unsqueeze(1).to(self.device)
        
        return obs, obs_next, acts, rews, done, mc_returns
    
    def sample_data_offline_only(self, batch_size: int):
        batch = self.replay_buffer_offline.sample_batch(batch_size)
        obs = torch.FloatTensor(batch['obs1']).to(self.device)
        obs_next = torch.FloatTensor(batch['obs2']).to(self.device)
        acts = torch.FloatTensor(batch['acts']).to(self.device)
        rews = torch.FloatTensor(batch['rews']).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(batch['done']).unsqueeze(1).to(self.device)
        mc_returns = torch.FloatTensor(batch['mc_returns']).unsqueeze(1).to(self.device)
        return obs, obs_next, acts, rews, done, mc_returns
    
    def compute_calql_q_loss(self, obs, obs_next, acts, rews, done, mc_returns, use_calibration=True):
        # Cal-QL regularizer: logsumexp(max(Q(s,a), mc_return)) - Q(s, a_data)
        batch_size = obs.shape[0]

        with torch.no_grad():
            next_actions, _, _, next_log_probs, _, _ = self.policy_net.forward(obs_next, return_log_prob=True)
            target_q_list = [q_target(torch.cat([obs_next, next_actions], 1)) for q_target in self.q_target_net_list]
            target_q = torch.min(torch.cat(target_q_list, dim=1), dim=1, keepdim=True)[0]
            td_target = rews + self.gamma * (1 - done) * (target_q - self.alpha * next_log_probs)

        q_data_list = [q_net(torch.cat([obs, acts], 1)) for q_net in self.q_net_list]

        random_actions = torch.FloatTensor(self.cql_n_actions, batch_size, self.act_dim).uniform_(-1, 1).to(self.device)
        random_log_prob = np.log(0.5 ** self.act_dim)

        policy_actions, policy_log_probs = [], []
        next_policy_actions, next_policy_log_probs = [], []
        for _ in range(self.cql_n_actions):
            pi_a, _, _, pi_lp, _, _ = self.policy_net.forward(obs, return_log_prob=True)
            policy_actions.append(pi_a.unsqueeze(0))
            policy_log_probs.append(pi_lp.unsqueeze(0))

            next_pi_a, _, _, next_pi_lp, _, _ = self.policy_net.forward(obs_next, return_log_prob=True)
            next_policy_actions.append(next_pi_a.unsqueeze(0))
            next_policy_log_probs.append(next_pi_lp.unsqueeze(0))

        policy_actions = torch.cat(policy_actions, dim=0)
        policy_log_probs = torch.cat(policy_log_probs, dim=0)
        next_policy_actions = torch.cat(next_policy_actions, dim=0)
        next_policy_log_probs = torch.cat(next_policy_log_probs, dim=0)

        cql_loss_total, td_loss_total = 0, 0
        q_values_log, cql_diff_log = [], []

        for q_i, q_net in enumerate(self.q_net_list):
            q_random = torch.stack([q_net(torch.cat([obs, random_actions[j]], 1)) for j in range(self.cql_n_actions)])
            q_policy = torch.stack([q_net(torch.cat([obs, policy_actions[j]], 1)) for j in range(self.cql_n_actions)])
            q_next_policy = torch.stack([q_net(torch.cat([obs, next_policy_actions[j]], 1)) for j in range(self.cql_n_actions)])

            if use_calibration:
                mc_expanded = mc_returns.unsqueeze(0).expand(self.cql_n_actions, -1, -1)
                q_random = torch.max(q_random, mc_expanded)
                q_policy = torch.max(q_policy, mc_expanded)
                q_next_policy = torch.max(q_next_policy, mc_expanded)

            q_random_is = q_random / self.cql_temp - random_log_prob
            q_policy_is = q_policy / self.cql_temp - policy_log_probs / self.cql_temp
            q_next_policy_is = q_next_policy / self.cql_temp - next_policy_log_probs / self.cql_temp

            cat_q = torch.cat([q_random_is, q_policy_is, q_next_policy_is], dim=0)
            cql_logsumexp = torch.logsumexp(cat_q, dim=0) * self.cql_temp

            cql_diff = cql_logsumexp - q_data_list[q_i]
            cql_loss = cql_diff.mean()
            td_loss = self.mse_criterion(q_data_list[q_i], td_target)

            cql_loss_total += cql_loss
            td_loss_total += td_loss
            q_values_log.append(q_data_list[q_i].mean().item())
            cql_diff_log.append(cql_diff.mean().item())

        if self.cql_lagrange:
            current_cql_alpha = torch.exp(self.log_cql_alpha).clamp(min=0.0, max=1e6)
            alpha_loss = -self.log_cql_alpha * (cql_loss_total.detach() / self.num_Q - self.cql_target_action_gap)
        else:
            current_cql_alpha = self.cql_alpha
            alpha_loss = torch.tensor(0.0)

        total_q_loss = td_loss_total + current_cql_alpha * cql_loss_total

        return (total_q_loss, td_loss_total.item() / self.num_Q, cql_loss_total.item() / self.num_Q,
                np.mean(q_values_log), np.mean(cql_diff_log), alpha_loss, current_cql_alpha)

    def compute_policy_loss(self, obs: torch.Tensor):
        actions, _, _, log_probs, _, _ = self.policy_net.forward(obs, return_log_prob=True)
        q_values = [q_net(torch.cat([obs, actions], 1)) for q_net in self.q_net_list]
        q_min = torch.min(torch.cat(q_values, dim=1), dim=1, keepdim=True)[0]
        return (self.alpha * log_probs - q_min).mean(), log_probs.mean().item()
    
    def train(self, current_env_step: int = None):
        num_update = 0 if self.buffer_size <= self.delay_update_steps else self.utd_ratio
        
        if num_update == 0:
            return 0.0, 0.0
        
        td_loss_val, policy_loss_val = 0.0, 0.0
        
        for i_update in range(num_update):
            if self.o2o and self.replay_buffer.size > 0:
                obs, obs_next, acts, rews, done, mc_returns = self.sample_data_mix(self.batch_size)
            else:
                obs, obs_next, acts, rews, done, mc_returns = self.sample_data_offline_only(self.batch_size)
            
            total_q_loss, td_loss_val, _, _, _, alpha_loss, _ = \
                self.compute_calql_q_loss(obs, obs_next, acts, rews, done, mc_returns, use_calibration=True)
            
            for q_opt in self.q_optimizer_list:
                q_opt.zero_grad()
            total_q_loss.backward()
            for q_opt in self.q_optimizer_list:
                q_opt.step()
            
            if self.cql_lagrange:
                self.cql_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.cql_alpha_optimizer.step()
            
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                policy_loss, _ = self.compute_policy_loss(obs)
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                policy_loss_val = policy_loss.item()
                
                if self.auto_alpha:
                    _, _, _, log_prob, _, _ = self.policy_net.forward(obs, return_log_prob=True)
                    self.update_alpha(log_prob)
            
            self.update_target_networks()
        
        return policy_loss_val, td_loss_val
    
    def train_offline(self, epochs: int, test_env=None, current_env_step: int = None, log_interval: int = 10000):
        # Logs use a separate 'OfflineStep' counter so online step counts can restart from 0.
        print(f"CalQL offline training: {epochs} steps")
        
        for e in range(epochs):
            obs, obs_next, acts, rews, done, mc_returns = self.sample_data_offline_only(self.batch_size)
            
            total_q_loss, _, _, _, _, alpha_loss, _ = \
                self.compute_calql_q_loss(obs, obs_next, acts, rews, done, mc_returns, use_calibration=True)
            
            for q_opt in self.q_optimizer_list:
                q_opt.zero_grad()
            total_q_loss.backward()
            for q_opt in self.q_optimizer_list:
                q_opt.step()
            
            if self.cql_lagrange:
                self.cql_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.cql_alpha_optimizer.step()
            
            policy_loss, _ = self.compute_policy_loss(obs)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            if self.auto_alpha:
                _, _, _, log_prob, _, _ = self.policy_net.forward(obs, return_log_prob=True)
                self.update_alpha(log_prob)
            
            self.update_target_networks()

            if (e + 1) % log_interval == 0:
                print(f"  Offline pretraining epoch {e+1}/{epochs}")
                if test_env is not None:
                    test_rw = test_agent(self, test_env, 1000, None)
                    eval_reward = np.mean(test_rw)
                    print(f"    EvalReward: {eval_reward:.2f}")
                    wandb.log({
                        "OfflineEvalReward": eval_reward,
                        "OfflineStep": e + 1,
                    })
        
        print(f"CalQL offline pretraining complete: {epochs} epochs")
        return epochs