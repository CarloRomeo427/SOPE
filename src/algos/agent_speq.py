import numpy as np
import torch
import wandb

from src.algos.agent_base import BaseAgent
from src.algos.core import test_agent


class SPEQAgent(BaseAgent):

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
        o2o: bool = False,
        offline_epochs: int = 75000,
        trigger_interval: int = 10000,
    ):
        super().__init__(
            env_name, obs_dim, act_dim, act_limit, device,
            hidden_sizes, replay_size, batch_size, lr, gamma, polyak,
            alpha, auto_alpha, target_entropy, start_steps, utd_ratio,
            num_Q, policy_update_delay, target_drop_rate, layer_norm, o2o
        )
        
        self.offline_epochs = offline_epochs
        self.trigger_interval = trigger_interval
        self.next_trigger_step = trigger_interval
        self.env_steps = 0
    
    def store_data(self, o, a, r, o2, d):
        self.replay_buffer.store(o, a, r, o2, d)
        self.env_steps += 1

    def check_should_trigger_offline_stabilization(self) -> bool:
        if self.env_steps >= self.next_trigger_step:
            self.next_trigger_step += self.trigger_interval
            print(f"[SPEQ] Stabilization @ step {self.env_steps}")
            return True
        return False
    
    @staticmethod
    def expectile_loss(diff: torch.Tensor, expectile: float = 0.5) -> torch.Tensor:
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)
    
    def finetune_offline(self, epochs: int = None, test_env=None, current_env_step: int = None) -> int:
        epochs = epochs or self.offline_epochs
        
        for e in range(epochs):
            if self.o2o and self.replay_buffer_offline.size > 0:
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
                print(f"  Offline epoch {e+1}/{epochs}")
                if test_env is not None and current_env_step is not None:
                    test_rw = test_agent(self, test_env, 1000, None)
                    wandb.log({"OfflineEvalReward": np.mean(test_rw)}, step=current_env_step)
        
        if current_env_step is not None:
            wandb.log({"OfflineEpochs": epochs}, step=current_env_step)
        
        return epochs
    
    def train(self, current_env_step: int = None):
        if self.buffer_size <= self.delay_update_steps:
            return 0.0, 0.0
        
        total_q_loss = 0.0
        total_pi_loss = 0.0
        
        for i_update in range(self.utd_ratio):
            if self.o2o and self.replay_buffer_offline.size > 0:
                obs, obs_next, acts, rews, done = self.sample_data_mix(self.batch_size)
            else:
                obs, obs_next, acts, rews, done = self.sample_data(self.batch_size)

            y_q = self.get_sac_q_target(obs_next, rews, done)
            q_preds = [q_net(torch.cat([obs, acts], 1)) for q_net in self.q_net_list]
            q_cat = torch.cat(q_preds, dim=1)
            y_q_expanded = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q

            q_loss = self.mse_criterion(q_cat, y_q_expanded) * self.num_Q
            
            for q_opt in self.q_optimizer_list:
                q_opt.zero_grad()
            q_loss.backward()
            
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == self.utd_ratio - 1:
                action, _, _, log_prob, _, _ = self.policy_net.forward(obs)
                
                for q_net in self.q_net_list:
                    q_net.requires_grad_(False)
                
                q_values = [q_net(torch.cat([obs, action], 1)) for q_net in self.q_net_list]
                q_mean = torch.mean(torch.cat(q_values, dim=1), dim=1, keepdim=True)
                policy_loss = (self.alpha * log_prob - q_mean).mean()
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                
                for q_net in self.q_net_list:
                    q_net.requires_grad_(True)
                
                if self.auto_alpha:
                    self.update_alpha(log_prob)
                
                total_pi_loss = policy_loss.item()
            
            for q_opt in self.q_optimizer_list:
                q_opt.step()
            
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == self.utd_ratio - 1:
                self.policy_optimizer.step()
            
            self.update_target_networks()
            total_q_loss = q_loss.item() / self.num_Q
        
        return total_pi_loss, total_q_loss
