import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
ACTION_BOUND_EPSILON = 1E-6
# Target-entropy values from the MBPO paper.
mbpo_target_entropy_dict = {'Hopper-v2': -1, 'HalfCheetah-v2': -3, 'Walker2d-v2': -3, 'Ant-v2': -4, 'Humanoid-v2': -2, 'HumanoidStandup-v2': -2}


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.episode_ids = np.zeros(size, dtype=np.int32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.current_episode_id = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.episode_ids[self.ptr] = self.current_episode_id

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if done:
            self.current_episode_id += 1

    def sample_batch(self, batch_size=32, idxs=None):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    idxs=idxs)

    def remove_indices(self, indices_to_remove):
        # Compact the buffer in place by dropping the given indices. O(n).
        if len(indices_to_remove) == 0:
            return

        indices_to_remove = np.unique(indices_to_remove)
        mask = np.ones(self.size, dtype=bool)
        valid_indices = indices_to_remove[indices_to_remove < self.size]
        mask[valid_indices] = False

        keep_indices = np.where(mask)[0]
        new_size = len(keep_indices)

        if new_size == 0:
            self.ptr = 0
            self.size = 0
            return

        self.obs1_buf[:new_size] = self.obs1_buf[keep_indices]
        self.obs2_buf[:new_size] = self.obs2_buf[keep_indices]
        self.acts_buf[:new_size] = self.acts_buf[keep_indices]
        self.rews_buf[:new_size] = self.rews_buf[keep_indices]
        self.done_buf[:new_size] = self.done_buf[keep_indices]
        self.episode_ids[:new_size] = self.episode_ids[keep_indices]

        self.size = new_size
        self.ptr = new_size % self.max_size


class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            hidden_activation=F.relu,
            target_drop_rate=0.0,
            layer_norm=False,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.hidden_layers = nn.ModuleList()
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc_layer)

            if target_drop_rate > 0.0:
                self.hidden_layers.append(nn.Dropout(p=target_drop_rate))
            if layer_norm:
                self.hidden_layers.append(nn.LayerNorm(fc_layer.out_features))

        # Apply activation only after the final module of each linear-(dropout)-(layernorm) block.
        self.apply_activation_per = 1
        if target_drop_rate > 0.0:
            self.apply_activation_per += 1
        if layer_norm:
            self.apply_activation_per += 1

        self.last_fc_layer = nn.Linear(in_size, output_size)
        self.apply(weights_init_)

    def forward(self, input):
        h = input
        for i, fc_layer in enumerate(self.hidden_layers):
            h = fc_layer(h)
            if ((i + 1) % self.apply_activation_per) == 0:
                h = self.hidden_activation(h)
        return self.last_fc_layer(h)


class TanhGaussianPolicy(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
            hidden_activation=F.relu,
            action_limit=1.0,
            layer_norm=False,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_activation = hidden_activation
        self.action_limit = action_limit
        self.layer_norm = layer_norm

        self.hidden_layers = nn.ModuleList()
        in_size = obs_dim

        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_size, hidden_size))
            if layer_norm:
                self.hidden_layers.append(nn.LayerNorm(hidden_size))
            in_size = hidden_size

        self.modules_per_layer = 2 if layer_norm else 1

        self.last_fc_layer = nn.Linear(in_size, action_dim)
        self.last_fc_log_std = nn.Linear(in_size, action_dim)

        self.apply(weights_init_)

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=True,
    ):
        h = obs
        for i, layer in enumerate(self.hidden_layers):
            h = layer(h)
            if (i + 1) % self.modules_per_layer == 0:
                h = self.hidden_activation(h)

        mean = self.last_fc_layer(h)
        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        normal = Normal(mean, std)

        if deterministic:
            pre_tanh_value = mean
            action = torch.tanh(mean)
        else:
            pre_tanh_value = normal.rsample()
            action = torch.tanh(pre_tanh_value)

        if return_log_prob:
            log_prob = normal.log_prob(pre_tanh_value)
            log_prob -= torch.log(1 - action.pow(2) + ACTION_BOUND_EPSILON)
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = None

        return (
            action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value,
        )


def soft_update_model1_with_model2(model1, model2, rou):
    # Polyak update: model1 <- rou * model1 + (1 - rou) * model2
    for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
        model1_param.data.copy_(rou * model1_param.data + (1 - rou) * model2_param.data)


def test_agent(agent, test_env, max_ep_len, logger, n_eval=10):
    ep_return_list = np.zeros(n_eval)
    ep_len_list = np.zeros(n_eval)
    for j in range(n_eval):
        o, info = test_env.reset()
        r, ep_ret, ep_len = 0, 0, 0
        terminated, truncated = False, False
        d = False

        while not (d or (ep_len == max_ep_len)):
            a = agent.get_test_action(o)
            o, r, terminated, truncated, _ = test_env.step(a)
            d = terminated or truncated
            ep_ret += r
            ep_len += 1
        ep_return_list[j] = ep_ret
        ep_len_list[j] = ep_len
    if logger is not None:
        logger.store(TestEpRet=ep_return_list, TestEpLen=ep_len_list)
    return ep_return_list