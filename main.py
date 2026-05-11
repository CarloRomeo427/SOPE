import os

# Some conda installs export a stale SSL_CERT_FILE/REQUESTS_CA_BUNDLE that points at a
# file which no longer exists. httpx (pulled in by huggingface_hub / minari / wandb) reads
# these eagerly when building its client and crashes before any HTTPS call is made. Repoint
# them at certifi's bundle (or drop them) before anything imports httpx.
for _var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    _path = os.environ.get(_var)
    if _path and not os.path.isfile(_path):
        try:
            import certifi
            os.environ[_var] = certifi.where()
        except ImportError:
            del os.environ[_var]

import argparse
import time
import numpy as np
import torch
import wandb
import gymnasium as gym


MUJOCO_ENVS = {
    'hopper', 'halfcheetah', 'walker2d', 'ant', 'swimmer', 'humanoid',
    'invertedpendulum', 'inverteddoublependulum', 'pusher', 'reacher'
}

MUJOCO_ENV_VERSION = 'v5'
MUJOCO_DATASET_VERSION = 'v0'

STEPS_PER_EPOCH = 1000
MAX_EP_LEN = 1000
START_STEPS = 5000


def normalize_env_name(env_name: str) -> str:
    """Map user-provided env name to canonical short name."""
    env_lower = env_name.lower().replace('_', '-')

    base = env_lower
    if '-v' in env_lower:
        parts = env_lower.rsplit('-v', 1)
        if parts[1].isdigit():
            base = parts[0]

    for mujoco_env in MUJOCO_ENVS:
        if base == mujoco_env or base.startswith(mujoco_env + '-'):
            return mujoco_env

    raise ValueError(f"Unknown environment: {env_name}. "
                     f"Supported MuJoCo envs: {', '.join(sorted(MUJOCO_ENVS))}")


def get_gymnasium_env_name(canonical_name: str) -> str:
    name_map = {
        'hopper': 'Hopper',
        'halfcheetah': 'HalfCheetah',
        'walker2d': 'Walker2d',
        'ant': 'Ant',
        'swimmer': 'Swimmer',
        'humanoid': 'Humanoid',
        'invertedpendulum': 'InvertedPendulum',
        'inverteddoublependulum': 'InvertedDoublePendulum',
        'pusher': 'Pusher',
        'reacher': 'Reacher',
    }
    return f"{name_map[canonical_name]}-{MUJOCO_ENV_VERSION}"


def get_minari_dataset_name(canonical_name: str, quality: str) -> str:
    return f"mujoco/{canonical_name}/{quality}-{MUJOCO_DATASET_VERSION}"


def get_display_name(canonical_name: str, quality: str) -> str:
    return f"{canonical_name}-v5_{quality}"


def get_dropout_rate(canonical_name: str) -> float:
    """Per-env target Q-network dropout rate (used by DroQ/SPEQ/SOPE)."""
    if canonical_name == 'humanoid':
        return 0.1
    if canonical_name == 'ant':
        return 0.01
    if canonical_name in ['walker2d', 'halfcheetah', 'pusher']:
        return 0.005
    if canonical_name in ['hopper', 'swimmer', 'reacher', 'invertedpendulum', 'inverteddoublependulum']:
        return 0.001
    return 0.005


def make_env(canonical_name: str):
    return gym.make(get_gymnasium_env_name(canonical_name))


def get_env_info(env):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    if hasattr(env, '_max_episode_steps'):
        max_ep_len = env._max_episode_steps
    elif hasattr(env, 'spec') and env.spec is not None and hasattr(env.spec, 'max_episode_steps'):
        max_ep_len = env.spec.max_episode_steps or MAX_EP_LEN
    else:
        max_ep_len = MAX_EP_LEN

    return obs_dim, act_dim, act_limit, max_ep_len


def load_minari_dataset(agent, canonical_name: str, quality: str, compute_mc_returns: bool = False):
    import minari

    dataset_name = get_minari_dataset_name(canonical_name, quality)
    print(f"Loading Minari dataset: {dataset_name}")

    try:
        dataset = minari.load_dataset(dataset_name)
    except Exception:
        print(f"Downloading dataset: {dataset_name}")
        minari.download_dataset(dataset_name)
        dataset = minari.load_dataset(dataset_name)

    print(f"Dataset: {dataset.total_episodes} episodes, {dataset.total_steps} steps")

    gamma = agent.gamma
    total = 0

    for episode in dataset.iterate_episodes():
        observations = episode.observations
        actions = episode.actions
        rewards = episode.rewards
        terminations = episode.terminations
        T = len(actions)

        if compute_mc_returns:
            mc_returns = np.zeros(T)
            mc_returns[-1] = rewards[-1]
            for t in range(T - 2, -1, -1):
                mc_returns[t] = rewards[t] + gamma * mc_returns[t + 1]

        for t in range(T):
            obs = observations[t].astype(np.float32)
            next_obs = observations[t + 1].astype(np.float32)
            action = actions[t].astype(np.float32)
            reward = float(rewards[t])
            done = bool(terminations[t])

            if compute_mc_returns:
                agent.store_data_offline(obs, action, reward, next_obs, done, mc_returns[t])
            else:
                agent.store_data_offline(obs, action, reward, next_obs, done)
            total += 1

    print(f"Loaded {total} transitions")
    return total


# Per-algorithm settings.
#   description      : one-line summary shown in --help
#   agent_class      : dotted import path of the agent class
#   use_offline_data : whether to load the Minari dataset before online training
#   per_env_dropout  : whether target_drop_rate should be auto-selected from the env
#   pretrain_steps   : number of offline pretraining steps (None = no pretrain)
#   config           : kwargs forwarded to the agent constructor
ALGO_DEFAULTS = {
    'sac': {
        'description': 'Soft Actor-Critic (online only, 2 critics)',
        'agent_class': 'src.algos.agent_sac.SACAgent',
        'use_offline_data': False,
        'per_env_dropout': False,
        'pretrain_steps': None,
        'config': {
            'num_Q': 2, 'utd_ratio': 1, 'policy_update_delay': 1,
            'o2o': False, 'target_drop_rate': 0.0,
        },
    },
    'sacfd': {
        'description': 'SAC from Demonstrations (online with 50/50 offline data mixing)',
        'agent_class': 'src.algos.agent_sac.SACAgent',
        'use_offline_data': True,
        'per_env_dropout': False,
        'pretrain_steps': None,
        'config': {
            'num_Q': 2, 'utd_ratio': 1, 'policy_update_delay': 1,
            'o2o': True, 'target_drop_rate': 0.0,
        },
    },
    'rlpd': {
        'description': 'RL with Prior Data (10-critic ensemble, UTD=20, 50/50 online-offline mix)',
        'agent_class': 'src.algos.agent_sac.SACAgent',
        'use_offline_data': True,
        'per_env_dropout': False,
        'pretrain_steps': None,
        'config': {
            'num_Q': 10, 'utd_ratio': 20, 'policy_update_delay': 1,
            'o2o': True, 'target_drop_rate': 0.0,
        },
    },
    'droq': {
        'description': 'Dropout Q-functions (high-UTD SAC with target Q-net dropout)',
        'agent_class': 'src.algos.agent_droq.DroQAgent',
        'use_offline_data': False,
        'per_env_dropout': True,
        'pretrain_steps': None,
        'config': {
            'num_Q': 2, 'utd_ratio': 20, 'policy_update_delay': 20,
            'o2o': False,
        },
    },
    'calql': {
        'description': 'Calibrated Q-Learning (1M-step offline pretrain + online finetune)',
        'agent_class': 'src.algos.agent_calql.CalQLAgent',
        'use_offline_data': True,
        'per_env_dropout': False,
        'pretrain_steps': 1_000_000,
        'config': {
            'cql_alpha': 5.0, 'cql_n_actions': 10, 'cql_lagrange': False,
            'mixing_ratio': 0.5, 'policy_update_delay': 1,
            'o2o': True, 'target_drop_rate': 0.0,
        },
    },
    'speq': {
        'description': 'Stabilized Policy-Enhanced Q-learning (online; 75k Q-updates every 10k env steps)',
        'agent_class': 'src.algos.agent_speq.SPEQAgent',
        'use_offline_data': False,
        'per_env_dropout': True,
        'pretrain_steps': None,
        'config': {
            'num_Q': 2, 'utd_ratio': 1, 'policy_update_delay': 20,
            'offline_epochs': 75000, 'trigger_interval': 10000,
            'o2o': False,
        },
    },
    'speq_o2o': {
        'description': 'SPEQ in offline-to-online mode (50/50 mixing during stabilization)',
        'agent_class': 'src.algos.agent_speq.SPEQAgent',
        'use_offline_data': True,
        'per_env_dropout': True,
        'pretrain_steps': None,
        'config': {
            'num_Q': 2, 'utd_ratio': 1, 'policy_update_delay': 20,
            'offline_epochs': 75000, 'trigger_interval': 10000,
            'o2o': True,
        },
    },
    'sope': {
        'description': 'SPEQ + policy-loss early stopping on a 10% held-out validation slice',
        'agent_class': 'src.algos.agent_sope.SOPEAgent',
        'use_offline_data': True,
        'per_env_dropout': True,
        'pretrain_steps': None,
        'config': {
            'num_Q': 2, 'utd_ratio': 1, 'policy_update_delay': 20,
            'offline_epochs': 75000, 'trigger_interval': 10000,
            'val_check_interval': 1000, 'val_patience': 10000, 'val_pct': 0.1,
            'o2o': True,
        },
    },
}


def get_agent_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def build_algo_config(algo_name: str, canonical_env: str) -> dict:
    spec = ALGO_DEFAULTS[algo_name]
    config = dict(spec['config'])
    if spec['per_env_dropout']:
        config['target_drop_rate'] = get_dropout_rate(canonical_env)
    return config


def evaluate(agent, env, max_ep_len: int, num_episodes: int = 10) -> float:
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_ret = 0
        for _ in range(max_ep_len):
            action = agent.get_test_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            if terminated or truncated:
                break
        rewards.append(ep_ret)
    return np.mean(rewards)


def train(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    spec = ALGO_DEFAULTS[args.algo]

    canonical_name = normalize_env_name(args.env)
    gym_env_name = get_gymnasium_env_name(canonical_name)
    display_name = get_display_name(canonical_name, args.dataset_quality)

    print(f"Environment: {args.env} -> {gym_env_name}")

    env = make_env(canonical_name)
    test_env = make_env(canonical_name)
    obs_dim, act_dim, act_limit, max_ep_len = get_env_info(env)
    max_ep_len = min(MAX_EP_LEN, max_ep_len)

    total_steps = STEPS_PER_EPOCH * args.epochs + 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    algo_config = build_algo_config(args.algo, canonical_name)
    AgentClass = get_agent_class(spec['agent_class'])

    agent = AgentClass(
        env_name=gym_env_name,
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        device=device,
        start_steps=START_STEPS,
        **algo_config,
    )

    drop = algo_config.get('target_drop_rate', 0.0)
    print(f"Algorithm: {args.algo}, Env: {display_name}, obs_dim: {obs_dim}, act_dim: {act_dim}, dropout: {drop}")

    if spec['use_offline_data']:
        compute_mc = args.algo == 'calql'
        load_minari_dataset(agent, canonical_name, args.dataset_quality, compute_mc)

        if spec['pretrain_steps'] is not None and spec['pretrain_steps'] > 0:
            print(f"Offline pretraining: {spec['pretrain_steps']} steps")
            agent.train_offline(epochs=spec['pretrain_steps'], test_env=test_env)

    obs, _ = env.reset(seed=args.seed)
    ep_ret, ep_len = 0, 0
    start_time = time.time()

    for t in range(total_steps):
        action = agent.get_exploration_action(obs, env)
        obs_next, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        ep_len += 1
        agent.store_data(obs, action, reward, obs_next, terminated and ep_len < max_ep_len)

        # SPEQ/SOPE trigger their offline stabilization phase here.
        if hasattr(agent, 'check_should_trigger_offline_stabilization'):
            if agent.check_should_trigger_offline_stabilization():
                agent.finetune_offline(test_env=test_env, current_env_step=t + 1)

        pi_loss, q_loss = agent.train(current_env_step=t + 1)

        obs = obs_next
        ep_ret += reward

        if done or ep_len >= max_ep_len:
            obs, _ = env.reset()
            ep_ret, ep_len = 0, 0

        if (t + 1) % STEPS_PER_EPOCH == 0:
            epoch = (t + 1) // STEPS_PER_EPOCH
            eval_reward = evaluate(agent, test_env, max_ep_len)

            print(f"Epoch {epoch:4d} | Step {t+1:7d} | EvalReward: {eval_reward:8.2f} | Time: {time.time()-start_time:.0f}s")

            wandb.log({
                "epoch": epoch,
                "policy_loss": pi_loss,
                "mean_q_loss": q_loss,
                "EvalReward": eval_reward,
            }, step=t + 1)

    print("Training complete!")


def _format_algo_table() -> str:
    width = max(len(name) for name in ALGO_DEFAULTS)
    lines = ["Available algorithms (selecting one auto-configures all related hyperparameters):"]
    for name in sorted(ALGO_DEFAULTS):
        lines.append(f"  {name:<{width}}  {ALGO_DEFAULTS[name]['description']}")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_format_algo_table(),
    )
    parser.add_argument("--algo", type=str, default='sac',
                        choices=sorted(ALGO_DEFAULTS.keys()),
                        metavar='ALGO',
                        help="Algorithm name (see list above). Default: sac")
    parser.add_argument("--env", type=str, default='hopper',
                        help=f"MuJoCo env: {', '.join(sorted(MUJOCO_ENVS))}. Default: hopper")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--dataset-quality", type=str, default='expert',
                        help="Minari dataset quality: expert / medium / simple (ignored when the algo runs online-only).")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    canonical_name = normalize_env_name(args.env)
    display_name = get_display_name(canonical_name, args.dataset_quality)

    if args.algo == 'sope':
        val_pct = ALGO_DEFAULTS['sope']['config']['val_pct']
        exp_name = f"sope{int(val_pct * 100)}_{display_name.capitalize()}"
    else:
        exp_name = f"{args.algo}_{display_name.capitalize()}"

    if args.epochs != 300:
        exp_name += f"_epochs{args.epochs}"

    wandb.init(
        name=exp_name,
        project="SPEQ",
        config=vars(args),
        mode='online' if args.log_wandb else 'disabled'
    )

    train(args)
    wandb.finish()
