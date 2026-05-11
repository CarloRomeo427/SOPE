# SOPE: Stabilizing Off-Policy Evaluation for Online RL with Prior Data

SOPE is an **online RL with prior data** method that replaces fixed-length
stabilization phases with an actor-aligned Off-Policy Policy Evaluation (OPE)
early-stopping signal: each offline update phase halts when the critic's loss on a
held-out validation split stops improving under the current policy. On 25
continuous-control tasks from the Minari benchmark, SOPE improves performance by up
to **45.6%** over baselines while cutting TFLOPs by up to **22×**, removing the
need for manual schedule tuning.

> **Paper:** _SOPE: Stabilizing Off-Policy Evaluation for Online RL with Prior Data_ — [arXiv:2605.05863](https://arxiv.org/abs/2605.05863)

---

## Algorithms

`main.py` exposes the algorithms below behind a single CLI. Selecting `--algo` is
enough — every other hyperparameter is filled in from the `ALGO_DEFAULTS` table in
`main.py`.

| `--algo`   | Paper | Description |
|------------|-------|-------------|
| `sac`      | [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290) | Soft Actor-Critic, 2 critics, online only. |
| `sacfd`    | [Vecerik et al., 2017](https://arxiv.org/abs/1707.08817) | SAC from Demonstrations: offline data loaded into the online replay buffer. |
| `rlpd`     | [Ball et al., 2023](https://arxiv.org/abs/2302.02948) | RL with Prior Data: 10-critic ensemble, UTD = 20, symmetric sampling. |
| `calql`    | [Nakamoto et al., 2023](https://arxiv.org/abs/2303.05479) | Calibrated Q-Learning: 1 M-step offline pretraining + online finetune. |
| `speq`     | [Romeo et al., 2025](https://arxiv.org/abs/2501.08669) | Periodic offline stabilization phases: 75 k Q-updates every 10 k env steps, online only. |
| `speq_o2o` | based on [Romeo et al., 2025](https://arxiv.org/abs/2501.08669) | SPEQ + symmetric sampling during both training and stabilization. |
| **`sope`** | [this paper](https://arxiv.org/abs/2605.05863) | **Ours.** Actor-aligned OPE early-stopping replaces SPEQ's fixed-length stabilization phase + symmetric sampling. |

All algorithms train on MuJoCo `-v5` environments via
[Gymnasium](https://gymnasium.farama.org/) and consume offline datasets through
[Minari](https://minari.farama.org/) (`mujoco/{env}/{quality}-v0`).

Supported environments:

```
hopper, halfcheetah, walker2d, ant, swimmer, humanoid,
invertedpendulum, inverteddoublependulum, pusher, reacher
```

Supported dataset qualities: `expert`, `medium`, `simple` (availability varies by env).

---

## Results

Aggregated across the full Minari benchmark suite (mean ± std across 10 seeds).
**Bold** = best per row.

|                 | RLPD             | Cal-QL            | SPEQ O2O          | SACfD             | **SOPE (ours)**       |
|-----------------|------------------|-------------------|-------------------|-------------------|-----------------------|
| TFLOPs          | 11 708.1         | 2 618.2           | 897.0             | 152.3             | **318.8 ± 18.7**      |
| Time (min)      | 1 006            | 681               | 124               | 30                | **64 ± 19.1**         |
| Norm. score     | 53.54 ± 39.65    | 68.49 ± 25.69     | 72.29 ± 25.20     | 67.62 ± 25.56     | **77.94 ± 25.11**     |

SOPE matches or beats every online-with-prior-data baseline on aggregate score
while using ~37× fewer TFLOPs and ~16× less wall-clock time than RLPD, the most
expensive one.

---

## Installation

```bash
git clone https://github.com/CarloRomeo427/SOPE.git
cd SOPE
conda create -n sope python=3.11 -y
conda activate sope
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py --algo sope --env hopper                 # SOPE on Hopper / expert
python main.py --algo sope --env humanoid --log-wandb   # log to Weights & Biases
python main.py --help                                   # full CLI
```

Algorithm-internal hyperparameters (critic count, UTD, dropout, offline epochs,
validation patience, …) are fixed per algorithm in the `ALGO_DEFAULTS` table in
`main.py`. Edit that table to tune them.

---

## Citation

If you use this code, please cite:

```bibtex
@article{sope2026,
  title   = {SOPE: Stabilizing Off-Policy Evaluation for Online RL with Prior Data},
  author  = {Carlo Romeo, Girolamo Macaluso, Alessandro Sestini, Andrew D. Bagdanov},
  journal = {arXiv preprint arXiv:2605.05863},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.05863}
}
```

---

## License

See [LICENSE](LICENSE).
