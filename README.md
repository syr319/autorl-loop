# autorl-loop

<div align="center">

**Universal AutoRL Hyperparameter Search Framework**

*LLM-driven • Platform-agnostic • Zero human intervention*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)

[English](#english) | [中文](#中文)

</div>

---

<a name="english"></a>
## English

### Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        autorl-loop                                  │
│                                                                     │
│   You provide:             Framework does:                          │
│   • train script    ──►   • Generate experiment configs             │
│   • param space     ──►   • Submit jobs in parallel                 │
│   • baseline        ──►   • Poll for completion                     │
│                           • Parse logs → extract metrics            │
│                           • Analyze results (rule / LLM)            │
│                           • Plan next round automatically           │
│                           • Loop until convergence                  │
└─────────────────────────────────────────────────────────────────────┘
```

### The Loop

```
                    ┌──────────────────────────────────┐
                    │                                  │
                    ▼                                  │
            ┌──────────────┐                          │
            │    SUBMIT    │  batch submit N exps      │
            │  (parallel)  │  to training cluster      │
            └──────┬───────┘                          │
                   │                                  │
                   ▼                                  │
            ┌──────────────┐                          │
            │    MONITOR   │  poll every 3 min        │
            │   (polling)  │  PENDING→RUNNING→DONE    │
            └──────┬───────┘                          │
                   │                                  │
                   ▼                                  │
            ┌──────────────┐                          │
            │    COLLECT   │  parse training logs      │
            │  (metrics)   │  extract best-step acc    │
            └──────┬───────┘                          │
                   │                                  │
                   ▼                                  │
            ┌──────────────┐   candidate →  keep  ───►│
            │    ANALYZE   │   discard   →  skip      │
            │  (LLM/rule)  │   unstable  →  lower lr  │
            └──────┬───────┘                          │
                   │                                  │
                   ▼                                  │
            ┌──────────────┐                          │
            │     PLAN     │  generate next N exps ───┘
            │  (next round)│  perturb best config
            └──────────────┘
```

### Quickstart

```bash
# Install
pip install autorl-loop
pip install "autorl-loop[llm]"   # for LLM-driven search

# Configure (copy & edit)
cp examples/minimal/autorl_config.yaml .

# Submit initial experiments
autorl init

# Start autonomous loop
nohup autorl run &

# Check progress anytime
autorl status
```

### Configuration

```yaml
experiment:
  baseline: 0.80        # accuracy to beat
  noise_floor: 0.005    # min delta to count as "candidate"

parameters:
  actor_lr:
    default: 5e-5
    type: float
    range: [1e-6, 1e-3]
  ppo_epochs:
    default: 4
    type: int
    range: [1, 8]
  # add as many params as you want...

backend:
  type: cloudml         # cloudml / slurm / local

search:
  strategy: llm         # perturbation / llm
  experiments_per_round: 5
  max_rounds: 20
```

### Platform Support

| Platform | Backend | Notes |
|----------|---------|-------|
| Local | `local` | Single machine, sequential or parallel |
| Slurm | `slurm` | Via `sbatch` / `squeue` / `sacct` |
| CloudML | `cloudml` | Via `cml` CLI |
| Custom | `BaseBackend` | Implement 3 methods |

### Search Strategies

| Strategy | How it works | Best for |
|----------|-------------|----------|
| `perturbation` | ±30% random perturbation of best config | Fast, reliable |
| `llm` | LLM analyzes results, suggests next params | Complex param interactions |

### Decision Logic

| Label | Meaning | Next action |
|-------|---------|-------------|
| `candidate` | delta > noise_floor, stable | Explore nearby |
| `marginal` | Small improvement | Follow cautiously |
| `neutral` | No change | Ignore |
| `discard` | Worse than baseline | Abandon direction |
| `+unstable` | grad_norm > 15 | Lower learning rate |

### Extending

**Custom backend:**
```python
from autorl.backends.base import BaseBackend

class MyBackend(BaseBackend):
    def submit(self, exp) -> str:          # return job_id
    def get_status(self, job_id) -> str:   # PENDING/RUNNING/SUCCEED/FAILED
    def generate_script(self, exp, dir):   # return script path
```

**Custom log parser:**
```python
from autorl.parsers.base import BaseLogParser

class MyParser(BaseLogParser):
    def extract_metrics(self, exp) -> dict:
        # return {"best_acc": float, "best_step": int}
```

### Project Structure

```
autorl-loop/
├── autorl/
│   ├── core/
│   │   ├── experiment.py    # Experiment dataclass
│   │   ├── runner.py        # Main loop orchestrator
│   │   └── tracker.py       # TSV result persistence
│   ├── backends/
│   │   ├── base.py          # Abstract interface
│   │   ├── cloudml.py       # CloudML platform
│   │   ├── slurm.py         # Slurm HPC clusters
│   │   └── local.py         # Local execution
│   ├── search/
│   │   ├── base.py          # Abstract interface
│   │   ├── perturbation.py  # ±30% perturbation
│   │   └── llm.py           # LLM-driven search
│   ├── parsers/
│   │   ├── base.py          # Abstract interface
│   │   └── verl.py          # verl framework parser
│   └── cli.py               # autorl CLI
├── examples/
│   ├── verl_cloudml/        # verl + CloudML example
│   └── minimal/             # minimal local example
└── pyproject.toml
```

---

<a name="中文"></a>
## 中文

### 简介

`autorl-loop` 是一个**通用的强化学习超参数自动搜索框架**，支持 LLM 驱动的智能决策。

你只需要提供：训练脚本、参数搜索空间、基线指标。框架自动完成所有剩余工作：并行提交实验 → 轮询等待 → 解析日志 → 分析结果 → 规划下一轮 → 循环直到收敛。全程无需人工干预。

### 运行流程

```
                    ┌──────────────────────────────────┐
                    │                                  │
                    ▼                                  │
            ┌──────────────┐                          │
            │   提交实验    │  批量并行提交到训练集群   │
            └──────┬───────┘                          │
                   │                                  │
                   ▼                                  │
            ┌──────────────┐                          │
            │   轮询状态    │  每3分钟查询一次          │
            │              │  PENDING→RUNNING→完成     │
            └──────┬───────┘                          │
                   │                                  │
                   ▼                                  │
            ┌──────────────┐                          │
            │   收集结果    │  解析训练日志             │
            │              │  提取最佳 step 指标       │
            └──────┬───────┘                          │
                   │                                  │
                   ▼                                  │
            ┌──────────────┐   candidate → 继续探索 ──►│
            │   分析决策    │   discard   → 放弃方向    │
            │ (LLM / 规则) │   unstable  → 降低 lr    │
            └──────┬───────┘                          │
                   │                                  │
                   ▼                                  │
            ┌──────────────┐                          │
            │  规划下一轮   │  在最优配置附近生成新实验 ─┘
            └──────────────┘
```

### 快速开始

```bash
# 安装
pip install autorl-loop
pip install "autorl-loop[llm]"   # 使用 LLM 决策引擎

# 复制配置模板并编辑
cp examples/minimal/autorl_config.yaml .

# 提交初始实验（baseline + 单参数扫描）
autorl init

# 启动全自动循环（后台运行）
nohup autorl run &

# 随时查看进度
autorl status
```

### 配置文件

```yaml
experiment:
  baseline: 0.9411      # 基线指标，需要超越的目标
  noise_floor: 0.001    # 最小有效提升，低于此视为无效

parameters:
  actor_lr:             # 参数名（可任意添加）
    default: 5e-5       # 默认值（baseline 使用）
    type: float         # float 或 int
    range: [1e-6, 1e-3] # 搜索范围
  ppo_epochs:
    default: 4
    type: int
    range: [1, 8]

backend:
  type: cloudml         # cloudml / slurm / local

search:
  strategy: llm         # perturbation（规则）/ llm（智能）
  experiments_per_round: 5
  max_rounds: 20
```

### 平台支持

| 平台 | 后端 | 说明 |
|------|------|------|
| 本地 | `local` | 单机顺序或并行执行 |
| Slurm | `slurm` | 通过 `sbatch` / `squeue` / `sacct` |
| CloudML | `cloudml` | 通过 `cml` CLI |
| 自定义 | `BaseBackend` | 实现3个方法即可接入任意平台 |

### 搜索策略

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| `perturbation` | 在最优配置基础上随机 ±30% 扰动 | 快速可靠，适合入门 |
| `llm` | LLM 分析全量结果，推断下一轮参数 | 参数之间存在复杂交互 |

LLM 策略相比传统方法的优势：不只是数值优化，还能理解实验失败的**原因**（梯度不稳定、过拟合等），并发现参数之间的交互规律。

### 判决逻辑

| 标签 | 含义 | 下一步动作 |
|------|------|-----------|
| `candidate` | 超出基线 > noise_floor 且稳定 | 继续在此方向探索 |
| `marginal` | 小幅提升 | 谨慎跟进 |
| `neutral` | 无变化 | 忽略 |
| `discard` | 低于基线 | 放弃此方向 |
| `+unstable` | 梯度范数 > 15 | 降低学习率重试 |

### 扩展接口

**自定义训练平台后端（3个方法）：**
```python
from autorl.backends.base import BaseBackend

class MyBackend(BaseBackend):
    def submit(self, exp) -> str:          # 提交任务，返回 job_id
    def get_status(self, job_id) -> str:   # 返回 PENDING/RUNNING/SUCCEED/FAILED
    def generate_script(self, exp, dir):   # 生成训练脚本，返回路径
```

**自定义日志解析器：**
```python
from autorl.parsers.base import BaseLogParser

class MyParser(BaseLogParser):
    def extract_metrics(self, exp) -> dict:
        # 解析你的训练日志，返回指标
        return {"best_acc": 0.95, "best_step": 150}
```

---

## License

MIT