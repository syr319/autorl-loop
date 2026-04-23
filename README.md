# autorl-loop

**Universal AutoRL hyperparameter search framework with LLM-driven decision making.**

Automates the full RL training loop: submit → monitor → collect → analyze → submit → ...

No human intervention required.

---

## What it does

Traditional hyperparameter tuning requires a researcher to:
1. Manually pick parameters
2. Submit training jobs
3. Wait for results
4. Analyze what worked
5. Decide what to try next
6. Repeat

`autorl-loop` automates all of this. You define your parameter space once, then the framework:
- Submits experiments in parallel to your training cluster
- Polls for job completion
- Parses training logs to extract best-step metrics
- Uses either rule-based perturbation or an LLM (Claude) to decide the next round
- Loops automatically until convergence or max rounds

---

## Quickstart

### Install

```bash
pip install autorl-loop

# For LLM-driven search (recommended):
pip install "autorl-loop[llm]"
```

### Configure

Copy and edit the example config:

```bash
cp examples/minimal/autorl_config.yaml .
```

Key fields:

```yaml
experiment:
  baseline: 0.80          # accuracy to beat
  noise_floor: 0.005      # minimum improvement to count as "candidate"

parameters:
  learning_rate:
    default: 1e-4
    type: float
    range: [1e-6, 1e-2]
  # add as many params as you want...

backend:
  type: local             # cloudml / slurm / local

search:
  strategy: llm           # perturbation / llm
  experiments_per_round: 5
  max_rounds: 20
```

### Run

```bash
# Submit initial experiments (single-param sweeps + baseline)
autorl init

# Start the full autonomous loop
nohup autorl run &

# Check progress anytime
autorl status
```

---

## Training Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Local    | ✅ | Single machine, sequential or parallel |
| Slurm    | ✅ | Via `sbatch` / `squeue` / `sacct` |
| CloudML  | ✅ | Xiaomi CloudML via `cml` CLI |
| Custom   | ✅ | Implement `BaseBackend` |

---

## Search Strategies

### Perturbation (rule-based)
Perturbs the best known configuration by ±30% randomly. Fast and reliable.

```yaml
search:
  strategy: perturbation
  perturb_pct: 0.3
```

### LLM-driven (Claude)
Uses Claude to analyze results across rounds and suggest intelligent next experiments. Understands parameter interactions and adapts strategy based on trends.

```yaml
search:
  strategy: llm
  
llm:
  model: claude-sonnet-4-20250514
  # Set ANTHROPIC_API_KEY environment variable
```

---

## Log Parsers

| Parser | Framework | Metric extracted |
|--------|-----------|-----------------|
| `verl` | volcengine/verl | val/accuracy (best step) |
| Custom | Any | Implement `BaseLogParser` |

### Custom parser

```python
from autorl.parsers.base import BaseLogParser

class MyParser(BaseLogParser):
    def extract_metrics(self, exp):
        # Parse your log files, return dict with "best_acc" and "best_step"
        return {"best_acc": 0.95, "best_step": 150}
```

---

## Adding a Custom Backend

```python
from autorl.backends.base import BaseBackend

class MyClusterBackend(BaseBackend):
    def submit(self, exp) -> str:
        # Submit job, return job_id
        ...

    def get_status(self, job_id) -> str:
        # Return: PENDING / RUNNING / SUCCEED / FAILED / STOPPED
        ...

    def generate_script(self, exp, output_dir) -> str:
        # Generate training script, return path
        ...
```

---

## Project Structure

```
autorl-loop/
├── autorl/
│   ├── core/
│   │   ├── experiment.py    # Experiment dataclass
│   │   ├── runner.py        # Main loop orchestrator
│   │   └── tracker.py       # TSV result persistence
│   ├── backends/
│   │   ├── base.py          # Abstract interface
│   │   ├── cloudml.py       # Xiaomi CloudML
│   │   ├── slurm.py         # Slurm HPC
│   │   └── local.py         # Local execution
│   ├── search/
│   │   ├── base.py          # Abstract interface
│   │   ├── perturbation.py  # ±30% random perturbation
│   │   └── llm.py           # Claude-powered search
│   ├── parsers/
│   │   ├── base.py          # Abstract interface
│   │   └── verl.py          # verl log parser
│   └── cli.py               # `autorl` CLI
├── examples/
│   ├── verl_cloudml/        # Full verl + CloudML example
│   └── minimal/             # Simple local example
└── pyproject.toml
```

---

## Decision Logic

After each experiment, results are classified:

| Decision | Meaning |
|----------|---------|
| `candidate` | delta > noise_floor, stable → follow this direction |
| `marginal` | small improvement → maybe worth following |
| `neutral` | no change |
| `discard` | worse than baseline |
| `+unstable` | grad_norm > 15 → learning rate too high |

---

## Real-world Example

This framework was developed to automate hyperparameter search for a speech rejection RL model, searching across:
- Training params: `actor_lr`, `ppo_epochs`, `entropy_coef`
- Reward weights: 15 configurable parameters

Running on Xiaomi CloudML with 5 parallel experiments per round, it achieved significant improvements over manual tuning within 3 rounds.

---

## License

MIT
