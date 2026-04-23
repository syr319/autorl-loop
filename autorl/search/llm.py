"""
LLM-driven search strategy — uses Claude API to analyze results
and suggest the next round of hyperparameters.
"""
from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict, List, Optional

from autorl.core.experiment import Experiment
from autorl.core.tracker import Tracker
from autorl.search.base import BaseSearchStrategy

logger = logging.getLogger(__name__)


class LLMSearchStrategy(BaseSearchStrategy):
    """
    Uses an LLM (Claude) to analyze experiment results and
    intelligently suggest the next round of hyperparameters.

    Advantages over pure perturbation:
    - Understands WHY experiments failed (unstable, overfit, etc.)
    - Can reason about parameter interactions
    - Adapts strategy based on trends across rounds
    """

    SYSTEM_PROMPT = """You are an expert ML researcher helping to optimize hyperparameters for a reinforcement learning training job.

You will be given:
1. A table of completed experiments with their results
2. The parameter search space
3. The baseline performance

Your task is to suggest the next batch of experiments. Return ONLY a JSON array, no markdown, no explanation.

Decision codes:
- candidate: delta > noise_floor and stable (good, explore nearby)
- marginal: small improvement (might be worth following)
- neutral: no change
- discard: worse than baseline
- +unstable: grad_norm > 15 (learning rate likely too high)

Strategy guidelines:
- Focus on directions that showed "candidate" results
- If something was "unstable", suggest lower learning rate
- Try combining parameters that individually showed improvement
- Avoid re-testing configurations similar to existing ones
- Generate exactly {n} experiments

Return format:
[
  {{
    "description": "brief description of what this tests",
    "params": {{
      "param_name": value,
      ...
    }}
  }},
  ...
]"""

    def __init__(self, config: dict):
        self.param_specs = config["parameters"]
        self.n_per_round = config["search"].get("experiments_per_round", 5)
        self.defaults = {k: v["default"] for k, v in self.param_specs.items()}
        self.baseline_acc = config["experiment"].get("baseline", 0.0)
        self.noise_floor = config["experiment"].get("noise_floor", 0.001)

        llm_cfg = config.get("llm", {})
        self.model = llm_cfg.get("model", "claude-sonnet-4-20250514")
        self.api_key = llm_cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        self.max_tokens = llm_cfg.get("max_tokens", 2000)

    def suggest_next(
        self,
        completed: List[Experiment],
        tracker: Tracker,
        next_id_start: int,
        round_num: int,
        n: Optional[int] = None,
    ) -> List[Experiment]:
        n = n or self.n_per_round

        # Build context for LLM
        results_table = self._build_results_table(tracker)
        param_space = self._build_param_space()

        prompt = f"""Experiment results so far:

{results_table}

Parameter search space:
{param_space}

Baseline accuracy: {self.baseline_acc}
Noise floor: {self.noise_floor}

Please suggest the next {n} experiments to run."""

        logger.info("Querying LLM for next experiment suggestions...")
        suggestions = self._call_llm(prompt, n)

        if not suggestions:
            logger.warning("LLM returned no suggestions, falling back to perturbation")
            from autorl.search.perturbation import PerturbationStrategy
            fallback = PerturbationStrategy({"parameters": self.param_specs, "search": {"experiments_per_round": n, "perturb_pct": 0.3}})
            return fallback.suggest_next(completed, tracker, next_id_start, round_num, n)

        experiments = []
        for i, s in enumerate(suggestions[:n]):
            params = self._merge_with_defaults(s.get("params", {}))
            exp = Experiment(
                exp_id=next_id_start + i,
                params=params,
                description=s.get("description", f"llm_round{round_num}_exp{i}"),
                round_num=round_num,
            )
            experiments.append(exp)

        return experiments

    def _call_llm(self, prompt: str, n: int) -> List[Dict[str, Any]]:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.SYSTEM_PROMPT.format(n=n),
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic")
            return []
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return []

    def _build_results_table(self, tracker: Tracker) -> str:
        rows = tracker.load_all()
        if not rows:
            return "No experiments completed yet."

        lines = ["exp_id | " + " | ".join(self.param_specs.keys()) + " | best_acc | delta | decision"]
        lines.append("-" * 80)
        for r in sorted(rows, key=lambda x: int(x.get("exp_id", 0))):
            param_vals = " | ".join(str(r.get(k, "—")) for k in self.param_specs)
            lines.append(
                f"{r['exp_id']} | {param_vals} | {r.get('best_acc','—')} | "
                f"{r.get('delta','—')} | {r.get('decision','—')}"
            )
        return "\n".join(lines)

    def _build_param_space(self) -> str:
        lines = []
        for name, spec in self.param_specs.items():
            lo, hi = spec.get("range", [None, None])
            lines.append(
                f"  {name}: default={spec['default']}, type={spec.get('type','float')}, "
                f"range=[{lo}, {hi}]"
            )
        return "\n".join(lines)

    def _merge_with_defaults(self, params: Dict) -> Dict:
        merged = self.defaults.copy()
        for k, v in params.items():
            if k in self.param_specs:
                spec = self.param_specs[k]
                try:
                    merged[k] = int(v) if spec.get("type") == "int" else float(v)
                except (ValueError, TypeError):
                    merged[k] = v
        return merged
