"""
±30% random perturbation search strategy.
"""
from __future__ import annotations
import logging
import random
from typing import Any, Dict, List, Optional

from autorl.core.experiment import Experiment
from autorl.core.tracker import Tracker
from autorl.search.base import BaseSearchStrategy

logger = logging.getLogger(__name__)


class PerturbationStrategy(BaseSearchStrategy):
    """
    Generates new experiments by perturbing the best known configuration by ±perturb_pct%.

    Strategy per round:
    - Pick top-1 stable candidate as reference
    - Generate n experiments with random ±perturb_pct% perturbations
    - Clamp values to configured bounds
    - Deduplicate against prior experiments
    """

    def __init__(self, config: dict):
        self.param_specs = config["parameters"]          # name → {default, type, range, ...}
        self.n_per_round = config["search"].get("experiments_per_round", 5)
        self.perturb_pct = config["search"].get("perturb_pct", 0.3)
        self.defaults = {k: v["default"] for k, v in self.param_specs.items()}

    def suggest_next(
        self,
        completed: List[Experiment],
        tracker: Tracker,
        next_id_start: int,
        round_num: int,
        n: Optional[int] = None,
    ) -> List[Experiment]:
        n = n or self.n_per_round
        ref = self._pick_reference(tracker)

        if ref is None:
            logger.warning("No stable candidate found — using defaults as reference")
            ref_params = self.defaults.copy()
        else:
            ref_params = ref

        existing = self._load_existing_params(tracker)
        experiments = []
        attempts = 0

        while len(experiments) < n and attempts < n * 20:
            attempts += 1
            new_params = self._perturb(ref_params)
            if self._is_duplicate(new_params, existing):
                continue

            exp = Experiment(
                exp_id=next_id_start + len(experiments),
                params=new_params,
                description=self._describe(new_params, ref_params),
                round_num=round_num,
            )
            experiments.append(exp)
            existing.append(new_params)

        if len(experiments) < n:
            logger.warning(f"Only generated {len(experiments)}/{n} unique experiments after {attempts} attempts")

        return experiments

    def _pick_reference(self, tracker: Tracker) -> Optional[Dict[str, Any]]:
        rows = tracker.load_all()
        candidates = [
            r for r in rows
            if r.get("decision") == "candidate"
            and r.get("best_acc") not in ("—", "", "None", None)
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda r: float(r["best_acc"]))
        # Reconstruct params dict from TSV row
        return {k: self._cast(k, best[k]) for k in self.param_specs if k in best}

    def _perturb(self, ref_params: Dict[str, Any]) -> Dict[str, Any]:
        new_params = {}
        for name, spec in self.param_specs.items():
            ref_val = float(ref_params.get(name, spec["default"]))
            factor = random.uniform(1 - self.perturb_pct, 1 + self.perturb_pct)
            new_val = ref_val * factor

            # Clamp to range
            lo, hi = spec.get("range", [None, None])
            if lo is not None:
                new_val = max(float(lo), new_val)
            if hi is not None:
                new_val = min(float(hi), new_val)

            # Cast to correct type
            param_type = spec.get("type", "float")
            if param_type == "int":
                new_val = max(1, round(new_val))
            else:
                # Round floats to 2 significant figures
                new_val = float(f"{new_val:.2e}")

            new_params[name] = new_val
        return new_params

    def _describe(self, new_params: Dict, ref_params: Dict) -> str:
        changed = []
        for k, v in new_params.items():
            ref_v = ref_params.get(k, self.defaults.get(k))
            if str(v) != str(ref_v):
                changed.append(f"{k}={v}")
        return "perturb: " + (", ".join(changed) if changed else "all defaults")

    def _load_existing_params(self, tracker: Tracker) -> List[Dict]:
        rows = tracker.load_all()
        result = []
        for r in rows:
            params = {k: self._cast(k, r[k]) for k in self.param_specs if k in r}
            result.append(params)
        return result

    def _is_duplicate(self, params: Dict, existing: List[Dict], tol: float = 0.01) -> bool:
        for e in existing:
            if all(
                abs(float(params.get(k, 0)) - float(e.get(k, 0))) / (abs(float(e.get(k, 1))) + 1e-10) < tol
                for k in params
            ):
                return True
        return False

    def _cast(self, name: str, value: Any) -> Any:
        spec = self.param_specs.get(name, {})
        param_type = spec.get("type", "float")
        try:
            return int(value) if param_type == "int" else float(value)
        except (ValueError, TypeError):
            return value
