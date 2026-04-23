"""
Experiment dataclass and result types.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Experiment:
    """Represents a single hyperparameter experiment."""
    exp_id: int
    params: Dict[str, Any]
    description: str = ""
    round_num: int = 0

    # Runtime state
    job_id: Optional[str] = None
    status: str = "PENDING"  # PENDING / RUNNING / SUCCEED / FAILED / STOPPED

    # Results (filled after collect)
    best_acc: Optional[float] = None
    best_f1: Optional[float] = None
    best_step: Optional[int] = None
    delta: Optional[float] = None
    decision: Optional[str] = None  # candidate / marginal / neutral / discard / +unstable
    max_grad: Optional[float] = None
    notes: str = ""

    def is_terminal(self) -> bool:
        return self.status in ("SUCCEED", "FAILED", "STOPPED")

    def is_success(self) -> bool:
        return self.status == "SUCCEED"

    def is_candidate(self) -> bool:
        return self.decision is not None and "candidate" in self.decision and "unstable" not in self.decision

    def to_tsv_row(self, param_keys: list[str]) -> str:
        cols = [
            str(self.exp_id),
            str(self.round_num),
            self.description,
        ] + [str(self.params.get(k, "")) for k in param_keys] + [
            str(self.best_acc or "—"),
            str(self.best_f1 or "—"),
            str(self.best_step or "—"),
            str(self.delta or "—"),
            str(self.decision or "pending"),
            self.notes,
        ]
        return "\t".join(cols)
