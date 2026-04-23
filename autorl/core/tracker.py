"""
Experiment result tracker — reads/writes TSV.
"""
from __future__ import annotations
import csv
import os
from pathlib import Path
from typing import List, Optional
from autorl.core.experiment import Experiment


class Tracker:
    """Persists experiment results to a TSV file."""

    def __init__(self, tsv_path: str, param_keys: List[str]):
        self.tsv_path = Path(tsv_path)
        self.param_keys = param_keys
        self._ensure_header()

    def _header(self) -> List[str]:
        return (
            ["exp_id", "round", "description"]
            + self.param_keys
            + ["best_acc", "best_f1", "best_step", "delta", "decision", "notes"]
        )

    def _ensure_header(self):
        if not self.tsv_path.exists():
            self.tsv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tsv_path, "w") as f:
                f.write("\t".join(self._header()) + "\n")

    def upsert(self, exp: Experiment):
        """Insert or update a row for this experiment."""
        rows = self._read_all()
        updated = False
        for i, row in enumerate(rows):
            if row.get("exp_id") == str(exp.exp_id):
                rows[i] = self._exp_to_row(exp)
                updated = True
                break
        if not updated:
            rows.append(self._exp_to_row(exp))
        self._write_all(rows)

    def load_all(self) -> List[dict]:
        return self._read_all()

    def next_exp_id(self) -> int:
        rows = self._read_all()
        if not rows:
            return 1
        return max(int(r["exp_id"]) for r in rows) + 1

    def _exp_to_row(self, exp: Experiment) -> dict:
        row = {
            "exp_id": str(exp.exp_id),
            "round": str(exp.round_num),
            "description": exp.description,
            "best_acc": str(exp.best_acc or "—"),
            "best_f1": str(exp.best_f1 or "—"),
            "best_step": str(exp.best_step or "—"),
            "delta": str(exp.delta or "—"),
            "decision": str(exp.decision or "pending"),
            "notes": exp.notes,
        }
        for k in self.param_keys:
            row[k] = str(exp.params.get(k, ""))
        return row

    def _read_all(self) -> List[dict]:
        if not self.tsv_path.exists():
            return []
        with open(self.tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            return [dict(r) for r in reader]

    def _write_all(self, rows: List[dict]):
        with open(self.tsv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._header(), delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def print_summary(self, baseline_acc: Optional[float] = None):
        rows = self._read_all()
        rows.sort(
            key=lambda r: float(r["best_acc"]) if r["best_acc"] not in ("—", "", "None") else -999,
            reverse=True,
        )
        header = f"{'exp':>6} | {'round':>5} | {'best_acc':>10} | {'delta':>7} | decision"
        print(header)
        print("-" * 60)
        for r in rows:
            print(
                f"{r['exp_id']:>6} | {r['round']:>5} | {r['best_acc']:>10} | "
                f"{r['delta']:>7} | {r['decision']}"
            )
        if baseline_acc:
            print(f"\nbaseline: acc={baseline_acc}")
