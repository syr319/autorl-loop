"""
Main AutoRL loop runner.
"""
from __future__ import annotations
import logging
import time
from typing import List, Optional

from autorl.core.experiment import Experiment
from autorl.core.tracker import Tracker

logger = logging.getLogger(__name__)


class Runner:
    """
    Orchestrates the full AutoRL loop:
      submit → monitor → collect → analyze → submit → ...
    """

    def __init__(self, config: dict, backend, search_strategy, log_parser, tracker: Tracker):
        self.config = config
        self.backend = backend
        self.search = search_strategy
        self.parser = log_parser
        self.tracker = tracker

        self.baseline_acc: float = config["experiment"].get("baseline", 0.0)
        self.noise_floor: float = config["experiment"].get("noise_floor", 0.001)
        self.max_rounds: int = config["search"].get("max_rounds", 20)
        self.poll_interval: int = config.get("poll_interval", 180)

    # ── Public commands ────────────────────────────────────────────────────

    def run_init(self, experiments: List[Experiment]):
        """Submit initial batch of experiments."""
        logger.info(f"Submitting {len(experiments)} initial experiments...")
        self._submit_batch(experiments)
        logger.info("Initial batch submitted. Run `autorl run` to start the loop.")

    def run_full(self, experiments: List[Experiment]):
        """Full autonomous loop: submit → monitor → collect → analyze → repeat."""
        round_num = 0
        while round_num < self.max_rounds:
            logger.info(f"\n{'='*60}")
            logger.info(f"  ROUND {round_num}")
            logger.info(f"{'='*60}")

            self._submit_batch(experiments)
            self._monitor()
            self._collect(experiments)

            # Print summary after each round
            self.tracker.print_summary(self.baseline_acc)

            # Analyze and plan next round
            next_experiments = self.search.suggest_next(
                completed=experiments,
                tracker=self.tracker,
                next_id_start=self.tracker.next_exp_id(),
                round_num=round_num + 1,
            )

            if not next_experiments:
                logger.info("Search strategy returned no new experiments. Stopping.")
                break

            experiments = next_experiments
            round_num += 1

        logger.info("\nAutoRL loop complete.")
        self.tracker.print_summary(self.baseline_acc)

    def run_monitor(self):
        """Poll until all running jobs finish."""
        self._monitor()

    def run_collect(self, experiments: List[Experiment]):
        """Collect results for completed experiments."""
        self._collect(experiments)
        self.tracker.print_summary(self.baseline_acc)

    # ── Internal phases ────────────────────────────────────────────────────

    def _submit_batch(self, experiments: List[Experiment]):
        for exp in experiments:
            try:
                job_id = self.backend.submit(exp)
                exp.job_id = job_id
                exp.status = "SUBMITTED"
                self.tracker.upsert(exp)
                logger.info(f"  exp{exp.exp_id} submitted: job_id={job_id}")
            except Exception as e:
                logger.error(f"  exp{exp.exp_id} submit failed: {e}")
                exp.status = "SUBMIT_FAILED"
                self.tracker.upsert(exp)
            time.sleep(2)

    def _monitor(self):
        logger.info("Monitoring jobs...")
        while True:
            rows = self.tracker.load_all()
            active = [r for r in rows if r["decision"] in ("pending", "running", "submitted")]

            if not active:
                logger.info("All jobs finished.")
                break

            statuses = []
            for row in active:
                job_id = row.get("job_id", "")
                if not job_id:
                    continue
                status = self.backend.get_status(job_id)
                statuses.append(f"exp{row['exp_id']}={status}")

                if status in ("SUCCEED", "FAILED", "STOPPED"):
                    row["decision"] = status.lower()
                    self.tracker._write_all(self.tracker._read_all())

            logger.info(f"  {', '.join(statuses)}")
            logger.info(f"  Next poll in {self.poll_interval}s...")
            time.sleep(self.poll_interval)

    def _collect(self, experiments: List[Experiment]):
        logger.info("Collecting results...")
        for exp in experiments:
            if exp.status not in ("SUCCEED", "submitted"):
                logger.info(f"  exp{exp.exp_id}: skipping (status={exp.status})")
                continue

            result = self.parser.extract_metrics(exp)
            if result is None:
                logger.warning(f"  exp{exp.exp_id}: no metrics found")
                exp.decision = "no_eval"
            else:
                exp.best_acc = result["best_acc"]
                exp.best_f1 = result.get("best_f1")
                exp.best_step = result.get("best_step")
                exp.max_grad = result.get("max_grad", 0.0)
                exp.delta = exp.best_acc - self.baseline_acc

                if exp.delta > self.noise_floor:
                    exp.decision = "candidate"
                elif exp.delta > 0:
                    exp.decision = "marginal"
                elif exp.delta > -self.noise_floor:
                    exp.decision = "neutral"
                else:
                    exp.decision = "discard"

                if exp.max_grad and exp.max_grad > 15:
                    exp.decision += "+unstable"

                logger.info(
                    f"  exp{exp.exp_id}: acc={exp.best_acc:.4f} delta={exp.delta:+.4f} "
                    f"decision={exp.decision}"
                )

            self.tracker.upsert(exp)
