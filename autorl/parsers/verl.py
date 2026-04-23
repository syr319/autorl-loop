"""
Log parser for verl (volcengine RL framework) training logs.
"""
from __future__ import annotations
import glob
import logging
import os
import re
from typing import Dict, Optional

from autorl.parsers.base import BaseLogParser
from autorl.core.experiment import Experiment

logger = logging.getLogger(__name__)


class VerlLogParser(BaseLogParser):
    """
    Parses training logs produced by verl's main_dapo trainer.
    Extracts best-step metrics (not final step).
    """

    def __init__(self, config: dict):
        self.log_root = config.get("log_dir", "logs")
        self.exp_name_prefix = config.get("exp_name_prefix", "autorl_exp")

        # Regex patterns for verl log format
        self.patterns = {
            "step":    re.compile(r'step:(\d+) '),
            "acc":     re.compile(r'val/accuracy:([\d.]+)'),
            "f1":      re.compile(r'val/macro_f1:([\d.]+)'),
            "c_f1":    re.compile(r'val/class_C_f1:(?:np\.float64\()?([\d.]+)\)?'),
            "grad":    re.compile(r'actor/grad_norm:(?:np\.float64\()?([\d.]+)\)?'),
        }

    def extract_metrics(self, exp: Experiment) -> Optional[Dict]:
        log_file = self._find_log_file(exp)
        if not log_file:
            logger.warning(f"exp{exp.exp_id}: no log file found in {self._log_dir(exp)}")
            return None

        logger.debug(f"exp{exp.exp_id}: parsing {log_file}")
        return self._parse_log(log_file)

    def _find_log_file(self, exp: Experiment) -> Optional[str]:
        log_dir = self._log_dir(exp)
        if not os.path.isdir(log_dir):
            return None
        files = sorted(glob.glob(os.path.join(log_dir, "train_*.log")), reverse=True)
        return files[0] if files else None

    def _log_dir(self, exp: Experiment) -> str:
        exp_name = f"{self.exp_name_prefix}{exp.exp_id}"
        return os.path.join(self.log_root, exp_name)

    def _parse_log(self, log_file: str) -> Optional[Dict]:
        best_acc = 0.0
        best_step = -1
        best_f1 = None
        best_c_f1 = None
        max_grad = 0.0

        with open(log_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                # Track max grad norm
                m = self.patterns["grad"].search(line)
                if m:
                    g = float(m.group(1))
                    if g > max_grad:
                        max_grad = g

                # Track best accuracy step
                sm = self.patterns["step"].search(line)
                am = self.patterns["acc"].search(line)
                if sm and am:
                    step = int(sm.group(1))
                    acc = float(am.group(1))
                    if acc > best_acc:
                        best_acc = acc
                        best_step = step
                        fm = self.patterns["f1"].search(line)
                        cm = self.patterns["c_f1"].search(line)
                        best_f1 = float(fm.group(1)) if fm else None
                        best_c_f1 = float(cm.group(1)) if cm else None

        if best_step == -1:
            logger.warning(f"No val/accuracy found in {log_file}")
            return None

        return {
            "best_acc": best_acc,
            "best_f1": best_f1,
            "best_c_f1": best_c_f1,
            "best_step": best_step,
            "max_grad": max_grad,
        }
