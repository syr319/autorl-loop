"""
Abstract log parser interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional

from autorl.core.experiment import Experiment


class BaseLogParser(ABC):
    """
    Abstract interface for parsing training logs.

    Implement this for your training framework's log format.
    """

    @abstractmethod
    def extract_metrics(self, exp: Experiment) -> Optional[Dict]:
        """
        Extract metrics from training logs for the given experiment.

        Returns a dict with at minimum:
            {"best_acc": float, "best_step": int}

        Optional keys: best_f1, best_c_f1, max_grad, ...

        Returns None if no metrics could be extracted.
        """
        ...
