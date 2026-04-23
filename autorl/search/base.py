"""
Abstract search strategy interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional

from autorl.core.experiment import Experiment
from autorl.core.tracker import Tracker


class BaseSearchStrategy(ABC):
    """
    Abstract interface for hyperparameter search strategies.
    Implement this to add new search algorithms.
    """

    @abstractmethod
    def suggest_next(
        self,
        completed: List[Experiment],
        tracker: Tracker,
        next_id_start: int,
        round_num: int,
        n: Optional[int] = None,
    ) -> List[Experiment]:
        """
        Given completed experiments, suggest the next batch.

        Args:
            completed:     experiments that just finished
            tracker:       full result history (all rounds)
            next_id_start: first exp_id to assign to new experiments
            round_num:     current round number
            n:             number of experiments to generate (uses config default if None)

        Returns:
            List of new Experiment objects ready to submit.
        """
        ...
