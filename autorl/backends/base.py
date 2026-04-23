"""
Abstract backend interface for training platforms.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from autorl.core.experiment import Experiment


class BaseBackend(ABC):
    """
    Abstract interface for a training platform backend.

    Implement this to support any cluster/cloud platform.
    """

    @abstractmethod
    def submit(self, exp: Experiment) -> str:
        """
        Submit an experiment and return the job_id string.
        Raises on failure.
        """
        ...

    @abstractmethod
    def get_status(self, job_id: str) -> str:
        """
        Return normalized job status:
          PENDING / RUNNING / SUCCEED / FAILED / STOPPED / UNKNOWN
        """
        ...

    @abstractmethod
    def generate_script(self, exp: Experiment, output_dir: str) -> str:
        """
        Generate the training script file for this experiment.
        Returns the path to the generated script.
        """
        ...
