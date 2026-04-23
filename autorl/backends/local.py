"""
Local backend — runs training scripts directly via subprocess.
Useful for single-machine or local multi-GPU experiments.
"""
from __future__ import annotations
import logging
import subprocess
from pathlib import Path
from typing import Dict

from autorl.backends.base import BaseBackend
from autorl.core.experiment import Experiment

logger = logging.getLogger(__name__)


class LocalBackend(BaseBackend):
    """
    Backend that runs training scripts locally via subprocess.
    Jobs run sequentially (or in parallel if run_parallel=True).
    """

    def __init__(self, config: dict):
        self.train_template = config["train_template"]
        self.gen_dir = config.get("gen_dir", "generated")
        self.run_parallel = config.get("run_parallel", False)
        self._processes: Dict[str, subprocess.Popen] = {}
        Path(self.gen_dir).mkdir(parents=True, exist_ok=True)

    def generate_script(self, exp: Experiment, output_dir: str = None) -> str:
        out_dir = Path(output_dir or self.gen_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        script_out = out_dir / f"train_exp{exp.exp_id}.sh"
        train_tmpl = Path(self.train_template).read_text()

        replacements = {"{{EXP_ID}}": str(exp.exp_id), "{{EXP_DESC}}": exp.description}
        for k, v in exp.params.items():
            replacements[f"{{{{{k.upper()}}}}}"] = str(v)

        for placeholder, value in replacements.items():
            train_tmpl = train_tmpl.replace(placeholder, value)

        script_out.write_text(train_tmpl)
        script_out.chmod(0o755)
        return str(script_out)

    def submit(self, exp: Experiment) -> str:
        script_path = self.generate_script(exp)
        job_id = f"local_{exp.exp_id}"

        if self.run_parallel:
            proc = subprocess.Popen(["bash", script_path])
            self._processes[job_id] = proc
            logger.info(f"  exp{exp.exp_id} started (PID={proc.pid})")
        else:
            # Sequential: block until done
            result = subprocess.run(["bash", script_path])
            exit_code = result.returncode
            self._processes[job_id] = exit_code  # store exit code
            logger.info(f"  exp{exp.exp_id} finished (exit={exit_code})")

        return job_id

    def get_status(self, job_id: str) -> str:
        proc = self._processes.get(job_id)
        if proc is None:
            return "UNKNOWN"

        if isinstance(proc, int):
            # Sequential mode: stored exit code
            return "SUCCEED" if proc == 0 else "FAILED"

        # Parallel mode: check if process is still running
        proc.poll()
        if proc.returncode is None:
            return "RUNNING"
        return "SUCCEED" if proc.returncode == 0 else "FAILED"
