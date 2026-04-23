"""
Slurm backend — submits jobs via `sbatch`.
"""
from __future__ import annotations
import logging
import re
import subprocess
from pathlib import Path

from autorl.backends.base import BaseBackend
from autorl.core.experiment import Experiment

logger = logging.getLogger(__name__)


class SlurmBackend(BaseBackend):
    """
    Backend for Slurm HPC clusters.
    Submits jobs via sbatch and monitors via squeue/sacct.
    """

    def __init__(self, config: dict):
        self.train_template = config["train_template"]
        self.slurm_template = config.get("slurm_template")  # optional sbatch header template
        self.gen_dir = config.get("gen_dir", "generated")
        self.partition = config.get("partition", "gpu")
        self.nodes = config.get("nodes", 1)
        self.gpus_per_node = config.get("gpus_per_node", 8)
        self.time_limit = config.get("time_limit", "24:00:00")
        Path(self.gen_dir).mkdir(parents=True, exist_ok=True)

    def generate_script(self, exp: Experiment, output_dir: str = None) -> str:
        out_dir = Path(output_dir or self.gen_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        script_out = out_dir / f"train_exp{exp.exp_id}.sh"
        train_tmpl = Path(self.train_template).read_text()

        # Inject Slurm directives at top
        slurm_header = self._build_slurm_header(exp)
        train_tmpl = slurm_header + "\n" + train_tmpl

        # Replace placeholders
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
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")

        # Parse "Submitted batch job 12345"
        m = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not m:
            raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout}")
        return m.group(1)

    def get_status(self, job_id: str) -> str:
        # Try sacct first (works for completed jobs too)
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader", "-P"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            raw = result.stdout.strip().split("\n")[0].lower()
            return self._normalize_slurm_state(raw)

        # Fallback: squeue (only works for running/pending)
        result = subprocess.run(
            ["squeue", "-j", job_id, "--format=%T", "--noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            raw = result.stdout.strip().lower()
            return self._normalize_slurm_state(raw)

        return "UNKNOWN"

    def _normalize_slurm_state(self, state: str) -> str:
        state = state.split("|")[0].strip()
        mapping = {
            "completed": "SUCCEED", "completing": "RUNNING",
            "running": "RUNNING", "pending": "PENDING",
            "failed": "FAILED", "timeout": "FAILED",
            "cancelled": "STOPPED", "node_fail": "FAILED",
        }
        return mapping.get(state, f"UNKNOWN({state})")

    def _build_slurm_header(self, exp: Experiment) -> str:
        return f"""#!/bin/bash
#SBATCH --job-name=autorl_exp{exp.exp_id}
#SBATCH --partition={self.partition}
#SBATCH --nodes={self.nodes}
#SBATCH --gres=gpu:{self.gpus_per_node}
#SBATCH --time={self.time_limit}
#SBATCH --output=logs/exp{exp.exp_id}_%j.out
#SBATCH --error=logs/exp{exp.exp_id}_%j.err
"""
