"""
CloudML backend (Xiaomi CloudML platform via `cml` CLI).
"""
from __future__ import annotations
import json
import logging
import os
import re
import subprocess
from pathlib import Path

from autorl.backends.base import BaseBackend
from autorl.core.experiment import Experiment

logger = logging.getLogger(__name__)


class CloudMLBackend(BaseBackend):
    """
    Backend for Xiaomi CloudML platform.
    Uses the `cml` CLI tool to submit and monitor jobs.
    """

    STATUS_MAP = {
        "succeed": "SUCCEED", "succeeded": "SUCCEED", "success": "SUCCEED",
        "fail": "FAILED", "failed": "FAILED", "failure": "FAILED",
        "running": "RUNNING", "run": "RUNNING",
        "pending": "PENDING", "queued": "PENDING", "waiting": "PENDING",
        "stopped": "STOPPED", "killed": "STOPPED", "cancelled": "STOPPED",
        "creating": "PENDING", "starting": "PENDING",
    }

    def __init__(self, config: dict):
        self.cml_path = config.get("cml_path", "cml")
        self.queue_id = config.get("queue_id", "")
        self.train_template = config["train_template"]
        self.yaml_template = config["yaml_template"]
        self.gen_dir = config.get("gen_dir", "generated")
        self.exp_name_prefix = config.get("exp_name_prefix", "autorl_exp")
        Path(self.gen_dir).mkdir(parents=True, exist_ok=True)

    def generate_script(self, exp: Experiment, output_dir: str = None) -> str:
        out_dir = Path(output_dir or self.gen_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_out = out_dir / f"train_exp{exp.exp_id}.sh"
        yaml_out = out_dir / f"cloudml_exp{exp.exp_id}.yaml"

        # Read templates
        train_tmpl = Path(self.train_template).read_text()
        yaml_tmpl = Path(self.yaml_template).read_text()

        # Replace all placeholders
        replacements = {
            "{{EXP_ID}}": str(exp.exp_id),
            "{{EXP_DESC}}": exp.description,
            "{{ROUND}}": str(exp.round_num),
        }
        for k, v in exp.params.items():
            replacements[f"{{{{{k.upper()}}}}}"] = str(v)

        for placeholder, value in replacements.items():
            train_tmpl = train_tmpl.replace(placeholder, value)
            yaml_tmpl = yaml_tmpl.replace(placeholder, value)

        # Replace queue_id if present
        if self.queue_id:
            yaml_tmpl = yaml_tmpl.replace("{{QUEUE_ID}}", self.queue_id)

        train_out.write_text(train_tmpl)
        train_out.chmod(0o755)
        yaml_out.write_text(yaml_tmpl)

        logger.debug(f"Generated: {train_out}")
        logger.debug(f"Generated: {yaml_out}")
        return str(yaml_out)

    def submit(self, exp: Experiment) -> str:
        yaml_path = self.generate_script(exp)

        result = subprocess.run(
            [self.cml_path, "custom_train", "submit", "--filename", yaml_path],
            capture_output=True, text=True
        )
        output = result.stdout + result.stderr

        logger.debug(f"cml submit output:\n{output}")

        if result.returncode != 0:
            raise RuntimeError(f"cml submit failed (code={result.returncode}): {output}")

        job_id = self._parse_job_id(output)
        if not job_id:
            logger.warning(f"Could not parse job_id from output:\n{output}")
            raise RuntimeError("Failed to parse job_id from cml output")

        return job_id

    def get_status(self, job_id: str) -> str:
        result = subprocess.run(
            [self.cml_path, "custom_train", "describe", job_id],
            capture_output=True, text=True
        )
        output = result.stdout + result.stderr

        # Try JSON parse first
        try:
            data = json.loads(output)
            raw = (data.get("status") or data.get("jobStatus") or "").lower()
            return self.STATUS_MAP.get(raw, f"UNKNOWN({raw})")
        except json.JSONDecodeError:
            pass

        # Fallback: regex
        m = re.search(r'(?:status|jobStatus)\s*[:=]\s*(\S+)', output, re.IGNORECASE)
        if m:
            raw = m.group(1).lower().strip('"\',:')
            return self.STATUS_MAP.get(raw, f"UNKNOWN({raw})")

        return "UNKNOWN"

    def _parse_job_id(self, output: str) -> str:
        """Try multiple patterns to extract job ID from cml output."""
        # Pattern 1: [t-XXXX] style (Xiaomi CloudML format)
        m = re.search(r'\[([t]-[\w-]+)\]', output)
        if m:
            return m.group(1)

        # Pattern 2: "job_id: xxx" or "jobId: xxx"
        m = re.search(r'(?:job_?id)\s*[:=]\s*(\S+)', output, re.IGNORECASE)
        if m:
            return m.group(1).strip('"\',:')

        # Pattern 3: JSON
        try:
            data = json.loads(output)
            return str(data.get("id") or data.get("jobId") or data.get("job_id") or "")
        except Exception:
            pass

        # Pattern 4: standalone ID on its own line
        m = re.search(r'^\s*(t-[\w-]+)\s*$', output, re.MULTILINE)
        if m:
            return m.group(1)

        return ""
