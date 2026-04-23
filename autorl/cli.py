"""
autorl CLI entry point.
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_backend(config: dict):
    backend_type = config["backend"]["type"]
    if backend_type == "cloudml":
        from autorl.backends.cloudml import CloudMLBackend
        return CloudMLBackend(config["backend"])
    elif backend_type == "slurm":
        from autorl.backends.slurm import SlurmBackend
        return SlurmBackend(config["backend"])
    elif backend_type == "local":
        from autorl.backends.local import LocalBackend
        return LocalBackend(config["backend"])
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def build_search(config: dict):
    strategy = config["search"].get("strategy", "perturbation")
    if strategy == "perturbation":
        from autorl.search.perturbation import PerturbationStrategy
        return PerturbationStrategy(config)
    elif strategy == "llm":
        from autorl.search.llm import LLMSearchStrategy
        return LLMSearchStrategy(config)
    elif strategy == "both":
        # Use LLM but fall back to perturbation on failure
        from autorl.search.llm import LLMSearchStrategy
        return LLMSearchStrategy(config)
    else:
        raise ValueError(f"Unknown search strategy: {strategy}")


def build_parser(config: dict):
    parser_type = config.get("log_parser", {}).get("type", "verl")
    if parser_type == "verl":
        from autorl.parsers.verl import VerlLogParser
        cfg = config.get("log_parser", {})
        cfg.setdefault("log_dir", config["experiment"].get("log_dir", "logs"))
        cfg.setdefault("exp_name_prefix", config["experiment"].get("exp_name_prefix", "autorl_exp"))
        return VerlLogParser(cfg)
    else:
        raise ValueError(f"Unknown log parser: {parser_type}")


def build_initial_experiments(config: dict) -> list:
    """Build the initial experiment grid from config."""
    from autorl.core.experiment import Experiment

    param_specs = config["parameters"]
    defaults = {k: v["default"] for k, v in param_specs.items()}
    initial_cfg = config.get("initial_experiments", {})
    experiments = []
    exp_id = 1

    # Always include a baseline run
    if initial_cfg.get("include_baseline", True):
        experiments.append(Experiment(
            exp_id=exp_id,
            params=defaults.copy(),
            description="baseline (all defaults)",
            round_num=0,
        ))
        exp_id += 1

    # Single-param sweeps
    sweeps = initial_cfg.get("sweeps", {})
    for param_name, values in sweeps.items():
        for val in values:
            params = defaults.copy()
            params[param_name] = val
            experiments.append(Experiment(
                exp_id=exp_id,
                params=params,
                description=f"{param_name}={val}",
                round_num=0,
            ))
            exp_id += 1

    return experiments


def cmd_init(config: dict, args):
    """Submit initial experiments."""
    from autorl.core.tracker import Tracker
    from autorl.core.runner import Runner

    tracker = Tracker(
        config["experiment"]["results_tsv"],
        list(config["parameters"].keys()),
    )
    backend = build_backend(config)
    search = build_search(config)
    parser = build_parser(config)
    runner = Runner(config, backend, search, parser, tracker)

    experiments = build_initial_experiments(config)
    logger.info(f"Submitting {len(experiments)} initial experiments...")

    if args.dryrun:
        for exp in experiments:
            logger.info(f"  [dryrun] exp{exp.exp_id}: {exp.description} | {exp.params}")
        return

    runner.run_init(experiments)


def cmd_run(config: dict, args):
    """Start the full autonomous loop."""
    from autorl.core.tracker import Tracker
    from autorl.core.runner import Runner

    tracker = Tracker(
        config["experiment"]["results_tsv"],
        list(config["parameters"].keys()),
    )
    backend = build_backend(config)
    search = build_search(config)
    parser = build_parser(config)
    runner = Runner(config, backend, search, parser, tracker)

    experiments = build_initial_experiments(config)
    runner.run_full(experiments)


def cmd_status(config: dict, args):
    """Print current results table."""
    from autorl.core.tracker import Tracker

    tracker = Tracker(
        config["experiment"]["results_tsv"],
        list(config["parameters"].keys()),
    )
    tracker.print_summary(config["experiment"].get("baseline"))


def cmd_collect(config: dict, args):
    """Collect results from completed experiments."""
    from autorl.core.tracker import Tracker
    from autorl.core.runner import Runner

    tracker = Tracker(
        config["experiment"]["results_tsv"],
        list(config["parameters"].keys()),
    )
    backend = build_backend(config)
    search = build_search(config)
    parser = build_parser(config)
    runner = Runner(config, backend, search, parser, tracker)

    # Load experiments from tracker
    rows = tracker.load_all()
    from autorl.core.experiment import Experiment
    experiments = [
        Experiment(exp_id=int(r["exp_id"]), params={}, round_num=int(r.get("round", 0)))
        for r in rows
    ]
    runner.run_collect(experiments)


def main():
    parser = argparse.ArgumentParser(
        prog="autorl",
        description="AutoRL: Universal RL hyperparameter search framework",
    )
    parser.add_argument(
        "--config", "-c",
        default="autorl_config.yaml",
        help="Path to config file (default: autorl_config.yaml)",
    )
    parser.add_argument("--dryrun", action="store_true", help="Generate scripts but don't submit")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("init",    help="Submit initial batch of experiments")
    subparsers.add_parser("run",     help="Start full autonomous search loop")
    subparsers.add_parser("status",  help="Print current results summary")
    subparsers.add_parser("collect", help="Collect results from completed experiments")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    commands = {
        "init":    cmd_init,
        "run":     cmd_run,
        "status":  cmd_status,
        "collect": cmd_collect,
    }
    commands[args.command](config, args)


if __name__ == "__main__":
    main()
