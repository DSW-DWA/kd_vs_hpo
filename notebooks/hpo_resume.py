"""Resume a noise experiment from its saved recovery files."""

import argparse
import logging

from hydra import compose, initialize_config_dir

from hpo_optuna import PROJECT_ROOT, build_experiment
from kd_vs_hpo.hpo.pipeline import run_hpo_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Root config name, e.g. hpo_11570_noise10_tpe_50")
    args = parser.parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "conf")):
        cfg = compose(config_name=args.config, overrides=["hpo.resume=true"])
    logging.basicConfig(level=logging.INFO)
    result = run_hpo_experiment(build_experiment(cfg))
    print(result.studies.to_string(index=False))
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
