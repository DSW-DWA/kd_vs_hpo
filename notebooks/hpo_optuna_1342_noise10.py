"""Run 50 TPE trials for architecture 1342 with 10% label noise."""

import hydra
from omegaconf import DictConfig

from hpo_optuna import main as run_hpo


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="hpo_1342_noise10_tpe_50",
)
def main(cfg: DictConfig) -> None:
    run_hpo(cfg)


if __name__ == "__main__":
    main()
