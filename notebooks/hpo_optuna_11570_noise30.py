"""Run 50 TPE trials for architecture 11570 with 30% label noise."""

import hydra
from omegaconf import DictConfig

from hpo_optuna import main as run_hpo


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="hpo_11570_noise30_tpe_50",
)
def main(cfg: DictConfig) -> None:
    run_hpo(cfg)


if __name__ == "__main__":
    main()
