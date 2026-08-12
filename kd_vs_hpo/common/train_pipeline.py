import logging
from pathlib import Path

import hydra
import torch
from lightning import seed_everything
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from xautodl.models import get_cell_based_tiny_net

from kd_vs_hpo.common.dataloader import build_cifar10_dataloaders
from kd_vs_hpo.common.flops import FlopsBudgetTracker, count_flops_params
from kd_vs_hpo.common.train_modules import KDLightningModule, build_trainer
from kd_vs_hpo.common.utils import get_architectures_from_json
from kd_vs_hpo.kd.teacher import TeacherEnsemble

logger = logging.getLogger(__name__)


DEFAULT_SCHEDULER_CLS = CosineAnnealingLR
DEFAULT_OPTIMIZER_CLS = torch.optim.SGD


def get_student_from_config(cfg: DictConfig, architectures):
    arch = next((a for a in architectures if a['arch_index'] == cfg.student), None)
    if arch is None: 
        raise ValueError(f"Architecture with index {cfg.student} not found.")
    model = get_cell_based_tiny_net(arch)
    return model, arch


def get_teachers_from_config(cfg: DictConfig, architectures):
    models = []
    arches = []
    if cfg.teachers_mapping is None:
        return None, arches
    for idx, path in cfg.teachers_mapping:
        arch = next((a for a in architectures if a['arch_index'] == idx), None)
        model = get_cell_based_tiny_net(arch)
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        models.append(model)
        arches.append(arch)
    return TeacherEnsemble(models), arches



def run_training_pipeline(
        model: nn.Module,
        criterion: nn.Module,
        train_step_flops: int,
        eval_step_flops: int,
        run_name: str,
        checkpoint_dir: Path,
        log_dir: Path,
        data_root: Path,
        max_epochs: int,
        deterministic: bool,
        amp: bool,
        grad_clip_norm: float,
        seed: int,
        batch_size: int,
        num_workers: int,
        validation_fraction: float,
        device: torch.device,
        optimizer_kwargs: dict,
        optimizer_cls: torch.optim.Optimizer = DEFAULT_OPTIMIZER_CLS,
        scheduler_cls: torch.optim.lr_scheduler._LRScheduler | None = DEFAULT_SCHEDULER_CLS,
        scheduler_kwargs: dict | None = None,
        teacher_ensemble: TeacherEnsemble | None = None,
        kd_loss: nn.Module | None = None,
        flops_tracker: FlopsBudgetTracker | None = None,
        num_classes: int = 10,
):  

    train_loader, val_loader, test_loader, *_ = build_cifar10_dataloaders(
        checkpoint_dir, log_dir, data_root, seed, batch_size, num_workers, validation_fraction, device
    )

    module = KDLightningModule(
        model=model,
        criterion=criterion,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        train_step_flops=train_step_flops,
        eval_step_flops=eval_step_flops,
        num_classes=num_classes,
        scheduler_cls=scheduler_cls,
        scheduler_kwargs=scheduler_kwargs,
        teacher_ensemble=teacher_ensemble,
        kd_loss=kd_loss,
        flops_tracker=flops_tracker
    )

    trainer = build_trainer(
        run_name,
        checkpoint_dir,
        log_dir,
        max_epochs,
        deterministic,
        amp,
        grad_clip_norm,
    )
    trainer.fit(module, train_loader, val_loader)
    trainer.test(module, test_loader)


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="config",
)
def main(cfg: DictConfig):

    general_cfg = cfg.general
    kd_cfg = cfg.kd
    seed_everything(general_cfg.seed)

    data_root = Path(general_cfg.data_root)
    checkpoint_dir = Path(kd_cfg.checkpoint_dir)
    log_dir = Path(kd_cfg.log_dir)

    data_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    architectires = get_architectures_from_json(general_cfg.architectures_path)
    model, student_arch = get_student_from_config(kd_cfg, architectires)

    teacher_ensemble, teacher_arches = get_teachers_from_config(kd_cfg, architectires)

    run_name = kd_cfg.run_name + f"arch_{student_arch['arch_index']}_teachers_{'_'.join(str(t['arch_index']) for t in teacher_arches)}_seed_{general_cfg.seed}"

    student_flops, _ = count_flops_params(model)
    teacher_flops = sum(
            count_flops_params(teacher)[0]
            for teacher in teacher_ensemble.teachers
        ) if teacher_ensemble is not None else 0
    train_step_flops = int(student_flops * general_cfg.train_step_multiplier + teacher_flops)
    eval_step_flops = 0

    run_training_pipeline(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer_kwargs=kd_cfg.optimizer_params,
        train_step_flops=train_step_flops,
        eval_step_flops=eval_step_flops,
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        data_root=data_root,
        max_epochs=general_cfg.max_epochs,
        deterministic=general_cfg.deterministic,
        amp=general_cfg.amp,
        grad_clip_norm=general_cfg.grad_clip_norm,
        seed=general_cfg.seed,
        batch_size=general_cfg.batch_size,
        num_workers=general_cfg.num_workers,
        validation_fraction=general_cfg.validation_fraction,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        scheduler_kwargs=kd_cfg.scheduler_params,
        teacher_ensemble=teacher_ensemble,
        kd_loss=kd_cfg.kd_loss,
        flops_tracker=FlopsBudgetTracker(kd_cfg.flops_budget, kd_cfg.flops_counter_mode),
        num_classes=10,
    )


if __name__ == "__main__":
    main()
