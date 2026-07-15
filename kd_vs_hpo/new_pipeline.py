import torch.nn as nn
import torch

from lightning import seed_everything

from kd_vs_hpo.dataloader import (
    build_cifar10_dataloaders,
)

from kd_vs_hpo.flops.tracker import (
    FlopsTracker,
)

from kd_vs_hpo.flops.utils import (
    count_flops_params,
)

from kd_vs_hpo.lightning.module import (
    KDLightningModule,
)

from kd_vs_hpo.lightning.trainer import (
    build_trainer,
)


def run_training_pipeline(

    model,

    config,

    optimizer_cls,

    optimizer_kwargs,

    scheduler_cls,

    scheduler_kwargs,

    teacher_ensemble=None,

    kd_loss=None,

):

    seed_everything(config.seed)

    train_loader, val_loader, test_loader, *_ = (
        build_cifar10_dataloaders(
            config,
            torch.device("cuda"),
        )
    )

    student_flops, _ = (
        count_flops_params(model)
    )

    teacher_flops = 0

    if teacher_ensemble is not None:

        teacher_flops = (
            teacher_ensemble.flops_per_sample
        )

    train_step_flops = int(
        student_flops
        * config.train_step_multiplier
        + teacher_flops
    )

    flops_tracker = FlopsTracker(
        enabled=True,
    )

    module = KDLightningModule(

        model=model,

        criterion=nn.CrossEntropyLoss(),

        optimizer_cls=optimizer_cls,

        optimizer_kwargs=optimizer_kwargs,

        scheduler_cls=scheduler_cls,

        scheduler_kwargs=scheduler_kwargs,

        teacher_ensemble=teacher_ensemble,

        kd_loss=kd_loss,

        flops_tracker=flops_tracker,

        train_step_flops=train_step_flops,

        eval_step_flops=student_flops,
    )

    trainer = build_trainer(
        config,
        config.log_dir,
    )

    trainer.fit(
        module,
        train_loader,
        val_loader,
    )

    trainer.test(
        module,
        test_loader,
    )

    return {

        "best_model_path":
        trainer.checkpoint_callback.best_model_path,

        "cumulative_flops":
        flops_tracker.spent,
    }


# run_training_pipeline(
#     model=student,
#     config=config,
#     optimizer_cls=torch.optim.SGD,
#     optimizer_kwargs={
#         "lr": 0.1,
#         "momentum": 0.9,
#     },
#     scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR,
#     scheduler_kwargs={
#         "T_max": config.max_epochs,
#     },
# )


# teachers = TeacherEnsemble(
#     [teacher1, teacher2],
# )

# kd_loss = KLDivKDLoss(
#     alpha=0.7,
#     temperature=4.0,
# )

# run_training_pipeline(
#     model=student,
#     config=config,
#     teacher_ensemble=teachers,
#     kd_loss=kd_loss,
#     optimizer_cls=torch.optim.SGD,
#     optimizer_kwargs=...,
#     scheduler_cls=...,
#     scheduler_kwargs=...,
# )