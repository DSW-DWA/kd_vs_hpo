import lightning as L
import torch
import torch.nn as nn

from kd_vs_hpo.utils import (
    extract_logits,
    accuracy_top1_from_logits,
)
from kd_vs_hpo.flops.tracker import FlopsTracker
from lightning import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

from kd_vs_hpo.metrics import (
    MetricsHistoryCallback,
)

class KDLightningModule(L.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer_cls,
        optimizer_kwargs: dict,
        scheduler_cls=None,
        scheduler_kwargs=None,
        teacher_ensemble=None,
        kd_loss=None,
        flops_tracker: FlopsTracker | None = None,
        train_step_flops: int | None = None,
        eval_step_flops: int | None = None,
    ):
        super().__init__()

        self.model = model
        self.criterion = criterion

        self.teacher_ensemble = teacher_ensemble
        self.kd_loss = kd_loss

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs

        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.flops_tracker = flops_tracker
        self.train_step_flops = train_step_flops
        self.eval_step_flops = eval_step_flops

        self.train_epoch_flops = 0
        self.val_epoch_flops = 0
        self.test_epoch_flops = 0


    def forward(self, x):

        return self.model(x)


    def compute_loss(
        self,
        images,
        targets,
    ):

        logits = extract_logits(
            self.model(images)
        )

        hard_loss = self.criterion(
            logits,
            targets,
        )

        if (
            self.teacher_ensemble is None
            or self.kd_loss is None
        ):
            return hard_loss, logits

        teacher_logits = (
            self.teacher_ensemble.logits(
                images
            )
        )

        kd_loss = self.kd_loss(
            student_logits=logits,
            teacher_logits=teacher_logits,
        )

        total_loss = (
            self.kd_loss.alpha * hard_loss
            + (1.0 - self.kd_loss.alpha) * kd_loss
        )

        return total_loss, logits


    def _track_flops(
        self,
        batch_size,
        stage,
    ):

        if stage == "train":
            flops = batch_size * self.train_step_flops
            self.train_epoch_flops += flops

        elif stage == "val":
            flops = batch_size * self.eval_step_flops
            self.val_epoch_flops += flops

        else:
            flops = batch_size * self.eval_step_flops
            self.test_epoch_flops += flops

        if self.flops_tracker is not None:
            self.flops_tracker.add(flops)


    def training_step(
        self,
        batch,
        batch_idx,
    ):

        images, targets = batch

        loss, logits = self.compute_loss(
            images,
            targets,
        )

        acc = accuracy_top1_from_logits(
            logits.detach(),
            targets,
        )

        self._track_flops(
            batch_size=targets.size(0),
            stage="train",
        )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
        )

        self.log(
            "train_acc",
            acc,
            prog_bar=True,
            on_epoch=True,
        )

        return loss


    def validation_step(
        self,
        batch,
        batch_idx,
    ):

        images, targets = batch

        loss, logits = self.compute_loss(
            images,
            targets,
        )

        acc = accuracy_top1_from_logits(
            logits,
            targets,
        )

        self._track_flops(
            targets.size(0),
            "val",
        )

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
        )

        self.log(
            "val_acc",
            acc,
            prog_bar=True,
            on_epoch=True,
        )


    def test_step(
        self,
        batch,
        batch_idx,
    ):

        images, targets = batch

        loss, logits = self.compute_loss(
            images,
            targets,
        )

        acc = accuracy_top1_from_logits(
            logits,
            targets,
        )

        self._track_flops(
            targets.size(0),
            "test",
        )

        self.log(
            "test_loss",
            loss,
            on_epoch=True,
        )

        self.log(
            "test_acc",
            acc,
            on_epoch=True,
        )


    def on_train_epoch_start(self):

        self.train_epoch_flops = 0

    def on_validation_epoch_start(self):

        self.val_epoch_flops = 0

    def on_test_epoch_start(self):

        self.test_epoch_flops = 0


    def configure_optimizers(self):

        optimizer = self.optimizer_cls(
            self.parameters(),
            **self.optimizer_kwargs,
        )

        if self.scheduler_cls is None:
            return optimizer

        scheduler = self.scheduler_cls(
            optimizer,
            **self.scheduler_kwargs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
    


def build_trainer(
    config,
    log_dir,
):

    callbacks = [

        LearningRateMonitor(),

        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=1,
        ),

        MetricsHistoryCallback(
            save_dir=log_dir,
        ),
    ]

    trainer = Trainer(

        max_epochs=config.max_epochs,

        accelerator="auto",

        precision=(
            "16-mixed"
            if config.amp
            else "32"
        ),

        deterministic=config.deterministic,

        gradient_clip_val=(
            config.grad_clip_norm
        ),

        callbacks=callbacks,

        log_every_n_steps=20,
    )

    return trainer