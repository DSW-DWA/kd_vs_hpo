import lightning as L
import torch
import torchmetrics
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn

from kd_vs_hpo.common.flops import FlopsBudgetTracker
from kd_vs_hpo.common.metrics import (
    MetricsHistoryCallback,
)
from kd_vs_hpo.common.utils import (
    extract_logits,
    resolve_dir,
)
from kd_vs_hpo.kd.teacher import TeacherEnsemble


class StopAfterEpochCallback(L.Callback):
    def __init__(self, epoch: int):
        self.epoch = epoch

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.current_epoch >= self.epoch: 
            print(f"Stopping training gracefully at the end of epoch {trainer.current_epoch}")
            trainer.should_stop = True



class KDLightningModule(L.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer_cls: torch.optim.Optimizer,
        optimizer_kwargs: dict,
        train_step_flops: int,
        eval_step_flops: int,
        num_classes: int = 10,
        scheduler_cls = None,
        scheduler_kwargs = None,
        teacher_ensemble: TeacherEnsemble | None = None,
        kd_loss: nn.Module | None = None,
        flops_tracker: FlopsBudgetTracker | None = None,
    ):
        super().__init__()
        # self.save_hyperparameters()

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
        self.eval_epoch_flops = 0
        self.test_epoch_flops = 0
        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )


    def forward(self, x):
        return self.model(x)


    def compute_loss(self, images, targets):
        logits = extract_logits(self.model(images))
        hard_loss = self.criterion(logits, targets)
        if self.teacher_ensemble is None or self.kd_loss is None:
            return hard_loss, logits
        teacher_logits = self.teacher_ensemble.logits(images)
        kd_loss = self.kd_loss(
            student_logits=logits,
            teacher_logits=teacher_logits,
        )
        total_loss = (
            self.kd_loss.alpha * hard_loss
            + (1.0 - self.kd_loss.alpha) * kd_loss
        )
        return total_loss, logits


    def _track_flops(self, batch_size, stage):
        if stage == "train":
            flops = batch_size * self.train_step_flops
            self.train_epoch_flops += flops
        elif stage == "val":
            flops = batch_size * self.eval_step_flops
            self.eval_epoch_flops += flops
        else:
            flops = batch_size * self.eval_step_flops
            self.test_epoch_flops += flops

        if self.flops_tracker is not None:
            self.flops_tracker.spend(flops)


    def training_step(self, batch, batch_idx):
        images, targets = batch

        loss, logits = self.compute_loss(images, targets)
        self.train_acc(logits, targets)
        self._track_flops(batch_size=targets.size(0), stage="train")

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("flops", self.flops_tracker.spent, prog_bar=True, on_epoch=True, on_step=False)

        return loss


    def validation_step(self, batch, batch_idx):
        images, targets = batch

        loss, logits = self.compute_loss(images, targets)
        self.val_acc(logits, targets)
        self._track_flops(targets.size(0), "val")

        self.log("val_loss",loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        images, targets = batch

        loss, logits = self.compute_loss(images, targets)
        self.test_acc(logits, targets)
        self._track_flops(targets.size(0), "test")

        self.log("test_loss", loss, on_epoch=True, on_step=False)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)


    def on_train_epoch_start(self):
        self.train_epoch_flops = 0

    def on_validation_epoch_start(self):
        self.eval_epoch_flops = 0

    def on_test_epoch_start(self):
        self.test_epoch_flops = 0


    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)
        if self.scheduler_cls is None:
            return optimizer
        
        scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


def build_trainer(
        run_name: str,
        checkpoint_dir: str,
        log_dir: str,
        max_epochs: int,
        deterministic: bool,
        amp: bool,
        grad_clip_norm: float,
        ):
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            filename=run_name + "epoch={epoch}-val_acc={val_acc:.4f}",
            dirpath=resolve_dir(f"{checkpoint_dir}/{run_name}"),
        ),
        MetricsHistoryCallback(
            save_dir=resolve_dir(f"{checkpoint_dir}/{run_name}"),
        ),
        # EarlyStopping(
        #     monitor="val_loss",
        #     patience=25,
        #     mode="min"
        #     ),
        # StopAfterEpochCallback(10)
    ]
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        precision=("16-mixed" if amp else "32"
        ),
        deterministic=deterministic,
        gradient_clip_val=grad_clip_norm,
        callbacks=callbacks,
        logger=TensorBoardLogger(
            save_dir=log_dir,
            name=run_name,
            ),
        devices=1,
        strategy="auto",
        accumulate_grad_batches=1,
    )
    return trainer