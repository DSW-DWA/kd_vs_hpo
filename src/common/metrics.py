from pathlib import Path

import lightning as L
import pandas as pd


class MetricsHistoryCallback(L.Callback):

    def __init__(
        self,
        save_dir: str | Path,
    ):
        self.save_dir = Path(save_dir)
        self.history = []

    def on_train_epoch_end(
        self,
        trainer,
        pl_module,
    ):
        metrics = trainer.callback_metrics
        row = {
            "epoch": trainer.current_epoch,
            "train_loss": metrics["train_loss"].item(),
            "train_acc": metrics["train_acc"].item(),
            "lr": trainer.optimizers[0].param_groups[0]["lr"],
            "train_flops": pl_module.train_epoch_flops,
            "val_flops": pl_module.eval_epoch_flops,
            "cumulative_flops": pl_module.flops_tracker.spent if pl_module.flops_tracker else 0,
        }

        self.history.append(row)


    def on_validation_epoch_end(
        self,
        trainer,
        pl_module,
    ):
        metrics = trainer.callback_metrics
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else None
        row = {
            "epoch": trainer.current_epoch,
            "val_loss": metrics["val_loss"].item(),
            "val_acc": metrics["val_acc"].item(),
            "lr": lr,
            "train_flops": pl_module.train_epoch_flops,
            "val_flops": pl_module.eval_epoch_flops,
            "cumulative_flops": pl_module.flops_tracker.spent if pl_module.flops_tracker else 0,
        }

        self.history.append(row)

    def on_fit_end(
        self,
        trainer,
        pl_module,
    ):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.history).to_csv(self.save_dir / "metrics.csv", index=False)