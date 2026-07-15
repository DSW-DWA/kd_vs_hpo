import torch.nn as nn
import torch.nn.functional as F
import torch

from kd_vs_hpo.utils import extract_logits


class TeacherEnsemble(nn.Module):

    def __init__(
        self,
        teachers: list[nn.Module],
        weights=None,
    ):
        super().__init__()

        self.teachers = nn.ModuleList(teachers)

        self.register_buffer(
            "weights",
            self._normalize_weights(weights)
        )

        self.freeze()

    @property
    def has_teachers(self):
        return len(self.teachers) > 0


    def _normalize_weights(self, weights: list[float] | None) -> list[float]:
        if weights is None:
            return torch.ones(len(self.teachers))

        return torch.tensor(weights) / torch.tensor(weights).sum()

    def freeze(self):
        for teacher in self.teachers:
            teacher.eval()

            for p in teacher.parameters():
                p.requires_grad_(False)

    @torch.inference_mode()
    def logits(self, images):
        ensemble = None

        for teacher, w in zip(self.teachers, self.weights):
            logits = extract_logits(teacher(images))

            if ensemble is None:
                ensemble = logits.mul(w)
            else:
                ensemble.add_(logits, alpha=w)

        return ensemble

    @torch.inference_mode()
    def probs(self, images, temperature=1.0):
        logits = self.logits(images)

        return F.softmax(
            logits / temperature,
            dim=-1,
        )