from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from kd_vs_hpo.utils import extract_logits


class AbstractKDLoss(nn.Module, ABC):

    def __init__(
        self,
        alpha=0.5,
        temperature=4.0,
    ):
        super().__init__()

        self.alpha = alpha
        self.temperature = temperature


    @abstractmethod
    def forward(
        self,
        student_logits,
        teacher_logits,
    ):
        pass