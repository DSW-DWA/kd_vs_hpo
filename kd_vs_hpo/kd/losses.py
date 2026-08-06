from abc import ABC, abstractmethod

import torch.nn.functional as F
from torch import nn


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


class KullbackLeiblerKDLoss(AbstractKDLoss): 

    def forward(
        self,
        student_logits,
        teacher_logits,
    ):
        
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        return F.kl_div(student_log_probs, teacher_logits, reduction="batchmean") * (self.temperature ** 2)


class MSELogitKDLoss(AbstractKDLoss):

    def forward(
        self,
        student_logits,
        teacher_logits,
    ):
        return F.mse_loss(student_logits, teacher_logits, reduction="mean")
    

class MSEProbKDLoss(AbstractKDLoss):

    def forward(
        self,
        student_logits,
        teacher_logits,
    ):
        student_probs = F.softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        return F.mse_loss(student_probs, teacher_probs, reduction="mean")
