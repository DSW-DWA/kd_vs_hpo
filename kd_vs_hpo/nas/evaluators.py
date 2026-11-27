import numpy as np
from archai.discrete_search.api import (
    ArchaiModel,
    ModelEvaluator,
)
from overrides import overrides


class PrecomputedArrayEvaluator(ModelEvaluator):
    def __init__(self, values: np.ndarray, name: str):
        self.values = np.asarray(values)
        self._name = name

    @overrides
    def evaluate(self, arch: ArchaiModel, budget=None) -> float:
        idx = (
            int(arch.metadata["idx"])
            if arch.metadata and "idx" in arch.metadata
            else int(arch.archid)
        )
        return float(self.values[idx])