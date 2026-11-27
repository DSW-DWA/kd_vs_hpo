from __future__ import annotations

import numpy as np
from archai.discrete_search.api import ArchaiModel, EvolutionarySearchSpace
from overrides import overrides

from kd_vs_hpo.nas.constants import NATSOperations
from kd_vs_hpo.nas.models import NATSArchaiModel


class NATSEvolutionarySearchSpace(EvolutionarySearchSpace):
    """NATS evolutionary search space."""

    def __init__(self, api, seed: int = 0) -> None:
        super().__init__()
        self.api = api
        self.rng = np.random.default_rng(seed)
        self.operations = list(NATSOperations)
        self.num_edges = 6

    @overrides
    def random_sample(self) -> NATSArchaiModel:
        ops = self.rng.choice(self.operations, size=self.num_edges).tolist()
        return NATSArchaiModel(self.api, ops)

    @overrides
    def mutate(self, model: NATSArchaiModel) -> NATSArchaiModel:  # type: ignore[override]
        child_ops = list(model.operations)
        edge = self.rng.integers(self.num_edges)
        available = [op for op in self.operations if op != child_ops[edge]]
        child_ops[edge] = self.rng.choice(available)
        return NATSArchaiModel(self.api, child_ops)

    @overrides
    def crossover(  # type: ignore[override]
        self, models: list[NATSArchaiModel]
    ) -> NATSArchaiModel:
        assert len(models) >= 2, "crossover requires at least two parents"
        parent_a, parent_b = models[0], models[1]
        child_ops = [
            self.rng.choice([parent_a.operations[i], parent_b.operations[i]])
            for i in range(self.num_edges)
        ]
        return NATSArchaiModel(self.api, child_ops)

    @overrides
    def save_arch(self, model: ArchaiModel, path: str):
        pass

    @overrides
    def load_arch(self, path: str):
        pass

    @overrides
    def save_model_weights(self, model: ArchaiModel, path: str):
        pass

    @overrides
    def load_model_weights(self, model: ArchaiModel, path: str):
        pass