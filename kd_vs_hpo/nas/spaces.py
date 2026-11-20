from __future__ import annotations

import json

import numpy as np
from archai.discrete_search.api import ArchaiModel, EvolutionarySearchSpace
from overrides import overrides

from kd_vs_hpo.nas.constants import (
    NATSOperations,
    get_arch_by_operations,
    get_operations_by_arch,
)
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
        payload = {
            "archid": model.archid,
            "nats_index": int(model.nats_index),
            "operations": [str(op) for op in model.operations],
            "arch_str": get_arch_by_operations(model.operations),
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file)

    @overrides
    def load_arch(self, path: str):
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)

        arch_id = payload.get(
            "nats_index", payload.get("arch_index", payload.get("archid"))
        )
        if arch_id is None:
            raise ValueError(f"Architecture file {path} does not contain an index")
        arch_id = int(arch_id)

        if payload.get("operations") is not None:
            operations = [NATSOperations(op) for op in payload["operations"]]
        elif payload.get("arch_str"):
            operations = get_operations_by_arch(payload["arch_str"])
        else:
            operations = get_operations_by_arch(self.api.arch(arch_id))

        return NATSArchaiModel(self.api, operations, arch_id=arch_id)

    @overrides
    def save_model_weights(self, model: ArchaiModel, path: str):
        pass

    @overrides
    def load_model_weights(self, model: ArchaiModel, path: str):
        pass