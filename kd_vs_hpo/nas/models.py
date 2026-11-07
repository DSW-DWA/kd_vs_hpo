from archai.discrete_search.api.archai_model import ArchaiModel

from kd_vs_hpo.nas.constants import NATSOperations, ops_to_arch_index


class NATSArchaiModel(ArchaiModel):
    """Lightweight wrapper: stores operations list and the NATS index."""

    def __init__(
        self, api, operations: list[NATSOperations], arch_id: int | None = None
    ):
        self.api = api
        self.operations = list(operations)
        if arch_id is None:
            arch_id = ops_to_arch_index(api, operations)
        super().__init__(arch=None, archid=str(arch_id))
        self.nats_index = arch_id

    def copy(self) -> "NATSArchaiModel":
        return NATSArchaiModel(self.api, list(self.operations), self.nats_index)