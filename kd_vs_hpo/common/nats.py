from typing import Any

import torch.nn as nn
from xautodl.config_utils import dict2config
from xautodl.models import get_cell_based_tiny_net


def create_nats_model(arch_record: dict[str, Any]) -> nn.Module:
    config = dict2config(
        {
            "name": arch_record.get("name", "infer.tiny"),
            "C": int(arch_record.get("C", 16)),
            "N": int(arch_record.get("N", 5)),
            "arch_str": arch_record["arch_str"],
            "num_classes": int(arch_record.get("num_classes", 10)),
        },
        None,
    )
    return get_cell_based_tiny_net(config)
