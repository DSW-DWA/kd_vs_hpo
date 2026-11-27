from enum import StrEnum


class NATSOperations(StrEnum):
    NONE = "none"
    SKIP = "skip_connect"
    CONV1X1 = "nor_conv_1x1"
    CONV3X3 = "nor_conv_3x3"
    AVG_POOL_3X3 = "avg_pool_3x3"


def get_arch_by_operations(operations: list[NATSOperations]):
    assert len(operations) == 6
    return f"|{operations[0]}~0|+|{operations[1]}~0|{operations[2]}~1|+|{operations[3]}~0|{operations[4]}~1|{operations[5]}~2|"


def ops_to_arch_index(api, operations: list[NATSOperations]) -> int:
    """Convert an operation list to a NATS-Bench architecture index."""
    arch_str = get_arch_by_operations(operations)
    return api.query_index_by_arch(arch_str)


def get_operations_by_arch(arch: str) -> list[NATSOperations]:
    ops = []
    for token in arch.split("|"):
        token = token.strip()
        if "~" in token:
            op_name = token.split("~")[0]
            ops.append(NATSOperations(op_name))
    return ops