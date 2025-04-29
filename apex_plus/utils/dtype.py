import enum


class DTYPE(enum.Enum):
    FLOAT32 = 0
    FLOAT16 = 1
    FLOAT8 = 2
    BFLOAT16 = 3


# FIXME(woosuk): This is hacky.
DTYPE.FLOAT32.size = 4
DTYPE.FLOAT16.size = 2
DTYPE.FLOAT8.size = 1
DTYPE.BFLOAT16.size = 2

_DTYPE_REGISTRY = {
    # Float32
    "float32": DTYPE.FLOAT32,
    "float": DTYPE.FLOAT32,
    # Float16
    "float16": DTYPE.FLOAT16,
    "half": DTYPE.FLOAT16,
    # Float8
    "float8": DTYPE.FLOAT8,
    # BFloat16
    "bfloat16": DTYPE.BFLOAT16,
}


def get_dtype(dtype: str) -> DTYPE:
    if dtype.lower() not in _DTYPE_REGISTRY:
        raise ValueError(f"Unknown dtype {dtype}")
    return _DTYPE_REGISTRY[dtype.lower()]


def dtype_to_str(dtype: DTYPE) -> str:
    if dtype == DTYPE.FLOAT32:
        return "float"
    elif dtype == DTYPE.FLOAT16:
        return "half"
    elif dtype == DTYPE.FLOAT8:
        return "float8"
    elif dtype == DTYPE.BFLOAT16:
        return "bfloat16"
    else:
        raise ValueError(f"Unknown dtype {dtype}")
