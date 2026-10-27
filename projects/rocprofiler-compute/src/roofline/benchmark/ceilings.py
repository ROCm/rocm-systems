# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Roofline benchmark ceilings, as plain data.

Each benchmarked data type writes one roofline.csv column, which analyze reads as
"$<column>_empirical_peak". The module imports nothing, so tools can read it on a host
without ROCm.
"""

# Data type -> roofline.csv column, per arch family.
FAMILY_CEILINGS: dict[str, dict[str, str]] = {
    "gfx9": {
        "HBM": "HBMBw",
        "MALL": "MALLBw",
        "L2": "L2Bw",
        "L1": "L1Bw",
        "LDS": "LDSBw",
        "F16": "FP16Flops",
        "F32": "FP32Flops",
        "F64": "FP64Flops",
        "I8": "I8Ops",
        "I32": "I32Ops",
        "I64": "I64Ops",
        "MFMA-F4": "MFMAF4Flops",
        "MFMA-F6": "MFMAF6Flops",
        "MFMA-F6F4": "MFMAF6F4Flops",
        "MFMA-MXF8": "MFMAMXF8Flops",
        "MFMA-F8": "MFMAF8Flops",
        "MFMA-F16": "MFMAF16Flops",
        "MFMA-BF16": "MFMABF16Flops",
        "MFMA-F32": "MFMAF32Flops",
        "MFMA-F64": "MFMAF64Flops",
        "MFMA-I8": "MFMAI8Ops",
    },
    "gfx11": {
        "HBM": "HBMBw",
        "MALL": "MALLBw",
        "L2": "L2Bw",
        "L1": "L1Bw",
        "L0": "L0Bw",
        "LDS": "LDSBw",
        "F16": "FP16Flops",
        "BF16": "BF16Flops",
        "F32": "FP32Flops",
        "F64": "FP64Flops",
        "I8": "I8Ops",
        "I32": "I32Ops",
        "I64": "I64Ops",
        "WMMA-F4": "WMMAF4Flops",
        "WMMA-F6": "WMMAF6Flops",
        "WMMA-F6F4": "WMMAF6F4Flops",
        "WMMA-MXF8": "WMMAMXF8Flops",
        "WMMA-F8": "WMMAF8Flops",
        "WMMA-F16": "WMMAF16Flops",
        "WMMA-BF16": "WMMABF16Flops",
        "WMMA-F32": "WMMAF32Flops",
        "WMMA-F64": "WMMAF64Flops",
        "WMMA-I8": "WMMAI8Ops",
    },
    "gfx12": {
        "HBM": "HBMBw",
        "MALL": "MALLBw",
        "L2": "L2Bw",
        "L1": "L1Bw",
        "L0": "L0Bw",
        "LDS": "LDSBw",
        "F16": "FP16Flops",
        "F32": "FP32Flops",
        "F64": "FP64Flops",
        "I8": "I8Ops",
        "I32": "I32Ops",
        "I64": "I64Ops",
        "WMMA-F4": "WMMAF4Flops",
        "WMMA-F6": "WMMAF6Flops",
        "WMMA-F6F4": "WMMAF6F4Flops",
        "WMMA-MXF8": "WMMAMXF8Flops",
        "WMMA-F8": "WMMAF8Flops",
        "WMMA-F16": "WMMAF16Flops",
        "WMMA-BF16": "WMMABF16Flops",
        "WMMA-F32": "WMMAF32Flops",
        "WMMA-F64": "WMMAF64Flops",
        "WMMA-I8": "WMMAI8Ops",
    },
}

# Every roofline.csv column any benchmark writes.
ALL_CEILINGS = {
    column for family in FAMILY_CEILINGS.values() for column in family.values()
}

_GFX11_UNSUPPORTED = (
    "HBM",
    "WMMA-F4",
    "WMMA-F6",
    "WMMA-F6F4",
    "WMMA-MXF8",
    "WMMA-F8",
    "WMMA-F16",
    "WMMA-BF16",
    "WMMA-F32",
    "WMMA-F64",
    "WMMA-I8",
)

# Data types each device's benchmark skips, for every device with a benchmark class.
# Tuples, so a benchmark instance cannot change the shared data.
UNSUPPORTED_DATA_TYPES: dict[str, tuple[str, ...]] = {
    "gfx90a": ("MALL", "MFMA-F4", "MFMA-F6", "MFMA-F6F4", "MFMA-MXF8", "MFMA-F8"),
    "gfx942": ("MFMA-F4", "MFMA-F6", "MFMA-F6F4", "MFMA-MXF8"),
    "gfx950": (),
    "gfx1150": (*_GFX11_UNSUPPORTED, "MALL"),
    "gfx1151": _GFX11_UNSUPPORTED,
    "gfx1152": (*_GFX11_UNSUPPORTED, "MALL"),
    "gfx1153": (*_GFX11_UNSUPPORTED, "MALL"),
    "gfx1250": ("L1", "MALL", "WMMA-F64"),
}

# gfx940 (MI300A_A0) and gfx941 (MI300X_A0) use the gfx942 (MI300_A1) benchmark.
BENCH_DEVICE_ALIASES = {"gfx940": "gfx942", "gfx941": "gfx942"}


def measured_ceilings(gpu_arch: str) -> set[str]:
    """Return the roofline.csv columns the benchmark measures on gpu_arch.

    An arch without a benchmark class, such as gfx908, measures none.
    """
    device = BENCH_DEVICE_ALIASES.get(gpu_arch, gpu_arch)
    if device not in UNSUPPORTED_DATA_TYPES:
        return set()
    return {
        column
        for data_type, column in FAMILY_CEILINGS[device[:-2]].items()
        if data_type not in UNSUPPORTED_DATA_TYPES[device]
    }
