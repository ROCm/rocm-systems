# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_base_gfx9.py
#
# Benchmarking base class for all gfx9-based products.
#
# -----------------------------------------------------------------------------


from . import benchmark_gfx9_base


class Bench_gfx950(benchmark_gfx9_base.Bench_gfx9):
    def __init__(self, device_ids: list) -> None:
        super().__init__(device_ids)

        self.lds_sizes = {
            "gfx908": 64 * 1024,
            "gfx90a": 64 * 1024,
            "gfx940": 64 * 1024,
            "gfx941": 64 * 1024,
            "gfx942": 64 * 1024,
            "gfx950": 64 * 1024,
        }

        self.unsupported_data_types = {
            # MI100 series
            "gfx908": [
                "MALL",
                "MFMA-F4",
                "MFMA-F6",
                "MFMA-F6F4",
                "MFMA-F8",
                "MFMA-F16",
                "MFMA-BF16",
                "MFMA-F64",
                "MFMA-I8",
            ],
            # MI200 series
            "gfx90a": ["MALL", "MFMA-F4", "MFMA-F6", "MFMA-F6F4", "MFMA-F8"],
            # MI300A_A0
            "gfx940": ["MFMA-F4", "MFMA-F6", "MFMA-F6F4"],
            # MI300X_A0
            "gfx941": ["MFMA-F4", "MFMA-F6", "MFMA-F6F4"],
            # MI300A_A1, MI300X_A1, MI308
            "gfx942": ["MFMA-F4", "MFMA-F6", "MFMA-F6F4"],
            # MI350, MI355
            "gfx950": [],
        }

        self.cache_kernel_selector = {
            "L1": {
                "gfx908": "Cache_bw<float, 16 * 1024, 256>",
                "gfx90a": "Cache_bw<float, 16 * 1024, 256>",
                "gfx940": "Cache_bw<float, 32 * 1024, 256>",
                "gfx941": "Cache_bw<float, 32 * 1024, 256>",
                "gfx942": "Cache_bw<float, 32 * 1024, 256>",
                "gfx950": "Cache_bw<float, 32 * 1024, 256>",
            },
            "L2": {
                "gfx908": "Cache_bw<float, 8 * 1024 * 1024, 256>",
                "gfx90a": "Cache_bw<float, 8 * 1024 * 1024, 256>",
                "gfx940": "Cache_bw<float, 4 * 1024 * 1024, 256>",
                "gfx941": "Cache_bw<float, 4 * 1024 * 1024, 256>",
                "gfx942": "Cache_bw<float, 4 * 1024 * 1024, 256>",
                "gfx950": "Cache_bw<float, 4 * 1024 * 1024, 256>",
            },
            "MALL": {
                "gfx940": "Cache_bw<float, 64 * 1024 * 1024, 256>",
                "gfx941": "Cache_bw<float, 64 * 1024 * 1024, 256>",
                "gfx942": "Cache_bw<float, 64 * 1024 * 1024, 256>",
                "gfx950": "Cache_bw<float, 64 * 1024 * 1024, 256>",
            },
        }

        self.matrix_ops = {
            "F4": {"gfx950": 131072},
            "F6": {"gfx950": 131072},
            "F6F4": {"gfx950": 131072},  # Mixed precision F6 x F4
            "F8": dict.fromkeys(
                ["gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"], 32768
            ),
            "F16": dict.fromkeys(["gfx90a", "gfx940", "gfx941", "gfx942"], 16384)
            | dict.fromkeys(["gfx950"], 32768),
            "F32": dict.fromkeys(
                ["gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"], 4096
            ),
            "BF16": dict.fromkeys(["gfx940", "gfx941", "gfx942"], 16384)
            | dict.fromkeys(["gfx90a"], 8192)
            | dict.fromkeys(["gfx950"], 32768),
            "I8": dict.fromkeys(["gfx940", "gfx941", "gfx942"], 32768)
            | dict.fromkeys(["gfx90a"], 16384)
            | dict.fromkeys(["gfx950"], 65536),
            "F64": dict.fromkeys(
                ["gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"], 2048
            ),
        }

        self.cache_sizes = {
            "L1": {
                "gfx908": 16 * 1024,
                "gfx90a": 16 * 1024,
                "gfx940": 32 * 1024,
                "gfx941": 32 * 1024,
                "gfx942": 32 * 1024,
                "gfx950": 32 * 1024,
            },
            "L2": {
                "gfx908": 8 * 1024 * 1024,
                "gfx90a": 8 * 1024 * 1024,
                "gfx940": 4 * 1024 * 1024,
                "gfx941": 4 * 1024 * 1024,
                "gfx942": 4 * 1024 * 1024,
                "gfx950": 4 * 1024 * 1024,
            },
            "MALL": {
                "gfx940": 64 * 1024 * 1024,
                "gfx941": 64 * 1024 * 1024,
                "gfx942": 64 * 1024 * 1024,
                "gfx950": 64 * 1024 * 1024,
            },
        }
