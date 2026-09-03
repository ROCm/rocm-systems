#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the pinned vfio-user guest rocBLAS SGEMM inventory."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

TOOLS = Path(__file__).parents[2] / "tools"
SCRIPT = TOOLS / "vfio_guest_sgemm_inventory.py"
SPEC = importlib.util.spec_from_file_location("vfio_guest_sgemm_inventory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
inventory_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory_tool)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def pack(value):
    if value is None:
        return b"\xc0"
    if value is False:
        return b"\xc2"
    if value is True:
        return b"\xc3"
    if isinstance(value, int):
        if 0 <= value <= 0x7F:
            return bytes([value])
        if 0 <= value <= 0xFFFFFFFF:
            return b"\xce" + value.to_bytes(4, "big")
        raise ValueError(value)
    if isinstance(value, float):
        return b"\xcb" + struct.pack(">d", value)
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        if len(encoded) <= 31:
            return bytes([0xA0 + len(encoded)]) + encoded
        return b"\xda" + len(encoded).to_bytes(2, "big") + encoded
    if isinstance(value, list):
        if len(value) <= 15:
            prefix = bytes([0x90 + len(value)])
        else:
            prefix = b"\xdc" + len(value).to_bytes(2, "big")
        return prefix + b"".join(pack(item) for item in value)
    if isinstance(value, dict):
        if len(value) <= 15:
            prefix = bytes([0x80 + len(value)])
        else:
            prefix = b"\xde" + len(value).to_bytes(2, "big")
        return prefix + b"".join(pack(key) + pack(item) for key, item in value.items())
    raise TypeError(type(value))


def solution(index: int, name: str, strided_batched: bool) -> dict[str, object]:
    return {
        "index": index,
        "name": name,
        "problemType": {
            "operationIdentifier": "Contraction_l_Ailk_Bljk_Cijk_Dijk",
            "aType": "Float",
            "bType": "Float",
            "cType": "Float",
            "dType": "Float",
            "stridedBatched": strided_batched,
        },
        "sizeMapping": {"workGroup": [64, 4, 1]},
    }


def package_record() -> dict[str, str]:
    return {
        "architecture": "amd64",
        "filename": "rocblas_7.14-test_amd64.deb",
        "name": "rocblas",
        "sha256": "1" * 64,
        "version": "7.14-test",
    }


class GuestSgemmInventoryTest(unittest.TestCase):
    def test_builds_inventory_with_pinned_selection_and_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            symbol = "ordinary"
            workload = {
                "api": "rocblas_sgemm",
                "m": 128,
                "n": 128,
                "k": 128,
                "guest_executable": "opt/rocjitsu/bin/m2-rocblas-sgemm",
            }
            source = work / "rocblas_sgemm.cpp"
            source.write_text("// workload\n", encoding="utf-8")
            guest_init = work / "init"
            guest_init.write_bytes(b"#!/bin/busybox sh\npoweroff -f\n")
            workload_binary = work / "m2-rocblas-sgemm"
            write_executable(
                workload_binary,
                "#!/bin/sh\nprintf '%s\\n' '" + json.dumps(workload) + "'\n",
            )
            code_object = work / "library.hsaco"
            code_object.write_bytes(b"fake ELF")
            metadata = work / "library.dat"
            metadata.write_bytes(
                pack(
                    {
                        "solutions": [
                            solution(17, "strided", True),
                            solution(23, symbol, False),
                        ],
                        "library": {
                            "type": "Matching",
                            "table": [
                                {
                                    "key": [128, 128, 128],
                                    "index": 17,
                                    "speed": 1.0,
                                },
                                {
                                    "key": [128, 128, 128],
                                    "index": 23,
                                    "speed": 2.0,
                                },
                            ],
                        },
                    }
                )
            )
            lazy_metadata = work / "lazy.dat"
            lazy_metadata.write_bytes(
                pack({"type": "Placeholder", "value": "pinned-library"})
            )
            contract = {
                "schema_version": 2,
                "artifacts": {
                    "code_object": {
                        "abi_version": 4,
                        "guest_path": "opt/rocm/library.hsaco",
                        "processor": "gfx1250",
                        "sha256": digest(code_object.read_bytes()),
                        "silicon_revision": "B0",
                        "target": "amdgcn-amd-amdhsa--gfx1250",
                    },
                    "metadata": {
                        "guest_path": "opt/rocm/library.dat",
                        "sha256": digest(metadata.read_bytes()),
                    },
                    "lazy_metadata": {
                        "guest_path": "opt/rocm/lazy.dat",
                        "sha256": digest(lazy_metadata.read_bytes()),
                    },
                },
                "feature_tasks": {},
                "guest_init": {
                    "guest_path": "init",
                    "sha256": digest(guest_init.read_bytes()),
                },
                "packages": [package_record()],
                "runtime": {
                    "backend": "Tensile",
                    "environment": {
                        "ROCBLAS_TENSILE_LIBPATH": "/opt/rocm/core-7.14/lib/rocblas/library",
                        "ROCBLAS_USE_HIPBLASLT": "0",
                    },
                },
                "selection": {
                    "matching_key": [128, 128, 128],
                    "matching_mode": "exact",
                    "strided_batched": False,
                    "expected_solution_indices": [23],
                    "expected_symbol": symbol,
                    "operation_identifier": "Contraction_l_Ailk_Bljk_Cijk_Dijk",
                    "placeholder": "pinned-library",
                },
                "source": {
                    "path": source.name,
                    "guest_path": "usr/src/rocjitsu/rocblas_sgemm.cpp",
                },
                "workload": workload,
            }
            contract_path = work / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            dependency = work / "libdependency.so.1"
            dependency.write_bytes(b"dependency")
            readobj = work / "llvm-readobj"
            write_executable(
                readobj,
                "#!/bin/sh\n"
                "[ \"$1\" = --version ] && { echo 'readobj test'; exit 0; }\n"
                "cat <<'EOF'\n"
                "Format: elf64-amdgpu\nArch: amdgcn\nOS/ABI: AMDGPU_HSA\n"
                "ABIVersion: 4\nMachine: EM_AMDGPU\nEF_AMDGPU_MACH_AMDGCN_GFX1250\n"
                "AMDGPU Metadata: ---\namdhsa.kernels:\n  - .args: []\n"
                "    .gfx1250_revision: B0\n"
                "    .group_segment_fixed_size: 64\n"
                "    .kernarg_segment_align: 8\n"
                "    .kernarg_segment_size: 16\n"
                f"    .name:           {symbol}\n"
                "    .private_segment_fixed_size: 0\n"
                "    .sgpr_count:     8\n    .sgpr_spill_count: 0\n"
                f"    .symbol:         {symbol}.kd\n"
                "    .vgpr_count:     8\n    .vgpr_spill_count: 0\n"
                "    .wavefront_size: 32\n"
                "amdhsa.target:   amdgcn-amd-amdhsa--gfx1250\n"
                "amdhsa.version:\n  - 1\n  - 1\n...\nEOF\n",
            )
            llvm_nm = work / "llvm-nm"
            write_executable(
                llvm_nm,
                "#!/bin/sh\n"
                "[ \"$1\" = --version ] && { echo 'nm test'; exit 0; }\n"
                f"printf '00001000 T {symbol}\\n00001100 T next_kernel\\n'\n",
            )
            llvm_objdump = work / "llvm-objdump"
            write_executable(
                llvm_objdump,
                "#!/bin/sh\n"
                "[ \"$1\" = --version ] && { echo 'objdump test'; exit 0; }\n"
                "printf '\\ts_barrier\\n\\tv_mfma_f32_32x32x2_f32 v0, v1, v2, v3\\n'\n",
            )
            ldd = work / "ldd"
            write_executable(
                ldd,
                "#!/bin/sh\n"
                "[ \"$1\" = --version ] && { echo 'ldd test'; exit 0; }\n"
                f"printf 'libdependency.so.1 => {dependency} (0x1)\\n'\n",
            )

            result = inventory_tool.inventory(
                contract_path,
                workload_binary,
                code_object,
                metadata,
                lazy_metadata,
                readobj,
                llvm_nm,
                llvm_objdump,
                ldd,
            )

            self.assertEqual(result["selected_solution"]["indices"], [23])
            self.assertEqual(
                result["code_object"]["selected_kernel_range"]["size"], 256
            )
            self.assertEqual(
                result["code_object"]["instructions"]["mfma"],
                ["v_mfma_f32_32x32x2_f32"],
            )
            self.assertIn(
                str(dependency).removeprefix("/"), result["required_guest_files"]
            )
            self.assertEqual(result["packages"], [package_record()])
            self.assertEqual(result["schema_version"], 2)
            self.assertEqual(result["runtime"]["backend"], "Tensile")
            self.assertEqual(
                result["guest_init"]["sha256"], digest(guest_init.read_bytes())
            )

    def test_tracked_guest_init_selects_the_contracted_tensile_backend(self) -> None:
        contract = json.loads(
            (TOOLS / "vfio_guest/rocblas_sgemm_contract.json").read_text(
                encoding="utf-8"
            )
        )
        environment = contract["runtime"]["environment"]
        init_path = TOOLS / "vfio_guest/init"
        init = init_path.read_text(encoding="utf-8")

        self.assertEqual(contract["runtime"]["backend"], "Tensile")
        for name, value in environment.items():
            self.assertIn(f'export {name}="{value}"', init)
        self.assertEqual(
            contract["guest_init"]["sha256"], inventory_tool.sha256_file(init_path)
        )

    def test_rejects_a_later_init_override_even_if_required_exports_remain(
        self,
    ) -> None:
        contract = json.loads(
            (TOOLS / "vfio_guest/rocblas_sgemm_contract.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary) / "init"
            altered.write_bytes(
                (TOOLS / "vfio_guest/init").read_bytes()
                + b'export ROCBLAS_USE_HIPBLASLT="1"\n'
            )
            with self.assertRaisesRegex(
                inventory_tool.InventoryError, "M2 guest init hash mismatch"
            ):
                inventory_tool.verify_hash(
                    altered,
                    contract["guest_init"]["sha256"],
                    "M2 guest init",
                )

    def test_rejects_name_and_version_only_package_assertions(self) -> None:
        with self.assertRaisesRegex(
            inventory_tool.InventoryError, r"packages\[0\]\.architecture"
        ):
            inventory_tool.package_records(
                [{"name": "rocblas", "version": "7.14-test"}]
            )

    def test_decodes_tensile_messagepack_without_external_modules(self) -> None:
        value = {
            "solutions": [solution(17, "kernel", False)],
            "library": {"type": "Matching", "table": []},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "library.dat"
            path.write_bytes(pack(value))

            self.assertEqual(inventory_tool.decode_msgpack(path), value)

    def test_selects_non_strided_exact_128_cube_solution(self) -> None:
        metadata = {
            "solutions": [
                solution(17, "strided", True),
                solution(23, "ordinary", False),
            ],
            "library": {
                "type": "Problem",
                "rows": [
                    {
                        "predicate": {"type": "TruePred"},
                        "library": {
                            "type": "Matching",
                            "table": [
                                {"key": [128, 128, 128], "index": 17, "speed": 1.0},
                                {"key": [128, 128, 128], "index": 23, "speed": 2.0},
                            ],
                        },
                    }
                ],
            },
        }
        selection = {
            "matching_key": [128, 128, 128],
            "matching_mode": "exact",
            "strided_batched": False,
            "expected_solution_indices": [23],
            "expected_symbol": "ordinary",
            "operation_identifier": "Contraction_l_Ailk_Bljk_Cijk_Dijk",
        }

        selected = inventory_tool.select_solution(metadata, selection)

        self.assertEqual(selected["indices"], [23])
        self.assertEqual(selected["candidate_indices"], [17, 23])
        self.assertEqual(selected["speeds"], [2.0])

    def test_selects_all_equivalent_nearest_fallback_entries(self) -> None:
        symbol = "fallback"
        metadata = {
            "solutions": [
                solution(17, "strided", True),
                solution(23, symbol, False),
                solution(29, symbol, False),
            ],
            "library": {
                "type": "Matching",
                "distance": "Euclidean",
                "table": [
                    {"key": [5504, 5504, 5504], "index": 17, "speed": 1.0},
                    {"key": [5504, 5504, 5504], "index": 23, "speed": 2.0},
                    {"key": [5504, 5504, 5504], "index": 29, "speed": 3.0},
                ],
            },
        }
        selection = {
            "matching_key": [128, 128, 128],
            "matching_mode": "nearest_euclidean",
            "strided_batched": False,
            "expected_solution_indices": [23, 29],
            "expected_symbol": symbol,
            "operation_identifier": "Contraction_l_Ailk_Bljk_Cijk_Dijk",
        }

        selected = inventory_tool.select_solution(metadata, selection)

        self.assertEqual(selected["indices"], [23, 29])
        self.assertEqual(selected["matched_keys"], [[5504, 5504, 5504]])
        self.assertEqual(selected["speeds"], [2.0, 3.0])

    def test_finds_lazy_library_placeholder(self) -> None:
        metadata = {
            "library": {
                "type": "Problem",
                "rows": [
                    {
                        "library": {
                            "type": "Placeholder",
                            "value": "pinned-library",
                        }
                    }
                ],
            }
        }

        self.assertTrue(inventory_tool.contains_placeholder(metadata, "pinned-library"))
        self.assertFalse(inventory_tool.contains_placeholder(metadata, "other"))

    def test_inventories_selected_kernel_instructions(self) -> None:
        disassembly = """
\ts_load_dword s0, s[0:1], 0x0
\tds_write_b32 v0, v1
\ts_barrier
\tv_mfma_f32_32x32x2_f32 v[0:15], v0, v1, v[0:15]
\tbuffer_store_dword v0, v1, s[0:3], 0 offen sc0 sc1
\ts_waitcnt vmcnt(0)
"""

        result = inventory_tool.parse_instructions(disassembly)

        self.assertEqual(result["instruction_count"], 6)
        self.assertEqual(result["barriers"], ["s_barrier"])
        self.assertEqual(result["lds"], ["ds_write_b32"])
        self.assertEqual(result["mfma"], ["v_mfma_f32_32x32x2_f32"])
        self.assertEqual(result["coherency_flags"], ["sc0", "sc1"])
        self.assertEqual(result["atomics"], [])

    def test_parses_selected_kernel_resource_metadata(self) -> None:
        symbol = "selected_kernel"
        output = f"""
AMDGPU Metadata: ---
amdhsa.kernels:
  - .args: []
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .kernarg_segment_size: 4
    .name:           setup_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .sgpr_spill_count: 0
    .symbol:         setup_kernel.kd
    .vgpr_count:     2
    .vgpr_spill_count: 0
    .wavefront_size: 32
  - .args: []
    .gfx1250_revision: B0
    .group_segment_fixed_size: 12544
    .kernarg_segment_align: 8
    .kernarg_segment_size: 160
    .name:           {symbol}
    .private_segment_fixed_size: 0
    .sgpr_count:     70
    .sgpr_spill_count: 0
    .symbol:         {symbol}.kd
    .vgpr_count:     96
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...
"""

        metadata = inventory_tool.parse_kernel_metadata(output, symbol)

        self.assertEqual(metadata["group_segment_fixed_size"], 12544)
        self.assertEqual(metadata["private_segment_fixed_size"], 0)
        self.assertEqual(metadata["kernarg_segment_size"], 160)
        self.assertEqual(metadata["metadata_version"], [1, 2])
        self.assertEqual(metadata["target"], "amdgcn-amd-amdhsa--gfx1250")
        self.assertEqual(metadata["silicon_revision"], "B0")

    def test_records_runtime_dependency_sources_and_guest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            binary = work / "workload"
            binary.write_bytes(b"workload")
            library = work / "libexample.so.1.2"
            library.write_bytes(b"library")
            alias = work / "libexample.so.1"
            alias.symlink_to(library.name)
            loader = work / "ld-linux-test.so.2"
            loader.write_bytes(b"loader")
            ldd = work / "ldd"
            ldd.write_text(
                "#!/bin/sh\n"
                f"printf 'libexample.so.1 => {alias} (0x1)\\n'\n"
                f"printf '{loader} (0x2)\\n'\n",
                encoding="utf-8",
            )
            ldd.chmod(0o755)

            dependencies = inventory_tool.runtime_dependencies(binary, ldd)

            self.assertEqual(len(dependencies), 2)
            example = next(
                item
                for item in dependencies
                if item["guest_path"].endswith("libexample.so.1")
            )
            self.assertEqual(example["source"], str(library.resolve()))
            self.assertEqual(example["sha256"], inventory_tool.sha256_file(library))


if __name__ == "__main__":
    unittest.main()
