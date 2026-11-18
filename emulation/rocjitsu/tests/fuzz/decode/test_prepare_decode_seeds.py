#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "fuzz" / "decode"))
import prepare_decode_seeds


def write_elf_header(path: Path, machine: int, flags: int = 0) -> None:
    header = bytearray(52)
    header[:4] = b"\x7fELF"
    header[4] = 2
    header[5] = 1
    header[6] = 1
    header[18:20] = machine.to_bytes(2, "little")
    header[48:52] = flags.to_bytes(4, "little")
    path.write_bytes(header)


class ElfIdentityTest(unittest.TestCase):
    def test_reads_little_endian_machine_and_processor_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "kernel.co"
            write_elf_header(source, prepare_decode_seeds.ELF_MACHINE_AMDGPU, 0x134E)
            self.assertEqual(
                prepare_decode_seeds.elf_identity(source),
                (prepare_decode_seeds.ELF_MACHINE_AMDGPU, 0x4E),
            )

    def test_rejects_non_elf_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "not-elf"
            source.write_bytes(b"not an ELF")
            self.assertIsNone(prepare_decode_seeds.elf_identity(source))


class WindowSelectionTest(unittest.TestCase):
    def select(self, sections, max_seeds=3, max_windows=8192):
        selected = []
        selected_windows = set()
        for source, data in sections:
            prepare_decode_seeds.add_text_windows(
                selected,
                selected_windows,
                Path(source),
                data,
                max_seeds,
                max_windows,
            )
        return selected, selected_windows

    def test_selection_is_deterministic_and_bounded(self):
        data = bytes(range(40))
        first, first_windows = self.select([("a.co", data)], max_seeds=2)
        second, second_windows = self.select([("a.co", data)], max_seeds=2)
        self.assertEqual(first, second)
        self.assertEqual(first_windows, second_windows)
        self.assertEqual(len(first), 2)

    def test_deduplicates_windows_across_objects(self):
        data = bytes(range(24))
        selected, windows = self.select([("a.co", data), ("b.co", data)])
        self.assertEqual(len(selected), len(windows))
        self.assertEqual(len(selected), 3)

    def test_short_sections_add_no_windows(self):
        selected, windows = self.select([("short.co", bytes(15))])
        self.assertEqual(selected, [])
        self.assertEqual(windows, set())


class MainTest(unittest.TestCase):
    def test_skips_non_amdgpu_elf_and_records_machine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gpu = root / "gpu.co"
            failed_gpu = root / "failed-gpu.co"
            host = root / "host"
            write_elf_header(gpu, prepare_decode_seeds.ELF_MACHINE_AMDGPU, 0x4F)
            write_elf_header(failed_gpu, prepare_decode_seeds.ELF_MACHINE_AMDGPU, 0x50)
            write_elf_header(host, 62)
            output = root / "seeds"
            args = Namespace(
                input=[root],
                output=output,
                llvm_objcopy=Path("llvm-objcopy"),
                max_seeds=4,
                max_windows_per_object=8,
            )

            def extract(_tool, source, destination):
                if source == failed_gpu:
                    return False
                self.assertEqual(source, gpu)
                destination.write_bytes(bytes(range(32)))
                return True

            with (
                mock.patch.object(
                    prepare_decode_seeds, "parse_args", return_value=args
                ),
                mock.patch.object(
                    prepare_decode_seeds,
                    "input_files",
                    return_value=[gpu, failed_gpu, host],
                ),
                mock.patch.object(
                    prepare_decode_seeds, "extract_text", side_effect=extract
                ) as extract_mock,
                mock.patch.object(
                    prepare_decode_seeds, "tool_version", return_value="LLVM test"
                ),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(prepare_decode_seeds.main(), 0)

            self.assertEqual(extract_mock.call_count, 2)
            manifest = json.loads((root / "seeds.manifest.json").read_text())
            self.assertEqual(manifest["examined_objects"], 3)
            self.assertEqual(manifest["objects_with_text"], 1)
            self.assertEqual(manifest["extraction_failures"], 1)
            self.assertEqual(manifest["skipped_inputs"], 1)
            self.assertEqual(manifest["skipped_elf_machines"], [62])
            self.assertTrue(manifest["seeds"])
            self.assertTrue(
                all(seed["ef_amdgpu_mach"] == 0x4F for seed in manifest["seeds"])
            )

    def test_reports_why_all_inputs_were_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            host = root / "host"
            write_elf_header(host, 62)
            args = Namespace(
                input=[host],
                output=root / "seeds",
                llvm_objcopy=Path("llvm-objcopy"),
                max_seeds=4,
                max_windows_per_object=8,
            )
            with mock.patch.object(
                prepare_decode_seeds, "parse_args", return_value=args
            ):
                with self.assertRaisesRegex(
                    RuntimeError, r"examined 1, skipped 1.*e_machine values: 62"
                ):
                    prepare_decode_seeds.main()

    def test_reports_when_all_text_extractions_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gpu = root / "gpu.co"
            write_elf_header(gpu, prepare_decode_seeds.ELF_MACHINE_AMDGPU)
            args = Namespace(
                input=[gpu],
                output=root / "seeds",
                llvm_objcopy=Path("llvm-objcopy"),
                max_seeds=4,
                max_windows_per_object=8,
            )
            with (
                mock.patch.object(
                    prepare_decode_seeds, "parse_args", return_value=args
                ),
                mock.patch.object(
                    prepare_decode_seeds, "extract_text", return_value=False
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, r"examined 1, skipped 0.*1 \.text extraction failures"
                ):
                    prepare_decode_seeds.main()


if __name__ == "__main__":
    unittest.main()
