#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import rocjitsu_consan_allowlist as allowlist  # noqa: E402


def _write_trace(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Kind", "Kernel_Name", "Dispatch_Id"))
        for index, (kind, name) in enumerate(rows):
            writer.writerow((kind, name, index))


class ConSanAllowlistTest(unittest.TestCase):
    def test_collects_sorted_unique_exact_names_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "host" / "123"
            nested.mkdir(parents=True)
            _write_trace(
                nested / "first_kernel_trace.csv",
                [
                    ("KERNEL_DISPATCH", "templated<float, int>.kd"),
                    ("KERNEL_DISPATCH", "plain_kernel"),
                ],
            )
            _write_trace(
                root / "second_kernel_trace.csv",
                [
                    ("KERNEL_DISPATCH", "plain_kernel"),
                    ("MEMORY_COPY", "not_a_kernel"),
                ],
            )
            self.assertEqual(
                allowlist.collect_kernel_names([root]),
                ("plain_kernel", "templated<float, int>"),
            )

    def test_rejects_missing_kernel_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "bad_kernel_trace.csv"
            trace.write_text("Kind,Dispatch_Id\nKERNEL_DISPATCH,1\n", encoding="utf-8")
            with self.assertRaisesRegex(allowlist.AllowlistError, "Kernel_Name"):
                allowlist.collect_kernel_names([trace])

    def test_rejects_trace_without_dispatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "empty_kernel_trace.csv"
            _write_trace(trace, [("MEMORY_COPY", "not_a_kernel")])
            with self.assertRaisesRegex(allowlist.AllowlistError, "no GPU dispatches"):
                allowlist.collect_kernel_names([trace])

    def test_cli_writes_plain_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "run_kernel_trace.csv"
            output = root / "kernels.txt"
            _write_trace(trace, [("KERNEL_DISPATCH", "kernel_with,comma")])
            self.assertEqual(
                allowlist.main(["--output", str(output), str(trace)]), 0
            )
            self.assertEqual(output.read_text(encoding="utf-8"), "kernel_with,comma\n")


if __name__ == "__main__":
    unittest.main()
