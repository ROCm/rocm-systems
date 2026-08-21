#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

TOOL_PATH = Path(__file__).with_name("record-gfx1250-dbt-sha-pairs.py")
SPEC = importlib.util.spec_from_file_location("gfx1250_dbt_sha_pairs", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class CollectionTest(unittest.TestCase):
    def test_collects_successes_and_skips_expected_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            corpus = root / "corpus"
            objects = corpus / "objects"
            objects.mkdir(parents=True)
            source = b"ELF input bytes"
            excluded_source = b"known large input"
            input_sha256 = hashlib.sha256(source).hexdigest()
            excluded_sha256 = hashlib.sha256(excluded_source).hexdigest()
            (objects / f"{input_sha256}.hsaco").write_bytes(source)
            (objects / f"{excluded_sha256}.hsaco").write_bytes(excluded_source)
            input_manifest = corpus / "SHA256SUMS"
            input_manifest.write_text(
                f"{input_sha256}  objects/{input_sha256}.hsaco\n"
                f"{excluded_sha256}  objects/{excluded_sha256}.hsaco\n",
                encoding="ascii",
            )
            expected_failures = corpus / "expected-failures.json"
            expected_failures.write_text(
                json.dumps({"expected_failures": {excluded_sha256: {}}}),
                encoding="utf-8",
            )
            translator = self._write_translator(root, mode="success")
            fragments = root / "fragments"

            self.assertEqual(
                TOOL.main(
                    self._collect_args(
                        translator,
                        corpus,
                        input_manifest,
                        expected_failures,
                        fragments,
                    )
                ),
                0,
            )

            record = json.loads(
                (fragments / f"{input_sha256}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["input_sha256"], input_sha256)
            self.assertEqual(
                record["output_sha256"], hashlib.sha256(source[::-1]).hexdigest()
            )
            self.assertEqual(record["input_bytes"], len(source))
            self.assertEqual(record["output_bytes"], len(source))
            self.assertFalse((fragments / f"{excluded_sha256}.json").exists())

    def test_rejects_translation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            corpus = root / "corpus"
            objects = corpus / "objects"
            objects.mkdir(parents=True)
            source = b"bad input"
            input_sha256 = hashlib.sha256(source).hexdigest()
            (objects / f"{input_sha256}.hsaco").write_bytes(source)
            input_manifest = corpus / "SHA256SUMS"
            input_manifest.write_text(
                f"{input_sha256}  objects/{input_sha256}.hsaco\n",
                encoding="ascii",
            )
            expected_failures = corpus / "expected-failures.json"
            expected_failures.write_text(
                json.dumps({"expected_failures": {}}), encoding="utf-8"
            )
            translator = self._write_translator(root, mode="failure")
            fragments = root / "fragments"

            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                status = TOOL.main(
                    self._collect_args(
                        translator,
                        corpus,
                        input_manifest,
                        expected_failures,
                        fragments,
                    )
                )
            self.assertEqual(status, TOOL.ERROR)
            self.assertIn("translation failed with status 7", diagnostics.getvalue())
            self.assertEqual(list(fragments.glob("*.json")), [])

    def test_rejects_translation_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            corpus, input_manifest, expected_failures = self._write_single_input(root)
            translator = self._write_translator(root, mode="timeout")
            arguments = self._collect_args(
                translator,
                corpus,
                input_manifest,
                expected_failures,
                root / "fragments",
            )
            timeout_index = arguments.index("--timeout") + 1
            arguments[timeout_index] = "0.05"

            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                status = TOOL.main(arguments)

            self.assertEqual(status, TOOL.ERROR)
            self.assertIn("translation timed out", diagnostics.getvalue())

    def test_rejects_empty_translation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            corpus, input_manifest, expected_failures = self._write_single_input(root)
            translator = self._write_translator(root, mode="empty")

            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                status = TOOL.main(
                    self._collect_args(
                        translator,
                        corpus,
                        input_manifest,
                        expected_failures,
                        root / "fragments",
                    )
                )

            self.assertEqual(status, TOOL.ERROR)
            self.assertIn(
                "translation produced an empty output", diagnostics.getvalue()
            )

    def test_accepts_relative_translator_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            corpus, input_manifest, expected_failures = self._write_single_input(root)
            self._write_translator(root, mode="success")
            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                status = TOOL.main(
                    self._collect_args(
                        Path("translator.py"),
                        corpus,
                        input_manifest,
                        expected_failures,
                        root / "fragments",
                    )
                )
            finally:
                os.chdir(previous_directory)

            self.assertEqual(status, 0)

    def test_rejects_nonempty_fragment_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            corpus, input_manifest, expected_failures = self._write_single_input(root)
            translator = self._write_translator(root, mode="success")
            fragments = root / "fragments"
            fragments.mkdir()
            (fragments / "stale.json").write_text("{}", encoding="utf-8")

            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                status = TOOL.main(
                    self._collect_args(
                        translator,
                        corpus,
                        input_manifest,
                        expected_failures,
                        fragments,
                    )
                )

            self.assertEqual(status, TOOL.ERROR)
            self.assertIn("fragment directory is not empty", diagnostics.getvalue())

    def test_record_fragment_rejects_conflicting_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fragments = Path(temporary_directory)
            record = {
                "input_bytes": 64,
                "input_sha256": "1" * 64,
                "output_bytes": 96,
                "output_sha256": "2" * 64,
            }
            TOOL._record_fragment(fragments, record)
            conflicting = {**record, "output_sha256": "3" * 64}

            with self.assertRaisesRegex(ValueError, "conflicting translation outputs"):
                TOOL._record_fragment(fragments, conflicting)

    @staticmethod
    def _write_translator(root: Path, *, mode: str) -> Path:
        translator = root / "translator.py"
        translator.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            "import sys\n"
            "import time\n"
            "if '--verify-idempotence' not in sys.argv:\n"
            "    print('missing extra argument', file=sys.stderr)\n"
            "    sys.exit(9)\n"
            "source = next(Path(arg) for arg in sys.argv[1:] if arg.endswith('.hsaco'))\n"
            f"mode = {mode!r}\n"
            "if mode == 'timeout':\n"
            "    time.sleep(60)\n"
            "if mode == 'failure':\n"
            "    print('expected failure', file=sys.stderr)\n"
            "    sys.exit(7)\n"
            "if mode == 'success':\n"
            "    sys.stdout.buffer.write(source.read_bytes()[::-1])\n"
            "sys.exit(0)\n",
            encoding="utf-8",
        )
        translator.chmod(0o755)
        return translator

    @staticmethod
    def _write_single_input(root: Path) -> tuple[Path, Path, Path]:
        corpus = root / "corpus"
        objects = corpus / "objects"
        objects.mkdir(parents=True)
        source = b"single input"
        input_sha256 = hashlib.sha256(source).hexdigest()
        (objects / f"{input_sha256}.hsaco").write_bytes(source)
        input_manifest = corpus / "SHA256SUMS"
        input_manifest.write_text(
            f"{input_sha256}  objects/{input_sha256}.hsaco\n", encoding="ascii"
        )
        expected_failures = corpus / "expected-failures.json"
        expected_failures.write_text(
            json.dumps({"expected_failures": {}}), encoding="utf-8"
        )
        return corpus, input_manifest, expected_failures

    @staticmethod
    def _collect_args(
        translator: Path,
        corpus: Path,
        input_manifest: Path,
        expected_failures: Path,
        fragments: Path,
    ) -> list[str]:
        return [
            "collect",
            "--translator",
            str(translator),
            "--corpus",
            str(corpus),
            "--input-manifest",
            str(input_manifest),
            "--expected-failures",
            str(expected_failures),
            "--fragments",
            str(fragments),
            "--workers",
            "2",
            "--timeout",
            "5",
        ]


class ManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.fragments = self.root / "fragments"
        self.fragments.mkdir()
        self.success_input = "1" * 64
        self.failed_input = "2" * 64
        self.output_digest = "3" * 64
        self._write_fragment(self.success_input, self.output_digest)

        self.input_manifest = self.root / "SHA256SUMS"
        self.input_manifest.write_text(
            f"{self.success_input}  objects/{self.success_input}.hsaco\n"
            f"{self.failed_input}  objects/{self.failed_input}.hsaco\n",
            encoding="ascii",
        )
        self.expected_failures = self.root / "expected-failures.json"
        self.expected_failures.write_text(
            json.dumps(
                {
                    "expected_failures": {
                        self.failed_input: {
                            "kind": "memory-limit",
                            "reason": "fixture",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        self.expected_rewrites = self.root / "expected-rewrites.json"
        self.expected_rewrites.write_text(
            json.dumps({"expected_rewrites": {}}), encoding="utf-8"
        )
        self.package_lock = self.root / "package-lock.json"
        self.package_lock.write_text(
            json.dumps({"schema_version": 1}), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_finalize_is_deterministic_and_complete(self) -> None:
        first = self.root / "first.json"
        second = self.root / "second.json"

        self.assertEqual(TOOL.main(self._finalize_args(first)), 0)
        self.assertEqual(TOOL.main(self._finalize_args(second)), 0)

        self.assertEqual(first.read_bytes(), second.read_bytes())
        manifest = json.loads(first.read_text(encoding="utf-8"))
        self.assertEqual(manifest["excluded_inputs"], [self.failed_input])
        self.assertEqual(len(manifest["pairs"]), 1)
        self.assertEqual(manifest["profile"], TOOL.PROFILE)
        self.assertTrue(manifest["profile"]["verify_idempotence"])
        self.assertEqual(manifest["provenance"]["rocm_systems_commit"], "a" * 40)

    def test_finalize_rejects_missing_successful_input(self) -> None:
        (self.fragments / f"{self.success_input}.json").unlink()

        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            status = TOOL.main(self._finalize_args(self.root / "bad.json"))
        self.assertEqual(status, TOOL.ERROR)
        self.assertIn("missing=1", diagnostics.getvalue())

    def test_finalize_rejects_extra_input(self) -> None:
        self._write_fragment("4" * 64, "5" * 64)

        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            status = TOOL.main(self._finalize_args(self.root / "bad.json"))

        self.assertEqual(status, TOOL.ERROR)
        self.assertIn("extra=1", diagnostics.getvalue())

    def test_finalize_rejects_intermediate_git_sha_length(self) -> None:
        arguments = self._finalize_args(self.root / "bad.json")
        source_index = arguments.index("--source-commit") + 1
        arguments[source_index] = "a" * 41

        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            status = TOOL.main(arguments)

        self.assertEqual(status, TOOL.ERROR)
        self.assertIn("full lowercase Git SHA", diagnostics.getvalue())

    def test_finalize_rejects_unsafe_sdk_version(self) -> None:
        arguments = self._finalize_args(self.root / "bad.json")
        version_index = arguments.index("--rocm-sdk-version") + 1
        arguments[version_index] = "7.15.0 | @reviewers"

        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            status = TOOL.main(arguments)

        self.assertEqual(status, TOOL.ERROR)
        self.assertIn("unsupported characters", diagnostics.getvalue())

    def test_compare_reports_changed_outputs(self) -> None:
        baseline = self.root / "baseline.json"
        candidate = self.root / "candidate.json"
        report = self.root / "report.md"
        self.assertEqual(TOOL.main(self._finalize_args(baseline)), 0)

        self._write_fragment(self.success_input, "4" * 64, output_bytes=128)
        candidate_args = self._finalize_args(candidate)
        source_index = candidate_args.index("--source-commit") + 1
        candidate_args[source_index] = "c" * 40
        self.assertEqual(TOOL.main(candidate_args), 0)

        status = TOOL.main(
            [
                "compare",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--markdown",
                str(report),
            ]
        )

        self.assertEqual(status, TOOL.COMPARE_CHANGED)
        rendered = report.read_text(encoding="utf-8")
        self.assertIn("changes 1 translated output", rendered)
        self.assertIn("| Input SHA-256 | Input bytes |", rendered)
        self.assertIn("| 64 |", rendered)
        self.assertIn("| 96 |", rendered)
        self.assertIn("| 128 |", rendered)

    def test_compare_accepts_unchanged_outputs(self) -> None:
        baseline = self.root / "baseline.json"
        candidate = self.root / "candidate.json"
        report = self.root / "report.md"
        self.assertEqual(TOOL.main(self._finalize_args(baseline)), 0)
        candidate_args = self._finalize_args(candidate)
        source_index = candidate_args.index("--source-commit") + 1
        candidate_args[source_index] = "c" * 40
        self.assertEqual(TOOL.main(candidate_args), 0)

        status = TOOL.main(
            [
                "compare",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--markdown",
                str(report),
            ]
        )

        self.assertEqual(status, TOOL.COMPARE_UNCHANGED)
        self.assertIn("No translated output", report.read_text(encoding="utf-8"))

    def test_compare_rejects_incompatible_provenance(self) -> None:
        baseline = self.root / "baseline.json"
        candidate = self.root / "candidate.json"
        report = self.root / "report.md"
        self.assertEqual(TOOL.main(self._finalize_args(baseline)), 0)
        candidate_args = self._finalize_args(candidate)
        corpus_index = candidate_args.index("--corpus-commit") + 1
        candidate_args[corpus_index] = "d" * 40
        self.assertEqual(TOOL.main(candidate_args), 0)

        status = TOOL.main(
            [
                "compare",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--markdown",
                str(report),
            ]
        )

        self.assertEqual(status, TOOL.COMPARE_INCOMPATIBLE)
        self.assertIn("not directly comparable", report.read_text(encoding="utf-8"))

    def _write_fragment(
        self,
        input_digest: str,
        output_digest: str,
        *,
        output_bytes: int = 96,
    ) -> None:
        path = self.fragments / f"{input_digest}.json"
        path.write_text(
            json.dumps(
                {
                    "input_bytes": 64,
                    "input_sha256": input_digest,
                    "output_bytes": output_bytes,
                    "output_sha256": output_digest,
                }
            ),
            encoding="utf-8",
        )

    def _finalize_args(self, output: Path) -> list[str]:
        return [
            "finalize",
            "--fragments",
            str(self.fragments),
            "--output",
            str(output),
            "--input-manifest",
            str(self.input_manifest),
            "--expected-failures",
            str(self.expected_failures),
            "--expected-rewrites",
            str(self.expected_rewrites),
            "--package-lock",
            str(self.package_lock),
            "--source-commit",
            "a" * 40,
            "--corpus-commit",
            "b" * 40,
            "--rocm-sdk-version",
            "7.15.0a20260728",
        ]


if __name__ == "__main__":
    unittest.main()
