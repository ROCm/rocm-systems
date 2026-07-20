#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


DBI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DBI_DIR))
from consan_run_provenance import initialize_contract, summarize_contract  # noqa: E402
from consan_validation_support import RESULT_SCHEMA_VERSION, read_row_result  # noqa: E402
from consan_validation_test_support import temporary_root  # noqa: E402


RUNNER = DBI_DIR / "consan_fault_runner.py"


class ConSanRunProvenanceTest(unittest.TestCase):
    def fixture(self, root: Path, rows: tuple[str, ...] = ("row-a", "row-b")) -> tuple[dict, dict]:
        source = root / "source"
        source.mkdir()
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(
            ["git", "-C", str(source), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
        (source / "tracked.txt").write_text("initial\n")
        subprocess.run(["git", "-C", str(source), "add", "tracked.txt"], check=True)
        subprocess.run(["git", "-C", str(source), "commit", "-qm", "initial"], check=True)
        plan = root / "plan.json"
        manifest = root / "manifest.json"
        hook = root / "hook.so"
        binary = root / "binary"
        plan.write_text(json.dumps({"schema_version": 1, "rows": [{"name": row} for row in rows]}))
        manifest.write_text(json.dumps({"schema_version": 1, "cases": []}))
        hook.write_bytes(b"hook")
        binary.write_bytes(b"binary")
        artifact = root / "artifacts"
        contract = initialize_contract(
            artifact,
            plan,
            manifest,
            [("hook", hook), ("binary", binary)],
            [source],
        )
        context = {
            "source": source,
            "plan": plan,
            "manifest": manifest,
            "hook": hook,
            "binary": binary,
            "artifact": artifact,
        }
        return contract, context

    def row_provenance(self, contract: dict) -> dict:
        records = {record["label"]: record for record in contract["files"]}
        return {
            "run_id": contract["run_id"],
            "contract_sha256": contract["contract_sha256"],
            "plan_canonical_sha256": records["plan"]["canonical_sha256"],
            "manifest_canonical_sha256": records["manifest"]["canonical_sha256"],
            "files": contract["files"],
            "sources": contract["sources"],
        }

    def write_rows(self, contract: dict, artifact: Path) -> None:
        for name in contract["declared_rows"]:
            row = artifact / name
            row.mkdir()
            (row / "result.json").write_text(
                json.dumps(
                    {
                        "schema_version": RESULT_SCHEMA_VERSION,
                        "run_provenance": self.row_provenance(contract),
                    }
                )
            )

    def test_accepts_only_exact_declared_current_run(self) -> None:
        with temporary_root() as root:
            contract, context = self.fixture(root)
            self.write_rows(contract, context["artifact"])
            summary = summarize_contract(context["artifact"])
            self.assertEqual(summary["status"], "accepted", summary["errors"])
            self.assertEqual(summary["rows"], ["row-a", "row-b"])

    def test_refuses_foreign_nonempty_artifact_root(self) -> None:
        with temporary_root() as root:
            contract, context = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "nonempty"):
                initialize_contract(
                    context["artifact"],
                    context["plan"],
                    context["manifest"],
                    [("hook", context["hook"]), ("binary", context["binary"])],
                    [context["source"]],
                )
            self.assertTrue(contract["run_id"])

    def test_rejects_mixed_schema_run_hash_revision_and_row_sets(self) -> None:
        mutations = {
            "schema": lambda result: result.update(schema_version=2),
            "run_id": lambda result: result["run_provenance"].update(run_id="foreign"),
            "contract": lambda result: result["run_provenance"].update(contract_sha256="bad"),
            "plan": lambda result: result["run_provenance"].update(plan_canonical_sha256="bad"),
            "manifest": lambda result: result["run_provenance"].update(
                manifest_canonical_sha256="bad"
            ),
            "input": lambda result: result["run_provenance"].update(files=[]),
            "revision": lambda result: result["run_provenance"].update(sources=[]),
        }
        expected = {
            "schema": "schema version 3",
            "run_id": "run_id mismatch",
            "contract": "contract hash mismatch",
            "plan": "plan hash mismatch",
            "manifest": "manifest hash mismatch",
            "input": "input file provenance mismatch",
            "revision": "source revision/fingerprint mismatch",
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), temporary_root() as root:
                contract, context = self.fixture(root, ("row-a",))
                self.write_rows(contract, context["artifact"])
                path = context["artifact"] / "row-a" / "result.json"
                result = read_row_result(context["artifact"], "row-a")
                mutate(result)
                path.write_text(json.dumps(result))
                summary = summarize_contract(context["artifact"])
                self.assertEqual(summary["status"], "rejected")
                self.assertEqual(summary["rows"], [])
                self.assertIn(expected[name], "\n".join(summary["errors"]))

        with temporary_root() as root:
            contract, context = self.fixture(root)
            self.write_rows(contract, context["artifact"])
            path = context["artifact"] / "row-b" / "result.json"
            result = read_row_result(context["artifact"], "row-b")
            result["run_provenance"]["run_id"] = "foreign-run"
            path.write_text(json.dumps(result))
            summary = summarize_contract(context["artifact"])
            self.assertEqual(summary["status"], "rejected")
            self.assertEqual(summary["rows"], [])
            self.assertIn("run_id mismatch", "\n".join(summary["errors"]))

        with temporary_root() as root:
            contract, context = self.fixture(root)
            self.write_rows(contract, context["artifact"])
            (context["artifact"] / "row-b" / "result.json").unlink()
            extra = context["artifact"] / "foreign"
            extra.mkdir()
            (extra / "result.json").write_text("{}")
            summary = summarize_contract(context["artifact"])
            rendered = "\n".join(summary["errors"])
            self.assertIn("missing declared row", rendered)
            self.assertIn("extra undeclared row", rendered)
            self.assertEqual(summary["rows"], [])

    def test_rejects_current_input_head_dirty_and_worktree_drift(self) -> None:
        with temporary_root() as root:
            contract, context = self.fixture(root, ("row-a",))
            self.write_rows(contract, context["artifact"])
            context["binary"].write_bytes(b"changed")
            summary = summarize_contract(context["artifact"])
            self.assertIn("input hash changed: binary", "\n".join(summary["errors"]))

        with temporary_root() as root:
            contract, context = self.fixture(root, ("row-a",))
            self.write_rows(contract, context["artifact"])
            (context["source"] / "tracked.txt").write_text("dirty\n")
            summary = summarize_contract(context["artifact"])
            rendered = "\n".join(summary["errors"])
            self.assertIn("dirty state changed", rendered)
            self.assertIn("worktree fingerprint changed", rendered)

        with temporary_root() as root:
            contract, context = self.fixture(root, ("row-a",))
            self.write_rows(contract, context["artifact"])
            (context["source"] / "tracked.txt").write_text("next\n")
            subprocess.run(
                ["git", "-C", str(context["source"]), "commit", "-qam", "next"],
                check=True,
            )
            summary = summarize_contract(context["artifact"])
            self.assertIn("source HEAD changed", "\n".join(summary["errors"]))

    def test_runner_stamps_contract_and_summary_accepts(self) -> None:
        with temporary_root() as root:
            contract, context = self.fixture(root, ("row-a",))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--artifact-root",
                    str(context["artifact"]),
                    "--run-contract",
                    str(context["artifact"] / "run-contract.json"),
                    "--name",
                    "row-a",
                    "--timeout",
                    "5",
                    "--",
                    sys.executable,
                    "-c",
                    "pass",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "CTEST_PARALLEL_LEVEL": "1"},
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = read_row_result(context["artifact"], "row-a")
            self.assertEqual(result["run_provenance"]["run_id"], contract["run_id"])
            self.assertEqual(summarize_contract(context["artifact"])["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
