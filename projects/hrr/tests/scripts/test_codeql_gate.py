#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for codeql_gate.py."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import codeql_gate  # noqa: E402

SRC = "projects/hrr/playback/hrr_reader.cpp"
OPEN_LINE = 'FILE* f = fopen(file_path.c_str(), "rb");'
ARGV_SRC = "user input (a command-line argument)"
FREAD_SRC = "user input (string read by fread)"
ARGV = f"This argument ... is derived from [{ARGV_SRC}](1) and then passed to fopen."
# CodeQL writes one line per flow when several inputs reach the same sink.
BOTH = (f"This argument ... is derived from [{FREAD_SRC}](1) and then passed to fopen.\n"
        f"This argument ... is derived from [{ARGV_SRC}](2) and then passed to fopen.")


def result(line: int, message: str = ARGV, rule: str = "cpp/path-injection",
           uri: str = SRC) -> dict:
    return {
        "ruleId": rule,
        "message": {"text": message},
        "locations": [{"physicalLocation": {
            "artifactLocation": {"uri": uri, "uriBaseId": "%SRCROOT%"},
            "region": {"startLine": line},
        }}],
    }


def entry(sources=(ARGV_SRC,), status: str = "false-positive",
          code: str = OPEN_LINE, **over) -> dict:
    e = {"rule": "cpp/path-injection", "file": SRC, "code": code,
         "sources": list(sources), "status": status, "reason": "reviewed"}
    e.update(over)
    return e


class Gate(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source(["// header", "  " + OPEN_LINE, "int x;"])

    def source(self, lines: list[str]) -> None:
        path = self.root / SRC
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")

    def run_gate(self, results: list[dict], entries: list[dict]) -> tuple[int, list, str]:
        alerts = self.root / "alerts.json"
        alerts.write_text(json.dumps({"alerts": entries}))
        sarif_in = self.root / "in.sarif"
        sarif_in.write_text(json.dumps({"runs": [{"results": results}]}))
        sarif_out = self.root / "out.sarif"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = codeql_gate.main(["--alerts", str(alerts), "--source-root", str(self.root),
                                   str(sarif_in), str(sarif_out)])
        kept = json.loads(sarif_out.read_text())["runs"][0]["results"] if sarif_out.exists() else None
        return rc, kept, out.getvalue()

    def test_false_positive_is_removed_and_passes(self):
        rc, kept, out = self.run_gate([result(2)], [entry()])
        self.assertEqual(rc, 0)
        self.assertEqual(kept, [])
        self.assertNotIn("::error", out)

    def test_unlisted_result_is_kept_and_fails(self):
        rc, kept, out = self.run_gate([result(3, rule="cpp/world-writable-file-creation")],
                                      [entry()])
        self.assertEqual(rc, 1)
        self.assertEqual(len(kept), 1)
        self.assertIn(f"::error file={SRC},line=3::new CodeQL alert "
                      "cpp/world-writable-file-creation", out)

    def test_entry_survives_the_line_moving(self):
        self.source(["// header", "// added above", "int y;", "  " + OPEN_LINE])
        rc, kept, _ = self.run_gate([result(4)], [entry()])
        self.assertEqual((rc, kept), (0, []))

    def test_changed_flagged_line_is_new(self):
        self.source(["// header", '  FILE* f = fopen(other.c_str(), "rb");'])
        rc, kept, out = self.run_gate([result(2)], [entry()])
        self.assertEqual(rc, 1)
        self.assertEqual(len(kept), 1)
        self.assertIn("stale entry", out)

    def test_same_line_reached_by_another_input_is_new(self):
        # The entry clears the command-line flow into this fopen, not a flow from
        # archive bytes that reaches the same call as well.
        rc, kept, _ = self.run_gate([result(2, message=BOTH)], [entry()])
        self.assertEqual(rc, 1)
        self.assertEqual(len(kept), 1)

    def test_each_result_matches_the_entry_with_its_sources(self):
        rc, kept, out = self.run_gate([result(2, message=BOTH), result(2)],
                                      [entry(), entry(sources=(FREAD_SRC, ARGV_SRC))])
        self.assertEqual((rc, kept), (0, []))
        self.assertNotIn("stale entry", out)

    def test_open_alert_stays_in_the_upload_and_passes(self):
        rc, kept, out = self.run_gate([result(2)], [entry(status="open")])
        self.assertEqual(rc, 0)
        self.assertEqual(len(kept), 1)
        self.assertNotIn("::error", out)

    def test_stale_entry_warns_without_failing(self):
        rc, kept, out = self.run_gate([], [entry()])
        self.assertEqual((rc, kept), (0, []))
        self.assertIn("::warning", out)
        self.assertIn("matched no result", out)

    def test_entry_without_reason_is_refused(self):
        rc, kept, out = self.run_gate([result(2)], [entry(reason=" ")])
        self.assertEqual(rc, 2)
        self.assertIsNone(kept)
        self.assertIn("lacks reason", out)

    def test_entry_without_sources_is_refused(self):
        bad = entry()
        del bad["sources"]
        rc, _, out = self.run_gate([result(2)], [bad])
        self.assertEqual(rc, 2)
        self.assertIn("needs sources", out)

    def test_unknown_status_is_refused(self):
        rc, _, out = self.run_gate([result(2)], [entry(status="dismissed")])
        self.assertEqual(rc, 2)
        self.assertIn("status 'dismissed'", out)


class CheckedInAlerts(unittest.TestCase):
    def test_every_entry_names_a_line_that_exists(self):
        alerts = TOOLS / "codeql-replay-alerts.json"
        repo = TOOLS.parents[2]
        entries = codeql_gate.load_alerts(alerts)
        self.assertTrue(entries)
        for e in entries:
            lines = [l.strip() for l in (repo / e["file"]).read_text().splitlines()]
            self.assertIn(e["code"].strip(), lines, f"{e['file']}: {e['code']}")


if __name__ == "__main__":
    unittest.main()
