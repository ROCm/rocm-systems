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
INPUTS = "projects/hrr/playback/hrr_playback.cpp"
INPUT_LINES = ["int main(int argc, char** argv) {",
               "  size_t got = fread(out.data(), 1, hdr_size, f);",
               "  if (fread(out.data() + hdr_size, 1, pl_size, f) != pl_size) {"]
ARGV_SRC = "user input (a command-line argument)"
FREAD_SRC = "user input (string read by fread)"
# An input reaching a sink: its caption and its line in INPUTS.
ARGV = (ARGV_SRC, 1)
FREAD_HEADER = (FREAD_SRC, 2)
FREAD_PAYLOAD = (FREAD_SRC, 3)


def result(line: int, flows=(ARGV,), rule: str = "cpp/path-injection", uri: str = SRC,
           input_lines=None) -> dict:
    # CodeQL writes one message line per flow when several inputs reach the same
    # sink; each link's number is the id of the related location of its input.
    input_lines = input_lines or [l for _, l in flows]
    return {
        "ruleId": rule,
        "message": {"text": "\n".join(
            f"This argument ... is derived from [{caption}]({i}) and then passed to fopen."
            for i, (caption, _) in enumerate(flows, 1))},
        "locations": [{"physicalLocation": {
            "artifactLocation": {"uri": uri, "uriBaseId": "%SRCROOT%"},
            "region": {"startLine": line},
        }}],
        "relatedLocations": [{
            "id": i,
            "physicalLocation": {"artifactLocation": {"uri": INPUTS, "uriBaseId": "%SRCROOT%"},
                                 "region": {"startLine": l, "startColumn": 3}},
            "message": {"text": caption},
        } for i, ((caption, _), l) in enumerate(zip(flows, input_lines), 1)],
    }


def source(flow) -> dict:
    caption, line = flow
    return {"caption": caption, "file": INPUTS, "code": INPUT_LINES[line - 1].strip()}


def entry(flows=(ARGV,), status: str = "false-positive",
          code: str = OPEN_LINE, **over) -> dict:
    e = {"rule": "cpp/path-injection", "file": SRC, "code": code,
         "sources": [source(f) for f in flows], "status": status, "reason": "reviewed"}
    e.update(over)
    return e


class Gate(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source(["// header", "  " + OPEN_LINE, "int x;"])
        self.source(INPUT_LINES, INPUTS)

    def source(self, lines: list[str], uri: str = SRC) -> None:
        path = self.root / uri
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
        rc, kept, out = self.run_gate([result(3, flows=(), rule="cpp/world-writable-file-creation")],
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
        rc, kept, _ = self.run_gate([result(2, flows=(FREAD_HEADER, ARGV))], [entry()])
        self.assertEqual(rc, 1)
        self.assertEqual(len(kept), 1)

    def test_new_input_with_a_listed_caption_is_new(self):
        # Both freads are "user input (string read by fread)". The entry clears
        # the header read; the payload read reaching the same fopen is new.
        rc, kept, out = self.run_gate([result(2, flows=(FREAD_HEADER, FREAD_PAYLOAD, ARGV))],
                                      [entry(flows=(FREAD_HEADER, ARGV))])
        self.assertEqual(rc, 1)
        self.assertEqual(len(kept), 1)
        self.assertIn("new CodeQL alert", out)

    def test_each_result_matches_the_entry_with_its_sources(self):
        rc, kept, out = self.run_gate(
            [result(2, flows=(FREAD_HEADER, FREAD_PAYLOAD, ARGV)), result(2)],
            [entry(), entry(flows=(FREAD_HEADER, FREAD_PAYLOAD, ARGV))])
        self.assertEqual((rc, kept), (0, []))
        self.assertNotIn("stale entry", out)

    def test_entry_survives_an_input_moving(self):
        self.source(["// added above"] + INPUT_LINES, INPUTS)
        rc, kept, _ = self.run_gate([result(2, flows=(ARGV,), input_lines=[2])], [entry()])
        self.assertEqual((rc, kept), (0, []))

    def test_input_without_a_location_is_new(self):
        unresolved = result(2)
        unresolved["relatedLocations"] = []
        rc, kept, _ = self.run_gate([unresolved], [entry()])
        self.assertEqual(rc, 1)
        self.assertEqual(len(kept), 1)

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

    def test_source_named_by_caption_only_is_refused(self):
        rc, _, out = self.run_gate([result(2)], [entry(sources=[ARGV_SRC])])
        self.assertEqual(rc, 2)
        self.assertIn("needs sources", out)

    def test_unknown_status_is_refused(self):
        rc, _, out = self.run_gate([result(2)], [entry(status="dismissed")])
        self.assertEqual(rc, 2)
        self.assertIn("status 'dismissed'", out)


class CheckedInAlerts(unittest.TestCase):
    def test_every_entry_names_lines_that_exist(self):
        alerts = TOOLS / "codeql-replay-alerts.json"
        repo = TOOLS.parents[2]
        entries = codeql_gate.load_alerts(alerts)
        self.assertTrue(entries)
        for e in entries:
            for named in [e] + e["sources"]:
                lines = [l.strip() for l in (repo / named["file"]).read_text().splitlines()]
                self.assertIn(named["code"].strip(), lines, f"{named['file']}: {named['code']}")


if __name__ == "__main__":
    unittest.main()
