# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / ".github" / "therock_ref.json"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
WRAPPER_NAMES = {
    "_therock_setup_multi_arch.yml",
    "_therock_multi_arch_ci_linux.yml",
    "_therock_multi_arch_ci_windows.yml",
}

USES_SHA_RE = re.compile(r"uses:\s*ROCm/TheRock/\S+@([0-9a-f]{40})")
REF_LINE_RE = re.compile(r"^\s*ref:\s*([0-9a-f]{40})\s*(?:#.*)?$")
THEROCK_REPO_RE = re.compile(r'repository:\s*"?ROCm/TheRock"?\s*$')

# How many lines around a `ref: <sha>` line to scan for a sibling
# `repository: ROCm/TheRock` entry in the same `with:` block.
PROXIMITY = 4


class TheRockRefDriftTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text())
        self.expected_ref = self.config["ref"]

    def test_wrapper_files_exist(self):
        for name in WRAPPER_NAMES:
            path = WORKFLOWS_DIR / name
            self.assertTrue(path.is_file(), f"Missing wrapper workflow: {path}")

    def test_wrapper_shas_match_config(self):
        for name in WRAPPER_NAMES:
            text = (WORKFLOWS_DIR / name).read_text()

            uses_matches = USES_SHA_RE.findall(text)
            self.assertEqual(
                len(uses_matches),
                1,
                f"Expected exactly one 'uses: ROCm/TheRock/...@<sha>' in {name}, "
                f"found {len(uses_matches)}",
            )
            self.assertEqual(
                uses_matches[0],
                self.expected_ref,
                f"{name}'s 'uses:' pin does not match .github/therock_ref.json",
            )

            ref_matches = [
                m.group(1) for line in text.splitlines() if (m := REF_LINE_RE.match(line))
            ]
            self.assertEqual(
                len(ref_matches),
                1,
                f"Expected exactly one 'ref: <sha>' in {name}, found {len(ref_matches)}",
            )
            self.assertEqual(
                ref_matches[0],
                self.expected_ref,
                f"{name}'s 'ref:' pin does not match .github/therock_ref.json",
            )

    def _find_therock_ref_pins(self, lines: list[str]) -> list[int]:
        """Line numbers (0-based) of a `ref: <sha>` near a ROCm/TheRock repository line."""
        hits = []
        for i, line in enumerate(lines):
            if not REF_LINE_RE.match(line):
                continue
            window = lines[max(0, i - PROXIMITY) : i + PROXIMITY + 1]
            if any(THEROCK_REPO_RE.search(w) for w in window):
                hits.append(i)
        return hits

    def test_no_stray_therock_pins_outside_wrappers(self):
        offenders = []
        for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            if path.name in WRAPPER_NAMES:
                continue
            text = path.read_text()

            if USES_SHA_RE.search(text):
                offenders.append(
                    f"{path.name}: hardcoded 'uses: ROCm/TheRock/...@<sha>'"
                )

            lines = text.splitlines()
            for line_no in self._find_therock_ref_pins(lines):
                offenders.append(
                    f"{path.name}:{line_no + 1}: hardcoded TheRock 'ref: <sha>' "
                    "(should use steps.therock_ref.outputs.ref)"
                )

        self.assertEqual(
            offenders,
            [],
            "Found hardcoded TheRock commit pins outside the allowed wrapper "
            f"workflows ({', '.join(sorted(WRAPPER_NAMES))}):\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
