# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""PRIVACY.md lists what capture writes into an archive.

The page is the customer text for AIRUNTIME-2733. Its device table has to name
exactly the fields hip_capture_metadata.cpp writes into a manifest, so a task
that adds, removes or blanks one cannot land without updating the page.

Run: python3 -m unittest discover -s projects/hrr/tools/tests
"""

import re
import unittest
from pathlib import Path

HRR = Path(__file__).resolve().parents[2]
PRIVACY = HRR / "PRIVACY.md"
METADATA_CPP = (
    HRR.parent / "clr" / "hipamd" / "src" / "hrr" / "hip_capture_metadata.cpp"
)

# The tasks under AIRUNTIME-2733 that the page links, one per statement.
EPIC_TASKS = (
    "AIRUNTIME-2847",
    "AIRUNTIME-2857",
    "AIRUNTIME-2858",
    "AIRUNTIME-2878",
    "AIRUNTIME-2880",
    "AIRUNTIME-2881",
)


def device_fields_written() -> set:
    """The keys append_prop_fields writes into each manifest device block."""
    text = METADATA_CPP.read_text(encoding="utf-8")
    body = text.split("void append_prop_fields(", 1)[1].split("\n}\n", 1)[0]
    return set(re.findall(r'\\"(\w+)\\": ', body)) - {"properties"}


def device_fields_documented() -> set:
    """The backticked names in the first column of the page's device table."""
    text = PRIVACY.read_text(encoding="utf-8")
    table = text.split("| Device field |", 1)[1].split("\n\n", 1)[0]
    fields = set()
    for row in table.splitlines()[2:]:
        fields.update(re.findall(r"`(\w+)`", row.split("|")[1]))
    return fields


class PrivacyDocTest(unittest.TestCase):
    def test_device_table_matches_the_manifest_capture_writes(self):
        written = device_fields_written()
        self.assertIn("uuid", written)
        self.assertEqual(device_fields_documented(), written)

    def test_states_that_an_archive_is_untrusted_executable_input(self):
        text = PRIVACY.read_text(encoding="utf-8")
        self.assertIn("untrusted executable input", text)
        self.assertIn("launches their kernels on the GPU", text)

    def test_links_each_epic_task(self):
        text = PRIVACY.read_text(encoding="utf-8")
        for key in EPIC_TASKS:
            self.assertIn(f"https://amd-hub.atlassian.net/browse/{key}", text)

    def test_readme_and_recording_skill_link_the_page(self):
        readme = (HRR / "README.md").read_text(encoding="utf-8")
        skill = (HRR / "skills" / "enable-recording" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("](PRIVACY.md)", readme)
        self.assertIn("](../../PRIVACY.md)", skill)


if __name__ == "__main__":
    unittest.main()
