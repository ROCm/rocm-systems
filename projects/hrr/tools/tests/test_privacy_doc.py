# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""PRIVACY.md lists what capture writes into an archive.

The page is the customer text for AIRUNTIME-2733. Its device table has to name
exactly the fields hip_capture_metadata.cpp writes into a manifest, so a task
that adds, removes or blanks one cannot land without updating the page. Its
inventory has to list the active marker, the hipHostRegister snapshot, the
hipLinkAddData inputs and the hipDeviceProp_t snapshots in events.bin.

Run: python3 -m unittest discover -s projects/hrr/tools/tests
"""

import re
import unittest
from pathlib import Path

HRR = Path(__file__).resolve().parents[2]
PRIVACY = HRR / "PRIVACY.md"
CAPTURE_SRC = HRR.parent / "clr" / "hipamd" / "src" / "hrr"
METADATA_CPP = CAPTURE_SRC / "hip_capture_metadata.cpp"

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


def inventory_rows() -> list:
    """The rows of the page's "What an archive contains" table."""
    text = PRIVACY.read_text(encoding="utf-8")
    table = text.split("| Item | Where |", 1)[1].split("\n\n", 1)[0]
    return table.splitlines()[2:]


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

    def test_inventory_lists_the_active_marker_capture_writes(self):
        writer = (CAPTURE_SRC / "hip_capture_writer.cpp").read_text(encoding="utf-8")
        name = re.search(r'kActiveMarker = "/?(\w+)"', writer).group(1)
        self.assertTrue(
            any(row.split("|")[2].strip() == f"`{name}`" for row in inventory_rows())
        )

    def assert_blob_row(self, api: str) -> tuple:
        """api's capture wrapper writes a blob and the inventory has its row."""
        capture = (CAPTURE_SRC / "hip_capture.cpp").read_text(encoding="utf-8")
        body = capture.split(f"hipError_t capture_{api}(", 1)[1]
        body = body.split("\n}\n", 1)[0]
        self.assertIn("write_blob(", body)
        rows = [r for r in inventory_rows() if f"`{api}`" in r and "`blobs/`" in r]
        self.assertTrue(rows, api)
        return body, rows[0]

    def test_inventory_lists_the_host_register_snapshot(self):
        self.assert_blob_row("hipHostRegister")

    def test_inventory_lists_linker_inputs_rejected_ones_included(self):
        body, row = self.assert_blob_row("hipLinkAddData")
        # The row says rejected inputs are recorded: true while the wrapper
        # does not gate on the call's result.
        self.assertNotIn("hipSuccess", body)
        self.assertIn("rejected", row)

    def test_lists_device_property_snapshots_outside_the_manifest(self):
        # Every shim that copies a caller's hipDeviceProp_t into its event
        # records the device name, PCI ids, UUID and LUID in events.bin.
        shims = (CAPTURE_SRC / "hip_capture_generated.cpp").read_text(
            encoding="utf-8"
        ) + (CAPTURE_SRC / "hip_capture.cpp").read_text(encoding="utf-8")
        apis = set()
        for shim in re.split(r"\n(?=(?:static )?\S+\*? capture_)", shims):
            name = re.match(r"(?:static )?\S+\*? capture_(\w+)\(", shim)
            if name and re.search(r"memcpy\(a\.\w+, \w+, sizeof\(hipDeviceProp", shim):
                apis.add(re.sub(r"R\d{4}$", "", name.group(1)))
        self.assertIn("hipChooseDevice", apis)
        text = PRIVACY.read_text(encoding="utf-8")
        strings = text.split("## Never recorded", 1)[1].split("\n## ", 1)[0]
        for api in apis:
            rows = [r for r in inventory_rows() if f"`{api}`" in r]
            self.assertTrue(any("`events.bin`" in r and "UUID" in r for r in rows), api)
            self.assertIn(f"`{api}`", strings)

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
