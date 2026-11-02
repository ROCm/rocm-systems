# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""The enable-recording skill knows the archive format it inspects.

`inspect_archive.py verify` finds the fat binaries in an archive by the
`__hipRegisterFatBinary` id of the archive's format version, from a table it
keeps itself, and leaves them unscanned for a version it does not know. A
format bump that misses that table makes every new archive skip the scan
without any error, so the table has to map the current HRR_VERSION to the id
hrr_api_args.h gives that API.

Run: python3 -m unittest discover -s projects/hrr/tools/tests
"""

import importlib.util
import re
import sys
import unittest
from pathlib import Path

HRR = Path(__file__).resolve().parents[2]
API_ARGS = HRR / "include" / "hrr" / "hrr_api_args.h"
INSPECT_ARCHIVE = HRR / "skills" / "enable-recording" / "scripts" / "inspect_archive.py"


def load_inspect_archive():
    spec = importlib.util.spec_from_file_location("inspect_archive", INSPECT_ARCHIVE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look the module up
    spec.loader.exec_module(module)
    return module


class SkillApiIdsTest(unittest.TestCase):
    def test_fat_binary_id_covers_the_current_format_version(self):
        text = API_ARGS.read_text(encoding="utf-8")
        version = int(re.search(r"#define HRR_VERSION \(\(uint16_t\)(\d+)u\)", text).group(1))
        api_id = int(re.search(r"\bHRR_API_HIPREGISTERFATBINARY = (\d+),", text).group(1))
        table = load_inspect_archive().REGISTER_FAT_BINARY
        self.assertEqual(
            table.get(version),
            api_id,
            f"hrr_api_args.h is format v{version} with __hipRegisterFatBinary = {api_id}; "
            "add that to REGISTER_FAT_BINARY in "
            "skills/enable-recording/scripts/inspect_archive.py",
        )


if __name__ == "__main__":
    unittest.main()
