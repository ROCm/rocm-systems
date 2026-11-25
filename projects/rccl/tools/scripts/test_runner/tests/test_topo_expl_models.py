#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""Each topo_expl model <cpu> carries the CPUID family a live host writes (AMD 23/25/26, Intel 6), not 143/175/191."""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

_MODELS = Path(__file__).resolve().parents[3] / "topo_expl" / "models"

_FAMILIES = {
    "AuthenticAMD": {23, 25, 26},
    "GenuineIntel": {6},
}


def _cpus(path):
    return [node for node in ET.parse(path).iter("cpu") if "vendor" in node.attrib]


class TopoExplModelCpuTest(unittest.TestCase):
    def test_models_exist(self):
        self.assertTrue(sorted(_MODELS.glob("*.xml")), _MODELS)

    def test_cpu_family_uses_the_cpuid_encoding(self):
        for path in sorted(_MODELS.glob("*.xml")):
            for cpu in _cpus(path):
                vendor = cpu.attrib["vendor"]
                if vendor not in _FAMILIES:
                    continue
                expected = sorted(_FAMILIES[vendor])
                with self.subTest(model=path.name, numaid=cpu.get("numaid")):
                    self.assertIn("familyid", cpu.attrib, f"{path.name}: {vendor} <cpu> has no familyid")
                    self.assertIn("modelid", cpu.attrib, f"{path.name}: {vendor} <cpu> has no modelid")
                    family = int(cpu.attrib["familyid"], 0)
                    self.assertIn(
                        family,
                        _FAMILIES[vendor],
                        f"{path.name}: {vendor} <cpu numaid={cpu.get('numaid')}> has familyid={family}; "
                        f"expected the CPUID family that topology detection writes ({expected}), "
                        "not family + (extFamily << 4) (AMD 143/175/191)",
                    )


if __name__ == "__main__":
    unittest.main()
