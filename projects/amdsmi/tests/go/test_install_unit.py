# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import unittest

import test_install


class HeaderVersionTests(unittest.TestCase):
    def test_supported_header_baseline(self) -> None:
        self.assertTrue(callable(getattr(test_install, "header_version", None)))
        for major, minor in ((27, 2), (27, 3)):
            text = "#define AMDSMI_LIB_VERSION_MAJOR {}\n#define AMDSMI_LIB_VERSION_MINOR {}\n"
            self.assertEqual(test_install.header_version(text.format(major, minor)), (major, minor))
        for text in (
            "#define AMDSMI_LIB_VERSION_MAJOR 27\n#define AMDSMI_LIB_VERSION_MINOR 1\n",
            "#define AMDSMI_LIB_VERSION_MAJOR 28\n#define AMDSMI_LIB_VERSION_MINOR 2\n",
            "#define AMDSMI_LIB_VERSION_MAJOR 27\n",
            "",
        ):
            with self.subTest(header=text), self.assertRaises(ValueError):
                test_install.header_version(text)


if __name__ == "__main__":
    unittest.main()
