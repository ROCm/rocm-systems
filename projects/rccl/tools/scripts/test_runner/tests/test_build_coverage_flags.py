import unittest

from lib.test_executor import configure_coverage_build


class CoverageBuildFlagsTest(unittest.TestCase):
    def test_coverage_report_selects_auto_device_coverage(self):
        flags, cmake_options = configure_coverage_build(
            ["--enable-device-coverage", "--no_clean"],
            "-DFOO=ON -DENABLE_DEVICE_COVERAGE=OFF",
            coverage_report=True,
        )

        self.assertIn("--debug", flags)
        self.assertIn("--enable-code-coverage", flags)
        self.assertNotIn("--enable-device-coverage", flags)
        self.assertTrue(cmake_options.endswith(
            "-DENABLE_CODE_COVERAGE=ON -DENABLE_DEVICE_COVERAGE=AUTO"
        ))

    def test_coverage_report_replaces_host_only_install_flag(self):
        flags, cmake_options = configure_coverage_build(
            ["--enable-code-coverage", "--no_clean"],
            "-DFOO=ON -DENABLE_DEVICE_COVERAGE=OFF",
            coverage_report=True,
        )

        self.assertIn("--debug", flags)
        self.assertIn("--enable-code-coverage", flags)
        self.assertNotIn("--enable-device-coverage", flags)
        self.assertTrue(cmake_options.endswith(
            "-DENABLE_CODE_COVERAGE=ON -DENABLE_DEVICE_COVERAGE=AUTO"
        ))

    def test_non_coverage_build_disables_cached_coverage_options(self):
        original_flags = [
            "--enable-code-coverage",
            "--enable-device-coverage",
            "--no_clean",
        ]

        flags, cmake_options = configure_coverage_build(
            original_flags,
            "-DFOO=ON",
            coverage_report=False,
        )

        self.assertEqual(original_flags, [
            "--enable-code-coverage",
            "--enable-device-coverage",
            "--no_clean",
        ])
        self.assertNotIn("--enable-code-coverage", flags)
        self.assertNotIn("--enable-device-coverage", flags)
        self.assertTrue(cmake_options.endswith(
            "-DENABLE_CODE_COVERAGE=OFF -DENABLE_DEVICE_COVERAGE=OFF"
        ))

    def test_coverage_preserves_debug_fast_without_duplicate_debug_flag(self):
        flags, _ = configure_coverage_build(
            ["--debug-fast"],
            "",
            coverage_report=True,
        )

        self.assertIn("--debug-fast", flags)
        self.assertNotIn("--debug", flags)

if __name__ == "__main__":
    unittest.main()
