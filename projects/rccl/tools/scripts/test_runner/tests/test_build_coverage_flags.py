import unittest

from lib.test_executor import configure_coverage_build


class CoverageBuildFlagsTest(unittest.TestCase):
    def test_coverage_report_selects_auto_full_coverage(self):
        for seed_flag in ("--enable-full-coverage",
                          "--enable-code-coverage"):
            with self.subTest(seed_flag=seed_flag):
                flags, cmake_options = configure_coverage_build(
                    [seed_flag, "--no_clean"],
                    "-DFOO=ON -DENABLE_CODE_COVERAGE=OFF -DENABLE_FULL_COVERAGE=OFF",
                    coverage_report=True,
                )

                self.assertIn("--debug", flags)
                self.assertIn("--enable-code-coverage", flags)
                self.assertNotIn("--enable-full-coverage", flags)
                self.assertTrue(cmake_options.endswith(
                    "-DENABLE_CODE_COVERAGE=ON -DENABLE_FULL_COVERAGE=AUTO"
                ))
                self.assertNotIn("-DENABLE_FULL_COVERAGE=OFF", cmake_options)
                self.assertNotIn("-DENABLE_CODE_COVERAGE=OFF", cmake_options)

    def test_non_coverage_build_disables_cached_coverage_options(self):
        original_flags = [
            "--enable-code-coverage",
            "--enable-full-coverage",
            "--no_clean",
        ]

        flags, cmake_options = configure_coverage_build(
            original_flags,
            "-DFOO=ON",
            coverage_report=False,
        )

        self.assertEqual(original_flags, [
            "--enable-code-coverage",
            "--enable-full-coverage",
            "--no_clean",
        ])
        self.assertNotIn("--enable-code-coverage", flags)
        self.assertNotIn("--enable-full-coverage", flags)
        self.assertTrue(cmake_options.endswith(
            "-DENABLE_CODE_COVERAGE=OFF -DENABLE_FULL_COVERAGE=OFF"
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
