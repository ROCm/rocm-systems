"""Contract checks for selection and failure reporting without ROCm hardware."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'systems_quick_parity.py'
spec = importlib.util.spec_from_file_location('systems_quick_parity', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class QuickParityTest(unittest.TestCase):
    def invoke(self, empty=False):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / 'rocm'
            for path in ['lib/librocm-dbgapi.so',
                         'TheRock/build_tools/github_actions/test_executable_scripts/test_rocrtst.py']:
                target = (root if path.startswith('lib/') else base) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("QUICK_TESTS = ['rocrtst.Test_Example']\n")
            calls = []

            def run(command, **kwargs):
                command = list(map(str, command))
                calls.append(command)
                return types.SimpleNamespace(returncode=0, stdout=json.dumps(
                    {'tests': [] if empty else [{'name': 'selected'}]}))

            with patch.object(runner.subprocess, 'run', side_effect=run), \
                 patch('sys.argv', [str(SCRIPT), '--root', str(root), '--logs', str(base / 'logs')]), \
                 patch.object(runner.Path, 'cwd', return_value=base):
                # TheRock harness path is relative to the checkout root.
                import os
                old = os.getcwd()
                try:
                    os.chdir(base)
                    status = runner.main()
                finally:
                    os.chdir(old)
            results = json.loads((base / 'logs/hw-suite-status.json').read_text())
            xmls = list((base / 'logs').glob('hw-*-execution.xml'))
            return status, results, calls, len(xmls)

    def test_all_six_suites_run_with_explicit_quick_selection(self):
        status, results, calls, xml_count = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(len(results), 6)
        self.assertEqual(xml_count, 6)
        ctests = [cmd for cmd in calls if cmd[0] == 'ctest' and '--show-only=json-v1' not in cmd]
        self.assertEqual(len(ctests), 4)
        self.assertTrue(all('--no-tests=error' in cmd for cmd in ctests))
        self.assertTrue(any('rocprofiler_sdk[.]unit[.]' in cmd for cmd in ctests))
        self.assertTrue(any('--max-retries' in cmd for cmd in calls))
        self.assertTrue(any('--gtest_filter=rocrtst.Test_Example' in cmd for cmd in calls))

    def test_empty_ctest_selection_fails_but_attempts_remaining_suites(self):
        status, results, calls, xml_count = self.invoke(empty=True)
        self.assertEqual(status, 1)
        self.assertEqual(len(results), 6)
        self.assertEqual(xml_count, 6)
        for name in ['hip-tests', 'rocprofiler-sdk', 'rocprofiler-compute', 'rocprofiler-systems']:
            self.assertNotEqual(results[name], 0)
        self.assertEqual(results['rocr-debug-agent'], 0)


if __name__ == '__main__':
    unittest.main()
