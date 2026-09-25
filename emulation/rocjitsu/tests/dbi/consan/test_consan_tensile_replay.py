import tempfile
import json
import unittest
from pathlib import Path
from unittest import mock
import consan_tensile_replay as replay

class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.wrapper = self.root / 'wrapper'
        self.contract = {'wrapper': str(self.wrapper), 'target': 'gfx1250', 'shape': [127]}
        self.ini = self.root / 'ClientParameters.ini'
        self.co = self.root / 'kernel.co'
        self.co.write_bytes(b'original exact ELF')
        self.library = self.root / 'library.yaml'
        self.library.write_text('original library')
        self.ini.write_text(f'library-file={self.library}\ncode-object={self.co}\nproblem-size=127,127,1,127\nresults-file=old.csv\n')
        self.script = self.root / '1_BenchmarkProblems/a/build/run.sh'
        self.script.parent.mkdir(parents=True)
        self.script.write_text(f'{self.wrapper} --config-file {self.ini}\n')
        self.manifest = replay.freeze(self.root, self.wrapper, self.contract)
        self.output = self.root / 'fresh'
        self.output.mkdir()

    def test_snapshot_retains_exact_input_bytes_after_source_changes(self):
        path = self.root / 'manifest.json'
        snapshot = self.output / 'snapshot.json'
        raw = json.dumps(self.manifest, indent=2).encode() + b'\n'
        path.write_bytes(raw)
        loaded = replay.snapshot_manifest(path, snapshot)
        path.write_text('{}')
        self.assertEqual(loaded, self.manifest)
        self.assertEqual(snapshot.read_bytes(), raw)
        self.assertNotEqual(replay.digest(path), replay.digest(snapshot))

    def test_preserves_inputs_and_redirects_results(self):
        before = self.ini.read_text()
        command = mock.Mock(return_value=(0, 'numeric output', False))
        code, output, timed = replay.run(self.manifest, self.contract, self.output, {}, 10, command)
        self.assertEqual((code, timed), (0, False))
        self.assertIn('clientExit=0 (PASS)', output)
        args = command.call_args.args
        self.assertEqual(args[0][:2], [str(self.wrapper), '--config-file'])
        copied = Path(args[0][2]).read_text()
        self.assertIn(f'code-object={self.co}', copied)
        self.assertIn('problem-size=127,127,1,127', copied)
        self.assertIn(f'results-file={self.output}/client-0.csv', copied)
        self.assertEqual(self.ini.read_text(), before)

    def test_reject_changed_inputs_before_execution(self):
        for path in [self.ini, self.co, self.library, self.script]:
            with self.subTest(path=path):
                before = path.read_bytes()
                path.write_bytes(before + b'changed')
                command = mock.Mock()
                with self.assertRaises(ValueError):
                    replay.run(self.manifest, self.contract, self.output, {}, 10, command)
                command.assert_not_called()
                path.write_bytes(before)

    def test_reject_changed_target_or_shape(self):
        for changed in [{'target': 'gfx950'}, {'shape': [128]}]:
            with self.assertRaises(ValueError):
                replay.verify(self.manifest, {**self.contract, **changed})

    def test_reject_mutation_during_execution(self):
        def mutate(*args):
            self.co.write_bytes(b'changed during execution')
            return 0, 'output', False
        with self.assertRaises(ValueError):
            replay.run(self.manifest, self.contract, self.output, {}, 10, mutate)

    def test_failure_or_timeout_never_gets_pass_marker(self):
        for result in [(1, 'numeric failure', False), (-15, 'partial', True)]:
            actual = replay.run(self.manifest, self.contract, self.output, {}, 10, mock.Mock(return_value=result))
            self.assertEqual((actual[0], actual[2]), (result[0], result[2]))
            self.assertNotIn('clientExit=0 (PASS)', actual[1])

    def test_one_deadline_shared_by_all_clients(self):
        second = self.script.parent.parent.parent / 'b/build/run.sh'
        second.parent.mkdir(parents=True)
        ini2 = self.root / 'second.ini'
        ini2.write_text(self.ini.read_text())
        second.write_text(f'{self.wrapper} --config-file {ini2}\n')
        manifest = replay.freeze(self.root, self.wrapper, self.contract)
        command = mock.Mock(return_value=(0, 'first', False))
        with mock.patch.object(replay.time, 'monotonic', side_effect=[0, 1, 11]):
            result = replay.run(manifest, self.contract, self.output, {}, 10, command)
        self.assertTrue(result[2])
        self.assertEqual(command.call_count, 1)

if __name__ == '__main__':
    unittest.main()
