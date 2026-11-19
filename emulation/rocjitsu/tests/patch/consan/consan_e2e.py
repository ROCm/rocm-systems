# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify compiler/application/analyzer handoffs and semantic replay in both environments."""

import argparse
import json
from pathlib import Path
import subprocess
import struct
import sys

from consan_runtime_e2e import (
    CASES,
    Driver,
    HardwareUnavailable,
    add_target_arguments,
    build_root,
    identity,
)


def clear_examples(result):
    """Normalize a result for comparison with --examples 0."""
    for execution in result['executions']:
        execution['cross_wave_examples'] = []
        execution['collision_examples'] = []
        execution['ordered_examples'] = []
        execution['unresolved_examples'] = []


class AnalysisDriver(Driver):
    def prepare_original(self, fixture, original):
        self.entry = super().prepare_original(fixture, original)
        if self.args.utf8_entry:
            data = original.read_bytes()
            old = self.entry.encode('utf-8')
            new = old[:-2] + '\u00e9'.encode('utf-8')
            assert len(old) == len(new) and data.count(old) == 6
            renamed = data.replace(old, new)
            shoff = struct.unpack_from('<Q', data, 40)[0]
            shnum = struct.unpack_from('<H', data, 60)[0]
            for index in range(shnum):
                section = struct.unpack_from('<IIQQQQIIQQ', data, shoff + 64 * index)
                if section[2] & 6 == 6:
                    offset, length = section[4:6]
                    assert (
                        data[offset : offset + length]
                        == renamed[offset : offset + length]
                    )
            original.write_bytes(renamed)
            self.entry = new.decode('utf-8')
            self.inputs['utf8-original'] = identity(original)
        return self.entry

    def execute(self):
        assert build_root(self.args.analyze) == build_root(self.args.prepare)
        self.inputs[str(self.args.analyze)] = identity(self.args.analyze)
        super().execute()
        capture = self.directory / 'capture.bin'
        if not capture.exists():
            # The base driver already asserted the precise pre-submit failure
            # and absence of a successfully published capture.
            return
        assert not (self.directory / 'artifact').exists()
        assert not (self.directory / 'original.hsaco').exists()
        case = self.args.case
        pairs = (
            self.args.workgroups
            if case == 'grouped_race'
            else (
                1
                if case
                in (
                    'race',
                    'high_lane',
                    'mixed_overlap',
                    'mixed_b16',
                    'barrier_missing_second',
                )
                else (2 if case == 'repeat_race' else 0)
            )
        )
        collisions = (
            (self.args.threads // self.args.wave_size)
            if case == 'wide_collision'
            else (1 if case == 'collision' else 0)
        )
        status = 1 if pairs or collisions else 0
        # The dynamic loader's own trace records the analyzer's actual modules
        # without linking a runtime or adding a live-registry input to the tool.
        environment = self.environment | {'LD_DEBUG': 'files'}
        analysis_command = [self.args.analyze, capture]
        if case.startswith('grouped_'):
            analysis_command += ['--examples', 64]
        first = self.run('analyze', analysis_command, status, environment)
        replay = self.run('analyze-replay', analysis_command, status)
        assert first == replay, 'semantic replay changed'
        dependencies = (self.directory / 'analyze.stderr').read_text()
        for forbidden in (
            'librocjitsu.so',
            'libhsa-runtime',
            'libhsa_consan',
            'libhsa_dbt',
            'libamdhip',
        ):
            assert forbidden not in dependencies, f'analyzer loaded {forbidden}'
        result = json.loads(first)
        assert result['target'] == int(self.args.target[3:])
        assert result['wave_size'] == self.args.wave_size
        assert result['entry'] == self.entry, (result['entry'], self.entry)
        assert result['cross_wave_pairs'] == pairs
        assert result['ordered_pairs'] == sum(
            execution['ordered_pairs'] for execution in result['executions']
        )
        assert result['collision_groups'] == collisions
        assert result['coverage_complete'] == (case not in ('exhausted',))
        assert result['verdict'] == ('findings' if status else 'no_observed_findings')
        raw = json.loads((self.directory / 'inspect.stdout').read_text())
        assert result['execution_conditions'] == raw['execution_conditions'] == 15
        assert len(raw['executions']) == len(result['executions'])
        for observed, analyzed in zip(raw['executions'], result['executions']):
            assert analyzed['serial'] == observed['serial']
            assert analyzed['valid_observations'] == sum(
                len(row['records']) for row in observed['rows']
            )
            assert analyzed['rejected_records'] == 0
            assert analyzed['lost_occurrences'] == (
                {
                    'exhausted': 1,
                }.get(case, 0)
            )
            assert analyzed['ordering_uncertain'] == (case in ('exhausted',))
            assert analyzed['cross_wave_pairs'] == (
                self.args.workgroups
                if case == 'grouped_race'
                else (
                    1
                    if case
                    in (
                        'race',
                        'repeat_race',
                        'high_lane',
                        'mixed_overlap',
                        'mixed_b16',
                        'barrier_missing_second',
                    )
                    else 0
                )
            )
            assert analyzed['collision_groups'] == collisions
            if case in ('race', 'repeat_race', 'high_lane'):
                finding = analyzed['cross_wave_examples'][0]
                assert finding['ordering'] == 'unordered'
                mask = (1 << (self.args.wave_size - 1)) if case == 'high_lane' else 1
                assert (
                    finding['first']['wave'] == 0 and finding['first']['mask'] == mask
                )
                assert finding['first']['access'] == 'write'
                assert (
                    finding['second']['wave'] == 1 and finding['second']['mask'] == mask
                )
                assert finding['second']['access'] == 'read'
                assert finding['first']['offset'] == finding['second']['offset'] == 0
                for participant in (finding['first'], finding['second']):
                    site = raw['sites'][participant['site']]
                    assert participant['original_offset'] == site['original_offset']
            if case.startswith('grouped_'):
                groups = self.args.workgroups
                waves = (
                    self.args.threads + self.args.wave_size - 1
                ) // self.args.wave_size
                assert (
                    analyzed['workgroup_count'] == groups
                    and analyzed['waves_per_workgroup'] == waves
                )
                if case == 'grouped_race':
                    examples = analyzed['cross_wave_examples']
                    assert {
                        finding['first']['workgroup_x'] for finding in examples
                    } == set(range(groups))
                    for finding in examples:
                        assert (
                            finding['first']['workgroup_x']
                            == finding['second']['workgroup_x']
                        )
                        assert (
                            finding['first']['wave'] == 0
                            and finding['second']['wave'] == 1
                        )
                        assert finding['ordering'] == 'unordered'
                if case in ('grouped_barrier', 'grouped_empty') or (
                    case == 'grouped_repeat' and self.args.target == 'gfx1250'
                ):
                    assert analyzed['ordered_pairs'] == groups * (waves - 1)
                    assert (
                        analyzed['required_barrier_waves']
                        == analyzed['completed_barrier_waves']
                        == groups * waves
                    )
                    examples = analyzed['ordered_examples']
                    assert {
                        finding['first']['workgroup_x'] for finding in examples
                    } == set(range(groups))
                    for finding in examples:
                        assert (
                            finding['first']['workgroup_x']
                            == finding['second']['workgroup_x']
                        )
                        assert (
                            finding['first']['wave'] == 0
                            and finding['second']['wave'] > 0
                        )
                        assert finding['ordering'] == 'ordered'
                else:
                    assert analyzed['ordered_pairs'] == 0
                    assert analyzed['required_barrier_waves'] == 0
                    assert analyzed['completed_barrier_waves'] == 0
                    assert (
                        result['scope'] == 'native_lds_single_range_no_sync_bounded_1d'
                    )
            if case == 'mixed_barrier':
                assert analyzed['ordered_pairs'] == 1
                assert (
                    analyzed['required_barrier_waves']
                    == analyzed['completed_barrier_waves']
                    == len(observed['rows'])
                )
                finding = analyzed['ordered_examples'][0]
                assert finding['ordering'] == 'ordered'
                assert (finding['first']['offset'], finding['first']['width']) == (0, 8)
                assert (finding['second']['offset'], finding['second']['width']) == (
                    4,
                    4,
                )
            if case.startswith('barrier'):
                assert (
                    result['scope']
                    == 'native_lds_single_range_epoch_workgroup_barriers'
                )
                assert analyzed['ordered_pairs'] == (
                    2 if case == 'barrier_multiple' else 1
                )
                assert analyzed['ordering_unresolved_pairs'] == 0
                assert analyzed['required_barrier_waves'] == len(observed['rows'])
                assert analyzed['completed_barrier_waves'] == len(observed['rows'])
                certificate = analyzed['ordered_examples'][0]
                assert certificate['ordering'] == 'ordered'
                assert certificate['first']['wave'] == 0
                assert certificate['second']['wave'] == 1
                mask = (1 << 31) if case == 'barrier_high' else 1
                assert (
                    certificate['first']['mask']
                    == certificate['second']['mask']
                    == mask
                )
                assert (
                    certificate['first']['offset']
                    == certificate['second']['offset']
                    == 0
                )
            if case in ('barrier_multiple', 'barrier_missing_second'):
                assert [
                    (finding['first']['site'], finding['second']['site'])
                    for finding in analyzed['ordered_examples']
                ] == ([(0, 1), (2, 1)] if case == 'barrier_multiple' else [(0, 1)])
                assert [
                    (finding['first']['site'], finding['second']['site'])
                    for finding in analyzed['cross_wave_examples']
                ] == ([(2, 1)] if case == 'barrier_missing_second' else [])
                if case == 'barrier_missing_second':
                    reuse = analyzed['cross_wave_examples'][0]
                    assert reuse['ordering'] == 'unordered'
                    assert reuse['first']['access'] == 'write'
                    assert reuse['first']['wave'] == 0 and reuse['first']['phase'] == 1
                    assert reuse['second']['access'] == 'read'
                    assert (
                        reuse['second']['wave'] == 1 and reuse['second']['phase'] == 1
                    )
            if case == 'collision':
                finding = analyzed['collision_examples'][0]
                assert finding['first']['mask'] == 3 and finding['lane_pairs'] == 1
                assert analyzed['valid_observations'] == 1
        hidden = json.loads(
            self.run(
                'analyze-no-examples',
                [self.args.analyze, capture, '--examples', 0],
                status,
            )
        )
        clear_examples(result)
        assert (
            result == hidden
        ), 'example limit changed full semantic counts or coverage'
        self.run(
            'analysis-reject-removed-option',
            [self.args.analyze, capture, '--require-complete'],
            2,
        )
        if case == 'disjoint':
            for name in ('truncated.bin', 'missing-facts.bin'):
                self.run(
                    f'analyze-reject-{name}',
                    [self.args.analyze, self.directory / name],
                    2,
                )
            command = [str(self.args.analyze), str(capture)]
            with open('/dev/full', 'wb') as sink:
                failed = subprocess.run(
                    command,
                    env=self.environment,
                    stdout=sink,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=60,
                    check=False,
                )
            self.commands.append(
                {
                    'label': 'analysis-output-failure',
                    'argv': command,
                    'stdout': '/dev/full',
                    'exit_status': failed.returncode,
                }
            )
            (self.directory / 'analysis-output-failure.stderr').write_text(
                failed.stderr
            )
            assert (
                failed.returncode == 3
            ), 'output failure was confused with input/finding status'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        'prepare',
        'runner',
        'inspect',
        'plugin',
        'fixtures',
        'source',
        'root',
        'analyze',
    ):
        parser.add_argument(
            f'--{name}', type=lambda value: Path(value).resolve(), required=True
        )
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--utf8-entry', action='store_true')
    add_target_arguments(parser)
    driver = AnalysisDriver(parser.parse_args())
    status = 1
    try:
        driver.execute()
        status = 0
    except HardwareUnavailable:
        status = 77
        print(
            f'{driver.args.target}/ROCm unavailable; physical coverage skipped',
            file=sys.stderr,
        )
    except Exception as error:
        status = 1
        print(str(error), file=sys.stderr)
    finally:
        (driver.directory / 'result.json').write_text(
            json.dumps(
                {
                    'status': status,
                    'case': driver.args.case,
                    'commands': driver.commands,
                    'identities': driver.inputs,
                },
                indent=2,
            )
            + '\n'
        )
        print(f'evidence: {driver.directory}')
    return status


if __name__ == '__main__':
    sys.exit(main())
