# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise separate preparation, real HSA recording, and capture-reader processes."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time

# Row publication flags from the shared ConSan wire contract.
ROW_ENTERED = 1
ROW_EXITED = 2
ROW_EXHAUSTED = 4

CASES = (
    'race',
    'disjoint',
    'read_read',
    'collision',
    'single_lane',
    'repeat',
    'repeat_race',
    'alias',
    'immediate',
    'four_ranges',
    'exhausted',
    'empty',
    'branched',
    'high_lane',
    'barrier',
    'barrier_empty',
    'barrier_high',
    'barrier_separated',
    'barrier_multiple',
    'barrier_missing_second',
    'wide_u16',
    'wide_i16',
    'wide_b64',
    'wide_b128',
    'wide_u16_high',
    'wide_i16_high',
    'wide_b64_high',
    'wide_b128_high',
    'wide_alias',
    'wide_collision',
    'mixed_overlap',
    'mixed_b16',
    'mixed_disjoint',
    'mixed_read_read',
    'mixed_barrier',
    'grouped_disjoint',
    'grouped_barrier',
    'grouped_empty',
    'grouped_race',
    'grouped_read_read',
    'grouped_isolation',
    'grouped_repeat',
    'grouped_oversized',
    'grouped_edge',
    'grouped_missing_id',
    'single_missing_id',
    'wrong-artifact',
    'bad-geometry',
    'profiling',
    'reused-signal',
    'missing-statistics',
    'batch-barrier-and',
    'batch-barrier-or',
    'late-barrier-and',
    'late-barrier-or',
)


def identity(path):
    content = path.read_bytes()
    return {
        'path': str(path),
        'bytes': len(content),
        'sha256': hashlib.sha256(content).hexdigest(),
    }


def build_root(path):
    for parent in path.parents:
        if (parent / 'CMakeCache.txt').is_file():
            return parent
    raise AssertionError(f'no worktree CMake build owns {path}')


class Driver:
    def __init__(self, args):
        self.args = args
        args.root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.directory = Path(
            tempfile.mkdtemp(prefix=f'{stamp}-runtime-{args.case}-', dir=args.root)
        )
        self.commands = []
        self.inputs = {}
        self.environment = os.environ.copy()
        for name in (
            'HSA_TOOLS_LIB',
            'RJ_CONSAN_PLUGIN_PATH',
            'RJ_CONSAN_ARTIFACT_DIR',
            'RJ_CONSAN_CAPTURE_PATH',
            'CONSAN_PROCESS_MAPS',
        ):
            self.environment.pop(name, None)

    def run(self, label, command, expected=0, environment=None):
        argv = [str(item) for item in command]
        if label in ('application', 'original-baseline'):
            argv += [
                '--target',
                self.args.target,
                '--wave-size',
                str(self.args.wave_size),
                '--threads',
                str(self.args.threads),
                '--workgroups',
                str(self.args.workgroups),
            ]
            if self.args.amd_signals:
                argv.append('--amd-signals')
            if self.args.case.startswith('wide_') and self.args.case.endswith('_high'):
                argv.append('--high-lane')
            if self.args.launcher:
                argv = [
                    str(self.args.launcher),
                    '--config',
                    str(self.args.config),
                    '--',
                    *argv,
                ]
        started = time.perf_counter()
        result = subprocess.run(
            argv,
            env=environment or self.environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        (self.directory / f'{label}.stdout').write_text(result.stdout)
        (self.directory / f'{label}.stderr').write_text(result.stderr)
        self.commands.append(
            {
                'label': label,
                'argv': argv,
                'exit_status': result.returncode,
                'elapsed_seconds': time.perf_counter() - started,
            }
        )
        if result.returncode == 77:
            if self.args.require_agent:
                raise AssertionError(
                    f'{label}: required simulated {self.args.target} agent unavailable'
                )
            raise HardwareUnavailable()
        assert (
            result.returncode == expected
        ), f'{label}: expected exit {expected}, got {result.returncode}\n{result.stderr}'
        return result.stdout

    def execute(self):
        args = self.args
        binaries = [args.prepare, args.runner, args.inspect, args.plugin]
        fixture_name = {
            'repeat': 'disjoint',
            'repeat_race': 'race',
            'grouped_repeat': (
                'grouped_barrier' if args.target == 'gfx1250' else 'grouped_disjoint'
            ),
            'grouped_oversized': 'grouped_disjoint',
            'grouped_edge': 'grouped_disjoint',
            'single_missing_id': 'disjoint',
            'wrong-artifact': 'race',
            'bad-geometry': 'disjoint',
            'profiling': 'disjoint',
            'reused-signal': 'disjoint',
            'missing-statistics': 'disjoint',
            'batch-barrier-and': 'disjoint',
            'batch-barrier-or': 'disjoint',
            'late-barrier-and': 'disjoint',
            'late-barrier-or': 'disjoint',
        }.get(args.case, args.case)
        fixture_directory = (
            args.fixtures / 'targets' / args.target / f'wave{args.wave_size}'
        )
        if args.fixture_kind == 'named-assembly':
            fixture = fixture_directory / f'{fixture_name}.hsaco'
        elif fixture_name.startswith('wide_'):
            name = {'wide_alias': 'b128-alias', 'wide_collision': 'b64-collision'}.get(
                fixture_name, fixture_name.removeprefix('wide_').replace('_', '-')
            )
            fixture = fixture_directory / f'width-{name}.hsaco'
        elif fixture_name.startswith('mixed_'):
            number = {
                'mixed_overlap': 1,
                'mixed_b16': 2,
                'mixed_disjoint': 3,
                'mixed_read_read': 4,
                'mixed_barrier': 5,
            }[fixture_name]
            fixture = fixture_directory / f'mixed-{number}.hsaco'
        elif args.case == 'single_missing_id':
            fixture = fixture_directory / 'native-no-workgroup-id.hsaco'
        elif fixture_name.startswith('grouped_'):
            name = (
                'native-no-workgroup-id'
                if fixture_name == 'grouped_missing_id'
                else 'group-'
                + str(
                    {
                        'grouped_disjoint': 1,
                        'grouped_barrier': 2,
                        'grouped_race': 3,
                        'grouped_read_read': 4,
                        'grouped_empty': 5,
                        'grouped_isolation': 6,
                    }[fixture_name]
                )
            )
            fixture = fixture_directory / f'{name}.hsaco'
        else:
            case_number = {
                'race': 1,
                'disjoint': 2,
                'read_read': 3,
                'alias': 6,
                'high_lane': 14,
                'barrier': 15,
                'barrier_empty': 16,
                'barrier_high': 20,
                'barrier_separated': 21,
                'barrier_multiple': 24,
                'barrier_missing_second': 27,
            }[fixture_name]
            prefix = 'hip' if args.fixture_kind == 'hip' else 'native'
            fixture = fixture_directory / f'{prefix}-{case_number}.hsaco'
        if args.launcher:
            binaries.append(args.launcher)
            assert args.config.is_file()
        roots = {build_root(path) for path in [*binaries, fixture]}
        assert len(roots) == 1, 'mixed build trees'
        cache = (roots.pop() / 'CMakeCache.txt').read_text()
        assert (
            f'CMAKE_HOME_DIRECTORY:INTERNAL={args.source}' in cache
        ), 'wrong source checkout'
        for path in [*binaries, fixture]:
            self.inputs[str(path)] = identity(path)
        original = self.directory / 'original.hsaco'
        entry = self.prepare_original(fixture, original)
        artifact = self.directory / 'artifact'
        launches = (
            2
            if args.case
            in (
                'repeat',
                'repeat_race',
                'branched',
                'reused-signal',
                'grouped_repeat',
                'barrier_multiple',
            )
            else 1
        )
        defined = fixture_name in ('disjoint', 'alias')
        synchronized = fixture_name.startswith('barrier')
        mode = 'synchronized' if synchronized else ('defined' if defined else 'observe')
        wide = fixture_name.startswith('wide_')
        if wide:
            mode = (
                'wide_b128'
                if fixture_name == 'wide_alias'
                else (
                    'wide_b64'
                    if fixture_name == 'wide_collision'
                    else fixture_name.removesuffix('_high')
                )
            )
        if fixture_name == 'mixed_barrier':
            mode = 'mixed'
        grouped = fixture_name in (
            'grouped_disjoint',
            'grouped_barrier',
            'grouped_empty',
        )
        if grouped:
            mode = 'grouped' if fixture_name == 'grouped_disjoint' else 'grouped_sync'
        # The missing-statistics diagnostic precedes device admission in the
        # runner. Check the requested agent first so an unavailable physical
        # target cannot pass that control without a matching device.
        if (
            synchronized
            or wide
            or fixture_name == 'mixed_barrier'
            or (grouped and args.case not in ('grouped_oversized', 'grouped_edge'))
            or (
                defined
                and args.case
                in (
                    'disjoint',
                    'alias',
                    'repeat',
                    'single_missing_id',
                    'missing-statistics',
                )
            )
        ):
            self.run(
                'original-baseline',
                [args.runner, original, entry, launches, mode, '--baseline'],
            )
        self.run('prepare', [args.prepare, original, args.target, entry, artifact])
        for name in ('prepared.hsaco', 'facts.bin', 'ready'):
            self.inputs[name] = identity(artifact / name)
        capture = self.directory / 'capture.bin'
        environment = self.environment | {
            'HSA_TOOLS_LIB': str(args.plugin),
            'RJ_CONSAN_PLUGIN_PATH': str(args.plugin),
            'RJ_CONSAN_ARTIFACT_DIR': str(artifact),
            'RJ_CONSAN_CAPTURE_PATH': str(capture),
            'CONSAN_PROCESS_MAPS': str(self.directory / 'process-maps.txt'),
        }
        options = {
            'bad-geometry': ['--bad-geometry'],
            'grouped_edge': ['--bad-geometry'],
            'profiling': ['--profiling'],
            'reused-signal': ['--reuse-signal'],
            'batch-barrier-and': ['--batch-barrier-and'],
            'batch-barrier-or': ['--batch-barrier-or'],
            'late-barrier-and': [
                '--batch-barrier-and',
                '--separate-barrier',
                '--bad-geometry',
            ],
            'late-barrier-or': [
                '--batch-barrier-or',
                '--separate-barrier',
                '--bad-geometry',
            ],
        }.get(args.case, [])
        if args.test_statistics:
            options.append('--test-statistics')
        if args.case == 'missing-statistics':
            assert not args.test_statistics
            self.run(
                'application',
                [args.runner, original, entry, launches, mode, '--test-statistics'],
                3,
                environment,
            )
            assert (
                'required ConSan test statistics interface is unavailable'
                in (self.directory / 'application.stderr').read_text()
            )
            assert not capture.exists(), 'missing test API published a capture'
            return
        expected_status = (
            20
            if args.case == 'wrong-artifact'
            else (
                21
                if args.case
                in (
                    'bad-geometry',
                    'grouped_edge',
                    'grouped_oversized',
                    'grouped_missing_id',
                    'reused-signal',
                    'batch-barrier-and',
                    'batch-barrier-or',
                    'late-barrier-and',
                    'late-barrier-or',
                )
                else 0
            )
        )
        if args.case == 'wrong-artifact':
            # A different, valid original image safely exercises exact matching.
            wrong = fixture_directory / 'disjoint.hsaco'
            self.inputs['mismatched-original'] = identity(wrong)
            shutil.copyfile(wrong, original)
        output = self.run(
            'application',
            [args.runner, original, entry, launches, mode, *options],
            expected_status,
            environment,
        )
        statistics = json.loads(output.splitlines()[-1])
        assert statistics['app_status'] == expected_status
        if expected_status:
            if args.case.startswith(('batch-barrier-', 'late-barrier-')):
                assert statistics['barrier_completion'] == -1
            count = 1 if args.case == 'reused-signal' else 0
            if args.test_statistics:
                assert statistics['submitted'] == count
            assert statistics['completed'] == count
            assert statistics['finish_status'] != 0
            assert not capture.exists(), 'failed run published a successful capture'
            return
        assert statistics['finish_status'] == 0
        assert statistics['completed'] == launches
        counter_fields = ('submitted', 'forwarded', 'copied', 'written', 'reused')
        if args.test_statistics:
            for field in counter_fields[:-1]:
                assert statistics[field] == launches, (field, statistics)
            if launches > 1:
                assert statistics['reused'] == launches - 1
        else:
            assert not set(counter_fields) & statistics.keys(), statistics
        if args.case == 'profiling':
            assert statistics['profiling_status'] != 0
        maps = (self.directory / 'process-maps.txt').read_text()
        assert str(args.plugin) in maps, 'runtime DSO was not loaded'
        for forbidden in (
            ('libhsa_dbt.so', 'libhsa_dbi.so')
            if args.launcher
            else ('librocjitsu.so', 'libhsa_dbt.so', 'libhsa_dbi.so')
        ):
            assert forbidden not in maps, f'unexpected runtime dependency {forbidden}'
        # Consumer has no path to the runtime artifact store or original object.
        artifact.rename(self.directory / 'retired-artifact')
        original.rename(self.directory / 'retired-original.hsaco')
        result = self.run('inspect', [args.inspect, capture])
        assert result == self.run('inspect-replay', [args.inspect, capture])
        decoded = json.loads(result)
        assert decoded['execution_conditions'] == 15
        assert len(decoded['executions']) == launches
        self.assert_observations(decoded, fixture_name)
        if launches > 1:
            first, second = decoded['executions']
            assert first['slot'] == second['slot']
            assert first['report_address'] == second['report_address']
            assert second['generation'] == first['generation'] + 1
        self.inputs['capture'] = identity(capture)
        if args.case == 'disjoint':
            truncated = self.directory / 'truncated.bin'
            truncated.write_bytes(capture.read_bytes()[:-1])
            self.run('reject-truncated', [args.inspect, truncated], 2)
            missing = bytearray(capture.read_bytes())
            struct.pack_into('<Q', missing, 40, 0)
            missing[-32:] = hashlib.sha256(missing[:-32]).digest()
            missing_path = self.directory / 'missing-facts.bin'
            missing_path.write_bytes(missing)
            self.run(
                'reject-missing-facts-valid-checksum', [args.inspect, missing_path], 2
            )

    def prepare_original(self, fixture, original):
        shutil.copyfile(fixture, original)
        return 'lds_producer_consumer'

    def assert_observations(self, decoded, fixture):
        if fixture.startswith(('wide_', 'mixed_')):
            self.assert_width_observations(decoded, fixture)
            return
        if fixture.startswith('grouped_'):
            self.assert_workgroup_observations(decoded, fixture)
            return
        expected = {
            'race': 2,
            'high_lane': 2,
            'barrier': 2,
            'barrier_empty': 2,
            'barrier_high': 2,
            'barrier_separated': 2,
            'barrier_multiple': 3,
            'barrier_missing_second': 3,
            'disjoint': 4,
            'read_read': 2,
            'collision': 1,
            'single_lane': 1,
            'alias': 4,
            'immediate': 2,
            'four_ranges': 4,
            'exhausted': 4,
            'empty': 0,
        }
        kinds = {site['id']: site['access'] for site in decoded['sites']}
        for index, execution in enumerate(decoded['executions']):
            assert execution['serial'] == index + 1
            assert execution['argument_u32'] == 7 + 16 * index
            rows = execution['rows']
            wave_count = (
                self.args.threads + self.args.wave_size - 1
            ) // self.args.wave_size
            assert len(rows) == wave_count
            observations = []
            for wave, row in enumerate(rows):
                assert row['wave'] == wave and row['serial'] == index + 1
                if fixture.startswith('barrier'):
                    phases = 2 if fixture == 'barrier_multiple' else 1
                    assert row['final_phase'] == phases
                    assert len(row['completions']) == phases
                    for group, completion in enumerate(row['completions']):
                        assert completion['commit'] == 0x43534243
                        assert (
                            completion['group'] == group
                            and completion['occurrence'] == 1
                        )
                        assert completion['phase'] == group + 1
                        assert completion['participants'] == wave_count
                        assert completion['sequence'] == (
                            1 if wave == 0 else (group if wave == 1 else 0)
                        )
                exhausted = fixture == 'exhausted' and wave == 0
                assert row['status'] == (
                    ROW_ENTERED | ROW_EXITED | (ROW_EXHAUSTED if exhausted else 0)
                )
                assert row['loss'] == (1 if exhausted else 0)
                sequences = set()
                for record in row['records']:
                    assert record['commit'] == 0x43534143
                    assert record['width'] == 4 and record['flags'] == 1
                    assert record['site'] in kinds
                    assert record['mask'] > 0
                    sequences.add(record['sequence'])
                    observations.append((wave, kinds[record['site']], record))
                assert sequences == set(range(1, row['final_sequence'] + 1))
            count = (
                (index + 1)
                if fixture == 'branched'
                else (
                    2 * wave_count
                    if fixture in ('disjoint', 'alias')
                    else (
                        wave_count
                        if self.args.fixture_kind == 'hip' and fixture == 'read_read'
                        else expected[fixture]
                    )
                )
            )
            assert len(observations) == count, observations
            if fixture in ('race', 'read_read', 'high_lane'):
                assert [
                    (wave, kind, record['mask'], record['offset'])
                    for wave, kind, record in observations
                ] == [
                    (
                        wave,
                        1 if fixture == 'read_read' or wave else 2,
                        (
                            (1 << (self.args.wave_size - 1))
                            if fixture == 'high_lane'
                            else 1
                        ),
                        0,
                    )
                    for wave in range(wave_count)
                ]
            if fixture in ('barrier_multiple', 'barrier_missing_second'):
                assert [
                    (wave, kind, record['site'], record['phase'], record['sequence'])
                    for wave, kind, record in observations
                ] == [
                    (0, 2, 0, 0, 1),
                    (0, 2, 2, 2 if fixture == 'barrier_multiple' else 1, 2),
                    (1, 1, 1, 1, 1),
                ]
                assert all(
                    record['mask'] == 1 and record['offset'] == 0
                    for _, _, record in observations
                )
            if fixture in ('disjoint', 'alias'):
                assert [
                    (wave, kind, record['mask'], record['offset'])
                    for wave, kind, record in observations
                ] == [
                    (wave, kind, 1, 4 * wave)
                    for wave in range(wave_count)
                    for kind in (2, 1)
                ]
            if fixture in ('collision', 'single_lane'):
                wave, kind, record = observations[0]
                assert (wave, kind, record['mask'], record['offset']) == (
                    0,
                    2,
                    3 if fixture == 'collision' else 1,
                    0,
                )
            if fixture in ('four_ranges', 'exhausted'):
                assert [
                    (wave, kind, record['mask'], record['offset'])
                    for wave, kind, record in observations
                ] == [(0, 2, 1 << lane, 4 * lane) for lane in range(4)]

    def assert_width_observations(self, decoded, fixture):
        wave_size = self.args.wave_size
        waves = (self.args.threads + wave_size - 1) // wave_size
        wide = fixture.startswith('wide_')
        width = (
            2
            if '16' in fixture
            else (16 if '128' in fixture or fixture == 'wide_alias' else 8)
        )
        high = fixture.endswith('_high')
        for index, execution in enumerate(decoded['executions']):
            assert execution['serial'] == index + 1
            assert len(execution['rows']) == waves
            total = 0
            for wave, row in enumerate(execution['rows']):
                assert row['wave'] == wave and row['serial'] == index + 1
                assert row['status'] == ROW_ENTERED | ROW_EXITED and row['loss'] == 0
                records = row['records']
                if wide:
                    active = (
                        not high or wave * wave_size + wave_size - 1 < self.args.threads
                    )
                    assert row['final_sequence'] == (2 if active else 0)
                    assert len(records) == (2 if active else 0)
                    mask = (
                        (1 << (wave_size - 1))
                        if high
                        else (
                            3
                            if fixture == 'wide_collision'
                            and wave * wave_size + 1 < self.args.threads
                            else 1
                        )
                    )
                    for sequence, record in enumerate(records, 1):
                        assert record['sequence'] == sequence
                        assert record['mask'] == mask
                        assert record['width'] == width
                        assert (
                            record['offset'] == wave * 16 + {2: 2, 8: 8, 16: 16}[width]
                        )
                        assert (
                            record['width'] == decoded['sites'][record['site']]['width']
                        )
                else:
                    assert len(records) == (1 if wave < 2 else 0)
                    assert row['final_sequence'] == (1 if wave < 2 else 0)
                    if records:
                        record = records[0]
                        widths = (4, 2) if fixture == 'mixed_b16' else (8, 4)
                        assert record['width'] == widths[wave] and record['mask'] == 1
                        assert record['offset'] == (
                            0
                            if wave == 0
                            else (
                                2
                                if fixture == 'mixed_b16'
                                else 8 if fixture == 'mixed_disjoint' else 4
                            )
                        )
                        assert decoded['sites'][record['site']]['access'] == (
                            1 if wave or fixture == 'mixed_read_read' else 2
                        )
                for record in records:
                    assert record['commit'] == 0x43534143 and record['flags'] == 1
                total += len(records)
            assert total > 0

    def assert_workgroup_observations(self, decoded, fixture):
        groups = self.args.workgroups
        waves = (self.args.threads + self.args.wave_size - 1) // self.args.wave_size
        for index, execution in enumerate(decoded['executions']):
            assert execution['serial'] == index + 1
            assert execution['argument_u32'] == 7 + 16 * index
            assert execution['grid_x'] == groups * self.args.threads
            assert execution['workgroup_size_x'] == self.args.threads
            assert len(execution['rows']) == groups * waves
            total = 0
            for row_index, row in enumerate(execution['rows']):
                group, wave = divmod(row_index, waves)
                assert (
                    row['serial'] == index + 1
                    and row['workgroup_x'] == group
                    and row['wave'] == wave
                )
                assert row['status'] == ROW_ENTERED | ROW_EXITED and row['loss'] == 0
                records = row['records']
                if fixture in ('grouped_disjoint', 'grouped_barrier', 'grouped_empty'):
                    assert len(records) == row['final_sequence'] == 2
                    for site, record in enumerate(records):
                        assert record['site'] == site and record['sequence'] == site + 1
                        assert record['offset'] == (
                            wave * 4
                            if site == 0 or fixture == 'grouped_disjoint'
                            else 0
                        )
                    if fixture != 'grouped_disjoint':
                        assert row['completions'][0]['commit'] == 0x43534243
                        assert (
                            row['completions'][0]['group'] == 0
                            and row['completions'][0]['occurrence'] == 1
                        )
                        assert (
                            row['completions'][0]['participants'] == waves
                            and row['completions'][0]['sequence'] == 1
                        )
                else:
                    expected = (
                        (wave < 2)
                        if fixture != 'grouped_isolation'
                        else (wave == (0 if group == 0 else 1))
                    )
                    assert len(records) == row['final_sequence'] == int(expected)
                    if records:
                        assert records[0]['site'] == (0 if wave == 0 else 1)
                        assert records[0]['offset'] == 0 and records[0]['sequence'] == 1
                for record in records:
                    assert record['commit'] == 0x43534143 and record['flags'] == 1
                    assert record['width'] == 4 and record['mask'] == 1
                    kind = decoded['sites'][record['site']]['access']
                    assert kind == (
                        1
                        if fixture == 'grouped_read_read' or record['site'] == 1
                        else 2
                    )
                total += len(records)
            assert total == (
                2 * groups * waves
                if fixture in ('grouped_disjoint', 'grouped_barrier', 'grouped_empty')
                else groups if fixture == 'grouped_isolation' else 2 * groups
            )


def add_target_arguments(parser):
    parser.add_argument('--test-statistics', action='store_true')
    parser.add_argument('--amd-signals', action='store_true')
    parser.add_argument(
        '--target',
        default='gfx1100',
        choices=('gfx942', 'gfx950', 'gfx1100', 'gfx1201', 'gfx1250'),
    )
    parser.add_argument('--wave-size', type=int, choices=(32, 64), default=32)
    parser.add_argument('--threads', type=int, default=64)
    parser.add_argument('--workgroups', type=int, default=1)
    parser.add_argument(
        '--fixture-kind',
        choices=('named-assembly', 'assembly', 'hip'),
        default='named-assembly',
    )
    parser.add_argument('--launcher', type=lambda value: Path(value).resolve())
    parser.add_argument('--config', type=lambda value: Path(value).resolve())
    parser.add_argument('--require-agent', action='store_true')


class HardwareUnavailable(Exception):
    pass


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
    ):
        parser.add_argument(
            f'--{name}', type=lambda value: Path(value).resolve(), required=True
        )
    parser.add_argument('--case', choices=CASES, required=True)
    add_target_arguments(parser)
    driver = Driver(parser.parse_args())
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
    except (
        AssertionError,
        OSError,
        ValueError,
        KeyError,
        IndexError,
        subprocess.SubprocessError,
    ) as error:
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
