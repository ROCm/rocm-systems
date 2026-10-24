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
            1 if case in ('race', 'high_lane') else (2 if case == 'repeat_race' else 0)
        )
        collisions = 1 if case == 'collision' else 0
        status = 1 if pairs or collisions else 0
        # The dynamic loader's own trace records the analyzer's actual modules
        # without linking a runtime or adding a live-registry input to the tool.
        environment = self.environment | {'LD_DEBUG': 'files'}
        first = self.run('analyze', [self.args.analyze, capture], status, environment)
        replay = self.run('analyze-replay', [self.args.analyze, capture], status)
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
        assert result['collision_groups'] == collisions
        assert result['coverage_complete'] == (case != 'exhausted')
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
            assert analyzed['lost_occurrences'] == (1 if case == 'exhausted' else 0)
            assert analyzed['ordering_uncertain'] == (case == 'exhausted')
            assert analyzed['cross_wave_pairs'] == (
                1 if case in ('race', 'repeat_race', 'high_lane') else 0
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
            if case.startswith('barrier'):
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
        for execution in result['executions']:
            execution['cross_wave_examples'] = []
            execution['collision_examples'] = []
            execution['ordered_examples'] = []
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
