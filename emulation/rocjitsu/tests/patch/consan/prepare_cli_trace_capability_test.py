# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep an unsupported object-wide trace capability as native compilation evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepare', 'fixture', 'root'):
        parser.add_argument(
            '--' + name, type=lambda value: Path(value).resolve(), required=True
        )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='trace-capability-', dir=args.root))
    print(f'evidence: {directory}', flush=True)
    original = args.fixture.read_bytes()
    commands = []
    for strict in (False, True):
        destination = directory / ('strict' if strict else 'native')
        command = [
            str(args.prepare),
            str(args.fixture),
            'gfx1250',
            str(destination),
            '--profile',
            'execution-trace',
            '--entry',
            'mk_a',
        ]
        if strict:
            command.append('--require-all')
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=False
        )
        commands.append(
            {
                'command': command,
                'exit': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
            }
        )
        (directory / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        assert result.returncode == (2 if strict else 0), commands[-1]
        if strict:
            assert not destination.exists()
            continue
        report = json.loads((destination / 'preparation.json').read_text())
        assert (
            report['schema_version'] == 2 and report['outcome'] == 'native_only'
        ), report
        assert report['original_sha256'] == hashlib.sha256(original).hexdigest(), report
        assert report['requested_scope'] == 'execution_trace', report
        assert report['inventory_entries'] == 3 and report['not_requested'] == 2, report
        assert report['no_probes_needed'] == 0 and report['blocked'] == 1, report
        assert (report['requested'], report['instrumented'], report['skipped']) == (
            1,
            0,
            1,
        ), report
        assert report['requested_coverage'] == 'not_performed', report
        assert report['selection_profile'] is None, report
        assert len(report['entries']) == 1, report
        skipped = report['entries'][0]
        assert (
            skipped['entry'] == 'mk_a' and skipped['disposition'] == 'blocked'
        ), skipped
        assert skipped['reason_code'] == 'unsupported_instruction', skipped
        assert skipped['stage'] == 'lowering', skipped
        assert skipped['reason'].startswith(
            'whole-image callable DBI lowering:'
        ), skipped
        assert skipped['no_probes_needed'] is None, skipped
        assert (destination / 'original.hsaco').read_bytes() == original
        for absent in ('ready', 'facts.bin', 'prepared.hsaco'):
            assert not (destination / absent).exists(), absent
    assert args.fixture.read_bytes() == original
    (directory / 'result.json').write_text(json.dumps({'passed': True}) + '\n')


if __name__ == '__main__':
    main()
