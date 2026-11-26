# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Require truthful partial/native-only preparation without hiding hard errors."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    prepare, fixture, root = (Path(value).resolve() for value in sys.argv[1:])
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='prepare-coverage-', dir=root))
    print(f'evidence: {directory}', flush=True)

    def run(name, entries, expected=0, require_all=False, original=fixture):
        destination = directory / name
        command = [str(prepare), str(original), 'gfx1250', str(destination)]
        for entry in entries:
            command.extend(('--entry', entry))
        if require_all:
            command.append('--require-all')
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=False
        )
        record = {
            'command': command,
            'exit_status': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
        (directory / f'{name}.json').write_text(json.dumps(record, indent=2) + '\n')
        assert result.returncode == expected, record
        if expected:
            assert not destination.exists(), record
            return None
        report = json.loads((destination / 'preparation.json').read_text())
        assert report['schema_version'] == 2, report
        assert (
            report['inventory_entries'] == report['requested'] + report['not_requested']
        ), report
        if report['outcome'] == 'instrumented':
            assert len(bytes.fromhex(report['selection_profile'])) == 32, report
        else:
            assert report['selection_profile'] is None, report
        assert (
            report['original_sha256']
            == hashlib.sha256(original.read_bytes()).hexdigest()
        )
        assert report['requested'] == len(entries), report
        assert report['requested'] == (
            report['instrumented'] + report['no_probes_needed'] + report['blocked']
        ), report
        assert report['skipped'] == report['blocked'], report
        assert result.stderr.count('[rj warn]') == int(report['skipped'] != 0), record
        assert [row['entry'] for row in report['entries']] == list(entries), report
        return report

    partial = run('partial', ('mk_a', 'mk_c'))
    assert partial['outcome'] == 'instrumented', partial
    assert partial['requested_coverage'] == 'partial', partial
    assert (partial['instrumented'], partial['skipped']) == (1, 1), partial
    assert (directory / 'partial/ready').is_file()
    assert (directory / 'partial/facts.bin').is_file()
    skipped = partial['entries'][1]
    assert skipped['disposition'] == 'blocked', skipped
    assert skipped['stage'] == 'qualification', skipped
    assert skipped['no_probes_needed'] is None, skipped
    assert skipped['reason_code'] == 'unsupported_instruction', skipped
    assert 'v_xor_b32' in skipped['reason'], skipped
    assert isinstance(skipped['text_offset'], int), skipped
    native = run('native-only', ('mk_c',))
    assert native['outcome'] == 'native_only', native
    assert native['requested_coverage'] == 'not_performed', native
    assert (native['instrumented'], native['skipped']) == (0, 1), native
    assert (
        directory / 'native-only/original.hsaco'
    ).read_bytes() == fixture.read_bytes()
    assert not (directory / 'native-only/ready').exists()
    assert not (directory / 'native-only/facts.bin').exists()
    assert not (directory / 'native-only/prepared.hsaco').exists()
    complete = run('complete', ('mk_a',))
    assert complete['requested_coverage'] == 'complete', complete
    assert (complete['instrumented'], complete['skipped']) == (1, 0), complete
    scalar = fixture.with_name('multi-entry-native-scalar.hsaco')
    no_probes = run('no-probes', ('mk_c',), original=scalar, require_all=True)
    assert no_probes['outcome'] == 'native_only', no_probes
    assert no_probes['requested_coverage'] == 'complete', no_probes
    assert (no_probes['inventory_entries'], no_probes['not_requested']) == (
        3,
        2,
    ), no_probes
    assert (
        no_probes['instrumented'],
        no_probes['no_probes_needed'],
        no_probes['blocked'],
    ) == (0, 1, 0), no_probes
    entry = no_probes['entries'][0]
    assert entry['disposition'] == 'no_probes_needed', entry
    assert entry['stage'] == 'applicability' and entry['reason_code'] == 'none', entry
    proof = entry['no_probes_needed']
    assert proof['version'] == 1 and proof['scope'] == 'lds', proof
    assert (
        proof['instruction_count'] > 0
        and proof['code_bytes'] >= proof['instruction_count'] * 4
    ), proof
    assert len(bytes.fromhex(proof['code_sha256'])) == 32, proof
    assert not (directory / 'no-probes/ready').exists()
    assert not (directory / 'no-probes/facts.bin').exists()
    assert (directory / 'no-probes/original.hsaco').read_bytes() == scalar.read_bytes()
    mixed = run(
        'instrumented-and-no-probes',
        ('mk_a', 'mk_c'),
        original=scalar,
        require_all=True,
    )
    assert (mixed['instrumented'], mixed['no_probes_needed'], mixed['blocked']) == (
        1,
        1,
        0,
    ), mixed
    assert mixed['requested_coverage'] == 'complete', mixed
    assert (directory / 'instrumented-and-no-probes/ready').is_file()
    run('strict', ('mk_a', 'mk_c'), expected=2, require_all=True)
    run('typo', ('mk_a', 'missing'), expected=2)
    run('duplicate', ('mk_a', 'mk_a'), expected=2)
    escaped = directory / 'escaped.hsaco'
    escaped_name = bytes((109, 34, 92, 99))
    escaped.write_bytes(fixture.read_bytes().replace(b'mk_c', escaped_name))
    escaped_report = run('escaped', (escaped_name.decode(),), original=escaped)
    assert (
        escaped_report['entries'][0]['entry'] == escaped_name.decode()
    ), escaped_report
    assert (
        escaped_report['entries'][0]['reason_code'] == 'unsupported_instruction'
    ), escaped_report
    invalid = directory / 'invalid.hsaco'
    invalid.write_bytes(b'not an ELF')
    run('invalid', ('mk_a',), expected=2, original=invalid)


if __name__ == '__main__':
    main()
