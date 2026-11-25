# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Report site and entry-stub branch capacity without publishing partial rewrites."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

ENTRIES = ('cap_a', 'cap_b')
RANGE_REASON = 'unsupported_control_transfer_range'


def symbols(image):
    """Read this fixture's ELF64 symbol bytes independently of preparation facts."""
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', image)
    assert header[0][:6] == b'\x7fELF\x02\x01'
    assert header[11] == 64
    sections = [
        struct.unpack_from('<IIQQQQIIQQ', image, header[6] + index * header[11])
        for index in range(header[12])
    ]
    result = {}
    for section in sections:
        if section[1] not in (2, 11):
            continue
        assert section[9] == 24 and section[5] % section[9] == 0
        strings = sections[section[6]]
        names = image[strings[4] : strings[4] + strings[5]]
        for offset in range(section[4], section[4] + section[5], section[9]):
            name, _info, _other, owner, value, size = struct.unpack_from(
                '<IBBHQQ', image, offset
            )
            if owner == 0 or owner >= len(sections):
                continue
            end = names.index(b'\0', name)
            label = names[name:end].decode()
            if label not in (*ENTRIES, *(entry + '.kd' for entry in ENTRIES)):
                continue
            address = sections[owner]
            assert value >= address[3] and value - address[3] + size <= address[5]
            begin = address[4] + value - address[3]
            entry = {'offset': begin, 'bytes': image[begin : begin + size]}
            assert entry['bytes'] and len(entry['bytes']) == size
            assert label not in result or result[label] == entry
            result[label] = entry
    assert set(result) == {*ENTRIES, *(entry + '.kd' for entry in ENTRIES)}
    return result


def descriptor_resources(descriptor):
    assert len(descriptor) == 64
    # The entry address is relocated; all native ABI/resource bytes must match.
    return descriptor[:16] + bytes(8) + descriptor[24:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        'prepare',
        'fixture',
        'prologue-fixture',
        'split-prologue-fixture',
        'root',
    ):
        parser.add_argument(
            '--' + name, type=lambda value: Path(value).resolve(), required=True
        )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='prepare-composition-', dir=args.root))
    print(f'evidence: {directory}', flush=True)
    original = args.fixture.read_bytes()
    original_symbols = symbols(original)
    original_digest = hashlib.sha256(original).hexdigest()
    commands = []

    def run(
        name,
        entries=(),
        *,
        require_all=False,
        expected=0,
        fixture=args.fixture,
        outcome='instrumented',
    ):
        destination = directory / name
        command = [str(args.prepare), str(fixture), 'gfx1250', str(destination)]
        for entry in entries:
            command.extend(('--entry', entry))
        if require_all:
            command.append('--require-all')
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
        commands.append(
            {
                'stage': name,
                'command': command,
                'exit': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
            }
        )
        (directory / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        assert result.returncode == expected, commands[-1]
        if expected:
            assert not destination.exists(), commands[-1]
            return None
        report = json.loads((destination / 'preparation.json').read_text())
        assert report['schema_version'] == 2 and report['target'] == 1250, report
        assert (
            report['original_sha256']
            == hashlib.sha256(fixture.read_bytes()).hexdigest()
        ), report
        assert report['original_bytes'] == fixture.stat().st_size, report
        assert report['inventory_entries'] == 2, report
        assert report['not_requested'] == 2 - report['requested'], report
        assert report['no_probes_needed'] == 0, report
        assert report['blocked'] == report['skipped'], report
        assert report['outcome'] == outcome, report
        if outcome == 'instrumented':
            assert (destination / 'ready').is_file()
            assert (destination / 'facts.bin').is_file()
            assert (destination / 'prepared.hsaco').is_file()
        else:
            assert report['selection_profile'] is None, report
            assert (destination / 'original.hsaco').read_bytes() == fixture.read_bytes()
            for artifact in ('ready', 'facts.bin', 'prepared.hsaco'):
                assert not (destination / artifact).exists(), artifact
        return report

    def check_image(name, selected, native, source_symbols=original_symbols):
        prepared = symbols((directory / name / 'prepared.hsaco').read_bytes())
        native_body = source_symbols[native]['bytes']
        # Appended caves may extend the final function's ELF symbol. Its owning
        # original body still ends at the fixture's explicit s_endpgm boundary.
        assert len(prepared[native]['bytes']) >= len(native_body)
        assert prepared[native]['bytes'][: len(native_body)] == native_body
        assert descriptor_resources(
            prepared[native + '.kd']['bytes']
        ) == descriptor_resources(source_symbols[native + '.kd']['bytes'])
        assert prepared[selected]['bytes'] != source_symbols[selected]['bytes']
        assert struct.unpack_from('<I', prepared[selected + '.kd']['bytes'], 8)[0] > 16

    for selected, native in (ENTRIES, tuple(reversed(ENTRIES))):
        name = 'single-' + selected
        report = run(name, (selected,))
        assert (report['requested'], report['instrumented'], report['skipped']) == (
            1,
            1,
            0,
        )
        assert report['requested_coverage'] == 'complete', report
        check_image(name, selected, native)

    # Automatic selection follows descriptor order. Reverse explicit order also
    # proves that either candidate can survive as the independently valid subset.
    for name, order in (('automatic', ENTRIES), ('reverse', tuple(reversed(ENTRIES)))):
        report = run(name, () if name == 'automatic' else order)
        assert (report['requested'], report['instrumented'], report['skipped']) == (
            2,
            1,
            1,
        )
        assert report['requested_coverage'] == 'partial', report
        assert [entry['entry'] for entry in report['entries']] == list(order), report
        selected, skipped = report['entries']
        assert selected['disposition'] == 'instrumented', report
        assert skipped['disposition'] == 'blocked', report
        assert skipped['reason_code'] == RANGE_REASON, skipped
        assert skipped['stage'] == 'lowering', skipped
        assert skipped['no_probes_needed'] is None, skipped
        assert isinstance(skipped['reason'], str) and skipped['reason'], skipped
        for entry in report['entries']:
            assert (
                entry['entry_key'] == original_symbols[entry['entry'] + '.kd']['offset']
            )
        check_image(name, *order)
        for artifact in ('prepared.hsaco', 'facts.bin', 'ready'):
            assert (directory / name / artifact).read_bytes() == (
                directory / ('single-' + order[0]) / artifact
            ).read_bytes(), artifact

    run('strict', require_all=True, expected=2)
    prologue_original = args.prologue_fixture.read_bytes()
    prologue_symbols = symbols(prologue_original)
    for name, entries, requested in (
        ('prologue-explicit', ('cap_a',), ('cap_a',)),
        ('prologue-automatic', (), ENTRIES),
    ):
        report = run(
            name,
            entries,
            fixture=args.prologue_fixture,
            outcome='native_only',
        )
        assert (
            report['requested'],
            report['instrumented'],
            report['blocked'],
        ) == (len(requested), 0, len(requested)), report
        assert report['requested_coverage'] == 'not_performed', report
        assert [entry['entry'] for entry in report['entries']] == list(
            requested
        ), report
        for entry in report['entries']:
            assert (
                entry['entry_key'] == prologue_symbols[entry['entry'] + '.kd']['offset']
            )
            assert entry['disposition'] == 'blocked', entry
            assert entry['stage'] == 'lowering', entry
            assert entry['reason_code'] == RANGE_REASON, entry
            assert entry['no_probes_needed'] is None, entry
            assert (
                'entry-prologue stub is out of branch range' in entry['reason']
            ), entry
    run(
        'prologue-strict',
        fixture=args.prologue_fixture,
        require_all=True,
        expected=2,
    )
    assert args.prologue_fixture.read_bytes() == prologue_original

    split_original = args.split_prologue_fixture.read_bytes()
    split_symbols = symbols(split_original)
    control_name = 'split-prologue-single-cap-b'
    control = run(
        control_name,
        ('cap_b',),
        fixture=args.split_prologue_fixture,
        require_all=True,
    )
    assert (control['requested'], control['instrumented'], control['blocked']) == (
        1,
        1,
        0,
    ), control
    assert control['requested_coverage'] == 'complete', control
    assert control['entries'][0]['entry'] == 'cap_b', control
    assert control['entries'][0]['disposition'] == 'instrumented', control
    assert (
        control['entries'][0]['entry_key'] == split_symbols['cap_b.kd']['offset']
    ), control
    check_image(control_name, 'cap_b', 'cap_a', split_symbols)
    for name, entries, requested in (
        ('split-prologue-automatic', (), ENTRIES),
        ('split-prologue-reverse', tuple(reversed(ENTRIES)), tuple(reversed(ENTRIES))),
    ):
        report = run(name, entries, fixture=args.split_prologue_fixture)
        assert (report['requested'], report['instrumented'], report['blocked']) == (
            2,
            1,
            1,
        ), report
        assert report['requested_coverage'] == 'partial', report
        assert report['selection_profile'] == control['selection_profile'], report
        assert [entry['entry'] for entry in report['entries']] == list(
            requested
        ), report
        by_name = {entry['entry']: entry for entry in report['entries']}
        selected = by_name['cap_b']
        blocked = by_name['cap_a']
        assert selected['disposition'] == 'instrumented', selected
        assert selected['no_probes_needed'] is None, selected
        assert blocked['disposition'] == 'blocked', blocked
        assert blocked['stage'] == 'lowering', blocked
        assert blocked['reason_code'] == RANGE_REASON, blocked
        assert blocked['no_probes_needed'] is None, blocked
        assert (
            'entry-prologue stub is out of branch range' in blocked['reason']
        ), blocked
        for entry in report['entries']:
            assert (
                entry['entry_key'] == split_symbols[entry['entry'] + '.kd']['offset']
            ), entry
        check_image(name, 'cap_b', 'cap_a', split_symbols)
        for artifact in ('prepared.hsaco', 'facts.bin', 'ready'):
            assert (directory / name / artifact).read_bytes() == (
                directory / control_name / artifact
            ).read_bytes(), artifact
        run(
            name + '-strict',
            entries,
            fixture=args.split_prologue_fixture,
            require_all=True,
            expected=2,
        )
    assert args.split_prologue_fixture.read_bytes() == split_original
    assert args.fixture.read_bytes() == original
    (directory / 'result.json').write_text(
        json.dumps(
            {
                'passed': True,
                'original_sha256': original_digest,
                'typed_skip_reason': RANGE_REASON,
                'selected_and_native_orders': [list(ENTRIES), list(reversed(ENTRIES))],
                'strict_published': False,
                'prologue_sha256': hashlib.sha256(prologue_original).hexdigest(),
                'prologue_native_only': True,
                'prologue_strict_published': False,
                'split_prologue_sha256': hashlib.sha256(split_original).hexdigest(),
                'split_prologue_retained_entry': 'cap_b',
                'split_prologue_strict_published': False,
            },
            indent=2,
        )
        + '\n'
    )


if __name__ == '__main__':
    main()
