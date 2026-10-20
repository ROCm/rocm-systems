# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise bounded input and expected filesystem failures at the CLI boundary."""

import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile


def limit_child_output():
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    _, hard_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024, hard_limit))


def main():
    prepare, fixture, root = (Path(value).resolve() for value in sys.argv[1:])
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='prepare-io-', dir=root))
    print(f'evidence: {directory}', flush=True)
    empty = directory / 'empty.hsaco'
    empty.touch()
    oversized = directory / 'oversized.hsaco'
    with oversized.open('wb') as stream:
        stream.truncate(256 * 1024 * 1024 + 1)
    occupied = directory / 'occupied'
    occupied.mkdir()
    sentinel = occupied / 'sentinel'
    sentinel.write_bytes(b'preserve existing destination')
    parent_file = directory / 'parent-file'
    parent_file.touch()
    cases = [
        ('missing', directory / 'missing.hsaco', directory / 'missing-output', 2),
        ('empty', empty, directory / 'empty-output', 2),
        ('oversized', oversized, directory / 'oversized-output', 2),
        ('occupied', fixture, occupied, 3),
        ('parent-file', fixture, parent_file / 'artifact', 3),
        ('missing-parent', fixture, directory / 'absent-parent' / 'artifact', 3),
    ]
    for name, original, destination, expected in cases:
        command = [
            str(prepare),
            str(original),
            'gfx1100',
            'lds_producer_consumer',
            str(destination),
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=False
        )
        record = {
            'command': command,
            'exit_status': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'ready_exists': (destination / 'ready').exists(),
        }
        (directory / f'{name}.json').write_text(json.dumps(record, indent=2) + '\n')
        assert result.returncode == expected, record
        assert result.stderr, record
        assert not (destination / 'ready').exists(), record
        if name != 'occupied':
            assert not destination.exists(), record
    for target in ('gfx1150', 'GFX1100', ''):
        destination = directory / f'unknown-target-{target}'
        command = [
            str(prepare),
            str(fixture),
            target,
            'lds_producer_consumer',
            str(destination),
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=False
        )
        record = {
            'command': command,
            'exit_status': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
        (directory / f'unknown-target-{target}.json').write_text(
            json.dumps(record, indent=2) + '\n'
        )
        assert result.returncode == 2, record
        assert 'unsupported concrete native LDS target' in result.stderr, record
        assert not destination.exists(), record
    for name, limited in (('normal-write', False), ('limited-write', True)):
        destination = directory / name
        command = [
            str(prepare),
            str(fixture),
            'gfx1100',
            'lds_producer_consumer',
            str(destination),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            preexec_fn=limit_child_output if limited else None,
        )
        files = (
            {child.name: child.stat().st_size for child in destination.iterdir()}
            if destination.exists()
            else {}
        )
        record = {
            'command': command,
            'exit_status': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'files': files,
            'file_size_limit': 1024 if limited else None,
            'sigxfsz_ignored': limited,
        }
        (directory / f'{name}.json').write_text(json.dumps(record, indent=2) + '\n')
        assert destination.is_dir(), record
        if limited:
            assert result.returncode == 3, record
            assert 'Artifact write/close failed' in result.stderr, record
            assert files.get('prepared.hsaco') == 1024, record
            assert 'facts.bin' not in files and 'ready' not in files, record
        else:
            assert result.returncode == 0, record
            assert files.get('prepared.hsaco', 0) > 1024, record
            assert files.get('facts.bin', 0) > 0 and files.get('ready') == 16, record
    assert sentinel.read_bytes() == b'preserve existing destination'
    assert list(occupied.iterdir()) == [sentinel]
    print(
        f'{len(cases)} I/O failure cases plus 3 unknown targets and 2 write controls; ready only on success'
    )


if __name__ == '__main__':
    main()
