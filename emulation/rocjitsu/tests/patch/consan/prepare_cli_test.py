# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check that an unloadable original never publishes a ready artifact."""

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def main():
    prepare, fixture, root = (Path(value).resolve() for value in sys.argv[1:])
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='prepare-cli-', dir=root))
    image = bytearray(fixture.read_bytes())
    phoff = struct.unpack_from('<Q', image, 32)[0]
    phnum = struct.unpack_from('<H', image, 56)[0]
    changed = 0
    for index in range(phnum):
        offset = phoff + index * 56
        kind, flags = struct.unpack_from('<II', image, offset)
        if kind == 1 and flags & 1:
            struct.pack_into('<I', image, offset + 4, flags & ~1)
            changed += 1
    assert changed == 1
    original = directory / 'non-executable.hsaco'
    original.write_bytes(image)
    artifact = directory / 'artifact'
    command = [
        str(prepare),
        str(original),
        'gfx1100',
        'lds_producer_consumer',
        str(artifact),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False
    )
    (directory / 'prepare.stdout').write_text(result.stdout)
    (directory / 'prepare.stderr').write_text(result.stderr)
    record = {
        'command': command,
        'exit_status': result.returncode,
        'ready_exists': (artifact / 'ready').exists(),
        'artifact_exists': artifact.exists(),
    }
    (directory / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'evidence: {directory}')
    assert result.returncode == 2, record
    assert 'executable .text' in result.stderr, result.stderr
    assert not artifact.exists(), record


if __name__ == '__main__':
    main()
