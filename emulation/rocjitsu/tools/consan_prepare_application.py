#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prepare every kernel in every supplied application code image."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

from consan_application_contract import (
    APPLICATION_SCHEMA_VERSION,
    validate_preparation_report,
)

MAGIC = b'RJCSAPP0'
MAX_IMAGES = 1024
MAX_PATH_BYTES = 4096
MAX_CATALOG_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 512 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode_catalog(producer, images):
    require(len(producer) == 16, 'producer contract must contain 16 bytes')
    require(0 < len(images) <= MAX_IMAGES, 'empty or oversized application inventory')
    encoded = bytearray(MAGIC + producer + struct.pack('<I', len(images)))
    for image in images:
        report = image['preparation']
        selected = report['outcome'] == 'instrumented'
        path = image['artifact'] if selected else image['native_image']
        capture = image['capture_absolute'] if selected else ''
        profile = bytes.fromhex(report['selection_profile']) if selected else bytes(32)
        encoded.extend(bytes.fromhex(report['original_sha256']))
        encoded.extend(
            struct.pack(
                '<QII', report['original_bytes'], report['target'], 1 if selected else 2
            )
        )
        path_bytes = path.encode('utf-8')
        capture_bytes = capture.encode('utf-8')
        require(
            0 < len(path_bytes) <= MAX_PATH_BYTES, 'image path exceeds catalog budget'
        )
        require(
            len(capture_bytes) <= MAX_PATH_BYTES, 'capture path exceeds catalog budget'
        )
        encoded.extend(struct.pack('<I', len(path_bytes)) + path_bytes)
        encoded.extend(profile)
        encoded.extend(struct.pack('<I', len(capture_bytes)) + capture_bytes)
    encoded.extend(hashlib.sha256(encoded).digest())
    require(len(encoded) <= MAX_CATALOG_BYTES, 'catalog exceeds its byte budget')
    return bytes(encoded)


def prepare_application(args):
    source_paths = list(args.images)
    if args.images_dir:
        require(args.images_dir.is_dir(), 'application images directory does not exist')
        source_paths.extend(sorted(args.images_dir.glob('*.hsaco')))
    require(source_paths, 'no application HSACO images were supplied')
    unique = {}
    for path in source_paths:
        path = path.resolve(strict=True)
        require(path.is_file(), f'not a regular image file: {path}')
        unique.setdefault(file_digest(path), path)
    require(len(unique) <= MAX_IMAGES, 'too many distinct application images')
    require(
        sum(path.stat().st_size for path in unique.values()) <= MAX_IMAGE_BYTES,
        'application images exceed aggregate byte budget',
    )
    args.output.mkdir(parents=True, exist_ok=False)
    images_directory = args.output / 'images'
    images_directory.mkdir()
    captures = args.output / 'captures'
    captures.mkdir()
    deadline = time.monotonic() + args.timeout
    records = []

    def run(stage, command):
        remaining = math.ceil(deadline - time.monotonic())
        require(remaining > 0, 'application preparation deadline exceeded')
        result = subprocess.run(
            [str(part) for part in command],
            capture_output=True,
            text=True,
            timeout=remaining,
            check=False,
        )
        records.append(
            {
                'stage': stage,
                'argv': [str(part) for part in command],
                'exit': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
            }
        )
        (args.output / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
        require(result.returncode == 0, f'{stage} failed: {result.stderr.strip()}')
        if result.stderr:
            sys.stderr.write(result.stderr)
        return result.stdout

    producer = bytes.fromhex(
        run('producer-contract', [args.prepare, '--producer-contract']).strip()
    )
    prepared = []
    for digest, original in sorted(unique.items()):
        artifact = images_directory / digest
        command = [
            args.prepare,
            original,
            args.target,
            artifact,
            '--profile',
            args.profile,
        ]
        if args.require_all:
            command.append('--require-all')
        run('prepare-' + digest, command)
        report = validate_preparation_report(
            json.loads((artifact / 'preparation.json').read_text())
        )
        require(
            report['original_sha256'] == digest
            and report['original_bytes'] == original.stat().st_size
            and report['requested_scope'] == args.profile.replace('-', '_'),
            'preparation report original identity or requested scope differs',
        )
        require(
            report['not_requested'] == 0,
            'application preparation did not request the complete original entry inventory',
        )
        selected = report['outcome'] == 'instrumented'
        require(
            selected or report['outcome'] == 'native_only',
            'unknown preparation outcome',
        )
        owned_original = artifact / 'original.hsaco'
        if selected:
            require(
                report['instrumented'] > 0 and len(report['selection_profile']) == 64,
                'selected image has no selection identity',
            )
            shutil.copyfile(original, owned_original)
        require(file_digest(owned_original) == digest, 'native original bytes differ')
        capture = Path('captures') / (digest + '.bin') if selected else None
        prepared.append(
            {
                'artifact': str(artifact),
                'native_image': str(owned_original),
                'capture_absolute': str(args.output / capture) if capture else '',
                'capture': capture.as_posix() if capture else None,
                'preparation': report,
            }
        )
    catalog = args.output / 'application.bin'
    catalog.write_bytes(encode_catalog(producer, prepared))
    run('validate-catalog', [args.prepare, '--validate-application-catalog', catalog])
    # The durable analysis index owns the reports. Offline analysis follows only
    # relative capture paths and never reopens any original/prepared input path.
    index = {
        'schema_version': APPLICATION_SCHEMA_VERSION,
        'scope': 'provided_application_images',
        'images': [
            {'capture': image['capture'], 'preparation': image['preparation']}
            for image in prepared
        ],
    }
    (args.output / 'application.json').write_text(json.dumps(index, indent=2) + '\n')
    environment = {
        'RJ_CONSAN_APPLICATION_PROFILE': '1',
        'RJ_CONSAN_APPLICATION_CATALOG': str(catalog),
    }
    (args.output / 'launch-environment.json').write_text(
        json.dumps(environment, indent=2) + '\n'
    )
    print(
        json.dumps(
            {'index': str(args.output / 'application.json'), 'environment': environment}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='*', type=Path)
    parser.add_argument('--images-dir', type=Path)
    parser.add_argument(
        '--prepare', type=lambda value: Path(value).resolve(), required=True
    )
    parser.add_argument('--target', required=True)
    parser.add_argument(
        '--output', type=lambda value: Path(value).resolve(), required=True
    )
    parser.add_argument('--profile', choices=('lds', 'execution-trace'), default='lds')
    parser.add_argument('--require-all', action='store_true')
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()
    require(args.timeout > 0, 'timeout must be positive')
    prepare_application(args)


if __name__ == '__main__':
    try:
        main()
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f'ConSan application preparation: {error}', file=sys.stderr)
        sys.exit(2)
