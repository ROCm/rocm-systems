#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Append one run path to a dashboard publication index."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile

NEW_RUN_PATH = re.compile(
    r'runs/(?:default-branch|side-branches)/(?P<name>[A-Za-z0-9._-]+)\.json'
)
EXISTING_RUN_PATH = re.compile(
    r'runs/(?:(?:default-branch|side-branches)/)?(?P<name>[A-Za-z0-9._-]+)\.json'
)


def safe_run_path(path, pattern):
    match = pattern.fullmatch(path)
    return match is not None and match.group('name') not in ('.', '..')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def read_index(path):
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)


def validate_index(index):
    if not isinstance(index, dict) or set(index) != {'generatedAt', 'runFiles'}:
        raise ValueError('Invalid index object')
    generated_at = index['generatedAt']
    if not isinstance(generated_at, str) or not re.fullmatch(
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})',
        generated_at,
    ):
        raise ValueError('Invalid index timestamp')
    try:
        datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('Invalid index timestamp') from error
    run_files = index['runFiles']
    if not isinstance(run_files, list) or not all(
        isinstance(path, str) and safe_run_path(path, EXISTING_RUN_PATH)
        for path in run_files
    ):
        raise ValueError('Invalid indexed run path')
    if len(run_files) != len(set(run_files)):
        raise ValueError('Duplicate indexed run path')
    return run_files


def write_index(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix='.dashboard-index-', dir=path.parent
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def reject_symlink_ancestors(path):
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError(f'Path contains a symlink: {candidate}')


def update_index(data_dir, run_path):
    if not safe_run_path(run_path, NEW_RUN_PATH):
        raise ValueError('Invalid new run path')
    root = Path(os.path.abspath(data_dir))
    reject_symlink_ancestors(root)
    path = root / 'index.json'
    if path.is_symlink():
        raise ValueError('index.json must not be a symlink')
    if path.exists():
        index = read_index(path)
        run_files = validate_index(index)
        if run_path in run_files:
            return
    else:
        run_files = []
    index = {
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'runFiles': [*run_files, run_path],
    }
    write_index(path, index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--run-path', required=True)
    args = parser.parse_args()
    try:
        update_index(args.data_dir, args.run_path)
    except (ValueError, OSError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
