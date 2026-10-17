# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Focused contracts for updating dashboard index.json."""

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / 'update-dashboard-index.py'


def invoke(data_dir, run_path):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            '--data-dir',
            str(data_dir),
            '--run-path',
            run_path,
        ],
        capture_output=True,
        text=True,
    )


def test_bootstrap_creates_index_with_run(tmp_path):
    data_dir = tmp_path / 'data'

    result = invoke(data_dir, 'runs/default-branch/new.json')

    assert result.returncode == 0, result.stderr
    index = json.loads((data_dir / 'index.json').read_text())
    assert set(index) == {'generatedAt', 'runFiles'}
    assert index['runFiles'] == ['runs/default-branch/new.json']
    offset = datetime.fromisoformat(
        index['generatedAt'].replace('Z', '+00:00')
    ).utcoffset()
    assert offset is not None and offset.total_seconds() == 0


def test_append_preserves_existing_history(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    original = {
        'generatedAt': '2026-01-01T02:00:00Z',
        'runFiles': [
            'runs/legacy.json',
            'runs/side-branches/old.json',
        ],
    }
    (data_dir / 'index.json').write_text(json.dumps(original))

    result = invoke(data_dir, 'runs/default-branch/new.json')

    assert result.returncode == 0, result.stderr
    index = json.loads((data_dir / 'index.json').read_text())
    assert index['generatedAt'] != original['generatedAt']
    assert index['runFiles'] == [
        'runs/legacy.json',
        'runs/side-branches/old.json',
        'runs/default-branch/new.json',
    ]


def test_duplicate_index_key_is_rejected_without_rewrite(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    path = data_dir / 'index.json'
    original = (
        b'{"generatedAt":"2026-01-01T02:00:00Z",'
        b'"generatedAt":"2026-01-02T02:00:00Z","runFiles":[]}\n'
    )
    path.write_bytes(original)

    result = invoke(data_dir, 'runs/default-branch/new.json')

    assert result.returncode != 0
    assert 'Duplicate JSON key' in result.stderr
    assert path.read_bytes() == original


def test_duplicate_index_run_path_is_rejected_without_rewrite(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    path = data_dir / 'index.json'
    original = json.dumps(
        {
            'generatedAt': '2026-01-01T02:00:00Z',
            'runFiles': ['runs/legacy.json', 'runs/legacy.json'],
        }
    ).encode()
    path.write_bytes(original)

    result = invoke(data_dir, 'runs/default-branch/new.json')

    assert result.returncode != 0
    assert 'Duplicate indexed run path' in result.stderr
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    'run_path',
    [
        'runs/legacy.json',
        'runs/default-branch/../escape.json',
        'runs/default-branch/nested/new.json',
        'runs/other/new.json',
        'runs/side-branches/.json',
        'runs/default-branch/...json',
    ],
)
def test_invalid_new_run_path_is_rejected(tmp_path, run_path):
    data_dir = tmp_path / 'data'

    result = invoke(data_dir, run_path)

    assert result.returncode != 0
    assert 'run path' in result.stderr.lower()
    assert not data_dir.exists()


def test_exact_retry_preserves_index_bytes_and_generated_at(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    path = data_dir / 'index.json'
    original = (
        b'{\n  "runFiles": ["runs/default-branch/new.json"],\n'
        b'  "generatedAt": "2026-01-01T02:00:00Z"\n}\n'
    )
    path.write_bytes(original)

    result = invoke(data_dir, 'runs/default-branch/new.json')

    assert result.returncode == 0, result.stderr
    assert path.read_bytes() == original


@pytest.mark.parametrize('target', ['ancestor', 'data-dir', 'index'])
def test_symlinked_data_dir_or_index_is_rejected(tmp_path, target):
    outside = tmp_path / 'outside'
    outside.mkdir()
    marker = outside / 'marker'
    marker.write_bytes(b'keep\n')
    data_dir = tmp_path / 'data'
    if target == 'ancestor':
        link = tmp_path / 'link'
        link.symlink_to(outside, target_is_directory=True)
        data_dir = link / 'data'
    elif target == 'data-dir':
        data_dir.symlink_to(outside, target_is_directory=True)
    else:
        data_dir.mkdir()
        (data_dir / 'index.json').symlink_to(marker)

    result = invoke(data_dir, 'runs/default-branch/new.json')

    assert result.returncode != 0
    assert 'symlink' in result.stderr.lower()
    assert marker.read_bytes() == b'keep\n'
    assert not (outside / 'data' / 'index.json').exists()


@pytest.mark.parametrize(
    'index',
    [
        {'generatedAt': '2026-01-01T02:00:00Z', 'runFiles': [], 'extra': True},
        {'generatedAt': '2026-01-01T02:00:00Z'},
        {'generatedAt': 'not-a-time', 'runFiles': []},
        {'generatedAt': '2026-01-01T02:00:00', 'runFiles': []},
        {'generatedAt': '2026-01-01T02:00:00Z', 'runFiles': 'runs/old.json'},
        {'generatedAt': '2026-01-01T02:00:00Z', 'runFiles': [1]},
        {'generatedAt': '2026-01-01T02:00:00Z', 'runFiles': ['other/old.json']},
        {'generatedAt': '2026-01-01T02:00:00Z', 'runFiles': ['runs/...json']},
        {
            'generatedAt': '2026-01-01T02:00:00Z',
            'runFiles': ['runs/default-branch/nested/old.json'],
        },
    ],
)
def test_existing_index_shape_and_values_are_strict(tmp_path, index):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    path = data_dir / 'index.json'
    original = json.dumps(index).encode()
    path.write_bytes(original)

    result = invoke(data_dir, 'runs/default-branch/new.json')

    assert result.returncode != 0
    assert 'index' in result.stderr.lower() or 'timestamp' in result.stderr.lower()
    assert path.read_bytes() == original
