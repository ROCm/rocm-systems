# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Publication layout, legacy-history and containment contracts."""

import copy
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publisher = load_module('publisher', HERE.parent / 'publish-dashboard-run.py')
fixtures = load_module('producer_tests', HERE / 'test_prepare_dashboard_data.py')


@pytest.fixture
def built(tmp_path):
    run, catalog = publisher.prepare.normalize_runs(
        [fixtures.raw()],
        run_id='new',
        branch='develop',
        commit_message='Fictional',
        machine_id='worker',
        trigger='auto',
    )
    root = tmp_path / 'built'
    publisher.prepare.write_built_run(root, run, catalog)
    return root, run, catalog


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob('*')
        if p.is_file()
    }


@pytest.mark.parametrize('version', [1, None])
def test_flat_legacy_history_remains_byte_preserved(built, tmp_path, version):
    _, run, catalog = built
    root = tmp_path / 'published'
    legacy = {'id': 'legacy', 'source': copy.deepcopy(run['source']), 'targets': []}
    if version is not None:
        legacy['schemaVersion'] = version
    publisher.prepare.write_json(root / 'runs/legacy.json', legacy)
    publisher.prepare.write_json(
        root / 'index.json',
        {'generatedAt': '2026-01-01T02:00:00Z', 'runFiles': ['runs/legacy.json']},
    )
    original = (root / 'runs/legacy.json').read_bytes()
    publisher.write_dataset(root, run, catalog)
    assert (root / 'runs/legacy.json').read_bytes() == original
    assert publisher.prepare.read_json(root / 'index.json')['runFiles'] == [
        'runs/legacy.json',
        'runs/default-branch/new.json',
    ]
    before = snapshot(root)
    publisher.write_dataset(root, run, catalog)
    assert snapshot(root) == before
    conflicting = copy.deepcopy(run)
    conflicting['id'] = 'later'
    conflicting['source']['committedAt'] = '2025-12-31T00:00:00Z'
    with pytest.raises(ValueError, match='conflicting committedAt'):
        publisher.write_dataset(root, conflicting, catalog)
    assert snapshot(root) == before


@pytest.mark.parametrize(
    'name,version',
    [
        ('runs/old.json', 2),
        ('runs/old.json', None),
        ('runs/default-branch/old.json', None),
        ('runs/side-branches/old.json', 2),
    ],
)
def test_invalid_historical_layout_or_unversioned_configurations_require_migration(
    built, tmp_path, name, version
):
    _, run, catalog = built
    root = tmp_path / 'published'
    historical = copy.deepcopy(run)
    historical['id'] = 'old'
    if version is None:
        historical.pop('schemaVersion')
    publisher.prepare.write_json(root / name, historical)
    publisher.prepare.write_json(
        root / 'index.json', {'generatedAt': '2026-01-01T02:00:00Z', 'runFiles': [name]}
    )
    before = snapshot(root)
    with pytest.raises(ValueError, match='migration|directory|schema'):
        publisher.write_dataset(root, run, catalog)
    assert snapshot(root) == before


@pytest.mark.parametrize(
    'name',
    [
        'runs/new.json',
        'runs/side-branches/new.json',
        'runs/side-branches/feature/new.json',
        'runs/default-branch/wrong.json',
    ],
)
def test_built_layout_matches_source_branch_and_id(built, name):
    root, _, _ = built
    destination = root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    (root / 'runs/default-branch/new.json').rename(destination)
    with pytest.raises(ValueError, match='filename|directory'):
        publisher.load_built(root)


@pytest.mark.parametrize('version', [None, 1, '2', True, 2.0])
def test_new_built_runs_require_explicit_numeric_schema_two(built, version):
    root, run, _ = built
    if version is None:
        run.pop('schemaVersion')
    else:
        run['schemaVersion'] = version
    publisher.prepare.write_json(root / 'runs/default-branch/new.json', run)
    with pytest.raises(ValueError, match='schemaVersion'):
        publisher.load_built(root)


def test_built_directory_requires_exactly_one_run_across_groups(built):
    root, run, _ = built
    publisher.prepare.write_json(root / 'runs/side-branches/second.json', run)
    with pytest.raises(ValueError, match='exactly one'):
        publisher.load_built(root)


@pytest.mark.parametrize('target', ['run', 'directory', 'catalog'])
def test_built_symlinks_rejected_without_touching_external_files(
    built, tmp_path, target
):
    root, run, _ = built
    path = (
        root
        / {
            'run': 'runs/default-branch/new.json',
            'directory': 'runs/default-branch',
            'catalog': run['testCatalog'],
        }[target]
    )
    outside = tmp_path / 'outside'
    path.rename(outside)
    path.symlink_to(outside, target_is_directory=target == 'directory')
    before = outside.read_bytes() if outside.is_file() else snapshot(outside)
    with pytest.raises(ValueError, match='symlink'):
        publisher.load_built(root)
    assert (outside.read_bytes() if outside.is_file() else snapshot(outside)) == before


def test_run_id_cannot_be_republished_in_a_different_branch_group(built, tmp_path):
    _, run, catalog = built
    root = tmp_path / 'published'
    publisher.write_dataset(root, run, catalog)
    changed = copy.deepcopy(run)
    changed['source']['branch'] = 'feature/example'
    before = snapshot(root)
    with pytest.raises(ValueError, match='Immutable resource conflict'):
        publisher.write_dataset(root, changed, catalog)
    assert snapshot(root) == before
    # An unindexed immutable file is protected as well.
    (root / 'index.json').unlink()
    before = snapshot(root)
    with pytest.raises(ValueError, match='Immutable resource conflict'):
        publisher.write_dataset(root, changed, catalog)
    assert snapshot(root) == before


def test_catalog_traversal_rejected(built):
    root, run, _ = built
    run['testCatalog'] = '../outside.json'
    publisher.prepare.write_json(root / 'runs/default-branch/new.json', run)
    with pytest.raises(ValueError, match='invalid catalog'):
        publisher.load_built(root)
