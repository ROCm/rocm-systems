#!/usr/bin/env python3
"""Append one built dashboard run to a publication snapshot.

prepare-dashboard-data.py writes the run and catalog only. This copies those
files into the publication directory and updates index.json in place: when the
run path is absent, append it and set generatedAt. An indexed run whose file
already matches is left unchanged, including generatedAt. Publication metadata
is never read or written: website configuration is bundled with the frontend.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import re


def load_prepare():
    path = Path(__file__).resolve().with_name('prepare-dashboard-data.py')
    spec = importlib.util.spec_from_file_location('prepare_dashboard_data', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_prepare()


def load_index(root, source):
    """Read existing history and reject source-timestamp conflicts."""
    if not (root / 'index.json').exists():
        return None
    index = prepare.read_json(root / 'index.json')
    names = index['runFiles']
    prepare.require(
        isinstance(names, list)
        and all(
            isinstance(name, str)
            and re.fullmatch(
                r'runs/(?:default-branch/|side-branches/)?[A-Za-z0-9._-]+\.json', name
            )
            for name in names
        ),
        'Invalid indexed run filename',
    )
    prepare.require(len(names) == len(set(names)), 'Duplicate indexed run filename')
    prepare.instant(index['generatedAt'])
    for name in names:
        previous_run = prepare.read_json(root / name)
        version = previous_run.get('schemaVersion', 1)
        prepare.require(
            type(version) is int and version in (1, 2),
            f'Unsupported run schema; explicit migration required: {name}',
        )
        if version == 1:
            prepare.require(
                isinstance(previous_run.get('targets'), list)
                and 'configurations' not in previous_run,
                f'Legacy run schema requires target groups; explicit migration required: {name}',
            )
        flat = re.fullmatch(r'runs/[A-Za-z0-9._-]+\.json', name)
        if flat:
            prepare.require(
                version == 1,
                f'Flat run paths require legacy schema 1; explicit migration required: {name}',
            )
        else:
            prepare.require(
                name == prepare.run_filename(previous_run),
                f'Indexed run directory does not match source.branch or id: {name}',
            )
        previous_source = previous_run['source']
        if previous_source['commit'] == source['commit']:
            prepare.require(
                prepare.instant(previous_source['committedAt'])
                == prepare.instant(source['committedAt']),
                'Commit has conflicting committedAt values',
            )
    return index


def write_dataset(root, run, catalog):
    """Preserve history and make the new run reachable by writing the index last.

    The workflow owns this private staging directory and serializes its writes.
    Full dataset validation runs separately before the workflow pushes to Git.
    """
    prepare.validate_output_path(root)
    prepare.canonical([run, catalog])  # Reject non-finite output before any writes.
    index = load_index(root, run['source'])
    run_path = prepare.run_filename(run)
    # Moving an id to a different branch group must not evade immutability,
    # even when the old file is an orphan or a historical flat legacy record.
    for directory in ('runs', 'runs/default-branch', 'runs/side-branches'):
        previous_path = f"{directory}/{run['id']}.json"
        prepare.require(
            previous_path == run_path or not (root / previous_path).exists(),
            f'Immutable resource conflict: {previous_path}',
        )
    resources = {run['testCatalog']: catalog, run_path: run}
    # Check all conflicts before writing anything, including unindexed resources.
    for name, value in resources.items():
        if (root / name).exists():
            prepare.require(
                prepare.canonical(prepare.read_json(root / name))
                == prepare.canonical(value),
                f'Immutable resource conflict: {name}',
            )
    run_files = index['runFiles'] if index else []
    if run_path not in run_files:
        index = {
            'generatedAt': datetime.now(timezone.utc).isoformat(),
            'runFiles': [*run_files, run_path],
        }
    for name, value in resources.items():
        prepare.write_json(root / name, value)
    prepare.write_json(root / 'index.json', index)


def load_built(source):
    """Read the single run built by prepare-dashboard-data.py. Ignore any index."""
    source = Path(os.path.abspath(source))
    prepare.validate_output_path(source)
    prepare.require(source.is_dir(), 'Built data directory is missing')
    run_dir = source / 'runs'
    prepare.require(
        run_dir.is_dir() and not run_dir.is_symlink(),
        'Built data must contain a runs directory',
    )
    runs = sorted(path for path in run_dir.rglob('*') if path.is_file())
    prepare.require(len(runs) == 1, 'Built data must contain exactly one run file')
    run_path = runs[0]
    prepare.require(
        re.fullmatch(
            r'runs/(?:default-branch|side-branches)/[A-Za-z0-9._-]+\.json',
            run_path.relative_to(source).as_posix(),
        ),
        'Invalid built run filename',
    )
    run = prepare.read_json(run_path)
    prepare.require(
        isinstance(run.get('id'), str) and run_path.name == f"{run['id']}.json",
        'Built run filename must match its id',
    )
    prepare.require(
        type(run.get('schemaVersion')) is int and run['schemaVersion'] == 2,
        'Built run requires numeric schemaVersion 2; migrate older runs explicitly',
    )
    prepare.require(
        run_path.relative_to(source).as_posix() == prepare.run_filename(run),
        'Built run directory does not match source.branch',
    )
    catalog_name = run.get('testCatalog')
    prepare.require(
        isinstance(catalog_name, str)
        and re.fullmatch(r'test-catalogs/[A-Za-z0-9._-]+\.json', catalog_name),
        'Built run references an invalid catalog',
    )
    catalog = prepare.read_json(source / catalog_name)
    return run, catalog


def publish(source, data_dir):
    run, catalog = load_built(source)
    write_dataset(Path(os.path.abspath(data_dir)), run, catalog)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--from',
        dest='source',
        required=True,
        help='Directory written by prepare-dashboard-data.py',
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Publication data directory whose index.json is updated',
    )
    args = parser.parse_args()
    try:
        run = publish(args.source, args.data_dir)
        print(f"Published schema-2 run {run['id']}")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
