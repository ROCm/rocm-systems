#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Convert util or rocjitsu statistics JSONL to normalized SQLite or CSV tables."""

import argparse
from contextlib import closing
import csv
import json
import math
from pathlib import Path
import sqlite3
import sys

SCHEMA = 'rocjitsu.statistics.v1'
SUPPORTED_SCHEMAS = (SCHEMA, 'util.statistics.v1')
MAX_U64 = (1 << 64) - 1
MAX_I64 = (1 << 63) - 1

TABLES = {
    'runs': ('run_id', 'schema', 'complete', 'metadata_json'),
    'metrics': (
        'run_id',
        'metric_id',
        'name',
        'kind',
        'value_type',
        'unit',
        'domain',
        'metadata_json',
    ),
    'scopes': ('run_id', 'scope_id', 'name', 'gpu_id', 'metadata_json'),
    'snapshots': ('run_id', 'epoch', 'snapshot_id', 'complete', 'metadata_json'),
    'samples': (
        'run_id',
        'epoch',
        'snapshot_id',
        'scope_id',
        'metric_id',
        'availability',
        'exact_value',
        'value_integer',
        'value_real',
        'integer_out_of_range',
        'overflow',
        'metadata_json',
    ),
    'histogram_buckets': (
        'run_id',
        'epoch',
        'snapshot_id',
        'scope_id',
        'metric_id',
        'bucket_index',
        'upper_bound',
        'exact_count',
        'count_integer',
        'integer_out_of_range',
        'overflow',
        'metadata_json',
    ),
}

KEY_COLUMNS = {
    'runs': 1,
    'metrics': 2,
    'scopes': 2,
    'snapshots': 3,
    'samples': 5,
    'histogram_buckets': 6,
}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identifier(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f'{name} must be a string or integer')
    if not str(value):
        raise ValueError(f'{name} must not be empty')
    return str(value)


def integer(value, name, minimum=0, maximum=MAX_U64):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f'{name} must be a decimal integer')
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f'{name} must be a decimal integer') from error
    if isinstance(value, str) and str(result) != value:
        raise ValueError(f'{name} must use canonical decimal notation')
    if not minimum <= result <= maximum:
        raise ValueError(f'{name} is outside [{minimum}, {maximum}]')
    return result


def real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number')
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f'{name} must be a finite number') from error
    if not math.isfinite(result):
        raise ValueError(f'{name} must be a finite number')
    return result


def boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f'{name} must be a boolean')
    return int(value)


def string(value, name):
    if not isinstance(value, str):
        raise ValueError(f'{name} must be a string')
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def merge_row(table, previous, row):
    if previous == row:
        return row
    if table == 'runs' and previous[:2] == row[:2]:
        before = json.loads(previous[3])
        after = json.loads(row[3])
        old_footer = before.get('session_end')
        new_footer = after.get('session_end')
        if canonical_json(before['run']) == canonical_json(after['run']):
            if old_footer is None or new_footer is None:
                footer = old_footer if new_footer is None else new_footer
            elif canonical_json(
                {key: value for key, value in old_footer.items() if key != 'complete'}
            ) == canonical_json(
                {key: value for key, value in new_footer.items() if key != 'complete'}
            ):
                footer = dict(new_footer, complete=bool(max(previous[2], row[2])))
            else:
                raise ValueError(f'conflicting duplicate runs record: {row[:1]}')
            metadata = {'run': before['run']}
            if footer is not None:
                metadata['session_end'] = footer
            return (*row[:2], max(previous[2], row[2]), canonical_json(metadata))
    key = row[: KEY_COLUMNS[table]]
    raise ValueError(f'conflicting duplicate {table} record: {key}')


class Statistics:
    """Validated rows; only snapshots with an end marker are retained."""

    def __init__(self):
        self.tables = {name: {} for name in TABLES}

    def insert(self, table, row):
        key = row[: KEY_COLUMNS[table]]
        previous = self.tables[table].get(key)
        if previous is not None:
            row = merge_row(table, previous, row)
        self.tables[table][key] = row

    def merge(self, other):
        for table in TABLES:
            for row in other.tables[table].values():
                self.insert(table, row)


def scalar_values(value, value_type):
    if value_type == 'u64':
        number = integer(value, 'value')
        return str(number), number if number <= MAX_I64 else None, None
    if value_type == 'i64':
        number = integer(value, 'value', -(1 << 63), MAX_I64)
        return str(number), number, None
    if value_type == 'f64':
        return None, None, real(value, 'value')
    raise ValueError(f'unknown numeric type: {value_type}')


def sample_row(record, statistics, snapshot):
    run_id, epoch, snapshot_id = snapshot[:3]
    scope_id = identifier(record['scope_id'], 'scope_id')
    metric_id = identifier(record['metric_id'], 'metric_id')
    if (run_id, scope_id) not in statistics.tables['scopes']:
        raise ValueError(f'unknown scope_id: {scope_id}')
    metric = statistics.tables['metrics'].get((run_id, metric_id))
    if metric is None:
        raise ValueError(f'unknown metric_id: {metric_id}')
    prefix = (run_id, epoch, snapshot_id, scope_id, metric_id)
    if record['record'] == 'histogram_bucket':
        if metric[3] != 'histogram':
            raise ValueError('histogram bucket refers to a non-histogram metric')
        index = integer(record['bucket_index'], 'bucket_index', maximum=MAX_I64)
        bound = record['upper_bound']
        if bound is not None:
            bound = real(bound, 'upper_bound')
        count = integer(record['count'], 'count')
        return 'histogram_buckets', (
            *prefix,
            index,
            bound,
            str(count),
            count if count <= MAX_I64 else None,
            int(count > MAX_I64),
            boolean(record.get('overflow', False), 'overflow'),
            canonical_json(record),
        )
    availability = record.get('availability', 'available')
    if availability not in ('available', 'unavailable', 'unsupported'):
        raise ValueError(f'unknown availability: {availability}')
    overflow = boolean(record.get('overflow', False), 'overflow')
    if availability == 'available':
        values = scalar_values(record['value'], metric[4])
    else:
        if record.get('value') is not None:
            raise ValueError('unavailable sample must not carry a value')
        values = (None, None, None)
    out_of_range = int(values[0] is not None and values[1] is None)
    return 'samples', (
        *prefix,
        availability,
        *values,
        out_of_range,
        overflow,
        canonical_json(record),
    )


def validate_histograms(statistics):
    groups = {}
    for row in statistics.tables['histogram_buckets'].values():
        groups.setdefault(row[:5], []).append(row)
    for identity, rows in groups.items():
        rows.sort(key=lambda row: row[5])
        if [row[5] for row in rows] != list(range(len(rows))):
            raise ValueError(f'histogram bucket indices must be contiguous: {identity}')
        bounds = [row[6] for row in rows]
        if bounds[-1] is not None or any(bound is None for bound in bounds[:-1]):
            raise ValueError(
                f'histogram must end with one unbounded bucket: {identity}'
            )
        if any(left >= right for left, right in zip(bounds[:-2], bounds[1:-1])):
            raise ValueError(f'histogram upper bounds must increase: {identity}')


def load_statistics(path):
    statistics = Statistics()
    run_id = None
    run_schema = None
    snapshot = None
    pending = None
    ended = False
    with Path(path).open('rb') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if ended:
                raise ValueError(f'{path}:{line_number}: record follows session_end')
            try:
                record = json.loads(line, object_pairs_hook=unique_object)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                if not line.endswith(b'\n'):
                    break
                raise ValueError(
                    f'{path}:{line_number}: invalid JSON: {error}'
                ) from error
            except ValueError as error:
                raise ValueError(f'{path}:{line_number}: {error}') from error
            try:
                if not isinstance(record, dict):
                    raise ValueError('record must be an object')
                if ended:
                    raise ValueError('record follows session_end')
                schema = record.get('schema')
                if 'schema' in record and schema not in SUPPORTED_SCHEMAS:
                    raise ValueError(f'unsupported schema: {schema}')
                kind = record['record']
                if kind == 'run':
                    if run_id is not None or schema is None:
                        raise ValueError('expected exactly one versioned run record')
                    run_id = identifier(record['run_id'], 'run_id')
                    run_schema = schema
                    statistics.insert(
                        'runs', (run_id, run_schema, 0, canonical_json({'run': record}))
                    )
                    continue
                if run_id is None:
                    raise ValueError('first record must be a run')
                if 'schema' in record and schema != run_schema:
                    raise ValueError('record has a different schema')
                if identifier(record.get('run_id', run_id), 'run_id') != run_id:
                    raise ValueError('record has a different run_id')
                if kind == 'metric':
                    if snapshot is not None:
                        raise ValueError('metric definition inside a snapshot')
                    value_type = record['value_type']
                    if value_type not in ('u64', 'i64', 'f64'):
                        raise ValueError(f'unknown numeric type: {value_type}')
                    metric_kind = record['kind']
                    if metric_kind not in ('counter', 'gauge', 'histogram', 'summary'):
                        raise ValueError(f'unknown metric kind: {metric_kind}')
                    statistics.insert(
                        'metrics',
                        (
                            run_id,
                            identifier(record['metric_id'], 'metric_id'),
                            string(record['name'], 'name'),
                            metric_kind,
                            value_type,
                            string(record['unit'], 'unit'),
                            string(record['domain'], 'domain'),
                            canonical_json(record),
                        ),
                    )
                elif kind == 'scope':
                    if snapshot is not None:
                        raise ValueError('scope definition inside a snapshot')
                    gpu_id = record.get('gpu_id')
                    statistics.insert(
                        'scopes',
                        (
                            run_id,
                            identifier(record['scope_id'], 'scope_id'),
                            string(record['name'], 'name'),
                            (
                                identifier(gpu_id, 'gpu_id')
                                if gpu_id is not None
                                else None
                            ),
                            canonical_json(record),
                        ),
                    )
                elif kind == 'snapshot_begin':
                    if snapshot is not None:
                        raise ValueError('nested snapshot')
                    snapshot = (
                        run_id,
                        str(integer(record['epoch'], 'epoch')),
                        str(integer(record['snapshot_id'], 'snapshot_id')),
                        boolean(record.get('complete', True), 'complete'),
                        canonical_json(record),
                    )
                    pending = Statistics()
                elif kind in ('sample', 'histogram_bucket', 'snapshot_end'):
                    if snapshot is None:
                        raise ValueError(f'{kind} outside a snapshot')
                    for field, expected in (
                        ('epoch', snapshot[1]),
                        ('snapshot_id', snapshot[2]),
                    ):
                        if str(integer(record.get(field, expected), field)) != expected:
                            raise ValueError(f'{kind} has a different {field}')
                    if kind == 'snapshot_end':
                        validate_histograms(pending)
                        pending.insert('snapshots', snapshot)
                        statistics.merge(pending)
                        snapshot = None
                        pending = None
                    else:
                        table, row = sample_row(record, statistics, snapshot)
                        pending.insert(table, row)
                elif kind == 'session_end':
                    if snapshot is not None:
                        raise ValueError('session_end before snapshot_end')
                    current = statistics.tables['runs'][(run_id,)]
                    statistics.insert(
                        'runs',
                        (
                            run_id,
                            run_schema,
                            boolean(record['complete'], 'complete'),
                            canonical_json(
                                {**json.loads(current[3]), 'session_end': record}
                            ),
                        ),
                    )
                    ended = True
                else:
                    raise ValueError(f'unknown record kind: {kind}')
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'{path}:{line_number}: {error}') from error
    if run_id is None:
        raise ValueError(f'{path}: missing run record')
    return statistics


def create_tables(connection):
    for table, columns in TABLES.items():
        definitions = []
        for column in columns:
            column_type = 'TEXT'
            if column in (
                'complete',
                'value_integer',
                'count_integer',
                'integer_out_of_range',
                'overflow',
                'bucket_index',
            ):
                column_type = 'INTEGER'
            elif column in ('value_real', 'upper_bound'):
                column_type = 'REAL'
            definitions.append(f'{column} {column_type}')
        keys = ', '.join(columns[: KEY_COLUMNS[table]])
        definitions.append(f'PRIMARY KEY ({keys})')
        if table != 'runs':
            definitions.append('FOREIGN KEY (run_id) REFERENCES runs(run_id)')
        if table in ('samples', 'histogram_buckets'):
            definitions.extend(
                (
                    'FOREIGN KEY (run_id, epoch, snapshot_id) REFERENCES snapshots(run_id, epoch, snapshot_id)',
                    'FOREIGN KEY (run_id, scope_id) REFERENCES scopes(run_id, scope_id)',
                    'FOREIGN KEY (run_id, metric_id) REFERENCES metrics(run_id, metric_id)',
                )
            )
        connection.execute(
            f'CREATE TABLE IF NOT EXISTS {table} ({", ".join(definitions)})'
        )


def write_sqlite(statistics, path):
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute('PRAGMA foreign_keys = ON')
        create_tables(connection)
        for table, columns in TABLES.items():
            keys = columns[: KEY_COLUMNS[table]]
            condition = ' AND '.join(f'{key} = ?' for key in keys)
            for row in sorted(statistics.tables[table].values()):
                identity = row[: len(keys)]
                existing = connection.execute(
                    f'SELECT {", ".join(columns)} FROM {table} WHERE {condition}',
                    identity,
                ).fetchone()
                if existing is not None:
                    merged = merge_row(table, existing, row)
                    if merged != existing:
                        connection.execute(
                            'UPDATE runs SET complete = ?, metadata_json = ? WHERE run_id = ?',
                            (merged[2], merged[3], merged[0]),
                        )
                    continue
                placeholders = ', '.join('?' for _ in columns)
                connection.execute(
                    f'INSERT INTO {table} ({", ".join(columns)}) VALUES ({placeholders})',
                    row,
                )


def write_csv(statistics, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f'{table}.csv' for table in TABLES]
    if any(path.exists() for path in paths):
        raise ValueError(f'{directory}: CSV output files already exist')
    for table, columns in TABLES.items():
        with (directory / f'{table}.csv').open(
            'x', encoding='utf-8', newline=''
        ) as output:
            writer = csv.writer(output)
            writer.writerow(columns)
            writer.writerows(sorted(statistics.tables[table].values()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument('--sqlite', type=Path)
    output.add_argument('--csv', type=Path)
    arguments = parser.parse_args(argv)
    try:
        statistics = Statistics()
        for path in arguments.inputs:
            statistics.merge(load_statistics(path))
        if arguments.sqlite:
            write_sqlite(statistics, arguments.sqlite)
        else:
            write_csv(statistics, arguments.csv)
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f'rocjitsu-stats: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
