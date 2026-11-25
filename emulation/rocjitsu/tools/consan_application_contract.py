# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate the preparation evidence shared by application tools and reports."""

PREPARATION_SCHEMA_VERSION = 2
APPLICATION_SCHEMA_VERSION = 2
MAX_ENTRIES = 1024
MAX_IMAGE_BYTES = 512 * 1024 * 1024
MAX_U64 = (1 << 64) - 1
BLOCKED_REASONS = {
    'unsupported_entry',
    'unsupported_access',
    'unsupported_synchronization',
    'unsupported_instruction',
    'unsupported_flow',
    'unsupported_control_transfer_range',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unsigned(value, name, *, maximum=MAX_U64, minimum=0):
    require(
        type(value) is int and minimum <= value <= maximum,
        f'invalid {name}',
    )
    return value


def digest(value, name):
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in '0123456789abcdef' for character in value)
        and value != '0' * 64,
        f'invalid {name}',
    )
    return value


def validate_no_probes_needed_proof(proof, original_bytes):
    """Validate a producer proof; its owning report binds image SHA and entry key.

    The digest records the producer's decoded original code. Without the original
    image this checks the saved contract, not an independent code audit or a
    security attestation of an untrusted producer.
    """
    require(
        isinstance(proof, dict)
        and set(proof)
        == {'version', 'scope', 'instruction_count', 'code_bytes', 'code_sha256'},
        'invalid no-probes-needed proof fields',
    )
    require(
        type(proof['version']) is int
        and proof['version'] == 1
        and proof['scope'] == 'lds',
        'unsupported no-probes-needed proof scope or version',
    )
    code_bytes = unsigned(
        proof['code_bytes'],
        'no-probes-needed code bytes',
        minimum=4,
        maximum=original_bytes,
    )
    require(code_bytes % 4 == 0, 'invalid no-probes-needed code alignment')
    unsigned(
        proof['instruction_count'],
        'no-probes-needed instruction count',
        minimum=1,
        maximum=code_bytes // 4,
    )
    digest(proof['code_sha256'], 'no-probes-needed code digest')
    return proof


def validate_preparation_report(report):
    """Return a validated schema-2 owning preparation report.

    Schema 1 cannot establish which native entries need no LDS probes. Reject it
    explicitly rather than deriving successful coverage from a legacy skip.
    """
    require(
        isinstance(report, dict)
        and type(report.get('schema_version')) is int
        and report['schema_version'] == PREPARATION_SCHEMA_VERSION,
        'unsupported preparation report schema; regenerate schema-2 evidence',
    )
    digest(report.get('original_sha256'), 'original image identity')
    original_bytes = unsigned(
        report.get('original_bytes'),
        'original image bytes',
        minimum=1,
        maximum=MAX_IMAGE_BYTES,
    )
    unsigned(
        report.get('target'), 'preparation target', minimum=1, maximum=(1 << 32) - 1
    )
    scope = report.get('requested_scope')
    require(scope in ('lds', 'execution_trace'), 'invalid requested preparation scope')
    requested = unsigned(
        report.get('requested'),
        'requested preparation count',
        minimum=1,
        maximum=MAX_ENTRIES,
    )
    inventory_entries = unsigned(
        report.get('inventory_entries'),
        'original inventory count',
        minimum=1,
        maximum=MAX_ENTRIES,
    )
    not_requested = unsigned(
        report.get('not_requested'),
        'not-requested preparation count',
        maximum=inventory_entries,
    )
    require(
        inventory_entries == requested + not_requested,
        'original inventory count differs from requested and not-requested counts',
    )
    entries = report.get('entries')
    require(
        isinstance(entries, list) and len(entries) == requested,
        'invalid preparation entry inventory',
    )
    counts = {'instrumented': 0, 'no_probes_needed': 0, 'blocked': 0}
    names = set()
    keys = set()
    for entry in entries:
        require(isinstance(entry, dict), 'invalid preparation entry')
        name = entry.get('entry')
        key = unsigned(entry.get('entry_key'), 'preparation entry key')
        require(
            isinstance(name, str) and name and name not in names and key not in keys,
            'invalid or duplicate preparation entry identity',
        )
        names.add(name)
        keys.add(key)
        disposition = entry.get('disposition')
        require(disposition in counts, 'unknown preparation entry disposition')
        counts[disposition] += 1
        require(
            entry.get('stage') in ('applicability', 'qualification', 'lowering'),
            'invalid preparation entry stage',
        )
        require(
            isinstance(entry.get('reason'), str), 'invalid preparation entry reason'
        )
        offset = entry.get('text_offset')
        if offset is not None:
            unsigned(offset, 'preparation text offset')
        require('no_probes_needed' in entry, 'missing preparation proof declaration')
        if disposition == 'no_probes_needed':
            require(
                scope == 'lds'
                and entry.get('stage') == 'applicability'
                and entry.get('reason_code') == 'none',
                'no-probes-needed outcome has invalid scope, stage or reason',
            )
            validate_no_probes_needed_proof(entry['no_probes_needed'], original_bytes)
        else:
            require(
                entry['no_probes_needed'] is None, 'unexpected no-probes-needed proof'
            )
            if disposition == 'blocked':
                require(
                    entry.get('reason_code') in BLOCKED_REASONS
                    and bool(entry['reason']),
                    'blocked entry has no supported blocking reason',
                )
            else:
                require(
                    entry.get('reason_code') == 'none',
                    'instrumented entry has a blocking reason',
                )
    for name, count in counts.items():
        unsigned(report.get(name), f'{name} preparation count', maximum=requested)
        require(report[name] == count, 'preparation counts do not match owning entries')
    require(
        type(report.get('skipped')) is int and report['skipped'] == counts['blocked'],
        'legacy skipped count must equal blocked count',
    )
    instrumented = counts['instrumented'] != 0
    require(
        report.get('outcome') == ('instrumented' if instrumented else 'native_only'),
        'preparation outcome does not match selected entries',
    )
    require(
        'selection_profile' in report,
        'missing preparation selection profile declaration',
    )
    if instrumented:
        digest(report.get('selection_profile'), 'selection profile')
    else:
        require(
            report.get('selection_profile') is None,
            'native-only image claims a selection profile',
        )
    coverage = (
        'complete'
        if counts['blocked'] == 0
        else 'not_performed' if counts['blocked'] == requested else 'partial'
    )
    require(
        report.get('requested_coverage') == coverage,
        'preparation coverage contradicts entry outcomes',
    )
    return report
