#!/usr/bin/env python3
"""Append safe full-coverage curated entries from real HIP declarations.

The existing hand-curated prefix in the YAML is preserved verbatim. For every
migration-inventory symbol not already in that prefix, this tool resolves the
real wrapper declaration through lttng_curated_verify.py's libclang machinery
and appends a compact-schema entry. It deliberately never emits `out:`:
pointer arguments are captured as addresses, not dereferenced after the HIP
call.

Implementation-only exported wrappers may be supplied with --source. Their
function definitions override public-header convenience overloads, while
--header still supplies the normal public API declarations.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lttng_curated_lib import parse_yaml_file
from lttng_curated_verify import (AmbiguousDeclarationError,
                                  AmbiguousInferenceError, infer_dsl_type,
                                  parse_declarations, resolve_declaration)


def inventory_names(paths):
    """Return migration-inventory names in first-seen order."""
    seen = set()
    names = []
    for path in paths:
        with open(path) as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                name = line.split('\t', 1)[0]
                if name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


def _bare_type(c_type):
    """Return (unqualified base spelling, pointer depth) for a C type."""
    spelling = c_type.strip()
    depth = spelling.count('*')
    spelling = spelling.rstrip('* \t')
    for qualifier in ('const ', 'volatile '):
        if spelling.startswith(qualifier):
            spelling = spelling[len(qualifier):].strip()
    return spelling, depth


def _is_cstring(c_type):
    base, depth = _bare_type(c_type)
    return base == 'char' and depth == 1


def _is_dim3_by_value(c_type):
    base, depth = _bare_type(c_type)
    return base == 'dim3' and depth == 0


def _skip_reason(c_type, canonical, error):
    """Classify a documented policy-permitted omission."""
    spelling = c_type.strip()
    if ('[' in spelling and ']' in spelling) or ('[' in canonical and ']' in canonical):
        return 'fixed-size C array parameter is not representable by the DSL'
    base, depth = _bare_type(spelling)
    if depth == 0 and '&' not in base:
        return 'by-value C struct is not representable by the DSL'
    return f'ambiguous DSL inference: {error}'


def render_entry(name, args, strings, pack_dim3):
    """Render one compact YAML entry without reserializing the existing prefix."""
    lines = [f'- api: {name}', f"  args: [{', '.join(args)}]"]
    if strings:
        lines.append(f"  strings: [{', '.join(strings)}]")
    if pack_dim3:
        lines.append(f"  pack_dim3: [{', '.join(pack_dim3)}]")
    return '\n'.join(lines)


def generate(yaml_path, inventory_paths, header_paths, source_paths, extra_args):
    with open(yaml_path) as file:
        original = file.read()
    existing = {api['api'] for api in parse_yaml_file(yaml_path)}
    declarations = parse_declarations(header_paths, source_paths, extra_args)

    additions = []
    skipped = []
    captured = 0
    for name in inventory_names(inventory_paths):
        if name in existing:
            continue
        try:
            params = resolve_declaration(name, declarations.get(name, []), [])
        except AmbiguousDeclarationError as error:
            raise SystemExit(f'ERROR: {error}')
        if params is None:
            raise SystemExit(f'ERROR: {name}: no declaration or wrapper definition found')

        args = []
        strings = []
        pack_dim3 = []
        for arg_name, c_type, canonical, is_enum in params:
            if _is_cstring(c_type):
                args.append(arg_name)
                strings.append(arg_name)
                captured += 1
                continue
            if _is_dim3_by_value(c_type):
                args.append(arg_name)
                pack_dim3.append(arg_name)
                captured += 1
                continue
            try:
                infer_dsl_type(c_type, canonical, is_enum, 'IN')
            except AmbiguousInferenceError as error:
                skipped.append({
                    'api': name,
                    'arg': arg_name,
                    'c_type': c_type,
                    'reason': _skip_reason(c_type, canonical, error),
                })
                continue
            args.append(arg_name)
            captured += 1
        additions.append(render_entry(name, args, strings, pack_dim3))

    if additions:
        generated = original.rstrip() + '\n\n# --- auto-curated (full migration-inventory coverage) ---\n\n'
        generated += '\n\n'.join(additions) + '\n'
    else:
        generated = original
    return generated, additions, captured, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', required=True)
    parser.add_argument('--inventory', required=True, action='append')
    parser.add_argument('--header', required=True, action='append')
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument('--extra-arg', action='append', default=[])
    parser.add_argument('--output', required=True,
                        help='Output YAML path; may be the same as --yaml.')
    parser.add_argument('--skip-report', required=True,
                        help='Write the policy-permitted skipped-argument ledger as JSON.')
    parser.add_argument('--check', action='store_true',
                        help='Do not write --output; fail if it lacks generated additions.')
    args = parser.parse_args()

    generated, additions, captured, skipped = generate(
        args.yaml, args.inventory, args.header, args.source, args.extra_arg)
    report = {
        'api_count_added': len(additions),
        'arguments_captured': captured,
        'arguments_skipped': len(skipped),
        'skipped': skipped,
    }
    with open(args.skip_report, 'w') as file:
        json.dump(report, file, indent=2)
        file.write('\n')

    if args.check:
        with open(args.output) as file:
            current = file.read()
        if current != generated:
            print(f'ERROR: {args.output} is missing generated curated entries', file=sys.stderr)
            return 1
    else:
        with open(args.output, 'w') as file:
            file.write(generated)
    print(f'curated additions: {len(additions)} APIs, {captured} args captured, '
          f'{len(skipped)} args skipped')
    return 0


if __name__ == '__main__':
    sys.exit(main())
