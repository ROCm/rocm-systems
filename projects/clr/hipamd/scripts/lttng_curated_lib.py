"""Shared parser/validator library for the LTTng curated-args DSL.

Used by:
  - lttng_curated_codegen.py   (generates tracepoint header + emit helpers)
  - lttng_curated_verify.py    (libclang vs YAML drift check; CI gate)
  - lttng_migrate.py           (selects _CURATED vs _CURATED_NOARGS macro variants)

Schema and field-budget rules are normative per:
  docs/superpowers/specs/2026-04-26-lttng-curated-args-design.md §4.

The library has NO libclang dependency — that is isolated to the verifier
script. PyYAML is the only third-party dependency.
"""
import yaml

# ---- DSL vocabulary (spec §4.1) ----
DSL_TYPES = frozenset([
    'handle', 'ptr', 'device_ptr', 'size',
    'int32', 'uint32', 'int64', 'uint64',
    'float', 'enum', 'bool', 'dim3', 'dim3_packed', 'cstring',
])
ALLOWED_DIRS = frozenset(['IN', 'OUT', 'INOUT'])
ALLOWED_CATEGORIES = frozenset([
    'streams', 'events', 'kernel_launch', 'memory', 'graphs', 'module',
    'hsa_queues', 'hsa_signals', 'hsa_memory',
])

# Per spec §4.4: budget is 10 LTTng fields total including corr_id => 9 payload max.
PAYLOAD_BUDGET = 9

# Type expansion (spec §4.4).
TYPE_EXPANSION = {
    'dim3': 3, 'dim3_packed': 1,
    # All others expand to 1.
}

# Direction expansion (spec §4.4): INOUT contributes 2 (input + <name>_out).
DIR_EXPANSION = {'IN': 1, 'OUT': 1, 'INOUT': 2}


class ParseError(Exception):
    """Schema-level error in the YAML (missing field, bad type, INOUT in v1, etc.)."""

class BudgetError(Exception):
    """Field-budget violation (>9 payload fields after expansion)."""


def _type_expand(arg):
    return TYPE_EXPANSION.get(arg['type'], 1)


def _dir_expand(arg):
    return DIR_EXPANSION[arg['dir']]


def expanded_field_count(api):
    """Return total payload field count (excluding corr_id) after both
    type-expansion and direction-expansion (spec §4.4 normative rule)."""
    return sum(_type_expand(a) * _dir_expand(a) for a in api['args'])


def IN_ARG_KIND(arg):
    """Return 'IN' | 'OUT' | 'INOUT' — the arg's direction."""
    return arg['dir']


def _require_keys(d, required, ctx):
    missing = [k for k in required if k not in d]
    if missing:
        raise ParseError(f"{ctx}: missing required field(s): {', '.join(missing)}")


def _validate_arg(arg, api_name):
    _require_keys(arg, ['name', 'type', 'dir'], f"{api_name} arg")
    if arg['type'] not in DSL_TYPES:
        raise ParseError(
            f"{api_name} arg {arg['name']}: unknown type {arg['type']!r}; "
            f"valid: {sorted(DSL_TYPES)}")
    if arg['dir'] not in ALLOWED_DIRS:
        raise ParseError(
            f"{api_name} arg {arg['name']}: unknown dir {arg['dir']!r}; "
            f"valid: {sorted(ALLOWED_DIRS)}")
    # Spec §4.4 INOUT-out-of-scope-v1: hard error in codegen + verifier.
    if arg['dir'] == 'INOUT':
        raise ParseError(
            f"{api_name} arg {arg['name']}: dir: INOUT is out-of-scope for v1 "
            f"(spec §4.4 'INOUT scope (v1)'); model as IN or OUT instead")


def validate_api(api):
    """Validate one API entry. Raises ParseError or BudgetError."""
    _require_keys(api, ['api', 'category', 'args'], "API entry")
    if not isinstance(api['args'], list):
        raise ParseError(f"{api['api']}: 'args' must be a list")
    if api['category'] not in ALLOWED_CATEGORIES:
        raise ParseError(
            f"{api['api']}: unknown category {api['category']!r}; "
            f"valid: {sorted(ALLOWED_CATEGORIES)}")
    for arg in api['args']:
        _validate_arg(arg, api['api'])
    n = expanded_field_count(api)
    if n > PAYLOAD_BUDGET:
        raise BudgetError(
            f"{api['api']}: payload has {n} fields, exceeds budget of "
            f"{PAYLOAD_BUDGET} (spec §4.4). Apply mitigation: type as "
            f"dim3_packed and/or omit low-value args.")


def parse_yaml_text(text):
    """Parse YAML text into a list of validated API dicts."""
    raw = yaml.safe_load(text)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ParseError(f"top level must be a YAML list; got {type(raw).__name__}")
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ParseError(f"each list entry must be a mapping; got {type(entry).__name__}")
        validate_api(entry)
        out.append(entry)
    # Sanity: no duplicate api names.
    seen = set()
    for a in out:
        if a['api'] in seen:
            raise ParseError(f"duplicate api: {a['api']}")
        seen.add(a['api'])
    return out


def parse_yaml_file(path):
    with open(path, 'r') as f:
        return parse_yaml_text(f.read())
