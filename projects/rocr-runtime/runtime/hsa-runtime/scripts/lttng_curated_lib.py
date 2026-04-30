"""Shared parser/validator library for the LTTng curated-args DSL.

Used by the coverage gate (lttng_coverage_check.py / lttng_coverage_gate.sh)
and by the unit tests (test_lttng_curated_lib.py). PyYAML is the only
third-party dependency.

Schema, field-budget rules, and the INOUT-not-supported-in-v1 rule are
normative for the curated-args DSL.
"""
import yaml

# ---- DSL vocabulary ----
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

# Field budget: LTTng-UST allows at most 10 fields per tracepoint event;
# none of those slots are reserved by the curated framework now that
# corr_id is no longer carried as an explicit field. Identity (vpid, vtid,
# timestamp) comes from channel context, not the event payload.
PAYLOAD_BUDGET = 10

# Type expansion (number of payload fields each DSL type emits).
TYPE_EXPANSION = {
    'dim3': 3, 'dim3_packed': 1,
    # All others expand to 1.
}

# Direction expansion: INOUT contributes 2 (input + <name>_out).
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
    """Return total payload field count after both type-expansion and
    direction-expansion."""
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
    # INOUT is out-of-scope for v1; model the parameter as IN or OUT instead.
    if arg['dir'] == 'INOUT':
        raise ParseError(
            f"{api_name} arg {arg['name']}: dir: INOUT is out-of-scope for v1; "
            f"model as IN or OUT instead")


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
            f"{PAYLOAD_BUDGET}. Apply mitigation: type as dim3_packed "
            f"and/or omit low-value args.")


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
