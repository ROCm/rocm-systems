"""Shared parser/validator library for the LTTng curated-args DSL.

Used by the coverage gate (lttng_coverage_check.py / lttng_coverage_gate.sh)
and by the unit tests (test_lttng_curated_lib.py). PyYAML is the only
third-party dependency.

Schema (Phase 3 compact form) per API entry:
    - api: <name>                  # must match the real C function name
      args: [<name>, ...]          # allowlist of captured params, in order
      out: [<name>, ...]           # optional; subset of args captured OUT
                                    # (default: every arg is IN)
      pack_dim3: [<name>, ...]     # optional; subset of args whose real
                                    # C type is `dim3` to pack into a single
                                    # uint64 field instead of expanding x/y/z
      strings: [<name>, ...]       # optional; subset of args whose real
                                    # C type is `char*`/`const char*` to
                                    # capture as a string, not a raw pointer
      types: {<name>: <DSL type>}  # optional; explicit DSL-type override
                                    # for an arg, only needed when the real
                                    # C type is genuinely ambiguous (see
                                    # lttng_curated_verify.infer_dsl_type())

`args`/`out`/`pack_dim3`/`strings`/`types` fully describe direction and
capture-shape policy (never inferable from the C type alone). Per-arg DSL
*type* for everything else is inferred from the real, live-libclang-parsed
C signature by lttng_curated_verify.expand_compact_apis() — this module
has no header access and cannot do that itself, so parse_yaml_file() here
only performs schema-shape validation and direction resolution; the
returned dicts do NOT have a 'type' key per arg until expand_compact_apis()
fills it in.

A second, fully-explicit per-arg shape (`args: [{name, type, dir}, ...]`)
is also still accepted, purely so libclang-independent test fixtures (see
test_lttng_curated_codegen.py's `--sigs`-driven cases) can hand-describe a
synthetic API's already-known types without needing a header to infer
against. Neither real curated_apis.yaml uses this form any more; only test
fixtures do. An explicit-shape entry is validated (including the field
budget, since its type is already known) and returned unchanged.

Field-budget rules are normative and enforced here (compact-schema
entries: post-expansion, by lttng_curated_verify.expand_compact_apis();
explicit-schema entries: immediately, since their types are already
known). `category` (removed) and `dir: INOUT` (never implemented, removed
from the direction vocabulary) are not part of the schema.
"""
import yaml

# ---- DSL vocabulary ----
DSL_TYPES = frozenset([
    'handle', 'ptr', 'device_ptr', 'size',
    'int32', 'uint32', 'int64', 'uint64',
    'float', 'enum', 'bool', 'dim3', 'dim3_packed', 'cstring',
])
ALLOWED_DIRS = frozenset(['IN', 'OUT'])

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

DIR_EXPANSION = {'IN': 1, 'OUT': 1}


class ParseError(Exception):
    """Schema-level error in the YAML (missing field, bad type, unknown
    dir, out/pack_dim3/strings/types referencing an arg not in `args`,
    etc.)."""

class BudgetError(Exception):
    """Field-budget violation (>10 payload fields after expansion)."""


def _type_expand(arg):
    return TYPE_EXPANSION.get(arg['type'], 1)


def _dir_expand(arg):
    return DIR_EXPANSION[arg['dir']]


def expanded_field_count(api):
    """Return total payload field count after both type-expansion and
    direction-expansion. `api['args']` entries must already have both
    'type' and 'dir' populated (i.e. this is the fully-expanded/resolved
    representation, not a freshly-parsed compact-schema entry)."""
    return sum(_type_expand(a) * _dir_expand(a) for a in api['args'])


def IN_ARG_KIND(arg):
    """Return 'IN' | 'OUT' — the arg's direction."""
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


def validate_api(api):
    """Validate one FULLY EXPANDED API entry ({api, args: [{name, type,
    dir}, ...]}) — every arg must already have a real DSL 'type'. Raises
    ParseError or BudgetError.

    `category` is intentionally not part of the schema: it had exactly
    one consumer (this function's own now-removed requirement) and zero
    downstream consumers (codegen, the coverage gates, and the verifier
    never read it) as of the Phase 3 audit, so it was dropped rather than
    kept as unused documentation cruft."""
    _require_keys(api, ['api', 'args'], "API entry")
    if not isinstance(api['args'], list):
        raise ParseError(f"{api['api']}: 'args' must be a list")
    for arg in api['args']:
        _validate_arg(arg, api['api'])
    n = expanded_field_count(api)
    if n > PAYLOAD_BUDGET:
        raise BudgetError(
            f"{api['api']}: payload has {n} fields, exceeds budget of "
            f"{PAYLOAD_BUDGET}. Apply mitigation: pack_dim3 the arg "
            f"and/or drop it from 'args' entirely.")


# ---------------------------------------------------------------------------
# Compact-schema top-level parsing
# ---------------------------------------------------------------------------

def _validate_name_list(names, args, api_name, field_name):
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ParseError(f"{api_name}: '{field_name}' must be a list of arg-name strings")
    unknown = [n for n in names if n not in args]
    if unknown:
        raise ParseError(
            f"{api_name}: '{field_name}' references arg(s) {unknown} not in 'args'")
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ParseError(f"{api_name}: '{field_name}' has duplicate name(s) {dupes}")


def _parse_explicit_entry(entry):
    """Fully-explicit per-arg shape (`args: [{name, type, dir}, ...]`) —
    already self-describing, validated and returned as-is (unchanged
    behavior from the pre-Phase-3 schema). Used only by libclang-
    independent test fixtures."""
    validate_api(entry)
    return entry


def parse_api_entry(entry):
    """Parse+statically validate one API entry. Dispatches on the shape
    of `args`:
      - list of {name, type, dir} mappings -> explicit shape, returned
        unchanged (already fully resolved and budget-checked).
      - list of plain name strings (compact shape, or empty) -> schema
        shape is validated and direction is resolved here (no header
        needed for that), but per-arg DSL 'type' is deliberately left
        unset; callers needing it must resolve real C types (e.g. via a
        live libclang parse) and call
        lttng_curated_verify.expand_compact_apis()."""
    _require_keys(entry, ['api', 'args'], "API entry")
    name = entry['api']
    args = entry['args']
    if not isinstance(args, list):
        raise ParseError(f"{name}: 'args' must be a list")

    if args and isinstance(args[0], dict):
        return _parse_explicit_entry(entry)

    if not all(isinstance(a, str) for a in args):
        raise ParseError(
            f"{name}: 'args' must be a list of arg-name strings (compact "
            f"schema) or a list of {{name, type, dir}} mappings (explicit "
            f"schema, test fixtures only)")
    if len(set(args)) != len(args):
        dupes = sorted({a for a in args if args.count(a) > 1})
        raise ParseError(f"{name}: 'args' has duplicate name(s) {dupes}")

    out = entry.get('out', [])
    pack_dim3 = entry.get('pack_dim3', [])
    strings_ = entry.get('strings', [])
    types_override = entry.get('types', {})

    _validate_name_list(out, args, name, 'out')
    _validate_name_list(pack_dim3, args, name, 'pack_dim3')
    _validate_name_list(strings_, args, name, 'strings')

    if not isinstance(types_override, dict):
        raise ParseError(f"{name}: 'types' must be a mapping of arg-name -> DSL type")
    _validate_name_list(list(types_override), args, name, 'types')
    for arg_name, ty in types_override.items():
        if ty not in DSL_TYPES:
            raise ParseError(
                f"{name} arg {arg_name}: unknown type override {ty!r}; "
                f"valid: {sorted(DSL_TYPES)}")

    overlap = sorted(set(pack_dim3) & set(strings_))
    if overlap:
        raise ParseError(
            f"{name}: arg(s) {overlap} listed in both 'pack_dim3' and 'strings'")

    out_set = set(out)
    return {
        'api': name,
        'args': [{'name': a, 'dir': 'OUT' if a in out_set else 'IN'} for a in args],
        'pack_dim3': pack_dim3,
        'strings': strings_,
        'types_override': types_override,
    }


def parse_yaml_text(text):
    """Parse YAML text into a list of API dicts. See parse_api_entry() for
    the two accepted per-entry shapes and what is/isn't resolved yet."""
    raw = yaml.safe_load(text)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ParseError(f"top level must be a YAML list; got {type(raw).__name__}")
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ParseError(f"each list entry must be a mapping; got {type(entry).__name__}")
        out.append(parse_api_entry(entry))
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
