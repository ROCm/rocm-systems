#!/usr/bin/env python3
"""Verifier: assert curated_apis.yaml matches the actual HIP/HSA header
declarations via libclang. Per spec §8.3.

Hard errors (exit 1):
- API listed in YAML but not declared in the header.
- YAML arg name not present in header (typo / stale name detection).
- Type mismatch on included args per spec §4.1.
- Arg type mismatch per the type-vocabulary mapping (§4.1) — including
  uint32 used for a C bool parameter (must be the bool DSL type).
- Over-budget API (§4.4 — re-checked via lttng_curated_lib).
- dir: INOUT (§4.4 INOUT-out-of-scope-v1).

Informational warning (does NOT cause exit 1):
- Header parameter declared but not in YAML (intentional omission per spec §4.4 mitigation).

Usage:
    python3 lttng_curated_verify.py \\
        --yaml   path/to/curated_apis.yaml \\
        --header path/to/hip_runtime_api.h \\
        [--header path/to/another_header.h ...] \\
        [--out-sidecar path/to/sigs.json] \\
        [--extra-arg=-I/some/include] [--extra-arg=...]

`--header` may be repeated; declarations from all headers are unioned
before checking YAML APIs (e.g. HSA needs hsa.h + hsa_ext_amd.h).
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lttng_curated_lib import (parse_yaml_file, expanded_field_count,
                               PAYLOAD_BUDGET, ParseError, BudgetError)

# C type -> set of acceptable DSL types.
# Multiple-DSL-type acceptance lets `int` map to either `int32` or `enum`
# (since enums in C are int-typed at the API boundary).
C_TO_DSL = {
    'void *':                  {'ptr'},
    'const void *':            {'ptr'},
    'void **':                 {'ptr'},   # OUT pointer
    'char *':                  {'ptr', 'cstring'},
    'const char *':            {'cstring', 'ptr'},
    'size_t':                  {'size'},
    'unsigned long':           {'size', 'uint64'},
    'unsigned long long':      {'uint64', 'size'},
    'int':                     {'int32', 'enum'},
    'unsigned int':            {'uint32', 'enum'},
    'int32_t':                 {'int32', 'enum'},
    'uint32_t':                {'uint32', 'enum'},
    'int64_t':                 {'int64'},
    'uint64_t':                {'uint64'},
    # HSA's hsa_signal_value_t typedef'd to int64_t under HSA_LARGE_MODEL,
    # but libclang canonicalizes to plain `long` (LP64). Accept both.
    'long':                    {'int64'},
    # Narrow integer types are upper-compatible with the wider DSL
    # uint32 type (used for hipMemsetD8/D16's `value` param). The
    # helper signature widens to uint32_t.
    'unsigned char':           {'uint32'},
    'unsigned short':          {'uint32'},
    'short':                   {'int32'},
    'signed char':             {'int32'},
    'float':                   {'float'},
    'bool':                    {'bool'},   # spec §4.1: hard error if YAML uses uint32
    '_Bool':                   {'bool'},
    'dim3':                    {'dim3', 'dim3_packed'},
    'hipDeviceptr_t':          {'device_ptr'},
    # All opaque handle typedefs map to the `handle` DSL type. The set
    # below is approximate; libclang gives us the underlying canonical type
    # (e.g. `struct ihipStream_t *` for hipStream_t), so we also accept
    # any pointer-to-struct type if the DSL type is `handle`.
}

# Pointer-to-struct types from HIP/HSA — accept as `handle`.
# Substring match against the canonical libclang spelling, e.g.
# `ihipStream_t *` for hipStream_t. Some HIP graph types canonicalize
# to names without the `_t` suffix (`ihipGraph`, `hipGraphExec`,
# `hipGraphNode`); list both shapes here.
HANDLE_TYPE_PATTERNS = (
    'ihipStream_t', 'ihipEvent_t', 'ihipModule_t', 'ihipFunction_t',
    'ihipModuleSymbol_t',
    'ihipGraph_t', 'ihipGraphExec_t', 'ihipGraphNode_t',
    'ihipGraph', 'hipGraphExec', 'hipGraphNode',
    'hipUserObject_t',
    'hsa_signal_t', 'hsa_queue_t', 'hsa_agent_t',
    # HSA value-type handles (struct foo_s) — match against the canonical
    # `hsa_amd_memory_pool_s` form too.
    'hsa_amd_memory_pool_t', 'hsa_amd_memory_pool_s',
)

def _type_is_handle(c_type):
    return any(p in c_type for p in HANDLE_TYPE_PATTERNS)


def _strip_one_pointer(c_type):
    """Strip a single trailing `*` (with optional whitespace) from a C type
    spelling. Used to normalize OUT params: header has `unsigned int *`,
    but the DSL `uint32` describes the value the helper writes through it."""
    s = c_type.rstrip()
    if s.endswith('*'):
        return s[:-1].rstrip()
    return c_type


def _is_compatible(c_type, dsl_type, canonical_type=None, is_enum=False):
    """Return True iff the C type is compatible with the DSL type.

    libclang gives us both the typedef spelling (`hipStream_t`, `size_t`)
    and the canonical spelling (`ihipStream_t *`, `unsigned long`). We
    consult both because:
      - Real HIP/HSA handles like `hipStream_t` only match the
        HANDLE_TYPE_PATTERNS via the canonical `ihipStream_t *` form.
      - `size_t` typedef is matched directly via C_TO_DSL, but
        platforms could canonicalize to `unsigned long` or
        `unsigned long long` — both are accepted via canonical lookup.
      - Typedef'd enums (`hipMemcpyKind`) only present as `enum` in the
        TypeKind, which we surface via `is_enum`.
    """
    candidates = [c_type.strip()]
    if canonical_type and canonical_type.strip() != c_type.strip():
        candidates.append(canonical_type.strip())
    for c in candidates:
        accepted = C_TO_DSL.get(c)
        if accepted and dsl_type in accepted:
            return True
        # Handle types: pointer-to-struct from HIP/HSA accepts `handle`.
        if dsl_type == 'handle' and _type_is_handle(c):
            return True
        # Generic pointer fallback for `ptr` DSL type.
        if dsl_type == 'ptr' and ('*' in c):
            return True
        # OUT pointer to T — accept device_ptr* or hipDeviceptr_t* for device_ptr.
        if dsl_type == 'device_ptr' and 'hipDeviceptr_t' in c:
            return True
        # `enum` DSL type accepts any enum-typed parameter (also handled
        # explicitly via is_enum below for typedef'd enums whose spelling
        # has no `enum` keyword).
        if dsl_type == 'enum' and ('enum' in c or c in ('int', 'unsigned int')):
            return True
    # Typedef'd enums (e.g. `typedef enum X { ... } X;`) come through
    # libclang as TypeKind.ENUM with no `enum` token in the spelling.
    if dsl_type == 'enum' and is_enum:
        return True
    return False


def parse_headers(header_paths, extra_args):
    """Parse one or more headers; return a unioned
    {api_name: [(name, c_type, canonical_type, is_enum), ...]}.

    We capture the typedef spelling (c_type), the canonical spelling
    (canonical_type, e.g. `ihipStream_t *` for `hipStream_t`), and an
    is_enum flag.

    is_enum is True if EITHER the param itself is an enum (typedef'd
    enum like `hipMemcpyKind`) OR the param is a pointer-to-enum (OUT
    enum, e.g. `hipStreamCaptureStatus *`). Both cases are needed by
    the type-compat checker since libclang strips the `enum` keyword
    from typedef'd enums.

    On duplicate api_name across headers (shouldn't happen for HSA, but
    defend against it), the LATER header wins and a warning is emitted.
    """
    try:
        from clang import cindex
    except ImportError:
        sys.exit("ERROR: libclang Python bindings not installed. Try: pip install libclang")
    args = ['-x', 'c++', '-std=c++17'] + list(extra_args)
    idx = cindex.Index.create()
    union = {}
    for hp in header_paths:
        tu = idx.parse(hp, args=args)
        for n in tu.cursor.walk_preorder():
            if n.kind != cindex.CursorKind.FUNCTION_DECL:
                continue
            params = []
            for arg in n.get_arguments():
                ct = arg.type
                canon = ct.get_canonical()
                is_enum = canon.kind == cindex.TypeKind.ENUM
                # OUT enum: param is pointer-to-enum.
                if (not is_enum and canon.kind == cindex.TypeKind.POINTER):
                    pointee = canon.get_pointee()
                    if pointee.kind == cindex.TypeKind.ENUM:
                        is_enum = True
                params.append((arg.spelling, ct.spelling, canon.spelling, is_enum))
            if n.spelling in union and union[n.spelling] != params:
                print(f"WARN: {n.spelling}: declaration in {hp} differs from "
                      f"earlier header; using {hp}", file=sys.stderr)
            union[n.spelling] = params
    return union


def verify(yaml_path, header_paths, extra_args, out_sidecar=None):
    # Parser-level errors (INOUT, over-budget, unknown type/dir, etc.)
    # are spec §8.3 hard errors. Surface them with a clean ERROR: line
    # rather than letting the traceback escape, since this script is a
    # CI gate and noisy tracebacks make failures hard to read.
    try:
        apis = parse_yaml_file(yaml_path)
    except (ParseError, BudgetError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    header_decls = parse_headers(header_paths, extra_args)
    errors = []
    warnings = []
    for api in apis:
        name = api['api']
        if name not in header_decls:
            errors.append(f"{name}: not declared in any of {header_paths}")
            continue
        hdr_params = header_decls[name]
        yaml_args = api['args']
        # Spec §4.4 explicitly allows omitting low-value header params as a
        # field-budget mitigation. Match by NAME, not by count/position.
        hdr_by_name = {hname: (htype, canon, is_enum)
                       for hname, htype, canon, is_enum in hdr_params}
        yaml_names = [a['name'] for a in yaml_args]
        # Hard-error: YAML arg name not in header (typo or stale name).
        for i, yaml_arg in enumerate(yaml_args):
            if yaml_arg['name'] not in hdr_by_name:
                errors.append(
                    f"{name} arg {i}: YAML name {yaml_arg['name']!r} not in "
                    f"header params {list(hdr_by_name)}")
                continue
            htype, canon, is_enum = hdr_by_name[yaml_arg['name']]
            # Spec §4.1: C bool MUST be DSL type bool, not uint32.
            if (htype.strip() in ('bool', '_Bool') or canon.strip() in ('bool', '_Bool')) \
                    and yaml_arg['type'] != 'bool':
                errors.append(
                    f"{name} arg {yaml_arg['name']}: C bool requires DSL type "
                    f"'bool' (spec §4.1), got {yaml_arg['type']!r}")
                continue
            ok = _is_compatible(htype, yaml_arg['type'],
                                canonical_type=canon, is_enum=is_enum)
            # OUT params: the header carries pointer-to-T (e.g.
            # `unsigned int *`), the DSL type describes T (e.g. `uint32`).
            # Retry with one `*` stripped for OUT.
            if not ok and yaml_arg['dir'] == 'OUT':
                ok = _is_compatible(_strip_one_pointer(htype),
                                    yaml_arg['type'],
                                    canonical_type=_strip_one_pointer(canon),
                                    is_enum=is_enum)
            if not ok:
                errors.append(
                    f"{name} arg {yaml_arg['name']}: type mismatch — C "
                    f"{htype!r} (canonical {canon!r}) not compatible with "
                    f"DSL {yaml_arg['type']!r}")
        # Informational: header params not in YAML (intentional partial
        # coverage per spec §4.4 / §8.3 'partial coverage of large APIs is
        # by design').
        for hname, _, _, _ in hdr_params:
            if hname not in yaml_names:
                warnings.append(
                    f"{name}: header param {hname!r} not in YAML "
                    f"(intentional omission per spec §4.4 mitigation?)")
        # Field-budget re-check (also enforced by parser, but verifier is the
        # CI gate so we report it again).
        if expanded_field_count(api) > PAYLOAD_BUDGET:
            errors.append(
                f"{name}: payload exceeds budget of {PAYLOAD_BUDGET} fields")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    for w in warnings:
        print(f"WARN: {w}")
    # Sidecar emission (debate-review C10 fix). Codegen cannot determine
    # provider-correct OUT-handle helper signatures from the YAML alone
    # (HIP `hipStream_t*` is pointer-to-typedef'd-pointer, but HSA
    # `hsa_signal_t*` is pointer-to-struct with `.handle` field). The
    # verifier already has libclang-resolved signatures in header_decls,
    # so dump them as JSON for codegen consumption. Sidecar JSON is
    # checked in alongside generated headers — build still doesn't need
    # libclang at compile time (§3.2 invariant).
    if out_sidecar:
        # Sidecar uses the typedef spelling (c_type) as that is what
        # codegen needs to emit provider-correct OUT-handle helpers
        # (e.g. `hipStream_t*`, not the typedef-stripped form).
        sidecar = {
            a['api']: [{'name': nm, 'c_type': ct}
                       for nm, ct, _canon, _is_enum in header_decls[a['api']]]
            for a in apis if a['api'] in header_decls
        }
        os.makedirs(os.path.dirname(out_sidecar) or '.', exist_ok=True)
        with open(out_sidecar, 'w') as f:
            json.dump(sidecar, f, indent=2, sort_keys=True)
        print(f"wrote signature sidecar: {out_sidecar} "
              f"({len(sidecar)} APIs)", file=sys.stderr)
    print(f"OK: {len(apis)} curated APIs verified against "
          f"{len(header_paths)} header(s)")
    return 0

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--yaml',   required=True)
    ap.add_argument('--header', required=True, action='append',
                    help='Header file to verify against. May be specified '
                         'multiple times; declarations from all headers are '
                         'unioned before checking YAML APIs (e.g. HSA needs '
                         'both hsa.h and hsa_ext_amd.h).')
    ap.add_argument('--extra-arg', action='append', default=[])
    ap.add_argument('--out-sidecar', default=None,
                    help='Optional path to dump verified API signatures as JSON '
                         '(used by codegen for provider-correct OUT-handle '
                         'helper signatures — see Task 3 sidecar mechanism).')
    args = ap.parse_args()
    sys.exit(verify(args.yaml, args.header, args.extra_arg,
                    out_sidecar=args.out_sidecar))

if __name__ == '__main__':
    main()
