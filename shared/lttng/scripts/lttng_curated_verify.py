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
import argparse, functools, json, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lttng_curated_lib import (parse_yaml_file, expanded_field_count,
                               PAYLOAD_BUDGET, ParseError, BudgetError,
                               validate_api)

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


# ---------------------------------------------------------------------------
# Type inference (Phase 3): given a real, live-libclang-resolved C
# signature, infer the DSL type for an arg that has no explicit `types:`
# override in the compact YAML schema. This is deliberately a STRICTER,
# single-answer sibling of C_TO_DSL/_is_compatible above (which is
# intentionally PERMISSIVE — it accepts multiple DSL types for one C type
# so verification of a human-declared type doesn't false-positive). See
# lttng_curated_lib.py's module docstring for the compact schema shape.
# ---------------------------------------------------------------------------

# Each handle typedef's own "natural" pointer depth — the number of `*`
# characters in the TYPEDEF SPELLING (not the canonical type) when the
# handle is captured directly as an IN arg by value. Needed because HIP's
# handle typedefs already hide one level of pointer inside the typedef
# itself (`hipStream_t` IN arg spelling is bare `hipStream_t`, depth 0),
# while HSA's are a mix: most are value-structs also captured bare (depth
# 0: hsa_agent_t, hsa_signal_t, hsa_amd_memory_pool_t), but hsa_queue_t is
# the one exception — HSA queues are heap-allocated structs the API always
# addresses via an explicit pointer, so even the plain IN form is spelled
# `hsa_queue_t *` (depth 1). An OUT arg of a given handle type is spelled
# with exactly one MORE `*` than its natural depth (writing the handle's
# value back through a pointer). This table is the direct analog, for
# inference, of HANDLE_TYPE_PATTERNS above (used for permissive
# compatibility checking) — extend both together when a new handle type
# is curated.
HANDLE_NATURAL_DEPTH = {
    'hipStream_t': 0, 'hipEvent_t': 0, 'hipModule_t': 0, 'hipFunction_t': 0,
    'hipGraph_t': 0, 'hipGraphExec_t': 0, 'hipGraphNode_t': 0,
    'hsa_agent_t': 0, 'hsa_signal_t': 0, 'hsa_amd_memory_pool_t': 0,
    'hsa_queue_t': 1,
}

# Named-typedef -> single DSL type, matched against the TYPEDEF spelling
# (c_type, not canonical) first. Deliberately single-valued (unlike
# C_TO_DSL's sets): by the time inference reaches this table, the is_enum
# check has already peeled off every genuine enum case, so a plain
# `uint32_t`/`int32_t` etc. arg is unambiguous.
NUMERIC_TYPEDEF_MAP = {
    'size_t': 'size', 'uint32_t': 'uint32', 'int32_t': 'int32',
    'uint64_t': 'uint64', 'int64_t': 'int64',
}

# Canonical-spelling fallback, used only when the typedef spelling itself
# isn't one of NUMERIC_TYPEDEF_MAP's keys (e.g. a plain `unsigned int`
# parameter with no named typedef, or a typedef like
# `hsa_queue_type32_t`/`hsa_amd_sdma_engine_id_t` whose canonical form is
# a builtin integer type). `unsigned long`/`unsigned long long` resolve
# to uint64 here (not `size`, despite size_t canonicalizing to `unsigned
# long` on LP64) because every genuine byte-count arg in both curated
# YAMLs is spelled `size_t` at the typedef level and is caught by
# NUMERIC_TYPEDEF_MAP first; a raw `unsigned long` with no size_t typedef
# is far more likely a plain wide integer than a size. If a future API
# genuinely needs `size` inferred from a bare `unsigned long`, add an
# explicit `types:` override rather than changing this default.
NUMERIC_CANONICAL_MAP = {
    'unsigned int': 'uint32', 'int': 'int32',
    'unsigned long': 'uint64', 'unsigned long long': 'uint64', 'long': 'int64',
    'unsigned char': 'uint32', 'unsigned short': 'uint32',
    'short': 'int32', 'signed char': 'int32',
}


def _bare_spelling(spelling):
    """Return (bare_name, pointer_depth) for a C type spelling: strip all
    trailing `*` (counting them as the depth), then strip a leading
    `const `/`volatile ` qualifier."""
    s = spelling.strip()
    depth = s.count('*')
    s = s.rstrip('* \t')
    for qual in ('const ', 'volatile '):
        if s.startswith(qual):
            s = s[len(qual):].strip()
    return s, depth


class AmbiguousInferenceError(Exception):
    """Raised when infer_dsl_type() cannot determine a single DSL type for
    an arg and no explicit `types:` override was given in the YAML."""


def infer_dsl_type(c_type, canonical, is_enum, dir_):
    """Infer the DSL type for one arg from its real, libclang-resolved C
    signature (c_type = typedef spelling, canonical = canonical spelling,
    is_enum = True if the (possibly pointed-to) type is a C enum) and its
    already-known direction. Raises AmbiguousInferenceError if no single
    DSL type can be determined — the caller (expand_compact_apis) turns
    that into a clear "add an explicit types: override" error.

    Deliberately does NOT decide dim3 vs dim3_packed or ptr vs cstring —
    those are `pack_dim3:`/`strings:` policy choices applied by the
    caller before/after this function, never inferred from the C type
    alone (a `dim3` struct's packing is a field-budget decision; a
    `char*`/`const char*` could be a string or a raw buffer)."""
    if '(*)' in canonical:
        # Function pointer (e.g. HIP's hipStreamCallback_t / HSA's queue
        # error callback) — always a raw pointer, and canonical spellings
        # embed their own parameter list which can spuriously substring-
        # match a handle/enum name, so this check must come first.
        return 'ptr'

    expected_extra = 1 if dir_ == 'OUT' else 0
    bare, depth = _bare_spelling(c_type)

    if bare == 'hipDeviceptr_t' and depth == expected_extra:
        return 'device_ptr'
    if bare == 'dim3' and depth == expected_extra:
        return 'dim3'
    if bare in HANDLE_NATURAL_DEPTH and depth == HANDLE_NATURAL_DEPTH[bare] + expected_extra:
        return 'handle'
    if is_enum and depth == expected_extra:
        return 'enum'
    if bare in ('bool', '_Bool') and depth == expected_extra:
        return 'bool'
    if bare == 'float' and depth == expected_extra:
        return 'float'
    if depth == expected_extra and bare in NUMERIC_TYPEDEF_MAP:
        return NUMERIC_TYPEDEF_MAP[bare]

    canon_bare, canon_depth = _bare_spelling(canonical)
    if canon_depth == expected_extra and canon_bare in NUMERIC_CANONICAL_MAP:
        return NUMERIC_CANONICAL_MAP[canon_bare]

    # Generic pointer fallback: any remaining pointer-shaped arg that
    # isn't a recognized handle/enum/numeric-typedef at the expected
    # depth (arrays of handles like `const hipGraphNode_t *`, `void*`/
    # `void**`, struct pointers like `const hipKernelNodeParams *`,
    # `char*`/`const char*` when not overridden via `strings:`, etc.)
    # captures as a raw pointer.
    if depth >= 1:
        return 'ptr'

    raise AmbiguousInferenceError(
        f"cannot infer a DSL type for C type {c_type!r} (canonical "
        f"{canonical!r}, is_enum={is_enum}, dir={dir_}); add an explicit "
        f"'types:' override for this arg")


def expand_compact_apis(raw_apis, header_decls):
    """Resolve every compact-schema api in `raw_apis` (as returned by
    lttng_curated_lib.parse_yaml_file()) against `header_decls` (as
    returned by parse_headers()) into the fully-expanded, budget-checked
    internal representation lttng_curated_codegen.py and
    lttng_curated_lib.validate_api() expect: {api, args: [{name, type,
    dir}, ...]}.

    Explicit-shape entries (already have 'type' per arg — see
    lttng_curated_lib.parse_api_entry()) are already fully resolved and
    pass through unchanged after a validate_api() re-check.

    Raises AmbiguousDeclarationError (duplicate/ambiguous header
    declaration), AmbiguousInferenceError (a real C type this function
    cannot map to a single DSL type and no `types:` override was given),
    or lttng_curated_lib.BudgetError (expanded payload exceeds the field
    budget)."""
    out = []
    for api in raw_apis:
        if api['args'] and 'type' in api['args'][0]:
            validate_api(api)
            out.append(api)
            continue

        name = api['api']
        arg_names = [a['name'] for a in api['args']]
        params = resolve_declaration(name, header_decls.get(name, []), arg_names) \
            if name in header_decls else None
        if params is None:
            raise AmbiguousDeclarationError(f"{name}: no declarations found")
        by_name = {p[0]: p for p in params}

        pack_dim3 = set(api.get('pack_dim3', []))
        strings_ = set(api.get('strings', []))
        types_override = api.get('types_override', {})

        expanded_args = []
        for a in api['args']:
            arg_name, dir_ = a['name'], a['dir']
            if arg_name in types_override:
                ty = types_override[arg_name]
            elif arg_name not in by_name:
                raise ParseError(
                    f"{name} arg {arg_name}: not present in header params "
                    f"{list(by_name)}")
            else:
                _, c_type, canon, is_enum = by_name[arg_name]
                if arg_name in strings_:
                    ty = 'cstring'
                else:
                    ty = infer_dsl_type(c_type, canon, is_enum, dir_)
                    if arg_name in pack_dim3:
                        if ty != 'dim3':
                            raise ParseError(
                                f"{name} arg {arg_name}: 'pack_dim3' given but "
                                f"real C type is not dim3 (inferred {ty!r})")
                        ty = 'dim3_packed'
            expanded_args.append({'name': arg_name, 'type': ty, 'dir': dir_})

        expanded = {'api': name, 'args': expanded_args}
        validate_api(expanded)
        out.append(expanded)
    return out


@functools.lru_cache(maxsize=1)
def _system_isystem_args():
    """Auto-detect the host C++ compiler's default system include search
    path and return it as a flat list of `-isystem <path>` args.

    The standalone libclang shared library shipped by `pip install
    libclang` is not invoked through the clang driver, so it has none of
    the driver's "find the local GCC/Clang installation's system header
    dirs" logic. Parsing any header that transitively pulls in
    <cstddef>/<climits>/<cstdlib> etc. then silently resolves types like
    `size_t` to nothing usable and falls back to plain `int` instead of
    erroring loudly — e.g. `size_t sizeBytes` reads back from libclang
    as C type `int`. That is silently WRONG (not merely "type
    unavailable"), and is exactly the kind of case Phase 3 type
    inference depends on getting right. Probe the host's own
    clang++/g++ for the exact system include list a normal `clang++
    file.cpp` invocation would use, and feed it to libclang explicitly.

    Best-effort and fully backward compatible: if no compiler is found
    or probing fails for any reason, return [] and libclang's own
    (possibly limited) default search behavior is unchanged from before
    this existed."""
    cxx = shutil.which('clang++') or shutil.which('g++')
    if not cxx:
        return []
    try:
        r = subprocess.run([cxx, '-E', '-x', 'c++', '-v', '-'], input='',
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    paths = []
    in_list = False
    for ln in r.stderr.splitlines():
        if ln.startswith('#include <...> search starts here'):
            in_list = True
            continue
        if ln.startswith('End of search list'):
            break
        if in_list:
            paths.append(ln.strip())
    args = []
    for p in paths:
        args += ['-isystem', p]
    return args


def parse_headers(header_paths, extra_args):
    """Parse one or more headers; return RAW (unresolved) declarations:
    {api_name: [(is_static, (param, ...)), ...]}, where each param is
    (name, c_type, canonical_type, is_enum).

    We capture the typedef spelling (c_type), the canonical spelling
    (canonical_type, e.g. `ihipStream_t *` for `hipStream_t`), and an
    is_enum flag.

    is_enum is True if EITHER the param itself is an enum (typedef'd
    enum like `hipMemcpyKind`) OR the param is a pointer-to-enum (OUT
    enum, e.g. `hipStreamCaptureStatus *`). Both cases are needed by
    the type-compat checker since libclang strips the `enum` keyword
    from typedef'd enums.

    Real headers commonly declare the same function name more than
    once — e.g. HIP's hipMallocAsync has a plain extern declaration
    plus a later `static inline` convenience overload with an extra
    `mem_pool` parameter — and the same name can appear (identically or
    not) across multiple `--header` files. This function does NOT pick
    a winner among duplicates; every DISTINCT (is_static, params)
    signature observed is kept as a candidate (deduplicated by exact
    content, so an identical redeclaration across headers doesn't
    produce a spurious extra candidate). Use resolve_declaration() /
    resolve_signatures() to pick the authoritative signature for a
    specific API once the caller knows which arg names it actually
    needs — silently keeping "whichever was seen last" here is exactly
    the bug this two-step design avoids (it used to make hipMallocAsync
    resolve to its 4-arg `mem_pool` overload instead of the real 3-arg
    extern API).
    """
    try:
        from clang import cindex
    except ImportError:
        sys.exit("ERROR: libclang Python bindings not installed. Try: pip install libclang")
    args = ['-x', 'c++', '-std=c++17'] + list(extra_args) + _system_isystem_args()
    idx = cindex.Index.create()
    candidates = {}
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
            is_static = n.storage_class == cindex.StorageClass.STATIC
            entry = (is_static, tuple(params))
            lst = candidates.setdefault(n.spelling, [])
            if entry not in lst:
                lst.append(entry)
    return candidates


class AmbiguousDeclarationError(Exception):
    """Raised when an API has multiple candidate declarations (see
    parse_headers()) and resolve_declaration() cannot automatically
    determine which one is authoritative."""


def _candidate_sig_str(name, is_static, params):
    args_str = ', '.join(f"{ct} {pn}" for pn, ct, _c, _e in params)
    return f"{'static ' if is_static else ''}{name}({args_str})"


def resolve_declaration(name, candidates, yaml_arg_names):
    """Pick the single authoritative declaration for `name` out of the
    candidates parse_headers() found, given the arg names the YAML/DSL
    actually references for this API.

    Resolution order:
      1. Exactly one candidate -> trivially authoritative.
      2. Candidates whose parameter names are a superset of the YAML's
         arg names ("eligible") are preferred over ones that are
         missing a name the YAML needs; if none are eligible, every
         candidate stays in play (so the ambiguity error below still
         has something to report against).
      3. Among the remaining candidates, prefer the sole non-`static`
         one. A `static inline` overload is an opt-in convenience
         wrapper layered on top of the real (extern) API, not the API
         itself — e.g. HIP's hipMallocAsync has a 3-arg extern
         declaration plus a later 4-arg `static inline` `mem_pool`
         overload; naive "last declaration wins" logic used to resolve
         to the overload, which is wrong.
      4. If exactly one candidate remains after that narrowing, it
         wins. Otherwise the choice is genuinely ambiguous: fail loudly
         with every candidate signature listed so a human can resolve
         it, rather than silently picking whichever libclang happened
         to walk last.
    """
    if not candidates:
        raise AmbiguousDeclarationError(f"{name}: no declarations found")
    if len(candidates) == 1:
        return list(candidates[0][1])

    yaml_names = set(yaml_arg_names)
    eligible = [c for c in candidates if yaml_names <= {p[0] for p in c[1]}]
    search_space = eligible or candidates
    non_static = [c for c in search_space if not c[0]]

    winner = None
    if len(non_static) == 1:
        winner = non_static[0]
    elif len(search_space) == 1:
        winner = search_space[0]

    if winner is not None:
        return list(winner[1])

    lines = [f"{name}: ambiguous declaration — {len(candidates)} candidate "
             f"signatures found for YAML args {sorted(yaml_names)} and none "
             f"can be preferred automatically:"]
    for is_static, params in candidates:
        lines.append(f"  {_candidate_sig_str(name, is_static, params)}")
    raise AmbiguousDeclarationError('\n'.join(lines))


def resolve_signatures(apis, header_decls):
    """Resolve every YAML api present in header_decls to its single
    winning declaration's param list. Returns {api_name: params}. Apis
    not found in header_decls are silently skipped — callers that need
    "must be declared somewhere" hard-error semantics (i.e. verify())
    check that separately, since it's a distinct error class.

    Raises AmbiguousDeclarationError if any api's candidates can't be
    resolved automatically (see resolve_declaration())."""
    resolved = {}
    for api in apis:
        name = api['api']
        if name not in header_decls:
            continue
        yaml_names = [a['name'] for a in api['args']]
        resolved[name] = resolve_declaration(name, header_decls[name], yaml_names)
    return resolved


def compute_sidecar(apis, header_decls):
    """Build the {api: [{name, c_type}, ...]} signature sidecar
    structure lttng_curated_codegen.py needs for provider-correct
    OUT-handle helper generation. Shared by lttng_curated_verify.py's
    --out-sidecar (CI drift gate) and lttng_curated_codegen.py's
    --header / --dump-resolved live-parsing mode — this is the one
    place duplicate-declaration resolution happens."""
    resolved = resolve_signatures(apis, header_decls)
    return {name: [{'name': nm, 'c_type': ct} for nm, ct, _canon, _is_enum in params]
            for name, params in resolved.items()}


def verify(yaml_path, header_paths, extra_args, out_sidecar=None):
    # Parser-level errors (bad type/dir, out/pack_dim3/strings/types
    # referencing an unknown arg, etc.) are hard errors. Surface them
    # with a clean ERROR: line rather than letting the traceback escape,
    # since this script is a CI gate and noisy tracebacks make failures
    # hard to read.
    try:
        raw_apis = parse_yaml_file(yaml_path)
    except (ParseError, BudgetError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    header_decls = parse_headers(header_paths, extra_args)
    errors = []
    warnings = []
    resolved_apis = []
    for api in raw_apis:
        name = api['api']
        if name not in header_decls:
            errors.append(f"{name}: not declared in any of {header_paths}")
            continue
        yaml_args = api['args']
        try:
            hdr_params = resolve_declaration(
                name, header_decls[name], [a['name'] for a in yaml_args])
        except AmbiguousDeclarationError as e:
            errors.append(str(e))
            continue
        # Field-budget mitigation is explicitly allowed to omit low-value
        # header params from the YAML. Match by NAME, not by count/position.
        hdr_by_name = {hname: (htype, canon, is_enum)
                       for hname, htype, canon, is_enum in hdr_params}
        yaml_names = [a['name'] for a in yaml_args]
        # Hard-error: YAML arg name not in header (typo or stale name).
        stale_name = False
        for i, yaml_arg in enumerate(yaml_args):
            if yaml_arg['name'] not in hdr_by_name:
                errors.append(
                    f"{name} arg {i}: YAML name {yaml_arg['name']!r} not in "
                    f"header params {list(hdr_by_name)}")
                stale_name = True
        if stale_name:
            continue

        is_explicit = bool(yaml_args) and 'type' in yaml_args[0]
        if is_explicit:
            # Explicit per-arg shape (test fixtures only): cross-check the
            # human-declared type against reality exactly as before Phase 3.
            arg_errors = False
            for yaml_arg in yaml_args:
                htype, canon, is_enum = hdr_by_name[yaml_arg['name']]
                # C bool MUST be DSL type bool, not uint32.
                if (htype.strip() in ('bool', '_Bool') or canon.strip() in ('bool', '_Bool')) \
                        and yaml_arg['type'] != 'bool':
                    errors.append(
                        f"{name} arg {yaml_arg['name']}: C bool requires DSL "
                        f"type 'bool', got {yaml_arg['type']!r}")
                    arg_errors = True
                    continue
                ok = _is_compatible(htype, yaml_arg['type'],
                                    canonical_type=canon, is_enum=is_enum)
                # OUT params: the header carries pointer-to-T (e.g.
                # `unsigned int *`), the DSL type describes T (e.g.
                # `uint32`). Retry with one `*` stripped for OUT.
                if not ok and yaml_arg['dir'] == 'OUT':
                    ok = _is_compatible(_strip_one_pointer(htype),
                                        yaml_arg['type'],
                                        canonical_type=_strip_one_pointer(canon),
                                        is_enum=is_enum)
                if not ok:
                    errors.append(
                        f"{name} arg {yaml_arg['name']}: type mismatch — C "
                        f"{htype!r} (canonical {canon!r}) not compatible "
                        f"with DSL {yaml_arg['type']!r}")
                    arg_errors = True
            if arg_errors:
                continue
            resolved = api
        else:
            # Compact schema: cross-check any explicit `types:` override
            # against reality (catches a genuinely wrong override) using
            # the same permissive compatibility table as the explicit
            # path above; everything else is inference via
            # expand_compact_apis(), which either succeeds (tautologically
            # compatible with what it was inferred from) or raises.
            override_errors = False
            for arg_name, ty in api.get('types_override', {}).items():
                htype, canon, is_enum = hdr_by_name[arg_name]
                dir_ = next(a['dir'] for a in yaml_args if a['name'] == arg_name)
                ok = _is_compatible(htype, ty, canonical_type=canon, is_enum=is_enum)
                if not ok and dir_ == 'OUT':
                    ok = _is_compatible(_strip_one_pointer(htype), ty,
                                        canonical_type=_strip_one_pointer(canon),
                                        is_enum=is_enum)
                if not ok:
                    errors.append(
                        f"{name} arg {arg_name}: 'types:' override {ty!r} "
                        f"not compatible with C {htype!r} (canonical {canon!r})")
                    override_errors = True
            if override_errors:
                continue
            try:
                resolved = expand_compact_apis([api], header_decls)[0]
            except (AmbiguousInferenceError, ParseError, BudgetError) as e:
                errors.append(f"{name}: {e}")
                continue

        # Informational: header params not in YAML (intentional partial
        # coverage — field-budget mitigation is by design).
        for hname, _, _, _ in hdr_params:
            if hname not in yaml_names:
                warnings.append(
                    f"{name}: header param {hname!r} not in YAML "
                    f"(intentional omission per field-budget mitigation?)")
        # Field-budget re-check (also enforced above via
        # expand_compact_apis()/validate_api(), but the verifier is the CI
        # gate so report it again defensively).
        if expanded_field_count(resolved) > PAYLOAD_BUDGET:
            errors.append(
                f"{name}: payload exceeds budget of {PAYLOAD_BUDGET} fields")
        resolved_apis.append(resolved)
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
    # libclang at compile time.
    if out_sidecar:
        # Sidecar uses the typedef spelling (c_type) as that is what
        # codegen needs to emit provider-correct OUT-handle helpers
        # (e.g. `hipStream_t*`, not the typedef-stripped form). Built
        # via compute_sidecar() so duplicate-declaration resolution
        # (resolve_declaration()) is applied here too — every api in
        # this loop already resolved cleanly above (an ambiguous one
        # would have added to `errors` and returned before this point).
        # compute_sidecar() only needs arg NAMES (raw_apis, not
        # resolved_apis, is sufficient and avoids re-plumbing 'type').
        sidecar = compute_sidecar(raw_apis, header_decls)
        os.makedirs(os.path.dirname(out_sidecar) or '.', exist_ok=True)
        with open(out_sidecar, 'w') as f:
            json.dump(sidecar, f, indent=2, sort_keys=True)
        print(f"wrote signature sidecar: {out_sidecar} "
              f"({len(sidecar)} APIs)", file=sys.stderr)
    print(f"OK: {len(raw_apis)} curated APIs verified against "
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
