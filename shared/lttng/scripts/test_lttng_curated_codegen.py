"""Tests for lttng_curated_codegen.py — the curated-args header generator.

Three kinds of coverage:
  1. Golden-file regression: the generator, run against the REAL checked-in
     curated_apis.yaml plus a LIVE libclang parse of the real HIP/HSA
     headers (--header, not a cached sidecar JSON), must reproduce the
     REAL checked-in generated headers exactly (--check mode, rc=0). This
     is the regression guard against future accidental hand edits to the
     generated headers (or drift between the YAML/real headers and the
     headers they're supposed to produce). The curated_apis.yaml + checked
     -in generated headers this depends on don't exist yet at every stage
     of this feature's patch series, so tests that need them are
     unittest.skipUnless-guarded on their presence and skip cleanly (not
     fail) until the patches that add them land. The real HIP/HSA base
     headers (hip_runtime_api.h, hsa.h, ...) are ordinary pre-existing
     repo headers, not part of that gating.
  2. Type-mapping unit tests: small synthetic YAML+sigs fixtures exercise
     each DSL type (including bool, dim3, dim3_packed, cstring, the HSA
     `.handle` OUT-deref pattern, and the HIP double-pointer OUT-deref
     pattern) in isolation and assert on the exact generated substrings.
     These use --sigs (a small hand-written JSON dict), not --header —
     --sigs remains supported specifically for fixtures like these that
     shouldn't need to depend on libclang.
  3. Live-header duplicate-declaration resolution: a synthetic fixture
     header mirroring HIP's real hipMallocAsync shape (a plain extern
     declaration plus a later `static inline` overload with an extra
     parameter) exercises resolve_declaration()'s preference logic and
     its loud-failure path for genuinely ambiguous cases.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

CODEGEN = os.path.join(HERE, 'lttng_curated_codegen.py')
COVERAGE_CHECK = os.path.join(HERE, 'lttng_coverage_check.py')

HIP_YAML = os.path.join(REPO_ROOT, 'projects/clr/hipamd/scripts/curated_apis.yaml')
HIP_TP_OUT = os.path.join(REPO_ROOT, 'projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h')
HIP_EMIT_OUT = os.path.join(REPO_ROOT, 'projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h')
HIP_EMIT_CPP_OUT = os.path.join(REPO_ROOT, 'projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.cpp')
# Same real header + flags used to (re)generate the checked-in HIP headers
# (see the regen command embedded in their banners).
HIP_HEADERS = [
    os.path.join(REPO_ROOT, 'projects/hip/include/hip/hip_runtime_api.h'),
    os.path.join(REPO_ROOT, 'projects/hip/include/hip/hip_ext.h'),
]
HIP_SOURCES = [
    os.path.join(REPO_ROOT, 'projects/clr/hipamd/src/hip_table_interface.cpp'),
    os.path.join(REPO_ROOT, 'projects/clr/hipamd/src/hip_table_interface_c.cpp'),
    os.path.join(REPO_ROOT, 'projects/clr/hipamd/src/hip_error.cpp'),
    os.path.join(REPO_ROOT, 'projects/clr/hipamd/src/profiler/hip_clr_profiler.cpp'),
]
HIP_EXTRA_ARGS = ['-D__HIP_PLATFORM_AMD__=1',
                   f'-I{os.path.join(REPO_ROOT, "projects/hip/include")}',
                   f'-I{os.path.join(REPO_ROOT, "projects/clr/hipamd/src")}',
                   f'-I{os.path.join(REPO_ROOT, "projects/clr/hipamd/include")}',
                   f'-I{os.path.join(REPO_ROOT, "projects/clr/rocclr")}',
                   f'-I{os.path.join(REPO_ROOT, "projects/rocr-runtime/runtime/hsa-runtime/inc")}']

HSA_YAML = os.path.join(
    REPO_ROOT, 'projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml')
HSA_TP_OUT = os.path.join(
    REPO_ROOT, 'projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_curated_tp.h')
HSA_EMIT_OUT = os.path.join(
    REPO_ROOT, 'projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit_curated.h')
HSA_EMIT_CPP_OUT = os.path.join(
    REPO_ROOT, 'projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit_curated.cpp')
_HSA_INC = os.path.join(REPO_ROOT, 'projects/rocr-runtime/runtime/hsa-runtime/inc')
_HSA_RUNTIME = os.path.dirname(_HSA_INC)
HSA_HEADERS = [os.path.join(_HSA_INC, h)
               for h in ('hsa.h', 'hsa_ext_amd.h', 'hsa_api_trace.h')]
HSA_HEADERS.append(os.path.join(_HSA_RUNTIME, 'core/inc/hsa_table_interface.h'))
HSA_EXTRA_ARGS = [f'-I{_HSA_INC}', f'-I{_HSA_RUNTIME}']

# The real HIP/HSA base headers (hip_runtime_api.h, hsa.h, ...) are ordinary
# checked-in repo headers unrelated to this feature and are present from the
# very first patch. curated_apis.yaml and the generated curated headers,
# however, are only added in the "HIP/HSA curated typed-arg capture" patches
# later in this series. Tests below that need those curated artifacts are
# skip-guarded on their presence so this file's content and assertions never
# need to change once those patches land -- only the skip condition flips.
HIP_CURATED_ARTIFACTS_EXIST = (os.path.exists(HIP_YAML)
                               and os.path.exists(HIP_TP_OUT)
                               and os.path.exists(HIP_EMIT_OUT)
                               and os.path.exists(HIP_EMIT_CPP_OUT))
HSA_CURATED_ARTIFACTS_EXIST = (os.path.exists(HSA_YAML)
                               and os.path.exists(HSA_TP_OUT)
                               and os.path.exists(HSA_EMIT_OUT)
                               and os.path.exists(HSA_EMIT_CPP_OUT))
_HIP_CURATED_SKIP_REASON = ('real HIP curated artifacts (curated_apis.yaml, '
                            'rocm_hip_curated_tp.h, rocm_trace_emit_curated.h) '
                            'not present yet in this patch series stage')
_HSA_CURATED_SKIP_REASON = ('real HSA curated artifacts (curated_apis.yaml, '
                            'rocm_hsa_curated_tp.h, rocm_trace_emit_curated.h) '
                            'not present yet in this patch series stage')

# Synthetic YAML+sigs fixture for the type-mapping unit tests below.
MINIMAL_SIGS = os.path.join(HERE, 'testdata', 'curated_minimal_sigs.json')

# Fixture header mirroring HIP's real hipMallocAsync duplicate-declaration
# shape, for the live-header resolution tests.
FAKE_OVERLOADS_HEADER = os.path.join(HERE, 'testdata', 'fake_hip_overloads.h')


def _run(args):
    return subprocess.run(['python3', CODEGEN] + args, capture_output=True, text=True)


def _generate(provider, yaml_path, tp_out, emit_out, sigs_path=None,
              header_paths=None, source_paths=None, extra_args=None,
              emit_cpp_out=None):
    """Run the generator and return (tp_text, emit_text, emit_cpp_text).

    `emit_cpp_out` defaults to a sibling <emit_out>.cpp so tests always
    exercise the header+cpp split; callers that only care about the header
    or tp.h can ignore the third element."""
    if emit_cpp_out is None:
        emit_cpp_out = os.path.splitext(emit_out)[0] + '.cpp'
    args = ['--provider', provider, '--yaml', yaml_path,
            '--tp-out', tp_out, '--emit-out', emit_out,
            '--emit-cpp-out', emit_cpp_out]
    if sigs_path is not None:
        args += ['--sigs', sigs_path]
    else:
        for h in header_paths:
            args += ['--header', h]
        for s in (source_paths or []):
            args += ['--source', s]
        for e in (extra_args or []):
            args.append(f'--extra-arg={e}')
    r = _run(args)
    assert r.returncode == 0, f"generation failed: {r.stderr}"
    with open(tp_out) as f:
        tp_text = f.read()
    with open(emit_out) as f:
        emit_text = f.read()
    with open(emit_cpp_out) as f:
        emit_cpp_text = f.read()
    return tp_text, emit_text, emit_cpp_text


# ---------------------------------------------------------------------------
# 1. Golden-file regression against the real checked-in HIP/HSA artifacts,
#    with signatures resolved via a live libclang parse of the real
#    headers (no checked-in sidecar JSON).
# ---------------------------------------------------------------------------
@unittest.skipUnless(HIP_CURATED_ARTIFACTS_EXIST, _HIP_CURATED_SKIP_REASON)
def test_hip_check_mode_matches_checked_in_headers():
    """The generator, run against the real HIP YAML + a live parse of the
    real HIP header, must reproduce the real checked-in
    rocm_hip_curated_tp.h and rocm_trace_emit_curated.h exactly (--check
    mode exits 0)."""
    r = _run(['--provider', 'hip', '--check',
               '--yaml', HIP_YAML,
               *sum((['--header', h] for h in HIP_HEADERS), []),
               *sum((['--source', s] for s in HIP_SOURCES), []),
              *[f'--extra-arg={a}' for a in HIP_EXTRA_ARGS],
              '--tp-out', HIP_TP_OUT, '--emit-out', HIP_EMIT_OUT,
              '--emit-cpp-out', HIP_EMIT_CPP_OUT])
    assert r.returncode == 0, f"HIP headers drifted from generator output:\n{r.stderr}"


@unittest.skipUnless(HSA_CURATED_ARTIFACTS_EXIST, _HSA_CURATED_SKIP_REASON)
def test_hsa_check_mode_matches_checked_in_headers():
    """Same as above for HSA."""
    args = ['--provider', 'hsa', '--check', '--yaml', HSA_YAML]
    for h in HSA_HEADERS:
        args += ['--header', h]
    args += [f'--extra-arg={a}' for a in HSA_EXTRA_ARGS]
    args += ['--tp-out', HSA_TP_OUT, '--emit-out', HSA_EMIT_OUT,
             '--emit-cpp-out', HSA_EMIT_CPP_OUT]
    r = _run(args)
    assert r.returncode == 0, f"HSA headers drifted from generator output:\n{r.stderr}"


@unittest.skipUnless(HIP_CURATED_ARTIFACTS_EXIST and HSA_CURATED_ARTIFACTS_EXIST,
                     'real HIP/HSA curated artifacts not present yet in this patch series stage')
def test_live_curated_wrapper_return_kinds_match_signatures():
    """Every curated wrapper macro must match the live libclang result type.

    The historic generic STATUS kind deliberately included ordinary 32-bit
    scalar returns. This full-inventory gate prevents the curated migration
    from treating HIP's int-returning hipGetStreamDeviceId or HSA's uint32_t
    signal waits as their provider status enums.
    """
    cases = [
        ('hip', HIP_YAML, os.path.join(REPO_ROOT, 'projects/clr/hipamd/src'),
         HIP_HEADERS, HIP_SOURCES, HIP_EXTRA_ARGS, 554),
        ('hsa', HSA_YAML,
         os.path.join(REPO_ROOT, 'projects/rocr-runtime/runtime/hsa-runtime'),
         HSA_HEADERS, [], HSA_EXTRA_ARGS, 217),
    ]
    for provider, yaml_path, src_dir, headers, sources, extra_args, count in cases:
        command = ['python3', COVERAGE_CHECK, 'return-kinds',
                   '--provider', provider, '--src-dir', src_dir, '--yaml', yaml_path]
        for header in headers:
            command += ['--header', header]
        for source in sources:
            command += ['--source', source]
        command += [f'--extra-arg={arg}' for arg in extra_args]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"{provider} curated return-kind audit failed:\n{result.stdout}{result.stderr}")
        assert f'RETURN-KINDS: {count} curated APIs verified' in result.stdout


@unittest.skipUnless(HIP_CURATED_ARTIFACTS_EXIST, _HIP_CURATED_SKIP_REASON)
def test_check_mode_detects_injected_drift():
    """--check must fail (rc=1) when the on-disk header doesn't match what
    the generator would produce — the whole point of the regression guard."""
    with tempfile.TemporaryDirectory() as d:
        tp_out = os.path.join(d, 'rocm_hip_curated_tp.h')
        emit_out = os.path.join(d, 'rocm_trace_emit_curated.h')
        emit_cpp_out = os.path.join(d, 'rocm_trace_emit_curated.cpp')
        _generate('hip', HIP_YAML, tp_out, emit_out,
                   header_paths=HIP_HEADERS, source_paths=HIP_SOURCES,
                   extra_args=HIP_EXTRA_ARGS, emit_cpp_out=emit_cpp_out)
        # Hand-corrupt the tp.h the generator just wrote.
        with open(tp_out, 'a') as f:
            f.write("\n/* hand edit that must be detected as drift */\n")
        r = _run(['--provider', 'hip', '--check',
                   '--yaml', HIP_YAML,
                   *sum((['--header', h] for h in HIP_HEADERS), []),
                   *sum((['--source', s] for s in HIP_SOURCES), []),
                  *[f'--extra-arg={a}' for a in HIP_EXTRA_ARGS],
                  '--tp-out', tp_out, '--emit-out', emit_out,
                  '--emit-cpp-out', emit_cpp_out])
        assert r.returncode == 1
        assert 'DRIFT' in r.stderr


# ---------------------------------------------------------------------------
# 1b. CLI validation + live-header duplicate-declaration resolution.
# ---------------------------------------------------------------------------
def test_sigs_and_header_are_mutually_exclusive():
    r = _run(['--provider', 'hip', '--yaml', HIP_YAML,
               '--sigs', MINIMAL_SIGS, '--header', HIP_HEADERS[0],
              '--tp-out', '/tmp/unused_tp.h', '--emit-out', '/tmp/unused_emit.h'])
    assert r.returncode != 0
    assert 'mutually exclusive' in (r.stdout + r.stderr).lower()


def test_sigs_or_header_is_required():
    r = _run(['--provider', 'hip', '--yaml', HIP_YAML,
              '--tp-out', '/tmp/unused_tp.h', '--emit-out', '/tmp/unused_emit.h'])
    assert r.returncode != 0
    assert 'required' in (r.stdout + r.stderr).lower()


def test_dump_resolved_writes_sidecar_json():
    """--dump-resolved writes the {api: [{name, c_type}, ...]} sidecar
    JSON from a live header parse and exits without needing --tp-out/
    --emit-out — the mechanism external consumers (e.g. the HIP curated
    coverage test harness) use instead of a checked-in JSON cache."""
    yaml_text = """\
- api: hipMemcpyAsync
  category: memory
  args:
    - {name: dst,       type: ptr,    dir: IN}
    - {name: src,       type: ptr,    dir: IN}
    - {name: sizeBytes, type: size,   dir: IN}
    - {name: kind,      type: enum,   dir: IN}
    - {name: stream,    type: handle, dir: IN}
"""
    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'y.yaml')
        _write(yaml_path, yaml_text)
        out_json = os.path.join(d, 'resolved.json')
        r = _run(['--provider', 'hip', '--dump-resolved', out_json,
                   '--yaml', yaml_path, '--header', HIP_HEADERS[0],
                  *[f'--extra-arg={a}' for a in HIP_EXTRA_ARGS]])
        assert r.returncode == 0, f"--dump-resolved failed:\n{r.stderr}"
        with open(out_json) as f:
            data = json.load(f)
    names = [e['name'] for e in data['hipMemcpyAsync']]
    assert names == ['dst', 'src', 'sizeBytes', 'kind', 'stream']


def test_dump_resolved_requires_header_not_sigs():
    r = _run(['--provider', 'hip', '--dump-resolved', '/tmp/unused.json',
              '--yaml', HIP_YAML, '--sigs', MINIMAL_SIGS])
    assert r.returncode != 0
    assert 'requires --header' in (r.stdout + r.stderr)


def test_live_header_prefers_extern_over_static_inline_overload():
    """Regression test for the real hipMallocAsync bug: a header that
    declares both a plain extern function and a later `static inline`
    overload with an extra parameter (fakeMallocAsync mirrors
    hipMallocAsync's dev_ptr/size/[mem_pool]/stream shape) must resolve
    to the extern declaration when the YAML's arg names are satisfied by
    it — NOT silently pick whichever declaration libclang walked last."""
    yaml_text = """\
- api: fakeMallocAsync
  category: memory
  args:
    - {name: dev_ptr, type: ptr,    dir: OUT}
    - {name: size,    type: size,   dir: IN}
    - {name: stream,  type: handle, dir: IN}
"""
    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'y.yaml')
        _write(yaml_path, yaml_text)
        out_json = os.path.join(d, 'resolved.json')
        r = _run(['--provider', 'hip', '--dump-resolved', out_json,
                  '--yaml', yaml_path, '--header', FAKE_OVERLOADS_HEADER])
        assert r.returncode == 0, f"--dump-resolved failed:\n{r.stderr}"
        with open(out_json) as f:
            data = json.load(f)
    names = [e['name'] for e in data['fakeMallocAsync']]
    assert names == ['dev_ptr', 'size', 'stream'], \
        f"expected the 3-arg extern signature (no mem_pool), got {names}"


def test_live_header_ambiguous_declaration_fails_loudly():
    """Two non-static overloads that both satisfy the YAML's arg names
    can't be disambiguated automatically — must fail loudly (naming the
    API and listing candidates), not silently pick one."""
    yaml_text = """\
- api: fakeAmbiguousApi
  category: memory
  args:
    - {name: a, type: ptr,    dir: IN}
    - {name: b, type: size,   dir: IN}
    - {name: c, type: handle, dir: IN}
"""
    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'y.yaml')
        _write(yaml_path, yaml_text)
        out_json = os.path.join(d, 'resolved.json')
        r = _run(['--provider', 'hip', '--dump-resolved', out_json,
                  '--yaml', yaml_path, '--header', FAKE_OVERLOADS_HEADER])
    assert r.returncode != 0, "expected failure on genuinely ambiguous declaration"
    combined = r.stdout + r.stderr
    assert 'fakeAmbiguousApi' in combined
    assert 'ambiguous' in combined.lower()


# ---------------------------------------------------------------------------
# 2. Type-mapping unit tests against small synthetic fixtures.
# ---------------------------------------------------------------------------
def _write(path, text):
    with open(path, 'w') as f:
        f.write(text)


def test_curated_minimal_fixture_generates():
    """The pre-existing testdata/curated_minimal.yaml fixture (ptr/size/
    enum/handle DSL types, an OUT ptr, and a zero-arg API) generates
    syntactically-plausible schema-v1 combined-event output: each event
    leads with `phase` and carries the per-API return field; each API gets
    an enter/exit helper pair and no shared lifecycle events."""
    yaml_path = os.path.join(HERE, 'testdata', 'curated_minimal.yaml')
    sigs_path = os.path.join(HERE, 'testdata', 'curated_minimal_sigs.json')
    with tempfile.TemporaryDirectory() as d:
        tp_text, emit_text, cpp_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)

    # hipMemcpyAsync: phase first, ptr/ptr/size/enum/handle (all IN), then
    # the STATUS return field retstatus.
    assert ('LTTNG_UST_TP_ARGS(int32_t, phase, uint64_t, dst, uint64_t, src, '
            'uint64_t, sizeBytes, int32_t, kind, uint64_t, stream, '
            'int32_t, retstatus)') in tp_text
    assert 'lttng_ust_field_integer(int32_t, phase, phase)' in tp_text
    assert 'lttng_ust_field_integer_hex(uint64_t, dst, dst)' in tp_text
    assert 'lttng_ust_field_integer(uint64_t, sizeBytes, sizeBytes)' in tp_text
    assert 'lttng_ust_field_integer(int32_t, kind, kind)' in tp_text
    assert 'lttng_ust_field_integer(int32_t, retstatus, retstatus)' in tp_text

    # hipMalloc: OUT ptr + IN size — sidecar says `void **` for ptr. The
    # combined event is phase + ptr + size + retstatus.
    assert 'LTTNG_UST_TP_ARGS(int32_t, phase, uint64_t, ptr, uint64_t, size, int32_t, retstatus)' \
        in tp_text
    # The header declares the helpers (no bodies); the OUT-ptr deref body
    # lives in the .cpp. enter takes only the IN size; exit takes the OUT
    # ptr pointer + status.
    assert 'void rocm_trace_emit_hipMalloc_enter(size_t);' in emit_text
    assert 'void rocm_trace_emit_hipMalloc_exit(void**, hipError_t);' in emit_text
    assert 'void** ptr_out_ptr' in cpp_text
    assert '(uint64_t)(uintptr_t)(*ptr_out_ptr)' in cpp_text

    # hipDeviceSynchronize: zero-arg (phase + retstatus only) event; the
    # enter body fires with phase=ENTER and a zero return field (in .cpp).
    assert 'LTTNG_UST_TP_ARGS(int32_t, phase, int32_t, retstatus)' in tp_text
    assert 'lttng_ust_do_tracepoint(rocm_hip, hipDeviceSynchronize,' in cpp_text

    # Per-API enter/exit helper DECLARATIONS in the header (non-static,
    # non-inline, no body); no shared lifecycle helpers/events anywhere.
    assert 'void rocm_trace_emit_hipDeviceSynchronize_enter(void);' in emit_text
    assert 'void rocm_trace_emit_hipDeviceSynchronize_exit(' in emit_text
    assert 'static' not in emit_text
    assert 'inline' not in emit_text
    assert '_args' not in emit_text
    assert '_args' not in cpp_text
    assert 'hip_api_enter' not in tp_text
    assert 'hip_api_exit_status' not in tp_text
    assert 'rocm_trace_emit_hip_api_enter' not in cpp_text
    assert 'rocm_trace_emit_hip_api_exit' not in cpp_text

    # No-op stub section (in the .cpp, non-inline): enter takes the IN size,
    # exit the OUT ptr + status.
    assert 'void rocm_trace_emit_hipMalloc_enter(size_t) {}' in cpp_text
    assert 'void rocm_trace_emit_hipMalloc_exit(void**, hipError_t) {}' in cpp_text


def test_high_arity_api_generates_ordered_event_chunks():
    """Preserve every representable arg while keeping each UST event at ten fields."""
    args = '\n'.join(
        f'    - {{name: a{i}, type: uint32, dir: IN}}' for i in range(11))
    yaml_text = f"""\
- api: fakeHighArityApi
  args:
{args}
"""
    sigs = {'fakeHighArityApi': [
        {'name': f'a{i}', 'c_type': 'uint32_t'} for i in range(11)]}
    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'y.yaml')
        sigs_path = os.path.join(d, 's.json')
        _write(yaml_path, yaml_text)
        _write(sigs_path, json.dumps(sigs))
        tp_text, emit_text, cpp_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    # Schema v1: event name drops the _args suffix; phase leads each chunk,
    # the STATUS retstatus field lands on the last chunk. With phase+retstatus
    # counting against the 10-field budget, 11 uint32 args split as
    # [phase + a0..a8] (10 fields) then [phase + a9 + a10 + retstatus].
    assert 'rocm_hip, fakeHighArityApi,' in tp_text
    assert 'rocm_hip, fakeHighArityApi_2,' in tp_text
    assert 'uint32_t, a8)' in tp_text
    assert ('LTTNG_UST_TP_ARGS(int32_t, phase, uint32_t, a9, uint32_t, a10, '
            'int32_t, retstatus)') in tp_text
    assert 'lttng_ust_do_tracepoint(rocm_hip, fakeHighArityApi,' in cpp_text
    assert 'lttng_ust_do_tracepoint(rocm_hip, fakeHighArityApi_2,' in cpp_text


def test_bool_maps_to_uint32_not_uint64():
    """Regression test for the historical bug (fixed by hand in commit
    7c99b427fb) where bool's TP_ARGS/field type was computed from the
    wrong lookup table and silently widened to uint64_t. The DSL's bool
    type must produce uint32_t in both the tracepoint event and the cast
    expression, everywhere, with no special-casing needed."""
    yaml_text = """\
- api: fakeBoolApi
  category: memory
  args:
    - {name: flag, type: bool, dir: IN}
"""
    sigs = {'fakeBoolApi': [{'name': 'flag', 'c_type': 'bool'}]}
    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'y.yaml')
        sigs_path = os.path.join(d, 's.json')
        _write(yaml_path, yaml_text)
        _write(sigs_path, json.dumps(sigs))
        tp_text, emit_text, cpp_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert 'LTTNG_UST_TP_ARGS(int32_t, phase, uint32_t, flag, int32_t, retstatus)' in tp_text
    assert 'lttng_ust_field_integer(uint32_t, flag, flag)' in tp_text
    assert '(uint32_t)(!!(flag))' in cpp_text


def test_dim3_expands_to_three_fields():
    """dim3 (unpacked) expands to <name>_x/_y/_z uint32_t fields per
    lttng_curated_lib.TYPE_EXPANSION — exercised here since the current
    real YAML only uses dim3_packed, never plain dim3."""
    yaml_text = """\
- api: fakeDim3Api
  category: kernel_launch
  args:
    - {name: blockDim, type: dim3, dir: IN}
"""
    sigs = {'fakeDim3Api': [{'name': 'blockDim', 'c_type': 'dim3'}]}
    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'y.yaml')
        sigs_path = os.path.join(d, 's.json')
        _write(yaml_path, yaml_text)
        _write(sigs_path, json.dumps(sigs))
        tp_text, emit_text, cpp_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert ('LTTNG_UST_TP_ARGS(int32_t, phase, uint32_t, blockDim_x, uint32_t, blockDim_y, '
            'uint32_t, blockDim_z, int32_t, retstatus)') in tp_text
    assert 'lttng_ust_field_integer(uint32_t, blockDim_x, blockDim_x)' in tp_text
    assert 'lttng_ust_field_integer(uint32_t, blockDim_y, blockDim_y)' in tp_text
    assert 'lttng_ust_field_integer(uint32_t, blockDim_z, blockDim_z)' in tp_text
    assert '(uint32_t)blockDim.x' in cpp_text
    assert '(uint32_t)blockDim.y' in cpp_text
    assert '(uint32_t)blockDim.z' in cpp_text


def test_cstring_uses_field_string_macro():
    """cstring DSL type uses lttng_ust_field_string (2-arg form, no type
    token) and the null-safe ternary cast at the call site."""
    yaml_text = """\
- api: fakeCstringApi
  category: module
  args:
    - {name: name, type: cstring, dir: IN}
"""
    sigs = {'fakeCstringApi': [{'name': 'name', 'c_type': 'const char *'}]}
    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'y.yaml')
        sigs_path = os.path.join(d, 's.json')
        _write(yaml_path, yaml_text)
        _write(sigs_path, json.dumps(sigs))
        tp_text, emit_text, cpp_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert 'lttng_ust_field_string(name, name)' in tp_text
    assert '(name ? name : "")' in cpp_text


def test_hsa_out_handle_uses_struct_field_deref():
    """HSA OUT handle types that are pointer-to-struct (e.g. hsa_signal_t*)
    must deref via `->handle`, NOT `*p` — the sidecar-driven C10 fix. This
    is the key behavioral difference from HIP's OUT-handle pattern."""
    yaml_text = """\
- api: fakeHsaSignalCreate
  category: hsa_signals
  args:
    - {name: signal, type: handle, dir: OUT}
"""
    sigs = {'fakeHsaSignalCreate': [{'name': 'signal', 'c_type': 'hsa_signal_t *'}]}
    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'y.yaml')
        sigs_path = os.path.join(d, 's.json')
        _write(yaml_path, yaml_text)
        _write(sigs_path, json.dumps(sigs))
        tp_text, emit_text, cpp_text = _generate(
            'hsa', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert 'hsa_signal_t * signal_out_ptr' in cpp_text
    assert '(signal_out_ptr->handle)' in cpp_text
    assert '*signal_out_ptr' not in cpp_text


def test_hsa_out_double_pointer_derefs_one_level():
    """HSA OUT types that are already pointer-to-pointer (e.g.
    hsa_queue_t**, where the pointee IS the handle) deref one level via
    `*p`, not `p->handle`."""
    yaml_text = """\
- api: fakeHsaQueueCreate
  category: hsa_queues
  args:
    - {name: queue, type: handle, dir: OUT}
"""
    sigs = {'fakeHsaQueueCreate': [{'name': 'queue', 'c_type': 'hsa_queue_t **'}]}
    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'y.yaml')
        sigs_path = os.path.join(d, 's.json')
        _write(yaml_path, yaml_text)
        _write(sigs_path, json.dumps(sigs))
        tp_text, emit_text, cpp_text = _generate(
            'hsa', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert 'hsa_queue_t ** queue_out_ptr' in cpp_text
    assert '(uint64_t)(uintptr_t)(*queue_out_ptr)' in cpp_text
    assert '->handle' not in cpp_text


def test_hsa_combined_event_no_shared_lifecycle():
    """Schema v1: HSA emits ONE combined `<api>` event (phase + args +
    retstatus) plus a per-API enter/exit helper pair, and NO shared
    hsa_api_enter/hsa_api_exit_* lifecycle events or helpers."""
    yaml_text = """\
- api: fakeHsaLifecycle
  args:
    - {name: value, type: uint32, dir: IN}
"""
    sigs = {'fakeHsaLifecycle': [{'name': 'value', 'c_type': 'uint32_t'}]}
    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'y.yaml')
        sigs_path = os.path.join(d, 's.json')
        _write(yaml_path, yaml_text)
        _write(sigs_path, json.dumps(sigs))
        tp_text, emit_text, cpp_text = _generate(
            'hsa', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    # Combined event with phase + arg + retstatus (defaults to STATUS kind).
    assert 'rocm_hsa, fakeHsaLifecycle,' in tp_text
    assert ('LTTNG_UST_TP_ARGS(int32_t, phase, uint32_t, value, int32_t, retstatus)') in tp_text
    # No shared lifecycle events or helpers.
    assert 'hsa_api_enter' not in tp_text
    assert 'hsa_api_exit_status' not in tp_text
    assert 'hsa_api_exit_u64' not in tp_text
    assert 'rocm_trace_emit_hsa_api_enter' not in emit_text
    assert 'rocm_trace_emit_hsa_api_exit' not in emit_text
    assert 'rocm_trace_emit_hsa_api_enter' not in cpp_text
    assert 'rocm_trace_emit_hsa_api_exit' not in cpp_text
    assert '_args' not in tp_text
    assert '_args' not in emit_text
    assert '_args' not in cpp_text
    # The combined event's enter/exit helper pair IS declared and defined.
    assert 'rocm_trace_emit_fakeHsaLifecycle_enter(' in emit_text
    assert 'rocm_trace_emit_fakeHsaLifecycle_exit(' in emit_text
    assert 'rocm_trace_emit_fakeHsaLifecycle_enter(' in cpp_text
    assert 'rocm_trace_emit_fakeHsaLifecycle_exit(' in cpp_text


def test_hip_out_handle_derefs_typedef_pointer():
    """HIP OUT handle types are typedef'd pointers (hipStream_t == void*),
    so the deref is one level via `*p`, matching the sidecar's real
    single-pointer spelling (hipStream_t*)."""
    yaml_text = """\
- api: fakeHipStreamCreate
  category: streams
  args:
    - {name: stream, type: handle, dir: OUT}
"""
    sigs = {'fakeHipStreamCreate': [{'name': 'stream', 'c_type': 'hipStream_t*'}]}
    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'y.yaml')
        sigs_path = os.path.join(d, 's.json')
        _write(yaml_path, yaml_text)
        _write(sigs_path, json.dumps(sigs))
        tp_text, emit_text, cpp_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert 'hipStream_t* stream_out_ptr' in cpp_text
    assert '(uint64_t)(uintptr_t)(*stream_out_ptr)' in cpp_text


def test_all_in_api_enter_takes_in_exit_takes_status():
    """Schema v1: an all-IN STATUS API's enter helper takes the IN arg(s),
    its exit helper takes the status (which is both the retstatus field and
    the OUT success gate). No OUT params, no corr_id anywhere."""
    yaml_text = """\
- api: fakeAllInApi
  category: streams
  args:
    - {name: x, type: uint32, dir: IN}
"""
    sigs = {'fakeAllInApi': [{'name': 'x', 'c_type': 'uint32_t'}]}
    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'y.yaml')
        sigs_path = os.path.join(d, 's.json')
        _write(yaml_path, yaml_text)
        _write(sigs_path, json.dumps(sigs))
        tp_text, emit_text, cpp_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    # Header declares (unnamed) params; the named-param definition is in .cpp.
    assert 'void rocm_trace_emit_fakeAllInApi_enter(uint32_t);' in emit_text
    assert 'void rocm_trace_emit_fakeAllInApi_exit(hipError_t);' in emit_text
    assert 'rocm_trace_emit_fakeAllInApi_enter(uint32_t x)' in cpp_text
    assert 'rocm_trace_emit_fakeAllInApi_exit(hipError_t status)' in cpp_text
    assert 'corr_id' not in tp_text
    assert 'corr_id' not in emit_text
    assert 'corr_id' not in cpp_text


def test_v5_ptr_return_field_and_helpers():
    """A PTR-returning API's combined event carries a `retptr` uint64 hex
    field on its last chunk; its exit helper takes the returned pointer as a
    uint64_t (not the provider status), and its enter helper takes the IN
    args. Uses the sigs-mode `return_kind` override."""
    yaml_text = """\
- api: fakePtrApi
  args:
    - {name: id, type: uint32, dir: IN}
"""
    sigs = {'fakePtrApi': {'params': [{'name': 'id', 'c_type': 'uint32_t'}],
                           'return_kind': 'PTR'}}
    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'y.yaml')
        sigs_path = os.path.join(d, 's.json')
        _write(yaml_path, yaml_text)
        _write(sigs_path, json.dumps(sigs))
        tp_text, emit_text, cpp_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert 'lttng_ust_field_integer_hex(uint64_t, retptr, retptr)' in tp_text
    assert 'LTTNG_UST_TP_ARGS(int32_t, phase, uint32_t, id, uint64_t, retptr)' in tp_text
    assert 'void rocm_trace_emit_fakePtrApi_enter(uint32_t);' in emit_text
    assert 'void rocm_trace_emit_fakePtrApi_exit(uint64_t);' in emit_text
    assert 'rocm_trace_emit_fakePtrApi_enter(uint32_t id)' in cpp_text
    assert 'rocm_trace_emit_fakePtrApi_exit(uint64_t retptr)' in cpp_text


def test_v5_void_api_has_no_return_field_and_no_exit_param():
    """A VOID-returning API's combined event has no return field, and its
    exit helper takes no parameter at all (just phase=EXIT)."""
    yaml_text = """\
- api: fakeVoidApi
  args:
    - {name: id, type: uint32, dir: IN}
"""
    sigs = {'fakeVoidApi': {'params': [{'name': 'id', 'c_type': 'uint32_t'}],
                            'return_kind': 'VOID'}}
    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'y.yaml')
        sigs_path = os.path.join(d, 's.json')
        _write(yaml_path, yaml_text)
        _write(sigs_path, json.dumps(sigs))
        tp_text, emit_text, cpp_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert 'LTTNG_UST_TP_ARGS(int32_t, phase, uint32_t, id)' in tp_text
    assert 'retstatus' not in tp_text
    assert 'retval' not in tp_text
    assert 'retptr' not in tp_text
    assert 'void rocm_trace_emit_fakeVoidApi_enter(uint32_t);' in emit_text
    assert 'void rocm_trace_emit_fakeVoidApi_exit(void);' in emit_text
    assert 'rocm_trace_emit_fakeVoidApi_enter(uint32_t id)' in cpp_text
    assert 'rocm_trace_emit_fakeVoidApi_exit(void)' in cpp_text


def test_v5_enter_and_exit_use_phase_discriminator():
    """The enter helper fires phase=ENTER(0) with the IN arg populated and
    the return field 0; the exit helper fires phase=EXIT(1) with the IN arg
    field 0 and the return field populated from status."""
    yaml_text = """\
- api: fakePhaseApi
  args:
    - {name: x, type: uint32, dir: IN}
"""
    sigs = {'fakePhaseApi': [{'name': 'x', 'c_type': 'uint32_t'}]}
    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'y.yaml')
        sigs_path = os.path.join(d, 's.json')
        _write(yaml_path, yaml_text)
        _write(sigs_path, json.dumps(sigs))
        _tp, _emit, cpp_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    enter = cpp_text.split('rocm_trace_emit_fakePhaseApi_enter')[1].split('}')[0]
    exit_ = cpp_text.split('rocm_trace_emit_fakePhaseApi_exit')[1].split('}')[0]
    # ENTER: phase 0, IN populated, return 0.
    assert '(int32_t)0' in enter
    assert '(uint32_t)(x)' in enter
    # EXIT: phase 1, IN field written 0, return from status.
    assert '(int32_t)1' in exit_
    assert '(int32_t)status' in exit_


def test_banner_has_correct_sha256_and_regen_command():
    """The provenance banner must carry the ACTUAL sha256 of the yaml file
    passed in (not a stale/incorrect one) and an explicit regen command
    naming this script — the whole point of this phase's fix."""
    import hashlib
    yaml_path = os.path.join(HERE, 'testdata', 'curated_minimal.yaml')
    sigs_path = os.path.join(HERE, 'testdata', 'curated_minimal_sigs.json')
    with open(yaml_path, 'rb') as f:
        real_sha256 = hashlib.sha256(f.read()).hexdigest()
    with tempfile.TemporaryDirectory() as d:
        tp_text, emit_text, cpp_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert real_sha256 in tp_text
    assert real_sha256 in emit_text
    assert real_sha256 in cpp_text
    assert 'lttng_curated_codegen.py' in tp_text
    assert 'lttng_curated_codegen.py' in emit_text
    assert 'lttng_curated_codegen.py' in cpp_text
    assert '--provider hip' in tp_text
    assert '--emit-cpp-out' in cpp_text


if __name__ == '__main__':
    import inspect
    failures = 0
    skipped = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn) and not inspect.isclass(fn):
            try:
                fn()
                print(f'  ok  {name}')
            except unittest.SkipTest as e:
                print(f'  skip {name}: {e}')
                skipped += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'  FAIL {name}: {e}')
                failures += 1
    print(f'\n{"PASS" if failures == 0 else "FAIL"}: {failures} failures, {skipped} skipped')
    sys.exit(1 if failures else 0)
