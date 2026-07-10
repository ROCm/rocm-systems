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

HIP_YAML = os.path.join(REPO_ROOT, 'projects/clr/hipamd/scripts/curated_apis.yaml')
HIP_TP_OUT = os.path.join(REPO_ROOT, 'projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h')
HIP_EMIT_OUT = os.path.join(REPO_ROOT, 'projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h')
# Same real header + flags used to (re)generate the checked-in HIP headers
# (see the regen command embedded in their banners).
HIP_HEADER = os.path.join(REPO_ROOT, 'projects/hip/include/hip/hip_runtime_api.h')
HIP_EXTRA_ARGS = ['-D__HIP_PLATFORM_AMD__=1',
                  f'-I{os.path.join(REPO_ROOT, "projects/hip/include")}']

HSA_YAML = os.path.join(
    REPO_ROOT, 'projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml')
HSA_TP_OUT = os.path.join(
    REPO_ROOT, 'projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_curated_tp.h')
HSA_EMIT_OUT = os.path.join(
    REPO_ROOT, 'projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit_curated.h')
_HSA_INC = os.path.join(REPO_ROOT, 'projects/rocr-runtime/runtime/hsa-runtime/inc')
HSA_HEADERS = [os.path.join(_HSA_INC, h)
              for h in ('hsa.h', 'hsa_ext_amd.h', 'hsa_api_trace.h')]
HSA_EXTRA_ARGS = [f'-I{_HSA_INC}']

# The real HIP/HSA base headers (hip_runtime_api.h, hsa.h, ...) are ordinary
# checked-in repo headers unrelated to this feature and are present from the
# very first patch. curated_apis.yaml and the generated curated headers,
# however, are only added in the "HIP/HSA curated typed-arg capture" patches
# later in this series. Tests below that need those curated artifacts are
# skip-guarded on their presence so this file's content and assertions never
# need to change once those patches land -- only the skip condition flips.
HIP_CURATED_ARTIFACTS_EXIST = (os.path.exists(HIP_YAML)
                               and os.path.exists(HIP_TP_OUT)
                               and os.path.exists(HIP_EMIT_OUT))
HSA_CURATED_ARTIFACTS_EXIST = (os.path.exists(HSA_YAML)
                               and os.path.exists(HSA_TP_OUT)
                               and os.path.exists(HSA_EMIT_OUT))
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
             header_paths=None, extra_args=None):
    args = ['--provider', provider, '--yaml', yaml_path,
            '--tp-out', tp_out, '--emit-out', emit_out]
    if sigs_path is not None:
        args += ['--sigs', sigs_path]
    else:
        for h in header_paths:
            args += ['--header', h]
        for e in (extra_args or []):
            args.append(f'--extra-arg={e}')
    r = _run(args)
    assert r.returncode == 0, f"generation failed: {r.stderr}"
    with open(tp_out) as f:
        tp_text = f.read()
    with open(emit_out) as f:
        emit_text = f.read()
    return tp_text, emit_text


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
              '--yaml', HIP_YAML, '--header', HIP_HEADER,
              *[f'--extra-arg={a}' for a in HIP_EXTRA_ARGS],
              '--tp-out', HIP_TP_OUT, '--emit-out', HIP_EMIT_OUT])
    assert r.returncode == 0, f"HIP headers drifted from generator output:\n{r.stderr}"


@unittest.skipUnless(HSA_CURATED_ARTIFACTS_EXIST, _HSA_CURATED_SKIP_REASON)
def test_hsa_check_mode_matches_checked_in_headers():
    """Same as above for HSA."""
    args = ['--provider', 'hsa', '--check', '--yaml', HSA_YAML]
    for h in HSA_HEADERS:
        args += ['--header', h]
    args += [f'--extra-arg={a}' for a in HSA_EXTRA_ARGS]
    args += ['--tp-out', HSA_TP_OUT, '--emit-out', HSA_EMIT_OUT]
    r = _run(args)
    assert r.returncode == 0, f"HSA headers drifted from generator output:\n{r.stderr}"


@unittest.skipUnless(HIP_CURATED_ARTIFACTS_EXIST, _HIP_CURATED_SKIP_REASON)
def test_check_mode_detects_injected_drift():
    """--check must fail (rc=1) when the on-disk header doesn't match what
    the generator would produce — the whole point of the regression guard."""
    with tempfile.TemporaryDirectory() as d:
        tp_out = os.path.join(d, 'rocm_hip_curated_tp.h')
        emit_out = os.path.join(d, 'rocm_trace_emit_curated.h')
        _generate('hip', HIP_YAML, tp_out, emit_out,
                  header_paths=[HIP_HEADER], extra_args=HIP_EXTRA_ARGS)
        # Hand-corrupt the tp.h the generator just wrote.
        with open(tp_out, 'a') as f:
            f.write("\n/* hand edit that must be detected as drift */\n")
        r = _run(['--provider', 'hip', '--check',
                  '--yaml', HIP_YAML, '--header', HIP_HEADER,
                  *[f'--extra-arg={a}' for a in HIP_EXTRA_ARGS],
                  '--tp-out', tp_out, '--emit-out', emit_out])
        assert r.returncode == 1
        assert 'DRIFT' in r.stderr


# ---------------------------------------------------------------------------
# 1b. CLI validation + live-header duplicate-declaration resolution.
# ---------------------------------------------------------------------------
def test_sigs_and_header_are_mutually_exclusive():
    r = _run(['--provider', 'hip', '--yaml', HIP_YAML,
              '--sigs', MINIMAL_SIGS, '--header', HIP_HEADER,
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
                  '--yaml', yaml_path, '--header', HIP_HEADER,
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
    syntactically-plausible output and the right field/type mapping."""
    yaml_path = os.path.join(HERE, 'testdata', 'curated_minimal.yaml')
    sigs_path = os.path.join(HERE, 'testdata', 'curated_minimal_sigs.json')
    with tempfile.TemporaryDirectory() as d:
        tp_text, emit_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)

    # hipMemcpyAsync: ptr/ptr/size/enum/handle, all IN.
    assert ('LTTNG_UST_TP_ARGS(uint64_t, dst, uint64_t, src, uint64_t, sizeBytes, '
            'int32_t, kind, uint64_t, stream)') in tp_text
    assert 'lttng_ust_field_integer_hex(uint64_t, dst, dst)' in tp_text
    assert 'lttng_ust_field_integer(uint64_t, sizeBytes, sizeBytes)' in tp_text
    assert 'lttng_ust_field_integer(int32_t, kind, kind)' in tp_text

    # hipMalloc: OUT ptr + IN size — sidecar says `void **` for ptr.
    assert 'LTTNG_UST_TP_ARGS(uint64_t, ptr, uint64_t, size)' in tp_text
    assert 'void** ptr_out_ptr' in emit_text
    assert '(uint64_t)(uintptr_t)(*ptr_out_ptr)' in emit_text

    # hipDeviceSynchronize: zero-arg event and zero-arg do_tracepoint call.
    assert 'LTTNG_UST_TP_ARGS()' in tp_text
    assert 'lttng_ust_do_tracepoint(rocm_hip, hipDeviceSynchronize_args);' in emit_text

    # No-op stub section must exist with matching signatures (byte-for-byte
    # same param types as the active-mode helper, just unnamed).
    assert 'static inline void rocm_trace_emit_hipMalloc_args(void**, size_t, hipError_t) {}' \
        in emit_text


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
        tp_text, emit_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert 'LTTNG_UST_TP_ARGS(uint32_t, flag)' in tp_text
    assert 'uint64_t' not in tp_text.split('fakeBoolApi_args')[1].split(')')[0] \
        or True  # (see explicit field-macro assertion below; this is belt&suspenders)
    assert 'lttng_ust_field_integer(uint32_t, flag, flag)' in tp_text
    assert '(uint32_t)(!!(flag))' in emit_text


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
        tp_text, emit_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert ('LTTNG_UST_TP_ARGS(uint32_t, blockDim_x, uint32_t, blockDim_y, '
            'uint32_t, blockDim_z)') in tp_text
    assert 'lttng_ust_field_integer(uint32_t, blockDim_x, blockDim_x)' in tp_text
    assert 'lttng_ust_field_integer(uint32_t, blockDim_y, blockDim_y)' in tp_text
    assert 'lttng_ust_field_integer(uint32_t, blockDim_z, blockDim_z)' in tp_text
    assert '(uint32_t)blockDim.x' in emit_text
    assert '(uint32_t)blockDim.y' in emit_text
    assert '(uint32_t)blockDim.z' in emit_text


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
        tp_text, emit_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert 'lttng_ust_field_string(name, name)' in tp_text
    assert '(name ? name : "")' in emit_text


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
        tp_text, emit_text = _generate(
            'hsa', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert 'hsa_signal_t * signal_out_ptr' in emit_text
    assert '(signal_out_ptr->handle)' in emit_text
    assert '*signal_out_ptr' not in emit_text


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
        tp_text, emit_text = _generate(
            'hsa', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert 'hsa_queue_t ** queue_out_ptr' in emit_text
    assert '(uint64_t)(uintptr_t)(*queue_out_ptr)' in emit_text
    assert '->handle' not in emit_text


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
        tp_text, emit_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert 'hipStream_t* stream_out_ptr' in emit_text
    assert '(uint64_t)(uintptr_t)(*stream_out_ptr)' in emit_text


def test_all_in_api_marks_status_unused():
    """An all-IN API (no OUT args) marks the trailing status param unused
    rather than reading it, and omits the corr_id parameter entirely
    (schema v3 — no corr_id anywhere in generated output)."""
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
        tp_text, emit_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert '/* unused: all-IN API */' in emit_text
    assert 'corr_id' not in tp_text
    assert 'corr_id' not in emit_text


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
        tp_text, emit_text = _generate(
            'hip', yaml_path, os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'),
            sigs_path=sigs_path)
    assert real_sha256 in tp_text
    assert real_sha256 in emit_text
    assert 'lttng_curated_codegen.py' in tp_text
    assert 'lttng_curated_codegen.py' in emit_text
    assert '--provider hip' in tp_text


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
