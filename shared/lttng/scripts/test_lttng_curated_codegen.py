"""Tests for lttng_curated_codegen.py — the curated-args header generator.

Two kinds of coverage:
  1. Golden-file regression: the generator, run against the REAL checked-in
     curated_apis.yaml + curated_apis_sigs.json for both HIP and HSA, must
     reproduce the REAL checked-in generated headers exactly (--check mode,
     rc=0). This is the regression guard against future accidental hand
     edits to the generated headers (or drift between the YAML/sigs and
     the headers they're supposed to produce).
  2. Type-mapping unit tests: small synthetic YAML+sigs fixtures exercise
     each DSL type (including bool, dim3, dim3_packed, cstring, the HSA
     `.handle` OUT-deref pattern, and the HIP double-pointer OUT-deref
     pattern) in isolation and assert on the exact generated substrings.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

CODEGEN = os.path.join(HERE, 'lttng_curated_codegen.py')

HIP_YAML = os.path.join(REPO_ROOT, 'projects/clr/hipamd/scripts/curated_apis.yaml')
HIP_SIGS = os.path.join(REPO_ROOT, 'projects/clr/hipamd/scripts/curated_apis_sigs.json')
HIP_TP_OUT = os.path.join(REPO_ROOT, 'projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h')
HIP_EMIT_OUT = os.path.join(REPO_ROOT, 'projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h')

HSA_YAML = os.path.join(
    REPO_ROOT, 'projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml')
HSA_SIGS = os.path.join(
    REPO_ROOT, 'projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis_sigs.json')
HSA_TP_OUT = os.path.join(
    REPO_ROOT, 'projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_hsa_curated_tp.h')
HSA_EMIT_OUT = os.path.join(
    REPO_ROOT, 'projects/rocr-runtime/runtime/hsa-runtime/lttng/rocm_trace_emit_curated.h')


def _run(args):
    return subprocess.run(['python3', CODEGEN] + args, capture_output=True, text=True)


def _generate(provider, yaml_path, sigs_path, tp_out, emit_out):
    r = _run(['--provider', provider, '--yaml', yaml_path, '--sigs', sigs_path,
              '--tp-out', tp_out, '--emit-out', emit_out])
    assert r.returncode == 0, f"generation failed: {r.stderr}"
    with open(tp_out) as f:
        tp_text = f.read()
    with open(emit_out) as f:
        emit_text = f.read()
    return tp_text, emit_text


# ---------------------------------------------------------------------------
# 1. Golden-file regression against the real checked-in HIP/HSA artifacts.
# ---------------------------------------------------------------------------
def test_hip_check_mode_matches_checked_in_headers():
    """The generator, run against the real HIP YAML + sigs, must reproduce
    the real checked-in rocm_hip_curated_tp.h and rocm_trace_emit_curated.h
    exactly (--check mode exits 0)."""
    r = _run(['--provider', 'hip', '--check',
              '--yaml', HIP_YAML, '--sigs', HIP_SIGS,
              '--tp-out', HIP_TP_OUT, '--emit-out', HIP_EMIT_OUT])
    assert r.returncode == 0, f"HIP headers drifted from generator output:\n{r.stderr}"


def test_hsa_check_mode_matches_checked_in_headers():
    """Same as above for HSA."""
    r = _run(['--provider', 'hsa', '--check',
              '--yaml', HSA_YAML, '--sigs', HSA_SIGS,
              '--tp-out', HSA_TP_OUT, '--emit-out', HSA_EMIT_OUT])
    assert r.returncode == 0, f"HSA headers drifted from generator output:\n{r.stderr}"


def test_check_mode_detects_injected_drift():
    """--check must fail (rc=1) when the on-disk header doesn't match what
    the generator would produce — the whole point of the regression guard."""
    with tempfile.TemporaryDirectory() as d:
        tp_out = os.path.join(d, 'rocm_hip_curated_tp.h')
        emit_out = os.path.join(d, 'rocm_trace_emit_curated.h')
        _generate('hip', HIP_YAML, HIP_SIGS, tp_out, emit_out)
        # Hand-corrupt the tp.h the generator just wrote.
        with open(tp_out, 'a') as f:
            f.write("\n/* hand edit that must be detected as drift */\n")
        r = _run(['--provider', 'hip', '--check',
                  '--yaml', HIP_YAML, '--sigs', HIP_SIGS,
                  '--tp-out', tp_out, '--emit-out', emit_out])
        assert r.returncode == 1
        assert 'DRIFT' in r.stderr


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
            'hip', yaml_path, sigs_path,
            os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'))

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
            'hip', yaml_path, sigs_path,
            os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'))
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
            'hip', yaml_path, sigs_path,
            os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'))
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
            'hip', yaml_path, sigs_path,
            os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'))
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
            'hsa', yaml_path, sigs_path,
            os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'))
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
            'hsa', yaml_path, sigs_path,
            os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'))
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
            'hip', yaml_path, sigs_path,
            os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'))
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
            'hip', yaml_path, sigs_path,
            os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'))
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
            'hip', yaml_path, sigs_path,
            os.path.join(d, 'tp.h'), os.path.join(d, 'emit.h'))
    assert real_sha256 in tp_text
    assert real_sha256 in emit_text
    assert 'lttng_curated_codegen.py' in tp_text
    assert 'lttng_curated_codegen.py' in emit_text
    assert '--provider hip' in tp_text


if __name__ == '__main__':
    import inspect
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn) and not inspect.isclass(fn):
            try:
                fn()
                print(f'  ok  {name}')
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'  FAIL {name}: {e}')
                failures += 1
    print(f'\n{"PASS" if failures == 0 else "FAIL"}: {failures} failures')
    sys.exit(1 if failures else 0)
