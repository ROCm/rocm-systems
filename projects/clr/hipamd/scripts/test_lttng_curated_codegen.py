"""Golden-file tests for lttng_curated_codegen.py."""
import os, sys, subprocess, tempfile, hashlib, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CODEGEN = os.path.join(HERE, 'lttng_curated_codegen.py')
YAML    = os.path.join(HERE, 'testdata', 'curated_minimal.yaml')

def _run_codegen(provider, status_type='hipError_t', status_success='hipSuccess',
                 sigs_path=None, yaml_path=None):
    """Invoke codegen, return (tp_h_text, emit_h_text)."""
    with tempfile.TemporaryDirectory() as d:
        tp = os.path.join(d, 'tp.h')
        em = os.path.join(d, 'emit.h')
        cmd = ['python3', CODEGEN,
               '--yaml', yaml_path or YAML,
               '--provider', provider,
               '--status-type', status_type,
               '--status-success', status_success,
               '--out-tp', tp,
               '--out-emit', em]
        if sigs_path:
            cmd += ['--sigs', sigs_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"codegen failed:\n{r.stderr}"
        with open(tp) as f: tp_text = f.read()
        with open(em) as f: em_text = f.read()
    return tp_text, em_text

def test_emits_tracepoint_event_per_api():
    tp, _ = _run_codegen('rocm_hip')
    assert 'hipMemcpyAsync_args' in tp
    assert 'hipMalloc_args' in tp
    assert 'hipDeviceSynchronize_args' in tp
    # Schema version comment present.
    assert 'AUTO-GENERATED' in tp
    # Provider correctly templated.
    assert 'rocm_hip,' in tp
    assert 'rocm_hsa' not in tp

def test_emits_corr_id_field_first():
    tp, _ = _run_codegen('rocm_hip')
    # Find the hipMemcpyAsync event block and assert corr_id is first field.
    block_start = tp.index('hipMemcpyAsync_args')
    block = tp[block_start:block_start+1200]
    fields_start = block.index('LTTNG_UST_TP_FIELDS')
    fields_section = block[fields_start:fields_start+800]
    # First field decl must be corr_id.
    first_field_decl_idx = fields_section.index('lttng_ust_field_')
    assert 'corr_id' in fields_section[first_field_decl_idx:first_field_decl_idx+80]

def test_emits_helper_per_api():
    _, em = _run_codegen('rocm_hip')
    assert 'rocm_trace_emit_hipMemcpyAsync_args' in em
    assert 'rocm_trace_emit_hipMalloc_args' in em
    assert 'rocm_trace_emit_hipDeviceSynchronize_args' in em

def test_helper_signature_takes_status_last_for_all_in_api():
    """Spec §6.2: every helper takes status as last param, even all-IN."""
    _, em = _run_codegen('rocm_hip')
    # Look for hipMemcpyAsync helper signature, last param must include hipError_t.
    sig_start = em.index('rocm_trace_emit_hipMemcpyAsync_args')
    sig_end = em.index(')', sig_start) + 1
    sig = em[sig_start:sig_end]
    assert 'hipError_t' in sig, f"signature missing hipError_t status: {sig!r}"

def test_out_param_helper_uses_status_to_gate_deref():
    """Spec §5.2 hipMalloc example: helper deref's *ptr_out only on success."""
    _, em = _run_codegen('rocm_hip')
    body_start = em.index('rocm_trace_emit_hipMalloc_args')
    body_end = em.index('}', body_start) + 1
    body = em[body_start:body_end]
    # Must reference hipSuccess as the success sentinel.
    assert 'hipSuccess' in body
    # Must guard the deref behind the status check (look for ternary).
    assert '?' in body and ':' in body  # ternary guard

def test_no_arg_helper_signature():
    """Spec §6.2 _NOARGS variant: helper signature is just (corr_id, status)."""
    _, em = _run_codegen('rocm_hip')
    sig_start = em.index('rocm_trace_emit_hipDeviceSynchronize_args')
    sig_end = em.index(')', sig_start) + 1
    sig = em[sig_start:sig_end]
    # Only 2 params: corr_id and status.
    assert sig.count(',') == 1, f"no-arg helper should have 2 params: {sig!r}"
    assert 'corr_id' in sig
    assert 'hipError_t' in sig

def test_no_op_branch_for_off_mode():
    """Spec §5.2: when HIP_ENABLE_LTTNG_UST=0, helpers are no-ops."""
    _, em = _run_codegen('rocm_hip')
    assert '#if defined(HIP_ENABLE_LTTNG_UST)' in em or 'HIP_ENABLE_LTTNG_UST' in em
    assert '#else' in em
    # Check that the #else branch contains no-op helper definitions.
    else_idx = em.index('#else')
    after_else = em[else_idx:]
    assert 'rocm_trace_emit_hipMemcpyAsync_args' in after_else

def test_provider_parameterization_for_hsa():
    """Same script generates HSA tracepoint events when --provider rocm_hsa."""
    tp, em = _run_codegen('rocm_hsa', status_type='hsa_status_t',
                           status_success='HSA_STATUS_SUCCESS')
    # HSA provider in tp definitions
    assert 'rocm_hsa,' in tp
    # HSA status type in helper signatures
    assert 'hsa_status_t' in em

def test_yaml_sha256_in_header():
    """Spec §5.1: header includes SHA256(curated_apis.yaml) comment."""
    tp, _ = _run_codegen('rocm_hip')
    with open(YAML, 'rb') as f:
        expected = hashlib.sha256(f.read()).hexdigest()
    assert expected[:16] in tp, f"sha256 prefix {expected[:16]} not in tp.h"


# ---- C10 fix: provider-aware OUT-handle helper signatures via --sigs sidecar ----

def test_out_handle_helper_uses_void_pp_without_sigs():
    """C10 fallback: without --sigs, OUT pointer/handle helpers use void**.

    Locks in the legacy fallback so existing HIP golden tests keep passing
    when no verifier sidecar is supplied.
    """
    _, em = _run_codegen('rocm_hip')  # no sigs
    # Find hipMalloc helper signature.
    sig_start = em.index('rocm_trace_emit_hipMalloc_args')
    sig_end = em.index(')', sig_start) + 1
    sig = em[sig_start:sig_end]
    assert 'void**' in sig or 'void **' in sig, \
        f"expected void** fallback in hipMalloc helper signature, got: {sig!r}"
    # Body deref expression should use (uint64_t)(uintptr_t)(*ptr_out_ptr).
    body_start = em.index('rocm_trace_emit_hipMalloc_args')
    # Limit body search to the active-mode definition (before #else).
    else_idx = em.index('#else')
    body = em[body_start:else_idx]
    assert '(uint64_t)(uintptr_t)(*ptr_out_ptr)' in body, \
        f"expected legacy fallback deref in body; got:\n{body}"

def test_out_handle_helper_uses_struct_handle_with_hsa_sigs():
    """C10 fix: with --sigs giving hsa_signal_t*, helper uses struct deref.

    Fixture: HSA-style YAML for hsa_signal_create with OUT 'signal' arg.
    Sidecar declares c_type 'hsa_signal_t *'. Helper signature should
    contain 'hsa_signal_t *signal_out_ptr' and the deref body should
    reference 'signal_out_ptr->handle' (NOT '*signal_out_ptr').
    """
    with tempfile.TemporaryDirectory() as d:
        # Minimal HSA-style YAML.
        ya = os.path.join(d, 'hsa.yaml')
        with open(ya, 'w') as f:
            f.write(
                "- api: hsa_signal_create\n"
                "  category: hsa_signals\n"
                "  args:\n"
                "    - {name: initial_value, type: uint64, dir: IN}\n"
                "    - {name: signal,        type: handle, dir: OUT}\n"
            )
        # Sidecar JSON keyed by api name.
        sigs = os.path.join(d, 'sigs.json')
        with open(sigs, 'w') as f:
            json.dump({
                'hsa_signal_create': [
                    {'name': 'initial_value', 'c_type': 'hsa_signal_value_t'},
                    {'name': 'signal',        'c_type': 'hsa_signal_t *'},
                ]
            }, f)
        _, em = _run_codegen('rocm_hsa',
                              status_type='hsa_status_t',
                              status_success='HSA_STATUS_SUCCESS',
                              sigs_path=sigs, yaml_path=ya)
    # Helper signature includes hsa_signal_t * for signal_out_ptr.
    sig_start = em.index('rocm_trace_emit_hsa_signal_create_args')
    sig_end = em.index(')', sig_start) + 1
    sig = em[sig_start:sig_end]
    assert 'hsa_signal_t' in sig and 'signal_out_ptr' in sig, \
        f"expected hsa_signal_t* in signal_out_ptr param; got: {sig!r}"
    # Body should deref via struct field, NOT *signal_out_ptr.
    else_idx = em.index('#else')
    body = em[sig_start:else_idx]
    assert 'signal_out_ptr->handle' in body, \
        f"expected struct-field deref ->handle in body; got:\n{body}"


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
