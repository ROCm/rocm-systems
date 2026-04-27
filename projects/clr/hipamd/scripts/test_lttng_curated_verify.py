"""Tests for lttng_curated_verify.py — libclang vs YAML drift gate."""
import os, sys, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

VERIFY = os.path.join(HERE, 'lttng_curated_verify.py')
HEADER = os.path.join(HERE, 'testdata', 'fake_hip_header.h')

def _run_verify(yaml_text, expect_pass, extra_cmd=None):
    """Write yaml_text to a temp file, invoke verifier, return (rc, output)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_text)
        yaml_path = f.name
    try:
        cmd = ['python3', VERIFY,
               '--yaml', yaml_path,
               '--header', HEADER]
        if extra_cmd:
            cmd += extra_cmd
        r = subprocess.run(cmd, capture_output=True, text=True)
        if expect_pass:
            assert r.returncode == 0, f"expected pass, got rc={r.returncode}\n{r.stderr}"
        else:
            assert r.returncode != 0, f"expected fail, got pass\n{r.stdout}"
        return r.returncode, r.stdout + r.stderr
    finally:
        os.unlink(yaml_path)

def test_matching_yaml_passes():
    yaml_text = """
- api: hipMemcpyAsync
  category: memory
  args:
    - {name: dst,       type: ptr,    dir: IN}
    - {name: src,       type: ptr,    dir: IN}
    - {name: sizeBytes, type: size,   dir: IN}
    - {name: kind,      type: enum,   dir: IN}
    - {name: stream,    type: handle, dir: IN}
"""
    _run_verify(yaml_text, expect_pass=True)

def test_arg_name_mismatch_fails():
    """Spec §8.3: arg-name mismatch is a hard error."""
    yaml_text = """
- api: hipMemcpyAsync
  category: memory
  args:
    - {name: WRONG_NAME, type: ptr,    dir: IN}
    - {name: src,        type: ptr,    dir: IN}
    - {name: sizeBytes,  type: size,   dir: IN}
    - {name: kind,       type: enum,   dir: IN}
    - {name: stream,     type: handle, dir: IN}
"""
    _, output = _run_verify(yaml_text, expect_pass=False)
    assert 'WRONG_NAME' in output

def test_header_param_omission_warns_but_passes():
    """Spec §4.4 + §8.3: omitting low-value header parameters from YAML
    is allowed (e.g., hsa_amd_memory_async_copy_on_engine drops
    dep_signals to fit field budget). Verifier should pass with an
    informational warning."""
    yaml_text = """
- api: hipMemcpyAsync
  category: memory
  args:
    - {name: dst,       type: ptr,    dir: IN}
    - {name: src,       type: ptr,    dir: IN}
    - {name: sizeBytes, type: size,   dir: IN}
    # kind and stream intentionally omitted (would be a real-world
    # field-budget mitigation if this API were over budget)
"""
    rc, output = _run_verify(yaml_text, expect_pass=True)
    # Should warn about the two omitted header params.
    assert 'kind' in output or 'stream' in output, \
        f"expected omission warnings, got:\n{output}"

def test_extra_yaml_arg_fails():
    """An arg in YAML that doesn't exist in the header is a hard error
    (typo / stale name detection)."""
    yaml_text = """
- api: hipMemcpyAsync
  category: memory
  args:
    - {name: dst,           type: ptr,    dir: IN}
    - {name: src,           type: ptr,    dir: IN}
    - {name: sizeBytes,     type: size,   dir: IN}
    - {name: kind,          type: enum,   dir: IN}
    - {name: stream,        type: handle, dir: IN}
    - {name: NONEXISTENT,   type: int32,  dir: IN}
"""
    _, output = _run_verify(yaml_text, expect_pass=False)
    assert 'NONEXISTENT' in output

def test_inout_rejected():
    """Spec §4.4 v1: dir: INOUT is hard error in verifier."""
    yaml_text = """
- api: hipMemcpyAsync
  category: memory
  args:
    - {name: dst,       type: ptr,    dir: INOUT}
    - {name: src,       type: ptr,    dir: IN}
    - {name: sizeBytes, type: size,   dir: IN}
    - {name: kind,      type: enum,   dir: IN}
    - {name: stream,    type: handle, dir: IN}
"""
    _, output = _run_verify(yaml_text, expect_pass=False)
    assert 'INOUT' in output

def test_over_budget_rejected():
    """Spec §4.4: over-budget is hard error."""
    yaml_text = """
- api: hipMemcpyAsync
  category: memory
  args:
""" + '\n'.join(f"    - {{name: a{i}, type: uint32, dir: IN}}" for i in range(11))
    _, output = _run_verify(yaml_text, expect_pass=False)
    assert 'budget' in output.lower() or 'fields' in output.lower()

def test_api_missing_in_header_fails():
    yaml_text = """
- api: hipNonExistent
  category: memory
  args: []
"""
    _, output = _run_verify(yaml_text, expect_pass=False)
    assert 'hipNonExistent' in output

def test_sidecar_emission():
    """Verifier with --out-sidecar dumps {api: [{name, c_type}, ...]} JSON
    for codegen consumption (debate-review C10 fix). Codegen needs the
    libclang-resolved C signatures to emit provider-correct OUT-handle
    helper signatures (HIP void** vs HSA struct-with-.handle) without
    itself needing libclang at build time."""
    import json
    yaml_text = """
- api: hipMalloc
  category: memory
  args:
    - {name: ptr,  type: ptr,  dir: OUT}
    - {name: size, type: size, dir: IN}
"""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        sidecar_path = f.name
    try:
        _run_verify(yaml_text, expect_pass=True,
                    extra_cmd=['--out-sidecar', sidecar_path])
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        assert 'hipMalloc' in sidecar, f"hipMalloc missing: {sidecar}"
        params = sidecar['hipMalloc']
        assert isinstance(params, list), f"expected list, got {type(params)}"
        names = [p['name'] for p in params]
        assert names == ['ptr', 'size'], f"got names {names}"
        # The first arg is an OUT void**.
        assert 'void **' in params[0]['c_type'] or 'void**' in params[0]['c_type'], \
            f"expected void** for ptr arg, got {params[0]['c_type']!r}"
    finally:
        os.unlink(sidecar_path)

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
