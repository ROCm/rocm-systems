"""Smoke test: migrator emits curated sentinels and macro variants
correctly on a synthetic source file."""
import os
import subprocess
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))


def test_curated_emit():
    yaml_text = textwrap.dedent("""\
    - api: hipMemcpyAsync
      category: memory
      args:
        - {name: dst,       type: ptr,    dir: IN}
        - {name: src,       type: ptr,    dir: IN}
        - {name: sizeBytes, type: size,   dir: IN}
        - {name: kind,      type: enum,   dir: IN}
        - {name: stream,    type: handle, dir: IN}

    - api: hipMalloc
      category: memory
      args:
        - {name: ptr,  type: ptr,  dir: OUT}
        - {name: size, type: size, dir: IN}

    - api: hipDeviceSynchronize
      category: streams
      args: []
    """)

    src_text = textwrap.dedent("""\
    #include <hip/hip_runtime.h>

    hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                              hipMemcpyKind kind, hipStream_t stream) {
        return hipSuccess;
    }

    hipError_t hipMalloc(void** ptr, size_t size) {
        return hipSuccess;
    }

    hipError_t hipDeviceSynchronize(void) {
        return hipSuccess;
    }
    """)

    with tempfile.TemporaryDirectory() as d:
        yaml_path = os.path.join(d, 'apis.yaml')
        src_path = os.path.join(d, 'wrappers.cpp')
        inv_path = os.path.join(d, 'inv.txt')
        with open(yaml_path, 'w') as f:
            f.write(yaml_text)
        with open(src_path, 'w') as f:
            f.write(src_text)
        cmd = ['python3', os.path.join(HERE, 'lttng_migrate.py'),
               '--source', src_path,
               '--include-path', '/opt/rocm/include',
               '--extra-arg=-D__HIP_PLATFORM_AMD__=1',
               '--inventory', inv_path,
               '--curated-yaml', yaml_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"migrate failed:\n{r.stderr}"
        with open(src_path) as f:
            out = f.read()

    # Sentinel for each curated API.
    assert '/* __ROCM_CURATED__: hipMemcpyAsync */' in out, out
    assert '/* __ROCM_CURATED__: hipMalloc */' in out, out
    assert '/* __ROCM_CURATED__: hipDeviceSynchronize */' in out, out

    # IN-locals for the IN args of hipMemcpyAsync.
    assert '__rocm_in_dst' in out, out
    assert '__rocm_in_sizeBytes' in out, out
    assert '__rocm_in_kind' in out, out
    assert '__rocm_in_stream' in out, out

    # hipMalloc: size is IN, so __rocm_in_size IS expected.
    assert '__rocm_in_size' in out, out
    # ptr is OUT — no __rocm_in_ptr.
    # (Be careful: __rocm_in_ptr could match __rocm_in_ptr_out — but no
    # such identifier is generated. Use full word boundary check.)
    import re
    assert re.search(r'__rocm_in_ptr\b', out) is None, \
        "OUT-only arg ptr should not have an __rocm_in_ local"

    # _CURATED macro variants.
    assert 'ROCM_TRACE_RET_STATUS_CURATED(hipMemcpyAsync,' in out, out
    assert 'ROCM_TRACE_RET_STATUS_CURATED(hipMalloc,' in out, out
    assert 'ROCM_TRACE_RET_STATUS_CURATED_NOARGS(hipDeviceSynchronize,' in out, out

    # C6 regression: IN-locals must use 'auto const' so that C++14+ type
    # deduction preserves the wrapper's actual parameter type. Previously
    # we used the libclang-resolved type, which silently truncated size_t
    # to int when system headers weren't available, corrupting any
    # allocation/copy >= 2^31 bytes at the assignment line.
    assert 'auto const __rocm_in_sizeBytes' in out, \
        f"Expected 'auto const __rocm_in_sizeBytes' in migrator output " \
        f"for type safety:\n{out}"
    assert 'auto const __rocm_in_size' in out, \
        f"Expected 'auto const __rocm_in_size' in migrator output for " \
        f"type safety:\n{out}"
    # Negative assertions: narrow-typed IN-locals are forbidden for size args.
    assert 'int const __rocm_in_sizeBytes' not in out, \
        f"REGRESSION (C6): int-typed IN-local for size_t parameter " \
        f"sizeBytes:\n{out}"
    assert 'int const __rocm_in_size' not in out, \
        f"REGRESSION (C6): int-typed IN-local for size_t parameter size:\n{out}"
    # Stronger guard: every IN-local declaration must be ' auto const '.
    # Catch any future regression where a libclang-resolved type leaks
    # into the IN-local declaration (the leading space is part of the
    # migrator's emission template).
    import re
    for m in re.finditer(r'__rocm_in_\w+\s*=', out):
        # Look at the ~30 chars preceding each IN-local assignment.
        start = max(0, m.start() - 32)
        prefix = out[start:m.start()]
        assert 'auto const ' in prefix, \
            f"REGRESSION (C6): IN-local '{m.group(0)}' not declared as " \
            f"'auto const'; preceding context: {prefix!r}\n{out}"

    print('PASS')


if __name__ == '__main__':
    test_curated_emit()
