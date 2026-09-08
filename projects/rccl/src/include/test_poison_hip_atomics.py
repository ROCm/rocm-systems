#!/usr/bin/env python3
"""Guard that #pragma GCC poison in poison_hip_atomics.h actually rejects
__hip_atomic_* builtins.

Nothing in RCCL CI runs ctest, so this is invoked from the host-test pipeline
(test/host/run_host_tests.sh `guards` phase, folded into `run`) rather than
add_test().  It compiles tiny HIP probes with amdclang++; no GPU and no
librccl.so are required (-nogpulib -fsyntax-only).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
POISON_H = os.path.join(HERE, "poison_hip_atomics.h")

# Names the pragma in poison_hip_atomics.h must reject. Keep in lockstep with
# that file: dropping a name from the pragma must fail this test.
POISONED = (
    "__hip_atomic_load",
    "__hip_atomic_store",
    "__hip_atomic_exchange",
    "__hip_atomic_compare_exchange_weak",
    "__hip_atomic_compare_exchange_strong",
    "__hip_atomic_fetch_add",
    "__hip_atomic_fetch_sub",
    "__hip_atomic_fetch_and",
    "__hip_atomic_fetch_or",
    "__hip_atomic_fetch_xor",
    "__hip_atomic_fetch_min",
    "__hip_atomic_fetch_max",
)

def _find_cxx():
    for env in ("HIPCXX", "CXX"):
        val = os.environ.get(env)
        if val and shutil.which(val):
            return val
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    for cand in (
        os.path.join(rocm, "bin", "amdclang++"),
        os.path.join(rocm, "llvm", "bin", "clang++"),
        os.path.join(rocm, "bin", "hipcc"),
    ):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    for name in ("amdclang++", "hipcc"):
        found = shutil.which(name)
        if found:
            return found
    return None


CXX = _find_cxx()


def _call_expr(name):
    """A well-typed call to `name` that survives HIP host+device parsing."""
    if name in ("__hip_atomic_load",):
        return f"{name}(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)"
    if name in ("__hip_atomic_store",):
        return f"{name}(p, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)"
    if name in ("__hip_atomic_exchange",):
        return f"{name}(p, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)"
    if name in (
        "__hip_atomic_compare_exchange_weak",
        "__hip_atomic_compare_exchange_strong",
    ):
        return (
            f"{name}(p, &expected, 1u, __ATOMIC_RELAXED, __ATOMIC_RELAXED, "
            "__HIP_MEMORY_SCOPE_AGENT)"
        )
    # fetch_* family
    return f"{name}(p, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)"


def _probe_src(call):
    return textwrap.dedent(
        f"""\
        #include <hip/hip_runtime.h>
        __global__ void k(unsigned int *p) {{
          unsigned int expected = 0;
          (void)expected;
          (void){call};
        }}
        """
    )


def _compile(src, *, poison, offload):
    """Return (returncode, combined stdout+stderr)."""
    cmd = [CXX, "-x", "hip", "-nogpulib", "-fsyntax-only"]
    if offload == "host":
        cmd += ["--offload-host-only"]
    elif offload == "device":
        cmd += ["--offload-device-only", "--offload-arch=gfx942"]
    else:
        raise ValueError(offload)
    if poison:
        cmd += [f"--include={POISON_H}"]
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        proc = subprocess.run(cmd + [path], capture_output=True, text=True)
        return proc.returncode, proc.stdout + proc.stderr
    finally:
        os.unlink(path)


@unittest.skipUnless(CXX, "amdclang++/hipcc not found (set ROCM_PATH or HIPCXX)")
class PoisonHipAtomicsTest(unittest.TestCase):
    def test_poison_header_exists(self):
        self.assertTrue(os.path.isfile(POISON_H), POISON_H)

    def test_hip_atomic_load_rejected_on_device(self):
        rc, out = _compile(
            _probe_src(_call_expr("__hip_atomic_load")),
            poison=True,
            offload="device",
        )
        self.assertNotEqual(rc, 0, msg=out)
        self.assertIn("poisoned identifier", out, msg=out)
        self.assertIn("__hip_atomic_load", out, msg=out)

    def test_hip_atomic_load_rejected_on_host(self):
        rc, out = _compile(
            _probe_src(_call_expr("__hip_atomic_load")),
            poison=True,
            offload="host",
        )
        self.assertNotEqual(rc, 0, msg=out)
        self.assertIn("poisoned identifier", out, msg=out)
        self.assertIn("__hip_atomic_load", out, msg=out)

    def test_hip_atomic_load_accepted_without_poison(self):
        rc, out = _compile(
            _probe_src(_call_expr("__hip_atomic_load")),
            poison=False,
            offload="device",
        )
        self.assertEqual(rc, 0, msg=out)

    def test_scoped_atomic_load_accepted_with_poison(self):
        src = textwrap.dedent(
            """\
            #include <hip/hip_runtime.h>
            __global__ void k(unsigned int *p, unsigned int *out) {
              *out = __scoped_atomic_load_n(p, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
            }
            """
        )
        rc, out = _compile(src, poison=True, offload="device")
        self.assertEqual(rc, 0, msg=out)

    def test_every_poisoned_builtin_is_rejected(self):
        missing = []
        for name in POISONED:
            rc, out = _compile(_probe_src(_call_expr(name)), poison=True, offload="device")
            if rc == 0 or "poisoned identifier" not in out:
                missing.append(f"{name}: rc={rc}\n{out}")
        self.assertFalse(
            missing,
            "poison header did not reject:\n" + "\n".join(missing),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
