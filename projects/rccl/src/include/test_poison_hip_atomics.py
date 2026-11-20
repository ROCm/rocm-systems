#!/usr/bin/env python3
"""Guard that #pragma GCC poison in poison_hip_atomics.h actually rejects
__hip_atomic_* builtins.

Nothing in RCCL CI runs ctest, so this is invoked from the host-test pipeline
(test/host/run_host_tests.sh `guards` phase, folded into `run`) rather than
add_test().  It compiles tiny HIP probes with amdclang++; no GPU and no
librccl.so are required (-nogpulib -fsyntax-only).

A missing HIP compiler is a failure here, not a skip: unittest exits 0 on a
fully skipped suite, so a toolchain-less run would report success without ever
compiling a probe.  Set RCCL_POISON_TEST_ALLOW_SKIP=1 to downgrade that to a
skip when running on a box without ROCm.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
POISON_H = os.path.join(HERE, "poison_hip_atomics.h")
RCCL_ROOT = os.path.dirname(os.path.dirname(HERE))
TEST_CMAKE = os.path.join(RCCL_ROOT, "test", "CMakeLists.txt")
HOST_TEST_CMAKE = os.path.join(RCCL_ROOT, "test", "host", "CMakeLists.txt")

# Names the pragma in poison_hip_atomics.h must reject.  Kept in lockstep with
# that file by test_pragma_matches_poisoned_tuple, which compares this tuple
# against the pragma itself -- drift in either direction fails.
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

# Opt out of the "no compiler is a failure" rule for local runs without ROCm.
ALLOW_SKIP = os.environ.get("RCCL_POISON_TEST_ALLOW_SKIP") == "1"

_PRAGMA_RE = re.compile(r"\s*#\s*pragma\s+GCC\s+poison\b")


def _pragma_poisoned_names(text):
    """Identifiers listed in the `#pragma GCC poison` directive in `text`.

    Joins backslash continuations so the multi-line spelling in the header
    parses the same as a single-line one.  Returns () when no pragma is found,
    which fails the comparison rather than silently matching an empty tuple.
    """
    lines = text.splitlines()
    for start, line in enumerate(lines):
        if not _PRAGMA_RE.match(line):
            continue
        parts = []
        i = start
        while i < len(lines):
            stripped = lines[i].rstrip()
            continued = stripped.endswith("\\")
            parts.append(stripped[:-1] if continued else stripped)
            i += 1
            if not continued:
                break
        joined = _PRAGMA_RE.sub("", " ".join(parts), count=1)
        return tuple(joined.split())
    return ()


_FORCE_INCLUDE_RE = re.compile(
    r"\badd_compile_options\(\s*--include=\S*/poison_hip_atomics\.h\s*\)"
)
_OPTION_IF_RE = re.compile(r"^\s*if\(\s*RCCL_POISON_HIP_ATOMICS\s*\)")
# ALIAS targets compile nothing, so the GTest aliases ahead of the poison in
# rccl_standalone_setup() are not a gap.
_TARGET_RE = re.compile(r"^\s*(add_executable|add_library|add_test_executable)\(")


def _cmake_code_lines(text):
    """(1-based line number, line) for every line that is not blank or a pure comment."""
    return [
        (n, line)
        for n, line in enumerate(text.splitlines(), 1)
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _poison_wiring_errors(lines, where):
    """Problems with the test force-include among `lines`, or [] if wired.

    It has to be a directory-scope add_compile_options() under
    if(RCCL_POISON_HIP_ATOMICS), ahead of the first target it is meant to reach:
    add_compile_options() only applies to targets defined after it.
    """
    # Searched across lines so a call wrapped one argument per line (gersemi's
    # style) still matches; the match start is the add_compile_options( line.
    text = "\n".join(l for _, l in lines)
    match = _FORCE_INCLUDE_RE.search(text)
    if match is None:
        return [f"{where}: no add_compile_options(--include=.../poison_hip_atomics.h)"]
    include_at = text[: match.start()].count("\n")
    errors = []
    if include_at == 0 or not _OPTION_IF_RE.match(lines[include_at - 1][1]):
        errors.append(
            f"{where}: force-include is not directly under if(RCCL_POISON_HIP_ATOMICS)"
        )
    first_target = next(
        ((n, l) for n, l in lines if _TARGET_RE.match(l) and "ALIAS" not in l), None
    )
    if first_target is not None and first_target[0] < lines[include_at][0]:
        errors.append(
            f"{where}: line {first_target[0]} defines a target before the force-include "
            f"at line {lines[include_at][0]}, so it is built without the poison:\n"
            f"    {first_target[1].strip()}"
        )
    return errors


def _function_body(lines, name):
    """The lines strictly inside `function(name)` ... `endfunction()`."""
    start = next(
        (
            i
            for i, (_, l) in enumerate(lines)
            if re.match(rf"^\s*function\(\s*{name}\b", l)
        ),
        None,
    )
    if start is None:
        return None
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if re.match(r"^\s*endfunction\(", lines[i][1])
        ),
        len(lines),
    )
    return lines[start + 1 : end]


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
    return textwrap.dedent(f"""\
        #include <hip/hip_runtime.h>
        __global__ void k(unsigned int *p) {{
          unsigned int expected = 0;
          (void)expected;
          (void){call};
        }}
        """)


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


class PoisonHeaderTest(unittest.TestCase):
    """Checks on the header text itself.  These need no HIP compiler, so they
    still run (and still gate) on a machine without ROCm."""

    def test_poison_header_exists(self):
        self.assertTrue(os.path.isfile(POISON_H), POISON_H)

    def test_pragma_matches_poisoned_tuple(self):
        """POISONED and the pragma are two hand-maintained lists; keep them equal.

        Without this, dropping a name from POISONED just shrinks the compile
        probes below and nothing fails -- the direction that silently loses
        coverage.
        """
        with open(POISON_H) as f:
            pragma = _pragma_poisoned_names(f.read())
        self.assertEqual(
            set(pragma),
            set(POISONED),
            "#pragma GCC poison in poison_hip_atomics.h and POISONED here have "
            f"drifted.\n  only in pragma:   {sorted(set(pragma) - set(POISONED))}"
            f"\n  only in POISONED: {sorted(set(POISONED) - set(pragma))}",
        )


class TestBuildWiringTest(unittest.TestCase):
    """The test binaries get the poison through their own force-includes, not
    the rccl target's.  Dropping or moving one of those would quietly let
    __hip_atomic_* back into the test tree, so pin both.  CMake text only; no
    HIP compiler or configured build is needed."""

    def test_in_tree_test_build_force_includes_poison(self):
        with open(TEST_CMAKE) as f:
            lines = _cmake_code_lines(f.read())
        errors = _poison_wiring_errors(lines, "test/CMakeLists.txt")
        self.assertFalse(errors, "\n".join(errors))

    def test_standalone_host_build_force_includes_poison(self):
        """run_host_tests.sh configures test/host on its own, so the root and
        test/ CMakeLists never run; rccl_standalone_setup() must add it."""
        with open(HOST_TEST_CMAKE) as f:
            body = _function_body(_cmake_code_lines(f.read()), "rccl_standalone_setup")
        self.assertIsNotNone(
            body, "rccl_standalone_setup() not found in test/host/CMakeLists.txt"
        )
        errors = _poison_wiring_errors(
            body, "test/host/CMakeLists.txt rccl_standalone_setup()"
        )
        self.assertFalse(errors, "\n".join(errors))

    def test_blank_and_comment_lines_do_not_separate_include_from_if(self):
        text = (
            "if(RCCL_POISON_HIP_ATOMICS)\n"
            "\n"
            "  # comment\n"
            "  add_compile_options(--include=${X}/poison_hip_atomics.h)\n"
            "endif()\n"
            "add_executable(t t.cpp)\n"
        )
        errors = _poison_wiring_errors(_cmake_code_lines(text), "inline")
        self.assertFalse(errors, "\n".join(errors))


class ToolchainTest(unittest.TestCase):
    """Makes a toolchain-less run red.

    Every compile probe below can only skip when no HIP compiler is found, and
    a fully skipped suite exits 0.  This is the test that tells "the poison
    works" apart from "nothing was ever compiled".
    """

    @unittest.skipIf(ALLOW_SKIP, "RCCL_POISON_TEST_ALLOW_SKIP=1")
    def test_hip_compiler_available(self):
        self.assertIsNotNone(
            CXX,
            "no HIP compiler found: set ROCM_PATH, HIPCXX or CXX, or put "
            "amdclang++/hipcc on PATH. Without one the poison probes cannot "
            "run, and skipping them would let this guard pass without "
            "checking anything. Set RCCL_POISON_TEST_ALLOW_SKIP=1 to skip "
            "instead when running locally without ROCm.",
        )


@unittest.skipUnless(CXX, "amdclang++/hipcc not found (set ROCM_PATH or HIPCXX)")
class PoisonHipAtomicsTest(unittest.TestCase):
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

    def test_every_probe_is_well_formed(self):
        """Each _call_expr() must compile with the poison off.

        #pragma GCC poison fires in the lexer, so the rejection tests pass on
        any spelling of the argument list.  This is what keeps the per-name
        arity table in _call_expr honest for all 12 names rather than just the
        one the load tests happen to use.
        """
        broken = []
        for name in POISONED:
            rc, out = _compile(
                _probe_src(_call_expr(name)), poison=False, offload="device"
            )
            if rc != 0:
                broken.append(f"{name}: rc={rc}\n{out}")
        self.assertFalse(
            broken,
            "probe is not a well-formed call without the poison:\n" + "\n".join(broken),
        )

    def test_scoped_atomic_load_accepted_with_poison(self):
        src = textwrap.dedent("""\
            #include <hip/hip_runtime.h>
            __global__ void k(unsigned int *p, unsigned int *out) {
              *out = __scoped_atomic_load_n(p, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
            }
            """)
        rc, out = _compile(src, poison=True, offload="device")
        self.assertEqual(rc, 0, msg=out)

    def test_scoped_atomic_compare_exchange_accepted_with_poison(self):
        """Pin the replacement compare-exchange shape for both widths.

        This is the one conversion in the scoped-atomics port whose argument
        list changed rather than just its name, gaining a weak flag ahead of
        the success/failure orders.  Mirrors the spelling atomic_ref uses in
        nccl_device/hip_compat.h: (ptr, expected, desired, weak, success,
        failure, scope).  It pins the builtin, not hip_compat.h -- a reshape
        of that header still needs review.
        """
        for cxx_ty in ("unsigned int", "unsigned long long"):
            with self.subTest(type=cxx_ty):
                src = textwrap.dedent(f"""\
                    #include <hip/hip_runtime.h>
                    __global__ void k({cxx_ty} *p, bool *out) {{
                      {cxx_ty} expected = 0;
                      *out = __scoped_atomic_compare_exchange_n(
                          p, &expected, ({cxx_ty})1, /*weak=*/true,
                          __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);
                    }}
                    """)
                rc, out = _compile(src, poison=True, offload="device")
                self.assertEqual(rc, 0, msg=out)

    def test_every_poisoned_builtin_is_rejected(self):
        missing = []
        for name in POISONED:
            rc, out = _compile(
                _probe_src(_call_expr(name)), poison=True, offload="device"
            )
            if rc == 0 or "poisoned identifier" not in out:
                missing.append(f"{name}: rc={rc}\n{out}")
        self.assertFalse(
            missing,
            "poison header did not reject:\n" + "\n".join(missing),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
