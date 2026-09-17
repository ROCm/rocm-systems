"""Tests for tracelogging_curated_codegen.py — the TraceLogging curated-emit
generator (backend swap of lttng_curated_codegen.py).

These assert the TraceLogging-specific invariants that the port's adversarial
review flagged as blocking, using the libclang-independent --sigs fixture:

  * Distinct enter/exit event names ("<api>_enter" / "<api>_exit"), never a
    single combined event name (LTTng forbids duplicate event names).
  * No duplicate event name within a provider.
  * Distinct provider name from the classic provider (rocm_hip_tlg, not
    rocm_hip; rocm_hsa_tlg, not rocm_hsa).
  * Emit helpers are ordinary non-inline functions (no `static`/`inline` on the
    generated definitions).
  * Correct type-macro mapping (handle/ptr -> TraceLoggingHexUInt64, etc).
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
LTTNG_TESTDATA = os.path.abspath(
    os.path.join(HERE, '..', '..', 'lttng', 'scripts', 'testdata'))
GEN = os.path.join(HERE, 'tracelogging_curated_codegen.py')


def _run(provider, extra):
    """Generate emit.h + emit.cpp from the minimal fixture and return their
    text. `extra` is appended to the arg list."""
    with tempfile.TemporaryDirectory() as td:
        emit_h = os.path.join(td, 'rocm_trace_emit_curated.h')
        emit_cpp = os.path.join(td, 'rocm_trace_emit_curated.cpp')
        cmd = [sys.executable, GEN, '--provider', provider,
               '--yaml', os.path.join(LTTNG_TESTDATA, 'curated_minimal.yaml'),
               '--sigs', os.path.join(LTTNG_TESTDATA, 'curated_minimal_sigs.json'),
               '--emit-out', emit_h, '--emit-cpp-out', emit_cpp] + extra
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"codegen failed: {r.stderr}"
        with open(emit_h) as f:
            h = f.read()
        with open(emit_cpp) as f:
            c = f.read()
        return h, c


class TestTraceLoggingCodegen(unittest.TestCase):
    def test_distinct_enter_exit_event_names(self):
        _h, c = _run('hip', [])
        # Both distinct event names present; NO combined "<api>" (bare) event.
        self.assertIn('"hipMemcpyAsync_enter"', c)
        self.assertIn('"hipMemcpyAsync_exit"', c)
        self.assertNotIn('"hipMemcpyAsync"', c)

    def test_no_duplicate_event_names_within_provider(self):
        for provider in ('hip', 'hsa'):
            _h, c = _run(provider, [])
            names = re.findall(r'TraceLoggingWrite\(\w+, "([^"]+)"', c)
            self.assertEqual(sorted(names), sorted(set(names)),
                             f"{provider}: duplicate event name in provider")

    def test_distinct_provider_name_from_classic(self):
        _h, c = _run('hip', [])
        self.assertIn('TRACELOGGING_DEFINE_PROVIDER', c)
        self.assertIn('rocm_hip_tlg', c)
        # Provider name string must be the _tlg name, never bare "rocm_hip".
        self.assertIn('"rocm_hip_tlg"', c)
        self.assertNotIn('"rocm_hip"', c)
        _h, c = _run('hsa', [])
        self.assertIn('"rocm_hsa_tlg"', c)
        self.assertNotIn('"rocm_hsa"', c)

    def test_emit_helpers_are_non_inline(self):
        _h, c = _run('hip', [])
        # Every emit-helper definition begins at column 0 as `void rocm_...`,
        # with no leading static/inline. Assert neither qualifier decorates an
        # emit helper.
        for m in re.finditer(r'^(\w[\w ]*?)\brocm_trace_emit_\w+_(?:enter|exit)\(',
                             c, re.M):
            prefix = m.group(1)
            self.assertNotIn('static', prefix)
            self.assertNotIn('inline', prefix)

    def test_type_macro_mapping(self):
        _h, c = _run('hip', [])
        # ptr -> HexUInt64, size -> UInt64, enum -> Int32, handle -> HexUInt64
        self.assertIn('TraceLoggingHexUInt64((uint64_t)(uintptr_t)(dst), "dst")', c)
        self.assertIn('TraceLoggingUInt64((uint64_t)(sizeBytes), "sizeBytes")', c)
        self.assertIn('TraceLoggingInt32((int32_t)(kind), "kind")', c)
        # OUT ptr deref (hipMalloc) success-gated on hipSuccess.
        self.assertIn('status == hipSuccess', c)

    def test_register_unregister_lifecycle_emitted(self):
        _h, c = _run('hip', [])
        self.assertIn('rocm_hip_tlg_register', c)
        self.assertIn('rocm_hip_tlg_unregister', c)
        self.assertIn('TraceLoggingRegister(rocm_hip_tlg)', c)
        self.assertIn('TraceLoggingUnregister(rocm_hip_tlg)', c)
        # Declarations exposed in the header for the runtime lifecycle callers.
        self.assertIn('void rocm_hip_tlg_register(void);', _h)
        self.assertIn('void rocm_hip_tlg_unregister(void);', _h)

    def test_check_mode_roundtrips(self):
        # Generating then re-running with --check against the same output must
        # report no drift (rc 0).
        with tempfile.TemporaryDirectory() as td:
            emit_h = os.path.join(td, 'rocm_trace_emit_curated.h')
            emit_cpp = os.path.join(td, 'rocm_trace_emit_curated.cpp')
            base = [sys.executable, GEN, '--provider', 'hsa',
                    '--yaml', os.path.join(LTTNG_TESTDATA, 'curated_minimal.yaml'),
                    '--sigs', os.path.join(LTTNG_TESTDATA, 'curated_minimal_sigs.json'),
                    '--emit-out', emit_h, '--emit-cpp-out', emit_cpp]
            self.assertEqual(subprocess.run(base, capture_output=True).returncode, 0)
            chk = subprocess.run(base + ['--check'], capture_output=True, text=True)
            self.assertEqual(chk.returncode, 0, chk.stderr)


if __name__ == '__main__':
    unittest.main()
