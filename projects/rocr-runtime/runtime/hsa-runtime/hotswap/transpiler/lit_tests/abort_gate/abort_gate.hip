// ARCH: gfx942
//
// Fixture for the SPE allow-list abort gate (lit_tests/abort_gate).
//
// The goal is to put an instruction into the decoded stream whose
// MCInstrDesc advertises an EXEC def but whose SemOp is NOT on the
// SPE-modelled allow-list in raiser.cpp (`isSPEModeledExecWriter`).
// `s_flbit_i32_b32` is a real, well-formed gfx942 instruction whose
// `S_FLBIT_I32_B32` SemOp is excluded from the allow-list (no
// handler in the SPE audit trail), and assembling it with an
// explicit `exec_lo` destination gives the EXEC-writer detector in
// raiser.cpp something concrete to scan for.
//
// When the raiser sees this instruction it must refuse to lower the
// kernel with:
//
//   transpiler: pre-translation abort: 's_flbit_i32_b32' writes EXEC
//   but its SemOp (N) is not in the SPE-modelled allow-list. ...
//
// The kernel itself is otherwise a no-op — we never read back the
// clobbered EXEC, so the dispatch would be meaningless. We only care
// that the raiser's pre-translation scan catches the unmodeled
// writer.
//
// ============================================================
// MAINTENANCE CONTRACT — read before changing the allow-list.
// ============================================================
// This fixture is valid only as long as `S_FLBIT_I32_B32` remains
// OFF the SPE allow-list in raiser.cpp. If you audit the
// S_FLBIT_I32_B32 handler path against SPE and add it to
// `isSPEModeledExecWriter` — a legitimate thing to do — this test
// stops being a negative test (the raise will succeed and the lit
// test will fail with "expected failure did not occur"). Do NOT
// delete the test. Instead, pick another EXEC-writing SemOp that
// is still off the allow-list and update this fixture's inline asm
// accordingly. Good candidates at the time of writing:
//
//   * `s_bitreplicate_b64_b32 exec, s0`  (SemOp::Unknown at time
//     of writing — semop.hpp does not list S_BITREPLICATE yet).
//   * `s_wqm_b64 exec, exec`             (SemOp::Unknown; WQM is
//     graphics-only and not modelled).
//   * any new SemOp whose handler has not been audited.
//
// Verify your replacement by inspecting `isSPEModeledExecWriter` in
// raiser.cpp and confirming the new SemOp is NOT on that switch.
// Also confirm the MCInstrDesc advertises an EXEC def so the
// `instructionWritesEXEC` scan in raiser.cpp flags it. Update the
// stderr CHECK strings in abort_gate.ll to match the new mnemonic.

#include <hip/hip_runtime.h>

extern "C" __global__ void unknown_exec_writer_kernel(int *out) {
  int lane = (int)threadIdx.x;
  asm volatile(
      // s_flbit_i32_b32 is an SOP1 scalar op that writes its 32-bit
      // dst. Routing the dst to exec_lo forces the raiser's EXEC
      // detector to flag the instruction; the SemOp it maps to
      // (`S_FLBIT_I32_B32`) is deliberately *not* on the SPE allow
      // list, so the allow-list abort fires. We use an immediate
      // source to keep the fixture hermetic — no kernarg / VGPR
      // dependency to confuse the detector.
      "s_flbit_i32_b32 exec_lo, 0x12345678\n\t"
      // Restore EXEC so any later code in the kernel sees a sane
      // mask. The restore itself (`s_mov_b64 exec, -1`) *is* on the
      // allow-list (`S_MOV_B64`), so it does not itself fire the
      // abort — only the s_flbit above does.
      "s_mov_b64 exec, -1\n\t"
      :
      :
      : "exec", "memory");
  out[lane] = lane;
}
