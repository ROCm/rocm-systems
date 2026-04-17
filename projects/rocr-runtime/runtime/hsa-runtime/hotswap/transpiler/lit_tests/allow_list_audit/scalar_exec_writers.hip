// ARCH: gfx942
//
// Fixture for the SPE allow-list audit suite
// (lit_tests/allow_list_audit). Exercises three distinct EXEC-writer
// SemOps in one kernel:
//
//   * V_CMPX                     — the lane-level predicate writer.
//   * S_AND_B64  → EXEC          — scalar bitwise write into EXEC.
//   * S_MOV_B64  → EXEC          — scalar reset of EXEC to all-ones.
//
// The lit test asserts that each of these routes through
// `regs.storeExec` (observable post-mem2reg as the subsequent
// `lshr i64 <new-exec>, %spe_lane_mod` consuming the updated EXEC
// SSA value rather than the initial `-1`).
//
// Duplicate of tests/divergent_exec.hip so the lit fixture is
// self-contained.
//
// Launches one wavefront of 64 threads. Each lane writes its own slot of
// `out[0..63]` using a single basic block that manually manipulates EXEC
// twice via `v_cmpx`. The intended per-lane semantic is:
//
//   lane  0..15:  out[lane] = 0xAA
//   lane 16..31:  out[lane] = 0xBB
//   lane 32..63:  out[lane] = 0   (from the pre-store)
//
// Native hardware executes this correctly because `v_cmpx` physically
// narrows the hardware EXEC mask. The current Salmon translator models
// EXEC as an alloca that is tracked in IR but NOT reconnected to the
// backend's hardware-EXEC management, so both `global_store_dword`s are
// emitted unmasked in the raised IR. After llc re-lowers to gfx942,
// every lane executes both stores and the final value (0xBB) wins
// everywhere.
//
// A correct translator (SPE) predicates each side-effectful op on the
// per-lane EXEC bit, so the backend re-inserts `v_cmpx` around the do-
// blocks and reproduces native hardware semantics.

#include <hip/hip_runtime.h>

extern "C" __global__ void divergent_exec_kernel(int *out) {
  int lane = (int)threadIdx.x;
  int *addr = out + lane;
  int valDef = 0;
  int valA = 0xAA;
  int valB = 0xBB;

  // One basic block, two v_cmpx-gated global_store_dwords with EXEC
  // restores between them. `volatile` + `"exec"` / `"memory"` clobbers
  // prevent the compiler from reordering, hoisting, or deleting any of
  // the ops across the asm boundary.
  asm volatile(
      // Pre-store: everyone writes 0 under full EXEC.
      "global_store_dword %[addr], %[valDef], off\n\t"
      "s_waitcnt vmcnt(0)\n\t"
      // Divergent region 1: lanes 0..15 write valA.
      "v_cmpx_lt_u32_e64 exec, %[lane], 16\n\t"
      "global_store_dword %[addr], %[valA], off\n\t"
      "s_waitcnt vmcnt(0)\n\t"
      "s_mov_b64 exec, -1\n\t"
      // Divergent region 2: lanes 16..31 write valB.
      "v_cmpx_ge_u32_e64 exec, %[lane], 16\n\t"
      "v_cmp_lt_u32_e64 s[4:5], %[lane], 32\n\t"
      "s_and_b64 exec, exec, s[4:5]\n\t"
      "global_store_dword %[addr], %[valB], off\n\t"
      "s_waitcnt vmcnt(0)\n\t"
      "s_mov_b64 exec, -1\n\t"
      :
      : [addr] "v"(addr), [lane] "v"(lane), [valDef] "v"(valDef),
        [valA] "v"(valA), [valB] "v"(valB)
      : "exec", "s4", "s5", "memory");
}
