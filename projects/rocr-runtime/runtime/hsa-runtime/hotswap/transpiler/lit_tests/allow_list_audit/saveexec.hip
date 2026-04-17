// ARCH: gfx942
//
// Fixture for the SPE allow-list audit
// (lit_tests/allow_list_audit/saveexec.ll). Exercises the
// `S_AND_SAVEEXEC_B64` SemOp — the one EXEC-writer family that is
// BOTH an EXEC writer AND an SGPR-pair producer (EXEC ← EXEC & src,
// dst ← old EXEC). The handler in handle_sop1.cpp is the only place
// this pattern is implemented, so a dedicated audit is warranted.
//
// The lit test asserts that after `s_and_saveexec_b64`:
//   * EXEC has been updated through storeExec (subsequent
//     `lshr i64 <new-exec>, ...` in the SPE diamond consumes the
//     post-saveexec SSA value).
//   * The dst sgpr pair captures the OLD EXEC (observable as an SSA
//     value that is the kernel's initial `-1`, not the post-saveexec
//     value).

#include <hip/hip_runtime.h>

extern "C" __global__ void saveexec_kernel(int *out) {
  int lane = (int)threadIdx.x;
  int *addr = out + lane;
  int val = 0xCC;

  asm volatile(
      // Compute a narrow mask in s[4:5] (lanes 0..15) using a
      // normal v_cmp (NOT v_cmpx — we only want the mask, not an
      // EXEC write here).
      "v_cmp_lt_u32_e64 s[4:5], %[lane], 16\n\t"
      // S_AND_SAVEEXEC_B64: saves old EXEC into s[6:7], sets EXEC
      // to EXEC & s[4:5]. This is the allow-listed writer under
      // test.
      "s_and_saveexec_b64 s[6:7], s[4:5]\n\t"
      // One side-effectful op under narrowed EXEC to pin down an
      // observable lshr-of-new-exec in the raised IR.
      "global_store_dword %[addr], %[val], off\n\t"
      "s_waitcnt vmcnt(0)\n\t"
      // Restore EXEC from the saved copy.
      "s_mov_b64 exec, s[6:7]\n\t"
      :
      : [addr] "v"(addr), [lane] "v"(lane), [val] "v"(val)
      : "exec", "s4", "s5", "s6", "s7", "memory");
}
