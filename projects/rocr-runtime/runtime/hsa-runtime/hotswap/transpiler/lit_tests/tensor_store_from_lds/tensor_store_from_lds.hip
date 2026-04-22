// ARCH: gfx1250
//
// Lit fixture for VIMAGE TENSOR `tensor_store_from_lds_d2` —
// cross-target TDM-emulation path. Companion to
// `tensor_load_to_lds.hip`; this one exercises the store direction
// (LDS -> global via Tensor Descriptor) in the _d2 form (up-to-2D
// tensor, groups 2/3 encoded as NULL sentinels).
//
// The load / store SemOps share the VIMAGE_TENSOR_Pseudo operand
// layout and the `marshalTDMArgs` helper; they only differ in (a)
// the SemOp (`TENSOR_LOAD_TO_LDS` vs `TENSOR_STORE_FROM_LDS`), (b)
// the emitted helper symbol (`salmon_tdm_load_to_lds` vs
// `salmon_tdm_store_from_lds`), and (c) one bit of the encoding
// (byte 1 = 0x40 for the store, 0x00 for the load — selects between
// opcode slots in the same VIMAGE `op=0x71` pseudo family). Zeroing
// groups 2/3 for the _d2 form is identical; so is the intrinsic
// signature parity with the (companion) same-target emit.
//
// ENCODING: 12 bytes.
//   dword0 = D0714001   (VIMAGE 0x34 @ 31-26, op 0x71 @ 21-14,
//                        dmask-like bit @ 14 selects STORE -> 0x40
//                        at byte 1)
//   dword1 = 7C000000   (NULL sentinel in vaddr4 per LLVM-23)
//   dword2 = 7C7C0428
//     bits 71-64 = 0x28 (vaddr0 = s40 — TD group 0)
//     bits 79-72 = 0x04 (vaddr1 = s4  — TD group 1)
//     bits 87-80 = 0x7C (vaddr2 = NULL — _d2 form)
//     bits 95-88 = 0x7C (vaddr3 = NULL — _d2 form)
//
// MAINTENANCE: kernel must NEVER be dispatched — only the static IR
// emission is exercised via `%raise_cli --emit-ir`. For a
// dispatchable end-to-end store fixture see
// `test_data/gfx1250/tdm_store_kernel.hip` + the parameterised
// `TdmGpu` gtest cases.

#include <hip/hip_runtime.h>

extern "C" __global__ void
tensor_store_from_lds_kernel(int *out) {
  int tid = (int)threadIdx.x;
  asm volatile(
      ".long 0xd0714001\n\t"
      ".long 0x7c000000\n\t"
      ".long 0x7c7c0428\n\t"
      :
      :
      : "s4",  "s5",  "s6",  "s7",
        "s8",  "s9",  "s10", "s11",
        "s40", "s41", "s42", "s43");
  out[tid] = tid;
}
