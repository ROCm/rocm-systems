// ARCH: gfx1250
//
// Lit fixture for VIMAGE TENSOR `tensor_store_from_lds_d4` —
// cross-target TDM-emulation path for the 4/5D Tensor Descriptor
// form, store direction. Completes the 2x2 encoding-coverage matrix:
//
//    form x direction    load                           store
//    _d2 (1-2D)          tensor_load_to_lds.ll          tensor_store_from_lds.ll
//    _d4 (3-5D)          tensor_load_to_lds_d4.ll       <this file>
//
// ENCODING (12 bytes, same layout as the _d4 load but with byte 1
// = 0x40 for STORE):
//   dword0 = D0714001
//   dword1 = 7C000000
//   dword2 = 10080428
//     bits 71-64 = 0x28 (vaddr0 = s40  — TD group 0, SReg_128)
//     bits 79-72 = 0x04 (vaddr1 = s4   — TD group 1, SReg_256)
//     bits 87-80 = 0x0C (vaddr2 = s12  — TD group 2, SReg_128) <-- _d4
//     bits 95-88 = 0x10 (vaddr3 = s16  — TD group 3, SReg_128) <-- _d4
//
// See `tensor_store_from_lds.hip` for the wire-up rationale and
// `tensor_load_to_lds_d4.hip` for the _d4 operand-count contract.
// Kernel must NEVER be dispatched — SGPR ranges carry garbage.

#include <hip/hip_runtime.h>

extern "C" __global__ void
tensor_store_from_lds_d4_kernel(int *out) {
  int tid = (int)threadIdx.x;
  asm volatile(
      ".long 0xd0714001\n\t"
      ".long 0x7c000000\n\t"
      ".long 0x10080428\n\t"
      :
      :
      : "s4",  "s5",  "s6",  "s7",
        "s8",  "s9",  "s10", "s11",
        "s12", "s13", "s14", "s15",
        "s16", "s17", "s18", "s19",
        "s40", "s41", "s42", "s43");
  out[tid] = tid;
}
