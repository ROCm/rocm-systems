// ARCH: gfx1250
//
// Lit fixture for VIMAGE TENSOR `tensor_load_to_lds_d4` — cross-target
// TDM-emulation path for the 4/5D Tensor Descriptor form. Companion
// to `tensor_load_to_lds.hip` (the _d2 form, groups 2/3 encoded as
// NULL sentinels) — this one pins the _d4 form where all four SGPR
// ranges carry real Tensor Descriptor groups.
//
// The _d2 / _d4 distinction is structural: both forms share the same
// opcode (0x71 in VIMAGE), same SemOp (`TENSOR_LOAD_TO_LDS`), and
// same LLVM intrinsic (`llvm.amdgcn.tensor.load.to.lds`). The handler
// recovers the form from `op.nSrcs()` (4 for _d2 after r128/cpol,
// 6 for _d4). For _d4, groups 2 (SReg_128) and 3 (SReg_128) must be
// marshalled from real SGPR ranges; zeroing them would change the
// observable 3D/4D/5D walk and silently miscompile kernels using the
// higher dims.
//
// ENCODING: 12-byte VIMAGE encoding, same hand-encoded layout as the
// _d2 fixture. The third dword packs vaddr0..vaddr3:
//
//   dword0 = D0710001   (VIMAGE encoding class 0x34 at bits 31-26,
//                        op 0x71 at bits 21-14, same as _d2)
//   dword1 = 7C000000   (byte 7 = 0x7C NULL sentinel in `vaddr4`
//                        slot per LLVM-23's VIMAGE format — the
//                        raise_cli bundled disassembler looks here;
//                        the ROCm 7.2.1 assembler puts the sentinel
//                        in byte 4 instead, so we hand-encode)
//   dword2 = 10080428
//     bits 71-64 = 0x28 (vaddr0 = s40  — TD group 0, SReg_128)
//     bits 79-72 = 0x04 (vaddr1 = s4   — TD group 1, SReg_256)
//     bits 87-80 = 0x0C (vaddr2 = s12  — TD group 2, SReg_128) <-- _d4
//     bits 95-88 = 0x10 (vaddr3 = s16  — TD group 3, SReg_128) <-- _d4
//
// Contrast with the _d2 encoding where bytes 10 and 11 are both
// `0x7C` (NULL sentinel) — that's how `OpcodeMap::canonicalize`
// recovers the form after the initial pseudo collapse.
//
// MAINTENANCE: the kernel must NEVER be dispatched — the SGPR
// ranges s[4:11], s[12:15], s[16:19], s[40:43] carry garbage at
// runtime. Only the raiser's static IR emission is exercised via
// `%raise_cli --emit-ir`.

#include <hip/hip_runtime.h>

extern "C" __global__ void
tensor_load_to_lds_d4_kernel(int *out) {
  int tid = (int)threadIdx.x;
  asm volatile(
      ".long 0xd0710001\n\t"
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
