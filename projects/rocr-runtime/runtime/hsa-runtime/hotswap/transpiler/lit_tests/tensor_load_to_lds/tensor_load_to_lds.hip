// ARCH: gfx1250
//
// Lit fixture for VIMAGE TENSOR `tensor_load_to_lds_d2` — cross-target
// TDM-emulation path. See SemOp::TENSOR_LOAD_TO_LDS and SemOp::
// TENSOR_STORE_FROM_LDS in semop.hpp, the opcode_map.cpp
// canonicalization entries, and the handleVIMAGE TDM emulation
// contract in transpiler/handle_vimage.cpp.
//
// The gfx1250 TENSOR cnt unit is the only hardware that can execute
// these tensor-descriptor memory ops (`MIMGInstructions.td:2049-2113`,
// `let SubtargetPredicate = isGFX125xOnly`). The cross-target lift
// on gfx942 emits a call to the link-merged HIP-authored runtime
// helper `salmon_tdm_load_to_lds` (`runtime/tdm.hip`, declared via
// `tdm_runtime.{hpp,cpp}`); the companion fixture
// `tensor_load_to_lds_same_target.ll` pins the gfx1250 -> gfx1250
// same-target intrinsic emit, the sibling `_d4.ll` fixture covers
// the _d4 form, and the `tensor_store_from_lds*` fixtures cover the
// store direction.
//
// ENCODING: We emit the 12-byte VIMAGE TENSOR encoding by hand using
// `.long` directives rather than the `tensor_load_to_lds` mnemonic
// or the `__builtin_amdgcn_tensor_load_to_lds` clang builtin. The
// host's ROCm 7.2.1 assembler and the raise_cli's bundled LLVM
// disassembler disagree on where the NULL SGPR sentinel `0x7C` sits
// in the 12-byte encoding (byte 4 = `vdata` vs. byte 7 = `vaddr4`).
// Hand-encoding side-steps the drift; the produced bytes
// `D0710001 7C000000 7C7C0428` exactly mirror the encoding emitted
// by Triton-generated kernels in the kerneldex corpus that
// originally surfaced this gap (e.g.
// `triton__f16_gemm_8warp_basic_gemm_tdm_pipelined_kernel`).
//
// MAINTENANCE: this fixture intentionally produces a
// `tensor_load_to_lds_d2` to a hand-zeroed descriptor. The kernel
// must NEVER be dispatched — the SGPR ranges referenced in the
// encoding are never assigned a valid Tensor Descriptor at runtime.
// Only the raiser's static IR emission is exercised (via `%raise_cli
// --emit-ir`). For a dispatchable end-to-end TDM fixture see
// `test_data/gfx1250/tdm_smoke_kernel.hip` + `TdmGpu.CrossTargetCorpus`.

#include <hip/hip_runtime.h>

extern "C" __global__ void
tensor_load_to_lds_kernel(int *out) {
  int tid = (int)threadIdx.x;
  // Hand-encoded `tensor_load_to_lds s[40:43], s[4:11]` (d2 form).
  // 12 bytes; layout per VIMAGE_VSAMPLE_Common in
  // SIInstrFormats.td:457 + VIMAGEe in SIInstrFormats.td:501:
  //   dword0 = D0710001
  //     bits 31-26 = 0x34 (VIMAGE encoding family)
  //     bits 21-14 = 0x71 (TENSOR_LOAD_TO_LDS_d2 op)
  //   dword1 = 7C000000
  //     bits 39-32 = 0x7C (vdata = NULL — required for the
  //                        no-vdata tensor encoding)
  //   dword2 = 7C7C0428
  //     bits 71-64 = 0x28 (vaddr0 = s40 — TD pointer dword 0)
  //     bits 79-72 = 0x04 (vaddr1 = s4  — TD pointer dword 1 …)
  //     bits 87-80 = 0x7C (vaddr2 = NULL, d2 form omits)
  //     bits 95-88 = 0x7C (vaddr3 = NULL, d2 form omits)
  asm volatile(
      ".long 0xd0710001\n\t"
      ".long 0x7c000000\n\t"
      ".long 0x7c7c0428\n\t"
      :
      :
      : "s4",  "s5",  "s6",  "s7",
        "s8",  "s9",  "s10", "s11",
        "s40", "s41", "s42", "s43");
  out[tid] = tid;
}
