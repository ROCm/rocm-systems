// ARCH: gfx1250
//
// Lit fixture for VIMAGE TENSOR `tensor_load_to_lds_d2` — cross-target
// loud refusal path. See SemOp::TENSOR_LOAD_TO_LDS and SemOp::
// TENSOR_STORE_FROM_LDS in semop.hpp, the opcode_map.cpp
// canonicalization entries, and the handleVIMAGE refusal contract
// in transpiler/handle_vimage.cpp.
//
// The gfx1250 TENSOR cnt unit is the only hardware that can execute
// these tensor-descriptor memory ops (`MIMGInstructions.td:2049-2113`,
// `let SubtargetPredicate = isGFX125xOnly`). On gfx942 and earlier
// generations there is NO equivalent instruction or instruction
// sequence; the per-dim addressing semantics of a Tensor Descriptor
// cannot be reconstructed from gfx9xx primitives without
// fundamentally altering the kernel's memory-access pattern. The
// matching LLVM intrinsics (`int_amdgcn_tensor_load_to_lds`,
// `int_amdgcn_tensor_store_from_lds`, IntrinsicsAMDGPU.td:4213) are
// themselves gated `isGFX125xOnly`, so even a same-shape intrinsic
// emit would fail at codegen on a non-gfx1250 target.
//
// Per the user-rules (no silent fallbacks), the handler refuses the
// cross-target lift via `RaiseFailure::unsupportedShape` with a
// precise diagnostic that names the offending mnemonic, the
// architectural mismatch, and the canonical same-target intrinsic.
// This fixture pins:
//
//   1. `%not` inversion of the exit code — the test passes only
//      when raise_cli actually fails.
//   2. The stderr diagnostic includes the offending mnemonic
//      (`tensor_load_to_lds`) and the encoding format
//      (`VIMAGE`). raise_cli's failure-line format is fixed
//      (raise_cli.cpp:213): `kernel '<name>' failed to raise:
//      <mnemonic> [<format>]`.
//
// ENCODING: We emit the 12-byte VIMAGE TENSOR encoding by hand using
// `.long` directives rather than the `tensor_load_to_lds` mnemonic
// or the `__builtin_amdgcn_tensor_load_to_lds` clang builtin. The
// host's ROCm 7.2.1 assembler emits `vdata=s0` (0x00) into bits
// 39:32 for `tensor_load_to_lds` (which has no vdata output), but
// the canonical encoding the corpus uses — and the encoding the
// raise_cli's bundled LLVM disassembler accepts — places the NULL
// SGPR sentinel `0x7C` in the vdata slot. Hand-encoding side-steps
// the assembler bug; the produced bytes
// `D0710001 7C000000 7C7C0428` exactly mirror the encoding emitted
// by Triton-generated kernels in the kerneldex corpus that
// originally surfaced this gap (e.g.
// `triton__f16_gemm_8warp_basic_gemm_tdm_pipelined_kernel`).
//
// MAINTENANCE: this fixture intentionally produces a
// `tensor_load_to_lds_d2` to a hand-zeroed descriptor. The kernel
// must NEVER be dispatched — only the raiser pre-translation
// refusal is exercised.

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
