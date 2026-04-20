// ARCH: gfx1250
//
// Lit fixture for the FLAT scratch family (`scratch_load_*`,
// `scratch_store_*`, `scratch_atomic_*`) — produces a kernel that
// the AMDGPU backend lowers using `scratch_*` reals so the raiser
// can exercise the documented refusal in `handle_flat.cpp` (the
// `SIInstrFlags::FlatScratch` early-return branch).
//
// === Why this kernel must NEVER be dispatched on Salmon's lifted KD ===
//
// The kernel deliberately exercises per-thread scratch (private
// memory, `addrspace(5)` in IR). Lifting + re-codegen via the Salmon
// pipeline would discard the source kernel's KD and emit a fresh KD
// from the AMDGPU backend, but Salmon's compilation-target ABI
// deliberately does NOT request `flat_scratch_init` (see the
// regression guard in `lit_tests/buffer_store_no_scratch_alloca/`).
// Any `scratch_*` real in the lifted output would fault at run time
// because `FLAT_SCRATCH` is undefined on entry. The raiser therefore
// refuses the entire FLAT scratch family with a documented
// `unsupportedShape` failure.
//
// === Companion lit test ===
//
// `scratch_refusal.ll` (this directory) — pins the loud-failure
// contract: raise_cli must exit non-zero AND emit the architecturally
// specific diagnostic (FLAT_SCRATCH ABI gap, with a back-pointer to
// the `buffer_store_no_scratch_alloca` regression guard) rather than
// the generic "Unsupported instruction" surface that signalled the
// gap before the documented refusal landed. The diagnostic-text pin
// keeps the failure surface stable for kerneldex's coverage diff and
// for users who read raise_cli's stderr directly.
//
// === Producer pattern (inline asm) ===
//
// Inline asm is used rather than a dynamic-indexed `volatile int[N]`
// spill array because the HIP/AMDGPU frontend aggressively lowers
// volatile scratch accesses to `flat_store_b32` (generic-aperture,
// FLAT encoding) on gfx1250 — NOT to `scratch_store_b32` (SCRATCH
// segment-specific encoding, `SIInstrFlags::FlatScratch` set). Only
// inline asm reliably pins the SCRATCH encoding we need to exercise
// the raiser's FlatScratch-tsFlags-bit refusal branch.
//
// We emit BOTH a store and a load through the SCRATCH encoding so the
// refusal-by-tsFlags covers reads as well as writes — the
// `IsFlatScratch` TSFlags bit (SIDefines.h: bit 56) is set on every
// `FLAT_Scratch_*_Pseudo` regardless of access direction.

#include <hip/hip_runtime.h>

extern "C" __global__ void
scratch_refusal_kernel(int *out) {
  int tid = (int)threadIdx.x;
  int val = tid * 3;
  int loaded;
  // scratch_store_b32 off, %[val], off offset:0
  //   — SCRATCH-segment store, VADDR/SADDR=off, inline-literal offset.
  // scratch_load_b32 %[loaded], off, off offset:0
  //   — paired SCRATCH-segment load so the refusal covers both
  //     directions through a single lit fixture.
  asm volatile(
      "scratch_store_b32 off, %[val], off offset:0\n\t"
      "scratch_load_b32  %[loaded], off, off offset:0\n\t"
      : [loaded] "=v"(loaded)
      : [val]    "v"(val)
      : "memory");
  out[tid] = loaded;
}
