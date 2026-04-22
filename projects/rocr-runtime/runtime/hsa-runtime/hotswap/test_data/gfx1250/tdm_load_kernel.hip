// Source for tdm_load_gfx1250.hsaco — dispatchable gfx1250 fixture that
// exercises the `tensor_load_to_lds` TDM emulation across arbitrary
// Tensor Descriptor dimensionality (1D through 5D).
//
// Unlike `tdm_smoke_kernel.hip` (1D-only, groups 2/3 encoded as zero
// VGPRs — reaches the helper with all-zero groups 2/3 and triggers
// the zero-bound walker contract), this kernel reads all four D#
// groups from host-provided pointers so the test harness can build
// any D# shape the walker supports:
//
//   * 1D: tensor_dim0 = T0, tensor_dim1..4 = 0, tile_dim0 = T0,
//         tile_dim1..4 = 0.
//   * 2D: tensor_dim{0,1} = T0,T1 ; tile_dim{0,1} = T0,T1 ;
//         tensor_dim0_stride = T0 (contiguous rows).
//   * 3D..5D: layered on the 2D case via g1/g2/g3 extra dims and
//         cumulative strides (see `TdmGpu.DescriptorCoverage` in
//         `tests/tdm_gpu_test.cpp` for the exact host build).
//
// The copy is always "dense contiguous tensor in, identical layout
// into LDS, identical layout back to `out`" — LDS slot i receives
// global element i, and thread `tid` copies `lds[tid]` to
// `out[tid]`. With `tile_total = T0*T1*T2*T3*T4 <= 64` that covers
// the whole copy in a single 64-lane wavefront.
//
// The kernel reads the D# groups into SGPRs via normal s_load_b128/
// s_load_b256 sequences; `__builtin_amdgcn_tensor_load_to_lds` in
// ROCm 7.2.1 maps to the `_d4` form (4 SReg operands). The lifted
// IR call that reaches `salmon_tdm_load_to_lds` carries whatever
// bit pattern the host put in `*g{2,3}_ptr`, which is how the test
// drives the walker's higher-dim branches.
//
// See `tdm_smoke_kernel.hip` for the hipcc + byte-patch regen recipe
// — same steps, just replace the filename. Kernel symbol is
// `tdm_load_kernel` (visible via `nm -D tdm_load_gfx1250.hsaco`).

#include <hip/hip_runtime.h>
#include <cstdint>

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

// 256 slots = max tile_total the TdmGpu.DescriptorCoverage driver
// uses. The D#.lds_byte_addr is always 0 so the helper writes from
// byte 0; the compiler places `lds` at offset 0 by default and the
// trailing `out[tid] = lds[tid]` reads from the same base.
#define LDS_SLOTS 256

extern "C" __global__ __launch_bounds__(64) void
tdm_load_kernel(const v4i *__restrict__ g0_ptr,
                const v8i *__restrict__ g1_ptr,
                const v4i *__restrict__ g2_ptr,
                const v4i *__restrict__ g3_ptr,
                uint32_t *__restrict__ out) {
  __shared__ uint32_t lds[LDS_SLOTS];
  int tid = threadIdx.x;

  // Poison LDS before the TDM copy so a silently-skipped emulation
  // path shows 0xdeadbeef in `out` rather than whatever stale LDS
  // happened to contain (distinguishable from the hardware-OOB
  // zero-fill that the walker produces on bounds violation).
  if (tid < LDS_SLOTS) lds[tid] = 0xdeadbeefu;
  __syncthreads();

  v4i g0 = *g0_ptr;
  v8i g1 = *g1_ptr;
  v4i g2 = *g2_ptr;
  v4i g3 = *g3_ptr;
  __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
  __builtin_amdgcn_s_wait_tensorcnt(0);
  __syncthreads();

  out[tid] = lds[tid];
}
