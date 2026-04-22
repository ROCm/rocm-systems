// Source for tdm_store_gfx1250.hsaco — dispatchable gfx1250 fixture
// that exercises the `tensor_store_from_lds` TDM emulation across
// arbitrary Tensor Descriptor dimensionality (1D through 5D).
//
// Store-direction companion of `tdm_load_kernel.hip`. Flow:
//
//   1. Each thread `tid` loads `in[tid]` from global and writes it
//      to `lds[tid]`. A `__syncthreads()` establishes the LDS
//      contents before the TDM call.
//   2. All lanes call `__builtin_amdgcn_tensor_store_from_lds` with
//      the host-provided D#. The lifted IR call resolves to
//      `salmon_tdm_store_from_lds`, which walks the descriptor and
//      stripes `buffer_store_b32` writes back out to the global
//      buffer the D#'s `global_addr` points at.
//   3. `s_wait_tensorcnt` fences the store.
//
// The host harness prepares `in[]`, lets the kernel pre-populate
// LDS from it (step 1), passes a D# whose `global_addr` points at a
// separate `out[]` buffer, then reads `out[]` back and compares to
// the expected layout. For a dense contiguous descriptor (the
// harness's default) out[i] must equal in[i] for every i in
// [0, tile_total).

#include <hip/hip_runtime.h>
#include <cstdint>

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

#define LDS_SLOTS 256

extern "C" __global__ __launch_bounds__(64) void
tdm_store_kernel(const v4i *__restrict__ g0_ptr,
                 const v8i *__restrict__ g1_ptr,
                 const v4i *__restrict__ g2_ptr,
                 const v4i *__restrict__ g3_ptr,
                 const uint32_t *__restrict__ in) {
  __shared__ uint32_t lds[LDS_SLOTS];
  int tid = threadIdx.x;

  // Stage the source array into LDS. The TDM emulation reads LDS
  // (from `lds_byte_addr=0`) and writes to the D#'s global target.
  //
  // The `volatile addrspace(3)*` cast is load-bearing:
  // `int_amdgcn_tensor_store_from_lds` is declared
  // `IntrInaccessibleMemOrArgMemOnly` in `IntrinsicsAMDGPU.td`, which
  // tells LLVM's alias analysis that the intrinsic never touches
  // addrspace(3) — only its vector arguments. With a plain
  // `lds[tid] = in[tid]`, the optimiser eliminates every LDS write
  // as dead (no alias-visible reader) and the emulation stores
  // garbage / zero to the global target.
  //
  // Writing through an explicit `volatile __attribute__((address_space(3)))`
  // pointer keeps the store in the DS pipe (no `src_shared_base` +
  // flat-store detour that a plain `__shared__ volatile` array would
  // trigger — that detour uses `s_mov_b64 sdst, src_shared_base`
  // which the salmon raiser does not lift today) AND defeats DCE
  // because volatile stores are preserved unconditionally. A more
  // targeted fix would be to widen the intrinsic's memory-effects
  // declaration upstream to include `MemoryEffects::shared`; for
  // the fixture, this cast is the minimal correct workaround.
  volatile __attribute__((address_space(3))) uint32_t *lds_v =
      (volatile __attribute__((address_space(3))) uint32_t *)lds;
  if (tid < LDS_SLOTS) lds_v[tid] = in[tid];
  __syncthreads();

  v4i g0 = *g0_ptr;
  v8i g1 = *g1_ptr;
  v4i g2 = *g2_ptr;
  v4i g3 = *g3_ptr;
  __builtin_amdgcn_tensor_store_from_lds(g0, g1, g2, g3, 0);
  __builtin_amdgcn_s_wait_tensorcnt(0);
}
