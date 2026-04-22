//===- tdm.hip - Tensor Data Mover emulation runtime (gfx942) -------------===//
//
// Cross-target lift of gfx1250 TENSOR_LOAD_TO_LDS / TENSOR_STORE_FROM_LDS
// onto gfx942. Two C-linkage entry points are emitted by the bitcode and
// link-merged into the kernel module by `linkTDMRuntime` (see
// `tdm_runtime.cpp` and `handle_vimage.cpp`); their signatures mirror the
// gfx1250 intrinsic operand bank.
//
// Strategy:
//   * X-axis OOB (and multi-dim row OOB folded onto num_records=0) is
//     delegated to gfx942's MUBUF unit by routing global memory through
//     `__builtin_amdgcn_raw_buffer_{load,store}_b{8,16,32,64}` against a
//     V# constructed once per row.
//   * `data_size` (1/2/4/8) is dispatched once per call in the entry
//     point; the rest of the walk is templated on `<bool IsLoad,
//     unsigned DS>` so per-element switching disappears entirely.
//   * Lane L of the hardware wave handles X = L, W, 2W, ... where W is
//     `__builtin_amdgcn_wavefrontsize()`. Unique X per lane, so the
//     stripe is race-free on both the global and LDS sides.
//   * Atomic-barrier emit is gated to lane 0.
//
// Field offsets follow the MI450 SPG Tensor DMA Resource Descriptor
// tables (79–84).
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef unsigned int v2u __attribute__((ext_vector_type(2)));

// Bits [lo+n-1 : lo] of `w`.
__device__ __forceinline__ uint32_t bits(uint32_t w, unsigned lo, unsigned n) {
  return (w >> lo) & ((n == 32) ? ~0u : ((1u << n) - 1u));
}

// Element-wise readfirstlane on the descriptor vectors.
//
// TDM is per-wave: every lane carries the same D# (the source ISA loaded
// it from SGPR-sourced operands). But the HIP / amdgpu_func ABI passes
// `<N x i32>` args in VGPRs, so without this hint clang treats every
// descriptor-derived value as VGPR-typed. That cascades into the inner
// X loop: the V# (which `buffer_load`'s SRSRC requires in SGPRs) goes
// through a `readfirstlane + cmp + s_and_saveexec + cbranch_execnz`
// uniformity-check loop wrapped around every buffer op, and the row-
// base stride math (`y*S0 + z*S1 + ...`) is emitted as `v_mad_u64_u32`
// instead of scalar `s_mul_*`.
//
// Forcing the descriptor through `readfirstlane` at walk entry promises
// the optimiser that all four vectors are uniform; everything derived
// (V#, tile/tensor dims, strides, padding, lds_byte_off, ...) then
// stays in SGPRs, and the X loop collapses to one buffer op + one DS
// op per iteration.
__device__ __forceinline__ v4i rfl_v4(v4i v) {
  v4i r;
  r[0] = __builtin_amdgcn_readfirstlane((int)v[0]);
  r[1] = __builtin_amdgcn_readfirstlane((int)v[1]);
  r[2] = __builtin_amdgcn_readfirstlane((int)v[2]);
  r[3] = __builtin_amdgcn_readfirstlane((int)v[3]);
  return r;
}
__device__ __forceinline__ v8i rfl_v8(v8i v) {
  v8i r;
  r[0] = __builtin_amdgcn_readfirstlane((int)v[0]);
  r[1] = __builtin_amdgcn_readfirstlane((int)v[1]);
  r[2] = __builtin_amdgcn_readfirstlane((int)v[2]);
  r[3] = __builtin_amdgcn_readfirstlane((int)v[3]);
  r[4] = __builtin_amdgcn_readfirstlane((int)v[4]);
  r[5] = __builtin_amdgcn_readfirstlane((int)v[5]);
  r[6] = __builtin_amdgcn_readfirstlane((int)v[6]);
  r[7] = __builtin_amdgcn_readfirstlane((int)v[7]);
  return r;
}

// D# field decoders. Free functions taking the relevant vector arg by
// value so the codegen keeps the descriptor in VGPRs throughout the
// walk. (An earlier revision wrapped the four vectors in a `Desc`
// struct with member accessors; the resulting alloca survived SROA
// and spilled the whole 80-byte descriptor to scratch — see git
// history.) Each accessor reads only what it needs; unused fields
// DCE.

// ---- Group 0 (4 dwords) -----------------------------------------------
__device__ __forceinline__ uint32_t count(v4i g0)               { return bits(g0[0], 0, 2); }
__device__ __forceinline__ bool     gather_mode(v4i g0)         { return bits(g0[0], 31, 1); }
__device__ __forceinline__ bool     gather_index_size32(v4i g0) { return bits(g0[0], 30, 1); }
__device__ __forceinline__ uint32_t lds_addr(v4i g0)            { return (uint32_t)g0[1]; } // bytes
__device__ __forceinline__ uint64_t global_addr(v4i g0) {
  return (uint64_t)(uint32_t)g0[2] | ((uint64_t)bits(g0[3], 0, 25) << 32);
}

// ---- Group 1 (8 dwords) -----------------------------------------------
__device__ __forceinline__ uint32_t data_size_log2(v8i g1)      { return bits(g1[0], 16, 2); }
__device__ __forceinline__ bool     atomic_barrier_en(v8i g1)   { return bits(g1[0], 18, 1); }
__device__ __forceinline__ bool     iterate_en(v8i g1)          { return bits(g1[0], 19, 1); }
__device__ __forceinline__ bool     pad_en(v8i g1)              { return bits(g1[0], 20, 1); }
__device__ __forceinline__ uint32_t pad_interval(v8i g1)        { return bits(g1[0], 22, 3); }
__device__ __forceinline__ uint32_t pad_amount(v8i g1)          { return bits(g1[0], 25, 7); }
__device__ __forceinline__ uint32_t atomic_barrier_addr(v8i g1) { return bits(g1[1], 0, 16); }
__device__ __forceinline__ uint32_t tensor_dim0(v8i g1)         { return ((uint32_t)g1[1] >> 16) | ((uint32_t)g1[2] << 16); }
__device__ __forceinline__ uint32_t tensor_dim1(v8i g1)         { return ((uint32_t)g1[2] >> 16) | ((uint32_t)g1[3] << 16); }
__device__ __forceinline__ uint32_t tile_dim0(v8i g1)           { return (uint32_t)g1[3] >> 16; }
__device__ __forceinline__ uint32_t tile_dim1(v8i g1)           { return bits(g1[4], 0, 16); }
__device__ __forceinline__ uint32_t tile_dim2(v8i g1)           { return (uint32_t)g1[4] >> 16; }
__device__ __forceinline__ uint64_t tensor_dim0_stride(v8i g1) {
  return (uint64_t)(uint32_t)g1[5] | ((uint64_t)bits(g1[6], 0, 16) << 32);
}
__device__ __forceinline__ uint64_t tensor_dim1_stride(v8i g1) {
  return ((uint64_t)(uint32_t)g1[6] >> 16) | ((uint64_t)(uint32_t)g1[7] << 16);
}

// ---- Group 2 (default — !iterate_en, !gather_mode) -------------------
__device__ __forceinline__ uint32_t tensor_dim2(v4i g2)         { return (uint32_t)g2[0]; }
__device__ __forceinline__ uint32_t tensor_dim3(v4i g2)         { return (uint32_t)g2[1]; }
__device__ __forceinline__ uint64_t tensor_dim2_stride(v4i g2) {
  return (uint64_t)(uint32_t)g2[2] | ((uint64_t)bits(g2[3], 0, 16) << 32);
}
__device__ __forceinline__ uint32_t tile_dim3(v4i g2)           { return (uint32_t)g2[3] >> 16; }

// ---- Group 2 (iterate_en) overlay ------------------------------------
__device__ __forceinline__ uint32_t lds_addr_increment(v4i g2)  { return (uint32_t)g2[1]; }
__device__ __forceinline__ uint64_t global_addr_increment(v4i g2) {
  return (uint64_t)(uint32_t)g2[2] | ((uint64_t)bits(g2[3], 0, 16) << 32);
}
__device__ __forceinline__ uint32_t iterate_count(v4i g2)       { return (uint32_t)g2[3] >> 16; }

// ---- Group 3 (default) -----------------------------------------------
__device__ __forceinline__ uint64_t tensor_dim3_stride(v4i g3) {
  return (uint64_t)(uint32_t)g3[0] | ((uint64_t)bits(g3[1], 0, 16) << 32);
}
__device__ __forceinline__ uint32_t tensor_dim4(v4i g3)         { return ((uint32_t)g3[1] >> 16) | ((uint32_t)g3[2] << 16); }
__device__ __forceinline__ uint32_t tile_dim4(v4i g3)           { return (uint32_t)g3[2] >> 16; }

// ---- Gather-mode overlay on g2/g3 ------------------------------------
// 16-bit indices: 8 in g2 + 8 in g3, two per dword.
// 32-bit indices: 4 in g2 + 4 in g3, one per dword.
__device__ __forceinline__ uint32_t gather_index(v4i g2, v4i g3,
                                                  unsigned i, bool size32) {
  if (size32) {
    v4i grp = (i < 4) ? g2 : g3;
    return (uint32_t)grp[i & 3];
  }
  v4i grp = (i < 8) ? g2 : g3;
  uint32_t w = (uint32_t)grp[(i & 7) >> 1];
  return (i & 1) ? (w >> 16) : (w & 0xffffu);
}

// V# for raw byte addressing with hardware bounds check.
// flags=0x00020000 = data_format=BUF_DATA_FORMAT_32, num_format=UINT,
// type=BUFFER (the canonical "raw byte access" gfx9 layout). With
// stride=0, num_records is in BYTES and `voffset >= num_records` triggers
// the OOB rule: loads return 0, stores are dropped.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_rsrc(uint64_t base,
                                                            uint32_t nrec_bytes) {
  return __builtin_amdgcn_make_buffer_rsrc(
      (void *)(uintptr_t)base, /*stride=*/(short)0,
      /*num_records=*/(int)nrec_bytes, /*flags=*/(int)0x00020000);
}

// Map data-size byte count to the matching scalar LDS element type.
template <unsigned DS> struct LdsT;
template <> struct LdsT<1> { using type = uint8_t;  };
template <> struct LdsT<2> { using type = uint16_t; };
template <> struct LdsT<4> { using type = uint32_t; };
template <> struct LdsT<8> { using type = uint64_t; };

// Typed buffer-load wrappers. The `_b64` builtin returns/takes <2 x i32>;
// `__builtin_bit_cast` reinterprets it as uint64_t with no IR ops.
__device__ __forceinline__ uint8_t  buf_load_t(uint8_t *,  __amdgpu_buffer_rsrc_t r, uint32_t v) { return __builtin_amdgcn_raw_buffer_load_b8 (r, v, 0u, 0); }
__device__ __forceinline__ uint16_t buf_load_t(uint16_t *, __amdgpu_buffer_rsrc_t r, uint32_t v) { return __builtin_amdgcn_raw_buffer_load_b16(r, v, 0u, 0); }
__device__ __forceinline__ uint32_t buf_load_t(uint32_t *, __amdgpu_buffer_rsrc_t r, uint32_t v) { return __builtin_amdgcn_raw_buffer_load_b32(r, v, 0u, 0); }
__device__ __forceinline__ uint64_t buf_load_t(uint64_t *, __amdgpu_buffer_rsrc_t r, uint32_t v) {
  return __builtin_bit_cast(uint64_t, __builtin_amdgcn_raw_buffer_load_b64(r, v, 0u, 0));
}
__device__ __forceinline__ void buf_store_t(uint8_t  x, __amdgpu_buffer_rsrc_t r, uint32_t v) { __builtin_amdgcn_raw_buffer_store_b8 (x, r, v, 0u, 0); }
__device__ __forceinline__ void buf_store_t(uint16_t x, __amdgpu_buffer_rsrc_t r, uint32_t v) { __builtin_amdgcn_raw_buffer_store_b16(x, r, v, 0u, 0); }
__device__ __forceinline__ void buf_store_t(uint32_t x, __amdgpu_buffer_rsrc_t r, uint32_t v) { __builtin_amdgcn_raw_buffer_store_b32(x, r, v, 0u, 0); }
__device__ __forceinline__ void buf_store_t(uint64_t x, __amdgpu_buffer_rsrc_t r, uint32_t v) {
  __builtin_amdgcn_raw_buffer_store_b64(__builtin_bit_cast(v2u, x), r, v, 0u, 0);
}

// Per-lane element copy: hardware bounds-check on V# (loads zero-fill,
// stores drop), no software OOB branch.
template <bool IsLoad, unsigned DS>
__device__ __forceinline__ void copy_one(__amdgpu_buffer_rsrc_t rsrc,
                                          uint32_t voffset,
                                          uint32_t lds_byte_off) {
  using T = typename LdsT<DS>::type;
  T __attribute__((address_space(3))) *ldsP =
      (T __attribute__((address_space(3))) *)(uintptr_t)lds_byte_off;
  if constexpr (IsLoad) *ldsP = buf_load_t((T *)nullptr, rsrc, voffset);
  else                  buf_store_t(*ldsP, rsrc, voffset);
}

// Common body shared by load and store. Templated on direction AND
// data-size: every per-element switch collapses to typed loads/stores,
// every `* DS` multiply folds to a constant or shift.
template <bool IsLoad, unsigned DS>
__device__ __forceinline__ void walk(v4i g0, v8i g1, v4i g2, v4i g3) {
  // Coerce the wave-uniform descriptor into SGPRs so all derived
  // computations (V#, dims, strides, padding, lds_byte_off) stay in
  // the scalar pipe. See the comment on rfl_v4/rfl_v8.
  g0 = rfl_v4(g0);
  g1 = rfl_v8(g1);
  g2 = rfl_v4(g2);
  g3 = rfl_v4(g3);

  // count()==0 is a NULL tensor: no copy, no atomic barrier.
  if (count(g0) == 0)
    return;

  const unsigned lane = __builtin_amdgcn_mbcnt_hi(~0u,
                            __builtin_amdgcn_mbcnt_lo(~0u, 0u));
  const unsigned wave = __builtin_amdgcn_wavefrontsize();

  const bool gm      = gather_mode(g0);
  const bool itr     = iterate_en(g1) && !gm;
  const bool sz32    = gather_index_size32(g0);
  const unsigned iter_count = itr ? (iterate_count(g2) + 1u) : 1u;

  // Unused-dimension normalisation. A low-dimensional descriptor
  // (e.g. a 2D tensor produced by a `_d2` encoding, or a `_d4`
  // encoding whose high-dim SGPRs were zeroed by the compiler
  // because they are unused) reports `tile_dim{1..4} == 0` on the
  // unused axes. Mirror the ffm runtime's descriptor patch: when a
  // tile dim is zero, set BOTH the tile extent and the companion
  // tensor bound to 1 so the loop iterates exactly once on that
  // axis and the `index 0 < bound 1` comparison never fires the
  // per-row OOB check.
  //
  //   - Dim 1 is only normalised outside gather mode; in gather
  //     mode `tile_dim1` encodes the gather count and a zero count
  //     must leave the yy-loop empty (no copies).
  //   - Dims 2..4 are unused in gather- or iterate-mode, so collapse
  //     unconditionally there; otherwise key off `tile_dim == 0`.
  const bool unused_d1 = !gm && tile_dim1(g1) == 0;
  const bool unused_d2 = (gm || itr) || tile_dim2(g1) == 0;
  const bool unused_d3 = (gm || itr) || tile_dim3(g2) == 0;
  const bool unused_d4 = (gm || itr) || tile_dim4(g3) == 0;

  // Tile dims: X is innermost and striped over lanes.
  const uint32_t T0 = tile_dim0(g1);
  const uint32_t T1 = unused_d1 ? 1u : tile_dim1(g1);
  const uint32_t T2 = unused_d2 ? 1u : tile_dim2(g1);
  const uint32_t T3 = unused_d3 ? 1u : tile_dim3(g2);
  const uint32_t T4 = unused_d4 ? 1u : tile_dim4(g3);

  // Tensor extents for OOB check. Unused dims pair with T{1..4}=1,
  // so `index 0 < bound 1` passes cleanly.
  const uint32_t Td0 = tensor_dim0(g1);
  const uint32_t Td1 = unused_d1 ? 1u : tensor_dim1(g1);
  const uint32_t Td2 = unused_d2 ? 1u : tensor_dim2(g2);
  const uint32_t Td3 = unused_d3 ? 1u : tensor_dim3(g2);
  const uint32_t Td4 = unused_d4 ? 1u : tensor_dim4(g3);

  // Strides (in element units).
  const uint64_t S0 = tensor_dim0_stride(g1);
  const uint64_t S1 = gm ? 0ull : tensor_dim1_stride(g1);
  const uint64_t S2 = (gm || itr) ? 0ull : tensor_dim2_stride(g2);
  const uint64_t S3 = (gm || itr) ? 0ull : tensor_dim3_stride(g3);

  // Padding. `pad_interval` encodes log2 of (2 dwords) -> bytes.
  const uint32_t pad_interval_bytes = pad_en(g1) ? (8u  << pad_interval(g1)) : 0u;
  const uint32_t pad_amount_bytes   = pad_en(g1) ? ((pad_amount(g1) + 1u) << 2) : 0u;

  const uint64_t global_inc = itr ? global_addr_increment(g2) : 0ull;
  const uint32_t lds_inc    = itr ? lds_addr_increment(g2)    : 0u;

  for (unsigned it = 0; it < iter_count; ++it) {
    uint32_t lds_byte_off = lds_addr(g0) + it * lds_inc * DS;
    uint64_t base_addr    = global_addr(g0)
                          + (uint64_t)it * global_inc * (uint64_t)DS;
    uint32_t bytes_since_pad = 0;

    for (uint32_t b = 0; b < T4; ++b)
    for (uint32_t a = 0; a < T3; ++a)
    for (uint32_t z = 0; z < T2; ++z)
    for (uint32_t yy = 0; yy < T1; ++yy) {
      const uint32_t y = gm ? gather_index(g2, g3, yy, sz32) : yy;

      const uint64_t row_base = base_addr +
          (uint64_t)DS *
          ((uint64_t)y * S0 + (uint64_t)z * S1 +
           (uint64_t)a * S2 + (uint64_t)b * S3);

      // Multi-dim row OOB → set num_records=0, hardware bounds-check
      // makes every X read OOB (zero-fill on load, drop on store).
      const bool row_oob = (yy >= Td1) || (z >= Td2) ||
                           (a  >= Td3) || (b >= Td4);
      auto rsrc = make_rsrc(row_base, row_oob ? 0u : Td0 * DS);

      // Stripe X across all hardware wave lanes; HW bounds-check on
      // `voffset >= num_records` covers the X-axis OOB rule.
      for (uint32_t x = lane; x < T0; x += wave)
        copy_one<IsLoad, DS>(rsrc, x * DS, lds_byte_off + x * DS);

      // Per-row LDS advance + padding accounting (uniform across lanes).
      lds_byte_off += T0 * DS;
      if (pad_interval_bytes) {
        bytes_since_pad += T0 * DS;
        while (bytes_since_pad >= pad_interval_bytes) {
          bytes_since_pad -= pad_interval_bytes;
          lds_byte_off += pad_amount_bytes;
        }
      }
    }
  }

  // Atomic barrier: the field stores bits [18:3] of a 64-bit-aligned
  // LDS pointer, low 3 bits implied zero.
  if (atomic_barrier_en(g1) && lane == 0) {
    auto *bar = (__attribute__((address_space(3))) uint32_t *)
                (uintptr_t)(atomic_barrier_addr(g1) << 3);
    __atomic_fetch_add(bar, 1u, __ATOMIC_RELEASE);
  }
}

// Per-call dispatch on `D#.data_size_log2` (g1[0] bits [17:16]).
// 2-bit field statically covers {1,2,4,8}; the default arm is
// unreachable so the optimiser drops it.
template <bool IsLoad>
__device__ __forceinline__ void dispatch(v4i g0, v8i g1, v4i g2, v4i g3) {
  switch (((unsigned)g1[0] >> 16) & 3u) {
  case 0: walk<IsLoad, 1>(g0, g1, g2, g3); return;
  case 1: walk<IsLoad, 2>(g0, g1, g2, g3); return;
  case 2: walk<IsLoad, 4>(g0, g1, g2, g3); return;
  case 3: walk<IsLoad, 8>(g0, g1, g2, g3); return;
  }
  __builtin_unreachable();
}

} // namespace

// `used` keeps the symbols alive through globaldce in the embedded
// bitcode (no in-module callers; the kernel that gets link-merged
// in references them by name). No `noinline` — let llc's inliner
// decide per call site once the helper is link-merged into a
// kernel module. Today llc's default heuristic almost never fires
// on a body this large, but a future post-link opt pass could
// benefit from constant-propagating kernel-side descriptor values
// into the helper.
//
// Signature mirrors the first four D# groups of the gfx1250
// intrinsic `int_amdgcn_tensor_{load_to_lds, store_from_lds}` but
// OMITS the intrinsic's trailing `<8 x i32> grp4` (declared
// "reserved for future targets, use zeroinitializer for now") and
// `i32 cpol` arguments. Neither is read by the emulation walk
// today, so carrying them through would just bloat the helper's
// IR prologue. The cross-target emit site in
// `handle_vimage.cpp::handleVIMAGE` elides those two arguments
// from the `CallInst` for the helper path while still passing all
// six to the same-target intrinsic emit (the intrinsic's TableGen
// definition still carries them).
extern "C" __device__ __attribute__((used))
void salmon_tdm_load_to_lds(v4i g0, v8i g1, v4i g2, v4i g3) {
  dispatch</*IsLoad=*/true>(g0, g1, g2, g3);
}

extern "C" __device__ __attribute__((used))
void salmon_tdm_store_from_lds(v4i g0, v8i g1, v4i g2, v4i g3) {
  dispatch</*IsLoad=*/false>(g0, g1, g2, g3);
}
