#pragma once

#include <type_traits>

#include "amd_hip_cooperative_groups.h"

namespace cooperative_groups {
namespace details {
template <typename TyGroup> struct can_group_do_async_copy : public std::false_type {};
template <unsigned int size, typename TyPar>
struct can_group_do_async_copy<cooperative_groups::thread_block_tile<size, TyPar>>
    : public std::true_type {};
template <> struct can_group_do_async_copy<cooperative_groups::coalesced_group>
    : public std::true_type {};
template <> struct can_group_do_async_copy<cooperative_groups::thread_block>
    : public std::true_type {};

// True when p is in the global aperture. Scratch and LDS are rejected: both accelerated
// copies take a global (addrspace 1) pointer, and casting a private address into that
// aperture reads the wrong memory. On MI210 that load is undefined; on MI300A
// global_load_lds is not replayed if the access takes an XNACK fault. Constant memory
// shares the global aperture on AMDHSA and is accepted.
__CG_STATIC_QUALIFIER__ bool is_global_mem(const void* p) {
#if __has_builtin(__builtin_amdgcn_is_shared) && __has_builtin(__builtin_amdgcn_is_private)
  const auto* q = (const __attribute__((address_space(0))) void*)p;
  return !__builtin_amdgcn_is_shared(q) && !__builtin_amdgcn_is_private(q);
#else
  (void)p;
  return false;
#endif
}

__CG_STATIC_QUALIFIER__ bool is_lds_mem(const void* p) {
#if __has_builtin(__builtin_amdgcn_is_shared) && __has_builtin(__builtin_amdgcn_is_private)
  const auto* q = (const __attribute__((address_space(0))) void*)p;
  return __builtin_amdgcn_is_shared(q) && !__builtin_amdgcn_is_private(q);
#else
  (void)p;
  return false;
#endif
}

#if __has_builtin(__builtin_amdgcn_global_store_async_from_lds_b128) and                           \
    __has_builtin(__builtin_amdgcn_global_load_async_to_lds_b128)
template <typename TyElem>
__CG_STATIC_QUALIFIER__ void accelerated_memcpy_global_to_lds(TyElem* __restrict__ dst,
                                                              const TyElem* __restrict__ src,
                                                              const size_t offset,
                                                              const size_t count) {
  typedef int __attribute__((ext_vector_type(2))) vint2;
  typedef int __attribute__((ext_vector_type(4))) vint4;

  // Some size sanity checks
  static_assert(sizeof(char) == 1);
  static_assert(sizeof(int) == 4);
  static_assert(sizeof(vint2) == 8);
  static_assert(sizeof(vint4) == 16);

  char* c_dst = ((char*)dst) + offset;
  char* c_src = ((char*)src) + offset;
  size_t bytes_left = count;

  while (bytes_left > 0) {
    if (bytes_left >= 16) {
      if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_global_load_async_to_lds_b128))
        __builtin_amdgcn_global_load_async_to_lds_b128(
            (__attribute__((address_space(1))) vint4*)c_src,
            (__attribute__((address_space(3))) vint4*)c_dst, 0 /* offset */, 0 /* cache policy */);
      bytes_left -= 16;
      c_src += 16;
      c_dst += 16;
    } else if (bytes_left >= 8) {
      if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_global_load_async_to_lds_b64))
        __builtin_amdgcn_global_load_async_to_lds_b64(
            (__attribute__((address_space(1))) vint2*)c_src,
            (__attribute__((address_space(3))) vint2*)c_dst, 0 /* offset */, 0 /* cache policy */);
      bytes_left -= 8;
      c_src += 8;
      c_dst += 8;
    } else if (bytes_left >= 4) {
      if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_global_load_async_to_lds_b32))
        __builtin_amdgcn_global_load_async_to_lds_b32(
            (__attribute__((address_space(1))) int*)c_src,
            (__attribute__((address_space(3))) int*)c_dst, 0 /* offset */, 0 /* cache policy */);
      bytes_left -= 4;
      c_src += 4;
      c_dst += 4;
    } else {
      if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_global_load_async_to_lds_b8))
        __builtin_amdgcn_global_load_async_to_lds_b8(
            (__attribute__((address_space(1))) char*)c_src,
            (__attribute__((address_space(3))) char*)c_dst, 0 /* offset */, 0 /* cache policy */);
      bytes_left--;
      c_src++;
      c_dst++;
    }
  }
}

template <typename TyElem>
__CG_STATIC_QUALIFIER__ void accelerated_memcpy_lds_to_global(TyElem* __restrict__ dst,
                                                              const TyElem* __restrict__ src,
                                                              const size_t offset,
                                                              const size_t count) {
  typedef int __attribute__((ext_vector_type(2))) vint2;
  typedef int __attribute__((ext_vector_type(4))) vint4;

  char* c_dst = ((char*)dst) + offset;
  char* c_src = ((char*)src) + offset;
  size_t bytes_left = count;

  while (bytes_left > 0) {
    if (bytes_left >= 16) {
      if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_global_store_async_from_lds_b128))
        __builtin_amdgcn_global_store_async_from_lds_b128(
            (__attribute__((address_space(1))) vint4*)c_dst,
            (__attribute__((address_space(3))) vint4*)c_src, 0 /* offset */, 0 /* cache policy */);
      bytes_left -= 16;
      c_src += 16;
      c_dst += 16;
    } else if (bytes_left >= 8) {
      if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_global_store_async_from_lds_b64))
        __builtin_amdgcn_global_store_async_from_lds_b64(
            (__attribute__((address_space(1))) vint2*)c_dst,
            (__attribute__((address_space(3))) vint2*)c_src, 0 /* offset */, 0 /* cache policy */);
      bytes_left -= 8;
      c_src += 8;
      c_dst += 8;
    } else if (bytes_left >= 4) {
      if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_global_store_async_from_lds_b32))
        __builtin_amdgcn_global_store_async_from_lds_b32(
            (__attribute__((address_space(1))) int*)c_dst,
            (__attribute__((address_space(3))) int*)c_src, 0 /* offset */, 0 /* cache policy */);
      bytes_left -= 4;
      c_src += 4;
      c_dst += 4;
    } else {
      if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_global_store_async_from_lds_b8))
        __builtin_amdgcn_global_store_async_from_lds_b8(
            (__attribute__((address_space(1))) char*)c_dst,
            (__attribute__((address_space(3))) char*)c_src, 0 /* offset */, 0 /* cache policy */);
      bytes_left--;
      c_src++;
      c_dst++;
    }
  }
}

template <class TyGroup, typename TyElem,
          typename std::enable_if<details::can_group_do_async_copy<TyGroup>{}, bool>::type = true>
__CG_STATIC_QUALIFIER__ bool dispatch_async_memcpy(const TyGroup& group, TyElem* __restrict__ dst,
                                                   const TyElem* __restrict__ src,
                                                   const size_t count) {
  if (count == 0) {
    return true;
  }

  // Per-lane LDS address. This is the gfx12.5 async-copy form, not the gfx9
  // LDS DMA form below: the destination pointer is the address this lane writes.
  const bool lds_to_global = is_lds_mem(src) && is_global_mem(dst);
  const bool global_to_lds = is_global_mem(src) && is_lds_mem(dst);
  if (!lds_to_global && !global_to_lds) {
    return false;
  }

  // We have total size in bytes: count
  // Total count of threads: group.size()
  // Each thread will have to do count / group.size() bytes copy in lockstep
  size_t group_size = group.size();
  size_t bytes_per_thread = count / group_size;
  if (bytes_per_thread > 0) {
    if (lds_to_global) {
      details::accelerated_memcpy_lds_to_global(dst, src, bytes_per_thread * group.thread_rank(),
                                                bytes_per_thread);
    } else {
      details::accelerated_memcpy_global_to_lds(dst, src, bytes_per_thread * group.thread_rank(),
                                                bytes_per_thread);
    }
  }

  // Now we handle data that could not be copied alongside all threads
  // example: user asked to copy 33 bytes on 32 threads, each thread will do 1 byte async-copy in
  // lock-step but for the last 1 byte we need to manually handle it and enqueue the memcpy
  size_t bytes_copied = bytes_per_thread * group_size;
  if (group.thread_rank() == 0 && count > bytes_copied) {
    if (lds_to_global) {
      details::accelerated_memcpy_lds_to_global(dst, src, bytes_copied, count - bytes_copied);
    } else {
      details::accelerated_memcpy_global_to_lds(dst, src, bytes_copied, count - bytes_copied);
    }
  }
  return true;
}
#endif

#if __has_builtin(__builtin_amdgcn_load_to_lds) && __CG_LDS_DMA_TARGET
// LDS DMA does not write to the per-lane address it is handed. The destination is a wave-uniform
// base and the hardware adds lane_id * 4 for a 4-byte transfer. The copy is therefore partitioned
// by rank stride, which puts consecutive lanes on consecutive dwords, rather than by the contiguous
// per-thread chunks dispatch_async_memcpy uses. The group's ranks must be contiguous within a wave,
// which holds for a thread_block but not for a tile or a coalesced group.
//
// The size argument stays 4 on every gfx9 target, including gfx950. gfx950 also accepts 12 and
// 16, but those sizes add lane_id * 16 to the LDS base and would scatter the bytes. gfx906,
// gfx908 and gfx90a (MI210) encode this as global_load_dword with the lds bit; gfx942
// (MI300A/MI300X) and gfx950 encode global_load_lds_dword. Both forms use the same
// M0 + lane_id * 4 addressing and both retire through vmcnt. gfx9 is always wave64, so lane is
// always 0..63 here. gfx10 is excluded by __CG_LDS_DMA_TARGET even though the builtin is
// invocable there; gfx11 and gfx12 lack vmem-to-lds-load-insts entirely, and gfx12.5 is handled
// by the async path above. Compilers without __builtin_amdgcn_load_to_lds at all, which includes
// the ROCm 7 clang, skip this block and use traditional_memcpy_bytes.
//
// global_load_lds is not replayed on an XNACK fault. The source must already be a resident global
// address. Scratch fails that requirement and is rejected below; on MI300A a private or system
// address cast into the global aperture would read the wrong memory or fault without replay.
template <typename TyGroup> struct can_group_use_lds_dma : public std::false_type {};
template <> struct can_group_use_lds_dma<cooperative_groups::thread_block>
    : public std::true_type {};

template <class TyGroup, typename TyElem,
          typename std::enable_if<!details::can_group_use_lds_dma<TyGroup>{}, bool>::type = true>
__CG_STATIC_QUALIFIER__ bool dispatch_lds_dma_memcpy(const TyGroup&, TyElem* __restrict__,
                                                     const TyElem* __restrict__, const size_t) {
  return false;
}

template <class TyGroup, typename TyElem,
          typename std::enable_if<details::can_group_use_lds_dma<TyGroup>{}, bool>::type = true>
__CG_STATIC_QUALIFIER__ bool dispatch_lds_dma_memcpy(const TyGroup& group, TyElem* __restrict__ dst,
                                                     const TyElem* __restrict__ src,
                                                     const size_t count) {
  if (count == 0) {
    return true;
  }

  // LDS DMA only moves resident global memory into LDS. !is_shared(src) is not
  // enough: a private (scratch) pointer is also not LDS, and addrspacecast of that
  // pointer to the global aperture is undefined on MI210 and faults without XNACK
  // replay on MI300A.
  if (!is_lds_mem(dst) || !is_global_mem(src)) {
    return false;
  }

  // The dword form faults if either address is not 4-byte aligned. Element
  // alignment is a compile-time Hint, but the runtime pointers may not honor it.
  if ((reinterpret_cast<unsigned long>(src) | reinterpret_cast<unsigned long>(dst)) & 3ul) {
    return false;
  }

  // Callers reach this only for element types aligned to at least a dword, so every dword
  // transfer below is naturally aligned when the pointers themselves are.
  char* c_dst = (char*)dst;
  const char* c_src = (const char*)src;

  const size_t ndwords = count / 4;
  const unsigned int num_threads = group.num_threads();
  const unsigned int rank = group.thread_rank();
  // The hardware lane this thread occupies. This is what the LDS DMA unit adds to the base, so
  // it has to be the physical lane and not rank % wave_size. gfx9 is wave64 throughout.
  const unsigned int lane = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
  // rank - lane is uniform in a thread_block. readfirstlane forces the SGPR that is copied to
  // M0; a divergent value would make every lane use the first active lane's LDS base.
  unsigned int wave_first_rank = rank - lane;
#if __has_builtin(__builtin_amdgcn_readfirstlane)
  wave_first_rank =
      static_cast<unsigned int>(__builtin_amdgcn_readfirstlane(static_cast<int>(wave_first_rank)));
#endif

  // base_idx is uniform across the wave, so it can be handed to the hardware as the destination
  // base while each lane supplies its own source address.
  for (size_t base_idx = wave_first_rank; base_idx < ndwords; base_idx += num_threads) {
    const size_t idx = base_idx + lane;
    if (idx < ndwords) {
      __builtin_amdgcn_load_to_lds((__attribute__((address_space(1))) int*)(c_src + idx * 4),
                                   (__attribute__((address_space(3))) int*)(c_dst + base_idx * 4),
                                   4 /* size */, 0 /* offset */, 0 /* cache policy */);
    }
  }

  // At most three bytes cannot fill a dword, so one thread stores them without LDS DMA.
  if (rank == 0) {
    for (size_t i = ndwords * 4; i < count; i++) {
      c_dst[i] = c_src[i];
    }
  }

  return true;
}
#endif

// Traditional Copy used when memcpy_async builtins are unavailable or not invocable on the target.
// Partition the memory into segments which each thread copies a portion of, similar to the
// accelerated copy.
template <class TyGroup, typename TyElem, typename TySize>
__CG_STATIC_QUALIFIER__ void traditional_memcpy_bytes(const TyGroup& group,
                                                      TyElem* __restrict__ dst,
                                                      const TyElem* __restrict__ src,
                                                      const TySize& count) {
  const size_t group_size = group.size();
  const size_t rank = group.thread_rank();
  const size_t bytes = static_cast<size_t>(count);
  unsigned char* c_dst = (unsigned char*)dst;
  const unsigned char* c_src = (const unsigned char*)src;

  // Rank strided, not rank contiguous. Neighbouring ranks have to touch neighbouring
  // addresses for a wave's accesses to coalesce into whole cache lines; giving each rank
  // its own contiguous chunk made one wave issue group_size separate lines per step.
  // The widest element size both pointers can sustain is chosen at run time, because the
  // only alignment known at compile time is alignof(TyElem) and the caller's pointers do
  // not have to be more aligned than that.
  const unsigned long addrs =
      reinterpret_cast<unsigned long>(c_dst) | reinterpret_cast<unsigned long>(c_src);

  if ((addrs & 15ul) == 0ul) {
    typedef unsigned int __attribute__((ext_vector_type(4))) cg_uint4;
    const size_t nvec = bytes / 16;
    for (size_t i = rank; i < nvec; i += group_size) {
      ((cg_uint4*)c_dst)[i] = ((const cg_uint4*)c_src)[i];
    }
    // At most 15 bytes are left over, so ranks 0..14 finish them one byte each.
    for (size_t i = nvec * 16 + rank; i < bytes; i += group_size) {
      c_dst[i] = c_src[i];
    }
  } else if ((addrs & 3ul) == 0ul) {
    const size_t ndwords = bytes / 4;
    for (size_t i = rank; i < ndwords; i += group_size) {
      ((unsigned int*)c_dst)[i] = ((const unsigned int*)c_src)[i];
    }
    for (size_t i = ndwords * 4 + rank; i < bytes; i += group_size) {
      c_dst[i] = c_src[i];
    }
  } else {
    for (size_t i = rank; i < bytes; i += group_size) {
      c_dst[i] = c_src[i];
    }
  }
}

template <class TyGroup, typename TyElem, typename TySize, size_t Hint = alignof(TyElem)>
__CG_STATIC_QUALIFIER__ void memcpy_async_bytes(const TyGroup& group, TyElem* __restrict__ dst,
                                                const TyElem* __restrict__ src,
                                                const TySize& count) {
#if __has_builtin(__builtin_amdgcn_global_store_async_from_lds_b128) and                           \
    __has_builtin(__builtin_amdgcn_global_load_async_to_lds_b128)
  // Use the async path only when the builtins are actually invocable on the target, otherwise the
  // accelerated loop would advance offsets without issuing any load/store and silently drop data.
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_global_store_async_from_lds_b128) &&
      __builtin_amdgcn_is_invocable(__builtin_amdgcn_global_load_async_to_lds_b128)) {
    // gfx12.5 only. These builtins require wavefrontsize32, so gfx950 (wave64)
    // cannot take this path. A direction this path cannot legally perform falls
    // through to the copies below instead of being dropped.
    if (details::dispatch_async_memcpy(group, dst, src, count)) {
      return;
    }
  }
#endif
#if __has_builtin(__builtin_amdgcn_load_to_lds) && __CG_LDS_DMA_TARGET
  // __CG_LDS_DMA_TARGET restricts this to gfx9, which is where the wave uniform LDS base
  // plus lane_id * 4 addressing has been verified. __builtin_amdgcn_is_invocable is kept as
  // well so a generic gfx9 target that lacks vmem-to-lds-load-insts still falls through.
  // gfx10, gfx11 and gfx12 stay on the traditional copy; gfx12.5 never reaches here when its
  // async builtins are invocable. Hint is a constant, so a sub-dword element type does not
  // emit this path.
  if ((Hint % 4) == 0 && __builtin_amdgcn_is_invocable(__builtin_amdgcn_load_to_lds) &&
      details::dispatch_lds_dma_memcpy(group, dst, src, count)) {
    return;
  }
#endif
  traditional_memcpy_bytes(group, dst, src, count);
}
}  // namespace details

/*
 * Enqueue a copy of `count` bytes.
 *
 * The copy is not complete when this function returns on targets that use an accelerated path
 * (gfx12.5 async copies, and gfx9 LDS DMA, including MI210, MI300A/MI300X, and gfx950).
 * Read the destination only after `group.sync()`. That is the portable rule, and on gfx12.5 it
 * is the only thing that works: those copies retire through the async counter, which no fence
 * or barrier drains. gfx9 LDS DMA instead retires through vmcnt, which the workgroup release
 * fence in any barrier already waits for, so `__syncthreads()` is also sufficient there.
 */
template <class TyGroup, typename TyElem, typename TySizeT>
__CG_STATIC_QUALIFIER__ void memcpy_async(const TyGroup& group, TyElem* __restrict__ dst,
                                          const TyElem* __restrict__ src, const TySizeT& count) {
  details::memcpy_async_bytes(group, dst, src, count);
}

/*
 * Enqueue a copy of min(dstLayout, srcLayout) elements.
 * The destination must not be read until `group.sync()`. See memcpy_async above.
 */
template <class TyGroup, class TyElem, typename DstLayout, typename SrcLayout>
__CG_STATIC_QUALIFIER__ void memcpy_async(const TyGroup& group, TyElem* __restrict__ dst,
                                          const DstLayout& dstLayout,
                                          const TyElem* __restrict__ src,
                                          const SrcLayout& srcLayout) {
  auto l_min = [](DstLayout d, SrcLayout s) { return d > s ? s : d; };
  auto count = l_min(dstLayout, srcLayout);
  details::memcpy_async_bytes(group, dst, src, count * sizeof(TyElem));
}
}  // namespace cooperative_groups