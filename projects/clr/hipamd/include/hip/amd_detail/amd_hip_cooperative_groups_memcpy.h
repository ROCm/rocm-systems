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
__CG_STATIC_QUALIFIER__ void dispatch_async_memcpy(const TyGroup& group, TyElem* __restrict__ dst,
                                                   const TyElem* __restrict__ src,
                                                   const size_t count) {
  if (count == 0) {
    return;
  }

  bool src_is_shared =
      __builtin_amdgcn_is_shared((const __attribute__((address_space(0))) void*)src);
  bool dst_is_shared =
      __builtin_amdgcn_is_shared((const __attribute__((address_space(0))) void*)dst);

  // We have total size in bytes: count
  // Total count of threads: group.size()
  // Each thread will have to do count / group.size() bytes copy in lockstep
  size_t group_size = group.size();
  size_t bytes_per_thread = count / group_size;
  if (src_is_shared && !dst_is_shared && bytes_per_thread > 0) {
    details::accelerated_memcpy_lds_to_global(dst, src, bytes_per_thread * group.thread_rank(),
                                              bytes_per_thread);
  } else if (!src_is_shared && dst_is_shared && bytes_per_thread > 0) {
    details::accelerated_memcpy_global_to_lds(dst, src, bytes_per_thread * group.thread_rank(),
                                              bytes_per_thread);
  }

  // Now we handle data that could not be copied alongside all threads
  // example: user asked to copy 33 bytes on 32 threads, each thread will do 1 byte async-copy in
  // lock-step but for the last 1 byte we need to manually handle it and enqueue the memcpy
  size_t bytes_copied = bytes_per_thread * group_size;
  if (group.thread_rank() == 0 && count > bytes_copied) {
    if (src_is_shared && !dst_is_shared) {
      details::accelerated_memcpy_lds_to_global(dst, src, bytes_copied, count - bytes_copied);
    } else if (!src_is_shared && dst_is_shared) {
      details::accelerated_memcpy_global_to_lds(dst, src, bytes_copied, count - bytes_copied);
    }
  }

  // Deliberately no wait here: the copies above are tracked by ASYNCcnt and the caller's group
  // sync drains it before releasing the barrier, which is what keeps the copy overlappable.
}
#endif

#if __has_builtin(__builtin_amdgcn_load_to_lds)
// LDS DMA does not write to the per-lane address it is handed: the destination is a wave uniform
// base and the hardware adds lane_id * 4. So the copy has to be partitioned by rank stride, which
// puts consecutive lanes on consecutive dwords, rather than by the contiguous per-thread chunks
// dispatch_async_memcpy uses. That also requires the group's ranks to be contiguous within a wave,
// which holds for a thread_block but not for a tile or a coalesced group, and it restricts the
// transfer to a dword: narrower transfers keep the four byte lane stride and cannot pack.
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
__CG_STATIC_QUALIFIER__ bool dispatch_lds_dma_memcpy(const TyGroup& group,
                                                     TyElem* __restrict__ dst,
                                                     const TyElem* __restrict__ src,
                                                     const size_t count) {
  if (count == 0) {
    return true;
  }

  // LDS DMA only moves global -> LDS.
  if (!__builtin_amdgcn_is_shared((const __attribute__((address_space(0))) void*)dst) ||
      __builtin_amdgcn_is_shared((const __attribute__((address_space(0))) void*)src)) {
    return false;
  }

  // Callers reach this only for element types aligned to at least a dword, so every dword
  // transfer below is naturally aligned.
  char* c_dst = (char*)dst;
  const char* c_src = (const char*)src;

  const size_t ndwords = count / 4;
  const unsigned int num_threads = group.num_threads();
  const unsigned int rank = group.thread_rank();
  const unsigned int lane = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
  const unsigned int wave_first_rank = rank - lane;

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
  size_t group_size = group.size();
  size_t bytes_per_thread = count / group_size; /* each thread will copy this much */
  unsigned char *c_src = ((unsigned char*)src) + (group.thread_rank() * bytes_per_thread),
                *c_dst = ((unsigned char*)dst) + (group.thread_rank() * bytes_per_thread);
  for (size_t i = 0; i < bytes_per_thread; i++) {
    c_dst[i] = c_src[i];
  }

  // copy remaining with 1 thread
  size_t bytes_copied = bytes_per_thread * group_size;
  if (group.thread_rank() == 0 && count > bytes_copied) {
    for (size_t i = bytes_copied; i < count; i++) {
      ((unsigned char*)dst)[i] = ((unsigned char*)src)[i];
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
  // It also only implements global<->LDS, so anything else (global->global,
  // LDS->LDS) has to take the traditional copy or no data would be moved at all.
  const bool src_is_shared =
      __builtin_amdgcn_is_shared((const __attribute__((address_space(0))) void*)src);
  const bool dst_is_shared =
      __builtin_amdgcn_is_shared((const __attribute__((address_space(0))) void*)dst);
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_global_store_async_from_lds_b128) &&
      __builtin_amdgcn_is_invocable(__builtin_amdgcn_global_load_async_to_lds_b128) &&
      src_is_shared != dst_is_shared) {
    details::dispatch_async_memcpy(group, dst, src, count);
    return;
  }
#endif
#if __has_builtin(__builtin_amdgcn_load_to_lds)
  // Targets without the async builtins can still move global->LDS without staging the data
  // through the VGPRs, which the traditional copy below cannot avoid. LDS DMA transfers a dword
  // per lane, so it is only used for element types that guarantee dword alignment; Hint is a
  // constant, which keeps the unused path out of the generated code.
  if ((Hint % 4) == 0 && __builtin_amdgcn_is_invocable(__builtin_amdgcn_load_to_lds) &&
      details::dispatch_lds_dma_memcpy(group, dst, src, count)) {
    return;
  }
#endif
  traditional_memcpy_bytes(group, dst, src, count);
}
}  // namespace details

/*
 * Enqueue memcpy async of `count` bytes.
 *
 * The copy is not complete when this returns. Call `group.sync()` before reading `dst`; the sync
 * both waits for the copy and makes it visible to the rest of the group.
 */
template <class TyGroup, typename TyElem, typename TySizeT>
__CG_STATIC_QUALIFIER__ void memcpy_async(const TyGroup& group, TyElem* __restrict__ dst,
                                          const TyElem* __restrict__ src, const TySizeT& count) {
  details::memcpy_async_bytes(group, dst, src, count);
}

/*
 * Enqueue memcpy async of min(dstLayout, srcLayout) elements.
 *
 * As above, the caller must `group.sync()` before reading `dst`.
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