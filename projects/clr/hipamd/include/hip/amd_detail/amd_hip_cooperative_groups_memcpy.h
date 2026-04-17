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
      __builtin_amdgcn_global_load_async_to_lds_b128(
          (__attribute__((address_space(1))) vint4*)c_src,
          (__attribute__((address_space(3))) vint4*)c_dst, 0 /* offset */, 0 /* cache policy */);
      bytes_left -= 16;
      c_src += 16;
      c_dst += 16;
    } else if (bytes_left >= 8) {
      __builtin_amdgcn_global_load_async_to_lds_b64((__attribute__((address_space(1))) vint2*)c_src,
                                                    (__attribute__((address_space(3))) vint2*)c_dst,
                                                    0 /* offset */, 0 /* cache policy */);
      bytes_left -= 8;
      c_src += 8;
      c_dst += 8;
    } else if (bytes_left >= 4) {
      __builtin_amdgcn_global_load_async_to_lds_b32((__attribute__((address_space(1))) int*)c_src,
                                                    (__attribute__((address_space(3))) int*)c_dst,
                                                    0 /* offset */, 0 /* cache policy */);
      bytes_left -= 4;
      c_src += 4;
      c_dst += 4;
    } else {
      __builtin_amdgcn_global_load_async_to_lds_b8((__attribute__((address_space(1))) char*)c_src,
                                                   (__attribute__((address_space(3))) char*)c_dst,
                                                   0 /* offset */, 0 /* cache policy */);
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
      __builtin_amdgcn_global_store_async_from_lds_b128(
          (__attribute__((address_space(1))) vint4*)c_dst,
          (__attribute__((address_space(3))) vint4*)c_src, 0 /* offset */, 0 /* cache policy */);
      bytes_left -= 16;
      c_src += 16;
      c_dst += 16;
    } else if (bytes_left >= 8) {
      __builtin_amdgcn_global_store_async_from_lds_b64(
          (__attribute__((address_space(1))) vint2*)c_dst,
          (__attribute__((address_space(3))) vint2*)c_src, 0 /* offset */, 0 /* cache policy */);
      bytes_left -= 8;
      c_src += 8;
      c_dst += 8;
    } else if (bytes_left >= 4) {
      __builtin_amdgcn_global_store_async_from_lds_b32(
          (__attribute__((address_space(1))) int*)c_dst,
          (__attribute__((address_space(3))) int*)c_src, 0 /* offset */, 0 /* cache policy */);
      bytes_left -= 4;
      c_src += 4;
      c_dst += 4;
    } else {
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
}
#endif

template <class TyGroup, typename TyElem, typename TySize, size_t Hint = alignof(TyElem)>
__CG_STATIC_QUALIFIER__ void memcpy_async_bytes(const TyGroup& group, TyElem* __restrict__ dst,
                                                const TyElem* __restrict__ src,
                                                const TySize& count) {
#if __has_builtin(__builtin_amdgcn_global_store_async_from_lds_b128) and                           \
    __has_builtin(__builtin_amdgcn_global_load_async_to_lds_b128)
  details::dispatch_async_memcpy(group, dst, src, count);
#else
  // Traditional Copy since we do not have memcpy_async builtins
  // Partition the memory in group of segments which each thread will copy portion of
  // Similar to what we do in accelerated copy
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
#endif
}
}  // namespace details

/*
 * Enqueue memcpy async of `count` bytes and await for completion
 */
template <class TyGroup, typename TyElem, typename TySizeT>
__CG_STATIC_QUALIFIER__ void memcpy_async(const TyGroup& group, TyElem* __restrict__ dst,
                                          const TyElem* __restrict__ src, const TySizeT& count) {
  details::memcpy_async_bytes(group, dst, src, count);
}

/*
 * Enqueue memcpy async min(dstLayout, srcLayout) elements bytes and await for completion
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