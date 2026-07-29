#pragma once

#include <hip/amd_detail/amd_hip_runtime.h>

namespace details {

#if __has_builtin(__builtin_amdgcn_global_store_async_from_lds_b128) and                           \
    __has_builtin(__builtin_amdgcn_global_load_async_to_lds_b128)

typedef int __attribute__((ext_vector_type(2))) vint2;
typedef int __attribute__((ext_vector_type(4))) vint4;

// Some size sanity checks
static_assert(sizeof(char) == 1);
static_assert(sizeof(int) == 4);
static_assert(sizeof(vint2) == 8);
static_assert(sizeof(vint4) == 16);

template <typename TyElem>
__device__ static __forceinline__ void accelerated_memcpy_global_to_lds(
    TyElem* __restrict__ dst, const TyElem* __restrict__ src, const size_t offset,
    const size_t count) {
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
__device__ static __forceinline__ void accelerated_memcpy_lds_to_global(
    TyElem* __restrict__ dst, const TyElem* __restrict__ src, const size_t offset,
    const size_t count) {
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

#endif

}  // namespace details
