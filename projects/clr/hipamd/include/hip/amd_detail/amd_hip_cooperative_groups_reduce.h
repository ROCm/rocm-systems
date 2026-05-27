/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_REDUCE_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_REDUCE_H

#if __cplusplus
#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/hip_cooperative_groups_helper.h>
#include <hip/amd_detail/amd_hip_cooperative_groups.h>
#endif

#if !defined(__HIPCC_RTC__)
#include <type_traits>
#endif

namespace cooperative_groups {
namespace impl {
  template <typename T, typename U>
  using is_param_type_same = __hip_internal::is_same<typename __hip_internal::remove_cvref<T>,
                                                     typename __hip_internal::remove_cvref<U>>;

  template <typename T, typename = void>
  struct has_add : __hip_internal::false_type {
  };

  template <typename T>
  struct has_add<T,
                 __hip_internal::void_t<decltype(__reduce_add_sync<unsigned long long>(0ull, T {}))>
    > : __hip_internal::true_type {};

  template <typename T, typename = void>
  struct has_min : __hip_internal::false_type {
  };

  template <typename T>
  struct has_min<T,
                 __hip_internal::void_t<decltype(__reduce_min_sync<unsigned long long>(0ull, T {}))>
    > : __hip_internal::true_type {};

  template <typename T, typename = void>
  struct has_max : __hip_internal::false_type {
  };

  template <typename T>
  struct has_max<T,
                 __hip_internal::void_t<decltype(__reduce_max_sync<unsigned long long>(0ull, T {}))>
    > : __hip_internal::true_type {};

  template <typename T, typename = void>
  struct has_and : __hip_internal::false_type {
  };

  template <typename T>
  struct has_and<T,
                 __hip_internal::void_t<decltype(__reduce_and_sync<unsigned long long>(0ull, T {}))>
    > : __hip_internal::true_type {};

  template <typename T, typename = void>
  struct has_or : __hip_internal::false_type {
  };

  template <typename T>
  struct has_or<T,
                 __hip_internal::void_t<decltype(__reduce_or_sync<unsigned long long>(0ull, T {}))>
    > : __hip_internal::true_type {};

  template <typename T, typename = void>
  struct has_xor : __hip_internal::false_type {
  };

  template <typename T>
  struct has_xor<T,
                 __hip_internal::void_t<decltype(__reduce_xor_sync<unsigned long long>(0ull, T {}))>
    > : __hip_internal::true_type {};

  // we can call reduce() only the block tiles that have a compile-time size
  template <class TyGroup>
  struct isTiledGroup : __hip_internal::false_type {
  };

  template <unsigned int N, class ParentCGTy>
  struct isTiledGroup<cooperative_groups::thread_block_tile<N, ParentCGTy>>
    : __hip_internal::integral_constant<bool,
          (N == 1  || N == 2  || N == 4  || N == 8 ||
           N == 16 || N == 32 || N == 64)> {
  };

  template <class TyGroup>
  struct isCoalescedGroup : __hip_internal::false_type {
  };

  template <>
  struct isCoalescedGroup<cooperative_groups::coalesced_group> : __hip_internal::true_type {
  };
}

/** \ingroup CooperativeGAPI
  *
  * \brief   Returns the result of the reduction of the specified cooperative group using a
  *          specified functor
  * \details A collective operation that can be used with a thread_block_tile whose size is known
  *          at compile-time or a coalesced_group.
  *
  * \tparam  TyGroup  The cooperative group class template parameter.
  * \tparam  TyVal    The type of the value this thread contributes for the reduction.
  * \tparam  TyFn     The type of the function object
  *
  * \param group  The group to reduce.
  * \param val    The value this thread contributes to the reduction.
  * \param op     The function object whose operator() will be called.
  */
template <typename TyGroup, typename TyVal, typename TyFn>
__CG_QUALIFIER__ auto reduce(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
  using Op = typename __hip_internal::remove_cvref<TyFn>::type;
  using Val = typename __hip_internal::remove_cvref<TyVal>::type;
  static_assert(impl::is_param_type_same<Val, decltype(op(val, val))>::value, "Operator input and output types differ");
  static_assert(__hip_internal::is_trivially_copyable<Val>::value, "val must be trivially copyable");
  static_assert(sizeof(Val) <= 32, "Can only reduce values of size up to 32 bytes");

  if constexpr (!impl::isTiledGroup<TyGroup>::value && !impl::isCoalescedGroup<TyGroup>::value) {
    static_assert(__hip_internal::is_void<TyGroup>::value, "This group does not exclusively represent a tile");
  }

  unsigned long long mask = ~0ull;

  // we cannot simply just use the __activemask() here, because more than one tile could have active
  // threads at a time; we need to mask away the threads that not part of this tile first
  if constexpr (!__hip_internal::is_same<TyGroup, cooperative_groups::coalesced_group>::value) {
    mask >>= (64 - group.num_threads());
    mask <<= (((threadIdx.x % warpSize) / group.num_threads()) * group.num_threads());
  }

  // for coalesced_groups, the mask is simply the activemask
  mask &= __activemask();

  if constexpr (__hip_internal::is_same<Op, cooperative_groups::plus<Val>>::value &&
                impl::has_add<Val>::value) {
    return __reduce_add_sync(mask, val);
  } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::less<Val>>::value &&
                impl::has_min<Val>::value) {
    return __reduce_min_sync(mask, val);
  } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::greater<Val>>::value &&
                impl::has_max<Val>::value) {
    return __reduce_max_sync(mask, val);
  } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::bit_and<Val>>::value &&
                impl::has_and<Val>::value) {
    return __reduce_and_sync(mask, val);
  } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::bit_or<Val>>::value &&
                impl::has_or<Val>::value) {
    return __reduce_or_sync(mask, val);
  } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::bit_xor<Val>>::value &&
                impl::has_xor<Val>::value) {
    return __reduce_xor_sync(mask, val);
  } else {
    return __reduce_op_sync(mask, val, op, nullptr);
  }
}
}

#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_REDUCE_H
