#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_SCAN_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_SCAN_H

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
  // these functions allow to make use of C++ function overloads, instead of having to code
  // a big if-constexpr according to operand type
  HIP_IMPL_GENERATE_SCAN_FUNC(add, i32, int);
  HIP_IMPL_GENERATE_SCAN_FUNC(add, u32, unsigned int);

  HIP_IMPL_GENERATE_SCAN_FUNC(min, i32, int);
  HIP_IMPL_GENERATE_SCAN_FUNC(min, u32, unsigned int);

  HIP_IMPL_GENERATE_SCAN_FUNC(max, i32, int);
  HIP_IMPL_GENERATE_SCAN_FUNC(max, u32, unsigned int);

  HIP_IMPL_GENERATE_SCAN_FUNC(and, i32, int);
  HIP_IMPL_GENERATE_SCAN_FUNC(and, u32, unsigned int);

  HIP_IMPL_GENERATE_SCAN_FUNC(or, i32, int);
  HIP_IMPL_GENERATE_SCAN_FUNC(or, u32, unsigned int);

  HIP_IMPL_GENERATE_SCAN_FUNC(xor, i32, int);
  HIP_IMPL_GENERATE_SCAN_FUNC(xor, u32, unsigned int);

  // extra types. Unlike cg::reduce() which depends on reduce_*_sync() functions being defined with
  // HIP_ENABLE_EXTRA_WARP_SYNC_TYPES to be able to use the ockl intrinsics, for scan we always
  // define them here
  HIP_IMPL_GENERATE_SCAN_FUNC(add, i64, long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(add, u64, unsigned long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(add, f32, float);
  HIP_IMPL_GENERATE_SCAN_FUNC(add, f64, double);

  HIP_IMPL_GENERATE_SCAN_FUNC(min, i64, long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(min, u64, unsigned long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(min, f32, float);
  HIP_IMPL_GENERATE_SCAN_FUNC(min, f64, double);

  HIP_IMPL_GENERATE_SCAN_FUNC(max, i64, long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(max, u64, unsigned long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(max, f32, float);
  HIP_IMPL_GENERATE_SCAN_FUNC(max, f64, double);

  HIP_IMPL_GENERATE_SCAN_FUNC(and, i64, long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(and, u64, unsigned long long);

  HIP_IMPL_GENERATE_SCAN_FUNC(or, i64, long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(or, u64, unsigned long long);

  HIP_IMPL_GENERATE_SCAN_FUNC(xor, i64, long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(xor, u64, unsigned long long);

  // not all types could be used with wfscan (e.g. user defined types), this predicate
  // indicates whether that is the case
  template <typename T, typename = void>
  struct has_arithmetic_scan : __hip_internal::false_type {
  };

  // all the arithmetic operations accept the same types, so we just based it on the overload being
  // present for scan_add
  template <typename T>
  struct has_arithmetic_scan<T,
                 __hip_internal::void_t<decltype(scan_add<false>(T {}))>
    > : __hip_internal::true_type {
  };

  template <typename T, typename = void>
  struct has_boolean_scan : __hip_internal::false_type {
  };

  // all the arithmetic operations accept the same types, so we just based it on the overload being
  // present for scan_and
  template <typename T>
  struct has_boolean_scan<T,
                 __hip_internal::void_t<decltype(scan_and<false>(T {}))>
    > : __hip_internal::true_type {
  };

  // given cooperative_groups template parameter, calls the right impl::scan function (that would contain __ockl_wfscan_*)
  // and that is overloaded by type
  template <class TyVal, class Op, bool Inclusive>
  __CG_QUALIFIER__ TyVal call_scan(const TyVal& val)
  {
    using Val = typename __hip_internal::remove_cvref<TyVal>::type;

    if constexpr (__hip_internal::is_same<Op, cooperative_groups::plus<Val>>::value) {
      return impl::scan_add<Inclusive>(val);
    } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::less<Val>>::value) {
      return impl::scan_min<Inclusive>(val);
    } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::greater<Val>>::value) {
      return impl::scan_max<Inclusive>(val);
    } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::bit_and<Val>>::value) {
      return impl::scan_and<Inclusive>(val);
    } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::bit_or<Val>>::value) {
      return impl::scan_or<Inclusive>(val);
    } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::bit_xor<Val>>::value) {
      return impl::scan_xor<Inclusive>(val);
    }
  }

  template <bool Inclusive, typename TyGroup, typename TyVal, typename TyFn>
  __CG_QUALIFIER__ auto scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val))
  {
    using Op = typename __hip_internal::remove_cvref<TyFn>::type;
    using Val = typename __hip_internal::remove_cvref<TyVal>::type;

    constexpr bool isPrimitiveType = impl::has_arithmetic_scan<Val>::value;
    using permuteType = typename __hip_internal::conditional<isPrimitiveType && (sizeof(Val) == 4 || sizeof(Val) == 2), Val, unsigned int>::type;

    // the number of backward permutes will be: size(Val) / 4 rounded up
    static constexpr int kNumOfPermutes = (sizeof(Val) <= 4)?
                                          1 :
                                          (sizeof(Val) + sizeof(unsigned int) - 1) / sizeof(unsigned int);
    static_assert(cooperative_groups::impl::is_param_type_same<Val, decltype(op(val, val))>::value, "Operator input and output types differ");
    static_assert(__hip_internal::is_trivially_copyable<Val>::value, "val must be trivially copyable");
    static_assert(sizeof(Val) <= 32, "scan only operate on values of size up to 32 bytes");

    if constexpr (!cooperative_groups::impl::isTiledGroup<TyGroup>::value && !cooperative_groups::impl::isCoalescedGroup<TyGroup>::value) {
      static_assert(__hip_internal::is_void<TyGroup>::value, "This group does not exclusively represent a tile");
    }

    unsigned int maskNumBits;
    int numIterations;
    // next bit to aggregate with
    unsigned int maskIdx;
    unsigned int laneId = __lane_id();
    int nextBit = laneId;
    unsigned long long mask = ~0ull;

    mask = impl::groupMask(group);
    maskNumBits = __popcll(mask);
    maskIdx = __popcll(((1ull << laneId) - 1) & mask);

    if (laneId) {
      mask <<= 64 - laneId;
    }

#ifdef __OPTIMIZE__  // at the time of this writing the ockl wfscan functions do not compile when
                      // using -O0
    if (impl::isTiledGroup<TyGroup>::value) {
      // for tiled_groups we know at compile time that whether we can call the ockl intrinsics or
      // not; if the block tile is actually the whole warp
      if (impl::tiledGroupSize<TyGroup>::value == warpSize) {
        if constexpr (impl::isArithmeticFunc<Val, Op>::value && impl::has_arithmetic_scan<Val>::value ||
                      impl::isBooleanFunc<Val, Op>::value && impl::has_boolean_scan<Val>::value) {
          return impl::call_scan<Val, Op, Inclusive>(val);
        }
      }
    } else if constexpr (impl::isCoalescedGroup<TyGroup>::value) {
      // for the coalesced_group case we do need to check at runtime, adding a slight overhead on
      // this branch
      if (maskNumBits == warpSize) {
        if constexpr (impl::isArithmeticFunc<Val, Op>::value && impl::has_arithmetic_scan<Val>::value ||
                      impl::isBooleanFunc<Val, Op>::value && impl::has_boolean_scan<Val>::value) {
          return impl::call_scan<Val, Op, Inclusive>(val);
        }
      }
    }
#endif

    // unsigned int[N] is used in some cases, e.g. when T is wider than 32-bit
    using ResultType = typename __hip_internal::conditional<
                         isPrimitiveType && (sizeof(Val) == 4 || sizeof(Val) == 2), permuteType,
                         permuteType[kNumOfPermutes]>::type;
    static constexpr int alignment = alignof(Val) <= 4? 4 : alignof(Val);
    alignas(alignment) ResultType result;
    alignas(alignment) ResultType permuteResult;

    if constexpr (isPrimitiveType && (sizeof(Val) == 2 || sizeof(Val) == 4)) {
      result = val;
    } else {
      __builtin_memcpy(result, &val, sizeof(Val));
    }

    // the number of iterations needs to be at least log2(number of bits on)
    numIterations = sizeof(int) * 8 - __clz(maskNumBits);

    if constexpr (impl::isTiledGroup<TyGroup>::value) {
      // the number of bits in the mask is always a power of 2 when
      // tiled blocks are used and in that case we need an iteration less
      numIterations -= 1;
    } else {
      // in the coalesced_threads case it depends, we are not sure whether
      // it is a power of 2, or not so we check
      if (!(maskNumBits & (maskNumBits - 1))) {
        numIterations -= 1;
      }
    }

    int modulo = 1;

    while (numIterations) {
      int offset = modulo >> 1;
      int increment = modulo - offset;
      int nextPos = maskIdx - offset - increment;
      bool insideLanes = nextPos >= 0;

      if (insideLanes) {
        int next;

        // find the position to aggregate with
        for (int i = 0; i < increment; i++) {
          next = __builtin_clzll(mask) + 1;
          mask <<= next;
          nextBit -= next;
        }
      }

      // clamp index; if out of bounds, the thread read its own value
      nextBit = (nextBit < (laneId & ~(warpSize - 1))) ? laneId : nextBit;
      bPermute<isPrimitiveType, permuteType, kNumOfPermutes>(permuteResult, result, nextBit);

      if (insideLanes) {
        if constexpr (!isPrimitiveType) {
          Val toReturn;
          toReturn = op(*reinterpret_cast<Val*>(permuteResult), *reinterpret_cast<Val*>(result));
        __builtin_memcpy(result, &toReturn, sizeof(Val));
        } else if constexpr (sizeof(Val) == 4 || sizeof(Val) == 2) {
          result = op(permuteResult, result);
        } else if constexpr (sizeof(Val) == 8) {
          Val tmp;
          unsigned long long rhs =
              (static_cast<unsigned long long>(permuteResult[1]) << 32) | permuteResult[0];
          __builtin_memcpy(&tmp, result, sizeof(Val));
          tmp = op(*reinterpret_cast<Val*>(&rhs), tmp);
          __builtin_memcpy(result, &tmp, sizeof(Val));
        }
      }

      modulo <<= 1;
      numIterations--;
    }

    if constexpr (Inclusive) {
      return *reinterpret_cast<Val*>(&result);
    } else {
      int nextBit = laneId;

      mask = impl::groupMask(group);

      if (laneId) {
        mask <<= 64 - laneId;
        nextBit -= mask? __builtin_clzll(mask) + 1 : 0;
      } else {
        mask = 0ull;
      }

      // clamp index; if out of bounds, the thread read its own value
      nextBit = (nextBit < (laneId & ~(warpSize - 1))) ? laneId : nextBit;
      bPermute<isPrimitiveType, permuteType, kNumOfPermutes>(permuteResult, result, nextBit);

      if (mask) {
        return *reinterpret_cast<Val*>(&permuteResult);
      } else {
        impl::CGIdentity<Val, Op> identity;

        return identity();
      }
    }
  }
}

  template <typename TyGroup, typename TyVal, typename TyFn>
  __CG_QUALIFIER__ auto inclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op)
    -> decltype(op(val, val))
  {
    return impl::scan<true>(group, val, op);
  }

  template <typename TyGroup, typename TyVal, typename TyFn>
  __CG_QUALIFIER__ auto exclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op)
    -> decltype(op(val, val))
  {
    return impl::scan<false>(group, val, op);
  }

  template <typename TyGroup, typename TyVal>
  __CG_QUALIFIER__ auto inclusive_scan(const TyGroup& group, TyVal&& val)
  {
    using Val = typename __hip_internal::remove_cvref<TyVal>::type;

    return impl::scan<true>(group, val, cooperative_groups::plus<Val>());
  }

  template <typename TyGroup, typename TyVal>
  __CG_QUALIFIER__ auto exclusive_scan(const TyGroup& group, TyVal&& val)
  {
    using Val = typename __hip_internal::remove_cvref<TyVal>::type;

    return impl::scan<false>(group, val, cooperative_groups::plus<Val>());
  }
}
#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_SCAN_H
