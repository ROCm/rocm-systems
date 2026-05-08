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

extern "C" __device__ __attribute__((const)) int __ockl_wfscan_add_i32(int x, bool inclusive);


namespace cooperative_groups {
namespace impl {
  // these functions allow to make use of C++ function overloads, instead of having to code
  // a big if-constexpr according to operand type
  #define GENERATE_SCAN_FUNC(OP, TYPE_ALIAS, TYPE) \
  extern "C" __device__ __attribute__((const)) TYPE __ockl_wfscan_ ## OP ## _ ## TYPE_ALIAS(TYPE, bool);\
\
    template <bool Inclusive>\
    __device__ inline TYPE scan_ ## OP(TYPE val)\
    {\
      return __ockl_wfscan_ ## OP ## _ ## TYPE_ALIAS(val, Inclusive);\
    }

  GENERATE_SCAN_FUNC(add, i32, int);
  GENERATE_SCAN_FUNC(add, u32, unsigned int);

  GENERATE_SCAN_FUNC(min, i32, int);
  GENERATE_SCAN_FUNC(min, u32, unsigned int);

  GENERATE_SCAN_FUNC(max, i32, int);
  GENERATE_SCAN_FUNC(max, u32, unsigned int);

  GENERATE_SCAN_FUNC(and, i32, int);
  GENERATE_SCAN_FUNC(and, u32, unsigned int);

  GENERATE_SCAN_FUNC(or, i32, int);
  GENERATE_SCAN_FUNC(or, u32, unsigned int);

  GENERATE_SCAN_FUNC(xor, i32, int);
  GENERATE_SCAN_FUNC(xor, u32, unsigned int);

#ifdef HIP_ENABLE_EXTRA_WARP_SYNC_TYPES
  GENERATE_SCAN_FUNC(add, i64, long long);
  GENERATE_SCAN_FUNC(add, u64, unsigned long long);
  GENERATE_SCAN_FUNC(add, f32, float);
  GENERATE_SCAN_FUNC(add, f64, double);

  GENERATE_SCAN_FUNC(min, i64, long long);
  GENERATE_SCAN_FUNC(min, u64, unsigned long long);
  GENERATE_SCAN_FUNC(min, f32, float);
  GENERATE_SCAN_FUNC(min, f64, double);

  GENERATE_SCAN_FUNC(max, i64, long long);
  GENERATE_SCAN_FUNC(max, u64, unsigned long long);
  GENERATE_SCAN_FUNC(max, f32, float);
  GENERATE_SCAN_FUNC(max, f64, double);

  GENERATE_SCAN_FUNC(and, i64, long long);
  GENERATE_SCAN_FUNC(and, u64, unsigned long long);

  GENERATE_SCAN_FUNC(or, i64, long long);
  GENERATE_SCAN_FUNC(or, u64, unsigned long long);

  GENERATE_SCAN_FUNC(xor, i64, long long);
  GENERATE_SCAN_FUNC(xor, u64, unsigned long long);
#endif

  // not all types could be used with wfscan (e.g. user defined types), this predicate
  // indicates whether that is the case
  template <typename T, typename = void>
  struct has_scan_add : __hip_internal::false_type {
  };

  template <typename T>
  struct has_scan_add<T,
                 __hip_internal::void_t<decltype(scan_add<false>(T {}))>
    > : __hip_internal::true_type {
  };

  template <typename T, typename = void>
  struct has_scan_min : __hip_internal::false_type {
  };

  template <typename T>
  struct has_scan_min<T,
                 __hip_internal::void_t<decltype(scan_min<false>(T {}))>
    > : __hip_internal::true_type {
  };
}

template <typename TyGroup, typename TyVal, typename TyFn>
__CG_QUALIFIER__ auto inclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
  using Op = typename __hip_internal::remove_cvref<TyFn>::type;
  using Val = typename __hip_internal::remove_cvref<TyVal>::type;
  // TODO g-h-c we might want to change this definition
  static constexpr bool isPrimitiveType = impl::has_scan_add<Val>::value;
  using permuteType = typename __hip_internal::conditional<isPrimitiveType && (sizeof(Val) == 4 || sizeof(Val) == 2), Val, unsigned int>::type;

  // the number of backward permutes will be: size(Val) / 4 rounded up
  static constexpr int kNumOfPermutes = (sizeof(Val) < 4)?
                                        1 :
                                        (sizeof(Val) + sizeof(unsigned int) - 1) / sizeof(unsigned int);
  static_assert(cooperative_groups::impl::is_param_type_same<Val, decltype(op(val, val))>::value, "Operator input and output types differ");
  static_assert(__hip_internal::is_trivially_copyable<Val>::value, "val must be trivially copyable");
  static_assert(sizeof(Val) <= 32, "scan only operate on values of size up to 32 bytes");

  if constexpr (!cooperative_groups::impl::isTiledGroup<TyGroup>::value && !cooperative_groups::impl::isCoalescedGroup<TyGroup>::value) {
    static_assert(__hip_internal::is_void<TyGroup>::value, "This group does not exclusively represent a tile");
  }

  // TODO g-h-c we should be able to calculate the mask AT COMPILE TIME, and thusly
  // do a loop unroll at compile time but only IF THE GROUP IS A BLOCK TILE of a compile-time size
  unsigned int maskNumBits;
  int numIterations;
  // next bit to aggregate with
  unsigned int maskIdx;
  unsigned int laneId = __lane_id();
  int nextBit = laneId;
  unsigned long long mask = ~0ull;

  // we cannot simply just use the __activemask() here, because more than one tile could have active
  // threads at a time; we need to mask away the threads that not part of this tile first
  if constexpr (!__hip_internal::is_same<TyGroup, cooperative_groups::coalesced_group>::value) {
    mask >>= (64 - group.num_threads());
    mask <<= (((threadIdx.x % warpSize) / group.num_threads()) * group.num_threads());
  }

  // for coalesced_groups, the mask is simply the activemask
  // for tiled groups, it is legal for some threads in a tile to not participate so we also
  // need to apply the active mask
  mask &= __activemask();
  maskNumBits = __popcll(mask);
  maskIdx = __popcll(((1ul << laneId) - 1) & mask);

  if (laneId) {
    mask <<= 64 - laneId;
  }

#ifdef __OPTIMIZE__  // at the time of this writing the ockl wfscan functions do not compile when
                     // using -O0
  if (impl::isTiledGroup<TyGroup>::value) {
    // for tiled_groups we know at compile time that whether we can call the ockl intrinsics or
    // not; if the block tile is actually the whole warp
    if (impl::tiledGroupSize<TyGroup>::value == warpSize) {
      if constexpr (__hip_internal::is_same<Op, cooperative_groups::plus<Val>>::value &&
                    impl::has_scan_add<Val>::value) {
        return impl::scan_add<true>(val);
      } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::less<Val>>::value &&
                    impl::has_scan_min<Val>::value) {
        return impl::scan_min<true>(val);
      }
    }
  } else if constexpr (__hip_internal::is_same<TyGroup, cooperative_groups::coalesced_group>::value) {
    // for the coalesced_group case we do need to check at runtime
    if (maskNumBits == warpSize) {
      if constexpr (__hip_internal::is_same<Op, cooperative_groups::plus<Val>>::value &&
                    impl::has_scan_add<Val>::value) {
        return impl::scan_add<true>(val);
      } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::less<Val>>::value &&
                    impl::has_scan_min<Val>::value) {
        return impl::scan_min<true>(val);
      }
    }
  }
#endif

  
  // unsigned int[N] is used in some cases, e.g. when T is wider than 32-bit
  typename __hip_internal::conditional<isPrimitiveType && (sizeof(Val) == 4 || sizeof(Val) == 2), permuteType,
                                       permuteType[kNumOfPermutes]>::type result, permuteResult;
  auto backwardPermute = [](int index, permuteType arg) {
    if constexpr (__hip_internal::is_floating_point<Val>::value &&
                sizeof(Val) <= 4) {
      return __hip_ds_bpermutef(index, arg);
    } else {
      return __hip_ds_bpermute(index, arg);
    }
  };

  if constexpr (isPrimitiveType && (sizeof(Val) == 2 || sizeof(Val) == 4)) {
    result = val;
  } else {
    __builtin_memcpy(result, &val, sizeof(result));
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

      // find the position to aggregate with; although we could just call fns64() that will probably
      // be very slow when called multiple times in this for loop; this is equivalent
      for (int i = 0; i < increment; i++) {
        next = __builtin_clzll(mask) + 1;
        mask <<= next;
        nextBit -= next;
      }
    }

    // clamp index; if out of bounds, the thread read its own value
    nextBit = (nextBit < (laneId & ~(warpSize - 1))) ? laneId : nextBit;

    if constexpr (!isPrimitiveType) {
      // ds_bpermute only deals with 32-bit sizes, so for other sizes
      // we need to call the permute multiple times
      for (int i = 0; i < kNumOfPermutes; i++) {
        permuteResult[i] = backwardPermute(nextBit << 2, result[i]);
      }
    } else if constexpr (sizeof(Val) == 2) {
      union {
        int i;
        Val f;
      } tmp;

      tmp.f = result;
      tmp.i = __hip_ds_bpermute(nextBit << 2, tmp.i);
      permuteResult = tmp.f;
    } else if constexpr (sizeof(Val) == 4) {
      auto bPermuteResult = backwardPermute(nextBit << 2, result);
      __builtin_memcpy(&permuteResult, &bPermuteResult, sizeof(result));
    } else {
      // ds_bpermute only deals with 32-bit sizes, so for 8 bytes, we
      // need to call it twice
      permuteResult[0] = backwardPermute(nextBit << 2, result[0]);
      permuteResult[1] = backwardPermute(nextBit << 2, result[1]);
    }

    if (insideLanes) {
      if constexpr (!isPrimitiveType) {
        Val toReturn;
        toReturn = op(*reinterpret_cast<Val*>(result), *reinterpret_cast<Val*>(permuteResult));
      __builtin_memcpy(result, &toReturn, sizeof(Val));
      } else if constexpr (sizeof(Val) == 4 || sizeof(Val) == 2) {
        result = op(result, permuteResult);
      }  if constexpr (sizeof(Val) == 8) {
        Val tmp;
        unsigned long long rhs =
            (static_cast<unsigned long long>(permuteResult[1]) << 32) | permuteResult[0];
       __builtin_memcpy(&tmp, result, sizeof(result));
        tmp = op(tmp, *reinterpret_cast<Val*>(&rhs));
        __builtin_memcpy(result, &tmp, sizeof(result));
      }
    }

    modulo <<= 1;
    numIterations--;
  }
  return *reinterpret_cast<Val*>(&result);
}
}
#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_SCAN_H
