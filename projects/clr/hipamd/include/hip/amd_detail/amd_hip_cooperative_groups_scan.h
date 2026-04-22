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
template <typename TyGroup, typename TyVal, typename TyFn>
__CG_QUALIFIER__ auto inclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
  using Op = typename __hip_internal::remove_cvref<TyFn>::type;
  using Val = typename __hip_internal::remove_cvref<TyVal>::type;
  using permuteType = int; //typename __hip_internal::conditional<isPrimitiveType && (sizeof(Val) == 4 || sizeof(Val) == 2), Val, unsigned int>::type;

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

#ifdef __OPTIMIZE__  // at the time of this writing the ockl wfred functions do not compile when
                     // using -O0
  if (maskNumBits == warpSize) {
    return __ockl_wfscan_add_i32(val, true);
  }
#endif

  if constexpr (__hip_internal::is_same<Op, cooperative_groups::plus<Val>>::value || 
                sizeof(Val) > 4) {
    Val result = val;
    Val rhs;
    auto backwardPermute = [](int index, permuteType arg) {
      if constexpr (__hip_internal::is_floating_point<Val>::value &&
                  sizeof(Val) <= 4) {
        return __hip_ds_bpermutef(index, arg);
      } else {
        return __hip_ds_bpermute(index, arg);
      }
    };

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
      rhs = backwardPermute(nextBit << 2, result);

      if (insideLanes) {
        result = op(result, rhs);
      }

      modulo <<= 1;
      numIterations--;
    }
    return result;
  } else {
    assert(false && "Unimplemented");
  }
}
}
#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_SCAN_H
