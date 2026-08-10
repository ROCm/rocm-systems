#include <gtest/gtest.h>

#include <type_traits>

#include "rccl_float8.h"
#include "symmetric/primitives.cuh"

namespace RcclUnitTesting {

TEST(SymmetricAccumTypeTraits, HipSumAccumulatesFp8InFloat) {
  static_assert(std::is_same_v<ncclSymkAccumType<FuncSum, rccl_float8, false>::Type, float>,
                "HIP fp8 sum must accumulate in float");
  static_assert(std::is_same_v<ncclSymkAccumType<FuncSum, rccl_bfloat8, false>::Type, float>,
                "HIP bf8 sum must accumulate in float");
  static_assert(std::is_same_v<ncclSymkAccumType<FuncSum, hip_bfloat16, false>::Type, float>,
                "HIP bf16 sum must accumulate in float");

  // Runtime expectations keep this visible in test output, while the static_assert
  // guards fail at compile-time if a specialization regresses.
  EXPECT_TRUE((std::is_same_v<ncclSymkAccumType<FuncSum, rccl_float8, false>::Type, float>));
  EXPECT_TRUE((std::is_same_v<ncclSymkAccumType<FuncSum, rccl_bfloat8, false>::Type, float>));
  EXPECT_TRUE((std::is_same_v<ncclSymkAccumType<FuncSum, hip_bfloat16, false>::Type, float>));
}

}  // namespace RcclUnitTesting
