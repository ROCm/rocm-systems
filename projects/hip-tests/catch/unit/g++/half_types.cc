#include <hip_test_common.hh>

bool test_fp16();
bool test_bf16();
bool test_half22float2();

HIP_TEST_CASE(Unit_gcc_half_types) {
  REQUIRE(test_fp16());
  REQUIRE(test_bf16());
  REQUIRE(test_half22float2());
}
