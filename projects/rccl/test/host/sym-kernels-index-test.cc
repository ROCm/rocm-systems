/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for generated sym_kernels_host.cc: ncclSymkGetKernelIndex must route each emitted combo exactly.

#include <gtest/gtest.h>

#include <ostream>
#include <set>
#include <string>
#include <vector>

#include "sym_kernels.h"

// Neuter __global__: a real definition needs hipLaunchKernel/__hipRegisterFunction, unavailable under -no-hip-rt.
#undef __global__
#define __global__

#include "fakes/sym_kernels_test_stubs.h"

#include SYM_KERNELS_HOST_CC_PATH

#undef __global__

namespace {

struct ExpectedKernelCase {
  std::string name;
  ncclSymkKernelId id;
  int red;
  ncclDataType_t ty;
  void* expectedKernel;
  void* expectedProfileKernel;
};

std::ostream& operator<<(std::ostream& os, const ExpectedKernelCase& c) { return os << c.name; }

class SymKernelIndexValidTest : public ::testing::TestWithParam<ExpectedKernelCase> {};

TEST_P(SymKernelIndexValidTest, MapsToExactExpectedKernel) {
  const ExpectedKernelCase& c = GetParam();
  int index = ncclSymkGetKernelIndex(c.id, c.red, c.ty);
  ASSERT_GE(index, 0);
  ASSERT_LT(index, ncclSymkKernelCount);
  EXPECT_EQ(ncclSymkKernelList[index], c.expectedKernel);
  EXPECT_EQ(ncclSymkKernelListProfile[index], c.expectedProfileKernel);
}

// Both cases lists are read off generate.py's own kernel_list, the same list that produces the switch
// below: this suite proves the switch routes that list correctly, not that enumerate_kernels() chose it.
const std::vector<ExpectedKernelCase>& ValidCases() {
  static const std::vector<ExpectedKernelCase> kCases = {
      {"AllGather_LL", ncclSymkKernelId_AllGather_LL, ncclDevSum, ncclFloat32, (void*)ncclSymkDevKernel_AllGather_LL,
       (void*)ncclSymkDevKernel_AllGather_LL_profile},
      {"AllGather_ST", ncclSymkKernelId_AllGather_ST, ncclDevSum, ncclFloat32, (void*)ncclSymkDevKernel_AllGather_ST,
       (void*)ncclSymkDevKernel_AllGather_ST_profile},

      {"AllReduce_AGxLL_R_sum_f32", ncclSymkKernelId_AllReduce_AGxLL_R, ncclDevSum, ncclFloat32,
       (void*)ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f32, (void*)ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f32_profile},
      {"AllReduce_AGxLL_R_sum_f16", ncclSymkKernelId_AllReduce_AGxLL_R, ncclDevSum, ncclFloat16,
       (void*)ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f16, (void*)ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f16_profile},
      {"AllReduce_AGxLL_R_sum_bf16", ncclSymkKernelId_AllReduce_AGxLL_R, ncclDevSum, ncclBfloat16,
       (void*)ncclSymkDevKernel_AllReduce_AGxLL_R_sum_bf16,
       (void*)ncclSymkDevKernel_AllReduce_AGxLL_R_sum_bf16_profile},
      {"AllReduce_AGxLL_R_sum_f8e4m3", ncclSymkKernelId_AllReduce_AGxLL_R, ncclDevSum, ncclFloat8e4m3,
       (void*)ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f8e4m3,
       (void*)ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f8e4m3_profile},
      {"AllReduce_AGxLL_R_sum_f8e5m2", ncclSymkKernelId_AllReduce_AGxLL_R, ncclDevSum, ncclFloat8e5m2,
       (void*)ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f8e5m2,
       (void*)ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f8e5m2_profile},

      {"AllReduce_RSxLD_AGxST_sum_f32", ncclSymkKernelId_AllReduce_RSxLD_AGxST, ncclDevSum, ncclFloat32,
       (void*)ncclSymkDevKernel_AllReduce_RSxLD_AGxST_sum_f32,
       (void*)ncclSymkDevKernel_AllReduce_RSxLD_AGxST_sum_f32_profile},
      {"AllReduce_RSxLD_AGxST_sum_f16", ncclSymkKernelId_AllReduce_RSxLD_AGxST, ncclDevSum, ncclFloat16,
       (void*)ncclSymkDevKernel_AllReduce_RSxLD_AGxST_sum_f16,
       (void*)ncclSymkDevKernel_AllReduce_RSxLD_AGxST_sum_f16_profile},
      {"AllReduce_RSxLD_AGxST_sum_bf16", ncclSymkKernelId_AllReduce_RSxLD_AGxST, ncclDevSum, ncclBfloat16,
       (void*)ncclSymkDevKernel_AllReduce_RSxLD_AGxST_sum_bf16,
       (void*)ncclSymkDevKernel_AllReduce_RSxLD_AGxST_sum_bf16_profile},
      {"AllReduce_RSxLD_AGxST_sum_f8e4m3", ncclSymkKernelId_AllReduce_RSxLD_AGxST, ncclDevSum, ncclFloat8e4m3,
       (void*)ncclSymkDevKernel_AllReduce_RSxLD_AGxST_sum_f8e4m3,
       (void*)ncclSymkDevKernel_AllReduce_RSxLD_AGxST_sum_f8e4m3_profile},
      {"AllReduce_RSxLD_AGxST_sum_f8e5m2", ncclSymkKernelId_AllReduce_RSxLD_AGxST, ncclDevSum, ncclFloat8e5m2,
       (void*)ncclSymkDevKernel_AllReduce_RSxLD_AGxST_sum_f8e5m2,
       (void*)ncclSymkDevKernel_AllReduce_RSxLD_AGxST_sum_f8e5m2_profile},

      {"ReduceScatter_LL_sum_f32", ncclSymkKernelId_ReduceScatter_LL, ncclDevSum, ncclFloat32,
       (void*)ncclSymkDevKernel_ReduceScatter_LL_sum_f32, (void*)ncclSymkDevKernel_ReduceScatter_LL_sum_f32_profile},
      {"ReduceScatter_LL_sum_f16", ncclSymkKernelId_ReduceScatter_LL, ncclDevSum, ncclFloat16,
       (void*)ncclSymkDevKernel_ReduceScatter_LL_sum_f16, (void*)ncclSymkDevKernel_ReduceScatter_LL_sum_f16_profile},
      {"ReduceScatter_LL_sum_bf16", ncclSymkKernelId_ReduceScatter_LL, ncclDevSum, ncclBfloat16,
       (void*)ncclSymkDevKernel_ReduceScatter_LL_sum_bf16, (void*)ncclSymkDevKernel_ReduceScatter_LL_sum_bf16_profile},
      {"ReduceScatter_LL_sum_f8e4m3", ncclSymkKernelId_ReduceScatter_LL, ncclDevSum, ncclFloat8e4m3,
       (void*)ncclSymkDevKernel_ReduceScatter_LL_sum_f8e4m3,
       (void*)ncclSymkDevKernel_ReduceScatter_LL_sum_f8e4m3_profile},
      {"ReduceScatter_LL_sum_f8e5m2", ncclSymkKernelId_ReduceScatter_LL, ncclDevSum, ncclFloat8e5m2,
       (void*)ncclSymkDevKernel_ReduceScatter_LL_sum_f8e5m2,
       (void*)ncclSymkDevKernel_ReduceScatter_LL_sum_f8e5m2_profile},
      {"ReduceScatter_LL_avg_f32", ncclSymkKernelId_ReduceScatter_LL, ncclDevSumPostDiv, ncclFloat32,
       (void*)ncclSymkDevKernel_ReduceScatter_LL_avg_f32, (void*)ncclSymkDevKernel_ReduceScatter_LL_avg_f32_profile},
      {"ReduceScatter_LL_avg_f16", ncclSymkKernelId_ReduceScatter_LL, ncclDevSumPostDiv, ncclFloat16,
       (void*)ncclSymkDevKernel_ReduceScatter_LL_avg_f16, (void*)ncclSymkDevKernel_ReduceScatter_LL_avg_f16_profile},
      {"ReduceScatter_LL_avg_bf16", ncclSymkKernelId_ReduceScatter_LL, ncclDevSumPostDiv, ncclBfloat16,
       (void*)ncclSymkDevKernel_ReduceScatter_LL_avg_bf16, (void*)ncclSymkDevKernel_ReduceScatter_LL_avg_bf16_profile},
      {"ReduceScatter_LL_avg_f8e4m3", ncclSymkKernelId_ReduceScatter_LL, ncclDevSumPostDiv, ncclFloat8e4m3,
       (void*)ncclSymkDevKernel_ReduceScatter_LL_avg_f8e4m3,
       (void*)ncclSymkDevKernel_ReduceScatter_LL_avg_f8e4m3_profile},
      {"ReduceScatter_LL_avg_f8e5m2", ncclSymkKernelId_ReduceScatter_LL, ncclDevSumPostDiv, ncclFloat8e5m2,
       (void*)ncclSymkDevKernel_ReduceScatter_LL_avg_f8e5m2,
       (void*)ncclSymkDevKernel_ReduceScatter_LL_avg_f8e5m2_profile},

      {"ReduceScatter_LD_sum_f32", ncclSymkKernelId_ReduceScatter_LD, ncclDevSum, ncclFloat32,
       (void*)ncclSymkDevKernel_ReduceScatter_LD_sum_f32, (void*)ncclSymkDevKernel_ReduceScatter_LD_sum_f32_profile},
      {"ReduceScatter_LD_sum_f16", ncclSymkKernelId_ReduceScatter_LD, ncclDevSum, ncclFloat16,
       (void*)ncclSymkDevKernel_ReduceScatter_LD_sum_f16, (void*)ncclSymkDevKernel_ReduceScatter_LD_sum_f16_profile},
      {"ReduceScatter_LD_sum_bf16", ncclSymkKernelId_ReduceScatter_LD, ncclDevSum, ncclBfloat16,
       (void*)ncclSymkDevKernel_ReduceScatter_LD_sum_bf16, (void*)ncclSymkDevKernel_ReduceScatter_LD_sum_bf16_profile},
      {"ReduceScatter_LD_sum_f8e4m3", ncclSymkKernelId_ReduceScatter_LD, ncclDevSum, ncclFloat8e4m3,
       (void*)ncclSymkDevKernel_ReduceScatter_LD_sum_f8e4m3,
       (void*)ncclSymkDevKernel_ReduceScatter_LD_sum_f8e4m3_profile},
      {"ReduceScatter_LD_sum_f8e5m2", ncclSymkKernelId_ReduceScatter_LD, ncclDevSum, ncclFloat8e5m2,
       (void*)ncclSymkDevKernel_ReduceScatter_LD_sum_f8e5m2,
       (void*)ncclSymkDevKernel_ReduceScatter_LD_sum_f8e5m2_profile},
      {"ReduceScatter_LD_avg_f32", ncclSymkKernelId_ReduceScatter_LD, ncclDevSumPostDiv, ncclFloat32,
       (void*)ncclSymkDevKernel_ReduceScatter_LD_avg_f32, (void*)ncclSymkDevKernel_ReduceScatter_LD_avg_f32_profile},
      {"ReduceScatter_LD_avg_f16", ncclSymkKernelId_ReduceScatter_LD, ncclDevSumPostDiv, ncclFloat16,
       (void*)ncclSymkDevKernel_ReduceScatter_LD_avg_f16, (void*)ncclSymkDevKernel_ReduceScatter_LD_avg_f16_profile},
      {"ReduceScatter_LD_avg_bf16", ncclSymkKernelId_ReduceScatter_LD, ncclDevSumPostDiv, ncclBfloat16,
       (void*)ncclSymkDevKernel_ReduceScatter_LD_avg_bf16, (void*)ncclSymkDevKernel_ReduceScatter_LD_avg_bf16_profile},
      {"ReduceScatter_LD_avg_f8e4m3", ncclSymkKernelId_ReduceScatter_LD, ncclDevSumPostDiv, ncclFloat8e4m3,
       (void*)ncclSymkDevKernel_ReduceScatter_LD_avg_f8e4m3,
       (void*)ncclSymkDevKernel_ReduceScatter_LD_avg_f8e4m3_profile},
      {"ReduceScatter_LD_avg_f8e5m2", ncclSymkKernelId_ReduceScatter_LD, ncclDevSumPostDiv, ncclFloat8e5m2,
       (void*)ncclSymkDevKernel_ReduceScatter_LD_avg_f8e5m2,
       (void*)ncclSymkDevKernel_ReduceScatter_LD_avg_f8e5m2_profile},

      {"ReduceScatter_RailA2A_LsaLD_sum_f32", ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD, ncclDevSum, ncclFloat32,
       (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_f32,
       (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_f32_profile},
      {"ReduceScatter_RailA2A_LsaLD_sum_f16", ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD, ncclDevSum, ncclFloat16,
       (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_f16,
       (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_f16_profile},
      {"ReduceScatter_RailA2A_LsaLD_sum_bf16", ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD, ncclDevSum, ncclBfloat16,
       (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_bf16,
       (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_bf16_profile},
      {"ReduceScatter_RailA2A_LsaLD_sum_f8e4m3", ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD, ncclDevSum,
       ncclFloat8e4m3, (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_f8e4m3,
       (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_f8e4m3_profile},
      {"ReduceScatter_RailA2A_LsaLD_sum_f8e5m2", ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD, ncclDevSum,
       ncclFloat8e5m2, (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_f8e5m2,
       (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_f8e5m2_profile},
      {"ReduceScatter_RailA2A_LsaLD_avg_f32", ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD, ncclDevSumPostDiv,
       ncclFloat32, (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_avg_f32,
       (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_avg_f32_profile},
      {"ReduceScatter_RailA2A_LsaLD_avg_f16", ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD, ncclDevSumPostDiv,
       ncclFloat16, (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_avg_f16,
       (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_avg_f16_profile},
      {"ReduceScatter_RailA2A_LsaLD_avg_bf16", ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD, ncclDevSumPostDiv,
       ncclBfloat16, (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_avg_bf16,
       (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_avg_bf16_profile},
      {"ReduceScatter_RailA2A_LsaLD_avg_f8e4m3", ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD, ncclDevSumPostDiv,
       ncclFloat8e4m3, (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_avg_f8e4m3,
       (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_avg_f8e4m3_profile},
      {"ReduceScatter_RailA2A_LsaLD_avg_f8e5m2", ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD, ncclDevSumPostDiv,
       ncclFloat8e5m2, (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_avg_f8e5m2,
       (void*)ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_avg_f8e5m2_profile},
  };
  return kCases;
}

INSTANTIATE_TEST_SUITE_P(SymAllEmittedCombinations, SymKernelIndexValidTest, ::testing::ValuesIn(ValidCases()),
                         [](const ::testing::TestParamInfo<ExpectedKernelCase>& info) { return info.param.name; });

// A 43rd generated kernel would otherwise fail elsewhere (link error or the Python EXPECTED_TOTAL) unnoticed here.
TEST(SymKernelIndexValidCasesTest, CoversEveryGeneratedKernel) {
  EXPECT_EQ(static_cast<int>(ValidCases().size()), ncclSymkKernelCount);
}

// The row count alone would not catch a duplicate row silently displacing a distinct kernel's coverage.
TEST(SymKernelIndexValidCasesTest, EveryRowMapsToADistinctIndex) {
  std::set<int> indices;
  for (const ExpectedKernelCase& c : ValidCases()) {
    indices.insert(ncclSymkGetKernelIndex(c.id, c.red, c.ty));
  }
  EXPECT_EQ(static_cast<int>(indices.size()), ncclSymkKernelCount);
}

// AllGather's case returns before ever inspecting red/ty; garbage values here prove that, not just a lucky match.
TEST(SymKernelIndexNonReductionTest, AllGatherLL_IgnoresRedAndType_ReturnsAllGatherLLKernel) {
  int index = ncclSymkGetKernelIndex(ncclSymkKernelId_AllGather_LL, 12345, (ncclDataType_t)9999);
  ASSERT_GE(index, 0);
  EXPECT_EQ(ncclSymkKernelList[index], (void*)ncclSymkDevKernel_AllGather_LL);
}

TEST(SymKernelIndexNonReductionTest, AllGatherST_IgnoresRedAndType_ReturnsAllGatherSTKernel) {
  int index = ncclSymkGetKernelIndex(ncclSymkKernelId_AllGather_ST, 12345, (ncclDataType_t)9999);
  ASSERT_GE(index, 0);
  EXPECT_EQ(ncclSymkKernelList[index], (void*)ncclSymkDevKernel_AllGather_ST);
}

struct InvalidKernelCase {
  std::string name;
  ncclSymkKernelId id;
  int red;
  ncclDataType_t ty;
};

std::ostream& operator<<(std::ostream& os, const InvalidKernelCase& c) { return os << c.name; }

class SymKernelIndexInvalidTest : public ::testing::TestWithParam<InvalidKernelCase> {};

TEST_P(SymKernelIndexInvalidTest, ReturnsNegativeOne) {
  const InvalidKernelCase& c = GetParam();
  EXPECT_EQ(ncclSymkGetKernelIndex(c.id, c.red, c.ty), -1);
}

const std::vector<InvalidKernelCase>& InvalidCases() {
  static const std::vector<InvalidKernelCase> kCases = {
      // Outer switch(id) default: every real enumerator generate.py never emits a case for.
      {"UnimplementedId_AllReduce_AGxLLMC_R", ncclSymkKernelId_AllReduce_AGxLLMC_R, ncclDevSum, ncclFloat32},
      {"UnimplementedId_AllReduce_RSxTmaLD_AGxTmaST", ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST, ncclDevSum,
       ncclFloat32},
      {"UnimplementedId_AllReduce_RSxLDMC_AGxSTMC", ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC, ncclDevSum,
       ncclFloat32},
      {"UnimplementedId_AllGather_LLMC", ncclSymkKernelId_AllGather_LLMC, ncclDevSum, ncclFloat32},
      {"UnimplementedId_AllGather_TmaST", ncclSymkKernelId_AllGather_TmaST, ncclDevSum, ncclFloat32},
      {"UnimplementedId_AllGather_TmaSTMC", ncclSymkKernelId_AllGather_TmaSTMC, ncclDevSum, ncclFloat32},
      {"UnimplementedId_AllGather_STMC", ncclSymkKernelId_AllGather_STMC, ncclDevSum, ncclFloat32},
      {"UnimplementedId_AllGather_RailRing_LsaSTMC", ncclSymkKernelId_AllGather_RailRing_LsaSTMC, ncclDevSum,
       ncclFloat32},
      {"UnimplementedId_ReduceScatter_TmaLD", ncclSymkKernelId_ReduceScatter_TmaLD, ncclDevSum, ncclFloat32},
      {"UnimplementedId_ReduceScatter_LDMC", ncclSymkKernelId_ReduceScatter_LDMC, ncclDevSum, ncclFloat32},
      {"UnimplementedId_ReduceScatter_RailA2A_LsaLDMC", ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC, ncclDevSum,
       ncclFloat32},
      // Outer switch(id) default: the sentinel past the last real enumerator.
      {"UnimplementedId_Count", ncclSymkKernelId_Count, ncclDevSum, ncclFloat32},
      // Outer switch(id) default: an in-range value that is not an enumerator. Casting -1 would fall
      // outside the enum's [0,31] value range, which is undefined per C++17, not unspecified.
      {"UnimplementedId_InRangeNonEnumerator", (ncclSymkKernelId)31, ncclDevSum, ncclFloat32},

      // Inner switch(red) default: a real ncclDevRedOp_t this id has no case for.
      {"AllReduce_AGxLL_R_UnknownRed_Prod", ncclSymkKernelId_AllReduce_AGxLL_R, ncclDevProd, ncclFloat32},
      // Inner switch(red) default: AllReduce implements sum only (generate.py skips avg for it).
      {"AllReduce_AGxLL_R_UnsupportedRed_Avg", ncclSymkKernelId_AllReduce_AGxLL_R, ncclDevSumPostDiv, ncclFloat32},
      {"ReduceScatter_LL_UnknownRed_Prod", ncclSymkKernelId_ReduceScatter_LL, ncclDevProd, ncclFloat32},

      // Innermost switch(ty) default: a real ncclDataType_t this (id, red) has no case for.
      {"AllReduce_AGxLL_R_UnknownType_Int32", ncclSymkKernelId_AllReduce_AGxLL_R, ncclDevSum, ncclInt32},
      {"ReduceScatter_LD_UnknownType_Int64", ncclSymkKernelId_ReduceScatter_LD, ncclDevSum, ncclInt64},
      {"ReduceScatter_RailA2A_LsaLD_UnknownType_Float64", ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD,
       ncclDevSumPostDiv, ncclFloat64},
  };
  return kCases;
}

INSTANTIATE_TEST_SUITE_P(SymUnhandledCombinations, SymKernelIndexInvalidTest, ::testing::ValuesIn(InvalidCases()),
                         [](const ::testing::TestParamInfo<InvalidKernelCase>& info) { return info.param.name; });

}  // namespace
