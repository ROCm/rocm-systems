/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <gtest/gtest.h>

#include "reshard_call_setup.h"
#include "reshard_internal.h"

namespace {

class ValidatedConfigTest : public ::testing::Test {
protected:
  bool savedExperimental = false;

  void SetUp() override {
    savedExperimental = gReshardEnableExperimental;
    gReshardEnableExperimental = false;
  }
  void TearDown() override { gReshardEnableExperimental = savedExperimental; }

  static ncclResult_t check(ReshardCopyAlgorithm algo, int ginType, int cudaDev = 0) {
    ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
    props.ginType = static_cast<ncclGinType_t>(ginType);
    props.cudaDev = cudaDev;
    return reshardCheckValidatedConfig("test", 0, algo, ncclSuccess, &props, cudaDev);
  }
};

TEST_F(ValidatedConfigTest, PackAndDirectWithProxyGinPass) {
  EXPECT_EQ(ncclSuccess, check(RESHARD_COPY_ALGO_PACK, NCCL_GIN_TYPE_PROXY));
  EXPECT_EQ(ncclSuccess, check(RESHARD_COPY_ALGO_DIRECT, NCCL_GIN_TYPE_PROXY));
  EXPECT_EQ(ncclSuccess, check(RESHARD_COPY_ALGO_PACK, NCCL_GIN_TYPE_NONE));
}

TEST_F(ValidatedConfigTest, DirectWithSeveralDevicesOnlyWarns) {
  EXPECT_EQ(ncclSuccess, check(RESHARD_COPY_ALGO_DIRECT, NCCL_GIN_TYPE_PROXY, 0));
  EXPECT_EQ(ncclSuccess, check(RESHARD_COPY_ALGO_DIRECT, NCCL_GIN_TYPE_PROXY, 1));
}

#if defined(__HIP_PLATFORM_AMD__)
TEST_F(ValidatedConfigTest, PipeNeedsExperimentalFlag) {
  EXPECT_EQ(ncclInvalidUsage, check(RESHARD_COPY_ALGO_PIPE, NCCL_GIN_TYPE_PROXY));
  gReshardEnableExperimental = true;
  EXPECT_EQ(ncclSuccess, check(RESHARD_COPY_ALGO_PIPE, NCCL_GIN_TYPE_PROXY));
}

TEST_F(ValidatedConfigTest, NonProxyGinNeedsExperimentalFlag) {
  EXPECT_EQ(ncclInvalidUsage, check(RESHARD_COPY_ALGO_PACK, NCCL_GIN_TYPE_ROCSHMEM_GDA));
  EXPECT_EQ(ncclInvalidUsage, check(RESHARD_COPY_ALGO_DIRECT, NCCL_GIN_TYPE_ANVIL_SDMA));
  gReshardEnableExperimental = true;
  EXPECT_EQ(ncclSuccess, check(RESHARD_COPY_ALGO_PACK, NCCL_GIN_TYPE_ROCSHMEM_GDA));
}
#endif

} // namespace
