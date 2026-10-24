// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/input.h"

#include <gtest/gtest.h>

#include <limits>

namespace rocjitsu::consan {
namespace {

TEST(ConSanRowLayout, SiteReservationsIncludeEveryAddressGroupAndTheRowHeader) {
  InputError error;
  for (size_t sites : {size_t{0}, size_t{1}, size_t{2}, size_t{ContractLimits{}.sites}}) {
    const auto stride = lds_race_row_stride(sites, error);
    ASSERT_TRUE(stride.succeeded()) << error.message;
    EXPECT_EQ(stride.value(), kRowHeaderBytes + sites * kAddressGroupsPerSite * kAccessRecordBytes);
  }
}

TEST(ConSanRowLayout, OversizedSiteCountsFailBeforeMultiplicationOrNarrowing) {
  for (const size_t sites :
       {size_t{ContractLimits{}.sites} + 1, std::numeric_limits<size_t>::max()}) {
    InputError error;
    EXPECT_TRUE(lds_race_row_stride(sites, error).failed());
    EXPECT_EQ(error.code, InputErrorCode::ResourceLimit);
    EXPECT_EQ(error.message, "no bounded LDS report-row layout");
  }
}

} // namespace
} // namespace rocjitsu::consan
