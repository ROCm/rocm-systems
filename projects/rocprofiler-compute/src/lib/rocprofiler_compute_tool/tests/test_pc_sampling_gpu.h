// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

// Must be defined before any <rocprofiler-sdk/...> include to expose PC-sampling types.
#define ROCPROFILER_SDK_EXPERIMENTAL

#include "sdk_wrapper.h"

#include <gtest/gtest.h>

#include <vector>

// GPU-gated integration fixture; the TEST_F self-skips on CPU-only CI.
class TestPcSamplingGpu : public ::testing::Test
{
protected:
    rocprofiler_compute_tool::SdkWrapperImpl sdk{};
};
