// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "torch_abi/runtime.h"
#include "torch_abi/torch_abi.h"

#include <gtest/gtest.h>

namespace
{

TEST(TorchAbiRuntimeTest, AcceptsEveryValidatedLibraryPair)
{
    for (const auto& runtime : torch_abi::kSupportedRuntimes)
    {
        EXPECT_TRUE(
            torch_abi::runtime_pair_is_supported(runtime.torch_cpu_build_id, runtime.c10_build_id));
    }
}

TEST(TorchAbiRuntimeTest, RejectsCrossedLibraryPair)
{
    ASSERT_GE(torch_abi::kSupportedRuntimes.size(), 2u);

    EXPECT_FALSE(torch_abi::runtime_pair_is_supported(torch_abi::kSupportedRuntimes[0].torch_cpu_build_id,
                                                      torch_abi::kSupportedRuntimes[1].c10_build_id));
}

TEST(TorchAbiRuntimeTest, RejectsUnknownLibraryPair)
{
    const torch_abi::GnuBuildId unknown_build_id{};

    EXPECT_FALSE(torch_abi::runtime_pair_is_supported(unknown_build_id, unknown_build_id));
}

}  // namespace
