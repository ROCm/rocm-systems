// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/rccl/device_resolver.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"

#include <gtest/gtest.h>

namespace rocprofsys::domains::callback
{
namespace
{

using test_support::mock_sdk;

}  // namespace

TEST(rccl_device_id_resolver_test, resolve_device_id_defaults_to_zero_for_null_comm)
{
    const auto device_id = rccl::device_id_resolver<mock_sdk>::resolve_device_id(nullptr);

    EXPECT_EQ(device_id, 0U);
}

// Regression guard: without configure_comm_cu_device_function() having been called (no
// ncclCommCuDevice resolved via dlsym), a non-null comm must still default to device 0
// rather than dereferencing a null function pointer.
TEST(rccl_device_id_resolver_test,
     resolve_device_id_defaults_to_zero_when_symbol_unresolved)
{
    int fake_comm{};

    const auto device_id =
        rccl::device_id_resolver<mock_sdk>::resolve_device_id(&fake_comm);

    EXPECT_EQ(device_id, 0U);
}

}  // namespace rocprofsys::domains::callback
