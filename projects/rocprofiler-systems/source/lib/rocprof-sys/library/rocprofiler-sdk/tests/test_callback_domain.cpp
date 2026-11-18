// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback_domain.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace rocprofsys::domains
{
namespace
{

using ::testing::Eq;
using ::testing::StrictMock;

using test_support::callback_tracing_kind_t;
using test_support::callback_tracing_record_t;
using test_support::context_id_t;
using test_support::g_mock;
using test_support::gmock_sdk_backend;
using test_support::mock_sdk;
using test_support::tracing_operation_t;
using test_support::user_data_t;

void
stub_on_configure()
{}

void
stub_on_record(callback_tracing_record_t, user_data_t*, void*)
{}

using sut_t = callback_domain<mock_sdk>;

constexpr domain_id_t k_domain_id = 17;

callback_domain_definition<mock_sdk>
make_definition()
{
    return callback_domain_definition<mock_sdk>{ .meta =
                                                     domain_descriptor{
                                                         .name = "test_callback_domain",
                                                         .id   = k_domain_id,
                                                         .mode =
                                                             collection_mode::callback,
                                                         .group = std::nullopt,
                                                     },
                                                 .on_record    = &stub_on_record,
                                                 .on_configure = &stub_on_configure };
}

// NOLINTNEXTLINE(readability-identifier-naming)
class callback_domain_test : public ::testing::Test
{
protected:
    void SetUp() override { g_mock = std::make_unique<StrictMock<gmock_sdk_backend>>(); }
    void TearDown() override { g_mock.reset(); }
};

TEST_F(callback_domain_test, name_returns_definition_name)
{
    const sut_t domain{ make_definition(), context_id_t{ 3 }, {} };
    EXPECT_EQ(domain.name(), "test_callback_domain");
}

TEST_F(callback_domain_test,
       configure_calls_configure_callback_tracing_service_with_exact_arguments)
{
    const context_id_t context{ 3 };

    constexpr int                    k_op_count = 5;
    std::vector<tracing_operation_t> operations{ 4, k_op_count };
    auto* const                      ops_ptr  = operations.data();
    const auto                       ops_size = operations.size();

    sut_t domain{ make_definition(), context, std::move(operations) };

    EXPECT_CALL(*g_mock,
                configure_callback_tracing_service(
                    Eq(context), Eq(static_cast<callback_tracing_kind_t>(k_domain_id)),
                    Eq(ops_ptr), Eq(ops_size), Eq(&stub_on_record),
                    Eq(static_cast<void*>(nullptr))))
        .Times(1);

    domain.configure();
}

}  // namespace
}  // namespace rocprofsys::domains
