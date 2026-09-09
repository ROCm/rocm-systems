// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback_domain.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace rocprofsys::domains
{
namespace
{

using ::testing::Eq;
using ::testing::StrictMock;

// Self-contained stand-in for SdkBackend: callback_domain<SdkBackend> only ever
// touches these members.
struct context_id_t
{
    std::uint64_t handle                                 = 0;
    auto          operator<=>(const context_id_t&) const = default;
};

struct user_data_t
{
    std::uint64_t value = 0;
};

struct callback_tracing_record_t
{
    std::uint64_t kind = 0;
};

using tracing_operation_t     = std::size_t;
using callback_tracing_kind_t = std::size_t;
using on_record_cb_t          = void (*)(callback_tracing_record_t, user_data_t*, void*);

void
stub_on_record(callback_tracing_record_t, user_data_t*, void*)
{}

struct gmock_sdk_backend
{
    MOCK_METHOD(void, configure_callback_tracing_service,
                (context_id_t context, callback_tracing_kind_t kind,
                 tracing_operation_t* operations, std::size_t num_operations,
                 on_record_cb_t on_record, void* callback_data));
};

std::unique_ptr<StrictMock<gmock_sdk_backend>> g_mock;

struct mock_sdk
{
    using context_id_t              = rocprofsys::domains::context_id_t;
    using user_data_t               = rocprofsys::domains::user_data_t;
    using callback_tracing_record_t = rocprofsys::domains::callback_tracing_record_t;
    using tracing_operation_t       = rocprofsys::domains::tracing_operation_t;
    using callback_tracing_kind_t   = rocprofsys::domains::callback_tracing_kind_t;

    static void configure_callback_tracing_service(context_id_t            context,
                                                   callback_tracing_kind_t kind,
                                                   tracing_operation_t*    operations,
                                                   std::size_t             num_operations,
                                                   on_record_cb_t          on_record,
                                                   void*                   callback_data)
    {
        g_mock->configure_callback_tracing_service(
            context, kind, operations, num_operations, on_record, callback_data);
    }
};

using sut_t = callback_domain<mock_sdk>;

constexpr domain_id_t k_domain_id = 17;

callback_domain_definition<mock_sdk>
make_definition()
{
    return callback_domain_definition<mock_sdk>{
        .meta =
            domain_descriptor{
                .name  = "test_callback_domain",
                .id    = k_domain_id,
                .mode  = collection_mode::callback,
                .group = std::nullopt,
            },
        .on_record = &stub_on_record,
    };
}

class callback_domain_test : public ::testing::Test
{
protected:
    void SetUp() override { g_mock = std::make_unique<StrictMock<gmock_sdk_backend>>(); }
    void TearDown() override { g_mock.reset(); }
};

TEST_F(callback_domain_test, name_returns_definition_name)
{
    sut_t domain{ make_definition(), context_id_t{ 3 }, {} };
    EXPECT_EQ(domain.name(), "test_callback_domain");
}

TEST_F(callback_domain_test,
       configure_calls_configure_callback_tracing_service_with_exact_arguments)
{
    const context_id_t context{ 3 };

    std::vector<tracing_operation_t> operations{ 4, 5 };
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
