// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/buffered_domain.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace rocprofsys::domains
{
namespace
{

using ::testing::DoAll;
using ::testing::Eq;
using ::testing::InSequence;
using ::testing::NotNull;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrictMock;

using test_support::buffer_id_t;
using test_support::buffer_tracing_kind_t;
using test_support::callback_thread_id_t;
using test_support::context_id_t;
using test_support::g_mock;
using test_support::gmock_sdk_backend;
using test_support::mock_sdk;
using test_support::record_header_t;
using test_support::tracing_operation_t;

void
stub_on_records(context_id_t, buffer_id_t, record_header_t**, std::size_t, void*,
                std::uint64_t)
{}

using sut_t = buffered_domain<mock_sdk>;

constexpr domain_id_t k_domain_id = 42;

buffered_domain_definition<mock_sdk>
make_definition()
{
    return buffered_domain_definition<mock_sdk>{
        .meta =
            domain_descriptor{
                .name  = "test_domain",
                .id    = k_domain_id,
                .mode  = collection_mode::buffered,
                .group = std::nullopt,
            },
        .on_records   = &stub_on_records,
        .on_configure = nullptr,
    };
}

// NOLINTNEXTLINE(readability-identifier-naming)
class buffered_domain_test : public ::testing::Test
{
protected:
    void SetUp() override { g_mock = std::make_unique<StrictMock<gmock_sdk_backend>>(); }
    void TearDown() override { g_mock.reset(); }

    // Sets up the strict, ordered expectations for one configure() call and returns
    // the buffer id that create_buffer() will report back through its out-parameter.
    // NOLINTNEXTLINE(readability-function-size)
    void expect_configure(const context_id_t& context, const buffer_id_t& buffer,
                          const callback_thread_id_t& thread,
                          tracing_operation_t* ops_ptr, std::size_t ops_size)
    {
        InSequence seq;

        EXPECT_CALL(
            *g_mock,
            create_buffer(Eq(context),
                          Eq(k_default_buffer_properties.buffer_size.to_bytes()),
                          Eq(k_default_buffer_properties.buffer_watermark.to_bytes()),
                          Eq(mock_sdk::BUFFER_POLICY_LOSSLESS), Eq(&stub_on_records),
                          Eq(static_cast<void*>(nullptr)), NotNull()))
            .Times(1)
            // NOLINTNEXTLINE(readability-magic-numbers)
            .WillOnce(DoAll(SetArgPointee<6>(buffer), Return()));

        EXPECT_CALL(*g_mock,
                    configure_buffer_tracing_service(
                        Eq(context), Eq(static_cast<buffer_tracing_kind_t>(k_domain_id)),
                        Eq(ops_ptr), Eq(ops_size), Eq(buffer)))
            .Times(1);

        EXPECT_CALL(*g_mock, create_callback_thread(NotNull()))
            .Times(1)
            .WillOnce(DoAll(SetArgPointee<0>(thread), Return()));

        EXPECT_CALL(*g_mock, assign_callback_thread(Eq(buffer), Eq(thread))).Times(1);
    }
};

constexpr int k_test_context_id = 7;

TEST_F(buffered_domain_test, name_returns_definition_name)
{
    const sut_t domain{ make_definition(), context_id_t{ k_test_context_id }, {} };
    EXPECT_EQ(domain.name(), "test_domain");
}

TEST_F(buffered_domain_test, buffer_id_is_default_before_configure)
{
    const sut_t domain{ make_definition(), context_id_t{ k_test_context_id }, {} };
    EXPECT_EQ(domain.buffer_id(), buffer_id_t{});
}

TEST_F(buffered_domain_test,
       configure_creates_buffer_configures_tracing_and_assigns_callback_thread)
{
    const context_id_t         context{ 7 };
    const buffer_id_t          buffer{ 99 };
    const callback_thread_id_t thread{ 5 };

    std::vector<tracing_operation_t> operations{ 1, 2, 3 };
    auto* const                      ops_ptr  = operations.data();
    const auto                       ops_size = operations.size();

    sut_t domain{ make_definition(), context, std::move(operations) };

    expect_configure(context, buffer, thread, ops_ptr, ops_size);
    domain.configure();

    EXPECT_EQ(domain.buffer_id(), buffer);

    EXPECT_CALL(*g_mock, destroy_buffer(Eq(buffer))).Times(1).WillOnce(Return(0));
}

TEST_F(buffered_domain_test, flush_does_not_call_flush_buffer_when_never_configured)
{
    const sut_t domain{ make_definition(), context_id_t{ k_test_context_id }, {} };
    domain.flush();
}

TEST_F(buffered_domain_test, flush_calls_flush_buffer_when_configured)
{
    const context_id_t         context{ 7 };
    const buffer_id_t          buffer{ 99 };
    const callback_thread_id_t thread{ 5 };

    std::vector<tracing_operation_t> operations{ 1 };
    auto* const                      ops_ptr  = operations.data();
    const auto                       ops_size = operations.size();

    sut_t domain{ make_definition(), context, std::move(operations) };
    expect_configure(context, buffer, thread, ops_ptr, ops_size);
    domain.configure();

    EXPECT_CALL(*g_mock, flush_buffer(Eq(buffer))).Times(1);
    domain.flush();

    EXPECT_CALL(*g_mock, destroy_buffer(Eq(buffer))).Times(1).WillOnce(Return(0));
}

TEST_F(buffered_domain_test, destructor_does_not_destroy_buffer_when_never_configured)
{
    const sut_t domain{ make_definition(), context_id_t{ k_test_context_id }, {} };
}

TEST_F(buffered_domain_test, destructor_destroys_buffer_when_configured)
{
    const context_id_t         context{ 7 };
    const buffer_id_t          buffer{ 99 };
    const callback_thread_id_t thread{ 5 };

    std::vector<tracing_operation_t> operations{ 1 };
    auto* const                      ops_ptr  = operations.data();
    const auto                       ops_size = operations.size();

    {
        sut_t domain{ make_definition(), context, std::move(operations) };
        expect_configure(context, buffer, thread, ops_ptr, ops_size);
        domain.configure();

        EXPECT_CALL(*g_mock, destroy_buffer(Eq(buffer))).Times(1).WillOnce(Return(0));
    }
}

TEST_F(buffered_domain_test, move_construction_transfers_buffer_ownership)
{
    const context_id_t         context{ 7 };
    const buffer_id_t          buffer{ 99 };
    const callback_thread_id_t thread{ 5 };

    std::vector<tracing_operation_t> operations{ 1 };
    auto* const                      ops_ptr  = operations.data();
    const auto                       ops_size = operations.size();

    sut_t domain_a{ make_definition(), context, std::move(operations) };
    expect_configure(context, buffer, thread, ops_ptr, ops_size);
    domain_a.configure();

    const sut_t domain_b{ std::move(domain_a) };

    // NOLINTNEXTLINE(bugprone-use-after-move) verifying the post-move state is empty
    EXPECT_EQ(domain_a.buffer_id(), buffer_id_t{});
    EXPECT_EQ(domain_b.buffer_id(), buffer);

    EXPECT_CALL(*g_mock, destroy_buffer(Eq(buffer))).Times(1).WillOnce(Return(0));
}

TEST_F(buffered_domain_test,
       move_assignment_destroys_target_buffer_then_takes_over_source)
{
    const context_id_t         context_a{ 7 };
    const context_id_t         context_b{ 8 };
    const buffer_id_t          buffer_a{ 11 };
    const buffer_id_t          buffer_b{ 22 };
    const callback_thread_id_t thread_a{ 5 };
    const callback_thread_id_t thread_b{ 6 };

    std::vector<tracing_operation_t> operations_a{ 1 };
    std::vector<tracing_operation_t> operations_b{ 2 };
    auto* const                      ops_a_ptr  = operations_a.data();
    const auto                       ops_a_size = operations_a.size();
    auto* const                      ops_b_ptr  = operations_b.data();
    const auto                       ops_b_size = operations_b.size();

    sut_t domain_a{ make_definition(), context_a, std::move(operations_a) };
    sut_t domain_b{ make_definition(), context_b, std::move(operations_b) };

    expect_configure(context_a, buffer_a, thread_a, ops_a_ptr, ops_a_size);
    domain_a.configure();

    expect_configure(context_b, buffer_b, thread_b, ops_b_ptr, ops_b_size);
    domain_b.configure();

    {
        InSequence seq;
        EXPECT_CALL(*g_mock, destroy_buffer(Eq(buffer_b))).Times(1).WillOnce(Return(0));
        EXPECT_CALL(*g_mock, destroy_buffer(Eq(buffer_a))).Times(1).WillOnce(Return(0));
    }

    domain_b = std::move(domain_a);

    // NOLINTNEXTLINE(bugprone-use-after-move) verifying the post-move state is empty
    EXPECT_EQ(domain_a.buffer_id(), buffer_id_t{});
    EXPECT_EQ(domain_b.buffer_id(), buffer_a);
}

}  // namespace
}  // namespace rocprofsys::domains
