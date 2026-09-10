// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/buffered_domain.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <compare>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
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

// Self-contained stand-in for SdkBackend: buffered_domain<SdkBackend> only ever
// touches these members.
struct context_id_t
{
    std::uint64_t handle                                 = 0;
    auto          operator<=>(const context_id_t&) const = default;
};

struct buffer_id_t
{
    std::uint64_t handle                                = 0;
    auto          operator<=>(const buffer_id_t&) const = default;
};

struct callback_thread_id_t
{
    std::uint64_t handle                                         = 0;
    auto          operator<=>(const callback_thread_id_t&) const = default;
};

struct record_header_t
{
    void* payload = nullptr;
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
using buffer_tracing_kind_t   = std::size_t;
using callback_tracing_kind_t = std::size_t;
using buffer_policy_t         = int;
using on_records_cb_t         = void (*)(context_id_t, buffer_id_t, record_header_t**,
                                 std::size_t, void*, std::uint64_t);
using on_record_cb_t          = void (*)(callback_tracing_record_t, user_data_t*, void*);

void
stub_on_records(context_id_t, buffer_id_t, record_header_t**, std::size_t, void*,
                std::uint64_t)
{}

struct gmock_sdk_backend
{
    MOCK_METHOD(void, create_buffer,
                (context_id_t context, std::size_t buffer_size,
                 std::size_t buffer_watermark, int policy, on_records_cb_t callback,
                 void* callback_data, buffer_id_t* buffer_out));
    MOCK_METHOD(void, configure_buffer_tracing_service,
                (context_id_t context, buffer_tracing_kind_t kind,
                 tracing_operation_t* operations, std::size_t num_operations,
                 buffer_id_t buffer));
    MOCK_METHOD(void, create_callback_thread, (callback_thread_id_t * thread));
    MOCK_METHOD(void, assign_callback_thread,
                (buffer_id_t buffer, callback_thread_id_t thread));
    MOCK_METHOD(void, flush_buffer, (buffer_id_t buffer));
    MOCK_METHOD(int, destroy_buffer, (buffer_id_t buffer));
};

std::unique_ptr<StrictMock<gmock_sdk_backend>> g_mock;

struct mock_sdk
{
    using context_id_t              = rocprofsys::domains::context_id_t;
    using buffer_id_t               = rocprofsys::domains::buffer_id_t;
    using tracing_operation_t       = rocprofsys::domains::tracing_operation_t;
    using callback_thread_id_t      = rocprofsys::domains::callback_thread_id_t;
    using record_header_t           = rocprofsys::domains::record_header_t;
    using buffer_tracing_kind_t     = rocprofsys::domains::buffer_tracing_kind_t;
    using callback_tracing_kind_t   = rocprofsys::domains::callback_tracing_kind_t;
    using user_data_t               = rocprofsys::domains::user_data_t;
    using callback_tracing_record_t = rocprofsys::domains::callback_tracing_record_t;
    using buffer_policy_t           = rocprofsys::domains::buffer_policy_t;
    using on_record_cb_t            = rocprofsys::domains::on_record_cb_t;
    using on_records_cb_t           = rocprofsys::domains::on_records_cb_t;

    static constexpr buffer_policy_t BUFFER_POLICY_LOSSLESS = 1;

    // domain_service_backend requires these; buffered_domain<SdkBackend> never calls
    // them, so the bodies are unreachable no-ops.
    static void create_context(context_id_t* /*context*/) {}
    static void start_context(context_id_t /*context*/) {}
    static void configure_callback_tracing_service(context_id_t /*context*/,
                                                   callback_tracing_kind_t /*kind*/,
                                                   tracing_operation_t* /*operations*/,
                                                   std::size_t /*num_operations*/,
                                                   on_record_cb_t /*on_record*/,
                                                   void* /*callback_data*/)
    {}

    struct tracing_names_t
    {
        struct entry_t
        {
            std::string_view              name;
            std::vector<std::string_view> operations;
            std::size_t                   value = 0;
        };

        std::vector<entry_t> entries;

        auto begin() const { return entries.begin(); }
        auto end() const { return entries.end(); }

        std::string_view at(std::size_t /*kind*/, std::uint32_t /*operation*/) const
        {
            return "operation";
        }
    };

    static tracing_names_t get_buffer_tracing_names() { return {}; }
    static tracing_names_t get_callback_tracing_names() { return {}; }

    static void create_buffer(context_id_t context, std::size_t buffer_size,
                              std::size_t buffer_watermark, int policy,
                              on_records_cb_t callback, void* callback_data,
                              buffer_id_t* buffer_out)
    {
        g_mock->create_buffer(context, buffer_size, buffer_watermark, policy, callback,
                              callback_data, buffer_out);
    }

    static void configure_buffer_tracing_service(context_id_t          context,
                                                 buffer_tracing_kind_t kind,
                                                 tracing_operation_t*  operations,
                                                 std::size_t           num_operations,
                                                 buffer_id_t           buffer)
    {
        g_mock->configure_buffer_tracing_service(context, kind, operations,
                                                 num_operations, buffer);
    }

    static void create_callback_thread(callback_thread_id_t* thread)
    {
        g_mock->create_callback_thread(thread);
    }

    static void assign_callback_thread(buffer_id_t buffer, callback_thread_id_t thread)
    {
        g_mock->assign_callback_thread(buffer, thread);
    }

    static void flush_buffer(buffer_id_t buffer) { g_mock->flush_buffer(buffer); }

    static int destroy_buffer(buffer_id_t buffer)
    {
        return g_mock->destroy_buffer(buffer);
    }
};

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
        .on_records = &stub_on_records,
    };
}

class buffered_domain_test : public ::testing::Test
{
protected:
    void SetUp() override { g_mock = std::make_unique<StrictMock<gmock_sdk_backend>>(); }
    void TearDown() override { g_mock.reset(); }

    // Sets up the strict, ordered expectations for one configure() call and returns
    // the buffer id that create_buffer() will report back through its out-parameter.
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

TEST_F(buffered_domain_test, name_returns_definition_name)
{
    sut_t domain{ make_definition(), context_id_t{ 7 }, {} };
    EXPECT_EQ(domain.name(), "test_domain");
}

TEST_F(buffered_domain_test, buffer_id_is_default_before_configure)
{
    sut_t domain{ make_definition(), context_id_t{ 7 }, {} };
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
    sut_t domain{ make_definition(), context_id_t{ 7 }, {} };
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
    sut_t domain{ make_definition(), context_id_t{ 7 }, {} };
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

    sut_t domain_b{ std::move(domain_a) };

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

    EXPECT_EQ(domain_a.buffer_id(), buffer_id_t{});
    EXPECT_EQ(domain_b.buffer_id(), buffer_a);
}

}  // namespace
}  // namespace rocprofsys::domains
