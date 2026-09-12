// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/buffered/kfd_event_page_fault.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_page_fault.hpp"
#include "library/rocprofiler-sdk/buffered/kfd_queue.hpp"
#include "library/rocprofiler-sdk/callback/code_object.hpp"
#include "library/rocprofiler-sdk/domain_selection.hpp"
#include "library/rocprofiler-sdk/domain_service.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys
{
namespace
{

using ::testing::DoAll;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::InSequence;
using ::testing::NotNull;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrictMock;

// domain_service<SdkBackend, Externals> pulls in the full domains::registry<>, so its
// fake must satisfy the union of everything library/rocprofiler-sdk/{buffered,
// callback}/*.hpp touch on SdkBackend and Externals, plus the context/table members
// domain_service itself calls directly. test_support::mock_sdk/externals already carry
// all of that (shared with test_domain_registry.cpp), so this file only adds the
// table-driven behavior (g_buffer_table/g_callback_table) that domain_service, but not
// registry<>, exercises.
using domains::test_support::agent_t;
using domains::test_support::externals;
using domains::test_support::g_buffer_table;
using domains::test_support::g_callback_table;
using domains::test_support::g_externals_mock;
using domains::test_support::g_mock;
using domains::test_support::gmock_externals;
using domains::test_support::gmock_sdk_backend;
using domains::test_support::mock_sdk;

using sut_t = domain_service<mock_sdk, externals>;

constexpr std::size_t k_unsupported_domain_value = 999;

// Matches a tracing_operation_t* argument whose first expected.size() elements equal
// expected exactly. The pointee address is an implementation-internal detail of
// domain_service::configure_domain() (a freshly built local vector), so identity
// cannot be asserted -- contents can, and are asserted exactly.
MATCHER_P(operations_equal, expected, "")
{
    return std::equal(expected.begin(), expected.end(), arg);
}

// NOLINTNEXTLINE(readability-identifier-naming)
class domain_service_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_mock           = std::make_unique<StrictMock<gmock_sdk_backend>>();
        g_externals_mock = std::make_unique<StrictMock<gmock_externals>>();
        g_buffer_table   = {};
        g_callback_table = {};
    }

    void TearDown() override
    {
        g_mock.reset();
        g_externals_mock.reset();
    }

    // Every production on_configure() body calls add_string(category_name) followed by
    // get_agents_by_type(k_agent_type_gpu), unconditionally and exactly once; asserting
    // both confirms on_configure() actually ran rather than merely not crashing.
    void expect_on_configure_ran(std::string_view category_name)
    {
        InSequence seq;
        EXPECT_CALL(*g_externals_mock, add_string(Eq(category_name))).Times(1);
        EXPECT_CALL(*g_externals_mock,
                    get_agents_by_type(Eq(externals::k_agent_type_gpu)))
            .Times(1)
            .WillOnce(Return(std::vector<std::shared_ptr<agent_t>>{}));
    }

    void expect_create_context(const mock_sdk::context_id_t& context)
    {
        EXPECT_CALL(*g_mock, create_context(NotNull()))
            .Times(1)
            .WillOnce(DoAll(SetArgPointee<0>(context), Return()));
    }

    void expect_start_context(const mock_sdk::context_id_t& context)
    {
        EXPECT_CALL(*g_mock, start_context(Eq(context))).Times(1);
    }

    // NOLINTNEXTLINE(readability-function-size)
    void expect_configure_buffered(
        const mock_sdk::context_id_t& context, const mock_sdk::buffer_id_t& buffer,
        const mock_sdk::callback_thread_id_t& thread,
        mock_sdk::buffer_tracing_kind_t kind, mock_sdk::on_records_cb_t on_records,
        const std::vector<mock_sdk::tracing_operation_t>& operations)
    {
        InSequence seq;

        EXPECT_CALL(
            *g_mock,
            create_buffer(
                Eq(context),
                Eq(domains::k_default_buffer_properties.buffer_size.to_bytes()),
                Eq(domains::k_default_buffer_properties.buffer_watermark.to_bytes()),
                Eq(mock_sdk::BUFFER_POLICY_LOSSLESS), Eq(on_records),
                Eq(static_cast<void*>(nullptr)), NotNull()))
            .Times(1)
            // NOLINTNEXTLINE(readability-magic-numbers)
            .WillOnce(DoAll(SetArgPointee<6>(buffer), Return()));

        EXPECT_CALL(*g_mock, configure_buffer_tracing_service(
                                 Eq(context), Eq(kind), operations_equal(operations),
                                 Eq(operations.size()), Eq(buffer)))
            .Times(1);

        EXPECT_CALL(*g_mock, create_callback_thread(NotNull()))
            .Times(1)
            .WillOnce(DoAll(SetArgPointee<0>(thread), Return()));

        EXPECT_CALL(*g_mock, assign_callback_thread(Eq(buffer), Eq(thread))).Times(1);
    }

    void expect_configure_callback(
        const mock_sdk::context_id_t& context, mock_sdk::callback_tracing_kind_t kind,
        mock_sdk::on_record_cb_t                          on_record,
        const std::vector<mock_sdk::tracing_operation_t>& operations)
    {
        EXPECT_CALL(*g_mock, configure_callback_tracing_service(
                                 Eq(context), Eq(kind), operations_equal(operations),
                                 Eq(operations.size()), Eq(on_record),
                                 Eq(static_cast<void*>(nullptr))))
            .Times(1);
    }

    void expect_destroy_buffer(const mock_sdk::buffer_id_t& buffer)
    {
        EXPECT_CALL(*g_mock, destroy_buffer(Eq(buffer))).Times(1).WillOnce(Return(0));
    }
};

TEST_F(domain_service_test,
       constructor_populates_available_domains_from_supported_sdk_tables)
{
    g_buffer_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "kfd_queue",
                       .operations = { "op0", "op1" },
                       .value      = mock_sdk::BUFFER_TRACING_KFD_QUEUE },
                     { .name       = "unsupported_domain",
                       .operations = {},
                       .value      = k_unsupported_domain_value } }
    };
    g_callback_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "code_object",
                       .operations = {},
                       .value      = mock_sdk::CALLBACK_TRACING_CODE_OBJECT } }
    };

    const sut_t service;

    const auto available = service.available_domains();
    ASSERT_EQ(available.size(), 2u);

    EXPECT_EQ(available[0].key.mode, domains::collection_mode::buffered);
    EXPECT_EQ(available[0].key.value, mock_sdk::BUFFER_TRACING_KFD_QUEUE);
    EXPECT_EQ(available[0].name, "kfd_queue");
    ASSERT_EQ(available[0].operations.size(), 2u);
    EXPECT_EQ(available[0].operations[0].id, 0u);
    EXPECT_EQ(available[0].operations[0].name, "op0");
    EXPECT_EQ(available[0].operations[1].id, 1u);
    EXPECT_EQ(available[0].operations[1].name, "op1");
    ASSERT_TRUE(available[0].group.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access) checked by ASSERT_TRUE above
    EXPECT_EQ(*available[0].group, "kfd_events");

    EXPECT_EQ(available[1].key.mode, domains::collection_mode::callback);
    EXPECT_EQ(available[1].key.value, mock_sdk::CALLBACK_TRACING_CODE_OBJECT);
    EXPECT_EQ(available[1].name, "code_object");
    EXPECT_TRUE(available[1].operations.empty());
    EXPECT_FALSE(available[1].group.has_value());
}

TEST_F(domain_service_test,
       configure_selects_and_fully_configures_matching_buffered_domain)
{
    g_buffer_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "kfd_queue",
                       .operations = { "op0", "op1" },
                       .value      = mock_sdk::BUFFER_TRACING_KFD_QUEUE } }
    };

    sut_t service;

    const mock_sdk::context_id_t         context{ 1 };
    const mock_sdk::buffer_id_t          buffer{ 50 };
    const mock_sdk::callback_thread_id_t thread{ 5 };

    expect_create_context(context);
    expect_configure_buffered(
        context, buffer, thread,
        static_cast<mock_sdk::buffer_tracing_kind_t>(mock_sdk::BUFFER_TRACING_KFD_QUEUE),
        domains::buffered::k_kfd_queue<mock_sdk, externals>.on_records, { 0, 1 });
    expect_on_configure_ran(externals::k_kfd_queue_category_name);
    expect_start_context(context);

    service.configure(std::vector<domain_selection>{ domain_selection{
        .name = "kfd_queue", .group = std::nullopt, .operations = std::nullopt } });

    const auto configuration = service.configuration();
    ASSERT_EQ(configuration.size(), 1u);
    EXPECT_EQ(configuration[0].key.mode, domains::collection_mode::buffered);
    EXPECT_EQ(configuration[0].key.value, mock_sdk::BUFFER_TRACING_KFD_QUEUE);
    EXPECT_THAT(configuration[0].operations, ElementsAre(0u, 1u));

    expect_destroy_buffer(buffer);
}

TEST_F(domain_service_test,
       configure_selects_and_fully_configures_matching_callback_domain)
{
    g_callback_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "code_object",
                       .operations = { "opA" },
                       .value      = mock_sdk::CALLBACK_TRACING_CODE_OBJECT } }
    };

    sut_t service;

    const mock_sdk::context_id_t context{ 2 };

    expect_create_context(context);
    expect_configure_callback(
        context,
        static_cast<mock_sdk::callback_tracing_kind_t>(
            mock_sdk::CALLBACK_TRACING_CODE_OBJECT),
        domains::callback::k_code_object<mock_sdk, externals>.on_record, { 0 });
    expect_start_context(context);

    service.configure(std::vector<domain_selection>{ domain_selection{
        .name = "code_object", .group = std::nullopt, .operations = std::nullopt } });

    const auto configuration = service.configuration();
    ASSERT_EQ(configuration.size(), 1u);
    EXPECT_EQ(configuration[0].key.mode, domains::collection_mode::callback);
    EXPECT_THAT(configuration[0].operations, ElementsAre(0u));
}

TEST_F(domain_service_test, configure_throws_runtime_error_for_unknown_domain_name)
{
    sut_t service;

    EXPECT_THROW(
        {
            service.configure(std::vector<domain_selection>{
                domain_selection{ .name       = "no_such_domain",
                                  .group      = std::nullopt,
                                  .operations = std::nullopt } });
        },
        std::runtime_error);
}

TEST_F(domain_service_test,
       configure_throws_runtime_error_when_selection_sets_both_name_and_group)
{
    sut_t service;

    EXPECT_THROW(
        {
            service.configure(std::vector<domain_selection>{
                domain_selection{ .name       = "kfd_queue",
                                  .group      = "kfd_events",
                                  .operations = std::nullopt } });
        },
        std::runtime_error);
}

TEST_F(domain_service_test,
       configure_throws_runtime_error_when_operations_set_without_name)
{
    sut_t service;

    EXPECT_THROW(
        {
            service.configure(std::vector<domain_selection>{
                domain_selection{ .name       = std::nullopt,
                                  .group      = std::nullopt,
                                  .operations = std::vector<std::string>{ "op0" } } });
        },
        std::runtime_error);
}

TEST_F(domain_service_test, configure_throws_runtime_error_for_unknown_operation_name)
{
    g_buffer_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "kfd_queue",
                       .operations = { "op0" },
                       .value      = mock_sdk::BUFFER_TRACING_KFD_QUEUE } }
    };

    sut_t service;

    EXPECT_THROW(
        {
            service.configure(std::vector<domain_selection>{ domain_selection{
                .name       = "kfd_queue",
                .group      = std::nullopt,
                .operations = std::vector<std::string>{ "no_such_operation" } } });
        },
        std::runtime_error);
}

TEST_F(domain_service_test,
       configure_merges_operations_when_multiple_selections_target_same_domain)
{
    g_buffer_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "kfd_queue",
                       .operations = { "op0", "op1" },
                       .value      = mock_sdk::BUFFER_TRACING_KFD_QUEUE } }
    };

    sut_t service;

    const mock_sdk::context_id_t         context{ 1 };
    const mock_sdk::buffer_id_t          buffer{ 50 };
    const mock_sdk::callback_thread_id_t thread{ 5 };

    expect_create_context(context);
    expect_configure_buffered(
        context, buffer, thread,
        static_cast<mock_sdk::buffer_tracing_kind_t>(mock_sdk::BUFFER_TRACING_KFD_QUEUE),
        domains::buffered::k_kfd_queue<mock_sdk, externals>.on_records, { 0, 1 });
    expect_on_configure_ran(externals::k_kfd_queue_category_name);
    expect_start_context(context);

    service.configure(std::vector<domain_selection>{
        domain_selection{ .name       = "kfd_queue",
                          .group      = std::nullopt,
                          .operations = std::vector<std::string>{ "op0" } },
        domain_selection{ .name       = "kfd_queue",
                          .group      = std::nullopt,
                          .operations = std::vector<std::string>{ "op1" } } });

    const auto configuration = service.configuration();
    ASSERT_EQ(configuration.size(), 1u);
    EXPECT_THAT(configuration[0].operations, ElementsAre(0u, 1u));

    expect_destroy_buffer(buffer);
}

TEST_F(domain_service_test, flush_calls_flush_on_each_configured_buffered_domain)
{
    g_buffer_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "kfd_queue",
                       .operations = { "op0" },
                       .value      = mock_sdk::BUFFER_TRACING_KFD_QUEUE } }
    };

    sut_t service;

    const mock_sdk::context_id_t         context{ 1 };
    const mock_sdk::buffer_id_t          buffer{ 50 };
    const mock_sdk::callback_thread_id_t thread{ 5 };

    expect_create_context(context);
    expect_configure_buffered(
        context, buffer, thread,
        static_cast<mock_sdk::buffer_tracing_kind_t>(mock_sdk::BUFFER_TRACING_KFD_QUEUE),
        domains::buffered::k_kfd_queue<mock_sdk, externals>.on_records, { 0 });
    expect_on_configure_ran(externals::k_kfd_queue_category_name);
    expect_start_context(context);

    service.configure(std::vector<domain_selection>{ domain_selection{
        .name = "kfd_queue", .group = std::nullopt, .operations = std::nullopt } });

    EXPECT_CALL(*g_mock, flush_buffer(Eq(buffer))).Times(1);
    service.flush();

    expect_destroy_buffer(buffer);
}

TEST_F(domain_service_test, configure_calls_on_configure_when_domain_defines_it)
{
    g_buffer_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "kfd_page_fault",
                       .operations = { "op0" },
                       .value      = mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT } }
    };

    sut_t service;

    const mock_sdk::context_id_t         context{ 1 };
    const mock_sdk::buffer_id_t          buffer{ 50 };
    const mock_sdk::callback_thread_id_t thread{ 5 };

    expect_create_context(context);
    expect_configure_buffered(
        context, buffer, thread,
        static_cast<mock_sdk::buffer_tracing_kind_t>(
            mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT),
        domains::buffered::k_kfd_page_fault<mock_sdk, externals>.on_records, { 0 });
    expect_on_configure_ran(externals::k_kfd_page_fault_category_name);
    expect_start_context(context);

    service.configure(std::vector<domain_selection>{ domain_selection{
        .name = "kfd_page_fault", .group = std::nullopt, .operations = std::nullopt } });

    expect_destroy_buffer(buffer);
}

TEST_F(domain_service_test, configure_skips_on_configure_when_domain_has_none)
{
    g_buffer_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "kfd_event_page_fault",
                       .operations = { "op0" },
                       .value      = mock_sdk::BUFFER_TRACING_KFD_EVENT_PAGE_FAULT } }
    };

    sut_t service;

    const mock_sdk::context_id_t         context{ 1 };
    const mock_sdk::buffer_id_t          buffer{ 50 };
    const mock_sdk::callback_thread_id_t thread{ 5 };

    expect_create_context(context);
    expect_configure_buffered(
        context, buffer, thread,
        static_cast<mock_sdk::buffer_tracing_kind_t>(
            mock_sdk::BUFFER_TRACING_KFD_EVENT_PAGE_FAULT),
        domains::buffered::k_kfd_event_page_fault<mock_sdk, externals>.on_records, { 0 });
    // No expect_on_configure_ran(): kfd_event_page_fault has no on_configure callback,
    // so StrictMock<gmock_externals> fails the test if add_string/get_agents_by_type
    // are called here.
    expect_start_context(context);

    service.configure(
        std::vector<domain_selection>{ domain_selection{ .name  = "kfd_event_page_fault",
                                                         .group = std::nullopt,
                                                         .operations = std::nullopt } });

    expect_destroy_buffer(buffer);
}

TEST_F(domain_service_test,
       configure_calls_on_configure_for_callback_domain_that_defines_it)
{
    // on_code_object_configure() is a no-op, so it has no Externals side effect to
    // assert on; this instead asserts the precondition domain_service's `if` branches
    // on (on_configure is non-null) and that invoking it does not throw or crash.
    constexpr const auto& k_code_object_definition =
        domains::callback::k_code_object<mock_sdk, externals>;
    ASSERT_NE(k_code_object_definition.on_configure, nullptr);

    g_callback_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "code_object",
                       .operations = {},
                       .value      = mock_sdk::CALLBACK_TRACING_CODE_OBJECT } }
    };

    sut_t service;

    const mock_sdk::context_id_t context{ 2 };

    expect_create_context(context);
    expect_configure_callback(
        context,
        static_cast<mock_sdk::callback_tracing_kind_t>(
            mock_sdk::CALLBACK_TRACING_CODE_OBJECT),
        domains::callback::k_code_object<mock_sdk, externals>.on_record, {});
    expect_start_context(context);

    service.configure(std::vector<domain_selection>{ domain_selection{
        .name = "code_object", .group = std::nullopt, .operations = std::nullopt } });

    const auto configuration = service.configuration();
    ASSERT_EQ(configuration.size(), 1u);
    EXPECT_EQ(configuration[0].key.value, mock_sdk::CALLBACK_TRACING_CODE_OBJECT);
}

TEST_F(domain_service_test,
       configure_selects_domains_by_group_case_insensitively_and_configures_all_matches)
{
    g_buffer_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "kfd_queue",
                       .operations = { "op0" },
                       .value      = mock_sdk::BUFFER_TRACING_KFD_QUEUE },
                     { .name       = "kfd_page_fault",
                       .operations = { "op0" },
                       .value      = mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT } }
    };

    sut_t service;

    const mock_sdk::context_id_t         context{ 1 };
    const mock_sdk::buffer_id_t          queue_buffer{ 50 };
    const mock_sdk::buffer_id_t          page_fault_buffer{ 51 };
    const mock_sdk::callback_thread_id_t queue_thread{ 5 };
    const mock_sdk::callback_thread_id_t page_fault_thread{ 6 };

    expect_create_context(context);
    expect_configure_buffered(
        context, queue_buffer, queue_thread,
        static_cast<mock_sdk::buffer_tracing_kind_t>(mock_sdk::BUFFER_TRACING_KFD_QUEUE),
        domains::buffered::k_kfd_queue<mock_sdk, externals>.on_records, { 0 });
    expect_on_configure_ran(externals::k_kfd_queue_category_name);
    expect_configure_buffered(
        context, page_fault_buffer, page_fault_thread,
        static_cast<mock_sdk::buffer_tracing_kind_t>(
            mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT),
        domains::buffered::k_kfd_page_fault<mock_sdk, externals>.on_records, { 0 });
    expect_on_configure_ran(externals::k_kfd_page_fault_category_name);
    expect_start_context(context);

    service.configure(std::vector<domain_selection>{ domain_selection{
        .name = std::nullopt, .group = "KFD_EVENTS", .operations = std::nullopt } });

    const auto configuration = service.configuration();
    ASSERT_EQ(configuration.size(), 2u);
    EXPECT_EQ(configuration[0].key.value, mock_sdk::BUFFER_TRACING_KFD_QUEUE);
    EXPECT_EQ(configuration[1].key.value, mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT);

    expect_destroy_buffer(queue_buffer);
    expect_destroy_buffer(page_fault_buffer);
}

TEST_F(domain_service_test,
       configure_selects_all_available_domains_when_selection_has_no_name_or_group)
{
    g_buffer_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "kfd_queue",
                       .operations = { "op0" },
                       .value      = mock_sdk::BUFFER_TRACING_KFD_QUEUE },
                     { .name       = "kfd_page_fault",
                       .operations = { "op0" },
                       .value      = mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT } }
    };

    sut_t service;

    const mock_sdk::context_id_t         context{ 1 };
    const mock_sdk::buffer_id_t          queue_buffer{ 50 };
    const mock_sdk::buffer_id_t          page_fault_buffer{ 51 };
    const mock_sdk::callback_thread_id_t queue_thread{ 5 };
    const mock_sdk::callback_thread_id_t page_fault_thread{ 6 };

    expect_create_context(context);
    expect_configure_buffered(
        context, queue_buffer, queue_thread,
        static_cast<mock_sdk::buffer_tracing_kind_t>(mock_sdk::BUFFER_TRACING_KFD_QUEUE),
        domains::buffered::k_kfd_queue<mock_sdk, externals>.on_records, { 0 });
    expect_on_configure_ran(externals::k_kfd_queue_category_name);
    expect_configure_buffered(
        context, page_fault_buffer, page_fault_thread,
        static_cast<mock_sdk::buffer_tracing_kind_t>(
            mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT),
        domains::buffered::k_kfd_page_fault<mock_sdk, externals>.on_records, { 0 });
    expect_on_configure_ran(externals::k_kfd_page_fault_category_name);
    expect_start_context(context);

    // No name and no group set: match_domains() falls through to its final branch,
    // which selects every available domain.
    service.configure(std::vector<domain_selection>{ domain_selection{
        .name = std::nullopt, .group = std::nullopt, .operations = std::nullopt } });

    const auto configuration = service.configuration();
    ASSERT_EQ(configuration.size(), 2u);
    EXPECT_EQ(configuration[0].key.value, mock_sdk::BUFFER_TRACING_KFD_QUEUE);
    EXPECT_EQ(configuration[1].key.value, mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT);

    expect_destroy_buffer(queue_buffer);
    expect_destroy_buffer(page_fault_buffer);
}

TEST_F(domain_service_test, configure_throws_runtime_error_for_unknown_group)
{
    g_buffer_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "kfd_queue",
                       .operations = { "op0" },
                       .value      = mock_sdk::BUFFER_TRACING_KFD_QUEUE } }
    };

    sut_t service;

    EXPECT_THROW(
        {
            service.configure(std::vector<domain_selection>{
                domain_selection{ .name       = std::nullopt,
                                  .group      = "no_such_group",
                                  .operations = std::nullopt } });
        },
        std::runtime_error);
}

}  // namespace
}  // namespace rocprofsys
