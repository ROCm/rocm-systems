// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/buffered/kfd/event_page_fault.hpp"
#include "library/rocprofiler-sdk/buffered/kfd/page_fault.hpp"
#include "library/rocprofiler-sdk/buffered/kfd/queue.hpp"
#include "library/rocprofiler-sdk/callback/code_object.hpp"
#include "library/rocprofiler-sdk/callback/hip/runtime_api.hpp"
#include "library/rocprofiler-sdk/callback/hip_stream.hpp"
#include "library/rocprofiler-sdk/callback/ompt/ompt.hpp"
#include "library/rocprofiler-sdk/domain_selection.hpp"
#include "library/rocprofiler-sdk/domain_service.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

using ::testing::_;
using ::testing::DoAll;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::InSequence;
using ::testing::IsEmpty;
using ::testing::NotNull;
using ::testing::Return;
using ::testing::Sequence;
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
using domains::test_support::externals_with_roctx_config;
using domains::test_support::g_buffer_table;
using domains::test_support::g_callback_table;
using domains::test_support::g_externals_mock;
using domains::test_support::g_metadata_registry_mock;
using domains::test_support::g_mock;
using domains::test_support::gmock_externals;
using domains::test_support::gmock_metadata_registry;
using domains::test_support::gmock_sdk_backend;
using domains::test_support::mock_sdk;

using sut_t       = domain_service<mock_sdk, externals>;
using roctx_sut_t = domain_service<mock_sdk, externals_with_roctx_config>;

constexpr std::size_t k_unsupported_domain_value = 999;

constexpr mock_sdk::context_id_t k_code_object_context{ 3 };

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
        g_metadata_registry_mock =
            std::make_unique<StrictMock<gmock_metadata_registry>>();
        g_buffer_table   = {};
        g_callback_table = {};
    }

    void TearDown() override
    {
        g_mock.reset();
        g_externals_mock.reset();
        g_metadata_registry_mock.reset();
    }

    // Every production on_configure() body calls add_string(category_name) followed by
    // get_agents_by_type(k_agent_type_gpu), unconditionally and exactly once; asserting
    // both confirms on_configure() actually ran rather than merely not crashing.
    void expect_on_configure_ran(std::string_view category_name)
    {
        InSequence seq;
        EXPECT_CALL(*g_metadata_registry_mock, add_string(Eq(category_name))).Times(1);
        EXPECT_CALL(*g_externals_mock,
                    get_agents_by_type(Eq(externals::k_agent_type_gpu)))
            .Times(1)
            .WillOnce(Return(std::vector<std::shared_ptr<agent_t>>{}));
    }

    void expect_create_context(const mock_sdk::context_id_t& context)
    {
        EXPECT_CALL(*g_mock, create_context(NotNull()))
            .Times(1)
            .InSequence(m_create_context_sequence)
            .WillOnce(DoAll(SetArgPointee<0>(context), Return()));
    }

    // configure() always finishes by subscribing code_object, with every operation, on
    // its own always-on context.
    void expect_code_object_domain_configured()
    {
        expect_create_context(k_code_object_context);
        expect_configure_callback(
            k_code_object_context,
            static_cast<mock_sdk::callback_tracing_kind_t>(
                mock_sdk::CALLBACK_TRACING_CODE_OBJECT),
            domains::callback::k_code_object<mock_sdk, externals>.on_record, {});
        expect_start_context(k_code_object_context);
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

private:
    // Both the main and the code_object context come from create_context(); the
    // sequence hands out context ids in the order the expectations are declared.
    Sequence m_create_context_sequence;

    void expect_roctx_enabled(bool pause_resume_enabled)
    {
        EXPECT_CALL(*g_externals_mock, is_roctx_enabled()).WillOnce(Return(true));
        EXPECT_CALL(*g_externals_mock, get_roctx_pause_resume_enabled())
            .WillOnce(Return(pause_resume_enabled));
    }

    void expect_roctx_core_configured(const mock_sdk::context_id_t& context)
    {
        expect_configure_callback(
            context,
            static_cast<mock_sdk::callback_tracing_kind_t>(
                mock_sdk::CALLBACK_TRACING_MARKER_CORE_API),
            domains::callback::roctx::k_core_api<mock_sdk, externals_with_roctx_config>.on_record,
            {});
        EXPECT_CALL(*g_metadata_registry_mock,
                    add_string(Eq(externals::rocm_marker_api_category_name)))
            .Times(1);
    }

    void expect_roctx_control_configured(const mock_sdk::context_id_t& context)
    {
        expect_configure_callback(
            context,
            static_cast<mock_sdk::callback_tracing_kind_t>(
                mock_sdk::CALLBACK_TRACING_MARKER_CONTROL_API),
            domains::callback::roctx::k_control_api<mock_sdk, externals_with_roctx_config>.on_record,
            { mock_sdk::MARKER_CONTROL_API_ID_roctxProfilerPause,
              mock_sdk::MARKER_CONTROL_API_ID_roctxProfilerResume });
    }

    // Configures roctx (core domain only) on @p roctx_context, returning an unstarted
    // service.
    void configure_roctx_core_only(roctx_sut_t&                  service,
                                   const mock_sdk::context_id_t& roctx_context)
    {
        expect_roctx_enabled(false);
        expect_create_context(roctx_context);
        expect_roctx_core_configured(roctx_context);

        service.configure(std::vector<domain_selection>{});
    }

    // The SDK must report code_object before the service is constructed: the
    // constructor snapshots the supported domains.
    static void report_code_object_domain()
    {
        g_callback_table = mock_sdk::tracing_names_t{
            .entries = { { .name       = "code_object",
                           .operations = {},
                           .value      = mock_sdk::CALLBACK_TRACING_CODE_OBJECT } }
        };
    }

    // Selects code_object (main context) and enables roctx (core only), so the service
    // owns two distinct contexts: @p main_context and @p roctx_context.
    void configure_main_and_roctx(roctx_sut_t&                  service,
                                  const mock_sdk::context_id_t& main_context,
                                  const mock_sdk::context_id_t& roctx_context)
    {
        expect_roctx_enabled(false);
        {
            const InSequence seq;
            expect_create_context(main_context);
            expect_create_context(roctx_context);
        }
        expect_configure_callback(
            main_context,
            static_cast<mock_sdk::callback_tracing_kind_t>(
                mock_sdk::CALLBACK_TRACING_CODE_OBJECT),
            domains::callback::k_code_object<mock_sdk, externals_with_roctx_config>.on_record,
            {});
        expect_roctx_core_configured(roctx_context);

        service.configure(std::vector<domain_selection>{ domain_selection{
            .name = "code_object", .group = std::nullopt, .operations = std::nullopt } });
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
        .entries = { { .name       = "hip_runtime_api",
                       .operations = {},
                       .value      = mock_sdk::CALLBACK_TRACING_HIP_RUNTIME_API } }
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
    EXPECT_EQ(available[1].key.value, mock_sdk::CALLBACK_TRACING_HIP_RUNTIME_API);
    EXPECT_EQ(available[1].name, "hip_runtime_api");
    EXPECT_TRUE(available[1].operations.empty());
    ASSERT_TRUE(available[1].group.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access) checked by ASSERT_TRUE above
    EXPECT_EQ(*available[1].group, "hip_api");
}

TEST_F(domain_service_test, constructor_hides_code_object_from_available_domains)
{
    g_callback_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "code_object",
                       .operations = {},
                       .value      = mock_sdk::CALLBACK_TRACING_CODE_OBJECT } }
    };

    const sut_t service;

    EXPECT_THAT(service.available_domains(), IsEmpty());
}

TEST_F(domain_service_test, configure_throws_runtime_error_when_code_object_is_selected)
{
    g_callback_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "code_object",
                       .operations = {},
                       .value      = mock_sdk::CALLBACK_TRACING_CODE_OBJECT } }
    };

    sut_t service;

    EXPECT_THROW(
        {
            service.configure(std::vector<domain_selection>{
                domain_selection{ .name       = "code_object",
                                  .group      = std::nullopt,
                                  .operations = std::nullopt } });
        },
        std::runtime_error);
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
        domains::buffered::kfd::k_queue<mock_sdk, externals>.on_records, {});
    expect_on_configure_ran(externals::k_kfd_queue_category_name);

    // No explicit .operations filter: resolve_operations() must resolve to an empty
    // list rather than enumerating every known operation id, so that
    // buffered_domain::configure() forwards (nullptr, 0) to the SDK -- some
    // buffer-tracing kinds silently drop all records when given a non-null array that
    // enumerates every operation id, even though the configure call reports success.
    expect_code_object_domain_configured();

    service.configure(std::vector<domain_selection>{ domain_selection{
        .name = "kfd_queue", .group = std::nullopt, .operations = std::nullopt } });

    const auto configuration = service.configuration();
    ASSERT_EQ(configuration.size(), 1u);
    EXPECT_EQ(configuration[0].key.mode, domains::collection_mode::buffered);
    EXPECT_EQ(configuration[0].key.value, mock_sdk::BUFFER_TRACING_KFD_QUEUE);
    EXPECT_THAT(configuration[0].operations, IsEmpty());

    expect_destroy_buffer(buffer);
}

TEST_F(domain_service_test,
       configure_selects_and_fully_configures_matching_callback_domain)
{
    g_callback_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "hip_runtime_api",
                       .operations = { "opA" },
                       .value      = mock_sdk::CALLBACK_TRACING_HIP_RUNTIME_API } }
    };

    sut_t service;

    const mock_sdk::context_id_t context{ 2 };

    expect_create_context(context);
    expect_configure_callback(
        context,
        static_cast<mock_sdk::callback_tracing_kind_t>(
            mock_sdk::CALLBACK_TRACING_HIP_RUNTIME_API),
        domains::callback::hip::k_runtime_api<mock_sdk, externals>.on_record, {});

    // No explicit .operations filter: resolve_operations() must resolve to an empty
    // list so callback_domain::configure() forwards (nullptr, 0) to the SDK instead of
    // enumerating every known operation id.
    expect_code_object_domain_configured();

    service.configure(std::vector<domain_selection>{ domain_selection{
        .name = "hip_runtime_api", .group = std::nullopt, .operations = std::nullopt } });

    const auto configuration = service.configuration();
    ASSERT_EQ(configuration.size(), 1u);
    EXPECT_EQ(configuration[0].key.mode, domains::collection_mode::callback);
    EXPECT_THAT(configuration[0].operations, IsEmpty());
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
        domains::buffered::kfd::k_queue<mock_sdk, externals>.on_records, { 0, 1 });
    expect_on_configure_ran(externals::k_kfd_queue_category_name);

    expect_code_object_domain_configured();

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
        domains::buffered::kfd::k_queue<mock_sdk, externals>.on_records, {});
    expect_on_configure_ran(externals::k_kfd_queue_category_name);

    expect_code_object_domain_configured();

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
        domains::buffered::kfd::k_page_fault<mock_sdk, externals>.on_records, {});
    expect_on_configure_ran(externals::k_kfd_page_fault_category_name);

    expect_code_object_domain_configured();

    service.configure(std::vector<domain_selection>{ domain_selection{
        .name = "kfd_page_fault", .group = std::nullopt, .operations = std::nullopt } });

    expect_destroy_buffer(buffer);
}

TEST_F(domain_service_test, configure_calls_on_configure_for_event_domain_that_defines_it)
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
        domains::buffered::kfd::k_event_page_fault<mock_sdk, externals>.on_records, {});
    expect_on_configure_ran(externals::k_kfd_event_page_fault_category_name);

    expect_code_object_domain_configured();

    service.configure(
        std::vector<domain_selection>{ domain_selection{ .name  = "kfd_event_page_fault",
                                                         .group = std::nullopt,
                                                         .operations = std::nullopt } });

    expect_destroy_buffer(buffer);
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
        domains::buffered::kfd::k_queue<mock_sdk, externals>.on_records, {});
    expect_on_configure_ran(externals::k_kfd_queue_category_name);
    expect_configure_buffered(
        context, page_fault_buffer, page_fault_thread,
        static_cast<mock_sdk::buffer_tracing_kind_t>(
            mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT),
        domains::buffered::kfd::k_page_fault<mock_sdk, externals>.on_records, {});
    expect_on_configure_ran(externals::k_kfd_page_fault_category_name);

    expect_code_object_domain_configured();

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
        domains::buffered::kfd::k_queue<mock_sdk, externals>.on_records, {});
    expect_on_configure_ran(externals::k_kfd_queue_category_name);
    expect_configure_buffered(
        context, page_fault_buffer, page_fault_thread,
        static_cast<mock_sdk::buffer_tracing_kind_t>(
            mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT),
        domains::buffered::kfd::k_page_fault<mock_sdk, externals>.on_records, {});
    expect_on_configure_ran(externals::k_kfd_page_fault_category_name);

    // No name and no group set: match_domains() falls through to its final branch,
    // which selects every available domain.
    expect_code_object_domain_configured();

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

// ─── finalize ────────────────────────────────────────────────────────────────

TEST_F(domain_service_test, finalize_is_a_noop_when_no_domains_are_configured)
{
    sut_t service;

    service.finalize();
}

TEST_F(domain_service_test, finalize_does_not_invoke_domains_that_leave_on_finalize_unset)
{
    // hip_stream does not define on_finalize, so k_hip_stream.on_finalize is the
    // default-initialized nullptr; finalize() must skip it rather than call through a
    // null function pointer.
    constexpr const auto& k_hip_stream_definition =
        domains::callback::k_hip_stream<mock_sdk, externals>;
    ASSERT_EQ(k_hip_stream_definition.on_finalize, nullptr);

    g_callback_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "hip_stream",
                       .operations = {},
                       .value      = mock_sdk::CALLBACK_TRACING_HIP_STREAM } }
    };

    sut_t service;

    const mock_sdk::context_id_t context{ 2 };

    expect_create_context(context);
    expect_configure_callback(context,
                              static_cast<mock_sdk::callback_tracing_kind_t>(
                                  mock_sdk::CALLBACK_TRACING_HIP_STREAM),
                              k_hip_stream_definition.on_record, {});
    expect_code_object_domain_configured();

    service.configure(std::vector<domain_selection>{ domain_selection{
        .name = "hip_stream", .group = std::nullopt, .operations = std::nullopt } });

    service.finalize();
}

TEST_F(domain_service_test,
       finalize_invokes_on_finalize_for_configured_domain_that_defines_it)
{
    // ompt is currently the only domain that defines on_finalize (it flushes any OMPT
    // region still open when the tool shuts down -- see ompt.hpp's
    // ompt_finalize_orphan_events). Populating its pending-callback storage and
    // asserting it drains after service.finalize() proves domain_service::finalize()
    // actually reaches the domain's on_finalize hook end-to-end, rather than merely
    // not crashing.
    constexpr const auto& k_ompt_definition =
        domains::callback::ompt::k_ompt_api<mock_sdk, externals>;
    ASSERT_NE(k_ompt_definition.on_finalize, nullptr);

    g_callback_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "ompt",
                       .operations = {},
                       .value      = mock_sdk::CALLBACK_TRACING_OMPT } }
    };

    sut_t service;

    const mock_sdk::context_id_t context{ 2 };

    expect_create_context(context);
    expect_configure_callback(
        context,
        static_cast<mock_sdk::callback_tracing_kind_t>(mock_sdk::CALLBACK_TRACING_OMPT),
        k_ompt_definition.on_record, {});
    expect_code_object_domain_configured();

    service.configure(std::vector<domain_selection>{ domain_selection{
        .name = "ompt", .group = std::nullopt, .operations = std::nullopt } });

    mock_sdk::callback_tracing_record_t record{};
    record.operation = static_cast<std::uint32_t>(mock_sdk::OMPT_ID_task_create);
    record.correlation_id.internal = 42U;

    auto& pending_standard_callbacks =
        domains::callback::ompt::detail::open_regions<mock_sdk>::s_standard;
    pending_standard_callbacks.emplace(
        record.correlation_id.internal,
        domains::callback::ompt::detail::pending_region<mock_sdk>{
            record, /*begin_timestamp=*/1, function_args_t{} });

    EXPECT_CALL(*g_metadata_registry_mock, add_string(_)).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_thread_info(_)).Times(1);

    service.finalize();

    EXPECT_TRUE(pending_standard_callbacks.empty());
}

// ─── roctx marker domains ───────────────────────────────────────────────────────

TEST_F(domain_service_test, configure_creates_no_roctx_context_when_roctx_disabled)
{
    roctx_sut_t service;

    EXPECT_CALL(*g_externals_mock, is_roctx_enabled()).WillOnce(Return(false));

    service.configure(std::vector<domain_selection>{});
}

TEST_F(domain_service_test,
       configure_creates_roctx_context_with_core_domain_only_without_pause_resume)
{
    roctx_sut_t service;

    const mock_sdk::context_id_t roctx_context{ 7 };

    // Only wired up: no start_context expectation, so starting it would fail the test.
    configure_roctx_core_only(service, roctx_context);
}

TEST_F(domain_service_test,
       configure_adds_control_domain_with_pause_resume_operations_on_same_roctx_context)
{
    roctx_sut_t service;

    const mock_sdk::context_id_t roctx_context{ 8 };

    expect_roctx_enabled(true);
    expect_create_context(roctx_context);
    expect_roctx_core_configured(roctx_context);
    expect_roctx_control_configured(roctx_context);

    service.configure(std::vector<domain_selection>{});
}

TEST_F(domain_service_test, roctx_context_is_separate_from_the_main_context)
{
    report_code_object_domain();
    roctx_sut_t service;

    const mock_sdk::context_id_t main_context{ 1 };
    const mock_sdk::context_id_t roctx_context{ 2 };

    configure_main_and_roctx(service, main_context, roctx_context);
}

TEST_F(domain_service_test, start_roctx_starts_valid_inactive_roctx_context)
{
    roctx_sut_t service;

    const mock_sdk::context_id_t roctx_context{ 9 };
    configure_roctx_core_only(service, roctx_context);

    EXPECT_CALL(*g_mock, context_is_valid(Eq(roctx_context))).WillOnce(Return(true));
    EXPECT_CALL(*g_mock, context_is_active(Eq(roctx_context))).WillOnce(Return(false));
    expect_start_context(roctx_context);

    service.start_roctx();
}

TEST_F(domain_service_test, start_roctx_skips_already_active_roctx_context)
{
    roctx_sut_t service;

    const mock_sdk::context_id_t roctx_context{ 9 };
    configure_roctx_core_only(service, roctx_context);

    EXPECT_CALL(*g_mock, context_is_valid(Eq(roctx_context))).WillOnce(Return(true));
    EXPECT_CALL(*g_mock, context_is_active(Eq(roctx_context))).WillOnce(Return(true));

    service.start_roctx();
}

TEST_F(domain_service_test, start_roctx_is_noop_when_roctx_was_not_configured)
{
    roctx_sut_t service;

    EXPECT_CALL(*g_externals_mock, is_roctx_enabled()).WillOnce(Return(false));
    service.configure(std::vector<domain_selection>{});

    EXPECT_CALL(*g_mock, context_is_valid(Eq(mock_sdk::context_id_t{})))
        .WillOnce(Return(false));

    service.start_roctx();
}

TEST_F(domain_service_test, pause_stops_only_the_main_context_not_the_roctx_context)
{
    report_code_object_domain();
    roctx_sut_t service;

    const mock_sdk::context_id_t main_context{ 1 };
    const mock_sdk::context_id_t roctx_context{ 2 };
    configure_main_and_roctx(service, main_context, roctx_context);

    EXPECT_CALL(*g_mock, context_is_valid(Eq(main_context))).WillOnce(Return(true));
    EXPECT_CALL(*g_mock, context_is_active(Eq(main_context))).WillOnce(Return(true));
    EXPECT_CALL(*g_mock, stop_context(Eq(main_context))).Times(1);

    service.pause();
}

TEST_F(domain_service_test, start_starts_only_the_main_context_not_the_roctx_context)
{
    report_code_object_domain();
    roctx_sut_t service;

    const mock_sdk::context_id_t main_context{ 1 };
    const mock_sdk::context_id_t roctx_context{ 2 };
    configure_main_and_roctx(service, main_context, roctx_context);

    EXPECT_CALL(*g_mock, context_is_valid(Eq(main_context))).WillOnce(Return(true));
    EXPECT_CALL(*g_mock, context_is_active(Eq(main_context))).WillOnce(Return(false));
    expect_start_context(main_context);

    service.start();
}

}  // namespace
}  // namespace rocprofsys
