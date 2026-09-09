// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/domain_service.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <compare>
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

using ::testing::DoAll;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::InSequence;
using ::testing::NotNull;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrictMock;

// domain_service<SdkBackend, Externals> pulls in the full domains::registry<>, so this
// fake must satisfy the union of everything library/rocprofiler-sdk/{buffered,
// callback}/*.hpp touch on SdkBackend and Externals, plus the context/table members
// domain_service itself calls directly.
struct mock_sdk
{
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
    struct record_header_t
    {
        void* payload = nullptr;
    };
    struct callback_thread_id_t
    {
        std::uint64_t handle                                         = 0;
        auto          operator<=>(const callback_thread_id_t&) const = default;
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

    static constexpr int         BUFFER_POLICY_LOSSLESS                  = 1;
    static constexpr std::size_t BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS = 20;
    static constexpr std::size_t BUFFER_TRACING_KFD_EVENT_PAGE_FAULT     = 21;
    static constexpr std::size_t BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE   = 22;
    static constexpr std::size_t BUFFER_TRACING_KFD_EVENT_QUEUE          = 23;
    static constexpr std::size_t BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU = 24;
    static constexpr std::size_t BUFFER_TRACING_KFD_PAGE_FAULT           = 25;
    static constexpr std::size_t BUFFER_TRACING_KFD_PAGE_MIGRATE         = 26;
    static constexpr std::size_t BUFFER_TRACING_KFD_QUEUE                = 27;
    static constexpr std::size_t CALLBACK_TRACING_CODE_OBJECT            = 1;

    struct address_t
    {
        std::uint64_t value = 0;
    };
    struct agent_handle_t
    {
        std::uint64_t handle = 0;
    };

    struct kfd_event_dropped_record
    {
        std::uint32_t operation = 0;
        std::int32_t  pid       = 0;
        std::uint64_t timestamp = 0;
        std::uint64_t count     = 0;
    };
    struct kfd_event_page_fault_record
    {};
    struct kfd_event_page_migrate_record
    {};
    struct kfd_event_queue_record
    {
        std::uint32_t  operation = 0;
        std::int32_t   pid       = 0;
        agent_handle_t agent_id  = {};
        std::uint64_t  timestamp = 0;
    };
    struct kfd_event_unmap_record
    {
        std::uint32_t  operation     = 0;
        std::int32_t   pid           = 0;
        agent_handle_t agent_id      = {};
        std::uint64_t  timestamp     = 0;
        address_t      start_address = {};
        address_t      end_address   = {};
    };
    struct kfd_page_fault_record
    {
        std::uint32_t  operation       = 0;
        std::int32_t   pid             = 0;
        agent_handle_t agent_id        = {};
        std::uint64_t  start_timestamp = 0;
        std::uint64_t  end_timestamp   = 0;
        address_t      address         = {};
    };
    struct kfd_page_migrate_record
    {
        std::uint32_t  operation       = 0;
        std::int32_t   pid             = 0;
        agent_handle_t src_agent       = {};
        agent_handle_t dst_agent       = {};
        std::uint64_t  start_timestamp = 0;
        std::uint64_t  end_timestamp   = 0;
        address_t      start_address   = {};
        address_t      end_address     = {};
    };
    struct kfd_queue_record
    {
        std::uint32_t  operation       = 0;
        std::int32_t   pid             = 0;
        agent_handle_t agent_id        = {};
        std::uint64_t  start_timestamp = 0;
        std::uint64_t  end_timestamp   = 0;
    };

    // Doubles as: (1) the per-domain table domain_service::filter_supported_domains
    // iterates at construction time, and (2) the operation-name lookup that the
    // (address-taken but never invoked in these tests) production on_records bodies
    // call through SdkBackend::get_buffer_tracing_names().at(kind, operation).
    struct tracing_names_t
    {
        struct entry_t
        {
            std::string_view              name;
            std::vector<std::string_view> operations;
            domains::domain_id_t          value = 0;
        };

        std::vector<entry_t> entries;

        auto begin() const { return entries.begin(); }
        auto end() const { return entries.end(); }

        std::string_view at(std::size_t /*kind*/, std::uint32_t /*operation*/) const
        {
            return "operation";
        }
    };

    static tracing_names_t get_buffer_tracing_names();
    static tracing_names_t get_callback_tracing_names();

    using on_records_cb_t = void (*)(context_id_t, buffer_id_t, record_header_t**,
                                     std::size_t, void*, std::uint64_t);
    using on_record_cb_t  = void (*)(callback_tracing_record_t, user_data_t*, void*);

    static void create_context(context_id_t* context);
    static void start_context(context_id_t context);
    static void create_buffer(context_id_t context, std::size_t buffer_size,
                              std::size_t buffer_watermark, int policy,
                              on_records_cb_t callback, void* callback_data,
                              buffer_id_t* buffer_out);
    static void configure_buffer_tracing_service(context_id_t          context,
                                                 buffer_tracing_kind_t kind,
                                                 tracing_operation_t*  operations,
                                                 std::size_t           num_operations,
                                                 buffer_id_t           buffer);
    static void create_callback_thread(callback_thread_id_t* thread);
    static void assign_callback_thread(buffer_id_t buffer, callback_thread_id_t thread);
    static void flush_buffer(buffer_id_t buffer);
    static int  destroy_buffer(buffer_id_t buffer);
    static void configure_callback_tracing_service(context_id_t            context,
                                                   callback_tracing_kind_t kind,
                                                   tracing_operation_t*    operations,
                                                   std::size_t             num_operations,
                                                   on_record_cb_t          on_record,
                                                   void*                   callback_data);
};

// Minimal stand-in for the agent/trace_cache::info shapes touched through Externals by
// the on_record callbacks. These callbacks are never invoked by domain_service in
// these tests, only address-taken, but address-of still requires the bodies to
// compile; on_configure() callbacks, in contrast, ARE invoked by domain_service.
struct agent_t
{
    int         type              = 0;
    std::size_t device_type_index = 0;
};

// Every production on_configure() body calls exactly these two Externals members
// unconditionally (add_string, then get_agents_by_type); mocked so tests can verify
// on_configure() actually ran instead of just not crashing.
struct gmock_externals
{
    MOCK_METHOD(void, add_string, (std::string_view value));
    MOCK_METHOD(std::vector<std::shared_ptr<agent_t>>, get_agents_by_type, (int type));
};

std::unique_ptr<StrictMock<gmock_externals>> g_externals_mock;

struct externals
{
    struct pmc_info_t
    {
        int           type             = 0;
        std::size_t   agent_type_index = 0;
        std::string   target_arch;
        std::size_t   event_code  = 0;
        std::size_t   instance_id = 0;
        std::string   name;
        std::string   symbol;
        std::string   description;
        std::string   long_description;
        std::string   component;
        std::string   units;
        std::string   value_type;
        std::string   block;
        std::string   expression;
        std::uint32_t is_constant = 0;
        std::uint32_t is_derived  = 0;
        std::string   extdata;
    };

    struct thread_info_t
    {
        std::int32_t  parent_process_id = 0;
        std::int32_t  process_id        = 0;
        std::uint64_t thread_id         = 0;
        std::uint32_t start             = 0;
        std::uint32_t end               = 0;
        std::string   extdata;
    };

    struct track_t
    {
        std::string   track_name;
        std::uint64_t thread_id = 0;
        std::string   extdata;
    };

    struct kfd_sample_t
    {
        std::uint64_t               thread_id = 0;
        std::string                 name;
        std::uint64_t               start_timestamp = 0;
        std::uint64_t               end_timestamp   = 0;
        std::string                 args_str;
        std::string                 category;
        std::string                 track_name;
        std::string                 event_metadata;
        std::uint32_t               device_id   = 0;
        std::uint8_t                device_type = 0;
        std::string                 pmc_info_name;
        double                      value = 0.0;
        std::optional<std::int64_t> system_tid;
    };

    using agent_t = rocprofsys::agent_t;

    struct agent_manager_t
    {
        std::vector<std::shared_ptr<agent_t>> get_agents_by_type(int type)
        {
            return g_externals_mock->get_agents_by_type(type);
        }

        agent_t& get_agent_by_handle(std::uint64_t /*handle*/)
        {
            static agent_t placeholder{};
            return placeholder;
        }
    };

    static constexpr int AGENT_TYPE_GPU = 1;
    static constexpr int AGENT_TYPE_CPU = 0;

    static agent_manager_t& get_agent_manager()
    {
        static agent_manager_t manager;
        return manager;
    }

    static void add_string(std::string_view value)
    {
        g_externals_mock->add_string(value);
    }
    static void add_thread_info(const thread_info_t& /*info*/) {}
    static void add_track(const track_t& /*info*/) {}
    static void add_pmc_info(const pmc_info_t& /*info*/) {}
    static void buffer_storage_store(kfd_sample_t&& /*sample*/) {}

    static std::int32_t get_pid() { return 0; }
    static std::int32_t get_ppid() { return 0; }

    static constexpr std::string_view pmc_value_type_absolute = "ABS";

    static constexpr std::string_view kfd_event_dropped_events_category_name =
        "rocm_kfd_event_dropped_events";
    static constexpr std::string_view kfd_event_dropped_events_category_description =
        "KFD Dropped Events";
    static constexpr std::string_view kfd_event_queue_category_name =
        "rocm_kfd_event_queue";
    static constexpr std::string_view kfd_event_queue_category_description =
        "KFD Event Queue";
    static constexpr std::string_view kfd_event_unmap_from_gpu_category_name =
        "rocm_kfd_event_unmap_from_gpu";
    static constexpr std::string_view kfd_event_unmap_from_gpu_category_description =
        "KFD Unmap from GPU";
    static constexpr std::string_view kfd_page_fault_category_name =
        "rocm_kfd_page_fault";
    static constexpr std::string_view kfd_page_fault_category_description =
        "KFD Page Fault";
    static constexpr std::string_view kfd_page_migrate_category_name =
        "rocm_kfd_page_migrate";
    static constexpr std::string_view kfd_page_migrate_category_description =
        "KFD Page Migrate";
    static constexpr std::string_view kfd_queue_category_name        = "rocm_kfd_queue";
    static constexpr std::string_view kfd_queue_category_description = "KFD Queue";
};

struct gmock_sdk_backend
{
    MOCK_METHOD(void, create_context, (mock_sdk::context_id_t * context));
    MOCK_METHOD(void, start_context, (mock_sdk::context_id_t context));
    MOCK_METHOD(void, create_buffer,
                (mock_sdk::context_id_t context, std::size_t buffer_size,
                 std::size_t buffer_watermark, int policy,
                 mock_sdk::on_records_cb_t callback, void* callback_data,
                 mock_sdk::buffer_id_t* buffer_out));
    MOCK_METHOD(void, configure_buffer_tracing_service,
                (mock_sdk::context_id_t context, mock_sdk::buffer_tracing_kind_t kind,
                 mock_sdk::tracing_operation_t* operations, std::size_t num_operations,
                 mock_sdk::buffer_id_t buffer));
    MOCK_METHOD(void, create_callback_thread, (mock_sdk::callback_thread_id_t * thread));
    MOCK_METHOD(void, assign_callback_thread,
                (mock_sdk::buffer_id_t buffer, mock_sdk::callback_thread_id_t thread));
    MOCK_METHOD(void, flush_buffer, (mock_sdk::buffer_id_t buffer));
    MOCK_METHOD(int, destroy_buffer, (mock_sdk::buffer_id_t buffer));
    MOCK_METHOD(void, configure_callback_tracing_service,
                (mock_sdk::context_id_t context, mock_sdk::callback_tracing_kind_t kind,
                 mock_sdk::tracing_operation_t* operations, std::size_t num_operations,
                 mock_sdk::on_record_cb_t on_record, void* callback_data));
};

std::unique_ptr<StrictMock<gmock_sdk_backend>> g_mock;
mock_sdk::tracing_names_t                      g_buffer_table;
mock_sdk::tracing_names_t                      g_callback_table;

mock_sdk::tracing_names_t
mock_sdk::get_buffer_tracing_names()
{
    return g_buffer_table;
}

mock_sdk::tracing_names_t
mock_sdk::get_callback_tracing_names()
{
    return g_callback_table;
}

void
mock_sdk::create_context(context_id_t* context)
{
    g_mock->create_context(context);
}

void
mock_sdk::start_context(context_id_t context)
{
    g_mock->start_context(context);
}

void
mock_sdk::create_buffer(context_id_t context, std::size_t buffer_size,
                        std::size_t buffer_watermark, int policy,
                        on_records_cb_t callback, void* callback_data,
                        buffer_id_t* buffer_out)
{
    g_mock->create_buffer(context, buffer_size, buffer_watermark, policy, callback,
                          callback_data, buffer_out);
}

void
mock_sdk::configure_buffer_tracing_service(context_id_t          context,
                                           buffer_tracing_kind_t kind,
                                           tracing_operation_t*  operations,
                                           std::size_t num_operations, buffer_id_t buffer)
{
    g_mock->configure_buffer_tracing_service(context, kind, operations, num_operations,
                                             buffer);
}

void
mock_sdk::create_callback_thread(callback_thread_id_t* thread)
{
    g_mock->create_callback_thread(thread);
}

void
mock_sdk::assign_callback_thread(buffer_id_t buffer, callback_thread_id_t thread)
{
    g_mock->assign_callback_thread(buffer, thread);
}

void
mock_sdk::flush_buffer(buffer_id_t buffer)
{
    g_mock->flush_buffer(buffer);
}

int
mock_sdk::destroy_buffer(buffer_id_t buffer)
{
    return g_mock->destroy_buffer(buffer);
}

void
mock_sdk::configure_callback_tracing_service(
    context_id_t context, callback_tracing_kind_t kind, tracing_operation_t* operations,
    std::size_t num_operations, on_record_cb_t on_record, void* callback_data)
{
    g_mock->configure_callback_tracing_service(context, kind, operations, num_operations,
                                               on_record, callback_data);
}

using sut_t = domain_service<mock_sdk, externals>;

// Matches a tracing_operation_t* argument whose first expected.size() elements equal
// expected exactly. The pointee address is an implementation-internal detail of
// domain_service::configure_domain() (a freshly built local vector), so identity
// cannot be asserted -- contents can, and are asserted exactly.
MATCHER_P(OperationsEqual, expected, "")
{
    return std::equal(expected.begin(), expected.end(), arg);
}

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
    // get_agents_by_type(AGENT_TYPE_GPU), unconditionally and exactly once; asserting
    // both confirms on_configure() actually ran rather than merely not crashing.
    void expect_on_configure_ran(std::string_view category_name)
    {
        InSequence seq;
        EXPECT_CALL(*g_externals_mock, add_string(Eq(category_name))).Times(1);
        EXPECT_CALL(*g_externals_mock, get_agents_by_type(Eq(externals::AGENT_TYPE_GPU)))
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
            .WillOnce(DoAll(SetArgPointee<6>(buffer), Return()));

        EXPECT_CALL(*g_mock, configure_buffer_tracing_service(
                                 Eq(context), Eq(kind), OperationsEqual(operations),
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
                                 Eq(context), Eq(kind), OperationsEqual(operations),
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
                     { .name = "unsupported_domain", .operations = {}, .value = 999 } }
    };
    g_callback_table = mock_sdk::tracing_names_t{
        .entries = { { .name       = "code_object",
                       .operations = {},
                       .value      = mock_sdk::CALLBACK_TRACING_CODE_OBJECT } }
    };

    sut_t service;

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
    expect_on_configure_ran(externals::kfd_queue_category_name);
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
    expect_on_configure_ran(externals::kfd_queue_category_name);
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
    expect_on_configure_ran(externals::kfd_queue_category_name);
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
    expect_on_configure_ran(externals::kfd_page_fault_category_name);
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
    constexpr const auto& code_object_definition =
        domains::callback::k_code_object<mock_sdk, externals>;
    ASSERT_NE(code_object_definition.on_configure, nullptr);

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
    expect_on_configure_ran(externals::kfd_queue_category_name);
    expect_configure_buffered(
        context, page_fault_buffer, page_fault_thread,
        static_cast<mock_sdk::buffer_tracing_kind_t>(
            mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT),
        domains::buffered::k_kfd_page_fault<mock_sdk, externals>.on_records, { 0 });
    expect_on_configure_ran(externals::kfd_page_fault_category_name);
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
    expect_on_configure_ran(externals::kfd_queue_category_name);
    expect_configure_buffered(
        context, page_fault_buffer, page_fault_thread,
        static_cast<mock_sdk::buffer_tracing_kind_t>(
            mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT),
        domains::buffered::k_kfd_page_fault<mock_sdk, externals>.on_records, { 0 });
    expect_on_configure_ran(externals::kfd_page_fault_category_name);
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
