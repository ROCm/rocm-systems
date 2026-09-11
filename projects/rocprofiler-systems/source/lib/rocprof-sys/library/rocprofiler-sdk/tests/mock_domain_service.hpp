// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <gmock/gmock.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::domains::test_support
{

// Shared stand-in for SdkBackend, satisfying domain_service_backend for every test
// that instantiates callback_domain<>, buffered_domain<>, registry<>, domain_service<>,
// or a single kfd_* on_record(s)/on_configure() pair. Carries every kfd_* domain's
// record type and BUFFER_TRACING_*/CALLBACK_TRACING_* id, since registry<>/
// domain_service<> pull in the full domains::registry<> regardless of which single
// domain a given test exercises.
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

struct agent_id_t
{
    std::uint64_t handle = 0;
};

struct address_t
{
    std::uint64_t value = 0;
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
    std::uint32_t operation = 0;
    std::int32_t  pid       = 0;
    agent_id_t    agent_id{};
    std::uint64_t timestamp = 0;
};
struct kfd_event_unmap_record
{
    std::uint32_t operation = 0;
    std::int32_t  pid       = 0;
    agent_id_t    agent_id{};
    std::uint64_t timestamp = 0;
    address_t     start_address{};
    address_t     end_address{};
};
struct kfd_page_fault_record
{
    std::uint32_t operation = 0;
    std::int32_t  pid       = 0;
    agent_id_t    agent_id{};
    address_t     address{};
    std::uint64_t start_timestamp = 0;
    std::uint64_t end_timestamp   = 0;
};
struct kfd_page_migrate_record
{
    std::uint32_t operation = 0;
    std::int32_t  pid       = 0;
    agent_id_t    src_agent{};
    agent_id_t    dst_agent{};
    address_t     start_address{};
    address_t     end_address{};
    std::uint64_t start_timestamp = 0;
    std::uint64_t end_timestamp   = 0;
};
struct kfd_queue_record
{
    std::uint32_t operation = 0;
    std::int32_t  pid       = 0;
    agent_id_t    agent_id{};
    std::uint64_t start_timestamp = 0;
    std::uint64_t end_timestamp   = 0;
};

// Satisfies domain_service_backend's requirement that get_{buffer,callback}_
// tracing_names() return a std::ranges::range of entries exposing
// name/operations/value. Kept as a plain (non-gmock) value: on_kfd_*<...> calls
// get_buffer_tracing_names().at(...) unconditionally, so it must work without a test
// having to set up an expectation for it.
struct tracing_names_t
{
    struct entry_t
    {
        std::string_view              name;
        std::vector<std::string_view> operations;
        std::size_t                   value = 0;
    };

    std::vector<entry_t> entries;

    [[nodiscard]] auto begin() const { return entries.begin(); }
    [[nodiscard]] auto end() const { return entries.end(); }

    [[nodiscard]] std::string_view at(std::size_t /*kind*/,
                                      std::uint32_t /*operation*/) const
    {
        return "operation";
    }
};

// Every domain_service_backend member that tests actually verify calls into, as a
// GMock method -- a test EXPECT_CALLs only the handful its scenario exercises;
// StrictMock fails it if anything else is touched. get_{buffer,callback}_tracing_names
// are deliberately NOT here; see tracing_names_t above.
struct gmock_sdk_backend
{
    MOCK_METHOD(void, create_context, (context_id_t * context));
    MOCK_METHOD(void, start_context, (context_id_t context));
    // NOLINTNEXTLINE(readability-function-size)
    MOCK_METHOD(void, create_buffer,
                (context_id_t context, std::size_t buffer_size,
                 std::size_t buffer_watermark, buffer_policy_t policy,
                 on_records_cb_t callback, void* callback_data, buffer_id_t* buffer_out));
    // NOLINTNEXTLINE(readability-function-size)
    MOCK_METHOD(void, configure_buffer_tracing_service,
                (context_id_t context, buffer_tracing_kind_t kind,
                 tracing_operation_t* operations, std::size_t num_operations,
                 buffer_id_t buffer));
    MOCK_METHOD(void, create_callback_thread, (callback_thread_id_t * thread));
    MOCK_METHOD(void, assign_callback_thread,
                (buffer_id_t buffer, callback_thread_id_t thread));
    MOCK_METHOD(void, flush_buffer, (buffer_id_t buffer));
    MOCK_METHOD(int, destroy_buffer, (buffer_id_t buffer));
    // NOLINTNEXTLINE(readability-function-size)
    MOCK_METHOD(void, configure_callback_tracing_service,
                (context_id_t context, callback_tracing_kind_t kind,
                 tracing_operation_t* operations, std::size_t num_operations,
                 on_record_cb_t on_record, void* callback_data));
};

inline std::unique_ptr<::testing::StrictMock<gmock_sdk_backend>> g_mock;

// Settable by domain_service<> tests to drive filter_supported_domains() with a
// specific set of supported buffered/callback domains; left empty (the default) by
// tests -- e.g. registry<>/callback_domain<>/buffered_domain<> tests -- that never
// call get_{buffer,callback}_tracing_names().
inline tracing_names_t g_buffer_table;
inline tracing_names_t g_callback_table;

// SdkBackend stand-in: every lifecycle member forwards to g_mock, so tests drive
// behavior entirely through EXPECT_CALL instead of hand-written shim bodies. Tests
// that never invoke a given member (e.g. the kfd_* on_record(s)/on_configure() tests,
// which call the free function directly instead of going through buffered_domain<>)
// simply never touch g_mock and can leave it unset.
struct mock_sdk
{
    using context_id_t              = test_support::context_id_t;
    using buffer_id_t               = test_support::buffer_id_t;
    using callback_thread_id_t      = test_support::callback_thread_id_t;
    using record_header_t           = test_support::record_header_t;
    using user_data_t               = test_support::user_data_t;
    using callback_tracing_record_t = test_support::callback_tracing_record_t;
    using tracing_operation_t       = test_support::tracing_operation_t;
    using buffer_tracing_kind_t     = test_support::buffer_tracing_kind_t;
    using callback_tracing_kind_t   = test_support::callback_tracing_kind_t;
    using buffer_policy_t           = test_support::buffer_policy_t;
    using on_records_cb_t           = test_support::on_records_cb_t;
    using on_record_cb_t            = test_support::on_record_cb_t;
    using tracing_names_t           = test_support::tracing_names_t;

    // NOLINTBEGIN(readability-identifier-naming)
    static constexpr std::size_t     compile_time_version                    = 90909;
    static constexpr buffer_policy_t BUFFER_POLICY_LOSSLESS                  = 1;
    static constexpr std::size_t     BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS = 20;
    static constexpr std::size_t     BUFFER_TRACING_KFD_EVENT_PAGE_FAULT     = 21;
    static constexpr std::size_t     BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE   = 22;
    static constexpr std::size_t     BUFFER_TRACING_KFD_EVENT_QUEUE          = 23;
    static constexpr std::size_t     BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU = 24;
    static constexpr std::size_t     BUFFER_TRACING_KFD_PAGE_FAULT           = 25;
    static constexpr std::size_t     BUFFER_TRACING_KFD_PAGE_MIGRATE         = 26;
    static constexpr std::size_t     BUFFER_TRACING_KFD_QUEUE                = 27;
    static constexpr std::size_t     CALLBACK_TRACING_CODE_OBJECT            = 1;
    // NOLINTEND(readability-identifier-naming)

    using kfd_event_dropped_record      = test_support::kfd_event_dropped_record;
    using kfd_event_page_fault_record   = test_support::kfd_event_page_fault_record;
    using kfd_event_page_migrate_record = test_support::kfd_event_page_migrate_record;
    using kfd_event_queue_record        = test_support::kfd_event_queue_record;
    using kfd_event_unmap_record        = test_support::kfd_event_unmap_record;
    using kfd_page_fault_record         = test_support::kfd_page_fault_record;
    using kfd_page_migrate_record       = test_support::kfd_page_migrate_record;
    using kfd_queue_record              = test_support::kfd_queue_record;

    static void create_context(context_id_t* context) { g_mock->create_context(context); }
    static void start_context(context_id_t context) { g_mock->start_context(context); }

    // NOLINTNEXTLINE(readability-function-size)
    static void create_buffer(context_id_t context, std::size_t buffer_size,
                              std::size_t buffer_watermark, buffer_policy_t policy,
                              on_records_cb_t callback, void* callback_data,
                              buffer_id_t* buffer_out)
    {
        g_mock->create_buffer(context, buffer_size, buffer_watermark, policy, callback,
                              callback_data, buffer_out);
    }

    // NOLINTNEXTLINE(readability-function-size)
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

    // NOLINTNEXTLINE(readability-function-size)
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

    static tracing_names_t get_buffer_tracing_names() { return g_buffer_table; }
    static tracing_names_t get_callback_tracing_names() { return g_callback_table; }
};

// Stand-in for the agent/trace_cache::info shapes every on_kfd_*<...> touches through
// Externals. Every field beyond type/device_type_index exists solely so agent_t
// satisfies policies::agent_policy (required transitively by
// domain_service_externals's agent_manager_policy check); tests never read them.
struct agent_t
{
    int           type                 = 0;
    std::uint64_t handle               = 0;
    std::uint64_t device_id            = 0;
    std::uint32_t node_id              = 0;
    std::int32_t  logical_node_id      = 0;
    std::int32_t  logical_node_type_id = 0;
    std::string   name;
    std::string   model_name;
    std::string   vendor_name;
    std::string   product_name;
    std::size_t   device_type_index = 0;
    std::string   agent_info;
    std::uint32_t location_id = 0;
    std::uint32_t domain      = 0;
    bool          hip_visible = true;
};

struct pmc_info_data_t
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

// Every production on_configure() body calls exactly these members
// unconditionally-or-conditionally; mocked so tests can verify they ran correctly
// instead of just not crashing.
struct gmock_externals
{
    MOCK_METHOD(void, add_string, (std::string_view value));
    MOCK_METHOD(std::vector<std::shared_ptr<agent_t>>, get_agents_by_type, (int type));
    MOCK_METHOD(void, add_pmc_info, (const pmc_info_data_t& info));
};

inline std::unique_ptr<::testing::StrictMock<gmock_externals>> g_externals_mock;

// Externals mirrors the real ExternalDeps policy surface used by every on_kfd_* and
// on_kfd_*_configure, plus domain_service<>/registry<>. The on_records-only members
// (add_thread_info/add_track/buffer_storage_store) stay plain no-ops -- only
// add_string/get_agents_by_type/add_pmc_info, which on_configure() exercises, are
// mocked. Category name/description constants for every kfd_* domain are carried
// here since domain_service_externals requires the full set regardless of which
// single domain a given test exercises.
struct externals
{
    using agent_t      = test_support::agent_t;
    using pmc_info_t   = pmc_info_data_t;
    using agent_type_t = int;

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

    // Satisfies policies::agent_manager_policy (required transitively by
    // domain_service_externals); only get_agents_by_type and the single-argument
    // get_agent_by_handle are ever exercised by these tests, so the rest are
    // unreachable stubs.
    struct agent_manager_t
    {
        agent_manager_t() = default;
        explicit agent_manager_t(const std::vector<std::shared_ptr<agent_t>>& /*agents*/)
        {}

        void insert_agent(agent_t& /*agent*/) {}

        [[nodiscard]] std::vector<std::shared_ptr<agent_t>> get_agents_by_type(
            int type) const
        {
            return g_externals_mock->get_agents_by_type(type);
        }

        [[nodiscard]] const agent_t& get_agent_by_type_index(std::size_t /*type_index*/,
                                                             int /*type*/) const
        {
            static const agent_t k_placeholder{};
            return k_placeholder;
        }

        [[nodiscard]] const agent_t& get_agent_by_id(std::size_t /*device_id*/,
                                                     int /*type*/) const
        {
            static const agent_t k_placeholder{};
            return k_placeholder;
        }

        [[nodiscard]] const agent_t& get_agent_by_handle(std::size_t /*handle*/,
                                                         int /*type*/) const
        {
            static const agent_t k_placeholder{};
            return k_placeholder;
        }

        [[nodiscard]] const agent_t& get_agent_by_handle(std::uint64_t /*handle*/) const
        {
            static const agent_t k_placeholder{};
            return k_placeholder;
        }

        [[nodiscard]] std::vector<std::shared_ptr<agent_t>> get_agents() const
        {
            return {};
        }

        [[nodiscard]] std::size_t get_gpu_agents_count() const { return 0; }
        [[nodiscard]] std::size_t get_cpu_agents_count() const { return 0; }
    };

    static constexpr int k_agent_type_gpu = 1;
    static constexpr int k_agent_type_cpu = 0;

    static agent_manager_t& get_agent_manager()
    {
        static agent_manager_t s_manager;
        return s_manager;
    }

    static void add_string(std::string_view value)
    {
        g_externals_mock->add_string(value);
    }
    static void add_thread_info(const thread_info_t& /*info*/) {}
    static void add_track(const track_t& /*info*/) {}
    static void add_pmc_info(const pmc_info_t& info)
    {
        g_externals_mock->add_pmc_info(info);
    }
    static void buffer_storage_store(kfd_sample_t&& /*sample*/) {}

    static std::int32_t get_pid() { return 0; }
    static std::int32_t get_ppid() { return 0; }

    static constexpr std::string_view k_pmc_value_type_absolute = "ABS";

    static constexpr std::string_view k_kfd_event_dropped_events_category_name =
        "rocm_kfd_event_dropped_events";
    static constexpr std::string_view k_kfd_event_dropped_events_category_description =
        "KFD Dropped Events";
    static constexpr std::string_view k_kfd_event_queue_category_name =
        "rocm_kfd_event_queue";
    static constexpr std::string_view k_kfd_event_queue_category_description =
        "KFD Event Queue";
    static constexpr std::string_view k_kfd_event_unmap_from_gpu_category_name =
        "rocm_kfd_event_unmap_from_gpu";
    static constexpr std::string_view k_kfd_event_unmap_from_gpu_category_description =
        "KFD Unmap from GPU";
    static constexpr std::string_view k_kfd_page_fault_category_name =
        "rocm_kfd_page_fault";
    static constexpr std::string_view k_kfd_page_fault_category_description =
        "KFD Page Fault Events";
    static constexpr std::string_view k_kfd_page_migrate_category_name =
        "rocm_kfd_page_migrate";
    static constexpr std::string_view k_kfd_page_migrate_category_description =
        "KFD Page Migrate Events";
    static constexpr std::string_view k_kfd_queue_category_name = "rocm_kfd_queue";
    static constexpr std::string_view k_kfd_queue_category_description =
        "KFD Queue Events";
};

}  // namespace rocprofsys::domains::test_support
