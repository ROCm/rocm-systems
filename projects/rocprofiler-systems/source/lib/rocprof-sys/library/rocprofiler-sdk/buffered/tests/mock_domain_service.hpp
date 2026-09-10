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

namespace rocprofsys::domains::buffered::test_support
{

// Shared stand-in for SdkBackend, satisfying domain_service_backend for every kfd_*
// buffered domain test. Each on_kfd_*<SdkBackend, Externals> only ever touches its own
// record type, but k_kfd_*<SdkBackend, Externals> requires SdkBackend to satisfy the
// full concept, so this mock carries every domain's record type and constant.
struct mock_sdk
{
    struct context_id_t
    {
        std::uint64_t handle = 0;
    };
    struct buffer_id_t
    {
        std::uint64_t handle = 0;
    };
    struct record_header_t
    {
        std::uint32_t category = 0;
        std::uint32_t kind     = 0;
        void*         payload  = nullptr;
    };
    struct callback_thread_id_t
    {
        std::uint64_t handle = 0;
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

    static constexpr buffer_policy_t BUFFER_POLICY_LOSSLESS                  = 1;
    static constexpr std::size_t     BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS = 20;
    static constexpr std::size_t     BUFFER_TRACING_KFD_EVENT_PAGE_FAULT     = 21;
    static constexpr std::size_t     BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE   = 22;
    static constexpr std::size_t     BUFFER_TRACING_KFD_EVENT_QUEUE          = 23;
    static constexpr std::size_t     BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU = 24;
    static constexpr std::size_t     BUFFER_TRACING_KFD_PAGE_FAULT           = 25;
    static constexpr std::size_t     BUFFER_TRACING_KFD_PAGE_MIGRATE         = 26;
    static constexpr std::size_t     BUFFER_TRACING_KFD_QUEUE                = 27;

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
    // name/operations/value; no test in this suite iterates it, so it stays empty.
    struct buffer_tracing_names_t
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

    static buffer_tracing_names_t get_buffer_tracing_names() { return {}; }
    static buffer_tracing_names_t get_callback_tracing_names() { return {}; }

    using on_records_cb_t = void (*)(context_id_t, buffer_id_t, record_header_t**,
                                     std::size_t, void*, std::uint64_t);
    using on_record_cb_t  = void (*)(callback_tracing_record_t, user_data_t*, void*);

    // domain_service_backend requires these members to exist and be callable with
    // the signatures below; no test in this suite exercises domain_service itself, so
    // the bodies are unreachable no-ops.
    static void create_context(context_id_t* /*context*/) {}
    static void start_context(context_id_t /*context*/) {}
    static void create_buffer(context_id_t /*context*/, std::size_t /*buffer_size*/,
                              std::size_t /*buffer_watermark*/, int /*policy*/,
                              on_records_cb_t /*callback*/, void* /*callback_data*/,
                              buffer_id_t* /*buffer_out*/)
    {}
    static void configure_buffer_tracing_service(context_id_t /*context*/,
                                                 buffer_tracing_kind_t /*kind*/,
                                                 tracing_operation_t* /*operations*/,
                                                 std::size_t /*num_operations*/,
                                                 buffer_id_t /*buffer*/)
    {}
    static void create_callback_thread(callback_thread_id_t* /*thread*/) {}
    static void assign_callback_thread(buffer_id_t /*buffer*/,
                                       callback_thread_id_t /*thread*/)
    {}
    static void flush_buffer(buffer_id_t /*buffer*/) {}
    static int  destroy_buffer(buffer_id_t /*buffer*/) { return 0; }
    static void configure_callback_tracing_service(context_id_t /*context*/,
                                                   callback_tracing_kind_t /*kind*/,
                                                   tracing_operation_t* /*operations*/,
                                                   std::size_t /*num_operations*/,
                                                   on_record_cb_t /*on_record*/,
                                                   void* /*callback_data*/)
    {}
};

// Stand-in for the agent/trace_cache::info shapes every on_kfd_*<...> touches through
// Externals. Every field beyond type/device_type_index exists solely so agent_t
// satisfies policies::agent_policy (required transitively by
// domain_service_externals's agent_manager_policy check); the tests never read them.
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
// on_kfd_*_configure. The on_records-only members
// (add_thread_info/add_track/buffer_storage_store) stay plain no-ops -- only
// add_string/get_agents_by_type/add_pmc_info, which on_configure() exercises, are
// mocked. Category name/description constants for every kfd_* domain are carried
// here since domain_service_externals requires the full set regardless of which
// single domain a given test file exercises.
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

        std::vector<std::shared_ptr<agent_t>> get_agents_by_type(int type) const
        {
            return g_externals_mock->get_agents_by_type(type);
        }

        const agent_t& get_agent_by_type_index(std::size_t /*type_index*/,
                                               int /*type*/) const
        {
            static agent_t placeholder{};
            return placeholder;
        }

        const agent_t& get_agent_by_id(std::size_t /*device_id*/, int /*type*/) const
        {
            static agent_t placeholder{};
            return placeholder;
        }

        const agent_t& get_agent_by_handle(std::size_t /*handle*/, int /*type*/) const
        {
            static agent_t placeholder{};
            return placeholder;
        }

        const agent_t& get_agent_by_handle(std::uint64_t /*handle*/) const
        {
            static agent_t placeholder{};
            return placeholder;
        }

        std::vector<std::shared_ptr<agent_t>> get_agents() const { return {}; }

        std::size_t get_gpu_agents_count() const { return 0; }
        std::size_t get_cpu_agents_count() const { return 0; }
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
    static void add_pmc_info(const pmc_info_t& info)
    {
        g_externals_mock->add_pmc_info(info);
    }
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
        "KFD Page Fault Events";
    static constexpr std::string_view kfd_page_migrate_category_name =
        "rocm_kfd_page_migrate";
    static constexpr std::string_view kfd_page_migrate_category_description =
        "KFD Page Migrate Events";
    static constexpr std::string_view kfd_queue_category_name        = "rocm_kfd_queue";
    static constexpr std::string_view kfd_queue_category_description = "KFD Queue Events";
};

}  // namespace rocprofsys::domains::buffered::test_support
