// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/domain_registry.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::domains
{
namespace
{

// registry<SdkBackend, Externals> instantiates every buffered/callback domain
// definition in library/rocprofiler-sdk/{buffered,callback}/*.hpp, so this fake must
// satisfy the union of everything those headers touch on SdkBackend and Externals --
// not just what a single domain needs.
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
        void* payload = nullptr;
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

    struct buffer_tracing_names_t
    {
        std::string_view at(std::size_t /*kind*/, std::uint32_t /*operation*/) const
        {
            return "operation";
        }
    };

    static buffer_tracing_names_t get_buffer_tracing_names() { return {}; }
};

// Minimal stand-in for the agent/trace_cache::info shapes touched through Externals by
// the on_record callbacks. These callbacks are never invoked by registry<>, only
// address-taken, but address-of still requires the function bodies to compile.
struct agent_t
{
    int         type              = 0;
    std::size_t device_type_index = 0;
};

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

    using agent_t = rocprofsys::domains::agent_t;

    struct agent_manager_t
    {
        std::vector<std::shared_ptr<agent_t>> get_agents_by_type(int /*type*/)
        {
            return {};
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

    static void add_string(std::string_view /*value*/) {}
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

using sut_t = registry<mock_sdk, externals>;

TEST(domain_registry_test, find_descriptor_finds_buffered_domain_case_insensitively)
{
    const domain_descriptor* descriptor = sut_t::find_descriptor("KfD_QuEuE");

    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(descriptor->name, "kfd_queue");
    EXPECT_EQ(descriptor->id, mock_sdk::BUFFER_TRACING_KFD_QUEUE);
    EXPECT_EQ(descriptor->mode, collection_mode::buffered);
}

TEST(domain_registry_test, find_descriptor_finds_callback_domain_case_insensitively)
{
    const domain_descriptor* descriptor = sut_t::find_descriptor("CODE_OBJECT");

    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(descriptor->name, "code_object");
    EXPECT_EQ(descriptor->id, mock_sdk::CALLBACK_TRACING_CODE_OBJECT);
    EXPECT_EQ(descriptor->mode, collection_mode::callback);
}

TEST(domain_registry_test, find_descriptor_returns_nullptr_for_unknown_name)
{
    EXPECT_EQ(sut_t::find_descriptor("not_a_real_domain"), nullptr);
}

TEST(domain_registry_test, get_buffered_returns_definition_matching_domain_id)
{
    const auto& definition = sut_t::get_buffered(mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT);

    EXPECT_EQ(definition.meta.name, "kfd_page_fault");
    EXPECT_EQ(definition.meta.id, mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT);
}

TEST(domain_registry_test, get_buffered_throws_runtime_error_for_unknown_domain_id)
{
    constexpr domain_id_t k_unknown_id = 9999;

    EXPECT_THROW(
        { static_cast<void>(sut_t::get_buffered(k_unknown_id)); }, std::runtime_error);
}

TEST(domain_registry_test, get_callback_returns_definition_matching_domain_id)
{
    const auto& definition = sut_t::get_callback(mock_sdk::CALLBACK_TRACING_CODE_OBJECT);

    EXPECT_EQ(definition.meta.name, "code_object");
    EXPECT_EQ(definition.meta.id, mock_sdk::CALLBACK_TRACING_CODE_OBJECT);
}

TEST(domain_registry_test, get_callback_throws_runtime_error_for_unknown_domain_id)
{
    constexpr domain_id_t k_unknown_id = 9999;

    EXPECT_THROW(
        { static_cast<void>(sut_t::get_callback(k_unknown_id)); }, std::runtime_error);
}

}  // namespace
}  // namespace rocprofsys::domains
