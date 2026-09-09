// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/buffered/kfd_page_fault.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::domains::buffered
{
namespace
{

using ::testing::AllOf;
using ::testing::Eq;
using ::testing::Field;
using ::testing::Return;
using ::testing::StrictMock;

// Self-contained stand-in for SdkBackend: kfd_page_fault<SdkBackend, Externals> and
// on_kfd_page_fault<SdkBackend, Externals> only ever touch these members.
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

    static constexpr std::size_t BUFFER_TRACING_KFD_PAGE_FAULT = 30;

    struct agent_id_t
    {
        std::uint64_t handle = 0;
    };

    struct address_t
    {
        std::uint64_t value = 0;
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

    struct buffer_tracing_names_t
    {
        std::string_view at(std::size_t /*kind*/, std::uint32_t /*operation*/) const
        {
            return "operation";
        }
    };

    static buffer_tracing_names_t get_buffer_tracing_names() { return {}; }
};

// Minimal stand-in for the agent/trace_cache::info shapes that
// on_kfd_page_fault<mock_sdk, externals> touches through Externals.
struct agent_t
{
    int         type              = 0;
    std::size_t device_type_index = 0;
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

// on_kfd_page_fault_configure<Externals> calls exactly these three members
// unconditionally-or-conditionally; mocked so tests can verify it ran correctly
// instead of just not crashing.
struct gmock_externals
{
    MOCK_METHOD(void, add_string, (std::string_view value));
    MOCK_METHOD(std::vector<std::shared_ptr<agent_t>>, get_agents_by_type, (int type));
    MOCK_METHOD(void, add_pmc_info, (const pmc_info_data_t& info));
};

std::unique_ptr<StrictMock<gmock_externals>> g_externals_mock;

// Externals mirrors the real ExternalDeps policy surface used by on_kfd_page_fault
// and on_kfd_page_fault_configure. The on_records-only members
// (add_thread_info/add_track/buffer_storage_store) stay plain no-ops -- only
// add_string/get_agents_by_type/add_pmc_info, which on_configure() exercises, are
// mocked.
struct externals
{
    using agent_t    = buffered::agent_t;
    using pmc_info_t = pmc_info_data_t;

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
    static void add_pmc_info(const pmc_info_t& info)
    {
        g_externals_mock->add_pmc_info(info);
    }
    static void buffer_storage_store(kfd_sample_t&& /*sample*/) {}

    static std::int32_t get_pid() { return 0; }
    static std::int32_t get_ppid() { return 0; }

    static constexpr std::string_view pmc_value_type_absolute = "ABS";
    static constexpr std::string_view kfd_page_fault_category_name =
        "rocm_kfd_page_fault";
    static constexpr std::string_view kfd_page_fault_category_description =
        "KFD Page Fault Events";
};

}  // namespace

TEST(kfd_page_fault_test, descriptor_reports_correct_metadata)
{
    using mock_dispatcher =
        buffered_callback_dispatcher<mock_sdk, mock_sdk::kfd_page_fault_record,
                                     on_kfd_page_fault<mock_sdk, externals>>;
    constexpr const auto& domain = k_kfd_page_fault<mock_sdk, externals>;

    EXPECT_EQ(domain.meta.name, "kfd_page_fault");
    EXPECT_EQ(domain.meta.id, mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT);
    EXPECT_EQ(domain.meta.mode, collection_mode::buffered);
    ASSERT_TRUE(domain.meta.group.has_value());
    EXPECT_EQ(domain.meta.group->name, "kfd_events");
    EXPECT_EQ(domain.on_records, &mock_dispatcher::callback);
}

TEST(kfd_page_fault_test, descriptor_uses_default_buffer_properties)
{
    constexpr const auto& domain = k_kfd_page_fault<mock_sdk, externals>;

    EXPECT_EQ(domain.buffer.buffer_size, k_default_buffer_properties.buffer_size);
    EXPECT_EQ(domain.buffer.buffer_watermark,
              k_default_buffer_properties.buffer_watermark);
}

TEST(kfd_page_fault_test, on_kfd_page_fault_handles_empty_record_batch_without_crashing)
{
    mock_sdk::kfd_page_fault_record record{};

    on_kfd_page_fault<mock_sdk, externals>(&record, nullptr);
}

TEST(kfd_page_fault_test,
     on_configure_registers_category_string_and_skips_pmc_info_without_gpu_agents)
{
    g_externals_mock = std::make_unique<StrictMock<gmock_externals>>();

    EXPECT_CALL(*g_externals_mock,
                add_string(Eq(externals::kfd_page_fault_category_name)))
        .Times(1);
    EXPECT_CALL(*g_externals_mock, get_agents_by_type(Eq(externals::AGENT_TYPE_GPU)))
        .Times(1)
        .WillOnce(Return(std::vector<std::shared_ptr<agent_t>>{}));

    on_kfd_page_fault_configure<externals>();

    g_externals_mock.reset();
}

TEST(kfd_page_fault_test, on_configure_registers_pmc_info_for_each_gpu_agent)
{
    g_externals_mock = std::make_unique<StrictMock<gmock_externals>>();

    auto gpu_agent = std::make_shared<agent_t>(
        agent_t{ .type = externals::AGENT_TYPE_GPU, .device_type_index = 1 });

    EXPECT_CALL(*g_externals_mock,
                add_string(Eq(externals::kfd_page_fault_category_name)))
        .Times(1);
    EXPECT_CALL(*g_externals_mock, get_agents_by_type(Eq(externals::AGENT_TYPE_GPU)))
        .Times(1)
        .WillOnce(Return(std::vector<std::shared_ptr<agent_t>>{ gpu_agent }));
    EXPECT_CALL(
        *g_externals_mock,
        add_pmc_info(AllOf(
            Field(&pmc_info_data_t::type, Eq(externals::AGENT_TYPE_GPU)),
            Field(&pmc_info_data_t::agent_type_index, Eq(std::size_t{ 1 })),
            Field(&pmc_info_data_t::target_arch, Eq(std::string{ "GPU" })),
            Field(&pmc_info_data_t::name,
                  Eq(std::string{ externals::kfd_page_fault_category_name })),
            Field(&pmc_info_data_t::symbol, Eq(std::string{ "KFD Page Fault Events" })),
            Field(&pmc_info_data_t::description,
                  Eq(std::string{ externals::kfd_page_fault_category_description })))))
        .Times(1);

    on_kfd_page_fault_configure<externals>();

    g_externals_mock.reset();
}

}  // namespace rocprofsys::domains::buffered
