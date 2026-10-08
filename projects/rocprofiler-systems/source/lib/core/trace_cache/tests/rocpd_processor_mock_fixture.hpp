// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/agent_manager.hpp"
#include "core/common_types.hpp"
#include "core/config.hpp"
#include "core/node_info.hpp"
#include "core/output_file_registry.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/rocpd_processor.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "mock_profiler_hub_writer.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <timemory/settings/settings.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>

namespace rocprofsys::trace_cache::test
{

inline constexpr int           k_pid           = 200;
inline constexpr int           k_ppid          = 100;
inline constexpr std::uint64_t k_thread_id     = 300;
inline constexpr std::uint64_t k_queue_id      = 10;
inline constexpr std::uint64_t k_stream_id     = 20;
inline constexpr std::uint64_t k_gpu_handle    = 1;
inline constexpr std::uint64_t k_cpu_handle    = 2;
inline constexpr std::size_t   k_process_start = 1000;
inline constexpr std::size_t   k_process_end   = 9000;

[[nodiscard]] inline writer_types::agent_unique_id_t
make_uid(std::string_view type, std::size_t type_index)
{
    return { .agent_type = type, .type_index = type_index };
}

[[nodiscard]] inline std::size_t
node_id()
{
    return rocprofsys::node_info::get_instance().id;
}

template <typename Category>
[[nodiscard]] std::string_view
category_name()
{
    return rocprofsys::trait::name<Category>::value;
}

// ---- trace_environment ------------------------------------------------------

struct expected_env
{
    std::optional<std::size_t>                     thread_id = k_thread_id;
    std::optional<writer_types::agent_unique_id_t> agent_id;
    std::optional<std::size_t>                     stream_id;
    std::optional<std::size_t>                     queue_id;
    std::optional<std::string_view>                track_name;
};

[[nodiscard]] inline bool
env_equals(const writer_types::trace_environment_t& env, const expected_env& expected)
{
    return env.node_id == node_id() &&
           env.process_id == static_cast<std::size_t>(k_pid) &&
           env.thread_id == expected.thread_id && env.agent_id == expected.agent_id &&
           env.stream_id == expected.stream_id && env.queue_id == expected.queue_id &&
           env.track_name == expected.track_name;
}

MATCHER_P(IsEnv, expected, "trace environment") { return env_equals(arg, expected); }

// ---- event_data -------------------------------------------------------------

struct expected_event
{
    std::optional<std::size_t> stack_id;
    std::optional<std::size_t> parent_stack_id;
    std::optional<std::size_t> correlation_id;
    std::string_view           category;
    std::string_view           extdata = "{}";
};

[[nodiscard]] inline bool
event_equals(const std::optional<writer_types::event_data_t>& event,
             const expected_event&                            expected)
{
    return event.has_value() && event->stack_id == expected.stack_id &&
           event->parent_stack_id == expected.parent_stack_id &&
           event->correlation_id == expected.correlation_id &&
           event->event_category == expected.category &&
           event->extdata == expected.extdata;
}

// ---- pmc event --------------------------------------------------------------

struct expected_pmc_event
{
    expected_event             event;
    double                     value = 0.0;
    std::size_t                timestamp{};
    std::string_view           track_name;
    std::optional<std::size_t> track_thread_id;
    std::string_view           pmc_extdata = "{}";
};

[[nodiscard]] inline bool
pmc_event_equals(const writer_types::pmc_event_data_t& data,
                 const expected_pmc_event&             expected)
{
    return event_equals(data.event, expected.event) && data.value == expected.value &&
           data.extdata == expected.pmc_extdata &&
           data.sample.timestamp == expected.timestamp &&
           data.sample.track.name == expected.track_name &&
           data.sample.track.node_id == node_id() &&
           data.sample.track.process_id == static_cast<std::size_t>(k_pid) &&
           data.sample.track.thread_id == expected.track_thread_id;
}

MATCHER_P(IsPmcEvent, expected, "pmc event") { return pmc_event_equals(arg, expected); }

MATCHER_P2(IsPmcId, name, agent_id, "pmc unique id")
{
    return arg.name == name && arg.agent_id == agent_id;
}

/// Fixture owning a `StrictMock` writer behind `mock_profiler_hub_writer`.
class rocpd_processor_mock_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_mock_profiler_hub_writer =
            std::make_unique<::testing::StrictMock<gmock_profiler_hub_writer>>();

        m_previous_output_path = tim::settings::output_path();
        m_output_dir = std::filesystem::temp_directory_path() / "rocpd_processor_mock";
        std::filesystem::create_directories(m_output_dir);
        tim::settings::output_path() = m_output_dir.string();
        rocprofsys::reset_database_path_memo();

        m_metadata = std::make_shared<metadata_registry>();
        info::process process{};
        process.pid     = k_pid;
        process.ppid    = k_ppid;
        process.command = "test_binary";
        process.start   = k_process_start;
        process.end     = k_process_end;
        m_metadata->set_process(process);
    }

    void TearDown() override
    {
        g_mock_profiler_hub_writer.reset();
        tim::settings::output_path() = m_previous_output_path;
        std::filesystem::remove_all(m_output_dir);
    }

    [[nodiscard]] std::unique_ptr<rocpd_processor_t<mock_profiler_hub_writer>>
    make_processor()
    {
        return std::make_unique<rocpd_processor_t<mock_profiler_hub_writer>>(
            m_metadata, m_agents, k_pid, k_ppid, m_registry);
    }

    void add_agent(const agent& entry)
    {
        auto copy = entry;
        m_agents->insert_agent(copy);
    }

    void add_agent(agent_type type, std::size_t type_index, std::uint64_t handle)
    {
        agent entry{};
        entry.type              = type;
        entry.device_type_index = type_index;
        entry.handle            = handle;
        entry.name              = "test_agent";
        add_agent(entry);
    }

    void add_pmc(agent_type type, std::string_view name, std::string_view target_arch,
                 std::string_view description = {}, std::string_view units = {},
                 std::size_t agent_type_index = 0)
    {
        info::pmc row{};
        row.type             = type;
        row.agent_type_index = agent_type_index;
        row.target_arch      = target_arch;
        row.name             = name;
        row.symbol           = name;
        row.description      = description.empty() ? name : description;
        row.units            = units;
        row.value_type       = "ABS";
        row.extdata          = "{}";
        m_metadata->add_pmc_info(row);
    }

    void seed_kernel_symbol(std::uint64_t kernel_id, const char* name)
    {
        m_code_object_uri = "file:///test_code_object.co";
        rocprofiler_callback_tracing_code_object_load_data_t code_object{};
        code_object.code_object_id = 1;
        code_object.uri            = m_code_object_uri.c_str();
#if(ROCPROFILER_VERSION >= 600)
        code_object.agent_id.handle = k_gpu_handle;
#else
        code_object.rocp_agent.handle = k_gpu_handle;
#endif
        code_object.storage_type = ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY;
        m_metadata->add_code_object(code_object);

        m_kernel_name = name;
        rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t symbol{};
        symbol.kernel_id      = kernel_id;
        symbol.code_object_id = 1;
        symbol.kernel_name    = m_kernel_name.c_str();
        m_metadata->add_kernel_symbol(symbol);
    }

    /// SDK buffer and callback names the registry pre-registers in every metadata.
    [[nodiscard]] int count_sdk_name_strings() const
    {
        std::size_t count = 0;
        for(const auto& info : m_metadata->get_buffer_name_info())
        {
            count += std::ranges::distance(info.items());
        }
        for(const auto& info : m_metadata->get_callback_tracing_info())
        {
            count += std::ranges::distance(info.items());
        }
        return static_cast<int>(count);
    }

    std::shared_ptr<metadata_registry> m_metadata;
    std::shared_ptr<agent_manager>     m_agents = std::make_shared<agent_manager>();
    output_file_registry               m_registry;

private:
    std::string           m_previous_output_path;
    std::string           m_kernel_name;
    std::string           m_code_object_uri;
    std::filesystem::path m_output_dir;
};

}  // namespace rocprofsys::trace_cache::test
