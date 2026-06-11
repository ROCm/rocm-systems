// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <cstdint>
#include <string>

namespace rocprofsys
{
namespace trace_cache
{
namespace test
{

[[nodiscard]] inline agent
make_cpu_agent(std::uint32_t node_id, std::string name = "AMD CPU")
{
    agent a{};
    a.type      = agent_type::CPU;
    a.node_id   = node_id;
    a.name      = std::move(name);
    a.device_id = node_id;
    return a;
}

[[nodiscard]] inline agent
make_gpu_agent(std::uint32_t node_id, std::string name = "gfx950")
{
    agent a{};
    a.type      = agent_type::GPU;
    a.node_id   = node_id;
    a.name      = std::move(name);
    a.device_id = node_id;
    return a;
}

inline constexpr double kDefaultMigrateSizeBytes = 1024.0;

[[nodiscard]] inline kfd_sample
make_kfd_page_migrate_sample_raw_args(
    std::string args_str, std::string trigger_name = "PAGE_MIGRATE_PAGEFAULT_GPU")
{
    kfd_sample s;
    s.thread_id       = 1;
    s.name            = std::move(trigger_name);
    s.start_timestamp = 0;
    s.end_timestamp   = 100;
    s.args_str        = std::move(args_str);
    s.category        = "rocm_kfd_page_migrate";
    s.device_id       = 0;
    s.device_type     = static_cast<std::uint8_t>(agent_type::CPU);
    s.value           = kDefaultMigrateSizeBytes;
    return s;
}

[[nodiscard]] inline kfd_sample
make_kfd_page_migrate_sample(std::uint32_t src_node, std::uint32_t dst_node,
                             std::uint64_t size_bytes, std::uint64_t duration_ns,
                             std::uint32_t device_id,
                             agent_type    device_type  = agent_type::CPU,
                             std::string   trigger_name = "PAGE_MIGRATE_PAGEFAULT_GPU")
{
    auto args = "0;;std::uint64_t;;start_address;;0x0;;"
                "1;;std::uint64_t;;end_address;;0x1000;;"
                "2;;string;;src_agent;;" +
                std::to_string(src_node) +
                ";;"
                "3;;string;;dst_agent;;" +
                std::to_string(dst_node) + ";;";
    auto s =
        make_kfd_page_migrate_sample_raw_args(std::move(args), std::move(trigger_name));
    s.end_timestamp = duration_ns;
    s.device_id     = device_id;
    s.device_type   = static_cast<std::uint8_t>(device_type);
    s.value         = static_cast<double>(size_bytes);
    return s;
}

[[nodiscard]] inline kfd_sample
make_kfd_page_fault_sample(std::uint32_t agent_id, bool is_read,
                           agent_type device_type = agent_type::GPU)
{
    kfd_sample s;
    s.thread_id       = 1;
    s.name            = is_read ? "PAGE_FAULT_Read" : "PAGE_FAULT_Write";
    s.start_timestamp = 0;
    s.end_timestamp   = 0;
    s.args_str        = "";
    s.category        = "rocm_kfd_page_fault";
    s.device_id       = agent_id;
    s.device_type     = static_cast<std::uint8_t>(device_type);
    s.value           = 0.0;
    return s;
}

}  // namespace test
}  // namespace trace_cache
}  // namespace rocprofsys
