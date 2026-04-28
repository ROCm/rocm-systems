// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/output_file_registry.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <gmock/gmock.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rocprofsys
{
namespace trace_cache
{
namespace test
{

// Mock standing in for rocprofsys::agent_manager. Only the methods actually
// invoked by unified_memory_processor_t are mocked.
class MockAgentManager
{
public:
    MOCK_METHOD(std::vector<std::shared_ptr<agent>>, get_agents, (), (const));
    MOCK_METHOD(const agent&, get_agent_by_type_index, (size_t, agent_type), (const));
};

// Mock standing in for rocprofsys::output_file_registry. The processor only
// calls register_file(path, format) — the 3-arg overload is unused here.
class MockOutputFileRegistry
{
public:
    MOCK_METHOD(void, register_file, (std::string, output_format));
};

[[nodiscard]] inline std::shared_ptr<agent>
make_cpu_agent(uint32_t node_id, std::string name = "AMD CPU")
{
    auto a     = std::make_shared<agent>();
    a->type    = agent_type::CPU;
    a->node_id = node_id;
    a->name    = std::move(name);
    return a;
}

[[nodiscard]] inline std::shared_ptr<agent>
make_gpu_agent(uint32_t node_id, std::string name = "gfx950")
{
    auto a     = std::make_shared<agent>();
    a->type    = agent_type::GPU;
    a->node_id = node_id;
    a->name    = std::move(name);
    return a;
}

// Default migration size used by the raw-args helper when no override is
// supplied. Picked to match what the malformed-args / out-of-range-id tests
// want for the size field while keeping the value out of the function body.
inline constexpr double kDefaultMigrateSizeBytes = 1024.0;

// Build a kfd_sample for a page-migrate event with a *caller-supplied*
// args_str. Used directly by tests that need malformed / out-of-range
// argument strings, and as the construction backbone for
// make_kfd_page_migrate_sample (which builds a well-formed args_str and
// delegates here).
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
    s.device_type     = static_cast<uint8_t>(agent_type::CPU);
    s.value           = kDefaultMigrateSizeBytes;
    return s;
}

// Build a kfd_sample matching what the rocprofiler-sdk KFD callback emits
// for a page-migrate event. args_str follows the producer's standard
// "<idx>;;<type>;;<name>;;<value>;;" 4-tuple layout (see kfd_events.cpp:421
// in the rocprof-sys library) so the canonical process_arguments_string()
// parser understands it. Delegates to make_kfd_page_migrate_sample_raw_args
// for the field defaults, then overrides what this overload parameterizes.
[[nodiscard]] inline kfd_sample
make_kfd_page_migrate_sample(uint32_t src_node, uint32_t dst_node, uint64_t size_bytes,
                             uint64_t duration_ns, uint32_t device_id,
                             agent_type  device_type  = agent_type::CPU,
                             std::string trigger_name = "PAGE_MIGRATE_PAGEFAULT_GPU")
{
    auto args = "0;;uint64_t;;start_address;;0x0;;"
                "1;;uint64_t;;end_address;;0x1000;;"
                "2;;string;;src_agent;;" +
                std::to_string(src_node) +
                ";;"
                "3;;string;;dst_agent;;" +
                std::to_string(dst_node) + ";;";
    auto s =
        make_kfd_page_migrate_sample_raw_args(std::move(args), std::move(trigger_name));
    s.end_timestamp = duration_ns;
    s.device_id     = device_id;
    s.device_type   = static_cast<uint8_t>(device_type);
    s.value         = static_cast<double>(size_bytes);
    return s;
}

[[nodiscard]] inline kfd_sample
make_kfd_page_fault_sample(uint32_t agent_id, bool is_read,
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
    s.device_type     = static_cast<uint8_t>(device_type);
    s.value           = 0.0;
    return s;
}

}  // namespace test
}  // namespace trace_cache
}  // namespace rocprofsys
