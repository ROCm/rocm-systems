// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "core/agent_manager.hpp"
#include "core/output_file_registry.hpp"
#include "core/trace_cache/sample_processor.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace rocprofsys
{
namespace trace_cache
{

struct migration_stats
{
    uint64_t count            = 0;
    uint64_t total_size_bytes = 0;
    uint64_t min_size_bytes   = std::numeric_limits<uint64_t>::max();
    uint64_t max_size_bytes   = 0;
    uint64_t total_time_ns    = 0;

    void add_migration(uint64_t size_bytes, uint64_t duration_ns) noexcept
    {
        count++;
        total_size_bytes += size_bytes;
        total_time_ns += duration_ns;
        if(size_bytes < min_size_bytes) min_size_bytes = size_bytes;
        if(size_bytes > max_size_bytes) max_size_bytes = size_bytes;
    }

    [[nodiscard]] double avg_size_bytes() const noexcept
    {
        return count > 0 ? static_cast<double>(total_size_bytes) / count : 0.0;
    }
    [[nodiscard]] double bandwidth_gbps() const noexcept
    {
        if(total_time_ns == 0) return 0.0;
        return static_cast<double>(total_size_bytes) / total_time_ns;
    }
};

struct device_migration_summary
{
    std::string device_name;

    migration_stats host_to_device;
    migration_stats device_to_host;
    migration_stats device_to_device;
};

struct migration_trigger_stats
{
    uint64_t gpu_page_fault = 0;
    uint64_t cpu_page_fault = 0;
    uint64_t prefetch       = 0;
    uint64_t ttm_eviction   = 0;
    uint64_t unknown        = 0;

    [[nodiscard]] uint64_t total() const noexcept
    {
        return gpu_page_fault + cpu_page_fault + prefetch + ttm_eviction + unknown;
    }
};

struct unified_memory_data
{
    std::map<uint32_t, device_migration_summary> devices;

    uint64_t                total_page_faults = 0;
    migration_trigger_stats triggers;
    bool                    xnack_enabled = false;
};

namespace detail
{
struct trigger_entry
{
    const char* kfd_name;  // nullptr marks the sentinel "unknown" row
    const char* json_key;
    const char* text_label;
    uint64_t migration_trigger_stats::*member;
};

inline constexpr std::array<trigger_entry, 5> kTriggerTable = { {
    { "PAGE_MIGRATE_PAGEFAULT_GPU", "gpu_page_fault", "GPU page fault",
      &migration_trigger_stats::gpu_page_fault },
    { "PAGE_MIGRATE_PAGEFAULT_CPU", "cpu_page_fault", "CPU page fault",
      &migration_trigger_stats::cpu_page_fault },
    { "PAGE_MIGRATE_PREFETCH", "prefetch", "Prefetch",
      &migration_trigger_stats::prefetch },
    { "PAGE_MIGRATE_TTM_EVICTION", "ttm_eviction", "TTM eviction",
      &migration_trigger_stats::ttm_eviction },
    { nullptr, "unknown", "Unknown", &migration_trigger_stats::unknown },
} };

static_assert(kTriggerTable.back().kfd_name == nullptr,
              "sentinel row must be last: handle_page_migrate falls through "
              "to it on no match");
}  // namespace detail

// NOT thread-safe. handle() and finalize_processing() must be called from a
// single thread; finalize_processing() is not idempotent.
template <typename AgentManagerT   = agent_manager,
          typename OutputRegistryT = output_file_registry>
class unified_memory_processor_t
: public processor_t<unified_memory_processor_t<AgentManagerT, OutputRegistryT>>
{
public:
    unified_memory_processor_t(std::shared_ptr<AgentManagerT> agent_mgr, int pid,
                               std::string output_dir, OutputRegistryT& output_registry);

    unified_memory_processor_t(const unified_memory_processor_t&)            = delete;
    unified_memory_processor_t(unified_memory_processor_t&&)                 = delete;
    unified_memory_processor_t& operator=(const unified_memory_processor_t&) = delete;
    unified_memory_processor_t& operator=(unified_memory_processor_t&&)      = delete;
    ~unified_memory_processor_t()                                            = default;

    void prepare_for_processing();
    void finalize_processing();

    void handle(const kfd_sample& sample);

    void handle(const in_time_sample&) {}
    void handle(const pmc_event_with_sample&) {}
    void handle(const region_sample&) {}
    void handle(const kernel_dispatch_sample&) {}
    void handle(const memory_copy_sample&) {}
    void handle(const memory_allocate_sample&) {}
    void handle(const scratch_memory_sample&) {}
    void handle(const gpu_pmc_sample&) {}
    void handle(const ainic_pmc_sample&) {}
    void handle(const cpu_pmc_sample&) {}
    void handle(const backtrace_region_sample&) {}

private:
    void handle_page_migrate(const kfd_sample& sample);

    enum class migration_direction
    {
        HOST_TO_DEVICE,
        DEVICE_TO_HOST,
        DEVICE_TO_DEVICE,
        UNKNOWN
    };

    [[nodiscard]] migration_direction classify_direction(
        const std::string& src_label, const std::string& dst_label) const;
    [[nodiscard]] std::optional<std::pair<std::string, std::string>>
    parse_agent_ids_from_args(const std::string& args_str) const;

    [[nodiscard]] std::string resolve_device_label(const kfd_sample&  sample,
                                                   const std::string& src_label,
                                                   const std::string& dst_label) const;

    [[nodiscard]] std::optional<std::pair<uint32_t, uint32_t>> parse_node_id_pair(
        const std::string& src_label, const std::string& dst_label) const;

    [[nodiscard]] std::string extract_gpu_name(const std::string& src_label,
                                               const std::string& dst_label) const;

    void write_text_output(std::ostream& out) const;
    void write_json_output(std::ostream& out) const;

    unified_memory_data            m_data;
    std::shared_ptr<AgentManagerT> m_agent_manager;
    int                            m_pid;
    std::string                    m_output_dir;
    OutputRegistryT&               m_output_registry;

    std::unordered_map<uint32_t, agent_type>  m_node_type_cache;
    std::unordered_map<uint32_t, std::string> m_gpu_name_cache;
};

extern template class unified_memory_processor_t<>;

}  // namespace trace_cache
}  // namespace rocprofsys
