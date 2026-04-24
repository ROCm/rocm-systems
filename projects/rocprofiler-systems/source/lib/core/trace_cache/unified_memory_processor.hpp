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
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_processor.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace rocprofsys
{
namespace trace_cache
{

struct migration_stats
{
    uint64_t count            = 0;
    uint64_t total_size_bytes = 0;
    uint64_t min_size_bytes   = UINT64_MAX;  // Sentinel for min tracking
    uint64_t max_size_bytes   = 0;
    uint64_t total_time_ns    = 0;

    void add_migration(uint64_t size_bytes, uint64_t duration_ns)
    {
        count++;
        total_size_bytes += size_bytes;
        total_time_ns += duration_ns;
        if(size_bytes < min_size_bytes) min_size_bytes = size_bytes;
        if(size_bytes > max_size_bytes) max_size_bytes = size_bytes;
    }

    double avg_size_bytes() const noexcept
    {
        return count > 0 ? static_cast<double>(total_size_bytes) / count : 0.0;
    }
    double bandwidth_gbps() const noexcept
    {
        if(total_time_ns == 0) return 0.0;
        // bytes/ns = GB/s
        return (static_cast<double>(total_size_bytes) / total_time_ns);
    }
};

struct device_migration_summary
{
    std::string device_name;
    uint32_t    device_id = 0;

    migration_stats host_to_device;
    migration_stats device_to_host;
    migration_stats device_to_device;
};

struct page_fault_stats
{
    uint64_t total_faults = 0;
    uint64_t read_faults  = 0;
    uint64_t write_faults = 0;

    void add_fault(bool is_read)
    {
        total_faults++;
        if(is_read)
            read_faults++;
        else
            write_faults++;
    }
};

struct migration_trigger_stats
{
    uint64_t gpu_page_fault = 0;
    uint64_t cpu_page_fault = 0;
    uint64_t prefetch       = 0;
    uint64_t ttm_eviction   = 0;
    uint64_t unknown        = 0;

    uint64_t total() const noexcept
    {
        return gpu_page_fault + cpu_page_fault + prefetch + ttm_eviction + unknown;
    }
};

struct unified_memory_data
{
    std::map<uint32_t, device_migration_summary> devices;
    std::map<uint32_t, page_fault_stats>         faults_by_agent;

    uint64_t                total_page_faults = 0;
    migration_trigger_stats triggers;
    bool                    xnack_enabled = false;
};

class unified_memory_processor_t : public processor_t<unified_memory_processor_t>
{
public:
    unified_memory_processor_t(const std::shared_ptr<metadata_registry>& metadata,
                               const std::shared_ptr<agent_manager>& agent_mgr, int pid,
                               const std::string&    output_dir,
                               output_file_registry& output_registry);

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
    void handle(const cpu_freq_sample&) {}
    void handle(const backtrace_region_sample&) {}

private:
    enum class migration_direction
    {
        HOST_TO_DEVICE,
        DEVICE_TO_HOST,
        DEVICE_TO_DEVICE,
        UNKNOWN
    };

    enum class migration_trigger
    {
        GPU_PAGE_FAULT,
        CPU_PAGE_FAULT,
        PREFETCH,
        TTM_EVICTION,
        UNKNOWN
    };

    migration_direction classify_direction(const std::string& src_label,
                                           const std::string& dst_label) const;
    migration_trigger   classify_trigger(const std::string& name) const;
    bool                is_read_fault(const std::string& name) const;
    [[nodiscard]] std::optional<std::pair<std::string, std::string>>
    parse_agent_ids_from_args(const std::string& args_str) const;

    /**
     * Extracts GPU name from migration event labels.
     * @param src_label Source agent numeric node ID from KFD event (e.g., "0", "2")
     * @param dst_label Destination agent numeric node ID from KFD event
     * @return GPU agent name if found (e.g., "gfx950"), fallback to "GPU" or "GPU {id}"
     */
    [[nodiscard]] std::string extract_gpu_name(const std::string& src_label,
                                               const std::string& dst_label) const;

    void write_text_output(std::ostream& out);
    void write_json_output(std::ostream& out);

    unified_memory_data                m_data;
    std::shared_ptr<metadata_registry> m_metadata;
    std::shared_ptr<agent_manager>     m_agent_manager;
    int                                m_pid;
    std::string                        m_output_dir;
    output_file_registry&              m_output_registry;

    // Performance optimization: cache node_id to agent_type mapping
    std::unordered_map<uint32_t, agent_type>  m_node_type_cache;
    std::unordered_map<uint32_t, std::string> m_gpu_name_cache;
};

}  // namespace trace_cache
}  // namespace rocprofsys
