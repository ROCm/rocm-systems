// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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

class output_file_sink_view
{
public:
    using register_file_fn_t = void (*)(void*, std::string, output_format);

    template <typename SinkT>
    // Non-owning sink view. The referenced sink object must outlive any
    // unified_memory_processor_t storing this view.
    explicit output_file_sink_view(SinkT& sink) noexcept
    : m_object{ std::addressof(sink) }
    , m_register_file_impl{ +[](void* obj, std::string path, output_format format) {
        static_cast<SinkT*>(obj)->register_file(std::move(path), format);
    } }
    {}

    output_file_sink_view(const output_file_sink_view&) noexcept            = default;
    output_file_sink_view(output_file_sink_view&&) noexcept                 = default;
    output_file_sink_view& operator=(const output_file_sink_view&) noexcept = default;
    output_file_sink_view& operator=(output_file_sink_view&&) noexcept      = default;

    void register_file(std::string path, output_format format) const
    {
        m_register_file_impl(m_object, std::move(path), format);
    }

private:
    void*              m_object;
    register_file_fn_t m_register_file_impl;
};

struct migration_stats
{
    std::uint64_t count            = 0;
    std::uint64_t total_size_bytes = 0;
    std::uint64_t min_size_bytes   = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_size_bytes   = 0;
    std::uint64_t total_time_ns    = 0;

    void add_migration(std::uint64_t size_bytes, std::uint64_t duration_ns) noexcept
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
        // bytes / ns == GB/s (decimal)
        return total_time_ns > 0 ? static_cast<double>(total_size_bytes) / total_time_ns
                                 : 0.0;
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
    std::uint64_t gpu_page_fault = 0;
    std::uint64_t cpu_page_fault = 0;
    std::uint64_t prefetch       = 0;
    std::uint64_t ttm_eviction   = 0;
    std::uint64_t unknown        = 0;

    [[nodiscard]] std::uint64_t total() const noexcept
    {
        return gpu_page_fault + cpu_page_fault + prefetch + ttm_eviction + unknown;
    }
};

struct unified_memory_data
{
    std::map<std::uint32_t, device_migration_summary> devices;

    std::uint64_t           total_page_faults = 0;
    migration_trigger_stats triggers;
    bool                    xnack_enabled = false;
};

namespace detail
{
struct trigger_entry
{
    const char*   kfd_name;  // nullptr marks the sentinel "unknown" row
    const char*   json_key;
    const char*   text_label;
    std::uint64_t migration_trigger_stats::*member;
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
class unified_memory_processor_t : public processor_t<unified_memory_processor_t>
{
public:
    unified_memory_processor_t(std::shared_ptr<agent_manager> agent_mgr, int pid,
                               output_file_sink_view output_sink);

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

    [[nodiscard]] std::optional<std::pair<std::uint32_t, std::uint32_t>>
    parse_node_id_pair(const std::string& src_label, const std::string& dst_label) const;

    [[nodiscard]] std::optional<std::uint32_t> resolve_gpu_bucket_id(
        const std::string& src_label, const std::string& dst_label,
        migration_direction direction) const;

    [[nodiscard]] std::string extract_gpu_name(const std::string& src_label,
                                               const std::string& dst_label) const;

    void write_text_output(std::ostream& out) const;
    void write_json_output(std::ostream& out) const;

    unified_memory_data            m_data;
    std::shared_ptr<agent_manager> m_agent_manager;
    int                            m_pid;
    std::string                    m_output_dir;
    output_file_sink_view          m_output_sink;

    std::unordered_map<std::uint32_t, agent_type>  m_node_type_cache;
    std::unordered_map<std::uint32_t, std::string> m_gpu_name_cache;
};

}  // namespace trace_cache
}  // namespace rocprofsys
