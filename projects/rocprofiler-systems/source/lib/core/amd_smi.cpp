// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/amd_smi.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "timemory.hpp"

#include "logger/debug.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <string_view>
#include <vector>

namespace rocprofsys
{
namespace amd_smi
{
namespace
{
std::string
get_setting_name(std::string_view input)
{
    constexpr auto prefix = std::string_view{ "rocprofsys_" };

    std::string result;
    result.reserve(input.size());
    for(auto c : input)
        result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

    if(result.compare(0, prefix.size(), prefix) == 0) return result.substr(prefix.size());

    return result;
}

#define ROCPROFSYS_CONFIG_SETTING(TYPE, ENV_NAME, DESCRIPTION, INITIAL_VALUE, ...)       \
    [&]() {                                                                              \
        auto _ret = _config->insert<TYPE, TYPE>(                                         \
            ENV_NAME, get_setting_name(ENV_NAME), DESCRIPTION, TYPE{ INITIAL_VALUE },    \
            std::set<std::string>{ "custom", "rocprofsys", "librocprof-sys",             \
                                   __VA_ARGS__ });                                       \
        if(!_ret.second)                                                                 \
        {                                                                                \
            LOG_WARNING("Duplicate setting: {} / {}", get_setting_name(ENV_NAME),        \
                        ENV_NAME);                                                       \
        }                                                                                \
        return _config->find(ENV_NAME)->second;                                          \
    }()

void
check_amdsmi_error(amdsmi_status_t code, const char* file, int line)
{
    if(code == AMDSMI_STATUS_SUCCESS) return;
    const char* msg = nullptr;
    auto        err = amdsmi_status_code_to_string(code, &msg);
    if(err != AMDSMI_STATUS_SUCCESS)
    {
        throw std::runtime_error(fmt::format(
            "amdsmi_status_code_to_string failed. No error message available. "
            "Error code {} originated at {}:{}",
            static_cast<int>(code), file, line));
    }
    throw std::runtime_error(fmt::format("[{}:{}] Error code {} :: {}", file, line,
                                         static_cast<int>(code), msg));
}

#define ROCPROFSYS_AMD_SMI_CALL(ERROR_CODE)                                              \
    check_amdsmi_error(ERROR_CODE, __FILE__, __LINE__)

std::atomic<bool> amdsmi_initialized{ false };
std::mutex        processor_mutex{};

std::vector<amdsmi_processor_handle> processor_handles{};

template <typename T>
constexpr bool
is_metric_supported(T value, T invalid_sentinel = std::numeric_limits<T>::max())
{
    return value != invalid_sentinel;
}

struct converted_metrics : probe_input
{};

template <typename T>
constexpr bool
populate_if_supported(T& dest, T src, T invalid_sentinel = std::numeric_limits<T>::max())
{
    const bool valid = is_metric_supported(src, invalid_sentinel);
    dest             = valid ? src : T{ 0 };
    return valid;
}

probe_input
convert_gpu_metrics(const amdsmi_gpu_metrics_t& raw)
{
    converted_metrics out{};
    out.current_socket_power = raw.current_socket_power;
    out.average_socket_power = raw.average_socket_power;
    out.hotspot_temperature  = raw.temperature_hotspot;
    out.edge_temperature     = raw.temperature_edge;
    out.gfx_activity         = raw.average_gfx_activity;
    out.umc_activity         = raw.average_umc_activity;
    out.mm_activity          = raw.average_mm_activity;

    for(std::size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
    {
        std::copy(std::begin(raw.xcp_stats[xcp].vcn_busy),
                  std::end(raw.xcp_stats[xcp].vcn_busy),
                  out.xcp_stats[xcp].vcn_busy.begin());

        constexpr std::size_t copy_count =
            std::min(static_cast<std::size_t>(sizeof(raw.xcp_stats[0].jpeg_busy) /
                                              sizeof(std::uint16_t)),
                     MAX_NUM_JPEG_V1);
        std::copy_n(std::begin(raw.xcp_stats[xcp].jpeg_busy), copy_count,
                    out.xcp_stats[xcp].jpeg_busy.begin());
    }

    std::copy(std::begin(raw.vcn_activity), std::end(raw.vcn_activity),
              out.vcn_activity.begin());
    std::copy(std::begin(raw.jpeg_activity), std::end(raw.jpeg_activity),
              out.jpeg_activity.begin());

    populate_if_supported(out.xgmi.width, raw.xgmi_link_width);
    populate_if_supported(out.xgmi.speed, raw.xgmi_link_speed);
    for(std::size_t idx = 0; idx < MAX_NUM_XGMI_LINKS; ++idx)
    {
        populate_if_supported(out.xgmi.data_acc.read[idx], raw.xgmi_read_data_acc[idx]);
        populate_if_supported(out.xgmi.data_acc.write[idx], raw.xgmi_write_data_acc[idx]);
    }

    populate_if_supported(out.pcie.width, raw.pcie_link_width);
    populate_if_supported(out.pcie.speed, raw.pcie_link_speed);
    populate_if_supported(out.pcie.bandwidth.acc, raw.pcie_bandwidth_acc);
    populate_if_supported(out.pcie.bandwidth.inst, raw.pcie_bandwidth_inst);

    populate_if_supported(out.gfx_clock_mhz,
                          static_cast<std::uint32_t>(raw.current_gfxclk),
                          static_cast<std::uint32_t>(0xFFFFU));
    populate_if_supported(out.mem_clock_mhz, static_cast<std::uint32_t>(raw.current_uclk),
                          static_cast<std::uint32_t>(0xFFFFU));

    return out;
}

metric_availability
compute_availability_impl(const probe_input& raw, bool memory_usage, bool sdma_supported)
{
    metric_availability result{};

    result.bits.memory_usage = memory_usage ? 1 : 0;
    result.bits.current_socket_power =
        is_metric_supported(raw.current_socket_power, METRIC_VALUE_NOT_SUPPORTED_16) ? 1
                                                                                     : 0;
    result.bits.average_socket_power =
        is_metric_supported(raw.average_socket_power, METRIC_VALUE_NOT_SUPPORTED_16) ? 1
                                                                                     : 0;
    result.bits.hotspot_temperature =
        is_metric_supported(raw.hotspot_temperature, METRIC_VALUE_NOT_SUPPORTED_16) ? 1
                                                                                    : 0;
    result.bits.edge_temperature =
        is_metric_supported(raw.edge_temperature, METRIC_VALUE_NOT_SUPPORTED_16) ? 1 : 0;
    result.bits.gfx_activity =
        is_metric_supported(raw.gfx_activity, METRIC_VALUE_NOT_SUPPORTED_16) ? 1 : 0;
    result.bits.umc_activity =
        is_metric_supported(raw.umc_activity, METRIC_VALUE_NOT_SUPPORTED_16) ? 1 : 0;
    result.bits.mm_activity =
        is_metric_supported(raw.mm_activity, METRIC_VALUE_NOT_SUPPORTED_16) ? 1 : 0;

    result.bits.vcn_busy =
        std::any_of(raw.xcp_stats.begin(), raw.xcp_stats.end(),
                    [](const probe_input::xcp_metrics& xcp) {
                        return std::any_of(
                            xcp.vcn_busy.begin(), xcp.vcn_busy.end(),
                            [](std::uint16_t val) { return is_metric_supported(val); });
                    })
            ? 1
            : 0;

    result.bits.jpeg_busy =
        std::any_of(raw.xcp_stats.begin(), raw.xcp_stats.end(),
                    [](const probe_input::xcp_metrics& xcp) {
                        return std::any_of(
                            xcp.jpeg_busy.begin(), xcp.jpeg_busy.end(),
                            [](std::uint16_t val) { return is_metric_supported(val); });
                    })
            ? 1
            : 0;

    result.bits.vcn_activity =
        (!result.bits.vcn_busy &&
         std::any_of(raw.vcn_activity.begin(), raw.vcn_activity.end(),
                     [](std::uint16_t val) { return is_metric_supported(val); }))
            ? 1
            : 0;

    result.bits.jpeg_activity =
        (!result.bits.jpeg_busy &&
         std::any_of(raw.jpeg_activity.begin(), raw.jpeg_activity.end(),
                     [](std::uint16_t val) { return is_metric_supported(val); }))
            ? 1
            : 0;

    result.bits.xgmi =
        (is_metric_supported(raw.xgmi.width) || is_metric_supported(raw.xgmi.speed) ||
         std::any_of(raw.xgmi.data_acc.read.begin(), raw.xgmi.data_acc.read.end(),
                     [](std::uint64_t val) { return is_metric_supported(val); }))
            ? 1
            : 0;

    result.bits.pcie =
        (is_metric_supported(raw.pcie.width) || is_metric_supported(raw.pcie.speed) ||
         is_metric_supported(raw.pcie.bandwidth.acc) ||
         is_metric_supported(raw.pcie.bandwidth.inst))
            ? 1
            : 0;

    result.bits.gfx_clock =
        is_metric_supported(raw.gfx_clock_mhz, METRIC_VALUE_NOT_SUPPORTED_16) ? 1 : 0;
    result.bits.mem_clock =
        is_metric_supported(raw.mem_clock_mhz, METRIC_VALUE_NOT_SUPPORTED_16) ? 1 : 0;
    result.bits.sdma_usage = sdma_supported ? 1 : 0;

    return result;
}

void
append_scalar_entry(std::vector<metric_entry>& entries, std::size_t device_index,
                    const std::string& name, const std::string& category,
                    const std::string& summary, bool available)
{
    entries.push_back(metric_entry{ name + ":device=" + std::to_string(device_index),
                                    category, summary, available });
}

struct metric_symbol_parts
{
    std::string name;
    std::size_t device = 0;
};

metric_symbol_parts
parse_metric_symbol(const std::string& symbol)
{
    constexpr auto suffix = std::string_view{ ":device=" };

    metric_symbol_parts parts{};
    parts.name = symbol;
    if(const auto pos = symbol.rfind(suffix); pos != std::string::npos)
    {
        parts.name = symbol.substr(0, pos);
        parts.device =
            static_cast<std::size_t>(std::stoull(symbol.substr(pos + suffix.size())));
    }
    return parts;
}

bool
compare_metric_entries(const metric_entry& lhs, const metric_entry& rhs)
{
    const auto l = parse_metric_symbol(lhs.symbol);
    const auto r = parse_metric_symbol(rhs.symbol);
    if(l.name != r.name) return l.name < r.name;
    return l.device < r.device;
}

void
build_metric_entries_impl(std::size_t device_index, const probe_input& raw,
                          const metric_availability& metrics,
                          std::vector<metric_entry>& entries)
{
    append_scalar_entry(entries, device_index, "CURRENT_SOCKET_POWER", "power",
                        "current_socket_power", metrics.bits.current_socket_power != 0);
    append_scalar_entry(entries, device_index, "AVERAGE_SOCKET_POWER", "power",
                        "average_socket_power", metrics.bits.average_socket_power != 0);
    append_scalar_entry(entries, device_index, "MEMORY_USAGE", "mem_usage",
                        "amdsmi_get_gpu_memory_usage (VRAM)",
                        metrics.bits.memory_usage != 0);
    append_scalar_entry(entries, device_index, "HOTSPOT_TEMPERATURE", "temp",
                        "temperature_hotspot", metrics.bits.hotspot_temperature != 0);
    append_scalar_entry(entries, device_index, "EDGE_TEMPERATURE", "temp",
                        "temperature_edge", metrics.bits.edge_temperature != 0);
    append_scalar_entry(entries, device_index, "GFX_ACTIVITY", "busy",
                        "average_gfx_activity", metrics.bits.gfx_activity != 0);
    append_scalar_entry(entries, device_index, "UMC_ACTIVITY", "busy",
                        "average_umc_activity", metrics.bits.umc_activity != 0);
    append_scalar_entry(entries, device_index, "MM_ACTIVITY", "busy",
                        "average_mm_activity", metrics.bits.mm_activity != 0);
    append_scalar_entry(entries, device_index, "VCN_BUSY", "vcn_activity", "vcn_busy",
                        metrics.bits.vcn_busy != 0);
    append_scalar_entry(entries, device_index, "VCN_ACTIVITY", "vcn_activity",
                        "vcn_activity", metrics.bits.vcn_activity != 0);
    append_scalar_entry(entries, device_index, "JPEG_BUSY", "jpeg_activity", "jpeg_busy",
                        metrics.bits.jpeg_busy != 0);
    append_scalar_entry(entries, device_index, "JPEG_ACTIVITY", "jpeg_activity",
                        "jpeg_activity", metrics.bits.jpeg_activity != 0);

    append_scalar_entry(entries, device_index, "XGMI_LINK_WIDTH", "xgmi",
                        "xgmi_link_width", is_metric_supported(raw.xgmi.width));
    append_scalar_entry(entries, device_index, "XGMI_LINK_SPEED", "xgmi",
                        "xgmi_link_speed", is_metric_supported(raw.xgmi.speed));

    const bool xgmi_read_supported =
        std::any_of(raw.xgmi.data_acc.read.begin(), raw.xgmi.data_acc.read.end(),
                    [](std::uint64_t val) { return is_metric_supported(val); });
    const bool xgmi_write_supported =
        std::any_of(raw.xgmi.data_acc.write.begin(), raw.xgmi.data_acc.write.end(),
                    [](std::uint64_t val) { return is_metric_supported(val); });
    append_scalar_entry(entries, device_index, "XGMI_READ", "xgmi", "xgmi_read_data",
                        xgmi_read_supported);
    append_scalar_entry(entries, device_index, "XGMI_WRITE", "xgmi", "xgmi_write_data",
                        xgmi_write_supported);

    append_scalar_entry(entries, device_index, "PCIE_LINK_WIDTH", "pcie",
                        "pcie_link_width", is_metric_supported(raw.pcie.width));
    append_scalar_entry(entries, device_index, "PCIE_LINK_SPEED", "pcie",
                        "pcie_link_speed", is_metric_supported(raw.pcie.speed));
    append_scalar_entry(entries, device_index, "PCIE_BANDWIDTH_ACC", "pcie",
                        "pcie_bandwidth_acc",
                        is_metric_supported(raw.pcie.bandwidth.acc));
    append_scalar_entry(entries, device_index, "PCIE_BANDWIDTH_INST", "pcie",
                        "pcie_bandwidth_inst",
                        is_metric_supported(raw.pcie.bandwidth.inst));
    append_scalar_entry(entries, device_index, "SDMA_USAGE", "sdma_usage",
                        "amdsmi_get_gpu_process_list", metrics.bits.sdma_usage != 0);
    append_scalar_entry(entries, device_index, "GFX_CLOCK", "gfx_clock",
                        "current_gfxclk (MHz)", metrics.bits.gfx_clock != 0);
    append_scalar_entry(entries, device_index, "MEM_CLOCK", "mem_clock",
                        "current_uclk (MHz)", metrics.bits.mem_clock != 0);
}

bool
query_memory_usage(amdsmi_processor_handle handle, std::uint64_t& usage)
{
    usage = 0;
    return amdsmi_get_gpu_memory_usage(handle, AMDSMI_MEM_TYPE_VRAM, &usage) ==
           AMDSMI_STATUS_SUCCESS;
}

bool
query_sdma_supported(amdsmi_processor_handle handle)
{
#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
    std::uint32_t num_processes = 0;
    return amdsmi_get_gpu_process_list(handle, &num_processes, nullptr) ==
           AMDSMI_STATUS_SUCCESS;
#else
    (void) handle;
    return false;
#endif
}

void
enumerate_processors()
{
    std::lock_guard<std::mutex> lock(processor_mutex);
    processor_handles.clear();

    std::uint32_t socket_count = 0;
    if(amdsmi_get_socket_handles(&socket_count, nullptr) != AMDSMI_STATUS_SUCCESS)
    {
        return;
    }

    std::vector<amdsmi_socket_handle> sockets(socket_count);
    if(amdsmi_get_socket_handles(&socket_count, sockets.data()) != AMDSMI_STATUS_SUCCESS)
    {
        return;
    }

    for(auto& socket : sockets)
    {
        std::uint32_t processor_count = 0;
        if(amdsmi_get_processor_handles(socket, &processor_count, nullptr) !=
           AMDSMI_STATUS_SUCCESS)
        {
            continue;
        }

        std::vector<amdsmi_processor_handle> processors(processor_count);
        if(amdsmi_get_processor_handles(socket, &processor_count, processors.data()) !=
           AMDSMI_STATUS_SUCCESS)
        {
            continue;
        }

        for(auto& processor : processors)
        {
            processor_type_t processor_type = {};
            if(amdsmi_get_processor_type(processor, &processor_type) !=
               AMDSMI_STATUS_SUCCESS)
            {
                continue;
            }
#ifdef AINIC_SUPPORTED
            if(processor_type == AMDSMI_PROCESSOR_TYPE_AMD_NIC)
            {
                continue;
            }
#endif
            if(processor_type != AMDSMI_PROCESSOR_TYPE_AMD_GPU)
            {
                continue;
            }
            processor_handles.push_back(processor);
        }
    }
}

device_probe_result
probe_processor_internal(amdsmi_processor_handle handle, std::size_t device_index)
{
    device_probe_result result{};
    result.device_index = device_index;

    amdsmi_asic_info_t asic_info{};
    if(amdsmi_get_gpu_asic_info(handle, &asic_info) == AMDSMI_STATUS_SUCCESS)
    {
        result.product_name = asic_info.market_name;
        result.vendor_name  = asic_info.vendor_name;
    }
    else
    {
        result.product_name = "Unknown GPU";
        result.vendor_name  = "AMD";
    }

    std::uint64_t memory_usage = 0;
    const bool    memory_ok    = query_memory_usage(handle, memory_usage);
    const bool    memory_supported =
        memory_ok && is_metric_supported(memory_usage, METRIC_VALUE_NOT_SUPPORTED_64);

    probe_input          raw{};
    amdsmi_gpu_metrics_t gpu_metrics{};
    if(amdsmi_get_gpu_metrics_info(handle, &gpu_metrics) == AMDSMI_STATUS_SUCCESS)
    {
        raw = convert_gpu_metrics(gpu_metrics);
    }

    const bool sdma_supported = query_sdma_supported(handle);
    result.metrics = compute_availability_impl(raw, memory_supported, sdma_supported);
    build_metric_entries_impl(device_index, raw, result.metrics, result.entries);
    return result;
}
}  // namespace

metric_availability
compute_availability(const probe_input& raw, bool memory_usage, bool sdma_supported)
{
    return compute_availability_impl(raw, memory_usage, sdma_supported);
}

void
build_metric_entries(std::size_t device_index, const probe_input& raw,
                     const metric_availability& metrics,
                     std::vector<metric_entry>& entries)
{
    build_metric_entries_impl(device_index, raw, metrics, entries);
}

bool
ensure_initialized()
{
    if(amdsmi_initialized.load()) return !processor_handles.empty();

    try
    {
        std::uint64_t init_flags = AMDSMI_INIT_AMD_GPUS;
#ifdef AINIC_SUPPORTED
        init_flags |= AMDSMI_INIT_AMD_NICS;
#endif
        ROCPROFSYS_AMD_SMI_CALL(::amdsmi_init(init_flags));
        enumerate_processors();
        amdsmi_initialized.store(true);
    } catch(const std::exception& e)
    {
        LOG_ERROR("Exception thrown initializing amd-smi: {}", e.what());
        amdsmi_initialized.store(false);
        processor_handles.clear();
        return false;
    }

    return !processor_handles.empty();
}

metric_availability
probe_processor(amdsmi_processor_handle handle)
{
    return probe_processor_internal(handle, 0).metrics;
}

std::vector<device_probe_result>
probe_devices()
{
    std::vector<device_probe_result> results;
    if(!ensure_initialized()) return results;

    results.reserve(processor_handles.size());
    for(std::size_t i = 0; i < processor_handles.size(); ++i)
    {
        results.emplace_back(probe_processor_internal(processor_handles[i], i));
    }
    return results;
}

std::string
format_settings_description(const std::vector<device_probe_result>& devices)
{
    metric_availability union_metrics{};
    for(const auto& device : devices)
    {
        union_metrics.value |= device.metrics.value;
    }

    std::vector<std::string> categories;
    auto                     add = [&](bool cond, const char* name) {
        if(cond) categories.emplace_back(name);
    };

    add(union_metrics.bits.gfx_activity || union_metrics.bits.umc_activity ||
            union_metrics.bits.mm_activity,
        "busy");
    add(union_metrics.bits.hotspot_temperature || union_metrics.bits.edge_temperature,
        "temp");
    add(union_metrics.bits.current_socket_power ||
            union_metrics.bits.average_socket_power,
        "power");
    add(union_metrics.bits.memory_usage, "mem_usage");
    add(union_metrics.bits.vcn_activity || union_metrics.bits.vcn_busy, "vcn_activity");
    add(union_metrics.bits.jpeg_activity || union_metrics.bits.jpeg_busy,
        "jpeg_activity");
    add(union_metrics.bits.xgmi, "xgmi");
    add(union_metrics.bits.pcie, "pcie");
    add(union_metrics.bits.sdma_usage, "sdma_usage");
    add(union_metrics.bits.gfx_clock, "gfx_clock");
    add(union_metrics.bits.mem_clock, "mem_clock");

    std::ostringstream desc;
    desc << "amd-smi metrics to collect: ";
    if(categories.empty())
    {
        desc << "none detected on this system";
    }
    else
    {
        for(std::size_t i = 0; i < categories.size(); ++i)
        {
            if(i > 0) desc << ", ";
            desc << categories[i];
        }
    }
    desc << ". An empty value implies 'all' and 'none' suppresses all.";
    return desc.str();
}

std::vector<metric_entry>
format_avail_entries(const std::vector<device_probe_result>& devices)
{
    std::vector<metric_entry> entries;
    for(const auto& device : devices)
    {
        entries.insert(entries.end(), device.entries.begin(), device.entries.end());
    }

    std::sort(entries.begin(), entries.end(), compare_metric_entries);
    return entries;
}

void
config_settings(const std::shared_ptr<settings>& _config)
{
    if(!get_use_amd_smi()) return;

    const auto devices = probe_devices();
    if(devices.empty()) return;

    const auto description = format_settings_description(devices);

    ROCPROFSYS_CONFIG_SETTING(std::string, "ROCPROFSYS_AMD_SMI_METRICS", description,
                              "busy, temp, power, mem_usage", "backend", "amd_smi",
                              "rocm", "process_sampling");
}
}  // namespace amd_smi
}  // namespace rocprofsys
