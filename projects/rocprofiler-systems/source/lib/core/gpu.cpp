// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "agent.hpp"
#include "agent_info.hpp"
#include <cstdint>
#define ROCPROFILER_SDK_CEREAL_NAMESPACE_BEGIN                                           \
    namespace tim                                                                        \
    {                                                                                    \
    namespace cereal                                                                     \
    {
#define ROCPROFILER_SDK_CEREAL_NAMESPACE_END                                             \
    }                                                                                    \
    }  // namespace ::tim::cereal

#include "common/defines.h"
#include "common/environment.hpp"
#include "common/pci_bdf.hpp"
#include "core/gpu_visibility.hpp"
#include "gpu.hpp"

#include <timemory/manager.hpp>

#include <set>
#include <string>

#include "core/agent_manager.hpp"

#include "core/amd_smi.hpp"
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/cxx/serialization.hpp>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/version.h>

#include "logger/debug.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <exception>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace rocprofsys
{
namespace gpu
{
namespace
{
#define ROCPROFSYS_AMD_SMI_CALL(ERROR_CODE)                                              \
    ::rocprofsys::gpu::check_amdsmi_error(ERROR_CODE, __FILE__, __LINE__)

void
check_amdsmi_error(amdsmi_status_t _code, const char* _file, int _line)
{
    if(_code == AMDSMI_STATUS_SUCCESS) return;
    const char* _msg = nullptr;
    auto        _err = amdsmi_status_code_to_string(_code, &_msg);
    if(_err != AMDSMI_STATUS_SUCCESS)
    {
        throw std::runtime_error(fmt::format(
            "amdsmi_status_code_to_string failed. No error message available. "
            "Error code {} originated at {}:{}",
            static_cast<int>(_code), _file, _line));
    }
    throw std::runtime_error(fmt::format("[{}:{}] Error code {} :: {}", _file, _line,
                                         static_cast<int>(_code), _msg));
}

std::atomic<bool> amdsmi_initialized{ false };

bool
amdsmi_init()
{
    if(amdsmi_initialized.exchange(true)) return true;

    try
    {
        // Currently, only AMDSMI_INIT_AMD_GPUS and AMDSMI_INIT_AMD_NICS are supported
        std::uint64_t init_flags = AMDSMI_INIT_AMD_GPUS;
#ifdef AINIC_SUPPORTED
        init_flags |= AMDSMI_INIT_AMD_NICS;
#endif
        ROCPROFSYS_AMD_SMI_CALL(::amdsmi_init(init_flags));
        get_processor_handles();
    } catch(std::exception& _e)
    {
        LOG_ERROR("Exception thrown initializing amd-smi: {}", _e.what());
        amdsmi_initialized.store(false);
        return false;
    }
    return true;
}

size_t
query_rocm_agents()
{
    size_t _dev_cnt = 0;
    auto   iterator = []([[maybe_unused]] rocprofiler_agent_version_t version,
                       const void** agents, size_t num_agents,
                       [[maybe_unused]] void* user_data) -> rocprofiler_status_t {
        auto& _agent_manager = get_agent_manager_instance();
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* _agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
            agent       cur_agent;
            cur_agent.type =
                (_agent->type == ROCPROFILER_AGENT_TYPE_GPU ? agent_type::GPU
                                                              : agent_type::CPU);
            cur_agent.handle               = _agent->id.handle;
            cur_agent.device_id            = _agent->device_id;
            cur_agent.node_id              = _agent->node_id;
            cur_agent.logical_node_id      = _agent->logical_node_id;
            cur_agent.logical_node_type_id = _agent->logical_node_type_id;
            cur_agent.location_id          = _agent->location_id;
            cur_agent.domain               = _agent->domain;
#if(ROCPROFILER_VERSION >= 600)
            // runtime_visibility.hip honors both ROCR_VISIBLE_DEVICES and
            // HIP_VISIBLE_DEVICES (hip visibility requires hsa visibility).
            cur_agent.hip_visible = (_agent->runtime_visibility.hip != 0);
#else
            cur_agent.hip_visible = true;
#endif
            cur_agent.name         = std::string(_agent->name);
            cur_agent.model_name   = std::string(_agent->model_name);
            cur_agent.vendor_name  = std::string(_agent->vendor_name);
            cur_agent.product_name = std::string(_agent->product_name);

            cur_agent.agent_info = agent_info::to_json_string(*_agent);

            _agent_manager.insert_agent(cur_agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    try
    {
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, iterator,
                                           sizeof(rocprofiler_agent_v0_t), nullptr);
    } catch(std::exception& _e)
    {
        LOG_ERROR("Exception thrown getting the rocm agents: {}. _dev_cnt={}", _e.what(),
                  _dev_cnt);
    }
    _dev_cnt = get_agent_manager_instance().get_gpu_agents_count();
    return _dev_cnt;
}

/// @brief Whether @p value is an increasing integer list (e.g. "4,5"), not a permutation
///        ("5,4") or UUID list. HIP ordinals follow the env-var order; this mapping uses
///        agent order, so only an increasing subset is reliable.
[[nodiscard]] bool
visibility_mask_is_ordered_subset(std::string_view value)
{
    std::size_t       previous      = 0;
    bool              have_previous = false;
    std::string       token;
    std::stringstream stream{ std::string{ value } };
    while(std::getline(stream, token, ','))
    {
        const auto first = token.find_first_not_of(" \t");
        if(first == std::string::npos)
        {
            continue;
        }
        const auto last = token.find_last_not_of(" \t");
        token           = token.substr(first, last - first + 1);

        std::size_t ordinal        = 0;
        const auto* token_end      = token.data() + token.size();
        const auto [parse_end, ec] = std::from_chars(token.data(), token_end, ordinal);
        if(ec != std::errc{} || parse_end != token_end)
        {
            return false;
        }

        if(have_previous && ordinal <= previous)
        {
            return false;
        }
        previous      = ordinal;
        have_previous = true;
    }
    return true;
}

struct visibility_env
{
    const char* name;
    std::string value;
};

[[nodiscard]] std::optional<visibility_env>
mask_if_unreliable(const char* name, std::string_view value)
{
    if(value.empty() || visibility_mask_is_ordered_subset(value))
    {
        return std::nullopt;
    }
    return visibility_env{ name, std::string{ value } };
}

/// @brief First visibility mask that would put HIP ordinals in a different order
///        from agent-manager (physical) order. @c ROCR_VISIBLE_DEVICES is always
///        considered; then @c HIP_VISIBLE_DEVICES, else CUDA / @c GPU_DEVICE_ORDINAL.
[[nodiscard]] std::optional<visibility_env>
unreliable_hip_ordinal_mask()
{
    if(auto rocr = mask_if_unreliable(
           "ROCR_VISIBLE_DEVICES",
           rocprofsys::get_env<std::string>("ROCR_VISIBLE_DEVICES", "")))
    {
        return rocr;
    }

    const auto hip = rocprofsys::get_env<std::string>("HIP_VISIBLE_DEVICES", "");
    if(!hip.empty())
    {
        return mask_if_unreliable("HIP_VISIBLE_DEVICES", hip);
    }

    // SDK also reads CUDA_VISIBLE_DEVICES/GPU_DEVICE_ORDINAL to determine visible GPUs
    const auto cuda = rocprofsys::get_env<std::string>("CUDA_VISIBLE_DEVICES", "");
    if(!cuda.empty())
    {
        return mask_if_unreliable("CUDA_VISIBLE_DEVICES", cuda);
    }

    return mask_if_unreliable("GPU_DEVICE_ORDINAL",
                              rocprofsys::get_env<std::string>("GPU_DEVICE_ORDINAL", ""));
}

/// @brief Whether HIP ordinal k can be treated as the k-th hip-visible agent.
///
/// Logs once and returns false when a permutation or UUID list would mislabel
/// hipFile slots. AMD SMI uses BDF membership and is not gated by this check.
[[nodiscard]] bool
hip_ordinal_mapping_is_reliable()
{
    static const bool reliable = [] {
        const auto unreliable = unreliable_hip_ordinal_mask();
        if(!unreliable)
        {
            return true;
        }

        LOG_WARNING("hipFile telemetry disabled: {}={} is a permutation or UUID "
                    "list, not an increasing integer subset. hipFile per-GPU stats "
                    "are indexed by HIP ordinal and would be recorded under the "
                    "wrong profiler GPU. Use an increasing mask such as "
                    "HIP_VISIBLE_DEVICES=4,5. AMD SMI and rocprofiler-sdk GPU "
                    "metrics are unaffected.",
                    unreliable->name, unreliable->value);
        return false;
    }();
    return reliable;
}
}  // namespace

int
device_count()
{
    static const int _num_devices = query_rocm_agents();
    return _num_devices;
}

std::optional<std::set<std::string>>
get_visible_gpu_bdfs()
{
    // Ensure the rocprofiler-sdk agents (and their runtime_visibility) are populated.
    // No GPU agents means the query found nothing to report on (or failed), so runtime
    // visibility is unknown rather than empty.
    if(device_count() == 0)
    {
        return std::nullopt;
    }

    std::set<std::string> _bdfs;
    for(const auto& _agent :
        get_agent_manager_instance().get_agents_by_type(agent_type::GPU))
    {
        if(!_agent)
        {
            continue;
        }
        if(!_agent->hip_visible)
        {
            continue;
        }
        _bdfs.insert(
            common::format_pci_bdf_from_location_id(_agent->domain, _agent->location_id));
    }
    return _bdfs;
}

std::vector<std::size_t>
get_visible_gpu_type_indices()
{
    // Ensure agents (and runtime_visibility.hip) are populated.
    if(device_count() == 0)
    {
        return {};
    }

    // hipFile indexes per_gpu_stats by HIP ordinal (hipGetDevice()). rocprofiler-sdk
    // agents only store a hip_visible bit, so HIP ordinal k is taken to be the k-th
    // hip_visible agent in physical order. That is correct for an increasing integer
    // mask (HIP_VISIBLE_DEVICES=4,5) and wrong for a permutation (5,4) or a UUID list.
    // When the hipFile project fills each slot's UUID or PCI BDF, match that identity
    // to the rocprofiler agent (the same join AMD SMI already does with BDF) and this
    // fail-closed gate can be replaced with a real mapping for permutations.
    if(!hip_ordinal_mapping_is_reliable())
    {
        return {};
    }

    std::vector<std::size_t> indices;
    for(const auto& gpu_agent :
        get_agent_manager_instance().get_agents_by_type(agent_type::GPU))
    {
        if(!gpu_agent || !gpu_agent->hip_visible)
        {
            continue;
        }
        indices.push_back(gpu_agent->device_type_index);
    }
    return indices;
}

bool
initialize_amdsmi()
{
    return amdsmi_init();
}

bool
reinitialize_amdsmi()
{
    static std::mutex                 mtx;
    const std::lock_guard<std::mutex> lock(mtx);
    amdsmi_initialized.store(false);
    return amdsmi_init();
}

template <typename ArchiveT>
void
add_device_metadata(ArchiveT& ar)
{
    namespace cereal = tim::cereal;
    using cereal::make_nvp;

    using agent_vec_t = std::vector<rocprofiler_agent_v0_t>;

    auto iterator_cb = []([[maybe_unused]] rocprofiler_agent_version_t version,
                          const void** agents, size_t num_agents,
                          [[maybe_unused]] void* user_data) -> rocprofiler_status_t {
        auto* agents_vec = static_cast<agent_vec_t*>(user_data);
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* _agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
            if(_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
            {
                agents_vec->push_back(*_agent);
            }
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    auto _agents_vec = agent_vec_t{};
    try
    {
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, iterator_cb,
                                           sizeof(rocprofiler_agent_v0_t), &_agents_vec);
    } catch(std::exception& _e)
    {
        LOG_ERROR("Exception thrown getting the rocm agents: {}", _e.what());
    }

    ar(make_nvp("rocm_agents", _agents_vec));
}

void
add_device_metadata()
{
    if(device_count() == 0) return;

    ::tim::manager::add_metadata([](auto& ar) {
        try
        {
            add_device_metadata(ar);
        } catch(std::runtime_error& _e)
        {
            LOG_ERROR("Exception thrown adding device metadata: {}", _e.what());
        }
    });
}

/*
 * Required amdsmi methods to get processors and handles
 */

std::uint32_t                        processors::total_processor_count  = 0;
std::vector<amdsmi_processor_handle> processors::processors_list        = {};
std::vector<bool>                    processors::vcn_device_level_only  = {};
std::vector<bool>                    processors::jpeg_device_level_only = {};
std::vector<bool>                    processors::vcn_busy_supported     = {};
std::vector<bool>                    processors::jpeg_busy_supported    = {};
std::vector<bool>                    processors::xgmi_supported         = {};
std::vector<bool>                    processors::pcie_supported         = {};

std::vector<amdsmi_processor_handle> processors::ainic_list        = {};
std::uint32_t                        processors::total_ainic_count = 0;

void
get_processor_handles()
{
    std::uint32_t socket_count;
    std::uint32_t processor_count;
    processors::processors_list.clear();
    processors::ainic_list.clear();

    // Passing nullptr will return us the number of sockets available for read in this
    // system
    auto ret = amdsmi_get_socket_handles(&socket_count, nullptr);
    if(ret != AMDSMI_STATUS_SUCCESS)
    {
        return;
    }
    std::vector<amdsmi_socket_handle> sockets(socket_count);
    ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
    for(auto& socket : sockets)
    {
        // Passing nullptr will return us the number of processors available for read for
        // this socket
        ret = amdsmi_get_processor_handles(socket, &processor_count, nullptr);
        if(ret != AMDSMI_STATUS_SUCCESS)
        {
            return;
        }
        std::vector<amdsmi_processor_handle> all_processors(processor_count);
        ret =
            amdsmi_get_processor_handles(socket, &processor_count, all_processors.data());
        if(ret != AMDSMI_STATUS_SUCCESS)
        {
            return;
        }

        for(auto& processor : all_processors)
        {
            processor_type_t processor_type = {};
            ret = amdsmi_get_processor_type(processor, &processor_type);
#ifdef AINIC_SUPPORTED
            if(processor_type == AMDSMI_PROCESSOR_TYPE_AMD_NIC)
            {
                processors::ainic_list.push_back(processor);
                continue;
            }
#endif  // AINIC_SUPPORTED
            if(processor_type != AMDSMI_PROCESSOR_TYPE_AMD_GPU)
            {
                throw std::runtime_error("Not AMD_GPU device type!");
            }
            processors::processors_list.push_back(processor);

            amdsmi_gpu_metrics_t gpu_metrics;
            bool                 vcn_supported = false, jpeg_supported = false;
            bool                 v_busy_supported = false, j_busy_supported = false;
            bool                 xgmi_supported = false, pcie_supported = false;
            // AMD SMI will not report VCN_activity and JPEG_activity, if VCN_busy or
            // JPEG_busy fields are available.
            if(amdsmi_get_gpu_metrics_info(processor, &gpu_metrics) ==
               AMDSMI_STATUS_SUCCESS)
            {
                // Helper lambda to check if any value in the array is valid (not
                // UINT16_MAX)
                auto has_valid_u16 = [](const auto& arr) {
                    return std::any_of(std::begin(arr), std::end(arr),
                                       [](auto val) { return val != UINT16_MAX; });
                };

                // Helper lambda to check if any value in the array is valid (not
                // UINT64_MAX)
                auto has_valid_u64 = [](const auto& arr) {
                    return std::any_of(std::begin(arr), std::end(arr),
                                       [](auto val) { return val != UINT64_MAX; });
                };

                vcn_supported  = has_valid_u16(gpu_metrics.vcn_activity);
                jpeg_supported = has_valid_u16(gpu_metrics.jpeg_activity);

                // Check if VCN and JPEG busy metrics are available
                for(const auto& xcp : gpu_metrics.xcp_stats)
                {
                    if(!v_busy_supported && has_valid_u16(xcp.vcn_busy))
                        v_busy_supported = true;
                    if(!j_busy_supported && has_valid_u16(xcp.jpeg_busy))
                        j_busy_supported = true;
                    if(v_busy_supported && j_busy_supported) break;
                }

                // Check if XGMI metrics are supported (any value not at max)
                xgmi_supported = (gpu_metrics.xgmi_link_width != UINT16_MAX) ||
                                 (gpu_metrics.xgmi_link_speed != UINT16_MAX) ||
                                 has_valid_u64(gpu_metrics.xgmi_read_data_acc) ||
                                 has_valid_u64(gpu_metrics.xgmi_write_data_acc);

                // Check if PCIe metrics are supported (any value not at max)
                pcie_supported = (gpu_metrics.pcie_link_width != UINT16_MAX) ||
                                 (gpu_metrics.pcie_link_speed != UINT16_MAX) ||
                                 (gpu_metrics.pcie_bandwidth_acc != UINT64_MAX) ||
                                 (gpu_metrics.pcie_bandwidth_inst != UINT64_MAX);
            }
            processors::vcn_device_level_only.push_back(vcn_supported);
            processors::jpeg_device_level_only.push_back(jpeg_supported);
            processors::vcn_busy_supported.push_back(v_busy_supported);
            processors::jpeg_busy_supported.push_back(j_busy_supported);
            processors::xgmi_supported.push_back(xgmi_supported);
            processors::pcie_supported.push_back(pcie_supported);
        }
    }
    processors::total_processor_count = processors::processors_list.size();
    processors::total_ainic_count     = processors::ainic_list.size();
}

bool
vcn_is_device_level_only(std::uint32_t dev_id)
{
    if(dev_id >= processors::vcn_device_level_only.size()) return false;
    return processors::vcn_device_level_only[dev_id];
}

bool
jpeg_is_device_level_only(std::uint32_t dev_id)
{
    if(dev_id >= processors::jpeg_device_level_only.size()) return false;
    return processors::jpeg_device_level_only[dev_id];
}

bool
is_vcn_busy_supported(std::uint32_t dev_id)
{
    if(dev_id >= processors::vcn_busy_supported.size()) return false;
    return processors::vcn_busy_supported[dev_id];
}

bool
is_jpeg_busy_supported(std::uint32_t dev_id)
{
    if(dev_id >= processors::jpeg_busy_supported.size()) return false;
    return processors::jpeg_busy_supported[dev_id];
}

bool
is_xgmi_supported(std::uint32_t dev_id)
{
    if(dev_id >= processors::xgmi_supported.size()) return false;
    return processors::xgmi_supported[dev_id];
}

bool
is_pcie_supported(std::uint32_t dev_id)
{
    if(dev_id >= processors::pcie_supported.size()) return false;
    return processors::pcie_supported[dev_id];
}

std::uint32_t
get_processor_count()
{
    return processors::total_processor_count;
}

amdsmi_processor_handle
get_handle_from_id(std::uint32_t dev_id)
{
    return processors::processors_list[dev_id];
}

}  // namespace gpu
}  // namespace rocprofsys
