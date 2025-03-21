// MIT License
//
// Copyright (c) 2024-2025 ROCm Developer Tools
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

#include "lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"
#include "lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter_types.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <sys/ioctl.h>

#include <fcntl.h>
#include <unistd.h>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <vector>

namespace rocprofiler
{
namespace pc_sampling
{
namespace ioctl
{
namespace
{
#define PC_SAMPLING_IOCTL_BITMASK 0xFFFF

#define PC_SAMPLING_IOCTL_COMPUTE_VERSION(major, minor) ROCPROFILER_COMPUTE_VERSION(major, minor, 0)

using pcs_ioctl_version_t = uint32_t;

#define KFD_ROCP_PCS_METHOD_PAIR(KFD_ENUM_VAL, ROCP_ENUM_VAL)                                      \
    template <>                                                                                    \
    struct pcs_method_pair<KFD_ENUM_VAL>                                                           \
    {                                                                                              \
        static constexpr auto rocp_enum_val = ROCP_ENUM_VAL;                                       \
    };

template <size_t Idx>
struct pcs_method_pair;

KFD_ROCP_PCS_METHOD_PAIR(ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_NONE,
                         ROCPROFILER_PC_SAMPLING_METHOD_NONE);
KFD_ROCP_PCS_METHOD_PAIR(ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_HOSTTRAP_V1,
                         ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP);
KFD_ROCP_PCS_METHOD_PAIR(ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_STOCHASTIC_V1,
                         ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC);

template <size_t Idx, size_t... Tail>
rocprofiler_pc_sampling_method_t
get_rocp_pcs_method(rocprofiler_ioctl_pc_sampling_method_kind_t kfd_method,
                    std::index_sequence<Idx, Tail...>)
{
    if(kfd_method == Idx) return pcs_method_pair<Idx>::rocp_enum_val;
    // recursion until tail empty
    if constexpr(sizeof...(Tail) > 0)
        return get_rocp_pcs_method(kfd_method, std::index_sequence<Tail...>{});
    // Return none value if matching fails
    return ROCPROFILER_PC_SAMPLING_METHOD_NONE;
}

rocprofiler_pc_sampling_method_t
get_rocp_pcs_method_from_kfd(rocprofiler_ioctl_pc_sampling_method_kind_t kfd_method)
{
    return get_rocp_pcs_method(
        kfd_method, std::make_index_sequence<ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_LAST>{});
}

int
kfd_open()
{
    int             fd              = -1;
    constexpr auto* kfd_device_name = "/dev/kfd";

    fd = open(kfd_device_name, O_RDWR | O_CLOEXEC);

    if(fd == -1)
    {
        ROCP_CI_LOG(WARNING) << fmt::format("Cannot open {} for pc sampling", kfd_device_name);
        return -1;
    }

    return fd;
}

/** Call ioctl, restarting if it is interrupted
 * Taken from libhsakmt.c
 */
int
ioctl(int fd, unsigned long request, void* arg)
{
    int ret;

    do
    {
        ret = ::ioctl(fd, request, arg);
    } while(ret == -1 && (errno == EINTR || errno == EAGAIN));

    if(ret == -1 && errno == EBADF)
    {
        /* In case pthread_atfork didn't catch it, this will
         * make any subsequent hsaKmt calls fail in CHECK_KFD_OPEN.
         */
        printf("Invalid KFD descriptor: %d\n", fd);
    }

    return ret * errno;
}

// More or less taken from the HsaKmt

/**
 * @brief Query KFD IOCTL version.
 *
 */
rocprofiler_status_t
get_ioctl_version(rocprofiler_ioctl_version_info_t& ioctl_version)
{
    struct kfd_ioctl_get_version_args args = {.major_version = 0, .minor_version = 0};
    if(ioctl(get_kfd_fd(), AMDKFD_IOC_GET_VERSION, &args) != 0)
    {
        // An error occured while querying KFD IOCTL version.
        return ROCPROFILER_STATUS_ERROR;
    }

    // Extract KFD IOCTL version
    ioctl_version.major_version = args.major_version;
    ioctl_version.minor_version = args.minor_version;
    return ROCPROFILER_STATUS_SUCCESS;
}

/**
 * @brief KFD IOCTL PC Sampling API version is provided via
 * the `kfd_ioctl_pc_sample_args.version` field by
 * @ref ::KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES` IOCTL function.
 * The latter function requires @p kfd_gpu_id
 * This mechanism is used for internal versioning of the PC sampling
 * implementation.
 *
 * @todo: Remove once KFD IOCTL is upstreamed.
 *
 * @param[in] kfd_gpu_id - KFD GPU identifier
 * @param[out] pcs_ioctl_version - The PC sampling IOCTL version. Invalid if
 * the return value is different than ::ROCPROFILER_STATUS_SUCCESS
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
get_pc_sampling_ioctl_version(uint32_t kfd_gpu_id, pcs_ioctl_version_t* pcs_ioctl_version)
{
    struct kfd_ioctl_pc_sample_args args;
    args.op              = KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES;
    args.gpu_id          = kfd_gpu_id;
    args.sample_info_ptr = 0;
    args.num_sample_info = 0;
    args.flags           = 0;
    args.version         = 0;

    auto ret = ioctl(get_kfd_fd(), AMDKFD_IOC_PC_SAMPLE, &args);

    if(ret == -EBUSY)
    {
        // The ROCProfiler-SDK is used inside the ROCgdb.
        // The `KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES` is not executed,
        // so the value of the args.version is irrelevant.
        // Report that PC sampling cannot be used from within the ROCgdb.
        return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
    }
    else if(ret == -EOPNOTSUPP)
    {
        // The GPU does not support PC sampling.
        return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
    }
    else if(ret != 0)
    {
        // An unexpected error occured, so we cannot be sure if the
        // context of the `version` is valid.
        return ROCPROFILER_STATUS_ERROR;
    }

    // `version` field contains PC Sampling IOCTL version
    auto version = args.version;
    // Lower 16 bits represent minor version
    auto minor_version = version & PC_SAMPLING_IOCTL_BITMASK;
    // Upper 16 bits represent major version
    auto major_version = (version >> 16) & PC_SAMPLING_IOCTL_BITMASK;
    // finally, compute the version
    *pcs_ioctl_version = PC_SAMPLING_IOCTL_COMPUTE_VERSION(major_version, minor_version);

    return ROCPROFILER_STATUS_SUCCESS;
}

/**
 * @brief Check if PC sampling feature is supported in KFD.
 *
 * Starting from KFD IOCTL 1.16, KFD delivers beta implementation of the PC sampling.
 *
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS PC sampling is supported in the driver.
 * Other values informs users about the reason why PC sampling is not supported.
 */
rocprofiler_status_t
is_pc_sampling_supported()
{
    // Verify KFD 1.16 version
    rocprofiler_ioctl_version_info_t ioctl_version = {.major_version = 0, .minor_version = 0};
    auto                             status        = get_ioctl_version(ioctl_version);
    if(status != ROCPROFILER_STATUS_SUCCESS)
        return status;
    else if(ioctl_version.major_version < 1 || ioctl_version.minor_version < 16)
    {
        // The KFD IOCTL version is the same for all available devices.
        // Thus, emit the message and skip all tests and samples on the system in use.
        ROCP_INFO << "PC sampling unavailable. Please update amdgpu driver to at least 1.16.";
        return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
    }

    // PC Sampling feature is supported in the driver.
    return ROCPROFILER_STATUS_SUCCESS;
}

/**
 * @brief Check if PC sampling method is supported on the agent.
 *
 * The function complements the @ref is_pc_sampling_supported function.
 * It introduces a strict check against the PC sampling IOCTL version
 * that tells us whether a certain PC sampling method is safe to be used
 * on the specific device architecture.
 *
 * @param method - PC sampling method to be checked
 * @param agent - The agent to be checked
 * @param pcs_ioctl_version - The PC sampling IOCTL version
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS - The method is supported
 * Other values informs users about the reason why the method is not supported.
 */
rocprofiler_status_t
is_pc_sampling_method_supported(rocprofiler_pc_sampling_method_t method,
                                const rocprofiler_agent_t*       agent,
                                pcs_ioctl_version_t              pcs_ioctl_version)
{
    std::string_view agent_name = agent->name;
    if(method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP)
    {
        if(agent_name == "gfx90a")
        {
            // 0.1 version enables host-trap PC sampling on gfx90a
            if(pcs_ioctl_version >= PC_SAMPLING_IOCTL_COMPUTE_VERSION(0, 1))
                return ROCPROFILER_STATUS_SUCCESS;
            else
                return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
        }
        else if(agent_name.find("gfx94") == 0)
        {
            // 0.3 version enables host-trap PC sampling on gfx940, gfx941, gfx942, etc.
            if(pcs_ioctl_version >= PC_SAMPLING_IOCTL_COMPUTE_VERSION(0, 3))
                return ROCPROFILER_STATUS_SUCCESS;
            else
                return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
        }
        else if(agent_name.find("gfx95") == 0)
        {
            // 1.2 version enables host-trap PC sampling on gfx950
            if(pcs_ioctl_version >= PC_SAMPLING_IOCTL_COMPUTE_VERSION(1, 2))
                return ROCPROFILER_STATUS_SUCCESS;
            else
                return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
        }
    }
    else if(method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC)
    {
        if(agent_name == "gfx90a")
        {
            // gfx90a doesn't support stochastic PC sampling
            return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
        }
        else if(agent_name.find("gfx94") == 0)
        {
            // 1.3 version enables stochastic PC sampling on gfx940, gfx941, gfx942, etc.
            if(pcs_ioctl_version >= PC_SAMPLING_IOCTL_COMPUTE_VERSION(1, 3))
                return ROCPROFILER_STATUS_SUCCESS;
            else
                return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
        }
        else if(agent_name.find("gfx95") == 0)
        {
            // 1.4 version enables stochastic PC sampling on gfx950
            if(pcs_ioctl_version >= PC_SAMPLING_IOCTL_COMPUTE_VERSION(1, 4))
                return ROCPROFILER_STATUS_SUCCESS;
            else
                return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
        }
    }
    else
    {
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    // Other architecture do not support the PC sampling method.
    return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
}

/**
 * @brief Returns the PC sampling IOCTL version if the PC sampling feature is supported in the
 * driver.
 *
 * First, check the minimal driver version via @ref is_pc_sampling_supported.
 * Then, determines the PC sampling IOCTL version via @ref get_pc_sampling_ioctl_version.
 *
 * @param [in] kfd_gpu_id - The KFD GPU identifier
 * @param [out] pcs_ioctl_version_t - The PC sampling IOCTL version
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
get_pcs_ioctl_version_if_kfd_supports(uint32_t kfd_gpu_id, pcs_ioctl_version_t* pcs_ioctl_version)
{
    // Check if the PC sampling feature is supported in the driver
    auto status = is_pc_sampling_supported();
    if(status != ROCPROFILER_STATUS_SUCCESS) return status;

    // Get the PC sampling IOCTL version
    status = get_pc_sampling_ioctl_version(kfd_gpu_id, pcs_ioctl_version);
    return status;
}

/**
 * @brief Same as @ref is_pc_sampling_method_supported.
 */
rocprofiler_status_t
is_pc_sampling_method_supported(rocprofiler_ioctl_pc_sampling_method_kind_t ioctl_method,
                                const rocprofiler_agent_t*                  agent,
                                pcs_ioctl_version_t                         pcs_ioctl_version)
{
    auto rocp_method = get_rocp_pcs_method_from_kfd(ioctl_method);
    return is_pc_sampling_method_supported(rocp_method, agent, pcs_ioctl_version);
}

/**
 * @kfd_gpu_id represents the gpu identifier read from the content of the
 * /sys/class/kfd/kfd/topology/nodes/<node-id>/gpu_id.
 */
rocprofiler_ioctl_status_t
ioctl_query_pc_sampling_capabilities(uint32_t  kfd_gpu_id,
                                     void*     sample_info,
                                     uint32_t  sample_info_sz,
                                     uint32_t* size)
{
    int                             ret;
    struct kfd_ioctl_pc_sample_args args;

    assert(sizeof(rocprofiler_ioctl_pc_sampling_info_t) == sizeof(struct kfd_pc_sample_info));

    ret                  = ROCPROFILER_IOCTL_STATUS_SUCCESS;
    args.op              = KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES;
    args.gpu_id          = kfd_gpu_id;
    args.sample_info_ptr = (uint64_t) sample_info;
    args.num_sample_info = sample_info_sz;
    args.flags           = 0;

    ret = ioctl(get_kfd_fd(), AMDKFD_IOC_PC_SAMPLE, &args);

    if(ret != 0)
    {
        if(ret == -EBUSY)
        {
            // Querying PC sampling capabilities is requsted from within the ROCgdb
            // which is not supported.
            return ROCPROFILER_IOCTL_STATUS_UNAVAILABLE;
        }
        ROCP_WARNING << "IOCTL failed to query PC sampling configs: " << ret << "\n";
    }
    *size = args.num_sample_info;

    if(ret == -ENOSPC) return ROCPROFILER_IOCTL_STATUS_BUFFER_TOO_SMALL;

    return ret != 0 ? ROCPROFILER_IOCTL_STATUS_ERROR : ROCPROFILER_IOCTL_STATUS_SUCCESS;
}

rocprofiler_status_t
convert_ioctl_pcs_config_to_rocp(const rocprofiler_ioctl_pc_sampling_info_t& ioctl_pcs_config,
                                 rocprofiler_pc_sampling_configuration_t&    rocp_pcs_config)
{
    // Sometimes, the KFD returns 0 for `method` and `units` as an error.
    // Note: the 0 is not of the matching enumeration.
    // Thus, the default case remains here to indicate that KFD edge case
    // and prevents failures inside rocprofiler.

    switch(ioctl_pcs_config.method)
    {
        case ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_HOSTTRAP_V1:
            rocp_pcs_config.method = ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;
            break;
        case ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_STOCHASTIC_V1:
            rocp_pcs_config.method = ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC;
            break;
        default:
            // Sampling method unsupported, return the error
            return ROCPROFILER_STATUS_ERROR;
    }

    switch(ioctl_pcs_config.units)
    {
        case ROCPROFILER_IOCTL_PC_SAMPLING_UNIT_INTERVAL_MICROSECONDS:
            rocp_pcs_config.unit = ROCPROFILER_PC_SAMPLING_UNIT_TIME;
            break;
        case ROCPROFILER_IOCTL_PC_SAMPLING_UNIT_INTERVAL_CYCLES:
            rocp_pcs_config.unit = ROCPROFILER_PC_SAMPLING_UNIT_CYCLES;
            break;
        case ROCPROFILER_IOCTL_PC_SAMPLING_UNIT_INTERVAL_INSTRUCTIONS:
            rocp_pcs_config.unit = ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS;
            break;
        default:
            // Sampling unit unsupported, return error
            return ROCPROFILER_STATUS_ERROR;
    }

    if(ioctl_pcs_config.interval != 0)
    {
        // The pc sampling is configured on the corresponding device.
        // The `interval` contains the value of the interval used for deliverying samples.
        // Values of `interval_min` and `interval_max` are irrelevant.
        rocp_pcs_config.min_interval = ioctl_pcs_config.interval;
        rocp_pcs_config.max_interval = ioctl_pcs_config.interval;
    }
    else
    {
        // No one configured PC sampling on the corresponding device.
        // Read the values of min and max interval provided by the KFD
        rocp_pcs_config.min_interval = ioctl_pcs_config.interval_min;
        rocp_pcs_config.max_interval = ioctl_pcs_config.interval_max;
    }

    rocp_pcs_config.flags = ioctl_pcs_config.flags;

    return ROCPROFILER_STATUS_SUCCESS;
}
}  // namespace

int
get_kfd_fd()
{
    static auto _v = kfd_open();
    return _v;
}

rocprofiler_status_t
ioctl_query_pcs_configs(const rocprofiler_agent_t* agent, rocp_pcs_cfgs_vec_t& rocp_configs)
{
    pcs_ioctl_version_t pcs_ioctl_version = 0;
    auto status = get_pcs_ioctl_version_if_kfd_supports(agent->gpu_id, &pcs_ioctl_version);
    if(status != ROCPROFILER_STATUS_SUCCESS) return status;

    uint32_t kfd_gpu_id = agent->gpu_id;

    const size_t ioctl_configs_num = 10;
    uint32_t     size              = 0;

    std::vector<rocprofiler_ioctl_pc_sampling_info_t> ioctl_configs(ioctl_configs_num);

    auto ret = ioctl_query_pc_sampling_capabilities(
        kfd_gpu_id, ioctl_configs.data(), ioctl_configs.size(), &size);
    if(ret == ROCPROFILER_IOCTL_STATUS_BUFFER_TOO_SMALL)
    {
        ioctl_configs.resize(size);
        ret = ioctl_query_pc_sampling_capabilities(
            kfd_gpu_id, ioctl_configs.data(), ioctl_configs.size(), &size);
    }

    if(ret == ROCPROFILER_IOCTL_STATUS_UNAVAILABLE)
    {
        // The PC sampling is accessed from within the ROCgdb which is not supported.
        return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
    }
    else if(ret != ROCPROFILER_IOCTL_STATUS_SUCCESS)
    {
        ROCP_WARNING << "......... Failed while iterating over PC sampling configurations\n";
        return ROCPROFILER_STATUS_ERROR;
    }

    for(auto const& ioctl_cfg : ioctl_configs)
    {
        // FIXME: Why this happens?
        if(ioctl_cfg.method == 0) continue;

        // Strict check whether the driver version (safely) supports the sampling method for
        // this specific device architecture.
        // If not, skip showing this configuration to the user, as it's not safe to use this
        // sampling method on this device.
        if(is_pc_sampling_method_supported(ioctl_cfg.method, agent, pcs_ioctl_version) !=
           ROCPROFILER_STATUS_SUCCESS)
            continue;

        auto rocp_cfg = common::init_public_api_struct(rocprofiler_pc_sampling_configuration_t{});
        auto rocp_ret = convert_ioctl_pcs_config_to_rocp(ioctl_cfg, rocp_cfg);
        if(rocp_ret != ROCPROFILER_STATUS_SUCCESS)
        {
            // This should never happened, unless the KFD is broken.
            continue;
        }
        rocp_configs.emplace_back(rocp_cfg);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
create_ioctl_pcs_config_from_rocp(rocprofiler_ioctl_pc_sampling_info_t& ioctl_cfg,
                                  rocprofiler_pc_sampling_method_t      method,
                                  rocprofiler_pc_sampling_unit_t        unit,
                                  uint64_t                              interval)
{
    switch(method)
    {
        case ROCPROFILER_PC_SAMPLING_METHOD_NONE: return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
        case ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC:
            ioctl_cfg.method = ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_STOCHASTIC_V1;
            break;
        case ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP:
            ioctl_cfg.method = ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_HOSTTRAP_V1;
            break;
        case ROCPROFILER_PC_SAMPLING_METHOD_LAST: return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    switch(unit)
    {
        case ROCPROFILER_PC_SAMPLING_UNIT_NONE: return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
        case ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS:
            ioctl_cfg.units = ROCPROFILER_IOCTL_PC_SAMPLING_UNIT_INTERVAL_INSTRUCTIONS;
            break;
        case ROCPROFILER_PC_SAMPLING_UNIT_CYCLES:
            ioctl_cfg.units = ROCPROFILER_IOCTL_PC_SAMPLING_UNIT_INTERVAL_CYCLES;
            break;
        case ROCPROFILER_PC_SAMPLING_UNIT_TIME:
            ioctl_cfg.units = ROCPROFILER_IOCTL_PC_SAMPLING_UNIT_INTERVAL_MICROSECONDS;
            break;
        case ROCPROFILER_PC_SAMPLING_UNIT_LAST: return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    ioctl_cfg.interval = interval;
    // TODO: Is it possible to use flags for interval values that are power of 2
    // when specifying stochastic on MI300?
    ioctl_cfg.flags        = 0;
    ioctl_cfg.interval_min = 0;
    ioctl_cfg.interval_max = 0;

    return ROCPROFILER_STATUS_SUCCESS;
}

/**
 * @brief Reserve PC sampling service on the device
 * @param[out] ioctl_pcs_id - If the return value is ROCPROFILER_STATUS_SUCCESS,
 * contains the id that uniquely identifies PC sampling session within IOCTL.
 */
rocprofiler_status_t
ioctl_pcs_create(const rocprofiler_agent_t*       agent,
                 rocprofiler_pc_sampling_method_t method,
                 rocprofiler_pc_sampling_unit_t   unit,
                 uint64_t                         interval,
                 uint32_t*                        ioctl_pcs_id)
{
    pcs_ioctl_version_t pcs_ioctl_version = 0;
    auto status = get_pcs_ioctl_version_if_kfd_supports(agent->gpu_id, &pcs_ioctl_version);
    if(status != ROCPROFILER_STATUS_SUCCESS) return status;

    // Strict check: whether the driver version (safely) supports the sampling method for
    // this specific device architecture. If not, return an error and prevent the user from
    // using this sampling method on this device.
    status = is_pc_sampling_method_supported(method, agent, pcs_ioctl_version);
    if(status != ROCPROFILER_STATUS_SUCCESS) return status;

    rocprofiler_ioctl_pc_sampling_info_t ioctl_cfg;
    auto ret = create_ioctl_pcs_config_from_rocp(ioctl_cfg, method, unit, interval);
    if(ret != ROCPROFILER_STATUS_SUCCESS)
    {
        return ret;
    }

    struct kfd_ioctl_pc_sample_args args;

    if(!ioctl_pcs_id) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    *ioctl_pcs_id = INVALID_TRACE_ID;

    args.op              = KFD_IOCTL_PCS_OP_CREATE;
    args.gpu_id          = agent->gpu_id;
    args.sample_info_ptr = (uint64_t)(&ioctl_cfg);
    args.num_sample_info = 1;
    args.trace_id        = INVALID_TRACE_ID;

    if(get_kfd_fd() == -1) return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;

    auto ioctl_ret = ioctl(get_kfd_fd(), AMDKFD_IOC_PC_SAMPLE, &args);
    *ioctl_pcs_id  = args.trace_id;

    if(ioctl_ret != 0)
    {
        if(errno == EBUSY || errno == EEXIST)
        {
            // Currently, KFD uses EBUSY when e.g., PC sampling create is requested from
            // withing the ROCgdb.
            // On the other hand, EEXIST is used when one tries to create a PC sampling
            // with a configuration different than the one already active.
            return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
        }
        else if(errno == EINVAL)
        {
            // invalid argument (e.g., interval must be power of 2, but a value that's
            // not power of 2 is provided)
            return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
        }
        else
        {
            // generic error
            return ROCPROFILER_STATUS_ERROR;
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

}  // namespace ioctl
}  // namespace pc_sampling
}  // namespace rocprofiler
