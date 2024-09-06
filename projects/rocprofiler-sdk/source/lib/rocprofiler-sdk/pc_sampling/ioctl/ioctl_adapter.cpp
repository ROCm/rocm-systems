// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter_types.hpp"

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

/**
 * @brief Used to determine the version of PC sampling
 * IOCTL implementation in the driver.
 *
 * @todo Remove this once the KFD IOCTL is upstreamed
 */
struct pc_sampling_ioctl_version_t
{
    uint32_t major_version;  /// PC sampling IOCTL major version
    uint32_t minor_version;  /// PC sampling IOCTL minor version
};

int
kfd_open()
{
    int               fd                = -1;
    static const char kfd_device_name[] = "/dev/kfd";

    fd = open(kfd_device_name, O_RDWR | O_CLOEXEC);

    if(fd == -1)
    {
        throw std::runtime_error("Cannot open /dev/kfd");
    }

    return fd;
}

int
get_kfd_fd()
{
    static auto _v = kfd_open();
    return _v;
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
get_pc_sampling_ioctl_version(uint32_t kfd_gpu_id, pc_sampling_ioctl_version_t& pcs_ioctl_version)
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
    pcs_ioctl_version.minor_version = version & PC_SAMPLING_IOCTL_BITMASK;
    // Upper 16 bits represent major version
    pcs_ioctl_version.major_version = (version >> 16) & PC_SAMPLING_IOCTL_BITMASK;

    return ROCPROFILER_STATUS_SUCCESS;
}

/**
 * @brief Check if PC sampling is supported on the device with @p kfd_gpu_id.
 *
 * Starting from KFD IOCTL 1.16, KFD delivers beta implementation of the PC sampling.
 * Furthermore, ROCProfiler-SDK expects PC sampling IOCTL 0.1 version.
 * @todo: Once KFD is upstreamed, ROCProfiler-SDK will rely only on KFD IOCTL version.
 *
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS PC sampling is supported in the driver.
 * Other values informs users about the reason why PC sampling is not supported.
 */
rocprofiler_status_t
is_pc_sampling_supported(const rocprofiler_agent_t* agent)
{
    auto             kfd_gpu_id = agent->gpu_id;
    std::string_view agent_name = agent->name;
    // Verify KFD 1.16 version
    rocprofiler_ioctl_version_info_t ioctl_version = {.major_version = 0, .minor_version = 0};
    auto                             status        = get_ioctl_version(ioctl_version);
    if(status != ROCPROFILER_STATUS_SUCCESS)
        return status;
    else if(ioctl_version.major_version < 1 || ioctl_version.minor_version < 16)
    {
        // The KFD IOCTL version is the same for all available devices.
        // Thus, emit the message and skip all tests and samples on the system in use.
        ROCP_ERROR << "PC sampling unavailable\n";
        return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
    }

    // TODO: remove once KFD is upstreamed
    // Verify PC sampling IOCTL version
    pc_sampling_ioctl_version_t pcs_ioctl_version = {.major_version = 0, .minor_version = 0};
    status = get_pc_sampling_ioctl_version(kfd_gpu_id, pcs_ioctl_version);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        // The reason for not emitting the "PC sampling unavailable" message is the following.
        // Assume that all devices except one support PC sampling on the system.
        // By emitting the message for that one device that doesn't support PC sampling,
        // all tests and samples are skipped. Instead, tests and samples will ignore
        // that one problematic device and continue using PC sampling on other devices
        // that support this feature.
        return status;
    }
    else if(agent_name == "gfx90a")
    {
        // For gfx90a, we expect PC sampling IOCTL to be at least 0.1.
        if(pcs_ioctl_version.major_version > 0 || pcs_ioctl_version.minor_version >= 1)
            return ROCPROFILER_STATUS_SUCCESS;
        else
            return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
    }
    else if(agent_name.find("gfx94") == 0)
    {
        // We expect PC sampling IOCTL to be at least 0.3 for gfx940, gfx941, gfx942, etc.
        if(pcs_ioctl_version.major_version > 0 || pcs_ioctl_version.minor_version >= 3)
            return ROCPROFILER_STATUS_SUCCESS;
        else
            return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
    }
    else
    {
        // The agent does not support PC sampling.
        return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
    }
}

/**
 * @kfd_gpu_id represents the gpu identifier read from the content of the
 * /sys/class/kfd/kfd/topology/nodes/<node-id>/gpu_id.
 */
ROCPROFILER_IOCTL_STATUS
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
        ROCP_ERROR << "IOCTL failed to query PC sampling configs: " << ret << "\n";
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

rocprofiler_status_t
ioctl_query_pcs_configs(const rocprofiler_agent_t* agent, rocp_pcs_cfgs_vec_t& rocp_configs)
{
    if(auto status = is_pc_sampling_supported(agent); status != ROCPROFILER_STATUS_SUCCESS)
        return status;

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
        ROCP_ERROR << "......... Failed while iterating over PC sampling configurations\n";
        return ROCPROFILER_STATUS_ERROR;
    }

    for(auto const& ioctl_cfg : ioctl_configs)
    {
        // FIXME: Why this happens?
        if(ioctl_cfg.method == 0) continue;
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
    if(auto status = is_pc_sampling_supported(agent); status != ROCPROFILER_STATUS_SUCCESS)
        return status;

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

    auto ioctl_ret = ioctl(get_kfd_fd(), AMDKFD_IOC_PC_SAMPLE, &args);
    *ioctl_pcs_id  = args.trace_id;

    if(ioctl_ret != 0 && (errno == EBUSY || errno == EEXIST))
    {
        // Currently, KFD uses EBUSY when e.g., PC sampling create is requested from
        // withing the ROCgdb.
        // On the other hand, EEXIST is used when one tries to create a PC sampling
        // with a configuration different than the one already active.
        return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
    }
    else if(ioctl_ret != 0)
    {
        return ROCPROFILER_STATUS_ERROR;
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

}  // namespace ioctl
}  // namespace pc_sampling
}  // namespace rocprofiler
