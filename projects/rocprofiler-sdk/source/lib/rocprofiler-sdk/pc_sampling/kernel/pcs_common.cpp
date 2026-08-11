// Shared PCS helpers used by both KFD and KGD backends.

#include "lib/rocprofiler-sdk/pc_sampling/kernel/pcs_common.hpp"

#include <rocprofiler-sdk/rocprofiler.h>

#include <sys/ioctl.h>

#include <algorithm>
#include <cerrno>
#include <string_view>
#include <utility>

namespace rocprofiler
{
namespace pc_sampling
{
namespace kernel
{
namespace pcs_common
{
namespace
{
constexpr uint32_t version_bitmask = 0xFFFF;

#define PCS_METHOD_PAIR(IOCTL_VAL, ROCP_VAL)                                                       \
    template <>                                                                                  \
    struct method_pair<IOCTL_VAL>                                                                \
    {                                                                                            \
        static constexpr auto rocp_val = ROCP_VAL;                                               \
    };

template <size_t Idx>
struct method_pair;

PCS_METHOD_PAIR(ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_HOSTTRAP_V1,
                ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP);
PCS_METHOD_PAIR(ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_STOCHASTIC_V1,
                ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC);

template <size_t Idx, size_t... Tail>
rocprofiler_pc_sampling_method_t
ioctl_method_to_rocp(rocprofiler_ioctl_pc_sampling_method_kind_t ioctl_method,
                     std::index_sequence<Idx, Tail...>)
{
    if(ioctl_method == Idx) return method_pair<Idx>::rocp_val;
    if constexpr(sizeof...(Tail) > 0)
        return ioctl_method_to_rocp(ioctl_method, std::index_sequence<Tail...>{});
    return ROCPROFILER_PC_SAMPLING_METHOD_NONE;
}

rocprofiler_pc_sampling_method_t
ioctl_method_to_rocp(rocprofiler_ioctl_pc_sampling_method_kind_t ioctl_method)
{
    return ioctl_method_to_rocp(
        ioctl_method, std::make_index_sequence<ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_LAST>{});
}
}  // namespace

pcs_impl_version_t
compute_pcs_version(uint32_t raw_version)
{
    const auto minor = raw_version & version_bitmask;
    const auto major = (raw_version >> 16) & version_bitmask;
    return ROCPROFILER_COMPUTE_VERSION(major, minor, 0);
}

rocprofiler_status_t
is_method_supported(rocprofiler_pc_sampling_method_t method,
                    const rocprofiler_agent_t*       agent,
                    pcs_impl_version_t               pcs_version)
{
    const std::string_view agent_name = agent->name;

    if(method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP)
    {
        if(agent_name == "gfx90a")
            return pcs_version >= ROCPROFILER_COMPUTE_VERSION(0, 1, 0)
                       ? ROCPROFILER_STATUS_SUCCESS
                       : ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
        if(agent_name.starts_with("gfx94"))
            return pcs_version >= ROCPROFILER_COMPUTE_VERSION(0, 3, 0)
                       ? ROCPROFILER_STATUS_SUCCESS
                       : ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
        if(agent_name.starts_with("gfx95"))
            return pcs_version >= ROCPROFILER_COMPUTE_VERSION(1, 2, 0)
                       ? ROCPROFILER_STATUS_SUCCESS
                       : ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
        if(agent_name.starts_with("gfx12"))
            return pcs_version >= ROCPROFILER_COMPUTE_VERSION(1, 5, 0)
                       ? ROCPROFILER_STATUS_SUCCESS
                       : ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
    }
    else if(method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC)
    {
        if(agent_name == "gfx90a") return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
        if(agent_name.starts_with("gfx94"))
            return pcs_version >= ROCPROFILER_COMPUTE_VERSION(1, 3, 0)
                       ? ROCPROFILER_STATUS_SUCCESS
                       : ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
        if(agent_name.starts_with("gfx95"))
            return pcs_version >= ROCPROFILER_COMPUTE_VERSION(1, 4, 0)
                       ? ROCPROFILER_STATUS_SUCCESS
                       : ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
        if(agent_name.starts_with("gfx1250"))
            return pcs_version >= ROCPROFILER_COMPUTE_VERSION(1, 7, 0)
                       ? ROCPROFILER_STATUS_SUCCESS
                       : ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
    }
    else if(method == ROCPROFILER_PC_SAMPLING_METHOD_NONE ||
            method == ROCPROFILER_PC_SAMPLING_METHOD_LAST)
    {
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
}

rocprofiler_status_t
is_method_supported(rocprofiler_ioctl_pc_sampling_method_kind_t ioctl_method,
                    const rocprofiler_agent_t*                  agent,
                    pcs_impl_version_t                        pcs_version)
{
    return is_method_supported(ioctl_method_to_rocp(ioctl_method), agent, pcs_version);
}

int
ioctl_retry(int fd, unsigned long request, void* arg)
{
    int ret = 0;
    do
    {
        ret = ::ioctl(fd, request, arg);
    } while(ret == -1 && (errno == EINTR || errno == EAGAIN));
    return ret == 0 ? 0 : -errno;
}

rocprofiler_status_t
convert_driver_config_to_rocp(const rocprofiler_ioctl_pc_sampling_info_t& driver_cfg,
                              rocprofiler_pc_sampling_configuration_t&    rocp_cfg)
{
    switch(driver_cfg.method)
    {
        case ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_HOSTTRAP_V1:
            rocp_cfg.method = ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;
            break;
        case ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_STOCHASTIC_V1:
            rocp_cfg.method = ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC;
            break;
        default: return ROCPROFILER_STATUS_ERROR;
    }

    switch(driver_cfg.units)
    {
        case ROCPROFILER_IOCTL_PC_SAMPLING_UNIT_INTERVAL_MICROSECONDS:
            rocp_cfg.unit = ROCPROFILER_PC_SAMPLING_UNIT_TIME;
            break;
        case ROCPROFILER_IOCTL_PC_SAMPLING_UNIT_INTERVAL_CYCLES:
            rocp_cfg.unit = ROCPROFILER_PC_SAMPLING_UNIT_CYCLES;
            break;
        case ROCPROFILER_IOCTL_PC_SAMPLING_UNIT_INTERVAL_INSTRUCTIONS:
            rocp_cfg.unit = ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS;
            break;
        default: return ROCPROFILER_STATUS_ERROR;
    }

    if(driver_cfg.interval != 0)
    {
        rocp_cfg.min_interval = driver_cfg.interval;
        rocp_cfg.max_interval = driver_cfg.interval;
    }
    else
    {
        rocp_cfg.min_interval = driver_cfg.interval_min;
        rocp_cfg.max_interval = std::min(driver_cfg.interval_max, 1ul << 20);
    }

    rocp_cfg.flags = driver_cfg.flags;
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
create_driver_config_from_rocp(rocprofiler_ioctl_pc_sampling_info_t& driver_cfg,
                               rocprofiler_pc_sampling_method_t      method,
                               rocprofiler_pc_sampling_unit_t        unit,
                               uint64_t                              interval)
{
    switch(method)
    {
        case ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC:
            driver_cfg.method = ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_STOCHASTIC_V1;
            break;
        case ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP:
            driver_cfg.method = ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_HOSTTRAP_V1;
            break;
        default: return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    switch(unit)
    {
        case ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS:
            driver_cfg.units = ROCPROFILER_IOCTL_PC_SAMPLING_UNIT_INTERVAL_INSTRUCTIONS;
            break;
        case ROCPROFILER_PC_SAMPLING_UNIT_CYCLES:
            driver_cfg.units = ROCPROFILER_IOCTL_PC_SAMPLING_UNIT_INTERVAL_CYCLES;
            break;
        case ROCPROFILER_PC_SAMPLING_UNIT_TIME:
            driver_cfg.units = ROCPROFILER_IOCTL_PC_SAMPLING_UNIT_INTERVAL_MICROSECONDS;
            break;
        default: return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    driver_cfg.interval     = interval;
    driver_cfg.flags        = 0;
    driver_cfg.interval_min = 0;
    driver_cfg.interval_max = 0;
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
map_create_errno_to_status()
{
    switch(errno)
    {
        case EBUSY:
        case EEXIST: return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
        case EINVAL: return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
        default: return ROCPROFILER_STATUS_ERROR;
    }
}

}  // namespace pcs_common
}  // namespace kernel
}  // namespace pc_sampling
}  // namespace rocprofiler
