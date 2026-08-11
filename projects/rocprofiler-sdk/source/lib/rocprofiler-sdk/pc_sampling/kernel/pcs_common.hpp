// Shared helpers for KFD and KGD PC sampling backends.

#pragma once

#include "lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter_types.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <cstdint>

namespace rocprofiler
{
namespace pc_sampling
{
namespace kernel
{
namespace pcs_common
{
constexpr uint32_t invalid_trace_id = 0;

using pcs_impl_version_t = uint32_t;

pcs_impl_version_t
compute_pcs_version(uint32_t raw_version);

/// Per-arch/method PCS implementation version gates (from driver QUERY_CAPABILITIES).
rocprofiler_status_t
is_method_supported(rocprofiler_pc_sampling_method_t method,
                    const rocprofiler_agent_t*       agent,
                    pcs_impl_version_t               pcs_version);

rocprofiler_status_t
is_method_supported(rocprofiler_ioctl_pc_sampling_method_kind_t ioctl_method,
                    const rocprofiler_agent_t*                  agent,
                    pcs_impl_version_t                        pcs_version);

/// Retry ioctl on EINTR/EAGAIN; returns 0 on success else negative errno.
int
ioctl_retry(int fd, unsigned long request, void* arg);

rocprofiler_status_t
convert_driver_config_to_rocp(const rocprofiler_ioctl_pc_sampling_info_t& driver_cfg,
                              rocprofiler_pc_sampling_configuration_t&    rocp_cfg);

rocprofiler_status_t
create_driver_config_from_rocp(rocprofiler_ioctl_pc_sampling_info_t& driver_cfg,
                               rocprofiler_pc_sampling_method_t      method,
                               rocprofiler_pc_sampling_unit_t        unit,
                               uint64_t                              interval);

rocprofiler_status_t
map_create_errno_to_status();

}  // namespace pcs_common
}  // namespace kernel
}  // namespace pc_sampling
}  // namespace rocprofiler
