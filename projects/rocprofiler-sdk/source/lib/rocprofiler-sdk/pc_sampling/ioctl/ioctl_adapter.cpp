// Public facade for PC sampling ioctls. Implementation lives under pc_sampling/kernel/.

#include "lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter.hpp"

#include "lib/rocprofiler-sdk/pc_sampling/kernel/kfd_backend.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/kernel/selector.hpp"

namespace rocprofiler
{
namespace pc_sampling
{
namespace ioctl
{

int
get_kfd_fd()
{
    return kernel::get_kfd_fd();
}

rocprofiler_status_t
ioctl_query_pcs_configs(const rocprofiler_agent_t* agent, rocp_pcs_cfgs_vec_t& rocp_configs)
{
    return kernel::select_backend(agent).query_configs(agent, rocp_configs);
}

rocprofiler_status_t
ioctl_pcs_create(const rocprofiler_agent_t*       agent,
                 rocprofiler_pc_sampling_method_t method,
                 rocprofiler_pc_sampling_unit_t   unit,
                 uint64_t                         interval,
                 uint32_t*                        ioctl_pcs_id)
{
    return kernel::select_backend(agent).create_session(agent, method, unit, interval, ioctl_pcs_id);
}

}  // namespace ioctl
}  // namespace pc_sampling
}  // namespace rocprofiler
