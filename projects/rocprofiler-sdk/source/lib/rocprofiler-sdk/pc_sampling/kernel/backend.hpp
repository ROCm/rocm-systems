// PC sampling kernel backend abstraction (KFD / KGD render node).
//
// rocprofiler-sdk uses this layer for QUERY_CAPABILITIES and CREATE.
// Start/stop/destroy still go through ROCr today (see kernel/README.md).

#pragma once

#include "lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter_types.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/types.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <cstdint>
#include <vector>

namespace rocprofiler
{
namespace pc_sampling
{
namespace kernel
{
using rocp_pcs_cfgs_vec_t = std::vector<rocprofiler_pc_sampling_configuration_t>;

/// Which kernel path is used for PC sampling ioctls.
enum class backend_kind_t
{
    kfd,  ///< /dev/kfd  + AMDKFD_IOC_PC_SAMPLE
    kgd,  ///< /dev/dri/renderD* + DRM_AMDGPU_IOCTL_PC_SAMPLE
};

/// Small interface implemented by KfdBackend and KgdBackend.
class PcSamplingBackend
{
public:
    virtual ~PcSamplingBackend() = default;

    virtual backend_kind_t kind() const = 0;

    /// Short name for logs, e.g. "kfd" or "kgd".
    virtual const char* name() const = 0;

    /// True when this backend can talk to the driver for @p agent (cheap probe).
    virtual bool probe(const rocprofiler_agent_t* agent) = 0;

    /// List supported configurations exposed by the driver.
    virtual rocprofiler_status_t query_configs(const rocprofiler_agent_t* agent,
                                               rocp_pcs_cfgs_vec_t&      configs) = 0;

    /// Reserve a PCS session; writes trace id into @p trace_id.
    virtual rocprofiler_status_t create_session(const rocprofiler_agent_t*       agent,
                                                  rocprofiler_pc_sampling_method_t method,
                                                  rocprofiler_pc_sampling_unit_t   unit,
                                                  uint64_t                         interval,
                                                  uint32_t*                        trace_id) = 0;
};

PcSamplingBackend&
kfd_backend();

PcSamplingBackend&
kgd_backend();

}  // namespace kernel
}  // namespace pc_sampling
}  // namespace rocprofiler
