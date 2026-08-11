// Select KFD vs KGD backend for PC sampling ioctls.

#pragma once

#include "lib/rocprofiler-sdk/pc_sampling/kernel/backend.hpp"

#include <rocprofiler-sdk/fwd.h>

namespace rocprofiler
{
namespace pc_sampling
{
namespace kernel
{
enum class kernel_iface_mode_t
{
    kfd,   ///< Force /dev/kfd (default — matches existing production behavior)
    kgd,   ///< Force render-node path
    auto_select,  ///< Try KGD first, fall back to KFD
};

/// Parse ROCPROFILER_KERNEL_IFACE (default: kfd).
kernel_iface_mode_t
parse_kernel_iface_mode();

/// Backend used for @p agent (cached for process lifetime).
PcSamplingBackend&
select_backend(const rocprofiler_agent_t* agent);

/// Log once which backend was chosen (ROCP_INFO).
void
log_backend_selection(const rocprofiler_agent_t* agent, PcSamplingBackend& backend);

}  // namespace kernel
}  // namespace pc_sampling
}  // namespace rocprofiler
