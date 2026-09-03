#pragma once

#include "lib/rocprofiler-sdk/pc_sampling/kernel/backend.hpp"

namespace rocprofiler
{
namespace pc_sampling
{
namespace kernel
{
/// Cached /dev/kfd descriptor shared with other ioctl users (e.g. PMC lock).
int
get_kfd_fd();

}  // namespace kernel
}  // namespace pc_sampling
}  // namespace rocprofiler
