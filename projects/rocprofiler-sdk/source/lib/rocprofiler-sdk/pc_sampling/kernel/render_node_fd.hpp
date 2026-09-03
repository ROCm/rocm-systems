// Cache of DRM render-node file descriptors keyed by drm_render_minor.

#pragma once

#include <cstdint>

namespace rocprofiler
{
namespace pc_sampling
{
namespace kernel
{
/// Returns an open O_RDWR|O_CLOEXEC fd for /dev/dri/renderD{128+minor}, or -1 on failure.
int
get_render_node_fd(uint32_t drm_render_minor);

}  // namespace kernel
}  // namespace pc_sampling
}  // namespace rocprofiler
