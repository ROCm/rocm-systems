#include "lib/rocprofiler-sdk/pc_sampling/kernel/render_node_fd.hpp"

#include "lib/common/logging.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <mutex>
#include <unordered_map>

namespace rocprofiler
{
namespace pc_sampling
{
namespace kernel
{
namespace
{
std::mutex                             render_fd_mutex{};
std::unordered_map<uint32_t, int>      render_fd_cache{};

int
open_render_node(uint32_t drm_render_minor)
{
    if(drm_render_minor == 0) return -1;

    char path[64];
    const int path_len =
        snprintf(path, sizeof(path), "/dev/dri/renderD%u", 128 + drm_render_minor);
    if(path_len <= 0 || static_cast<size_t>(path_len) >= sizeof(path)) return -1;

    return open(path, O_RDWR | O_CLOEXEC);
}
}  // namespace

int
get_render_node_fd(uint32_t drm_render_minor)
{
    if(drm_render_minor == 0) return -1;

    std::lock_guard<std::mutex> lock(render_fd_mutex);
    if(auto it = render_fd_cache.find(drm_render_minor); it != render_fd_cache.end())
        return it->second;

    const int fd = open_render_node(drm_render_minor);
    if(fd >= 0) render_fd_cache.emplace(drm_render_minor, fd);
    else
        ROCP_CI_LOG(WARNING) << fmt::format(
            "Cannot open render node for drm_render_minor {}", drm_render_minor);

    return fd;
}

}  // namespace kernel
}  // namespace pc_sampling
}  // namespace rocprofiler
