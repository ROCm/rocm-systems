#include "lib/rocprofiler-sdk/pc_sampling/kernel/selector.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/kernel/kfd_backend.hpp"

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
std::mutex                                             selection_mutex{};
std::unordered_map<uint64_t, backend_kind_t>           agent_backend_kind{};
std::unordered_map<backend_kind_t, PcSamplingBackend*> backend_by_kind{
    {backend_kind_t::kfd, &kfd_backend()}, {backend_kind_t::kgd, &kgd_backend()}};

backend_kind_t
pick_kind(const rocprofiler_agent_t* agent, kernel_iface_mode_t mode)
{
    switch(mode)
    {
        case kernel_iface_mode_t::kfd: return backend_kind_t::kfd;
        case kernel_iface_mode_t::kgd: return backend_kind_t::kgd;
        case kernel_iface_mode_t::auto_select:
        {
            if(kgd_backend().probe(agent)) return backend_kind_t::kgd;
            if(kfd_backend().probe(agent)) return backend_kind_t::kfd;
            return backend_kind_t::kfd;
        }
    }
    return backend_kind_t::kfd;
}
}  // namespace

kernel_iface_mode_t
parse_kernel_iface_mode()
{
    const auto value = common::get_env("ROCPROFILER_KERNEL_IFACE", std::string_view{"kfd"});
    if(value == "kgd") return kernel_iface_mode_t::kgd;
    if(value == "auto") return kernel_iface_mode_t::auto_select;
    return kernel_iface_mode_t::kfd;
}

PcSamplingBackend&
select_backend(const rocprofiler_agent_t* agent)
{
    const auto agent_key = agent->id.handle;

    {
        std::lock_guard<std::mutex> lock(selection_mutex);
        if(auto it = agent_backend_kind.find(agent_key); it != agent_backend_kind.end())
            return *backend_by_kind.at(it->second);
    }

    const auto mode = parse_kernel_iface_mode();
    const auto kind = pick_kind(agent, mode);

    if(mode == kernel_iface_mode_t::kgd && kind == backend_kind_t::kgd && !kgd_backend().probe(agent))
    {
        ROCP_INFO << "ROCPROFILER_KERNEL_IFACE=kgd but render-node PCS probe failed for agent "
                  << agent->id.handle << "; operations will fail until driver is ready.\n";
    }

    {
        std::lock_guard<std::mutex> lock(selection_mutex);
        agent_backend_kind[agent_key] = kind;
    }

    auto& backend = *backend_by_kind.at(kind);
    log_backend_selection(agent, backend);
    return backend;
}

void
log_backend_selection(const rocprofiler_agent_t* agent, PcSamplingBackend& backend)
{
    static std::mutex log_mutex{};
    std::lock_guard     lock(log_mutex);

    ROCP_INFO << fmt::format(
        "PC sampling kernel backend: {} (agent id {}, drm_render_minor {}, gpu_id {})\n",
        backend.name(),
        agent->id.handle,
        agent->drm_render_minor,
        agent->gpu_id);
}

}  // namespace kernel
}  // namespace pc_sampling
}  // namespace rocprofiler
