// KGD backend: per-GPU render node + DRM_AMDGPU_IOCTL_PC_SAMPLE

#include "lib/rocprofiler-sdk/pc_sampling/kernel/backend.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/details/amdgpu_drm_pcs.h"
#include "lib/rocprofiler-sdk/pc_sampling/kernel/pcs_common.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/kernel/render_node_fd.hpp"

#include <cassert>
#include <cerrno>
#include <vector>

namespace rocprofiler
{
namespace pc_sampling
{
namespace kernel
{
namespace
{
class KgdBackend final : public PcSamplingBackend
{
public:
    backend_kind_t kind() const override { return backend_kind_t::kgd; }

    const char* name() const override { return "kgd"; }

    bool probe(const rocprofiler_agent_t* agent) override
    {
        if(!agent || agent->drm_render_minor == 0) return false;
        const int fd = get_render_node_fd(agent->drm_render_minor);
        if(fd < 0) return false;

        // Capability probe: QUERY_CAPABILITIES with zero buffer (version only).
        // TODO(kernel): confirm whether a separate KGD capability bit is needed in addition.
        drm_amdgpu_pc_sample_args args{};
        args.op     = KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES;
        args.gpu_id = agent->gpu_id;

        const int ret = pcs_common::ioctl_retry(fd, DRM_AMDGPU_IOCTL_PC_SAMPLE, &args);
        return ret == 0 || ret == -ENOSPC;
    }

    rocprofiler_status_t query_configs(const rocprofiler_agent_t* agent,
                                       rocp_pcs_cfgs_vec_t&      configs) override
    {
        const int fd = get_render_node_fd(agent->drm_render_minor);
        if(fd < 0) return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;

        pcs_common::pcs_impl_version_t pcs_version = 0;
        if(auto st = query_pcs_version(agent, fd, &pcs_version); st != ROCPROFILER_STATUS_SUCCESS)
            return st;

        const size_t initial_capacity = 10;
        uint32_t     count            = 0;
        std::vector<rocprofiler_ioctl_pc_sampling_info_t> driver_cfgs(initial_capacity);

        auto ioctl_st = query_capabilities(
            fd, agent->gpu_id, driver_cfgs.data(), static_cast<uint32_t>(driver_cfgs.size()), &count);
        if(ioctl_st == ROCPROFILER_IOCTL_STATUS_BUFFER_TOO_SMALL)
        {
            driver_cfgs.resize(count);
            ioctl_st = query_capabilities(fd,
                                          agent->gpu_id,
                                          driver_cfgs.data(),
                                          static_cast<uint32_t>(driver_cfgs.size()),
                                          &count);
        }

        if(ioctl_st == ROCPROFILER_IOCTL_STATUS_UNAVAILABLE)
            return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
        if(ioctl_st != ROCPROFILER_IOCTL_STATUS_SUCCESS) return ROCPROFILER_STATUS_ERROR;

        for(const auto& driver_cfg : driver_cfgs)
        {
            if(driver_cfg.method == 0) continue;
            if(pcs_common::is_method_supported(driver_cfg.method, agent, pcs_version) !=
               ROCPROFILER_STATUS_SUCCESS)
                continue;

            auto rocp_cfg = common::init_public_api_struct(rocprofiler_pc_sampling_configuration_t{});
            if(pcs_common::convert_driver_config_to_rocp(driver_cfg, rocp_cfg) !=
               ROCPROFILER_STATUS_SUCCESS)
                continue;
            configs.emplace_back(rocp_cfg);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    }

    rocprofiler_status_t create_session(const rocprofiler_agent_t*       agent,
                                        rocprofiler_pc_sampling_method_t method,
                                        rocprofiler_pc_sampling_unit_t   unit,
                                        uint64_t                         interval,
                                        uint32_t*                        trace_id) override
    {
        if(!trace_id) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

        const int fd = get_render_node_fd(agent->drm_render_minor);
        if(fd < 0) return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;

        pcs_common::pcs_impl_version_t pcs_version = 0;
        if(auto st = query_pcs_version(agent, fd, &pcs_version); st != ROCPROFILER_STATUS_SUCCESS)
            return st;

        if(pcs_common::is_method_supported(method, agent, pcs_version) !=
           ROCPROFILER_STATUS_SUCCESS)
            return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;

        rocprofiler_ioctl_pc_sampling_info_t driver_cfg{};
        if(auto st = pcs_common::create_driver_config_from_rocp(driver_cfg, method, unit, interval);
           st != ROCPROFILER_STATUS_SUCCESS)
            return st;

        drm_amdgpu_pc_sample_args args{};
        args.op              = KFD_IOCTL_PCS_OP_CREATE;
        args.gpu_id          = agent->gpu_id;
        args.sample_info_ptr = reinterpret_cast<uint64_t>(&driver_cfg);
        args.num_sample_info = 1;
        args.trace_id        = pcs_common::invalid_trace_id;

        if(pcs_common::ioctl_retry(fd, DRM_AMDGPU_IOCTL_PC_SAMPLE, &args) != 0)
            return pcs_common::map_create_errno_to_status();

        *trace_id = args.trace_id;
        return ROCPROFILER_STATUS_SUCCESS;
    }

private:
    static rocprofiler_status_t query_pcs_version(const rocprofiler_agent_t*         agent,
                                                  int                                  fd,
                                                  pcs_common::pcs_impl_version_t* pcs_version)
    {
        drm_amdgpu_pc_sample_args args{};
        args.op     = KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES;
        args.gpu_id = agent->gpu_id;

        const int ret = pcs_common::ioctl_retry(fd, DRM_AMDGPU_IOCTL_PC_SAMPLE, &args);
        if(ret == -EBUSY) return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
        if(ret == -EOPNOTSUPP || ret == -ENOTTY) return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
        if(ret != 0) return ROCPROFILER_STATUS_ERROR;

        *pcs_version = pcs_common::compute_pcs_version(args.version);
        return ROCPROFILER_STATUS_SUCCESS;
    }

    static rocprofiler_ioctl_status_t query_capabilities(int       fd,
                                                         uint32_t  gpu_id,
                                                         void*     sample_info,
                                                         uint32_t  sample_info_sz,
                                                         uint32_t* size)
    {
        assert(sizeof(rocprofiler_ioctl_pc_sampling_info_t) == sizeof(kfd_pc_sample_info));

        drm_amdgpu_pc_sample_args args{};
        args.op              = KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES;
        args.gpu_id          = gpu_id;
        args.sample_info_ptr = reinterpret_cast<uint64_t>(sample_info);
        args.num_sample_info = sample_info_sz;

        const int ret = pcs_common::ioctl_retry(fd, DRM_AMDGPU_IOCTL_PC_SAMPLE, &args);
        *size         = args.num_sample_info;
        if(ret == -EBUSY) return ROCPROFILER_IOCTL_STATUS_UNAVAILABLE;
        if(ret == -ENOSPC) return ROCPROFILER_IOCTL_STATUS_BUFFER_TOO_SMALL;
        return ret == 0 ? ROCPROFILER_IOCTL_STATUS_SUCCESS : ROCPROFILER_IOCTL_STATUS_ERROR;
    }
};
}  // namespace

PcSamplingBackend&
kgd_backend()
{
    static KgdBackend instance{};
    return instance;
}

}  // namespace kernel
}  // namespace pc_sampling
}  // namespace rocprofiler
