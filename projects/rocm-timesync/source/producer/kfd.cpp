#include <vector>
#include <cstdio>

#include <rocm-timesync/rocm_timesync.hpp>

#include <hsakmt/hsakmt.h>
#include <hsakmt/hsakmttypes.h>

#include "kfd.hpp"

int kfd_enumerate_gpus(std::vector<gpu_context_t>& gpus)
{
    HSAKMT_STATUS status = hsaKmtOpenKFD();
    if (status != HSAKMT_STATUS_SUCCESS) {
        fprintf(stderr, "hsaKmtOpenKFD failed: %d\n", status);
        return -1;
    }

    HsaSystemProperties sys_props{};
    status = hsaKmtAcquireSystemProperties(&sys_props);
    if (status != HSAKMT_STATUS_SUCCESS) {
        fprintf(stderr, "hsaKmtAcquireSystemProperties failed: %d\n", status);
        return -1;
    }

    for (uint32_t node_id = 0; node_id < sys_props.NumNodes; node_id++) {
        HsaNodeProperties props{};
        status = hsaKmtGetNodeProperties(node_id, &props);
        if (status != HSAKMT_STATUS_SUCCESS) {
            fprintf(stderr, "hsaKmtGetNodeProperties failed: %d\n", status);
            continue;
        }

        // Skip CPU nodes
        if (props.KFDGpuID == 0)
            continue;

        gpus.push_back({
            .kfd_gpu_id = props.KFDGpuID,
            .node_id = node_id
        });
    }

    return 0;
}

// TODO: produce a true crosststamp, possibly through a new ioctl() call
int kfd_get_crosststamp(const gpu_context_t& gpu, crosststamp_t& tstamp)
{
    HsaClockCounters ctrs;
    int status = hsaKmtGetClockCounters(gpu.node_id, &ctrs);
    if (status != HSAKMT_STATUS_SUCCESS)
        return -1;

    tstamp.gpu_timestamp = ctrs.GPUClockCounter;
    tstamp.system_timestamp = ctrs.SystemClockCounter;

    return 0;
}
