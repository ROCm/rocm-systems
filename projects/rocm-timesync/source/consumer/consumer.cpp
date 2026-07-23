#include <rocm-timesync/rocm_timesync.hpp>
#include <core/ipc.hpp>

#include <iostream>
#include <cassert>

namespace rocm
{
namespace timesync
{

int translate_time(uint32_t agent_kfd_gpu_id, uint64_t agent_timestamp, uint64_t *system_timestamp)
{
    std::cerr << "translate_time request for kfd gpu id: " << agent_kfd_gpu_id << std::endl << std::flush;
    return 0;
}

} // namespace timesync
} // namespace rocm
