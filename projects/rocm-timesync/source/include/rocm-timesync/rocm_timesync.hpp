#pragma once

#include <cstdint>

namespace rocm
{
namespace timesync
{

int translate_time(uint32_t agent_kfd_gpu_id, uint64_t agent_timestamp, uint64_t *system_timestamp);

} // namespace timesync
} // namespace rocm
