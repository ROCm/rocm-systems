#pragma once

#include <atomic>
namespace rocm_timesync
{

namespace ipc
{

void atomic_notify_one(std::atomic<uint32_t>& atomic);
void atomic_notify_all(std::atomic<uint32_t>& atomic);
void atomic_wait(const std::atomic<uint32_t>& atomic, uint32_t expected);

} // namespace ipc

} // namespace rocm_timesync
