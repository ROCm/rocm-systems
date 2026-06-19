#include <atomic>

#include <climits>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace rocm_timesync
{

namespace ipc
{

void atomic_notify_one(std::atomic<uint32_t>& atomic)
{
    syscall(
        SYS_futex,
        reinterpret_cast<uint32_t*>(&atomic),
        FUTEX_WAKE,
        1,
        nullptr,
        nullptr,
        0
    );
}

void atomic_notify_all(std::atomic<uint32_t>& atomic)
{
    syscall(
        SYS_futex,
        reinterpret_cast<uint32_t*>(&atomic),
        FUTEX_WAKE,
        INT_MAX,
        nullptr,
        nullptr,
        0
    );
}

void atomic_wait(const std::atomic<uint32_t>& atomic, uint32_t expected)
{
    syscall(
        SYS_futex,
        const_cast<uint32_t*>(
            reinterpret_cast<const uint32_t*>(&atomic)
        ),
        FUTEX_WAIT,
        expected,
        nullptr,
        nullptr,
        0
    );
}

} // namespace ipc

} // namespace rocm_timesync
