#include <atomic>

#include <climits>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>

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

void atomic_wait_for(const std::atomic<uint32_t>& atomic, uint32_t expected, int64_t timeout_ms)
{
    struct timespec ts{};

    if (timeout_ms >= 0)
    {
        ts.tv_sec = timeout_ms / 1000;
        ts.tv_nsec = (timeout_ms % 1000) * 1000000ULL;
    }

    syscall(
        SYS_futex,
        const_cast<uint32_t*>(
            reinterpret_cast<const uint32_t*>(&atomic)
        ),
        FUTEX_WAIT,
        expected,
        timeout_ms < 0 ? nullptr : &ts,
        nullptr,
        0
    );
}

void atomic_wait(const std::atomic<uint32_t>& atomic, uint32_t expected)
{
    return atomic_wait_for(atomic, expected, -1);
}

} // namespace ipc

} // namespace rocm_timesync
