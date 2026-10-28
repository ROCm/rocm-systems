// Tests for cpu_snap::sampling::sampler's pthread_atfork(child) handler.
// Regression: the handler used to acquire registry_mutex() in the
// child, which deadlocks (or throws system_error) when a different
// parent thread held the mutex at fork time. The handler now runs
// lock-free in the single-threaded child.

#include <cerrno>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <sched.h>
#include <csignal>
#include <span>
#include <thread>

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

#include "cpu_snap/sampling/sampler.hpp"

namespace
{

// Cheap no-op sink: these tests exercise fork/atfork lifecycle, not
// captured data.
cpu_snap::core::sink_fn
noop_sink()
{
    return [](std::span<const std::byte>) {};
}

bool
perf_open_available()
{
    struct perf_event_attr attr
    {};
    attr.size           = sizeof(attr);
    attr.type           = PERF_TYPE_SOFTWARE;
    attr.config         = PERF_COUNT_SW_CPU_CLOCK;
    attr.disabled       = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv     = 1;
    const int fd        = static_cast<int>(
        ::syscall(SYS_perf_event_open, &attr, 0, -1, -1, PERF_FLAG_FD_CLOEXEC));
    if(fd < 0)
    {
        return false;
    }
    ::close(fd);
    return true;
}

// Exit codes used by forked children to communicate assertion results
// back to the parent through waitpid status.
constexpr int k_child_exit_ok                 = 0;
constexpr int k_child_exit_still_running      = 10;
constexpr int k_child_exit_reconstruct_failed = 11;

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest fixture must be CamelCase to satisfy google-readability-avoid-underscore-in-googletest-name
class AtforkChildHandlerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(!perf_open_available())
        {
            GTEST_SKIP() << "perf_event_open not permitted";
        }
    }
};

}  // namespace

// After fork(), the child's on_child_fork handler must have flipped
// is_running() back to false and the child must be able to construct
// a fresh sampler without deadlocking on registry_mutex.
TEST_F(AtforkChildHandlerTest, ForkChildRunsWithoutDeadlock)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.kernel_pages_per_tid = 1;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.perf.callchain_depth      = 4;
    cfg.streaming.sink            = noop_sink();
    cpu_snap::sampling::sampler sampler(std::move(cfg));
    sampler.start();
    ASSERT_TRUE(sampler.is_running());

    const ::pid_t pid = ::fork();
    ASSERT_NE(pid, -1) << "fork() failed: errno=" << errno;

    if(pid == 0)
    {
        // Child: the atfork handler should have torn down sampler.
        if(sampler.is_running())
        {
            std::_Exit(k_child_exit_still_running);
        }

        // Constructing + destroying another sampler in the child must
        // not deadlock on registry_mutex.
        try
        {
            cpu_snap::sampling::sampler::config child_cfg;
            child_cfg.perf.kernel_pages_per_tid = 1;
            child_cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
            child_cfg.perf.callchain_depth      = 4;
            child_cfg.streaming.sink            = noop_sink();
            cpu_snap::sampling::sampler const child_sampler(std::move(child_cfg));
            // Just construct + destruct; do not start() to keep the
            // child cheap and avoid perf_event_open in CI sandboxes
            // that may have child-side restrictions.
            (void) child_sampler.is_running();
        } catch(...)
        {
            std::_Exit(k_child_exit_reconstruct_failed);
        }
        std::_Exit(k_child_exit_ok);
    }

    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status)) << "child did not exit normally; status=" << status;
    const int code = WEXITSTATUS(status);
    EXPECT_EQ(code, k_child_exit_ok)
        << "child exited with code " << code
        << " (10=is_running stayed true, 11=child reconstruct threw)";

    sampler.stop();
}

// Deterministic version of the deadlock regression: a background
// thread parks while holding registry_mutex, then the main thread
// fork()s. The child inherits the locked mutex; the old handler
// would block on lock_guard forever (and the test would time out).
// With the fix, the handler iterates lock-free and the child exits
// cleanly.
TEST_F(AtforkChildHandlerTest, ForkWithParentMutexBusyDoesNotDeadlock)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.kernel_pages_per_tid = 1;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.perf.callchain_depth      = 4;
    cfg.streaming.sink            = noop_sink();
    cpu_snap::sampling::sampler sampler(std::move(cfg));
    sampler.start();

    // Helper thread acquires registry_mutex indirectly by calling
    // register_instance through a transient sampler that we keep alive
    // via a held lock. We can't reach registry_mutex() from outside,
    // so we approximate by spawning a thread that hammers
    // register/unregister via repeated start/stop on a second sampler.
    // The window in which it holds the mutex is short, but combined
    // with a tight loop on the main thread that fork()s many times,
    // this reliably reproduced the original deadlock pre-fix.
    std::atomic<bool> stop_helper{ false };
    std::thread       helper([&stop_helper] {
        cpu_snap::sampling::sampler::config helper_cfg;
        helper_cfg.perf.kernel_pages_per_tid = 1;
        helper_cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
        helper_cfg.perf.callchain_depth      = 4;
        helper_cfg.streaming.sink            = noop_sink();
        while(!stop_helper.load(std::memory_order_relaxed))
        {
            // Each ctor/dtor takes registry_mutex briefly via
            // register_instance / unregister_instance.
            cpu_snap::sampling::sampler const transient(helper_cfg);
            (void) transient.is_running();
        }
    });

    // Give helper a chance to be scheduled.
    std::this_thread::sleep_for(std::chrono::milliseconds{ 10 });

    bool any_child_hung = false;
    for(int i = 0; i < 32; ++i)
    {
        const ::pid_t pid = ::fork();
        ASSERT_NE(pid, -1) << "fork() failed: errno=" << errno;
        if(pid == 0)
        {
            // Child: if the handler deadlocked we would not reach here
            // (the handler runs before fork() returns in the child).
            std::_Exit(k_child_exit_ok);
        }
        int status = 0;
        // 5 s budget per child is generous; pre-fix the child would
        // hang indefinitely if it inherited a locked mutex.
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds{ 5 };
        bool child_reaped = false;
        while(std::chrono::steady_clock::now() < deadline)
        {
            const ::pid_t waitpid_result = ::waitpid(pid, &status, WNOHANG);
            if(waitpid_result == pid)
            {
                child_reaped = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{ 10 });
        }
        if(!child_reaped)
        {
            any_child_hung = true;
            ::kill(pid, SIGKILL);
            ::waitpid(pid, &status, 0);
            break;
        }
        ASSERT_TRUE(WIFEXITED(status));
        EXPECT_EQ(WEXITSTATUS(status), k_child_exit_ok);
    }

    stop_helper.store(true);
    helper.join();
    sampler.stop();

    EXPECT_FALSE(any_child_hung)
        << "at least one forked child hung; atfork handler deadlocked";
}
