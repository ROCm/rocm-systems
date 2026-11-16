// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Signal handler integration test.
//
// Tests signal handling behavior with rocprofv3 in various process configurations.
// The test runs indefinitely until killed by an external SIGINT (use `timeout`).
//
// Usage:
//   timeout --signal=INT 3s signal-handler-test [--app-signal-handler|--no-app-signal-handler]
//                                               [--single-process|--fork|--fork-exec|--spawn]
//
// Good case (--app-signal-handler): app installs handler, coordinates shutdown.
//   Used with rocprofv3 --disable-signal-handlers. Profiler flushes via atexit.
//
// Bad case (--no-app-signal-handler): app has SIG_DFL everywhere.
//   Profiler's signal handler is the only thing that can flush data before death.
//
// Fork after work (--fork-after-work): no signal. The app records HIP API calls and markers,
//   then forks children that exit on their own. Records made before fork() belong to the parent
//   only.
//
// TODO: --raw-fork (_Fork without exec) is not supported. _Fork skips
// pthread_atfork handlers, leaving stale profiler state. Extremely rare in practice.

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include <rocprofiler-sdk/registration.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#include <atomic>
#include <stdexcept>
#include <thread>

namespace
{
std::atomic<bool> g_shutdown{false};

template <typename... Args>
std::string
str_join(Args&&... _args)
{
    auto _ss = std::stringstream{};
    ((_ss << std::forward<Args>(_args)), ...);
    return _ss.str();
}

template <typename... Args>
void
emit_roctx_marker(const char* fmt, ...)
{
    char    buf[128];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    roctxMark(buf);
}

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t err = (call);                                                                   \
        if(err != hipSuccess)                                                                      \
        {                                                                                          \
            char msg[256];                                                                         \
            snprintf(msg,                                                                          \
                     sizeof(msg),                                                                  \
                     "HIP error %d at %s:%d: %s",                                                  \
                     err,                                                                          \
                     __FILE__,                                                                     \
                     __LINE__,                                                                     \
                     hipGetErrorString(err));                                                      \
            fprintf(stderr, "%s\n", msg);                                                          \
            throw std::runtime_error(msg);                                                         \
        }                                                                                          \
    } while(0)

__global__ void
test_kernel(float* out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n)
    {
        float v = static_cast<float>(idx);
        for(int i = 0; i < 50; i++)
            v = v * 0.999f + 0.001f;
        out[idx] = v;
    }
}

void
sigint_handler(int)
{
    g_shutdown.store(true, std::memory_order_relaxed);
}

// Runs HIP kernels in a loop until g_shutdown is set.
void
run_kernels(const char* label)
{
    roctxRangePush(str_join(label, "_pid_", getpid()).c_str());

    float* d_buf = nullptr;
    HIP_CHECK(hipMalloc(&d_buf, 1024 * sizeof(float)));

    int iter = 0;
    while(!g_shutdown.load(std::memory_order_relaxed))
    {
        roctxRangePush(str_join(label, "_iter_", iter).c_str());
        test_kernel<<<4, 256>>>(d_buf, 1024);
        roctxRangePop();

        iter++;
        usleep(10000);
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_buf));
    roctxRangePop();

    fprintf(stderr, "  %s PID=%d: completed %d iterations\n", label, getpid(), iter);
}

// Fork child worker for the good case (app handles signals).
// Child ignores SIGINT and waits for 'q' on pipe from parent.
void
fork_child_worker(int id, int pipe_rd)
{
    signal(SIGINT, SIG_IGN);
    fcntl(pipe_rd, F_SETFL, O_NONBLOCK);

    float* d_buf = nullptr;
    HIP_CHECK(hipMalloc(&d_buf, 1024 * sizeof(float)));

    roctxRangePush(str_join("child_", id, "_pid_", getpid()).c_str());

    int iter = 0;
    while(true)
    {
        roctxRangePush(str_join("child_", id, "_iter_", iter).c_str());

        test_kernel<<<4, 256>>>(d_buf, 1024);

        roctxRangePop();
        iter++;

        char cmd = 0;
        if(read(pipe_rd, &cmd, 1) == 1 && cmd == 'q') break;
        usleep(10000);
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_buf));
    roctxRangePop();

    emit_roctx_marker("exit_marker child fork ppid:%d pid:%d", getppid(), getpid());
    fprintf(stderr, "  child_%d PID=%d: exiting after %d iters\n", id, getpid(), iter);
    close(pipe_rd);
}

// ============================================================================
// GOOD CASE modes: app handles signals, coordinates shutdown
// ============================================================================

int
mode_good_single_process()
{
    fprintf(stderr, "Mode: good/single-process, PID=%d\n", getpid());

    struct sigaction sa = {};
    sa.sa_handler       = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);

    run_kernels("parent");

    emit_roctx_marker("exit_marker parent single-process ppid:%d pid:%d", getppid(), getpid());
    fprintf(stderr, "Parent PID=%d: clean exit\n", getpid());
    return 0;
}

int
mode_fork_after_work()
{
    fprintf(stderr, "Mode: fork-after-work, PID=%d\n", getpid());

    // HIP API calls only: a child forked after kernel launches cannot finalize the GPU runtime.
    for(int i = 0; i < 20; ++i)
    {
        float* d_buf = nullptr;
        HIP_CHECK(hipMalloc(&d_buf, 1024 * sizeof(float)));
        HIP_CHECK(hipFree(d_buf));
    }

    // Enough markers that the profiler offloads them to its tmp file before fork().
    for(int i = 0; i < 5000; ++i)
        emit_roctx_marker("prefork_marker %d", i);

    // All traced work is complete before fork: deterministic plain-fork child output requires
    // producers to be quiescent at this boundary. Concurrent trace production is outside this
    // regression's supported scope.
    constexpr int NUM_CHILDREN = 2;
    pid_t         children[NUM_CHILDREN];
    for(int i = 0; i < NUM_CHILDREN; ++i)
    {
        pid_t pid = fork();
        if(pid == 0)
        {
            emit_roctx_marker(
                "exit_marker child fork-after-work ppid:%d pid:%d", getppid(), getpid());
            exit(0);
        }
        children[i] = pid;
    }
    for(int i = 0; i < NUM_CHILDREN; ++i)
        waitpid(children[i], nullptr, 0);

    emit_roctx_marker("exit_marker parent fork-after-work ppid:%d pid:%d", getppid(), getpid());
    fprintf(stderr, "Parent PID=%d: clean exit\n", getpid());
    return 0;
}

int
mode_fork_prepare_marker()
{
    fprintf(stderr, "Mode: fork-prepare-marker, PID=%d\n", getpid());

    auto child = fork();
    if(child == 0)
    {
        emit_roctx_marker(
            "exit_marker child fork-prepare-marker ppid:%d pid:%d", getppid(), getpid());
        exit(0);
    }
    if(child < 0) throw std::runtime_error{"fork failed"};

    waitpid(child, nullptr, 0);
    emit_roctx_marker("exit_marker parent fork-prepare-marker ppid:%d pid:%d", getppid(), getpid());
    fprintf(stderr, "Parent PID=%d: clean exit\n", getpid());
    return 0;
}

volatile sig_atomic_t custom_signal_seen = 0;

void
lifecycle_custom_signal(int signo, siginfo_t* info, void*)
{
    sigset_t mask{};
    sigprocmask(SIG_SETMASK, nullptr, &mask);
    custom_signal_seen = signo == SIGTERM && info && sigismember(&mask, SIGUSR1) ? 1 : -1;
}

int
mode_fork_during_lifecycle(const char* mode, const char* self_path)
{
    const bool  detaching         = std::strstr(mode, "detach") != nullptr;
    const bool  handoff           = std::strstr(mode, "handoff") != nullptr;
    const bool  tool_only         = std::strcmp(mode, "fork-during-tool-finalization") == 0;
    const bool  full_child_matrix = std::strcmp(mode, "fork-during-finalization") == 0;
    const char* phase             = handoff     ? "detach-handoff"
                                    : detaching ? "detach"
                                    : tool_only ? "finalization"
                                                : mode + std::strlen("fork-during-");
    int         ready[2]{};
    int         release[2]{};
    int         attempt[2]{};
    if(handoff)
    {
        if(pipe(attempt) != 0) throw std::runtime_error{"test admission pipe failed"};
        setenv("ROCPROF_INTERNAL_TEST_FINALIZE_ATTEMPT_FD", std::to_string(attempt[1]).c_str(), 1);
    }
    if(pipe(ready) != 0 || pipe(release) != 0) throw std::runtime_error{"test pipe failed"};
    setenv("ROCPROF_INTERNAL_TEST_FORK_PHASE", phase, 1);
    setenv("ROCPROF_INTERNAL_TEST_FORK_READY_FD", std::to_string(ready[1]).c_str(), 1);
    setenv("ROCPROF_INTERNAL_TEST_FORK_RELEASE_FD", std::to_string(release[0]).c_str(), 1);
    emit_roctx_marker("exit_marker parent %s", mode);
    roctxRangePushA("inherited thread range");
    const auto inherited_range = roctxRangeStartA("inherited process range");

    using query_type  = rocprofiler_status_t (*)(int*);
    using attach_type = rocprofiler_status_t (*)();
    auto initialized =
        reinterpret_cast<query_type>(dlsym(RTLD_DEFAULT, "rocprofiler_is_initialized"));
    auto finalized = reinterpret_cast<query_type>(dlsym(RTLD_DEFAULT, "rocprofiler_is_finalized"));
    auto attach    = reinterpret_cast<attach_type>(dlsym(RTLD_DEFAULT, "rocprofiler_attach"));
    auto detach_entry = reinterpret_cast<attach_type>(dlsym(RTLD_DEFAULT, "rocprofiler_detach"));
    if(!initialized || !finalized || !attach || !detach_entry)
        throw std::runtime_error{"missing SDK queries"};
    struct sigaction action
    {};
    action.sa_sigaction = lifecycle_custom_signal;
    action.sa_flags     = SA_SIGINFO | SA_RESTART;
    sigemptyset(&action.sa_mask);
    sigaddset(&action.sa_mask, SIGUSR1);
    if(sigaction(SIGTERM, &action, nullptr) != 0) throw std::runtime_error{"test sigaction failed"};

    std::thread operation;
    if(detaching)
    {
        operation = std::thread{[detach_entry]() {
            if(detach_entry() != ROCPROFILER_STATUS_SUCCESS) std::_Exit(80);
        }};
    }
    else if(tool_only)
    {
        auto fini =
            reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "rocprofiler_tool_test_finalize"));
        if(!fini) throw std::runtime_error{"missing tool test entry"};
        operation = std::thread{fini};
    }
    else
        std::thread{[]() { std::exit(0); }}.detach();

    char    token  = 0;
    ssize_t result = 0;
    do
    {
        result = read(ready[0], &token, 1);
    } while(result < 0 && errno == EINTR);
    if(result != 1) throw std::runtime_error{"test lifecycle did not become ready"};
    if(handoff)
    {
        // Detach owns SDK/tool lifecycle state and has not published its output decision yet.
        std::thread{[]() { std::exit(0); }}.detach();
        do
        {
            result = read(attempt[0], &token, 1);
        } while(result < 0 && errno == EINTR);
        if(result != 1 || write(release[1], &token, 1) != 1)
            throw std::runtime_error{"competing finalization did not reach admission"};
        operation.join();
        for(;;)
            pause();
    }

    const int child_mode_count = full_child_matrix ? 6 : 1;
    for(int child_mode = 0; child_mode < child_mode_count; ++child_mode)
    {
        const auto child = fork();
        if(child == 0)
        {
            struct rlimit no_core
            {
                0, 0
            };
            setrlimit(RLIMIT_CORE, &no_core);
            int init = -2;
            int fini = -2;
            if(initialized(&init) != ROCPROFILER_STATUS_SUCCESS || init != 0 ||
               finalized(&fini) != ROCPROFILER_STATUS_SUCCESS || fini != 1 ||
               attach() != ROCPROFILER_STATUS_ERROR_FINALIZED ||
               detach_entry() != ROCPROFILER_STATUS_ERROR_FINALIZED)
                std::_Exit(81);
            // These inherited stacks must be bypassed by the range wrappers, including pop/stop.
            roctxRangePop();
            roctxRangeStop(inherited_range);
            roctxRangePushA("abandoned new thread range");
            roctxRangePop();
            roctxRangeStop(roctxRangeStartA("abandoned new process range"));
            if(child_mode == 1)
            {
                signal(SIGTERM, SIG_DFL);
                raise(SIGTERM);
            }
            if(child_mode == 2)
            {
                signal(SIGTERM, SIG_IGN);
                raise(SIGTERM);
            }
            if(child_mode == 3)
            {
                raise(SIGTERM);
                struct sigaction restored
                {};
                if(custom_signal_seen != 1 || sigaction(SIGTERM, nullptr, &restored) != 0 ||
                   (restored.sa_flags & (SA_SIGINFO | SA_RESTART)) != (SA_SIGINFO | SA_RESTART) ||
                   !sigismember(&restored.sa_mask, SIGUSR1))
                    std::_Exit(82);
            }
            if(child_mode == 4) std::abort();
            if(child_mode == 5)
            {
                // exec discards every inherited flag and initializes a new profiling session.
                unsetenv("ROCPROF_INTERNAL_TEST_FORK_PHASE");
                execl(self_path, self_path, "--fork-exec-fresh", nullptr);
                std::_Exit(127);
            }
            emit_roctx_marker("exit_marker child abandoned lifecycle");
            std::exit(0);
        }
        if(child < 0) throw std::runtime_error{"test fork failed"};
        int       status          = 0;
        const int expected_signal = child_mode == 1 ? SIGTERM : child_mode == 4 ? SIGABRT : 0;
        if(waitpid(child, &status, 0) != child ||
           (expected_signal ? (!WIFSIGNALED(status) || WTERMSIG(status) != expected_signal)
                            : (!WIFEXITED(status) || WEXITSTATUS(status) != 0)))
            throw std::runtime_error{"child lifecycle/signal behavior failed"};
    }
    fprintf(stderr, "%s: children completed while parent output remained blocked\n", mode);
    if(write(release[1], &token, 1) != 1) throw std::runtime_error{"test lifecycle release failed"};
    if(operation.joinable()) operation.join();
    if(!detaching && !tool_only)
        for(;;)
            pause();
    return 0;
}

int
mode_good_fork()
{
    fprintf(stderr, "Mode: good/fork, PID=%d\n", getpid());

    constexpr int NUM_CHILDREN = 2;
    int           pipes[NUM_CHILDREN][2];
    pid_t         children[NUM_CHILDREN];

    for(int i = 0; i < NUM_CHILDREN; i++)
    {
        if(pipe(pipes[i]) != 0)
        {
            for(int j = 0; j < i; j++)
            {
                close(pipes[j][0]);
                close(pipes[j][1]);
            }

            throw std::runtime_error(
                std::string("signal-handler-test pipe() failed with error code ") +
                std::to_string(errno));
        }
        pid_t pid = fork();
        if(pid == 0)
        {
            close(pipes[i][1]);
            fork_child_worker(i, pipes[i][0]);
            exit(0);
        }
        children[i] = pid;
        close(pipes[i][0]);
    }

    struct sigaction sa = {};
    sa.sa_handler       = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);

    run_kernels("parent");

    fprintf(stderr, "Parent: sending shutdown to children\n");
    for(int i = 0; i < NUM_CHILDREN; i++)
    {
        write(pipes[i][1], "q", 1);
        close(pipes[i][1]);
    }
    for(int i = 0; i < NUM_CHILDREN; i++)
        waitpid(children[i], nullptr, 0);

    emit_roctx_marker("exit_marker parent fork ppid:%d pid:%d", getppid(), getpid());
    fprintf(stderr, "Parent PID=%d: clean exit\n", getpid());
    return 0;
}

int
mode_good_fork_exec(const char* self_path)
{
    fprintf(stderr, "Mode: good/fork-exec, PID=%d\n", getpid());

    constexpr int NUM_CHILDREN = 2;
    pid_t         children[NUM_CHILDREN];

    for(int i = 0; i < NUM_CHILDREN; i++)
    {
        pid_t pid = fork();
        if(pid == 0)
        {
            execl(self_path, self_path, "--single-process", "--app-signal-handler", nullptr);
            _exit(127);
        }
        children[i] = pid;
    }

    struct sigaction sa = {};
    sa.sa_handler       = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);

    run_kernels("parent");

    for(int i = 0; i < NUM_CHILDREN; i++)
        waitpid(children[i], nullptr, 0);

    emit_roctx_marker("exit_marker parent fork-exec ppid:%d pid:%d", getppid(), getpid());
    fprintf(stderr, "Parent PID=%d: clean exit\n", getpid());
    return 0;
}

int
mode_good_spawn(const char* self_path)
{
    fprintf(stderr, "Mode: good/spawn, PID=%d\n", getpid());

    constexpr int NUM_CHILDREN = 2;
    pid_t         children[NUM_CHILDREN];

    for(int i = 0; i < NUM_CHILDREN; i++)
    {
        pid_t pid = vfork();
        if(pid == 0)
        {
            execl(self_path, self_path, "--single-process", "--app-signal-handler", nullptr);
            _exit(127);
        }
        children[i] = pid;
    }

    struct sigaction sa = {};
    sa.sa_handler       = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);

    run_kernels("parent");

    for(int i = 0; i < NUM_CHILDREN; i++)
        waitpid(children[i], nullptr, 0);

    emit_roctx_marker("exit_marker parent spawn ppid:%d pid:%d", getppid(), getpid());
    fprintf(stderr, "Parent PID=%d: clean exit\n", getpid());
    return 0;
}

// ============================================================================
// BAD CASE modes: app does NOT handle signals (SIG_DFL everywhere)
// Profiler's signal handler is the only thing that can flush data.
// ============================================================================

int
mode_bad_single_process()
{
    fprintf(stderr, "Mode: bad/single-process, PID=%d (no signal handler)\n", getpid());
    run_kernels("parent");
    emit_roctx_marker("exit_marker parent single-process ppid:%d pid:%d", getppid(), getpid());
    fprintf(stderr, "Parent PID=%d: exit\n", getpid());
    return 0;
}

int
mode_bad_fork()
{
    fprintf(stderr, "Mode: bad/fork, PID=%d (no signal handler)\n", getpid());

    constexpr int NUM_CHILDREN = 2;
    pid_t         children[NUM_CHILDREN];

    for(int i = 0; i < NUM_CHILDREN; i++)
    {
        pid_t pid = fork();
        if(pid == 0)
        {
            run_kernels(str_join("child_", i).c_str());
            emit_roctx_marker("exit_marker child fork ppid:%d pid:%d", getppid(), getpid());
            exit(0);
        }
        children[i] = pid;
    }

    run_kernels("parent");

    for(int i = 0; i < NUM_CHILDREN; i++)
        waitpid(children[i], nullptr, 0);

    emit_roctx_marker("exit_marker parent fork ppid:%d pid:%d", getppid(), getpid());
    fprintf(stderr, "Parent PID=%d: exit\n", getpid());
    return 0;
}

int
mode_bad_fork_exec(const char* self_path)
{
    fprintf(stderr, "Mode: bad/fork-exec, PID=%d (no signal handler)\n", getpid());

    constexpr int NUM_CHILDREN = 2;
    pid_t         children[NUM_CHILDREN];

    for(int i = 0; i < NUM_CHILDREN; i++)
    {
        pid_t pid = fork();
        if(pid == 0)
        {
            execl(self_path, self_path, "--single-process", "--no-app-signal-handler", nullptr);
            _exit(127);
        }
        children[i] = pid;
    }

    run_kernels("parent");

    for(int i = 0; i < NUM_CHILDREN; i++)
        waitpid(children[i], nullptr, 0);

    emit_roctx_marker("exit_marker parent fork-exec ppid:%d pid:%d", getppid(), getpid());
    fprintf(stderr, "Parent PID=%d: exit\n", getpid());
    return 0;
}

int
mode_bad_spawn(const char* self_path)
{
    fprintf(stderr, "Mode: bad/spawn, PID=%d (no signal handler)\n", getpid());

    constexpr int NUM_CHILDREN = 2;
    pid_t         children[NUM_CHILDREN];

    for(int i = 0; i < NUM_CHILDREN; i++)
    {
        pid_t pid = vfork();
        if(pid == 0)
        {
            execl(self_path, self_path, "--single-process", "--no-app-signal-handler", nullptr);
            _exit(127);
        }
        children[i] = pid;
    }

    run_kernels("parent");

    for(int i = 0; i < NUM_CHILDREN; i++)
        waitpid(children[i], nullptr, 0);

    emit_roctx_marker("exit_marker parent spawn ppid:%d pid:%d", getppid(), getpid());
    fprintf(stderr, "Parent PID=%d: exit\n", getpid());
    return 0;
}

// ============================================================================
// PR# 6717 regression: profiled app leaves a long-lived child alive at NORMAL exit.
//
// No signal is involved. Child outliving the parent must not hang
// ============================================================================
int
mode_lingering_child()
{
    fprintf(stderr, "Mode: lingering-child, PID=%d (normal exit, long-lived child)\n", getpid());

    // Fork BEFORE any HIP/HSA init so the child holds no GPU state (GPU is not fork-safe).
    pid_t pid = fork();
    if(pid == 0)
    {
        // Compile-worker-pool style: ignore term signals and outlive the parent's exit.
        signal(SIGINT, SIG_IGN);
        signal(SIGTERM, SIG_IGN);

        // Detach stdio so the harness capturing our output gets EOF when the parent exits;
        // otherwise this child holds the inherited pipe write-end open and the reader blocks.
        int devnull = open("/dev/null", O_RDWR);
        if(devnull >= 0)
        {
            dup2(devnull, STDIN_FILENO);
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            if(devnull > STDERR_FILENO) close(devnull);
        }

        // Exit once the original parent is gone (robust to subreapers that adopt orphans),
        // with a hard cap as a safety net.
        const pid_t orig_parent = getppid();
        for(int i = 0; i < 30 && getppid() == orig_parent; ++i)
        {
            sleep(1);
        }

        _exit(0);
    }

    // Parent does a bounded amount of GPU work (so there is data to flush), then returns
    // normally. It deliberately does NOT wait for or kill the child.
    float* d_buf = nullptr;
    HIP_CHECK(hipMalloc(&d_buf, 1024 * sizeof(float)));
    roctxRangePush(str_join("parent_pid_", getpid()).c_str());
    for(int iter = 0; iter < 20; ++iter)
    {
        roctxRangePush(str_join("parent_iter_", iter).c_str());
        test_kernel<<<4, 256>>>(d_buf, 1024);
        roctxRangePop();
        usleep(10000);
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_buf));
    roctxRangePop();

    emit_roctx_marker("exit_marker parent lingering-child ppid:%d pid:%d", getppid(), getpid());
    fprintf(stderr,
            "Parent PID=%d: normal exit with lingering child %d: clean exit\n",
            getpid(),
            static_cast<int>(pid));
    return 0;  // NORMAL exit -> rocprofv3 finalize must complete without hanging
}

}  // namespace
// ============================================================================
// Main
// ============================================================================

int
main(int argc, char** argv)
{
    const char* mode                = "--single-process";
    bool        app_handles_signals = true;

    for(int i = 1; i < argc; i++)
    {
        if(std::strcmp(argv[i], "--single-process") == 0 || std::strcmp(argv[i], "--fork") == 0 ||
           std::strcmp(argv[i], "--fork-exec") == 0 || std::strcmp(argv[i], "--spawn") == 0 ||
           std::strcmp(argv[i], "--lingering-child") == 0 ||
           std::strcmp(argv[i], "--fork-after-work") == 0 ||
           std::strcmp(argv[i], "--fork-prepare-marker") == 0 ||
           std::strcmp(argv[i], "--fork-prepare-detach") == 0 ||
           std::strcmp(argv[i], "--startup-fork-check") == 0 ||
           std::strcmp(argv[i], "--fork-child-transition-abort") == 0 ||
           std::strcmp(argv[i], "--detach-output-retry") == 0 ||
           std::strcmp(argv[i], "--detach-output-retry-reattach") == 0 ||
           std::strcmp(argv[i], "--fork-during-finalization") == 0 ||
           std::strcmp(argv[i], "--fork-during-detach") == 0 ||
           std::strcmp(argv[i], "--fork-during-detach-sync") == 0 ||
           std::strcmp(argv[i], "--fork-during-sdk-finalization") == 0 ||
           std::strcmp(argv[i], "--fork-during-tool-finalization") == 0 ||
           std::strcmp(argv[i], "--fork-exec-fresh") == 0 ||
           std::strcmp(argv[i], "--detach-finalization-handoff") == 0 ||
           std::strcmp(argv[i], "--detach-finalization-handoff-sync") == 0)
        {
            mode = argv[i];
        }
        else if(std::strcmp(argv[i], "--app-signal-handler") == 0)
        {
            app_handles_signals = true;
        }
        else if(std::strcmp(argv[i], "--no-app-signal-handler") == 0)
        {
            app_handles_signals = false;
        }
    }

    fprintf(stderr,
            "signal-handler-test: mode=%s app_handles_signals=%d PID=%d\n",
            mode,
            static_cast<int>(app_handles_signals),
            getpid());

    // Normal-exit regression mode (no signal); independent of app_handles_signals.
    if(std::strcmp(mode, "--lingering-child") == 0) return mode_lingering_child();
    if(std::strcmp(mode, "--fork-after-work") == 0) return mode_fork_after_work();
    if(std::strcmp(mode, "--fork-prepare-marker") == 0) return mode_fork_prepare_marker();
    if(std::strcmp(mode, "--fork-prepare-detach") == 0)
    {
        emit_roctx_marker("exit_marker parent fork-prepare-detach");
        const auto child = fork();
        if(child == 0) std::exit(0);
        int status = 0;
        if(child < 0 || waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
           WEXITSTATUS(status) != 0)
            throw std::runtime_error{"prepare-detach child failed"};
        return 0;
    }
    if(std::strcmp(mode, "--startup-fork-check") == 0)
    {
        emit_roctx_marker("exit_marker parent startup-fork-check");
        return 0;
    }
    if(std::strcmp(mode, "--fork-child-transition-abort") == 0)
    {
        struct rlimit no_core
        {
            0, 0
        };
        setrlimit(RLIMIT_CORE, &no_core);
        signal(SIGABRT, SIG_DFL);
        const auto child = fork();
        if(child == 0) std::_Exit(96);  // the earlier child handler must abort before fork returns
        int status = 0;
        if(child < 0 || waitpid(child, &status, 0) != child || !WIFSIGNALED(status) ||
           WTERMSIG(status) != SIGABRT)
            throw std::runtime_error{"pre-rebuild abort did not preserve child signal behavior"};
        emit_roctx_marker("exit_marker parent fork-child-transition-abort");
        return 0;
    }
    if(std::strncmp(mode, "--detach-output-retry", std::strlen("--detach-output-retry")) == 0)
    {
        emit_roctx_marker("exit_marker parent %s", mode + 2);
        auto detach =
            reinterpret_cast<rocprofiler_status_t (*)()>(dlsym(RTLD_DEFAULT, "rocprofiler_detach"));
        if(!detach || detach() != ROCPROFILER_STATUS_SUCCESS)
            throw std::runtime_error{"async detach retry regression failed to detach"};
        if(std::strcmp(mode, "--detach-output-retry-reattach") == 0)
        {
            auto attach = reinterpret_cast<rocprofiler_status_t (*)()>(
                dlsym(RTLD_DEFAULT, "rocprofiler_attach"));
            if(!attach || attach() != ROCPROFILER_STATUS_SUCCESS)
                throw std::runtime_error{"retry regression failed to reattach"};
            emit_roctx_marker("exit_marker reattached session after failed detach");
        }
        return 0;  // normal finalization joins the failed output thread and retries preserved input
    }
    if(std::strcmp(mode, "--fork-exec-fresh") == 0)
    {
        using query_type = rocprofiler_status_t (*)(int*);
        auto initialized =
            reinterpret_cast<query_type>(dlsym(RTLD_DEFAULT, "rocprofiler_is_initialized"));
        auto finalized =
            reinterpret_cast<query_type>(dlsym(RTLD_DEFAULT, "rocprofiler_is_finalized"));
        int init = 0;
        int fini = 1;
        if(!initialized || !finalized || initialized(&init) != ROCPROFILER_STATUS_SUCCESS ||
           finalized(&fini) != ROCPROFILER_STATUS_SUCCESS || init != 1 || fini != 0)
            throw std::runtime_error{"exec did not initialize a fresh SDK session"};
        emit_roctx_marker("exit_marker exec fresh initialization");
        return 0;
    }
    if(std::strncmp(mode,
                    "--detach-finalization-handoff",
                    std::strlen("--detach-finalization-handoff")) == 0 ||
       std::strncmp(mode, "--fork-during-", 14) == 0)
        return mode_fork_during_lifecycle(mode + 2, argv[0]);

    if(app_handles_signals)
    {
        if(std::strcmp(mode, "--single-process") == 0) return mode_good_single_process();
        if(std::strcmp(mode, "--fork") == 0) return mode_good_fork();
        if(std::strcmp(mode, "--fork-exec") == 0) return mode_good_fork_exec(argv[0]);
        if(std::strcmp(mode, "--spawn") == 0) return mode_good_spawn(argv[0]);
    }
    else
    {
        if(std::strcmp(mode, "--single-process") == 0) return mode_bad_single_process();
        if(std::strcmp(mode, "--fork") == 0) return mode_bad_fork();
        if(std::strcmp(mode, "--fork-exec") == 0) return mode_bad_fork_exec(argv[0]);
        if(std::strcmp(mode, "--spawn") == 0) return mode_bad_spawn(argv[0]);
    }

    fprintf(stderr, "Unknown mode: %s\n", mode);
    return 1;
}
