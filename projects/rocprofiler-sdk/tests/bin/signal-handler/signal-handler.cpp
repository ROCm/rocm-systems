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
// _Fork skips pthread_atfork handlers and is outside rocprofv3's contract. vfork is used only for
// the immediate exec/_exit spawn path below.

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdarg>
#include <cstring>
#include <sstream>
#include <string>

#include <atomic>
#include <stdexcept>
#include <vector>

namespace
{
std::atomic<bool> g_shutdown{false};

using profiler_status_query_t = int (*)(int*);

profiler_status_query_t g_profiler_is_initialized = nullptr;
profiler_status_query_t g_profiler_is_finalized   = nullptr;

void
resolve_profiler_status_queries()
{
    g_profiler_is_initialized = reinterpret_cast<profiler_status_query_t>(
        dlsym(RTLD_DEFAULT, "rocprofiler_is_initialized"));
    g_profiler_is_finalized =
        reinterpret_cast<profiler_status_query_t>(dlsym(RTLD_DEFAULT, "rocprofiler_is_finalized"));
    if(!g_profiler_is_initialized || !g_profiler_is_finalized)
        throw std::runtime_error{"profiler status queries are unavailable"};
}

bool
profiler_is_disabled_in_child()
{
    int init = -1;
    int fini = -1;
    return g_profiler_is_initialized && g_profiler_is_finalized &&
           g_profiler_is_initialized(&init) == 0 && g_profiler_is_finalized(&fini) == 0 &&
           init == 0 && fini == 1;
}

struct waiting_child
{
    pid_t pid      = -1;
    int   write_fd = -1;
};

std::vector<waiting_child>
start_waiting_plain_fork_children(int count)
{
    resolve_profiler_status_queries();

    auto children = std::vector<waiting_child>{};
    children.reserve(count);

    for(int i = 0; i < count; ++i)
    {
        int control_pipe[2] = {-1, -1};
        if(pipe(control_pipe) != 0)
            throw std::runtime_error{"plain-fork control pipe creation failed"};

        auto pid = fork();
        if(pid == 0)
        {
            close(control_pipe[1]);
            signal(SIGINT, SIG_IGN);

            char       command  = 0;
            const auto received = read(control_pipe[0], &command, sizeof(command));
            close(control_pipe[0]);

            const bool valid = profiler_is_disabled_in_child() &&
                               received == static_cast<ssize_t>(sizeof(command)) && command == 'q';
            _exit(valid ? 0 : 78);
        }
        if(pid < 0)
        {
            close(control_pipe[0]);
            close(control_pipe[1]);
            throw std::runtime_error{"plain-fork child creation failed"};
        }

        close(control_pipe[0]);
        children.emplace_back(waiting_child{pid, control_pipe[1]});
    }

    return children;
}

void
release_waiting_plain_fork_children(std::vector<waiting_child>& children)
{
    for(auto& child : children)
    {
        const char command = 'q';
        if(write(child.write_fd, &command, sizeof(command)) !=
           static_cast<ssize_t>(sizeof(command)))
            throw std::runtime_error{"plain-fork child release failed"};
        close(child.write_fd);
        child.write_fd = -1;
    }

    for(const auto& child : children)
    {
        int status = 0;
        if(waitpid(child.pid, &status, 0) != child.pid || !WIFEXITED(status) ||
           WEXITSTATUS(status) != 0)
            throw std::runtime_error{"plain-fork child contract check failed"};
    }
}

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
mode_good_fork()
{
    fprintf(stderr, "Mode: good/fork, PID=%d\n", getpid());

    auto children = start_waiting_plain_fork_children(2);

    struct sigaction sa = {};
    sa.sa_handler       = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);

    run_kernels("parent");
    release_waiting_plain_fork_children(children);

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

    auto children = start_waiting_plain_fork_children(2);
    release_waiting_plain_fork_children(children);

    run_kernels("parent");

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
           std::strcmp(argv[i], "--lingering-child") == 0)
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
