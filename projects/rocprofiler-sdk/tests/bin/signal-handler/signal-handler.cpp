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
// TODO: --raw-fork (_Fork without exec) is not supported. _Fork skips
// pthread_atfork handlers, leaving stale profiler state. Extremely rare in practice.

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <cstdarg>
#include <cstring>
#include <sstream>
#include <string>

#include <atomic>
#include <stdexcept>

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
           std::strcmp(argv[i], "--fork-exec") == 0 || std::strcmp(argv[i], "--spawn") == 0)
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
