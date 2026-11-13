/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "GpuCountProbe.hpp"

#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <system_error>

#include <hip/hip_runtime_api.h>

namespace RcclUnitTesting
{

namespace
{

// Names the write end of the result pipe in the re-exec'd image. Its presence
// is what distinguishes a probe child from an ordinary test process.
constexpr char const* kProbeFdEnvVar = "RCCL_UT_GPU_COUNT_PROBE_FD";

// Upper bound on how long the parent waits for the probe. The child does an
// execv() and one HIP call, so this is a backstop against a wedged runtime
// rather than an expected duration (observed cost is a few hundred ms):
// exceeding it yields kGpuCountProbeFailed instead of hanging the test run.
constexpr std::chrono::milliseconds kProbeTimeout{30'000};

// Entrypoint for the re-exec'd image: reports the device count over the
// inherited pipe and _exit()s. Returns normally in every other process.
//
// Runs before main() (see the file-scope initializer at the bottom of this
// file) because the probe child re-execs the test binary with no arguments;
// were it to fall through, it would run the whole test suite instead.
//
// Deliberately avoids iostreams and anything else with its own static state:
// during static initialization other translation units' globals may not be
// constructed yet. Raw write()/_exit() have no such dependency.
void serviceProbeRequestIfChild()
{
    char const* fdStr = std::getenv(kProbeFdEnvVar);
    if(!fdStr)
        return;

    // Unset first: a HIP call below must not be able to spawn a nested probe.
    const std::string_view fdText{fdStr};
    unsetenv(kProbeFdEnvVar);

    // from_chars rather than atoi: a malformed value must not silently parse
    // as 0 and send the result to stdin.
    int fd = -1;
    if(std::from_chars(fdText.data(), fdText.data() + fdText.size(), fd).ec != std::errc{}
       || fd < 0)
        _exit(EXIT_FAILURE);

    int count = 0;
    if(hipGetDeviceCount(&count) != hipSuccess || count < 0)
        _exit(EXIT_FAILURE); // Parent sees a short read and reports failure.

    char   bytes[sizeof(count)];
    std::memcpy(bytes, &count, sizeof(count));

    size_t written = 0;
    while(written < sizeof(bytes))
    {
        const ssize_t n = write(fd, bytes + written, sizeof(bytes) - written);
        if(n < 0 && errno == EINTR)
            continue;
        if(n <= 0)
            _exit(EXIT_FAILURE);
        written += static_cast<size_t>(n);
    }

    close(fd);
    _exit(EXIT_SUCCESS);
}

// Reads exactly sizeof(int) bytes from fd, giving up once kProbeTimeout has
// elapsed. The deadline is computed once and each poll() gets only the time
// remaining, so EINTR and partial reads cannot extend the total wait: passing
// the full timeout to every poll() would make it unbounded under repeated
// interruption, which is the hang this timeout exists to prevent.
bool readCountWithTimeout(int fd, int* out)
{
    using Clock = std::chrono::steady_clock;
    const auto deadline = Clock::now() + kProbeTimeout;

    char   buf[sizeof(int)];
    size_t haveRead = 0;

    while(haveRead < sizeof(buf))
    {
        const auto remaining
            = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now());
        if(remaining.count() <= 0)
            return false;

        pollfd    pfd{fd, POLLIN, 0};
        const int ready = poll(&pfd, 1, static_cast<int>(remaining.count()));
        if(ready < 0 && errno == EINTR)
            continue;
        if(ready <= 0)
            return false; // Timed out, or poll() failed.

        const ssize_t n = read(fd, buf + haveRead, sizeof(buf) - haveRead);
        if(n < 0 && errno == EINTR)
            continue;
        if(n <= 0)
            return false; // Child exited without writing a full count.
        haveRead += static_cast<size_t>(n);
    }

    std::memcpy(out, buf, sizeof(buf));
    return true;
}

// Reaps pid, killing it first if it is still running. Called once the parent
// has its answer (or has given up), so the probe never becomes a zombie and a
// wedged child never outlives the call.
void reap(pid_t pid, bool killFirst)
{
    if(killFirst)
        kill(pid, SIGKILL);

    int status = 0;
    while(waitpid(pid, &status, 0) < 0 && errno == EINTR)
    {
    }
}

} // namespace

int gpuCountViaFreshImage()
{
    int pipeFds[2];
    if(pipe(pipeFds) != 0)
        return kGpuCountProbeFailed;

    fflush(nullptr);
    const pid_t pid = fork();
    if(pid < 0)
    {
        close(pipeFds[0]);
        close(pipeFds[1]);
        return kGpuCountProbeFailed;
    }

    if(pid == 0)
    {
        // Child: hand the write end to the re-exec'd image through the
        // environment, then replace this image so the HIP call runs in a
        // fresh, fork-free process. Dropping the profiler tool hooks keeps the
        // tiny probe from being traced itself.
        //
        // Between fork() and execv() only async-signal-safe work is sound --
        // no allocation, no locks, nothing that can throw -- hence the C-level
        // calls here. EnvVars::DetectGpuInfo (EnvVars.cpp:211-224) does the
        // same for the same reason.
        close(pipeFds[0]);
        char      fdStr[16];
        const auto written = std::to_chars(fdStr, fdStr + sizeof(fdStr) - 1, pipeFds[1]);
        if(written.ec != std::errc{})
            _exit(EXIT_FAILURE);
        *written.ptr = '\0';

        setenv(kProbeFdEnvVar, fdStr, 1);
        unsetenv("ROCP_TOOL_LIBRARIES");
        unsetenv("HSA_TOOLS_LIB");

        char  self[] = "/proc/self/exe";
        char* argv[] = {self, nullptr};
        execv(self, argv);
        _exit(EXIT_FAILURE); // execv() only returns on failure.
    }

    close(pipeFds[1]);
    int        count = 0;
    const bool gotIt = readCountWithTimeout(pipeFds[0], &count);
    close(pipeFds[0]);
    reap(pid, !gotIt);

    return gotIt ? count : kGpuCountProbeFailed;
}

namespace
{
// Installs the probe entrypoint before main(). A file-scope initializer rather
// than a call from each main(): the test binaries that link this TU have
// several distinct mains (common/main.cpp, common/main_altrsmi.cpp,
// host/main.cpp, ...), and a probe child that reached any of them would run
// the suite instead of answering.
struct ProbeEntrypointInstaller
{
    ProbeEntrypointInstaller() { serviceProbeRequestIfChild(); }
};

const ProbeEntrypointInstaller g_probeEntrypointInstaller;
} // namespace

} // namespace RcclUnitTesting
