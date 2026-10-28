// Smoke test running bench_streaming at reduced scale (the full run is too
// heavy for default ctest).
//   full run: bench_streaming --threads=256 --freq=5000 --duration=30

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace
{

bool
perf_open_available()
{
    struct perf_event_attr a
    {};
    a.size           = sizeof(a);
    a.type           = PERF_TYPE_SOFTWARE;
    a.config         = PERF_COUNT_SW_CPU_CLOCK;
    a.disabled       = 1;
    a.exclude_kernel = 1;
    a.exclude_hv     = 1;
    const int fd     = static_cast<int>(
        ::syscall(SYS_perf_event_open, &a, 0, -1, -1, PERF_FLAG_FD_CLOEXEC));
    if(fd < 0) return false;
    ::close(fd);
    return true;
}

}  // namespace

#ifndef SAMPLING_BENCH_STREAMING_PATH
#    define SAMPLING_BENCH_STREAMING_PATH "./bench_streaming"
#endif

TEST(BenchStreamingSmoke, RunsAtReducedScaleAndExitsClean)
{
    if(!perf_open_available()) GTEST_SKIP() << "perf_event_open not permitted";

    const std::string cmd = std::string(SAMPLING_BENCH_STREAMING_PATH) +
                            " --duration=3 --threads=8 --freq=500 --null-sink";
    const int rc = std::system(cmd.c_str());
    ASSERT_NE(rc, -1) << "std::system(" << cmd << ") failed to spawn";
    // Exit code 0 = PASS on overhead + size budgets; 1 = budget FAIL
    // but bench still ran cleanly. Either means the binary is healthy;
    // we just want non-crash + valid output here.
    const int exit_code = WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
    EXPECT_GE(exit_code, 0) << "bench_streaming did not exit normally";
    EXPECT_LE(exit_code, 1) << "bench_streaming exit code beyond budget-fail range";
}
