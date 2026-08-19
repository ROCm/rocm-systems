// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <ctime>

namespace
{
constexpr long SLEEP_NS    = 50000000L;  // 50 ms
constexpr long SHORT_BY_NS = 5000000L;   // 5 ms
constexpr long NS_PER_SEC  = 1000000000L;

long
elapsed_ns(timespec start, timespec stop)
{
    return (stop.tv_sec - start.tv_sec) * NS_PER_SEC + (stop.tv_nsec - start.tv_nsec);
}
}  // namespace

// Uses nanosleep directly rather than std::this_thread::sleep_for, which
// retries on EINTR internally and would hide the interruptions being counted.
int
main(int argc, const char* const* argv)
{
    const int iterations =
        (argc > 1) ? static_cast<int>(std::strtol(argv[1], nullptr, 10)) : 60;

    int interrupted = 0;
    int short_sleep = 0;

    for(int i = 0; i < iterations; ++i)
    {
        const timespec request = { 0, SLEEP_NS };
        timespec       start{};
        timespec       stop{};
        // NOLINTBEGIN(misc-include-cleaner)
        clock_gettime(CLOCK_MONOTONIC, &start);
        const int status = nanosleep(&request, nullptr);
        clock_gettime(CLOCK_MONOTONIC, &stop);
        // NOLINTEND(misc-include-cleaner)
        if(status != 0 && errno == EINTR)
        {
            ++interrupted;
        }
        if(elapsed_ns(start, stop) < SLEEP_NS - SHORT_BY_NS)
        {
            ++short_sleep;
        }
    }

    printf("sleep_interrupts: iterations=%d interrupted=%d short_sleeps=%d\n", iterations,
           interrupted, short_sleep);
    return 0;
}
