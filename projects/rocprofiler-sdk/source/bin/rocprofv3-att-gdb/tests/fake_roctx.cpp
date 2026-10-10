// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static int
record(const char* name, const char* gate_variable)
{
    if(auto* file = std::fopen(std::getenv("ATT_TEST_CALLS"), "a"))
    {
        std::fprintf(file, "%s\n", name);
        std::fclose(file);
    }
    if(auto* gate = std::getenv(gate_variable))
    {
        std::printf("WAITING_%s\n", name);
        std::fflush(stdout);
        while(access(gate, F_OK) != 0)
            usleep(1000);
    }
    return 0;
}
extern "C" int roctxProfilerResume(uint64_t) { return record("start", "ATT_TEST_RESUME_GATE"); }
extern "C" int roctxProfilerPause(uint64_t) { return record("stop", "ATT_TEST_PAUSE_GATE"); }
extern "C" int
rocprofiler_is_initialized(int* ready)
{
    *ready = std::getenv("ATT_TEST_NOT_READY") ? 0 : 1;
    return 0;
}
