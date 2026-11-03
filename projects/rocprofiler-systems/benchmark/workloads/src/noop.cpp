// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "bench_hooks.h"
#include "common.hpp"

int
main(int argc, char** argv)
{
    workload::parse_args(argc, argv, {}, "noop",
                         "Does nothing; measures process start/stop cost.");
    bench_hook(BENCH_HOOK_WORK_BEGIN);
    bench_hook(BENCH_HOOK_WORK_END);
    workload::print_checksum(0);
    return 0;
}
