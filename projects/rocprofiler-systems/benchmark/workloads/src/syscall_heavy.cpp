// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "bench_hooks.h"
#include "common.hpp"
#include <cstdint>

#include <fcntl.h>
#include <unistd.h>

namespace
{
std::uint64_t
run(const workload::args& a, int fd)
{
    std::uint64_t sum = a.seed;
    std::uint64_t buf = 0;
    for(std::uint64_t i = 0; i < a.iters; ++i)
    {
        const ssize_t n = pread(fd, &buf, sizeof(buf), 0);  // /dev/zero: always zeros
        sum             = workload::mix(sum, static_cast<std::uint64_t>(n) + buf + i);
    }
    return sum;
}
}  // namespace

int
main(int argc, char** argv)
{
    const auto a = workload::parse_args(
        argc, argv, { 0, 15000000, 1 }, "syscall_heavy",
        "iters = number of pread(/dev/zero) syscalls; --size is ignored.");
    const int fd = open("/dev/zero", O_RDONLY);
    if(fd < 0) return 1;
    bench_hook(BENCH_HOOK_WORK_BEGIN);
    const std::uint64_t sum = run(a, fd);
    bench_hook(BENCH_HOOK_WORK_END);
    close(fd);
    workload::print_checksum(sum);
    return 0;
}
