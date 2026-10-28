#pragma once

// bench_common.hpp
//
// Shared helpers between bench_overhead and bench_streaming:
//   - hw_counters: self-attached PMU counter pair (cycles + instructions)
//   - read_counter: scaled read of a perf_event_open counter
//   - ns_diff: timespec subtraction in nanoseconds
//
// The synthetic workload chain (leaf_compute -> ... -> workload_step
// + deeper frames) is re-exported from examples/common/workload.hpp
// so every demo + bench drives the same call graph.
//
// Internal to the bench/ scope; not installed.

#include <cerrno>
#include <cstdint>
#include <ctime>

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "workload.hpp"

namespace bench_common
{

using cs_examples::hot_loop;
using cs_examples::leaf_compute;
using cs_examples::matmul_inner;
using cs_examples::matmul_outer;
using cs_examples::workload_step;

// Self-attached HW counter group: cycles + instructions for THIS thread.
// Returned struct holds the two fds; ok=false on any setup failure.
struct hw_counters
{
    int  fd_cycles = -1;
    int  fd_insns  = -1;
    bool ok        = false;
};

inline hw_counters
open_self_counters() noexcept
{
    hw_counters h{};
    struct perf_event_attr cy{};
    cy.size           = sizeof(cy);
    cy.type           = PERF_TYPE_HARDWARE;
    cy.config         = PERF_COUNT_HW_CPU_CYCLES;
    cy.disabled       = 1;
    cy.exclude_kernel = 1;
    cy.exclude_hv     = 1;
    cy.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

    h.fd_cycles = static_cast<int>(
        ::syscall(SYS_perf_event_open, &cy, 0, -1, -1, PERF_FLAG_FD_CLOEXEC));
    if(h.fd_cycles < 0) return h;

    struct perf_event_attr in{};
    in.size           = sizeof(in);
    in.type           = PERF_TYPE_HARDWARE;
    in.config         = PERF_COUNT_HW_INSTRUCTIONS;
    in.disabled       = 0;
    in.exclude_kernel = 1;
    in.exclude_hv     = 1;
    in.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

    h.fd_insns = static_cast<int>(
        ::syscall(SYS_perf_event_open, &in, 0, -1, h.fd_cycles, PERF_FLAG_FD_CLOEXEC));
    if(h.fd_insns < 0)
    {
        ::close(h.fd_cycles);
        h.fd_cycles = -1;
        return h;
    }

    h.ok = true;
    return h;
}

/// Single scaled counter value (handles multiplexing via te/tr ratio).
inline std::uint64_t
read_counter(int fd) noexcept
{
    struct {
        std::uint64_t v, te, tr;
    } buf{};
    if(::read(fd, &buf, sizeof(buf)) != sizeof(buf)) return 0;
    if(buf.tr != 0 && buf.te != buf.tr)
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
        using wide_uint = unsigned __int128;
#pragma GCC diagnostic pop
        return static_cast<std::uint64_t>(static_cast<wide_uint>(buf.v) * buf.te / buf.tr);
    }
    return buf.v;
}

/// timespec subtraction yielding nanoseconds as a double.
inline double
ns_diff(const timespec& a, const timespec& b) noexcept
{
    return static_cast<double>(a.tv_sec - b.tv_sec) * 1e9 +
           static_cast<double>(a.tv_nsec - b.tv_nsec);
}

}  // namespace bench_common
