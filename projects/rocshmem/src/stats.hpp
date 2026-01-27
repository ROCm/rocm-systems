/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_STATS_HPP_
#define LIBRARY_SRC_STATS_HPP_

#include <time.h>

#include <atomic>

#include "rocshmem/rocshmem.hpp"
#include "util.hpp"

namespace rocshmem {

enum rocshmem_stats {
  NUM_PUT = 0,
  NUM_PUT_NBI,
  NUM_P,
  NUM_GET,
  NUM_G,
  NUM_GET_NBI,
  NUM_FENCE,
  NUM_QUIET,
  NUM_PE_QUIET,
  NUM_TO_ALL,
  NUM_BARRIER_ALL,
  NUM_BARRIER_ALL_WAVE,
  NUM_BARRIER_ALL_WG,
  NUM_BARRIER,
  NUM_BARRIER_WAVE,
  NUM_BARRIER_WG,
  NUM_WAIT_UNTIL,
  NUM_WAIT_UNTIL_ANY,
  NUM_WAIT_UNTIL_ALL,
  NUM_WAIT_UNTIL_SOME,
  NUM_WAIT_UNTIL_ALL_VECTOR,
  NUM_WAIT_UNTIL_ANY_VECTOR,
  NUM_WAIT_UNTIL_SOME_VECTOR,
  NUM_FINALIZE,
  NUM_MSG_COAL,
  NUM_ATOMIC_FADD,
  NUM_ATOMIC_FCSWAP,
  NUM_ATOMIC_FINC,
  NUM_ATOMIC_FETCH,
  NUM_ATOMIC_ADD,
  NUM_ATOMIC_SET,
  NUM_ATOMIC_SWAP,
  NUM_ATOMIC_FETCH_AND,
  NUM_ATOMIC_AND,
  NUM_ATOMIC_FETCH_OR,
  NUM_ATOMIC_OR,
  NUM_ATOMIC_FETCH_XOR,
  NUM_ATOMIC_XOR,
  NUM_ATOMIC_CSWAP,
  NUM_ATOMIC_INC,
  NUM_TEST,
  NUM_SHMEM_PTR,
  NUM_SYNC_ALL,
  NUM_SYNC_ALL_WAVE,
  NUM_SYNC_ALL_WG,
  NUM_SYNC,
  NUM_SYNC_WAVE,
  NUM_SYNC_WG,
  NUM_BROADCAST,
  NUM_PUT_WG,
  NUM_PUT_NBI_WG,
  NUM_GET_WG,
  NUM_GET_NBI_WG,
  NUM_PUT_WAVE,
  NUM_PUT_NBI_WAVE,
  NUM_GET_WAVE,
  NUM_GET_NBI_WAVE,
  NUM_CREATE,
  NUM_ALLTOALL,
  NUM_FCOLLECT,
  NUM_PUT_SIGNAL,
  NUM_PUT_SIGNAL_WG,
  NUM_PUT_SIGNAL_WAVE,
  NUM_PUT_SIGNAL_NBI,
  NUM_PUT_SIGNAL_NBI_WG,
  NUM_PUT_SIGNAL_NBI_WAVE,
  NUM_STATS
};

enum rocshmem_host_stats {
  NUM_HOST_PUT = 0,
  NUM_HOST_PUT_NBI,
  NUM_HOST_P,
  NUM_HOST_GET,
  NUM_HOST_G,
  NUM_HOST_GET_NBI,
  NUM_HOST_FENCE,
  NUM_HOST_QUIET,
  NUM_HOST_TO_ALL,
  NUM_HOST_BARRIER_ALL,
  NUM_HOST_WAIT_UNTIL,
  NUM_HOST_WAIT_UNTIL_ANY,
  NUM_HOST_WAIT_UNTIL_ALL,
  NUM_HOST_WAIT_UNTIL_SOME,
  NUM_HOST_WAIT_UNTIL_ALL_VECTOR,
  NUM_HOST_WAIT_UNTIL_ANY_VECTOR,
  NUM_HOST_WAIT_UNTIL_SOME_VECTOR,
  NUM_HOST_FINALIZE,
  NUM_HOST_ATOMIC_FADD,
  NUM_HOST_ATOMIC_FCSWAP,
  NUM_HOST_ATOMIC_FINC,
  NUM_HOST_ATOMIC_FETCH,
  NUM_HOST_ATOMIC_ADD,
  NUM_HOST_ATOMIC_SET,
  NUM_HOST_ATOMIC_SWAP,
  NUM_HOST_ATOMIC_FETCH_AND,
  NUM_HOST_ATOMIC_AND,
  NUM_HOST_ATOMIC_FETCH_OR,
  NUM_HOST_ATOMIC_OR,
  NUM_HOST_ATOMIC_FETCH_XOR,
  NUM_HOST_ATOMIC_XOR,
  NUM_HOST_ATOMIC_CSWAP,
  NUM_HOST_ATOMIC_INC,
  NUM_HOST_TEST,
  NUM_HOST_SHMEM_PTR,
  NUM_HOST_SYNC_ALL,
  NUM_HOST_BROADCAST,
  NUM_HOST_ALLTOALL,
  NUM_HOST_STATS
};

typedef unsigned long long StatType;

typedef std::atomic_ullong AtomicStatType;

template <int I>
class Stats {
  StatType stats[I] = {0};

 public:
  __device__ uint64_t startTimer() const { return wall_clock64(); }

  __device__ void endTimer(uint64_t start, int index) {
    incStat(index, wall_clock64() - start);
  }

  __device__ void incStat(int index, int value = 1) {
    atomicAdd(&stats[index], value);
  }

  __device__ void accumulateStats(const Stats<I> &otherStats) {
    for (int i = 0; i < I; i++) incStat(i, otherStats.getStat(i));
  }

  __host__ __device__ void resetStats() {
    memset(&stats, 0, sizeof(StatType) * I);
  }

  __host__ __device__ StatType getStat(int index) const { return stats[index]; }
};

template <int I>
class HostStats {
  AtomicStatType stats[I] = {};

 public:
  __host__ uint64_t startTimer() const { return wtime(); }

  __host__ void endTimer(uint64_t start, int index) {
    incStat(index, wtime() - start);
  }

  __host__ void incStat(int index, int value = 1) { stats[index] += value; }

  __host__ void accumulateStats(const HostStats<I> &otherStats) {
    for (int i = 0; i < I; i++) incStat(i, otherStats.getStat(i));
  }

  __host__ void resetStats() {
    /* Using loop to ensure atomic writes */
    for (int i = 0; i < I; i++) stats[i] = 0;
  }

  __host__ StatType getStat(int index) const { return stats[index].load(); }
 private:
  double wtime(void) {
    double wt;
    struct timespec tp;

    (void) clock_gettime(CLOCK_MONOTONIC, &tp);
    wt = (static_cast<double>(tp.tv_nsec))/1.0e+9;
    wt += static_cast<double>(tp.tv_sec);
    return wt;
  }
};

// clang-format off
NOWARN(-Wunused-parameter,
template <int I>
class NullStats {
 public:
  __host__ __device__ uint64_t startTimer() const { return 0; }
  __host__ __device__ void endTimer(uint64_t start, int index) {}
  __host__ __device__ void incStat(int index, int value = 1) {}
  __host__ __device__ void accumulateStats(const NullStats<I> &otherStats) {}
  __host__ __device__ void resetStats() {}
  __host__ __device__ StatType getStat(int index) const { return 0; }
};
)
// clang-format on

#ifdef PROFILE
typedef Stats<NUM_STATS> ROCStats;
typedef HostStats<NUM_HOST_STATS> ROCHostStats;
#else
typedef NullStats<NUM_STATS> ROCStats;
typedef NullStats<NUM_STATS> ROCHostStats;
#endif

}  // namespace rocshmem

#endif  // LIBRARY_SRC_STATS_HPP_
