/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// The kernel that reproducers 10 and 11 keep running across many transfers,
// and the host's side of its conversation with that kernel. Unlike
// repro-common.h, this header is part of what fails: read it with either
// program.
//
// The kernel is launched once and stays alive. Each of its work-items is a
// watcher, with one 32-bit word of the buffer to read, on a 64-byte line no
// other watcher reads. The watchers' words are spread over the buffer in
// index order. In every iteration:
//
//   host                                  each watcher
//   ----                                  ------------
//   stores Shared::started                acquires at system scope and reads
//                                         its word: the old data
//   waits for Workgroup::primed           reads its word over and over, with
//   puts the new data in the buffer       the acquire under test before each
//   stores the flag                       read, until it reads the new data
//   waits for Workgroup::finished         or has seen the flag: the read
//                                         after the flag must return the
//                                         new data
//
// This is hipfileread-gpu-watcher-visibility with everything but that removed.

#pragma once

#include "repro-common.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace repro {

/// @brief Most workgroups the kernel can have
constexpr unsigned int MAX_WORKGROUPS{512};
/// @brief Value of Shared::started that makes the kernel return
constexpr uint32_t STOP{UINT32_MAX};
/// @brief Times a watcher that read old data after the flag reads its word again
constexpr uint32_t REREADS{10000};

/// @brief Scope of the acquire a watcher does before each read of its word
enum Acquire : int {
    /// @brief No acquire
    ACQUIRE_NONE,
    /// @brief What other work-items of the watcher's workgroup wrote
    ACQUIRE_WORKGROUP,
    /// @brief What anything on the same GPU wrote
    ACQUIRE_AGENT,
    /// @brief What anything wrote: other GPUs, the CPU, a drive. This is the
    /// scope HIP's memory model asks for here.
    ACQUIRE_SYSTEM,
};

/// @brief The data iteration i puts in every word of the buffer, for a program
/// whose data repeats every so many iterations: 1 to generations
///
/// Iteration i finds dataOf(i - 1, generations) in the buffer.
constexpr uint32_t
dataOf(uint32_t iteration, uint32_t generations)
{
    return iteration % generations + 1;
}

/// @brief Distance, in 32-bit words, between the words of two neighbouring
/// watchers that spreads the watchers over the buffer
///
/// The distance is odd, on purpose. With the words a power of two apart,
/// many watchers read the new data without any acquire (README.md has the
/// counts). We infer that such words push each other out of the GPU's
/// caches, and that a watcher whose line has left the cache reads VRAM again.
constexpr size_t
spacingOf(size_t word_count, size_t watchers)
{
    return (word_count / watchers - 1) | 1;
}

/// @brief What one workgroup tells the host
struct Workgroup {
    /// @brief The last iteration in which every watcher has read the old data
    uint32_t primed;
    /// @brief The last iteration every watcher has finished
    uint32_t finished;
    /// @brief Watchers that, at the start of the iteration, after a
    /// system-scope acquire, did not read the data of the iteration before
    uint32_t unseen;
    /// @brief Watchers that read something other than the new data after the flag
    uint32_t stale;
    /// @brief Stale watchers that then read the new data within REREADS more reads
    uint32_t late;
    /// @brief The lowest index, within the whole kernel, of a stale watcher
    uint32_t lowest;
    /// @brief What one stale watcher read
    uint32_t observed;
};

/// @brief What the host and the kernel tell each other, in host memory from
/// hipHostMalloc with hipHostMallocCoherent
struct Shared {
    /// @brief The iteration the host has started, or STOP
    alignas(64) uint32_t started;
    /// @brief The flag: the iteration whose data the host has put in the buffer
    alignas(64) uint32_t arrived;
    /// @brief One element per workgroup
    alignas(64) Workgroup workgroups[MAX_WORKGROUPS];
};

/// @brief The acquire under test
///
/// What each scope compiles to depends on the GPU; README.md has the table.
static __device__ void
acquire(int scope)
{
    switch (scope) {
        case ACQUIRE_NONE:
            // Emits no instruction on any GPU. It only keeps the compiler from
            // reading the word once and reusing what it read.
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            break;
        case ACQUIRE_WORKGROUP:
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
            break;
        case ACQUIRE_AGENT:
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            break;
        default:
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
            break;
    }
}

/// @brief The long-running kernel
///
/// @param buffer The buffer the host puts new data in
/// @param spacing Distance between two neighbouring watchers' words, in
///        32-bit words: at least 16, so that no two watchers share a line
/// @param generations Number of iterations after which the data repeats
/// @param shared What the host and the kernel tell each other
/// @param flag Where the host stores the iteration once its data is in the
///        buffer: &shared->arrived, or a word of VRAM
/// @param scope The Acquire each watcher does before each read of its word
static __global__ void
watch(const uint32_t *buffer, size_t spacing, uint32_t generations, Shared *shared, const uint32_t *flag,
      int scope)
{
    __shared__ uint32_t unseen_count;
    __shared__ uint32_t stale_count;
    __shared__ uint32_t late_count;
    __shared__ uint32_t lowest;
    __shared__ uint32_t observed;

    const uint32_t  watcher{blockIdx.x * blockDim.x + threadIdx.x};
    const uint32_t *word{buffer + watcher * spacing};
    Workgroup      &workgroup{shared->workgroups[blockIdx.x]};

    for (uint32_t iteration{1};; iteration++) {
        const uint32_t old_data{dataOf(iteration - 1, generations)};
        const uint32_t new_data{dataOf(iteration, generations)};

        // Wait for the host to start the iteration. These loads bypass the
        // GPU's caches, and invalidate nothing.
        for (;;) {
            const uint32_t started{
                __hip_atomic_load(&shared->started, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)};
            if (started == STOP) {
                return;
            }
            if (started == iteration) {
                break;
            }
            __builtin_amdgcn_s_sleep(127);
        }

        // Read the old data after a system-scope acquire. No watcher is
        // between the flag and its read now, as the host has waited for all
        // of them, so this acquire cannot hide what another watcher's own
        // acquire misses.
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
        const bool unseen{*word != old_data};
        if (threadIdx.x == 0) {
            unseen_count = 0;
            stale_count  = 0;
            late_count   = 0;
            lowest       = UINT32_MAX;
        }
        __syncthreads();
        if (unseen) {
            atomicAdd(&unseen_count, 1U);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            workgroup.unseen = unseen_count;
            __hip_atomic_store(&workgroup.primed, iteration, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }

        // Watch the word until the flag says the new data is in the buffer.
        // The word is loaded after the flag, so it must hold the new data
        // once the flag does.
        uint32_t value{};
        for (;;) {
            const uint32_t arrived{__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)};
            acquire(scope);
            value = *word;
            // A watcher that reads the new data stops watching, too. The
            // loop has to use what it reads: the compiler moves a read whose
            // result the loop ignores to after the loop.
            if (arrived == iteration || value == new_data) {
                break;
            }
        }

        if (value != new_data) {
            // Stale. Does the new data turn up with the same acquire, a
            // little later?
            observed = value;
            atomicAdd(&stale_count, 1U);
            atomicMin(&lowest, watcher);
            for (uint32_t reread{0}; reread < REREADS && value != new_data; reread++) {
                acquire(scope);
                value = *word;
            }
            if (value == new_data) {
                atomicAdd(&late_count, 1U);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            workgroup.stale    = stale_count;
            workgroup.late     = late_count;
            workgroup.lowest   = lowest;
            workgroup.observed = observed;
            __hip_atomic_store(&workgroup.finished, iteration, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

/// @brief Stores a value the kernel is waiting for in host memory
inline void
tell(uint32_t *where, uint32_t value)
{
    std::atomic_ref<uint32_t>{*where}.store(value, std::memory_order_release);
}

/// @brief Waits until every workgroup has stored an iteration in one of its members
///
/// @param member &Workgroup::primed or &Workgroup::finished
inline void
waitForWorkgroups(Shared *shared, unsigned int workgroups, uint32_t Workgroup::*member, uint32_t iteration)
{
    const auto give_up{std::chrono::steady_clock::now() + std::chrono::seconds{60}};
    for (unsigned int i{0}; i < workgroups; i++) {
        while (std::atomic_ref<uint32_t>{shared->workgroups[i].*member}.load(std::memory_order_acquire) !=
               iteration) {
            if (std::chrono::steady_clock::now() > give_up) {
                // _Exit, as HIP's cleanup at exit could wait for the kernel
                fprintf(stderr, "error: the kernel has not answered for 60 s\n");
                _Exit(2);
            }
        }
    }
}

/// @brief What the watchers of every workgroup found, added up
struct Found {
    /// @brief Sum of Workgroup::unseen
    uint32_t unseen{0};
    /// @brief Sum of Workgroup::stale
    uint32_t stale{0};
    /// @brief Sum of Workgroup::late
    uint32_t late{0};
    /// @brief Lowest Workgroup::lowest
    uint32_t lowest{UINT32_MAX};
    /// @brief Workgroup::observed of one workgroup with a stale watcher
    uint32_t observed{0};
};

/// @brief Adds up what the workgroups reported for the iteration they last
/// primed (Found::unseen) and last finished (the rest)
inline Found
found(const Shared *shared, unsigned int workgroups)
{
    Found sum;
    for (unsigned int i{0}; i < workgroups; i++) {
        const Workgroup &workgroup{shared->workgroups[i]};
        sum.unseen += workgroup.unseen;
        sum.stale += workgroup.stale;
        sum.late += workgroup.late;
        if (workgroup.stale != 0) {
            sum.lowest   = workgroup.lowest < sum.lowest ? workgroup.lowest : sum.lowest;
            sum.observed = workgroup.observed;
        }
    }
    return sum;
}

/// @brief Stops the program on a GPU where a fallback hipFileRead with a
/// kernel running has hung the amdgpu driver, unless the user allows it
///
/// 09-fallback-read-during-kernel hung RX 9070 (gfx1201) nodes until a
/// reboot. Programs 10 and 11 have not been run on that GPU.
inline void
refuseFallbackWhereItHung()
{
    int             device{};
    hipDeviceProp_t properties{};
    HIP_OK(hipGetDevice(&device));
    HIP_OK(hipGetDeviceProperties(&properties, device));
    if (std::string{properties.gcnArchName}.rfind("gfx12", 0) == 0 && !getenv("REPRO_MAY_HANG_DRIVER")) {
        die("a fallback hipFileRead with a kernel running has hung the amdgpu driver on gfx1201; set "
            "REPRO_MAY_HANG_DRIVER=1 to run it");
    }
}

}
