/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime_api.h>

namespace hipFileTest {

/// @brief Threads in each workgroup of the visibility kernel
constexpr unsigned int VISIBILITY_KERNEL_THREADS{256};

/// @brief State of the current iteration's hipFileRead, as a running kernel sees it
enum class ReadState : uint32_t {
    /// @brief hipFileRead has not returned yet
    InProgress,
    /// @brief hipFileRead returned and read the whole buffer
    Succeeded,
    /// @brief hipFileRead returned an error or read less than the whole buffer
    Failed,
};

/// @brief What one workgroup of the visibility kernel found in its part of the buffer
struct VisibilityResult {
    /// @brief Values that did not hold the previous pattern before hipFileRead
    uint32_t previous_mismatches;
    /// @brief Values that still held the previous pattern after hipFileRead returned
    uint32_t stale;
    /// @brief Values that held neither pattern after hipFileRead returned
    uint32_t unexpected;
    /// @brief Nonzero if the workgroup gave up waiting for hipFileRead to return
    uint32_t timed_out;
    /// @brief The lowest stale or unexpected value's word index in the upper 32 bits and the value in
    /// the lower 32 bits, or UINT64_MAX if there was none
    uint64_t first_bad;
};

/// @brief Arguments of the visibility kernel
///
/// The buffer holds one of two patterns: word i of pattern p is i + p * word_count, so the base of
/// pattern p is p * word_count.
struct VisibilityKernelArgs {
    /// @brief GPU buffer that hipFileRead fills
    const uint32_t *buffer;
    /// @brief Number of 32-bit words in the buffer
    size_t word_count;
    /// @brief Base of the pattern the buffer holds before hipFileRead
    uint32_t previous_base;
    /// @brief Base of the pattern hipFileRead reads into the buffer
    uint32_t current_base;
    /// @brief Read the buffer and check for the previous pattern first, which also brings it into the
    /// GPU's caches
    bool check_previous;
    /// @brief Check for the current pattern last
    bool check_current;
    /// @brief ReadState of hipFileRead, in fine-grained host memory, or nullptr if hipFileRead has
    /// already returned. If set, ready must be too, and each workgroup waits for hipFileRead to
    /// return before checking for the current pattern.
    uint32_t *read_state;
    /// @brief One flag per workgroup, in fine-grained host memory, or nullptr. If set, each
    /// workgroup stores 1 in its element once it has read its share of the buffer to check for
    /// the previous pattern, which tells the host the kernel no longer needs the old data.
    uint32_t *ready;
    /// @brief Wall clock ticks to wait for hipFileRead before giving up
    uint64_t timeout_ticks;
    /// @brief One result per workgroup, in GPU memory
    VisibilityResult *results;
};

/// @brief Launches the visibility kernel without waiting for it
///
/// Each workgroup checks a contiguous, equal share of the buffer.
///
/// @param args The kernel's arguments
/// @param workgroups Number of workgroups. If args.read_state is set, they must all fit on the GPU at
///        once.
/// @param stream Stream to launch the kernel on
///
/// @return The launch's error, or hipSuccess. Any error earlier HIP calls left
///         for hipGetLastError is cleared and not returned.
hipError_t launchVisibilityKernel(const VisibilityKernelArgs &args, unsigned int workgroups,
                                  hipStream_t stream);

}
