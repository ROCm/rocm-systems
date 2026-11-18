/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime_api.h>

namespace hipFileTest {

/// @brief Work-items in each workgroup of the producer kernel
constexpr unsigned int PRODUCER_KERNEL_WORK_ITEMS{256};

/// @brief State of the current iteration's hipFileWrite, as a running kernel sees it
enum class WriteState : uint32_t {
    /// @brief hipFileWrite has not returned yet
    InProgress,
    /// @brief hipFileWrite returned
    Finished,
};

/// @brief Gets the value a producer stores in a word of the buffer
///
/// Every store of the buffer has its own generation, so a value says both
/// which word it belongs to and which store wrote it.
///
/// @param generation The store's generation
/// @param word Index of the 64-bit word in the buffer
///
/// @return The generation in the upper 32 bits and the word's index in the lower 32 bits
constexpr uint64_t
patternValue(uint32_t generation, size_t word)
{
    return (uint64_t{generation} << 32) | static_cast<uint32_t>(word);
}

/// @brief Arguments of the producer kernel
struct ProducerKernelArgs {
    /// @brief GPU buffer the kernel stores a generation in, which hipFileWrite then writes to the file
    uint64_t *buffer;
    /// @brief Number of 64-bit words in the buffer
    size_t word_count;
    /// @brief Generation to store
    uint32_t generation;
    /// @brief One flag per workgroup, in fine-grained host memory, or nullptr. If set, each
    /// work-item releases its stores at system scope, and each workgroup then stores 1 in its
    /// element, which tells the host the workgroup's share of the buffer holds the generation.
    uint32_t *ready;
    /// @brief WriteState of hipFileWrite, in fine-grained host memory, or nullptr. If set, ready
    /// and timed_out must be too, and each workgroup keeps running until hipFileWrite returns.
    uint32_t *write_state;
    /// @brief Wall clock ticks to wait for hipFileWrite before giving up
    uint64_t timeout_ticks;
    /// @brief One flag per workgroup, in fine-grained host memory, or nullptr. Each workgroup
    /// stores 1 in its element if it gave up waiting for hipFileWrite to return.
    uint32_t *timed_out;
};

/// @brief Launches the producer kernel without waiting for it
///
/// Each workgroup stores the generation in a contiguous, equal share of the buffer.
///
/// @param args The kernel's arguments
/// @param workgroups Number of workgroups. If args.write_state is set, they must all fit on the
///        GPU at once.
/// @param stream Stream to launch the kernel on
///
/// @return The launch's error, or hipSuccess. Any error earlier HIP calls left
///         for hipGetLastError is cleared and not returned.
hipError_t launchProducerKernel(const ProducerKernelArgs &args, unsigned int workgroups, hipStream_t stream);

}
