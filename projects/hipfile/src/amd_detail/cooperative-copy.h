/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <sys/types.h>

namespace hipFile {

struct Hip;
struct Sys;
enum class IoType;

// One CPU worker issues copy commands. One GPU workgroup consumes them in stream order.
// The CPU owns a slot until it publishes requested. The GPU returns ownership by clearing it.
struct CopySlot {
    static constexpr size_t Capacity = 16 * 1024 * 1024;
    uint32_t                requested{};
    size_t                  offset{};
    size_t                  count{};
    alignas(64) std::array<unsigned char, Capacity> data;
};

// The CPU publishes ticket only on fallback. A matching kernel publishes resident before
// polling the slots. At termination the CPU requests stop, the GPU acknowledges stopped,
// and the CPU publishes finished after releasing the file/buffer and recording the byte count.
struct CopyExchange {
    uint64_t                ticket{};
    uint64_t                resident{};
    uint32_t                stop{};
    uint32_t                stopped{};
    uint32_t                finished{};
    std::array<CopySlot, 2> slots;
};

const void *cooperativeCopyKernel();

/// @internal Pinned host storage and its device alias, reused in registered stream order.
class CooperativeCopy {
public:
    explicit CooperativeCopy(Hip &hip);
    void    launch(hipStream_t stream, void *buffer, bool read, uint64_t ticket);
    void    prepare(uint64_t ticket);
    bool    resident(uint64_t ticket) const;
    ssize_t transfer(Sys &sys, IoType type, int fd, size_t size, int64_t file_offset, size_t buffer_offset);
    void    stop(IoType type);
    void    finish();

private:
    struct Deleter {
        void operator()(CopyExchange *ptr) const noexcept;
    };
    std::unique_ptr<CopyExchange, Deleter> host;
    CopyExchange                          *device;
};

}
