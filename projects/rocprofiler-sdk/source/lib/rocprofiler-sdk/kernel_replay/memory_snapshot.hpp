// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstddef>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
// Minimal save/restore of device memory for kernel replay.
//
// snap() copies every tracked device allocation into host memory; restore() copies it back. This
// keeps each replay pass running against identical inputs. The implementation is deliberately
// simple: a full copy of each region (no dirty-page diffing) of directly-allocated device memory.
namespace memory_snapshot
{
// Saved copy of a single device allocation.
struct mem_block
{
    void*             gpu_addr = nullptr;  // live device allocation base pointer
    std::vector<char> host_copy;          // pre-kernel contents held in host memory
};

class Snapshot
{
public:
    Snapshot()  = default;
    ~Snapshot() = default;

    Snapshot(const Snapshot&) = delete;
    Snapshot& operator=(const Snapshot&) = delete;

    // Copy every tracked allocation device->host. Returns number of regions captured.
    size_t snap();

    // Copy each saved region host->device. Returns number of regions restored.
    size_t restore();

    bool empty() const { return blocks_.empty(); }

private:
    std::vector<mem_block> blocks_;
};
}  // namespace memory_snapshot
}  // namespace hsa
}  // namespace rocprofiler
