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

#include "lib/rocprofiler-sdk/hsa/memory_tracker.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
// GPU memory snapshot/restore for kernel replay. Ports Kerncap's chunked-staging snap
// (intellikit/kerncap, kerncap.hip:765-903) and implements restore() per design doc Section 4.3,
// correcting the three ROCScope rocmhook.cc bugs called out in Section 3.3:
//   1. never overwrite the GPU address with the staging pointer (kept as separate fields);
//   2. recompute checksums fresh on every restore() call (no stale diff reuse);
//   3. hash CURRENT GPU memory (not the on-disk snapshot) to compute the dirty set.
namespace memory_snapshot
{
// Per-allocation snapshot record. gpu_addr and the staging buffer are deliberately distinct
// (Section 3.3 bug 1).
struct mem_block
{
    void*                 gpu_addr = nullptr;  // live GPU allocation base pointer
    size_t                size     = 0;        // allocation size in bytes
    bool                  is_vmem  = false;
    std::string           path;                // on-disk snapshot path
    std::vector<uint32_t> checksums1;          // Murmur3 hashes of 1 MB pages (pre-kernel)
};

// Module-variable (HSA_SYMBOL_KIND_VARIABLE) snapshot record. Kernels launched via constant-memory
// shims (e.g. Kokkos hip_parallel_launch_constant_memory) read state that lives in the executable's
// data segment, not a tracked allocation -- without restoring it, replay reads stale/NULL pointers.
// Restored by full overwrite since module variables are small (design Section 4.3, Kerncap
// kerncap.hip:905-1141).
struct var_block
{
    uint64_t    address = 0;
    size_t      size    = 0;
    std::string path;  // on-disk snapshot path
};

class Snapshot
{
public:
    Snapshot();
    ~Snapshot();

    Snapshot(const Snapshot&)            = delete;
    Snapshot& operator=(const Snapshot&) = delete;

    // DMA every tracked allocation GPU->disk via a single reused staging buffer, recording
    // per-page checksums into each block's checksums1. Returns number of regions captured.
    size_t snap();

    // For each allocation: DMA current GPU memory->staging, hash pages into a fresh local
    // checksums2, diff against checksums1, and write back only dirty pages from the disk snapshot.
    // Returns number of dirty pages written back.
    size_t restore();

    // Remove on-disk snapshot files. Called automatically by the destructor.
    void cleanup();

    bool empty() const { return blocks_.empty(); }

private:
    // Walk loaded executables (code_object::iterate_loaded_code_objects), filter
    // HSA_SYMBOL_KIND_VARIABLE, and DMA each variable's device bytes to disk. Must run at snap
    // time, not executable-load time -- constant memory may not be populated at load.
    size_t snap_module_variables(std::vector<char>& staging);
    // Full-overwrite each captured module variable from its disk blob.
    size_t restore_module_variables(std::vector<char>& staging);

    std::string            output_dir_;
    std::vector<mem_block> blocks_;
    std::vector<var_block> vars_;
    size_t                 chunk_bytes_ = 0;  // staging buffer size (Kerncap default 64 MiB)
};
}  // namespace memory_snapshot
}  // namespace hsa
}  // namespace rocprofiler
