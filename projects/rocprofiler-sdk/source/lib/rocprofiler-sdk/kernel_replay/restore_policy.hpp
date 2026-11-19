// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cstdint>
#include <optional>
#include <string_view>

namespace rocprofiler
{
namespace kernel_replay
{
// How a replay restores memory between passes. A hashed restore compares each captured tile's
// digest with the one taken at capture and copies back only the tiles that changed.
namespace restore_policy
{
enum class restore_mode : uint8_t
{
    automatic,        // requested: let choose() decide
    full,             // copy everything back before every pass
    hash_every_pass,  // hash before every pass
    hash_once,        // hash before the second pass, reuse its list, and check the rest at the end
};

// What earlier dispatches of a kernel on an agent found.
struct history_t
{
    uint32_t tiles    = 0;      // hashed by the last hashed restore, 0 before the first
    uint32_t changed  = 0;      // of those, the ones it found changed
    bool     unlisted = false;  // tiles changed off a reused list
};

struct kernel_key_t
{
    uint64_t agent         = 0;
    uint64_t kernel_object = 0;
};

struct inputs_t
{
    restore_mode requested    = restore_mode::automatic;
    uint64_t     total_passes = 0;  // 0 when the tool decides as the replay goes
    history_t    history      = {};
    size_t       gpu_bytes    = 0;
    size_t       pinned_bytes = 0;
};

// ROCPROFILER_INTERNAL_REPLAY_RESTORE_FORCE_MODE values: auto, full, hash-every-pass and
// hash-once.
std::optional<restore_mode>
parse_mode(std::string_view value);

std::string_view
name_of(restore_mode mode);

// The mode ROCPROFILER_INTERNAL_REPLAY_RESTORE_FORCE_MODE requests, read once.
restore_mode
requested_mode();

// Whether the capture should keep digests: choose() may still pick hashing once the snapshot's
// size is known.
bool
may_hash(const inputs_t& inputs);

restore_mode
choose(const inputs_t& inputs);

history_t
history_of(const kernel_key_t& key);

// What a dispatch's hashed restores found, for the kernel's next dispatch.
void
record(const kernel_key_t& key, uint32_t tiles, uint32_t changed, uint32_t unlisted_changed);
}  // namespace restore_policy
}  // namespace kernel_replay
}  // namespace rocprofiler
