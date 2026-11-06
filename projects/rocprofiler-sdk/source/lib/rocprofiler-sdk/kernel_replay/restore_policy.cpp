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

#include "lib/rocprofiler-sdk/kernel_replay/restore_policy.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/synchronized.hpp"

#include <fmt/format.h>

#include <limits>
#include <map>
#include <string>
#include <utility>

namespace rocprofiler
{
namespace kernel_replay
{
namespace restore_policy
{
namespace
{
// Below these sizes a full restore takes about as long as the hashing launches. Pinned memory
// restores over PCIe, so hashing pays off much sooner there.
constexpr size_t min_gpu_bytes    = size_t{64} << 20;
constexpr size_t min_pinned_bytes = size_t{1} << 20;

// Whether restoring everything moves fewer bytes than `mode`, for a kernel that changes as many of
// its T captured bytes per pass as its last hashed restore found, c. A full restore reads and
// writes all of them, 2(N - 1)T over N passes. Hashing reads all of them and copies the changed
// ones back: (N - 1)(T + 2c) when it hashes before every pass, which is more once c reaches T/2;
// and 2T + (2N - 3)c when it hashes once, reuses the list and checks the rest at the end, which is
// more once c reaches (1 - 1/(2N - 3))T. That tends to T as N grows, so without a pass count,
// hashing once is never worse.
bool
full_moves_less(restore_mode mode, uint64_t passes, const history_t& history)
{
    if(history.tiles == 0) return false;
    const auto tiles   = uint64_t{history.tiles};
    const auto changed = uint64_t{history.changed};
    if(mode == restore_mode::hash_every_pass) return 2 * changed >= tiles;
    if(passes == 0) return false;
    return (2 * passes - 3) * changed >= 2 * (passes - 2) * tiles;
}

using history_map_t = std::map<std::pair<uint64_t, uint64_t>, history_t>;

common::Synchronized<history_map_t>&
histories()
{
    // An application can launch kernels from its own static destructors, so this is never
    // destroyed.
    static auto* value = new common::Synchronized<history_map_t>{};
    return *value;
}
}  // namespace

std::optional<restore_mode>
parse_mode(std::string_view value)
{
    if(value.empty() || value == "auto") return restore_mode::automatic;
    if(value == "full") return restore_mode::full;
    if(value == "hash-every-pass") return restore_mode::hash_every_pass;
    if(value == "hash-once") return restore_mode::hash_once;
    return std::nullopt;
}

std::string_view
name_of(restore_mode mode)
{
    switch(mode)
    {
        case restore_mode::automatic: return "auto";
        case restore_mode::full: return "full";
        case restore_mode::hash_every_pass: return "hash-every-pass";
        case restore_mode::hash_once: return "hash-once";
    }
    return "auto";
}

restore_mode
requested_mode()
{
    static const auto mode = [] {
        const auto value =
            common::get_env("ROCPROFILER_INTERNAL_REPLAY_RESTORE_FORCE_MODE", std::string{"auto"});
        const auto parsed = parse_mode(value);
        if(!parsed)
            ROCP_WARNING << fmt::format("ROCPROFILER_INTERNAL_REPLAY_RESTORE_FORCE_MODE={} is not "
                                        "auto, full, hash-every-pass or hash-once; using auto",
                                        value);
        return parsed.value_or(restore_mode::automatic);
    }();
    return mode;
}

bool
may_hash(const inputs_t& inputs)
{
    auto largest         = inputs;
    largest.gpu_bytes    = std::numeric_limits<size_t>::max();
    largest.pinned_bytes = std::numeric_limits<size_t>::max();
    return choose(largest) != restore_mode::full;
}

restore_mode
choose(const inputs_t& inputs)
{
    if(inputs.requested == restore_mode::full || inputs.total_passes == 1)
        return restore_mode::full;
    if(inputs.requested != restore_mode::automatic) return inputs.requested;

    // Such a kernel may not write the same tiles every pass.
    if(inputs.history.unlisted) return restore_mode::full;
    // With two or three passes, hashing before each costs no more than reusing a list and checking
    // it, and does not assume the kernel writes the same tiles every pass.
    const auto mode = (inputs.total_passes != 0 && inputs.total_passes <= 3)
                          ? restore_mode::hash_every_pass
                          : restore_mode::hash_once;
    // A pinned restore crosses PCIe, so hashing pays off there even when most tiles change.
    if(inputs.pinned_bytes < min_pinned_bytes &&
       (inputs.gpu_bytes < min_gpu_bytes ||
        full_moves_less(mode, inputs.total_passes, inputs.history)))
        return restore_mode::full;
    return mode;
}

history_t
history_of(const kernel_key_t& key)
{
    return histories().rlock([&](const auto& map) {
        const auto itr = map.find({key.agent, key.kernel_object});
        return itr == map.end() ? history_t{} : itr->second;
    });
}

void
record(const kernel_key_t& key, uint32_t tiles, uint32_t changed, uint32_t unlisted_changed)
{
    histories().wlock([&](auto& map) {
        auto& history    = map[{key.agent, key.kernel_object}];
        history.tiles    = tiles;
        history.changed  = changed;
        history.unlisted = history.unlisted || unlisted_changed > 0;
    });
}
}  // namespace restore_policy
}  // namespace kernel_replay
}  // namespace rocprofiler
