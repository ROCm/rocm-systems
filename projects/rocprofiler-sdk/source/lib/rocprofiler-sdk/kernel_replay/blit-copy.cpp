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

#include "lib/rocprofiler-sdk/kernel_replay/blit-copy.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/blit-copy-kernel.hpp"

#include <fmt/format.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace kernel_replay
{
namespace blit
{
namespace
{
using kernel_abi::capture_tiles_args_t;
using kernel_abi::changed_count_slots;
using kernel_abi::check_tiles_args_t;
using kernel_abi::chunks_for;
using kernel_abi::copy_descriptor_t;
using kernel_abi::copy_tiles_args_t;
using kernel_abi::copy_workgroups_per_cu;
using kernel_abi::count_tiles_args_t;
using kernel_abi::counter_count;
using kernel_abi::expand_tiles_args_t;
using kernel_abi::restore_changed_args_t;
using kernel_abi::restore_listed_args_t;
using kernel_abi::scan_tiles_args_t;
using kernel_abi::tiles_for;
using kernel_abi::unlisted_counter;
using kernel_abi::workgroup_size;

// A copy runs as four dispatches, in this order: three build the tile list on the GPU from the
// region descriptors, and the last copies it, digesting the tiles when it is a capture that keeps
// them. The restore kernels then work from that capture's tile list, one dispatch each.
enum kernel_index : size_t
{
    count_tiles = 0,
    scan_tiles,
    expand_tiles,
    copy_tiles,
    capture_tiles,
    restore_changed_tiles,
    restore_listed_tiles,
    check_tiles,
    kernel_count
};

constexpr auto kernel_names = std::array<const char*, kernel_count>{
    "__kernel_replay_count_tiles",
    "__kernel_replay_scan_tiles",
    "__kernel_replay_expand_tiles",
    "__kernel_replay_copy_tiles",
    "__kernel_replay_capture_tiles",
    "__kernel_replay_restore_changed_tiles",
    "__kernel_replay_restore_listed_tiles",
    "__kernel_replay_check_tiles",
};

constexpr auto kernel_arg_sizes = std::array<size_t, kernel_count>{
    sizeof(count_tiles_args_t),
    sizeof(scan_tiles_args_t),
    sizeof(expand_tiles_args_t),
    sizeof(copy_tiles_args_t),
    sizeof(capture_tiles_args_t),
    sizeof(restore_changed_args_t),
    sizeof(restore_listed_args_t),
    sizeof(check_tiles_args_t),
};

constexpr size_t max_dispatches = 4;

struct kernel_info_t
{
    uint64_t object               = 0;
    uint32_t private_segment_size = 0;
    uint32_t group_segment_size   = 0;
    uint32_t kernarg_size         = 0;
};

// Where the tile-list kernels keep their intermediate results, and a capture that keeps digests
// keeps them, in one device allocation.
struct workspace_layout_t
{
    size_t first_tiles   = 0;
    size_t chunk_totals  = 0;
    size_t tile_count    = 0;
    size_t counters      = 0;
    size_t tiles         = 0;
    size_t digests       = 0;
    size_t changed_flags = 0;
    size_t changed       = 0;
    size_t size          = 0;
};

struct kernel_state_t
{
    int                                     file = -1;
    hsa_code_object_reader_t                reader{};
    hsa_executable_t                        executable{};
    std::array<kernel_info_t, kernel_count> kernels{};
    hsa_signal_t                            completion{};
    void*                                   kernarg           = nullptr;
    size_t                                  kernarg_capacity  = 0;
    bool                                    kernarg_on_device = false;
    std::vector<void*>                      retired_kernargs{};
    void*                                   workspace          = nullptr;
    size_t                                  workspace_capacity = 0;
    std::atomic<bool>                       resource_in_use{false};

    // The workspace's tile list is a capture's, kept for its restores, possibly with digests.
    // Restores alternate between the changed-list counts, and changed_slot is the last list's.
    bool               kept             = false;
    bool               digests          = false;
    uint32_t           kept_tiles       = 0;
    workspace_layout_t kept_layout      = {};
    uint32_t           changed_restores = 0;
    uint32_t           changed_slot     = 0;

    // The host reads the counters once per replay. Reading device memory it can map takes one
    // PCIe round trip, where hsa_memory_copy takes tens of microseconds, so they live there when
    // the CPU can map device memory and in the workspace otherwise.
    void*    mapped_counters = nullptr;
    bool     counters_placed = false;
    uint64_t counters        = 0;
};

constexpr size_t
align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

workspace_layout_t
workspace_layout(uint32_t descriptor_count, uint32_t tile_count, bool digests)
{
    constexpr auto alignment = size_t{256};
    static_assert(counter_count * sizeof(uint32_t) <= alignment);

    auto layout         = workspace_layout_t{};
    layout.chunk_totals = align_up(size_t{descriptor_count} * sizeof(uint32_t), alignment);
    layout.tile_count =
        layout.chunk_totals +
        align_up(size_t{chunks_for(descriptor_count)} * sizeof(uint32_t), alignment);
    layout.counters = layout.tile_count + alignment;
    layout.tiles    = layout.counters + alignment;
    layout.size     = layout.tiles + size_t{tile_count} * sizeof(copy_descriptor_t);
    if(!digests) return layout;

    layout.digests = align_up(layout.size, alignment);
    layout.changed_flags =
        align_up(layout.digests + size_t{tile_count} * sizeof(uint64_t), alignment);
    layout.changed = align_up(layout.changed_flags + tile_count, alignment);
    layout.size    = layout.changed + size_t{tile_count} * sizeof(uint32_t);
    return layout;
}

using state_map_t = std::unordered_map<uint64_t, kernel_state_t*>;

common::Synchronized<state_map_t>&
states()
{
    // Keep one code object and reusable packet allocation per agent for process lifetime. HSA may
    // be gone during static destruction, so deliberately do not destroy these objects there.
    static auto* value = new common::Synchronized<state_map_t>{};
    return *value;
}

std::string
find_code_object(const hsa::AgentCache& agent)
{
    const auto filename = fmt::format("{}_kernel_replay_blit.hsaco", agent.name());
    for(const auto* directory :
        {ROCPROFILER_KERNEL_REPLAY_BLIT_BUILD_DIR, ROCPROFILER_KERNEL_REPLAY_BLIT_INSTALL_DIR})
    {
        auto path = fmt::format("{}/{}", directory, filename);
        if(access(path.c_str(), R_OK) == 0) return path;
    }
    return {};
}

hsa_status_t
get_symbol_info(hsa_executable_symbol_t symbol, hsa_executable_symbol_info_t attribute, void* value)
{
    return hsa::get_core_table()->hsa_executable_symbol_get_info_fn(symbol, attribute, value);
}

hsa_status_t
initialize_kernel(hsa_executable_t executable,
                  hsa_agent_t      agent,
                  const char*      name,
                  kernel_info_t&   info)
{
    auto* core   = hsa::get_core_table();
    auto  symbol = hsa_executable_symbol_t{};
    auto  status = core->hsa_executable_get_symbol_by_name_fn(executable, name, &agent, &symbol);
    if(status != HSA_STATUS_SUCCESS)
    {
        const auto descriptor_name = fmt::format("{}.kd", name);
        status                     = core->hsa_executable_get_symbol_by_name_fn(
            executable, descriptor_name.c_str(), &agent, &symbol);
    }
    if(status != HSA_STATUS_SUCCESS) return status;

    status = get_symbol_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &info.object);
    if(status != HSA_STATUS_SUCCESS) return status;

    status = get_symbol_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &info.private_segment_size);
    if(status != HSA_STATUS_SUCCESS) return status;
    status = get_symbol_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &info.group_segment_size);
    if(status != HSA_STATUS_SUCCESS) return status;
    return get_symbol_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &info.kernarg_size);
}

hsa_status_t
initialize_state(const hsa::AgentCache& agent, kernel_state_t& state)
{
    auto* core = hsa::get_core_table();
    auto* ext  = hsa::get_amd_ext_table();
    if(!core || !ext || !ext->hsa_amd_signal_create_fn) return HSA_STATUS_ERROR;

    const auto path = find_code_object(agent);
    if(path.empty())
    {
        ROCP_ERROR << fmt::format("kernel replay blit: no code object for agent '{}'",
                                  agent.name());
        return HSA_STATUS_ERROR;
    }

    state.file = open(path.c_str(), O_RDONLY);
    if(state.file < 0) return HSA_STATUS_ERROR;

    auto status = core->hsa_code_object_reader_create_from_file_fn(state.file, &state.reader);
    if(status != HSA_STATUS_SUCCESS) return status;

    status = core->hsa_executable_create_alt_fn(
        HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &state.executable);
    if(status != HSA_STATUS_SUCCESS) return status;

    status = core->hsa_executable_load_agent_code_object_fn(
        state.executable, agent.get_hsa_agent(), state.reader, nullptr, nullptr);
    if(status != HSA_STATUS_SUCCESS) return status;

    status = core->hsa_executable_freeze_fn(state.executable, nullptr);
    if(status != HSA_STATUS_SUCCESS) return status;

    const auto hsa_agent = agent.get_hsa_agent();
    for(size_t index = 0; index < kernel_count; ++index)
    {
        auto& kernel = state.kernels.at(index);
        status = initialize_kernel(state.executable, hsa_agent, kernel_names.at(index), kernel);
        if(status != HSA_STATUS_SUCCESS) return status;
        // The kernels are dispatched with only their explicit arguments; a larger segment would
        // mean they read implicit arguments that are never written.
        if(kernel.kernarg_size != kernel_arg_sizes.at(index))
        {
            ROCP_ERROR << fmt::format("kernel replay blit: {} takes {}B of arguments, expected {}B",
                                      kernel_names.at(index),
                                      kernel.kernarg_size,
                                      kernel_arg_sizes.at(index));
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        }
    }

    status = ext->hsa_amd_signal_create_fn(1, 0, nullptr, 0, &state.completion);
    if(status != HSA_STATUS_SUCCESS) return status;

    ROCP_INFO << fmt::format("kernel replay blit: loaded '{}' for agent '{}'", path, agent.name());
    return HSA_STATUS_SUCCESS;
}

// Releases what initialize_state() created before it failed.
void
release_partial_state(kernel_state_t& state)
{
    auto* core = hsa::get_core_table();
    if(core && state.executable.handle != 0) core->hsa_executable_destroy_fn(state.executable);
    if(core && state.reader.handle != 0) core->hsa_code_object_reader_destroy_fn(state.reader);
    if(state.file >= 0) close(state.file);
}

kernel_state_t*
get_state(const hsa::AgentCache& agent)
{
    const auto handle = agent.get_hsa_agent().handle;
    const auto cached = states().rlock([&](const auto& map) -> std::optional<kernel_state_t*> {
        auto itr = map.find(handle);
        if(itr == map.end()) return std::nullopt;
        return itr->second;
    });
    if(cached) return *cached;

    return states().wlock([&](auto& map) -> kernel_state_t* {
        // An agent whose load failed stays null, so the failure is reported and paid for once.
        auto [itr, inserted] = map.emplace(handle, nullptr);
        if(!inserted) return itr->second;

        auto candidate = std::make_unique<kernel_state_t>();
        if(initialize_state(agent, *candidate) != HSA_STATUS_SUCCESS)
        {
            release_partial_state(*candidate);
            return nullptr;
        }
        itr->second = candidate.release();
        return itr->second;
    });
}

// Small device allocations are carved out of larger blocks, so hostBaseAddress is set even when
// the CPU cannot map the block. The block's host-access flag is what decides it.
bool
host_writable(void* ptr)
{
#if HSA_AMD_INTERFACE_VERSION_MAJOR > 1 ||                                                         \
    (HSA_AMD_INTERFACE_VERSION_MAJOR == 1 && HSA_AMD_INTERFACE_VERSION_MINOR >= 18)
    auto* ext = hsa::get_amd_ext_table();
    if(!ext || !ext->hsa_amd_pointer_info_fn) return false;
    auto info = hsa_amd_pointer_info_t{};
    info.size = sizeof(info);
    if(ext->hsa_amd_pointer_info_fn(ptr, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS)
        return false;
    return (info.alloc_flags & HSA_AMD_POINTER_INFO_ALLOC_FLAG_HOST_ACCESS) != 0;
#else
    (void) ptr;
    return false;
#endif
}

// Device memory the CPU can map, or null when it cannot map device memory.
void*
allocate_mapped(const hsa::AgentCache& agent, size_t size)
{
    auto* ext = hsa::get_amd_ext_table();
    void* ptr = nullptr;
    if(ext->hsa_amd_memory_pool_allocate_fn(
           agent.gpu_pool(), size, HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, &ptr) != HSA_STATUS_SUCCESS)
        return nullptr;
    if(host_writable(ptr)) return ptr;
    ext->hsa_amd_memory_pool_free_fn(ptr);
    return nullptr;
}

// Every wave reads the blit's arguments and copy descriptors, and reading them from host memory
// makes each restore depend on PCIe read latency. Keep them in device memory the CPU can write,
// and fall back to the host kernarg pool when it cannot. Returns the buffer and whether it is in
// device memory.
std::pair<void*, bool>
allocate_kernarg(const hsa::AgentCache& agent, size_t size)
{
    if(auto* mapped = allocate_mapped(agent, size)) return {mapped, true};

    auto* ext = hsa::get_amd_ext_table();
    void* ptr = nullptr;
    if(ext->hsa_amd_memory_pool_allocate_fn(agent.kernarg_pool(), size, 0, &ptr) !=
       HSA_STATUS_SUCCESS)
        return {nullptr, false};
    const auto hsa_agent = agent.get_hsa_agent();
    if(ext->hsa_amd_agents_allow_access_fn(1, &hsa_agent, nullptr, ptr) != HSA_STATUS_SUCCESS)
    {
        ext->hsa_amd_memory_pool_free_fn(ptr);
        return {nullptr, false};
    }
    return {ptr, false};
}

// Writes to device memory cross PCIe and may still be in flight when the packet is published, so
// fence them and read the last byte back before the doorbell.
void
write_kernarg(void* dest, const std::vector<std::byte>& image, bool on_device)
{
    std::memcpy(dest, image.data(), image.size());
    if(!on_device || image.empty()) return;

    auto* last = static_cast<volatile std::byte*>(dest) + image.size() - 1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *last = image.back();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    [[maybe_unused]] const auto landed = *last;
}

// The dispatches of one submission, in order, and where each one's arguments start in the
// argument buffer. Each segment is 16-byte aligned, and anything else the kernels read from the
// buffer follows the last one.
struct chain_t
{
    std::array<kernel_index, max_dispatches> kernels    = {};
    std::array<uint32_t, max_dispatches>     workgroups = {};
    std::array<size_t, max_dispatches>       offsets    = {};
    size_t                                   count      = 0;
    size_t                                   end        = 0;

    void add(const kernel_state_t& state, kernel_index kernel, uint32_t workgroup_count)
    {
        kernels.at(count)    = kernel;
        workgroups.at(count) = workgroup_count;
        offsets.at(count)    = end;
        end                  = align_up(end + state.kernels.at(kernel).kernarg_size, 16);
        ++count;
    }
};

// Grows the argument buffer. A packet that read the old one may still sit in the queue, so it is
// retired rather than freed.
bool
reserve_arguments(const hsa::AgentCache& agent, kernel_state_t& state, size_t size)
{
    if(size <= state.kernarg_capacity) return true;
    if(!hsa::get_amd_ext_table()) return false;

    auto capacity = std::max(size_t{4096}, state.kernarg_capacity);
    while(capacity < size)
    {
        if(capacity > std::numeric_limits<size_t>::max() / 2)
        {
            capacity = size;
            break;
        }
        capacity *= 2;
    }

    const auto [kernarg, on_device] = allocate_kernarg(agent, capacity);
    if(!kernarg) return false;
    ROCP_INFO << fmt::format("kernel replay blit: {}B argument buffer in {} memory for agent '{}'",
                             capacity,
                             on_device ? "device" : "host",
                             agent.name());

    if(state.kernarg) state.retired_kernargs.emplace_back(state.kernarg);
    state.kernarg           = kernarg;
    state.kernarg_capacity  = capacity;
    state.kernarg_on_device = on_device;
    return true;
}

// Grows the workspace. The caller holds resource_in_use, so no earlier blit can still be reading
// it.
bool
reserve_workspace(const hsa::AgentCache& agent, kernel_state_t& state, size_t size)
{
    if(size <= state.workspace_capacity) return true;

    auto* ext = hsa::get_amd_ext_table();
    if(!ext) return false;
    if(state.workspace) ext->hsa_amd_memory_pool_free_fn(state.workspace);
    state.workspace          = nullptr;
    state.workspace_capacity = 0;

    const auto capacity  = align_up(size, size_t{2} << 20);
    void*      workspace = nullptr;
    if(ext->hsa_amd_memory_pool_allocate_fn(agent.gpu_pool(), capacity, 0, &workspace) !=
       HSA_STATUS_SUCCESS)
        return false;
    ROCP_INFO << fmt::format(
        "kernel replay blit: {}B tile workspace for agent '{}'", capacity, agent.name());
    state.workspace          = workspace;
    state.workspace_capacity = capacity;
    return true;
}
}  // namespace

hsa_status_t
prepare(const hsa::Queue& queue)
{
    return get_state(queue.get_agent()) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

struct packet_info::implementation
{
    std::array<hsa::rocprofiler_packet, max_dispatches + 1> packets{};
    size_t                                                  packet_count = 0;
    kernel_state_t*                                         state        = nullptr;
    hsa_signal_t                                            completion{};
    void*                                                   kernarg   = nullptr;
    bool                                                    submitted = false;
    bool                                                    completed = false;
    hsa_status_t                                            status    = HSA_STATUS_SUCCESS;

    ~implementation()
    {
        if(state) state->resource_in_use.store(false, std::memory_order_release);
    }
};

packet_info::packet_info() = default;

packet_info::packet_info(std::unique_ptr<implementation> impl)
: m_impl{std::move(impl)}
{}

packet_info::~packet_info()
{
    if(!m_impl) return;
    if(m_impl->submitted && !m_impl->completed) wait();
}

packet_info::packet_info(packet_info&&) noexcept = default;

packet_info&
packet_info::operator=(packet_info&& rhs) noexcept
{
    if(this == &rhs) return *this;
    if(m_impl && m_impl->submitted && !m_impl->completed) wait();
    m_impl = std::move(rhs.m_impl);
    return *this;
}

const void*
packet_info::get_raw()
{
    if(!m_impl) return nullptr;
    m_impl->submitted = true;
    return m_impl->packets.data();
}

uint64_t
packet_info::packet_count() const
{
    return m_impl ? m_impl->packet_count : 0;
}

hsa_status_t
packet_info::wait()
{
    if(!m_impl) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    if(m_impl->completed) return m_impl->status;
    if(!m_impl->submitted) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    using namespace std::chrono_literals;
    auto value = hsa_signal_value_t{1};
    for(int i = 0; i < 12 && value >= 1; ++i)
    {
        value = hsa::get_core_table()->hsa_signal_wait_scacquire_fn(
            m_impl->completion,
            HSA_SIGNAL_CONDITION_LT,
            1,
            std::chrono::nanoseconds{5s}.count(),
            HSA_WAIT_STATE_BLOCKED);
    }

    m_impl->status    = (value == 0) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
    m_impl->completed = true;
    return m_impl->status;
}

hsa_status_t
packet_info::retire()
{
    if(!m_impl || !m_impl->submitted) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    if(m_impl->completed) return m_impl->status;

    const auto value  = hsa::get_core_table()->hsa_signal_load_scacquire_fn(m_impl->completion);
    m_impl->status    = (value == 0) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
    m_impl->completed = true;
    return m_impl->status;
}

struct packet_builder
{
    // Claims the agent's blit resources: until the packet is destroyed, no other blit can reuse
    // the argument buffer, the workspace or the completion signal.
    static std::unique_ptr<packet_info::implementation> claim(kernel_state_t& state)
    {
        auto impl     = std::make_unique<packet_info::implementation>();
        auto expected = false;
        if(!state.resource_in_use.compare_exchange_strong(
               expected, true, std::memory_order_acq_rel))
            return nullptr;
        impl->state      = &state;
        impl->completion = state.completion;
        return impl;
    }

    // Writes `image` to the argument buffer and the chain's packets. Each dispatch waits for the
    // one before it. The first acquires at system scope because the CPU wrote the arguments; the
    // last releases at system scope, signals completion, and is followed by a barrier on it.
    static packet_info finish(std::unique_ptr<packet_info::implementation> impl,
                              const chain_t&                               chain,
                              const std::vector<std::byte>&                image)
    {
        const auto& state = *impl->state;
        hsa::get_core_table()->hsa_signal_store_screlease_fn(impl->completion, 1);
        write_kernarg(impl->kernarg, image, state.kernarg_on_device);

        for(size_t index = 0; index < chain.count; ++index)
        {
            const auto& kernel = state.kernels.at(chain.kernels.at(index));
            const auto  first  = (index == 0);
            const auto  last   = (index + 1 == chain.count);

            auto packet   = hsa_kernel_dispatch_packet_t{};
            packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
            packet.header |= 1 << HSA_PACKET_HEADER_BARRIER;
            packet.header |= (first ? HSA_FENCE_SCOPE_SYSTEM : HSA_FENCE_SCOPE_AGENT)
                             << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
            packet.header |= (last ? HSA_FENCE_SCOPE_SYSTEM : HSA_FENCE_SCOPE_AGENT)
                             << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
            packet.setup                = 1;
            packet.workgroup_size_x     = static_cast<uint16_t>(workgroup_size);
            packet.workgroup_size_y     = 1;
            packet.workgroup_size_z     = 1;
            packet.grid_size_x          = chain.workgroups.at(index) * workgroup_size;
            packet.grid_size_y          = 1;
            packet.grid_size_z          = 1;
            packet.private_segment_size = kernel.private_segment_size;
            packet.group_segment_size   = kernel.group_segment_size;
            packet.kernel_object        = kernel.object;
            packet.kernarg_address =
                static_cast<std::byte*>(impl->kernarg) + chain.offsets.at(index);
            packet.completion_signal = last ? impl->completion : hsa_signal_t{0};
            impl->packets.at(index)  = hsa::rocprofiler_packet{packet};
        }

        auto barrier   = hsa_barrier_and_packet_t{};
        barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        barrier.header |= 1 << HSA_PACKET_HEADER_BARRIER;
        barrier.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
        barrier.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
        barrier.dep_signal[0]         = impl->completion;
        impl->packets.at(chain.count) = hsa::rocprofiler_packet{barrier};
        impl->packet_count            = chain.count + 1;
        return packet_info{std::move(impl)};
    }

    // Cuts `regions` into tiles on the GPU and copies them. With `keep`, the tile list stays in the
    // workspace for the restore kernels; with `digests`, the copy also digests the tiles for them.
    static std::optional<packet_info> create_list(const hsa::Queue&                 queue,
                                                  const std::vector<copy_region_t>& regions,
                                                  bool                              keep,
                                                  bool                              digests)
    {
        if(regions.empty()) return std::nullopt;
        const auto& agent = queue.get_agent();

        auto* state = get_state(agent);
        auto* core  = hsa::get_core_table();
        if(!state || !core || !core->hsa_signal_store_screlease_fn) return std::nullopt;

        const auto cu_count = agent.get_rocp_agent()->cu_count;
        if(cu_count == 0) return std::nullopt;

        // Descriptor and tile indices are 32-bit on the GPU, and the copy kernel's work items must
        // not wrap when a wave steps past the last one.
        constexpr auto max_count = uint64_t{std::numeric_limits<uint32_t>::max()} / 2;
        if(regions.size() > max_count) return std::nullopt;

        auto descriptor_data = std::vector<copy_descriptor_t>{};
        descriptor_data.reserve(regions.size());
        auto total_tiles = uint64_t{0};
        for(const auto& region : regions)
        {
            if(region.size == 0 || !region.source || !region.dest) return std::nullopt;
            total_tiles += tiles_for(region.size);
            if(total_tiles > max_count) return std::nullopt;
            descriptor_data.emplace_back(copy_descriptor_t{
                region.size,
                reinterpret_cast<uint64_t>(region.source),
                reinterpret_cast<uint64_t>(region.dest),
            });
        }

        const auto descriptor_count = static_cast<uint32_t>(descriptor_data.size());
        const auto tile_count       = static_cast<uint32_t>(total_tiles);
        const auto chunk_count      = chunks_for(descriptor_count);
        const auto copy_workgroups  = static_cast<uint32_t>(cu_count) * copy_workgroups_per_cu;
        const auto expand_workgroups =
            std::min(static_cast<uint32_t>((total_tiles + workgroup_size - 1) / workgroup_size),
                     copy_workgroups);

        auto impl = claim(*state);
        if(!impl) return std::nullopt;

        // This list replaces the one a capture kept. Without room for digests, a capture keeps just
        // its list, and its replay restores everything.
        state->kept    = false;
        state->digests = false;
        auto layout    = workspace_layout(descriptor_count, tile_count, digests);
        if(!reserve_workspace(agent, *state, layout.size))
        {
            if(!digests) return std::nullopt;
            digests = false;
            layout  = workspace_layout(descriptor_count, tile_count, false);
            if(!reserve_workspace(agent, *state, layout.size)) return std::nullopt;
        }

        auto chain = chain_t{};
        chain.add(*state, count_tiles, chunk_count);
        chain.add(*state, scan_tiles, chunk_count);
        chain.add(*state, expand_tiles, expand_workgroups);
        chain.add(*state, digests ? capture_tiles : copy_tiles, copy_workgroups);
        const auto descriptor_offset = chain.end;
        const auto image_size =
            descriptor_offset + size_t{descriptor_count} * sizeof(copy_descriptor_t);
        if(!reserve_arguments(agent, *state, image_size)) return std::nullopt;
        impl->kernarg = state->kernarg;

        auto image = std::vector<std::byte>{};
        try
        {
            image.resize(image_size);
        } catch(const std::bad_alloc&)
        {
            return std::nullopt;
        }

        const auto descriptors  = reinterpret_cast<uint64_t>(impl->kernarg) + descriptor_offset;
        const auto workspace    = reinterpret_cast<uint64_t>(state->workspace);
        const auto first_tiles  = workspace + layout.first_tiles;
        const auto chunk_totals = workspace + layout.chunk_totals;
        const auto tile_total   = workspace + layout.tile_count;
        const auto tiles        = workspace + layout.tiles;
        const auto digest_data  = digests ? workspace + layout.digests : 0;
        const auto flags        = digests ? workspace + layout.changed_flags : 0;

        if(digests && !state->counters_placed && hsa::get_amd_ext_table())
        {
            state->mapped_counters = allocate_mapped(agent, counter_count * sizeof(uint32_t));
            state->counters_placed = true;
            ROCP_INFO << fmt::format(
                "kernel replay blit: restore counters in {} for agent '{}'",
                state->mapped_counters ? "device memory the CPU reads" : "the workspace",
                agent.name());
        }
        const auto counters = !digests ? 0
                              : state->mapped_counters
                                  ? reinterpret_cast<uint64_t>(state->mapped_counters)
                                  : workspace + layout.counters;

        const auto write_args = [&](size_t index, const auto& args) {
            std::memcpy(image.data() + chain.offsets.at(index), &args, sizeof(args));
        };
        write_args(0, count_tiles_args_t{descriptors, chunk_totals, descriptor_count, 0});
        write_args(
            1,
            scan_tiles_args_t{
                descriptors, chunk_totals, first_tiles, tile_total, descriptor_count, chunk_count});
        write_args(2,
                   expand_tiles_args_t{descriptors,
                                       first_tiles,
                                       tile_total,
                                       tiles,
                                       digest_data,
                                       flags,
                                       counters,
                                       descriptor_count,
                                       expand_workgroups});
        if(digests)
            write_args(3, capture_tiles_args_t{tiles, tile_total, digest_data, copy_workgroups, 0});
        else
            write_args(3, copy_tiles_args_t{tiles, tile_total, copy_workgroups, 0});
        std::memcpy(image.data() + descriptor_offset,
                    descriptor_data.data(),
                    descriptor_data.size() * sizeof(descriptor_data[0]));

        if(keep)
        {
            state->kept             = true;
            state->digests          = digests;
            state->kept_tiles       = tile_count;
            state->kept_layout      = layout;
            state->changed_restores = 0;
            state->changed_slot     = 0;
            state->counters         = counters;
        }
        return finish(std::move(impl), chain, image);
    }

    // One restore kernel over the tiles of the capture that kept them. Only the full restore works
    // without digests.
    static std::optional<packet_info> create_restore(const hsa::Queue& queue, kernel_index kernel)
    {
        const auto& agent = queue.get_agent();
        auto*       state = get_state(agent);
        auto*       core  = hsa::get_core_table();
        if(!state || !state->kept || !core || !core->hsa_signal_store_screlease_fn)
            return std::nullopt;
        if(!state->digests && kernel != copy_tiles) return std::nullopt;

        const auto cu_count = agent.get_rocp_agent()->cu_count;
        if(cu_count == 0) return std::nullopt;

        auto impl = claim(*state);
        if(!impl) return std::nullopt;

        const auto workgroups = static_cast<uint32_t>(cu_count) * copy_workgroups_per_cu;
        auto       chain      = chain_t{};
        chain.add(*state, kernel, workgroups);
        if(!reserve_arguments(agent, *state, chain.end)) return std::nullopt;
        impl->kernarg = state->kernarg;

        const auto& layout     = state->kept_layout;
        const auto  workspace  = reinterpret_cast<uint64_t>(state->workspace);
        const auto  tiles      = workspace + layout.tiles;
        const auto  tile_total = workspace + layout.tile_count;
        const auto  digests    = workspace + layout.digests;
        const auto  flags      = workspace + layout.changed_flags;
        const auto  changed    = workspace + layout.changed;
        const auto  counters   = state->counters;

        auto       image      = std::vector<std::byte>(chain.end);
        const auto write_args = [&](const auto& args) {
            std::memcpy(image.data(), &args, sizeof(args));
        };
        switch(kernel)
        {
            case copy_tiles: write_args(copy_tiles_args_t{tiles, tile_total, workgroups, 1}); break;
            case restore_changed_tiles:
            {
                // Successive lists alternate between the two counts.
                state->changed_slot = state->changed_restores % changed_count_slots;
                state->changed_restores += 1;
                write_args(restore_changed_args_t{tiles,
                                                  tile_total,
                                                  digests,
                                                  flags,
                                                  changed,
                                                  counters,
                                                  state->changed_slot,
                                                  workgroups});
                break;
            }
            case restore_listed_tiles:
                write_args(restore_listed_args_t{tiles,
                                                 changed,
                                                 counters + state->changed_slot * sizeof(uint32_t),
                                                 workgroups,
                                                 0});
                break;
            case check_tiles:
                write_args(
                    check_tiles_args_t{tiles, tile_total, digests, flags, counters, workgroups, 0});
                break;
            default: return std::nullopt;
        }
        return finish(std::move(impl), chain, image);
    }
};

namespace
{
hsa_status_t
submit(hsa_amd_queue_intercept_packet_writer writer,
       std::optional<packet_info>            packet,
       std::vector<packet_info>&             pending)
{
    if(!packet) return HSA_STATUS_ERROR;
    writer(packet->get_raw(), packet->packet_count());
    pending.emplace_back(std::move(*packet));
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
list_and_copy(const hsa::Queue&                     queue,
              hsa_amd_queue_intercept_packet_writer writer,
              const std::vector<copy_region_t>&     regions,
              bool                                  keep,
              bool                                  digests,
              std::vector<packet_info>&             pending)
{
    if(!writer) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    if(regions.empty()) return HSA_STATUS_SUCCESS;
    if(!pending.empty()) return HSA_STATUS_ERROR;
    return submit(writer, packet_builder::create_list(queue, regions, keep, digests), pending);
}

hsa_status_t
restore_kernel(const hsa::Queue&                     queue,
               hsa_amd_queue_intercept_packet_writer writer,
               kernel_index                          kernel,
               std::vector<packet_info>&             pending)
{
    if(!writer) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    if(!pending.empty()) return HSA_STATUS_ERROR;
    return submit(writer, packet_builder::create_restore(queue, kernel), pending);
}
}  // namespace

hsa_status_t
copy(const hsa::Queue&                     queue,
     hsa_amd_queue_intercept_packet_writer writer,
     const std::vector<copy_region_t>&     regions,
     std::vector<packet_info>&             pending)
{
    return list_and_copy(queue, writer, regions, false, false, pending);
}

hsa_status_t
capture(const hsa::Queue&                     queue,
        hsa_amd_queue_intercept_packet_writer writer,
        const std::vector<copy_region_t>&     regions,
        bool                                  digests,
        std::vector<packet_info>&             pending)
{
    return list_and_copy(queue, writer, regions, true, digests, pending);
}

bool
has_kept_list(const hsa::Queue& queue)
{
    const auto* state = get_state(queue.get_agent());
    return state && state->kept;
}

bool
has_digests(const hsa::Queue& queue)
{
    const auto* state = get_state(queue.get_agent());
    return state && state->kept && state->digests;
}

void
forget_kept_list(const hsa::Queue& queue)
{
    if(auto* state = get_state(queue.get_agent()))
    {
        state->kept    = false;
        state->digests = false;
    }
}

hsa_status_t
restore_all(const hsa::Queue&                     queue,
            hsa_amd_queue_intercept_packet_writer writer,
            std::vector<packet_info>&             pending)
{
    return restore_kernel(queue, writer, copy_tiles, pending);
}

hsa_status_t
restore_changed(const hsa::Queue&                     queue,
                hsa_amd_queue_intercept_packet_writer writer,
                std::vector<packet_info>&             pending)
{
    return restore_kernel(queue, writer, restore_changed_tiles, pending);
}

hsa_status_t
restore_listed(const hsa::Queue&                     queue,
               hsa_amd_queue_intercept_packet_writer writer,
               std::vector<packet_info>&             pending)
{
    return restore_kernel(queue, writer, restore_listed_tiles, pending);
}

hsa_status_t
check_unlisted(const hsa::Queue&                     queue,
               hsa_amd_queue_intercept_packet_writer writer,
               std::vector<packet_info>&             pending)
{
    return restore_kernel(queue, writer, check_tiles, pending);
}

std::optional<digest_counts_t>
read_digest_counts(const hsa::Queue& queue)
{
    const auto* state = get_state(queue.get_agent());
    if(!state || !state->kept || !state->digests) return std::nullopt;
    if(state->resource_in_use.load(std::memory_order_acquire)) return std::nullopt;

    auto counters = std::array<uint32_t, counter_count>{};
    if(state->mapped_counters)
    {
        std::atomic_thread_fence(std::memory_order_acquire);
        std::memcpy(counters.data(), state->mapped_counters, sizeof(counters));
    }
    else
    {
        auto*       core = hsa::get_core_table();
        const auto* source =
            static_cast<const std::byte*>(state->workspace) + state->kept_layout.counters;
        if(!core || !core->hsa_memory_copy_fn ||
           core->hsa_memory_copy_fn(counters.data(), source, sizeof(counters)) !=
               HSA_STATUS_SUCCESS)
            return std::nullopt;
    }
    return digest_counts_t{
        state->kept_tiles, counters.at(state->changed_slot), counters.at(unlisted_counter)};
}
}  // namespace blit
}  // namespace kernel_replay
}  // namespace rocprofiler
