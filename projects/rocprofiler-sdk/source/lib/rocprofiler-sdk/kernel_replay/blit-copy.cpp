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
using kernel_abi::chunks_for;
using kernel_abi::copy_descriptor_t;
using kernel_abi::copy_tiles_args_t;
using kernel_abi::copy_workgroups_per_cu;
using kernel_abi::count_tiles_args_t;
using kernel_abi::expand_tiles_args_t;
using kernel_abi::scan_tiles_args_t;
using kernel_abi::tiles_for;
using kernel_abi::workgroup_size;

// The restore runs as four dispatches, in this order: three build the tile list on the GPU from
// the region descriptors, and the last copies it.
enum kernel_index : size_t
{
    count_tiles = 0,
    scan_tiles,
    expand_tiles,
    copy_tiles,
    kernel_count
};

constexpr auto kernel_names = std::array<const char*, kernel_count>{
    "kernel_replay_count_tiles",
    "kernel_replay_scan_tiles",
    "kernel_replay_expand_tiles",
    "kernel_replay_copy_tiles",
};

constexpr auto kernel_arg_sizes = std::array<size_t, kernel_count>{
    sizeof(count_tiles_args_t),
    sizeof(scan_tiles_args_t),
    sizeof(expand_tiles_args_t),
    sizeof(copy_tiles_args_t),
};

struct kernel_info_t
{
    uint64_t object               = 0;
    uint32_t private_segment_size = 0;
    uint32_t group_segment_size   = 0;
    uint32_t kernarg_size         = 0;
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
};

// Where the tile-list kernels keep their intermediate results, in one device allocation.
struct workspace_layout_t
{
    size_t first_tiles  = 0;
    size_t chunk_totals = 0;
    size_t tile_count   = 0;
    size_t tiles        = 0;
    size_t size         = 0;
};

constexpr size_t
align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

workspace_layout_t
workspace_layout(uint32_t descriptor_count, uint32_t tile_count)
{
    constexpr auto alignment = size_t{256};
    auto           layout    = workspace_layout_t{};
    layout.chunk_totals      = align_up(size_t{descriptor_count} * sizeof(uint32_t), alignment);
    layout.tile_count =
        layout.chunk_totals +
        align_up(size_t{chunks_for(descriptor_count)} * sizeof(uint32_t), alignment);
    layout.tiles = layout.tile_count + alignment;
    layout.size  = layout.tiles + size_t{tile_count} * sizeof(copy_descriptor_t);
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

kernel_state_t*
get_state(const hsa::AgentCache& agent)
{
    auto* state = states().rlock([&](const auto& map) -> kernel_state_t* {
        auto itr = map.find(agent.get_hsa_agent().handle);
        return (itr == map.end()) ? nullptr : itr->second;
    });
    if(state) return state;

    return states().wlock([&](auto& map) -> kernel_state_t* {
        auto& slot = map[agent.get_hsa_agent().handle];
        if(slot) return slot;

        auto candidate = std::make_unique<kernel_state_t>();
        if(initialize_state(agent, *candidate) != HSA_STATUS_SUCCESS) return nullptr;
        slot = candidate.release();
        return slot;
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

// Every wave reads the blit's arguments and copy descriptors, and reading them from host memory
// makes each restore depend on PCIe read latency. Keep them in device memory the CPU can write,
// and fall back to the host kernarg pool when it cannot. Returns the buffer and whether it is in
// device memory.
std::pair<void*, bool>
allocate_kernarg(const hsa::AgentCache& agent, size_t size)
{
    auto* ext = hsa::get_amd_ext_table();
    void* ptr = nullptr;
    if(ext->hsa_amd_memory_pool_allocate_fn(
           agent.gpu_pool(), size, HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, &ptr) == HSA_STATUS_SUCCESS)
    {
        if(host_writable(ptr)) return {ptr, true};
        ext->hsa_amd_memory_pool_free_fn(ptr);
    }

    ptr = nullptr;
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
}  // namespace

hsa_status_t
prepare(const hsa::Queue& queue)
{
    return get_state(queue.get_agent()) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

struct packet_info::implementation
{
    std::array<hsa::rocprofiler_packet, kernel_count + 1> packets{};
    kernel_state_t*                                       state = nullptr;
    hsa_signal_t                                          completion{};
    void*                                                 kernarg   = nullptr;
    bool                                                  submitted = false;
    bool                                                  completed = false;
    hsa_status_t                                          status    = HSA_STATUS_SUCCESS;

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
    return m_impl ? m_impl->packets.size() : 0;
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

std::optional<packet_info>
create(const hsa::Queue& queue, const std::vector<copy_region_t>& regions)
{
    if(regions.empty()) return std::nullopt;
    const auto& agent = queue.get_agent();

    auto* state = get_state(agent);
    auto* core  = hsa::get_core_table();
    auto* ext   = hsa::get_amd_ext_table();
    if(!state || !core || !ext || !core->hsa_signal_store_screlease_fn) return std::nullopt;

    const auto cu_count = agent.get_rocp_agent()->cu_count;
    if(cu_count == 0) return std::nullopt;

    // Descriptor and tile indices are 32-bit on the GPU, and the copy kernel's work items must not
    // wrap when a wave steps past the last one.
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
    const auto workgroups = std::array<uint32_t, kernel_count>{
        chunk_count, chunk_count, expand_workgroups, copy_workgroups};

    // One 16-byte aligned argument segment per kernel, then the descriptors.
    auto kernarg_offsets = std::array<size_t, kernel_count>{};
    auto kernarg_end     = size_t{0};
    for(size_t index = 0; index < kernel_count; ++index)
    {
        kernarg_offsets.at(index) = kernarg_end;
        kernarg_end = align_up(kernarg_end + state->kernels.at(index).kernarg_size, 16);
    }
    const auto descriptor_offset = kernarg_end;
    const auto allocation_size =
        descriptor_offset + size_t{descriptor_count} * sizeof(copy_descriptor_t);
    const auto layout = workspace_layout(descriptor_count, tile_count);

    auto impl     = std::make_unique<packet_info::implementation>();
    auto expected = false;
    if(!state->resource_in_use.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        return std::nullopt;

    impl->state      = state;
    impl->completion = state->completion;

    if(allocation_size > state->kernarg_capacity)
    {
        auto new_capacity = std::max(size_t{4096}, state->kernarg_capacity);
        while(new_capacity < allocation_size)
        {
            if(new_capacity > std::numeric_limits<size_t>::max() / 2)
            {
                new_capacity = allocation_size;
                break;
            }
            new_capacity *= 2;
        }

        const auto [new_kernarg, on_device] = allocate_kernarg(agent, new_capacity);
        if(!new_kernarg) return std::nullopt;
        ROCP_INFO << fmt::format("kernel replay blit: {}B argument buffer in {} memory for agent "
                                 "'{}'",
                                 new_capacity,
                                 on_device ? "device" : "host",
                                 agent.name());

        if(state->kernarg) state->retired_kernargs.emplace_back(state->kernarg);
        state->kernarg           = new_kernarg;
        state->kernarg_capacity  = new_capacity;
        state->kernarg_on_device = on_device;
    }
    impl->kernarg = state->kernarg;

    if(layout.size > state->workspace_capacity)
    {
        // This blit holds resource_in_use, so no earlier one can still be reading the workspace.
        if(state->workspace) ext->hsa_amd_memory_pool_free_fn(state->workspace);
        state->workspace          = nullptr;
        state->workspace_capacity = 0;

        const auto capacity  = align_up(layout.size, size_t{2} << 20);
        void*      workspace = nullptr;
        if(ext->hsa_amd_memory_pool_allocate_fn(agent.gpu_pool(), capacity, 0, &workspace) !=
           HSA_STATUS_SUCCESS)
            return std::nullopt;
        ROCP_INFO << fmt::format(
            "kernel replay blit: {}B tile workspace for agent '{}'", capacity, agent.name());
        state->workspace          = workspace;
        state->workspace_capacity = capacity;
    }

    auto image = std::vector<std::byte>{};
    try
    {
        image.resize(allocation_size);
    } catch(const std::bad_alloc&)
    {
        return std::nullopt;
    }

    const auto kernarg_base       = reinterpret_cast<uint64_t>(impl->kernarg);
    const auto workspace_base     = reinterpret_cast<uint64_t>(state->workspace);
    const auto descriptors        = kernarg_base + descriptor_offset;
    const auto first_tiles        = workspace_base + layout.first_tiles;
    const auto chunk_totals       = workspace_base + layout.chunk_totals;
    const auto tile_count_address = workspace_base + layout.tile_count;
    const auto tiles              = workspace_base + layout.tiles;

    const auto write_args = [&](kernel_index index, const auto& args) {
        std::memcpy(image.data() + kernarg_offsets.at(index), &args, sizeof(args));
    };
    write_args(count_tiles, count_tiles_args_t{descriptors, chunk_totals, descriptor_count, 0});
    write_args(scan_tiles,
               scan_tiles_args_t{descriptors,
                                 chunk_totals,
                                 first_tiles,
                                 tile_count_address,
                                 descriptor_count,
                                 chunk_count});
    write_args(expand_tiles,
               expand_tiles_args_t{descriptors,
                                   first_tiles,
                                   tile_count_address,
                                   tiles,
                                   descriptor_count,
                                   expand_workgroups});
    write_args(copy_tiles, copy_tiles_args_t{tiles, tile_count_address, copy_workgroups, 0});
    std::memcpy(image.data() + descriptor_offset,
                descriptor_data.data(),
                descriptor_data.size() * sizeof(descriptor_data[0]));

    core->hsa_signal_store_screlease_fn(impl->completion, 1);
    write_kernarg(impl->kernarg, image, state->kernarg_on_device);

    // Each dispatch waits for the one before it. The first acquires at system scope because the
    // CPU wrote the descriptors; the copy releases at system scope and signals completion.
    for(size_t index = 0; index < kernel_count; ++index)
    {
        const auto& kernel = state->kernels.at(index);
        const auto  first  = (index == 0);
        const auto  last   = (index + 1 == kernel_count);

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
        packet.grid_size_x          = workgroups.at(index) * workgroup_size;
        packet.grid_size_y          = 1;
        packet.grid_size_z          = 1;
        packet.private_segment_size = kernel.private_segment_size;
        packet.group_segment_size   = kernel.group_segment_size;
        packet.kernel_object        = kernel.object;
        packet.kernarg_address = static_cast<std::byte*>(impl->kernarg) + kernarg_offsets.at(index);
        packet.completion_signal = last ? impl->completion : hsa_signal_t{0};
        impl->packets.at(index)  = hsa::rocprofiler_packet{packet};
    }

    auto barrier   = hsa_barrier_and_packet_t{};
    barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    barrier.header |= 1 << HSA_PACKET_HEADER_BARRIER;
    barrier.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    barrier.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    barrier.dep_signal[0]          = impl->completion;
    impl->packets.at(kernel_count) = hsa::rocprofiler_packet{barrier};

    return packet_info{std::move(impl)};
}

hsa_status_t
copy(const hsa::Queue&                     queue,
     hsa_amd_queue_intercept_packet_writer writer,
     const std::vector<copy_region_t>&     regions,
     std::vector<packet_info>&             pending)
{
    if(!writer) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    if(regions.empty()) return HSA_STATUS_SUCCESS;
    if(!pending.empty()) return HSA_STATUS_ERROR;

    auto packet = create(queue, regions);
    if(!packet) return HSA_STATUS_ERROR;
    writer(packet->get_raw(), packet->packet_count());
    pending.emplace_back(std::move(*packet));
    return HSA_STATUS_SUCCESS;
}
}  // namespace blit
}  // namespace kernel_replay
}  // namespace rocprofiler
