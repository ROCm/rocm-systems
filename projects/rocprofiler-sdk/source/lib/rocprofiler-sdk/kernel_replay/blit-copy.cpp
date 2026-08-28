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
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace kernel_replay
{
namespace blit
{
namespace
{
constexpr auto full_kernel_name   = "kernel_replay_blit";
constexpr auto stride_kernel_name = "kernel_replay_blit_stride";

using kernel_abi::bytes_per_tile;
using kernel_abi::copy_descriptor_t;
using kernel_abi::kernel_args_t;
using kernel_abi::workgroup_size;

struct kernel_info_t
{
    uint64_t object               = 0;
    uint32_t private_segment_size = 0;
    uint32_t group_segment_size   = 0;
    uint32_t kernarg_size         = 0;
};

struct kernel_state_t
{
    int                      file = -1;
    hsa_code_object_reader_t reader{};
    hsa_executable_t         executable{};
    kernel_info_t            full_kernel{};
    kernel_info_t            stride_kernel{};
    hsa_signal_t             completion{};
    void*                    kernarg          = nullptr;
    size_t                   kernarg_capacity = 0;
    std::vector<void*>       retired_kernargs{};
    std::atomic<bool>        resource_in_use{false};
};

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
    status = get_symbol_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &info.kernarg_size);
    if(status != HSA_STATUS_SUCCESS) return status;
    return (info.kernarg_size >= sizeof(void*)) ? HSA_STATUS_SUCCESS
                                                : HSA_STATUS_ERROR_INVALID_ARGUMENT;
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
    status = initialize_kernel(state.executable, hsa_agent, full_kernel_name, state.full_kernel);
    if(status != HSA_STATUS_SUCCESS) return status;
    status =
        initialize_kernel(state.executable, hsa_agent, stride_kernel_name, state.stride_kernel);
    if(status != HSA_STATUS_SUCCESS) return status;

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
}  // namespace

hsa_status_t
prepare(const hsa::Queue& queue)
{
    return get_state(queue.get_agent()) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

struct packet_info::implementation
{
    std::array<hsa::rocprofiler_packet, 2> packets{};
    kernel_state_t*                        state = nullptr;
    hsa_signal_t                           completion{};
    void*                                  kernarg   = nullptr;
    bool                                   submitted = false;
    bool                                   completed = false;
    hsa_status_t                           status    = HSA_STATUS_SUCCESS;

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

    auto descriptor_data = std::vector<copy_descriptor_t>{};
    descriptor_data.reserve(regions.size());
    auto total_tiles = uint64_t{0};
    for(const auto& region : regions)
    {
        if(!region.dst || !region.src || region.size == 0) return std::nullopt;
        const auto tile_count =
            region.size / bytes_per_tile + static_cast<uint64_t>(region.size % bytes_per_tile != 0);
        if(tile_count > std::numeric_limits<uint64_t>::max() - total_tiles) return std::nullopt;
        descriptor_data.emplace_back(copy_descriptor_t{reinterpret_cast<uint64_t>(region.src),
                                                       reinterpret_cast<uint64_t>(region.dst),
                                                       region.size,
                                                       total_tiles,
                                                       tile_count});
        total_tiles += tile_count;
    }

    constexpr auto max_launched_tiles =
        uint64_t{std::numeric_limits<uint32_t>::max()} / workgroup_size;
    const auto use_stride = (total_tiles > max_launched_tiles);

    auto launched_tiles = total_tiles;
    if(use_stride)
    {
        constexpr auto blocks_per_cu = uint64_t{2};
        const auto     cu_count      = agent.get_rocp_agent()->cu_count;
        if(cu_count == 0) return std::nullopt;
        launched_tiles = std::min(total_tiles, static_cast<uint64_t>(cu_count) * blocks_per_cu);
    }
    if(launched_tiles == 0 || launched_tiles > max_launched_tiles) return std::nullopt;

    const auto& kernel = use_stride ? state->stride_kernel : state->full_kernel;
    if(kernel.kernarg_size < sizeof(kernel_args_t)) return std::nullopt;
    const auto descriptor_offset = (kernel.kernarg_size + 15) & ~size_t{15};
    if(regions.size() > std::numeric_limits<uint32_t>::max() ||
       regions.size() >
           (std::numeric_limits<size_t>::max() - descriptor_offset) / sizeof(copy_descriptor_t))
        return std::nullopt;
    const auto allocation_size = descriptor_offset + regions.size() * sizeof(copy_descriptor_t);

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

        void* new_kernarg = nullptr;
        auto  status      = ext->hsa_amd_memory_pool_allocate_fn(
            agent.kernarg_pool(), new_capacity, 0, &new_kernarg);
        if(status != HSA_STATUS_SUCCESS) return std::nullopt;

        const auto hsa_agent = agent.get_hsa_agent();
        status = ext->hsa_amd_agents_allow_access_fn(1, &hsa_agent, nullptr, new_kernarg);
        if(status != HSA_STATUS_SUCCESS)
        {
            ext->hsa_amd_memory_pool_free_fn(new_kernarg);
            return std::nullopt;
        }

        if(state->kernarg) state->retired_kernargs.emplace_back(state->kernarg);
        state->kernarg          = new_kernarg;
        state->kernarg_capacity = new_capacity;
    }
    impl->kernarg = state->kernarg;

    core->hsa_signal_store_screlease_fn(impl->completion, 1);

    std::memset(impl->kernarg, 0, allocation_size);
    auto* descriptor_memory = static_cast<std::byte*>(impl->kernarg) + descriptor_offset;
    std::memcpy(descriptor_memory,
                descriptor_data.data(),
                descriptor_data.size() * sizeof(descriptor_data[0]));
    const auto kernel_args = kernel_args_t{reinterpret_cast<uint64_t>(descriptor_memory),
                                           static_cast<uint64_t>(descriptor_data.size()),
                                           total_tiles,
                                           launched_tiles};
    std::memcpy(impl->kernarg, &kernel_args, sizeof(kernel_args));

    const auto grid_size_x = static_cast<uint32_t>(launched_tiles * workgroup_size);

    auto packet   = hsa_kernel_dispatch_packet_t{};
    packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
    packet.header |= 1 << HSA_PACKET_HEADER_BARRIER;
    packet.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    packet.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    packet.setup                = 1;
    packet.workgroup_size_x     = workgroup_size;
    packet.workgroup_size_y     = 1;
    packet.workgroup_size_z     = 1;
    packet.grid_size_x          = grid_size_x;
    packet.grid_size_y          = 1;
    packet.grid_size_z          = 1;
    packet.private_segment_size = kernel.private_segment_size;
    packet.group_segment_size   = kernel.group_segment_size;
    packet.kernel_object        = kernel.object;
    packet.kernarg_address      = impl->kernarg;
    packet.completion_signal    = impl->completion;
    impl->packets[0]            = hsa::rocprofiler_packet{packet};

    auto barrier   = hsa_barrier_and_packet_t{};
    barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    barrier.header |= 1 << HSA_PACKET_HEADER_BARRIER;
    barrier.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    barrier.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    barrier.dep_signal[0] = impl->completion;
    impl->packets[1]      = hsa::rocprofiler_packet{barrier};

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
