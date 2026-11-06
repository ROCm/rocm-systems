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

// Runs the kernel replay blit's code object on the first GPU the way the SDK does: raw AQL
// dispatches with only the explicit kernel arguments written. Each layout is copied once and
// checked byte for byte, including the gaps between regions. The restore kernels are checked
// against digests and changed tiles worked out on the host.

#include "lib/rocprofiler-sdk/kernel_replay/blit-copy-kernel.hpp"

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace
{
namespace abi = rocprofiler::kernel_replay::blit::kernel_abi;

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

constexpr auto kernel_arg_sizes = std::array<uint32_t, kernel_count>{
    sizeof(abi::count_tiles_args_t),
    sizeof(abi::scan_tiles_args_t),
    sizeof(abi::expand_tiles_args_t),
    sizeof(abi::copy_tiles_args_t),
    sizeof(abi::capture_tiles_args_t),
    sizeof(abi::restore_changed_args_t),
    sizeof(abi::restore_listed_args_t),
    sizeof(abi::check_tiles_args_t),
};

constexpr auto kernarg_stride = size_t{64};
constexpr auto sentinel       = std::byte{0xa5};

struct kernel_t
{
    uint64_t object               = 0;
    uint32_t private_segment_size = 0;
    uint32_t group_segment_size   = 0;
    uint32_t kernarg_size         = 0;
};

// One dispatch of a chain, with its arguments.
struct step_t
{
    kernel_index                          kernel     = copy_tiles;
    uint32_t                              workgroups = 0;
    std::array<std::byte, kernarg_stride> args       = {};
};

template <typename Args>
step_t
make_step(kernel_index kernel, uint32_t workgroups, const Args& args)
{
    static_assert(sizeof(Args) <= kernarg_stride);
    auto step = step_t{kernel, workgroups};
    std::memcpy(step.args.data(), &args, sizeof(args));
    return step;
}

uint64_t
address(const void* ptr)
{
    return reinterpret_cast<uint64_t>(ptr);
}

struct pool_search_t
{
    uint32_t              flags = 0;
    hsa_amd_memory_pool_t pool{};
    bool                  found = false;
};

hsa_status_t
find_pool(hsa_amd_memory_pool_t pool, void* data)
{
    auto* search  = static_cast<pool_search_t*>(data);
    auto  segment = hsa_amd_segment_t{};
    auto  flags   = uint32_t{0};
    auto  allowed = false;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &allowed);
    if(segment != HSA_AMD_SEGMENT_GLOBAL || !allowed || (flags & search->flags) == 0)
        return HSA_STATUS_SUCCESS;
    search->pool  = pool;
    search->found = true;
    return HSA_STATUS_INFO_BREAK;
}

std::optional<hsa_amd_memory_pool_t>
pool_with(hsa_agent_t agent, uint32_t flags)
{
    auto search = pool_search_t{flags};
    hsa_amd_agent_iterate_memory_pools(agent, find_pool, &search);
    if(!search.found) return std::nullopt;
    return search.pool;
}

hsa_status_t
find_agent(hsa_agent_t agent, void* data)
{
    auto* found = static_cast<std::pair<hsa_device_type_t, hsa_agent_t>*>(data);
    auto  type  = hsa_device_type_t{};
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if(type != found->first) return HSA_STATUS_SUCCESS;
    found->second = agent;
    return HSA_STATUS_INFO_BREAK;
}

std::optional<hsa_agent_t>
first_agent(hsa_device_type_t type)
{
    auto found = std::pair<hsa_device_type_t, hsa_agent_t>{type, hsa_agent_t{0}};
    hsa_iterate_agents(find_agent, &found);
    if(found.second.handle == 0) return std::nullopt;
    return found.second;
}

class gpu_t
{
public:
    static gpu_t& get()
    {
        static auto* value = new gpu_t{};
        return *value;
    }

    bool               available() const { return m_reason.empty(); }
    const std::string& reason() const { return m_reason; }
    const auto&        kernels() const { return m_kernels; }
    uint32_t           copy_workgroups() const { return m_cu_count * abi::copy_workgroups_per_cu; }

    void* allocate_device(size_t size) const
    {
        void* ptr = nullptr;
        if(hsa_amd_memory_pool_allocate(m_device_pool, std::max<size_t>(size, 64), 0, &ptr) !=
           HSA_STATUS_SUCCESS)
            return nullptr;
        return ptr;
    }

    // Host memory the GPU reads over PCIe, like the SDK's pinned snapshot arena.
    void* allocate_pinned(size_t size) const { return allocate_host(m_staging_pool, size); }

    // Dispatches `steps` with the SDK's fences and returns whether they completed in time.
    bool run(const std::vector<step_t>& steps) const;

private:
    gpu_t();

    void* allocate_host(hsa_amd_memory_pool_t pool, size_t size) const
    {
        void* ptr = nullptr;
        if(hsa_amd_memory_pool_allocate(pool, std::max<size_t>(size, 64), 0, &ptr) !=
           HSA_STATUS_SUCCESS)
            return nullptr;
        if(hsa_amd_agents_allow_access(1, &m_gpu, nullptr, ptr) != HSA_STATUS_SUCCESS)
        {
            hsa_amd_memory_pool_free(ptr);
            return nullptr;
        }
        return ptr;
    }

    std::string                        m_reason{};
    hsa_agent_t                        m_gpu{};
    hsa_amd_memory_pool_t              m_device_pool{};
    hsa_amd_memory_pool_t              m_kernarg_pool{};
    hsa_amd_memory_pool_t              m_staging_pool{};
    hsa_queue_t*                       m_queue    = nullptr;
    uint32_t                           m_cu_count = 0;
    std::array<kernel_t, kernel_count> m_kernels{};
    hsa_executable_t                   m_executable{};
    hsa_code_object_reader_t           m_reader{};
};

gpu_t::gpu_t()
{
    if(hsa_init() != HSA_STATUS_SUCCESS)
    {
        m_reason = "HSA is not available";
        return;
    }
    const auto gpu = first_agent(HSA_DEVICE_TYPE_GPU);
    const auto cpu = first_agent(HSA_DEVICE_TYPE_CPU);
    if(!gpu || !cpu)
    {
        m_reason = "no GPU agent";
        return;
    }
    m_gpu = *gpu;

    const auto device_pool  = pool_with(m_gpu, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED);
    const auto kernarg_pool = pool_with(*cpu, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT);
    const auto staging_pool = pool_with(*cpu, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED);
    if(!device_pool || !kernarg_pool || !staging_pool)
    {
        m_reason = "no device, kernarg or staging memory pool";
        return;
    }
    m_device_pool  = *device_pool;
    m_kernarg_pool = *kernarg_pool;
    m_staging_pool = *staging_pool;
    hsa_agent_get_info(
        m_gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT), &m_cu_count);

    char name[64] = {};
    hsa_agent_get_info(m_gpu, HSA_AGENT_INFO_NAME, name);
    const auto path =
        std::string{KERNEL_REPLAY_BLIT_CODE_OBJECT_DIR} + "/" + name + "_kernel_replay_blit.hsaco";
    const auto file = open(path.c_str(), O_RDONLY);
    if(file < 0)
    {
        m_reason = "no code object at " + path;
        return;
    }
    if(hsa_code_object_reader_create_from_file(file, &m_reader) != HSA_STATUS_SUCCESS ||
       hsa_executable_create_alt(
           HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &m_executable) !=
           HSA_STATUS_SUCCESS ||
       hsa_executable_load_agent_code_object(m_executable, m_gpu, m_reader, nullptr, nullptr) !=
           HSA_STATUS_SUCCESS ||
       hsa_executable_freeze(m_executable, nullptr) != HSA_STATUS_SUCCESS)
    {
        m_reason = "could not load " + path;
        return;
    }

    for(size_t index = 0; index < m_kernels.size(); ++index)
    {
        auto symbol = hsa_executable_symbol_t{};
        if(hsa_executable_get_symbol_by_name(
               m_executable, kernel_names.at(index), &m_gpu, &symbol) != HSA_STATUS_SUCCESS &&
           hsa_executable_get_symbol_by_name(m_executable,
                                             (std::string{kernel_names.at(index)} + ".kd").c_str(),
                                             &m_gpu,
                                             &symbol) != HSA_STATUS_SUCCESS)
        {
            m_reason = std::string{"no kernel "} + kernel_names.at(index) + " in " + path;
            return;
        }
        auto& kernel = m_kernels.at(index);
        hsa_executable_symbol_get_info(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel.object);
        hsa_executable_symbol_get_info(symbol,
                                       HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                       &kernel.private_segment_size);
        hsa_executable_symbol_get_info(symbol,
                                       HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                       &kernel.group_segment_size);
        hsa_executable_symbol_get_info(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernel.kernarg_size);
    }

    if(hsa_queue_create(m_gpu,
                        1024,
                        HSA_QUEUE_TYPE_SINGLE,
                        nullptr,
                        nullptr,
                        UINT32_MAX,
                        UINT32_MAX,
                        &m_queue) != HSA_STATUS_SUCCESS)
        m_reason = "could not create a queue";
}

bool
gpu_t::run(const std::vector<step_t>& steps) const
{
    auto* kernargs = allocate_host(m_kernarg_pool, kernarg_stride * steps.size());
    auto  done     = hsa_signal_t{};
    if(!kernargs || hsa_signal_create(1, 0, nullptr, &done) != HSA_STATUS_SUCCESS)
    {
        if(kernargs) hsa_amd_memory_pool_free(kernargs);
        return false;
    }
    auto* args = static_cast<std::byte*>(kernargs);
    for(size_t index = 0; index < steps.size(); ++index)
        std::memcpy(args + index * kernarg_stride, steps.at(index).args.data(), kernarg_stride);

    // The same packet chain as the SDK: every dispatch waits for the previous one, system scope
    // at the two ends, and the last one signals completion.
    const auto first_index = hsa_queue_add_write_index_scacq_screl(m_queue, steps.size());
    while(first_index + steps.size() - hsa_queue_load_read_index_scacquire(m_queue) > m_queue->size)
    {}
    auto* packets = static_cast<hsa_kernel_dispatch_packet_t*>(m_queue->base_address);
    auto  headers = std::vector<uint16_t>(steps.size());
    for(size_t index = 0; index < steps.size(); ++index)
    {
        const auto& kernel          = m_kernels.at(steps.at(index).kernel);
        const auto  first           = (index == 0);
        const auto  last            = (index + 1 == steps.size());
        auto&       packet          = packets[(first_index + index) & (m_queue->size - 1)];
        packet.reserved0            = 0;
        packet.reserved2            = 0;
        packet.workgroup_size_x     = static_cast<uint16_t>(abi::workgroup_size);
        packet.workgroup_size_y     = 1;
        packet.workgroup_size_z     = 1;
        packet.grid_size_x          = steps.at(index).workgroups * abi::workgroup_size;
        packet.grid_size_y          = 1;
        packet.grid_size_z          = 1;
        packet.private_segment_size = kernel.private_segment_size;
        packet.group_segment_size   = kernel.group_segment_size;
        packet.kernel_object        = kernel.object;
        packet.kernarg_address      = args + index * kernarg_stride;
        packet.completion_signal    = last ? done : hsa_signal_t{0};
        headers.at(index)           = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                            (1 << HSA_PACKET_HEADER_BARRIER) |
                            ((first ? HSA_FENCE_SCOPE_SYSTEM : HSA_FENCE_SCOPE_AGENT)
                             << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                            ((last ? HSA_FENCE_SCOPE_SYSTEM : HSA_FENCE_SCOPE_AGENT)
                             << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    }
    for(size_t index = 0; index < steps.size(); ++index)
    {
        auto* word =
            reinterpret_cast<uint32_t*>(&packets[(first_index + index) & (m_queue->size - 1)]);
        __atomic_store_n(word, uint32_t{headers.at(index)} | (uint32_t{1} << 16), __ATOMIC_RELEASE);
    }
    hsa_signal_store_screlease(m_queue->doorbell_signal, first_index + steps.size() - 1);

    using clock         = std::chrono::steady_clock;
    const auto deadline = clock::now() + std::chrono::seconds{60};
    auto       value    = hsa_signal_value_t{1};
    while(value != 0 && clock::now() < deadline)
        value = hsa_signal_wait_scacquire(
            done, HSA_SIGNAL_CONDITION_EQ, 0, 1000000000, HSA_WAIT_STATE_BLOCKED);

    hsa_signal_destroy(done);
    hsa_amd_memory_pool_free(kernargs);
    return value == 0;
}

template <typename Tp>
std::vector<Tp>
read_device(const void* device, size_t count)
{
    auto values = std::vector<Tp>(count);
    if(count != 0) hsa_memory_copy(values.data(), device, count * sizeof(Tp));
    return values;
}

// The buffers the SDK keeps in its workspace for one tile list, here allocated separately, and
// the dispatches it makes over them.
class workspace_t
{
public:
    workspace_t(const std::vector<abi::copy_descriptor_t>& regions, bool digests)
    : m_descriptor_count{static_cast<uint32_t>(regions.size())}
    {
        for(const auto& region : regions)
            m_tile_count += static_cast<uint32_t>(abi::tiles_for(region.size));
        const auto descriptors_size = regions.size() * sizeof(abi::copy_descriptor_t);
        m_descriptors               = allocate(descriptors_size);
        m_first_tiles               = allocate(size_t{m_descriptor_count} * sizeof(uint32_t));
        m_chunk_totals = allocate(size_t{abi::chunks_for(m_descriptor_count)} * sizeof(uint32_t));
        m_tile_total   = allocate(sizeof(uint32_t));
        m_counters     = allocate(abi::counter_count * sizeof(uint32_t));
        m_tiles        = allocate(size_t{m_tile_count} * sizeof(abi::copy_descriptor_t));
        if(digests)
        {
            m_digests = allocate(size_t{m_tile_count} * sizeof(uint64_t));
            m_flags   = allocate(m_tile_count);
            m_changed = allocate(size_t{m_tile_count} * sizeof(uint32_t));
        }
        m_ok = m_ok && hsa_memory_copy(m_descriptors, regions.data(), descriptors_size) ==
                           HSA_STATUS_SUCCESS;
    }

    ~workspace_t()
    {
        for(void* buffer : m_buffers)
            hsa_amd_memory_pool_free(buffer);
    }

    workspace_t(const workspace_t&) = delete;
    workspace_t& operator=(const workspace_t&) = delete;

    bool     ok() const { return m_ok; }
    uint32_t tile_count() const { return m_tile_count; }

    // The SDK's copy: three dispatches build the tile list, and the last copies it, digesting the
    // tiles when the workspace has digests.
    std::vector<step_t> copy_steps() const
    {
        const auto chunks            = abi::chunks_for(m_descriptor_count);
        const auto copy_workgroups   = gpu_t::get().copy_workgroups();
        const auto expand_workgroups = std::min(
            (m_tile_count + abi::workgroup_size - 1) / abi::workgroup_size, copy_workgroups);
        auto steps = std::vector<step_t>{
            make_step(count_tiles,
                      chunks,
                      abi::count_tiles_args_t{
                          address(m_descriptors), address(m_chunk_totals), m_descriptor_count, 0}),
            make_step(scan_tiles,
                      chunks,
                      abi::scan_tiles_args_t{address(m_descriptors),
                                             address(m_chunk_totals),
                                             address(m_first_tiles),
                                             address(m_tile_total),
                                             m_descriptor_count,
                                             chunks}),
            make_step(expand_tiles,
                      expand_workgroups,
                      abi::expand_tiles_args_t{address(m_descriptors),
                                               address(m_first_tiles),
                                               address(m_tile_total),
                                               address(m_tiles),
                                               address(m_digests),
                                               address(m_flags),
                                               m_digests ? address(m_counters) : 0,
                                               m_descriptor_count,
                                               expand_workgroups}),
        };
        if(m_digests)
            steps.push_back(make_step(capture_tiles,
                                      copy_workgroups,
                                      abi::capture_tiles_args_t{address(m_tiles),
                                                                address(m_tile_total),
                                                                address(m_digests),
                                                                copy_workgroups,
                                                                0}));
        else
            steps.push_back(
                make_step(copy_tiles,
                          copy_workgroups,
                          abi::copy_tiles_args_t{
                              address(m_tiles), address(m_tile_total), copy_workgroups, 0}));
        return steps;
    }

    // A full restore: the copy kernel over the capture's list, each tile back to its source.
    step_t restore_all() const
    {
        const auto workgroups = gpu_t::get().copy_workgroups();
        return make_step(
            copy_tiles,
            workgroups,
            abi::copy_tiles_args_t{address(m_tiles), address(m_tile_total), workgroups, 1});
    }

    step_t restore_changed(uint32_t slot) const
    {
        const auto workgroups = gpu_t::get().copy_workgroups();
        return make_step(restore_changed_tiles,
                         workgroups,
                         abi::restore_changed_args_t{address(m_tiles),
                                                     address(m_tile_total),
                                                     address(m_digests),
                                                     address(m_flags),
                                                     address(m_changed),
                                                     address(m_counters),
                                                     slot,
                                                     workgroups});
    }

    step_t restore_listed(uint32_t slot) const
    {
        const auto workgroups = gpu_t::get().copy_workgroups();
        return make_step(restore_listed_tiles,
                         workgroups,
                         abi::restore_listed_args_t{address(m_tiles),
                                                    address(m_changed),
                                                    address(m_counters) + slot * sizeof(uint32_t),
                                                    workgroups,
                                                    0});
    }

    step_t check() const
    {
        const auto workgroups = gpu_t::get().copy_workgroups();
        return make_step(check_tiles,
                         workgroups,
                         abi::check_tiles_args_t{address(m_tiles),
                                                 address(m_tile_total),
                                                 address(m_digests),
                                                 address(m_flags),
                                                 address(m_counters),
                                                 workgroups,
                                                 0});
    }

    std::vector<uint32_t> counters() const
    {
        return read_device<uint32_t>(m_counters, abi::counter_count);
    }
    std::vector<uint64_t> digests() const { return read_device<uint64_t>(m_digests, m_tile_count); }
    std::vector<uint8_t>  flags() const { return read_device<uint8_t>(m_flags, m_tile_count); }
    std::vector<uint32_t> changed(uint32_t count) const
    {
        return read_device<uint32_t>(m_changed, count);
    }

    void set_counter(uint32_t index, uint32_t value) const
    {
        hsa_memory_copy(static_cast<uint32_t*>(m_counters) + index, &value, sizeof(value));
    }

private:
    void* allocate(size_t size)
    {
        auto* buffer = gpu_t::get().allocate_device(size);
        if(buffer)
            m_buffers.push_back(buffer);
        else
            m_ok = false;
        return buffer;
    }

    bool               m_ok               = true;
    uint32_t           m_descriptor_count = 0;
    uint32_t           m_tile_count       = 0;
    std::vector<void*> m_buffers          = {};
    void*              m_descriptors      = nullptr;
    void*              m_first_tiles      = nullptr;
    void*              m_chunk_totals     = nullptr;
    void*              m_tile_total       = nullptr;
    void*              m_counters         = nullptr;
    void*              m_tiles            = nullptr;
    void*              m_digests          = nullptr;
    void*              m_flags            = nullptr;
    void*              m_changed          = nullptr;
};

// Source and destination spans with the regions at the same offsets in both.
struct layout_t
{
    std::vector<std::pair<uint64_t, uint64_t>> regions{};  // offset, size
    uint64_t                                   span = 0;
};

layout_t
make_layout(const std::vector<uint64_t>& sizes, uint64_t alignment, uint64_t max_gap, uint64_t seed)
{
    auto rng    = std::mt19937_64{seed};
    auto gap    = std::uniform_int_distribution<uint64_t>{0, max_gap};
    auto layout = layout_t{};
    auto offset = uint64_t{0};
    for(const auto size : sizes)
    {
        offset += gap(rng);
        offset = (offset + alignment - 1) / alignment * alignment;
        layout.regions.emplace_back(offset, size);
        offset += size;
    }
    layout.span = offset + max_gap + 1;
    return layout;
}

std::vector<uint64_t>
random_sizes(size_t count, uint64_t smallest, uint64_t largest, uint64_t seed)
{
    auto rng   = std::mt19937_64{seed};
    auto scale = std::uniform_real_distribution<double>{std::log(double(smallest)),
                                                        std::log(double(largest))};
    auto sizes = std::vector<uint64_t>{};
    for(size_t index = 0; index < count; ++index)
        sizes.push_back(static_cast<uint64_t>(std::exp(scale(rng))));
    return sizes;
}

std::vector<std::byte>
pattern(uint64_t size)
{
    auto bytes = std::vector<std::byte>(size);
    for(size_t index = 0; index < bytes.size(); ++index)
        bytes.at(index) = static_cast<std::byte>((index * 131 + 7) >> 3);
    return bytes;
}

// The digest blit-copy-kernel.hpp defines, worked out byte by byte on the (little-endian) host.
uint64_t
host_digest(const std::byte* data, uint64_t size)
{
    auto digest = uint64_t{0};
    for(auto offset = uint64_t{0}; offset < size; offset += 8)
    {
        auto word = uint64_t{0};
        std::memcpy(&word, data + offset, std::min<uint64_t>(8, size - offset));
        digest += abi::digest_term(word, offset / 8);
    }
    return digest;
}

size_t
count_mismatches(const std::vector<std::byte>& actual, const std::vector<std::byte>& expected)
{
    auto count = size_t{0};
    for(size_t index = 0; index < actual.size(); ++index)
        count += (actual.at(index) != expected.at(index));
    return count;
}

void
check_layout(const layout_t& layout)
{
    auto& gpu = gpu_t::get();
    if(!gpu.available()) GTEST_SKIP() << gpu.reason();

    const auto source = pattern(layout.span);
    auto       dest   = std::vector<std::byte>(layout.span, sentinel);

    auto* device_source = gpu.allocate_device(layout.span);
    auto* device_dest   = gpu.allocate_device(layout.span);
    ASSERT_NE(device_source, nullptr);
    ASSERT_NE(device_dest, nullptr);
    ASSERT_EQ(hsa_memory_copy(device_source, source.data(), layout.span), HSA_STATUS_SUCCESS);
    ASSERT_EQ(hsa_memory_copy(device_dest, dest.data(), layout.span), HSA_STATUS_SUCCESS);

    auto regions = std::vector<abi::copy_descriptor_t>{};
    for(const auto& [offset, size] : layout.regions)
        regions.push_back({size, address(device_source) + offset, address(device_dest) + offset});
    {
        const auto workspace = workspace_t{regions, false};
        ASSERT_TRUE(workspace.ok());
        ASSERT_TRUE(gpu.run(workspace.copy_steps())) << "the blit did not complete";
    }
    ASSERT_EQ(hsa_memory_copy(dest.data(), device_dest, layout.span), HSA_STATUS_SUCCESS);
    hsa_amd_memory_pool_free(device_source);
    hsa_amd_memory_pool_free(device_dest);

    auto covered = std::vector<bool>(layout.span, false);
    for(const auto& [offset, size] : layout.regions)
        std::fill_n(covered.begin() + offset, size, true);
    auto mismatches = size_t{0};
    for(size_t index = 0; index < layout.span && mismatches < 8; ++index)
    {
        const auto expected = covered.at(index) ? source.at(index) : sentinel;
        if(dest.at(index) == expected) continue;
        ++mismatches;
        ADD_FAILURE() << "byte " << index << (covered.at(index) ? " in a region" : " in a gap")
                      << " is " << int(dest.at(index)) << ", expected " << int(expected);
    }
    EXPECT_EQ(mismatches, 0);
}

// A capture with digests of `layout` from live device memory into a snapshot in device memory,
// or in host memory the GPU reads over PCIe like the pinned arena.
class capture_t
{
public:
    capture_t(const layout_t& layout, bool host_snapshot)
    : capture_t{layout, host_snapshot, pattern(layout.span)}
    {}

    capture_t(const layout_t& layout, bool host_snapshot, std::vector<std::byte> original)
    : m_original{std::move(original)}
    {
        auto& gpu = gpu_t::get();
        m_live    = gpu.allocate_device(layout.span);
        m_snapshot =
            host_snapshot ? gpu.allocate_pinned(layout.span) : gpu.allocate_device(layout.span);
        if(!m_live || !m_snapshot ||
           hsa_memory_copy(m_live, m_original.data(), layout.span) != HSA_STATUS_SUCCESS)
            return;

        auto regions = std::vector<abi::copy_descriptor_t>{};
        for(const auto& [offset, size] : layout.regions)
        {
            const auto region = abi::copy_descriptor_t{
                size, address(m_live) + offset, address(m_snapshot) + offset};
            regions.push_back(region);
            for(uint64_t index = 0; index < abi::tiles_for(size); ++index)
            {
                const auto tile = abi::tile_of(region, index);
                m_tiles.emplace_back(tile.source_address - address(m_live), tile.size);
            }
        }
        m_workspace = std::make_unique<workspace_t>(regions, true);
        m_ok        = m_workspace->ok() && gpu.run(m_workspace->copy_steps());
    }

    ~capture_t()
    {
        if(m_live) hsa_amd_memory_pool_free(m_live);
        if(m_snapshot) hsa_amd_memory_pool_free(m_snapshot);
    }

    capture_t(const capture_t&) = delete;
    capture_t& operator=(const capture_t&) = delete;

    bool                          ok() const { return m_ok; }
    const workspace_t&            workspace() const { return *m_workspace; }
    const std::vector<std::byte>& original() const { return m_original; }

    // Offset and size of each tile, in the order the kernels list them.
    const std::vector<std::pair<uint64_t, uint64_t>>& tiles() const { return m_tiles; }

    std::vector<std::byte> live() const
    {
        return read_device<std::byte>(m_live, m_original.size());
    }
    std::vector<std::byte> snapshot() const
    {
        return read_device<std::byte>(m_snapshot, m_original.size());
    }
    void write_live(const std::vector<std::byte>& bytes) const
    {
        hsa_memory_copy(m_live, bytes.data(), bytes.size());
    }

    // Changes one byte in each of `tiles` on top of `bytes`.
    std::vector<std::byte> touched(std::vector<std::byte>    bytes,
                                   const std::set<uint32_t>& tiles,
                                   std::byte                 value) const
    {
        for(const auto tile : tiles)
        {
            const auto [offset, size] = m_tiles.at(tile);
            bytes.at(offset + size / 2) ^= value;
        }
        return bytes;
    }

private:
    bool                                       m_ok       = false;
    std::vector<std::byte>                     m_original = {};
    void*                                      m_live     = nullptr;
    void*                                      m_snapshot = nullptr;
    std::vector<std::pair<uint64_t, uint64_t>> m_tiles    = {};
    std::unique_ptr<workspace_t>               m_workspace{};
};

std::set<uint32_t>
every(uint32_t step, uint32_t first, uint32_t count)
{
    auto tiles = std::set<uint32_t>{};
    for(auto tile = first; tile < count; tile += step)
        tiles.insert(tile);
    return tiles;
}

void
check_digests(const layout_t& layout)
{
    auto& gpu = gpu_t::get();
    if(!gpu.available()) GTEST_SKIP() << gpu.reason();

    const auto capture = capture_t{layout, false};
    ASSERT_TRUE(capture.ok());
    const auto snapshot = capture.snapshot();
    for(const auto& [offset, size] : layout.regions)
        ASSERT_TRUE(std::equal(snapshot.begin() + offset,
                               snapshot.begin() + offset + size,
                               capture.original().begin() + offset))
            << "region at " << offset << " was not captured";

    const auto digests = capture.workspace().digests();
    ASSERT_EQ(digests.size(), capture.tiles().size());
    for(size_t tile = 0; tile < digests.size(); ++tile)
    {
        const auto [offset, size] = capture.tiles().at(tile);
        ASSERT_EQ(digests.at(tile), host_digest(capture.original().data() + offset, size))
            << "tile " << tile << " at " << offset << ", " << size << " bytes";
    }
}

void
check_restore_changed(bool host_snapshot)
{
    auto& gpu = gpu_t::get();
    if(!gpu.available()) GTEST_SKIP() << gpu.reason();

    const auto capture =
        capture_t{make_layout(random_sizes(40, 1, 200000, 11), 16, 64, 12), host_snapshot};
    ASSERT_TRUE(capture.ok());
    const auto tile_count = capture.workspace().tile_count();
    const auto changed    = every(5, 0, tile_count);

    // The first restore, before pass 1, also clears the count the next one uses.
    capture.write_live(capture.touched(capture.original(), changed, std::byte{0x5a}));
    capture.workspace().set_counter(1, 77);
    ASSERT_TRUE(gpu.run({capture.workspace().restore_changed(0)}));
    EXPECT_EQ(count_mismatches(capture.live(), capture.original()), 0);
    auto counters = capture.workspace().counters();
    EXPECT_EQ(counters.at(0), changed.size());
    EXPECT_EQ(counters.at(1), 0);
    const auto listed = capture.workspace().changed(counters.at(0));
    EXPECT_EQ(std::set<uint32_t>(listed.begin(), listed.end()), changed);
    const auto flags = capture.workspace().flags();
    for(uint32_t tile = 0; tile < tile_count; ++tile)
        EXPECT_EQ(flags.at(tile) != 0, changed.count(tile) == 1) << "tile " << tile;

    // Hashing before every pass alternates between the two counts.
    const auto next = every(7, 3, tile_count);
    capture.write_live(capture.touched(capture.original(), next, std::byte{0x33}));
    ASSERT_TRUE(gpu.run({capture.workspace().restore_changed(1)}));
    EXPECT_EQ(count_mismatches(capture.live(), capture.original()), 0);
    counters = capture.workspace().counters();
    EXPECT_EQ(counters.at(0), 0);
    EXPECT_EQ(counters.at(1), next.size());
}

TEST(kernel_replay_blit_kernels, use_no_scratch_and_only_explicit_arguments)
{
    auto& gpu = gpu_t::get();
    if(!gpu.available()) GTEST_SKIP() << gpu.reason();
    for(size_t index = 0; index < gpu.kernels().size(); ++index)
    {
        EXPECT_EQ(gpu.kernels().at(index).private_segment_size, 0) << kernel_names.at(index);
        EXPECT_EQ(gpu.kernels().at(index).kernarg_size, kernel_arg_sizes.at(index))
            << kernel_names.at(index);
    }
}

TEST(kernel_replay_blit_kernels, partial_items_and_tile_tails)
{
    check_layout(make_layout(
        {1, 15, 16, 17, 4095, 4096, 4097, 16383, 16384, 16385, 32768, 49153}, 16, 64, 1));
}

TEST(kernel_replay_blit_kernels, regions_at_any_byte_offset)
{
    check_layout(make_layout({1, 3, 17, 255, 4097, 16385, 40961, 100003}, 1, 31, 2));
}

TEST(kernel_replay_blit_kernels, one_large_region)
{
    check_layout(make_layout({(std::uint64_t{64} << 20) + 7}, 16, 0, 3));
}

TEST(kernel_replay_blit_kernels, short_list_split_into_pieces)
{
    check_layout(make_layout({16384, 16384, 16384}, 4096, 4096, 4));
}

TEST(kernel_replay_blit_kernels, descriptors_across_chunks)
{
    check_layout(make_layout(std::vector<uint64_t>(abi::workgroup_size + 1, 4096), 4096, 4096, 5));
}

TEST(kernel_replay_blit_kernels, many_tiny_regions)
{
    check_layout(make_layout(random_sizes(5000, 1, 64, 6), 16, 48, 7));
}

TEST(kernel_replay_blit_kernels, mixed_sizes)
{
    check_layout(make_layout(random_sizes(500, 1, std::uint64_t{1} << 20, 8), 16, 256, 9));
}

TEST(kernel_replay_blit_kernels, capture_digests_partial_words_and_tile_tails)
{
    check_digests(make_layout(
        {1, 7, 8, 9, 15, 16, 17, 24, 4095, 4097, 16383, 16384, 16385, 49153}, 16, 64, 21));
}

TEST(kernel_replay_blit_kernels, capture_digests_at_any_byte_offset)
{
    check_digests(make_layout({1, 3, 17, 255, 4097, 16385, 40961, 100003}, 1, 31, 22));
}

TEST(kernel_replay_blit_kernels, capture_digests_tiles_split_into_pieces)
{
    check_digests(make_layout({16384, 16384, 9000}, 4096, 4096, 23));
}

TEST(kernel_replay_blit_kernels, capture_digests_large_and_tiny_regions)
{
    check_digests(make_layout(random_sizes(300, 1, std::uint64_t{4} << 20, 24), 16, 256, 25));
}

TEST(kernel_replay_blit_kernels, restore_all_runs_the_capture_list_backwards)
{
    auto& gpu = gpu_t::get();
    if(!gpu.available()) GTEST_SKIP() << gpu.reason();

    for(const auto host_snapshot : {false, true})
    {
        const auto layout  = make_layout(random_sizes(50, 1, 300000, 41), 16, 64, 42);
        const auto capture = capture_t{layout, host_snapshot};
        ASSERT_TRUE(capture.ok());

        auto written = capture.original();
        for(auto& byte : written)
            byte ^= std::byte{0xff};
        capture.write_live(written);
        ASSERT_TRUE(gpu.run({capture.workspace().restore_all()}));

        // The regions come back; the gaps between them keep what was written.
        auto expected = written;
        for(const auto& [offset, size] : layout.regions)
            std::copy_n(capture.original().begin() + offset, size, expected.begin() + offset);
        EXPECT_EQ(count_mismatches(capture.live(), expected), 0)
            << (host_snapshot ? "from a host snapshot" : "from a device snapshot");
    }
}

TEST(kernel_replay_blit_kernels, restore_changed_from_a_device_snapshot)
{
    check_restore_changed(false);
}

TEST(kernel_replay_blit_kernels, restore_changed_from_a_host_snapshot)
{
    check_restore_changed(true);
}

// Any one changed word must change its tile's digest, including {b - 1, 0} to {0xffffffff, b}
// (32-bit halves) at the start of a tile, which collide under swap-plus-product terms.
static_assert(abi::digest_term(0, 0) != abi::digest_term(0x00000001ffffffffULL, 0));
static_assert(abi::digest_term(4, 0) != abi::digest_term(0x00000005ffffffffULL, 0));

TEST(kernel_replay_blit_kernels, restore_changed_catches_a_single_changed_word)
{
    auto& gpu = gpu_t::get();
    if(!gpu.available()) GTEST_SKIP() << gpu.reason();

    const auto layout     = make_layout({3 * abi::tile_bytes}, 16, 0, 1);
    const auto write_word = [](std::vector<std::byte>& bytes, uint64_t offset, uint64_t word) {
        std::memcpy(bytes.data() + offset, &word, sizeof(word));
    };

    // Zeros, but for {4, 0} at the start of the second tile. Then the first word of the first two
    // tiles changes, and nothing else.
    auto original = std::vector<std::byte>(layout.span);
    write_word(original, abi::tile_bytes, 4);
    const auto capture = capture_t{layout, false, original};
    ASSERT_TRUE(capture.ok());
    auto written = original;
    write_word(written, 0, 0x00000001ffffffffULL);
    write_word(written, abi::tile_bytes, 0x00000005ffffffffULL);
    capture.write_live(written);

    ASSERT_TRUE(gpu.run({capture.workspace().restore_changed(0)}));
    EXPECT_EQ(count_mismatches(capture.live(), original), 0);
    const auto count = capture.workspace().counters().at(0);
    ASSERT_EQ(count, 2);
    const auto listed = capture.workspace().changed(count);
    EXPECT_EQ(std::set<uint32_t>(listed.begin(), listed.end()), (std::set<uint32_t>{0, 1}));
}

TEST(kernel_replay_blit_kernels, restore_listed_and_check_the_rest)
{
    auto& gpu = gpu_t::get();
    if(!gpu.available()) GTEST_SKIP() << gpu.reason();

    const auto capture = capture_t{make_layout(random_sizes(60, 1, 120000, 31), 16, 64, 32), false};
    ASSERT_TRUE(capture.ok());
    const auto tile_count = capture.workspace().tile_count();
    const auto listed     = every(4, 1, tile_count);
    capture.write_live(capture.touched(capture.original(), listed, std::byte{0x5a}));
    ASSERT_TRUE(gpu.run({capture.workspace().restore_changed(0)}));
    ASSERT_EQ(capture.workspace().counters().at(0), listed.size());

    // A later pass writes the same tiles. The check after it ignores them, and the list restores
    // them.
    capture.write_live(capture.touched(capture.original(), listed, std::byte{0x0f}));
    ASSERT_TRUE(gpu.run({capture.workspace().check()}));
    EXPECT_EQ(capture.workspace().counters().at(abi::unlisted_counter), 0);
    ASSERT_TRUE(gpu.run({capture.workspace().restore_listed(0)}));
    EXPECT_EQ(count_mismatches(capture.live(), capture.original()), 0);

    // One that also writes a tile off the list: the check counts it, and the list misses it.
    const auto unlisted = std::set<uint32_t>{0};
    auto       written  = capture.touched(capture.original(), listed, std::byte{0x0f});
    written             = capture.touched(written, unlisted, std::byte{0x40});
    capture.write_live(written);
    ASSERT_TRUE(gpu.run({capture.workspace().check()}));
    EXPECT_EQ(capture.workspace().counters().at(abi::unlisted_counter), 1);
    ASSERT_TRUE(gpu.run({capture.workspace().restore_listed(0)}));
    const auto live           = capture.live();
    const auto [offset, size] = capture.tiles().at(0);
    EXPECT_NE(live.at(offset + size / 2), capture.original().at(offset + size / 2));
    EXPECT_EQ(count_mismatches(live, capture.original()), 1);
}
}  // namespace
