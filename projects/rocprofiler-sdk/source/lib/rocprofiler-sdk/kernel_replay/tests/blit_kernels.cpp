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
// dispatches with only the explicit kernel arguments written. Each layout is restored once and
// checked byte for byte, including the gaps between regions.

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
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace
{
namespace abi = rocprofiler::kernel_replay::blit::kernel_abi;

constexpr auto kernel_names = std::array<const char*, 4>{
    "kernel_replay_count_tiles",
    "kernel_replay_scan_tiles",
    "kernel_replay_expand_tiles",
    "kernel_replay_copy_tiles",
};

constexpr auto kernel_arg_sizes = std::array<uint32_t, 4>{
    sizeof(abi::count_tiles_args_t),
    sizeof(abi::scan_tiles_args_t),
    sizeof(abi::expand_tiles_args_t),
    sizeof(abi::copy_tiles_args_t),
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

    void* allocate_device(size_t size) const
    {
        void* ptr = nullptr;
        if(hsa_amd_memory_pool_allocate(m_device_pool, std::max<size_t>(size, 64), 0, &ptr) !=
           HSA_STATUS_SUCCESS)
            return nullptr;
        return ptr;
    }

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

    // Copies `regions` with the four kernels and returns whether the copy completed in time.
    bool restore(const std::vector<abi::copy_descriptor_t>& regions) const;

private:
    gpu_t();

    std::string              m_reason{};
    hsa_agent_t              m_gpu{};
    hsa_amd_memory_pool_t    m_device_pool{};
    hsa_amd_memory_pool_t    m_kernarg_pool{};
    hsa_amd_memory_pool_t    m_staging_pool{};
    hsa_queue_t*             m_queue    = nullptr;
    uint32_t                 m_cu_count = 0;
    std::array<kernel_t, 4>  m_kernels{};
    hsa_executable_t         m_executable{};
    hsa_code_object_reader_t m_reader{};
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
gpu_t::restore(const std::vector<abi::copy_descriptor_t>& regions) const
{
    const auto descriptor_count = static_cast<uint32_t>(regions.size());
    auto       total_tiles      = uint64_t{0};
    for(const auto& region : regions)
        total_tiles += abi::tiles_for(region.size);
    const auto tile_count = static_cast<uint32_t>(total_tiles);

    const auto descriptors_size = regions.size() * sizeof(abi::copy_descriptor_t);
    auto*      descriptors      = allocate_device(descriptors_size);
    auto*      first_tiles      = allocate_device(size_t{descriptor_count} * sizeof(uint32_t));
    auto*      chunk_totals =
        allocate_device(size_t{abi::chunks_for(descriptor_count)} * sizeof(uint32_t));
    auto*      tile_count_buffer = allocate_device(sizeof(uint32_t));
    auto*      tiles    = allocate_device(size_t{tile_count} * sizeof(abi::copy_descriptor_t));
    auto*      kernargs = allocate_host(m_kernarg_pool, kernarg_stride * m_kernels.size());
    auto*      staging  = allocate_host(m_staging_pool, descriptors_size);
    auto       done     = hsa_signal_t{};
    const auto release  = [&] {
        if(done.handle != 0) hsa_signal_destroy(done);
        for(void* ptr :
            {descriptors, first_tiles, chunk_totals, tile_count_buffer, tiles, kernargs, staging})
            if(ptr) hsa_amd_memory_pool_free(ptr);
    };
    if(!descriptors || !first_tiles || !chunk_totals || !tile_count_buffer || !tiles || !kernargs ||
       !staging || hsa_signal_create(1, 0, nullptr, &done) != HSA_STATUS_SUCCESS)
    {
        release();
        return false;
    }

    std::memcpy(staging, regions.data(), descriptors_size);
    if(hsa_memory_copy(descriptors, staging, descriptors_size) != HSA_STATUS_SUCCESS)
    {
        release();
        return false;
    }

    const auto address           = [](const void* ptr) { return reinterpret_cast<uint64_t>(ptr); };
    const auto chunks            = abi::chunks_for(descriptor_count);
    const auto copy_workgroups   = m_cu_count * abi::copy_workgroups_per_cu;
    const auto expand_workgroups = std::min(
        static_cast<uint32_t>((total_tiles + abi::workgroup_size - 1) / abi::workgroup_size),
        copy_workgroups);
    const auto workgroups =
        std::array<uint32_t, 4>{chunks, chunks, expand_workgroups, copy_workgroups};

    auto*      args = static_cast<std::byte*>(kernargs);
    const auto count_args =
        abi::count_tiles_args_t{address(descriptors), address(chunk_totals), descriptor_count, 0};
    const auto scan_args   = abi::scan_tiles_args_t{address(descriptors),
                                                  address(chunk_totals),
                                                  address(first_tiles),
                                                  address(tile_count_buffer),
                                                  descriptor_count,
                                                  chunks};
    const auto expand_args = abi::expand_tiles_args_t{address(descriptors),
                                                      address(first_tiles),
                                                      address(tile_count_buffer),
                                                      address(tiles),
                                                      descriptor_count,
                                                      expand_workgroups};
    const auto copy_args =
        abi::copy_tiles_args_t{address(tiles), address(tile_count_buffer), copy_workgroups, 0};
    std::memcpy(args + 0 * kernarg_stride, &count_args, sizeof(count_args));
    std::memcpy(args + 1 * kernarg_stride, &scan_args, sizeof(scan_args));
    std::memcpy(args + 2 * kernarg_stride, &expand_args, sizeof(expand_args));
    std::memcpy(args + 3 * kernarg_stride, &copy_args, sizeof(copy_args));

    // The same packet chain as the SDK: every dispatch waits for the previous one, system scope
    // at the two ends, and the copy signals completion.
    const auto first_index = hsa_queue_add_write_index_scacq_screl(m_queue, m_kernels.size());
    while(first_index + m_kernels.size() - hsa_queue_load_read_index_scacquire(m_queue) >
          m_queue->size)
    {}
    auto* packets = static_cast<hsa_kernel_dispatch_packet_t*>(m_queue->base_address);
    auto  headers = std::array<uint16_t, 4>{};
    for(size_t index = 0; index < m_kernels.size(); ++index)
    {
        const auto& kernel          = m_kernels.at(index);
        const auto  first           = (index == 0);
        const auto  last            = (index + 1 == m_kernels.size());
        auto&       packet          = packets[(first_index + index) & (m_queue->size - 1)];
        packet.reserved0            = 0;
        packet.reserved2            = 0;
        packet.workgroup_size_x     = static_cast<uint16_t>(abi::workgroup_size);
        packet.workgroup_size_y     = 1;
        packet.workgroup_size_z     = 1;
        packet.grid_size_x          = workgroups.at(index) * abi::workgroup_size;
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
    for(size_t index = 0; index < m_kernels.size(); ++index)
    {
        auto* word =
            reinterpret_cast<uint32_t*>(&packets[(first_index + index) & (m_queue->size - 1)]);
        __atomic_store_n(word, uint32_t{headers.at(index)} | (uint32_t{1} << 16), __ATOMIC_RELEASE);
    }
    hsa_signal_store_screlease(m_queue->doorbell_signal, first_index + m_kernels.size() - 1);

    using clock         = std::chrono::steady_clock;
    const auto deadline = clock::now() + std::chrono::seconds{60};
    auto       value    = hsa_signal_value_t{1};
    while(value != 0 && clock::now() < deadline)
        value = hsa_signal_wait_scacquire(
            done, HSA_SIGNAL_CONDITION_EQ, 0, 1000000000, HSA_WAIT_STATE_BLOCKED);

    release();
    return value == 0;
}

// Source and destination spans with the regions at the same offsets in both. The destination
// starts as a sentinel, so a copy outside a region shows up in the gaps.
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

void
check_layout(const layout_t& layout)
{
    auto& gpu = gpu_t::get();
    if(!gpu.available()) GTEST_SKIP() << gpu.reason();

    auto source = std::vector<std::byte>(layout.span);
    for(size_t index = 0; index < source.size(); ++index)
        source.at(index) = static_cast<std::byte>((index * 131 + 7) >> 3);
    auto dest = std::vector<std::byte>(layout.span, sentinel);

    auto* device_source = gpu.allocate_device(layout.span);
    auto* device_dest   = gpu.allocate_device(layout.span);
    ASSERT_NE(device_source, nullptr);
    ASSERT_NE(device_dest, nullptr);
    ASSERT_EQ(hsa_memory_copy(device_source, source.data(), layout.span), HSA_STATUS_SUCCESS);
    ASSERT_EQ(hsa_memory_copy(device_dest, dest.data(), layout.span), HSA_STATUS_SUCCESS);

    auto regions = std::vector<abi::copy_descriptor_t>{};
    for(const auto& [offset, size] : layout.regions)
        regions.push_back({size,
                           reinterpret_cast<uint64_t>(device_source) + offset,
                           reinterpret_cast<uint64_t>(device_dest) + offset});
    ASSERT_TRUE(gpu.restore(regions)) << "the blit did not complete";
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
}  // namespace
