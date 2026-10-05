// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "packet_golden_utils.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <list>
#include <new>
#include <optional>
#include <stdexcept>

namespace aqlprofile
{
namespace packet_dump
{
namespace
{
std::optional<uintptr_t>
deterministic_address(uintptr_t base, size_t index, size_t size)
{
    if(size == 0 || size > deterministic_address_stride) return std::nullopt;
    if(base >= userspace_address_limit ||
       index > (userspace_address_limit - base) / deterministic_address_stride)
        return std::nullopt;

    const uintptr_t address = base + (index * deterministic_address_stride);
    if(address >= userspace_address_limit || size > userspace_address_limit - address)
        return std::nullopt;
    return address;
}

size_t
page_aligned_size(size_t size)
{
    static const size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    return ((size + page_size - 1) / page_size) * page_size;
}

template <typename Tp>
class fixed_base_allocator
{
public:
    using value_type = Tp;

    explicit fixed_base_allocator(uintptr_t address)
    : address{address}
    {}

    template <typename Up>
    fixed_base_allocator(const fixed_base_allocator<Up>& rhs)
    : address{rhs.address}
    {}

    Tp* allocate(size_t count)
    {
        if(count > std::numeric_limits<size_t>::max() / sizeof(Tp))
            throw std::bad_array_new_length{};

        const size_t size              = count * sizeof(Tp);
        void* const  requested_address = reinterpret_cast<void*>(address);
        void* const  mapping           = ::mmap(requested_address,
                                     page_aligned_size(size),
                                     PROT_READ | PROT_WRITE,
                                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
                                     -1,
                                     0);
        if(mapping == MAP_FAILED) throw std::bad_alloc{};
        if(mapping != requested_address)
        {
            ::munmap(mapping, page_aligned_size(size));
            throw std::bad_alloc{};
        }
        return static_cast<Tp*>(mapping);
    }

    void deallocate(Tp* ptr, size_t count) { ::munmap(ptr, page_aligned_size(count * sizeof(Tp))); }

    template <typename Up>
    bool operator==(const fixed_base_allocator<Up>& rhs) const
    {
        return address == rhs.address;
    }

    template <typename Up>
    bool operator!=(const fixed_base_allocator<Up>& rhs) const
    {
        return !(*this == rhs);
    }

    uintptr_t address = 0;
};

using fixed_base_vector = std::vector<uint8_t, fixed_base_allocator<uint8_t>>;

struct packet_capture
{
    explicit packet_capture(uintptr_t base)
    : address_base{base}
    {}

    uintptr_t                         address_base     = 0;
    size_t                            allocation_count = 0;
    std::list<fixed_base_vector>      allocations{};
    std::vector<std::vector<uint8_t>> copies{};
};

hsa_status_t
allocate_memory(void** ptr, uint64_t size, aqlprofile_buffer_desc_flags_t, void* userdata)
{
    auto* capture = static_cast<packet_capture*>(userdata);
    if(size == 0 || size > std::numeric_limits<size_t>::max())
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

    try
    {
        const auto address = deterministic_address(
            capture->address_base, capture->allocation_count, static_cast<size_t>(size));
        if(!address) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

        auto& allocation = capture->allocations.emplace_back(
            static_cast<size_t>(size), uint8_t{0}, fixed_base_allocator<uint8_t>{*address});
        *ptr = allocation.data();
        ++capture->allocation_count;
        return HSA_STATUS_SUCCESS;
    } catch(const std::bad_alloc&)
    {
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
}

void
deallocate_memory(void* ptr, void* userdata)
{
    auto*      capture    = static_cast<packet_capture*>(userdata);
    const auto allocation = std::find_if(capture->allocations.begin(),
                                         capture->allocations.end(),
                                         [ptr](const auto& entry) { return entry.data() == ptr; });
    if(allocation != capture->allocations.end()) capture->allocations.erase(allocation);
}

hsa_status_t
memcpy_memory(void* dst, const void* src, size_t size, void* userdata)
{
    auto*       capture = static_cast<packet_capture*>(userdata);
    const auto* bytes   = static_cast<const uint8_t*>(src);
    capture->copies.emplace_back(bytes, bytes + size);
    std::memcpy(dst, src, size);
    return HSA_STATUS_SUCCESS;
}
}  // namespace

pmc_packet_streams
generate_pmc_packet_streams(aqlprofile_agent_handle_t     agent,
                            const aqlprofile_pmc_event_t& event,
                            uintptr_t                     allocation_base)
{
    const aqlprofile_pmc_profile_t profile = {
        agent,
        &event,
        1,
    };

    packet_capture               capture{allocation_base};
    aqlprofile_handle_t          handle{};
    aqlprofile_pmc_aql_packets_t packets{};
    const hsa_status_t           status = aqlprofile_pmc_create_packets(
        &handle, &packets, profile, allocate_memory, deallocate_memory, memcpy_memory, &capture);
    if(status != HSA_STATUS_SUCCESS)
        throw std::runtime_error{"aqlprofile_pmc_create_packets failed with status " +
                                 std::to_string(static_cast<int>(status))};
    if(capture.copies.size() != 3)
    {
        aqlprofile_pmc_delete_packets(handle);
        throw std::runtime_error{"aqlprofile generated an unexpected number of command streams"};
    }

    pmc_packet_streams result{
        std::move(capture.copies.at(1)),
        std::move(capture.copies.at(0)),
        std::move(capture.copies.at(2)),
    };
    aqlprofile_pmc_delete_packets(handle);
    return result;
}

std::vector<uint32_t>
normalize_pmc_read_addresses(const std::vector<uint8_t>& bytes)
{
    if(bytes.size() % sizeof(uint32_t) != 0)
        throw std::runtime_error{"PM4 stream size is not dword aligned"};

    std::vector<uint32_t> dwords(bytes.size() / sizeof(uint32_t));
    std::memcpy(dwords.data(), bytes.data(), bytes.size());

    constexpr uint32_t packet_type_3             = 3;
    constexpr uint32_t packet_3_nop              = 0x10;
    constexpr uint32_t packet_3_copy             = 0x40;
    constexpr uint32_t packet_3_acquire          = 0x58;
    constexpr uint32_t copy_dst_sel_gfx12_memory = 2;
    constexpr uint32_t copy_dst_sel_gfx9_memory  = 5;

    size_t offset = 0;
    while(offset < dwords.size())
    {
        const uint32_t header = dwords.at(offset);
        if((header >> 30) != packet_type_3)
            throw std::runtime_error{"unexpected non-Type-3 PM4 packet"};

        const uint32_t count         = (header >> 16) & 0x3FFF;
        const uint32_t opcode        = (header >> 8) & 0xFF;
        const size_t   packet_dwords = (opcode == packet_3_nop && count == 0x3FFF) ? 1 : count + 2;
        if(packet_dwords > dwords.size() - offset) throw std::runtime_error{"truncated PM4 packet"};

        const uint32_t copy_dst_sel = packet_dwords > 1 ? ((dwords.at(offset + 1) >> 8) & 0xF) : 0;
        const bool     copy_to_memory =
            copy_dst_sel == copy_dst_sel_gfx12_memory || copy_dst_sel == copy_dst_sel_gfx9_memory;
        if(opcode == packet_3_copy && copy_to_memory)
        {
            if(packet_dwords < 6) throw std::runtime_error{"short COPY_DATA packet"};
            dwords.at(offset + 4) = 0;
            dwords.at(offset + 5) = 0;
        }
        else if(opcode == packet_3_acquire)
        {
            if(packet_dwords < 7) throw std::runtime_error{"short ACQUIRE_MEM packet"};
            dwords.at(offset + 4) = 0;
            dwords.at(offset + 5) = 0;
        }
        offset += packet_dwords;
    }
    return dwords;
}

aqlprofile_agent_handle_t
register_packet_agent(const std::string& gfx)
{
    aqlprofile_agent_info_v2_t agent_info{};
    agent_info.agent_gfxip = gfx.c_str();
    if(gfx == "gfx1201")
    {
        agent_info.xcc_num              = 1;
        agent_info.se_num               = 4;
        agent_info.cu_num               = 64;
        agent_info.shader_arrays_per_se = 2;
        agent_info.domain               = 0;
        agent_info.location_id          = 0x8300;
        for(uint32_t se = 0; se < agent_info.se_num; ++se)
            for(uint32_t sa = 0; sa < agent_info.shader_arrays_per_se; ++sa)
                agent_info.cu_bitmap.bits[se][sa] = 0xFF;
    }
    else if(gfx == "gfx942")
    {
        agent_info.xcc_num              = 8;
        agent_info.se_num               = 32;
        agent_info.cu_num               = 304;
        agent_info.shader_arrays_per_se = 1;
        agent_info.domain               = 0;
        agent_info.location_id          = 0x0C00;
    }
    else
    {
        throw std::runtime_error{"unsupported packet-test architecture " + gfx};
    }

    aqlprofile_agent_handle_t agent{};
    const hsa_status_t        status =
        aqlprofile_register_agent_info(&agent, &agent_info, AQLPROFILE_AGENT_VERSION_V2);
    if(status != HSA_STATUS_SUCCESS)
        throw std::runtime_error{"aqlprofile_register_agent_info failed with status " +
                                 std::to_string(static_cast<int>(status))};
    return agent;
}

hsa_ven_amd_aqlprofile_block_name_t
resolve_block(aqlprofile_agent_handle_t agent, const std::string& block)
{
    const aqlprofile_pmc_profile_t profile = {
        agent,
        nullptr,
        0,
    };
    hsa_ven_amd_aqlprofile_id_query_t query = {
        block.c_str(),
        0,
        0,
    };
    const hsa_status_t status = aqlprofile_get_pmc_info(&profile, AQLPROFILE_INFO_BLOCK_ID, &query);
    if(status != HSA_STATUS_SUCCESS) throw std::runtime_error{"failed to resolve block " + block};
    return static_cast<hsa_ven_amd_aqlprofile_block_name_t>(query.id);
}

}  // namespace packet_dump
}  // namespace aqlprofile
