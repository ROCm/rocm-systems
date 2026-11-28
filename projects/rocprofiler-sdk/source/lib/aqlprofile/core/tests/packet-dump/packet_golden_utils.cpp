// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "packet_golden_utils.hpp"

#include <gtest/gtest.h>

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <list>
#include <new>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>

namespace aqlprofile
{
namespace packet_dump
{
size_t
system_page_size()
{
    static const size_t value = []() {
        const long result = ::sysconf(_SC_PAGESIZE);
        if(result <= 0) throw std::runtime_error{"failed to query the system page size"};
        return static_cast<size_t>(result);
    }();
    return value;
}

std::vector<uintptr_t>
relocated_allocation_bases(uint64_t seed, size_t random_count)
{
    const size_t page_size = system_page_size();
    if(deterministic_address_base % page_size != 0 || deterministic_address_stride % page_size != 0)
        throw std::runtime_error{"deterministic allocation addresses are not page aligned"};

    std::mt19937_64                         generator{seed};
    std::uniform_int_distribution<uint64_t> high_slot_distribution{1, 1024};
    std::uniform_int_distribution<uint64_t> low_page_distribution{
        1, (deterministic_address_stride / page_size) - 1};

    std::vector<uintptr_t> bases = {
        deterministic_address_base + page_size,
        deterministic_address_base + deterministic_address_stride - page_size,
        deterministic_address_base + deterministic_address_stride,
        deterministic_address_base + deterministic_address_stride + page_size,
        deterministic_address_base + deterministic_address_stride + 2 * page_size,
    };
    for(size_t i = 0; i < random_count; ++i)
        bases.emplace_back(deterministic_address_base +
                           (high_slot_distribution(generator) * deterministic_address_stride) +
                           (low_page_distribution(generator) * page_size));
    return bases;
}

// gtest derives the seed from the clock unless --gtest_random_seed is passed.
uint64_t
relocation_seed()
{
    const int seed = ::testing::UnitTest::GetInstance()->random_seed();
    std::cout << "relocation seed: " << seed << " (rerun with --gtest_random_seed=" << seed << ")"
              << std::endl;
    return static_cast<uint64_t>(seed);
}

uint32_t
load_le_dword(const uint8_t* bytes)
{
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
}

uintptr_t
aql_ib_target(const hsa_ext_amd_aql_pm4_packet_t& packet)
{
    const auto* bytes = reinterpret_cast<const uint8_t*>(&packet);
    return static_cast<uintptr_t>((uint64_t{load_le_dword(bytes + 12)} << 32) |
                                  (load_le_dword(bytes + 8) & 0xFFFFFFFCu));
}

void
expect_only_ib_target_moved(const hsa_ext_amd_aql_pm4_packet_t& baseline,
                            uintptr_t                           baseline_command_base,
                            const hsa_ext_amd_aql_pm4_packet_t& moved,
                            uintptr_t                           moved_command_base)
{
    constexpr size_t ib_address_lo_dword = 2;
    constexpr size_t ib_address_hi_dword = 3;
    const auto*      baseline_bytes      = reinterpret_cast<const uint8_t*>(&baseline);
    const auto*      moved_bytes         = reinterpret_cast<const uint8_t*>(&moved);
    for(size_t dword = 0; dword < sizeof(baseline) / sizeof(uint32_t); ++dword)
    {
        if(dword == ib_address_hi_dword) continue;
        const uint32_t mask = dword == ib_address_lo_dword ? 0x3u : 0xFFFFFFFFu;
        EXPECT_EQ(load_le_dword(baseline_bytes + dword * sizeof(uint32_t)) & mask,
                  load_le_dword(moved_bytes + dword * sizeof(uint32_t)) & mask)
            << "dword " << dword;
    }
    EXPECT_EQ(aql_ib_target(moved) - moved_command_base,
              aql_ib_target(baseline) - baseline_command_base);
}

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
    const size_t page_size = system_page_size();
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

struct allocation_observation
{
    void*                          pointer = nullptr;
    size_t                         size    = 0;
    aqlprofile_buffer_desc_flags_t flags{};
};

struct copy_observation
{
    void*                destination = nullptr;
    std::vector<uint8_t> bytes{};
};

struct packet_capture
{
    explicit packet_capture(uintptr_t base)
    : address_base{base}
    {}

    uintptr_t                           address_base         = 0;
    size_t                              allocation_count     = 0;
    size_t                              deallocation_count   = 0;
    bool                                invalid_deallocation = false;
    std::list<fixed_base_vector>        allocations{};
    std::vector<allocation_observation> allocation_records{};
    std::vector<copy_observation>       copies{};
};

hsa_status_t
allocate_memory(void**                         ptr,
                uint64_t                       size,
                aqlprofile_buffer_desc_flags_t flags,
                void*                          userdata) noexcept
{
    auto* capture = static_cast<packet_capture*>(userdata);
    if(!capture || !ptr || size == 0 || size > std::numeric_limits<size_t>::max())
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    *ptr = nullptr;

    try
    {
        const auto address = deterministic_address(
            capture->address_base, capture->allocation_count, static_cast<size_t>(size));
        if(!address) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

        auto& allocation = capture->allocations.emplace_back(
            static_cast<size_t>(size), uint8_t{0}, fixed_base_allocator<uint8_t>{*address});
        *ptr = allocation.data();
        capture->allocation_records.push_back({*ptr, static_cast<size_t>(size), flags});
        ++capture->allocation_count;
        return HSA_STATUS_SUCCESS;
    } catch(const std::bad_alloc&)
    {
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    } catch(...)
    {
        return HSA_STATUS_ERROR;
    }
}

void
deallocate_memory(void* ptr, void* userdata) noexcept
{
    auto* capture = static_cast<packet_capture*>(userdata);
    if(!capture) return;
    const auto allocation = std::find_if(capture->allocations.begin(),
                                         capture->allocations.end(),
                                         [ptr](const auto& entry) { return entry.data() == ptr; });
    if(allocation == capture->allocations.end())
    {
        capture->invalid_deallocation = true;
        return;
    }
    capture->allocations.erase(allocation);
    ++capture->deallocation_count;
}

hsa_status_t
memcpy_memory(void* dst, const void* src, size_t size, void* userdata) noexcept
{
    auto* capture = static_cast<packet_capture*>(userdata);
    if(!capture || (!dst && size != 0) || (!src && size != 0)) return HSA_STATUS_ERROR;

    try
    {
        const auto* bytes = static_cast<const uint8_t*>(src);
        capture->copies.push_back({dst, {bytes, bytes + size}});
        std::memcpy(dst, src, size);
        return HSA_STATUS_SUCCESS;
    } catch(const std::bad_alloc&)
    {
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    } catch(...)
    {
        return HSA_STATUS_ERROR;
    }
}

std::optional<std::pair<size_t, size_t>>
locate_allocation(const packet_capture& capture, const void* pointer, size_t size)
{
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    for(size_t i = 0; i < capture.allocation_records.size(); ++i)
    {
        const auto& allocation = capture.allocation_records.at(i);
        const auto  base       = reinterpret_cast<uintptr_t>(allocation.pointer);
        if(address >= base && address - base <= allocation.size &&
           size <= allocation.size - (address - base))
            return std::pair<size_t, size_t>{i, address - base};
    }
    return std::nullopt;
}

struct command_stream_layout
{
    size_t                          command_index = 0;
    std::vector<packet_allocation>  allocations{};
    std::vector<packet_memory_copy> memory_copies{};
};

command_stream_layout
classify_command_streams(const packet_capture&                                  capture,
                         const std::vector<std::pair<packet_phase, uintptr_t>>& ib_targets)
{
    if(capture.copies.size() != ib_targets.size())
        throw std::runtime_error{"aqlprofile generated an unexpected number of command streams"};
    if(capture.allocation_records.size() != 2)
        throw std::runtime_error{"aqlprofile generated an unexpected number of allocations"};

    constexpr size_t      missing_index = std::numeric_limits<size_t>::max();
    command_stream_layout layout{missing_index};
    std::vector<bool>     phase_found(ib_targets.size(), false);
    for(size_t i = 0; i < capture.copies.size(); ++i)
    {
        const auto& copy     = capture.copies.at(i);
        const auto  location = locate_allocation(capture, copy.destination, copy.bytes.size());
        if(!location) throw std::runtime_error{"memory copy destination is not allocated"};
        if(layout.command_index == missing_index)
            layout.command_index = location->first;
        else if(layout.command_index != location->first)
            throw std::runtime_error{"command streams use different allocations"};

        const auto target =
            std::find_if(ib_targets.begin(), ib_targets.end(), [&copy](const auto& entry) {
                return entry.second == reinterpret_cast<uintptr_t>(copy.destination);
            });
        if(target == ib_targets.end())
            throw std::runtime_error{"memory copy does not match an AQL IB target"};
        phase_found.at(static_cast<size_t>(target - ib_targets.begin())) = true;
        layout.memory_copies.push_back(
            {target->first, i, location->first, location->second, copy.bytes.size()});
    }
    if(std::find(phase_found.begin(), phase_found.end(), false) != phase_found.end())
        throw std::runtime_error{"failed to identify packet command streams"};

    for(size_t i = 0; i < capture.allocation_records.size(); ++i)
    {
        const auto& allocation = capture.allocation_records.at(i);
        layout.allocations.push_back({
            i == layout.command_index ? packet_allocation_role::command
                                      : packet_allocation_role::output,
            i,
            reinterpret_cast<uintptr_t>(allocation.pointer),
            allocation.size,
            allocation.flags,
        });
    }
    return layout;
}

std::vector<uint8_t>
take_phase_stream(packet_capture&                        capture,
                  const std::vector<packet_memory_copy>& copies,
                  packet_phase                           phase)
{
    const auto copy = std::find_if(
        copies.begin(), copies.end(), [phase](const auto& entry) { return entry.phase == phase; });
    if(copy == copies.end()) throw std::runtime_error{"missing packet phase stream"};
    return std::move(capture.copies.at(copy->observed_call_index).bytes);
}

void
verify_allocations_released(const packet_capture& capture)
{
    if(capture.invalid_deallocation ||
       capture.deallocation_count != capture.allocation_records.size() ||
       !capture.allocations.empty())
        throw std::runtime_error{
            "aqlprofile did not deallocate every packet allocation exactly once"};
}

struct sample_capture
{
    aqlprofile_agent_handle_t  agent{};
    size_t                     slot_count = 0;
    std::vector<bool>          reported{};
    std::vector<packet_sample> samples{};
};

struct sample_coordinate_capture
{
    packet_sample* sample = nullptr;
    hsa_status_t   status = HSA_STATUS_SUCCESS;
};

hsa_status_t
collect_sample_coordinate(int         position,
                          int         id,
                          int         extent,
                          int         coordinate,
                          const char* name,
                          void*       userdata) noexcept
{
    auto* capture = static_cast<sample_coordinate_capture*>(userdata);
    if(!capture || capture->status != HSA_STATUS_SUCCESS) return HSA_STATUS_SUCCESS;
    if(!capture->sample || position < 0 ||
       static_cast<size_t>(position) != capture->sample->coordinates.size() || !name)
    {
        capture->status = HSA_STATUS_ERROR;
        return HSA_STATUS_SUCCESS;
    }

    try
    {
        capture->sample->coordinates.push_back({name,
                                                static_cast<uint32_t>(position),
                                                static_cast<uint32_t>(id),
                                                static_cast<uint64_t>(extent),
                                                static_cast<uint64_t>(coordinate)});
    } catch(const std::bad_alloc&)
    {
        capture->status = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    } catch(...)
    {
        capture->status = HSA_STATUS_ERROR;
    }
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
collect_sample(aqlprofile_pmc_event_t event,
               uint64_t               counter_id,
               uint64_t               counter_value,
               void*                  userdata) noexcept
{
    try
    {
        constexpr uint64_t tag_mask   = 0xFFFF000000000000;
        constexpr uint64_t tag_prefix = 0xA5C0000000000000;
        if((counter_value & tag_mask) != tag_prefix) return HSA_STATUS_ERROR;

        auto* capture = static_cast<sample_capture*>(userdata);
        if(!capture) return HSA_STATUS_ERROR;
        const size_t slot = static_cast<size_t>(counter_value & ~tag_mask);
        if(slot >= capture->slot_count || capture->reported.at(slot)) return HSA_STATUS_ERROR;

        auto& sample               = capture->samples.emplace_back();
        sample.event               = event;
        sample.counter_id          = counter_id;
        sample.callback_index      = capture->samples.size() - 1;
        sample.output_offset_bytes = slot * sizeof(uint64_t);

        sample_coordinate_capture coordinates{&sample};
        const hsa_status_t        status = aqlprofile_iterate_event_coord(
            capture->agent, event, counter_id, collect_sample_coordinate, &coordinates);
        if(status != HSA_STATUS_SUCCESS) return status;
        if(coordinates.status != HSA_STATUS_SUCCESS) return coordinates.status;

        capture->reported.at(slot) = true;
        return HSA_STATUS_SUCCESS;
    } catch(const std::bad_alloc&)
    {
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    } catch(...)
    {
        return HSA_STATUS_ERROR;
    }
}

hsa_status_t
collect_dimension(int position, int id, int extent, int, const char* name, void* userdata)
{
    static_cast<std::vector<packet_dimension>*>(userdata)->push_back(
        {name,
         static_cast<uint32_t>(position),
         static_cast<uint32_t>(id),
         static_cast<uint64_t>(extent)});
    return HSA_STATUS_SUCCESS;
}
}  // namespace

pmc_packet_streams
generate_pmc_packet_streams(aqlprofile_agent_handle_t     agent,
                            const aqlprofile_pmc_event_t& event,
                            uintptr_t                     allocation_base,
                            bool                          capture_samples)
{
    return generate_pmc_packet_streams(
        agent, std::vector<aqlprofile_pmc_event_t>{event}, allocation_base, capture_samples);
}

pmc_packet_streams
generate_pmc_packet_streams(aqlprofile_agent_handle_t                  agent,
                            const std::vector<aqlprofile_pmc_event_t>& events,
                            uintptr_t                                  allocation_base,
                            bool                                       capture_samples)
{
    if(events.empty() || events.size() > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument{"PMC event list is empty or too large"};

    const aqlprofile_pmc_profile_t profile = {
        agent,
        events.data(),
        static_cast<uint32_t>(events.size()),
    };

    packet_capture               capture{allocation_base};
    aqlprofile_handle_t          handle{};
    aqlprofile_pmc_aql_packets_t packets{};
    const hsa_status_t           status = aqlprofile_pmc_create_packets(
        &handle, &packets, profile, allocate_memory, deallocate_memory, memcpy_memory, &capture);
    if(status != HSA_STATUS_SUCCESS)
        throw std::runtime_error{"aqlprofile_pmc_create_packets failed with status " +
                                 std::to_string(static_cast<int>(status))};

    bool packets_deleted = false;
    try
    {
        auto layout =
            classify_command_streams(capture,
                                     {
                                         {packet_phase::read, aql_ib_target(packets.read_packet)},
                                         {packet_phase::start, aql_ib_target(packets.start_packet)},
                                         {packet_phase::stop, aql_ib_target(packets.stop_packet)},
                                     });

        pmc_packet_streams result{};
        result.aql           = packets;
        result.command_base  = layout.allocations.at(layout.command_index).base;
        result.allocations   = std::move(layout.allocations);
        result.memory_copies = std::move(layout.memory_copies);

        const size_t output_index = layout.command_index == 0 ? 1 : 0;
        if(capture_samples)
        {
            constexpr uint64_t tag_prefix = 0xA5C0000000000000;
            auto&              output     = capture.allocation_records.at(output_index);
            const size_t       slot_count = output.size / sizeof(uint64_t);
            for(size_t slot = 0; slot < slot_count; ++slot)
            {
                const uint64_t tag = tag_prefix | slot;
                std::memcpy(
                    static_cast<uint8_t*>(output.pointer) + slot * sizeof(tag), &tag, sizeof(tag));
            }

            sample_capture     samples{agent, slot_count, std::vector<bool>(slot_count, false), {}};
            const hsa_status_t iterate_status =
                aqlprofile_pmc_iterate_data(handle, collect_sample, &samples);
            if(iterate_status != HSA_STATUS_SUCCESS)
                throw std::runtime_error{"aqlprofile_pmc_iterate_data failed with status " +
                                         std::to_string(static_cast<int>(iterate_status))};
            result.samples = std::move(samples.samples);
            for(size_t slot = 0; slot < samples.reported.size(); ++slot)
                if(!samples.reported.at(slot))
                    result.unreported_output_offsets.emplace_back(slot * sizeof(uint64_t));
        }

        result.read  = take_phase_stream(capture, result.memory_copies, packet_phase::read);
        result.start = take_phase_stream(capture, result.memory_copies, packet_phase::start);
        result.stop  = take_phase_stream(capture, result.memory_copies, packet_phase::stop);

        aqlprofile_pmc_delete_packets(handle);
        packets_deleted = true;
        verify_allocations_released(capture);
        return result;
    } catch(...)
    {
        if(!packets_deleted) aqlprofile_pmc_delete_packets(handle);
        throw;
    }
}

spm_packet_streams
generate_spm_packet_streams(aqlprofile_agent_handle_t                      agent,
                            const std::vector<aqlprofile_pmc_event_t>&     events,
                            const std::vector<aqlprofile_spm_parameter_t>& parameters,
                            uintptr_t                                      allocation_base)
{
    if(events.empty()) throw std::invalid_argument{"SPM event list is empty"};

    packet_capture                          capture{allocation_base};
    std::vector<aqlprofile_spm_parameter_t> profile_parameters = parameters;
    const aqlprofile_spm_profile_t          profile{
        agent,
        hsa_agent_t{0},
        events.data(),
        events.size(),
        profile_parameters.data(),
        profile_parameters.size(),
        0,
        allocate_memory,
        deallocate_memory,
        memcpy_memory,
        &capture,
    };

    constexpr uint64_t           unassigned_handle = std::numeric_limits<uint64_t>::max();
    aqlprofile_handle_t          handle{unassigned_handle};
    aqlprofile_spm_buffer_desc_t descriptor{};
    aqlprofile_spm_aql_packets_t packets{};
    const hsa_status_t           status =
        aqlprofile_spm_create_packets(&handle, &descriptor, &packets, profile, 0);
    if(status != HSA_STATUS_SUCCESS)
    {
        if(handle.handle != unassigned_handle) aqlprofile_spm_delete_packets(handle);
        throw std::runtime_error{"aqlprofile_spm_create_packets failed with status " +
                                 std::to_string(static_cast<int>(status))};
    }

    bool packets_deleted = false;
    try
    {
        auto layout =
            classify_command_streams(capture,
                                     {
                                         {packet_phase::start, aql_ib_target(packets.start_packet)},
                                         {packet_phase::stop, aql_ib_target(packets.stop_packet)},
                                     });
        const auto descriptor_location =
            locate_allocation(capture, descriptor.data, descriptor.size);
        if(!descriptor_location || descriptor_location->first == layout.command_index)
            throw std::runtime_error{"SPM descriptor is not inside the output allocation"};

        spm_packet_streams result{};
        result.aql                         = packets;
        result.command_base                = layout.allocations.at(layout.command_index).base;
        result.allocations                 = std::move(layout.allocations);
        result.memory_copies               = std::move(layout.memory_copies);
        result.descriptor_allocation_index = descriptor_location->first;
        result.descriptor_offset_bytes     = descriptor_location->second;

        const auto* descriptor_bytes = static_cast<const uint8_t*>(descriptor.data);
        result.descriptor.assign(descriptor_bytes, descriptor_bytes + descriptor.size);
        for(size_t query = 0; query < result.decode_queries.size(); ++query)
        {
            const hsa_status_t query_status =
                aqlprofile_spm_decode_query(descriptor,
                                            static_cast<aqlprofile_spm_decode_query_t>(query),
                                            &result.decode_queries.at(query));
            if(query_status != HSA_STATUS_SUCCESS)
                throw std::runtime_error{"aqlprofile_spm_decode_query failed with status " +
                                         std::to_string(static_cast<int>(query_status))};
        }

        result.start = take_phase_stream(capture, result.memory_copies, packet_phase::start);
        result.stop  = take_phase_stream(capture, result.memory_copies, packet_phase::stop);

        aqlprofile_spm_delete_packets(handle);
        packets_deleted = true;
        verify_allocations_released(capture);
        return result;
    } catch(...)
    {
        if(!packets_deleted) aqlprofile_spm_delete_packets(handle);
        throw;
    }
}

bool
operator==(const packet_allocation& lhs, const packet_allocation& rhs)
{
    return lhs.role == rhs.role && lhs.observed_call_index == rhs.observed_call_index &&
           lhs.base == rhs.base && lhs.size_bytes == rhs.size_bytes &&
           lhs.flags.raw == rhs.flags.raw;
}

bool
operator==(const packet_memory_copy& lhs, const packet_memory_copy& rhs)
{
    return lhs.phase == rhs.phase && lhs.observed_call_index == rhs.observed_call_index &&
           lhs.allocation_index == rhs.allocation_index && lhs.offset_bytes == rhs.offset_bytes &&
           lhs.size_bytes == rhs.size_bytes;
}

bool
same_sample_layout(const pmc_packet_streams& lhs, const pmc_packet_streams& rhs)
{
    if(lhs.samples.size() != rhs.samples.size()) return false;
    for(size_t i = 0; i < lhs.samples.size(); ++i)
    {
        const auto& lhs_sample = lhs.samples.at(i);
        const auto& rhs_sample = rhs.samples.at(i);
        if(lhs_sample.counter_id != rhs_sample.counter_id ||
           lhs_sample.callback_index != rhs_sample.callback_index ||
           lhs_sample.output_offset_bytes != rhs_sample.output_offset_bytes ||
           lhs_sample.coordinates.size() != rhs_sample.coordinates.size())
            return false;

        for(size_t coordinate = 0; coordinate < lhs_sample.coordinates.size(); ++coordinate)
        {
            const auto& lhs_coordinate = lhs_sample.coordinates.at(coordinate);
            const auto& rhs_coordinate = rhs_sample.coordinates.at(coordinate);
            if(lhs_coordinate.name != rhs_coordinate.name ||
               lhs_coordinate.position != rhs_coordinate.position ||
               lhs_coordinate.id != rhs_coordinate.id ||
               lhs_coordinate.extent != rhs_coordinate.extent ||
               lhs_coordinate.coordinate != rhs_coordinate.coordinate)
                return false;
        }
    }
    return true;
}

std::vector<pm4_packet_range>
split_pm4_stream(const std::vector<uint8_t>& bytes)
{
    if(bytes.size() % sizeof(uint32_t) != 0)
        throw std::runtime_error{"PM4 stream size is not dword aligned"};

    std::vector<uint32_t> dwords(bytes.size() / sizeof(uint32_t));
    std::memcpy(dwords.data(), bytes.data(), bytes.size());

    constexpr uint32_t packet_type_3 = 3;
    constexpr uint32_t packet_3_nop  = 0x10;

    std::vector<pm4_packet_range> packets{};
    size_t                        offset = 0;
    while(offset < dwords.size())
    {
        const uint32_t header = dwords.at(offset);
        if((header >> 30) != packet_type_3)
            throw std::runtime_error{"unexpected non-Type-3 PM4 packet"};

        const uint32_t count         = (header >> 16) & 0x3FFF;
        const uint32_t opcode        = (header >> 8) & 0xFF;
        const size_t   packet_dwords = (opcode == packet_3_nop && count == 0x3FFF) ? 1 : count + 2;
        if(packet_dwords > dwords.size() - offset) throw std::runtime_error{"truncated PM4 packet"};

        packets.push_back({offset * sizeof(uint32_t), packet_dwords * sizeof(uint32_t), opcode});
        offset += packet_dwords;
    }
    return packets;
}

namespace
{
struct read_address_field
{
    size_t dword_offset = 0;
    // ACQUIRE_MEM stores its base address shifted right by 8.
    bool shifted_by_256 = false;
};

std::vector<uint32_t>
stream_dwords(const std::vector<uint8_t>& bytes)
{
    if(bytes.size() % sizeof(uint32_t) != 0)
        throw std::runtime_error{"PM4 stream size is not dword aligned"};

    std::vector<uint32_t> dwords(bytes.size() / sizeof(uint32_t));
    std::memcpy(dwords.data(), bytes.data(), bytes.size());
    return dwords;
}

std::vector<read_address_field>
pmc_read_address_fields(const std::vector<uint8_t>& bytes, const std::vector<uint32_t>& dwords)
{
    constexpr uint32_t packet_3_copy             = 0x40;
    constexpr uint32_t packet_3_acquire          = 0x58;
    constexpr uint32_t copy_dst_sel_gfx12_memory = 2;
    constexpr uint32_t copy_dst_sel_gfx9_memory  = 5;

    std::vector<read_address_field> fields{};
    for(const auto& packet : split_pm4_stream(bytes))
    {
        const size_t   offset        = packet.byte_offset / sizeof(uint32_t);
        const size_t   packet_dwords = packet.byte_size / sizeof(uint32_t);
        const uint32_t copy_dst_sel  = packet_dwords > 1 ? ((dwords.at(offset + 1) >> 8) & 0xF) : 0;
        const bool     copy_to_memory =
            copy_dst_sel == copy_dst_sel_gfx12_memory || copy_dst_sel == copy_dst_sel_gfx9_memory;
        if(packet.opcode == packet_3_copy && copy_to_memory)
        {
            if(packet_dwords < 6) throw std::runtime_error{"short COPY_DATA packet"};
            fields.push_back({offset + 4, false});
        }
        else if(packet.opcode == packet_3_acquire)
        {
            if(packet_dwords < 7) throw std::runtime_error{"short ACQUIRE_MEM packet"};
            fields.push_back({offset + 4, true});
        }
    }
    return fields;
}
}  // namespace

std::vector<uint32_t>
normalize_pmc_read_addresses(const std::vector<uint8_t>& bytes)
{
    auto dwords = stream_dwords(bytes);
    for(const auto& field : pmc_read_address_fields(bytes, dwords))
    {
        dwords.at(field.dword_offset)     = 0;
        dwords.at(field.dword_offset + 1) = 0;
    }
    return dwords;
}

std::vector<uintptr_t>
pmc_read_addresses(const std::vector<uint8_t>& bytes)
{
    const auto             dwords = stream_dwords(bytes);
    std::vector<uintptr_t> addresses{};
    for(const auto& field : pmc_read_address_fields(bytes, dwords))
    {
        const uint64_t value = uint64_t{dwords.at(field.dword_offset)} |
                               (uint64_t{dwords.at(field.dword_offset + 1)} << 32);
        addresses.push_back(static_cast<uintptr_t>(field.shifted_by_256 ? value << 8 : value));
    }
    return addresses;
}

aqlprofile_agent_info_v2_t
packet_agent_info(const std::string& gfx)
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
    return agent_info;
}

aqlprofile_agent_handle_t
register_packet_agent(const std::string& gfx)
{
    const auto                agent_info = packet_agent_info(gfx);
    aqlprofile_agent_handle_t agent{};
    const hsa_status_t        status =
        aqlprofile_register_agent_info(&agent, &agent_info, AQLPROFILE_AGENT_VERSION_V2);
    if(status != HSA_STATUS_SUCCESS)
        throw std::runtime_error{"aqlprofile_register_agent_info failed with status " +
                                 std::to_string(static_cast<int>(status))};
    return agent;
}

packet_block
query_block(aqlprofile_agent_handle_t agent, const std::string& block)
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
    return {static_cast<hsa_ven_amd_aqlprofile_block_name_t>(query.id), query.instance_count};
}

hsa_ven_amd_aqlprofile_block_name_t
resolve_block(aqlprofile_agent_handle_t agent, const std::string& block)
{
    return query_block(agent, block).id;
}

uint32_t
block_counter_count(aqlprofile_agent_handle_t agent, hsa_ven_amd_aqlprofile_block_name_t block)
{
    const aqlprofile_pmc_event_t event{
        0,
        0,
        {},
        block,
    };
    const aqlprofile_pmc_profile_t profile{
        agent,
        &event,
        1,
    };
    uint32_t           count = 0;
    const hsa_status_t status =
        aqlprofile_get_pmc_info(&profile, AQLPROFILE_INFO_BLOCK_COUNTERS, &count);
    if(status != HSA_STATUS_SUCCESS)
        throw std::runtime_error{"failed to query block counter count"};
    return count;
}

std::vector<packet_dimension>
event_dimensions(aqlprofile_agent_handle_t agent, const aqlprofile_pmc_event_t& event)
{
    std::vector<packet_dimension> dimensions{};
    const hsa_status_t            status =
        aqlprofile_iterate_event_coord(agent, event, 0, collect_dimension, &dimensions);
    if(status != HSA_STATUS_SUCCESS)
        throw std::runtime_error{"aqlprofile_iterate_event_coord failed with status " +
                                 std::to_string(static_cast<int>(status))};
    return dimensions;
}

uint32_t
block_instance_count(const std::vector<packet_dimension>& dimensions)
{
    const auto instance =
        std::find_if(dimensions.begin(), dimensions.end(), [](const auto& dimension) {
            return dimension.name == "INSTANCE";
        });
    if(instance == dimensions.end()) return 1;
    if(instance->extent > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error{"block instance count exceeds uint32_t"};
    return static_cast<uint32_t>(instance->extent);
}

std::vector<aqlprofile_pmc_event_t>
all_instance_events(hsa_ven_amd_aqlprofile_block_name_t block,
                    uint32_t                            event_id,
                    uint32_t                            instance_count)
{
    std::vector<aqlprofile_pmc_event_t> events{};
    events.reserve(instance_count);
    for(uint32_t block_index = 0; block_index < instance_count; ++block_index)
    {
        auto& event       = events.emplace_back();
        event.block_index = block_index;
        event.event_id    = event_id;
        event.block_name  = block;
    }
    return events;
}

}  // namespace packet_dump
}  // namespace aqlprofile
