// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "include/rocprofiler-sdk/cxx/codeobj/code_printing.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/code_object.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/translation.hpp"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

template <>
struct std::hash<device_handle>
{
    size_t operator()(const device_handle& d) const { return d.handle; }
};
bool inline
operator==(device_handle a, device_handle b)
{
    return a.handle == b.handle;
}

namespace Parser
{
struct dispatch_correlation_ids_t
{
    rocprofiler_dispatch_id_t          dispatch_id;
    rocprofiler_async_correlation_id_t correlation_id;
};

/**
 * @brief Struct immitating the correlation_id returned by the trap handler in raw PC samples.
 */
union trap_correlation_id_t
{
    uint64_t raw;
    struct
    {
        uint64_t dispatch_index : 25;
        uint64_t _reserved0     : 7;
        uint64_t doorbell_id    : 10;
        uint64_t _reserved1     : 22;
    } wrapped;
};

struct DispatchPkt
{
    trap_correlation_id_t correlation_id_in;  //! Correlation ID seen by the trap handler
    device_handle         dev;                //! Which device this is run
};

struct cache_type_t
{
    trap_correlation_id_t      id_in{.raw = ~0ul};
    dispatch_correlation_ids_t id_out{};
    uint64_t                   dev_id    = ~0ul;
    size_t                     increment = 0;
    size_t                     object_id = 0;
};

inline bool
operator==(const trap_correlation_id_t& a, const trap_correlation_id_t& b)
{
    return a.raw == b.raw;
}

inline bool
operator==(const DispatchPkt& a, const DispatchPkt& b)
{
    return a.correlation_id_in == b.correlation_id_in && a.dev == b.dev;
}
}  // namespace Parser

template <>
struct std::hash<Parser::DispatchPkt>
{
    size_t operator()(const Parser::DispatchPkt& d) const
    {
        return (d.correlation_id_in.raw << 8) ^ d.dev.handle;
    }
};

namespace Parser
{
// 64B for performance reasons
constexpr auto pcs_parser_sample_record_size = 64;
static_assert(sizeof(generic_sample_t) == pcs_parser_sample_record_size);
static_assert(sizeof(generic_sample_t) == sizeof(perf_sample_snapshot_v1));
static_assert(sizeof(generic_sample_t) == sizeof(perf_sample_host_trap_v1));
static_assert(sizeof(generic_sample_t) == sizeof(upcoming_samples_t));
static_assert(sizeof(generic_sample_t) == sizeof(dispatch_pkt_id_t));

/**
 * Coordinates DispatchMap and DoorBellMap to reconstruct the original correlation_id
 * from the correlation_id seen by the trap handler.
 */
class CorrelationMap
{
public:
    CorrelationMap()
    {
        static std::atomic<size_t> _ids{1};
        object_id = _ids.fetch_add(1);
    };

    /**
     * Checks wether a dispatch pkt will generate a collision.
     * @returns true on collision and false when slot is available.
     */
    bool checkDispatch(const dispatch_pkt_id_t& pkt) const
    {
        auto trap = trap_correlation_id(pkt.doorbell_id, pkt.write_index, pkt.queue_size);
        return dispatch_to_correlation.find({trap, pkt.device}) != dispatch_to_correlation.end();
    }

    /**
     * @brief Updates the mapping of dispatch_id to correlation_id
     */
    void newDispatch(const dispatch_pkt_id_t& pkt)
    {
        std::unique_lock<std::mutex> lk(mut);
        auto trap_id = trap_correlation_id(pkt.doorbell_id, pkt.write_index, pkt.queue_size);
        dispatch_to_correlation[{trap_id, pkt.device}] = {pkt.dispatch_id, pkt.correlation_id};
        cache_reset_count.fetch_add(1);
    }

    /**
     * @brief Allows the parser to forget a correlation_id, to save memory.
     */
    void forget(const dispatch_pkt_id_t& pkt)
    {
        std::unique_lock<std::mutex> lk(mut);
        auto trap_id = trap_correlation_id(pkt.doorbell_id, pkt.write_index, pkt.queue_size);
        dispatch_to_correlation.erase({trap_id, pkt.device});
        cache_reset_count.fetch_add(1);
    }

    /**
     * Given a device dev, doorbell and and wrapped dispatch_id,
     * @returns the correlation_id set by dispatch_pkt_id_t
     */
    dispatch_correlation_ids_t get(device_handle dev, trap_correlation_id_t correlation_in)
    {
#ifndef _PARSER_CORRELATION_DISABLE_CACHE
        static thread_local cache_type_t cache{};
        size_t                           new_increment = cache_reset_count.load();

        if(cache.increment == new_increment && cache.object_id == this->object_id &&
           cache.dev_id == dev.handle && cache.id_in == correlation_in)
            return cache.id_out;

        // Using unique_lock showed better performance over the shared_lock
        std::unique_lock<std::mutex> lk(mut);
        cache.increment = cache_reset_count.load();
        cache.object_id = object_id;
        cache.id_out    = dispatch_to_correlation.at({correlation_in, dev});
        cache.dev_id    = dev.handle;
        cache.id_in     = correlation_in;
        return cache.id_out;
#else
        std::unique_lock<std::mutex> lk(mut);
        return dispatch_to_correlation.at({correlation_in, dev});
#endif
    }

    /**
     * Returns the correlation_id as seen by the trap handler, consisting of a
     * - wrapped dispatch_pkt
     * - doorbell_id divibed by 8 Bytes
     * @param[in] doorbell The doorbell handler returned by HSA
     * @param[in] write_idx The dispatch packet write index, [optional] not wrapped
     * @param[in] queue_size The queue size. [optional] If write_index is already wrapped,
     *                       then this value can just be a large integer > queue_size.
     * @returns The correlation_id immitating the ones returned by the trap handler.
     */
    static trap_correlation_id_t trap_correlation_id(uint64_t doorbell,
                                                     uint64_t write_idx,
                                                     uint64_t queue_size)
    {
        trap_correlation_id_t trap{.raw = 0};
        trap.wrapped.dispatch_index = write_idx % queue_size;
        trap.wrapped.doorbell_id    = doorbell >> 3;
        return trap;
    }

private:
    std::unordered_map<DispatchPkt, dispatch_correlation_ids_t> dispatch_to_correlation{};
    std::atomic<size_t>                                         cache_reset_count{1};
    size_t                                                      object_id = 0;

    std::mutex mut;
};
}  // namespace Parser

using address_range_t = rocprofiler::sdk::codeobj::segment::address_range_t;

template <typename GFXIP, typename PcSamplingRecordT>
inline pcsample_status_t
add_upcoming_samples(const device_handle     device,
                     const generic_sample_t* buffer,
                     const size_t            available_samples,
                     Parser::CorrelationMap* corr_map,
                     PcSamplingRecordT*      samples)
{
    pcsample_status_t status           = PCSAMPLE_STATUS_SUCCESS;
    auto              cache_addr_range = address_range_t{0, 0, ROCPROFILER_CODE_OBJECT_ID_NONE};

    auto* table = rocprofiler::pc_sampling::code_object::CodeobjTableTranslatorSynchronized::Get();
    // To achieve better performance, we exported mutex outside of the translator class.
    table->clear_backlog();
    auto table_read_lock = table->acquire_query_lock();

    for(uint64_t p = 0; p < available_samples; p++)
    {
        const auto* snap = reinterpret_cast<const perf_sample_snapshot_v1*>(buffer + p);

        auto& pc_sample = samples[p];
        pc_sample       = copySample<GFXIP, PcSamplingRecordT>(static_cast<const void*>(snap));
        // skip invalid samples
        if(pc_sample.size == 0) continue;

        // Correct PC address of the original sample (if needed) prior to decoding it.
        auto pc_address = correct_pc_address<GFXIP, PcSamplingRecordT>(snap);

        // Convert PC -> (loaded code object id containing PC, offset within code object)
        if(!cache_addr_range.inrange(pc_address.value))
            cache_addr_range = table->find_codeobj_in_range(pc_address.value);

        pc_sample.pc.code_object_id     = cache_addr_range.id;
        pc_sample.pc.code_object_offset = pc_address.value - cache_addr_range.addr;

        try
        {
            Parser::trap_correlation_id_t trap{.raw = snap->correlation_id};
            auto                          dispatch_correlation_ids = corr_map->get(device, trap);
            pc_sample.dispatch_id    = dispatch_correlation_ids.dispatch_id;
            pc_sample.correlation_id = dispatch_correlation_ids.correlation_id;

            if(pc_sample.pc.code_object_id == ROCPROFILER_CODE_OBJECT_ID_NONE &&
               pc_sample.correlation_id.internal != ROCPROFILER_CORRELATION_ID_INTERNAL_NONE)
            {
                // We observed the stochastic sampling error samples, that was not being
                // tag as the error sample due to high contention in the trap handler.
                // Thus, we declare sample invalid, by setting its size to zero.
                pc_sample.size = 0;
            }
        } catch(std::exception& e)
        {
            // TODO: introduce ROCPROFILER_DISPATCH_ID_INTERNAL_NONE
            pc_sample.dispatch_id    = 0;
            pc_sample.correlation_id = {.internal = ROCPROFILER_CORRELATION_ID_INTERNAL_NONE,
                                        .external = rocprofiler_user_data_t{
                                            .value = ROCPROFILER_CORRELATION_ID_INTERNAL_NONE}};
            status                   = PCSAMPLE_STATUS_PARSER_ERROR;
        }
    }
    return status;
}

template <typename GFXIP, typename PcSamplingRecordT>
inline pcsample_status_t
_parse_buffer(generic_sample_t*                  buffer,
              uint64_t                           buffer_size,
              user_callback_t<PcSamplingRecordT> callback,
              void*                              userdata,
              Parser::CorrelationMap*            corr_map)
{
    // Maximum size
    uint64_t          index  = 0;
    pcsample_status_t status = PCSAMPLE_STATUS_SUCCESS;

    while(index < buffer_size)
    {
        switch(buffer[index].type)
        {
            case AMD_DISPATCH_PKT_ID:
            {
                const auto& pkt = *reinterpret_cast<const dispatch_pkt_id_t*>(buffer + index);
                if(pkt.queue_size >= (1 << 25)) status = PCSAMPLE_STATUS_PARSER_ERROR;
                index += 1;
                corr_map->newDispatch(pkt);
                break;
            }
            case AMD_UPCOMING_SAMPLES:
            {
                const auto& pkt = *reinterpret_cast<const upcoming_samples_t*>(buffer + index);
                index += 1;

                uint64_t pkt_counter = pkt.num_samples;
                if(index + pkt_counter > buffer_size) return PCSAMPLE_STATUS_OUT_OF_BOUNDS_ERROR;

                // I don't think we need this.
                // bool bIsHostTrap = pkt.which_sample_type == AMD_HOST_TRAP_V1;

                while(pkt_counter > 0)
                {
                    PcSamplingRecordT* samples = nullptr;
                    uint64_t available_samples = callback(&samples, pkt_counter, userdata);

                    if(available_samples == 0 || available_samples > pkt_counter)
                        return PCSAMPLE_STATUS_CALLBACK_ERROR;

                    // I don't think we need if-else here
                    // if(bIsHostTrap)
                    // {
                    //     status |= add_upcoming_samples<GFXIP>(
                    //         pkt.device, buffer + index, available_samples, corr_map, samples);
                    // }
                    // else
                    // {
                    //     status |= add_upcoming_samples<GFXIP>(
                    //         pkt.device, buffer + index, available_samples, corr_map, samples);
                    // }

                    status |= add_upcoming_samples<GFXIP>(
                        pkt.device, buffer + index, available_samples, corr_map, samples);

                    index += available_samples;
                    pkt_counter -= available_samples;
                }
                break;
            }
            default: return PCSAMPLE_STATUS_INVALID_SAMPLE;
        }
    }
    return status;
};

/**
 * @brief Parses a given set of pc samples.
 * @param[in] buffer Pointer to a buffer containing metadata and pcsamples.
 * @param[in] buffer_size The number of elements in the buffer.
 * @param[in] gfxip_major GFXIP major version of the samples.
 * @param[in] callback A callback function that accepts a double pointer to write the samples to,
 * a size requested parameter (number of pc_sample_t) and a void* to userdata.
 * The callback is expected to allocate 64B-aligned memory where the parsed samples are going to
 * be written to, and return the size of memory that was allocated, in multiples of
 * sizeof(generic_sample_t). If the callback returns 0 or a larger size than requested,
 * parse_buffer() will return PCSAMPLE_STATUS_CALLBACK_ERROR. If the callback returns
 * a size smaller than requested, then it may be called again requesting more memory.
 * @param[in] userdata parameter forwarded to the user callback.
 */
template <typename PcSamplingRecordT>
pcsample_status_t inline parse_buffer(generic_sample_t*                  buffer,
                                      uint64_t                           buffer_size,
                                      int                                gfxip_major,
                                      int                                gfxip_minor,
                                      user_callback_t<PcSamplingRecordT> callback,
                                      void*                              userdata)
{
    static auto corr_map = std::make_unique<Parser::CorrelationMap>();

    auto parseSample_func = _parse_buffer<GFX9, PcSamplingRecordT>;
    if(gfxip_major == 9)
    {
        if(gfxip_minor == 5)
        {
            parseSample_func = _parse_buffer<GFX950, PcSamplingRecordT>;
        }
    }
    else if(gfxip_major == 11)
    {
        parseSample_func = _parse_buffer<GFX11, PcSamplingRecordT>;
    }
    else
    {
        return PCSAMPLE_STATUS_INVALID_GFXIP;
    }

    return parseSample_func(buffer, buffer_size, callback, userdata, corr_map.get());
};
