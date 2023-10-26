/*
   Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#pragma once

#include <cstdint>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "lib/rocprofiler/pc_sampling/parser/translation.hpp"

#if 0
template <>
struct std::hash<device_handle>
{
    size_t operator()(const device_handle& d) const { return d.handle; }
};
bool
operator==(device_handle a, device_handle b)
{
    return a.handle == b.handle;
}
#endif
namespace Parser
{
/*
struct DispatchPkt
{
    uint64_t      write_id;  //! The location where this dispatch is written to
    uint64_t      doorbell_id;  //! The doorbell non-unique ID
    device_handle dev;          //! Which device this is run
}; */
struct DispatchPkt
{
    uint64_t      correlation_id_in;  //! Correlation ID seen by the trap handler
    device_handle dev;                //! Which device this is run
};
#if 0
bool
operator==(const DispatchPkt& a, const DispatchPkt& b)
{
    return a.correlation_id_in == b.correlation_id_in && a.dev == b.dev;
}
#endif
}  // namespace Parser

template <>
struct std::hash<Parser::DispatchPkt>
{
    size_t operator()(const Parser::DispatchPkt& d) const
    {
        return (d.correlation_id_in << 8) ^ d.dev.handle;
    }
};

namespace Parser
{
/**
 * Coordinates DispatchMap and DoorBellMap to reconstruct the original correlation_id
 * from the correlation_id seen by the trap handler.
 */
class CorrelationMap
{
public:
    CorrelationMap() = default;

    /**
     * Checks wether a dispatch pkt will generate a collision.
     * Returns true on collision and false when slot is available.
     */
    bool checkDispatch(const dispatch_pkt_id_t& pkt) const;

    /**
     * Updates the mapping of dispatch_id to correlation_id
     */
    void newDispatch(const dispatch_pkt_id_t& pkt);

    void forget(const dispatch_pkt_id_t& pkt);

    /**
     * Given a device dev, doorbell and and wrapped dispatch_id, returns the
     * correlation_id set by dispatch_pkt_id_t
     */
    uint64_t get(device_handle dev, uint64_t correlation_in);

    static uint64_t wrap_correlation_id(uint64_t doorbell, uint64_t write_idx, uint64_t queue_size);

private:
    std::unordered_map<DispatchPkt, uint64_t> dispatch_to_correlation{};

    // Making get() const and these cache variables mutable causes performance to be unstable
    uint64_t cache_correlation_id_in  = ~0ul;  // Invalid value in cache
    uint64_t cache_correlation_id_out = ~0ul;
    uint64_t cache_dev_id             = ~0ul;  // Invalid device Id in cache
};
}  // namespace Parser

template <bool bHostTrap, typename GFXIP>
inline pcsample_status_t
add_upcoming_samples(const device_handle     device,
                     const generic_sample_t* buffer,
                     const size_t            available_samples,
                     Parser::CorrelationMap* corr_map,
                     pcsample_v1_t*          samples)
{
    pcsample_status_t status = PCSAMPLE_STATUS_SUCCESS;
    for(uint64_t p = 0; p < available_samples; p++)
    {
        const auto* snap = reinterpret_cast<const perf_sample_snapshot_v1*>(buffer + p);
        samples[p]       = copySample<bHostTrap, GFXIP>((const void*) (buffer + p));
        try
        {
            samples[p].correlation_id = corr_map->get(device, snap->correlation_id);
        } catch(std::exception& e)
        {
            status = PCSAMPLE_STATUS_PARSER_ERROR;
        }
    }
    return status;
}

template <typename GFXIP>
pcsample_status_t
_parse_buffer(generic_sample_t*       buffer,
              uint64_t                buffer_size,
              user_callback_t         callback,
              void*                   userdata,
              Parser::CorrelationMap* corr_map)
{
    // Maximum size
    uint64_t index = 0;

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

                bool bIsHostTrap = pkt.which_sample_type == AMD_HOST_TRAP_V1;

                while(pkt_counter > 0)
                {
                    pcsample_v1_t* samples           = nullptr;
                    uint64_t       available_samples = callback(&samples, pkt_counter, userdata);

                    if(available_samples == 0 || available_samples > pkt_counter)
                        return PCSAMPLE_STATUS_CALLBACK_ERROR;

                    if(bIsHostTrap)
                    {
                        status |= add_upcoming_samples<true, GFXIP>(
                            pkt.device, buffer + index, available_samples, corr_map, samples);
                    }
                    else
                    {
                        status |= add_upcoming_samples<false, GFXIP>(
                            pkt.device, buffer + index, available_samples, corr_map, samples);
                    }

                    index += available_samples;
                    pkt_counter -= available_samples;
                }
                break;
            }
            default:
                std::cerr << "Index " << index << " - Invalid sample type: " << buffer[index].type
                          << std::endl;
                return PCSAMPLE_STATUS_INVALID_SAMPLE;
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
pcsample_status_t
parse_buffer(generic_sample_t* buffer,
             uint64_t          buffer_size,
             int               gfxip_major,
             user_callback_t   callback,
             void*             userdata);