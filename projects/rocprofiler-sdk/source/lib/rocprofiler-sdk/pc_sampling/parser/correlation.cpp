// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "correlation.hpp"

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

namespace Parser
{
bool
operator==(const DispatchPkt& a, const DispatchPkt& b)
{
    return a.correlation_id_in == b.correlation_id_in && a.dev == b.dev;
}
}  // namespace Parser

namespace Parser
{
/**
 * Coordinates DispatchMap and DoorBellMap to reconstruct the original correlation_id
 * from the correlation_id seen by the trap handler.
 */

/**
 * Checks wether a dispatch pkt will generate a collision.
 * Returns true on collision and false when slot is available.
 */
bool
CorrelationMap::checkDispatch(const dispatch_pkt_id_t& pkt) const
{
    uint64_t trap = wrap_correlation_id(pkt.doorbell_id, pkt.write_index, pkt.queue_size);
    return dispatch_to_correlation.find({trap, pkt.device}) != dispatch_to_correlation.end();
}

/**
 * Updates the mapping of dispatch_id to correlation_id
 */
void
CorrelationMap::newDispatch(const dispatch_pkt_id_t& pkt)
{
    cache_dev_id     = ~0ul;
    uint64_t trap_id = wrap_correlation_id(pkt.doorbell_id, pkt.write_index, pkt.queue_size);
    dispatch_to_correlation[{trap_id, pkt.device}] = pkt.correlation_id;
}

void
CorrelationMap::forget(const dispatch_pkt_id_t& pkt)
{
    cache_dev_id     = ~0ul;
    uint64_t trap_id = wrap_correlation_id(pkt.doorbell_id, pkt.write_index, pkt.queue_size);
    dispatch_to_correlation.erase({trap_id, pkt.device});
}

/**
 * Given a device dev, doorbell and and wrapped dispatch_id, returns the
 * correlation_id set by dispatch_pkt_id_t
 */
uint64_t
CorrelationMap::get(device_handle dev, uint64_t correlation_in)
{
#ifndef _PARSER_CORRELATION_DISABLE_CACHE
    if(dev.handle == cache_dev_id && correlation_in == cache_correlation_id_in)
        return cache_correlation_id_out;
#endif
    cache_dev_id             = dev.handle;
    cache_correlation_id_in  = correlation_in;
    cache_correlation_id_out = dispatch_to_correlation.at({correlation_in, dev});
    return cache_correlation_id_out;
}

uint64_t
CorrelationMap::wrap_correlation_id(uint64_t doorbell, uint64_t write_idx, uint64_t queue_size)
{
    static constexpr uint64_t WRITE_WRAP = (1 << 25) - 1;
    return ((write_idx % queue_size) & WRITE_WRAP) | (uint64_t(doorbell) << 32);
}

}  // namespace Parser

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
             void*             userdata)
{
    static auto corr_map = std::make_unique<Parser::CorrelationMap>();

    auto parseSample_func = _parse_buffer<GFX9>;
    if(gfxip_major == 9)
        parseSample_func = _parse_buffer<GFX9>;
    else if(gfxip_major == 11)
        parseSample_func = _parse_buffer<GFX11>;
    else if(gfxip_major == 0)
        parseSample_func = _parse_buffer<gfx_unknown>;
    else
        return PCSAMPLE_STATUS_INVALID_GFXIP;

    return parseSample_func(buffer, buffer_size, callback, userdata, corr_map.get());
};
