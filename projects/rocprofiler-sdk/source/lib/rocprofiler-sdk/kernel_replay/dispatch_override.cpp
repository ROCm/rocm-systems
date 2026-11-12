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

#include "lib/rocprofiler-sdk/kernel_replay/dispatch_override.hpp"

#include "lib/common/utility.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rocprofiler
{
namespace kernel_replay
{
namespace
{
enum class packet_kind
{
    other,
    dispatch,
    ext_dispatch,
};

packet_kind
kind_of(const hsa::rocprofiler_packet& pkt)
{
    const auto type = (pkt.kernel_dispatch.header >> HSA_PACKET_HEADER_TYPE) &
                      ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1);
    if(type == HSA_PACKET_TYPE_KERNEL_DISPATCH) return packet_kind::dispatch;
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
    if(type == HSA_PACKET_TYPE_VENDOR_SPECIFIC &&
       pkt.ext_kernel_dispatch.amd_format == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH)
        return packet_kind::ext_dispatch;
#endif
    return packet_kind::other;
}

bool
nonzero(const rocprofiler_dim3_t& v)
{
    return v.x != 0 && v.y != 0 && v.z != 0;
}

bool
all_zero(const rocprofiler_dim3_t& v)
{
    return v.x == 0 && v.y == 0 && v.z == 0;
}

bool
fits_u16(const rocprofiler_dim3_t& v)
{
    constexpr auto max = std::numeric_limits<uint16_t>::max();
    return v.x <= max && v.y <= max && v.z <= max;
}

// The dispatch dimension count the grid and workgroup need, but never fewer than the application's
// packet declared: the kernel may read its dimension count, and lowering it would change what it
// sees.
uint16_t
setup_with_dimensions(uint16_t app_setup, const rocprofiler_kernel_replay_dispatch_t& dispatch)
{
    constexpr uint16_t dims_mask = (1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS) - 1;

    uint16_t dims = 1;
    if(dispatch.grid_size.y > 1 || dispatch.workgroup_size.y > 1) dims = 2;
    if(dispatch.grid_size.z > 1 || dispatch.workgroup_size.z > 1) dims = 3;

    const uint16_t app_dims =
        (app_setup >> HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS) & dims_mask;
    if(app_dims > dims) dims = app_dims;

    return static_cast<uint16_t>(
        (app_setup & ~(dims_mask << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS)) |
        (dims << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS));
}
}  // namespace

std::optional<rocprofiler_kernel_replay_dispatch_t>
describe_dispatch(const hsa::rocprofiler_packet& pkt)
{
    auto dispatch = common::init_public_api_struct(rocprofiler_kernel_replay_dispatch_t{});
    switch(kind_of(pkt))
    {
        case packet_kind::dispatch:
        {
            const auto& s            = pkt.kernel_dispatch;
            dispatch.kernel_object   = s.kernel_object;
            dispatch.kernarg_address = s.kernarg_address;
            dispatch.workgroup_size  = {s.workgroup_size_x, s.workgroup_size_y, s.workgroup_size_z};
            dispatch.grid_size       = {s.grid_size_x, s.grid_size_y, s.grid_size_z};
            dispatch.cluster_size    = {0, 0, 0};
            dispatch.group_segment_size   = s.group_segment_size;
            dispatch.private_segment_size = s.private_segment_size;
            return dispatch;
        }
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
        case packet_kind::ext_dispatch:
        {
            const auto& e            = pkt.ext_kernel_dispatch;
            dispatch.kernel_object   = e.kernel_object;
            dispatch.kernarg_address = e.kernarg_address;
            dispatch.workgroup_size  = {e.workgroup_size_x, e.workgroup_size_y, e.workgroup_size_z};
            dispatch.cluster_size    = {e.cluster_size_x, e.cluster_size_y, e.cluster_size_z};
            dispatch.grid_size       = {
                static_cast<uint32_t>(e.cluster_count_x) * e.cluster_size_x * e.workgroup_size_x,
                static_cast<uint32_t>(e.cluster_count_y) * e.cluster_size_y * e.workgroup_size_y,
                static_cast<uint32_t>(e.cluster_count_z) * e.cluster_size_z * e.workgroup_size_z};
            dispatch.group_segment_size   = e.group_segment_size;
            dispatch.private_segment_size = e.private_segment_size;
            return dispatch;
        }
#endif
        default: break;
    }
    return std::nullopt;
}

rocprofiler_status_t
build_dispatch_packet(const hsa::rocprofiler_packet&              app,
                      const rocprofiler_kernel_replay_dispatch_t& dispatch,
                      hsa::rocprofiler_packet&                    out)
{
    constexpr auto invalid = ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    // Every member through private_segment_size is required; a larger size from a newer tool is
    // fine.
    constexpr size_t required_size =
        offsetof(rocprofiler_kernel_replay_dispatch_t, private_segment_size) +
        sizeof(rocprofiler_kernel_replay_dispatch_t{}.private_segment_size);
    if(dispatch.size < required_size) return invalid;
    if(dispatch.kernel_object == 0) return invalid;
    if(!nonzero(dispatch.workgroup_size) || !nonzero(dispatch.grid_size)) return invalid;
    if(!fits_u16(dispatch.workgroup_size)) return invalid;

    auto pkt = app;
    switch(kind_of(app))
    {
        case packet_kind::dispatch:
        {
            if(!all_zero(dispatch.cluster_size)) return invalid;

            auto& s                = pkt.kernel_dispatch;
            s.setup                = setup_with_dimensions(s.setup, dispatch);
            s.workgroup_size_x     = static_cast<uint16_t>(dispatch.workgroup_size.x);
            s.workgroup_size_y     = static_cast<uint16_t>(dispatch.workgroup_size.y);
            s.workgroup_size_z     = static_cast<uint16_t>(dispatch.workgroup_size.z);
            s.grid_size_x          = dispatch.grid_size.x;
            s.grid_size_y          = dispatch.grid_size.y;
            s.grid_size_z          = dispatch.grid_size.z;
            s.private_segment_size = dispatch.private_segment_size;
            s.group_segment_size   = dispatch.group_segment_size;
            s.kernel_object        = dispatch.kernel_object;
            s.kernarg_address      = dispatch.kernarg_address;
            break;
        }
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
        case packet_kind::ext_dispatch:
        {
            const auto& cs = dispatch.cluster_size;
            const auto& wg = dispatch.workgroup_size;
            const auto& gs = dispatch.grid_size;
            if(!nonzero(cs)) return invalid;
            constexpr uint32_t max_u8 = std::numeric_limits<uint8_t>::max();
            if(cs.x > max_u8 || cs.y > max_u8 || cs.z > max_u8) return invalid;

            // Work-items per cluster along each axis; the grid has to be a whole number of them.
            const uint64_t cx = uint64_t{cs.x} * wg.x;
            const uint64_t cy = uint64_t{cs.y} * wg.y;
            const uint64_t cz = uint64_t{cs.z} * wg.z;
            if(gs.x % cx != 0 || gs.y % cy != 0 || gs.z % cz != 0) return invalid;

            const uint64_t     count_x = gs.x / cx;
            const uint64_t     count_y = gs.y / cy;
            const uint64_t     count_z = gs.z / cz;
            constexpr uint64_t max_u16 = std::numeric_limits<uint16_t>::max();
            if(count_y > max_u16 || count_z > max_u16) return invalid;

            auto& e                = pkt.ext_kernel_dispatch;
            e.setup                = static_cast<uint8_t>(setup_with_dimensions(e.setup, dispatch));
            e.workgroup_size_x     = static_cast<uint16_t>(wg.x);
            e.workgroup_size_y     = static_cast<uint16_t>(wg.y);
            e.workgroup_size_z     = static_cast<uint16_t>(wg.z);
            e.cluster_count_x      = static_cast<uint32_t>(count_x);
            e.cluster_count_y      = static_cast<uint16_t>(count_y);
            e.cluster_count_z      = static_cast<uint16_t>(count_z);
            e.cluster_size_x       = static_cast<uint8_t>(cs.x);
            e.cluster_size_y       = static_cast<uint8_t>(cs.y);
            e.cluster_size_z       = static_cast<uint8_t>(cs.z);
            e.private_segment_size = dispatch.private_segment_size;
            e.group_segment_size   = dispatch.group_segment_size;
            e.kernel_object        = dispatch.kernel_object;
            e.kernarg_address      = dispatch.kernarg_address;
            break;
        }
#endif
        default: return invalid;
    }

    out = pkt;
    return ROCPROFILER_STATUS_SUCCESS;
}
}  // namespace kernel_replay
}  // namespace rocprofiler
