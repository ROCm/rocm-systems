// MIT License
//
/* Copyright (c) 2022-2025 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "lib/rocprofiler-sdk/hsa/queue.hpp"

namespace fmt
{
template <>
struct formatter<hsa_ext_amd_aql_pm4_packet_t>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(hsa_ext_amd_aql_pm4_packet_t const& pkt, Ctx& ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            "[AQL_PM4_PKT, header={}, pm4_commands=[{:x}], completion_signal={}]",
            pkt.header,
            fmt::join(std::string_view((const char*) pkt.pm4_command, sizeof(pkt.pm4_command)),
                      " "),
            pkt.completion_signal.handle);
    }
};

template <>
struct formatter<hsa_kernel_dispatch_packet_t>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(hsa_kernel_dispatch_packet_t const& pkt, Ctx& ctx) const
    {
        return fmt::format_to(ctx.out(),
                              "[KERNEL_DISPATCH, header={}, dim={}, workgroup_size=[{}, {}, {}], "
                              "grid_size=[{}, {}, {}], private_size={}, group_size={}, "
                              "kernel_object={:x}, kern_arg={}, completion_signal={}]",
                              pkt.header,
                              pkt.setup,
                              pkt.workgroup_size_x,
                              pkt.workgroup_size_y,
                              pkt.workgroup_size_z,
                              pkt.grid_size_x,
                              pkt.grid_size_y,
                              pkt.grid_size_z,
                              pkt.private_segment_size,
                              pkt.group_segment_size,
                              pkt.kernel_object,
                              pkt.kernarg_address,
                              pkt.completion_signal.handle);
    }
};

template <>
struct formatter<hsa_barrier_and_packet_t>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(hsa_barrier_and_packet_t const& pkt, Ctx& ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            "[BARRIER_AND, header={}, dep_signals=[{},{},{},{},{}], completion_signal={}]",
            pkt.header,
            pkt.dep_signal[0].handle,
            pkt.dep_signal[1].handle,
            pkt.dep_signal[2].handle,
            pkt.dep_signal[3].handle,
            pkt.dep_signal[4].handle,
            pkt.completion_signal.handle);
    }
};

template <>
struct formatter<hsa_barrier_or_packet_t>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(hsa_barrier_or_packet_t const& pkt, Ctx& ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            "[BARRIER_OR, header={}, dep_signals=[{},{},{},{},{}], completion_signal={}]",
            pkt.header,
            pkt.dep_signal[0].handle,
            pkt.dep_signal[1].handle,
            pkt.dep_signal[2].handle,
            pkt.dep_signal[3].handle,
            pkt.dep_signal[4].handle,
            pkt.completion_signal.handle);
    }
};

// fmt::format support for rocprofiler_packet
template <>
struct formatter<rocprofiler::hsa::rocprofiler_packet>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(rocprofiler::hsa::rocprofiler_packet const& pkt, Ctx& ctx) const
    {
        static const char* type_names[] = {"HSA_PACKET_TYPE_VENDOR_SPECIFIC",
                                           "HSA_PACKET_TYPE_INVALID",
                                           "HSA_PACKET_TYPE_KERNEL_DISPATCH",
                                           "HSA_PACKET_TYPE_BARRIER_AND",
                                           "HSA_PACKET_TYPE_AGENT_DISPATCH",
                                           "HSA_PACKET_TYPE_BARRIER_OR"};
        uint8_t            t            = ((pkt.ext_amd_aql_pm4.header >> HSA_PACKET_HEADER_TYPE) &
                     ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1));
        switch(t)
        {
            case 0:
                // PM4 packet
                return fmt::format_to(ctx.out(), "{}", pkt.ext_amd_aql_pm4);
            case 2:
                // Kernel dispatch
                return fmt::format_to(ctx.out(), "{}", pkt.kernel_dispatch);
            case 3:
                // Barrier AND packet
                return fmt::format_to(ctx.out(), "{}", pkt.barrier_and);
            case 5:
                // Barrier OR packet
                return fmt::format_to(ctx.out(), "{}", pkt.barrier_or);
            default:
                return fmt::format_to(ctx.out(), "[Unprintable Packet of type {}]", type_names[t]);
        }
    }
};
}  // namespace fmt
