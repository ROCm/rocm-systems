// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/rocdecode/api_args.h>

#include "fmt/core.h"

#define ROCP_SDK_ROCDECODE_FORMATTER(TYPE, ...)                                                    \
    template <>                                                                                    \
    struct formatter<TYPE> : rocprofiler::rocdecode::details::base_formatter                       \
    {                                                                                              \
        template <typename Ctx>                                                                    \
        auto format(const TYPE& v, Ctx& ctx) const                                                 \
        {                                                                                          \
            return fmt::format_to(ctx.out(), __VA_ARGS__);                                         \
        }                                                                                          \
    };

#define ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(PREFIX, SUFFIX)                                        \
    case PREFIX##_##SUFFIX: return fmt::format_to(ctx.out(), #SUFFIX)

namespace rocprofiler
{
namespace rocdecode
{
namespace details
{
struct base_formatter
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }
};
}  // namespace details
}  // namespace rocdecode
}  // namespace rocprofiler

namespace fmt
{
template <>
struct formatter<rocDecStatus> : rocprofiler::rocdecode::details::base_formatter
{
    template <typename Ctx>
    auto format(rocDecStatus v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(ROCDEC, DEVICE_INVALID);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(ROCDEC, CONTEXT_INVALID);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(ROCDEC, RUNTIME_ERROR);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(ROCDEC, OUTOF_MEMORY);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(ROCDEC, INVALID_PARAMETER);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(ROCDEC, NOT_IMPLEMENTED);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(ROCDEC, NOT_INITIALIZED);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(ROCDEC, NOT_SUPPORTED);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(ROCDEC, SUCCESS);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<rocDecDecodeStatus> : rocprofiler::rocdecode::details::base_formatter
{
    template <typename Ctx>
    auto format(rocDecDecodeStatus v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecodeStatus, Invalid);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecodeStatus, InProgress);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecodeStatus, Success);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecodeStatus, Error);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecodeStatus, Error_Concealed);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecodeStatus, Displaying);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<rocDecVideoCodec> : rocprofiler::rocdecode::details::base_formatter
{
    template <typename Ctx>
    auto format(rocDecVideoCodec v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoCodec, MPEG1);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoCodec, MPEG2);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoCodec, MPEG4);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoCodec, AVC);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoCodec, HEVC);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoCodec, AV1);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoCodec, VP8);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoCodec, VP9);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoCodec, JPEG);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoCodec, NumCodecs);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoCodec, YUV420);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoCodec, YV12);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoCodec, NV12);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoCodec, YUYV);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoCodec, UYVY);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<rocDecVideoChromaFormat> : rocprofiler::rocdecode::details::base_formatter
{
    template <typename Ctx>
    auto format(rocDecVideoChromaFormat v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoChromaFormat, Monochrome);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoChromaFormat, 420);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoChromaFormat, 422);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoChromaFormat, 444);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<rocDecVideoSurfaceFormat> : rocprofiler::rocdecode::details::base_formatter
{
    template <typename Ctx>
    auto format(rocDecVideoSurfaceFormat v, Ctx& ctx) const
    {
        switch(v)
        {
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoSurfaceFormat, NV12);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoSurfaceFormat, P016);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoSurfaceFormat, YUV444);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoSurfaceFormat, YUV444_16Bit);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoSurfaceFormat, YUV420);
            ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT(rocDecVideoSurfaceFormat, YUV420_16Bit);
        }
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

ROCP_SDK_ROCDECODE_FORMATTER(
    _RocdecParserParams,
    "{}codec_type={},max_num_decode_surfaces={}, clock_rate={}, error_threshold={}, "
    "max_display_delay={}, annex_b={}, reserved={}, user_data={}, ext_video_info={}{}",
    '{',
    v.codec_type,
    v.max_num_decode_surfaces,
    v.clock_rate,
    v.error_threshold,
    v.max_display_delay,
    v.annex_b,
    v.reserved,
    v.user_data,
    static_cast<void*>(v.ext_video_info),
    '}')

ROCP_SDK_ROCDECODE_FORMATTER(RocdecTimeStamp, "{}RocdecTimeStamp={}{}", '{', v, '}')

ROCP_SDK_ROCDECODE_FORMATTER(_RocdecSourceDataPacket,
                             "{}flags={}, payload_size={}, payload={}, pts={}{}",
                             '{',
                             v.flags,
                             v.payload_size,
                             static_cast<const void*>(v.payload),
                             v.pts,
                             '}')

ROCP_SDK_ROCDECODE_FORMATTER(
    _RocDecoderCreateInfo,
    "{}device_id={}, width={}, height={}, num_decode_surfaces={}, codec_type={}, chroma_format={}, "
    "bit_depth_minus_8={}, intra_decode_only={}, max_width={}, max_height={}, "
    "display_rect={}left={}, right={}, top={}, bottom={}{}, target_width={}, target_height={}, "
    "num_output_surfaces={}, target_rect={}left={}, right={}, top={}, bottom={}{}{}",
    '{',
    v.device_id,
    v.width,
    v.height,
    v.num_decode_surfaces,
    v.codec_type,
    v.chroma_format,
    v.bit_depth_minus_8,
    v.intra_decode_only,
    v.max_width,
    v.max_height,
    '{',
    v.display_rect.left,
    v.display_rect.right,
    v.display_rect.top,
    v.display_rect.bottom,
    '}',
    v.target_width,
    v.target_height,
    v.num_output_surfaces,
    '{',
    v.target_rect.left,
    v.target_rect.right,
    v.target_rect.top,
    v.target_rect.bottom,
    '}',
    '}')

ROCP_SDK_ROCDECODE_FORMATTER(_RocdecReconfigureDecoderInfo,
                             "{}width={}, height={}, target_width={}, target_height={}, "
                             "num_decode_surfaces={}, display_rect={}left={}, right={}, top={}, "
                             "bottom={}{}, target_rect={}left={}, right={}, top={}, bottom={}{}{}",
                             '{',
                             v.width,
                             v.height,
                             v.target_width,
                             v.target_height,
                             v.num_decode_surfaces,
                             '{',
                             v.display_rect.left,
                             v.display_rect.right,
                             v.display_rect.top,
                             v.display_rect.bottom,
                             '}',
                             '{',
                             v.target_rect.left,
                             v.target_rect.right,
                             v.target_rect.top,
                             v.target_rect.bottom,
                             '}',
                             '}')

ROCP_SDK_ROCDECODE_FORMATTER(_RocdecDecodeStatus,
                             "{}decode_status={}, p_reserved={}{}",
                             '{',
                             v.decode_status,
                             v.p_reserved,
                             '}')

ROCP_SDK_ROCDECODE_FORMATTER(
    _RocdecPicParams,
    "{}pic_width={}, pic_height={}, curr_pic_idx={}, field_pic_flag={}, bottom_field_flag={}, "
    "second_field={}, bitstream_data_len={}, bitstream_data={}, num_slices={}, ref_pic_flag={}, "
    "intra_pic_flag={}{}",
    '{',
    v.pic_width,
    v.pic_height,
    v.curr_pic_idx,
    v.field_pic_flag,
    v.bottom_field_flag,
    v.second_field,
    v.bitstream_data_len,
    static_cast<const void*>(v.bitstream_data),
    v.num_slices,
    v.ref_pic_flag,
    v.intra_pic_flag,
    '}')

ROCP_SDK_ROCDECODE_FORMATTER(
    _RocdecDecodeCaps,
    "{}device_id={}, codec_type={}, chroma_format={}, bit_depth_minus_8={}, is_supported={}, "
    "num_decoders={}, output_format_mask={}, max_width={}, max_height={}, min_width={}, "
    "min_height={}{}",
    '{',
    v.device_id,
    v.codec_type,
    v.chroma_format,
    v.bit_depth_minus_8,
    v.is_supported,
    v.num_decoders,
    v.output_format_mask,
    v.max_width,
    v.max_height,
    v.min_width,
    v.min_height,
    '}')

ROCP_SDK_ROCDECODE_FORMATTER(
    _RocdecProcParams,
    "{}progressive_frame={}, top_field_first={}, raw_input_dptr={}, raw_input_pitch={}, "
    "raw_input_format={}, raw_output_dptr={}, raw_output_pitch={}, raw_output_format={}{}",
    '{',
    v.progressive_frame,
    v.top_field_first,
    v.raw_input_dptr,
    v.raw_input_pitch,
    v.raw_input_format,
    v.raw_output_dptr,
    v.raw_output_pitch,
    v.raw_output_format,
    '}')
}  // namespace fmt

#undef ROCP_SDK_ROCDECODE_FORMATTER
#undef ROCP_SDK_ROCDECODE_FORMAT_CASE_STMT
