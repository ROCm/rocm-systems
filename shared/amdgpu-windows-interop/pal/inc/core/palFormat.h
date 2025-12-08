/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2014-2025 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 **********************************************************************************************************************/
/**
 ***********************************************************************************************************************
 * @file  palFormat.h
 * @brief Common include for the Platform Abstraction Library (PAL) interface.  Defines format types.
 ***********************************************************************************************************************
 */

#pragma once

#include "palUtil.h"

/// Library-wide namespace encapsulating all PAL entities.
namespace Pal
{

/// Specifies the format for an image or GPU memory view.
///
/// This defines the bit layout of the channels and how the value in each channel is interpreted.
///
/// Channels are listed in DX10+-style order, where the least significant channels are listed first.  For example, for
/// a uint32 val with an X8Y8Z8W8 value: X = val & 0xFF, Y = (val >> 8) & 0xFF, Z = (val >> 16) & 0xFF,
/// W = (val >> 24) & 0xFF.  Enums without a more detailed explanation can be decoded in this manner.  Multimedia,
/// or "YUV" formats are all exceptions to this rule.  Each of those formats explicitly describes how their channels are
/// organized.
///
/// Many of the multimedia (i.e., "YUV") formats are encoded such that the chrominance (chroma, CbCr, UV) samples are
/// stored at a lower resolution than the luminance (luma, Y) samples as a form of compression.  The ratio of the
/// subsampling is often referred to using an A:B:C notation, where the trio of numbers A,B,C are used to describe the
/// number of luma and chroma samples in a hypothetical region which is A pixels wide and 2 pixels high.  The three
/// numbers usually refer to the following quantities by convention:
///  A --> Width of the conceptual region of pixels, and is usually 4.
///  B --> Number of chroma samples in the first of two rows of A pixels.
///  C --> Number of changes of chroma samples between the first and second rows of A pixels.
///
/// Common examples of this notation are described below:
///  4:4:4 --> No chroma subsampling because luma and chroma both have 4 samples per row, and the number of chroma and
///            luma rows is the same.
///  4:1:1 --> Each row of 4 pixels has 1 chroma sample, and the number of chroma and luma rows is the same.
///  4:2:0 --> Each row of 4 pixels has 2 chroma samples, and there is only 1 chroma row for every 2 luma rows.
///  4:2:2 --> Each row of 4 pixels has 2 chroma samples, and the number of chroma and luma rows is the same.
///
/// Because of the subsampling ratios for multimedia formats, there are some restrictions on what dimensions can be used
/// when creating Images of these formats. 4:1:1 formats must have widths specified as a multiple of 4. 4:2:0 formats
/// must have widths and heights specified as multiples of 2. 4:2:2 formats must have widths specified as a multiple of
/// 2. 4:4:4 formats have no dimensional restrictions.
///
/// Additionally, the YUV formats are broadly grouped into two categories: packed and planar formats.  Packed formats
/// interleave the luma and chroma samples in each row of pixels.  Planar formats are organized so that all of the luma
/// samples are together, followed by all of the chroma samples.  Some planar formats interleave the U and V chroma
/// data, while some choose to have separate U and V planes.  Both packed and planar formats can have any subsampling
/// ratio between the luma and chroma data.
enum class ChNumFormat : Util::uint32
{
    Undefined                = 0x0,     ///< Used in situations where no format is needed, like raw memory views, or to
                                        ///  indicate no color/depth target will be attached when creating a graphics
                                        ///  pipeline.
    X1_Unorm                 = 0x1,     ///< _Untested._
    X1_Uscaled               = 0x2,     ///< _Untested._
    X4Y4_Unorm               = 0x3,
    X4Y4_Uscaled             = 0x4,
    L4A4_Unorm               = 0x5,
    X4Y4Z4W4_Unorm           = 0x6,
    X4Y4Z4W4_Uscaled         = 0x7,
    X5Y6Z5_Unorm             = 0x8,
    X5Y6Z5_Uscaled           = 0x9,
    X5Y5Z5W1_Unorm           = 0xA,
    X5Y5Z5W1_Uscaled         = 0xB,
    X1Y5Z5W5_Unorm           = 0xC,
    X1Y5Z5W5_Uscaled         = 0xD,
    X8_Unorm                 = 0xE,
    X8_Snorm                 = 0xF,
    X8_Uscaled               = 0x10,
    X8_Sscaled               = 0x11,
    X8_Uint                  = 0x12,
    X8_Sint                  = 0x13,
    X8_Srgb                  = 0x14,
    A8_Unorm                 = 0x15,
    L8_Unorm                 = 0x16,
    P8_Unorm                 = 0x17,
    X8Y8_Unorm               = 0x18,
    X8Y8_Snorm               = 0x19,
    X8Y8_Uscaled             = 0x1A,
    X8Y8_Sscaled             = 0x1B,
    X8Y8_Uint                = 0x1C,
    X8Y8_Sint                = 0x1D,
    X8Y8_Srgb                = 0x1E,
    L8A8_Unorm               = 0x1F,
    X8Y8Z8W8_Unorm           = 0x20,
    X8Y8Z8W8_Snorm           = 0x21,
    X8Y8Z8W8_Uscaled         = 0x22,
    X8Y8Z8W8_Sscaled         = 0x23,
    X8Y8Z8W8_Uint            = 0x24,
    X8Y8Z8W8_Sint            = 0x25,
    X8Y8Z8W8_Srgb            = 0x26,
    U8V8_Snorm_L8W8_Unorm    = 0x27,    ///< Mixed signed/unsigned format. Valid Image and Color-Target View formats
                                        ///  are X8Y8Z8W8_Snorm (to target U8V8_Snorm) and X8Y8Z8W8_Unorm (to target
                                        ///  L8W8_Unorm).
    X10Y11Z11_Float          = 0x28,
    X11Y11Z10_Float          = 0x29,
    X10Y10Z10W2_Unorm        = 0x2A,
    X10Y10Z10W2_Snorm        = 0x2B,
    X10Y10Z10W2_Uscaled      = 0x2C,
    X10Y10Z10W2_Sscaled      = 0x2D,
    X10Y10Z10W2_Uint         = 0x2E,
    X10Y10Z10W2_Sint         = 0x2F,
    X10Y10Z10W2Bias_Unorm    = 0x30,    ///< A four-component, 32-bit 2.8-biased fixed-point format that supports 10
                                        ///  bits for each color channel and 2-bit alpha. A shader must be aware of
                                        ///  *Bias* and must perform its own bias and scale on any data that is read
                                        ///  from or written.
    U10V10W10_Snorm_A2_Unorm = 0X31,    ///< Mixed signed/unsigned format. Valid Image and Color-Target View formats
                                        ///  are X10Y10Z10W2_Snorm (to target U10V10W10_Snorm) and X10Y10Z10W2_Unorm
                                        ///  (to target A2_Unorm).
    X16_Unorm                = 0x32,
    X16_Snorm                = 0x33,
    X16_Uscaled              = 0x34,
    X16_Sscaled              = 0x35,
    X16_Uint                 = 0x36,
    X16_Sint                 = 0x37,
    X16_Float                = 0x38,
    L16_Unorm                = 0x39,
    X16Y16_Unorm             = 0x3A,
    X16Y16_Snorm             = 0x3B,
    X16Y16_Uscaled           = 0x3C,
    X16Y16_Sscaled           = 0x3D,
    X16Y16_Uint              = 0x3E,
    X16Y16_Sint              = 0x3F,
    X16Y16_Float             = 0x40,
    X16Y16Z16W16_Unorm       = 0x41,
    X16Y16Z16W16_Snorm       = 0x42,
    X16Y16Z16W16_Uscaled     = 0x43,
    X16Y16Z16W16_Sscaled     = 0x44,
    X16Y16Z16W16_Uint        = 0x45,
    X16Y16Z16W16_Sint        = 0x46,
    X16Y16Z16W16_Float       = 0x47,
    X32_Uint                 = 0x48,
    X32_Sint                 = 0x49,
    X32_Float                = 0x4A,
    X32Y32_Uint              = 0x4B,
    X32Y32_Sint              = 0x4C,
    X32Y32_Float             = 0x4D,
    X32Y32Z32_Uint           = 0x4E,
    X32Y32Z32_Sint           = 0x4F,
    X32Y32Z32_Float          = 0x50,
    X32Y32Z32W32_Uint        = 0x51,
    X32Y32Z32W32_Sint        = 0x52,
    X32Y32Z32W32_Float       = 0x53,
    D16_Unorm_S8_Uint        = 0x54,
    D32_Float_S8_Uint        = 0x55,
    X9Y9Z9E5_Float           = 0x56,    ///< Three partial-precision floating-point numbers encoded into a single 32-bit
                                        ///  value all sharing the same 5-bit exponent (variant of s10e5, which is sign
                                        ///  bit, 10-bit mantissa, and 5-bit biased (15) exponent). There is no sign
                                        ///  bit, and there is a shared 5-bit biased (15) exponent and a 9-bit mantissa
                                        ///  for each channelShared exponent format.
    Bc1_Unorm                = 0x57,    ///< BC1 compressed texture format.
    Bc1_Srgb                 = 0x58,    ///< BC1 compressed texture format.
    Bc2_Unorm                = 0x59,    ///< BC2 compressed texture format.
    Bc2_Srgb                 = 0x5A,    ///< BC2 compressed texture format.
    Bc3_Unorm                = 0x5B,    ///< BC3 compressed texture format.
    Bc3_Srgb                 = 0x5C,    ///< BC3 compressed texture format.
    Bc4_Unorm                = 0x5D,    ///< BC4 compressed texture format.
    Bc4_Snorm                = 0x5E,    ///< BC4 compressed texture format.
    Bc5_Unorm                = 0x5F,    ///< BC5 compressed texture format.
    Bc5_Snorm                = 0x60,    ///< BC5 compressed texture format.
    Bc6_Ufloat               = 0x61,    ///< BC6 unsigned compressed texture format.
    Bc6_Sfloat               = 0x62,    ///< BC6 signed compressed texture format.
    Bc7_Unorm                = 0x63,    ///< BC7 compressed texture format.
    Bc7_Srgb                 = 0x64,    ///< BC7 compressed texture format.
    Etc2X8Y8Z8_Unorm         = 0x65,
    Etc2X8Y8Z8_Srgb          = 0x66,
    Etc2X8Y8Z8W1_Unorm       = 0x67,
    Etc2X8Y8Z8W1_Srgb        = 0x68,
    Etc2X8Y8Z8W8_Unorm       = 0x69,
    Etc2X8Y8Z8W8_Srgb        = 0x6A,
    Etc2X11_Unorm            = 0x6B,
    Etc2X11_Snorm            = 0x6C,
    Etc2X11Y11_Unorm         = 0x6D,
    Etc2X11Y11_Snorm         = 0x6E,
    AstcLdr4x4_Unorm         = 0x6F,
    AstcLdr4x4_Srgb          = 0x70,
    AstcLdr5x4_Unorm         = 0x71,
    AstcLdr5x4_Srgb          = 0x72,
    AstcLdr5x5_Unorm         = 0x73,
    AstcLdr5x5_Srgb          = 0x74,
    AstcLdr6x5_Unorm         = 0x75,
    AstcLdr6x5_Srgb          = 0x76,
    AstcLdr6x6_Unorm         = 0x77,
    AstcLdr6x6_Srgb          = 0x78,
    AstcLdr8x5_Unorm         = 0x79,
    AstcLdr8x5_Srgb          = 0x7A,
    AstcLdr8x6_Unorm         = 0x7B,
    AstcLdr8x6_Srgb          = 0x7C,
    AstcLdr8x8_Unorm         = 0x7D,
    AstcLdr8x8_Srgb          = 0x7E,
    AstcLdr10x5_Unorm        = 0x7F,
    AstcLdr10x5_Srgb         = 0x80,
    AstcLdr10x6_Unorm        = 0x81,
    AstcLdr10x6_Srgb         = 0x82,
    AstcLdr10x8_Unorm        = 0x83,
    AstcLdr10x8_Srgb         = 0x84,
    AstcLdr10x10_Unorm       = 0x85,
    AstcLdr10x10_Srgb        = 0x86,
    AstcLdr12x10_Unorm       = 0x87,
    AstcLdr12x10_Srgb        = 0x88,
    AstcLdr12x12_Unorm       = 0x89,
    AstcLdr12x12_Srgb        = 0x8A,
    AstcHdr4x4_Float         = 0x8B,
    AstcHdr5x4_Float         = 0x8C,
    AstcHdr5x5_Float         = 0x8D,
    AstcHdr6x5_Float         = 0x8E,
    AstcHdr6x6_Float         = 0x8F,
    AstcHdr8x5_Float         = 0x90,
    AstcHdr8x6_Float         = 0x91,
    AstcHdr8x8_Float         = 0x92,
    AstcHdr10x5_Float        = 0x93,
    AstcHdr10x6_Float        = 0x94,
    AstcHdr10x8_Float        = 0x95,
    AstcHdr10x10_Float       = 0x96,
    AstcHdr12x10_Float       = 0x97,
    AstcHdr12x12_Float       = 0x98,
    X8Y8_Z8Y8_Unorm          = 0x99,    ///< _Untested._
    X8Y8_Z8Y8_Uscaled        = 0x9A,    ///< _Untested._
    Y8X8_Y8Z8_Unorm          = 0x9B,    ///< _Untested._
    Y8X8_Y8Z8_Uscaled        = 0x9C,    ///< _Untested._
    AYUV                     = 0x9D,    ///< YUV 4:4:4 packed format.  Valid Image and Color-Target view formats are
                                        ///  { X8Y8Z8W8, Unorm } and { X8Y8Z8W8, Uint }.  Each view fully maps the
                                        ///  entire YUV subresource, with the V,U,Y,A channels mapped to the X,Y,Z,W
                                        ///  channels respectively.  Additionally, Image views can use the { X32, Uint }
                                        ///  format where all four channels are packed into a single uint32.
    UYVY                     = 0x9E,    ///< YUV 4:2:2 packed format.  The Image data is subsampled such that each 32bit
                                        ///  element contains two Y samples and one U and V sample.  Valid Image view
                                        ///  formats are { X8Y8Z8W8, Unorm } and { X8Y8Z8W8, Uint }.  Each view fully
                                        ///  maps the entire YUV subresource, with the X,Y,Z,W channels mapped to the
                                        ///  U0,Y0,V0,Y1 channels respectively. Additionally, Image views can use the
                                        ///  { X32, Uint } format where all four channels are packed into a single
                                        ///  uint32. Image views can also use the { X8Y8_Z8Y8, Unorm } format to access
                                        ///  these as well. In this case, the width of the Image view would appear to be
                                        ///  twice as wide as it normally does, and the X0,Y0,Z0,Y1 channels map to the
                                        ///  U0,Y0,V0,Y1 channels respectively.
    VYUY                     = 0x9F,    ///< YUV 4:2:2 packed format.  The image data is encoded just like the
                                        ///  @ref ChNumFormat::UYVY format, except with a different channel ordering.
                                        ///  Image views with X8Y8Z8W8 channel formats map the X,Y,Z,W channels to the
                                        ///  V0,Y0,U0,Y1 channels respectively. Image views with the X8Y8_Z8Y8 channel
                                        ///  format map the X0,Y0,Z0,Y1 channels to the V0,Y0,U0,Y1 channels
                                        ///  respectively.
    YUY2                     = 0xA0,    ///< YUV 4:2:2 packed format.  The image data is encoded just like the
                                        ///  @ref ChNumFormat::UYVY format, except with a different channel ordering.
                                        ///  X8Y8Z8W8 Image view formats map the X,Y,Z,W channels to the Y0,U0,Y1,V0
                                        ///  channels respectively. Image views can use the { Y8X8_Y8Z8, Unorm } format
                                        ///  where the Y0,X0,Y1,Z0 channels are mapped to the Y0,U0,Y1,V0 channels.
    YVY2                     = 0xA1,    ///< YUV 4:2:2 packed format.  The image data is encoded just like the
                                        ///  @ref ChNumFormat::YUY2 format, except with a different channel ordering.
                                        ///  X8Y8Z8W8 Image view formats map the X,Y,Z,W channels to the Y0,V0,Y1,U0
                                        ///  channels respectively. Image views can use the { Y8X8_Y8Z8, Unorm } format
                                        ///  where the Y0,X0,Y1,Z0 channels are mapped to the Y0,V0,Y1,U0 channels.
    YV12                     = 0xA2,    ///< YVU 4:2:0 planar format, with 8 bits per luma and chroma sample.  The Y
                                        ///  plane is first, containg a uint8 per sample.  Next is the V plane and the U
                                        ///  plane, both of which have a uint8 per sample.  Valid Image view formats are
                                        ///  { X8, Unorm } and { X8, Uint }.  Each view only has access to one of the Y,
                                        ///  V, or U planes.
    NV11                     = 0xA3,    ///< YUV 4:1:1 planar format, with 8 bits per luma and chroma sample.  The Y
                                        ///  plane is first, containing a uint8 per sample.  Next is a UV plane which
                                        ///  has interleaved U and V samples, each stored as a uint8.  Valid Image and
                                        ///  Color-Target view formats are { X8, Unorm }, { X8, Uint }, { X8Y8, Unorm }
                                        ///  and { X8Y8, Uint }.  When using an X8 channel format for the View, the view
                                        ///  only has access to the Y plane.  When using X8Y8, the view only has access
                                        ///  to the UV plane.
    NV12                     = 0xA4,    ///< YUV 4:2:0 planar format, with 8 bits per luma and chroma sample.  The Y
                                        ///  plane is first, containing a uint8 per sample.  Next is a UV plane which
                                        ///  has interleaved U and V samples, each stored as a uint8.  Valid Image and
                                        ///  Color-Target view formats are { X8, Unorm }, { X8, Uint }, { X8Y8, Unorm }
                                        ///  and { X8Y8, Uint }.  When using an X8 channel format for the View, the view
                                        ///  only has access to the Y plane.  When using X8Y8, the view only has access
                                        ///  to the UV plane.
    NV21                     = 0xA5,    ///< YUV 4:2:0 planar format, with 8 bits per luma and chroma sample.  This is
                                        ///  identical to @ref ChNumFormat::NV12, except that the second plane swaps the
                                        ///  ordering of the U and V samples. Image views behave just like with
                                        ///  @ref ChNumFormat::NV12.
    P016                     = 0xA6,    ///< YUV 4:2:0 planar format, with 16 bits per luma and chroma sample.  The
                                        ///  plane ordering is identical to @ref ChNumFormat::NV12.  Instead of uint8
                                        ///  samples, this format uses 8.8 fixed point sample encoding.  Image views
                                        ///  behave just like with @ref ChNumFormat::NV12, except R16 channel formats
                                        ///  are used for the Y plane, and X16Y16 channel formats are used for the UV
                                        ///  plane.
    P010                     = 0xA7,    ///< YUV 4:2:0 planar format, with 10 bits per luma and chroma sample.  This is
                                        ///  identical to @ref ChNumFormat::P016, except that the lowest 6 bits of each
                                        ///  luma and chroma sample are ignored. This allows the source data to be
                                        ///  interpreted as either P016 or P010 interchangably.
    P210                     = 0xA8,    ///< YUV 4:2:2 planar format, with 10 bits per luma and chroma sample. This is
                                        ///  similar to @ref ChNumFormat::P010, except that the UV planes are sub-sampled
                                        ///  only in the horizontal direction, but still by a factor of 2 so the UV plane
                                        ///  ends up having the same number of lines as the Y plane.
    X8_MM_Unorm              = 0xA9,    ///< Multi-media format used with DCC for non-interleaved planes in YUV planar
                                        ///  surfaces. Such as the Y plane or any plane in YV12.
    X8_MM_Uint               = 0xAA,    ///< Multi-media format used with DCC for non-interleaved planes in YUV planar
                                        ///  surfaces. Such as the Y plane or any plane in YV12.
    X8Y8_MM_Unorm            = 0xAB,    ///< Multi-media format used with DCC for the interleaved UV plane in YUV planar
                                        ///  surfaces.
    X8Y8_MM_Uint             = 0xAC,    ///< Multi-media format used with DCC for the interleaved UV plane in YUV planar
                                        ///  surfaces.
    X16_MM10_Unorm           = 0xAD,    ///< Multi-media format used with DCC for non-interleaved planes in YUV planar
                                        ///  surfaces (10-bit). Such as the Y plane or any plane in YV12.
    X16_MM10_Uint            = 0xAE,    ///< Multi-media format used with DCC for non-interleaved planes in YUV planar
                                        ///  surfaces (10-bit). Such as the Y plane or any plane in YV12.
    X16Y16_MM10_Unorm        = 0xAF,    ///< Multi-media format used with DCC for the interleaved UV plane in YUV planar
                                        ///  surfaces (10-bit).
    X16Y16_MM10_Uint         = 0xB0,    ///< Multi-media format used with DCC for the interleaved UV plane in YUV planar
                                        ///  surfaces (10-bit).
    P208                     = 0xB1,    ///< YUV 4:2:2 planar format, with 8 bits per luma and chroma sample. This is
                                        ///  similar to @ref ChNumFormat::NV12, except that the UV planes are sub-sampled
                                        ///  only in the horizontal direction, but still by a factor of 2 so the UV plane
                                        ///  ends up having the same number of lines as the Y plane. This format is
                                        ///  sometimes referred to as NV16.
    X16_MM12_Unorm           = 0xB2,    ///< Multi-media format used with DCC for non-interleaved planes in YUV planar
                                        ///  surfaces (12-bit).
    X16_MM12_Uint            = 0xB3,    ///< Multi-media format used with DCC for non-interleaved planes in YUV planar
                                        ///  surfaces (12-bit).
    X16Y16_MM12_Unorm        = 0xB4,    ///< Multi-media format used with DCC for the interleaved UV plane in YUV planar
                                        ///  surfaces (12-bit).
    X16Y16_MM12_Uint         = 0xB5,    ///< Multi-media format used with DCC for the interleaved UV plane in YUV planar
                                        ///  surfaces (12-bit).
    P012                     = 0xB6,    ///< YUV 4:2:0 planar format, with 12 bits per luma and chroma sample.  This is
                                        ///  identical to @ref ChNumFormat::P010, except that the lowest 4 bits of each
                                        ///  luma and chroma sample are ignored.
    P212                     = 0xB7,    ///< YUV 4:2:2 planar format, with 12 bits per luma and chroma sample.  This is
                                        ///  identical to @ref ChNumFormat::P210, except that the lowest 4 bits of each
                                        ///  luma and chroma sample are ignored.
    P412                     = 0xB8,    ///< YUV 4:4:4 planar format, with 12 bits per luma and chroma sample. It consists
                                        ///  of a Y-plane followed by an interleaved UV plane.
    X10Y10Z10W2_Float        = 0xB9,    ///< RGBA format with three 10-bit floats (6e4) and a 2-bit unorm as alpha.
    Y216                     = 0xBA,    ///< YUV 4:2:2 packed, with 16 bits per luma or chroma sample. No alpha.
    Y210                     = 0xBB,    ///< YUV 4:2:2 packed, with 10 bits per luma or chroma sample. No alpha.
                                        ///  Same memory layout as @ref ChNumFormat::Y216.
                                        ///  The lowest 6 bits of each sample are ignored.
    Y416                     = 0xBC,    ///< YUV 4:4:4 packed, with 16 bits per luma or chroma sample.
    Y410                     = 0xBD,    ///< YUV 4:4:4 packed, with 10 bits per luma or chroma sample and 2 bits for alpha.
    _ReservedBE              = 0xBE,
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 923
    P216                     = 0xBF,    ///< YUV 4:2:2 planar format, with 16 bits per luma and chroma sample. It consists
                                        ///  of a Y-plane followed by interleaved UV plane.
#endif
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 924
    YUV_420P10               = 0xC0,    ///< YUV 4:2:0 tri-planar format, with 10 bits per luma and chroma sample.
    YUV_422P10               = 0xC1,    ///< YUV 4:2:2 tri-planar format, with 10 bits per luma and chroma sample.
    YUV_444P10               = 0xC2,    ///< YUV 4:4:4 tri-planar format, with 10 bits per luma and chroma sample.
    YUV_420P12               = 0xC3,    ///< YUV 4:2:0 tri-planar format, with 12 bits per luma and chroma sample.
    YUV_422P12               = 0xC4,    ///< YUV 4:2:2 tri-planar format, with 12 bits per luma and chroma sample.
    YUV_444P12               = 0xC5,    ///< YUV 4:4:4 tri-planar format, with 12 bits per luma and chroma sample.
    YUV_420P16               = 0xC6,    ///< YUV 4:2:0 tri-planar format, with 16 bits per luma and chroma sample.
    YUV_422P16               = 0xC7,    ///< YUV 4:2:2 tri-planar format, with 16 bits per luma and chroma sample.
    YUV_444P16               = 0xC8,    ///< YUV 4:4:4 tri-planar format, with 16 bits per luma and chroma sample.
#endif
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 925
    YV16                     = 0xC9,    ///< YVU 4:2:2 tri-planar format, with 8 bits per luma and chroma sample. This
                                        ///  is similar to @ref ChNumFormat::YV12, except chroma is not subsampled in
                                        ///  vertical direction.
    YV24                     = 0xCA,    ///< YVU 4:4:4 tri-planar format, with 8 bits per luma and chroma sample. This
                                        ///  is similar to @ref ChNumFormat::YV12, except chroma is not subsampled.
    NV24                     = 0xCB,    ///< YUV 4:4:4 bi-planar format, with 8 bits per luma and chroma sample. This
                                        ///  is similar to @ref ChNumFormat::NV12, except chroma is not subsampled.
                                        ///  This format is sometimes referred to as P408.
    P410                     = 0xCC,    ///< YUV 4:4:4 planar format, with 10 bits per luma and chroma sample. It consists
                                        ///  of a Y-plane followed by interleaved UV plane.
    P416                     = 0xCD,    ///< YUV 4:4:4 planar format, with 16 bits per luma and chroma sample. It consists
                                        ///  of a Y-plane followed by interleaved UV plane.
#endif
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 929
    X16Y16Z16W16_MM10_Unorm  = 0xCE,    ///< A four component format with 16 bits of storage per component. Lowest 6 bits of
                                        ///  each component are ignored.
    X16Y16Z16W16_MM10_Uint   = 0xCF,    ///< A four component format with 16 bits of storage per component. Lowest 6 bits of
                                        ///  each component are ignored.
    X16Y16Z16W16_MM12_Unorm  = 0xD0,    ///< A four component format with 16 bits of storage per component. Lowest 4 bits of
                                        ///  each component are ignored.
    X16Y16Z16W16_MM12_Uint   = 0xD1,    ///< A four component format with 16 bits of storage per component. Lowest 4 bits of
                                        ///  each component are ignored.
#endif
    Count,

};

/// Specifies which channel of a resource should be mapped to a particular component of an image view.
///
/// @ingroup ResourceBinding
enum class ChannelSwizzle : Util::uint8
{
    Zero = 0x0,  ///< Ignore resource data and always fetch a 0 into this component.
    One  = 0x1,  ///< Ignore resource data and always fetch a 1 into this component.
    X    = 0x2,  ///< Use the X channel from resource for this component.
    Y    = 0x3,  ///< Use the Y channel from resource for this component.
    Z    = 0x4,  ///< Use the Z channel from resource for this component.
    W    = 0x5,  ///< Use the W channel from resource for this component.
    Count
};

/// Specifies a mapping for each component of an image or buffer view to a channel in its associated resource.
///
/// @ingroup ResourceBinding
struct ChannelMapping
{
    union
    {
        struct
        {
            ChannelSwizzle r;          ///< Red component swizzle.
            ChannelSwizzle g;          ///< Green component swizzle.
            ChannelSwizzle b;          ///< Blue component swizzle.
            ChannelSwizzle a;          ///< Alpha component swizzle.
        };
        ChannelSwizzle     swizzle[4]; ///< All four swizzles packed into one array.
        Util::uint32       swizzleValue;
    };
};

/// Specifies a pixel format for an image or memory view and its corresponding channel swizzle.
struct SwizzledFormat
{
    ChNumFormat    format;  ///< Pixel format.
    ChannelMapping swizzle; ///< Compatible channel swizzle for the above pixel format.
};

inline constexpr bool operator==(const SwizzledFormat& lhs, const SwizzledFormat& rhs)
{
    return (lhs.format == rhs.format) && (lhs.swizzle.swizzleValue == rhs.swizzle.swizzleValue);
}

/// Constant for undefined formats.
constexpr SwizzledFormat UndefinedSwizzledFormat =
{
    ChNumFormat::Undefined,
    { { { ChannelSwizzle::X, ChannelSwizzle::Zero, ChannelSwizzle::Zero, ChannelSwizzle::One } } },
};

/// Flags structure reporting available capabilities of a particular format.
enum FormatFeatureFlags : Util::uint32
{
    FormatFeatureCopy                = 0x00001,  ///< Images of this format can be used as a copy source or destination.
    FormatFeatureFormatConversion    = 0x00002,  ///< Images of this format support format conversion in copy
                                                 ///  operations.
    FormatFeatureImageShaderRead     = 0x00004,  ///< Images of this format can be read from a shader.
    FormatFeatureImageShaderWrite    = 0x00008,  ///< Images of this format can be written from a shader.
    FormatFeatureImageShaderAtomics  = 0x00010,  ///< Images of this format can be written atomically from a shader.
    FormatFeatureMemoryShaderRead    = 0x00020,  ///< Memory views of this format can be read from a shader.
    FormatFeatureMemoryShaderWrite   = 0x00040,  ///< Memory views of this format can be written from a shader.
    FormatFeatureMemoryShaderAtomics = 0x00080,  ///< Memory views of this format can be written atomically from a
                                                 ///  shader.
    FormatFeatureColorTargetWrite    = 0x00100,  ///< Images of this format can be bound as a color target.
    FormatFeatureColorTargetBlend    = 0x00200,  ///< Images of this format can be bound as a color target for blending.
    FormatFeatureDepthTarget         = 0x00400,  ///< Images of this format can be bound as a depth target.
    FormatFeatureStencilTarget       = 0x00800,  ///< Images of this format can be bound as a stencil target.
    FormatFeatureMsaaTarget          = 0x01000,  ///< Images of this format can support multisampling.
    FormatFeatureWindowedPresent     = 0x02000,  ///< Images of this format can support windowed-mode presents.
                                                 ///  Fullscreen present capability is queried using the @ref
                                                 ///  IScreen::GetScreenModeList method.
    FormatFeatureImageFilterLinear   = 0x04000,  ///< Images of this format can be linearly filtered.
    FormatFeatureImageFilterMinMax   = 0x08000,  ///< Images of this format can be min/max filtered.
    FormatFeatureFormatConversionSrc = 0x10000,  ///< Images of this format support format conversion in copy
                                                 ///  operations as the source image.
                                                 ///  @note This is aliased to FormatFeatureFormatConversionDst for
                                                 ///  backwards compatibility.
    FormatFeatureFormatConversionDst = 0x20000,  ///< Images of this format support format conversion in copy
                                                 ///  operations as the destination image.
                                                 ///  @note This is aliased to FormatFeatureFormatConversionSrc for
                                                 ///  backwards compatibility.
};

/// Enumeration for indexing into the format properties table based on tiling.
enum FormatPropertiesTiling : Util::uint32
{
    IsLinear  = 0,  ///< Format properties requested is for linearly-tiled surfaces.
    IsNonLinear,    ///< Format properties requested is for non-linearly tiled surfaces.
    Count,          ///< Number of format property tile types.
};

/// The format properties lookup table.  Contains information about which device access features are available for all
/// formats and tiling modes.  The tiling features for non-linear tiling modes are identical so we only store linear
/// and non-linear tiling features.  From left to right, it is indexed by format and "is-non-linear".
/// Returned by IDevice::GetFormatProperties().
struct MergedFormatPropertiesTable
{
    FormatFeatureFlags features[static_cast<size_t>(ChNumFormat::Count)][FormatPropertiesTiling::Count];
};

} // Pal
