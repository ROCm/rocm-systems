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
 * @file  palFormatInfo.h
 * @brief Defines the Platform Abstraction Library (PAL) Format utility functions.
 ***********************************************************************************************************************
 */

#pragma once

#include "palDevice.h"
#include "palImage.h"
#include "palInlineFuncs.h"
#include "palMath.h"

namespace Pal
{

/// Namespace encapsulating all PAL format utility functions.
namespace Formats
{

/// Specifies flags which indicate properties of each PAL channel format.
enum PropertyFlags : uint32
{
    BitCountInaccurate = 0x1,   ///< Indicates that format's bit count array is inaccurate
    BlockCompressed    = 0x2,   ///< Indicates channel format is block-compressed
    MacroPixelPacked   = 0x4,   ///< Indicates channel format has multiple pixels' data packed together into
                                ///  one "macro pixel"
    YuvPlanar          = 0x8,   ///< Indicates channel format is YUV-planar
    YuvPacked          = 0x10,  ///< Indicates channel format is YUV packed
};

/// Specifies numeric support of a specified format.
enum class NumericSupportFlags : uint32
{
    Undefined,      ///< No numeric support.
    Unorm,          ///< Unsigned normalized.
    Snorm,          ///< Signed normalized.
    Uscaled,        ///< _Untested._ Treated as an unsigned integer inside the resource, but received by
                    ///  the shader as a floating point number.
    Sscaled,        ///< _Untested._ Treated as a signed integer inside the resource, but received by
                    ///  the shader as a floating point number.
    Uint,           ///< Unsigned integer.
    Sint,           ///< Signed integer.
    Float,          ///< Floating point number.
    Srgb,           ///< sRGB.
    DepthStencil,   ///< Depth/stencil support.
    Yuv,            ///< YUV support.
};

/// Specifies flags which indicate the presence of each color channel in a PAL channel format.
enum ChannelFlags : uint32
{
    X     = 0x1,    ///< Indicates the X channel is present.
    Y     = 0x2,    ///< Indicates the Y channel is present.
    Z     = 0x4,    ///< Indicates the Z channel is present.
    W     = 0x8,    ///< Indicates the W channel is present.
};

/// An entry in the channel-format info lookup table. Contains intrinsic properties describing a channel format.
struct FormatInfo
{
    uint32              bitsPerPixel;    ///< Total count of bits in a signel pixel (or block).
    uint32              componentCount;  ///< Number of color components (channels) present.

    uint32              bitCount[4];     ///< Number of bits for each component in the format. These members are
                                         ///  only reliable if the 'bitCountInaccurate' flag is not set.
                                         ///  Listed in order: X, Y, Z, and W.

    uint32              channelMask;     ///< Mask of @ref ChannelFlags values indicating which channels are present.
    uint32              properties;      ///< Mask of @ref PropertyFlags values indicating which properties a format
                                         ///  has.
    NumericSupportFlags numericSupport;  ///< Which numeric format this format represents. Used for easy identification.
};

/// BC block dimension (4x4)
static constexpr uint32 CompressedBcBlockDim = 4;

/// ETC block dimension (4x4)
static constexpr uint32 CompressedEtcBlockDim = 4;

/// Lookup table for intrinsic properties describing each channel format. Callers should access the members of this
/// table via BitsPerPixel() and related functions.
extern const FormatInfo FormatInfoTable[static_cast<size_t>(ChNumFormat::Count)];

/// Convert a floating-point representation of a color value in RGBA order to the appropriate bit representation for
/// each channel based on the specified format. Swizzling is enabled by default to maintain backwards compatability.
/// There will be no swizzling functionality going forwards.
extern void ConvertColor(
    SwizzledFormat format,
    const float*   pColorIn,
    uint32*        pColorOut);

/// Convert an unsigned integer representation of a color value in YUVA order to the appropriate bit representation for
/// each channel based on the specified format.
extern void ConvertYuvColor(
    SwizzledFormat format,
    uint32         plane,
    const uint32*  pColorIn,
    uint32*        pColorOut);

/// Packs a clear color value in RGBA order to a single element of the provided format and stores it in the
/// memory provided. Swizzling is enabled by default to maintain backwards compatability. There will be
/// no swizzling functionality going forwards.
extern void PackRawClearColor(
    SwizzledFormat format,
    const uint32*  pColor,
    void*          pBufferMemory);

/// Swizzles the color according to the provided format swizzle.
extern void SwizzleColor(SwizzledFormat format, const uint32* pColorIn, uint32* pColorOut);

/// Compares two SwizzledFormats and checks for equality.
///
/// @param lhs [in] Left hand side of comparison
/// @param rhs [in] Right hand side of comparison
///
/// @return True if the formats are equal, false otherwise.
constexpr bool IsSameFormat(
    const SwizzledFormat& lhs,
    const SwizzledFormat& rhs)
{
    return ((lhs.format == rhs.format) && (lhs.swizzle.swizzleValue == rhs.swizzle.swizzleValue));
}

/// Queries the number of components for a particular channel format.
///
/// @param [in] format The channel format to query for.
///
/// @returns The number of components of the specified channel format.
inline uint32 NumComponents(
    ChNumFormat format)
{
    return FormatInfoTable[static_cast<size_t>(format)].componentCount;
}

/// Queries the component mask for a particular format.
///
/// @param [in] format The format to query for.
///
/// @returns The component mask of @ref ChannelFlags for the specified format.
inline uint32 ComponentMask(
    ChNumFormat format)
{
    uint32 mask = FormatInfoTable[static_cast<size_t>(format)].channelMask;
    PAL_ASSERT((mask & 0xF) == mask);
    return mask;
}

/// Checks if a format is undefined.
///
/// @param [in] format Pixel format.
///
/// @returns True if the pixel format is undefined. False otherwise.
constexpr bool IsUndefined(
    ChNumFormat format)
{
    return (format == ChNumFormat::Undefined);
}

/// Checks if a format's numeric representation is unsigned normalized.
///
/// @param [in] format Pixel format.
///
/// @returns True if the pixel format is unsigned normalized. False otherwise.
inline bool IsUnorm(
    ChNumFormat format)
{
    return (FormatInfoTable[static_cast<size_t>(format)].numericSupport == NumericSupportFlags::Unorm);
}

/// Checks if a format's numeric representation is signed normalized.
///
/// @param [in] format Pixel format.
///
/// @returns True if the pixel format is signed normalized. False otherwise.
inline bool IsSnorm(
    ChNumFormat format)
{
    return (FormatInfoTable[static_cast<size_t>(format)].numericSupport == NumericSupportFlags::Snorm);
}

/// Checks if a format's numeric representation is unsigned scaled.
///
/// @param [in] format Pixel format.
///
/// @returns True if the pixel format is unsigned scaled. False otherwise.
inline bool IsUscaled(
    ChNumFormat format)
{
    return (FormatInfoTable[static_cast<size_t>(format)].numericSupport == NumericSupportFlags::Uscaled);
}

/// Checks if a format's numeric representation is signed scaled.
///
/// @param [in] format Pixel format.
///
/// @returns True if the pixel format is signed scaled. False otherwise.
inline bool IsSscaled(
    ChNumFormat format)
{
    return (FormatInfoTable[static_cast<size_t>(format)].numericSupport == NumericSupportFlags::Sscaled);
}

/// Checks if a format's numeric representation is unsigned integer.
///
/// @param [in] format Pixel format.
///
/// @returns True if the pixel format is unsigned integer. False otherwise.
inline bool IsUint(
    ChNumFormat format)
{
    return (FormatInfoTable[static_cast<size_t>(format)].numericSupport == NumericSupportFlags::Uint);
}

/// Checks if a format's numeric representation is signed integer.
///
/// @param [in] format Pixel format.
///
/// @returns True if the pixel format is signed integer. False otherwise.
inline bool IsSint(
    ChNumFormat format)
{
    return (FormatInfoTable[static_cast<size_t>(format)].numericSupport == NumericSupportFlags::Sint);
}

/// Checks if a format's numeric representation is floating point.
///
/// @param [in] format Pixel format.
///
/// @returns True if the pixel format is floating point. False otherwise.
inline bool IsFloat(
    ChNumFormat format)
{
    return (FormatInfoTable[static_cast<size_t>(format)].numericSupport == NumericSupportFlags::Float);
}

/// Checks if a format's numeric representation is gamma-corrected sRGB.
///
/// @param [in] format Pixel format.
///
/// @returns True if the pixel format is sRGB. False otherwise.
inline bool IsSrgb(
    ChNumFormat format)
{
    return (FormatInfoTable[static_cast<size_t>(format)].numericSupport == NumericSupportFlags::Srgb);
}

/// Checks if a format's numeric representation is normalized.
///
/// @param [in] format Pixel format.
///
/// @returns True if the pixel format is normalized. False otherwise.
inline bool IsNormalized(
    ChNumFormat format)
{
    return IsUnorm(format) || IsSnorm(format);
}

/// Checks if a format's numeric representation is an integer format.
///
/// @param [in] format Pixel format.
///
/// @returns True if the pixel format is an integer format. False otherwise.
inline bool IsInteger(
    ChNumFormat format)
{
    return IsUint(format) || IsSint(format);
}

/// Checks if a format is a depth/stencil only format.
///
/// @param [in] format Pixel format.
///
/// @returns True if the pixel format is a depth/stencil only format. False otherwise.
inline bool IsDepthStencilOnly(
    ChNumFormat format)
{
    return (FormatInfoTable[static_cast<size_t>(format)].numericSupport == NumericSupportFlags::DepthStencil);
}

/// Checks if the specified format is one of the YUV-planar ones.
///
/// @param [in] format The format to check.
///
/// @returns True if the specified format is YUV-planar. False otherwise.
inline bool IsYuvPlanar(
    ChNumFormat format)
{
    return ((FormatInfoTable[static_cast<size_t>(format)].properties & YuvPlanar) != 0);
}

/// Checks if the specified format is one of the YUV-packed ones.
///
/// @param [in] format The format to check.
///
/// @returns True if the specified format is YUV-packed. False otherwise.
inline bool IsYuvPacked(
    ChNumFormat format)
{
    return ((FormatInfoTable[static_cast<size_t>(format)].properties & YuvPacked) != 0);
}

/// Checks if the specified format is one of the YUV ones.
///
/// @param [in] format The format to check.
///
/// @returns True if the specified format is for YUV data. False otherwise.
inline bool IsYuv(
    ChNumFormat format)
{
    return (FormatInfoTable[static_cast<size_t>(format)].numericSupport == NumericSupportFlags::Yuv);
}

/// Checks if a format has alpha.
///
/// @param [in] format Pixel format.
///
/// @returns True if the pixel format has an alpha channel. False otherwise.
constexpr bool HasAlpha(
    SwizzledFormat format)
{
    return ((format.format == ChNumFormat::A8_Unorm)                ||
            (format.format == ChNumFormat::L4A4_Unorm)              ||
            (format.format == ChNumFormat::L8A8_Unorm)              ||
            ((ComponentMask(format.format) & ChannelFlags::W) != 0) ||
            ((format.swizzle.a != ChannelSwizzle::Zero) && (format.swizzle.a != ChannelSwizzle::One)));
}

/// Checks if a format has an unused alpha channel.
///
/// @param [in] format Pixel format.
///
/// @returns True if the pixel format is a four channel format and has an unused alpha channel. False otherwise.
inline bool HasUnusedAlpha(
    SwizzledFormat format)
{
    return ((NumComponents(format.format) == 4)     &&
            (format.swizzle.r != ChannelSwizzle::W) &&
            (format.swizzle.g != ChannelSwizzle::W) &&
            (format.swizzle.b != ChannelSwizzle::W) &&
            (format.swizzle.a != ChannelSwizzle::W));
}

/// Converts format into its Unorm equivalent.
///
/// @param [in] format Pixel format.
///
/// @returns Format equivalent of input format. Undefined if none exist.
extern ChNumFormat PAL_STDCALL ConvertToUnorm(ChNumFormat format);

/// Converts format into its Snorm equivalent.
///
/// @param [in] format Pixel format.
///
/// @returns Format equivalent of input format. Undefined if none exist.
extern ChNumFormat PAL_STDCALL ConvertToSnorm(ChNumFormat format);

/// Converts format into its Uscaled equivalent.
///
/// @param [in] format Pixel format.
///
/// @returns Format equivalent of input format. Undefined if none exist.
extern ChNumFormat PAL_STDCALL ConvertToUscaled(ChNumFormat format);

/// Converts format into its Sscaled equivalent.
///
/// @param [in] format Pixel format.
///
/// @returns Format equivalent of input format. Undefined if none exist.
extern ChNumFormat PAL_STDCALL ConvertToSscaled(ChNumFormat format);

/// Converts format into its Uint equivalent.
///
/// @param [in] format Pixel format.
///
/// @returns Uint format equivalent of input format. Undefined if none exist.
extern ChNumFormat PAL_STDCALL ConvertToUint(ChNumFormat format);

/// Converts format into its Sint equivalent.
///
/// @param [in] format Pixel format.
///
/// @returns Sint format equivalent of input format. Undefined if none exist.
extern ChNumFormat PAL_STDCALL ConvertToSint(ChNumFormat format);

/// Converts format into its Float equivalent.
///
/// @param [in] format Pixel format.
///
/// @returns Float format equivalent of input format. Undefined if none exist.
extern ChNumFormat PAL_STDCALL ConvertToFloat(ChNumFormat format);

/// Converts format into its Srgb equivalent.
///
/// @param [in] format Pixel format.
///
/// @returns Srgb format equivalent of input format. Undefined if none exist.
extern ChNumFormat PAL_STDCALL ConvertToSrgb(ChNumFormat format);

/// Converts source numeric format to the provided destination numeric format.
///
/// @param [in] srcFormat Source Pixel format.
/// @param [in] dstFormat Destination Pixel format.
///
/// @returns Source format with equivalent numeric format of destination format. Undefined if none exist.
extern ChNumFormat ConvertToDstNumFmt(ChNumFormat srcFormat, ChNumFormat dstFormat);

/// Determines whether the srcFormat and the dstFormat have the same channel formats.
///
/// @param [in] srcFormat Source channel pixel format.
/// @param [in] dstFormat Destination channel pixel format.
///
/// @returns True if both formats share the same channel format. False otherwise.
extern bool ShareChFmt(ChNumFormat srcFormat, ChNumFormat dstFormat);

/// Determines whether the srcFormat and the dstFormat have the same numeric formats.
///
/// @param [in] srcFormat Source channel pixel format.
/// @param [in] dstFormat Destination channel pixel format.
///
/// @returns True if both formats share the same numeric format. False otherwise.
inline bool HaveSameNumFmt(
    ChNumFormat srcFormat,
    ChNumFormat dstFormat)
{
    return (FormatInfoTable[static_cast<size_t>(srcFormat)].numericSupport ==
            FormatInfoTable[static_cast<size_t>(dstFormat)].numericSupport);
}

/// Returns the block dimension for a compressed format.
///
/// @param [in] format Format.
///
/// @returns Corresponding block dimensions for the compressed format.
inline Extent3d CompressedBlockDim(
    ChNumFormat format)
{
    Extent3d blockDim = {};

    switch (format)
    {
    case ChNumFormat::Bc1_Unorm:
    case ChNumFormat::Bc1_Srgb:
    case ChNumFormat::Bc2_Unorm:
    case ChNumFormat::Bc2_Srgb:
    case ChNumFormat::Bc3_Unorm:
    case ChNumFormat::Bc3_Srgb:
    case ChNumFormat::Bc4_Unorm:
    case ChNumFormat::Bc4_Snorm:
    case ChNumFormat::Bc5_Unorm:
    case ChNumFormat::Bc5_Snorm:
    case ChNumFormat::Bc6_Ufloat:
    case ChNumFormat::Bc6_Sfloat:
    case ChNumFormat::Bc7_Unorm:
    case ChNumFormat::Bc7_Srgb:
        blockDim.width  = CompressedBcBlockDim;
        blockDim.height = CompressedBcBlockDim;
        blockDim.depth  = 1;
        break;
    case ChNumFormat::Etc2X8Y8Z8_Unorm:
    case ChNumFormat::Etc2X8Y8Z8_Srgb:
    case ChNumFormat::Etc2X8Y8Z8W1_Unorm:
    case ChNumFormat::Etc2X8Y8Z8W1_Srgb:
    case ChNumFormat::Etc2X8Y8Z8W8_Unorm:
    case ChNumFormat::Etc2X8Y8Z8W8_Srgb:
    case ChNumFormat::Etc2X11_Unorm:
    case ChNumFormat::Etc2X11_Snorm:
    case ChNumFormat::Etc2X11Y11_Unorm:
    case ChNumFormat::Etc2X11Y11_Snorm:
        blockDim.width  = CompressedEtcBlockDim;
        blockDim.height = CompressedEtcBlockDim;
        blockDim.depth  = 1;
        break;
    case ChNumFormat::AstcLdr4x4_Unorm:
    case ChNumFormat::AstcLdr4x4_Srgb:
    case ChNumFormat::AstcHdr4x4_Float:
        blockDim.width  = 4;
        blockDim.height = 4;
        blockDim.depth  = 1;
        break;
    case ChNumFormat::AstcLdr5x4_Unorm:
    case ChNumFormat::AstcLdr5x4_Srgb:
    case ChNumFormat::AstcHdr5x4_Float:
        blockDim.width  = 5;
        blockDim.height = 4;
        blockDim.depth  = 1;
        break;
    case ChNumFormat::AstcLdr5x5_Unorm:
    case ChNumFormat::AstcLdr5x5_Srgb:
    case ChNumFormat::AstcHdr5x5_Float:
        blockDim.width  = 5;
        blockDim.height = 5;
        blockDim.depth  = 1;
        break;
    case ChNumFormat::AstcLdr6x5_Unorm:
    case ChNumFormat::AstcLdr6x5_Srgb:
    case ChNumFormat::AstcHdr6x5_Float:
        blockDim.width  = 6;
        blockDim.height = 5;
        blockDim.depth  = 1;
        break;
    case ChNumFormat::AstcLdr6x6_Unorm:
    case ChNumFormat::AstcLdr6x6_Srgb:
    case ChNumFormat::AstcHdr6x6_Float:
        blockDim.width  = 6;
        blockDim.height = 6;
        blockDim.depth  = 1;
        break;
    case ChNumFormat::AstcLdr8x5_Unorm:
    case ChNumFormat::AstcLdr8x5_Srgb:
    case ChNumFormat::AstcHdr8x5_Float:
        blockDim.width  = 8;
        blockDim.height = 5;
        blockDim.depth  = 1;
        break;
    case ChNumFormat::AstcLdr8x6_Unorm:
    case ChNumFormat::AstcLdr8x6_Srgb:
    case ChNumFormat::AstcHdr8x6_Float:
        blockDim.width  = 8;
        blockDim.height = 6;
        blockDim.depth  = 1;
        break;
    case ChNumFormat::AstcLdr8x8_Unorm:
    case ChNumFormat::AstcLdr8x8_Srgb:
    case ChNumFormat::AstcHdr8x8_Float:
        blockDim.width  = 8;
        blockDim.height = 8;
        blockDim.depth  = 1;
        break;
    case ChNumFormat::AstcLdr10x5_Unorm:
    case ChNumFormat::AstcLdr10x5_Srgb:
    case ChNumFormat::AstcHdr10x5_Float:
        blockDim.width  = 10;
        blockDim.height = 5;
        blockDim.depth  = 1;
        break;
    case ChNumFormat::AstcLdr10x6_Unorm:
    case ChNumFormat::AstcLdr10x6_Srgb:
    case ChNumFormat::AstcHdr10x6_Float:
        blockDim.width  = 10;
        blockDim.height = 6;
        blockDim.depth  = 1;
        break;
    case ChNumFormat::AstcLdr10x8_Unorm:
    case ChNumFormat::AstcLdr10x8_Srgb:
    case ChNumFormat::AstcHdr10x8_Float:
        blockDim.width  = 10;
        blockDim.height = 8;
        blockDim.depth  = 1;
        break;
    case ChNumFormat::AstcLdr10x10_Unorm:
    case ChNumFormat::AstcLdr10x10_Srgb:
    case ChNumFormat::AstcHdr10x10_Float:
        blockDim.width  = 10;
        blockDim.height = 10;
        blockDim.depth  = 1;
        break;
    case ChNumFormat::AstcLdr12x10_Unorm:
    case ChNumFormat::AstcLdr12x10_Srgb:
    case ChNumFormat::AstcHdr12x10_Float:
        blockDim.width  = 12;
        blockDim.height = 10;
        blockDim.depth  = 1;
        break;
    case ChNumFormat::AstcLdr12x12_Unorm:
    case ChNumFormat::AstcLdr12x12_Srgb:
    case ChNumFormat::AstcHdr12x12_Float:
        blockDim.width  = 12;
        blockDim.height = 12;
        blockDim.depth  = 1;
        break;
    default:
        // This function should not be called on a non-compressed format.
        PAL_ASSERT_ALWAYS();
        break;
    }

    return blockDim;
}

/// Convert a compressed format block coordinate to texels.
///
/// @param [in]  format      Format.
/// @param [in]  width       Block width.
/// @param [in]  height      Block height.
/// @param [in]  depth       Block depth.
///
/// @returns Structure containing the texel width, height and depth
inline Extent3d CompressedBlocksToTexels(
    ChNumFormat format,
    uint32      width,
    uint32      height,
    uint32      depth)
{
    Extent3d dims = CompressedBlockDim(format);
    dims.width  *= width;
    dims.height *= height;
    dims.depth  *= depth;
    return dims;
}

/// Convert a compressed format texel coordinate to blocks.
///
/// @param [in] format      Format.
/// @param [in] width       Texel width.
/// @param [in] height      Texel height.
/// @param [in] depth       Texel depth.
///
/// @returns Structure containing the block width, height and depth
inline Extent3d CompressedTexelsToBlocks(
    ChNumFormat format,
    uint32      width,
    uint32      height,
    uint32      depth)
{
    Extent3d dims = CompressedBlockDim(format);
    dims.width  = Util::RoundUpQuotient(width,  dims.width);
    dims.height = Util::RoundUpQuotient(height, dims.height);
    dims.depth  = Util::RoundUpQuotient(depth,  dims.depth);
    return dims;
}

/// Queries the number of bits in a pixel or element for the given format.
///
/// @param format The format to query for.
///
/// @return The number of bits per pixel for the given channel format.
inline uint32 BitsPerPixel(
    ChNumFormat format)
{
    return FormatInfoTable[static_cast<size_t>(format)].bitsPerPixel;
}

/// Queries the number of bits in a pixel or element for the given format.
///
/// @param format The format to query for.
///
/// @return The number of bytes per pixel for the given channel format.
inline uint32 BytesPerPixel(
    ChNumFormat format)
{
    return (BitsPerPixel(format) >> 3);
}

/// Checks if the specified channel swizzle is allowed with the given format.
///
/// @param [in] format  The pixel format to check against.
/// @param [in] swizzle The specified channel swizzle to check with.
///
/// @returns True if the specified channel swizzle is valid for the given format. False otherwise.
inline bool IsValidChannelSwizzle(
    ChNumFormat    format,
    ChannelSwizzle swizzle)
{
    const uint32 mask = ComponentMask(format);

    bool valid = false;
    switch (swizzle)
    {
    case ChannelSwizzle::Zero:
    case ChannelSwizzle::One:
        valid = true;
        break;
    case ChannelSwizzle::X:
        valid = ((mask & ChannelFlags::X) != 0);
        break;
    case ChannelSwizzle::Y:
        valid = ((mask & ChannelFlags::Y) != 0);
        break;
    case ChannelSwizzle::Z:
        valid = ((mask & ChannelFlags::Z) != 0);
        break;
    case ChannelSwizzle::W:
        valid = ((mask & ChannelFlags::W) != 0);
        break;
    default:
        PAL_NEVER_CALLED();
        break;
    }

    return valid;
}

/// Queries the per-component bit counts for a particular format.
///
/// @param [in] format The format to query for.
///
/// @returns The corresponding component swizzles for the specified format. Returned as an array of four counts.
inline const uint32* ComponentBitCounts(
    ChNumFormat format)
{
    return &FormatInfoTable[static_cast<size_t>(format)].bitCount[0];
}

/// Determines the maximum bit-count of any component in the format.
///
/// @param [in] format The channel format to query for.
///
/// @returns The maximum bit-count of any component in the format.
inline uint32 MaxComponentBitCount(
    ChNumFormat format)
{
    const FormatInfo& info = FormatInfoTable[static_cast<size_t>(format)];

    return Util::Max(Util::Max(info.bitCount[0], info.bitCount[1]), Util::Max(info.bitCount[2], info.bitCount[3]));
}

/// Checks if the specified format is one of the block-compressed ones.
///
/// @param [in] format The format to check.
///
/// @returns True if the specified format is block-compressed. False otherwise.
inline bool IsBlockCompressed(
    ChNumFormat format)
{
    return ((FormatInfoTable[static_cast<size_t>(format)].properties & BlockCompressed) != 0);
}

/// Checks if the specified format is one of the macro-pixel-packed ones.
///
/// @param [in] format The format to check.
///
/// @returns True if the specified format is macro-pixel-packed. False otherwise.
inline bool IsMacroPixelPacked(
    ChNumFormat format)
{
    return ((FormatInfoTable[static_cast<size_t>(format)].properties & MacroPixelPacked) != 0);
}

/// Checks if the specified format is one of the rgb macro-pixel-packed ones.
///
/// @param [in] format The format to check.
///
/// @returns True if the specified format is a rgb macro-pixel-packed. False otherwise.
inline bool IsMacroPixelPackedRgbOnly(
    ChNumFormat format)
{
    return (IsMacroPixelPacked(format) && (IsYuv(format) == false));
}

/// Returns the base-2 logarithm of of the subsampling ratio between the luma plane and chroma plane(s) of a YUV planar
/// format. The dimensions of the luma plane should be right-shifted by these amounts to determine the dimensions of the
/// chroma plane(s).
///
/// @param [in] format  Format.
/// @param [in] plane   Image plane to query for.
///
/// @returns Corresponding scaling factors between the luma plane and chroma plane(s).
inline Extent3d Log2SubsamplingRatio(
    ChNumFormat format,
    uint32      plane)
{
    // All planes for formats which are not YUV planar, and the 0th plane of a YUV planar format (the luma plane) are
    // sampled at full rate, so the ratio is { log2(1), log2(1), log2(1) }, which equates to { 0,0,0 }.
    Extent3d ratio = { };

    if (IsYuvPlanar(format) && (plane != 0))
    {
        PAL_ASSERT((plane == 1) || (plane == 2));
        switch (format)
        {
        // 4:4:4 formats have the same number of samples in every direction.
        case ChNumFormat::P412:
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 924
        case ChNumFormat::YUV_444P10:
        case ChNumFormat::YUV_444P12:
        case ChNumFormat::YUV_444P16:
#endif
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 925
        case ChNumFormat::YV24:
        case ChNumFormat::NV24:
        case ChNumFormat::P410:
        case ChNumFormat::P416:
#endif
            break;
        // 4:2:0 formats have 1/2 as many samples in both the horizontal and vertical directions.
        case ChNumFormat::YV12:
        case ChNumFormat::NV12:
        case ChNumFormat::NV21:
        case ChNumFormat::P010:
        case ChNumFormat::P012:
        case ChNumFormat::P016:
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 924
        case ChNumFormat::YUV_420P10:
        case ChNumFormat::YUV_420P12:
        case ChNumFormat::YUV_420P16:
#endif
            ratio.width  = 1;  // log2(1/2) = -1
            ratio.height = 1;
            break;
        // 4:2:2 formats have 1/2 as many samples in the horizontal direction, and the same number of samples
        // in the vertical direction.
        case ChNumFormat::P208:
        case ChNumFormat::P210:
        case ChNumFormat::P212:
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 923
        case ChNumFormat::P216:
#endif
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 924
        case ChNumFormat::YUV_422P10:
        case ChNumFormat::YUV_422P12:
        case ChNumFormat::YUV_422P16:
#endif
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 925
        case ChNumFormat::YV16:
#endif
            ratio.width = 1;
            break;
        // 4:1:1 formats have 1/4 as many samples in the horizontal direction, and the same number of samples
        // in the vertical direction.
        case ChNumFormat::NV11:
            ratio.width = 2;   // log2(1/4) = -2
            break;
        default:
            PAL_NEVER_CALLED(); // Did we miss a new YUV planar format?
            break;
        }
    }

    return ratio;
}

/// Converts a linearly-scaled color value to gamma-corrected sRGB.
///
/// @param [in] linear Linear color value
///
/// @returns Gamma-corrected sRGB color value
extern float LinearToGamma(float linear);

/// Converts a gamma-corrected sRGB color value to linear color space.
///
/// @param [in] gammaCorrectedVal Gamma-corrected sRGB color value
///
/// @returns Linear color value
extern float GammaToLinear(float gammaCorrectedVal);

/// Checks to see if a given format is a MM format
///
///
/// @returns bool is it an MM format
extern bool IsMmFormat(ChNumFormat format);

/// Checks to see if a given format is a MM12 format
///
///
/// @returns bool is it an MM12 format
extern bool IsMm12Format(ChNumFormat format);

/// Checks to see if a given format is a MM10 format
///
///
/// @returns bool is it an MM10 format
extern bool IsMm10Format(ChNumFormat format);

} // Formats
} // Pal
