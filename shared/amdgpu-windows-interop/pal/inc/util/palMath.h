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
 * @file  palMath.h
 * @brief PAL utility collection function/constant declarations for the Math sub-namespace.
 ***********************************************************************************************************************
 */

#pragma once

#include "palSysMemory.h"

#include <limits>

namespace Util
{

/// Util sub-namespace defining several useful math routines and constants.
namespace Math
{

/// Exponent mask of a single-precision IEEE float.
constexpr uint32 FloatExponentMask      = 0x7F800000;
/// Exponent bias of a single-precision IEEE float.
constexpr uint32 FloatExponentBias      = 127;
/// Number of bits in the mantissa of a single-precision IEEE float.
constexpr uint32 FloatNumMantissaBits   = 23;
/// Mantissa mask of a single-precision IEEE float.
constexpr uint32 FloatMantissaMask      = 0x007FFFFF;
/// Sign bit mask of a single precision IEEE float.
constexpr uint32 FloatSignBitMask       = 0x80000000;
/// Mask of all non-sign bits of a single-precision IEEE float.
constexpr uint32 FloatMaskOutSignBit    = 0x7FFFFFFF;
/// Minimum number of float bits in a normalized IEE float.
constexpr uint32 MinNormalizedFloatBits = 0x00800000;

/// Positive one.
constexpr float FloatOne      = 1.0f;
/// Negative one.
constexpr float FloatNegOne   = -1.0f;
/// Zero.
constexpr float FloatZero     = 0.0f;
/// Positive infinity.
constexpr float FloatInfinity = std::numeric_limits<float>::infinity();

/// Fraction structure.
struct Fraction
{
    uint32 num; ///< Numerator
    uint32 den; ///< Denominator
};

/// Returns the bits of a floating point value as an unsigned integer.
inline uint32 FloatToBits(float f)
{
    return (*(reinterpret_cast<uint32*>(&f)));
}

/// Assigns the bits contained in an unsigned integer to the float pointer location
inline void SetBitsToFloat(float* f, uint32 u)
{
    *(reinterpret_cast<uint32*>(f)) = u;
}

/// Returns true if the specified float is denormalized.
extern bool IsDenorm(float f);
/// Returns true if the specified float is +/- infinity.
extern bool IsInf(float f);
/// Returns true if the specified float is a NaN.
extern bool IsNaN(float f);

/// Determines if a floating-point number is either +/-Infinity or NaN.
inline bool IsInfOrNaN(float f)
{
    return (IsInf(f) || IsNaN(f));
}

/// @brief Converts a floating point number to a signed fixed point number with the given integer and fractional bits.
///
/// If the number of integer bits is zero, the incoming value is treated as normalized, i.e. [-1.0, 1.0].  If the
/// intBits is zero, the fracBits is assumed to include 1 sign bit, otherwise the sign bit is assumed to be part of the
/// intBits.  A typical use for enableRounding would be when converting SNORM/UNORM values to fixed point.
///
/// @param [in] f              Floating point value to convert.
/// @param [in] intBits        Number of integer bits (including the sign bit) in the fixed point output.
/// @param [in] fracBits       Number of fractional bits in the fixed point output.
/// @param [in] enableRounding Round before conversion.
///
/// @returns Fixed point number in a uint32.
extern uint32 FloatToSFixed(float f, uint32 intBits, uint32 fracBits, bool enableRounding = false);

/// @brief Converts a floating point number to an unsigned fixed point number with the given integer and
///        fractional bits.
///
/// If the number of integer bits is zero, the incoming value is treated as normalized, i.e. [-1.0, 1.0].  A typical use
/// for enableRounding would be when converting SNORM/UNORM values to fixed point.
///
/// @param [in] f              Floating point value to convert.
/// @param [in] intBits        Number of integer bits (including the sign bit) in the fixed point output.
/// @param [in] fracBits       Number of fractional bits in the fixed point output.
/// @param [in] enableRounding Round before conversion.
///
/// @returns Fixed point number in a uint32.
extern uint32 FloatToUFixed(float f, uint32 intBits, uint32 fracBits, bool enableRounding = false);

/// @brief Converts a signed fixed point number with the given integer and fractional bits to a floating point number.
///
/// If the number of integer bits is zero, the incoming value is treated as normalized, i.e. [-1.0, 1.0].  If numIntBits
/// is 0, numFracBits is assumed to have 1 bit for the sign, otherwise the sign bit is assumed to be part of the integer
/// bits.
///
/// @param [in] fixedPtNum Fixed point number to convert.
/// @param [in] intBits    Number of integer bits (including the sign bit).
/// @param [in] fracBits   Number of fractional bits.
///
/// @returns Converted floating point number.
extern float SFixedToFloat(int32 fixedPtNum, uint32 intBits, uint32 fracBits);

/// @brief Converts a unsigned fixed point number with the given integer and fractional bits to a floating point number.
///
/// If the number of integer bits is zero, the incoming value is treated as normalized, i.e. [0, 1.0].
///
/// @param [in] fixedPtNum Fixed point number to convert.
/// @param [in] intBits    Number of integer bits (including the sign bit).
/// @param [in] fracBits   Number of fractional bits.
///
/// @returns Converted floating point number.
extern float UFixedToFloat(uint32 fixedPtNum, uint32 intBits, uint32 fracBits);

/// Converts a 32-bit IEEE floating point number to a 16-bit signed floating point number.
extern uint32 Float32ToFloat16(float f);

/// Converts a 32-bit IEEE floating point number to an 11-bit signed floating point number.
extern uint32 Float32ToFloat11(float f);

/// Converts a 32-bit IEEE floating point number to a 10-bit signed floating point number.
extern uint32 Float32ToFloat10(float f);

/// Converts a 32-bit IEEE floating-point number to a 10-bit unsigned floating-point number.
extern uint32 Float32ToFloat10_6e4(float f);

/// Converts a 10-bit signed floating point number to a 32-bit IEEE floating point number.
extern float Float10_6e4ToFloat32(uint32 fBits);

/// Converts a 32-bit IEEE floating point number to a N-bit signed floating point number.
extern uint32 Float32ToNumBits(float float32, uint32 numBits);

/// Converts a 16-bit signed floating point number to a 32-bit IEEE floating point number.
extern float Float16ToFloat32(uint32 fBits);

/// Converts an 11-bit signed floating point number to a 32-bit IEEE floating point number.
extern float Float11ToFloat32(uint32 fBits);

/// Converts a 10-bit signed floating point number to a 32-bit IEEE floating point number.
extern float Float10ToFloat32(uint32 fBits);

/// Converts an N-bit signed floating point number to a 32-bit IEEE floating point number.
extern float FloatNumBitsToFloat32(uint32 input, uint32  numBits);

/// Converts a 32-bit IEEE floating point number to a fraction.
extern Fraction Float32ToFraction(float float32);

/// Returns the square root of the specified value.
extern float Sqrt(float f);

/// Returns the result of an exponent operation (base^exponent).
extern float Pow(float base, float exponent);

/// Returns the unsigned integer absolute value.
extern uint32 Absu(int32 number);

/// Return sign-preserved zero if input is denorm, otherwise input value
extern float FlushDenormToZero(float input);

/// Return value in 1.7 signed magnitude format. Valid input range is (-127, 127)
extern uint8 IntToSignedMagnitude(int8 input);

/// @brief Performs unsigned fixed-point rounding operation.
///
/// @param [in] value      Fixed point number to convert in Qm.f format.
/// @param [in] n          Number of fractional bits.
///
/// @returns rounded fixed point number in Q0 format (unsigned integer).
constexpr uint32 UFixedRoundToUint32(uint32 value, uint8 n)
{
    PAL_CONSTEXPR_ASSERT((0 < n) && (n < 31));
    return ((value + (((1 << n) >> 1))) >> n);
}

/// @brief Performs signed fixed-point rounding operation.
///
/// @param [in] value      Fixed point number to convert in Qm.f format.
/// @param [in] n          Number of fractional bits.
///
/// @returns rounded fixed point number in Q0 format (signed integer).
constexpr int32 SFixedRoundToInt32(int32 value, uint8 n)
{
    PAL_CONSTEXPR_ASSERT((0 < n) && (n < 30));
    return ((value + (((1 << n) >> 1))) >> n);
}

} // Math
} // Util
