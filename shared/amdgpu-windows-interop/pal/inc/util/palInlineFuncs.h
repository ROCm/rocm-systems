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
 * @file  palInlineFuncs.h
 * @brief PAL utility collection inline functions.
 ***********************************************************************************************************************
 */

#pragma once

#include "palAssert.h"
#include "palStringUtil.h"
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <iterator>
#include <type_traits>
#include <limits>

namespace Util
{

/// Describes a value type, primarily used for loading settings values.
enum class ValueType : uint32
{
    Boolean,       ///< Boolean type.
    Int8,          ///< 8-bit integer type.
    Uint8,         ///< 8-bit unsigned integer type.
    Int16,         ///< 16-bit integer type.
    Uint16,        ///< 16-bit unsigned integer type.
    Int32,         ///< 32-bit integer type.
    Uint32,        ///< 32-bit unsigned integer type.
    Int64,         ///< 64-bit integer type.
    Uint64,        ///< 64-bit unsigned integer type.
    Float,         ///< Floating point type.
    Str,           ///< String type.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 905
    Int = Int32,   ///< Signed integer type.
    Uint = Uint32, ///< Unsigned integer type.
#endif
};

/// Determines the length of an array at compile-time.
///
/// @returns The length of the array.
template <typename T, size_t N>
constexpr size_t ArrayLen(
    const T (&array)[N]) ///< The array of arbitrary type T.
{
    return N;
}

/// Determines the 32-bit length of an array at compile-time.
///
/// @returns The length of the array.
template <typename T, uint32 N>
constexpr uint32 ArrayLen32(
    const T (&array)[N]) ///< The array of arbitrary type T.
{
    return N;
}

/// Increments a const pointer by nBytes by first casting it to a const uint8*.
///
/// @returns Incremented pointer.
constexpr const void* VoidPtrInc(
    const void* p,         ///< [in] Pointer to be incremented.
    size_t      numBytes)  ///< Number of bytes to increment the pointer by.
{
    return (static_cast<const uint8*>(p) + numBytes);
}

/// Increments a pointer by nBytes by first casting it to a uint8*.
///
/// @returns Incremented pointer.
constexpr void* VoidPtrInc(
    void*  p,         ///< [in] Pointer to be incremented.
    size_t numBytes)  ///< Number of bytes to increment the pointer by.
{
    return (static_cast<uint8*>(p) + numBytes);
}

/// Decrements a const pointer by nBytes by first casting it to a const uint8*.
///
/// @returns Decremented pointer.
constexpr const void* VoidPtrDec(
    const void* p,         ///< [in] Pointer to be decremented.
    size_t      numBytes)  ///< Number of bytes to decrement the pointer by.
{
    return (static_cast<const uint8*>(p) - numBytes);
}

/// Decrements a pointer by nBytes by first casting it to a uint8*.
///
/// @returns Decremented pointer.
constexpr void* VoidPtrDec(
    void*  p,         ///< [in] Pointer to be decremented.
    size_t numBytes)  ///< Number of bytes to decrement the pointer by.
{
    return (static_cast<uint8*>(p) - numBytes);
}

/// Finds the number of bytes between two pointers by first casting them to uint8*.
///
/// This function expects the first pointer to not be smaller than the second.
///
/// @returns Number of bytes between the two pointers.
constexpr size_t VoidPtrDiff(
    const void* p1,  ///< [in] First pointer (higher address).
    const void* p2)  ///< [in] Second pointer (lower address).
{
    PAL_CONSTEXPR_ASSERT(p1 >= p2);
    return (static_cast<const uint8*>(p1) - static_cast<const uint8*>(p2));
}

/// Returns the high 32 bits of a 64-bit integer.
///
/// @returns Returns the high 32 bits of a 64-bit integer.
constexpr uint32 HighPart(
    uint64 value)  ///< 64-bit input value.
{
    return (value & 0xFFFFFFFF00000000) >> 32;
}

/// Returns the low 32 bits of a 64-bit integer.
///
/// @returns Returns the low 32 bits of a 64-bit integer.
constexpr uint32 LowPart(
    uint64 value)  ///< 64-bit input value.
{
    return (value & 0x00000000FFFFFFFF);
}

/// Returns the high 32 bits of a 64-bit integer as a 64-bit integer.
///
/// @returns Returns the high 32 bits of a 64-bit integer as a 64-bit integer
/// without shifting
constexpr uint64 HighPart64(
    uint64 value)  ///< 64-bit input value.
{
    return (value & 0xFFFFFFFF00000000);
}

/// Combines the low and high 32 bits of a 64-bit integer.
///
/// @returns Returns the 64-bit integer.
constexpr uint64 Uint64CombineParts(
    uint32 lowPart,
    uint32 highPart)
{
    return (uint64(highPart) << 32) | uint64(lowPart);
}

/// Returns a larger value from repeating a single byte
constexpr uint32 ReplicateByteAcrossDword(
    uint8 value)  ///< 8-bit input value.
{
    return (value | (value << 8) | (value << 16) | (value << 24));
}

/// Returns a larger value from repeating a single byte
constexpr uint64 ReplicateByteAcrossQword(
    uint8 value)  ///< 8-bit input value.
{
    return ((static_cast<uint64>(ReplicateByteAcrossDword(value)) << 32) | ReplicateByteAcrossDword(value));
}

/// Combines four characters into a uint32-based four-character-code "string". There's no null terminator so it's not a
/// real c-string, it just looks like there's a string if you view the uint in a hex editor or memcmp against a string.
///
/// For example, FourCC('A', 'B', 'C', 'D') turns into 0x44434241. 'A' is 0x41 and it ends up in the first byte.
/// This function assumes we're running on a little endian platform (PAL only supports little-endian platforms).
///
/// @returns Returns a uin32 four-character-code made from the given chars.
constexpr uint32 FourCc(
    char c1, ///< The 1st character (lowest byte).
    char c2, ///< The 2nd character.
    char c3, ///< The 3rd character.
    char c4) ///< The 4th character (highest byte).
{
    return (uint32(c4) << 24) | (uint32(c3) << 16) | (uint32(c2) << 8) | uint32(c1);
}

/// Returns a bitfield from within some value.
///
/// @returns Returns a bitfield from within some value.
template <typename T>
constexpr T BitExtract(
    T      value,    ///< Extract a bitfield from here.
    uint32 firstBit, ///< The zero-based index of the first bit to extract.
    uint32 lastBit)  ///< The zero-based index of the last bit to extract.
{
    return (value >> firstBit) & ((1 << (lastBit - firstBit + 1)) - 1);
}

/// Determines if any of the bits set in "test" are also set in "src".
///
/// @returns True if any bits in "test" are set in "src", false otherwise.
constexpr bool TestAnyFlagSet(
    uint32 src,   ///< Source pattern.
    uint32 test)  ///< Test pattern.
{
    return ((src & test) != 0);
}

/// Determines if all of the bits set in "test" are also set in "src".
///
/// @returns True if all bits set in "test" are also set in "src", false otherwise.
constexpr bool TestAllFlagsSet(
    uint32 src,   ///< Source pattern.
    uint32 test)  ///< Test pattern.
{
    return ((src & test) == test);
}

/// Determines if any of the bits set in "test" are also set in "src".
///
/// @returns True if any bits in "test" are set in "src", false otherwise.
constexpr bool TestAnyFlagSet64(
    uint64 src,   ///< Source pattern.
    uint64 test)  ///< Test pattern.
{
    return ((src & test) != 0);
}

/// Determines if all of the bits set in "test" are also set in "src".
///
/// @returns True if all bits set in "test" are also set in "src", false otherwise.
constexpr bool TestAllFlagsSet64(
    uint64 src,   ///< Source pattern.
    uint64 test)  ///< Test pattern.
{
    return ((src & test) == test);
}

/// Tests if a single bit in a bitfield is set.
///
/// @param [in] bitfield  Bitfield being tested
/// @param [in] bit       Bit index to test
///
/// @returns True if the flag is set.
template <typename T>
constexpr bool BitfieldIsSet(
    const T bitfield,
    uint32  bit)
{
    PAL_CONSTEXPR_ASSERT(bit < (sizeof(T) * 8));
    return (bitfield & (static_cast<T>(1) << bit));
}

/// Sets a single bit in a bitfield to one.
///
/// @param [in] bitfield  Reference to the bitfield being modified
/// @param [in] bit       Index of the bit to set
template <typename T>
void BitfieldSetBit(
    T      &bitfield,
    uint32 bit)
{
    PAL_CONSTEXPR_ASSERT(bit < (sizeof(T) * 8));
    bitfield |= (static_cast<T>(1) << bit);
}

///@{
/// Counts the number of one bits (population count) in an unsigned integer using some bitwise magic explained in the
/// Software Optimization Guide for AMD64 Processors.
///
/// @param [in] value  The value need to be counted.
///
/// @returns Number of one bits in the input
template <typename T>
constexpr uint32 CountSetBits(
    T  value)
{
    uint32 x = static_cast<uint32>(value);

    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> ((sizeof(uint32) - 1) << 3);

    return x;
}

constexpr uint32 CountSetBits(
    uint64  value)
{
    uint64 x = value;

    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    x = (((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full) * 0x0101010101010101ull) >> ((sizeof(uint64) - 1) << 3);

    return static_cast<uint32>(x);
}
///@}

/// Update a subfield of a bitfield.
///
/// @param [in] bitFieldToUpdate  Bitfield being updated
/// @param [in] updateValue       Source value to update
/// @param [in] updateMask        Bitmask to update
///
/// @returns True if the flag is set.
template <typename T>
void BitfieldUpdateSubfield(
    T*      pBitFieldToUpdate,
    const T updateValue,
    const T updateMask)
{
    *pBitFieldToUpdate = ((*pBitFieldToUpdate) & ~updateMask) |
                          (updateValue         &  updateMask);
}

/// Tests if a single bit in a "wide bitfield" is set. A "wide bifield" is a bitfield which spans an array of
/// integers because there are more flags than bits in one integer.
///
/// @param [in] bitfield  Reference to the bitfield being tested
/// @param [in] bit       Index of the flag to test
///
/// @returns True if the flag is set.
template <typename T, size_t N>
constexpr bool WideBitfieldIsSet(
    const T (&bitfield)[N],
    uint32  bit)
{
    const uint32 index = (bit / (sizeof(T) << 3));
    const T      mask  = (static_cast<T>(1) << (bit & ((sizeof(T) << 3) - 1)));

    return (0 != (bitfield[index] & mask));
}

/// Checks if any bit is set in a wide bitfield. A "wide bitfield" is a bitfield which spans an array of
/// integers because there are more flags than bits in one integer.
///
/// @param [in] bitfield  Wide bitfield to count.
///
/// @returns True if the wide bitfield is non-zero; false otherwise.
template <typename T, size_t N>
bool WideBitfieldIsAnyBitSet(
    const T(&bitfield)[N])
{
    bool isBitSet = false;
    for (uint32 i = 0; i < N; i++)
    {
        isBitSet |= (bitfield[i] != 0);
    }

    return isBitSet;
}

/// Sets a single bit in a "wide bitfield" to one. A "wide bifield" is a bitfield which spans an array of
/// integers because there are more flags than bits in one integer.
///
/// @param [in] bitfield  Reference to the bitfield being modified
/// @param [in] bit       Index of the flag to set
template <typename T, size_t N>
void WideBitfieldSetBit(
    T      (&bitfield)[N],
    uint32 bit)
{
    const uint32 index = (bit / (sizeof(T) << 3));
    const T      mask  = (static_cast<T>(1) << (bit & ((sizeof(T) << 3) - 1)));

    bitfield[index] |= mask;
}

/// Clears a single bit in a "wide bitfield" to zero. A "wide bifield" is a bitfield which spans an array of
/// integers because there are more flags than bits in one integer.
///
/// @param [in] bitfield  Reference to the bitfield being modified
/// @param [in] bit       Index of the flag to set
template <typename T, size_t N>
void WideBitfieldClearBit(
    T      (&bitfield)[N],
    uint32 bit)
{
    const uint32 index = (bit / (sizeof(T) << 3));
    const T      mask  = (static_cast<T>(1) << (bit & ((sizeof(T) << 3) - 1)));

    bitfield[index] &= ~mask;
}

/// Sets consecutive bits in a "wide bitfield" to one. A "wide bifield" is a bitfield which spans an array of
/// integers because there are more flags than bits in one integer.
///
/// @param [in] bitfield    Reference to the bitfield being modified
/// @param [in] startingBit Index of the first flag to set
/// @param [in] numBits     Count of consecutive flags to set
template <typename T, size_t N>
void WideBitfieldSetRange(
    T      (&bitfield)[N],
    uint32 startingBit,
    uint32 numBits)
{
    constexpr uint32 SizeInBits = (sizeof(T) << 3);

    PAL_ASSERT((startingBit + numBits) <= (SizeInBits * N));

    uint32 index = (startingBit / SizeInBits);

    startingBit &= (SizeInBits - 1);

    while (numBits > 0)
    {
        const uint32 maxNumBits     = SizeInBits - startingBit;
        const uint32 curNumBits     = (maxNumBits < numBits) ? maxNumBits : numBits;
        const T      bitMask        = (curNumBits == SizeInBits) ? -1 : ((static_cast<T>(1) << curNumBits) - 1);

        bitfield[index++] |= (bitMask << startingBit);

        startingBit  = 0;
        numBits     -= curNumBits;
    }
}

/// XORs all of the bits in two "wide bitfields". A "wide bifield" is a bitfield which spans an array of integers
/// because there are more flags than bits in one integer.
///
/// @param [in]  bitfield1 Reference to the first bitfield.
/// @param [in]  bitfield2 Reference to the second bitfield.
/// @param [out] pOut      Result of (bitfield1 ^ bitfield2)
template <typename T, size_t N>
void WideBitfieldXorBits(
    const T (&bitfield1)[N],
    const T (&bitfield2)[N],
    T*       pOut)
{
    for (uint32 i = 0; i < N; i++)
    {
        pOut[i] = (bitfield1[i] ^ bitfield2[i]);
    }
}

/// ANDs all of the bits in two "wide bitfields". A "wide bifield" is a bitfield which spans an array of integers
/// because there are more flags than bits in one integer.
///
/// @param [in]  bitfield1 Reference to the first bitfield.
/// @param [in]  bitfield2 Reference to the second bitfield.
/// @param [out] pOut      Result of (bitfield1 & bitfield2)
template <typename T, size_t N>
void WideBitfieldAndBits(
    const T (&bitfield1)[N],
    const T (&bitfield2)[N],
    T*      pOut)
{
    for (uint32 i = 0; i < N; i++)
    {
        pOut[i] = (bitfield1[i] & bitfield2[i]);
    }
}

/// Counts the number of one bits (population count) in a wide bitfield. A "wide bitfield" is a bitfield which spans
/// an array of integers because there are more flags than bits in one integer.
///
/// @param [in] bitfield  Wide bitfield to count.
///
/// @returns Number of one bits in the input
template <typename T, size_t N>
uint32 WideBitfieldCountSetBits(
    const T(&bitfield)[N])
{
    uint32 count = 0;
    for (uint32 i = 0; i < N; i++)
    {
        count += CountSetBits(bitfield[i]);
    }

    return count;
}

/// Unsets the least-significant '1' bit in the given number.
/// Usually used in conjunction with BitMaskScanForward
///
/// @param [in] value  The value to be modified
///
/// @returns A copy of value with the lowest '1' bit unset.
template <typename T>
T UnsetLeastBit(
    T val)
{
    static_assert(std::is_unsigned<T>::value, "Must use unsigned ints here");
    return val & (val - 1);
}

/// Scans the specified bit-mask for the least-significant '1' bit.
///
/// @returns True if the input was nonzero; false otherwise.
template <typename T>
bool BitMaskScanForward(
    uint32* pIndex,  ///< [out] Index of least-significant '1' bit.  Undefined if input is zero.
    T       mask)    ///< Bit-mask to scan.
{
    // Bitscan intrinsics may compile to flaky code in certain situations. Discarding bitscan flags avoids this. The key
    // is to forward declare result, and set it in a conditional branch after the bitscan. Be careful if modifying this.
    bool result = false;

    if (mask != 0)
    {
#if defined(_WIN64) && defined(_M_X64)
        *pIndex = (sizeof(T) > 4) ? static_cast<uint32>(::_tzcnt_u64(mask)) : (::_tzcnt_u32(static_cast<uint32>(mask)));
#elif defined(_WIN64)
        auto*const pOut = reinterpret_cast<unsigned long*>(pIndex);
        (sizeof(T) > 4) ? (::_BitScanForward64(pOut, mask)) : (::_BitScanForward(pOut, static_cast<uint32>(mask)));
#elif defined(_WIN32) && defined(_M_IX86)
        const uint32 lowPart = LowPart(mask);
        uint32 index = (::_tzcnt_u32(((sizeof(T) > 4) && (lowPart == 0)) ? HighPart(mask) : lowPart));
        *pIndex = ((sizeof(T) > 4) && (lowPart == 0)) ? (index + 32u) : index;
#elif defined(_WIN32)
        unsigned long index;
        const uint32 lowPart = LowPart(mask);
        (::_BitScanForward(&index, ((sizeof(T) > 4) && (lowPart == 0)) ? HighPart(mask) : lowPart));
        *pIndex = ((sizeof(T) > 4) && (lowPart == 0)) ? (index + 32u) : index;
#elif defined(__GNUC__)
        *pIndex = (sizeof(T) > 4) ? __builtin_ctzll(mask) : __builtin_ctz(static_cast<uint32>(mask));
#else
        uint32 index = 0;
        for (; ((mask & 0x1) == 0); mask >>= 1, ++index);
        *pIndex = index;
#endif

        result = true;
    }
    return result;
}

/// Scans the specified bit-mask for the most-significant '1' bit.
///
/// @returns True if the input was nonzero; false otherwise.
template <typename T>
bool BitMaskScanReverse(
    uint32* pIndex,  ///< [out] Index of most-significant '1' bit.  Undefined if input is zero.
    T       mask)    ///< Bit-mask to scan.
{
    // Bitscan intrinsics may compile to flaky code in certain situations. Discarding bitscan flags avoids this. The key
    // is to forward declare result, and set it in a conditional branch after the bitscan. Be careful if modifying this.
    bool result = false;

    if (mask != 0)
    {
#if defined(_WIN64)
        auto*const pOut = reinterpret_cast<unsigned long*>(pIndex);
        (sizeof(T) > 4) ? (::_BitScanReverse64(pOut, mask)) : (::_BitScanReverse(pOut, static_cast<uint32>(mask)));
#elif defined(_WIN32)
        unsigned long index;
        const uint32 highPart = (sizeof(T) > 4) ? HighPart(mask) : 0;
        (::_BitScanReverse(&index, ((sizeof(T) > 4) && (highPart != 0)) ? highPart : LowPart(mask)));
        *pIndex = ((sizeof(T) > 4) && (highPart != 0)) ? (index + 32u) : index;
#elif defined(__GNUC__)
        *pIndex = (sizeof(T) > 4) ? (63u - __builtin_clzll(mask)) : (31u - __builtin_clz(static_cast<uint32>(mask)));
#else
        uint32 index = 31u;
        for (; (((mask >> index) & 0x1) == 0); --index);
        *pIndex = index;
#endif

        result = true;
    }
    return result;
}

/// Scans the specified wide bit-mask for the least-significant '1' bit.
///
/// @returns True if input was nonzero; false otherwise.
template <typename T, size_t N>
bool WideBitMaskScanForward(
    uint32* pIndex,        ///< [out] Index of least-significant '1' bit.  Undefined if input is zero.
    const T (&mask)[N])    ///< Bit-mask to scan.
{
    uint32 maskIndex = ((*pIndex) / (sizeof(T) << 3));

    // Check to see if the wide bitmask has some bits set.
    uint32 index = 0;
    while ((mask[index] == 0) && (++index < N));
    bool result = (index < N);

    while (result == true)
    {
        result = BitMaskScanForward(pIndex, mask[maskIndex]);

        if (result == false)
        {
            ++maskIndex;
            result = (maskIndex < N);
        }
        else
        {
            (*pIndex) = (*pIndex) + (maskIndex * (sizeof(T) << 3));
            break;
        }
    }

    return result;
}

/// Scans the specified wide bit-mask for the most-significant '1' bit.
///
/// @returns True if input was nonzero; false otherwise.
template <typename T, size_t N>
bool WideBitMaskScanReverse(
    uint32* pIndex,        ///< [out] Index of most-significant '1' bit.  Undefined if input is zero.
    const T (&mask)[N])    ///< Bit-mask to scan.
{
    uint32 maskIndex = ((*pIndex) / (sizeof(T) << 3));

    // Check to see if the wide bitmask has some bits set.
    uint32 index = N - 1;
    while ((mask[index] == 0) && (--index > 0));
    bool result = (mask[index] != 0);

    while (result == true)
    {
        result = BitMaskScanReverse(pIndex, mask[maskIndex]);

        if (result == false)
        {
            const uint32 oldIndex = maskIndex--;
            result = (oldIndex != 0);
        }
        else
        {
            (*pIndex) = (*pIndex) + (maskIndex * (sizeof(T) << 3));
            break;
        }
    }

    return result;
}

/// Generates a bitmask.
///
/// @param [in] numBits  Number of bits to set (starting at 0)
///
/// @returns Bitmask in storage of type T with bits [0:numBits-1] set.
template <typename T>
constexpr T BitfieldGenMask(
    T numBits)
{
    PAL_CONSTEXPR_ASSERT(numBits <= (sizeof(T) * 8));

    const T mask = (numBits < (sizeof(T) * 8)) ? ((static_cast<T>(1) << (numBits)) - static_cast<T>(1)) : static_cast<T>(-1);
    return mask;
}

/// Determines if a value is a power of two.
///
/// @returns True if it is a power of two, false otherwise.
constexpr bool IsPowerOfTwo(
    uint64 value)  ///< Value to check.
{
    return (value == 0) ? false : ((value & (value - 1)) == 0);
}

/// Determines if 'value' is at least aligned to the specified power-of-2 alignment.
///
/// @returns True if aligned, false otherwise.
constexpr bool IsPow2Aligned(
    uint64 value,      ///< Value to check.
    uint64 alignment)  ///< Desired alignment.
{
    PAL_CONSTEXPR_ASSERT(IsPowerOfTwo(alignment));
    return ((value & (alignment - 1)) == 0);
}

/// Determines if 'ptr' is at least aligned to the specified power-of-2 alignment.
///
/// @returns True if aligned, false otherwise.
inline bool VoidPtrIsPow2Aligned(
    const void* ptr,   ///< Pointer to check.
    uint64 alignment)  ///< Desired alignment.
{
    PAL_ASSERT(IsPowerOfTwo(alignment));
    return ((reinterpret_cast<size_t>(ptr) & (alignment - 1)) == 0);
}

/// Rounds the specified uint 'value' up to the nearest value meeting the specified 'alignment'.  Only power of 2
/// alignments are supported by this function.
///
/// @returns Aligned value.
template <typename T>
constexpr T Pow2Align(
    T      value,      ///< Value to align.
    uint64 alignment)  ///< Desired alignment (must be a power of 2).
{
    PAL_CONSTEXPR_ASSERT(IsPowerOfTwo(alignment));
    return ((value + static_cast<T>(alignment) - 1) & ~(static_cast<T>(alignment) - 1));
}

/// Rounds the specified uint 'value' up to the nearest power of 2
///
/// @param [in] value  The value to pad.
///
/// @returns Power of 2 padded value.
template <typename T>
T Pow2Pad(
    T value)
{
    T ret = value;

    if ((value & (value - 1)) != 0)
    {
        uint32 lastBitIndex = 0;
        BitMaskScanReverse(&lastBitIndex, value);
        ret = (static_cast<T>(0x2) << lastBitIndex);
    }

    return ret;
}

/// Computes the base-2 logarithm of an unsigned integer.
///
/// If the given integer is not a power of 2, this function will not provide an exact answer.
///
/// @param [in] u  Value to compute the logarithm of.
///
/// @returns log_2(u)
template <typename T>
uint32 Log2(
    T u)
{
    uint32 logValue = 0;
    return BitMaskScanReverse(&logValue, u) ? logValue : 0;
}

/// Computes the base-2 logarithm of an unsigned 64-bit integer based on ceiling
///
/// If the given integer is not a power of 2, this function will not provide an exact answer.
///
/// @returns ceilLog_2(u)
template <typename T>
uint32 CeilLog2(
    T u)  ///< Value to compute the ceil logarithm of.
{
    const uint32 logValue = Log2(u);
    return ((static_cast<T>(0x1ul) << logValue) < u) ? (logValue + 1) : logValue;
}

/// Implements an alternative version of integer division in which the quotient is always rounded up instead of down.
///
/// @returns The rounded quotient.
template <typename T>
constexpr T RoundUpQuotient(
    T dividend, ///< Value to divide.
    T divisor)  ///< Value to divide by.
{
    return ((dividend + (divisor - 1)) / divisor);
}

/// Rounds up the specified integer to the nearest multiple of the specified alignment value.
///
/// @returns Rounded value.
template <typename T>
constexpr T RoundUpToMultiple(
    T operand,   ///< Value to be aligned.
    T alignment) ///< Alignment desired.
{
    return (((operand + (alignment - 1)) / alignment) * alignment);
}

/// Rounds down the specified integer to the nearest multiple of the specified alignment value.
///
/// @returns Rounded value.
template <typename T>
constexpr T RoundDownToMultiple(
    T operand,    ///< Value to be aligned.
    T alignment)  ///< Alignment desired.
{
    return ((operand / alignment) * alignment);
}

/// Rounds the specified 'value' down to the nearest value meeting the specified 'alignment'.  Only power of 2
/// alignments are supported by this function.
///
/// @returns Rounded value.
template <typename T>
constexpr T Pow2AlignDown(
    T      value,      ///< Value to align.
    uint64 alignment)  ///< Desired alignment (must be a power of 2).
{
    PAL_CONSTEXPR_ASSERT(IsPowerOfTwo(alignment));
    return (value & ~(alignment - 1));
}

/// Determines the maximum of two numbers.
///
/// @returns The larger of the two inputs.
template <typename T>
constexpr T Max(
    T value1,  ///< First value to check.
    T value2)  ///< Second value to check.
{
    return ((value1 > value2) ? value1 : value2);
}

/// Determines the maximum of N numbers.
///
/// @returns The largest of all the inputs.
template <typename T, typename... Ts>
constexpr T Max(
    T     value1,  ///< First value to check.
    T     value2,  ///< Second value to check.
    Ts... values)  ///< Additional values to check.
{
    return Max(((value1 > value2) ? value1 : value2), values...);
}

/// Determines the minimum of two numbers.
///
/// @returns The smaller of the two inputs.
template <typename T>
constexpr T Min(
    T value1,  ///< First value to check.
    T value2)  ///< Second value to check.
{
    return ((value1 < value2) ? value1 : value2);
}

/// Determines the minimum of N numbers.
///
/// @returns The smallest of all the inputs.
template <typename T, typename... Ts>
constexpr T Min(
    T     value1,  ///< First value to check.
    T     value2,  ///< Second value to check.
    Ts... values)  ///< Additional values to check.
{
    return Min(((value1 < value2) ? value1 : value2), values...);
}

/// Clamps the input number so that it falls in-between the lower and upper bounds (inclusive).
///
/// @returns Clamped input number.
template <typename T>
constexpr T Clamp(
    T input,      ///< Input number to clamp.
    T lowBound,   ///< Lower-bound to clamp to.
    T highBound)  ///< Upper-bound to clamp to.
{
    return ((input <= lowBound)  ? lowBound  :
            (input >= highBound) ? highBound : input);
}

/// Determines if the input is within the range specified (inclusive).
///
/// @returns True if within range, False otherwise.
template <typename T>
constexpr bool InRange(
    T input,     ///< Input number to range check.
    T lowBound,  ///< Low bound of the range to check (inclusive).
    T highBound) ///< High bound of the range to check (inclusive).
{
    return (lowBound <= input) && (input <= highBound);
}

/// Converts a byte value to the equivalent number of DWORDs (uint32) rounded up.  I.e., 3 bytes will return 1 dword.
///
/// @returns Number of dwords necessary to cover numBytes.
constexpr uint32 NumBytesToNumDwords(
    uint32 numBytes)  ///< Byte count to convert.
{
    return Pow2Align(numBytes, static_cast<uint32>(sizeof(uint32))) / sizeof(uint32);
}

/// Compare two strings ignoring case
inline int Strcasecmp(
    const char* pSrc,     ///< [in] The source string to be compared.
    const char* pDst)     ///< [in] The dest string to compare.
{
    PAL_ASSERT(pSrc != nullptr);
    PAL_ASSERT(pDst != nullptr);

#if defined(_WIN32)
    return _stricmp(pDst, pSrc);
#else
    return strcasecmp(pDst, pSrc);
#endif
}

/// Performs a safe strcpy by requiring the destination buffer size.
inline void Strncpy(
    char*       pDst,     ///< [out] Destination string.
    const char* pSrc,     ///< [in] Source string to be copied into destination.
    size_t      dstSize)  ///< Size of the destination buffer in bytes.
{
    PAL_ASSERT(pDst != nullptr);
    PAL_ASSERT(pSrc != nullptr);
    PAL_ALERT(strlen(pSrc) >= dstSize);

    if (dstSize > 0)
    {
#if defined(_WIN32)
        // Clamp the copy to the size of the dst buffer (1 char reserved for the null terminator).
        strncpy_s(pDst, dstSize, pSrc, _TRUNCATE);
#else
        strncpy(pDst, pSrc, (dstSize - 1));
        pDst[dstSize - 1] = '\0';
#endif
    }
}

/// Simple wrapper for wcscpy_s or wcsncpy, which are available on Windows and Linux, respectively.
inline void Wcsncpy(
    wchar_t*       pDst,    ///< [out] Destination string.
    const wchar_t* pSrc,    ///< [in] Source string to copy.
    size_t         dstSize) ///< Length of the destination buffer, in wchar_t's.
{
#if defined(_WIN32)
    wcscpy_s(pDst, dstSize, pSrc);
#else
    wcsncpy(pDst, pSrc, (dstSize - 1));
    pDst[dstSize - 1] = L'\0';
#endif
}

// Wrapper for wcscat or wcscat_s which provides a safe version of wcscat
inline void Wcscat(
    wchar_t*       pDst,
    const wchar_t* pSrc,
    size_t         dstSize)
{
#if defined(_WIN32)
    wcsncat_s(pDst, dstSize, pSrc, _TRUNCATE);
#else
    const size_t dstLen = std::wcslen(pDst);
    wcsncat(pDst, pSrc, (dstSize - dstLen - 1));
    pDst[dstSize - 1] = L'\0';
#endif
}

/// Simple wrapper for strncat or strncat_s which provides a safe version of strncat.
inline void Strncat(
    char*       pDst,     ///< [in,out] Destination string.
    size_t      sizeDst,  ///< Length of the destination string, including the null terminator.
    const char* pSrc)     ///< [in] Source string.
{
    PAL_ASSERT((pDst != nullptr) && (pSrc != nullptr));

#if defined(_WIN32)
    // MS compilers provide strncat_s, which will truncate the copy to prevent buffer overruns and always guarantee that
    // pDst is null-terminated.
    strncat_s(pDst, sizeDst, pSrc, _TRUNCATE);
#else
    // Compute the length of the destination string to prevent buffer overruns.
    const size_t dstLength = strlen(pDst);
    strncat(pDst, pSrc, (sizeDst - dstLength - 1));
#endif
}

/// Simple wrapper for strtok_s or strtok_r which provides a safe version of strtok.
inline char* Strtok(
    char*       str,    ///< [in] Token string.
    const char* delim,  ///< [in] Token delimit.
    char**      buf)    ///< [in,out] Buffer to store the rest of the string.
{
    PAL_ASSERT((delim != nullptr) && (buf != nullptr));

    char* pToken = nullptr;

#if defined(_WIN32)
    pToken = strtok_s(str, delim, buf);
#else
    pToken = strtok_r(str, delim, buf);
#endif

    return pToken;
}

/// Rounds the specified pointer up to the nearest value meeting the specified 'alignment'.  Only power of 2 alignments
/// are supported by this function.
///
/// @returns Aligned pointer.
inline void* VoidPtrAlign(
    void*  ptr,        ///< Pointer to align.
    size_t alignment)  ///< Desired alignment.
{
    // This function only works for POW2 alignment
    PAL_ASSERT(IsPowerOfTwo(alignment));

    return reinterpret_cast<void*>(
               (reinterpret_cast<size_t>(ptr) + (alignment - 1)) & ~(alignment - 1));
}

/// Converts a raw string value to the correct data type.
inline void StringToValueType(
    const char* pStrValue,  ///< [in] Setting value in string form.
    ValueType   type,       ///< Data type of the value being converted.
    size_t      valueSize,  ///< Size of pValue buffer.
    void*       pValue)     ///< [out] Converted setting value buffer.
{
    switch (type)
    {
    case ValueType::Boolean:
        *(static_cast<bool*>(pValue)) = ((atoi(pStrValue)) ? true : false);
        break;
    case ValueType::Int8:
        *(static_cast<int8*>(pValue)) = static_cast<int8>(strtoll(pStrValue, nullptr, 0));
        break;
    case ValueType::Uint8:
        *(static_cast<uint8*>(pValue)) = static_cast<uint8>(strtoull(pStrValue, nullptr, 0));
        break;
    case ValueType::Int16:
        *(static_cast<int16*>(pValue)) = static_cast<int16>(strtoll(pStrValue, nullptr, 0));
        break;
    case ValueType::Uint16:
        *(static_cast<uint16*>(pValue)) = static_cast<uint16>(strtoull(pStrValue, nullptr, 0));
        break;
    case ValueType::Int32:
        *(static_cast<int32*>(pValue)) = static_cast<int32>(strtoll(pStrValue, nullptr, 0));
        break;
    case ValueType::Uint32:
        *(static_cast<uint32*>(pValue)) = static_cast<uint32>(strtoull(pStrValue, nullptr, 0));
        break;
    case ValueType::Int64:
        *(static_cast<int64*>(pValue)) = static_cast<int64>(strtoll(pStrValue, nullptr, 0));
        break;
    case ValueType::Uint64:
        *(static_cast<uint64*>(pValue)) = static_cast<uint64>(strtoull(pStrValue, nullptr, 0));
        break;
    case ValueType::Float:
        *(static_cast<float*>(pValue)) = static_cast<float>(atof(pStrValue));
        break;
    case ValueType::Str:
        Strncpy(static_cast<char*>(pValue), pStrValue, valueSize);
        break;
    }
}

/// Converts a raw string value to the correct data type, returning 'true' if parsed correctly.
/// When not parsed correctly, the value will be unchanged.
///
/// @note: A string that is truncated returns false.
/// @note: If the destination type is integer, the string is parsed as either int64 or uint64, and the parsed value is
/// clamped to fit the range of the destination type.
[[nodiscard]] inline bool StringToValueTypeChecked(
    const char* pStrValue,  ///< [in] Setting value in string form.
    ValueType   type,       ///< Data type of the value being converted.
    size_t      valueSize,  ///< Size of pValue buffer.
    void*       pValue)     ///< [out] Converted setting value buffer.
{
    auto CheckTrailingCharacters = [](char* pChar, const char* pEnd) -> bool {
        while ((pChar < pEnd) && isspace(*pChar))
        {
            // ignore trailing whitespace. strtoX handles leading whitespace
            pChar++;
        }
        return (pChar == pEnd);
    };

    const size_t len = strlen(pStrValue);
    const char* pTerminator = pStrValue + len;
    char* pEndptr = nullptr;
    bool valid = false;

    switch (type)
    {
    case ValueType::Boolean:
        {
            bool value = (strtol(pStrValue, &pEndptr, 0) != 0);
            valid = CheckTrailingCharacters(pEndptr, pTerminator);
            if (valid)
            {
                *(static_cast<bool*>(pValue)) = value;
            }
        }
        break;
    case ValueType::Int8:
        {
            const int64 parsedValue = strtoll(pStrValue, &pEndptr, 0);
            const int64 value = Clamp(parsedValue,
                                      int64((std::numeric_limits<int8>::min)()),
                                      int64((std::numeric_limits<int8>::max)()));
            valid = CheckTrailingCharacters(pEndptr, pTerminator);
            if (valid)
            {
                *(static_cast<int8*>(pValue)) = static_cast<int8>(value);
            }
        }
        break;
    case ValueType::Uint8:
        {
            const uint64 parsedValue = strtoull(pStrValue, &pEndptr, 0);
            const uint64 value = Clamp(parsedValue,
                                       uint64((std::numeric_limits<uint8>::min)()),
                                       uint64((std::numeric_limits<uint8>::max)()));
            valid = CheckTrailingCharacters(pEndptr, pTerminator);
            if (valid)
            {
                *(static_cast<uint8*>(pValue)) = static_cast<uint8>(value);
            }
        }
        break;
    case ValueType::Int16:
        {
            const int64 parsedValue = strtoll(pStrValue, &pEndptr, 0);
            const int64 value = Clamp(parsedValue,
                                      int64((std::numeric_limits<int16>::min)()),
                                      int64((std::numeric_limits<int16>::max)()));
            valid = CheckTrailingCharacters(pEndptr, pTerminator);
            if (valid)
            {
                *(static_cast<int16*>(pValue)) = static_cast<int16>(value);
            }
        }
        break;
    case ValueType::Uint16:
        {
            const uint64 parsedValue = strtoull(pStrValue, &pEndptr, 0);
            const uint64 value = Clamp(parsedValue,
                                       uint64((std::numeric_limits<uint16>::min)()),
                                       uint64((std::numeric_limits<uint16>::max)()));
            valid = CheckTrailingCharacters(pEndptr, pTerminator);
            if (valid)
            {
                *(static_cast<uint16*>(pValue)) = static_cast<uint16>(value);
            }
        }
        break;
    case ValueType::Int32:
        {
            const int64 parsedValue = strtoll(pStrValue, &pEndptr, 0);
            const int64 value = Clamp(parsedValue,
                                      int64((std::numeric_limits<int32>::min)()),
                                      int64((std::numeric_limits<int32>::max)()));
            valid = CheckTrailingCharacters(pEndptr, pTerminator);
            if (valid)
            {
                *(static_cast<int32*>(pValue)) = static_cast<int32>(value);
            }
        }
        break;
    case ValueType::Uint32:
        {
            const uint64 parsedValue = strtoull(pStrValue, &pEndptr, 0);
            const uint64 value = Clamp(parsedValue,
                                       uint64((std::numeric_limits<uint32>::min)()),
                                       uint64((std::numeric_limits<uint32>::max)()));
            valid = CheckTrailingCharacters(pEndptr, pTerminator);
            if (valid)
            {
                *(static_cast<uint32*>(pValue)) = static_cast<uint32>(value);
            }
        }
        break;
    case ValueType::Int64:
        {
            const int64 value = strtoll(pStrValue, &pEndptr, 0);
            valid = CheckTrailingCharacters(pEndptr, pTerminator);
            if (valid)
            {
                *(static_cast<int64*>(pValue)) = value;
            }
        }
        break;
    case ValueType::Uint64:
        {
            const uint64 value = strtoull(pStrValue, &pEndptr, 0);
            valid = CheckTrailingCharacters(pEndptr, pTerminator);
            if (valid)
            {
                *(static_cast<uint64*>(pValue)) = value;
            }
        }
        break;
    case ValueType::Float:
        {
            float value = static_cast<float>(strtof(pStrValue, &pEndptr));
            valid = CheckTrailingCharacters(pEndptr, pTerminator);
            if (valid)
            {
                *(static_cast<float*>(pValue)) = value;
            }
        }
        break;
    case ValueType::Str:
        if (len + 1 <= valueSize)
        {
            valid = true;
            Strncpy(static_cast<char*>(pValue), pStrValue, valueSize);
        }
        break;
    }
    return valid;
}

/// Hashes the provided string using FNV1a hashing (http://www.isthe.com/chongo/tech/comp/fnv/) algorithm.
///
/// @returns 32-bit hash generated from the provided string.
template <class Char>
constexpr uint32 HashString(
    const Char* pStr,     ///< [in] String to be hashed.
    size_t      strSize)  ///< Size of the input string.
{
    PAL_CONSTEXPR_ASSERT((pStr != nullptr) && (strSize > 0));

    constexpr uint32 FnvPrime  = 16777619u;
    constexpr uint32 FnvOffset = 2166136261u;

    uint32 hash = FnvOffset;

    for (size_t i = 0; i < strSize; i++)
    {
        Char c = pStr[i];
        for (uint32 j = 0; j < sizeof(Char); ++j)
        {
            hash ^= uint8(c);
            hash *= FnvPrime;
            c = (c >> 8);
        }
    }

    return hash;
}

/// Hashes the provided string using FNV1a hashing (http://www.isthe.com/chongo/tech/comp/fnv/) algorithm.
///
/// @returns 32-bit hash generated from the provided string.
template <class Char>
constexpr uint32 HashString(
    const Char* pString)
{
    return HashString(pString, StringLength(pString));
}

/// Hashes the provided string using FNV1a hashing (http://www.isthe.com/chongo/tech/comp/fnv/) algorithm.
/// Same as HashString() except consteval enforces that this can only be called at compile-time.
///
/// @returns 32-bit hash generated from the provided string.
template <class Char>
#if defined(__cpp_consteval)
consteval
#else
constexpr
#endif
uint32 CompileTimeHashString(
    const Char* pString)
{
    return HashString(pString);
}

/// Indicates that an object may be moved from.
/// Can be understood as preparation for possible move operation.
///
/// @warning Do not read object after it has been moved from!
///
/// @param [in] object Universal reference to an object that may be moved from.
///
/// @returns Rvalue reference to the parameter object.
template <typename T>
constexpr typename std::remove_reference<T>::type&& Move(T&& object)
{
    // Cast universal reference to rvalue reference.
    return static_cast<typename std::remove_reference<T>::type&&>(object);
}

/// Exchanges values between two variables.
///
/// @param [in] left  First variable used in swap operation.
/// @param [in] right Second variable used in swap operation.
template <typename T>
constexpr void Swap(T& left, T& right)
{
    T tmp = Move(left);
    left  = Move(right);
    right = Move(tmp);
}

/// Convenient alias for C style arrays.
template <typename Element, size_t Size>
using Array = Element[Size];

/// Prevent swapping arrays because of the cost of this operation.
template <typename Element, size_t Size>
void Swap(Array<Element, Size>& a, Array<Element, Size>& b);

/// Compacts an array by moving all empty slots to the end of the array.
///         +---+---+---+---+---+---+---+---+---+---+
///  Input: | A |   | C | D |   | E |   | A | X | J |
///         +---+---+---+---+---+---+---+---+---+---+
///         +---+---+---+---+---+---+---+---+---+---+
/// Output: | A | C | D | E | A | X | J |   |   |   |
///         +---+---+---+---+---+---+---+---+---+---+
template <typename Element, size_t Size>
void PackArray(Array<Element, Size>& array, const Element& emptySlot)
{
    int lastOccupiedSlot = -1;

    for (size_t i = 0; i < Size; ++i)
    {
        if (array[i] != emptySlot)
        {
            Swap(array[i], array[lastOccupiedSlot + 1]);
            ++lastOccupiedSlot;
        }
    }
}

/// Performs a safe mbstowcs by requiring the destination buffer size.
inline void Mbstowcs(
    wchar_t*      pDst,           ///< [out] dst string
    const char*   pSrc,           ///< [in] src string
    size_t        dstSizeInWords) ///< size of the destination buffer in words
{
    PAL_ASSERT(pDst != nullptr);
    PAL_ASSERT(pSrc != nullptr);

    bool result = false;
    // clamp the conversion to the size of the dst buffer (1 char reserved for the NULL terminator)
#if defined(_WIN32)
    size_t bytesConverted = 0;
    errno_t retCode = mbstowcs_s(&bytesConverted, pDst, dstSizeInWords, pSrc, (dstSizeInWords - 1));

    result = (retCode != 0) ? false : true;
#else
    size_t retCode = mbstowcs(pDst, pSrc, dstSizeInWords);

    result = (retCode == static_cast<size_t>(-1)) ? false : true;

    if (retCode == dstSizeInWords)
    {
        // Alert the user when the string has been truncated.
        PAL_ALERT_ALWAYS();

        // NULL terminate the string.
        pDst[dstSizeInWords - 1] = '\0';
    }
#endif

    if (result == false)
    {
        // A non-convertible character was encountered or the string was truncated on the mbstowcs_s or
        // ConvertCharStringToUtf16 code paths.
        PAL_ALERT_ALWAYS();
        pDst[0] = '\0';
    }
}

/// Performs a safe wcstombs by requiring the destination buffer size.
inline void Wcstombs(
    char*          pDst,           ///< [out] dst string
    const wchar_t* pSrc,           ///< [in] src string
    size_t         dstSizeInBytes) ///< size of the destination buffer in bytes
{
    PAL_ASSERT(pDst != nullptr);
    PAL_ASSERT(pSrc != nullptr);

    bool result = false;
    // clamp the conversion to the size of the dst buffer (1 char reserved for the NULL terminator)
#if defined(_WIN32)
    size_t bytesConverted = 0;
    errno_t retCode = wcstombs_s(&bytesConverted, pDst, dstSizeInBytes, pSrc, (dstSizeInBytes - 1));

    result = (retCode != 0) ? false : true;
#else
    size_t retCode = wcstombs(pDst, pSrc, (dstSizeInBytes - 1));

    result = (retCode == static_cast<size_t>(-1)) ? false : true;
#endif

    if (result == false)
    {
        // A non-convertible character was encountered.
        PAL_ASSERT_ALWAYS();
        pDst[0] = '\0';
    }

    if (wcslen(pSrc) >= dstSizeInBytes)
    {
        // Assert to alert the user when the string has been truncated.
        PAL_ASSERT_ALWAYS();

        // NULL terminate the string.
        pDst[dstSizeInBytes - 1] = '\0';
    }
}

/// Computes the Greatest Common Divisor of two numbers
///
/// @returns The GCD of the two inputs.
template<typename T1, typename T2>
inline typename std::common_type<T1, T2>::type Gcd(
    T1 value1,
    T2 value2)
{
    static_assert((std::is_integral<T1>::value == true) &&
                  (std::is_integral<T2>::value == true),
                  "GCD requires integral types");

    static_assert((std::is_unsigned<T1>::value == true) &&
                  (std::is_unsigned<T2>::value == true),
                  "GCD requires unsigned types");

    static_assert((std::is_same<T1, bool>::value == false) &&
                  (std::is_same<T2, bool>::value == false),
                  "GCD requires nonboolean types");

    using T = typename std::common_type<T1, T2>::type;
    T ret = 0u;

    if (value1 == 0u)
    {
        ret = static_cast<T>(value2);
    }
    else if (value2 == 0u)
    {
        ret = static_cast<T>(value1);
    }
    else
    {
        uint32 value1TrailingZeros = 0u;
        BitMaskScanForward(&value1TrailingZeros, value1);
        uint32 value2TrailingZeros = 0u;
        BitMaskScanForward(&value2TrailingZeros, value2);

        const uint32 shift = Min(value1TrailingZeros, value2TrailingZeros);
        value1 >>= value1TrailingZeros;
        value2 >>= shift;

        do
        {
            BitMaskScanForward(&value2TrailingZeros, value2);
            value2 >>= value2TrailingZeros;

            if (value1 > value2)
            {
                T tmp = value1;
                value1 = value2;
                value2 = tmp;
            }

            value2 -= value1;
        }
        while (value2 != 0);

        ret = static_cast<T>(value1 << shift);
    }

    return ret;
}

/// Computes the Greatest Common Divisor of N numbers
///
/// @returns The GCD of the all inputs.
template<typename T1,
         typename T2,
         typename... Ts>
inline typename std::common_type<T1, T2, typename std::common_type<Ts...>::type>::type Gcd(
    T1 value1,
    T2 value2,
    Ts... values)
{
    return Gcd(Gcd(value1, value2), values...);
}

/// Computes the Least Common Multiple of two numbers
///
/// @returns The LCM of the two inputs.
template<typename T1, typename T2>
constexpr typename std::common_type<T1, T2>::type Lcm(
    T1 value1,
    T2 value2)
{
    static_assert((std::is_integral<T1>::value == true) &&
                  (std::is_integral<T2>::value == true),
                  "LCM requires integral types");

    static_assert((std::is_unsigned<T1>::value == true) &&
                  (std::is_unsigned<T2>::value == true),
                  "LCM requires unsigned types");

    static_assert((std::is_same<T1, bool>::value == false) &&
                  (std::is_same<T2, bool>::value == false),
                  "LCM requires nonboolean types");

    using T = typename std::common_type<T1, T2>::type;

    return (value1 != 0u) && (value2 != 0u) ? static_cast<T>((value1 / Gcd(value1, value2)) * value2) : 0u;
}

/// Computes the Least Common Multiple of N numbers
///
/// @returns The LCM of all the inputs.
template<typename T1,
         typename T2,
         typename... Ts>
constexpr typename std::common_type<T1, T2, typename std::common_type<Ts...>::type>::type Lcm(
    T1 value1,
    T2 value2,
    Ts... values)
{
    return Lcm(Lcm(value1, value2), values...);
}

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 919
/// Returns the length of a wchar_t based string.
///
/// @returns The length of the given string in wide characters
inline size_t Wcslen(
    const wchar_t* pWideStr)
{
    return wcslen(pWideStr);
}

/// Performs a reverse string find of wide character wc.
///
/// @returns The matching character at the end of the string or nullptr if not found.
inline wchar_t* Wcsrchr(wchar_t *pStr, wchar_t wc)
{
    return wcsrchr(pStr, wc);
}
#endif

/// Compile-time function to report if two values from unrelated strong enums are equivalent.  This is useful for
/// static asserts ensuring it is safe to cast an enum without a conversion lookup table.
template <typename T1, typename T2>
inline constexpr bool EnumSameVal(
    T1 lhs,
    T2 rhs)
{
    return (static_cast<uint64>(lhs) == static_cast<uint64>(rhs));
}

/// Comparison function for Sort() below.
template<typename ElementTy> int PAL_CDECL SortComparisonFunc(
    const void* pLhs,
    const void* pRhs)
{
    return int(*static_cast<const ElementTy*>(pRhs) < *static_cast<const ElementTy*>(pLhs)) -
           int(*static_cast<const ElementTy*>(pLhs) < *static_cast<const ElementTy*>(pRhs));
}

/// In-place sort of an array. Uses C library qsort, so is probably a non-order-preserving quicksort.
/// Sorts the array given by the random iterator range [pStart,pEnd).
/// The element type (the type you get by dereferencing RandomIt) must have an operator<.
template<typename RandomIt> void Sort(
    RandomIt pStart,
    RandomIt pEnd)
{
    using ElementTy = typename std::iterator_traits<RandomIt>::value_type;
    qsort(&pStart[0], pEnd - pStart, sizeof(ElementTy), SortComparisonFunc<ElementTy>);
}

} // Util
