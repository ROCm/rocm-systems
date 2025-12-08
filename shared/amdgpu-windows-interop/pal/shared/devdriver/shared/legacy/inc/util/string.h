/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#include <string.h>
#include <util/hashFunc.h>

namespace DevDriver
{
// A String class that stores the string inline with a compile-time maximum size.
// This class facilitiates passing bounded sized C Strings around without dynamic allocation. It has POD semantics
// when copied or passed by value into functions, and can be stored in a vector.
template<size_t FixedSize>
class FixedString
{
private:
    char m_data[FixedSize];

public:
    FixedString()
    {
        // Only the first byte needs to be initialized - we actively do not want to zero the entire array!
        m_data[0] = 0;
    }

    FixedString(const FixedString<FixedSize>&) = default;
    FixedString(FixedString<FixedSize>&&)      = default;

    ~FixedString() {}

    FixedString<FixedSize>& operator=(FixedString<FixedSize>& pOther)
    {
        Platform::Strncpy(m_data, pOther.m_data, sizeof(m_data));
        return *this;
    }

    FixedString<FixedSize>& operator=(FixedString<FixedSize>&& pOther)
    {
        Platform::Strncpy(m_data, pOther.m_data, sizeof(m_data));
        return *this;
    }

    bool operator==(const FixedString<FixedSize>& other) const
    {
        return strncmp(this->AsCStr(), other.AsCStr(), FixedSize) == 0;
    }

    bool operator!=(const FixedString<FixedSize>& other) const
    {
        return strncmp(this->AsCStr(), other.AsCStr(), FixedSize) != 0;
    }

    // Create a FixedString from a C String, truncating the copy if pString is too long
    FixedString(const char* pString) { Platform::Strncpy(m_data, pString, sizeof(m_data)); }

    // Return a pointer to the inline C String.
    const char* AsCStr() const { return m_data; }

    char* AsCStr() { return m_data; }

    // Computes the length of the string.
    // Note! This is an O(N) operation!
    size_t Size() const { return strlen(m_data); }
};

// Sanity check for class size.
static_assert(sizeof(FixedString<16>) == 16, "FixedString<16> should be exactly 16 bytes");

/// ====================================================================================================================
// Hashes a FixedString<> Key using Metrohash
template<size_t Size>
struct DefaultHashFunc<FixedString<Size>>
{
    uint32 operator()(const FixedString<Size>& key) const { return DefaultHashFunc<const char*>()(key.AsCStr()); }
};

/// ====================================================================================================================
/// Utility functions for strings
/// ====================================================================================================================

enum struct HexStringFmt
{
    Lowercase,
    Uppercase,
};

/// ====================================================================================================================
// Encode not more than `numBytes` from `pBytes` into hexadecimal, storing not more than stringBufferSize characters
// into pStringBuffer.
//
// This is the compliment of DecodeFromHexString() and is suitable for saving large binary blocks in text formats such
// as Json.
//
// This function NULL terminates its output if it writes anything.
// Hex pairs are written to `pStrBuff` in pairs - either both digits are written or neither is. A lone nibble
//      is never written to the buffer.
// Thus, Hex strings are always an even length (+ a NULL byte)
//
// Returns the number of characters written out through `pStrBuff` (including the NULL terminator).
template <HexStringFmt fmt = HexStringFmt::Lowercase>
inline size_t EncodeToHexString(const void* pBytesIn, size_t numBytes, char* pStrBuff, size_t strBuffSize)
{
    const uint8* pBytes = static_cast<const uint8*>(pBytesIn);

    // Character offset that we've written into pStrBuff
    size_t charsProcessed = 0;

    if ((pBytes != nullptr) && (numBytes != 0) && (pStrBuff != nullptr) && (strBuffSize != 0))
    {

        // Both lookups are indexed by nibble
        constexpr const char kHexStringLookupLower[] = "0123456789abcdef";
        constexpr const char kHexStringLookupUpper[] = "0123456789ABCDEF";

        // This is the index where our next character pair goes.
        // We save this outside of the loop to NULL terminate correctly.
        size_t strIdx = 0;
        for (size_t byteIdx = 0; byteIdx < numBytes; byteIdx += 1)
        {
            // We're going to write two bytes this loop, but need to exit early if we're out of bounds.
            // We need room for:
            //      - the high nibble
            //      - the low nibble
            //      - the NULL terminator
            // Offsets (from stdIdx) of 0, 1, and 2 must be within the buffer bounds.
            if ((strIdx + 2) < strBuffSize)
            {
                const uint8 byte = pBytes[byteIdx];

                if (fmt == HexStringFmt::Lowercase)
                {
                    pStrBuff[strIdx + 0] = kHexStringLookupLower[byte >>  4]; // High nibble first
                    pStrBuff[strIdx + 1] = kHexStringLookupLower[byte & 0xf]; // Low nibble
                }
                else
                {
                    pStrBuff[strIdx + 0] = kHexStringLookupUpper[byte >>  4]; // High nibble first
                    pStrBuff[strIdx + 1] = kHexStringLookupUpper[byte & 0xf]; // Low nibble
                }

                strIdx += 2;
            }
            else
            {
                break;
            }
        }

        pStrBuff[strIdx] = '\0';
        charsProcessed += strIdx + 1; // Hex characters (if any) + NULL
    }

    return charsProcessed;
}

/// ====================================================================================================================
// Helper function that translates hex digits into numeric values.
// Returns 0xff if the value is not a hex digit
#if DD_CPLUSPLUS_SUPPORTS(CPP17)
constexpr uint8 HexDigitToValue(char c)
#else
inline uint8 HexDigitToValue(char c)
#endif
{
    // We use a switch case here to get the point across
    // gcc9, clang8, and MSVC all turn this into a lookup table indexing with c (sometimes subtracting from it first)

    switch (c)
    {
        // clang-format off
        case '0':
        case '1': case '2': case '3':
        case '4': case '5': case '6':
        case '7': case '8': case '9':
                                    return c - '0';

        case 'a': case 'A':         return 0xa;
        case 'b': case 'B':         return 0xb;
        case 'c': case 'C':         return 0xc;
        case 'd': case 'D':         return 0xd;
        case 'e': case 'E':         return 0xe;
        case 'f': case 'F':         return 0xf;

        default:
                                    return 0xff;
        // clang-format on
    }
}

#if DD_CPLUSPLUS_SUPPORTS(CPP17)
    static_assert(HexDigitToValue('Z') == 0xff);

    static_assert(HexDigitToValue('0') == 0);
    static_assert(HexDigitToValue('1') == 1);
    static_assert(HexDigitToValue('2') == 2);
    static_assert(HexDigitToValue('3') == 3);
    static_assert(HexDigitToValue('4') == 4);
    static_assert(HexDigitToValue('5') == 5);
    static_assert(HexDigitToValue('6') == 6);
    static_assert(HexDigitToValue('7') == 7);
    static_assert(HexDigitToValue('8') == 8);
    static_assert(HexDigitToValue('9') == 9);

    static_assert(HexDigitToValue('a') == 10);
    static_assert(HexDigitToValue('b') == 11);
    static_assert(HexDigitToValue('c') == 12);
    static_assert(HexDigitToValue('d') == 13);
    static_assert(HexDigitToValue('e') == 14);
    static_assert(HexDigitToValue('f') == 15);

    static_assert(HexDigitToValue('A') == 10);
    static_assert(HexDigitToValue('B') == 11);
    static_assert(HexDigitToValue('C') == 12);
    static_assert(HexDigitToValue('D') == 13);
    static_assert(HexDigitToValue('E') == 14);
    static_assert(HexDigitToValue('F') == 15);
#endif

/// ====================================================================================================================
// Decode not more than `strLength` hex characters from `pStrBuff` into their binary representation, storing
// not more than `numBytes` into `pBytesOut`.
//
// This is the compliment of EncodeToHexString() and is suitable for decoding large binary blocks out of text formats
// such as Json.
//
// Returns the number of bytes written out through `pBytesOut`.
inline size_t DecodeFromHexString(const char* pStrBuff, size_t strLength, void* pBytesOut, size_t numBytes)
{
    uint8* pBytes = static_cast<uint8*>(pBytesOut);

    // Byte offset that we've written into pBytes
    size_t bytesProcessed = 0;

    // Note: Only even-length hex strings are supported
    if ((strLength % 2 == 0) && (pBytes != nullptr) && (numBytes != 0) && (pStrBuff != nullptr) && (strLength != 0))
    {
        size_t byteIdx = 0;

        // Process two characters (one byte) per iteration.
        // This loop is bounded on two sizes: the string buffer and the byte buffer
        for (size_t strIdx = 0;
            ((strIdx + 1) < strLength) && (byteIdx < numBytes);
            strIdx += 2, byteIdx += 1)
        {
            const uint8 hi = HexDigitToValue(pStrBuff[strIdx + 0]); // High nibble first
            const uint8 lo = HexDigitToValue(pStrBuff[strIdx + 1]); // Low nibble

            if ((lo != 0xff) && (hi != 0xff))
            {
                pBytes[byteIdx] = (hi << 4) | lo;
                bytesProcessed += 1;
            }
            else
            {
                // Non-hex digit encountered, this is a parsing error.
                // This log statement is compiled out, but may be useful for debugging something funny.
                DD_PRINT(LogLevel::Never,
                    "[DecodeFromHexString] Expected hex digits ([0-9a-fA-F]), but found \"%c%c\"",
                    pStrBuff[strIdx + 0],
                    pStrBuff[strIdx + 1]);
                break;
            }
        }
    }

    return bytesProcessed;
}

} // namespace DevDriver
