/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
 * @file  palStringUtil.h
 * @brief PAL String utility collection functions.
 ***********************************************************************************************************************
 */

#pragma once

#include <cstring>
#include <cwchar>
#include <type_traits>

#include "palUtil.h"

namespace Util
{
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 919
/// Returns the length of a wchar_t based string.
///
/// @param [in]  wide string to query
///
/// @returns The length of the given string in wide characters
extern size_t PalWcslen(
    const wchar_t* pWideStr);

/// Performs a reverse string find of wide character wc.
///
/// @param [in]  wide string to scan
/// @param [in]  wide character to find
///
/// @returns The matching character at the end of the string or nullptr if not found.
extern wchar_t* PalWcsrchr(
    wchar_t *pStr,
    wchar_t wc);
#endif

/// When the -fshort-char compiler option is specified, wchar_t is 16 bits, but mbstowcs still treats the dest
/// as 32 bit so we provide our own implementation.
///
/// @param [out]  dst string
/// @param [in]   src string
/// @param [in]   size of the destination buffer in words
///
/// @returns Returns whether or not the conversion was successful.
extern bool ConvertCharStringToUtf16(
    wchar_t*      pDst,
    const char*   pSrc,
    size_t        dstSizeInWords);

/// When the -fshort-char compiler option is specified, wchar_t is 16 bits, but wcstombs still treats the src
/// as 32 bit so we provide our own implementation.
///
/// @param [out]  dst string
/// @param [in]   src string
/// @param [in]   size of the destination buffer in bytes
///
/// @returns Returns whether or not the conversion was successful.
extern bool ConvertUtf16StringToUtf8(
    char*          pDst,
    const wchar_t* pSrc,
    size_t         dstSizeInBytes);

/// Convert wchar_t string to UTF-8 string. Works whether wchar_t is 16 or 32 bits.
/// If wchar_t is 16 bits, this decodes UTF-16.
///
/// @param [out]  dst string
/// @param [in]   src string
/// @param [in]   size of the destination buffer in bytes
///
/// @returns Returns whether or not the conversion was successful.
bool ConvertWcharStringToUtf8(char* pDst, const wchar_t* pSrc, size_t dstSizeInBytes);

/// When the -fshort-char compiler option is specified, wchar_t is 16 bits, but wcsncpy still treats its arguments
/// as 32 bit so we provide our own implementation.
///
/// @param [out] pDst     Destination string.
/// @param [in]  pSrc     Source string to copy.
/// @param [in]  dstSize  Length of the destination buffer, in wchar_t's.
extern void CopyUtf16String(
    wchar_t*       pDst,
    const wchar_t* pSrc,
    size_t         dstSize);

/// A shared helper function which takes an arbitrary blob of data and formats it into a human readable "memory view"
/// string. This is intended to be used by logging code.
///
/// Imagine your input buffer is: { 0xef, 0xbe, 0xad, 0xde, 0x78, 0x56, 0x34, 0x12, 0xab }, then the string looks like
/// this with a blockSize of 4: "0xdeadbeef 0x12345678 0xab". So the block size determines how many bytes are combined
/// into one "0x" character block. The whole block is effetively cast into an integer of that size and printed in big
/// endian. Trailing bytes are printed without being size-extended. If a block won't fit at the end of the string it
/// is skipped (update your buffer pointer and call again to continue).
///
/// The return value is the number of bytes consumed from pBuffer. The idea is that you can loop until the full size
/// is consumed, printing a new line for each call.
///
/// @param [out] pDst       The caller-provided destination string.
/// @param [in]  dstSize    The length of pDst in bytes.
/// @param [in]  pBuffer    The arbitrary data blob to turn into a string.
/// @param [in]  bufferSize The length of pBuffer in bytes.
/// @param [in]  blockSize  How many bytes to combine into one hexidecimal big endian string.
///
/// @returns The number of bytes from pBuffer that were formatted into pDst.
extern size_t BytesToStr(
    char*       pDst,
    size_t      dstSize,
    const void* pBuffer,
    size_t      bufferSize,
    size_t      blockSize);

/// Returns the length of the string.
///
/// @returns String length.
constexpr uint32 StringLength(
    const char* pString)
{
    // TODO: On C++23 we can replace this with consteval-if.
    // TODO: When we upgrade PAL_CPLUSPLUS then we can rely that std::is_constant_evaluated() is always defined.
#if defined(__cpp_lib_is_constant_evaluated)
    if (std::is_constant_evaluated())
#else
    if (__builtin_is_constant_evaluated())
#endif
    {
        uint32 length = 0;
        while (pString[length] != '\0')
        {
            length++;
        }
        return length;
    }
    else
    {
        return uint32(std::strlen(pString));
    }
}
/// Returns the length of the string.
///
/// @returns String length.
constexpr uint32 StringLength(
    const wchar_t* pString)
{
    // TODO: On C++23 we can replace this with consteval-if.
    // TODO: When we upgrade PAL_CPLUSPLUS then we can rely that std::is_constant_evaluated() is always defined.
#if defined(__cpp_lib_is_constant_evaluated)
    if (std::is_constant_evaluated())
#else
    if (__builtin_is_constant_evaluated())
#endif
    {
        uint32 length = 0;
        while (pString[length] != L'\0')
        {
            length++;
        }
        return length;
    }
    else
    {
        return uint32(std::wcslen(pString));
    }
}

} // Util
