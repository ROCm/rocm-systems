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

#include <ddPlatform.h>
#include <metrohash.h>

namespace DevDriver
{

namespace MetroHash
{
/// 128-bit hash structure
struct Hash
{
    union
    {
        uint32 dwords[4]; ///< Output hash in dwords.
        uint8  bytes[16]; ///< Output hash in bytes.
    };
};

// Compacts a 128-bit hash into a 64-bit one by XOR'ing the low and high 64-bits together.
inline uint64 Compact64(
    const Hash* pHash)
{
    return (static_cast<uint64>(pHash->dwords[3] ^ pHash->dwords[1]) |
           (static_cast<uint64>(pHash->dwords[2] ^ pHash->dwords[0]) << 32));
}

// Compacts a 64-bit hash checksum into a 32-bit one by XOR'ing each 32-bit chunk together.
inline uint32 Compact32(
    const Hash* pHash)
{
    return pHash->dwords[3] ^ pHash->dwords[2] ^ pHash->dwords[1] ^ pHash->dwords[0];
}

// Compacts a 64-bit hash checksum into a 32-bit one by XOR'ing each 32-bit chunk together.
inline uint32 Compact32(
    const uint64 hash)
{
    return static_cast<uint32>(hash) ^ static_cast<uint32>(hash >> 32);
}

inline uint64 MetroHash64(const uint8* pData, const uint64 dataSize)
{
    uint64 hash = 0;
    Util::MetroHash64::Hash(pData, dataSize, reinterpret_cast<uint8*>(&hash));
    return hash;
}

inline uint32 MetroHash32(const uint8* pData, const uint64 dataSize)
{
    return Compact32(MetroHash64(pData, dataSize));
}

inline uint64 HashCStr64(const char* pString)
{
    return MetroHash64(reinterpret_cast<const uint8*>(pString), strlen(pString));
}

} // MetroHash
} // DevDriver
