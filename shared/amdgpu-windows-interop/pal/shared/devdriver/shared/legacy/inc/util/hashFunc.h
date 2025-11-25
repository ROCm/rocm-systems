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
#include <util/ddMetroHash.h>

namespace DevDriver
{

/// ====================================================================================================================
/// Hashes the bytes of a Key using MetroHash
template<typename Key>
struct DefaultHashFunc
{
    uint32 operator()(const Key& key) const
    {
        return MetroHash::MetroHash32(reinterpret_cast<const uint8*>(&key), sizeof(Key));
    }
};

/// ====================================================================================================================
/// Hashes a const char* CString Key using Metrohash
template<>
struct DefaultHashFunc<const char*>
{
    uint32 operator()(const char* pKey) const
    {
        // We cannot pass NULL strings to strlen() and friends, so guard against it anyway.
        uint32 hash = 0;
        DD_ASSERT(pKey != nullptr);
        if (pKey != nullptr)
        {
            hash = MetroHash::MetroHash32(reinterpret_cast<const uint8*>(pKey), strlen(pKey));
        }

        return hash;
    }
};

/// Pointer keys are usually a mistake, so this version is explicitly 'delete'd
/// Overload this template if you're sure you need this. (See: const char* above)
template<typename T>
struct DefaultHashFunc<T*>
{
    uint32 operator()(const T* pKey) const = delete;
};

/// Generic compare functor for types that have defined the comparison operator
///
/// Used by @ref HashBase to prevent defining compare functions for each type.
template<typename Key>
struct DefaultEqualFunc
{
    bool operator()(const Key& key1, const Key& key2) const { return (key1 == key2); }
};

/// String compare functor for use with C-style strings
template<>
struct DefaultEqualFunc<const char*>
{
    bool operator()(const char* pKey1, const char* pKey2) const
    {
        DD_ASSERT(pKey1 != nullptr);
        DD_ASSERT(pKey2 != nullptr);
        return (strcmp(pKey1, pKey2) == 0);
    }
};

/// Generic compare functor for types with arbitrary size
///
/// Used by @ref HashBase to prevent defining compare functions for each type.
template<typename Key>
struct BitwiseEqualFunc
{
    bool operator()(const Key& key1, const Key& key2) const { return (memcmp(&key1, &key2, sizeof(Key)) == 0); }
};

} // namespace DevDriver
