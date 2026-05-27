/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef HOST_BUFFER_HELPERS_HPP
#define HOST_BUFFER_HELPERS_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

/**
 * @file HostBufferHelpers.hpp
 * @brief Template-based host buffer utilities for RCCL tests
 *
 * Provides type-safe, reusable functions for host buffer operations:
 * - Initialization with test patterns
 * - Data verification with error reporting
 *
 * NOTE: All functions expect HOST memory pointers allocated with malloc/new.
 *       For device memory operations, use DeviceBufferHelpers.hpp instead.
 */

namespace RCCLTestHelpers
{

// ============================================================================
// Host Buffer Initialization
// ============================================================================

/**
 * @brief Fill host buffer with pattern function
 *
 * @tparam T Element type (uint8_t, float, etc.)
 * @tparam PatternFunc Callable: T pattern_func(size_t index)
 * @param buffer Host memory pointer
 * @param num_elements Number of elements
 * @param pattern_func Function that generates value for each index
 */
template<typename T, typename PatternFunc>
void fillHostBufferWithPattern(void* buffer, size_t num_elements, PatternFunc pattern_func)
{
    if(!buffer || num_elements == 0)
    {
        return;
    }

    T* typed = static_cast<T*>(buffer);
    for(size_t i = 0; i < num_elements; i++)
    {
        typed[i] = pattern_func(i);
    }
}

/**
 * @brief Zero-initialize host buffer
 *
 * @param buffer Host memory pointer
 * @param num_bytes Number of bytes to zero
 */
inline void zeroHostBuffer(void* buffer, size_t num_bytes)
{
    if(buffer && num_bytes > 0)
    {
        memset(buffer, 0, num_bytes);
    }
}

// ============================================================================
// Host Buffer Verification
// ============================================================================

/**
 * @brief Verify host buffer data against a pattern function
 *
 * @tparam T Element type
 * @tparam PatternFunc Callable: T pattern_func(size_t index)
 * @param buffer Host memory pointer
 * @param num_elements Total number of elements in buffer
 * @param pattern_func Function that generates expected value for each index
 * @param num_samples Number of elements to verify (0 = all)
 * @param tolerance Tolerance for floating-point comparison (ignored for integer types)
 * @param[out] first_error_index If verification fails, set to index of first mismatch
 * @param[out] expected_value If verification fails, set to expected value
 * @param[out] actual_value If verification fails, set to actual value
 * @return true if all samples match, false otherwise
 */
template<typename T, typename PatternFunc>
bool verifyHostBufferData(const void* buffer,
                          size_t      num_elements,
                          PatternFunc pattern_func,
                          size_t      num_samples       = 0,
                          double      tolerance         = 1e-5,
                          size_t*     first_error_index = nullptr,
                          T*          expected_value    = nullptr,
                          T*          actual_value      = nullptr)
{
    if(!buffer || num_elements == 0)
    {
        return false;
    }

    if(num_samples == 0)
    {
        num_samples = num_elements;
    }
    else
    {
        num_samples = std::min(num_samples, num_elements);
    }

    const T* typed = static_cast<const T*>(buffer);

    for(size_t i = 0; i < num_samples; i++)
    {
        T expected = pattern_func(i);
        T actual   = typed[i];

        bool matches = false;

        if constexpr(std::is_floating_point_v<T>)
        {
            matches = (std::abs(actual - expected) <= tolerance);
        }
        else
        {
            matches = (actual == expected);
        }

        if(!matches)
        {
            if(first_error_index)
                *first_error_index = i;
            if(expected_value)
                *expected_value = expected;
            if(actual_value)
                *actual_value = actual;
            return false;
        }
    }

    return true;
}

// ============================================================================
// Standard Test Pattern Factory
// ============================================================================

/**
 * @brief Returns a callable that generates the standard sequential byte pattern.
 *
 * Produces: static_cast<uint8_t>((seed + index) % 256)
 *
 * Intended for use with fillHostBufferWithPattern, verifyHostBufferData,
 * initializeBufferWithPattern, and verifyBufferData. Centralises the formula
 * so call sites do not repeat it.
 *
 * Example:
 * @code
 * fillHostBufferWithPattern<uint8_t>(buf, sz, makeBytePattern(seed));
 * verifyHostBufferData<uint8_t>(buf, sz, makeBytePattern(seed));
 * verifyHostBufferData<uint8_t>(buf, sz, makeBytePattern(seed), 0, 0.0, &errIdx, &errExp, &errGot);
 * @endcode
 *
 * @param seed Starting offset for the pattern.
 * @return Lambda: uint8_t f(size_t index)
 */
inline auto makeBytePattern(int seed)
{
    return [seed](size_t i) { return static_cast<uint8_t>((seed + i) % 256); };
}

} // namespace RCCLTestHelpers

#endif // HOST_BUFFER_HELPERS_HPP