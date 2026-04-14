/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include "nccl.h"
#include <cmath>
#include <hip/hip_runtime.h>
#include <type_traits>
#include <vector>

/**
 * @file DeviceBufferHelpers.hpp
 * @brief Template-based device buffer utilities for RCCL tests
 *
 * Provides type-safe, reusable functions for device buffer operations:
 * - Initialization with test patterns (Host -> Device)
 * - Host <-> Device transfers
 * - Data verification (Device -> Host)
 * - NCCL datatype mapping
 *
 * NOTE: All functions expect DEVICE memory pointers allocated with hipMalloc().
 *       For host memory operations, use direct CPU operations instead.
 */

namespace RCCLTestHelpers
{

// ============================================================================
// NCCL Datatype Mapping
// ============================================================================

/**
 * @brief Maps C++ types to NCCL data types at compile time
 * @tparam T C++ data type
 */
template<typename T>
struct NcclTypeTraits;

/**
 * @brief Macro to define NcclTypeTraits specializations
 *
 * ncclDataType_t mapping and the string name using the stringification
 * operator (#) for each supported type.
 *
 * @param cpp_type The C++ type (e.g., uint64_t, float)
 * @param nccl_type The corresponding NCCL type (e.g., ncclUint64, ncclFloat)
 */
#define DEFINE_NCCL_TYPE_TRAIT(cpp_type, nccl_type)                  \
    template<>                                                       \
    struct NcclTypeTraits<cpp_type>                                  \
    {                                                                \
        static constexpr ncclDataType_t value = nccl_type;           \
        static constexpr const char*    name  = #cpp_type;           \
    }

// Define all supported type mappings
DEFINE_NCCL_TYPE_TRAIT(float,    ncclFloat);
DEFINE_NCCL_TYPE_TRAIT(double,   ncclDouble);
DEFINE_NCCL_TYPE_TRAIT(int8_t,   ncclInt8);
DEFINE_NCCL_TYPE_TRAIT(uint8_t,  ncclUint8);
DEFINE_NCCL_TYPE_TRAIT(int32_t,  ncclInt32);
DEFINE_NCCL_TYPE_TRAIT(uint32_t, ncclUint32);
DEFINE_NCCL_TYPE_TRAIT(int64_t,  ncclInt64);
DEFINE_NCCL_TYPE_TRAIT(uint64_t, ncclUint64);

// Undefine macro to avoid polluting namespace
#undef DEFINE_NCCL_TYPE_TRAIT

/**
 * @brief Helper function to get NCCL datatype for a C++ type
 * @tparam T C++ data type
 * @return Corresponding ncclDataType_t
 */
template<typename T>
constexpr ncclDataType_t getNcclDataType()
{
    return NcclTypeTraits<T>::value;
}

/**
 * @brief Helper function to get type name string
 * @tparam T C++ data type
 * @return Type name as string
 */
template<typename T>
constexpr const char* getTypeName()
{
    return NcclTypeTraits<T>::name;
}

// ============================================================================
// Device Buffer Initialization
// ============================================================================

/**
 * @brief Initialize device buffer with pattern function
 *
 * Generic function that allows any pattern generation via lambda or function pointer.
 *
 * Example usage:
 * @code
 * // Rank-based pattern: rank * multiplier + index
 * initializeBufferWithPattern<float>(buffer, size,
 *     [rank, multiplier](size_t i) { return rank * multiplier + i; });
 *
 * // Constant value pattern
 * initializeBufferWithPattern<int>(buffer, size,
 *     [](size_t i) { return 42; });
 *
 * // Custom pattern
 * initializeBufferWithPattern<double>(buffer, size,
 *     [](size_t i) { return std::sin(i * 0.1); });
 * @endcode
 *
 * @tparam T Element type (float, int, etc.)
 * @tparam PatternFunc Callable type (lambda, function pointer, functor)
 * @param device_buffer Device memory pointer (from hipMalloc)
 * @param num_elements Number of elements
 * @param pattern_func Function that generates value for each index: T pattern_func(size_t index)
 * @return hipError_t from hipMemcpy, or hipSuccess
 */
template<typename T, typename PatternFunc>
hipError_t initializeBufferWithPattern(void*       device_buffer,
                                       size_t      num_elements,
                                       PatternFunc pattern_func)
{
    if(!device_buffer || num_elements == 0)
    {
        return hipErrorInvalidValue;
    }

    std::vector<T> host_data(num_elements);
    for(size_t i = 0; i < num_elements; i++)
    {
        host_data[i] = pattern_func(i);
    }

    return hipMemcpy(device_buffer,
                     host_data.data(),
                     num_elements * sizeof(T),
                     hipMemcpyHostToDevice);
}

/**
 * @brief Zero-initialize device buffer
 *
 * @tparam T Element type
 * @param device_buffer Device memory pointer (from hipMalloc)
 * @param num_elements Number of elements
 * @return hipError_t from hipMemset
 */
template<typename T>
hipError_t zeroInitializeBuffer(void* device_buffer, size_t num_elements)
{
    if(!device_buffer || num_elements == 0)
    {
        return hipErrorInvalidValue;
    }

    return hipMemset(device_buffer, 0, num_elements * sizeof(T));
}

// ============================================================================
// Device Buffer Verification
// ============================================================================

/**
 * @brief Verify device buffer data with pattern function
 *
 * Generic function that allows any verification pattern via lambda or function pointer.
 * Downloads data from device and verifies elements against expected values.
 * Uses appropriate comparison for floating-point vs integer types.
 *
 * Example usage:
 * @code
 * // Rank-based pattern verification: rank * multiplier + index
 * verifyBufferData<float>(buffer, size,
 *     [rank, multiplier](size_t i) { return rank * multiplier + i; },
 *     num_samples, tolerance);
 *
 * // Constant value verification
 * verifyBufferData<int>(buffer, size,
 *     [](size_t i) { return 42; });
 *
 * // Custom pattern verification
 * verifyBufferData<double>(buffer, size,
 *     [](size_t i) { return std::sin(i * 0.1); },
 *     size, 1e-6);  // verify all elements with tighter tolerance
 * @endcode
 *
 * @tparam T Element type
 * @tparam PatternFunc Callable type (lambda, function pointer, functor)
 * @param device_buffer Device memory pointer (from hipMalloc)
 * @param num_elements Total number of elements in buffer
 * @param pattern_func Function that generates expected value for each index: T pattern_func(size_t index)
 * @param num_samples Number of elements to verify (default: all, capped at num_elements)
 * @param tolerance Tolerance for floating-point comparison (default: 1e-5, ignored for integer types)
 * @param[out] first_error_index If verification fails, set to index of first mismatch
 * @param[out] expected_value If verification fails, set to expected value
 * @param[out] actual_value If verification fails, set to actual value
 * @return true if all samples match, false otherwise
 */
template<typename T, typename PatternFunc>
bool verifyBufferData(const void* device_buffer,
                      size_t      num_elements,
                      PatternFunc pattern_func,
                      size_t      num_samples       = 0,  // 0 means verify all
                      double      tolerance         = 1e-5,
                      size_t*     first_error_index = nullptr,
                      T*          expected_value    = nullptr,
                      T*          actual_value      = nullptr)
{
    if(!device_buffer || num_elements == 0)
    {
        return false;
    }

    // Default to verifying all elements if num_samples is 0
    if(num_samples == 0)
    {
        num_samples = num_elements;
    }
    else
    {
        // Cap num_samples at num_elements
        num_samples = std::min(num_samples, num_elements);
    }

    // Download data from device
    std::vector<T> host_data(num_elements);
    hipError_t     err = hipMemcpy(host_data.data(),
                               device_buffer,
                               num_elements * sizeof(T),
                               hipMemcpyDeviceToHost);
    if(err != hipSuccess)
    {
        return false;
    }

    // Verify samples
    for(size_t i = 0; i < num_samples; i++)
    {
        T expected = pattern_func(i);
        T actual   = host_data[i];

        bool matches = false;

        // Use appropriate comparison based on type
        if constexpr(std::is_floating_point_v<T>)
        {
            // Floating-point: use tolerance-based comparison
            matches = (std::abs(actual - expected) <= tolerance);
        }
        else
        {
            // Integer: exact comparison
            matches = (actual == expected);
        }

        if(!matches)
        {
            // Record error details
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
// Combined Operations
// ============================================================================

// Forward declaration for downloadBuffer (used in allocateAndInitialize)
template<typename T>
std::pair<hipError_t, std::vector<T>> downloadBuffer(const void* device_buffer, size_t num_elements);

/**
 * @brief Allocate, initialize, and return RAII-guarded device buffers
 *
 * Convenience function that combines allocation and initialization.
 * Returns host vector for later verification if needed.
 *
 * @tparam T Element type
 * @param[out] device_buffer Pointer to receive device buffer address
 * @param num_elements Number of elements
 * @param rank MPI rank for pattern generation
 * @param multiplier Pattern multiplier
 * @return std::pair<hipError_t, std::vector<T>> - error code and host data copy
 */
template<typename T>
std::pair<hipError_t, std::vector<T>> allocateAndInitialize(void** device_buffer,
                                                            size_t num_elements,
                                                            int    rank,
                                                            int    multiplier = 1000)
{
    if(!device_buffer)
    {
        return {hipErrorInvalidValue, {}};
    }

    // Allocate device memory
    hipError_t err = hipMalloc(device_buffer, num_elements * sizeof(T));
    if(err != hipSuccess)
    {
        return {err, {}};
    }

    // Initialize using generic pattern function
    err = initializeBufferWithPattern<T>(
        *device_buffer, num_elements,
        [rank, multiplier](size_t i) { return static_cast<T>(rank * multiplier + i); });

    if(err != hipSuccess)
    {
        return {err, {}};
    }

    // Download and return host copy for verification
    return downloadBuffer<T>(*device_buffer, num_elements);
}

/**
 * @brief Copy data from one device buffer to another
 *
 * @tparam T Element type (used for size calculation)
 * @param dst Destination device buffer (from hipMalloc)
 * @param src Source device buffer (from hipMalloc)
 * @param num_elements Number of elements to copy
 * @return hipError_t from hipMemcpy
 */
template<typename T>
hipError_t copyDeviceBuffer(void* dst, const void* src, size_t num_elements)
{
    if(!dst || !src || num_elements == 0)
    {
        return hipErrorInvalidValue;
    }

    return hipMemcpy(dst, src, num_elements * sizeof(T), hipMemcpyDeviceToDevice);
}

/**
 * @brief Download device buffer to host vector
 *
 * @tparam T Element type
 * @param device_buffer Device memory pointer (from hipMalloc)
 * @param num_elements Number of elements
 * @return std::pair<hipError_t, std::vector<T>> - error code and host data
 */
template<typename T>
std::pair<hipError_t, std::vector<T>> downloadBuffer(const void* device_buffer, size_t num_elements)
{
    std::vector<T> host_data(num_elements);

    if(!device_buffer || num_elements == 0)
    {
        return {hipErrorInvalidValue, {}};
    }

    hipError_t err = hipMemcpy(host_data.data(),
                               device_buffer,
                               num_elements * sizeof(T),
                               hipMemcpyDeviceToHost);

    return {err, std::move(host_data)};
}

} // namespace RCCLTestHelpers


