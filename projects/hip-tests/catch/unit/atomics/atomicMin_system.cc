/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "min_max_common.hh"

#include <hip_test_common.hh>

#if HT_NVIDIA
#define TEST_TYPES int, unsigned int, unsigned long, unsigned long long
#else
#define TEST_TYPES int, unsigned int, unsigned long, unsigned long long, float, double
#endif

/**
 * @addtogroup atomicMin_system atomicMin_system
 * @{
 * @ingroup AtomicsTest
 * `atomicMin_system(TestType* address, TestType* val)` -
 * performs system-wide atomic minimum between address and val, returns old value.
 */

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMin_system from multiple threads on the same address.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMin_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
#define ATOMIC_MIN_SYSTEM_POSITIVE_PEER_GPUS_SAME_ADDRESS_TEST(alloc_type)                         \
  TEMPLATE_TEST_CASE("Unit_atomicMin_system_Positive_Peer_GPUs_Same_Address_" #alloc_type, "",     \
                     TEST_TYPES) {                                                                 \
    for (auto current = 0; current < cmd_options.atomic_iterations; ++current) {                   \
      DYNAMIC_SECTION("Same address " << current) {                                                \
        MinMax::MultipleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMinSystem>(   \
            2, 2, 1, sizeof(TestType), LinearAllocs::alloc_type);                                  \
      }                                                                                            \
    }                                                                                              \
  }

ATOMIC_MIN_SYSTEM_POSITIVE_PEER_GPUS_SAME_ADDRESS_TEST(hipHostMalloc)
// ATOMIC_MIN_SYSTEM_POSITIVE_PEER_GPUS_SAME_ADDRESS_TEST(hipMallocManaged)
// ATOMIC_MIN_SYSTEM_POSITIVE_PEER_GPUS_SAME_ADDRESS_TEST(mallocAndRegister)

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMin_system from multiple threads on adjacent addresses.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMin_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
#define ATOMIC_MIN_SYSTEM_POSITIVE_PEER_GPUS_ADJACENT_ADDRESSES_TEST(alloc_type)                   \
  TEMPLATE_TEST_CASE("Unit_atomicMin_system_Positive_Peer_GPUs_Adjacent_Addresses_" #alloc_type,   \
                     "", TEST_TYPES) {                                                             \
    int warp_size = 0;                                                                             \
    HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));                   \
                                                                                                   \
    for (auto current = 0; current < cmd_options.atomic_iterations; ++current) {                   \
      DYNAMIC_SECTION("Adjacent address " << current) {                                            \
        MinMax::MultipleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMinSystem>(   \
            2, 2, warp_size, sizeof(TestType), LinearAllocs::alloc_type);                          \
      }                                                                                            \
    }                                                                                              \
  }

ATOMIC_MIN_SYSTEM_POSITIVE_PEER_GPUS_ADJACENT_ADDRESSES_TEST(hipHostMalloc)
// ATOMIC_MIN_SYSTEM_POSITIVE_PEER_GPUS_ADJACENT_ADDRESSES_TEST(hipMallocManaged)
// ATOMIC_MIN_SYSTEM_POSITIVE_PEER_GPUS_ADJACENT_ADDRESSES_TEST(mallocAndRegister)

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMin_system from multiple threads on scaterred addresses.
 *  - Uses multiple devices and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMin_system.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 5.2
 */
#define ATOMIC_MIN_SYSTEM_POSITIVE_PEER_GPUS_SCATTERED_ADDRESSES_TEST(alloc_type)                  \
  TEMPLATE_TEST_CASE("Unit_atomicMin_system_Positive_Peer_GPUs_Scattered_Addresses_" #alloc_type,  \
                     "", TEST_TYPES) {                                                             \
    int warp_size = 0;                                                                             \
    HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));                   \
    const auto cache_line_size = 128u;                                                             \
                                                                                                   \
    for (auto current = 0; current < cmd_options.atomic_iterations; ++current) {                   \
      DYNAMIC_SECTION("Scattered address " << current) {                                           \
        MinMax::MultipleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMinSystem>(   \
            2, 2, warp_size, cache_line_size, LinearAllocs::alloc_type);                           \
      }                                                                                            \
    }                                                                                              \
  }

ATOMIC_MIN_SYSTEM_POSITIVE_PEER_GPUS_SCATTERED_ADDRESSES_TEST(hipHostMalloc)
// ATOMIC_MIN_SYSTEM_POSITIVE_PEER_GPUS_SCATTERED_ADDRESSES_TEST(hipMallocManaged)
// ATOMIC_MIN_SYSTEM_POSITIVE_PEER_GPUS_SCATTERED_ADDRESSES_TEST(mallocAndRegister)

/**
 * End doxygen group AtomicsTest.
 * @}
 */
