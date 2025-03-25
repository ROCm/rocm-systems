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

#include <hip_test_common.hh>

#include "memory_order_common.hh"

#define HIP_ACQUIRE_RELEASE_SECTION(operation, alloc_type)                                         \
  SECTION("ACQUIRE/RELEASE") {                                                                     \
    SECTION("WAVEFRONT") {                                                                         \
      AcquireRelease::Test<BuiltinAtomicOperation::operation, __ATOMIC_ACQUIRE,                    \
                           __HIP_MEMORY_SCOPE_WAVEFRONT>(LinearAllocs::alloc_type);                \
    }                                                                                              \
    SECTION("WORKGROUP") {                                                                         \
      AcquireRelease::Test<BuiltinAtomicOperation::kLoadStore, __ATOMIC_ACQUIRE,                   \
                           __HIP_MEMORY_SCOPE_WORKGROUP>(LinearAllocs::alloc_type);                \
    }                                                                                              \
    SECTION("AGENT") {                                                                             \
      AcquireRelease::Test<BuiltinAtomicOperation::kLoadStore, __ATOMIC_ACQUIRE,                   \
                           __HIP_MEMORY_SCOPE_AGENT>(LinearAllocs::alloc_type);                    \
    }                                                                                              \
  }

#define HIP_ACQ_REL_SECTION(operation, alloc_type)                                                 \
  SECTION("ACQ_REL") {                                                                             \
    SECTION("WAVEFRONT") {                                                                         \
      AcquireRelease::Test<BuiltinAtomicOperation::operation, __ATOMIC_ACQ_REL,                    \
                           __HIP_MEMORY_SCOPE_WAVEFRONT>(LinearAllocs::alloc_type);                \
    }                                                                                              \
    SECTION("WORKGROUP") {                                                                         \
      AcquireRelease::Test<BuiltinAtomicOperation::operation, __ATOMIC_ACQ_REL,                    \
                           __HIP_MEMORY_SCOPE_WORKGROUP>(LinearAllocs::alloc_type);                \
    }                                                                                              \
    SECTION("AGENT") {                                                                             \
      AcquireRelease::Test<BuiltinAtomicOperation::operation, __ATOMIC_ACQ_REL,                    \
                           __HIP_MEMORY_SCOPE_AGENT>(LinearAllocs::alloc_type);                    \
    }                                                                                              \
  }

#define HIP_SEQ_CST_SECTION(operation, alloc_type)                                                 \
  SECTION("SEQ_CST") {                                                                             \
    SECTION("WAVEFRONT") {                                                                         \
      AcquireRelease::Test<BuiltinAtomicOperation::operation, __ATOMIC_SEQ_CST,                    \
                           __HIP_MEMORY_SCOPE_WAVEFRONT>(LinearAllocs::alloc_type);                \
    }                                                                                              \
    SECTION("WORKGROUP") {                                                                         \
      AcquireRelease::Test<BuiltinAtomicOperation::kLoadStore, __ATOMIC_SEQ_CST,                   \
                           __HIP_MEMORY_SCOPE_WORKGROUP>(LinearAllocs::alloc_type);                \
    }                                                                                              \
    SECTION("AGENT") {                                                                             \
      AcquireRelease::Test<BuiltinAtomicOperation::kLoadStore, __ATOMIC_SEQ_CST,                   \
                           __HIP_MEMORY_SCOPE_AGENT>(LinearAllocs::alloc_type);                    \
    }                                                                                              \
  }

#define HIP_ACQUIRE_RELEASE_SYSTEM_SECTION(operation, alloc_type)                                  \
  SECTION("ACQUIRE/RELEASE") {                                                                     \
    SECTION("SYSTEM") {                                                                            \
      AcquireRelease::SystemTest<BuiltinAtomicOperation::operation, __ATOMIC_ACQUIRE>(             \
          LinearAllocs::alloc_type);                                                               \
    }                                                                                              \
  }

#define HIP_ACQ_REL_SYSTEM_SECTION(operation, alloc_type)                                          \
  SECTION("ACQ_REL") {                                                                             \
    SECTION("SYSTEM") {                                                                            \
      AcquireRelease::SystemTest<BuiltinAtomicOperation::operation, __ATOMIC_ACQ_REL>(             \
          LinearAllocs::alloc_type);                                                               \
    }                                                                                              \
  }

#define HIP_SEQ_CST_SYSTEM_SECTION(operation, alloc_type)                                          \
  SECTION("SEQ_CST") {                                                                             \
    SECTION("SYSTEM") {                                                                            \
      AcquireRelease::SystemTest<BuiltinAtomicOperation::operation, __ATOMIC_SEQ_CST>(             \
          LinearAllocs::alloc_type);                                                               \
    }                                                                                              \
  }

#define HIP_POSITIVE_ACQUIRE_SEQ_CST_TEST(operation, operation_str, alloc_type)                    \
  TEST_CASE("Unit___hip_atomic_" operation_str "_Positive_Acquire_Release_" #alloc_type) {         \
    HIP_ACQUIRE_RELEASE_SECTION(operation, alloc_type)                                             \
    HIP_SEQ_CST_SECTION(operation, alloc_type)                                                     \
  }

#define HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(operation, operation_str, alloc_type)            \
  TEST_CASE("Unit___hip_atomic_" operation_str "_Positive_Acquire_Release_" #alloc_type) {         \
    HIP_ACQUIRE_RELEASE_SECTION(operation, alloc_type)                                             \
    HIP_ACQ_REL_SECTION(operation, alloc_type)                                                     \
    HIP_SEQ_CST_SECTION(operation, alloc_type)                                                     \
  }

#define HIP_POSITIVE_ACQUIRE_SEQ_CST_SYSTEM_TEST(operation, operation_str, alloc_type)             \
  TEST_CASE("Unit___hip_atomic_" operation_str "_Positive_Acquire_Release_system_" #alloc_type) {  \
    HIP_ACQUIRE_RELEASE_SYSTEM_SECTION(operation, alloc_type)                                      \
    HIP_SEQ_CST_SYSTEM_SECTION(operation, alloc_type)                                              \
  }

#define HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(operation, operation_str, alloc_type)     \
  TEST_CASE("Unit___hip_atomic_" operation_str "_Positive_Acquire_Release_system_" #alloc_type) {  \
    HIP_ACQUIRE_RELEASE_SYSTEM_SECTION(operation, alloc_type)                                      \
    HIP_ACQ_REL_SYSTEM_SECTION(operation, alloc_type)                                              \
    HIP_SEQ_CST_SYSTEM_SECTION(operation, alloc_type)                                              \
  }

// HIP_POSITIVE_ACQUIRE_SEQ_CST_TEST(kLoadStore, "load_store", hipMalloc)
// HIP_POSITIVE_ACQUIRE_SEQ_CST_TEST(kLoadStore, "load_store", hipMallocManaged)
// HIP_POSITIVE_ACQUIRE_SEQ_CST_SYSTEM_TEST(kLoadStore, "load_store", hipHostMalloc)
// HIP_POSITIVE_ACQUIRE_SEQ_CST_SYSTEM_TEST(kLoadStore, "load_store", hipMallocManaged)

// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kExchange, "exchange", hipMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kExchange, "exchange", hipMallocManaged)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kExchange, "exchange", hipHostMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kExchange, "exchange", hipMallocManaged)

// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kCompareExchangeStrong, "compare_exchange_strong",
//                                           hipMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kCompareExchangeStrong, "compare_exchange_strong",
//                                           hipMallocManaged)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kCompareExchangeStrong, "compare_exchange_strong",
//                                                  hipHostMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kCompareExchangeStrong, "compare_exchange_strong",
//                                                  hipMallocManaged)

// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kCompareExchangeWeak, "compare_exchange_weak", hipMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kCompareExchangeWeak, "compare_exchange_weak",
//                                           hipMallocManaged)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kCompareExchangeWeak, "compare_exchange_weak",
//                                                  hipHostMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kCompareExchangeWeak, "compare_exchange_weak",
//                                                  hipMallocManaged)

// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kAdd, "fetch_add", hipMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kAdd, "fetch_add", hipMallocManaged)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kAdd, "fetch_add", hipHostMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kAdd, "fetch_add", hipMallocManaged)

// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kAnd, "fetch_and", hipMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kAnd, "fetch_and", hipMallocManaged)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kAnd, "fetch_and", hipHostMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kAnd, "fetch_and", hipMallocManaged)

// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kOr, "fetch_or", hipMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kOr, "fetch_or", hipMallocManaged)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kOr, "fetch_or", hipHostMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kOr, "fetch_or", hipMallocManaged)

// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kXor, "fetch_xor", hipMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kXor, "fetch_xor", hipMallocManaged)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kXor, "fetch_xor", hipHostMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kXor, "fetch_xor", hipMallocManaged)

// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kMin, "fetch_min", hipMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kMin, "fetch_min", hipMallocManaged)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kMin, "fetch_min", hipHostMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kMin, "fetch_min", hipMallocManaged)

// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kMax, "fetch_max", hipMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_TEST(kMax, "fetch_max", hipMallocManaged)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kMax, "fetch_max", hipHostMalloc)
// HIP_POSITIVE_ACQUIRE_ACQ_REL_SEQ_CST_SYSTEM_TEST(kMax, "fetch_max", hipMallocManaged)
