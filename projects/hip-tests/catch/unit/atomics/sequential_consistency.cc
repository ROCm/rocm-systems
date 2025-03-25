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

#define HIP_SEQUENTIAL_CONSISTENCY_SECTION(operation, operation_str, alloc_type)                   \
  TEST_CASE("Unit___hip_atomic_" operation_str "_Positive_Sequential_Consistency_" #alloc_type) {  \
    SECTION("WAVEFRONT") {                                                                         \
      SequentialConsistency::Test<BuiltinAtomicOperation::operation,                               \
                                  __HIP_MEMORY_SCOPE_WAVEFRONT>(LinearAllocs::alloc_type);         \
    }                                                                                              \
    SECTION("WORKGROUP") {                                                                         \
      SequentialConsistency::Test<BuiltinAtomicOperation::operation,                               \
                                  __HIP_MEMORY_SCOPE_WORKGROUP>(LinearAllocs::alloc_type);         \
    }                                                                                              \
    SECTION("AGENT") {                                                                             \
      SequentialConsistency::Test<BuiltinAtomicOperation::operation, __HIP_MEMORY_SCOPE_AGENT>(    \
          LinearAllocs::alloc_type);                                                               \
    }                                                                                              \
  }

#define HIP_SEQUENTIAL_CONSISTENCY_SYSTEM_SECTION(operation, operation_str, alloc_type)            \
  TEST_CASE("Unit___hip_atomic_" operation_str                                                     \
            "_Positive_Sequential_Consistency_system_" #alloc_type) {                              \
    SECTION("SYSTEM") {                                                                            \
      SequentialConsistency::SystemTest<BuiltinAtomicOperation::operation>(                        \
          LinearAllocs::alloc_type);                                                               \
    }                                                                                              \
  }

// HIP_SEQUENTIAL_CONSISTENCY_SECTION(kLoadStore, "load_store", hipMalloc)
// HIP_SEQUENTIAL_CONSISTENCY_SYSTEM_SECTION(kLoadStore, "load_store", hipMallocManaged)

// HIP_SEQUENTIAL_CONSISTENCY_SECTION(kExchange, "exchange", hipMalloc)
// HIP_SEQUENTIAL_CONSISTENCY_SYSTEM_SECTION(kExchange, "exchange", hipMallocManaged)

// HIP_SEQUENTIAL_CONSISTENCY_SECTION(kCompareExchangeStrong, "compare_exchange_strong", hipMalloc)
// HIP_SEQUENTIAL_CONSISTENCY_SYSTEM_SECTION(kCompareExchangeStrong, "compare_exchange_strong", hipMallocManaged)

// HIP_SEQUENTIAL_CONSISTENCY_SECTION(kCompareExchangeWeak, "compare_exchange_weak", hipMalloc)
// HIP_SEQUENTIAL_CONSISTENCY_SYSTEM_SECTION(kCompareExchangeWeak, "compare_exchange_weak", hipMallocManaged)

// HIP_SEQUENTIAL_CONSISTENCY_SECTION(kAdd, "fetch_add", hipMalloc)
// HIP_SEQUENTIAL_CONSISTENCY_SYSTEM_SECTION(kAdd, "fetch_add", hipMallocManaged)

// HIP_SEQUENTIAL_CONSISTENCY_SECTION(kAnd, "fetch_and", hipMalloc)
// HIP_SEQUENTIAL_CONSISTENCY_SYSTEM_SECTION(kAnd, "fetch_and", hipMallocManaged)

// HIP_SEQUENTIAL_CONSISTENCY_SECTION(kOr, "fetch_or", hipMalloc)
// HIP_SEQUENTIAL_CONSISTENCY_SYSTEM_SECTION(kOr, "fetch_or", hipMallocManaged)

// HIP_SEQUENTIAL_CONSISTENCY_SECTION(kXor, "fetch_xor", hipMalloc)
// HIP_SEQUENTIAL_CONSISTENCY_SYSTEM_SECTION(kXor, "fetch_xor", hipMallocManaged)

// HIP_SEQUENTIAL_CONSISTENCY_SECTION(kMin, "fetch_min", hipMalloc)
// HIP_SEQUENTIAL_CONSISTENCY_SYSTEM_SECTION(kMin, "fetch_min", hipMallocManaged)

// HIP_SEQUENTIAL_CONSISTENCY_SECTION(kMax, "fetch_max", hipMalloc)
// HIP_SEQUENTIAL_CONSISTENCY_SYSTEM_SECTION(kMax, "fetch_max", hipMallocManaged)
