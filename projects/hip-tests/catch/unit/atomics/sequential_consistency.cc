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

TEST_CASE("Unit___hip_atomic_load_store_Positive_Sequential_Consistency") {
  SECTION("WAVEFRONT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kLoadStore, __HIP_MEMORY_SCOPE_WAVEFRONT>();
  }
  SECTION("WORKGROUP") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kLoadStore, __HIP_MEMORY_SCOPE_WORKGROUP>();
  }
  SECTION("AGENT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kLoadStore, __HIP_MEMORY_SCOPE_AGENT>();
  }
  SECTION("SYSTEM") { SequentialConsistency::SystemTest<BuiltinAtomicOperation::kLoadStore>(); }
}

TEST_CASE("Unit___hip_atomic_exchange_Positive_Sequential_Consistency") {
  SECTION("WAVEFRONT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kExchange, __HIP_MEMORY_SCOPE_WAVEFRONT>();
  }
  SECTION("WORKGROUP") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kExchange, __HIP_MEMORY_SCOPE_WORKGROUP>();
  }
  SECTION("AGENT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kExchange, __HIP_MEMORY_SCOPE_AGENT>();
  }
  SECTION("SYSTEM") { SequentialConsistency::SystemTest<BuiltinAtomicOperation::kExchange>(); }
}

TEST_CASE("Unit___hip_atomic_compare_exchange_strong_Positive_Sequential_Consistency") {
  SECTION("WAVEFRONT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kCompareExchangeStrong,
                                __HIP_MEMORY_SCOPE_WAVEFRONT>();
  }
  SECTION("WORKGROUP") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kCompareExchangeStrong,
                                __HIP_MEMORY_SCOPE_WORKGROUP>();
  }
  SECTION("AGENT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kCompareExchangeStrong,
                                __HIP_MEMORY_SCOPE_AGENT>();
  }
  SECTION("SYSTEM") {
    SequentialConsistency::SystemTest<BuiltinAtomicOperation::kCompareExchangeStrong>();
  }
}

TEST_CASE("Unit___hip_atomic_compare_exchange_weak_Positive_Sequential_Consistency") {
  SECTION("WAVEFRONT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kCompareExchangeWeak,
                                __HIP_MEMORY_SCOPE_WAVEFRONT>();
  }
  SECTION("WORKGROUP") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kCompareExchangeWeak,
                                __HIP_MEMORY_SCOPE_WORKGROUP>();
  }
  SECTION("AGENT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kCompareExchangeWeak,
                                __HIP_MEMORY_SCOPE_AGENT>();
  }
  SECTION("SYSTEM") {
    SequentialConsistency::SystemTest<BuiltinAtomicOperation::kCompareExchangeWeak>();
  }
}

TEST_CASE("Unit___hip_atomic_fetch_add_Positive_Sequential_Consistency") {
  SECTION("WAVEFRONT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kAdd, __HIP_MEMORY_SCOPE_WAVEFRONT>();
  }
  SECTION("WORKGROUP") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kAdd, __HIP_MEMORY_SCOPE_WORKGROUP>();
  }
  SECTION("AGENT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kAdd, __HIP_MEMORY_SCOPE_AGENT>();
  }
  SECTION("SYSTEM") { SequentialConsistency::SystemTest<BuiltinAtomicOperation::kAdd>(); }
}

TEST_CASE("Unit___hip_atomic_fetch_and_Positive_Sequential_Consistency") {
  SECTION("WAVEFRONT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kAnd, __HIP_MEMORY_SCOPE_WAVEFRONT>();
  }
  SECTION("WORKGROUP") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kAnd, __HIP_MEMORY_SCOPE_WORKGROUP>();
  }
  SECTION("AGENT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kAnd, __HIP_MEMORY_SCOPE_AGENT>();
  }
  SECTION("SYSTEM") { SequentialConsistency::SystemTest<BuiltinAtomicOperation::kAnd>(); }
}

TEST_CASE("Unit___hip_atomic_fetch_or_Positive_Sequential_Consistency") {
  SECTION("WAVEFRONT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kOr, __HIP_MEMORY_SCOPE_WAVEFRONT>();
  }
  SECTION("WORKGROUP") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kOr, __HIP_MEMORY_SCOPE_WORKGROUP>();
  }
  SECTION("AGENT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kOr, __HIP_MEMORY_SCOPE_AGENT>();
  }
  SECTION("SYSTEM") { SequentialConsistency::SystemTest<BuiltinAtomicOperation::kOr>(); }
}

TEST_CASE("Unit___hip_atomic_fetch_xor_Positive_Sequential_Consistency") {
  SECTION("WAVEFRONT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kXor, __HIP_MEMORY_SCOPE_WAVEFRONT>();
  }
  SECTION("WORKGROUP") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kXor, __HIP_MEMORY_SCOPE_WORKGROUP>();
  }
  SECTION("AGENT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kXor, __HIP_MEMORY_SCOPE_AGENT>();
  }
  SECTION("SYSTEM") { SequentialConsistency::SystemTest<BuiltinAtomicOperation::kXor>(); }
}

TEST_CASE("Unit___hip_atomic_fetch_min_Positive_Sequential_Consistency") {
  SECTION("WAVEFRONT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kMin, __HIP_MEMORY_SCOPE_WAVEFRONT>();
  }
  SECTION("WORKGROUP") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kMin, __HIP_MEMORY_SCOPE_WORKGROUP>();
  }
  SECTION("AGENT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kMin, __HIP_MEMORY_SCOPE_AGENT>();
  }
  SECTION("SYSTEM") { SequentialConsistency::SystemTest<BuiltinAtomicOperation::kMin>(); }
}

TEST_CASE("Unit___hip_atomic_fetch_max_Positive_Sequential_Consistency") {
  SECTION("WAVEFRONT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kMax, __HIP_MEMORY_SCOPE_WAVEFRONT>();
  }
  SECTION("WORKGROUP") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kMax, __HIP_MEMORY_SCOPE_WORKGROUP>();
  }
  SECTION("AGENT") {
    SequentialConsistency::Test<BuiltinAtomicOperation::kMax, __HIP_MEMORY_SCOPE_AGENT>();
  }
  SECTION("SYSTEM") { SequentialConsistency::SystemTest<BuiltinAtomicOperation::kMax>(); }
}