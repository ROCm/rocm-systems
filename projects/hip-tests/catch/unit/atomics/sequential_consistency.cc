/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include "memory_order_common.hh"

HIP_TEST_CASE(Unit___hip_atomic_load_store_Positive_Sequential_Consistency) {
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

HIP_TEST_CASE(Unit___hip_atomic_exchange_Positive_Sequential_Consistency) {
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

HIP_TEST_CASE(Unit___hip_atomic_compare_exchange_strong_Positive_Sequential_Consistency) {
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

HIP_TEST_CASE(Unit___hip_atomic_compare_exchange_weak_Positive_Sequential_Consistency) {
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

HIP_TEST_CASE(Unit___hip_atomic_fetch_add_Positive_Sequential_Consistency) {
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

HIP_TEST_CASE(Unit___hip_atomic_fetch_and_Positive_Sequential_Consistency) {
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

HIP_TEST_CASE(Unit___hip_atomic_fetch_or_Positive_Sequential_Consistency) {
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

HIP_TEST_CASE(Unit___hip_atomic_fetch_xor_Positive_Sequential_Consistency) {
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

HIP_TEST_CASE(Unit___hip_atomic_fetch_min_Positive_Sequential_Consistency) {
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

HIP_TEST_CASE(Unit___hip_atomic_fetch_max_Positive_Sequential_Consistency) {
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
