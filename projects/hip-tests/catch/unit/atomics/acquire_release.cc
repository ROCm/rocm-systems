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

TEST_CASE("Unit___hip_atomic_load_store_Positive_Acquire_Release") {
  SECTION("ACQUIRE/RELEASE") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kLoadStore, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kLoadStore, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kLoadStore, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kLoadStore, __ATOMIC_ACQUIRE>();
    }
  }
  SECTION("SEQ_CST") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kLoadStore, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kLoadStore, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kLoadStore, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kLoadStore, __ATOMIC_SEQ_CST>();
    }
  }
}

TEST_CASE("Unit___hip_atomic_exchange_Positive_Acquire_Release") {
  SECTION("ACQUIRE/RELEASE") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kExchange, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kExchange, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kExchange, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kExchange, __ATOMIC_ACQUIRE>();
    }
  }
  SECTION("ACQ_REL") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kExchange, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kExchange, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kExchange, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kExchange, __ATOMIC_ACQ_REL>();
    }
  }
  SECTION("SEQ_CST") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kExchange, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kExchange, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kExchange, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kExchange, __ATOMIC_SEQ_CST>();
    }
  }
}

TEST_CASE("Unit___hip_atomic_compare_exchange_strong_Positive_Acquire_Release") {
  SECTION("ACQUIRE/RELEASE") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeStrong, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeStrong, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeStrong, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kCompareExchangeStrong,
                                 __ATOMIC_ACQUIRE>();
    }
  }
  SECTION("ACQ_REL") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeStrong, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeStrong, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeStrong, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kCompareExchangeStrong,
                                 __ATOMIC_ACQ_REL>();
    }
  }
  SECTION("SEQ_CST") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeStrong, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeStrong, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeStrong, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kCompareExchangeStrong,
                                 __ATOMIC_SEQ_CST>();
    }
  }
}

TEST_CASE("Unit___hip_atomic_compare_exchange_weak_Positive_Acquire_Release") {
  SECTION("ACQUIRE/RELEASE") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeWeak, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeWeak, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeWeak, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kCompareExchangeWeak, __ATOMIC_ACQUIRE>();
    }
  }
  SECTION("ACQ_REL") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeWeak, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeWeak, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeWeak, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kCompareExchangeWeak, __ATOMIC_ACQ_REL>();
    }
  }
  SECTION("SEQ_CST") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeWeak, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeWeak, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kCompareExchangeWeak, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kCompareExchangeWeak, __ATOMIC_SEQ_CST>();
    }
  }
}

TEST_CASE("Unit___hip_atomic_fetch_add_Positive_Acquire_Release") {
  SECTION("ACQUIRE/RELEASE") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAdd, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAdd, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAdd, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kAdd, __ATOMIC_ACQUIRE>();
    }
  }
  SECTION("ACQ_REL") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAdd, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAdd, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAdd, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kAdd, __ATOMIC_ACQ_REL>();
    }
  }
  SECTION("SEQ_CST") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAdd, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAdd, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAdd, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kAdd, __ATOMIC_SEQ_CST>();
    }
  }
}

TEST_CASE("Unit___hip_atomic_fetch_and_Positive_Acquire_Release") {
  SECTION("ACQUIRE/RELEASE") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAnd, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAnd, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAnd, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kAnd, __ATOMIC_ACQUIRE>();
    }
  }
  SECTION("ACQ_REL") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAnd, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAnd, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAnd, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kAnd, __ATOMIC_ACQ_REL>();
    }
  }
  SECTION("SEQ_CST") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAnd, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAnd, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kAnd, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kAnd, __ATOMIC_SEQ_CST>();
    }
  }
}

TEST_CASE("Unit___hip_atomic_fetch_or_Positive_Acquire_Release") {
  SECTION("ACQUIRE/RELEASE") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kOr, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kOr, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kOr, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kOr, __ATOMIC_ACQUIRE>();
    }
  }
  SECTION("ACQ_REL") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kOr, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kOr, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kOr, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kOr, __ATOMIC_ACQ_REL>();
    }
  }
  SECTION("SEQ_CST") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kOr, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kOr, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kOr, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kOr, __ATOMIC_SEQ_CST>();
    }
  }
}

TEST_CASE("Unit___hip_atomic_fetch_xor_Positive_Acquire_Release") {
  SECTION("ACQUIRE/RELEASE") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kXor, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kXor, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kXor, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kXor, __ATOMIC_ACQUIRE>();
    }
  }
  SECTION("ACQ_REL") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kXor, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kXor, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kXor, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kXor, __ATOMIC_ACQ_REL>();
    }
  }
  SECTION("SEQ_CST") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kXor, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kXor, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kXor, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kXor, __ATOMIC_SEQ_CST>();
    }
  }
}

TEST_CASE("Unit___hip_atomic_fetch_min_Positive_Acquire_Release") {
  SECTION("ACQUIRE/RELEASE") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMin, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMin, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMin, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kMin, __ATOMIC_ACQUIRE>();
    }
  }
  SECTION("ACQ_REL") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMin, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMin, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMin, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kMin, __ATOMIC_ACQ_REL>();
    }
  }
  SECTION("SEQ_CST") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMin, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMin, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMin, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kMin, __ATOMIC_SEQ_CST>();
    }
  }
}

TEST_CASE("Unit___hip_atomic_fetch_max_Positive_Acquire_Release") {
  SECTION("ACQUIRE/RELEASE") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMax, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMax, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMax, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kMax, __ATOMIC_ACQUIRE>();
    }
  }
  SECTION("ACQ_REL") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMax, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMax, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMax, __ATOMIC_ACQ_REL,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kMax, __ATOMIC_ACQ_REL>();
    }
  }
  SECTION("SEQ_CST") {
    SECTION("WAVEFRONT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMax, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WAVEFRONT>();
    }
    SECTION("WORKGROUP") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMax, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_WORKGROUP>();
    }
    SECTION("AGENT") {
      AcquireRelease::Test<BuiltinAtomicOperation::kMax, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_AGENT>();
    }
    SECTION("SYSTEM") {
      AcquireRelease::SystemTest<BuiltinAtomicOperation::kMax, __ATOMIC_SEQ_CST>();
    }
  }
}