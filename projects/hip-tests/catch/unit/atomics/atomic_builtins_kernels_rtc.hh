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

#pragma once

/*
Positive and negative kernels used for the builtin atomic Test Cases that are using RTC.
*/

static constexpr auto kBuiltinStore{R"(
  constexpr int kMemOrder = __ATOMIC_RELAXED;
  constexpr int kMemScope = __HIP_MEMORY_SCOPE_SYSTEM;

  class DummyTC {
  public:
    __device__ DummyTC() {}
    __device__ ~DummyTC() = default;
    __device__ DummyTC(const DummyTC&) = default;
    __device__ DummyTC& operator=(const DummyTC&) = default;
    __device__ DummyTC(DummyTC&&) = default;
    __device__ DummyTC& operator=(DummyTC&&) = default;
  };

  class Dummy {
  public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };

  __global__ void StoreCompileKernel(int* x) {
    __hip_atomic_store(x, 1, __ATOMIC_RELAXED, kMemScope);
    __hip_atomic_store(x, 1, __ATOMIC_RELEASE, kMemScope);
    __hip_atomic_store(x, 1, __ATOMIC_SEQ_CST, kMemScope);

    __hip_atomic_store(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
    __hip_atomic_store(*x, 1, kMemOrder, kMemScope);
    __hip_atomic_store(x, 1, __ATOMIC_CONSUME, kMemScope);
    __hip_atomic_store(x, 1, __ATOMIC_ACQUIRE, kMemScope);
    __hip_atomic_store(x, 1, __ATOMIC_ACQ_REL, kMemScope);
    __hip_atomic_store(x, 1, -1, kMemScope);
    __hip_atomic_store(x, 1, 10, kMemScope);
    __hip_atomic_store(x, 1, kMemOrder, -1);
    __hip_atomic_store(x, 1, kMemOrder, 10);

    Dummy dummy_a{};
    Dummy dummy_b{};
    __hip_atomic_store(&dummy_a, dummy_b, kMemOrder, kMemScope);

    DummyTC dummytc_a{};
    DummyTC dummytc_b{};
    __hip_atomic_store(&dummytc_a, dummytc_b, kMemOrder, kMemScope);
  }
)"};

static constexpr auto kBuiltinLoad{R"(
  constexpr int kMemOrder = __ATOMIC_RELAXED;
  constexpr int kMemScope = __HIP_MEMORY_SCOPE_SYSTEM;

  class DummyTC {
  public:
    __device__ DummyTC() {}
    __device__ ~DummyTC() = default;
    __device__ DummyTC(const DummyTC&) = default;
    __device__ DummyTC& operator=(const DummyTC&) = default;
    __device__ DummyTC(DummyTC&&) = default;
    __device__ DummyTC& operator=(DummyTC&&) = default;
  };

  class Dummy {
  public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };

  __global__ void LoadCompileKernel(int* x, int* y) {
    *y = __hip_atomic_load(x, __ATOMIC_RELAXED, kMemScope);
    *y = __hip_atomic_load(x, __ATOMIC_CONSUME, kMemScope);
    *y = __hip_atomic_load(x, __ATOMIC_ACQUIRE, kMemScope);
    *y = __hip_atomic_load(x, __ATOMIC_SEQ_CST, kMemScope);

    *y = __hip_atomic_load(*x, kMemOrder, kMemScope);
    *y = __hip_atomic_load(x, __ATOMIC_RELEASE, kMemScope);
    *y = __hip_atomic_load(x, __ATOMIC_ACQ_REL, kMemScope);
    *y = __hip_atomic_load(x, -1, kMemScope);
    *y = __hip_atomic_load(x, 10, kMemScope);
    *y = __hip_atomic_load(x, kMemOrder, -1);
    *y = __hip_atomic_load(x, kMemOrder, 10);

    Dummy dummy_a{};
    Dummy dummy_b{};
    dummy_a = __hip_atomic_load(&dummy_b, kMemOrder, kMemScope);

    DummyTC dummytc_a{};
    DummyTC dummytc_b{};
    dummytc_a = __hip_atomic_load(&dummytc_b, kMemOrder, kMemScope);
  }
)"};

static constexpr auto kBuiltinCompExWeak{R"(
  constexpr int kMemOrder = __ATOMIC_RELAXED;
  constexpr int kMemScope = __HIP_MEMORY_SCOPE_SYSTEM;

  class DummyTC {
  public:
    __device__ DummyTC() {}
    __device__ ~DummyTC() = default;
    __device__ DummyTC(const DummyTC&) = default;
    __device__ DummyTC& operator=(const DummyTC&) = default;
    __device__ DummyTC(DummyTC&&) = default;
    __device__ DummyTC& operator=(DummyTC&&) = default;
  };

  class Dummy {
  public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };

  __global__ void CompareWeakCompileKernel(int* x, int* expected) {
    bool res{false};
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_CONSUME, __ATOMIC_RELAXED,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_CONSUME, __ATOMIC_CONSUME,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_ACQUIRE, __ATOMIC_CONSUME,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_RELEASE, __ATOMIC_RELAXED,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_RELEASE, __ATOMIC_CONSUME,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_ACQ_REL, __ATOMIC_CONSUME,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_CONSUME,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_ACQ_REL,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST,
                                            kMemScope);

    res = __hip_atomic_compare_exchange_weak(x, expected, 1, kMemOrder, __ATOMIC_RELEASE, kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, kMemOrder, __ATOMIC_ACQ_REL, kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, __ATOMIC_RELAXED, __ATOMIC_SEQ_CST,
                                            kMemScope);
    res = __hip_atomic_compare_exchange_weak(reinterpret_cast<const int*>(x), expected, 1, kMemOrder,
                                            kMemOrder, kMemScope);
    res = __hip_atomic_compare_exchange_weak(*x, expected, 1, kMemOrder, kMemOrder, kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, -1, kMemOrder, kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, 10, kMemOrder, kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, kMemOrder, -1, kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, kMemOrder, 10, kMemScope);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, kMemOrder, kMemOrder, -1);
    res = __hip_atomic_compare_exchange_weak(x, expected, 1, kMemOrder, kMemOrder, 10);

    Dummy dummy_a{};
    Dummy dummy_b{};
    Dummy dummy_c{};
    res = __hip_atomic_compare_exchange_weak(&dummy_a, &dummy_b, dummy_c, kMemOrder, kMemOrder,
                                            kMemScope);
    DummyTC dummytc_a{};
    DummyTC dummytc_b{};
    DummyTC dummytc_c{};
    res = __hip_atomic_compare_exchange_weak(&dummytc_a, &dummytc_b, dummytc_c, kMemOrder, kMemOrder,
                                            kMemScope);
  }
)"};

static constexpr auto kBuiltinCompExStrong{R"(
  constexpr int kMemOrder = __ATOMIC_RELAXED;
  constexpr int kMemScope = __HIP_MEMORY_SCOPE_SYSTEM;

  class DummyTC {
  public:
    __device__ DummyTC() {}
    __device__ ~DummyTC() = default;
    __device__ DummyTC(const DummyTC&) = default;
    __device__ DummyTC& operator=(const DummyTC&) = default;
    __device__ DummyTC(DummyTC&&) = default;
    __device__ DummyTC& operator=(DummyTC&&) = default;
  };

  class Dummy {
  public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };

  __global__ void CompareStrongCompileKernel(int* x, int* expected) {
    bool res{false};
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_CONSUME, __ATOMIC_RELAXED,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_CONSUME, __ATOMIC_CONSUME,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_ACQUIRE, __ATOMIC_CONSUME,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_RELEASE, __ATOMIC_RELAXED,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_RELEASE, __ATOMIC_CONSUME,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_ACQ_REL, __ATOMIC_CONSUME,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_CONSUME,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_ACQ_REL,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST,
                                              kMemScope);

    res =
        __hip_atomic_compare_exchange_strong(x, expected, 1, kMemOrder, __ATOMIC_RELEASE, kMemScope);
    res =
        __hip_atomic_compare_exchange_strong(x, expected, 1, kMemOrder, __ATOMIC_ACQ_REL, kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, __ATOMIC_RELAXED, __ATOMIC_SEQ_CST,
                                              kMemScope);
    res = __hip_atomic_compare_exchange_strong(reinterpret_cast<const int*>(x), expected, 1,
                                              kMemOrder, kMemOrder, kMemScope);
    res = __hip_atomic_compare_exchange_strong(*x, expected, 1, kMemOrder, kMemOrder, kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, -1, kMemOrder, kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, 10, kMemOrder, kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, kMemOrder, -1, kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, kMemOrder, 10, kMemScope);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, kMemOrder, kMemOrder, -1);
    res = __hip_atomic_compare_exchange_strong(x, expected, 1, kMemOrder, kMemOrder, 10);

    Dummy dummy_a{};
    Dummy dummy_b{};
    Dummy dummy_c{};
    res = __hip_atomic_compare_exchange_strong(&dummy_a, &dummy_b, dummy_c, kMemOrder, kMemOrder,
                                              kMemScope);
    DummyTC dummytc_a{};
    DummyTC dummytc_b{};
    DummyTC dummytc_c{};
    res = __hip_atomic_compare_exchange_strong(&dummytc_a, &dummytc_b, dummytc_c, kMemOrder,
                                              kMemOrder, kMemScope);
  }
)"};

static constexpr auto kBuiltinExchange{R"(
  constexpr int kMemOrder = __ATOMIC_RELAXED;
  constexpr int kMemScope = __HIP_MEMORY_SCOPE_SYSTEM;

  class DummyTC {
  public:
    __device__ DummyTC() {}
    __device__ ~DummyTC() = default;
    __device__ DummyTC(const DummyTC&) = default;
    __device__ DummyTC& operator=(const DummyTC&) = default;
    __device__ DummyTC(DummyTC&&) = default;
    __device__ DummyTC& operator=(DummyTC&&) = default;
  };

  class Dummy {
  public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };

  __global__ void ExchangeCompileKernel(int* x) {
    int old{};
    old = __hip_atomic_exchange(x, 1, __ATOMIC_RELAXED, kMemScope);
    old = __hip_atomic_exchange(x, 1, __ATOMIC_CONSUME, kMemScope);
    old = __hip_atomic_exchange(x, 1, __ATOMIC_ACQUIRE, kMemScope);
    old = __hip_atomic_exchange(x, 1, __ATOMIC_RELEASE, kMemScope);
    old = __hip_atomic_exchange(x, 1, __ATOMIC_ACQ_REL, kMemScope);
    old = __hip_atomic_exchange(x, 1, __ATOMIC_SEQ_CST, kMemScope);

    old = __hip_atomic_exchange(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
    old = __hip_atomic_exchange(*x, 1, kMemOrder, kMemScope);
    old = __hip_atomic_exchange(x, 1, -1, kMemScope);
    old = __hip_atomic_exchange(x, 1, 10, kMemScope);
    old = __hip_atomic_exchange(x, 1, kMemOrder, -1);
    old = __hip_atomic_exchange(x, 1, kMemOrder, 10);

    Dummy dummy_a{};
    Dummy dummy_b{};
    dummy_b = __hip_atomic_exchange(&dummy_a, dummy_b, kMemOrder, kMemScope);

    DummyTC dummytc_a{};
    DummyTC dummytc_b{};
    dummytc_b = __hip_atomic_exchange(&dummytc_a, dummytc_b, kMemOrder, kMemScope);
  }
)"};

static constexpr auto kBuiltinFetchAdd{R"(
  constexpr int kMemOrder = __ATOMIC_RELAXED;
  constexpr int kMemScope = __HIP_MEMORY_SCOPE_SYSTEM;

  class DummyTC {
  public:
    __device__ DummyTC() {}
    __device__ ~DummyTC() = default;
    __device__ DummyTC(const DummyTC&) = default;
    __device__ DummyTC& operator=(const DummyTC&) = default;
    __device__ DummyTC(DummyTC&&) = default;
    __device__ DummyTC& operator=(DummyTC&&) = default;
  };

  class Dummy {
  public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };

  __global__ void FetchAddCompileKernel(int* x) {
    int old{};
    old = __hip_atomic_fetch_add(x, 1, __ATOMIC_RELAXED, kMemScope);
    old = __hip_atomic_fetch_add(x, 1, __ATOMIC_CONSUME, kMemScope);
    old = __hip_atomic_fetch_add(x, 1, __ATOMIC_ACQUIRE, kMemScope);
    old = __hip_atomic_fetch_add(x, 1, __ATOMIC_RELEASE, kMemScope);
    old = __hip_atomic_fetch_add(x, 1, __ATOMIC_ACQ_REL, kMemScope);
    old = __hip_atomic_fetch_add(x, 1, __ATOMIC_SEQ_CST, kMemScope);

    old = __hip_atomic_fetch_add(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
    old = __hip_atomic_fetch_add(*x, 1, kMemOrder, kMemScope);
    old = __hip_atomic_fetch_add(x, 1, -1, kMemScope);
    old = __hip_atomic_fetch_add(x, 1, 10, kMemScope);
    old = __hip_atomic_fetch_add(x, 1, kMemOrder, -1);
    old = __hip_atomic_fetch_add(x, 1, kMemOrder, 10);

    Dummy dummy{};
    old = __hip_atomic_fetch_add(&dummy, 1, kMemOrder, kMemScope);
  }
)"};

static constexpr auto kBuiltinFetchAnd{R"(
  constexpr int kMemOrder = __ATOMIC_RELAXED;
  constexpr int kMemScope = __HIP_MEMORY_SCOPE_SYSTEM;

  class DummyTC {
  public:
    __device__ DummyTC() {}
    __device__ ~DummyTC() = default;
    __device__ DummyTC(const DummyTC&) = default;
    __device__ DummyTC& operator=(const DummyTC&) = default;
    __device__ DummyTC(DummyTC&&) = default;
    __device__ DummyTC& operator=(DummyTC&&) = default;
  };

  class Dummy {
  public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };

  __global__ void FetchAndCompileKernel(int* x) {
    int old{};
    old = __hip_atomic_fetch_and(x, 1, __ATOMIC_RELAXED, kMemScope);
    old = __hip_atomic_fetch_and(x, 1, __ATOMIC_CONSUME, kMemScope);
    old = __hip_atomic_fetch_and(x, 1, __ATOMIC_ACQUIRE, kMemScope);
    old = __hip_atomic_fetch_and(x, 1, __ATOMIC_RELEASE, kMemScope);
    old = __hip_atomic_fetch_and(x, 1, __ATOMIC_ACQ_REL, kMemScope);
    old = __hip_atomic_fetch_and(x, 1, __ATOMIC_SEQ_CST, kMemScope);

    old = __hip_atomic_fetch_and(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
    old = __hip_atomic_fetch_and(*x, 1, kMemOrder, kMemScope);
    old = __hip_atomic_fetch_and(x, 1, -1, kMemScope);
    old = __hip_atomic_fetch_and(x, 1, 10, kMemScope);
    old = __hip_atomic_fetch_and(x, 1, kMemOrder, -1);
    old = __hip_atomic_fetch_and(x, 1, kMemOrder, 10);

    Dummy dummy{};
    old = __hip_atomic_fetch_and(&dummy, 1, kMemOrder, kMemScope);
    float float_var{1.5f};
    old = __hip_atomic_fetch_and(&float_var, 1, kMemOrder, kMemScope);
    double double_var{1.5};
    old = __hip_atomic_fetch_and(&double_var, 1, kMemOrder, kMemScope);
  }
)"};

static constexpr auto kBuiltinFetchOr{R"(
  constexpr int kMemOrder = __ATOMIC_RELAXED;
  constexpr int kMemScope = __HIP_MEMORY_SCOPE_SYSTEM;

  class DummyTC {
  public:
    __device__ DummyTC() {}
    __device__ ~DummyTC() = default;
    __device__ DummyTC(const DummyTC&) = default;
    __device__ DummyTC& operator=(const DummyTC&) = default;
    __device__ DummyTC(DummyTC&&) = default;
    __device__ DummyTC& operator=(DummyTC&&) = default;
  };

  class Dummy {
  public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };

  __global__ void FetchOrCompileKernel(int* x) {
    int old{};
    old = __hip_atomic_fetch_or(x, 1, __ATOMIC_RELAXED, kMemScope);
    old = __hip_atomic_fetch_or(x, 1, __ATOMIC_CONSUME, kMemScope);
    old = __hip_atomic_fetch_or(x, 1, __ATOMIC_ACQUIRE, kMemScope);
    old = __hip_atomic_fetch_or(x, 1, __ATOMIC_RELEASE, kMemScope);
    old = __hip_atomic_fetch_or(x, 1, __ATOMIC_ACQ_REL, kMemScope);
    old = __hip_atomic_fetch_or(x, 1, __ATOMIC_SEQ_CST, kMemScope);

    old = __hip_atomic_fetch_or(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
    old = __hip_atomic_fetch_or(*x, 1, kMemOrder, kMemScope);
    old = __hip_atomic_fetch_or(x, 1, -1, kMemScope);
    old = __hip_atomic_fetch_or(x, 1, 10, kMemScope);
    old = __hip_atomic_fetch_or(x, 1, kMemOrder, -1);
    old = __hip_atomic_fetch_or(x, 1, kMemOrder, 10);

    Dummy dummy{};
    old = __hip_atomic_fetch_or(&dummy, 1, kMemOrder, kMemScope);
    float float_var{1.5f};
    old = __hip_atomic_fetch_or(&float_var, 1, kMemOrder, kMemScope);
    double double_var{1.5};
    old = __hip_atomic_fetch_or(&double_var, 1, kMemOrder, kMemScope);
  }
)"};

static auto constexpr kBuiltinFetchXor{R"(
  constexpr int kMemOrder = __ATOMIC_RELAXED;
  constexpr int kMemScope = __HIP_MEMORY_SCOPE_SYSTEM;

  class DummyTC {
  public:
    __device__ DummyTC() {}
    __device__ ~DummyTC() = default;
    __device__ DummyTC(const DummyTC&) = default;
    __device__ DummyTC& operator=(const DummyTC&) = default;
    __device__ DummyTC(DummyTC&&) = default;
    __device__ DummyTC& operator=(DummyTC&&) = default;
  };

  class Dummy {
  public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };

  __global__ void FetchXorCompileKernel(int* x) {
    int old{};
    old = __hip_atomic_fetch_xor(x, 1, __ATOMIC_RELAXED, kMemScope);
    old = __hip_atomic_fetch_xor(x, 1, __ATOMIC_CONSUME, kMemScope);
    old = __hip_atomic_fetch_xor(x, 1, __ATOMIC_ACQUIRE, kMemScope);
    old = __hip_atomic_fetch_xor(x, 1, __ATOMIC_RELEASE, kMemScope);
    old = __hip_atomic_fetch_xor(x, 1, __ATOMIC_ACQ_REL, kMemScope);
    old = __hip_atomic_fetch_xor(x, 1, __ATOMIC_SEQ_CST, kMemScope);

    old = __hip_atomic_fetch_xor(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
    old = __hip_atomic_fetch_xor(*x, 1, kMemOrder, kMemScope);
    old = __hip_atomic_fetch_xor(x, 1, -1, kMemScope);
    old = __hip_atomic_fetch_xor(x, 1, 10, kMemScope);
    old = __hip_atomic_fetch_xor(x, 1, kMemOrder, -1);
    old = __hip_atomic_fetch_xor(x, 1, kMemOrder, 10);

    Dummy dummy{};
    old = __hip_atomic_fetch_xor(&dummy, 1, kMemOrder, kMemScope);
    float float_var{1.5f};
    old = __hip_atomic_fetch_xor(&float_var, 1, kMemOrder, kMemScope);
    double double_var{1.5};
    old = __hip_atomic_fetch_xor(&double_var, 1, kMemOrder, kMemScope);
  }
)"};

static constexpr auto kBuiltinFetchMax{R"(
  constexpr int kMemOrder = __ATOMIC_RELAXED;
  constexpr int kMemScope = __HIP_MEMORY_SCOPE_SYSTEM;

  class DummyTC {
  public:
    __device__ DummyTC() {}
    __device__ ~DummyTC() = default;
    __device__ DummyTC(const DummyTC&) = default;
    __device__ DummyTC& operator=(const DummyTC&) = default;
    __device__ DummyTC(DummyTC&&) = default;
    __device__ DummyTC& operator=(DummyTC&&) = default;
  };

  class Dummy {
  public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };

  __global__ void FetchMaxCompileKernel(int* x) {
    int old{};
    old = __hip_atomic_fetch_max(x, 1, __ATOMIC_RELAXED, kMemScope);
    old = __hip_atomic_fetch_max(x, 1, __ATOMIC_CONSUME, kMemScope);
    old = __hip_atomic_fetch_max(x, 1, __ATOMIC_ACQUIRE, kMemScope);
    old = __hip_atomic_fetch_max(x, 1, __ATOMIC_RELEASE, kMemScope);
    old = __hip_atomic_fetch_max(x, 1, __ATOMIC_ACQ_REL, kMemScope);
    old = __hip_atomic_fetch_max(x, 1, __ATOMIC_SEQ_CST, kMemScope);

    old = __hip_atomic_fetch_max(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
    old = __hip_atomic_fetch_max(*x, 1, kMemOrder, kMemScope);
    old = __hip_atomic_fetch_max(x, 1, -1, kMemScope);
    old = __hip_atomic_fetch_max(x, 1, 10, kMemScope);
    old = __hip_atomic_fetch_max(x, 1, kMemOrder, -1);
    old = __hip_atomic_fetch_max(x, 1, kMemOrder, 10);

    Dummy dummy{};
    old = __hip_atomic_fetch_max(&dummy, 1, kMemOrder, kMemScope);
  }
)"};

static constexpr auto kBuiltinFetchMin{R"(
  constexpr int kMemOrder = __ATOMIC_RELAXED;
  constexpr int kMemScope = __HIP_MEMORY_SCOPE_SYSTEM;

  class DummyTC {
  public:
    __device__ DummyTC() {}
    __device__ ~DummyTC() = default;
    __device__ DummyTC(const DummyTC&) = default;
    __device__ DummyTC& operator=(const DummyTC&) = default;
    __device__ DummyTC(DummyTC&&) = default;
    __device__ DummyTC& operator=(DummyTC&&) = default;
  };

  class Dummy {
  public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };

  __global__ void FetchMinCompileKernel(int* x) {
    int old{};
    old = __hip_atomic_fetch_min(x, 1, __ATOMIC_RELAXED, kMemScope);
    old = __hip_atomic_fetch_min(x, 1, __ATOMIC_CONSUME, kMemScope);
    old = __hip_atomic_fetch_min(x, 1, __ATOMIC_ACQUIRE, kMemScope);
    old = __hip_atomic_fetch_min(x, 1, __ATOMIC_RELEASE, kMemScope);
    old = __hip_atomic_fetch_min(x, 1, __ATOMIC_ACQ_REL, kMemScope);
    old = __hip_atomic_fetch_min(x, 1, __ATOMIC_SEQ_CST, kMemScope);

    old = __hip_atomic_fetch_min(reinterpret_cast<const int*>(x), 1, kMemOrder, kMemScope);
    old = __hip_atomic_fetch_min(*x, 1, kMemOrder, kMemScope);
    old = __hip_atomic_fetch_min(x, 1, -1, kMemScope);
    old = __hip_atomic_fetch_min(x, 1, 10, kMemScope);
    old = __hip_atomic_fetch_min(x, 1, kMemOrder, -1);
    old = __hip_atomic_fetch_min(x, 1, kMemOrder, 10);

    Dummy dummy{};
    old = __hip_atomic_fetch_min(&dummy, 1, kMemOrder, kMemScope);
  }
)"};
