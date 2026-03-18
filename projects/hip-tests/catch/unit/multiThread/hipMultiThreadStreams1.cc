/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>


int p_iters = 10;
int N = 8000000;
unsigned blocksPerCU = 6;
unsigned threadsPerBlock = 256;

//---
// Test simple H2D copies and back.
// Designed to stress a small number of simple smoke tests

template <typename T = float, class P = HipTest::Unpinned, class C = HipTest::Memcpy>
void simpleVectorAdd(size_t numElements, int iters, hipStream_t stream) {
  using HipTest::MemTraits;
  size_t Nbytes = numElements * sizeof(T);

  T *A_d, *B_d, *C_d;
  T *A_h, *B_h, *C_h;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, P::isPinned);
  for (size_t i = 0; i < numElements; i++) {
    A_h[i] = 1000.0f;
    B_h[i] = 2000.0f;
    C_h[i] = -1;
  }

  MemTraits<C>::Copy(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream);
  MemTraits<C>::Copy(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream);
  MemTraits<C>::Copy(C_d, C_h, Nbytes, hipMemcpyHostToDevice, stream);
  HIPCHECK(hipDeviceSynchronize());

  for (size_t i = 0; i < numElements; i++) {
    A_h[i] = 1.0f;
    B_h[i] = 2.0f;
    C_h[i] = -1;
  }


  for (int i = 0; i < iters; i++) {
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);

    MemTraits<C>::Copy(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream);
    MemTraits<C>::Copy(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream);

    hipLaunchKernelGGL(HipTest::vectorADDReverse, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                       static_cast<const T*>(A_d), static_cast<const T*>(B_d), C_d, numElements);
    HIP_CHECK(hipGetLastError());

    MemTraits<C>::Copy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream);

    HIPCHECK(hipDeviceSynchronize());

    HipTest::checkVectorADD(A_h, B_h, C_h, numElements);
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, P::isPinned);
  HIPCHECK(hipDeviceSynchronize());
}

template <typename T, class C>
void test_multiThread_1(hipStream_t stream0, hipStream_t stream1, bool serialize) {
  size_t numElements = N;

  // Test 2 threads operating on same stream:
  std::thread t1(simpleVectorAdd<T, HipTest::Pinned, C>, numElements, p_iters /*iters*/, stream0);
  if (serialize) {
    t1.join();
  }
  std::thread t2(simpleVectorAdd<T, HipTest::Pinned, C>, numElements, p_iters /*iters*/, stream1);
  if (serialize) {
    t2.join();
  }

  if (!serialize) {
    t1.join();
    t2.join();
  }

  HIPCHECK(hipDeviceSynchronize());
};

TEST_CASE(Unit_hipMultiThreadStreams1_AsyncSync) {
  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  simpleVectorAdd<float, HipTest::Pinned, HipTest::MemcpyAsync>(N /*mb*/, 10 /*iters*/, stream);
  simpleVectorAdd<float, HipTest::Pinned, HipTest::Memcpy>(N /*mb*/, 10 /*iters*/, stream);

  HIPCHECK(hipStreamDestroy(stream));
}

TEST_CASE(Unit_hipMultiThreadStreams1_AsyncAsync) {
  hipStream_t stream0, stream1;
  HIPCHECK(hipStreamCreate(&stream0));
  HIPCHECK(hipStreamCreate(&stream1));

  // Easy tests to verify the test works - these don't allow overlap between the threads:
  test_multiThread_1<float, HipTest::MemcpyAsync>(NULL, NULL, true);
  test_multiThread_1<float, HipTest::MemcpyAsync>(stream0, stream1, true);

  HIPCHECK(hipStreamDestroy(stream0));
  HIPCHECK(hipStreamDestroy(stream1));
}
TEST_CASE(Unit_hipMultiThreadStreams1_AsyncSame) {
  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  // test_multiThread_1<float, HipTest::MemcpyAsync> ("Multithread with NULL stream", NULL,
  // NULL, false); test_multiThread_1<float, HipTest::MemcpyAsync> ("Multithread with two
  // streams", stream0, stream1, false);
  test_multiThread_1<float, HipTest::MemcpyAsync>(stream, stream, false);

  HIPCHECK(hipStreamDestroy(stream));
}
