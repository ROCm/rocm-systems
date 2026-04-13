/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
 #include <hip_test_common.hh>
 #include <hip_test_defgroups.hh>
 #include <resource_guards.hh>

 #include "memcpyBatchAsync_common.hh"
 /**
  * @addtogroup hipMemcpyBatchAsync hipMemcpyBatchAsync
  * @{
  * @ingroup MemoryTest
  * `hipError_t hipMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes,
  size_t count, hipMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs,
                                size_t* failIdx, hipStream_t stream __dparm(0))`
  -
  * Perform Batch of 1D copies.
  *
  * Multi-GPU peer tests: catch/unit/memory/hipMemcpyBatchAsync_p2p.cc
  */
 /**
  * Test Description
  * ------------------------
  * - Test case to verify the 1D batch memory copy.
  * 1. Create Array of device pointers(Src, Dst).
  * 2. Set the MemcpyBatch params. As of now no support for memcpy Attributes.
  * 3. Perform batch memcpy operation from deviceptr to deviceptr.
  * 4. Validate data on host.
  * Test source
  * ------------------------
  * - catch/unit/memory/hipMemcpyBatchAsync.cc
  * Test requirements
  * ------------------------
  *  - HIP_VERSION >= 7.1
  */
 #if HT_AMD
 
 TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_D2D_Functional, char, int, float) {
   const size_t count = 2;
   size_t numAttrs = 0;
   const size_t arrSize = 4096;
   const size_t size = 4096 * sizeof(TestType);
   hipStream_t stream;
   HIP_CHECK(hipStreamCreate(&stream));
   const auto values = get_test_values<TestType>();
   const TestType val1 = values.first;
   const TestType val2 = values.second;
 
   // Allocate buffers for pointer-ptr copy
   void *srcPtr[count], *dstPtr[count];
   std::vector<std::vector<TestType>> hostPtr1(count, std::vector<TestType>(arrSize, val1));
   std::vector<std::vector<TestType>> hostPtr2(count, std::vector<TestType>(arrSize, val2));
   size_t sizes[2];
   size_t attrsIdxs[1];
   for (int i = 0; i < count; i++) {
     HIP_CHECK(hipMalloc(&srcPtr[i], size));
     HIP_CHECK(hipMalloc(&dstPtr[i], size));
     HIP_CHECK(hipMemcpy(srcPtr[i], hostPtr2[i].data(), size, hipMemcpyHostToDevice));
     sizes[i] = size;
   }
   attrsIdxs[0] = 0;
   size_t failIdx;
 
   HIP_CHECK(hipMemcpyBatchAsync(dstPtr, srcPtr, sizes, count, nullptr, attrsIdxs, numAttrs,
                                 &failIdx, stream));
   HIP_CHECK(hipStreamSynchronize(stream));
   // validation
   for (int i = 0; i < count; i++) {
     HIP_CHECK(hipMemcpy(hostPtr1[i].data(), dstPtr[i], size, hipMemcpyDeviceToHost));
     for (int j = 0; j < arrSize; j++) {
       INFO("Array FAILURE at Index: " << i << " " << j << "\nval : " << hostPtr1[i][j]);
       REQUIRE(hostPtr1[i][j] == val2);
     }
   }
   // Clean up
   for (int i = 0; i < count; i++) {
     HIP_CHECK(hipFree(srcPtr[i]));
     HIP_CHECK(hipFree(dstPtr[i]));
   }
   HIP_CHECK(hipStreamDestroy(stream));
 }
 
 /**
  * Test Description
  * ------------------------
  * - Test case to verify the 1D batch memory copy.
  * 1. Create Array of device pointers(Src, Dst).
  * 2. Set the MemcpyBatch params. As of now no support for memcpy Attributes.
  * 3. Perform batch memcpy operation From Host to Device.
  * 4. Validate data on host.
  * Test source
  * ------------------------
  * - catch/unit/memory/hipMemcpyBatchAsync.cc
  * Test requirements
  * ------------------------
  *  - HIP_VERSION >= 7.1
  */
 TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_H2D_Functional, char, int, float) {
   const size_t count = 2;
   size_t numAttrs = 0;
   const size_t arrSize = 4096;
   const size_t size = 4096 * sizeof(TestType);
   hipStream_t stream;
   HIP_CHECK(hipStreamCreate(&stream));
   const auto values = get_test_values<TestType>();
   const TestType val1 = values.first;
   const TestType val2 = values.second;
 
   // Allocate buffers for pointer-ptr copy
   void *hostSrcPtr[count], *dstPtr[count];
   std::vector<std::vector<TestType>> hostPtr(count, std::vector<TestType>(arrSize, val2));
   std::array<TestType, arrSize> arr;
   arr.fill(val1);
   size_t sizes[2];
   size_t attrsIdxs[1];
   for (int i = 0; i < count; i++) {
     hostSrcPtr[i] = arr.data();
     HIP_CHECK(hipMalloc(&dstPtr[i], size));
     sizes[i] = size;
   }
   attrsIdxs[0] = 0;
   size_t failIdx;
 
   HIP_CHECK(hipMemcpyBatchAsync(dstPtr, hostSrcPtr, sizes, count, nullptr, attrsIdxs, numAttrs,
                                 &failIdx, stream));
   HIP_CHECK(hipStreamSynchronize(stream));
   // validation
   for (int i = 0; i < count; i++) {
     HIP_CHECK(hipMemcpy(hostPtr[i].data(), dstPtr[i], size, hipMemcpyDeviceToHost));
     for (int j = 0; j < arrSize; j++) {
       INFO("Array FAILURE at Index: " << i << " " << j << "\nval : " << hostPtr[i][j]);
       REQUIRE(hostPtr[i][j] == val1);
     }
   }
   // Clean up
   for (int i = 0; i < count; i++) {
     HIP_CHECK(hipFree(dstPtr[i]));
   }
   HIP_CHECK(hipStreamDestroy(stream));
 }
 
 /**
  * Test Description
  * ------------------------
  * - Test case to verify the 1D batch memory copy.
  * 1. Create Array of device pointers(Src, Dst).
  * 2. Set the MemcpyBatch params. As of now no support for memcpy Attributes.
  * 3. Perform batch memcpy operation From Device to Host.
  * 4. Validate data on host.
  * Test source
  * ------------------------
  * - catch/unit/memory/hipMemcpyBatchAsync.cc
  * Test requirements
  * ------------------------
  *  - HIP_VERSION >= 7.1
  */
 TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_D2H_Functional, char, int, float) {
   const size_t count = 2;
   size_t numAttrs = 0;
   const size_t arrSize = 4096;
   const size_t size = 4096 * sizeof(TestType);
   hipStream_t stream;
   HIP_CHECK(hipStreamCreate(&stream));
   const auto values = get_test_values<TestType>();
   const TestType val1 = values.first;
   const TestType val2 = values.second;
 
   // Allocate buffers for pointer-ptr copy
   TestType* hostDstPtr[count];
   void* deviceSrcPtr[count];
   std::vector<std::vector<TestType>> hostPtr(count, std::vector<TestType>(arrSize, val1));
   std::array<TestType, arrSize> arr;
   arr.fill(val2);
   size_t sizes[2];
   size_t attrsIdxs[1];
   for (int i = 0; i < count; i++) {
     hostDstPtr[i] = arr.data();
     HIP_CHECK(hipMalloc(&deviceSrcPtr[i], size));
     HIP_CHECK(hipMemcpy(deviceSrcPtr[i], hostPtr[i].data(), size, hipMemcpyHostToDevice));
     sizes[i] = size;
   }
   attrsIdxs[0] = 0;
   size_t failIdx;
 
   HIP_CHECK(hipMemcpyBatchAsync(reinterpret_cast<void**>(hostDstPtr), deviceSrcPtr, sizes, count,
                                 nullptr, attrsIdxs, numAttrs, &failIdx, stream));
   HIP_CHECK(hipStreamSynchronize(stream));
   // validation
   for (int i = 0; i < count; i++) {
     for (int j = 0; j < arrSize; j++) {
       INFO("Array FAILURE at Index: " << i << " " << j << "\nval : " << hostDstPtr[i][j]);
       REQUIRE(hostDstPtr[i][j] == val1);
     }
   }
   // Clean up
   for (int i = 0; i < count; i++) {
     HIP_CHECK(hipFree(deviceSrcPtr[i]));
   }
   HIP_CHECK(hipStreamDestroy(stream));
 }
 
 /**
  * Test Description
  * ------------------------
  * - Test case to verify the 1D batch memory copy.
  * 1. Create Array of device pointers(Src, Dst).
  * 2. Set the MemcpyBatch params. As of now no support for memcpy Attributes.
  * 3. Perform batch memcpy operation From Host to Host.
  * 4. Validate data on host.
  * Test source
  * ------------------------
  * - catch/unit/memory/hipMemcpyBatchAsync.cc
  * Test requirements
  * ------------------------
  *  - HIP_VERSION >= 7.1
  */
 TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_H2H_Functional, char, int, float) {
   const size_t count = 2;
   size_t numAttrs = 0;
   const size_t arrSize = 4096;
   hipStream_t stream;
   HIP_CHECK(hipStreamCreate(&stream));
   const auto values = get_test_values<TestType>();
   const TestType val1 = values.first;
   const TestType val2 = values.second;
 
   // Allocate buffers for pointer-ptr copy
   TestType *hostDstPtr[count], *hostSrcPtr[count];
   std::array<TestType, arrSize> arr1, arr2;
   arr1.fill(val1);
   arr2.fill(val2);
   size_t sizes[2];
   size_t attrsIdxs[1];
   for (int i = 0; i < count; i++) {
     hostDstPtr[i] = arr1.data();
     hostSrcPtr[i] = arr2.data();
     sizes[i] = arrSize * sizeof(TestType);
   }
   attrsIdxs[0] = 0;
   size_t failIdx;
 
   HIP_CHECK(hipMemcpyBatchAsync(reinterpret_cast<void**>(hostDstPtr),
                                 reinterpret_cast<void**>(hostSrcPtr), sizes, count, nullptr,
                                 attrsIdxs, numAttrs, &failIdx, stream));
   HIP_CHECK(hipStreamSynchronize(stream));
   // validation
   for (int i = 0; i < count; i++) {
     for (int j = 0; j < arrSize; j++) {
       INFO("Array FAILURE at Index: " << i << " " << j << "\nval : " << hostDstPtr[i][j]);
       REQUIRE(hostDstPtr[i][j] == val2);
     }
   }
   // Clean up
   HIP_CHECK(hipStreamDestroy(stream));
 }
 
 template <typename TestType> struct SwapCopyTest {
   SwapCopyTest(int count, int num_elements, LinearAllocs allocTypeB, LinearAllocs allocTypeA,
                hipError_t expectedError)
       : count{count},
         num_elements{num_elements},
         size_in_bytes{num_elements * sizeof(TestType)},
         allocTypeB(allocTypeB),
         allocTypeA(allocTypeA),
         initialValuesA(count,
                        std::vector<TestType>(num_elements, get_test_values<TestType>().first)),
         initialValuesB(count,
                        std::vector<TestType>(num_elements, get_test_values<TestType>().second)),
         swapPtrsA(count),
         swapPtrsB(count),
         expectedReturnValue(expectedError) {}
 
   void runTest() {
     initializeMem();
     execute();
     if (expectedReturnValue == hipSuccess) {
       verifyResults();
     }
     freeMem();
   }
 
  private:
   void initializeMem() {
     HIP_CHECK(hipStreamCreate(&stream));
 
     for (int i = 0; i < count; ++i) {
       LinearAllocGuard<TestType> allocB(allocTypeB, size_in_bytes);
       swapPtrsB[i] = allocB.ptr();
       allocations.push_back(std::move(allocB));
       hipMemcpyKind fillKindB =
           allocTypeB == LinearAllocs::hipMalloc ? hipMemcpyHostToDevice : hipMemcpyHostToHost;
       HIP_CHECK(hipMemcpy(swapPtrsB[i], initialValuesB[i].data(), size_in_bytes, fillKindB));
 
       LinearAllocGuard<TestType> allocA(allocTypeA, size_in_bytes);
       swapPtrsA[i] = allocA.ptr();
       allocations.push_back(std::move(allocA));
       hipMemcpyKind fillKindA =
           allocTypeA == LinearAllocs::hipMalloc ? hipMemcpyHostToDevice : hipMemcpyHostToHost;
       HIP_CHECK(hipMemcpy(swapPtrsA[i], initialValuesA[i].data(), size_in_bytes, fillKindA));
     }
   }
 
   void execute() {
     std::vector<size_t> sizes(count, size_in_bytes);
 
     const size_t num_attributes = 1;
 
     std::vector<hipMemcpyAttributes> attributes(num_attributes);
     attributes[0].flags = hipMemcpyFlagExtOpSwap;
     attributes[0].srcAccessOrder = hipMemcpySrcAccessOrderStream;
 
     std::vector<size_t> attributes_indexes(num_attributes);
     attributes_indexes[0] = 0;
 
     size_t fail_index;
 
     HIP_CHECK_ERROR(hipMemcpyBatchAsync(swapPtrsA.data(), swapPtrsB.data(), sizes.data(), count,
                                         attributes.data(), attributes_indexes.data(),
                                         num_attributes, &fail_index, stream),
                     expectedReturnValue);
     if (expectedReturnValue != hipSuccess) return;
 
     HIP_CHECK(hipStreamSynchronize(stream));
   }
 
   void verifyResults() {
     std::vector<std::vector<TestType>> hostPtrAOut(count, std::vector<TestType>(num_elements));
     std::vector<std::vector<TestType>> hostPtrBOut(count, std::vector<TestType>(num_elements));
     for (int i = 0; i < count; i++) {
       hipMemcpyKind readKindA =
           allocTypeA == LinearAllocs::hipMalloc ? hipMemcpyDeviceToHost : hipMemcpyHostToHost;
       HIP_CHECK(hipMemcpy(hostPtrAOut[i].data(), swapPtrsA[i], size_in_bytes, readKindA));
 
       hipMemcpyKind readKindB =
           allocTypeB == LinearAllocs::hipMalloc ? hipMemcpyDeviceToHost : hipMemcpyHostToHost;
       HIP_CHECK(hipMemcpy(hostPtrBOut[i].data(), swapPtrsB[i], size_in_bytes, readKindB));
       for (int j = 0; j < num_elements; j++) {
         REQUIRE(hostPtrAOut[i][j] == initialValuesB[i][j]);
         REQUIRE(hostPtrBOut[i][j] == initialValuesA[i][j]);
       }
     }
   }
 
   void freeMem() { HIP_CHECK(hipStreamDestroy(stream)); }
 
   const int count;
   const int num_elements;
   const size_t size_in_bytes;
 
   const LinearAllocs allocTypeB;
   const LinearAllocs allocTypeA;
 
   std::vector<std::vector<TestType>> initialValuesA;
   std::vector<std::vector<TestType>> initialValuesB;
 
   std::vector<LinearAllocGuard<TestType>> allocations;
 
   /*Stream where the memcpy will be enqueued to. */
   hipStream_t stream;
 
   /* Pointers that will be used as src and dst for the copy. */
   std::vector<void*> swapPtrsA;
   std::vector<void*> swapPtrsB;
 
   hipError_t expectedReturnValue;
 };



 TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_Swap, char, int, float) {
   const size_t count = GENERATE(2, 3, 8);
   const size_t num_elements = 4096;
   const LinearAllocs allocTypeSrc =
       GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc,
                LinearAllocs::hipMalloc);
   const LinearAllocs allocTypeDst =
       GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc,
                LinearAllocs::hipMalloc);
   CAPTURE(count, allocTypeSrc, allocTypeDst);

   hipError_t expectedError = getSwapExpectedReturn(allocTypeSrc, allocTypeDst);
   SwapCopyTest<TestType> test(count, num_elements, allocTypeSrc, allocTypeDst,
                               expectedError);
   test.runTest();
 }

 template <typename TestType> struct MulticastCopyTest {
   MulticastCopyTest(int count, int num_elements, LinearAllocs srcAllocType,
                     LinearAllocs dstAllocType, hipError_t expectedError)
       : count{count},
         num_elements{num_elements},
         size_in_bytes{num_elements * sizeof(TestType)},
         srcAllocType(srcAllocType),
         dstAllocType(dstAllocType),
         initialValues(num_elements, get_test_values<TestType>().first),
         srcPtrs(count),
         dstPtrs(count),
         expectedReturnValue(expectedError) {}
 
   void runTest() {
     initializeMem();
     execute();
     if (expectedReturnValue == hipSuccess) {
       verifyResults();
     }
     freeMem();
   }
 
  private:
   void initializeMem() {
     HIP_CHECK(hipStreamCreate(&stream));
 
     LinearAllocGuard<TestType> srcAlloc(srcAllocType, size_in_bytes);
     srcMem = srcAlloc.ptr();
     hipMemcpyKind fillKind =
         srcAllocType == LinearAllocs::hipMalloc ? hipMemcpyHostToDevice : hipMemcpyHostToHost;
     HIP_CHECK(hipMemcpy(srcMem, initialValues.data(), size_in_bytes, fillKind));
     allocations.push_back(std::move(srcAlloc));
 
     for (int i = 0; i < count; ++i) {
       srcPtrs[i] = srcMem;
 
       LinearAllocGuard<TestType> dstAlloc(dstAllocType, size_in_bytes);
       dstPtrs[i] = dstAlloc.ptr();
       allocations.push_back(std::move(dstAlloc));
     }
   }
 
   void execute() {
     std::vector<size_t> sizes(count, size_in_bytes);
 
     const size_t num_attributes = 1;
 
     std::vector<hipMemcpyAttributes> attributes(num_attributes);
     attributes[0].srcAccessOrder = hipMemcpySrcAccessOrderStream;
 
     std::vector<size_t> attributes_indexes(num_attributes);
     attributes_indexes[0] = 0;
 
     size_t fail_index;
     HIP_CHECK_ERROR(
         hipMemcpyBatchAsync(dstPtrs.data(), srcPtrs.data(), sizes.data(), count, attributes.data(),
                             attributes_indexes.data(), num_attributes, &fail_index, stream),
         expectedReturnValue);
     if (expectedReturnValue != hipSuccess) return;
 
     HIP_CHECK(hipStreamSynchronize(stream));
   }
 
   void verifyResults() {
     std::vector<std::vector<TestType>> hostPtrOut(count, std::vector<TestType>(num_elements));
     for (int i = 0; i < count; i++) {
       hipMemcpyKind readKind =
           dstAllocType == LinearAllocs::hipMalloc ? hipMemcpyDeviceToHost : hipMemcpyHostToHost;
       HIP_CHECK(hipMemcpy(hostPtrOut[i].data(), dstPtrs[i], size_in_bytes, readKind));
       for (int j = 0; j < num_elements; j++) {
         REQUIRE(hostPtrOut[i][j] == initialValues[j]);
       }
     }
   }
 
   void freeMem() { HIP_CHECK(hipStreamDestroy(stream)); }
 
   const int count;
   const int num_elements;
   const size_t size_in_bytes;
 
   const LinearAllocs srcAllocType;
   const LinearAllocs dstAllocType;
 
   /*Host memory used to assign initial values to device memory. */
   std::vector<TestType> initialValues;
 
   std::vector<LinearAllocGuard<TestType>> allocations;
 
   /*Stream where the memcpy will be enqueued to. */
   hipStream_t stream;
 
   /* Memory that will be used as src for the copy. Same src will be used for multiple copies*/
   void* srcMem;
 
   /* Pointers that will be used as src and dst for the copy. */
   std::vector<void*> srcPtrs;
   std::vector<void*> dstPtrs;
 
   hipError_t expectedReturnValue;
 };
 
 /**
  * Batched multicast copy: one shared source, multiple destinations.
  */
 TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_Multicast, char, int, float) {
   const size_t count = GENERATE(2, 3, 8);
   const size_t num_elements = 4096;
   const LinearAllocs allocTypeSrc =
       GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc);
   const LinearAllocs allocTypeDst =
       GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc);
   CAPTURE(count, allocTypeSrc, allocTypeDst);
 
   hipError_t expectedError = hipSuccess;
   MulticastCopyTest<TestType> test(count, num_elements, allocTypeSrc, allocTypeDst, expectedError);
   test.runTest();
 }
 
 /**
  * Batched multicast copy with large per-operation size.
  */
 TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_Multicast_Large, char, int, float) {
   const size_t count = GENERATE(2, 3, 8);
   const LinearAllocs allocTypeSrc =
       GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc);
   const LinearAllocs allocTypeDst =
       GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc);
   const size_t num_elements = 1024 * 1024;
   CAPTURE(count, allocTypeSrc, allocTypeDst);
 
   hipError_t expectedError = hipSuccess;
   MulticastCopyTest<TestType> test(count, num_elements, allocTypeSrc, allocTypeDst, expectedError);
   test.runTest();
 }
 
 /**
  * Batch D2D copies where most entries share one source (multicast-friendly) but one entry uses a
  * different source, e.g. srcA, srcA, srcA, srcB, srcA, srcA, srcA. Validates correctness when the
  * batch cannot be lowered to a single multicast operation.
  */
 template <typename TestType> struct MixedMulticastSourcesBatchTest {
   void run() {
     constexpr int k_count = 7;
     const int num_elements = 4096;
     const size_t size_in_bytes = static_cast<size_t>(num_elements) * sizeof(TestType);
     const auto values = get_test_values<TestType>();
 
     std::vector<TestType> pattern_a(num_elements, values.first);
     std::vector<TestType> pattern_b(num_elements, values.second);
 
     LinearAllocGuard<TestType> srcAllocA(LinearAllocs::hipMalloc, size_in_bytes);
     LinearAllocGuard<TestType> srcAllocB(LinearAllocs::hipMalloc, size_in_bytes);
     void* const srcMemA = srcAllocA.ptr();
     void* const srcMemB = srcAllocB.ptr();
 
     std::vector<LinearAllocGuard<TestType>> allocations;
     allocations.push_back(std::move(srcAllocA));
     allocations.push_back(std::move(srcAllocB));
 
     HIP_CHECK(hipMemcpy(srcMemA, pattern_a.data(), size_in_bytes, hipMemcpyHostToDevice));
     HIP_CHECK(hipMemcpy(srcMemB, pattern_b.data(), size_in_bytes, hipMemcpyHostToDevice));
 
     std::vector<void*> dst_ptrs;
     for (int i = 0; i < k_count; ++i) {
       LinearAllocGuard<TestType> dstAlloc(LinearAllocs::hipMalloc, size_in_bytes);
       HIP_CHECK(hipMemset(dstAlloc.ptr(), 0, size_in_bytes));
       dst_ptrs.push_back(dstAlloc.ptr());
       allocations.push_back(std::move(dstAlloc));
     }
 
     std::vector<void*> src_ptrs = {srcMemA, srcMemA, srcMemA, srcMemB, srcMemA, srcMemA, srcMemA};
     std::vector<size_t> sizes(k_count, size_in_bytes);
 
     hipMemcpyAttributes attr{};
     attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
     attr.flags = hipMemcpyFlagDefault;
     std::vector<hipMemcpyAttributes> attrs = {attr};
     std::vector<size_t> attrs_idxs = {0};
     constexpr size_t k_num_attrs = 1;
 
     hipStream_t stream{};
     HIP_CHECK(hipStreamCreate(&stream));
 
     size_t fail_index = 0;
     HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), k_count,
                                   attrs.data(), attrs_idxs.data(), k_num_attrs, &fail_index,
                                   stream));
     HIP_CHECK(hipStreamSynchronize(stream));
 
     std::vector<TestType> host_out(num_elements);
     for (int i = 0; i < k_count; ++i) {
       HIP_CHECK(hipMemcpy(host_out.data(), dst_ptrs[i], size_in_bytes, hipMemcpyDeviceToHost));
       const std::vector<TestType>& expected = (i == 3) ? pattern_b : pattern_a;
       for (int j = 0; j < num_elements; ++j) {
         REQUIRE(host_out[j] == expected[j]);
       }
     }
 
     HIP_CHECK(hipStreamDestroy(stream));
   }
 };
 
 /**
  * Batched D2D copy where one entry uses a different device source than the others (multicast
  * grouping interrupted mid-batch).
  */
 TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_D2D_MixedMulticastSources, char, int, float) {
   MixedMulticastSourcesBatchTest<TestType> test;
   test.run();
 }
 #endif
 
 /**
  * Test Description
  * ------------------------
  * - Test case to verify the negative cases of hipMemcpyBatchAsync.
  * 1. Dst Array as nullptr.
  * 2. Src Array as nullptr.
  * 3. Operations Count as 0.
  * 4. Num of attributes as 0.
  * 5. Sizes Array as nullptr.
  * 6. Attr Array as nullptr.
  * 7. AttrsIdxs Array as nullptr.
  * Test source
  * ------------------------
  * - catch/unit/memory/hipMemcpyBatchAsync.cc
  * Test requirements
  * ------------------------
  *  - HIP_VERSION >= 7.1
  */
 TEST_CASE("Unit_hipMemcpyBatchAsync_NegativeTsts") {
   const size_t count = 2;
   size_t numAttrs = 0;
   size_t sizes[2];
   size_t attrsIdxs[1];
   const size_t size = 4096 * sizeof(char);
   hipStream_t stream = NULL;
   HIP_CHECK(hipStreamCreate(&stream));
   void *srcPtr[count], *dstPtr[count];
   for (int i = 0; i < count; i++) {
     HIP_CHECK(hipMalloc(&srcPtr[i], size));
     HIP_CHECK(hipMalloc(&dstPtr[i], size));
     sizes[i] = size;
   }
 
   attrsIdxs[0] = 0;
   size_t failIdx;
   SECTION("Dst Array as nullptr") {
     HIP_CHECK_ERROR(hipMemcpyBatchAsync(nullptr, srcPtr, sizes, count, nullptr, attrsIdxs, numAttrs,
                                         &failIdx, stream),
                     hipErrorInvalidValue);
   }
   SECTION("Src Array as nullptr") {
     HIP_CHECK_ERROR(hipMemcpyBatchAsync(dstPtr, nullptr, sizes, count, nullptr, attrsIdxs, numAttrs,
                                         &failIdx, stream),
                     hipErrorInvalidValue);
   }
   SECTION("Count as zero") {
     HIP_CHECK_ERROR(hipMemcpyBatchAsync(dstPtr, srcPtr, sizes, 0, nullptr, attrsIdxs, numAttrs,
                                         &failIdx, stream),
                     hipErrorInvalidValue);
   }
   SECTION("sizes Array as nullptr") {
     HIP_CHECK_ERROR(hipMemcpyBatchAsync(dstPtr, srcPtr, nullptr, count, nullptr, attrsIdxs,
                                         numAttrs, &failIdx, stream),
                     hipErrorInvalidValue);
   }
 #if 0  // Enable these tests when support for memcpy attributes is enabled.
    SECTION("Number of Attributes as zero") {
      HIP_CHECK_ERROR(
          hipMemcpyBatchAsync(dstPtr, srcPtr, sizes, count, attr, attrsIdxs, 0, &failIdx, stream),
          hipErrorInvalidValue);
    }
    SECTION("Attr Array as nullptr") {
      HIP_CHECK_ERROR(hipMemcpyBatchAsync(dstPtr, srcPtr, sizes, count, nullptr, attrsIdxs, numAttrs,
                                          &failIdx, stream),
                      hipErrorInvalidValue);
    }
  
    SECTION("attrsIdxs Array as nullptr") {
      HIP_CHECK_ERROR(hipMemcpyBatchAsync(dstPtr, srcPtr, sizes, count, attr, nullptr, numAttrs,
                                          &failIdx, stream),
                      hipErrorInvalidValue);
    }
 #endif
   // Clean up
   for (int i = 0; i < count; i++) {
     HIP_CHECK(hipFree(srcPtr[i]));
     HIP_CHECK(hipFree(dstPtr[i]));
   }
   HIP_CHECK(hipStreamDestroy(stream));
 }
 /**
  * End doxygen group MemoryTest.
  * @}
  */