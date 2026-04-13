/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
 
 #include <hip_test_common.hh>
 #include <hip_test_defgroups.hh>
 #include <resource_guards.hh>

 #include "memcpyBatchAsync_common.hh"
 
 #if HT_AMD
 
 /**
  * Batch device-to-device copies across GPUs: each entry uses a distinct source allocation on
  * `device_for_src` and a distinct destination on `device_for_dst`. Peer access is enabled from
  * `device_for_dst` to `device_for_src`.
  */
 template <typename TestType> struct BasicCopyP2PTest {
   BasicCopyP2PTest(size_t count, size_t num_elements, int device_src, int device_dst)
       : count{count},
         num_elements{num_elements},
         size_in_bytes{num_elements * sizeof(TestType)},
         device_for_src{device_src},
         device_for_dst{device_dst},
         initial_values(num_elements, get_test_values<TestType>().first),
         src_ptrs(count),
         dst_ptrs(count) {}
 
   void runTest() {
     initialize_mem();
     execute();
     verify_results();
     free_mem();
     disable_peer_access();
   }
 
  private:
   void initialize_mem() {
     HIP_CHECK(hipSetDevice(device_for_dst));
     HIP_CHECK(hipDeviceEnablePeerAccess(device_for_src, 0));
 
     HIP_CHECK(hipStreamCreate(&stream));
 
     for (size_t i = 0; i < count; ++i) {
       HIP_CHECK(hipSetDevice(device_for_src));
       LinearAllocGuard<TestType> src_alloc(LinearAllocs::hipMalloc, size_in_bytes);
       src_ptrs[i] = src_alloc.ptr();
       HIP_CHECK(
           hipMemcpy(src_ptrs[i], initial_values.data(), size_in_bytes, hipMemcpyHostToDevice));
       allocations.push_back(std::move(src_alloc));
 
       HIP_CHECK(hipSetDevice(device_for_dst));
       LinearAllocGuard<TestType> dst_alloc(LinearAllocs::hipMalloc, size_in_bytes);
       dst_ptrs[i] = dst_alloc.ptr();
       HIP_CHECK(hipMemset(dst_ptrs[i], 0, size_in_bytes));
       allocations.push_back(std::move(dst_alloc));
     }
   }
 
   void execute() {
     HIP_CHECK(hipSetDevice(device_for_dst));
     std::vector<size_t> sizes(count, size_in_bytes);
     const size_t num_attributes = 0;
     std::vector<size_t> attributes_indexes(1);
     attributes_indexes[0] = 0;
     size_t fail_index = 0;
     HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), count, nullptr,
                                   attributes_indexes.data(), num_attributes, &fail_index, stream));
     HIP_CHECK(hipStreamSynchronize(stream));
   }
 
   void verify_results() {
     std::vector<std::vector<TestType>> host_out(count, std::vector<TestType>(num_elements));
     HIP_CHECK(hipSetDevice(device_for_dst));
     for (size_t i = 0; i < count; ++i) {
       HIP_CHECK(hipMemcpy(host_out[i].data(), dst_ptrs[i], size_in_bytes, hipMemcpyDeviceToHost));
       for (size_t j = 0; j < num_elements; ++j) {
         REQUIRE(host_out[i][j] == initial_values[j]);
       }
     }
   }
 
   void free_mem() { HIP_CHECK(hipStreamDestroy(stream)); }
 
   void disable_peer_access() {
     HIP_CHECK(hipSetDevice(device_for_dst));
     HIP_CHECK(hipDeviceDisablePeerAccess(device_for_src));
   }
 
   const size_t count;
   const size_t num_elements;
   const size_t size_in_bytes;
   const int device_for_src;
   const int device_for_dst;
 
   std::vector<TestType> initial_values;
   std::vector<LinearAllocGuard<TestType>> allocations;
   hipStream_t stream{};
   std::vector<void*> src_ptrs;
   std::vector<void*> dst_ptrs;
 };
 
 /**
  * Batched D2D copies from device 0 to device 1: each entry has its own src allocation on
  * device 0 and dst allocation on device 1. The stream and hipMemcpyBatchAsync run on device 1
  * with peer access enabled from device 1 to device 0 (see BasicCopyP2PTest).
  */
 TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_P2P_Basic, char, int, float) {
   if (HipTest::getDeviceCount() < 2) {
     HipTest::HIP_SKIP_TEST("Skipping because fewer than 2 devices are available");
     return;
   }
 
   int can_access_peer = 0;
   HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 1, 0));
 
   if (!can_access_peer) {
     HipTest::HIP_SKIP_TEST("Skipping because device 1 cannot access peer memory on device 0");
     return;
   }
 
   const size_t batch_count = GENERATE(1, 3, 8);
   const size_t num_elements = 4096;
   BasicCopyP2PTest<TestType> test(batch_count, num_elements, 0, 1);
   test.runTest();
 }
 
 /**
  * For each batch entry, the contents of the two device buffers are exchanged using
  * hipMemcpyFlagExtOpSwap. Buffers live on two different GPUs with mutual peer access; the batch is
  * issued from one device while the other side is reachable via P2P.
  */
 template <typename TestType> struct SwapCopyP2pTest {
   SwapCopyP2pTest(size_t count, size_t num_elements, int device_for_a, int device_for_b,
                   hipError_t expectedError)
       : count{count},
         num_elements{num_elements},
         size_in_bytes{num_elements * sizeof(TestType)},
         device_for_a{device_for_a},
         device_for_b{device_for_b},
         initial_values_a(count,
                          std::vector<TestType>(num_elements, get_test_values<TestType>().first)),
         initial_values_b(count,
                          std::vector<TestType>(num_elements, get_test_values<TestType>().second)),
         swap_ptrs_a(count),
         swap_ptrs_b(count),
         expectedReturnValue(expectedError) {}
 
   void runTest() {
     initialize_mem();
     execute();
     if (expectedReturnValue == hipSuccess) {
       verify_results();
     }
     free_mem();
     disable_peer_access();
   }
 
  private:
   void initialize_mem() {
     HIP_CHECK(hipSetDevice(device_for_a));
     HIP_CHECK(hipDeviceEnablePeerAccess(device_for_b, 0));
     HIP_CHECK(hipSetDevice(device_for_b));
     HIP_CHECK(hipDeviceEnablePeerAccess(device_for_a, 0));
 
     HIP_CHECK(hipSetDevice(device_for_a));
     HIP_CHECK(hipStreamCreate(&stream));
 
     for (size_t i = 0; i < count; ++i) {
       HIP_CHECK(hipSetDevice(device_for_b));
       LinearAllocGuard<TestType> alloc_b(LinearAllocs::hipMalloc, size_in_bytes);
       swap_ptrs_b[i] = alloc_b.ptr();
       allocations.push_back(std::move(alloc_b));
       HIP_CHECK(hipMemcpy(swap_ptrs_b[i], initial_values_b[i].data(), size_in_bytes,
                           hipMemcpyHostToDevice));
 
       HIP_CHECK(hipSetDevice(device_for_a));
       LinearAllocGuard<TestType> alloc_a(LinearAllocs::hipMalloc, size_in_bytes);
       swap_ptrs_a[i] = alloc_a.ptr();
       allocations.push_back(std::move(alloc_a));
       HIP_CHECK(hipMemcpy(swap_ptrs_a[i], initial_values_a[i].data(), size_in_bytes,
                           hipMemcpyHostToDevice));
     }
   }
 
   void execute() {
     HIP_CHECK(hipSetDevice(device_for_a));
     std::vector<size_t> sizes(count, size_in_bytes);
     const size_t num_attributes = 1;
     std::vector<hipMemcpyAttributes> attributes(num_attributes);
     attributes[0].flags = hipMemcpyFlagExtOpSwap;
     attributes[0].srcAccessOrder = hipMemcpySrcAccessOrderStream;
     std::vector<size_t> attributes_indexes(num_attributes);
     attributes_indexes[0] = 0;
     size_t fail_index = 0;
 
     HIP_CHECK_ERROR(hipMemcpyBatchAsync(swap_ptrs_a.data(), swap_ptrs_b.data(), sizes.data(),
                                         count, attributes.data(), attributes_indexes.data(),
                                         num_attributes, &fail_index, stream),
                     expectedReturnValue);
     if (expectedReturnValue != hipSuccess) return;
 
     HIP_CHECK(hipStreamSynchronize(stream));
   }
 
   void verify_results() {
     std::vector<std::vector<TestType>> host_a_out(count, std::vector<TestType>(num_elements));
     std::vector<std::vector<TestType>> host_b_out(count, std::vector<TestType>(num_elements));
     HIP_CHECK(hipSetDevice(device_for_a));
     for (size_t i = 0; i < count; ++i) {
       HIP_CHECK(
           hipMemcpy(host_a_out[i].data(), swap_ptrs_a[i], size_in_bytes, hipMemcpyDeviceToHost));
     }
     HIP_CHECK(hipSetDevice(device_for_b));
     for (size_t i = 0; i < count; ++i) {
       HIP_CHECK(
           hipMemcpy(host_b_out[i].data(), swap_ptrs_b[i], size_in_bytes, hipMemcpyDeviceToHost));
     }
     for (size_t i = 0; i < count; ++i) {
       for (size_t j = 0; j < num_elements; ++j) {
         REQUIRE(host_a_out[i][j] == initial_values_b[i][j]);
         REQUIRE(host_b_out[i][j] == initial_values_a[i][j]);
       }
     }
   }
 
   void free_mem() { HIP_CHECK(hipStreamDestroy(stream)); }
 
   void disable_peer_access() {
     HIP_CHECK(hipSetDevice(device_for_a));
     HIP_CHECK(hipDeviceDisablePeerAccess(device_for_b));
     HIP_CHECK(hipSetDevice(device_for_b));
     HIP_CHECK(hipDeviceDisablePeerAccess(device_for_a));
   }
 
   const size_t count;
   const size_t num_elements;
   const size_t size_in_bytes;
   const int device_for_a;
   const int device_for_b;
 
   std::vector<std::vector<TestType>> initial_values_a;
   std::vector<std::vector<TestType>> initial_values_b;
   std::vector<LinearAllocGuard<TestType>> allocations;
   hipStream_t stream{};
   std::vector<void*> swap_ptrs_a;
   std::vector<void*> swap_ptrs_b;
 
   hipError_t expectedReturnValue;
 };
 
 /**
  * Batched buffer exchange between device 0 and device 1: for each batch entry, one allocation on
  * each GPU is swapped via hipMemcpyFlagExtOpSwap with mutual peer access enabled; the stream and
  * hipMemcpyBatchAsync run on device 0 (see SwapCopyP2pTest).
  */
 TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_P2P_Swap, char, int, float) {
   if (HipTest::getDeviceCount() < 2) {
     HipTest::HIP_SKIP_TEST(
         "Skipping because fewer than 2 devices are available");
     return;
   }

   const int dev_a = 0;
   const int dev_b = 1;
   int can_access_peer_a_to_b = 0;
   int can_access_peer_b_to_a = 0;
   HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer_a_to_b, dev_a, dev_b));
   HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer_b_to_a, dev_b, dev_a));

   if (!can_access_peer_a_to_b || !can_access_peer_b_to_a) {
     HipTest::HIP_SKIP_TEST("Skipping because peer access is not supported in "
                            "both directions between device 0 and "
                            "device 1");
     return;
   }

   const size_t count = GENERATE(2, 3, 8);
   const size_t num_elements = 4096;

   hipError_t expectedError =
       getSwapExpectedReturn(LinearAllocs::hipMalloc, LinearAllocs::hipMalloc);
   SwapCopyP2pTest<TestType> test(count, num_elements, dev_a, dev_b,
                                  expectedError);
   test.runTest();
 }

 /**
  * Batched multicast device-to-device copies across GPUs: all entries share one source allocation on
  * `device_src` and each uses a distinct destination on `device_dst`. Peer access is enabled from
  * `device_dst` to `device_src`; the stream is created on `device_dst`.
  */
 template <typename TestType> struct MulticastCopyP2pTest {
   MulticastCopyP2pTest(size_t count, size_t num_elements, int device_src, int device_dst)
       : count{count},
         num_elements{num_elements},
         size_in_bytes{num_elements * sizeof(TestType)},
         device_for_src{device_src},
         device_for_dst{device_dst},
         initial_values(num_elements, get_test_values<TestType>().first),
         src_ptrs(count),
         dst_ptrs(count) {}
 
   void runTest() {
     initialize_mem();
     execute();
     verify_results();
     free_mem();
     disable_peer_access();
   }
 
  private:
   void initialize_mem() {
     HIP_CHECK(hipSetDevice(device_for_dst));
     HIP_CHECK(hipDeviceEnablePeerAccess(device_for_src, 0));
 
     HIP_CHECK(hipSetDevice(device_for_src));
     LinearAllocGuard<TestType> src_alloc(LinearAllocs::hipMalloc, size_in_bytes);
     src_mem = src_alloc.ptr();
     HIP_CHECK(hipMemcpy(src_mem, initial_values.data(), size_in_bytes, hipMemcpyHostToDevice));
     allocations.push_back(std::move(src_alloc));
 
     HIP_CHECK(hipSetDevice(device_for_dst));
     HIP_CHECK(hipStreamCreate(&stream));
 
     for (size_t i = 0; i < count; ++i) {
       src_ptrs[i] = src_mem;
       LinearAllocGuard<TestType> dst_alloc(LinearAllocs::hipMalloc, size_in_bytes);
       dst_ptrs[i] = dst_alloc.ptr();
       allocations.push_back(std::move(dst_alloc));
     }
   }
 
   void execute() {
     HIP_CHECK(hipSetDevice(device_for_dst));
     std::vector<size_t> sizes(count, size_in_bytes);
     const size_t num_attributes = 1;
     std::vector<hipMemcpyAttributes> attributes(num_attributes);
     attributes[0].srcAccessOrder = hipMemcpySrcAccessOrderStream;
     std::vector<size_t> attributes_indexes(num_attributes);
     attributes_indexes[0] = 0;
     size_t fail_index = 0;
     HIP_CHECK(hipMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), count,
                                   attributes.data(), attributes_indexes.data(), num_attributes,
                                   &fail_index, stream));
     HIP_CHECK(hipStreamSynchronize(stream));
   }
 
   void verify_results() {
     std::vector<std::vector<TestType>> host_out(count, std::vector<TestType>(num_elements));
     HIP_CHECK(hipSetDevice(device_for_dst));
     for (size_t i = 0; i < count; ++i) {
       HIP_CHECK(hipMemcpy(host_out[i].data(), dst_ptrs[i], size_in_bytes, hipMemcpyDeviceToHost));
       for (size_t j = 0; j < num_elements; ++j) {
         REQUIRE(host_out[i][j] == initial_values[j]);
       }
     }
   }
 
   void free_mem() { HIP_CHECK(hipStreamDestroy(stream)); }
 
   void disable_peer_access() {
     HIP_CHECK(hipSetDevice(device_for_dst));
     HIP_CHECK(hipDeviceDisablePeerAccess(device_for_src));
   }
 
   const size_t count;
   const size_t num_elements;
   const size_t size_in_bytes;
   const int device_for_src;
   const int device_for_dst;
 
   std::vector<TestType> initial_values;
   std::vector<LinearAllocGuard<TestType>> allocations;
   hipStream_t stream{};
   void* src_mem{};
   std::vector<void*> src_ptrs;
   std::vector<void*> dst_ptrs;
 };
 
 /**
  * Cross-GPU batched multicast: one shared source on the peer GPU, multiple destinations on the
  * local GPU.
  */
 TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_P2P_Multicast, char, int, float) {
   if (HipTest::getDeviceCount() < 2) {
     HipTest::HIP_SKIP_TEST("Skipping because fewer than 2 devices are available");
     return;
   }
 
   int can_access_peer = 0;
   HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 1, 0));
 
   if (!can_access_peer) {
     HipTest::HIP_SKIP_TEST("Skipping because device 1 cannot access peer memory on device 0");
     return;
   }
 
   const size_t count = GENERATE(2, 3, 8);
   const size_t num_elements = 4096;
 
   MulticastCopyP2pTest<TestType> test(count, num_elements, 0, 1);
   test.runTest();
 }
 
 /**
  * Cross-GPU batched multicast with large per-copy size.
  */
 TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_P2P_Multicast_Large, char, int, float) {
   if (HipTest::getDeviceCount() < 2) {
     HipTest::HIP_SKIP_TEST("Skipping because fewer than 2 devices are available");
     return;
   }
 
   int can_access_peer = 0;
   HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 1, 0));
 
   if (!can_access_peer) {
     HipTest::HIP_SKIP_TEST("Skipping because device 1 cannot access peer memory on device 0");
     return;
   }
 
   const size_t count = GENERATE(2, 3, 8);
   const size_t num_elements = 1024 * 1024;
 
   MulticastCopyP2pTest<TestType> test(count, num_elements, 0, 1);
   test.runTest();
 }
 #endif
 