// Copyright (c) 2017 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

/*
 * Modifications Copyright (C)2023 Advanced
 * Micro Devices, Inc. All rights reserved.
 */
#include <chrono>
#include <thread>
#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>
#include "hipSVMCommon.h"
// #define DEBUG_ATOMIC  // To provide additional data for debugging
#ifdef DEBUG_ATOMIC
// #define DEBUG_ATOMIC_PRINT_THREAD
#endif

typedef struct BinNode {
#ifdef DEBUG_ATOMIC
  unsigned int n;
  unsigned int d;
  unsigned int i;
#endif
  unsigned int value;
  struct BinNode* pNext;
} BinNode;

__global__ void build_hash_table_on_device(unsigned int* input, size_t inputSize, BinNode* pNodes,
                                           unsigned int* pNumNodes, unsigned int numBins,
                                           unsigned int dev) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= inputSize) return;

  unsigned int n = atomicAdd_system((unsigned int*)pNumNodes, 1u);
  BinNode* pNew = &pNodes[n];
  unsigned int b = input[i] % numBins;

  pNew->value = input[i];
#ifdef DEBUG_ATOMIC
  pNew->d = dev;
  pNew->i = i;
  pNew->n = n;
#endif
  unsigned long long next = 0;
  unsigned long long old = atomicAdd_system((unsigned long long*)&(pNodes[b].pNext),
                                            0ull);  // Because of no atomicLoad()
  do {
    next = old;
    // Use CAS to ensure atomic operation
    // pNew->pNext =  (BinNode*)next;
    atomicExch((unsigned long long*)&(pNew->pNext), next);
    old = atomicCAS_system((unsigned long long*)&(pNodes[b].pNext), next, (unsigned long long)pNew);
  } while (old != next);
#ifdef DEBUG_ATOMIC_PRINT_THREAD
  printf(
      "k%u: i=%zu, n=%u, pNew=%p(n=%2u, d=%u, i=%4u, value=%4u, next=%p), pNodes[%u]=%p,"
      " old=%p, input[%zu]=%u\n",
      dev, i, n, pNew, pNew->n, pNew->d, pNew->i, pNew->value, pNew->pNext, b, &pNodes[b],
      (void*)old, i, input[i]);
#else
  (void)dev;
#endif
}

void build_hash_table_on_host(unsigned int* input, size_t inputSize, BinNode* pNodes,
                              unsigned int* pNumNodes, unsigned int numBins, unsigned int dev) {
  // wait until we see some activity from a device (try to run host side simultaneously).
  while (numBins == AtomicLoad32(pNumNodes));
  for (unsigned int i = 0; i < inputSize; i++) {
    unsigned int n = AtomicFetchAdd32(pNumNodes, 1u);
    BinNode* pNew = &pNodes[n];
    unsigned int b = input[i] % numBins;
#ifdef DEBUG_ATOMIC
    pNew->d = dev;
    pNew->i = i;
    pNew->n = n;
#endif
    pNew->value = input[i];
    BinNode* next = AtomicFetchAdd64(&pNodes[b].pNext, (BinNode*)0ll);
    do {
      AtomicExchange64(&(pNew->pNext), next);
      // always inserting at head of list
    } while (!AtomicCompareExchange64(&(pNodes[b].pNext), &next, (BinNode*)pNew));
#ifdef DEBUG_ATOMIC_PRINT_THREAD
    fprintf(stderr,
            "k%u: i=%u, n=%u, pNew=%p(n=%2u, d=%u, i=%4u, value=%4u, next=%p), pNodes[%u]=%p, "
            "input[%u]=%u\n",
            dev, i, n, pNew, pNew->n, pNew->d, pNew->i, pNew->value, pNew->pNext, b, &pNodes[b], i,
            input[i]);
#else
    (void)dev;
#endif
  }
}

void launch_kernels_and_verify(std::vector<hipStream_t>& streams, unsigned int num_devices,
                               unsigned int numBins, size_t num_pixels) {
  unsigned int* pInputImage = nullptr;
  BinNode* pNodes = nullptr;
  unsigned int* pNumNodes = nullptr;
  unsigned int total_items = num_pixels * (num_devices + 1);
  HIP_CHECK(hipHostMalloc(&pInputImage, sizeof(unsigned int) * num_pixels, hipHostMallocCoherent));
  HIP_CHECK(
      hipHostMalloc(&pNodes, sizeof(BinNode) * (total_items + numBins), hipHostMallocCoherent));
  HIP_CHECK(hipHostMalloc(&pNumNodes, sizeof(unsigned int), hipHostMallocCoherent));

  *pNumNodes = numBins;  // using the first numBins nodes to hold the list heads.
  for (unsigned int i = 0; i < numBins; i++) pNodes[i].pNext = nullptr;
  for (unsigned int i = 0; i < num_pixels; i++) pInputImage[i] = i;

  // Get all the devices going simultaneously, each device (and the host) will insert
  // all the pixels.
  for (unsigned int d = 0; d < num_devices; d++) {
    HIP_CHECK(hipSetDevice(d));
    build_hash_table_on_device<<<(num_pixels + 255) / 256, 256, 0, streams[d]>>>(
        pInputImage, num_pixels, pNodes, pNumNodes, numBins, d);
    HIP_CHECK(hipGetLastError());
  }

  std::vector<std::thread> threads;
  threads.push_back(std::thread(build_hash_table_on_host, pInputImage, num_pixels, pNodes,
                                pNumNodes, numBins, num_devices));
  for (unsigned int d = 0; d < num_devices; d++) {
    threads.push_back(std::thread(
        [](hipStream_t s) {
          HIP_CHECK(hipStreamSynchronize(s));  // To workarround batch dispatching on Windows
        },
        streams[d]));
  }
  std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });

  for (unsigned int d = 0; d < num_devices; d++) {
    HIP_CHECK(hipSetDevice(d));
    HIP_CHECK(hipDeviceSynchronize());
  }
  HIP_CHECK(hipSetDevice(0));
  unsigned int num_items = 0;
  // check correctness of each bin in the hash table.
  for (unsigned int i = 0; i < numBins; i++) {
    BinNode* pNode = pNodes[i].pNext;
    unsigned int num_items_bin = 0;
    unsigned int total_num_items_bin =
        (num_pixels % numBins <= i) ? (num_pixels / numBins) : (num_pixels / numBins + 1);
    total_num_items_bin *= (num_devices + 1);  // The item number of the list in i-th bin
    while (pNode) {
#ifdef DEBUG_ATOMIC_PRINT_THREAD
      fprintf(stderr, "v%u/%u: %u, pNode=%p(n=%2u, d=%u, i=%4u, value=%4u, next=%p)\n", i, numBins,
              num_items_bin, pNode, pNode->n, pNode->d, pNode->i, pNode->value, pNode->pNext);
#endif
      if ((pNode->value % numBins) != i) {
        fprintf(stderr,
                "Something went wrong at i=%u, item is in wrong hash bucket:"
                "pNode->value=%u, numBins=%u\n",
                i, pNode->value, numBins);
        REQUIRE(false);
      }
      num_items++;
      num_items_bin++;
      if (num_items_bin > total_num_items_bin) {
        fprintf(stderr,
                "Something went wrong at i=%u/%u, num_items_bin(%u)>total_num_items_bin(%u)\n", i,
                numBins, num_items_bin, total_num_items_bin);
        REQUIRE(false);
      }
      pNode = pNode->pNext;
    }
    if (num_items_bin != total_num_items_bin) {
      fprintf(stderr,
              "Something went wrong at i=%u/%u, num_items_bin(%u)!=total_num_items_bin(%u)\n", i,
              numBins, num_items_bin, total_num_items_bin);
    }
  }
  HIP_CHECK(hipHostFree(pInputImage));
  HIP_CHECK(hipHostFree(pNodes));
  HIP_CHECK(hipHostFree(pNumNodes));

  // each device and the host inserted all of the pixels, check that none are missing.
  if (num_items != total_items) {
    fprintf(stderr, "The hash table is not correct, num items %u != expected num items: %u\n",
            num_items, total_items);
    REQUIRE(false);  // test did not pass
  }
  REQUIRE(true);
}

/**
* Test Description
* ------------------------
* - The suite will test the following functions,
      hipHostMalloc() with following flags,
        hipHostMallocCoherent(CL_MEM_SVM_FINE_GRAIN_BUFFER + CL_MEM_SVM_ATOMICS)
      atomicAdd_system()(in kernel)
      atomicCAS_system()(in kernel)
      atomicExch()(in kernel)
      InterlockedExchangeAdd()(in WINDOWS host)
      __sync_add_and_fetch()(in LINUX host)
      InterlockedExchangeAdd64()(in WINDOWS host)
      InterlockedExchangePointer()(in WINDOWS host)
      __sync_lock_test_and_set()(in LINUX host)
      InterlockedCompareExchange64()(in WINDOWS host)
      __sync_val_compare_and_swap()(in LINUX host)
      hipDeviceSynchronize()
*   It will demonstrate use of SVM's atomics to do fine grain synchronization among
*   devices and the host.
*   Concept: Each device and the host simultaneously insert values into a single hash table.
*   Each bin in the hash table is a linked list.  Each bin is protected against simultaneous
*   update using a lock free technique.  The correctness of the list is verified on the host.
* Test source
* ------------------------
* - catch/unit/memory/hipSVMTestFineGrainMemoryConsistency.cpp
* Test requirements
* ------------------------
*  - Host specific (WINDOWS and LINUX)
*  - Fine grain access and atomics supported on devices and host
*  - HIP_VERSION >= 5.7
*/
TEST_CASE("test_svm_fine_grain_memory_consistency") {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));

  for (int id = 0; id < num_devices; id++) {
    int pcieAtomic = 0;
    HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, id));
    if (!pcieAtomic) {
      fprintf(stderr, "Device doesn't support pcie atomic, Skipped\n");
      REQUIRE(true);
      return;
    }
  }
  const int num_elements = 2167;
  std::vector<hipStream_t> streams(num_devices);

  for (int d = 0; d < num_devices; d++) {
    HIP_CHECK(hipSetDevice(d));
    HIP_CHECK(hipStreamCreate(&streams[d]));
  }
  HIP_CHECK(hipSetDevice(0));

  // all work groups in all devices and the host code will hammer on this one lock.
  unsigned int numBins = 1;
  launch_kernels_and_verify(streams, num_devices, numBins, num_elements);

  numBins = 2;  // 2 locks within in same cache line will get hit from different devices and host.
  launch_kernels_and_verify(streams, num_devices, numBins, num_elements);

  numBins = 29;  // locks span a few cache lines.
  launch_kernels_and_verify(streams, num_devices, numBins, num_elements);

  for (unsigned int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
}
