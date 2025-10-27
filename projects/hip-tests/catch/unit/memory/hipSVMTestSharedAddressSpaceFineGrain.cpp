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

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>
#include "hipSVMCommon.h"

// const char *linked_list_create_and_verify_kernels[] = {
typedef struct Node {
  unsigned int global_id;
  unsigned int position_in_list;
  struct Node* pNext;
} Node;

// The allocation_index parameter must be initialized on the host to N work-items
// The first N nodes in pNodes will be the heads of the lists.
__global__ void create_linked_lists_on_device(Node* pNodes, unsigned int* allocation_index,
                                              unsigned int list_length) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  Node* pNode = &pNodes[i];

  pNode->global_id = i;
  pNode->position_in_list = 0;

  Node* pNew;
  for (unsigned int j = 1; j < list_length; j++) {
    pNew = &pNodes[atomicAdd(allocation_index, 1u)];  // allocate a new node
    pNew->global_id = i;
    pNew->position_in_list = j;
    pNode->pNext = pNew;  // link new node onto end of list
    pNode = pNew;         // move to end of list
  }
}

__global__ void verify_linked_lists_on_device(Node* pNodes, unsigned int* num_correct,
                                              unsigned int list_length) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  Node* pNode = &pNodes[i];

  for (unsigned int j = 0; j < list_length; j++) {
    if (pNode->global_id == i && pNode->position_in_list == j) {
      atomicAdd(num_correct, 1u);
    } else {
      break;
    }
    pNode = pNode->pNext;
  }
}

// The first N nodes in pNodes will be the heads of the lists.
void create_linked_lists_on_host(Node* pNodes, unsigned int num_lists, unsigned int list_length) {
  unsigned int allocation_index = num_lists;  // heads of lists are in first num_lists nodes.
  for (unsigned int i = 0; i < num_lists; i++) {
    Node* pNode = &pNodes[i];
    pNode->global_id = i;
    pNode->position_in_list = 0;
    Node* pNew;
    for (unsigned int j = 1; j < list_length; j++) {
      pNew = &pNodes[allocation_index++];  // allocate a new node
      pNew->global_id = i;
      pNew->position_in_list = j;
      pNode->pNext = pNew;  // link new node onto end of list
      pNode = pNew;         // move to end of list
    }
  }
}

void verify_linked_lists_on_host(Node* pNodes, unsigned int num_lists, unsigned int list_length) {
  unsigned int numCorrect = 0;
  for (unsigned int i = 0; i < num_lists; i++) {
    Node* pNode = &pNodes[i];
    for (int j = 0; j < list_length; j++) {
      if (pNode->global_id == i && pNode->position_in_list == j) {
        numCorrect++;
      } else {
        break;
      }
      pNode = pNode->pNext;
    }
  }
  if (numCorrect != list_length * num_lists) {
    fprintf(stderr, "Failed: numCorrect = %d, list_length=%u, numLists = %u\n", numCorrect,
            list_length, num_lists);
    REQUIRE(false);
  }
}

void create_linked_lists_on_device(hipStream_t stream, Node* pNodes, unsigned int* pAllocator,
                                   unsigned int numLists, unsigned int ListLength) {
  // reset allocator index
  *pAllocator = numLists;  // the first numLists elements of the nodes array are already
                           // allocated (they hold the head of each list).
  create_linked_lists_on_device<<<(numLists + 255) / 256, 256, 0, stream>>>(pNodes, pAllocator,
                                                                            ListLength);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(stream));
}

void verify_linked_lists_on_device(hipStream_t stream, Node* pNodes, unsigned int* pNumCorrect,
                                   unsigned int numLists, unsigned int ListLength) {
  *pNumCorrect = 0;  // reset numCorrect to zero

  verify_linked_lists_on_device<<<(numLists + 255) / 256, 256, 0, stream>>>(pNodes, pNumCorrect,
                                                                            ListLength);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(stream));

  int correct_count = *pNumCorrect;
  if (correct_count != ListLength * numLists) {
    fprintf(stderr, "Failed: correct_count = %d, ListLength=%u, numLists = %u\n", correct_count,
            ListLength, numLists);
    REQUIRE(false);
  }
}

/**
* Test Description
* ------------------------
* - The suite will test the following functions,
      hipHostMalloc() with following flags,
        hipHostMallocNonCoherent(CL_MEM_SVM_FINE_GRAIN_BUFFER)
        hipHostMallocCoherent(CL_MEM_SVM_FINE_GRAIN_BUFFER + CL_MEM_SVM_ATOMICS)
      atomicAdd()(in kernel)
      hipStreamCreate()
      hipStreamSynchronize()
*   It will test that all devices and the host share a common address space using fine-grain
*   host buffers.
*   Concept: This is done by creating a linked list on a device and then verifying the
*   correctness of the list on another device or the host.  This basic test is performed for all
*   combinations of devices and the host that exist within the platform. The test passes only if
*   every combination passes.
* Test source
* ------------------------
* - catch/unit/memory/hipSVMTestSharedAddressSpaceFineGrain.cpp
* Test requirements
* ------------------------
*  - Host specific (WINDOWS and LINUX)
*  - Fine grain access supported on devices and host
*  - HIP_VERSION >= 5.7
*/
TEST_CASE("test_svm_shared_address_space_fine_grain_buffers") {
  const unsigned int num_elements = 1024;
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));

  for (int id = 0; id < num_devices; id++) {
    int pcieAtomic = 0;
    HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, id));
    if (!pcieAtomic) {
      fprintf(stderr, "Device %d doesn't support pcie atomic, Skipped\n", id);
      REQUIRE(true);
      return;
    }
  }

  int num_devices_plus_host = num_devices + 1;
  std::vector<hipStream_t> streams(num_devices);

  for (int d = 0; d < num_devices; d++) {
    HIP_CHECK(hipSetDevice(d));
    HIP_CHECK(hipStreamCreate(&streams[d]));
  }
  HIP_CHECK(hipSetDevice(0));

  unsigned int numLists = num_elements;
  unsigned int ListLength = 32;
  Node* pNodes = nullptr;
  unsigned int* pAllocator = nullptr;
  unsigned int* pNumCorrect = nullptr;
  HIP_CHECK(hipHostMalloc(&pNodes, sizeof(Node) * ListLength * numLists, hipHostMallocNonCoherent));
  HIP_CHECK(hipHostMalloc(&pAllocator, sizeof(unsigned int), hipHostMallocCoherent));
  HIP_CHECK(hipHostMalloc(&pNumCorrect, sizeof(unsigned int), hipHostMallocCoherent));

  // Create linked list on one device and verify on another device (or the host).
  // Do this for all possible combinations of devices and host within the platform.
  // ci is CreationIndex, index of device/q to create linked list on
  for (int ci = 0; ci < num_devices_plus_host; ci++) {
    // vi is VerificationIndex, index of device/q to verify linked list on
    for (int vi = 0; vi < num_devices_plus_host; vi++) {
      if (ci == num_devices)  // last device index represents the host, note the num_device+1 above.
      {
        create_linked_lists_on_host(pNodes, numLists, ListLength);
      } else {
        HIP_CHECK(hipSetDevice(ci));
        create_linked_lists_on_device(streams[ci], pNodes, pAllocator, numLists, ListLength);
      }

      if (vi == num_devices) {
        verify_linked_lists_on_host(pNodes, numLists, ListLength);
      } else {
        HIP_CHECK(hipSetDevice(vi));
        verify_linked_lists_on_device(streams[vi], pNodes, pNumCorrect, numLists, ListLength);
      }
    }
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipHostFree(pNodes));
  HIP_CHECK(hipHostFree(pAllocator));
  HIP_CHECK(hipHostFree(pNumCorrect));
  for (int d = 0; d < num_devices; d++) {
    HIP_CHECK(hipStreamDestroy(streams[d]));
  }
  REQUIRE(true);
}

/**
* Test Description
* ------------------------
* - The suite will test the following functions,
      align_malloc()
      atomicAdd()(in kernel)
      hipStreamCreate()
      hipStreamSynchronize()
*   It will test that all devices and the host share a common address space using fine-grain mode
*   with regular host buffers.
*   Concept: This is done by creating a linked list on a device and then verifying the
*   correctness of the list on another device or the host.  This basic test is performed for all
*   combinations of devices and the host that exist within the platform.  The test passes only if
*   every combination passes.
* Test source
* ------------------------
* - catch/unit/memory/hipSVMTestSharedAddressSpaceFineGrain.cpp
* Test requirements
* ------------------------
*  - Host specific (WINDOWS and LINUX)
*  - Unified memory supported on devices
*  - System fine grain access supported on devices
*  - HIP_VERSION >= 5.7
*/
TEST_CASE("test_svm_shared_address_space_fine_grain_system") {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));

  for (int id = 0; id < num_devices; id++) {
    int pcieAtomic = 0;
    HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, id));
    if (!pcieAtomic) {
      fprintf(stderr, "Device %d doesn't support pcie atomic, Skipped\n", id);
      REQUIRE(true);
      return;
    }

    int pageableAccess = 0;
    // This need xnack+ on MiXXX. If xnack is off on MiXXX, try ENV HSA_XNACK=1
    HIP_CHECK(hipDeviceGetAttribute(&pageableAccess, hipDeviceAttributePageableMemoryAccess, id));
    if (!pageableAccess) {
      fprintf(stderr, "Device %d doesn't support access to pageable address. Skipped\n", id);
      REQUIRE(true);
      return;
    }
  }

  const unsigned int num_elements = 1024;
  int num_devices_plus_host = num_devices + 1;
  std::vector<hipStream_t> streams(num_devices);

  for (int d = 0; d < num_devices; d++) {
    HIP_CHECK(hipSetDevice(d));
    HIP_CHECK(hipStreamCreate(&streams[d]));
  }
  HIP_CHECK(hipSetDevice(0));

  unsigned int numLists = num_elements;
  unsigned int ListLength = 32;

  // this allocation holds the linked list nodes.
  Node* pNodes = (Node*)align_malloc(numLists * ListLength * sizeof(Node), 128);
  // this allocation holds an index into the nodes buffer, it is used for node allocation
  unsigned int* pAllocator = (unsigned int*)align_malloc(sizeof(unsigned int), 128);
  // this allocation holds the count of correct nodes, which is computed by the verify kernel.
  unsigned int* pNumCorrect = (unsigned int*)align_malloc(sizeof(unsigned int), 128);

  // ci is CreationIndex, index of device/q to create linked list on
  for (int ci = 0; ci < num_devices_plus_host; ci++) {
    // vi is VerificationIndex, index of device/q to verify linked list on
    for (int vi = 0; vi < num_devices_plus_host; vi++) {
      if (ci == num_devices)  // last device index represents the host, note the num_device+1 above.
      {
        create_linked_lists_on_host(pNodes, numLists, ListLength);
      } else {
        HIP_CHECK(hipSetDevice(ci));
        create_linked_lists_on_device(streams[ci], pNodes, pAllocator, numLists, ListLength);
      }

      if (vi == num_devices) {
        verify_linked_lists_on_host(pNodes, numLists, ListLength);
      } else {
        HIP_CHECK(hipSetDevice(vi));
        verify_linked_lists_on_device(streams[vi], pNodes, pNumCorrect, numLists, ListLength);
      }
    }
  }

  HIP_CHECK(hipSetDevice(0));
  align_free(pNodes);
  align_free(pAllocator);
  align_free(pNumCorrect);
  for (int d = 0; d < num_devices; d++) {
    HIP_CHECK(hipStreamDestroy(streams[d]));
  }
  REQUIRE(true);
}
