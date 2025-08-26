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

#include <vector>
#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

constexpr bool enable_validation = false;

template <typename F> void WaitExternalSemaphoreCommon(F f) {
  VulkanTest vkt(enable_validation);

  constexpr uint32_t count = 1;
  const auto src_storage = vkt.CreateMappedStorage<int>(count, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  const auto dst_storage = vkt.CreateMappedStorage<int>(count, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  const auto command_buffer = vkt.GetCommandBuffer();

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK_RESULT(vkBeginCommandBuffer(command_buffer, &begin_info));
  VkBufferCopy buffer_copy = {};
  buffer_copy.size = count * sizeof(*src_storage.host_ptr);
  vkCmdCopyBuffer(command_buffer, src_storage.buffer, dst_storage.buffer, 1, &buffer_copy);
  VK_CHECK_RESULT(vkEndCommandBuffer(command_buffer));

  const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_BINARY);
  const auto hip_sem_handle_desc =
      vkt.BuildSemaphoreDescriptor(semaphore, VK_SEMAPHORE_TYPE_BINARY);

  hipExternalSemaphore_t hip_ext_semaphore;
  HIP_CHECK(hipImportExternalSemaphore(&hip_ext_semaphore, &hip_sem_handle_desc));

  hipExternalSemaphoreWaitParams hip_ext_semaphore_wait_params = {};
  hip_ext_semaphore_wait_params.flags = 0;
  hip_ext_semaphore_wait_params.params.fence.value = 0;
  HIP_CHECK(f(&hip_ext_semaphore, &hip_ext_semaphore_wait_params, 1, nullptr));
  PollStream(nullptr, hipErrorNotReady);

  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = &semaphore;

  *src_storage.host_ptr = 42;

  const auto fence = vkt.CreateFence();
  VK_CHECK_RESULT(vkQueueSubmit(vkt.GetQueue(), 1, &submit_info, fence));
  VK_CHECK_RESULT(
      vkWaitForFences(vkt.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000 /*5 seconds*/));

  PollStream(nullptr, hipSuccess);

  REQUIRE(42 == *dst_storage.host_ptr);

  HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
}

#if HT_NVIDIA
template <typename F> void WaitExternalTimelineSemaphoreCommon(F f) {
  VulkanTest vkt(enable_validation);

  const auto [wait_value, signal_value] =
      GENERATE(std::make_pair(2, 2), std::make_pair(2, 3), std::make_pair(3, 2));
  INFO("Wait value: " << wait_value << ", signal value: " << signal_value);

  const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_TIMELINE);
  const auto hip_sem_handle_desc =
      vkt.BuildSemaphoreDescriptor(semaphore, VK_SEMAPHORE_TYPE_TIMELINE);
  hipExternalSemaphore_t hip_ext_semaphore;
  HIP_CHECK(hipImportExternalSemaphore(&hip_ext_semaphore, &hip_sem_handle_desc));

  hipExternalSemaphoreWaitParams hip_ext_semaphore_wait_params = {};
  hip_ext_semaphore_wait_params.flags = 0;
  hip_ext_semaphore_wait_params.params.fence.value = wait_value;
  HIP_CHECK(f(&hip_ext_semaphore, &hip_ext_semaphore_wait_params, 1, nullptr));
  PollStream(nullptr, hipErrorNotReady);

  VkSemaphoreSignalInfo signal_info = {};
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.semaphore = semaphore;
  signal_info.value = signal_value;
  VK_CHECK_RESULT(vkSignalSemaphore(vkt.GetDevice(), &signal_info));
  if (wait_value > signal_value) {
    PollStream(nullptr, hipErrorNotReady);
    signal_info.value = wait_value;
    VK_CHECK_RESULT(vkSignalSemaphore(vkt.GetDevice(), &signal_info));
  }
  PollStream(nullptr, hipSuccess);

  HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
}
#endif

template <typename F> void WaitExternalMultipleSemaphoresCommon(F f) {
  VulkanTest vkt(enable_validation);

#if HT_AMD
  constexpr auto second_semaphore_type = VK_SEMAPHORE_TYPE_BINARY;
#else
  constexpr auto second_semaphore_type = VK_SEMAPHORE_TYPE_TIMELINE;
#endif

  constexpr uint32_t count = 1;
  const auto src_storage = vkt.CreateMappedStorage<int>(count, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  const auto dst_storage = vkt.CreateMappedStorage<int>(count, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  const auto command_buffer = vkt.GetCommandBuffer();

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK_RESULT(vkBeginCommandBuffer(command_buffer, &begin_info));
  VkBufferCopy buffer_copy = {};
  buffer_copy.size = count * sizeof(*src_storage.host_ptr);
  vkCmdCopyBuffer(command_buffer, src_storage.buffer, dst_storage.buffer, 1, &buffer_copy);
  VK_CHECK_RESULT(vkEndCommandBuffer(command_buffer));

  const auto binary_semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_BINARY);
  const auto hip_binary_sem_handle_desc =
      vkt.BuildSemaphoreDescriptor(binary_semaphore, VK_SEMAPHORE_TYPE_BINARY);
  hipExternalSemaphore_t hip_binary_ext_semaphore;
  HIP_CHECK(hipImportExternalSemaphore(&hip_binary_ext_semaphore, &hip_binary_sem_handle_desc));

  const auto timeline_semaphore = vkt.CreateExternalSemaphore(second_semaphore_type);
  const auto hip_timeline_sem_handle_desc =
      vkt.BuildSemaphoreDescriptor(timeline_semaphore, second_semaphore_type);
  hipExternalSemaphore_t hip_timeline_ext_semaphore;
  HIP_CHECK(hipImportExternalSemaphore(&hip_timeline_ext_semaphore, &hip_timeline_sem_handle_desc));

  hipExternalSemaphoreWaitParams binary_semaphore_wait_params = {};
  binary_semaphore_wait_params.params.fence.value = 0;

  hipExternalSemaphoreWaitParams timeline_semaphore_wait_params = {};
  timeline_semaphore_wait_params.params.fence.value =
      second_semaphore_type == VK_SEMAPHORE_TYPE_TIMELINE ? 1 : 0;

  hipExternalSemaphore_t ext_semaphores[] = {hip_binary_ext_semaphore, hip_timeline_ext_semaphore};
  hipExternalSemaphoreWaitParams wait_params[] = {binary_semaphore_wait_params,
                                                  timeline_semaphore_wait_params};
  HIP_CHECK(f(ext_semaphores, wait_params, 2, nullptr));

  PollStream(nullptr, hipErrorNotReady);

  if (second_semaphore_type == VK_SEMAPHORE_TYPE_TIMELINE) {
    VkSemaphoreSignalInfo signal_info = {};
    signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signal_info.semaphore = timeline_semaphore;
    signal_info.value = 1;
    VK_CHECK_RESULT(vkSignalSemaphore(vkt.GetDevice(), &signal_info));

    PollStream(nullptr, hipErrorNotReady);
  }

  VkSubmitInfo submit_info = {};
  VkSemaphore signal_semaphores[] = {binary_semaphore, timeline_semaphore};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;
  submit_info.signalSemaphoreCount = second_semaphore_type == VK_SEMAPHORE_TYPE_TIMELINE ? 1 : 2;
  submit_info.pSignalSemaphores =
      second_semaphore_type == VK_SEMAPHORE_TYPE_MAX_ENUM ? &binary_semaphore : signal_semaphores;

  const auto fence = vkt.CreateFence();
  VK_CHECK_RESULT(vkQueueSubmit(vkt.GetQueue(), 1, &submit_info, fence));
  VK_CHECK_RESULT(
      vkWaitForFences(vkt.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000 /*5 seconds*/));

  PollStream(nullptr, hipSuccess);

  HIP_CHECK(hipDestroyExternalSemaphore(hip_timeline_ext_semaphore));
  HIP_CHECK(hipDestroyExternalSemaphore(hip_binary_ext_semaphore));
}

static inline bool operator==(const hipExternalSemaphoreWaitNodeParams& lhs,
                              const hipExternalSemaphoreWaitNodeParams& rhs) {
  bool equal = true;
  if (lhs.numExtSems != rhs.numExtSems) {
    return false;
  }
  for (unsigned int i = 0; i < lhs.numExtSems; i++) {
    if ((lhs.extSemArray[i] != rhs.extSemArray[i]) ||
        (lhs.paramsArray[i].params.fence.value != rhs.paramsArray[i].params.fence.value)) {
      equal = false;
      break;
    }
  }
  return equal;
}

template <bool set_params = false>
hipError_t GraphExtSemaphoreWaitWrapper(hipExternalSemaphore_t* extSemArray,
                                        hipExternalSemaphoreWaitParams* paramsArray,
                                        unsigned int numExtSems, hipStream_t stream) {
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t node = nullptr;
  hipExternalSemaphoreWaitNodeParams retrieved_params = {};
  memset(&retrieved_params, 0, sizeof(retrieved_params));

  hipExternalSemaphoreWaitNodeParams node_params = {};
  node_params.extSemArray = extSemArray;
  node_params.paramsArray = paramsArray;
  node_params.numExtSems = numExtSems;

  if constexpr (set_params) {
    hipExternalSemaphoreWaitParams* wait_params = new hipExternalSemaphoreWaitParams[numExtSems];
    for (unsigned int i = 0; i < numExtSems; i++) {
      wait_params[i].flags = 0;
      wait_params[i].params.fence.value = 10 + i;
    }

    hipExternalSemaphoreWaitNodeParams initial_params = {};
    initial_params.extSemArray = extSemArray;
    initial_params.paramsArray = wait_params;
    initial_params.numExtSems = numExtSems;

    HIP_CHECK(hipGraphAddExternalSemaphoresWaitNode(&node, graph, nullptr, 0, &initial_params));

    HIP_CHECK(hipGraphExternalSemaphoresWaitNodeGetParams(node, &retrieved_params));
    REQUIRE(initial_params == retrieved_params);
    HIP_CHECK(hipGraphExternalSemaphoresWaitNodeSetParams(node, &node_params));

    delete[] wait_params;
  } else {
    HIP_CHECK(hipGraphAddExternalSemaphoresWaitNode(&node, graph, nullptr, 0, &node_params));
  }

  HIP_CHECK(hipGraphExternalSemaphoresWaitNodeGetParams(node, &retrieved_params));
  REQUIRE(node_params == retrieved_params);

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));

  return hipSuccess;
}
