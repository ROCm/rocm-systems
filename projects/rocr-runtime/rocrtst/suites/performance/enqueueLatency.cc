/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */
#include <fcntl.h>
#include <algorithm>
#include <string>

#include "suites/performance/enqueueLatency.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/os.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

#define RET_IF_HSA_ERR(err) { \
  if ((err) != HSA_STATUS_SUCCESS) { \
    const char* msg = 0; \
    hsa_status_string(err, &msg); \
    std::cout << "hsa api call failure at line " << __LINE__ << ", file: " << \
                          __FILE__ << ". Call returned " << err << std::endl; \
    std::cout << msg << std::endl; \
    return (err); \
  } \
}

EnqueueLatency::
EnqueueLatency(bool enqueueSinglePacket) : TestBase(),
                                    enqueue_single_(enqueueSinglePacket) {
  queue_size_ = 0;
#if ROCRTST_EMULATOR_BUILD
  num_of_pkts_ = 2;
  set_num_iteration(1);
#else
  num_of_pkts_ = 100000;
  set_num_iteration(100);
#endif

  memset(&aql(), 0, sizeof(hsa_kernel_dispatch_packet_t));
  enqueue_time_mean_ = 0.0;

  std::string name;
  std::string desc;

  name = "Average Enqueue Time";
  desc = "This test measures the time when the packet enqueue to the"
      " queue and before the door bell is ring to notify the command processor "
      "to execute the packet";



  if (enqueueSinglePacket) {
    name += ", Single Packet";
    desc += " One Packet at a time in queue.";
  } else {
    name += ", Multiple Packets";
    desc += " Multiple i.e. maximum Packets equeued to queue at one time";
  }

  set_title(name);
  set_description(desc);
}

EnqueueLatency::~EnqueueLatency() {
}

void EnqueueLatency::SetUp() {
  hsa_status_t err;
  TestBase::SetUp();
  // If it indicates to use default signal, set env var properly

  err = SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
}

void EnqueueLatency::Run() {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  hsa_status_t err;
  TestBase::Run();

  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (unsigned int i = 0 ; i< gpus.size(); ++i) {
    hsa_agent_t* gpu_dev = &gpus[i];
    char agent_name[64];
    err = hsa_agent_get_info(*gpu_dev, HSA_AGENT_INFO_NAME, agent_name);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    set_agent_name(agent_name);

    // Create a queue
    hsa_queue_t* q = nullptr;
    rocrtst::CreateQueue(*gpu_dev, &q);
    ASSERT_NE(q, nullptr);
    set_main_queue(q);

    set_kernel_file_name("dispatch_time_kernels.hsaco");
    set_kernel_name("empty_kernel");
    err = rocrtst::LoadKernelFromObjFile(this, gpu_dev);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // Fill up the kernel packet except header
    err = rocrtst::InitializeAQLPacket(this, &aql());
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    aql().workgroup_size_x = 1;
    aql().grid_size_x = 1;

    // Here, modify the batch size if it is larger than the queue size
    if (enqueue_single_) {
      EnqueueSinglePacket();
    } else {
      hsa_status_t err;
      uint32_t size = 0;
      err = hsa_agent_get_info(*gpu_dev, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &size);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);

      num_of_pkts_ = num_of_pkts_ > size ? size : num_of_pkts_;
      EnqueueMultiPackets();
    }
    hsa_queue_destroy(q);
    set_main_queue(nullptr);
  }
}



size_t EnqueueLatency::RealIterationNum() {
  return num_iteration() * 1.2 + 1;
}

void EnqueueLatency::EnqueueSinglePacket() {
  std::vector<double> timer;

  int it = RealIterationNum();
  const uint32_t queue_mask = main_queue()->size - 1;

  // queue should be empty
  ASSERT_EQ(hsa_queue_load_read_index_scacquire(main_queue()),
            hsa_queue_load_write_index_scacquire(main_queue()));

  hsa_kernel_dispatch_packet_t *q_base_addr =
                      reinterpret_cast<hsa_kernel_dispatch_packet_t *>(
                                                  main_queue()->base_address);
  rocrtst::PerfTimer p_timer;
  for (int i = 0; i < it; i++) {
    // Get timing stamp and ring the doorbell to dispatch the kernel.
    int id = p_timer.CreateTimer();
    p_timer.StartTimer(id);
    // Obtain the current queue write index.
    uint64_t index = hsa_queue_add_write_index_relaxed(main_queue(), 1);

    ASSERT_LT(index, main_queue()->size + index);

    // Write the aql packet at the calculated queue index address.
    rocrtst::WriteAQLToQueueLoc(main_queue(), index, &aql());

    rocrtst::AtomicSetPacketHeader(
        HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE,
        aql().setup,
        reinterpret_cast<hsa_kernel_dispatch_packet_t *>
                                     (&(q_base_addr)[index & queue_mask]));

    p_timer.StopTimer(id);

    timer.push_back(p_timer.ReadTimer(id));
    hsa_signal_store_screlease(main_queue()->doorbell_signal, index);

    // Wait on the dispatch signal until the kernel is finished.
    while (hsa_signal_wait_scacquire(aql().completion_signal,
         HSA_SIGNAL_CONDITION_LT, 1, (uint64_t) - 1, HSA_WAIT_STATE_ACTIVE)) {
    }

    hsa_signal_store_screlease(aql().completion_signal, 1);

    if (verbosity() >= VERBOSE_PROGRESS) {
      std::cout << ".";
      fflush(stdout);
    }
  }

  if (verbosity() >= VERBOSE_PROGRESS) {
    std::cout << std::endl;
  }

  // Abandon the first result and after sort, delete the last 2% value
  timer.erase(timer.begin());
  std::sort(timer.begin(), timer.end());

  timer.erase(timer.begin() + num_iteration(), timer.end());

  enqueue_time_mean_ = rocrtst::CalcMean(timer);

  return;
}

void EnqueueLatency::EnqueueMultiPackets() {
  std::vector<double> timer;
  int it = RealIterationNum();
  const uint32_t queue_mask = main_queue()->size - 1;

  // queue should be empty
  ASSERT_EQ(hsa_queue_load_read_index_scacquire(main_queue()),
            hsa_queue_load_write_index_scacquire(main_queue()));

  rocrtst::PerfTimer p_timer;

  hsa_kernel_dispatch_packet_t *q_base_addr =
                      reinterpret_cast<hsa_kernel_dispatch_packet_t *>(
                                                  main_queue()->base_address);

  for (int i = 0; i < it; i++) {
    // Get timing stamp and ring the doorbell to dispatch the kernel.
    int id = p_timer.CreateTimer();
    p_timer.StartTimer(id);
    uint64_t* index =
           reinterpret_cast<uint64_t*>(malloc(sizeof(uint64_t) * num_of_pkts_));

    ASSERT_NE(index, nullptr);

    hsa_signal_store_screlease(aql().completion_signal, num_of_pkts_);

    for (uint32_t j = 0; j < num_of_pkts_; j++) {
      // index[j] = hsa_queue_add_write_index_scacq_screl(main_queue(), 1);
      index[j] = hsa_queue_add_write_index_relaxed(main_queue(), 1);

      // Write the aql packet at the calculated queue index address.
      rocrtst::WriteAQLToQueueLoc(main_queue(), index[j], &aql());
    }
    // Write the aql packet at the calculated queue index address.

    rocrtst::AtomicSetPacketHeader(
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
        (1 << HSA_PACKET_HEADER_BARRIER),
        aql().setup,
        reinterpret_cast<hsa_kernel_dispatch_packet_t *>
                      (&(q_base_addr)[index[num_of_pkts_ - 1] & queue_mask]));


    // Set packet header reversly; set all headers except the very first
    // one, for now.
    for (int32_t j = num_of_pkts_ - 1; j >= 0; j--) {
      rocrtst::AtomicSetPacketHeader(
          HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE,
          aql().setup,
          reinterpret_cast<hsa_kernel_dispatch_packet_t *>
                                     (&(q_base_addr)[index[j] & queue_mask]));
    }

    p_timer.StopTimer(id);

    timer.push_back(p_timer.ReadTimer(id));

    for (uint32_t j = 0; j < num_of_pkts_; j++) {
      hsa_signal_store_screlease(main_queue()->doorbell_signal, index[j]);
    }

    // Wait on the dispatch signal until the kernel is finished.
    while (hsa_signal_wait_scacquire(aql().completion_signal,
        HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) != 0) {
    }


    hsa_signal_store_screlease(aql().completion_signal, 1);

    free(index);

    if (verbosity() >= VERBOSE_PROGRESS) {
      std::cout << ".";
      fflush(stdout);
    }
  }

  std::cout << std::endl;

  // Abandon the first result and after sort, delete the last 2% value
  timer.erase(timer.begin());
  std::sort(timer.begin(), timer.end());

  timer.erase(timer.begin() + num_iteration(), timer.end());

  enqueue_time_mean_ = rocrtst::CalcMean(timer);

  return;
}



void EnqueueLatency::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void EnqueueLatency::DisplayResults(void) const {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::DisplayResults();

  std::cout << "Average Time to Completion: ";
  if (enqueue_single_) {
    std::cout << enqueue_time_mean_ * 1e6;
  } else {
    std::cout << enqueue_time_mean_ * 1e6 / num_of_pkts_;
  }

  std::cout << " uS" << std::endl;
  return;
}

void EnqueueLatency::Close() {
  TestBase::Close();
  return;
}
