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

#include <algorithm>
#include <string>

#include "suites/performance/dispatch_time.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/os.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

DispatchTime::
DispatchTime(bool defaultInterrupt, bool launchSingleKernel) : TestBase(),
              use_default_interupt_(defaultInterrupt),
                                          launch_single_(launchSingleKernel) {
  queue_size_ = 0;
#ifdef ROCRTST_EMULATOR_BUILD
  num_batch_ = 2;
  set_num_iteration(1);
#else
  num_batch_ = 100000;
  set_num_iteration(100);
#endif

  memset(&aql(), 0, sizeof(hsa_kernel_dispatch_packet_t));
  dispatch_time_mean_ = 0.0;

  set_kernel_file_name("dispatch_time_kernels.hsaco");
  set_kernel_name("empty_kernel");

  std::string name;
  std::string desc;

  name = "Average Dispatch Time";
  desc = "This test measures the time to handle AQL packets that "
      "do no work. Time is measured from when the packet is made available to"
      " the Command Processor to when the target agent notifies the host that "
      "the packet has been executed.  ";

  if (defaultInterrupt) {
    name += ", Default Interrupts";
    desc += "Interrupts are controlled by HSA_ENABLE_INTERRUPT environment "
                                                                "variable. ";
  } else {
    name += ", Interrupts Enabled";
    desc += "Interrupts are enabled. ";
  }

  if (launchSingleKernel) {
    name += ", Single Kernel";
    desc += " One kernel at a time is and executed.";
  } else {
    name += ", Multiple Kernels";
    desc += " Enough kernels to fill the queue are dispatched at one time";
  }

  set_title(name);
  set_description(desc);
}

DispatchTime::~DispatchTime() {
}

void DispatchTime::SetUp() {
  hsa_status_t err;

  // This need to happen before TestBase::SetUp()
  if (use_default_interupt_) {
    set_enable_interrupt(false);
  } else {
    set_enable_interrupt(true);
  }

  TestBase::SetUp();
  // If it indicates to use default signal, set env var properly

  err = SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  hsa_agent_t* gpu_dev = gpu_device1();

  // Create a queue
  hsa_queue_t* q = nullptr;
  rocrtst::CreateQueue(*gpu_dev, &q);
  ASSERT_NE(q, nullptr);
  set_main_queue(q);

  // Here, modify the batch size if it is larger than the queue size
  if (!launch_single_) {
    hsa_status_t err;
    uint32_t size = 0;
    err = hsa_agent_get_info(*gpu_dev, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &size);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    num_batch_ = num_batch_ > size ? size : num_batch_;
  }

  err = rocrtst::LoadKernelFromObjFile(this, gpu_dev);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Fill up the kernel packet except header
  err = rocrtst::InitializeAQLPacket(this, &aql());
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  aql().workgroup_size_x = 1;
  aql().grid_size_x = 1;
}

void DispatchTime::Run() {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
  if (launch_single_) {
    RunSingle();
  } else {
    RunMulti();
  }
}

size_t DispatchTime::RealIterationNum() {
  return num_iteration() * 1.2 + 1;
}

void DispatchTime::RunSingle() {
  std::vector<double> timer;

  uint32_t it = RealIterationNum();
  const uint32_t queue_mask = main_queue()->size - 1;

  // queue should be empty
  ASSERT_EQ(hsa_queue_load_read_index_scacquire(main_queue()),
            hsa_queue_load_write_index_scacquire(main_queue()));

  hsa_kernel_dispatch_packet_t *q_base_addr =
      reinterpret_cast<hsa_kernel_dispatch_packet_t *>
                                                 (main_queue()->base_address);

  if (it > main_queue()->size) {
    it = main_queue()->size;
  }
  for (uint32_t i = 0; i < it; i++) {
    // Obtain the current queue write index.
    uint64_t index = hsa_queue_add_write_index_relaxed(main_queue(), 1);

    // Write the aql packet at the calculated queue index address.
    rocrtst::WriteAQLToQueueLoc(main_queue(), index, &aql());

    // Get timing stamp and ring the doorbell to dispatch the kernel.
    rocrtst::PerfTimer p_timer;
    int id = p_timer.CreateTimer();
    p_timer.StartTimer(id);

    rocrtst::AtomicSetPacketHeader(
        HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE,
        aql().setup,
        reinterpret_cast<hsa_kernel_dispatch_packet_t *>
                                        (&(q_base_addr)[index & queue_mask]));

    hsa_signal_store_screlease(main_queue()->doorbell_signal, index);

    // Wait on the dispatch signal until the kernel is finished.
    while (hsa_signal_wait_scacquire(aql().completion_signal,
         HSA_SIGNAL_CONDITION_LT, 1, (uint64_t) - 1, HSA_WAIT_STATE_ACTIVE)) {
    }

    p_timer.StopTimer(id);

    timer.push_back(p_timer.ReadTimer(id));
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

  dispatch_time_mean_ = rocrtst::CalcMean(timer);

  return;
}

void DispatchTime::RunMulti() {
  std::vector<double> timer;
  int it = RealIterationNum();
  const uint32_t queue_mask = main_queue()->size - 1;
  hsa_kernel_dispatch_packet_t *q_base_addr =
      reinterpret_cast<hsa_kernel_dispatch_packet_t *>
                                                 (main_queue()->base_address);

  // queue should be empty
  ASSERT_EQ(hsa_queue_load_read_index_scacquire(main_queue()),
            hsa_queue_load_write_index_scacquire(main_queue()));

  rocrtst::PerfTimer p_timer;

  for (int i = 0; i < it; i++) {
    uint64_t* index =
           reinterpret_cast<uint64_t*>(malloc(sizeof(uint64_t) * num_batch_));

    ASSERT_NE(index, nullptr);

    hsa_signal_store_screlease(aql().completion_signal, num_batch_);

    for (uint32_t j = 0; j < num_batch_; j++) {
      // index[j] = hsa_queue_add_write_index_scacq_screl(main_queue(), 1);
      index[j] = hsa_queue_add_write_index_relaxed(main_queue(), 1);

      // Write the aql packet at the calculated queue index address.
      rocrtst::WriteAQLToQueueLoc(main_queue(), index[j], &aql());
    }

    rocrtst::AtomicSetPacketHeader(
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
        (1 << HSA_PACKET_HEADER_BARRIER),
        aql().setup,
        reinterpret_cast<hsa_kernel_dispatch_packet_t *>
                          (&q_base_addr[index[num_batch_ - 1] & queue_mask]));

    // Set packet header reversly; set all headers except the very first
    // one, for now.
    for (uint32_t j = num_batch_ - 1; j > 0; j--) {
      rocrtst::AtomicSetPacketHeader(
          HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE,
          aql().setup,
          reinterpret_cast<hsa_kernel_dispatch_packet_t *>
                      (&q_base_addr[index[j] & queue_mask]));
    }

    // Get timing stamp and ring the doorbell to dispatch the kernel.
    int id = p_timer.CreateTimer();
    p_timer.StartTimer(id);
    // Set the very first header...
    rocrtst::AtomicSetPacketHeader(
        HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE,
        aql().setup,
        reinterpret_cast<hsa_kernel_dispatch_packet_t *>
                                     (&(q_base_addr)[index[0] & queue_mask]));

    hsa_signal_store_screlease(main_queue()->doorbell_signal, index[num_batch_ - 1]);

    // Wait on the dispatch signal until the kernel is finished.
    while (hsa_signal_wait_scacquire(aql().completion_signal,
        HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) != 0) {
    }

    p_timer.StopTimer(id);

    timer.push_back(p_timer.ReadTimer(id));
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

  dispatch_time_mean_ = rocrtst::CalcMean(timer);

  return;
}

void DispatchTime::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void DispatchTime::DisplayResults(void) const {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::DisplayResults();

  std::cout << "Average Time to Completion: ";
  if (launch_single_) {
    std::cout << dispatch_time_mean_ * 1e6;
  } else {
    std::cout << dispatch_time_mean_ * 1e6 / num_batch_;
  }

  std::cout << " uS" << std::endl;
  return;
}

void DispatchTime::Close() {
  TestBase::Close();
  return;
}
