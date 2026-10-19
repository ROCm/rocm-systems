/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include "rdc_modules/rdc_rocr/TestBase.h"

#include <assert.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <string>

#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rocr/base_rocr_utils.h"

namespace amd {
namespace rdc {

static const char kLabelDelimiter[] = "####";
static const char kSetupLabel[] = "TEST SETUP";
static const char kCloseLabel[] = "TEST CLEAN UP";

LoggedStep::LoggedStep(uint32_t entity, const std::string& step) : entity_(entity), step_(step) {
  RDC_LOG(RDC_DEBUG, "GPU " << entity_ << " " << step_ << ": start.");
}

LoggedStep::~LoggedStep() {
  // A destructor must not throw: the line is put together inside a catch-all and dropped rather
  // than ending the process when it cannot be.
  try {
    RDC_LOG(RDC_DEBUG, "GPU " << entity_ << " " << step_ << ": done.");
  } catch (...) {
  }
}

void TestBase::say(rdc_diag_result_t outcome, const std::string& sentence) {
  GpuOutcome* const report = release_context_.outcome();
  if (report != nullptr) {
    report->add(outcome, sentence);
  }
}

TestBase::TestBase(uint32_t gpu_index, uint32_t node_id, GpuOutcome* outcome)
    : gpu_index_(gpu_index), node_id_(node_id), release_context_(gpu_index, outcome) {
  // SetUp()'s hsa_init() is the last step of this constructor, and nothing that can throw may
  // follow it: a throw from here would skip ~TestBase, and with it the hsa_shut_down() that
  // releases the reference, while a throw from a derived constructor or from a test body runs it.
  throw_if_error(SetUp(), "Fail to call hsa_init().");
}
TestBase::~TestBase() { Close(); }

static void MakeHeaderStr(const char* inStr, std::string* outStr) {
  assert(outStr != nullptr);
  assert(inStr != nullptr);

  outStr->clear();
  *outStr = kLabelDelimiter;
  *outStr += " ";
  *outStr += inStr;
  *outStr += " ";
  *outStr += kLabelDelimiter;
}

hsa_status_t TestBase::SetUp(void) {
  std::string label;
  MakeHeaderStr(kSetupLabel, &label);
  RDC_LOG(RDC_DEBUG, label);

  // This object's own reference to ROCr. The run holds one more while it checks its GPUs, so
  // ROCr stays loaded between the test objects of a check.
  return hsa_init();
}

void TestBase::Close(void) {
  // This runs from ~TestBase, which must not throw: the label is put together inside a
  // catch-all and dropped rather than ending the process when it cannot be.
  try {
    std::string label;
    MakeHeaderStr(kCloseLabel, &label);
    RDC_LOG(RDC_DEBUG, label);
  } catch (...) {
  }

  if (abandoned()) {
    // Work on the GPU could not be stopped, and its queue is still alive: this reference is kept
    // on purpose, as a shutdown that took ROCr's count to zero under that queue would make every
    // later hsa_init() of the process fail.
    return;
  }

  const hsa_status_t err = hsa_shut_down();
  if (err != HSA_STATUS_SUCCESS) {
    // A destructor must not throw: the failure is logged and is that GPU's failure.
    release_context_.release_failed("hsa_shut_down", err);
  }
}

uint64_t TestBase::wait_hint(void) {
  if (wait_hint_ != 0) {
    return wait_hint_;
  }
  uint64_t frequency = 0;
  throw_if_error(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency),
                 "Fail to call hsa_system_get_info().");
  if (frequency == 0) {
    throw_failure("ROCr reports a system timestamp frequency of 0.");
  }
  // A tenth of a second, in the unit a wait's timeout is given in, and never 0: a
  // frequency below ten ticks a second would leave a hint of no ticks, which is a wait that
  // returns at once and spins to the deadline.
  wait_hint_ = std::max<uint64_t>(frequency / 10, 1);
  return wait_hint_;
}

void TestBase::check_queue_error(const QueueStatus* status) {
  if (status == nullptr) {
    return;
  }
  const hsa_status_t reported = status->load(std::memory_order_acquire);
  if (reported != HSA_STATUS_SUCCESS) {
    throw_if_error(reported, "The queue reported an error.");
  }
}

void TestBase::before_doorbell(const QueueStatus* status) {
  // The hint of the waits to come is read while this check has dispatched nothing, so that a
  // frequency ROCr cannot report is this GPU's failure rather than a wait without a bound.
  (void)wait_hint();
  check_queue_error(status);
}

void TestBase::wait_for_kernel(hsa_signal_t signal, const QueueStatus* status) {
  wait_for(signal, status, true);
}

void TestBase::wait_for_copy(hsa_signal_t signal) { wait_for(signal, nullptr, false); }

void TestBase::wait_for(hsa_signal_t signal, const QueueStatus* status, bool kernel) {
  const uint64_t hint = wait_hint();
  const uint64_t deadline = wait_deadline_ms();
  const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();

  for (;;) {
    // An error ROCr reported suspends the queue, so what was put on it never completes: it is
    // this GPU's failure, with everything this check made released as it would be.
    check_queue_error(status);

    const hsa_signal_value_t value =
        hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, hint, HSA_WAIT_STATE_BLOCKED);
    if (value == 0) {
      return;
    }
    if (value < 0) {
      // ROCr reports an operation that failed by a negative value on its signal; a value of 1 or
      // more is the hint running out, which the loop takes up to the deadline.
      throw_failure(std::string(kernel ? "The kernel" : "The memory copy") +
                    " reported a failure through its signal, value " + std::to_string(value) + ".");
    }

    const uint64_t waited =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - started)
                                  .count());
    if (waited >= deadline) {
      // The work on the GPU could not be stopped: nothing it can touch is released, this object
      // keeps its reference to ROCr, and the GPU is recorded, which disables the checks.
      abandon();
      throw_failure(std::string(kernel ? "The kernel did not complete in "
                                       : "A memory copy did not complete in ") +
                    wait_deadline_text(deadline) + ".");
    }
  }
}

void TestBase::get_agent_by_node_id(uint32_t node_id, hsa_agent_t* agent) {
  std::vector<hsa_agent_t> gpus;
  hsa_status_t err = hsa_iterate_agents(IterateGPUAgents, &gpus);
  throw_if_error(err, "Fail to iterate agents.");
  for (const hsa_agent_t& gpu : gpus) {
    uint32_t node = 0;
    err = hsa_agent_get_info(gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
                             &node);
    throw_if_error(err, "Fail to read the KFD node of an agent.");
    if (node == node_id) {
      *agent = gpu;
      return;
    }
  }
  throw_if_skip("ROCr does not list the GPU.");
}

}  // namespace rdc
}  // namespace amd
