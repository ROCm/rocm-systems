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
#ifndef RDC_MODULES_RDC_ROCR_TESTBASE_H_
#define RDC_MODULES_RDC_ROCR_TESTBASE_H_

#include <memory>
#include <string>
#include <vector>

#include "rdc_modules/rdc_rocr/GpuOutcome.h"
#include "rdc_modules/rdc_rocr/RdcRocrBase.h"
#include "rdc_modules/rdc_rocr/RocrHandles.h"

namespace amd {
namespace rdc {

// A step of a check, bracketed in the log by two debug lines: "GPU <entity> <step>: start." when
// the step begins and "GPU <entity> <step>: done." when it ends, whatever ended it, so that the
// two lines are always a pair. The steps are the pools' allocations, the kernel loads, the
// accesses the memory checks make and each memory copy and dispatch, with its wait.
class LoggedStep {
 public:
  LoggedStep(uint32_t entity, const std::string& step);
  ~LoggedStep();

  LoggedStep(const LoggedStep&) = delete;
  LoggedStep& operator=(const LoggedStep&) = delete;

 private:
  const uint32_t entity_;
  const std::string step_;
};

class TestBase : public RdcRocrBase {
 public:
  // The GPU's entity id, which names it in a text or a log line, its KFD node,
  // which names its agent, and the outcome this object's sentences go into, which
  // outlives it: a release that fails runs in its destructor.
  TestBase(uint32_t gpu_index, uint32_t node_id, GpuOutcome* outcome);

  virtual ~TestBase(void);

  // @Brief: Takes this object's reference to ROCr. It is called from the constructor
  // and Close() from the destructor, where an override of a derived class would not
  // run, so neither is virtual.
  hsa_status_t SetUp(void);

  // @Brief: Releases this object's reference to ROCr. It never throws: a failure is
  // logged and is that GPU's failure, as a throw here would end the process.
  void Close(void);

  // The GPU agent whose KFD node is the given one, which is how a check finds the GPU
  // it was asked about: HSA lists the agents in the KFD topology's order, which is not
  // RDC's. It throws a SkipException when no agent carries that node, and the
  // iteration's failure when a call fails.
  void get_agent_by_node_id(uint32_t node_id, hsa_agent_t* agent);

  // What this sub-test found, in the sentence the GPU's report reads: a sub-test that
  // checks several things, as the System memory check does its pools, says what each of
  // them came to rather than one sentence for all of them.
  void say(rdc_diag_result_t outcome, const std::string& sentence);

  // Where this object's holders report a release that failed, and the flag they read:
  // once work on the GPU could not be stopped, nothing that work can touch is released.
  ReleaseContext* release_context(void) { return &release_context_; }
  void abandon(void) { release_context_.abandon(); }
  bool abandoned(void) const { return release_context_.abandoned(); }

  // The hint each wait carries, a tenth of a second in the ticks ROCr counts a wait's timeout
  // in, read once per test object: a frequency ROCr cannot report, or one of 0, is this GPU's
  // failure naming the call.
  uint64_t wait_hint(void);

  // What a check does just before it makes a packet ready: an error ROCr reported earlier
  // suspended the queue, so the dispatch would never complete, and that error is this GPU's
  // failure before the doorbell; the hint above is read here too, while nothing is dispatched.
  void before_doorbell(const QueueStatus* status);

  // The wait for a dispatch, which ends on an error of its queue as well, and the wait for a
  // memory copy, which runs on ROCr's own path and so ends only on its own signal. Each waits
  // in hints until the signal is below 1, until the signal reports a failure of its own, which
  // is this GPU's failure, or until the deadline passes, which abandons the work.
  void wait_for_kernel(hsa_signal_t signal, const QueueStatus* status);
  void wait_for_copy(hsa_signal_t signal);

 protected:
  uint32_t gpu_index_;  // the entity id of the GPU this object checks
  uint32_t node_id_;    // its KFD node, the agent's key
  ReleaseContext release_context_;

 private:
  // The wait both of the above are, and the error of a queue read where there is one.
  void wait_for(hsa_signal_t signal, const QueueStatus* status, bool kernel);
  void check_queue_error(const QueueStatus* status);

  uint64_t wait_hint_ = 0;  // 0 until the system timestamp frequency has been read
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCR_TESTBASE_H_
