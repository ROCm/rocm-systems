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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc_lib/DiagText.h"
#include "rdc_lib/RdcDiagnosticLibInterface.h"
#include "rdc_lib/impl/SmiUtils.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rocr/ComputeQueueTest.h"
#include "rdc_modules/rdc_rocr/GpuOutcome.h"
#include "rdc_modules/rdc_rocr/MemoryAccess.h"
#include "rdc_modules/rdc_rocr/MemoryTest.h"
#include "rdc_modules/rdc_rocr/RocrHandles.h"
#include "rdc_modules/rdc_rocr/common.h"

namespace {

using amd::rdc::DiagText;
using amd::rdc::GpuOutcome;

// What a check says about a GPU on which everything it ran passed, the first thing a reader needs
// about that GPU.
constexpr char kComputeQueuePassed[] = "A kernel ran on a compute queue.";
constexpr char kSysMemPassed[] =
    "Its memory pools and the CPU's and the GPU's access to memory were checked.";

// What one sub-test that passed says: what it did, which is what a reader of the report wants
// to know. The System memory check's pool sub-test says what it found pool by pool, so it has no
// sentence here.
constexpr char kCpuAccessPassed[] = "The CPU wrote and read back 8 KiB of GPU memory.";
constexpr char kGpuAccessPassed[] = "The GPU read and wrote 4 KiB of the CPU's memory.";

// The GPU could not be found, each cause with its own sentence: an entity id AMD SMI does not
// know, and an AMD SMI that cannot report the GPU's KFD node.
constexpr char kNoProcessor[] = "RDC cannot identify the GPU.";
constexpr char kNoKfdNode[] = "AMD SMI reports no KFD node for the GPU.";

// An exception that is not a std::exception says nothing about itself.
constexpr char kUnknownError[] = "An unknown error stopped the check.";

// ROCr could not be loaded for this run, which leaves no GPU to check.
constexpr char kRocrNotLoaded[] = "Fail to call hsa_init().";

// What a GPU an earlier check could not stop its work on is told. The other GPUs of such a run
// are skipped with the text below, which names every GPU in the record.
constexpr char kChecksStopped[] =
    "An earlier check could not stop its work on this GPU; the ROCr checks are disabled until "
    "the process running them restarts.";

// What the check says when it could not put its results together.
constexpr char kCheckStopped[] = "The check could not finish.";

// The reference to ROCr the run holds while it checks its GPUs, so that ROCr is loaded once for
// the check rather than once per test object, and so that nothing a test object leaves behind can
// take ROCr's count to zero in the middle of a run. It is taken before the run's first HSA call,
// which is the first agent lookup, and released when the check returns, on every path out of it.
// A load that failed is the failure of every GPU of the run that needs HSA, with no further call:
// ROCr repeats its whole load at each try, which costs seconds on a host where it cannot load at
// all, and the next run tries again. The count is the process's, so in a daemon whose profiler
// module holds a reference of its own ROCr stays loaded between checks.
class RunReference {
 public:
  hsa_status_t acquire() {
    if (reference_ == nullptr) {
      reference_.reset(new amd::rdc::RocrReference());
    }
    return reference_->status();
  }

 private:
  std::unique_ptr<amd::rdc::RocrReference> reference_;
};

// The outcome and the sentence the exception being handled gives a GPU.
void add_current_exception(GpuOutcome* outcome) {
  try {
    throw;
  } catch (const amd::rdc::SkipException& error) {
    outcome->add(RDC_DIAG_RESULT_SKIP, error.what());
  } catch (const std::exception& error) {
    outcome->add(RDC_DIAG_RESULT_FAIL, error.what());
  } catch (...) {
    outcome->add(RDC_DIAG_RESULT_FAIL, kUnknownError);
  }
}

// One sub-test, in a try of its own, so that one sub-test's skip or failure does not drop
// another's result. A sub-test that says what it found itself, pool by pool, is given no
// sentence of its own: it has already written one.
template <typename Body>
void run_sub_test(GpuOutcome* outcome, const char* passed, Body body) {
  try {
    body();
    if (passed != nullptr) {
      outcome->add(RDC_DIAG_RESULT_PASS, passed);
    }
  } catch (...) {
    add_current_exception(outcome);
  }
}

// The sub-tests of one check, on the GPU whose entity id and KFD node these are.
void run_check(rdc_diag_test_cases_t test_case, uint32_t entity, uint32_t node_id,
               GpuOutcome* outcome) {
  // Each test object writes into the outcome, which outlives it, so that a release that fails in
  // its destructor still makes that GPU's failure.
  if (test_case == RDC_DIAG_COMPUTE_QUEUE) {
    run_sub_test(outcome, kComputeQueuePassed, [entity, node_id, outcome]() {
      amd::rdc::ComputeQueueTest test(entity, node_id, outcome);
      test.RunBinarySearchTest();
    });
    return;
  }

  run_sub_test(outcome, nullptr, [entity, node_id, outcome]() {
    amd::rdc::MemoryTest test(entity, node_id, outcome);
    test.MaxSingleAllocationTest();
  });

  try {
    // The two access sub-tests share one test object, and each of them runs on its own.
    amd::rdc::MemoryAccessTest test(entity, node_id, outcome);
    run_sub_test(outcome, kCpuAccessPassed, [&test]() { test.CPUAccessToGPUMemoryTest(); });
    run_sub_test(outcome, kGpuAccessPassed, [&test]() { test.GPUAccessToCPUMemoryTest(); });
  } catch (...) {
    add_current_exception(outcome);
  }
}

// The GPUs an earlier check could not stop its work on, as the skip of the other GPUs names
// them: "GPU 1", or "GPUs 0, 1 and 2" where several are recorded.
std::string disabled_text(const std::vector<uint32_t>& entities) {
  std::string named = entities.size() > 1 ? "GPUs " : "GPU ";
  for (size_t i = 0; i < entities.size(); i++) {
    if (i > 0) {
      named += (i + 1 == entities.size()) ? " and " : ", ";
    }
    named += std::to_string(entities[i]);
  }
  return "The ROCr checks are disabled: an earlier check could not stop its work on " + named +
         "; restart the process that runs them.";
}

// One GPU's check. The record of the work a check could not stop is read first, before the
// mapping and before any HSA call, so that a run which is in flight sees it at its next GPU and
// makes no call on a GPU that is recorded. The entity id is then mapped to a KFD node, so that a
// GPU RDC cannot map costs no ROCr call, and the checks find its agent by that node: HSA lists
// the agents in the KFD topology's order, which is not RDC's.
void run_gpu(rdc_diag_test_cases_t test_case, uint32_t entity, RunReference* rocr,
             GpuOutcome* outcome) {
  const std::vector<uint32_t> disabled = amd::rdc::abandoned_gpus();
  if (!disabled.empty()) {
    const bool recorded = std::find(disabled.begin(), disabled.end(), entity) != disabled.end();
    outcome->add(recorded ? RDC_DIAG_RESULT_FAIL : RDC_DIAG_RESULT_SKIP,
                 recorded ? std::string(kChecksStopped) : disabled_text(disabled));
    return;
  }

  uint32_t node_id = 0;
  const amdsmi_status_t mapped = amd::rdc::get_kfd_node_id(entity, &node_id);
  if (mapped != AMDSMI_STATUS_SUCCESS) {
    outcome->add(RDC_DIAG_RESULT_SKIP,
                 mapped == AMDSMI_STATUS_NOT_FOUND ? kNoProcessor : kNoKfdNode);
    return;
  }

  // The run's reference to ROCr, which this GPU's agent lookup is the first call to need.
  const hsa_status_t loaded = rocr->acquire();
  if (loaded != HSA_STATUS_SUCCESS) {
    outcome->add(RDC_DIAG_RESULT_FAIL, amd::rdc::hsa_error_text(kRocrNotLoaded, loaded));
    return;
  }

  run_check(test_case, entity, node_id, outcome);
}

}  // namespace

rdc_status_t rdc_diag_init(uint64_t) { return RDC_ST_OK; }

rdc_status_t rdc_diag_destroy() { return RDC_ST_OK; }

rdc_status_t rdc_diag_test_cases_query(rdc_diag_test_cases_t test_cases[MAX_TEST_CASES],
                                       uint32_t* test_case_count) {
  if (test_case_count == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  // The caller logs the count whatever the status, so it is set before any other answer.
  *test_case_count = 0;
  if (test_cases == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  *test_case_count = 2;
  test_cases[0] = RDC_DIAG_COMPUTE_QUEUE;
  test_cases[1] = RDC_DIAG_SYS_MEM_CHECK;

  return RDC_ST_OK;
}

rdc_status_t rdc_diag_test_case_run(rdc_diag_test_cases_t test_case,
                                    uint32_t gpu_index[RDC_MAX_NUM_DEVICES], uint32_t gpu_count,
                                    const char* /*config*/, size_t /*config_size*/,
                                    rdc_diag_test_result_t* result, rdc_diag_callback_t* callback) {
  if (result == nullptr || (gpu_index == nullptr && gpu_count != 0)) {
    return RDC_ST_BAD_PARAMETER;
  }
  if (test_case != RDC_DIAG_COMPUTE_QUEUE && test_case != RDC_DIAG_SYS_MEM_CHECK) {
    return RDC_ST_OK;  // nothing is written, so the dispatcher's Skip for this check stands
  }

  *result = {};
  result->test_case = test_case;
  result->status = RDC_DIAG_RESULT_PASS;
  result->per_gpu_result_count = 0;
  if (gpu_count == 0) {
    return amd::rdc::skip_empty_group(result);
  }
  // gpu_index[] holds the ids the caller passed (the dispatcher passes at most
  // RDC_GROUP_MAX_ENTITIES); RDC_MAX_NUM_DEVICES is the bound of result->gpu_results[].
  gpu_count = std::min<uint32_t>(gpu_count, RDC_MAX_NUM_DEVICES);

  const char* const passed =
      test_case == RDC_DIAG_COMPUTE_QUEUE ? kComputeQueuePassed : kSysMemPassed;
  // One reference to ROCr for the whole check, released when this function returns, whichever
  // path it takes: with it held, no test object's hsa_shut_down() takes ROCr's count to zero.
  RunReference rocr;
  try {
    DiagText info;
    for (uint32_t i = 0; i < gpu_count; i++) {
      const uint32_t entity = gpu_index[i];
      if (callback != nullptr && callback->callback != nullptr && callback->cookie != nullptr) {
        // The GPU is named by its entity id, as its result is; an exception from the caller's
        // callback is the check's failure, not this GPU's.
        std::string line = "ROCR test on GPU " + std::to_string(entity);
        callback->callback(callback->cookie, line.data());
      }

      GpuOutcome outcome;
      run_gpu(test_case, entity, &rocr, &outcome);

      // A GPU's row is the next one written, so that the count covers the rows written and an
      // exception leaves the GPUs after it with no row rather than with a row of zeros, which
      // reads as GPU 0's result with no message.
      rdc_diag_per_gpu_result_t& row = result->gpu_results[result->per_gpu_result_count++];
      row.gpu_index = entity;
      outcome.copy_to(row.gpu_result.msg);
      info.add(outcome.summary(entity, passed));
      // Each check's result is the highest of its GPUs' outcomes, which rests on the order
      // rdc.h declares rdc_diag_result_t in.
      static_assert(RDC_DIAG_RESULT_PASS < RDC_DIAG_RESULT_SKIP &&
                        RDC_DIAG_RESULT_SKIP < RDC_DIAG_RESULT_WARN &&
                        RDC_DIAG_RESULT_WARN < RDC_DIAG_RESULT_FAIL,
                    "rdc_diag_result_t must run from Pass to Fail for std::max to take the highest "
                    "outcome");
      result->status = std::max(result->status, outcome.status());
    }
    info.copy_to(result->info);
  } catch (...) {
    // Each GPU's check is caught above; what is left here is the caller's progress callback and
    // the work of putting the texts together.
    result->status = RDC_DIAG_RESULT_FAIL;
    strncpy_with_null(result->info, kCheckStopped, MAX_DIAG_MSG_LENGTH);
  }

  return RDC_ST_OK;
}
