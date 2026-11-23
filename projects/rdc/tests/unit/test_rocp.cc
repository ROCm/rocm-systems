/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

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

// The ROCp module (RdcRocpBase, its counter sampler and the module API of RdcTelemetryLib.cc)
// against the link-time fake of rocprofiler-sdk and HSA in fake_rocprofiler.cc, with the fake
// AMD SMI for the GPU table. The module and both fakes keep process-wide state, and the fake
// AMD SMI reads its knobs once per process, so each case runs in a child process.

#include <gtest/gtest.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <vector>

#include "fake_rocprofiler.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcTelemetryLibInterface.h"
#include "rdc_lib/impl/SmiUtils.h"
#include "rdc_modules/rdc_rocp/RdcRocpBase.h"

extern char** environ;

namespace {

using amd::rdc::RdcRocpBase;
using rdc_fake_rocp::Call;

// Runs body in a child process with an AMD SMI session over `gpus` fake GPUs and the fake
// profiler that config describes. The test fails unless the child exits 0; a failure of its
// own makes it exit 1, and a crash shows as its signal in the wait status.
void in_child(uint32_t gpus, const rdc_fake_rocp::Config& config,
              const std::function<void()>& body) {
  const ::testing::TestResult& result =
      *::testing::UnitTest::GetInstance()->current_test_info()->result();
  const int parts_before = result.total_part_count();
  fflush(nullptr);
  const pid_t pid = fork();
  ASSERT_NE(pid, -1) << strerror(errno);
  if (pid == 0) {
    std::vector<std::string> inherited;
    for (char** e = environ; *e != nullptr; ++e) {
      if (strncmp(*e, "RDC_FAKE_", 9) == 0) inherited.emplace_back(*e, strcspn(*e, "="));
    }
    for (const auto& name : inherited) unsetenv(name.c_str());
    setenv("RDC_FAKE_GPUS", std::to_string(gpus).c_str(), 1);
    // rocprofiler_configure() logs to this stream instead of a file in the working directory.
    setenv("ROCPROFILER_SAMPLE_OUTPUT_FILE", "stderr", 1);
    // rdc_module_init() takes the metrics path from here instead of searching for the SDK.
    setenv("ROCPROFILER_METRICS_PATH", "/nonexistent", 1);
    unsetenv("RDC_DISABLE_ROCP");
    try {
      amd::rdc::SmiSession session;
      rdc_fake_rocp::configure(config);
      body();
    } catch (const std::exception& e) {
      fprintf(stderr, "child threw: %s\n", e.what());
      fflush(nullptr);
      _exit(1);
    } catch (...) {
      fprintf(stderr, "child threw\n");
      fflush(nullptr);
      _exit(1);
    }
    bool failed = false;
    for (int i = parts_before; i < result.total_part_count(); ++i) {
      failed = failed || result.GetTestPartResult(i).failed();
    }
    fflush(nullptr);
    _exit(failed ? 1 : 0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid) << strerror(errno);
  EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << "child wait status " << status;
}

// One field of one GPU through the bulk read: its return, the field's status and its value.
struct Read {
  rdc_status_t status = RDC_ST_UNKNOWN_ERROR;
  rdc_status_t field_status = RDC_ST_UNKNOWN_ERROR;
  double value = NAN;
};

Read bulk(RdcRocpBase& rocp, uint32_t gpu, rdc_field_t field) {
  std::vector<rdc_field_value_data> values;
  std::vector<rdc_field_type_t> types;
  std::vector<rdc_status_t> statuses;
  Read r;
  r.status = rocp.rocp_lookup_bulk({{gpu, field}}, values, types, statuses);
  if (!statuses.empty()) r.field_status = statuses[0];
  if (!values.empty()) r.value = values[0].dbl;
  return r;
}

rdc_status_t lookup(RdcRocpBase& rocp, uint32_t gpu, rdc_field_t field, double* value) {
  rdc_field_value_data data = {};
  rdc_field_type_t type = INTEGER;
  const rdc_status_t status = rocp.rocp_lookup({gpu, field}, &data, &type);
  *value = data.dbl;
  return status;
}

// The first lookup starts the profiler, and the profiler gives ROCr back its reference.
TEST(Rocp, TheFirstLookupStartsTheProfiler) {
  in_child(2, {}, [] {
    {
      RdcRocpBase rocp;
      double value = 0;
      ASSERT_EQ(lookup(rocp, 0, RDC_FI_PROF_ELAPSED_CYCLES, &value), RDC_ST_OK);
      EXPECT_EQ(value, rdc_fake_rocp::value_of(0, "GRBM_COUNT"));
    }
    EXPECT_EQ(rdc_fake_rocp::hsa_inits(), 1u);
    EXPECT_EQ(rdc_fake_rocp::hsa_shut_downs(), 1u);
  });
}

// A read whose sample fails gives no value, not the previous read's, which was another GPU's.
TEST(Rocp, AFailedSampleGivesNoValueOfAnotherGpu) {
  rdc_fake_rocp::Config config;
  config.failures = {{Call::Sample, 1}};
  in_child(2, config, [] {
    RdcRocpBase rocp;
    ASSERT_EQ(bulk(rocp, 0, RDC_FI_PROF_ELAPSED_CYCLES).field_status, RDC_ST_OK);
    double value = 0;
    ASSERT_EQ(lookup(rocp, 0, RDC_FI_PROF_ELAPSED_CYCLES, &value), RDC_ST_OK);
    EXPECT_EQ(value, rdc_fake_rocp::value_of(0, "GRBM_COUNT"));
    EXPECT_NE(lookup(rocp, 1, RDC_FI_PROF_ELAPSED_CYCLES, &value), RDC_ST_OK);
    EXPECT_NE(bulk(rocp, 1, RDC_FI_PROF_ELAPSED_CYCLES).field_status, RDC_ST_OK);
  });
}

// With no GPU agent, reads fail; nothing dereferences a sampler that does not exist.
TEST(Rocp, NoGpuAgentFailsTheReads) {
  rdc_fake_rocp::Config config;
  config.agents = 0;
  in_child(2, config, [] {
    RdcRocpBase rocp;
    const Read r = bulk(rocp, 0, RDC_FI_PROF_ELAPSED_CYCLES);
    EXPECT_NE(r.status, RDC_ST_OK);
    EXPECT_NE(r.field_status, RDC_ST_OK);
    double value = 0;
    EXPECT_NE(lookup(rocp, 0, RDC_FI_PROF_ELAPSED_CYCLES, &value), RDC_ST_OK);
  });
}

// A stop that fails fails the read, and the next read stops the context before it starts it
// again, so a failed stop does not keep the counters.
TEST(Rocp, AFailedStopFailsTheReadAndTheNextReadWorks) {
  rdc_fake_rocp::Config config;
  config.failures = {{Call::Stop, 0, 1, 1}};
  in_child(1, config, [] {
    RdcRocpBase rocp;
    EXPECT_NE(bulk(rocp, 0, RDC_FI_PROF_ELAPSED_CYCLES).field_status, RDC_ST_OK);
    const Read r = bulk(rocp, 0, RDC_FI_PROF_ELAPSED_CYCLES);
    EXPECT_EQ(r.field_status, RDC_ST_OK);
    EXPECT_EQ(r.value, rdc_fake_rocp::value_of(0, "GRBM_COUNT"));
    EXPECT_FALSE(rdc_fake_rocp::any_context_active());
  });
}

// A counter whose size cannot be queried fails the read; the size is not taken from what the
// failed query left (the fake writes 3 instances where the counter has 1).
TEST(Rocp, AFailedCounterQueryFailsTheRead) {
  rdc_fake_rocp::Config config;
  config.failures = {{Call::CounterInfoV1}};
  in_child(1, config, [] {
    RdcRocpBase rocp;
    EXPECT_NE(bulk(rocp, 0, RDC_FI_PROF_ELAPSED_CYCLES).field_status, RDC_ST_OK);
    double value = 0;
    EXPECT_NE(lookup(rocp, 0, RDC_FI_PROF_ELAPSED_CYCLES, &value), RDC_ST_OK);
  });
}

// The FLOPS percent fields divide by the SIMDs per CU: a GPU that reports none has no value.
TEST(Rocp, FlopsPercentNeedsSimdsPerCu) {
  rdc_fake_rocp::Config config;
  config.simd_per_cu = 0;
  in_child(1, config, [] {
    RdcRocpBase rocp;
    EXPECT_EQ(bulk(rocp, 0, RDC_FI_PROF_EVAL_FLOPS_32_PERCENT).field_status, RDC_ST_NOT_SUPPORTED);
  });
  config.simd_per_cu = 4;
  in_child(1, config, [] {
    RdcRocpBase rocp;
    const Read r = bulk(rocp, 0, RDC_FI_PROF_EVAL_FLOPS_32_PERCENT);
    ASSERT_EQ(r.field_status, RDC_ST_OK);
    // The counter per millisecond of the 100 ms window, over 256 / 4 operations per SIMD cycle.
    EXPECT_DOUBLE_EQ(r.value,
                     rdc_fake_rocp::value_of(0, "RDC_OPS_32_PER_SIMDCYCLE") / 100.0 / 64.0);
  });
}

rdc_status_t collect(rdc_gpu_field_value_t* values, uint32_t count, void* user_data) {
  auto* out = static_cast<std::vector<rdc_gpu_field_value_t>*>(user_data);
  out->insert(out->end(), values, values + count);
  return RDC_ST_OK;
}

// The module's value get passes every field to the callback, a GPU's whose read failed with
// its status, so that the watch table records the attempt. GPU 1 has no profiler agent here.
TEST(Rocp, AGpuWhoseReadFailsReachesTheCallback) {
  rdc_fake_rocp::Config config;
  config.agents = 1;
  in_child(2, config, [] {
    ASSERT_EQ(rdc_module_init(0), RDC_ST_OK);
    rdc_gpu_field_t fields[] = {{0, RDC_FI_PROF_ELAPSED_CYCLES}, {1, RDC_FI_PROF_ELAPSED_CYCLES}};
    std::vector<rdc_gpu_field_value_t> got;
    rdc_telemetry_fields_value_get(fields, 2, collect, &got);
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[0].gpu_index, 0u);
    EXPECT_EQ(got[0].field_value.field_id, RDC_FI_PROF_ELAPSED_CYCLES);
    EXPECT_EQ(got[0].field_value.status, RDC_ST_OK);
    EXPECT_EQ(got[1].gpu_index, 1u);
    EXPECT_EQ(got[1].field_value.field_id, RDC_FI_PROF_ELAPSED_CYCLES);
    EXPECT_NE(got[1].field_value.status, RDC_ST_OK);
    EXPECT_EQ(rdc_module_destroy(), RDC_ST_OK);
  });
}

// A profiler whose hsa_init() fails fails every read with a status, throws nothing (nothing
// on the update thread's way catches), does not try again, and gives back no reference it did
// not take.
TEST(Rocp, AProfilerThatFailsToStartReportsIt) {
  rdc_fake_rocp::Config config;
  config.failures = {{Call::HsaInit}};
  in_child(1, config, [] {
    {
      RdcRocpBase rocp;
      for (int i = 0; i < 2; ++i) {
        Read r;
        EXPECT_NO_THROW(r = bulk(rocp, 0, RDC_FI_PROF_ELAPSED_CYCLES));
        EXPECT_NE(r.status, RDC_ST_OK);
        EXPECT_NE(r.field_status, RDC_ST_OK);
        double value = 0;
        rdc_status_t status = RDC_ST_OK;
        EXPECT_NO_THROW(status = lookup(rocp, 0, RDC_FI_PROF_ELAPSED_CYCLES, &value));
        EXPECT_NE(status, RDC_ST_OK);
      }
    }
    EXPECT_EQ(rdc_fake_rocp::calls(Call::HsaInit), 1u);
    EXPECT_EQ(rdc_fake_rocp::hsa_shut_downs(), 0u);
  });
}

// A profiler whose agents cannot be listed at its start (the SDK call fails, and the sampler's
// wrapper throws) fails the read with a status, throws nothing, and gives ROCr back the
// reference its hsa_init() took.
TEST(Rocp, AProfilerWhoseAgentsCannotBeListedReportsIt) {
  rdc_fake_rocp::Config config;
  config.failures = {{Call::QueryAgents, -1, 2}};  // the start's query, after tool_init()'s
  in_child(1, config, [] {
    {
      RdcRocpBase rocp;
      Read r;
      EXPECT_NO_THROW(r = bulk(rocp, 0, RDC_FI_PROF_ELAPSED_CYCLES));
      EXPECT_NE(r.status, RDC_ST_OK);
      EXPECT_NE(r.field_status, RDC_ST_OK);
    }
    EXPECT_EQ(rdc_fake_rocp::hsa_inits(), 1u);
    EXPECT_EQ(rdc_fake_rocp::hsa_shut_downs(), 1u);
  });
}

}  // namespace
