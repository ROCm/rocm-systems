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

// The three AMD SMI checks of `rdci diag` (compute processes, node topology, GPU
// parameters) and the module that runs the diagnostic levels:
// - topology pairs each GPU with the others, names GPUs by index, and skips when
//   it has no pair to check;
// - GPU parameters reports each GPU under its own entry, reads the critical
//   voltage limits with the GPU's handle, compares no limit that reads as 0, and
//   skips a GPU that compared nothing;
// - compute processes asks AMD SMI for each process's GPUs with a real list
//   (AMD SMI rejects a NULL one), retries a process list that grew between its
//   count and its fill, bounds the count, reports only the group's GPUs, and
//   leaves out the process that runs it;
// - every text is cut after a whole sentence or entry and says how many are left out;
// - a group without GPUs, and a module that returns without a result, give Skip;
// - the module bounds the test cases it gathers and the group size, and logs a
//   custom-config run at debug level only.
//
// This binary builds RdcSmiDiagnosticImpl.cc and RdcDiagnosticModule.cc itself,
// with the AMD SMI calls they make replaced by the fakes below, so it needs
// neither a GPU nor librdc. It uses no symbol the files did not have before, so
// it also builds against the old code. It is linked with --wrap=getpid (see
// kOwnPid).

#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcDiagnostic.h"
#include "rdc_lib/impl/RdcDiagnosticModule.h"
#include "rdc_lib/impl/RdcSmiDiagnosticImpl.h"
#include "rdc_lib/impl/SmiUtils.h"

namespace {

// ---------------------------------------------------------------------------
// The fake AMD SMI
// ---------------------------------------------------------------------------

// One entry per GPU index; the fake processor handle of GPU i points at entry i.
struct FakeGpu {
  bool available = true;  // false: the handle lookup fails with NOT_FOUND
};
std::vector<FakeGpu> g_gpus;

int gpu_of(amdsmi_processor_handle handle) {
  for (size_t i = 0; i < g_gpus.size(); ++i) {
    if (handle == &g_gpus[i]) return static_cast<int>(i);
  }
  return -1;
}

// What one metric read returns. A key with no entry reads as NOT_SUPPORTED.
struct Reading {
  amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
  int64_t value = 0;
  bool writes = true;  // false: succeed without writing the output (a broken library)
};
using MetricKey = std::tuple<int, int, int>;  // GPU, sensor or voltage type, metric
std::map<MetricKey, Reading> g_temps;
std::map<MetricKey, Reading> g_volts;
int g_volt_calls_without_handle = 0;

// Every relevant sensor reads like the fake GPUs of the e2e suite: current 50,
// maximum 100, critical 105, emergency 110 (no minimum, no critical minimum);
// Vddgfx current 800, maximum 1100, minimum 700, critical maximum 1200,
// critical minimum 650. That is 3 x 3 + 4 = 13 limits.
constexpr int kLimitsPerHealthyGpu = 13;
const amdsmi_temperature_type_t kSensors[] = {
    AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMPERATURE_TYPE_JUNCTION, AMDSMI_TEMPERATURE_TYPE_VRAM};

void set_temp(int gpu, amdsmi_temperature_type_t sensor, amdsmi_temperature_metric_t metric,
              Reading r) {
  g_temps[MetricKey(gpu, sensor, metric)] = r;
}
void set_volt(int gpu, amdsmi_voltage_metric_t metric, Reading r) {
  g_volts[MetricKey(gpu, AMDSMI_VOLT_TYPE_VDDGFX, metric)] = r;
}
Reading value(int64_t v) { return Reading{AMDSMI_STATUS_SUCCESS, v, true}; }
Reading failing() { return Reading{AMDSMI_STATUS_UNKNOWN_ERROR, 0, false}; }

void healthy_sensors(int gpu) {
  for (auto sensor : kSensors) {
    set_temp(gpu, sensor, AMDSMI_TEMP_CURRENT, value(50));
    set_temp(gpu, sensor, AMDSMI_TEMP_MAX, value(100));
    set_temp(gpu, sensor, AMDSMI_TEMP_CRITICAL, value(105));
    set_temp(gpu, sensor, AMDSMI_TEMP_EMERGENCY, value(110));
  }
  set_volt(gpu, AMDSMI_VOLT_CURRENT, value(800));
  set_volt(gpu, AMDSMI_VOLT_MAX, value(1100));
  set_volt(gpu, AMDSMI_VOLT_MIN, value(700));
  set_volt(gpu, AMDSMI_VOLT_MAX_CRIT, value(1200));
  set_volt(gpu, AMDSMI_VOLT_MIN_CRIT, value(650));
}

// Topology: every pair's weight is 15; the calls are recorded by GPU index.
std::vector<std::pair<int, int>> g_weight_calls;
std::pair<int, int> g_failing_pair{-1, -1};

// The process list: what a count query reports and what a fill writes, in the
// way rocm_smi does (a list without room for every process gets the process IDs
// only, keeps *num_items and returns INSUFFICIENT_SIZE).
struct ProcessList {
  std::vector<amdsmi_process_info_t> procs;
  std::vector<uint32_t> add_before_call;  // processes added before process-list call k
  int64_t count_report = -1;              // what a count query reports instead
  amdsmi_status_t count_status = AMDSMI_STATUS_SUCCESS;  // a count query that fails this way
  amdsmi_status_t fill_status = AMDSMI_STATUS_SUCCESS;   // a fill that fails this way
  int64_t fill_report = -1;  // what a successful fill reports instead of the processes written
};
ProcessList g_plist;
int g_plist_calls = 0;
std::vector<uint32_t> g_fill_capacities;
uint32_t g_next_pid = 5000;

// A process's GPUs. A pid without an entry gets NOT_FOUND.
struct GpuList {
  amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
  std::vector<uint32_t> gpus;  // the entries written (up to the room given)
  int64_t report = -1;         // the count a SUCCESS reports instead of gpus.size()
};
std::map<uint32_t, GpuList> g_gpu_lists;
int g_gpu_list_calls_without_list = 0;

// The compute process check leaves out the process that runs it, getpid(). The
// binary is linked with --wrap=getpid, so getpid() in the code built into it
// returns kOwnPid: no fake process here has that ID, whatever the binary's own
// ID is, and no Linux process can (IDs stay below PID_MAX_LIMIT, 4194304).
constexpr pid_t kOwnPid = 4194304;

amdsmi_process_info_t make_process(uint32_t pid) {
  amdsmi_process_info_t p = {};
  p.process_id = pid;
  p.vram_usage = 4096;
  p.sdma_usage = 0;
  p.cu_occupancy = 1;
  return p;
}

// A process on the given GPUs.
void add_process(uint32_t pid, std::vector<uint32_t> gpus) {
  g_plist.procs.push_back(make_process(pid));
  g_gpu_lists[pid].gpus = std::move(gpus);
}

void reset_fake(size_t gpus) {
  g_gpus.assign(gpus, FakeGpu{});
  g_temps.clear();
  g_volts.clear();
  g_volt_calls_without_handle = 0;
  for (size_t i = 0; i < gpus; ++i) healthy_sensors(static_cast<int>(i));
  g_weight_calls.clear();
  g_failing_pair = {-1, -1};
  g_plist = ProcessList{};
  g_plist_calls = 0;
  g_fill_capacities.clear();
  g_next_pid = 5000;
  g_gpu_lists.clear();
  g_gpu_list_calls_without_list = 0;
}

// ---------------------------------------------------------------------------
// Running the checks
// ---------------------------------------------------------------------------

using CheckFn = rdc_status_t (amd::rdc::RdcSmiDiagnosticImpl::*)(uint32_t*, uint32_t,
                                                                 rdc_diag_test_result_t*);

// What the last check that ran returned.
rdc_status_t g_check_status = RDC_ST_OK;

// Runs one check over `group` and keeps what it returns in g_check_status. The
// result starts out as junk (0xA5), so every field the test reads must have been
// written by the check.
std::unique_ptr<rdc_diag_test_result_t> run_check(CheckFn check,
                                                  const std::vector<uint32_t>& group) {
  auto result = std::make_unique<rdc_diag_test_result_t>();
  std::memset(result.get(), 0xA5, sizeof(*result));
  // The RDC_MAX_NUM_DEVICES entries the checks' parameter declares, or more for a
  // test of a larger group.
  std::vector<uint32_t> ids(std::max<size_t>(group.size(), RDC_MAX_NUM_DEVICES));
  std::copy(group.begin(), group.end(), ids.begin());
  amd::rdc::RdcSmiDiagnosticImpl impl;
  g_check_status = (impl.*check)(ids.data(), static_cast<uint32_t>(group.size()), result.get());
  return result;
}

// A result followed by room for two more per-GPU entries, all 0xA5: a check that
// writes past gpu_results changes the room instead of memory the test does not own.
struct PaddedResult {
  rdc_diag_test_result_t result;
  rdc_diag_per_gpu_result_t room[2];
};

std::unique_ptr<PaddedResult> run_padded(CheckFn check, const std::vector<uint32_t>& group) {
  auto padded = std::make_unique<PaddedResult>();
  std::memset(padded.get(), 0xA5, sizeof(*padded));
  std::vector<uint32_t> ids(std::max<size_t>(group.size(), RDC_MAX_NUM_DEVICES));
  std::copy(group.begin(), group.end(), ids.begin());
  amd::rdc::RdcSmiDiagnosticImpl impl;
  g_check_status = (impl.*check)(ids.data(), static_cast<uint32_t>(group.size()), &padded->result);
  return padded;
}

bool room_untouched(const PaddedResult& padded) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(padded.room);
  return std::all_of(bytes, bytes + sizeof(padded.room), [](unsigned char b) { return b == 0xA5; });
}

std::unique_ptr<rdc_diag_test_result_t> topology(const std::vector<uint32_t>& group) {
  return run_check(&amd::rdc::RdcSmiDiagnosticImpl::check_smi_topo_info, group);
}
std::unique_ptr<rdc_diag_test_result_t> parameters(const std::vector<uint32_t>& group) {
  return run_check(&amd::rdc::RdcSmiDiagnosticImpl::check_smi_param_info, group);
}
std::unique_ptr<rdc_diag_test_result_t> processes(const std::vector<uint32_t>& group) {
  return run_check(&amd::rdc::RdcSmiDiagnosticImpl::check_smi_process_info, group);
}

std::string text(const char (&s)[MAX_DIAG_MSG_LENGTH]) {
  return std::string(s, strnlen(s, MAX_DIAG_MSG_LENGTH));
}
std::string info(const rdc_diag_test_result_t& r) { return text(r.info); }

// The per-GPU entries as (GPU index, message); fails the test on a count over 128.
std::vector<std::pair<uint32_t, std::string>> entries(const rdc_diag_test_result_t& r) {
  std::vector<std::pair<uint32_t, std::string>> out;
  EXPECT_LE(r.per_gpu_result_count, static_cast<uint32_t>(RDC_MAX_NUM_DEVICES));
  for (uint32_t i = 0; i < std::min<uint32_t>(r.per_gpu_result_count, RDC_MAX_NUM_DEVICES); ++i) {
    out.emplace_back(r.gpu_results[i].gpu_index, text(r.gpu_results[i].gpu_result.msg));
  }
  return out;
}

std::vector<uint32_t> first_gpus(uint32_t n) {
  std::vector<uint32_t> group(n);
  for (uint32_t i = 0; i < n; ++i) group[i] = i;
  return group;
}

bool ends_with(const std::string& s, const std::string& end) {
  return s.size() >= end.size() && s.compare(s.size() - end.size(), end.size(), end) == 0;
}

// The " (N more not shown)" a cut text ends with; -1 when there is none.
long more_not_shown(const std::string& s) {
  const std::string tail = " more not shown)";
  if (!ends_with(s, tail)) return -1;
  size_t open = s.rfind(" (");
  if (open == std::string::npos) return -1;
  return std::stol(s.substr(open + 2, s.size() - tail.size() - open - 2));
}

}  // namespace

// getpid() in the code built into this binary (see kOwnPid). Declared first, as
// no header declares it.
extern "C" pid_t __wrap_getpid();
extern "C" pid_t __wrap_getpid() { return kOwnPid; }

// The AMD SMI helpers RdcSmiDiagnosticImpl.cc uses from librdc.
namespace amd {
namespace rdc {
amdsmi_status_t get_processor_handle_from_id(uint32_t gpu_id,
                                             amdsmi_processor_handle* processor_handle) {
  if (gpu_id >= g_gpus.size() || !g_gpus[gpu_id].available) {
    return AMDSMI_STATUS_NOT_FOUND;
  }
  *processor_handle = &g_gpus[gpu_id];
  return AMDSMI_STATUS_SUCCESS;
}
rdc_status_t Smi2RdcError(amdsmi_status_t status) {
  return status == AMDSMI_STATUS_SUCCESS ? RDC_ST_OK : RDC_ST_SMI_ERROR;
}
}  // namespace rdc
}  // namespace amd

amdsmi_status_t amdsmi_get_temp_metric(amdsmi_processor_handle processor_handle,
                                       amdsmi_temperature_type_t sensor_type,
                                       amdsmi_temperature_metric_t metric, int64_t* temperature) {
  const int gpu = gpu_of(processor_handle);
  if (gpu < 0 || temperature == nullptr) return AMDSMI_STATUS_INVAL;
  auto it = g_temps.find(MetricKey(gpu, sensor_type, metric));
  if (it == g_temps.end()) return AMDSMI_STATUS_NOT_SUPPORTED;
  if (it->second.writes) *temperature = it->second.value;
  return it->second.status;
}

amdsmi_status_t amdsmi_get_gpu_volt_metric(amdsmi_processor_handle processor_handle,
                                           amdsmi_voltage_type_t sensor_type,
                                           amdsmi_voltage_metric_t metric, int64_t* voltage) {
  if (processor_handle == nullptr) {
    ++g_volt_calls_without_handle;
    return AMDSMI_STATUS_INVAL;  // as AMD SMI does
  }
  const int gpu = gpu_of(processor_handle);
  if (gpu < 0 || voltage == nullptr) return AMDSMI_STATUS_INVAL;
  auto it = g_volts.find(MetricKey(gpu, sensor_type, metric));
  if (it == g_volts.end()) return AMDSMI_STATUS_NOT_SUPPORTED;
  if (it->second.writes) *voltage = it->second.value;
  return it->second.status;
}

amdsmi_status_t amdsmi_topo_get_link_weight(amdsmi_processor_handle processor_handle_src,
                                            amdsmi_processor_handle processor_handle_dst,
                                            uint64_t* weight) {
  const int src = gpu_of(processor_handle_src);
  const int dst = gpu_of(processor_handle_dst);
  g_weight_calls.emplace_back(src, dst);
  if (src < 0 || dst < 0 || weight == nullptr) return AMDSMI_STATUS_INVAL;
  if (std::make_pair(src, dst) == g_failing_pair) return AMDSMI_STATUS_UNKNOWN_ERROR;
  *weight = 15;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_compute_process_info(amdsmi_process_info_t* procs,
                                                    uint32_t* num_items) {
  if (num_items == nullptr) return AMDSMI_STATUS_INVAL;
  const size_t call = static_cast<size_t>(g_plist_calls++);
  if (call < g_plist.add_before_call.size()) {
    for (uint32_t k = 0; k < g_plist.add_before_call[call]; ++k) {
      add_process(g_next_pid++, {0});
    }
  }
  const uint32_t found = static_cast<uint32_t>(g_plist.procs.size());
  if (procs == nullptr) {
    if (g_plist.count_status != AMDSMI_STATUS_SUCCESS) return g_plist.count_status;
    *num_items = g_plist.count_report >= 0 ? static_cast<uint32_t>(g_plist.count_report) : found;
    return AMDSMI_STATUS_SUCCESS;
  }
  const uint32_t room = *num_items;
  g_fill_capacities.push_back(room);
  if (g_plist.fill_status != AMDSMI_STATUS_SUCCESS) return g_plist.fill_status;
  if (room < found) {
    for (uint32_t i = 0; i < room; ++i) {
      procs[i] = amdsmi_process_info_t{};
      procs[i].process_id = g_plist.procs[i].process_id;  // the process IDs only
    }
    return AMDSMI_STATUS_INSUFFICIENT_SIZE;
  }
  for (uint32_t i = 0; i < found; ++i) procs[i] = g_plist.procs[i];
  *num_items = g_plist.fill_report >= 0 ? static_cast<uint32_t>(g_plist.fill_report) : found;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_compute_process_gpus(uint32_t pid, uint32_t* dv_indices,
                                                    uint32_t* num_devices) {
  if (num_devices == nullptr) return AMDSMI_STATUS_INVAL;
  if (dv_indices == nullptr) {
    ++g_gpu_list_calls_without_list;
    return AMDSMI_STATUS_INVAL;  // as every shipped AMD SMI does
  }
  auto it = g_gpu_lists.find(pid);
  if (it == g_gpu_lists.end()) return AMDSMI_STATUS_NOT_FOUND;
  const GpuList& list = it->second;
  const uint32_t room = *num_devices;
  if (list.status == AMDSMI_STATUS_SUCCESS || list.status == AMDSMI_STATUS_INSUFFICIENT_SIZE) {
    const size_t n = std::min<size_t>(room, list.gpus.size());
    for (size_t i = 0; i < n; ++i) dv_indices[i] = list.gpus[i];
  }
  if (list.status == AMDSMI_STATUS_SUCCESS) {
    *num_devices = list.report >= 0 ? static_cast<uint32_t>(list.report)
                                    : static_cast<uint32_t>(list.gpus.size());
  }
  // INSUFFICIENT_SIZE keeps the room in *num_devices, as rocm_smi does; other
  // failures write nothing and keep it too, as ROCm 7.3.0's AMD SMI does.
  return list.status;
}

// ---------------------------------------------------------------------------
// Node topology
// ---------------------------------------------------------------------------

TEST(SmiDiagTopology, PairsEachGpuWithTheOthersAndNamesThemByIndex) {
  reset_fake(3);
  auto r = topology({1, 2});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_PASS);
  EXPECT_EQ(g_weight_calls, (std::vector<std::pair<int, int>>{{1, 2}, {2, 1}}));
  EXPECT_EQ(info(*r), "1=>2 weight:15 2=>1 weight:15");
}

TEST(SmiDiagTopology, OneGpuHasNoPairToCheck) {
  reset_fake(2);
  auto r = topology({0});
  EXPECT_EQ(g_check_status, RDC_ST_OK);
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_SKIP);
  EXPECT_EQ(info(*r), "No pair of available GPUs to check.");
}

TEST(SmiDiagTopology, NoAvailablePairToCheck) {
  reset_fake(2);
  g_gpus[1].available = false;
  auto r = topology({0, 1});
  EXPECT_EQ(g_check_status, RDC_ST_OK);
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_SKIP);
  EXPECT_EQ(info(*r), "No pair of available GPUs to check.");
  EXPECT_TRUE(g_weight_calls.empty());
}

TEST(SmiDiagTopology, AWeightErrorNamesBothGpus) {
  reset_fake(3);
  g_failing_pair = {2, 1};
  auto r = topology({1, 2});
  EXPECT_EQ(g_check_status, RDC_ST_SMI_ERROR);
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_EQ(r->details.code, static_cast<uint32_t>(AMDSMI_STATUS_UNKNOWN_ERROR));
  EXPECT_EQ(text(r->details.msg), "amdsmi_topo_get_link_weight(2,1, &weight) fail");
  EXPECT_EQ(info(*r), "amdsmi_topo_get_link_weight(2,1, &weight) fail");
}

TEST(SmiDiagTopology, AGroupLargerThanTheParameterIsCut) {
  // Hardening: gpu_index holds RDC_MAX_NUM_DEVICES entries.
  reset_fake(RDC_MAX_NUM_DEVICES + 2);
  auto padded = run_padded(&amd::rdc::RdcSmiDiagnosticImpl::check_smi_topo_info,
                           first_gpus(RDC_MAX_NUM_DEVICES + 2));
  EXPECT_TRUE(room_untouched(*padded));
  size_t beyond = 0;
  for (const auto& call : g_weight_calls) {
    if (call.first >= RDC_MAX_NUM_DEVICES || call.second >= RDC_MAX_NUM_DEVICES) ++beyond;
  }
  EXPECT_FALSE(g_weight_calls.empty());
  EXPECT_EQ(beyond, 0u);
}

TEST(SmiDiagTopology, SixtyFourGpusAreCutAfterAWholePair) {
  reset_fake(64);
  auto r = topology(first_gpus(64));
  const std::string s = info(*r);
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_PASS);
  EXPECT_LE(s.size(), static_cast<size_t>(MAX_DIAG_MSG_LENGTH - 1));
  const long dropped = more_not_shown(s);
  ASSERT_GT(dropped, 0) << s.substr(s.size() > 80 ? s.size() - 80 : 0);
  // The kept text is whole pairs: "i=>j weight:15" units joined by single spaces.
  const std::string kept = s.substr(0, s.rfind(" ("));
  size_t pairs = 0;
  for (size_t pos = 0; pos < kept.size();) {
    size_t end = kept.find(' ', kept.find("weight:", pos));
    std::string unit = kept.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    unsigned a = 0, b = 0, w = 0;
    char tail = 0;
    ASSERT_EQ(std::sscanf(unit.c_str(), "%u=>%u weight:%u%c", &a, &b, &w, &tail), 3) << unit;
    EXPECT_EQ(w, 15u);
    ++pairs;
    pos = end == std::string::npos ? kept.size() : end + 1;
  }
  EXPECT_EQ(pairs + static_cast<size_t>(dropped), 64u * 63u);
}

// ---------------------------------------------------------------------------
// GPU parameters
// ---------------------------------------------------------------------------

TEST(SmiDiagParameters, EachGpuHasItsOwnEntryAndCountsItsLimits) {
  // GPUs 1 and 2 of three, so that entry i's index and text are GPU i's, not i.
  reset_fake(3);
  set_temp(2, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, value(101));
  auto r = parameters({1, 2});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_WARN);
  EXPECT_EQ(entries(*r),
            (std::vector<std::pair<uint32_t, std::string>>{
                {1, ""}, {2, "Edge temperature 101 is at or above its maximum 100."}}));
  EXPECT_EQ(info(*r), "GPU 1: 13 limits in range. GPU 2: 1 of 13 limits crossed.");
}

TEST(SmiDiagParameters, TheCriticalVoltagesAreReadWithTheGpusHandle) {
  reset_fake(1);
  auto r = parameters({0});
  EXPECT_EQ(g_volt_calls_without_handle, 0);
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_PASS);
  EXPECT_EQ(info(*r), "GPU 0: " + std::to_string(kLimitsPerHealthyGpu) + " limits in range.");
}

TEST(SmiDiagParameters, LimitsThatReadAsZeroAreNotCompared) {
  reset_fake(1);
  // A library that reports success without writing: every temperature reads as 0.
  for (auto sensor : kSensors) {
    for (auto metric : {AMDSMI_TEMP_CURRENT, AMDSMI_TEMP_MAX, AMDSMI_TEMP_MIN, AMDSMI_TEMP_CRITICAL,
                        AMDSMI_TEMP_EMERGENCY, AMDSMI_TEMP_CRIT_MIN}) {
      set_temp(0, sensor, metric, Reading{AMDSMI_STATUS_SUCCESS, 0, false});
    }
  }
  auto r = parameters({0});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_PASS);
  EXPECT_EQ(info(*r), "GPU 0: 4 limits in range.");  // the voltage limits only
  EXPECT_EQ(entries(*r), (std::vector<std::pair<uint32_t, std::string>>{{0, ""}}));

  // The same for the voltage limits.
  reset_fake(1);
  for (auto metric :
       {AMDSMI_VOLT_MAX, AMDSMI_VOLT_MIN, AMDSMI_VOLT_MAX_CRIT, AMDSMI_VOLT_MIN_CRIT}) {
    set_volt(0, metric, Reading{AMDSMI_STATUS_SUCCESS, 0, false});
  }
  r = parameters({0});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_PASS);
  EXPECT_EQ(info(*r), "GPU 0: 9 limits in range.");  // the temperature limits only
  EXPECT_EQ(entries(*r), (std::vector<std::pair<uint32_t, std::string>>{{0, ""}}));
}

TEST(SmiDiagParameters, ANegativeMinimumTemperatureIsCompared) {
  reset_fake(1);
  set_temp(0, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_MIN, value(-10));
  auto r = parameters({0});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_PASS);
  EXPECT_EQ(info(*r), "GPU 0: 14 limits in range.");

  set_temp(0, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, value(-20));
  r = parameters({0});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_WARN);
  EXPECT_EQ(info(*r), "GPU 0: 1 of 14 limits crossed.");
  EXPECT_EQ(entries(*r)[0].second, "Edge temperature -20 is at or below its minimum -10.");
}

TEST(SmiDiagParameters, EachCrossedLimitIsStatedTheRightWayRound) {
  reset_fake(1);
  set_temp(0, AMDSMI_TEMPERATURE_TYPE_JUNCTION, AMDSMI_TEMP_CURRENT, value(111));
  set_temp(0, AMDSMI_TEMPERATURE_TYPE_VRAM, AMDSMI_TEMP_MIN, value(60));
  set_temp(0, AMDSMI_TEMPERATURE_TYPE_VRAM, AMDSMI_TEMP_CRIT_MIN, value(55));
  set_volt(0, AMDSMI_VOLT_CURRENT, value(1250));
  auto r = parameters({0});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_EQ(info(*r), "GPU 0: 7 of 15 limits crossed.");
  EXPECT_EQ(entries(*r)[0].second,
            "Junction temperature 111 is at or above its maximum 100. "
            "Junction temperature 111 is at or above its critical limit 105. "
            "Junction temperature 111 is at or above its emergency limit 110. "
            "Memory temperature 50 is at or below its minimum 60. "
            "Memory temperature 50 is at or below its critical minimum 55. "
            "Vddgfx voltage 1250 is at or above its maximum 1100. "
            "Vddgfx voltage 1250 is at or above its critical maximum 1200.");

  reset_fake(1);
  set_volt(0, AMDSMI_VOLT_CURRENT, value(600));
  r = parameters({0});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_EQ(entries(*r)[0].second,
            "Vddgfx voltage 600 is at or below its minimum 700. "
            "Vddgfx voltage 600 is at or below its critical minimum 650.");
}

TEST(SmiDiagParameters, ALimitIsCrossedAtItsValue) {
  reset_fake(1);
  set_temp(0, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, value(100));  // its maximum
  set_volt(0, AMDSMI_VOLT_CURRENT, value(700));                                // its minimum
  auto r = parameters({0});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_WARN);
  EXPECT_EQ(info(*r), "GPU 0: 2 of 13 limits crossed.");
  EXPECT_EQ(entries(*r), (std::vector<std::pair<uint32_t, std::string>>{
                             {0,
                              "Edge temperature 100 is at or above its maximum 100. "
                              "Vddgfx voltage 700 is at or below its minimum 700."}}));
}

TEST(SmiDiagParameters, AGpuThatComparedNothingIsSkipped) {
  reset_fake(2);
  for (auto sensor : kSensors) set_temp(0, sensor, AMDSMI_TEMP_CURRENT, failing());
  set_volt(0, AMDSMI_VOLT_CURRENT, failing());
  auto r = parameters({0, 1});
  EXPECT_EQ(g_check_status, RDC_ST_OK);
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_SKIP);
  EXPECT_EQ(entries(*r), (std::vector<std::pair<uint32_t, std::string>>{
                             {0, "No temperature or voltage limit could be compared."}, {1, ""}}));
  EXPECT_EQ(info(*r), "GPU 0: no limit could be compared. GPU 1: 13 limits in range.");

  // A GPU whose handle lookup fails compares nothing too.
  reset_fake(2);
  g_gpus[1].available = false;
  r = parameters({0, 1});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_SKIP);
  EXPECT_EQ(entries(*r)[1], (std::pair<uint32_t, std::string>{
                                1, "No temperature or voltage limit could be compared."}));
}

TEST(SmiDiagParameters, OneComparedLimitIsEnough) {
  reset_fake(1);
  g_temps.clear();
  g_volts.clear();
  set_temp(0, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, value(50));
  set_temp(0, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CRITICAL, value(105));
  auto r = parameters({0});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_PASS);
  EXPECT_EQ(info(*r), "GPU 0: 1 limit in range.");

  set_temp(0, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, value(106));
  r = parameters({0});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_EQ(info(*r), "GPU 0: 1 of 1 limit crossed.");
  EXPECT_EQ(entries(*r)[0].second, "Edge temperature 106 is at or above its critical limit 105.");
}

TEST(SmiDiagParameters, WarnAndFailBeatSkip) {
  reset_fake(2);
  for (auto sensor : kSensors) set_temp(0, sensor, AMDSMI_TEMP_CURRENT, failing());
  set_volt(0, AMDSMI_VOLT_CURRENT, failing());
  set_volt(1, AMDSMI_VOLT_CURRENT, value(1150));
  auto r = parameters({0, 1});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_WARN);
  set_volt(1, AMDSMI_VOLT_CURRENT, value(1250));
  r = parameters({0, 1});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
}

TEST(SmiDiagParameters, AGroupLargerThanTheResultIsCut) {
  // Hardening: gpu_index and gpu_results hold RDC_MAX_NUM_DEVICES entries.
  reset_fake(RDC_MAX_NUM_DEVICES + 2);
  auto padded = run_padded(&amd::rdc::RdcSmiDiagnosticImpl::check_smi_param_info,
                           first_gpus(RDC_MAX_NUM_DEVICES + 2));
  EXPECT_EQ(padded->result.per_gpu_result_count, static_cast<uint32_t>(RDC_MAX_NUM_DEVICES));
  EXPECT_TRUE(room_untouched(*padded));
}

TEST(SmiDiagParameters, SixtyFourGpusFitWhole) {
  reset_fake(64);
  auto r = parameters(first_gpus(64));
  const std::string s = info(*r);
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_PASS);
  EXPECT_EQ(more_not_shown(s), -1);
  EXPECT_TRUE(ends_with(s, "GPU 63: 13 limits in range."))
      << s.substr(s.size() > 80 ? s.size() - 80 : 0);
  EXPECT_EQ(r->per_gpu_result_count, 64u);
  for (uint32_t i = 0; i < 64; ++i) EXPECT_EQ(r->gpu_results[i].gpu_index, i);
}

// ---------------------------------------------------------------------------
// Compute processes
// ---------------------------------------------------------------------------

TEST(SmiDiagProcesses, AProcessIsPlacedThroughARealList) {
  reset_fake(2);
  add_process(100, {1});
  auto r = processes({0, 1});
  EXPECT_EQ(g_gpu_list_calls_without_list, 0);
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_EQ(entries(*r),
            (std::vector<std::pair<uint32_t, std::string>>{{1, "Running process: 100"}}));
  EXPECT_EQ(info(*r), "Process: 100, vram_usage: 4096, sdma_usage: 0, cu_occupancy: 1.");
}

TEST(SmiDiagProcesses, AnInsufficientSizeGpuListUsesTheEntriesWritten) {
  reset_fake(3);
  add_process(100, {1, 2});
  g_gpu_lists[100].status = AMDSMI_STATUS_INSUFFICIENT_SIZE;
  auto r = processes({0, 1, 2});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_EQ(entries(*r), (std::vector<std::pair<uint32_t, std::string>>{
                             {1, "Running process: 100"}, {2, "Running process: 100"}}));
}

TEST(SmiDiagProcesses, NoProcessIsAPass) {
  reset_fake(1);
  auto r = processes({0});
  EXPECT_EQ(g_check_status, RDC_ST_OK);
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_PASS);
  EXPECT_EQ(info(*r), "No processes running on any devices.");
  EXPECT_EQ(r->per_gpu_result_count, 0u);
  EXPECT_TRUE(g_fill_capacities.empty());
}

TEST(SmiDiagProcesses, AFailedCountIsSkippedWithItsText) {
  reset_fake(1);
  add_process(100, {0});
  g_plist.count_status = AMDSMI_STATUS_UNKNOWN_ERROR;
  auto r = processes({0});
  EXPECT_EQ(g_check_status, RDC_ST_SMI_ERROR);
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_SKIP);
  EXPECT_EQ(info(*r), "Fail to retrieve process information from amd_smi_lib");
  EXPECT_TRUE(g_fill_capacities.empty());
}

TEST(SmiDiagProcesses, ItsFailuresLogNothingAtErrorLevel) {
  // A client can run the check, so none of these may grow rdcd's log at its
  // default level.
  testing::internal::CaptureStdout();
  reset_fake(2);
  add_process(100, {0});
  g_plist.count_status = AMDSMI_STATUS_UNKNOWN_ERROR;  // the count call fails
  processes({0, 1});
  reset_fake(2);
  add_process(100, {0});
  g_plist.fill_status = AMDSMI_STATUS_UNKNOWN_ERROR;  // the fill fails
  processes({0, 1});
  reset_fake(2);
  add_process(100, {0});
  g_plist.count_report = 65537;  // too many processes
  processes({0, 1});
  reset_fake(2);
  add_process(100, {0});
  g_plist.add_before_call = {0, 20, 0, 20, 0, 20};  // the list grows on every read
  processes({0, 1});
  reset_fake(2);
  add_process(100, {0});
  add_process(101, {1});
  g_gpu_lists[100].status = AMDSMI_STATUS_UNKNOWN_ERROR;      // a GPU list fails
  g_gpu_lists[101].status = AMDSMI_STATUS_INSUFFICIENT_SIZE;  // one is too long
  processes({0, 1});
  const std::string log = testing::internal::GetCapturedStdout();
  EXPECT_EQ(log.find(" ERROR "), std::string::npos) << log;
}

TEST(SmiDiagProcesses, AGroupLargerThanTheResultIsCut) {
  // Hardening: gpu_index and gpu_results hold RDC_MAX_NUM_DEVICES entries.
  reset_fake(RDC_MAX_NUM_DEVICES + 2);
  for (uint32_t gpu = 0; gpu < RDC_MAX_NUM_DEVICES + 2; ++gpu) add_process(1000 + gpu, {gpu});
  auto padded = run_padded(&amd::rdc::RdcSmiDiagnosticImpl::check_smi_process_info,
                           first_gpus(RDC_MAX_NUM_DEVICES + 2));
  EXPECT_EQ(padded->result.per_gpu_result_count, static_cast<uint32_t>(RDC_MAX_NUM_DEVICES));
  EXPECT_TRUE(room_untouched(*padded));
}

TEST(SmiDiagProcesses, AListThatGrowsIsReadAgainUpToThreeTimes) {
  // The fill gets room for the count plus 16; 20 more processes before a fill
  // overflow it. Attempts 1 and 2 overflow, attempt 3 fits.
  reset_fake(1);
  add_process(100, {0});
  g_plist.add_before_call = {0, 20, 0, 20, 0, 0};
  auto r = processes({0});
  EXPECT_EQ(g_fill_capacities, (std::vector<uint32_t>{17, 37, 57}));
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
  ASSERT_EQ(entries(*r).size(), 1u);

  // One that needs a 4th attempt is skipped.
  reset_fake(1);
  add_process(100, {0});
  g_plist.add_before_call = {0, 20, 0, 20, 0, 20};
  r = processes({0});
  EXPECT_EQ(g_check_status, RDC_ST_OK);
  EXPECT_EQ(g_fill_capacities.size(), 3u);
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_SKIP);
  EXPECT_EQ(info(*r), "The process list changed while it was read.");
  EXPECT_EQ(r->per_gpu_result_count, 0u);
}

TEST(SmiDiagProcesses, ACountAboveTheBoundIsRefusedWithoutAFill) {
  reset_fake(1);
  add_process(100, {0});
  g_plist.count_report = 65536;  // the bound: still read
  auto r = processes({0});
  EXPECT_EQ(g_fill_capacities, (std::vector<uint32_t>{65536}));
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);

  reset_fake(1);
  add_process(100, {0});
  g_plist.count_report = 65537;
  r = processes({0});
  EXPECT_EQ(g_check_status, RDC_ST_OK);
  EXPECT_TRUE(g_fill_capacities.empty());
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_SKIP);
  EXPECT_EQ(info(*r), "Too many processes to check.");
}

// A count of UINT32_MAX would size a 137 GB list (4294967295 x 32 bytes). The
// child that runs it limits its address space first, so the old code's
// allocation fails there instead of exhausting the host.
TEST(SmiDiagProcesses, ACountOfUint32MaxIsRefusedWithoutAllocating) {
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
  GTEST_SKIP() << "ASan and TSan reserve terabytes of address space: no RLIMIT_AS can bound this";
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
  GTEST_SKIP() << "ASan and TSan reserve terabytes of address space: no RLIMIT_AS can bound this";
#endif
#endif
  std::fflush(stdout);
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    long pages = 0;
    std::ifstream("/proc/self/statm") >> pages;
    const rlim_t size = static_cast<rlim_t>(pages) * static_cast<rlim_t>(sysconf(_SC_PAGESIZE));
    const rlim_t limit = size + (rlim_t{1} << 30);
    struct rlimit as = {limit, limit};
    setrlimit(RLIMIT_AS, &as);
    struct rlimit core = {};
    if (getrlimit(RLIMIT_CORE, &core) == 0 && core.rlim_max != 0) {
      struct rlimit one = {1, 1};
      setrlimit(RLIMIT_CORE, &one);  // no core file; a failure here is harmless
    }
    void* control =
        mmap(nullptr, size_t{64} << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (control == MAP_FAILED) {
      std::printf("address space %lu bytes: a 64 MB control allocation failed\n",
                  static_cast<unsigned long>(size));
      std::fflush(stdout);
      _exit(42);
    }
    munmap(control, size_t{64} << 20);
    try {
      reset_fake(1);
      add_process(100, {0});
      g_plist.count_report = UINT32_MAX;
      auto r = processes({0});
      EXPECT_EQ(g_check_status, RDC_ST_OK);
      EXPECT_TRUE(g_fill_capacities.empty());
      EXPECT_EQ(r->status, RDC_DIAG_RESULT_SKIP);
      EXPECT_EQ(info(*r), "Too many processes to check.");
    } catch (...) {
      _exit(3);
    }
    std::fflush(stdout);
    _exit(::testing::Test::HasFailure() ? 1 : 0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  if (WIFEXITED(status) && WEXITSTATUS(status) == 42) {
    GTEST_SKIP() << "the child's 64 MB control allocation failed under its RLIMIT_AS";
  }
  EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0)
      << "child status " << status << (WIFSIGNALED(status) ? " (signal)" : "");
}

TEST(SmiDiagProcesses, AFillThatFailsOtherwiseIsSkippedWithItsText) {
  reset_fake(1);
  add_process(100, {0});
  g_plist.fill_status = AMDSMI_STATUS_UNKNOWN_ERROR;
  auto r = processes({0});
  EXPECT_EQ(g_check_status, RDC_ST_SMI_ERROR);
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_SKIP);
  EXPECT_EQ(info(*r), "Fail to retrieve process information from amd_smi_lib");
}

TEST(SmiDiagProcesses, AProcessOnNoGpuIsIgnored) {
  reset_fake(1);
  add_process(100, {});  // read, empty: AMD SMI's answer for a process that has gone
  add_process(101, {0});
  g_plist.procs.push_back(make_process(102));  // NOT_FOUND
  auto r = processes({0});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_EQ(entries(*r),
            (std::vector<std::pair<uint32_t, std::string>>{{0, "Running process: 101"}}));
  EXPECT_EQ(info(*r), "Process: 101, vram_usage: 4096, sdma_usage: 0, cu_occupancy: 1.");
}

TEST(SmiDiagProcesses, TheProcessRunningTheCheckIsLeftOut) {
  // rdcd can stay listed on a GPU after ROCr's checks have used it; it is no
  // workload on the group's GPUs.
  ASSERT_EQ(getpid(), kOwnPid);  // the wrap is in place
  const uint32_t self = kOwnPid;
  reset_fake(2);
  add_process(self, {1});
  g_plist.procs[0].cu_occupancy = UINT32_MAX;  // not available, as on lsttdev6's GPUs
  auto r = processes({0, 1});
  EXPECT_EQ(g_check_status, RDC_ST_OK);
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_PASS);
  EXPECT_EQ(info(*r), "No processes running on the group's GPUs.");
  EXPECT_EQ(r->per_gpu_result_count, 0u);

  // Left out before its GPUs are read: a GPU list that cannot be read counts nothing either.
  reset_fake(2);
  add_process(self, {1});
  g_gpu_lists[self].status = AMDSMI_STATUS_UNKNOWN_ERROR;
  r = processes({0, 1});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_PASS);
  EXPECT_EQ(info(*r), "No processes running on the group's GPUs.");

  // Any other process on the group's GPUs is still found, and only it is named.
  reset_fake(2);
  add_process(self, {1});
  add_process(100, {1});
  r = processes({0, 1});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_EQ(entries(*r),
            (std::vector<std::pair<uint32_t, std::string>>{{1, "Running process: 100"}}));
  EXPECT_EQ(info(*r), "Process: 100, vram_usage: 4096, sdma_usage: 0, cu_occupancy: 1.");
}

TEST(SmiDiagProcesses, EntriesFollowTheGroupAndNoEntryIsLeftFromTheLastProcess) {
  // A is on GPUs 1, 2 and 3. B's list reports 3 entries but has 1 written (GPU 2).
  // Without filling the list before each call, B also lands on what A left there
  // (or, from a list of zeros, on GPU 0).
  reset_fake(4);
  add_process(10, {1, 2, 3});
  add_process(11, {2});
  g_gpu_lists[11].report = 3;
  auto r = processes({0, 3, 1, 2});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_EQ(
      entries(*r),
      (std::vector<std::pair<uint32_t, std::string>>{
          {3, "Running process: 10"}, {1, "Running process: 10"}, {2, "Running process: 10 11"}}));
}

TEST(SmiDiagProcesses, ProcessesOnlyOutsideTheGroupPass) {
  reset_fake(4);
  add_process(100, {3});
  auto r = processes({1, 2});
  EXPECT_EQ(g_check_status, RDC_ST_OK);
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_PASS);
  EXPECT_EQ(info(*r), "No processes running on the group's GPUs.");
  EXPECT_EQ(r->per_gpu_result_count, 0u);

  // The texts cover the group's GPUs only.
  add_process(101, {1});
  r = processes({1, 2});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_EQ(entries(*r),
            (std::vector<std::pair<uint32_t, std::string>>{{1, "Running process: 101"}}));
  EXPECT_EQ(info(*r), "Process: 101, vram_usage: 4096, sdma_usage: 0, cu_occupancy: 1.");
}

TEST(SmiDiagProcesses, ALongPerGpuMessageIsCutAfterAWholeProcessId) {
  // 600 IDs of 7 digits after "Running process:" make 16 + 600 x 8 = 4816 bytes.
  reset_fake(1);
  for (uint32_t pid = 1000000; pid < 1000600; ++pid) add_process(pid, {0});
  auto r = processes({0});
  ASSERT_EQ(entries(*r).size(), 1u);
  std::string want = "Running process:";
  for (uint32_t pid = 1000000; pid < 1000507; ++pid) want += " " + std::to_string(pid);
  want += " (93 more not shown)";
  EXPECT_EQ(want.size(), 4092u);
  EXPECT_EQ(entries(*r)[0].second, want);

  // 510 IDs make 4096 bytes, one more than fits.
  reset_fake(1);
  for (uint32_t pid = 1000000; pid < 1000510; ++pid) add_process(pid, {0});
  r = processes({0});
  ASSERT_EQ(entries(*r).size(), 1u);
  want = "Running process:";
  for (uint32_t pid = 1000000; pid < 1000507; ++pid) want += " " + std::to_string(pid);
  want += " (3 more not shown)";
  EXPECT_EQ(want.size(), 4091u);
  EXPECT_EQ(entries(*r)[0].second, want);
}

TEST(SmiDiagProcesses, TheInfoTextStopsAtTheFirstSentenceThatDoesNotFit) {
  // 35 sentences of 111 bytes and one of 67 take 3987 bytes. The next sentence,
  // of 111, does not fit; the one after it, of 67, would, but is not shown.
  reset_fake(1);
  auto add_long = [](uint32_t pid) {
    add_process(pid, {0});
    g_plist.procs.back().vram_usage = UINT64_MAX - 1;
    g_plist.procs.back().sdma_usage = UINT64_MAX - 1;
    g_plist.procs.back().cu_occupancy = UINT32_MAX - 1;
  };
  for (uint32_t pid = 1000000; pid < 1000035; ++pid) add_long(pid);
  add_process(2000000, {0});
  add_long(1000035);
  add_process(2000001, {0});
  auto r = processes({0});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
  std::string want;
  for (uint32_t pid = 1000000; pid < 1000035; ++pid) {
    want += "Process: " + std::to_string(pid) +
            ", vram_usage: 18446744073709551614, sdma_usage: 18446744073709551614,"
            " cu_occupancy: 4294967294. ";
  }
  want += "Process: 2000000, vram_usage: 4096, sdma_usage: 0, cu_occupancy: 1. (2 more not shown)";
  EXPECT_EQ(want.size(), 4006u);
  EXPECT_EQ(info(*r), want);
}

TEST(SmiDiagProcesses, TheUnreadableCountStartsTheInfoText) {
  reset_fake(1);
  for (uint32_t pid = 1000000; pid < 1000600; ++pid) add_process(pid, {0});
  add_process(7, {0});
  g_gpu_lists[7].status = AMDSMI_STATUS_UNKNOWN_ERROR;
  auto r = processes({0});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
  const std::string s = info(*r);
  EXPECT_EQ(s.rfind("Could not read the GPUs of 1 process. Process: 1000000, ", 0), 0u)
      << s.substr(0, 80);
  EXPECT_LE(s.size(), static_cast<size_t>(MAX_DIAG_MSG_LENGTH - 1));
  EXPECT_GT(more_not_shown(s), 0);
}

TEST(SmiDiagProcesses, AFailedGpuListPlacesTheProcessNowhere) {
  // ROCm 7.3.0's AMD SMI answers a process without a KFD entry with INVAL and
  // writes nothing, leaving the room in the count.
  reset_fake(2);
  add_process(100, {0, 1});
  g_gpu_lists[100].status = AMDSMI_STATUS_INVAL;
  add_process(101, {1});
  auto r = processes({0, 1});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_EQ(entries(*r),
            (std::vector<std::pair<uint32_t, std::string>>{{1, "Running process: 101"}}));
  EXPECT_EQ(info(*r),
            "Could not read the GPUs of 1 process. "
            "Process: 101, vram_usage: 4096, sdma_usage: 0, cu_occupancy: 1.");
}

TEST(SmiDiagProcesses, NoReadableGpuListIsSkipped) {
  reset_fake(2);
  add_process(100, {0});
  add_process(101, {1});
  g_gpu_lists[100].status = AMDSMI_STATUS_UNKNOWN_ERROR;
  g_gpu_lists[101].status = AMDSMI_STATUS_UNKNOWN_ERROR;
  auto r = processes({0, 1});
  EXPECT_EQ(g_check_status, RDC_ST_OK);
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_SKIP);
  EXPECT_EQ(info(*r), "Could not read the GPUs of 2 processes.");
  EXPECT_EQ(r->per_gpu_result_count, 0u);
}

TEST(SmiDiagProcesses, ValuesAmdSmiCannotReadPrintAsNotAvailable) {
  reset_fake(1);
  add_process(100, {0});
  g_plist.procs[0].cu_occupancy = UINT32_MAX;
  g_plist.procs[0].vram_usage = UINT64_MAX;
  g_plist.procs[0].sdma_usage = UINT64_MAX;
  auto r = processes({0});
  EXPECT_EQ(r->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_EQ(info(*r), "Process: 100, vram_usage: n/a, sdma_usage: n/a, cu_occupancy: n/a.");
}

TEST(SmiDiagProcesses, AListReportedLongerThanWrittenAddsNoProcess) {
  // Hardening: a library that reports more than it wrote, or more than the room.
  // An entry it did not write reads as process 0 with a CU occupancy of 0, which
  // is skipped: were it not, the GPU list below would place it on GPU 0.
  reset_fake(1);
  add_process(100, {0});
  g_gpu_lists[0].gpus = {0};
  g_plist.fill_report = 3;
  auto r = processes({0});
  EXPECT_EQ(entries(*r),
            (std::vector<std::pair<uint32_t, std::string>>{{0, "Running process: 100"}}));

  reset_fake(1);
  add_process(100, {0});
  g_gpu_lists[0].gpus = {0};
  g_plist.fill_report = 1000;      // more than the room of 17
  g_gpu_lists[100].report = 1000;  // more than the room of 128
  r = processes({0});
  EXPECT_EQ(entries(*r),
            (std::vector<std::pair<uint32_t, std::string>>{{0, "Running process: 100"}}));
}

// ---------------------------------------------------------------------------
// A group without GPUs, and modules that return without a result
// ---------------------------------------------------------------------------

TEST(SmiDiagEmptyGroup, EachCheckIsSkipped) {
  reset_fake(2);
  add_process(100, {0});
  using RunFn = std::unique_ptr<rdc_diag_test_result_t> (*)(const std::vector<uint32_t>&);
  for (RunFn run : {topology, parameters, processes}) {
    auto r = run({});
    EXPECT_EQ(g_check_status, RDC_ST_OK);
    EXPECT_EQ(r->status, RDC_DIAG_RESULT_SKIP);
    EXPECT_EQ(info(*r), "The group has no GPU.");
    EXPECT_EQ(r->per_gpu_result_count, 0u);
  }
}

namespace {

// A diagnostic module with fixed test cases, which can return without writing
// a result and records the GPU count it was given.
class FakeModule : public amd::rdc::RdcDiagnostic {
 public:
  explicit FakeModule(std::vector<rdc_diag_test_cases_t> cases, uint32_t reported = UINT32_MAX,
                      rdc_status_t query_status = RDC_ST_OK)
      : cases_(std::move(cases)), reported_(reported), query_status_(query_status) {}
  rdc_status_t rdc_diag_test_cases_query(rdc_diag_test_cases_t test_cases[MAX_TEST_CASES],
                                         uint32_t* test_case_count) override {
    // Writes at most MAX_TEST_CASES entries, as the declared parameter allows.
    const size_t n = std::min<size_t>(cases_.size(), MAX_TEST_CASES);
    std::copy(cases_.begin(), cases_.begin() + n, test_cases);
    *test_case_count = reported_ != UINT32_MAX ? reported_ : static_cast<uint32_t>(cases_.size());
    return query_status_;
  }
  rdc_status_t rdc_test_case_run(rdc_diag_test_cases_t, uint32_t*, uint32_t gpu_count, const char*,
                                 size_t, rdc_diag_test_result_t*, rdc_diag_callback_t*) override {
    gpu_counts.push_back(gpu_count);
    return RDC_ST_BAD_PARAMETER;  // like ROCr's checks for an empty group: nothing written
  }
  rdc_status_t rdc_diagnostic_run(const rdc_group_info_t&, rdc_diag_level_t, const char*, size_t,
                                  rdc_diag_response_t*, rdc_diag_callback_t*) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_diag_init(uint64_t) override { return RDC_ST_OK; }
  rdc_status_t rdc_diag_destroy() override { return RDC_ST_OK; }

  std::vector<uint32_t> gpu_counts;

 private:
  std::vector<rdc_diag_test_cases_t> cases_;
  uint32_t reported_;
  rdc_status_t query_status_;
};

std::vector<rdc_diag_test_cases_t> cases_from(int first, int n) {
  std::vector<rdc_diag_test_cases_t> cases;
  for (int i = 0; i < n; ++i) cases.push_back(static_cast<rdc_diag_test_cases_t>(first + i));
  return cases;
}

rdc_group_info_t group_of(uint32_t count) {
  rdc_group_info_t gpus = {};
  gpus.count = count;
  for (uint32_t i = 0; i < std::min<uint32_t>(count, RDC_GROUP_MAX_ENTITIES); ++i) {
    gpus.entity_ids[i] = i;
  }
  return gpus;
}

}  // namespace

TEST(SmiDiagModule, AnEntryAModuleDidNotWriteIsSkipped) {
  for (int fill : {0x00, 0xA5}) {
    auto module =
        std::make_shared<FakeModule>(std::vector<rdc_diag_test_cases_t>{RDC_DIAG_COMPUTE_QUEUE});
    amd::rdc::RdcDiagnosticModule diag({module});

    auto response = std::make_unique<rdc_diag_response_t>();  // 7.1 MB: not on the stack
    std::memset(response.get(), fill, sizeof(*response));
    rdc_group_info_t gpus = group_of(0);
    EXPECT_EQ(
        diag.rdc_diagnostic_run(gpus, RDC_DIAG_LVL_SHORT, nullptr, 0, response.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(response->results_count, 1u);
    const rdc_diag_test_result_t& entry = response->diag_info[0];
    EXPECT_EQ(entry.test_case, RDC_DIAG_COMPUTE_QUEUE) << "fill " << fill;
    EXPECT_EQ(entry.status, RDC_DIAG_RESULT_SKIP) << "fill " << fill;
    EXPECT_EQ(entry.info[0], '\0') << "fill " << fill;
    EXPECT_EQ(entry.details.msg[0], '\0') << "fill " << fill;
    EXPECT_EQ(entry.per_gpu_result_count, 0u) << "fill " << fill;

    auto result = std::make_unique<rdc_diag_test_result_t>();
    std::memset(result.get(), fill, sizeof(*result));
    uint32_t ids[RDC_MAX_NUM_DEVICES] = {};
    diag.rdc_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 0, nullptr, 0, result.get(), nullptr);
    EXPECT_EQ(result->test_case, RDC_DIAG_COMPUTE_QUEUE) << "fill " << fill;
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_SKIP) << "fill " << fill;
    EXPECT_EQ(result->info[0], '\0') << "fill " << fill;
    EXPECT_EQ(result->details.msg[0], '\0') << "fill " << fill;
    EXPECT_EQ(result->per_gpu_result_count, 0u) << "fill " << fill;
  }
}

TEST(SmiDiagModule, TheTestCasesGatheredStayInTheCallersArray) {
  // Hardening: two modules of 10 cases each would write 20 entries into 14.
  auto first = std::make_shared<FakeModule>(cases_from(0, 10));
  auto second = std::make_shared<FakeModule>(cases_from(4, 10));
  amd::rdc::RdcDiagnosticModule diag({first, second});
  struct {
    rdc_diag_test_cases_t cases[MAX_TEST_CASES];
    rdc_diag_test_cases_t guard[8];
  } out;
  std::fill(std::begin(out.cases), std::end(out.cases), RDC_DIAG_TEST_LAST);
  std::fill(std::begin(out.guard), std::end(out.guard), RDC_DIAG_TEST_LAST);
  uint32_t count = 0;
  EXPECT_EQ(diag.rdc_diag_test_cases_query(out.cases, &count), RDC_ST_OK);
  EXPECT_EQ(count, static_cast<uint32_t>(MAX_TEST_CASES));
  for (auto g : out.guard) EXPECT_EQ(g, RDC_DIAG_TEST_LAST);

  // A module that reports more cases than it wrote is held to MAX_TEST_CASES.
  auto overreporting = std::make_shared<FakeModule>(cases_from(0, MAX_TEST_CASES), 20);
  amd::rdc::RdcDiagnosticModule diag2({overreporting});
  count = 0;
  EXPECT_EQ(diag2.rdc_diag_test_cases_query(out.cases, &count), RDC_ST_OK);
  EXPECT_EQ(count, static_cast<uint32_t>(MAX_TEST_CASES));
  for (auto g : out.guard) EXPECT_EQ(g, RDC_DIAG_TEST_LAST);
}

TEST(SmiDiagModule, EntriesAModuleReportsButDoesNotWriteAreSkipped) {
  // Hardening: the first module reports 14 cases and writes 3; the second has
  // case 0, which an entry left at 0 would also name.
  auto first = std::make_shared<FakeModule>(cases_from(4, 3), MAX_TEST_CASES);
  auto second = std::make_shared<FakeModule>(cases_from(0, 1));
  amd::rdc::RdcDiagnosticModule diag({first, second});
  struct {
    rdc_diag_test_cases_t cases[MAX_TEST_CASES];
    rdc_diag_test_cases_t guard[8];
  } out;
  std::fill(std::begin(out.cases), std::end(out.cases), RDC_DIAG_TEST_LAST);
  std::fill(std::begin(out.guard), std::end(out.guard), RDC_DIAG_TEST_LAST);
  uint32_t count = 0;
  EXPECT_EQ(diag.rdc_diag_test_cases_query(out.cases, &count), RDC_ST_OK);
  std::vector<rdc_diag_test_cases_t> want = cases_from(4, 3);
  want.push_back(RDC_DIAG_COMPUTE_PROCESS);
  EXPECT_EQ(std::vector<rdc_diag_test_cases_t>(
                out.cases, out.cases + std::min<uint32_t>(count, MAX_TEST_CASES)),
            want);
  for (auto g : out.guard) EXPECT_EQ(g, RDC_DIAG_TEST_LAST);

  // Case 0 runs on the module that wrote it, and RDC_DIAG_TEST_LAST on none.
  auto result = std::make_unique<rdc_diag_test_result_t>();
  uint32_t ids[RDC_MAX_NUM_DEVICES] = {};
  diag.rdc_test_case_run(RDC_DIAG_COMPUTE_PROCESS, ids, 1, nullptr, 0, result.get(), nullptr);
  EXPECT_EQ(second->gpu_counts.size(), 1u);
  EXPECT_EQ(diag.rdc_test_case_run(RDC_DIAG_TEST_LAST, ids, 1, nullptr, 0, result.get(), nullptr),
            RDC_ST_NOT_SUPPORTED);
  EXPECT_TRUE(first->gpu_counts.empty());
}

TEST(SmiDiagModule, AModuleWhoseQueryFailsAddsNoTestCase) {
  auto failing = std::make_shared<FakeModule>(cases_from(1, 1), UINT32_MAX, RDC_ST_NOT_SUPPORTED);
  auto working = std::make_shared<FakeModule>(cases_from(0, 1));
  amd::rdc::RdcDiagnosticModule diag({failing, working});
  rdc_diag_test_cases_t cases[MAX_TEST_CASES] = {};
  uint32_t count = 0;
  EXPECT_EQ(diag.rdc_diag_test_cases_query(cases, &count), RDC_ST_OK);
  ASSERT_EQ(count, 1u);
  EXPECT_EQ(cases[0], RDC_DIAG_COMPUTE_PROCESS);
  auto result = std::make_unique<rdc_diag_test_result_t>();
  uint32_t ids[RDC_MAX_NUM_DEVICES] = {};
  EXPECT_EQ(
      diag.rdc_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, result.get(), nullptr),
      RDC_ST_NOT_SUPPORTED);
  EXPECT_TRUE(failing->gpu_counts.empty());
}

TEST(SmiDiagModule, AGroupCountAboveTheGroupSizeIsClamped) {
  // Hardening: the group's entity list has RDC_GROUP_MAX_ENTITIES entries.
  auto module =
      std::make_shared<FakeModule>(std::vector<rdc_diag_test_cases_t>{RDC_DIAG_COMPUTE_QUEUE});
  amd::rdc::RdcDiagnosticModule diag({module});
  auto response = std::make_unique<rdc_diag_response_t>();
  rdc_group_info_t gpus = group_of(100);
  diag.rdc_diagnostic_run(gpus, RDC_DIAG_LVL_SHORT, nullptr, 0, response.get(), nullptr);
  auto result = std::make_unique<rdc_diag_test_result_t>();
  uint32_t ids[RDC_MAX_NUM_DEVICES] = {};
  diag.rdc_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 100, nullptr, 0, result.get(), nullptr);
  EXPECT_EQ(module->gpu_counts,
            (std::vector<uint32_t>{RDC_GROUP_MAX_ENTITIES, RDC_GROUP_MAX_ENTITIES}));
}

TEST(SmiDiagModule, ACustomRunLogsNothingAtErrorLevel) {
  auto module =
      std::make_shared<FakeModule>(std::vector<rdc_diag_test_cases_t>{RDC_DIAG_COMPUTE_QUEUE});
  amd::rdc::RdcDiagnosticModule diag({module});
  auto response = std::make_unique<rdc_diag_response_t>();
  rdc_group_info_t gpus = group_of(1);
  testing::internal::CaptureStdout();
  for (int i = 0; i < 3; ++i) {
    diag.rdc_diagnostic_run(gpus, RDC_DIAG_LVL_SHORT, "abc", 3, response.get(), nullptr);
  }
  const std::string log = testing::internal::GetCapturedStdout();
  size_t errors = 0;
  for (size_t pos = 0; (pos = log.find(" ERROR ", pos)) != std::string::npos; ++pos) ++errors;
  EXPECT_EQ(errors, 0u) << log;
}
