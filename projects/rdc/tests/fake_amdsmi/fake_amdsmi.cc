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

// Fake amd-smi for GPU-less RDC tests.
//
// Preload it (LD_PRELOAD=librdc_fake_amdsmi.so) into rdcd, rdci or a process that uses RDC
// in embedded mode, and every amdsmi_* call RDC makes is answered for simulated GPUs instead
// of reaching the driver. The readings come from fake_values.inc, differ per GPU and per
// sensor, and are read by the Python tests too. The environment variables below reproduce
// lower-layer behavior that has caused RDC defects: failing calls, output buffers left
// unwritten, "not available" sentinels, partitions and split XGMI hives.
//
//   RDC_FAKE_GPUS=N              GPUs (sockets) to simulate, 1..128 (default 4)
//   RDC_FAKE_PARTITIONS=P        compute partitions per GPU, 1..8 (default: not partitionable)
//   RDC_FAKE_HIVES=H             XGMI hives; GPU g belongs to hive g % H (default 1)
//   RDC_FAKE_FAIL=rule,...       rule = api[@gpu][=status]: the call returns status (a number
//                                or an AMDSMI_STATUS_* name, default
//                                AMDSMI_STATUS_UNKNOWN_ERROR). api "*" matches every call
//                                that takes a GPU. Calls without a GPU argument, such as
//                                amdsmi_init, match only rules without @gpu.
//   RDC_FAKE_POISON=1            a failing call first fills its outputs with 0xA5 bytes, which
//                                is what a caller that ignores the status reads from a dirty
//                                stack
//   RDC_FAKE_SENTINEL=api[@gpu],...  the call succeeds but reports amd-smi's "not available"
//                                value (all bits set); honored by amdsmi_get_power_info,
//                                amdsmi_get_gpu_activity, amdsmi_get_gpu_busy_percent,
//                                amdsmi_get_clock_info and amdsmi_get_gpu_metrics_info
//   RDC_FAKE_ARM=path            FAIL and SENTINEL rules apply only once path exists, so a
//                                test can set RDC up first
//   RDC_FAKE_EVENT_WAIT_MS=ms    upper bound for the event-notification wait (default 100)
//   RDC_FAKE_LOG=path            append "api gpu" for every call (gpu -1 for calls without a
//                                GPU), plus " !" when a rule failed the call
//
// Only the amdsmi_* functions RDC calls are implemented. The definitions are checked against
// amd_smi/amdsmi.h at compile time, so an API change breaks the build instead of the tests.

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "amd_smi/amdsmi.h"

namespace {

constexpr uint32_t kMaxGpus = 128;
constexpr uint32_t kMaxParts = 8;
constexpr uint64_t kGiB = 1024ULL * 1024 * 1024;

enum FakeValueId {
#define FAKE_VALUE(name, base, step, modulo) k##name,
#include "fake_values.inc"
#undef FAKE_VALUE
};

struct FakeValueDef {
  int64_t base;
  int64_t step;
  int64_t modulo;
};

constexpr FakeValueDef kFakeValues[] = {
#define FAKE_VALUE(name, base, step, modulo) {base, step, modulo},
#include "fake_values.inc"
#undef FAKE_VALUE
};

// Reading `id` of GPU `gpu`, see fake_values.inc.
int64_t V(FakeValueId id, uint32_t gpu) {
  const FakeValueDef& d = kFakeValues[id];
  const int64_t v = d.base + d.step * static_cast<int64_t>(gpu);
  return d.modulo != 0 ? v % d.modulo : v;
}

struct FakeProc {
  uint32_t gpu;   // socket (physical GPU) index
  uint32_t part;  // partition index within the GPU
  uint32_t flat;  // position in amd-smi enumeration order
};

struct FakeSocket {
  uint32_t gpu;
};

struct Rule {
  std::string api;
  int64_t gpu;  // -1 = every GPU
  amdsmi_status_t status;
};

struct GpuState {
  uint64_t power_cap_uw = 750ULL * 1000 * 1000;
  uint64_t gfx_clk_limit[2] = {500, 2100};
  amdsmi_dev_perf_level_t perf_level = AMDSMI_DEV_PERF_LEVEL_AUTO;
  uint64_t event_mask = 0;
};

struct Config {
  uint32_t gpus = 4;
  uint32_t parts = 1;
  bool partitionable = false;
  uint32_t hives = 1;
  bool poison = false;
  uint32_t event_wait_ms = 100;
  std::string arm_path;  // rules apply once this file exists
  std::vector<Rule> fail;
  std::vector<Rule> sentinel;
};

std::mutex g_mutex;  // guards everything below
Config g_cfg;        // read from the environment once, at the first amdsmi_init()
bool g_loaded = false;
int g_init_refs = 0;
FakeProc g_procs[kMaxGpus * kMaxParts];
FakeSocket g_sockets[kMaxGpus];
GpuState g_state[kMaxGpus];
std::map<uint64_t, std::pair<uint32_t, uint64_t>> g_counters;  // handle -> (gpu, start ms)
uint64_t g_next_counter = 1;
FILE* g_log = nullptr;
const auto g_start = std::chrono::steady_clock::now();

uint64_t now_ms() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - g_start)
                                   .count());
}

amdsmi_status_t parse_status(const std::string& s, amdsmi_status_t dflt) {
  static const std::pair<const char*, amdsmi_status_t> kNames[] = {
      {"AMDSMI_STATUS_INVAL", AMDSMI_STATUS_INVAL},
      {"AMDSMI_STATUS_NOT_SUPPORTED", AMDSMI_STATUS_NOT_SUPPORTED},
      {"AMDSMI_STATUS_NOT_YET_IMPLEMENTED", AMDSMI_STATUS_NOT_YET_IMPLEMENTED},
      {"AMDSMI_STATUS_NO_PERM", AMDSMI_STATUS_NO_PERM},
      {"AMDSMI_STATUS_OUT_OF_RESOURCES", AMDSMI_STATUS_OUT_OF_RESOURCES},
      {"AMDSMI_STATUS_INTERNAL_EXCEPTION", AMDSMI_STATUS_INTERNAL_EXCEPTION},
      {"AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS", AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS},
      {"AMDSMI_STATUS_INIT_ERROR", AMDSMI_STATUS_INIT_ERROR},
      {"AMDSMI_STATUS_NOT_FOUND", AMDSMI_STATUS_NOT_FOUND},
      {"AMDSMI_STATUS_BUSY", AMDSMI_STATUS_BUSY},
      {"AMDSMI_STATUS_NO_DATA", AMDSMI_STATUS_NO_DATA},
      {"AMDSMI_STATUS_UNEXPECTED_DATA", AMDSMI_STATUS_UNEXPECTED_DATA},
      {"AMDSMI_STATUS_UNKNOWN_ERROR", AMDSMI_STATUS_UNKNOWN_ERROR},
  };
  if (s.empty()) return dflt;
  for (const auto& n : kNames) {
    if (s == n.first) return n.second;
  }
  char* end = nullptr;
  const long n = std::strtol(s.c_str(), &end, 0);
  if (end == s.c_str() || *end != '\0') {
    // A typo must not turn into AMDSMI_STATUS_SUCCESS and a test that cannot fail.
    fprintf(stderr, "fake amd-smi: unknown status '%s'\n", s.c_str());
    abort();
  }
  return static_cast<amdsmi_status_t>(n);
}

std::vector<Rule> parse_rules(const char* env, amdsmi_status_t dflt) {
  std::vector<Rule> rules;
  const char* v = getenv(env);
  if (v == nullptr) return rules;
  std::string all(v);
  size_t pos = 0;
  while (pos <= all.size()) {
    size_t end = all.find(',', pos);
    if (end == std::string::npos) end = all.size();
    std::string item = all.substr(pos, end - pos);
    pos = end + 1;
    if (item.empty()) continue;
    Rule r{item, -1, dflt};
    size_t eq = r.api.find('=');
    if (eq != std::string::npos) {
      r.status = parse_status(r.api.substr(eq + 1), dflt);
      r.api.resize(eq);
    }
    size_t at = r.api.find('@');
    if (at != std::string::npos) {
      r.gpu = std::strtoll(r.api.c_str() + at + 1, nullptr, 0);
      r.api.resize(at);
    }
    rules.push_back(r);
  }
  return rules;
}

uint32_t env_u32(const char* name, uint32_t dflt, uint32_t lo, uint32_t hi) {
  const char* v = getenv(name);
  if (v == nullptr || *v == '\0') return dflt;
  long n = std::strtol(v, nullptr, 0);
  return static_cast<uint32_t>(std::min<long>(std::max<long>(n, lo), hi));
}

void load_config_locked() {
  Config c;
  c.gpus = env_u32("RDC_FAKE_GPUS", 4, 1, kMaxGpus);
  c.partitionable = getenv("RDC_FAKE_PARTITIONS") != nullptr;
  c.parts = env_u32("RDC_FAKE_PARTITIONS", 1, 1, kMaxParts);
  c.hives = env_u32("RDC_FAKE_HIVES", 1, 1, kMaxGpus);
  c.poison = env_u32("RDC_FAKE_POISON", 0, 0, 1) != 0;
  c.event_wait_ms = env_u32("RDC_FAKE_EVENT_WAIT_MS", 100, 0, 60000);
  if (const char* arm = getenv("RDC_FAKE_ARM")) c.arm_path = arm;
  c.fail = parse_rules("RDC_FAKE_FAIL", AMDSMI_STATUS_UNKNOWN_ERROR);
  c.sentinel = parse_rules("RDC_FAKE_SENTINEL", AMDSMI_STATUS_SUCCESS);
  g_cfg = c;
  uint32_t flat = 0;
  for (uint32_t g = 0; g < c.gpus; ++g) {
    g_sockets[g].gpu = g;
    for (uint32_t p = 0; p < c.parts; ++p, ++flat) g_procs[flat] = {g, p, flat};
  }
  const char* log = getenv("RDC_FAKE_LOG");
  if (log != nullptr) g_log = fopen(log, "ae");
  g_loaded = true;
}

bool initialized() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_init_refs > 0;
}

// Rules apply once RDC_FAKE_ARM names an existing file, or always when it is unset, so
// a test can set up RDC before the faults start.
bool armed() { return g_cfg.arm_path.empty() || access(g_cfg.arm_path.c_str(), F_OK) == 0; }

// gpu is -1 for calls without a GPU argument; "*" matches only calls that take one.
const Rule* match(const std::vector<Rule>& rules, const char* api, int64_t gpu) {
  if (rules.empty() || !armed()) return nullptr;
  for (const auto& r : rules) {
    const bool same_api = r.api == api || (r.api == "*" && gpu >= 0);
    if (same_api && (r.gpu < 0 || r.gpu == gpu)) return &r;
  }
  return nullptr;
}

// One line per call, "api gpu", plus " !" when a rule failed the call.
void write_log_locked(const char* api, int64_t gpu, bool failed) {
  if (g_log == nullptr) return;
  fprintf(g_log, "%s %lld%s\n", api, static_cast<long long>(gpu), failed ? " !" : "");
  fflush(g_log);
}

void log_call(const char* api, int64_t gpu, bool failed = false) {
  std::lock_guard<std::mutex> lock(g_mutex);
  write_log_locked(api, gpu, failed);
}

const FakeProc* to_proc(amdsmi_processor_handle h) {
  // Integer compare: h may point anywhere, so a pointer compare would be undefined.
  const auto addr = reinterpret_cast<uintptr_t>(h);
  const auto first = reinterpret_cast<uintptr_t>(g_procs);
  const uintptr_t n = g_cfg.gpus * g_cfg.parts;
  if (addr < first || addr >= first + n * sizeof(FakeProc) || (addr - first) % sizeof(FakeProc)) {
    return nullptr;
  }
  return static_cast<const FakeProc*>(h);
}

// Fills every output with 0xA5 when poisoning is on.
void poison() {}
template <typename T, typename... Rest>
void poison(T* out, Rest*... rest) {
  if (out != nullptr && g_cfg.poison) memset(static_cast<void*>(out), 0xA5, sizeof(T));
  poison(rest...);
}

// Common entry for calls that take a GPU: validates the handle, logs the call and applies
// RDC_FAKE_FAIL. Returns SUCCESS when the call should go ahead.
template <typename... Outs>
amdsmi_status_t enter(const char* api, amdsmi_processor_handle h, const FakeProc** p,
                      Outs*... outs) {
  if (!initialized()) return AMDSMI_STATUS_NOT_INIT;
  *p = to_proc(h);
  if (*p == nullptr) return AMDSMI_STATUS_INVAL;
  const Rule* r = match(g_cfg.fail, api, (*p)->flat);
  log_call(api, (*p)->flat, r != nullptr);
  if (r != nullptr) {
    poison(outs...);
    return r->status;
  }
  return AMDSMI_STATUS_SUCCESS;
}

// Same for calls without a GPU argument.
template <typename... Outs>
amdsmi_status_t enter_global(const char* api, Outs*... outs) {
  if (!initialized()) return AMDSMI_STATUS_NOT_INIT;
  const Rule* r = match(g_cfg.fail, api, -1);
  log_call(api, -1, r != nullptr);
  if (r != nullptr) {
    poison(outs...);
    return r->status;
  }
  return AMDSMI_STATUS_SUCCESS;
}

bool sentinel(const char* api, const FakeProc* p) {
  return match(g_cfg.sentinel, api, p->flat) != nullptr;
}

template <typename T>
amdsmi_status_t fill_sentinel(T* out) {
  memset(static_cast<void*>(out), 0xFF, sizeof(T));
  return AMDSMI_STATUS_SUCCESS;
}

bool same_hive(const FakeProc* a, const FakeProc* b) {
  return a->gpu % g_cfg.hives == b->gpu % g_cfg.hives;
}

}  // namespace

extern "C" {

amdsmi_status_t amdsmi_init(uint64_t init_flags) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_loaded) load_config_locked();
  const Rule* r = match(g_cfg.fail, "amdsmi_init", -1);
  write_log_locked("amdsmi_init", -1, r != nullptr);
  if (r != nullptr) return r->status;
  // No simulated CPUs: amd-smi fails a CPU init on hosts without the HSMP driver.
  if ((init_flags & AMDSMI_INIT_AMD_CPUS) != 0) return AMDSMI_STATUS_NOT_SUPPORTED;
  ++g_init_refs;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_shut_down(void) {
  std::lock_guard<std::mutex> lock(g_mutex);
  write_log_locked("amdsmi_shut_down", -1, false);
  if (g_init_refs > 0) --g_init_refs;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_lib_version(amdsmi_version_t* version) {
  {
    // Valid before amdsmi_init(): open the log here if nothing else has.
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded) load_config_locked();
    write_log_locked(__func__, -1, false);
  }
  if (version == nullptr) return AMDSMI_STATUS_INVAL;
  version->major = 27;
  version->minor = 0;
  version->release = 0;
  version->build = "fake";
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_socket_handles(uint32_t* socket_count,
                                          amdsmi_socket_handle* socket_handles) {
  if (auto st = enter_global(__func__, socket_count)) return st;
  if (socket_count == nullptr) return AMDSMI_STATUS_INVAL;
  if (socket_handles == nullptr) {
    *socket_count = g_cfg.gpus;
    return AMDSMI_STATUS_SUCCESS;
  }
  uint32_t n = std::min(*socket_count, g_cfg.gpus);
  for (uint32_t i = 0; i < n; ++i) socket_handles[i] = &g_sockets[i];
  *socket_count = n;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_handles(amdsmi_socket_handle socket_handle,
                                             uint32_t* processor_count,
                                             amdsmi_processor_handle* processor_handles) {
  if (auto st = enter_global(__func__, processor_count)) return st;
  auto* s = static_cast<const FakeSocket*>(socket_handle);
  if (processor_count == nullptr || s < g_sockets || s >= g_sockets + g_cfg.gpus) {
    return AMDSMI_STATUS_INVAL;
  }
  if (processor_handles == nullptr) {
    *processor_count = g_cfg.parts;
    return AMDSMI_STATUS_SUCCESS;
  }
  uint32_t n = std::min(*processor_count, g_cfg.parts);
  for (uint32_t i = 0; i < n; ++i) processor_handles[i] = &g_procs[s->gpu * g_cfg.parts + i];
  *processor_count = n;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_type(amdsmi_processor_handle processor_handle,
                                          amdsmi_processor_type_t* processor_type) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, processor_type)) return st;
  if (processor_type == nullptr) return AMDSMI_STATUS_INVAL;
  *processor_type = AMDSMI_PROCESSOR_TYPE_AMD_GPU;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_asic_info(amdsmi_processor_handle processor_handle,
                                         amdsmi_asic_info_t* info) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, info)) return st;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  memset(info, 0, sizeof(*info));
  snprintf(info->market_name, sizeof(info->market_name), "Fake AMD Instinct GPU %u", p->gpu);
  snprintf(info->vendor_name, sizeof(info->vendor_name), "Advanced Micro Devices, Inc.");
  snprintf(info->asic_serial, sizeof(info->asic_serial), "FAKE%04X", p->gpu);
  info->vendor_id = 0x1002;
  info->subvendor_id = 0x1002;
  info->device_id = 0x74a1;
  info->rev_id = 0;
  info->oam_id = p->gpu;
  info->num_of_compute_units = 304 / g_cfg.parts;
  info->target_graphics_version = 90402;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_kfd_info(amdsmi_processor_handle processor_handle,
                                        amdsmi_kfd_info_t* info) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, info)) return st;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  memset(info, 0, sizeof(*info));
  info->kfd_id = 0x1000 + p->flat;
  info->node_id = p->flat + 1;
  info->current_partition_id = p->part;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_temp_metric(amdsmi_processor_handle processor_handle,
                                       amdsmi_temperature_type_t sensor_type,
                                       amdsmi_temperature_metric_t metric, int64_t* temperature) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, temperature)) return st;
  if (temperature == nullptr) return AMDSMI_STATUS_INVAL;
  FakeValueId current;
  switch (sensor_type) {
    case AMDSMI_TEMPERATURE_TYPE_EDGE:
      current = kTEMP_EDGE_C;
      break;
    case AMDSMI_TEMPERATURE_TYPE_HOTSPOT:
      current = kTEMP_HOTSPOT_C;
      break;
    case AMDSMI_TEMPERATURE_TYPE_VRAM:
      current = kTEMP_MEM_C;
      break;
    default:
      return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  switch (metric) {
    case AMDSMI_TEMP_CURRENT:
      *temperature = V(current, p->gpu);
      return AMDSMI_STATUS_SUCCESS;
    case AMDSMI_TEMP_MAX:
      *temperature = 100;
      return AMDSMI_STATUS_SUCCESS;
    case AMDSMI_TEMP_CRITICAL:
      *temperature = 105;
      return AMDSMI_STATUS_SUCCESS;
    case AMDSMI_TEMP_EMERGENCY:
    case AMDSMI_TEMP_SHUTDOWN:
      *temperature = 110;
      return AMDSMI_STATUS_SUCCESS;
    default:
      return AMDSMI_STATUS_NOT_SUPPORTED;
  }
}

amdsmi_status_t amdsmi_get_gpu_volt_metric(amdsmi_processor_handle processor_handle,
                                           amdsmi_voltage_type_t sensor_type,
                                           amdsmi_voltage_metric_t metric, int64_t* voltage) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, voltage)) return st;
  if (voltage == nullptr) return AMDSMI_STATUS_INVAL;
  (void)sensor_type;
  switch (metric) {
    case AMDSMI_VOLT_CURRENT:
      *voltage = V(kVOLTAGE_MV, p->gpu);
      return AMDSMI_STATUS_SUCCESS;
    case AMDSMI_VOLT_MIN:
      *voltage = 700;
      return AMDSMI_STATUS_SUCCESS;
    case AMDSMI_VOLT_MAX:
      *voltage = 1100;
      return AMDSMI_STATUS_SUCCESS;
    case AMDSMI_VOLT_MIN_CRIT:
      *voltage = 650;
      return AMDSMI_STATUS_SUCCESS;
    case AMDSMI_VOLT_MAX_CRIT:
      *voltage = 1200;
      return AMDSMI_STATUS_SUCCESS;
    default:
      return AMDSMI_STATUS_NOT_SUPPORTED;
  }
}

amdsmi_status_t amdsmi_get_power_info(amdsmi_processor_handle processor_handle,
                                      amdsmi_power_info_t* info) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, info)) return st;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  if (sentinel(__func__, p)) return fill_sentinel(info);
  memset(info, 0, sizeof(*info));
  info->current_socket_power = V(kSOCKET_POWER_W, p->gpu);
  info->average_socket_power = V(kSOCKET_POWER_W, p->gpu);
  info->socket_power = info->current_socket_power;
  info->power_limit = 750;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_power_cap_info(amdsmi_processor_handle processor_handle,
                                          uint32_t sensor_ind, amdsmi_power_cap_info_t* info) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, info)) return st;
  if (info == nullptr || sensor_ind != 0) return AMDSMI_STATUS_INVAL;
  memset(info, 0, sizeof(*info));
  std::lock_guard<std::mutex> lock(g_mutex);
  info->power_cap = g_state[p->gpu].power_cap_uw;
  info->default_power_cap = 750ULL * 1000 * 1000;
  info->min_power_cap = 300ULL * 1000 * 1000;
  info->max_power_cap = 750ULL * 1000 * 1000;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_power_cap(amdsmi_processor_handle processor_handle, uint32_t sensor_ind,
                                     uint64_t cap) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p)) return st;
  if (sensor_ind != 0 || cap < 300ULL * 1000 * 1000 || cap > 750ULL * 1000 * 1000) {
    return AMDSMI_STATUS_INVAL;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  g_state[p->gpu].power_cap_uw = cap;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_activity(amdsmi_processor_handle processor_handle,
                                        amdsmi_engine_usage_t* info) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, info)) return st;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  if (sentinel(__func__, p)) return fill_sentinel(info);
  memset(info, 0, sizeof(*info));
  info->gfx_activity = V(kGFX_ACTIVITY_PCT, p->gpu);
  info->umc_activity = V(kUMC_ACTIVITY_PCT, p->gpu);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_busy_percent(amdsmi_processor_handle processor_handle,
                                            uint32_t* gpu_busy_percent) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, gpu_busy_percent)) return st;
  if (gpu_busy_percent == nullptr) return AMDSMI_STATUS_INVAL;
  if (sentinel(__func__, p)) return fill_sentinel(gpu_busy_percent);
  *gpu_busy_percent = V(kGFX_ACTIVITY_PCT, p->gpu);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_clock_info(amdsmi_processor_handle processor_handle,
                                      amdsmi_clk_type_t clk_type, amdsmi_clk_info_t* info) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, info)) return st;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  if (sentinel(__func__, p)) return fill_sentinel(info);
  memset(info, 0, sizeof(*info));
  if (clk_type == AMDSMI_CLK_TYPE_GFX) {
    info->clk = V(kGFX_CLK_MHZ, p->gpu);
    info->min_clk = 500;
    info->max_clk = 2100;
  } else if (clk_type == AMDSMI_CLK_TYPE_MEM) {
    info->clk = V(kMEM_CLK_MHZ, p->gpu);
    info->min_clk = 900;
    info->max_clk = 1300;
  } else {
    info->clk = V(kSOC_CLK_MHZ, p->gpu);
    info->min_clk = 100;
    info->max_clk = 1000;
  }
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_clk_freq(amdsmi_processor_handle processor_handle,
                                    amdsmi_clk_type_t clk_type, amdsmi_frequencies_t* f) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, f)) return st;
  if (f == nullptr) return AMDSMI_STATUS_INVAL;
  memset(f, 0, sizeof(*f));
  // Same model as amdsmi_get_clock_info(): the current level is the per-GPU clock.
  const uint64_t mhz = 1000000;
  const bool mem = clk_type == AMDSMI_CLK_TYPE_MEM;
  f->num_supported = 3;
  f->current = 1;
  f->frequency[0] = (mem ? 900 : 500) * mhz;
  f->frequency[1] = V(mem ? kMEM_CLK_MHZ : kGFX_CLK_MHZ, p->gpu) * mhz;
  f->frequency[2] = (mem ? 1300 : 2100) * mhz;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_gpu_clk_limit(amdsmi_processor_handle processor_handle,
                                         amdsmi_clk_type_t clk_type,
                                         amdsmi_clk_limit_type_t limit_type, uint64_t clk_value) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p)) return st;
  if (clk_type != AMDSMI_CLK_TYPE_GFX || clk_value < 500 || clk_value > 2100) {
    return AMDSMI_STATUS_INVAL;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  g_state[p->gpu].gfx_clk_limit[limit_type == AMDSMI_CLK_LIMIT_MIN ? 0 : 1] = clk_value;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_gpu_perf_level(amdsmi_processor_handle processor_handle,
                                          amdsmi_dev_perf_level_t perf_lvl) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p)) return st;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_state[p->gpu].perf_level = perf_lvl;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_memory_total(amdsmi_processor_handle processor_handle,
                                            amdsmi_memory_type_t mem_type, uint64_t* total) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, total)) return st;
  if (total == nullptr) return AMDSMI_STATUS_INVAL;
  *total =
      (mem_type == AMDSMI_MEM_TYPE_GTT ? 128 : V(kVRAM_TOTAL_GIB, p->gpu)) * kGiB / g_cfg.parts;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_memory_usage(amdsmi_processor_handle processor_handle,
                                            amdsmi_memory_type_t mem_type, uint64_t* used) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, used)) return st;
  if (used == nullptr) return AMDSMI_STATUS_INVAL;
  *used = (mem_type == AMDSMI_MEM_TYPE_GTT ? 1 : V(kVRAM_USED_GIB, p->gpu)) * kGiB / g_cfg.parts;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_vram_info(amdsmi_processor_handle processor_handle,
                                         amdsmi_vram_info_t* info) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, info)) return st;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  memset(info, 0, sizeof(*info));
  info->vram_type = AMDSMI_VRAM_TYPE_HBM3;
  snprintf(info->vram_vendor, sizeof(info->vram_vendor), "FAKE");
  info->vram_size = V(kVRAM_TOTAL_GIB, p->gpu) * 1024 / g_cfg.parts;  // MB
  info->vram_bit_width = 8192;
  info->vram_max_bandwidth = 5300;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_metrics_info(amdsmi_processor_handle processor_handle,
                                            amdsmi_gpu_metrics_t* pgpu_metrics) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, pgpu_metrics)) return st;
  if (pgpu_metrics == nullptr) return AMDSMI_STATUS_INVAL;
  // amd-smi reports every metric the firmware does not provide as all bits set.
  memset(static_cast<void*>(pgpu_metrics), 0xFF, sizeof(*pgpu_metrics));
  pgpu_metrics->apu_metrics = nullptr;  // not an APU
  if (sentinel(__func__, p)) return AMDSMI_STATUS_SUCCESS;
  amdsmi_gpu_metrics_t& m = *pgpu_metrics;
  const uint64_t t = now_ms();
  m.temperature_edge = V(kTEMP_EDGE_C, p->gpu);
  m.temperature_hotspot = V(kTEMP_HOTSPOT_C, p->gpu);
  m.temperature_mem = V(kTEMP_MEM_C, p->gpu);
  m.average_gfx_activity = V(kGFX_ACTIVITY_PCT, p->gpu);
  m.average_umc_activity = V(kUMC_ACTIVITY_PCT, p->gpu);
  m.average_socket_power = V(kSOCKET_POWER_W, p->gpu);
  m.current_socket_power = V(kSOCKET_POWER_W, p->gpu);
  m.energy_accumulator = t * (p->gpu + 1);
  m.current_gfxclk = V(kGFX_CLK_MHZ, p->gpu);
  m.current_uclk = V(kMEM_CLK_MHZ, p->gpu);
  m.gfx_activity_acc = t * V(kGFX_ACTIVITY_PCT, p->gpu);
  m.mem_activity_acc = t * V(kUMC_ACTIVITY_PCT, p->gpu);
  m.firmware_timestamp = t * 100000;  // 10 ns units
  m.system_clock_counter = t * 1000000;
  m.pcie_link_width = 16;
  m.pcie_link_speed = 320;  // 0.1 GT/s units
  m.pcie_bandwidth_acc = t * (p->gpu + 1);
  m.pcie_bandwidth_inst = V(kPCIE_BANDWIDTH_MBPS, p->gpu);
  m.xgmi_link_width = 16;
  m.xgmi_link_speed = 32;
  for (uint32_t i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i) {
    m.xgmi_read_data_acc[i] = t * (p->gpu + 1) * (i + 1);
    m.xgmi_write_data_acc[i] = t * (p->gpu + 1) * (i + 2);
  }
  for (uint32_t i = 0; i < AMDSMI_MAX_NUM_GFX_CLKS; ++i) m.current_gfxclks[i] = m.current_gfxclk;
  m.num_partition = static_cast<uint16_t>(g_cfg.parts);
  m.vram_max_bandwidth = 5300;
  m.pcie_nak_sent_count_acc = p->gpu;
  m.pcie_nak_rcvd_count_acc = p->gpu;
  m.pcie_lc_perf_other_end_recovery = p->gpu;
  m.pcie_replay_count_acc = 0;
  m.pcie_replay_rover_count_acc = 0;
  m.pcie_l0_to_recov_count_acc = 0;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_pcie_info(amdsmi_processor_handle processor_handle,
                                     amdsmi_pcie_info_t* info) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, info)) return st;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  memset(info, 0, sizeof(*info));
  info->pcie_static.max_pcie_width = 16;
  info->pcie_static.max_pcie_speed = 32000;
  info->pcie_static.pcie_interface_version = 5;
  info->pcie_metric.pcie_width = 16;
  info->pcie_metric.pcie_speed = 32000;
  info->pcie_metric.pcie_bandwidth = V(kPCIE_BANDWIDTH_MBPS, p->gpu);
  info->pcie_metric.pcie_nak_sent_count = p->gpu;
  info->pcie_metric.pcie_nak_received_count = p->gpu;
  info->pcie_metric.pcie_lc_perf_other_end_recovery_count = p->gpu;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_pci_throughput(amdsmi_processor_handle processor_handle,
                                              uint64_t* sent, uint64_t* received,
                                              uint64_t* max_pkt_sz) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, sent, received, max_pkt_sz)) return st;
  if (sent == nullptr || received == nullptr || max_pkt_sz == nullptr) return AMDSMI_STATUS_INVAL;
  *sent = V(kPCI_SENT_BYTES, p->gpu);
  *received = V(kPCI_RECEIVED_BYTES, p->gpu);
  *max_pkt_sz = 256;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_energy_count(amdsmi_processor_handle processor_handle,
                                        uint64_t* energy_accumulator, float* counter_resolution,
                                        uint64_t* timestamp) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, energy_accumulator, counter_resolution,
                      timestamp)) {
    return st;
  }
  if (energy_accumulator == nullptr || counter_resolution == nullptr || timestamp == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  const uint64_t t = now_ms();
  *energy_accumulator = t * (p->gpu + 1);
  *counter_resolution = 15.3f;
  *timestamp = t * 1000000;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_ecc_count(amdsmi_processor_handle processor_handle,
                                         amdsmi_gpu_block_t block, amdsmi_error_count_t* ec) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, ec)) return st;
  if (ec == nullptr) return AMDSMI_STATUS_INVAL;
  memset(ec, 0, sizeof(*ec));
  if (block == AMDSMI_GPU_BLOCK_UMC) ec->correctable_count = V(kECC_UMC_CORRECTABLE, p->gpu);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_ecc_status(amdsmi_processor_handle processor_handle,
                                          amdsmi_gpu_block_t block, amdsmi_ras_err_state_t* state) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, state)) return st;
  if (state == nullptr) return AMDSMI_STATUS_INVAL;
  (void)block;
  *state = AMDSMI_RAS_ERR_STATE_ENABLED;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_bad_page_info(amdsmi_processor_handle processor_handle,
                                             uint32_t* num_pages,
                                             amdsmi_retired_page_record_t* info) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, num_pages)) return st;
  if (num_pages == nullptr) return AMDSMI_STATUS_INVAL;
  const uint32_t pages = p->gpu % 3;
  if (info == nullptr) {
    *num_pages = pages;
    return AMDSMI_STATUS_SUCCESS;
  }
  const uint32_t n = std::min(*num_pages, pages);
  for (uint32_t i = 0; i < n; ++i) {
    info[i].page_address = 0x100000ULL * (i + 1);
    info[i].page_size = 4096;
    info[i].status = i == 0 ? AMDSMI_MEM_PAGE_STATUS_PENDING : AMDSMI_MEM_PAGE_STATUS_RESERVED;
  }
  *num_pages = n;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_bad_page_threshold(amdsmi_processor_handle processor_handle,
                                                  uint32_t* threshold) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, threshold)) return st;
  if (threshold == nullptr) return AMDSMI_STATUS_INVAL;
  *threshold = 128;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_gpu_validate_ras_eeprom(amdsmi_processor_handle processor_handle) {
  const FakeProc* p;
  return enter(__func__, processor_handle, &p);
}

amdsmi_status_t amdsmi_gpu_xgmi_error_status(amdsmi_processor_handle processor_handle,
                                             amdsmi_xgmi_status_t* status) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, status)) return st;
  if (status == nullptr) return AMDSMI_STATUS_INVAL;
  *status = AMDSMI_XGMI_STATUS_NO_ERRORS;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_xgmi_link_status(amdsmi_processor_handle processor_handle,
                                                amdsmi_xgmi_link_status_t* link_status) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, link_status)) return st;
  if (link_status == nullptr) return AMDSMI_STATUS_INVAL;
  memset(link_status, 0, sizeof(*link_status));
  // One link per possible peer; links to GPUs in the same hive are up.
  link_status->total_links = AMDSMI_MAX_NUM_XGMI_LINKS;
  for (uint32_t i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i) {
    const uint32_t peer = (p->gpu + 1 + i) % g_cfg.gpus;
    bool up = peer != p->gpu && i + 1 < g_cfg.gpus && peer % g_cfg.hives == p->gpu % g_cfg.hives;
    link_status->status[i] = up ? AMDSMI_XGMI_LINK_UP : AMDSMI_XGMI_LINK_DOWN;
  }
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_topo_get_link_type(amdsmi_processor_handle processor_handle_src,
                                          amdsmi_processor_handle processor_handle_dst,
                                          uint64_t* hops, amdsmi_link_type_t* type) {
  const FakeProc* a;
  if (auto st = enter(__func__, processor_handle_src, &a, hops, type)) return st;
  const FakeProc* b = to_proc(processor_handle_dst);
  if (b == nullptr || hops == nullptr || type == nullptr) return AMDSMI_STATUS_INVAL;
  if (a->gpu == b->gpu) {
    *hops = 0;
    *type = AMDSMI_LINK_TYPE_INTERNAL;
  } else if (same_hive(a, b)) {
    *hops = 1;
    *type = AMDSMI_LINK_TYPE_XGMI;
  } else {
    *hops = 2;
    *type = AMDSMI_LINK_TYPE_PCIE;
  }
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_topo_get_link_weight(amdsmi_processor_handle processor_handle_src,
                                            amdsmi_processor_handle processor_handle_dst,
                                            uint64_t* weight) {
  const FakeProc* a;
  if (auto st = enter(__func__, processor_handle_src, &a, weight)) return st;
  const FakeProc* b = to_proc(processor_handle_dst);
  if (b == nullptr || weight == nullptr) return AMDSMI_STATUS_INVAL;
  *weight = a->gpu == b->gpu ? 0 : (same_hive(a, b) ? 15 : 40);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_minmax_bandwidth_between_processors(
    amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst,
    uint64_t* min_bandwidth, uint64_t* max_bandwidth) {
  const FakeProc* a;
  if (auto st = enter(__func__, processor_handle_src, &a, min_bandwidth, max_bandwidth)) {
    return st;
  }
  const FakeProc* b = to_proc(processor_handle_dst);
  if (b == nullptr || min_bandwidth == nullptr || max_bandwidth == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  if (a->gpu == b->gpu || !same_hive(a, b)) return AMDSMI_STATUS_NOT_SUPPORTED;
  *min_bandwidth = 50000;
  *max_bandwidth = 64000;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_is_P2P_accessible(amdsmi_processor_handle processor_handle_src,
                                         amdsmi_processor_handle processor_handle_dst,
                                         bool* accessible) {
  const FakeProc* src;
  if (auto st = enter(__func__, processor_handle_src, &src, accessible)) return st;
  const FakeProc* dst = to_proc(processor_handle_dst);
  if (dst == nullptr || accessible == nullptr) return AMDSMI_STATUS_INVAL;
  *accessible = src->gpu != dst->gpu && same_hive(src, dst);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_topo_get_numa_node_number(amdsmi_processor_handle processor_handle,
                                                 uint32_t* numa_node) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, numa_node)) return st;
  if (numa_node == nullptr) return AMDSMI_STATUS_INVAL;
  *numa_node = p->gpu % 2;
  return AMDSMI_STATUS_SUCCESS;
}

// The partition modes the fake lists, as an MI300X does: SPX, DPX, QPX and CPX (1, 2, 4 and 8
// partitions), and the configured count as a mode of its own when it is none of those. The
// active profile has the index of the configured count's mode.
struct PartitionModes {
  uint32_t parts[5];
  uint32_t count;
  uint32_t active;
};

PartitionModes partition_modes() {
  PartitionModes m = {{1, 2, 4, 8}, 4, 0};
  while (m.active < m.count && m.parts[m.active] != g_cfg.parts) ++m.active;
  if (m.active == m.count) m.parts[m.count++] = g_cfg.parts;
  return m;
}

amdsmi_status_t amdsmi_get_gpu_accelerator_partition_profile(
    amdsmi_processor_handle processor_handle, amdsmi_accelerator_partition_profile_t* profile,
    uint32_t* partition_id) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, profile, partition_id)) return st;
  if (profile == nullptr || partition_id == nullptr) return AMDSMI_STATUS_INVAL;
  if (!g_cfg.partitionable) return AMDSMI_STATUS_NOT_SUPPORTED;
  memset(profile, 0, sizeof(*profile));
  profile->profile_type = g_cfg.parts == 1   ? AMDSMI_ACCELERATOR_PARTITION_SPX
                          : g_cfg.parts == 2 ? AMDSMI_ACCELERATOR_PARTITION_DPX
                          : g_cfg.parts == 4 ? AMDSMI_ACCELERATOR_PARTITION_QPX
                                             : AMDSMI_ACCELERATOR_PARTITION_CPX;
  profile->num_partitions = g_cfg.parts;
  profile->profile_index = partition_modes().active;
  profile->memory_caps.nps_flags.nps1_cap = 1;
  *partition_id = p->part;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_accelerator_partition_profile_config(
    amdsmi_processor_handle processor_handle,
    amdsmi_accelerator_partition_profile_config_t* profile_config) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, profile_config)) return st;
  if (profile_config == nullptr) return AMDSMI_STATUS_INVAL;
  if (!g_cfg.partitionable) return AMDSMI_STATUS_NOT_SUPPORTED;
  memset(profile_config, 0, sizeof(*profile_config));
  const PartitionModes modes = partition_modes();
  profile_config->num_profiles = modes.count;
  for (uint32_t i = 0; i < modes.count; ++i) {
    profile_config->profiles[i].num_partitions = modes.parts[i];
    profile_config->profiles[i].profile_index = i;
  }
  // One XCC and one decoder resource profile per partition mode (8 XCCs, 4 decoders).
  uint32_t n = 0;
  for (uint32_t i = 0; i < modes.count; ++i) {
    auto& xcc = profile_config->resource_profiles[n++];
    xcc.profile_index = i;
    xcc.resource_type = AMDSMI_ACCELERATOR_XCC;
    xcc.partition_resource = 8 / modes.parts[i];
    xcc.num_partitions_share_resource = 1;
    auto& dec = profile_config->resource_profiles[n++];
    dec.profile_index = i;
    dec.resource_type = AMDSMI_ACCELERATOR_DECODER;
    dec.partition_resource = std::max(1u, 4 / modes.parts[i]);
    dec.num_partitions_share_resource = modes.parts[i] > 4 ? 2 : 1;
  }
  profile_config->num_resource_profiles = n;
  return AMDSMI_STATUS_SUCCESS;
}

// No compute processes are simulated.
amdsmi_status_t amdsmi_get_gpu_compute_process_info(amdsmi_process_info_t* procs,
                                                    uint32_t* num_items) {
  if (auto st = enter_global(__func__, num_items)) return st;
  if (num_items == nullptr) return AMDSMI_STATUS_INVAL;
  (void)procs;
  *num_items = 0;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_compute_process_gpus(uint32_t pid, uint32_t* dv_indices,
                                                    uint32_t* num_devices) {
  if (auto st = enter_global(__func__, num_devices)) return st;
  if (num_devices == nullptr) return AMDSMI_STATUS_INVAL;
  (void)pid;
  (void)dv_indices;
  return AMDSMI_STATUS_NOT_FOUND;
}

amdsmi_status_t amdsmi_init_gpu_event_notification(amdsmi_processor_handle processor_handle) {
  const FakeProc* p;
  return enter(__func__, processor_handle, &p);
}

amdsmi_status_t amdsmi_set_gpu_event_notification_mask(amdsmi_processor_handle processor_handle,
                                                       uint64_t mask) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p)) return st;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_state[p->gpu].event_mask = mask;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_event_notification(int timeout_ms, uint32_t* num_elem,
                                                  amdsmi_evt_notification_data_t* data) {
  if (auto st = enter_global(__func__, num_elem)) return st;
  if (num_elem == nullptr || data == nullptr) return AMDSMI_STATUS_INVAL;
  const int wait = std::min<int>(std::max(timeout_ms, 0), static_cast<int>(g_cfg.event_wait_ms));
  std::this_thread::sleep_for(std::chrono::milliseconds(wait));
  *num_elem = 0;
  return AMDSMI_STATUS_NO_DATA;
}

amdsmi_status_t amdsmi_stop_gpu_event_notification(amdsmi_processor_handle processor_handle) {
  const FakeProc* p;
  return enter(__func__, processor_handle, &p);
}

amdsmi_status_t amdsmi_gpu_counter_group_supported(amdsmi_processor_handle processor_handle,
                                                   amdsmi_event_group_t group) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p)) return st;
  return group == AMDSMI_EVNT_GRP_XGMI || group == AMDSMI_EVNT_GRP_XGMI_DATA_OUT
             ? AMDSMI_STATUS_SUCCESS
             : AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_gpu_available_counters(amdsmi_processor_handle processor_handle,
                                                  amdsmi_event_group_t grp, uint32_t* available) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, available)) return st;
  if (available == nullptr) return AMDSMI_STATUS_INVAL;
  (void)grp;
  *available = 4;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_gpu_create_counter(amdsmi_processor_handle processor_handle,
                                          amdsmi_event_type_t type,
                                          amdsmi_event_handle_t* evnt_handle) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, evnt_handle)) return st;
  if (evnt_handle == nullptr) return AMDSMI_STATUS_INVAL;
  (void)type;
  std::lock_guard<std::mutex> lock(g_mutex);
  *evnt_handle = g_next_counter;
  g_counters[g_next_counter++] = {p->gpu, now_ms()};
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_gpu_control_counter(amdsmi_event_handle_t evt_handle,
                                           amdsmi_counter_command_t cmd, void* cmd_args) {
  if (auto st = enter_global(__func__)) return st;
  (void)cmd;
  (void)cmd_args;
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_counters.count(evt_handle) != 0 ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_INVAL;
}

amdsmi_status_t amdsmi_gpu_read_counter(amdsmi_event_handle_t evt_handle,
                                        amdsmi_counter_value_t* value) {
  if (auto st = enter_global(__func__, value)) return st;
  if (value == nullptr) return AMDSMI_STATUS_INVAL;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_counters.find(evt_handle);
  if (it == g_counters.end()) return AMDSMI_STATUS_INVAL;
  const uint64_t elapsed = now_ms() - it->second.second;
  value->value = elapsed * 1000 * (it->second.first + 1);
  value->time_enabled = elapsed * 1000000;
  value->time_running = value->time_enabled;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_gpu_destroy_counter(amdsmi_event_handle_t evnt_handle) {
  if (auto st = enter_global(__func__)) return st;
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_counters.erase(evnt_handle) != 0 ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_INVAL;
}

amdsmi_status_t amdsmi_get_utilization_count(amdsmi_processor_handle processor_handle,
                                             amdsmi_utilization_counter_t utilization_counters[],
                                             uint32_t count, uint64_t* timestamp) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, timestamp)) return st;
  if (utilization_counters == nullptr || timestamp == nullptr || count == 0) {
    return AMDSMI_STATUS_INVAL;
  }
  const uint64_t t = now_ms();
  for (uint32_t i = 0; i < count; ++i) {
    utilization_counters[i].value = t * (p->gpu + 1);
    utilization_counters[i].fine_value_count = 0;
  }
  *timestamp = t * 1000000;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_violation_status(amdsmi_processor_handle processor_handle,
                                            amdsmi_violation_status_t* info) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, info)) return st;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  memset(static_cast<void*>(info), 0, sizeof(*info));
  info->reference_timestamp = now_ms() * 1000;
  info->acc_counter = now_ms();
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_cper_entries(amdsmi_processor_handle processor_handle,
                                            uint32_t severity_mask, char* cper_data,
                                            uint64_t* buf_size, amdsmi_cper_hdr_t** cper_hdrs,
                                            uint64_t* entry_count, uint64_t* cursor) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, entry_count)) return st;
  if (cper_data == nullptr || buf_size == nullptr || cper_hdrs == nullptr ||
      entry_count == nullptr || cursor == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  (void)severity_mask;
  *buf_size = 0;
  *entry_count = 0;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_afids_from_cper(char* cper_buffer, uint32_t buf_size, uint64_t* afids,
                                           uint32_t* num_afids) {
  if (auto st = enter_global(__func__, num_afids)) return st;
  if (cper_buffer == nullptr || afids == nullptr || num_afids == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  (void)buf_size;
  *num_afids = 0;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_reset_gpu(amdsmi_processor_handle processor_handle) {
  const FakeProc* p;
  return enter(__func__, processor_handle, &p);
}

#ifdef ENABLE_ESMI_LIB
// No simulated CPUs: every CPU call RDC makes (the CPU fields in RdcMetricFetcherImpl.cc)
// reports not supported.
// clang-format off
#define FAKE_NO_CPU(name, ...)                    \
  amdsmi_status_t name(__VA_ARGS__) {             \
    if (auto st = enter_global(#name)) return st; \
    return AMDSMI_STATUS_NOT_SUPPORTED;           \
  }
FAKE_NO_CPU(amdsmi_get_cpu_socket_count, uint32_t*)
FAKE_NO_CPU(amdsmi_get_cpu_family, uint32_t*)
FAKE_NO_CPU(amdsmi_get_cpu_model, uint32_t*)
FAKE_NO_CPU(amdsmi_get_cpu_cores_per_socket, uint32_t, amdsmi_sock_info_t*)
FAKE_NO_CPU(amdsmi_get_cpu_cclk_limit, amdsmi_processor_handle, uint32_t*)
FAKE_NO_CPU(amdsmi_get_cpu_fclk_mclk, amdsmi_processor_handle, uint32_t*, uint32_t*)
FAKE_NO_CPU(amdsmi_get_cpu_hsmp_driver_version, amdsmi_processor_handle, amdsmi_hsmp_driver_version_t*)
FAKE_NO_CPU(amdsmi_get_cpu_hsmp_proto_ver, amdsmi_processor_handle, uint32_t*)
FAKE_NO_CPU(amdsmi_get_cpu_model_name, amdsmi_processor_handle, amdsmi_cpu_info_t*)
FAKE_NO_CPU(amdsmi_get_cpu_smu_fw_version, amdsmi_processor_handle, amdsmi_smu_fw_version_t*)
FAKE_NO_CPU(amdsmi_get_cpu_socket_c0_residency, amdsmi_processor_handle, uint32_t*)
FAKE_NO_CPU(amdsmi_get_cpu_socket_current_active_freq_limit, amdsmi_processor_handle, uint16_t*, char**)
FAKE_NO_CPU(amdsmi_get_cpu_socket_energy, amdsmi_processor_handle, uint64_t*)
FAKE_NO_CPU(amdsmi_get_cpu_socket_freq_range, amdsmi_processor_handle, uint16_t*, uint16_t*)
FAKE_NO_CPU(amdsmi_get_cpu_socket_lclk_dpm_level, amdsmi_processor_handle, uint8_t, amdsmi_dpm_level_t*)
#undef FAKE_NO_CPU
// clang-format on
#endif  // ENABLE_ESMI_LIB

}  // extern "C"
