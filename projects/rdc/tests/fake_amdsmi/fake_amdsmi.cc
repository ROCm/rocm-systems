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
// sensor, and are read by the Python tests too. RDC_FAKE_* environment variables reproduce
// lower-layer behavior that has caused RDC defects (failing calls, output buffers left
// unwritten, "not available" sentinels, partitions, split XGMI hives, stale handles, counter
// resets) and, where RDC must not trust its library blindly, break amd-smi's output contract
// on purpose. README.md in this directory lists them and the handle rules.
//
// Only the amdsmi_* functions RDC calls are implemented. The definitions are checked against
// amd_smi/amdsmi.h at compile time, so an API change breaks the build instead of the tests.

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
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
  uint32_t ms;  // RDC_FAKE_DELAY_MS: the delay
};

struct GpuState {
  uint64_t power_cap_uw = 750ULL * 1000 * 1000;
  uint64_t gfx_clk_limit[2] = {500, 2100};
  amdsmi_dev_perf_level_t perf_level = AMDSMI_DEV_PERF_LEVEL_AUTO;
  uint64_t event_mask = 0;
};

// One RDC_FAKE_EVENTS entry.
struct EventSpec {
  uint32_t event;  // amdsmi_evt_notification_type_t value, 0..15; RDC knows 1..13
  int64_t gpu;     // processor index, or -1 for a handle the fake never handed out
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
  std::vector<Rule> fail_once;      // each rule fails one call
  std::vector<Rule> unfilled;       // SUCCESS without writing the outputs
  std::vector<Rule> delay;          // sleep ms before the call
  int64_t cper_count = -1;          // CPER entries to report, -1 = none
  std::vector<EventSpec> events;    // events delivered once
  int64_t events_count = -1;        // event count to report, -1 = events.size()
  int64_t num_freq = -1;            // levels in every clock frequency table, -1 = normal
  uint32_t procs = 0;               // simulated compute processes
  bool procs_grow = false;          // one more process on every process-list call
  bool counter_reset = false;       // accumulated counters restart when armed
  bool version_build_null = false;  // amdsmi_get_lib_version() build string is NULL
  bool long_strings = false;        // ASIC info names fill their buffers, no NUL
  uint32_t gfxclk_holes = 0;        // leading current_gfxclks[] entries not available
};

std::mutex g_mutex;  // guards everything below that is not atomic
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
// Handles are (generation << 32) | index values, never pointers (socket handles also set bit
// 31). The amdsmi_shut_down() that drops the last reference starts a new generation, so an older
// handle is unknown, as to amd-smi after its last shutdown. The generation starts at 1: no handle
// is NULL.
static_assert(sizeof(uintptr_t) == 8, "handles hold a 32-bit generation above a 32-bit index");
std::atomic<uint64_t> g_generation{1};
std::vector<bool> g_fail_once_fired;       // one flag per RDC_FAKE_FAIL_ONCE rule
size_t g_events_next = 0;                  // RDC_FAKE_EVENTS: the next event to deliver
std::atomic<uint32_t> g_process_calls{0};  // RDC_FAKE_PROCS_GROW: process-list calls so far
std::atomic<uint64_t> g_arm_ms{0};  // 1 + now_ms() when RDC_FAKE_ARM first existed, 0 = not yet
amdsmi_cper_hdr_t g_cper_hdr;       // the header of every simulated CPER entry

uint64_t now_ms() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - g_start)
                                   .count());
}

bool armed();

// Milliseconds an accumulated counter has run. RDC_FAKE_COUNTER_RESET restarts every accumulated
// counter (energy, activity, PCIe, XGMI, utilization, violation and XGMI event counters) from 0
// when the RDC_FAKE_ARM file first exists, so they step backwards once, as after a GPU reset.
// Without RDC_FAKE_ARM nothing changes.
uint64_t counter_ms(uint64_t start_ms = 0) {
  const uint64_t t = now_ms();
  if (g_cfg.counter_reset && armed()) {
    const uint64_t arm = g_arm_ms.load();
    if (arm != 0) start_ms = std::max(start_ms, arm - 1);
  }
  return t > start_ms ? t - start_ms : 0;
}

// A knob value that does not parse aborts: it must not turn into a test that cannot fail.
int64_t parse_int(const char* knob, const std::string& s, int64_t lo, int64_t hi) {
  char* end = nullptr;
  errno = 0;
  const long long n = std::strtoll(s.c_str(), &end, 0);
  if (s.empty() || *end != '\0' || errno != 0 || n < lo || n > hi) {
    fprintf(stderr, "fake amd-smi: %s: '%s' is not a number from %lld to %lld\n", knob, s.c_str(),
            static_cast<long long>(lo), static_cast<long long>(hi));
    abort();
  }
  return n;
}

amdsmi_status_t parse_status(const char* env, const std::string& s) {
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
  for (const auto& n : kNames) {
    if (s == n.first) return n.second;
  }
  // Anything else must be a number: a typo must not turn into a test that cannot fail.
  const std::string knob = std::string(env) + " status";
  return static_cast<amdsmi_status_t>(parse_int(knob.c_str(), s, 0, 0xFFFFFFFF));
}

int64_t env_int(const char* name, int64_t dflt, int64_t lo, int64_t hi) {
  const char* v = getenv(name);
  return v == nullptr ? dflt : parse_int(name, v, lo, hi);
}

// last is the last processor index. With delay set, every rule needs "=ms" (0..60000) instead
// of an optional status.
std::vector<Rule> parse_rules(const char* env, amdsmi_status_t dflt, int64_t last,
                              bool delay = false) {
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
    Rule r{item, -1, dflt, 0};
    size_t eq = r.api.find('=');
    if (eq != std::string::npos) {
      const std::string value = r.api.substr(eq + 1);
      if (delay) {
        r.ms = static_cast<uint32_t>(parse_int(env, value, 0, 60000));
      } else {
        r.status = parse_status(env, value);
      }
      r.api.resize(eq);
    } else if (delay) {
      fprintf(stderr, "fake amd-smi: %s rule '%s' has no =ms\n", env, item.c_str());
      abort();
    }
    size_t at = r.api.find('@');
    if (at != std::string::npos) {
      r.gpu = parse_int(env, r.api.substr(at + 1), 0, last);
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

// RDC_FAKE_EVENTS=event@gpu,...[,count=N]: event is 0..15, the values
// amdsmi_evt_notification_type_t can hold; gpu is a processor index or x (a handle the fake never
// handed out); count is the number the call reports, which may exceed the events it writes.
void parse_events(const char* v, Config* c) {
  if (v == nullptr) return;
  std::string all(v);
  size_t pos = 0;
  while (pos <= all.size()) {
    size_t end = all.find(',', pos);
    if (end == std::string::npos) end = all.size();
    std::string item = all.substr(pos, end - pos);
    pos = end + 1;
    if (item.empty()) continue;
    if (item.rfind("count=", 0) == 0) {
      c->events_count = parse_int("RDC_FAKE_EVENTS count", item.substr(6), 0, 65536);
      continue;
    }
    const size_t at = item.find('@');
    if (at == std::string::npos) {
      fprintf(stderr, "fake amd-smi: RDC_FAKE_EVENTS item '%s' has no @gpu\n", item.c_str());
      abort();
    }
    const std::string gpu = item.substr(at + 1);
    const int64_t last = int64_t{c->gpus} * c->parts - 1;
    c->events.push_back(
        {static_cast<uint32_t>(parse_int("RDC_FAKE_EVENTS event", item.substr(0, at), 0, 15)),
         gpu == "x" ? -1 : parse_int("RDC_FAKE_EVENTS gpu", gpu, 0, last)});
  }
  if (c->events.empty() && c->events_count >= 0) {
    fprintf(stderr, "fake amd-smi: RDC_FAKE_EVENTS has a count but no event\n");
    abort();
  }
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
  const int64_t last = int64_t{c.gpus} * c.parts - 1;
  c.fail = parse_rules("RDC_FAKE_FAIL", AMDSMI_STATUS_UNKNOWN_ERROR, last);
  c.sentinel = parse_rules("RDC_FAKE_SENTINEL", AMDSMI_STATUS_SUCCESS, last);
  c.fail_once = parse_rules("RDC_FAKE_FAIL_ONCE", AMDSMI_STATUS_UNKNOWN_ERROR, last);
  c.unfilled = parse_rules("RDC_FAKE_UNFILLED", AMDSMI_STATUS_SUCCESS, last);
  c.delay = parse_rules("RDC_FAKE_DELAY_MS", AMDSMI_STATUS_SUCCESS, last, true);
  c.cper_count = env_int("RDC_FAKE_CPER_COUNT", -1, 0, 1 << 20);
  parse_events(getenv("RDC_FAKE_EVENTS"), &c);
  c.num_freq = env_int("RDC_FAKE_NUM_FREQ", -1, 0, AMDSMI_MAX_NUM_FREQUENCIES);
  c.procs = static_cast<uint32_t>(env_int("RDC_FAKE_PROCS", 0, 0, 1024));
  c.procs_grow = env_int("RDC_FAKE_PROCS_GROW", 0, 0, 1) != 0;
  c.counter_reset = env_int("RDC_FAKE_COUNTER_RESET", 0, 0, 1) != 0;
  if (c.counter_reset && c.arm_path.empty()) {
    fprintf(stderr,
            "fake amd-smi: RDC_FAKE_COUNTER_RESET resets when RDC_FAKE_ARM exists: set it\n");
    abort();
  }
  c.version_build_null = env_int("RDC_FAKE_VERSION_BUILD_NULL", 0, 0, 1) != 0;
  c.long_strings = env_int("RDC_FAKE_LONG_STRINGS", 0, 0, 1) != 0;
  c.gfxclk_holes =
      static_cast<uint32_t>(env_int("RDC_FAKE_GFXCLK_HOLES", 0, 0, AMDSMI_MAX_NUM_GFX_CLKS));
  g_cfg = c;
  g_fail_once_fired.assign(g_cfg.fail_once.size(), false);
  memcpy(g_cper_hdr.signature, "CPER", 4);
  g_cper_hdr.signature_end = 0xFFFFFFFF;
  g_cper_hdr.error_severity = AMDSMI_CPER_SEV_NON_FATAL_CORRECTED;
  g_cper_hdr.record_length = sizeof(g_cper_hdr);
  uint32_t flat = 0;
  for (uint32_t g = 0; g < c.gpus; ++g) {
    g_sockets[g].gpu = g;
    for (uint32_t p = 0; p < c.parts; ++p, ++flat) g_procs[flat] = {g, p, flat};
  }
  const char* log = getenv("RDC_FAKE_LOG");
  if (log != nullptr && (g_log = fopen(log, "ae")) == nullptr) {
    fprintf(stderr, "fake amd-smi: RDC_FAKE_LOG: cannot open %s\n", log);
    abort();
  }
  g_loaded = true;
}

bool initialized() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_init_refs > 0;
}

// Rules apply once RDC_FAKE_ARM names an existing file, and from then on, or always when it is
// unset, so a test can set up RDC before the faults start.
bool armed() {
  if (g_cfg.arm_path.empty() || g_arm_ms.load() != 0) return true;
  if (access(g_cfg.arm_path.c_str(), F_OK) != 0) return false;
  uint64_t not_yet = 0;
  g_arm_ms.compare_exchange_strong(not_yet, now_ms() + 1);
  return true;
}

// gpu is -1 for calls without a GPU argument; "*" matches only calls that take one.
bool matches(const Rule& r, const char* api, int64_t gpu) {
  return (r.api == api || (r.api == "*" && gpu >= 0)) && (r.gpu < 0 || r.gpu == gpu);
}

const Rule* match(const std::vector<Rule>& rules, const char* api, int64_t gpu) {
  if (rules.empty() || !armed()) return nullptr;
  for (const auto& r : rules) {
    if (matches(r, api, gpu)) return &r;
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

constexpr uint64_t kSocketBit = 1ULL << 31;

amdsmi_processor_handle make_proc_handle(uint64_t flat) {
  return reinterpret_cast<amdsmi_processor_handle>(
      static_cast<uintptr_t>((g_generation.load() << 32) | flat));
}

amdsmi_socket_handle make_socket_handle(uint64_t gpu) {
  return reinterpret_cast<amdsmi_socket_handle>(
      static_cast<uintptr_t>((g_generation.load() << 32) | kSocketBit | gpu));
}

// Like amd-smi: AMDSMI_STATUS_INVAL for NULL, AMDSMI_STATUS_NOT_FOUND for a value this generation
// did not hand out.
amdsmi_status_t lookup_proc(amdsmi_processor_handle h, const FakeProc** p) {
  const auto v = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(h));
  if (v == 0) return AMDSMI_STATUS_INVAL;
  const uint64_t low = v & 0xFFFFFFFFULL;
  if ((v >> 32) != g_generation.load() || (low & kSocketBit) != 0 ||
      low >= uint64_t{g_cfg.gpus} * g_cfg.parts) {
    return AMDSMI_STATUS_NOT_FOUND;
  }
  *p = &g_procs[low];
  return AMDSMI_STATUS_SUCCESS;
}

// Like amd-smi: AMDSMI_STATUS_INVAL for NULL and for any value this generation did not hand out.
amdsmi_status_t lookup_socket(amdsmi_socket_handle h, const FakeSocket** s) {
  const auto v = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(h));
  const uint64_t low = v & 0xFFFFFFFFULL;
  if ((v >> 32) != g_generation.load() || (low & kSocketBit) == 0 ||
      (low & ~kSocketBit) >= g_cfg.gpus) {
    return AMDSMI_STATUS_INVAL;
  }
  *s = &g_sockets[low & ~kSocketBit];
  return AMDSMI_STATUS_SUCCESS;
}

// Fills every output with 0xA5 when poisoning is on.
void poison() {}
template <typename T, typename... Rest>
void poison(T* out, Rest*... rest) {
  if (out != nullptr && g_cfg.poison) memset(static_cast<void*>(out), 0xA5, sizeof(T));
  poison(rest...);
}

// enter() result for RDC_FAKE_UNFILLED: the call returns SUCCESS without writing its outputs.
// Every call returns done(st) for whatever enter() gave it.
constexpr amdsmi_status_t kUnfilled = static_cast<amdsmi_status_t>(0x7FFFFFF0);

amdsmi_status_t done(amdsmi_status_t st) { return st == kUnfilled ? AMDSMI_STATUS_SUCCESS : st; }

// The first RDC_FAKE_FAIL_ONCE rule for this call that has not failed a call yet.
const Rule* fail_once(const char* api, int64_t gpu) {
  if (g_cfg.fail_once.empty() || !armed()) return nullptr;
  std::lock_guard<std::mutex> lock(g_mutex);
  for (size_t i = 0; i < g_cfg.fail_once.size(); ++i) {
    const Rule& r = g_cfg.fail_once[i];
    if (!g_fail_once_fired[i] && matches(r, api, gpu)) {
      g_fail_once_fired[i] = true;
      return &r;
    }
  }
  return nullptr;
}

// Applies RDC_FAKE_DELAY_MS, RDC_FAKE_FAIL, RDC_FAKE_FAIL_ONCE and RDC_FAKE_UNFILLED and logs the
// call. Returns SUCCESS when the call should go ahead.
template <typename... Outs>
amdsmi_status_t apply_rules(const char* api, int64_t gpu, Outs*... outs) {
  if (const Rule* d = match(g_cfg.delay, api, gpu)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(d->ms));
  }
  const Rule* r = match(g_cfg.fail, api, gpu);
  if (r == nullptr) r = fail_once(api, gpu);
  log_call(api, gpu, r != nullptr);
  if (r != nullptr) {
    poison(outs...);
    return r->status;
  }
  return match(g_cfg.unfilled, api, gpu) != nullptr ? kUnfilled : AMDSMI_STATUS_SUCCESS;
}

// Common entry for calls that take a GPU: validates the handle, then applies the rules.
template <typename... Outs>
amdsmi_status_t enter(const char* api, amdsmi_processor_handle h, const FakeProc** p,
                      Outs*... outs) {
  if (!initialized()) return AMDSMI_STATUS_NOT_INIT;
  if (auto st = lookup_proc(h, p)) return st;
  return apply_rules(api, (*p)->flat, outs...);
}

// Same for calls without a GPU argument.
template <typename... Outs>
amdsmi_status_t enter_global(const char* api, Outs*... outs) {
  if (!initialized()) return AMDSMI_STATUS_NOT_INIT;
  return apply_rules(api, -1, outs...);
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
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded) load_config_locked();
  }
  // Rules without @gpu apply; RDC_FAKE_UNFILLED changes nothing, as there are no outputs.
  if (auto st = done(apply_rules(__func__, -1))) return st;
  // No simulated CPUs: amd-smi fails a CPU init on hosts without the HSMP driver.
  if ((init_flags & AMDSMI_INIT_AMD_CPUS) != 0) return AMDSMI_STATUS_NOT_SUPPORTED;
  std::lock_guard<std::mutex> lock(g_mutex);
  ++g_init_refs;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_shut_down(void) {
  std::lock_guard<std::mutex> lock(g_mutex);
  write_log_locked("amdsmi_shut_down", -1, false);
  // Every processor and socket handle is stale now. Counter handles survive, as amd-smi's do.
  if (g_init_refs > 0 && --g_init_refs == 0) ++g_generation;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_lib_version(amdsmi_version_t* version) {
  {
    // Valid before amdsmi_init(): read the environment here if nothing else has.
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded) load_config_locked();
  }
  if (auto st = apply_rules(__func__, -1, version)) return done(st);
  if (version == nullptr) return AMDSMI_STATUS_INVAL;
  version->major = 27;
  version->minor = 0;
  version->release = 0;
  // RDC_FAKE_VERSION_BUILD_NULL breaks the contract: amd-smi's build string is never NULL.
  version->build = g_cfg.version_build_null ? nullptr : "fake";
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_socket_handles(uint32_t* socket_count,
                                          amdsmi_socket_handle* socket_handles) {
  if (auto st = enter_global(__func__, socket_count)) return done(st);
  if (socket_count == nullptr) return AMDSMI_STATUS_INVAL;
  if (socket_handles == nullptr) {
    *socket_count = g_cfg.gpus;
    return AMDSMI_STATUS_SUCCESS;
  }
  uint32_t n = std::min(*socket_count, g_cfg.gpus);
  for (uint32_t i = 0; i < n; ++i) socket_handles[i] = make_socket_handle(i);
  *socket_count = n;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_handles(amdsmi_socket_handle socket_handle,
                                             uint32_t* processor_count,
                                             amdsmi_processor_handle* processor_handles) {
  if (auto st = enter_global(__func__, processor_count)) return done(st);
  if (processor_count == nullptr) return AMDSMI_STATUS_INVAL;
  const FakeSocket* s = nullptr;
  if (auto st = lookup_socket(socket_handle, &s)) return st;
  if (processor_handles == nullptr) {
    *processor_count = g_cfg.parts;
    return AMDSMI_STATUS_SUCCESS;
  }
  uint32_t n = std::min(*processor_count, g_cfg.parts);
  for (uint32_t i = 0; i < n; ++i)
    processor_handles[i] = make_proc_handle(s->gpu * g_cfg.parts + i);
  *processor_count = n;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_type(amdsmi_processor_handle processor_handle,
                                          amdsmi_processor_type_t* processor_type) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, processor_type)) return done(st);
  if (processor_type == nullptr) return AMDSMI_STATUS_INVAL;
  *processor_type = AMDSMI_PROCESSOR_TYPE_AMD_GPU;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_asic_info(amdsmi_processor_handle processor_handle,
                                         amdsmi_asic_info_t* info) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, info)) return done(st);
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  memset(info, 0, sizeof(*info));
  snprintf(info->market_name, sizeof(info->market_name), "Fake AMD Instinct GPU %u", p->gpu);
  snprintf(info->vendor_name, sizeof(info->vendor_name), "Advanced Micro Devices, Inc.");
  snprintf(info->asic_serial, sizeof(info->asic_serial), "FAKE%04X", p->gpu);
  if (g_cfg.long_strings) {
    // RDC_FAKE_LONG_STRINGS breaks the contract: amd-smi terminates these strings.
    memset(info->market_name, 'M', sizeof(info->market_name));
    memset(info->asic_serial, 'S', sizeof(info->asic_serial));
  }
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
  if (auto st = enter(__func__, processor_handle, &p, info)) return done(st);
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
  if (auto st = enter(__func__, processor_handle, &p, temperature)) return done(st);
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
  if (auto st = enter(__func__, processor_handle, &p, voltage)) return done(st);
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
  if (auto st = enter(__func__, processor_handle, &p, info)) return done(st);
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
  if (auto st = enter(__func__, processor_handle, &p, info)) return done(st);
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
  if (auto st = enter(__func__, processor_handle, &p)) return done(st);
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
  if (auto st = enter(__func__, processor_handle, &p, info)) return done(st);
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  memset(info, 0, sizeof(*info));
  if (sentinel(__func__, p)) {
    // amd-smi copies gpu_metrics' uint16_t marker into these uint32_t fields.
    info->gfx_activity = info->umc_activity = info->mm_activity = UINT16_MAX;
    return AMDSMI_STATUS_SUCCESS;
  }
  info->gfx_activity = V(kGFX_ACTIVITY_PCT, p->gpu);
  info->umc_activity = V(kUMC_ACTIVITY_PCT, p->gpu);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_busy_percent(amdsmi_processor_handle processor_handle,
                                            uint32_t* gpu_busy_percent) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, gpu_busy_percent)) return done(st);
  if (gpu_busy_percent == nullptr) return AMDSMI_STATUS_INVAL;
  *gpu_busy_percent = V(kGFX_ACTIVITY_PCT, p->gpu);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_clock_info(amdsmi_processor_handle processor_handle,
                                      amdsmi_clk_type_t clk_type, amdsmi_clk_info_t* info) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, info)) return done(st);
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
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
  // amd-smi: the range is known, the current clock is not.
  if (sentinel(__func__, p)) info->clk = UINT32_MAX;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_clk_freq(amdsmi_processor_handle processor_handle,
                                    amdsmi_clk_type_t clk_type, amdsmi_frequencies_t* f) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, f)) return done(st);
  if (f == nullptr) return AMDSMI_STATUS_INVAL;
  memset(f, 0, sizeof(*f));
  // Same model as amdsmi_get_clock_info(): the current level is the per-GPU clock.
  const uint64_t mhz = 1000000;
  const bool mem = clk_type == AMDSMI_CLK_TYPE_MEM;
  if (g_cfg.num_freq >= 0 && armed()) {
    // RDC_FAKE_NUM_FREQ: a table with that many levels and no current level (UINT32_MAX), as
    // amd-smi reports when the driver marks none. An empty table breaks the contract: amd-smi
    // fails the call instead.
    f->num_supported = static_cast<uint32_t>(g_cfg.num_freq);
    f->current = UINT32_MAX;
    for (uint32_t i = 0; i < f->num_supported; ++i) f->frequency[i] = (500 + 100 * i) * mhz;
    return AMDSMI_STATUS_SUCCESS;
  }
  f->num_supported = 3;
  f->current = 1;
  f->frequency[0] = (mem ? 900 : 500) * mhz;
  f->frequency[1] = V(mem ? kMEM_CLK_MHZ : kGFX_CLK_MHZ, p->gpu) * mhz;
  f->frequency[2] = (mem ? 1300 : 2100) * mhz;
  // amd-smi: no level is current, as for a power-gated clock domain.
  if (sentinel(__func__, p)) f->current = UINT32_MAX;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_gpu_clk_limit(amdsmi_processor_handle processor_handle,
                                         amdsmi_clk_type_t clk_type,
                                         amdsmi_clk_limit_type_t limit_type, uint64_t clk_value) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p)) return done(st);
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
  if (auto st = enter(__func__, processor_handle, &p)) return done(st);
  std::lock_guard<std::mutex> lock(g_mutex);
  g_state[p->gpu].perf_level = perf_lvl;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_memory_total(amdsmi_processor_handle processor_handle,
                                            amdsmi_memory_type_t mem_type, uint64_t* total) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, total)) return done(st);
  if (total == nullptr) return AMDSMI_STATUS_INVAL;
  *total =
      (mem_type == AMDSMI_MEM_TYPE_GTT ? 128 : V(kVRAM_TOTAL_GIB, p->gpu)) * kGiB / g_cfg.parts;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_memory_usage(amdsmi_processor_handle processor_handle,
                                            amdsmi_memory_type_t mem_type, uint64_t* used) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, used)) return done(st);
  if (used == nullptr) return AMDSMI_STATUS_INVAL;
  *used = (mem_type == AMDSMI_MEM_TYPE_GTT ? 1 : V(kVRAM_USED_GIB, p->gpu)) * kGiB / g_cfg.parts;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_vram_info(amdsmi_processor_handle processor_handle,
                                         amdsmi_vram_info_t* info) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, info)) return done(st);
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
  if (auto st = enter(__func__, processor_handle, &p, pgpu_metrics)) return done(st);
  if (pgpu_metrics == nullptr) return AMDSMI_STATUS_INVAL;
  // amd-smi reports every metric the firmware does not provide as all bits set.
  memset(static_cast<void*>(pgpu_metrics), 0xFF, sizeof(*pgpu_metrics));
  pgpu_metrics->apu_metrics = nullptr;  // not an APU
  if (sentinel(__func__, p)) return AMDSMI_STATUS_SUCCESS;
  amdsmi_gpu_metrics_t& m = *pgpu_metrics;
  const uint64_t t = now_ms();
  const uint64_t acc = counter_ms();
  m.temperature_edge = V(kTEMP_EDGE_C, p->gpu);
  m.temperature_hotspot = V(kTEMP_HOTSPOT_C, p->gpu);
  m.temperature_mem = V(kTEMP_MEM_C, p->gpu);
  m.average_gfx_activity = V(kGFX_ACTIVITY_PCT, p->gpu);
  m.average_umc_activity = V(kUMC_ACTIVITY_PCT, p->gpu);
  m.average_socket_power = V(kSOCKET_POWER_W, p->gpu);
  m.current_socket_power = V(kSOCKET_POWER_W, p->gpu);
  m.energy_accumulator = acc * (p->gpu + 1);
  m.current_gfxclk = V(kGFX_CLK_MHZ, p->gpu);
  m.current_uclk = V(kMEM_CLK_MHZ, p->gpu);
  m.gfx_activity_acc = acc * V(kGFX_ACTIVITY_PCT, p->gpu);
  m.mem_activity_acc = acc * V(kUMC_ACTIVITY_PCT, p->gpu);
  m.firmware_timestamp = t * 100000;  // 10 ns units
  m.system_clock_counter = t * 1000000;
  m.pcie_link_width = 16;
  m.pcie_link_speed = 320;  // 0.1 GT/s units
  m.pcie_bandwidth_acc = acc * (p->gpu + 1);
  m.pcie_bandwidth_inst = V(kPCIE_BANDWIDTH_MBPS, p->gpu);
  m.xgmi_link_width = 16;
  m.xgmi_link_speed = 32;
  for (uint32_t i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i) {
    m.xgmi_read_data_acc[i] = acc * (p->gpu + 1) * (i + 1);
    m.xgmi_write_data_acc[i] = acc * (p->gpu + 1) * (i + 2);
  }
  for (uint32_t i = 0; i < AMDSMI_MAX_NUM_GFX_CLKS; ++i) {
    m.current_gfxclks[i] = i < g_cfg.gfxclk_holes ? UINT16_MAX : m.current_gfxclk;
  }
  m.vcn_activity[0] = V(kVCN_ACTIVITY_PCT, p->gpu);  // one decoder, the others not available
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
  if (auto st = enter(__func__, processor_handle, &p, info)) return done(st);
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
  if (auto st = enter(__func__, processor_handle, &p, sent, received, max_pkt_sz)) return done(st);
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
    return done(st);
  }
  if (energy_accumulator == nullptr || counter_resolution == nullptr || timestamp == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  *energy_accumulator = counter_ms() * (p->gpu + 1);
  *counter_resolution = 15.3f;
  *timestamp = now_ms() * 1000000;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_ecc_count(amdsmi_processor_handle processor_handle,
                                         amdsmi_gpu_block_t block, amdsmi_error_count_t* ec) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, ec)) return done(st);
  if (ec == nullptr) return AMDSMI_STATUS_INVAL;
  memset(ec, 0, sizeof(*ec));
  if (block == AMDSMI_GPU_BLOCK_UMC) ec->correctable_count = V(kECC_UMC_CORRECTABLE, p->gpu);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_ecc_status(amdsmi_processor_handle processor_handle,
                                          amdsmi_gpu_block_t block, amdsmi_ras_err_state_t* state) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, state)) return done(st);
  if (state == nullptr) return AMDSMI_STATUS_INVAL;
  (void)block;
  *state = AMDSMI_RAS_ERR_STATE_ENABLED;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_bad_page_info(amdsmi_processor_handle processor_handle,
                                             uint32_t* num_pages,
                                             amdsmi_retired_page_record_t* info) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, num_pages)) return done(st);
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
  if (auto st = enter(__func__, processor_handle, &p, threshold)) return done(st);
  if (threshold == nullptr) return AMDSMI_STATUS_INVAL;
  *threshold = 128;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_gpu_validate_ras_eeprom(amdsmi_processor_handle processor_handle) {
  const FakeProc* p;
  return done(enter(__func__, processor_handle, &p));
}

amdsmi_status_t amdsmi_gpu_xgmi_error_status(amdsmi_processor_handle processor_handle,
                                             amdsmi_xgmi_status_t* status) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, status)) return done(st);
  if (status == nullptr) return AMDSMI_STATUS_INVAL;
  *status = AMDSMI_XGMI_STATUS_NO_ERRORS;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_xgmi_link_status(amdsmi_processor_handle processor_handle,
                                                amdsmi_xgmi_link_status_t* link_status) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, link_status)) return done(st);
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
  if (auto st = enter(__func__, processor_handle_src, &a, hops, type)) return done(st);
  const FakeProc* b;
  if (auto st = lookup_proc(processor_handle_dst, &b)) return st;
  if (hops == nullptr || type == nullptr) return AMDSMI_STATUS_INVAL;
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
  if (auto st = enter(__func__, processor_handle_src, &a, weight)) return done(st);
  const FakeProc* b;
  if (auto st = lookup_proc(processor_handle_dst, &b)) return st;
  if (weight == nullptr) return AMDSMI_STATUS_INVAL;
  *weight = a->gpu == b->gpu ? 0 : (same_hive(a, b) ? 15 : 40);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_minmax_bandwidth_between_processors(
    amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst,
    uint64_t* min_bandwidth, uint64_t* max_bandwidth) {
  const FakeProc* a;
  if (auto st = enter(__func__, processor_handle_src, &a, min_bandwidth, max_bandwidth)) {
    return done(st);
  }
  const FakeProc* b;
  if (auto st = lookup_proc(processor_handle_dst, &b)) return st;
  if (min_bandwidth == nullptr || max_bandwidth == nullptr) return AMDSMI_STATUS_INVAL;
  if (a->gpu == b->gpu || !same_hive(a, b)) return AMDSMI_STATUS_NOT_SUPPORTED;
  *min_bandwidth = 50000;
  *max_bandwidth = 64000;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_is_P2P_accessible(amdsmi_processor_handle processor_handle_src,
                                         amdsmi_processor_handle processor_handle_dst,
                                         bool* accessible) {
  const FakeProc* src;
  if (auto st = enter(__func__, processor_handle_src, &src, accessible)) return done(st);
  const FakeProc* dst;
  if (auto st = lookup_proc(processor_handle_dst, &dst)) return st;
  if (accessible == nullptr) return AMDSMI_STATUS_INVAL;
  *accessible = src->gpu != dst->gpu && same_hive(src, dst);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_topo_get_numa_node_number(amdsmi_processor_handle processor_handle,
                                                 uint32_t* numa_node) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, numa_node)) return done(st);
  if (numa_node == nullptr) return AMDSMI_STATUS_INVAL;
  *numa_node = p->gpu % 2;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_accelerator_partition_profile(
    amdsmi_processor_handle processor_handle, amdsmi_accelerator_partition_profile_t* profile,
    uint32_t* partition_id) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, profile, partition_id)) return done(st);
  if (profile == nullptr || partition_id == nullptr) return AMDSMI_STATUS_INVAL;
  if (!g_cfg.partitionable) return AMDSMI_STATUS_NOT_SUPPORTED;
  memset(profile, 0, sizeof(*profile));
  profile->profile_type = g_cfg.parts == 1   ? AMDSMI_ACCELERATOR_PARTITION_SPX
                          : g_cfg.parts == 2 ? AMDSMI_ACCELERATOR_PARTITION_DPX
                          : g_cfg.parts == 4 ? AMDSMI_ACCELERATOR_PARTITION_QPX
                                             : AMDSMI_ACCELERATOR_PARTITION_CPX;
  profile->num_partitions = g_cfg.parts;
  profile->profile_index = g_cfg.parts == 1 ? 0 : (g_cfg.parts == 2 ? 1 : 2);
  profile->memory_caps.nps_flags.nps1_cap = 1;
  *partition_id = p->part;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_accelerator_partition_profile_config(
    amdsmi_processor_handle processor_handle,
    amdsmi_accelerator_partition_profile_config_t* profile_config) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, profile_config)) return done(st);
  if (profile_config == nullptr) return AMDSMI_STATUS_INVAL;
  if (!g_cfg.partitionable) return AMDSMI_STATUS_NOT_SUPPORTED;
  memset(profile_config, 0, sizeof(*profile_config));
  const uint32_t kParts[] = {1, 2, 8};
  profile_config->num_profiles = 3;
  for (uint32_t i = 0; i < 3; ++i) {
    profile_config->profiles[i].num_partitions = kParts[i];
    profile_config->profiles[i].profile_index = i;
  }
  // One XCC and one decoder resource profile per partition mode (8 XCCs, 4 decoders).
  uint32_t n = 0;
  for (uint32_t i = 0; i < 3; ++i) {
    auto& xcc = profile_config->resource_profiles[n++];
    xcc.profile_index = i;
    xcc.resource_type = AMDSMI_ACCELERATOR_XCC;
    xcc.partition_resource = 8 / kParts[i];
    xcc.num_partitions_share_resource = 1;
    auto& dec = profile_config->resource_profiles[n++];
    dec.profile_index = i;
    dec.resource_type = AMDSMI_ACCELERATOR_DECODER;
    dec.partition_resource = std::max(1u, 4 / kParts[i]);
    dec.num_partitions_share_resource = kParts[i] > 4 ? 2 : 1;
  }
  profile_config->num_resource_profiles = n;
  return AMDSMI_STATUS_SUCCESS;
}

// RDC_FAKE_PROCS compute processes, pid 1000 + i on GPU (socket) i % RDC_FAKE_GPUS; with
// RDC_FAKE_PROCS_GROW one more on every call. Same count rules as amd-smi (rocm-smi): a buffer too
// small for every process gets AMDSMI_STATUS_INSUFFICIENT_SIZE and keeps *num_items.
amdsmi_status_t amdsmi_get_gpu_compute_process_info(amdsmi_process_info_t* procs,
                                                    uint32_t* num_items) {
  if (auto st = enter_global(__func__, num_items)) return done(st);
  if (num_items == nullptr) return AMDSMI_STATUS_INVAL;
  uint32_t found = g_cfg.procs;
  if (g_cfg.procs_grow) found += g_process_calls.fetch_add(1);
  if (procs != nullptr) {
    for (uint32_t i = 0; i < std::min(*num_items, found); ++i) {
      if (*num_items < found) {
        procs[i].process_id = 1000 + i;  // amd-smi fills in nothing else then
        continue;
      }
      memset(&procs[i], 0, sizeof(procs[i]));
      procs[i].process_id = 1000 + i;
      procs[i].vram_usage = (i + 1) * kGiB / 4;
      procs[i].cu_occupancy = 1 + i % 64;
    }
    if (*num_items < found) return AMDSMI_STATUS_INSUFFICIENT_SIZE;
  }
  if (procs == nullptr || *num_items > found) *num_items = found;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_compute_process_gpus(uint32_t pid, uint32_t* dv_indices,
                                                    uint32_t* num_devices) {
  if (auto st = enter_global(__func__, num_devices)) return done(st);
  if (num_devices == nullptr) return AMDSMI_STATUS_INVAL;
  // The processes the last process-list call reported.
  const uint32_t calls = g_process_calls.load();
  if (pid < 1000 || pid - 1000 >= g_cfg.procs + (calls > 0 ? calls - 1 : 0)) {
    return AMDSMI_STATUS_NOT_FOUND;
  }
  // Each process runs on one GPU. As with amd-smi, a NULL list asks for the count, and a list
  // without room gets AMDSMI_STATUS_INSUFFICIENT_SIZE and keeps *num_devices.
  if (dv_indices != nullptr) {
    if (*num_devices < 1) return AMDSMI_STATUS_INSUFFICIENT_SIZE;
    dv_indices[0] = (pid - 1000) % g_cfg.gpus;
  }
  *num_devices = 1;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_init_gpu_event_notification(amdsmi_processor_handle processor_handle) {
  const FakeProc* p;
  return done(enter(__func__, processor_handle, &p));
}

amdsmi_status_t amdsmi_set_gpu_event_notification_mask(amdsmi_processor_handle processor_handle,
                                                       uint64_t mask) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p)) return done(st);
  std::lock_guard<std::mutex> lock(g_mutex);
  g_state[p->gpu].event_mask = mask;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_event_notification(int timeout_ms, uint32_t* num_elem,
                                                  amdsmi_evt_notification_data_t* data) {
  if (auto st = enter_global(__func__, num_elem)) return done(st);
  // amd-smi rejects a call without room for an event.
  if (num_elem == nullptr || data == nullptr || *num_elem == 0) return AMDSMI_STATUS_INVAL;
  if (!g_cfg.events.empty() && armed()) {
    // RDC_FAKE_EVENTS: once armed, each event is delivered once, as many per call as *num_elem
    // has room for, like amd-smi. With count= the first call that delivers reports that count
    // instead, which breaks the contract: amd-smi reports only the events it wrote.
    std::lock_guard<std::mutex> lock(g_mutex);
    const size_t first = g_events_next;
    const size_t n = std::min<size_t>(g_cfg.events.size() - first, *num_elem);
    if (n > 0) {
      g_events_next += n;
      for (size_t i = 0; i < n; ++i) {
        const EventSpec& e = g_cfg.events[first + i];
        amdsmi_evt_notification_data_t& d = data[i];
        memset(&d, 0, sizeof(d));
        // gpu -1 (x): a handle from generation 0, which this fake never hands out.
        d.processor_handle = e.gpu >= 0
                                 ? make_proc_handle(static_cast<uint64_t>(e.gpu))
                                 : reinterpret_cast<amdsmi_processor_handle>(uintptr_t{0xDEAD});
        d.event = static_cast<amdsmi_evt_notification_type_t>(e.event);
        snprintf(d.message, sizeof(d.message), "fake event %u", e.event);
      }
      *num_elem = static_cast<uint32_t>(
          first == 0 && g_cfg.events_count >= 0 ? static_cast<uint64_t>(g_cfg.events_count) : n);
      return AMDSMI_STATUS_SUCCESS;
    }
  }
  const int wait = std::min<int>(std::max(timeout_ms, 0), static_cast<int>(g_cfg.event_wait_ms));
  std::this_thread::sleep_for(std::chrono::milliseconds(wait));
  *num_elem = 0;
  return AMDSMI_STATUS_NO_DATA;
}

amdsmi_status_t amdsmi_stop_gpu_event_notification(amdsmi_processor_handle processor_handle) {
  const FakeProc* p;
  return done(enter(__func__, processor_handle, &p));
}

amdsmi_status_t amdsmi_gpu_counter_group_supported(amdsmi_processor_handle processor_handle,
                                                   amdsmi_event_group_t group) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p)) return done(st);
  return group == AMDSMI_EVNT_GRP_XGMI || group == AMDSMI_EVNT_GRP_XGMI_DATA_OUT
             ? AMDSMI_STATUS_SUCCESS
             : AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_gpu_available_counters(amdsmi_processor_handle processor_handle,
                                                  amdsmi_event_group_t grp, uint32_t* available) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, available)) return done(st);
  if (available == nullptr) return AMDSMI_STATUS_INVAL;
  (void)grp;
  *available = 4;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_gpu_create_counter(amdsmi_processor_handle processor_handle,
                                          amdsmi_event_type_t type,
                                          amdsmi_event_handle_t* evnt_handle) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, evnt_handle)) return done(st);
  if (evnt_handle == nullptr) return AMDSMI_STATUS_INVAL;
  (void)type;
  std::lock_guard<std::mutex> lock(g_mutex);
  *evnt_handle = g_next_counter;
  g_counters[g_next_counter++] = {p->gpu, now_ms()};
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_gpu_control_counter(amdsmi_event_handle_t evt_handle,
                                           amdsmi_counter_command_t cmd, void* cmd_args) {
  if (auto st = enter_global(__func__)) return done(st);
  (void)cmd;
  (void)cmd_args;
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_counters.count(evt_handle) != 0 ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_INVAL;
}

amdsmi_status_t amdsmi_gpu_read_counter(amdsmi_event_handle_t evt_handle,
                                        amdsmi_counter_value_t* value) {
  if (auto st = enter_global(__func__, value)) return done(st);
  if (value == nullptr) return AMDSMI_STATUS_INVAL;
  std::pair<uint32_t, uint64_t> counter;  // (gpu, start ms)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_counters.find(evt_handle);
    if (it == g_counters.end()) return AMDSMI_STATUS_INVAL;
    counter = it->second;
  }
  const uint64_t elapsed = counter_ms(counter.second);
  value->value = elapsed * 1000 * (counter.first + 1);
  value->time_enabled = elapsed * 1000000;
  value->time_running = value->time_enabled;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_gpu_destroy_counter(amdsmi_event_handle_t evnt_handle) {
  if (auto st = enter_global(__func__)) return done(st);
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_counters.erase(evnt_handle) != 0 ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_INVAL;
}

amdsmi_status_t amdsmi_get_utilization_count(amdsmi_processor_handle processor_handle,
                                             amdsmi_utilization_counter_t utilization_counters[],
                                             uint32_t count, uint64_t* timestamp) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, timestamp)) return done(st);
  if (utilization_counters == nullptr || timestamp == nullptr || count == 0) {
    return AMDSMI_STATUS_INVAL;
  }
  const uint64_t t = now_ms();
  const uint64_t acc = counter_ms();
  for (uint32_t i = 0; i < count; ++i) {
    // As in amd-smi, the decoder activity is the average of the GPU metrics' VCN activities
    // that are available, a percentage: here vcn_activity[0] alone. The other counters
    // accumulate.
    const bool decoder = utilization_counters[i].type == AMDSMI_COARSE_DECODER_ACTIVITY;
    utilization_counters[i].value = decoder ? V(kVCN_ACTIVITY_PCT, p->gpu) : acc * (p->gpu + 1);
    utilization_counters[i].fine_value_count = 0;
  }
  *timestamp = t * 1000000;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_violation_status(amdsmi_processor_handle processor_handle,
                                            amdsmi_violation_status_t* info) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, info)) return done(st);
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  // amd-smi starts every counter at UINT64_MAX and fills in what the GPU provides.
  if (sentinel(__func__, p)) return fill_sentinel(info);
  memset(static_cast<void*>(info), 0, sizeof(*info));
  info->reference_timestamp = now_ms() * 1000;
  info->acc_counter = counter_ms();
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_cper_entries(amdsmi_processor_handle processor_handle,
                                            uint32_t severity_mask, char* cper_data,
                                            uint64_t* buf_size, amdsmi_cper_hdr_t** cper_hdrs,
                                            uint64_t* entry_count, uint64_t* cursor) {
  const FakeProc* p;
  if (auto st = enter(__func__, processor_handle, &p, entry_count)) return done(st);
  if (cper_data == nullptr || buf_size == nullptr || cper_hdrs == nullptr ||
      entry_count == nullptr || cursor == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  (void)severity_mask;
  if (g_cfg.cper_count >= 0 && armed()) {
    // RDC_FAKE_CPER_COUNT: report that many entries. Past the caller's capacity this breaks the
    // contract (amd-smi stops there with AMDSMI_STATUS_MORE_DATA); header pointers are written
    // only up to the capacity.
    const uint64_t n = static_cast<uint64_t>(g_cfg.cper_count);
    for (uint64_t i = 0; i < std::min(n, *entry_count); ++i) cper_hdrs[i] = &g_cper_hdr;
    *entry_count = n;
    *buf_size = 0;
    return AMDSMI_STATUS_SUCCESS;
  }
  *buf_size = 0;
  *entry_count = 0;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_afids_from_cper(char* cper_buffer, uint32_t buf_size, uint64_t* afids,
                                           uint32_t* num_afids) {
  if (auto st = enter_global(__func__, num_afids)) return done(st);
  if (cper_buffer == nullptr || afids == nullptr || num_afids == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  (void)buf_size;
  *num_afids = 0;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_reset_gpu(amdsmi_processor_handle processor_handle) {
  const FakeProc* p;
  return done(enter(__func__, processor_handle, &p));
}

#ifdef ENABLE_ESMI_LIB
// No simulated CPUs: every CPU call RDC makes (the CPU fields in RdcMetricFetcherImpl.cc)
// reports not supported.
// clang-format off
#define FAKE_NO_CPU(name, ...)                    \
  amdsmi_status_t name(__VA_ARGS__) {             \
    if (auto st = enter_global(#name)) return done(st); \
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
