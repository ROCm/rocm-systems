// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// rocprofiler-sdk backend plugin for gpumetrics. Samples GPU hardware
// performance counters via the rocprofiler-sdk "device counting" service, which
// reads counters directly from an agent without intercepting kernel launches, a
// good fit for a polling metrics tool.
//
// Bootstrap: the plugin exports `rocprofiler_configure` with default
// visibility. When the core dlopens this .so RTLD_GLOBAL and hsa_init() runs,
// rocprofiler-sdk finds that symbol in the global namespace and invokes
// tool_init(), which enumerates GPU agents and builds one CounterSampler each.
// init() first unsets HSA_TOOLS_LIB (a rocprofiler-v1 leftover that breaks the
// SDK), then calls hsa_init(). rocprofiler is process-global (single HSA), so
// tool_init is guarded to build samplers only once.

#include <dlfcn.h>
#include <hsa/hsa.h>
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/counter_config.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpumetrics/plugin_abi.h"
#include "gpumetrics/types.h"

namespace {

// Window counters accumulate over before a synchronous read: long enough for
// the free-running clock to advance, short enough to keep a poll responsive.
constexpr uint64_t kSampleWindowUs = 10000;

uint64_t NowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// CounterSampler: one per GPU agent. Owns a rocprofiler context for the
// device-counting service and greedily packs counter sets into as few HW configs
// as the slots allow. Not thread-safe; the plugin serializes reads with a mutex.
class CounterSampler {
 public:
  explicit CounterSampler(rocprofiler_agent_id_t agent) : agent_(agent) {
    if (rocprofiler_create_context(&ctx_) != ROCPROFILER_STATUS_SUCCESS) {
      ctx_ = {};
      return;
    }
    // NULL buffer id => synchronous sampling path.
    rocprofiler_configure_device_counting_service(
        ctx_, {.handle = 0}, agent,
        [](rocprofiler_context_id_t context_id, rocprofiler_agent_id_t,
           rocprofiler_device_counting_agent_cb_t set_config, void* user_data) {
          if (user_data) {
            static_cast<CounterSampler*>(user_data)->SetProfile(context_id, set_config);
          }
        },
        this);
  }

  rocprofiler_agent_id_t agent() const { return agent_; }

  // Name -> counter id for every counter this agent supports (cached).
  const std::unordered_map<std::string, rocprofiler_counter_id_t>& SupportedCounters() {
    if (!supported_loaded_) {
      supported_ = QuerySupportedCounters(agent_);
      supported_loaded_ = true;
    }
    return supported_;
  }

  // Sample counter names, packing them into HW configs. Sums counter_value
  // across all dimension-instance records for the same counter. `out_values`
  // maps counter name -> summed value; `elapsed_seconds` receives total wall
  // time in the sampling windows (denominator for per-second transforms). Only
  // counters supported on this agent are sampled.
  rocprofiler_status_t Sample(const std::vector<std::string>& counters,
                              std::map<std::string, double>* out_values, double* elapsed_seconds) {
    out_values->clear();
    *elapsed_seconds = 0.0;

    std::vector<std::string> sorted = counters;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    if (sorted.empty()) return ROCPROFILER_STATUS_SUCCESS;

    auto cached = profile_sets_.find(sorted);
    if (cached == profile_sets_.end()) {
      cached = profile_sets_.emplace(sorted, BuildProfileSet(sorted)).first;
    }

    rocprofiler_status_t last_status = ROCPROFILER_STATUS_SUCCESS;
    const auto start = std::chrono::steady_clock::now();
    for (const auto& profile : cached->second) {
      std::vector<rocprofiler_counter_record_t> records(profile.expected_size);
      profile_ = profile.config;
      rocprofiler_start_context(ctx_);
      size_t out_size = records.size();
      usleep(kSampleWindowUs);
      auto status = rocprofiler_sample_device_counting_service(
          ctx_, {}, ROCPROFILER_COUNTER_FLAG_NONE, records.data(), &out_size);
      rocprofiler_stop_context(ctx_);
      if (status != ROCPROFILER_STATUS_SUCCESS) {
        last_status = status;
        continue;
      }
      records.resize(out_size);
      for (const auto& record : records) {
        (*out_values)[DecodeName(record)] += record.counter_value;
      }
    }
    *elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return last_status;
  }

  static std::vector<rocprofiler_agent_v0_t> AvailableAgents() {
    std::vector<rocprofiler_agent_v0_t> agents;
    auto cb = [](rocprofiler_agent_version_t ver, const void** arr, size_t num, void* udata) {
      if (ver != ROCPROFILER_AGENT_INFO_VERSION_0) return ROCPROFILER_STATUS_ERROR;
      auto* out = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
      for (size_t i = 0; i < num; ++i) {
        const auto* a = static_cast<const rocprofiler_agent_v0_t*>(arr[i]);
        if (a->type == ROCPROFILER_AGENT_TYPE_GPU) out->emplace_back(*a);
      }
      return ROCPROFILER_STATUS_SUCCESS;
    };
    rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, cb,
                                       sizeof(rocprofiler_agent_t),
                                       const_cast<void*>(static_cast<const void*>(&agents)));
    return agents;
  }

 private:
  struct Profile {
    rocprofiler_counter_config_id_t config;
    size_t expected_size;
  };

  void SetProfile(rocprofiler_context_id_t ctx, rocprofiler_device_counting_agent_cb_t cb) const {
    if (profile_.handle != 0) cb(ctx, profile_);
  }

  const std::string& DecodeName(const rocprofiler_counter_record_t& rec) {
    rocprofiler_counter_id_t id = {.handle = 0};
    rocprofiler_query_record_counter_id(rec.id, &id);
    auto it = id_to_name_.find(id.handle);
    if (it != id_to_name_.end()) return it->second;
    rocprofiler_counter_info_v0_t info;
    if (rocprofiler_query_counter_info(id, ROCPROFILER_COUNTER_INFO_VERSION_0, &info) ==
        ROCPROFILER_STATUS_SUCCESS) {
      return id_to_name_.emplace(id.handle, info.name).first->second;
    }
    static const std::string kUnknown = "UNKNOWN_COUNTER";
    return kUnknown;
  }

  // Greedy bin-packing: add counters to a config until it can't be created,
  // then start a fresh config for the overflow. Returns configs covering all
  // packable counters.
  std::vector<Profile> BuildProfileSet(const std::vector<std::string>& counters) {
    std::vector<Profile> profiles;
    const auto& supported = SupportedCounters();
    std::vector<std::string> remaining;
    for (const auto& c : counters) {
      if (supported.count(c)) remaining.push_back(c);
    }

    while (!remaining.empty()) {
      std::vector<std::string> current;
      std::vector<std::string> overflow;
      rocprofiler_counter_config_id_t last_valid = {};
      size_t last_size = 0;

      for (const auto& name : remaining) {
        current.push_back(name);
        std::vector<rocprofiler_counter_id_t> ids;
        size_t size = 0;
        for (const auto& n : current) {
          auto it = supported.find(n);
          ids.push_back(it->second);
          size += CounterSize(it->second);
        }
        rocprofiler_counter_config_id_t config = {};
        auto status = rocprofiler_create_counter_config(agent_, ids.data(), ids.size(), &config);
        if (status == ROCPROFILER_STATUS_SUCCESS) {
          last_valid = config;
          last_size = size;
        } else {
          // Does not fit (or otherwise cannot be created together): defer it.
          current.pop_back();
          overflow.push_back(name);
        }
      }

      if (!current.empty() && last_valid.handle != 0) {
        profiles.push_back({last_valid, last_size});
      } else if (!overflow.empty()) {
        // No progress made; drop the head to avoid an infinite loop.
        overflow.erase(overflow.begin());
      }
      remaining = overflow;
    }
    return profiles;
  }

  static size_t CounterSize(rocprofiler_counter_id_t counter) {
    rocprofiler_counter_info_v1_t info;
    if (rocprofiler_query_counter_info(counter, ROCPROFILER_COUNTER_INFO_VERSION_1, &info) !=
        ROCPROFILER_STATUS_SUCCESS) {
      return 1;
    }
    // The instance-record count field was renamed from `instance_ids_count`
    // (rocprofiler-sdk 1.0.x) to `dimensions_instances_count` (>=1.1.0).
    // ROCPROFILER_VERSION encodes 10000*major + 100*minor + patch, so 1.1.0 ==
    // 10100.
#if defined(ROCPROFILER_VERSION) && ROCPROFILER_VERSION >= 10100
    return info.dimensions_instances_count;
#else
    return info.instance_ids_count;
#endif
  }

  static std::unordered_map<std::string, rocprofiler_counter_id_t> QuerySupportedCounters(
      rocprofiler_agent_id_t agent) {
    std::unordered_map<std::string, rocprofiler_counter_id_t> out;
    std::vector<rocprofiler_counter_id_t> ids;
    rocprofiler_iterate_agent_supported_counters(
        agent,
        [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* counters, size_t num, void* udata) {
          auto* v = static_cast<std::vector<rocprofiler_counter_id_t>*>(udata);
          for (size_t i = 0; i < num; ++i) v->push_back(counters[i]);
          return ROCPROFILER_STATUS_SUCCESS;
        },
        &ids);
    for (auto id : ids) {
      rocprofiler_counter_info_v0_t info;
      if (rocprofiler_query_counter_info(id, ROCPROFILER_COUNTER_INFO_VERSION_0, &info) ==
          ROCPROFILER_STATUS_SUCCESS) {
        out.emplace(info.name, id);
      }
    }
    return out;
  }

  rocprofiler_agent_id_t agent_ = {};
  rocprofiler_context_id_t ctx_ = {};
  rocprofiler_counter_config_id_t profile_ = {.handle = 0};

  bool supported_loaded_ = false;
  std::unordered_map<std::string, rocprofiler_counter_id_t> supported_;
  std::map<std::vector<std::string>, std::vector<Profile>> profile_sets_;
  std::unordered_map<uint64_t, std::string> id_to_name_;
};

// Process-global rocprofiler state, populated by tool_init.
std::vector<rocprofiler_agent_v0_t> g_agents;
std::vector<std::unique_ptr<CounterSampler>> g_samplers;
std::once_flag g_tool_init_once;

}  // namespace

// rocprofiler tool registration. rocprofiler-sdk finds these symbols in the
// global namespace at hsa_init(); they must have default visibility.
extern "C" __attribute__((visibility("default"))) int tool_init(rocprofiler_client_finalize_t,
                                                                void*) {
  std::call_once(g_tool_init_once, []() {
    g_agents = CounterSampler::AvailableAgents();
    for (const auto& agent : g_agents) {
      g_samplers.push_back(std::make_unique<CounterSampler>(agent.id));
    }
  });
  return 0;
}

extern "C" __attribute__((visibility("default"))) void tool_fini(void*) {}

extern "C" __attribute__((visibility("default"))) rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /*version*/, const char* /*runtime_version*/, uint32_t /*priority*/,
                      rocprofiler_client_id_t* id) {
  id->name = "gpumetrics_rocprofiler";
  static auto cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
  return &cfg;
}

namespace {

// Metric table: dotted gpumetrics key -> rocprofiler counter + transform. Only
// entries whose counter is supported by an agent are advertised for it.
enum class Transform {
  kRaw,        // value as reported (summed across dimensions)
  kDiv100,     // value / 100 (GPU_UTIL is reported as hundredths)
  kPerSecond,  // value / sample-window seconds (rates/bandwidth)
};

struct MetricDef {
  const char* key;
  const char* counter;
  const char* unit;
  Transform transform;
  const char* description;
};

// clang-format off
constexpr MetricDef kMetrics[] = {
    {"prof.active_cycles",  "GRBM_GUI_ACTIVE",   "cycles",   Transform::kRaw,       "Cycles the graphics pipeline was active"},
    {"prof.elapsed_cycles", "GRBM_COUNT",        "cycles",   Transform::kRaw,       "Free-running GRBM clock cycles elapsed"},
    {"prof.active_waves",   "SQ_WAVES",          "",         Transform::kRaw,       "Waves created and dispatched to the SQ"},
    {"prof.occupancy_pct",  "OccupancyPercent",  "%",        Transform::kRaw,       "Wavefront occupancy percentage"},
    {"prof.gpu_util_pct",   "GPU_UTIL",          "%",        Transform::kDiv100,    "Overall GPU utilization percentage"},
    {"prof.sm_active_pct",  "VALUBusy",          "%",        Transform::kRaw,       "Percentage of cycles the VALU was busy"},
    {"prof.valu_util_pct",  "ValuPipeIssueUtil", "%",        Transform::kRaw,       "VALU pipe issue utilization percentage"},
    {"prof.mem_read_bytes", "FETCH_SIZE",        "bytes",    Transform::kRaw,       "Bytes fetched from memory during the window"},
    {"prof.mem_write_bytes","WRITE_SIZE",        "bytes",    Transform::kRaw,       "Bytes written to memory during the window"},
    {"prof.mem_read_bps",   "FETCH_SIZE",        "bytes/s",  Transform::kPerSecond, "Memory read bandwidth (bytes per second)"},
    {"prof.mem_write_bps",  "WRITE_SIZE",        "bytes/s",  Transform::kPerSecond, "Memory write bandwidth (bytes per second)"},
    {"prof.flops_16",       "TOTAL_16_OPS",      "ops",      Transform::kRaw,       "Total 16-bit ops during the window"},
    {"prof.flops_32",       "TOTAL_32_OPS",      "ops",      Transform::kRaw,       "Total 32-bit ops during the window"},
    {"prof.flops_64",       "TOTAL_64_OPS",      "ops",      Transform::kRaw,       "Total 64-bit ops during the window"},
    // Raw counters exposed directly for advanced use.
    {"prof.raw.grbm_gui_active", "GRBM_GUI_ACTIVE", "cycles", Transform::kRaw,      "Raw GRBM_GUI_ACTIVE counter"},
    {"prof.raw.grbm_count",      "GRBM_COUNT",      "cycles", Transform::kRaw,      "Raw GRBM_COUNT counter"},
    {"prof.raw.sq_waves",        "SQ_WAVES",        "",       Transform::kRaw,      "Raw SQ_WAVES counter"},
    {"prof.raw.sq_busy_cycles",  "SQ_BUSY_CYCLES",  "cycles", Transform::kRaw,      "Raw SQ_BUSY_CYCLES counter"},
    {"prof.raw.sq_insts_valu",   "SQ_INSTS_VALU",   "",       Transform::kRaw,      "Raw SQ_INSTS_VALU counter"},
};
// clang-format on

const MetricDef* FindMetric(const char* key) {
  for (const auto& m : kMetrics) {
    if (std::strcmp(m.key, key) == 0) return &m;
  }
  return nullptr;
}

double ApplyTransform(const MetricDef& m, double raw, double elapsed_seconds) {
  switch (m.transform) {
    case Transform::kDiv100:
      return raw / 100.0;
    case Transform::kPerSecond:
      return elapsed_seconds > 0.0 ? raw / elapsed_seconds : 0.0;
    case Transform::kRaw:
    default:
      return raw;
  }
}

// location_id packs the PCIe bus/device/function; combined with the domain.
uint8_t BdfFunction(const rocprofiler_agent_v0_t& a) {
  return static_cast<uint8_t>(a.location_id & 0x7);
}
uint64_t PackBdf(const rocprofiler_agent_v0_t& a) {
  const uint8_t bus = (a.location_id >> 8) & 0xff;
  const uint8_t dev = (a.location_id >> 3) & 0x1f;
  return gpum_bdf_pack(a.domain, bus, dev, BdfFunction(a));
}

// Plugin context. Built during init(); afterward the per-read sampling mutates
// sampler caches under a lock.
struct PluginContext {
  bool ready = false;
  std::mutex sample_mutex;
  std::vector<gpum_device_desc> devices;
  std::vector<gpum_metric_desc> metrics;
  std::vector<std::vector<bool>> metric_supported;  // [agent][metric_index]
};

void CopyString(char* dst, const char* src, size_t cap) {
  if (!src) src = "";
  std::strncpy(dst, src, cap - 1);
  dst[cap - 1] = '\0';
}

}  // namespace

extern "C" {

struct gpum_plugin_ctx {
  PluginContext impl;
};

}  // extern "C"

namespace {

gpum_status PluginInit(gpum_plugin_ctx** out_ctx) {
  if (!out_ctx) return GPUM_ERR_INVALID_ARG;

  // HSA_TOOLS_LIB is a rocprofiler-v1 leftover that confuses the SDK; clear it.
  unsetenv("HSA_TOOLS_LIB");

  // hsa_init() makes rocprofiler-sdk scan the global namespace for
  // rocprofiler_configure (this .so must be dlopened RTLD_GLOBAL) and run
  // tool_init(), which populates g_agents / g_samplers.
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    return GPUM_ERR_BACKEND;
  }

  int rocp_status = 0;
  if (rocprofiler_is_initialized(&rocp_status) != ROCPROFILER_STATUS_SUCCESS || rocp_status != 1) {
    return GPUM_ERR_NOT_INITIALIZED;
  }

  if (g_samplers.empty() || g_agents.empty()) {
    return GPUM_ERR_NO_DATA;
  }

  auto ctx = std::make_unique<gpum_plugin_ctx>();
  PluginContext& s = ctx->impl;

  // Build device descriptors from agents. Report raw identity; the core does
  // grouping. On CPX each agent is one partition with its own PCIe function
  // (partition index = that function, 0 = whole-GPU handle). The agent has no
  // oam_id, so grouping relies on the masked BDF the amdsmi plugin shares.
  for (uint32_t i = 0; i < g_agents.size(); ++i) {
    const auto& a = g_agents[i];
    gpum_device_desc desc = {};
    const uint8_t func = BdfFunction(a);
    desc.identity.bdf = PackBdf(a);
    desc.identity.oam_id = GPUM_ID_UNKNOWN;
    desc.identity.kfd_node_id = a.node_id;
    desc.identity.drm_render_minor = a.drm_render_minor;
    std::memcpy(desc.identity.uuid, a.uuid.bytes, sizeof(desc.identity.uuid));
    desc.identity.partition_index = (func == 0) ? -1 : static_cast<int32_t>(func);
    desc.identity.plugin_local_index = i;
    desc.identity.socket_id = GPUM_SOCKET_UNKNOWN;  // amdsmi is socket authority
    const char* name = a.product_name && a.product_name[0] ? a.product_name
                       : (a.model_name && a.model_name[0]) ? a.model_name
                                                           : a.name;
    CopyString(desc.name, name, sizeof(desc.name));
    s.devices.push_back(desc);
  }

  // Per agent, which metrics are supported; advertise a metric if supported on
  // at least one agent.
  s.metric_supported.assign(g_agents.size(), std::vector<bool>(std::size(kMetrics), false));
  std::vector<bool> advertise(std::size(kMetrics), false);
  for (uint32_t ai = 0; ai < g_samplers.size(); ++ai) {
    const auto& supported = g_samplers[ai]->SupportedCounters();
    for (size_t mi = 0; mi < std::size(kMetrics); ++mi) {
      if (supported.count(kMetrics[mi].counter)) {
        s.metric_supported[ai][mi] = true;
        advertise[mi] = true;
      }
    }
  }
  for (size_t mi = 0; mi < std::size(kMetrics); ++mi) {
    if (!advertise[mi]) continue;
    gpum_metric_desc d = {};
    CopyString(d.key, kMetrics[mi].key, sizeof(d.key));
    CopyString(d.unit, kMetrics[mi].unit, sizeof(d.unit));
    CopyString(d.description, kMetrics[mi].description, sizeof(d.description));
    d.type = GPUM_TYPE_F64;
    d.scope = GPUM_SCOPE_GPU;
    s.metrics.push_back(d);
  }

  s.ready = true;
  *out_ctx = ctx.release();
  return GPUM_OK;
}

void PluginShutdown(gpum_plugin_ctx* ctx) {
  // rocprofiler/HSA are process-global and cannot be cleanly re-initialized
  // in-process, so we do NOT call hsa_shut_down(); only release our context.
  delete ctx;
}

gpum_status PluginEnumerate(gpum_plugin_ctx* ctx, const gpum_device_desc** out_devices,
                            uint32_t* out_count) {
  if (!ctx || !out_devices || !out_count) return GPUM_ERR_INVALID_ARG;
  *out_devices = ctx->impl.devices.data();
  *out_count = static_cast<uint32_t>(ctx->impl.devices.size());
  return GPUM_OK;
}

gpum_status PluginListMetrics(gpum_plugin_ctx* ctx, const gpum_metric_desc** out_metrics,
                              uint32_t* out_count) {
  if (!ctx || !out_metrics || !out_count) return GPUM_ERR_INVALID_ARG;
  *out_metrics = ctx->impl.metrics.data();
  *out_count = static_cast<uint32_t>(ctx->impl.metrics.size());
  return GPUM_OK;
}

gpum_status SampleStatusToGpum(rocprofiler_status_t st) {
  switch (st) {
    case ROCPROFILER_STATUS_SUCCESS:
      return GPUM_OK;
    case ROCPROFILER_STATUS_ERROR:  // HSA not loaded yet / transient
      return GPUM_ERR_NO_DATA;
    default:
      return GPUM_ERR_BACKEND;
  }
}

gpum_status PluginRead(gpum_plugin_ctx* ctx, const gpum_read_req* reqs, uint32_t n,
                       gpum_sample* out) {
  if (!ctx || (!reqs && n) || (!out && n)) return GPUM_ERR_INVALID_ARG;
  PluginContext& s = ctx->impl;
  if (!s.ready) return GPUM_ERR_NOT_INITIALIZED;

  const uint64_t ts = NowNs();
  for (uint32_t i = 0; i < n; ++i) {
    out[i] = gpum_sample{};
    out[i].type = GPUM_TYPE_F64;
    out[i].timestamp_ns = ts;
    out[i].status = GPUM_ERR_INTERNAL;
  }

  // Group request indices by agent.
  std::map<uint32_t, std::vector<uint32_t>> by_agent;
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t ai = reqs[i].plugin_local_index;
    if (ai >= g_samplers.size()) {
      out[i].status = GPUM_ERR_NOT_FOUND;
      continue;
    }
    const MetricDef* m = FindMetric(reqs[i].key ? reqs[i].key : "");
    if (!m) {
      out[i].status = GPUM_ERR_NOT_FOUND;
      continue;
    }
    by_agent[ai].push_back(i);
  }

  std::lock_guard<std::mutex> lock(s.sample_mutex);
  for (auto& [ai, indices] : by_agent) {
    // Collect the distinct counters this agent can serve for the batch.
    std::vector<std::string> counters;
    for (uint32_t i : indices) {
      const MetricDef* m = FindMetric(reqs[i].key);
      size_t mi = static_cast<size_t>(m - kMetrics);
      if (!s.metric_supported[ai][mi]) {
        out[i].status = GPUM_ERR_UNSUPPORTED;
        continue;
      }
      counters.push_back(m->counter);
    }

    std::map<std::string, double> values;
    double elapsed = 0.0;
    rocprofiler_status_t st = ROCPROFILER_STATUS_SUCCESS;
    if (!counters.empty()) {
      // Retry a few times if HSA is momentarily not ready.
      for (int attempt = 0; attempt < 3; ++attempt) {
        st = g_samplers[ai]->Sample(counters, &values, &elapsed);
        if (st != ROCPROFILER_STATUS_ERROR) break;
        usleep(2000);
      }
    }

    for (uint32_t i : indices) {
      if (out[i].status == GPUM_ERR_UNSUPPORTED) continue;
      const MetricDef* m = FindMetric(reqs[i].key);
      auto it = values.find(m->counter);
      if (it == values.end()) {
        // Supported but produced no record this pass.
        out[i].status = st == ROCPROFILER_STATUS_SUCCESS ? GPUM_ERR_NO_DATA : SampleStatusToGpum(st);
        continue;
      }
      out[i].status = GPUM_OK;
      out[i].type = GPUM_TYPE_F64;
      out[i].value.type = GPUM_TYPE_F64;
      out[i].value.f64 = ApplyTransform(*m, it->second, elapsed);
    }
  }
  return GPUM_OK;
}

const gpum_plugin_v1 g_vtable = {
    /*abi_version=*/GPUM_PLUGIN_ABI_V1,
    /*name=*/"rocprofiler",
    /*init=*/PluginInit,
    /*shutdown=*/PluginShutdown,
    /*enumerate=*/PluginEnumerate,
    /*list_metrics=*/PluginListMetrics,
    /*read=*/PluginRead,
};

}  // namespace

extern "C" __attribute__((visibility("default"))) const gpum_plugin_v1* gpum_plugin_entry_v1(void) {
  return &g_vtable;
}
