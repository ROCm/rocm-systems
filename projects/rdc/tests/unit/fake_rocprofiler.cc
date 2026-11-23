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

// A link-time fake of the rocprofiler-sdk and HSA functions the ROCp module calls (controls
// in fake_rocprofiler.h). Single-threaded, as the tests that use it.
#include "fake_rocprofiler.h"

#include <hsa/hsa.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

using rdc_fake_rocp::Call;

struct Counter {
  uint64_t id;
  const char* name;
};

// Every agent's counters: names the ROCp module maps fields to.
const Counter kCounters[] = {
    {1, "GRBM_COUNT"},
    {2, "SQ_WAVES"},
    {3, "GRBM_GUI_ACTIVE"},
    {4, "TOTAL_32_OPS"},
    {5, "RDC_OPS_32_PER_SIMDCYCLE"},
};

const rocprofiler_counter_record_dimension_info_t kDimension = {"DIMENSION_INSTANCE", 1, 0};
const rocprofiler_counter_record_dimension_info_t* kDimensions[] = {&kDimension};

constexpr uint64_t kAgentBase = 100;  // agent i has the handle kAgentBase + i

struct Context {
  int agent = -1;
  rocprofiler_device_counting_service_cb_t service = nullptr;
  void* user_data = nullptr;
  uint64_t config = 0;
  bool active = false;
};

struct State {
  rdc_fake_rocp::Config config;
  std::vector<rocprofiler_agent_v0_t> agents;
  std::map<uint64_t, Context> contexts;
  std::map<uint64_t, std::vector<uint64_t>> configs;  // config handle -> counter ids
  uint64_t next_context = 1;
  uint64_t next_config = 1;
  std::map<Call, uint32_t> calls;
  std::map<std::pair<Call, int>, uint32_t> agent_calls;
  uint32_t hsa_refs = 0;
  uint32_t hsa_inits = 0;
  uint32_t hsa_shut_downs = 0;
  bool registered = false;
};

State& state() {
  static State s;
  return s;
}

const Counter* find_counter(uint64_t id) {
  for (const Counter& c : kCounters) {
    if (c.id == id) return &c;
  }
  return nullptr;
}

// Counts the call and says whether a configured failure applies to it.
bool fails(Call call, int agent) {
  State& s = state();
  const uint32_t all = ++s.calls[call];
  const uint32_t own = ++s.agent_calls[{call, agent}];
  for (const rdc_fake_rocp::Failure& f : s.config.failures) {
    if (f.call != call || (f.agent != -1 && f.agent != agent)) continue;
    const uint32_t n = f.agent == -1 ? all : own;
    if (n >= f.from && (f.count == 0 || n < f.from + f.count)) return true;
  }
  return false;
}

Context* find_context(rocprofiler_context_id_t ctx) {
  auto it = state().contexts.find(ctx.handle);
  return it == state().contexts.end() ? nullptr : &it->second;
}

rocprofiler_status_t set_config(rocprofiler_context_id_t ctx,
                                rocprofiler_counter_config_id_t config) {
  Context* c = find_context(ctx);
  if (c == nullptr) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
  c->config = config.handle;
  return ROCPROFILER_STATUS_SUCCESS;
}

void finalize_client(rocprofiler_client_id_t) {}

}  // namespace

namespace rdc_fake_rocp {

void configure(const Config& config) {
  State& s = state();
  s = State{};
  s.config = config;
  for (uint32_t i = 0; i < config.agents; ++i) {
    rocprofiler_agent_v0_t a = {};
    a.size = sizeof(a);
    a.id.handle = kAgentBase + i;
    a.type = ROCPROFILER_AGENT_TYPE_GPU;
    a.gpu_id = 0x1000 + i;  // the fake AMD SMI's KFD id of GPU i
    a.location_id = i;
    a.name = "gfx942";
    a.simd_per_cu = config.simd_per_cu;
    a.node_id = i + 1;
    s.agents.push_back(a);
  }
}

double value_of(uint32_t agent, const std::string& counter) {
  for (const Counter& c : kCounters) {
    if (counter == c.name) return 1000.0 * (agent + 1) + static_cast<double>(c.id);
  }
  return 0.0;
}

uint32_t calls(Call call) { return state().calls[call]; }
uint32_t hsa_inits() { return state().hsa_inits; }
uint32_t hsa_shut_downs() { return state().hsa_shut_downs; }

bool any_context_active() {
  for (const auto& [handle, c] : state().contexts) {
    if (c.active) return true;
  }
  return false;
}

}  // namespace rdc_fake_rocp

// rocprofiler-sdk

rocprofiler_status_t rocprofiler_is_initialized(int* status) {
  *status = state().registered ? 1 : 0;
  return ROCPROFILER_STATUS_SUCCESS;
}

const char* rocprofiler_get_status_string(rocprofiler_status_t) { return "fake rocprofiler error"; }

rocprofiler_status_t rocprofiler_query_available_agents(rocprofiler_agent_version_t version,
                                                        rocprofiler_query_available_agents_cb_t cb,
                                                        size_t, void* user_data) {
  if (fails(Call::QueryAgents, -1)) return ROCPROFILER_STATUS_ERROR;
  std::vector<const void*> agents;
  for (const rocprofiler_agent_v0_t& a : state().agents) agents.push_back(&a);
  return cb(version, agents.data(), agents.size(), user_data);
}

rocprofiler_status_t rocprofiler_create_context(rocprofiler_context_id_t* ctx) {
  State& s = state();
  ctx->handle = s.next_context++;
  s.contexts[ctx->handle] = Context{};
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_configure_device_counting_service(
    rocprofiler_context_id_t ctx, rocprofiler_buffer_id_t, rocprofiler_agent_id_t agent,
    rocprofiler_device_counting_service_cb_t cb, void* user_data) {
  Context* c = find_context(ctx);
  if (c == nullptr) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
  c->agent = static_cast<int>(agent.handle - kAgentBase);
  c->service = cb;
  c->user_data = user_data;
  return ROCPROFILER_STATUS_SUCCESS;
}

// Starting a context asks its service for the counter config, as rocprofiler-sdk does.
rocprofiler_status_t rocprofiler_start_context(rocprofiler_context_id_t ctx) {
  Context* c = find_context(ctx);
  if (c == nullptr) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
  if (fails(Call::Start, c->agent)) return ROCPROFILER_STATUS_ERROR;
  if (c->active) return ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT;
  c->active = true;
  if (c->service != nullptr) {
    c->service(ctx, {kAgentBase + static_cast<uint64_t>(c->agent)}, &set_config, c->user_data);
  }
  return ROCPROFILER_STATUS_SUCCESS;
}

// A stop that fails leaves the context counting.
rocprofiler_status_t rocprofiler_stop_context(rocprofiler_context_id_t ctx) {
  Context* c = find_context(ctx);
  if (c == nullptr) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
  if (fails(Call::Stop, c->agent)) return ROCPROFILER_STATUS_ERROR;
  c->active = false;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_context_is_active(rocprofiler_context_id_t ctx, int* status) {
  Context* c = find_context(ctx);
  if (c == nullptr) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
  *status = c->active ? 1 : 0;
  return ROCPROFILER_STATUS_SUCCESS;
}

// A sample that fails writes nothing and leaves rec_count as it was.
rocprofiler_status_t rocprofiler_sample_device_counting_service(rocprofiler_context_id_t ctx,
                                                                rocprofiler_user_data_t,
                                                                rocprofiler_counter_flag_t,
                                                                rocprofiler_counter_record_t* out,
                                                                size_t* rec_count) {
  Context* c = find_context(ctx);
  if (c == nullptr) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
  if (fails(Call::Sample, c->agent)) return ROCPROFILER_STATUS_ERROR;
  if (!c->active) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED;
  const auto config = state().configs.find(c->config);
  if (config == state().configs.end()) return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;
  size_t n = 0;
  for (uint64_t id : config->second) {
    if (n == *rec_count) break;
    out[n] = {};
    out[n].id = id << 32;
    out[n].counter_value = rdc_fake_rocp::value_of(c->agent, find_counter(id)->name);
    out[n].agent_id.handle = kAgentBase + static_cast<uint64_t>(c->agent);
    ++n;
  }
  *rec_count = n;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_iterate_agent_supported_counters(
    rocprofiler_agent_id_t agent, rocprofiler_available_counters_cb_t cb, void* user_data) {
  std::vector<rocprofiler_counter_id_t> ids;
  for (const Counter& c : kCounters) ids.push_back({c.id});
  return cb(agent, ids.data(), ids.size(), user_data);
}

// A version-1 query that fails has written part of its output: an instance count of 3 where
// the counter has 1, as a structure the caller left unset can hold anything.
rocprofiler_status_t rocprofiler_query_counter_info(rocprofiler_counter_id_t counter_id,
                                                    rocprofiler_counter_info_version_id_t version,
                                                    void* info) {
  const Counter* c = find_counter(counter_id.handle);
  if (c == nullptr) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
  if (version == ROCPROFILER_COUNTER_INFO_VERSION_0) {
    auto* v0 = static_cast<rocprofiler_counter_info_v0_t*>(info);
    *v0 = {};
    v0->id = counter_id;
    v0->name = c->name;
    return ROCPROFILER_STATUS_SUCCESS;
  }
  auto* v1 = static_cast<rocprofiler_counter_info_v1_t*>(info);
  if (fails(Call::CounterInfoV1, -1)) {
    v1->dimensions_instances_count = 3;
    return ROCPROFILER_STATUS_ERROR;
  }
  *v1 = {};
  v1->size = sizeof(*v1);
  v1->id = counter_id;
  v1->name = c->name;
  v1->dimensions_count = 1;
  v1->dimensions = kDimensions;
  v1->dimensions_instances_count = 1;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_create_counter_config(rocprofiler_agent_id_t,
                                                       rocprofiler_counter_id_t* counters,
                                                       size_t count,
                                                       rocprofiler_counter_config_id_t* config) {
  if (count > 4) return ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT;
  State& s = state();
  std::vector<uint64_t> ids;
  for (size_t i = 0; i < count; ++i) ids.push_back(counters[i].handle);
  config->handle = s.next_config++;
  s.configs[config->handle] = ids;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_query_record_counter_id(rocprofiler_counter_instance_id_t id,
                                                         rocprofiler_counter_id_t* counter_id) {
  counter_id->handle = id >> 32;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_query_record_dimension_position(rocprofiler_counter_instance_id_t,
                                                                 rocprofiler_counter_dimension_id_t,
                                                                 size_t* pos) {
  *pos = 0;
  return ROCPROFILER_STATUS_SUCCESS;
}

// HSA

// The first hsa_init() that succeeds registers the module as a rocprofiler-sdk tool, which
// makes its samplers (tool_init).
hsa_status_t hsa_init() {
  State& s = state();
  if (fails(Call::HsaInit, -1)) return HSA_STATUS_ERROR;
  ++s.hsa_refs;
  ++s.hsa_inits;
  if (!s.registered) {
    s.registered = true;
    rocprofiler_client_id_t client = {};
    rocprofiler_tool_configure_result_t* tool = rocprofiler_configure(10100, "fake", 0, &client);
    if (tool != nullptr && tool->initialize != nullptr) {
      tool->initialize(&finalize_client, tool->tool_data);
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_shut_down() {
  State& s = state();
  ++s.hsa_shut_downs;
  if (s.hsa_refs == 0) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  --s.hsa_refs;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_status_string(hsa_status_t, const char** status_string) {
  *status_string = "fake hsa error";
  return HSA_STATUS_SUCCESS;
}
