// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// A hardware-free mock plugin for testing the Collector end-to-end (dlopen,
// correlation, routing, batched read). Fabricates two GPUs and metrics whose
// values are a deterministic function of device + key, so tests assert exact
// routing.
//
// Two .so's are built from this file via compile defines, exercising the
// two-plugin correlation path in one process:
//   MOCK_NAME        -> vtable name (e.g. "mockA")
//   MOCK_LOCAL_BASE  -> plugin-local index base (forces distinct indices)

#include <cstring>
#include <string>
#include <vector>

#include "gpumetrics/plugin_abi.h"

#ifndef MOCK_NAME
#define MOCK_NAME "mock"
#endif
#ifndef MOCK_LOCAL_BASE
#define MOCK_LOCAL_BASE 0
#endif

namespace {

struct MockCtx {
  std::vector<gpum_device_desc> devices;
  std::vector<gpum_metric_desc> metrics;
};

void fill_str(char* dst, size_t n, const std::string& s) {
  std::strncpy(dst, s.c_str(), n - 1);
  dst[n - 1] = '\0';
}

gpum_status mock_init(gpum_plugin_ctx** out) {
  auto* c = new MockCtx();
  const uint32_t base = MOCK_LOCAL_BASE;

  for (uint32_t i = 0; i < 2; ++i) {
    gpum_device_desc d{};
    d.identity.bdf = 0x6300 + (uint64_t(i) << 8);
    d.identity.oam_id = GPUM_ID_UNKNOWN;  // correlate on BDF
    d.identity.kfd_node_id = 100 + i;
    d.identity.socket_id = 0;
    d.identity.partition_index = -1;
    d.identity.plugin_local_index = base + i;
    fill_str(d.name, GPUM_STRING_MAX, std::string(MOCK_NAME) + "-gpu" + std::to_string(i));
    c->devices.push_back(d);
  }

  auto add_metric = [&](const char* key, const char* unit, gpum_value_type t) {
    gpum_metric_desc m{};
    fill_str(m.key, GPUM_STRING_MAX, key);
    fill_str(m.unit, sizeof(m.unit), unit);
    fill_str(m.description, GPUM_STRING_MAX, key);
    m.type = t;
    m.scope = GPUM_SCOPE_GPU;
    c->metrics.push_back(m);
  };
  add_metric("mock.temp", "C", GPUM_TYPE_F64);
  add_metric("mock.shared", "", GPUM_TYPE_U64);  // offered by every instance

  *out = reinterpret_cast<gpum_plugin_ctx*>(c);
  return GPUM_OK;
}

void mock_shutdown(gpum_plugin_ctx* ctx) { delete reinterpret_cast<MockCtx*>(ctx); }

gpum_status mock_enumerate(gpum_plugin_ctx* ctx, const gpum_device_desc** out, uint32_t* n) {
  auto* c = reinterpret_cast<MockCtx*>(ctx);
  *out = c->devices.data();
  *n = static_cast<uint32_t>(c->devices.size());
  return GPUM_OK;
}

gpum_status mock_list_metrics(gpum_plugin_ctx* ctx, const gpum_metric_desc** out, uint32_t* n) {
  auto* c = reinterpret_cast<MockCtx*>(ctx);
  *out = c->metrics.data();
  *n = static_cast<uint32_t>(c->metrics.size());
  return GPUM_OK;
}

gpum_status mock_read(gpum_plugin_ctx* /*ctx*/, const gpum_read_req* reqs, uint32_t n,
                      gpum_sample* out) {
  for (uint32_t i = 0; i < n; ++i) {
    out[i] = gpum_sample{};
    out[i].timestamp_ns = 1;
    std::string key = reqs[i].key ? reqs[i].key : "";
    if (key == "mock.temp") {
      out[i].status = GPUM_OK;
      out[i].type = GPUM_TYPE_F64;
      out[i].value.type = GPUM_TYPE_F64;
      out[i].value.f64 = 40.0 + reqs[i].plugin_local_index;  // encodes local idx
    } else if (key == "mock.shared") {
      out[i].status = GPUM_OK;
      out[i].type = GPUM_TYPE_U64;
      out[i].value.type = GPUM_TYPE_U64;
      // Last char of MOCK_NAME ('A'/'B') encodes which instance served it.
      out[i].value.u64 = 1000u + uint32_t(std::string(MOCK_NAME).back());
    } else {
      out[i].status = GPUM_ERR_NOT_FOUND;
    }
  }
  return GPUM_OK;
}

gpum_plugin_v1 g_vt = {
    GPUM_PLUGIN_ABI_V1, MOCK_NAME, mock_init, mock_shutdown,
    mock_enumerate,     mock_list_metrics, mock_read,
};

}  // namespace

extern "C" const gpum_plugin_v1* gpum_plugin_entry_v1(void) { return &g_vt; }
