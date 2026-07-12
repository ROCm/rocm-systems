// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Flat C API (capi.h): a thin adapter over the C++ Collector.

#include "gpumetrics/capi.h"

#include <cstring>
#include <string>
#include <vector>

#include "gpumetrics/gpumetrics.h"

using namespace gpumetrics;

struct gpum_collector {
  std::unique_ptr<Collector> impl;
};

namespace {

void copy_str(char* dst, size_t cap, const std::string& s) {
  if (!dst || cap == 0) return;
  std::strncpy(dst, s.c_str(), cap - 1);
  dst[cap - 1] = '\0';
}

void to_c_sample(const Sample& s, gpum_sample* out) {
  out->status = s.status;
  out->type = s.type;
  out->timestamp_ns = s.timestamp_ns;
  out->value = gpum_value{};
  out->value.type = s.type;
  if (s.status != GPUM_OK) return;
  switch (s.type) {
    case GPUM_TYPE_U64:
      out->value.u64 = std::get<uint64_t>(s.value);
      break;
    case GPUM_TYPE_I64:
      out->value.i64 = std::get<int64_t>(s.value);
      break;
    case GPUM_TYPE_F64:
      out->value.f64 = std::get<double>(s.value);
      break;
    case GPUM_TYPE_STRING:
      copy_str(out->value.str, GPUM_STRING_MAX, std::get<std::string>(s.value));
      break;
  }
}

template <typename T>
std::vector<std::string> to_vec(const T* const* arr, uint32_t n) {
  std::vector<std::string> v;
  for (uint32_t i = 0; i < n && arr; ++i)
    if (arr[i]) v.emplace_back(arr[i]);
  return v;
}

}  // namespace

extern "C" {

gpum_collector* gpum_collector_create(const gpum_collector_options* opts, gpum_status* status) {
  CollectorOptions o;
  if (opts) {
    o.plugin_paths = to_vec(opts->plugin_paths, opts->plugin_paths_count);
    o.plugins = to_vec(opts->plugins, opts->plugins_count);
    auto pp = to_vec(opts->provider_priority, opts->provider_priority_count);
    if (!pp.empty()) o.provider_priority = pp;
    if (opts->read_timeout_ms) o.read_timeout_ms = opts->read_timeout_ms;
  }
  gpum_status st;
  auto impl = Collector::Create(o, &st);
  if (status) *status = st;
  if (!impl) return nullptr;
  auto* c = new gpum_collector();
  c->impl = std::move(impl);
  return c;
}

void gpum_collector_destroy(gpum_collector* c) { delete c; }

uint32_t gpum_collector_gpu_count(const gpum_collector* c) {
  return c ? static_cast<uint32_t>(c->impl->devices().size()) : 0;
}

uint32_t gpum_collector_socket_count(const gpum_collector* c) {
  return c ? static_cast<uint32_t>(c->impl->sockets().size()) : 0;
}

gpum_status gpum_collector_gpu_entity(const gpum_collector* c, uint32_t gpu, gpum_entity_id* out) {
  if (!c || !out) return GPUM_ERR_INVALID_ARG;
  const auto& devs = c->impl->devices();
  if (gpu >= devs.size()) return GPUM_ERR_NOT_FOUND;
  *out = devs[gpu].id;
  return GPUM_OK;
}

gpum_status gpum_collector_gpu_info(const gpum_collector* c, uint32_t gpu, char* name_buf,
                                    size_t name_cap, gpum_device_identity* identity,
                                    uint32_t* partition_count) {
  if (!c) return GPUM_ERR_INVALID_ARG;
  const auto& devs = c->impl->devices();
  if (gpu >= devs.size()) return GPUM_ERR_NOT_FOUND;
  const auto& d = devs[gpu];
  copy_str(name_buf, name_cap, d.name);
  if (identity) *identity = d.identity;
  if (partition_count) *partition_count = static_cast<uint32_t>(d.partitions.size());
  return GPUM_OK;
}

gpum_status gpum_collector_gpu_partitions(const gpum_collector* c, uint32_t gpu,
                                          int32_t* out_partitions, uint32_t cap,
                                          uint32_t* out_count) {
  if (!c) return GPUM_ERR_INVALID_ARG;
  const auto& devs = c->impl->devices();
  if (gpu >= devs.size()) return GPUM_ERR_NOT_FOUND;
  const auto& p = devs[gpu].partitions;
  if (out_count) *out_count = static_cast<uint32_t>(p.size());
  for (uint32_t i = 0; i < p.size() && i < cap; ++i) out_partitions[i] = p[i];
  return GPUM_OK;
}

uint32_t gpum_collector_metric_count(const gpum_collector* c) {
  return c ? static_cast<uint32_t>(c->impl->metrics().size()) : 0;
}

gpum_status gpum_collector_metric_at(const gpum_collector* c, uint32_t i, char* key_buf,
                                     size_t key_cap, char* unit_buf, size_t unit_cap,
                                     char* provider_buf, size_t provider_cap, gpum_value_type* type,
                                     uint32_t* scope) {
  if (!c) return GPUM_ERR_INVALID_ARG;
  const auto& m = c->impl->metrics();
  if (i >= m.size()) return GPUM_ERR_NOT_FOUND;
  copy_str(key_buf, key_cap, m[i].key);
  copy_str(unit_buf, unit_cap, m[i].unit);
  copy_str(provider_buf, provider_cap, m[i].provider);
  if (type) *type = m[i].type;
  if (scope) *scope = m[i].scope;
  return GPUM_OK;
}

gpum_status gpum_collector_resolve(const gpum_collector* c, const char* selector,
                                   gpum_entity_id* out) {
  if (!c || !selector || !out) return GPUM_ERR_INVALID_ARG;
  auto e = c->impl->resolve(selector);
  if (!e) return GPUM_ERR_NOT_FOUND;
  *out = e->id;
  return GPUM_OK;
}

gpum_status gpum_collector_read(gpum_collector* c, const gpum_entity_id* entity, const char* key,
                                gpum_sample* out_sample) {
  if (!c || !entity || !key || !out_sample) return GPUM_ERR_INVALID_ARG;
  Sample s = c->impl->read(*entity, key);
  to_c_sample(s, out_sample);
  return GPUM_OK;
}

gpum_status gpum_collector_read_batch(gpum_collector* c, const gpum_entity_id* entity,
                                      const char* const* keys, uint32_t n,
                                      gpum_sample* out_samples) {
  if (!c || !entity || (!keys && n) || (!out_samples && n)) return GPUM_ERR_INVALID_ARG;
  std::vector<std::string> ks;
  ks.reserve(n);
  for (uint32_t i = 0; i < n; ++i) ks.emplace_back(keys[i] ? keys[i] : "");
  auto samples = c->impl->read(*entity, ks);
  for (uint32_t i = 0; i < n; ++i) to_c_sample(samples[i], &out_samples[i]);
  return GPUM_OK;
}

}  // extern "C"
