// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Public C++ API for the gpumetrics library, used by the CLI and embedders. A
// flat C API for Rust/FFI is layered on top (capi.h).

#ifndef GPUMETRICS_GPUMETRICS_H_
#define GPUMETRICS_GPUMETRICS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "gpumetrics/types.h"

namespace gpumetrics {

using Value = std::variant<uint64_t, int64_t, double, std::string>;

// A metric registry entry: the single source of truth for a metric's
// type/unit/scope.
struct MetricDescriptor {
  std::string key;
  gpum_value_type type = GPUM_TYPE_F64;
  std::string unit;
  uint32_t scope = 0;  // OR of gpum_scope_flags
  std::string description;
  std::string provider;  // plugin name that serves this key
};

// A single reading.
struct Sample {
  std::string key;
  gpum_status status = GPUM_OK;
  gpum_value_type type = GPUM_TYPE_F64;
  Value value{};
  uint64_t timestamp_ns = 0;

  bool ok() const { return status == GPUM_OK; }
  double as_double() const;   // numeric coercion (0 if string/!ok)
  std::string to_string() const;  // formatted value, no unit
};

// A correlated physical GPU with its partitions, grouped under a socket.
struct Device {
  gpum_entity_id id;              // kind==GPU
  std::string name;               // marketing/product name
  gpum_device_identity identity;  // merged identity (best keys across plugins)
  std::vector<int32_t> partitions;  // empty if unpartitioned
  std::vector<std::string> providers;  // plugins that serve this GPU
};

struct Socket {
  uint32_t index = 0;
  std::string name;
  std::vector<uint32_t> gpus;  // canonical GPU ordinals in this socket
};

// A parsed user selector, e.g. "gpu:0", "g0.1", "socket:1".
struct Entity {
  gpum_entity_id id{};
  std::string label;  // canonical rendering, e.g. "g0.1"
};

// Options controlling collection.
struct CollectorOptions {
  // Plugin .so search dirs, in priority order. If empty, defaults are used
  // ($GPUMETRICS_PLUGIN_PATH, cwd, install libdir).
  std::vector<std::string> plugin_paths;
  // Explicit plugin filenames; if empty, all discoverable plugins load.
  std::vector<std::string> plugins;
  // On a duplicate metric key, plugins earlier in this list win. Unlisted
  // plugins keep first-seen order after listed ones.
  std::vector<std::string> provider_priority = {"amdsmi", "rocprofiler"};
  // Per-read backend timeout.
  uint32_t read_timeout_ms = 5000;
};

// The top-level object. Loads plugins, correlates devices, routes reads.
class Collector {
 public:
  static std::unique_ptr<Collector> Create(const CollectorOptions& opts, gpum_status* status);
  virtual ~Collector() = default;

  // Discovered topology.
  virtual const std::vector<Socket>& sockets() const = 0;
  virtual const std::vector<Device>& devices() const = 0;

  // The metric registry (all keys any loaded plugin can serve).
  virtual const std::vector<MetricDescriptor>& metrics() const = 0;
  virtual std::optional<MetricDescriptor> describe(const std::string& key) const = 0;

  // Parse a user selector into an entity; nullopt if it doesn't resolve.
  virtual std::optional<Entity> resolve(const std::string& selector) const = 0;

  // All entities of a kind (e.g. all GPUs, or all partitions).
  virtual std::vector<Entity> entities(gpum_entity_kind kind) const = 0;

  // Read one metric for one entity.
  virtual Sample read(const gpum_entity_id& entity, const std::string& key) = 0;

  // Read many (entity,key) pairs; batched per plugin.
  virtual std::vector<Sample> read(const gpum_entity_id& entity,
                                   const std::vector<std::string>& keys) = 0;
  virtual std::vector<Sample> readMany(
      const std::vector<std::pair<gpum_entity_id, std::string>>& reqs) = 0;

  // Names of successfully loaded plugins.
  virtual std::vector<std::string> loadedPlugins() const = 0;
};

// Render a value with its unit, e.g. "42 C", "1400 MHz".
std::string FormatSample(const MetricDescriptor& desc, const Sample& s);

// Canonical label for an entity id ("gpu:0", "g0.1", "socket:1").
std::string EntityLabel(const gpum_entity_id& id);

}  // namespace gpumetrics

#endif  // GPUMETRICS_GPUMETRICS_H_
