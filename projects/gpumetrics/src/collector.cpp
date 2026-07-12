// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <sstream>
#include <unordered_map>

#include "device_registry.h"
#include "gpumetrics/gpumetrics.h"
#include "plugin_host.h"

namespace fs = std::filesystem;

namespace gpumetrics {

double Sample::as_double() const {
  if (status != GPUM_OK) return 0.0;
  switch (type) {
    case GPUM_TYPE_U64:
      return static_cast<double>(std::get<uint64_t>(value));
    case GPUM_TYPE_I64:
      return static_cast<double>(std::get<int64_t>(value));
    case GPUM_TYPE_F64:
      return std::get<double>(value);
    case GPUM_TYPE_STRING:
      return 0.0;
  }
  return 0.0;
}

std::string Sample::to_string() const {
  if (status != GPUM_OK) return gpum_status_string(status);
  char buf[64];
  switch (type) {
    case GPUM_TYPE_U64:
      std::snprintf(buf, sizeof(buf), "%llu",
                    static_cast<unsigned long long>(std::get<uint64_t>(value)));
      return buf;
    case GPUM_TYPE_I64:
      std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(std::get<int64_t>(value)));
      return buf;
    case GPUM_TYPE_F64: {
      double d = std::get<double>(value);
      std::snprintf(buf, sizeof(buf), "%.3f", d);
      return buf;
    }
    case GPUM_TYPE_STRING:
      return std::get<std::string>(value);
  }
  return "";
}

std::string FormatSample(const MetricDescriptor& desc, const Sample& s) {
  if (!s.ok()) return gpum_status_string(s.status);
  std::string v = s.to_string();
  if (!desc.unit.empty() && s.type != GPUM_TYPE_STRING) return v + " " + desc.unit;
  return v;
}

std::string EntityLabel(const gpum_entity_id& id) {
  switch (id.kind) {
    case GPUM_ENTITY_SOCKET:
      return "socket:" + std::to_string(id.socket);
    case GPUM_ENTITY_GPU:
      return "gpu:" + std::to_string(id.gpu);
    case GPUM_ENTITY_GPU_PARTITION:
      return "g" + std::to_string(id.gpu) + "." + std::to_string(id.partition);
  }
  return "?";
}

namespace {

Value from_c_value(const gpum_sample& s) {
  switch (s.type) {
    case GPUM_TYPE_U64:
      return s.value.u64;
    case GPUM_TYPE_I64:
      return s.value.i64;
    case GPUM_TYPE_F64:
      return s.value.f64;
    case GPUM_TYPE_STRING:
      return std::string(s.value.str);
  }
  return uint64_t{0};
}

std::vector<std::string> DefaultPluginDirs() {
  std::vector<std::string> dirs;
  if (const char* env = std::getenv("GPUMETRICS_PLUGIN_PATH")) {
    std::stringstream ss(env);
    std::string tok;
    while (std::getline(ss, tok, ':'))
      if (!tok.empty()) dirs.push_back(tok);
  }
  // Conventional build/install layout relative to cwd, plus common install dirs.
  dirs.push_back(".");
  dirs.push_back("./plugins/amdsmi");
  dirs.push_back("./plugins/rocprofiler");
  dirs.push_back("/opt/rocm/lib/gpumetrics");
  dirs.push_back("/usr/local/lib/gpumetrics");
  return dirs;
}

// Parse "0000:63:00.0" (or bus:dev.fn without domain) into a packed bdf.
bool ParseBdf(const std::string& s, uint64_t* out) {
  unsigned dom, bus, dev, fn;
  if (std::sscanf(s.c_str(), "%x:%x:%x.%x", &dom, &bus, &dev, &fn) == 4) {
    *out = gpum_bdf_pack(dom, static_cast<uint8_t>(bus), static_cast<uint8_t>(dev),
                         static_cast<uint8_t>(fn));
    return true;
  }
  if (std::sscanf(s.c_str(), "%x:%x.%x", &bus, &dev, &fn) == 3) {
    *out = gpum_bdf_pack(0, static_cast<uint8_t>(bus), static_cast<uint8_t>(dev),
                         static_cast<uint8_t>(fn));
    return true;
  }
  return false;
}

bool ParseUuid(const std::string& s, uint8_t out[16]) {
  std::string hex;
  for (char c : s)
    if (std::isxdigit(static_cast<unsigned char>(c))) hex.push_back(c);
  if (hex.size() != 32) return false;
  for (int i = 0; i < 16; ++i) {
    out[i] = static_cast<uint8_t>(std::stoul(hex.substr(i * 2, 2), nullptr, 16));
  }
  return true;
}

}  // namespace

class CollectorImpl : public Collector {
 public:
  CollectorImpl() = default;

  gpum_status init(const CollectorOptions& opts) {
    opts_ = opts;
    std::vector<std::string> dirs = opts.plugin_paths;
    if (dirs.empty()) dirs = DefaultPluginDirs();
    auto paths = DiscoverPluginPaths(dirs, opts.plugins);

    for (const auto& p : paths) {
      std::string err;
      auto lp = LoadedPlugin::Open(p, &err);
      if (!lp) {
        load_errors_.push_back(p + ": " + err);
        continue;
      }
      // Skip duplicate plugin names (e.g. found in two dirs).
      if (byName(lp->name())) continue;
      plugins_.push_back(std::move(lp));
    }

    std::vector<PluginDeviceRef> refs;
    for (const auto& lp : plugins_) {
      for (const auto& d : lp->devices()) {
        refs.push_back({lp->name(), d.identity, d.name});
      }
    }
    registry_ = DeviceRegistry::Build(refs, opts.provider_priority);

    buildMetricRegistry();
    return GPUM_OK;
  }

  const std::vector<Socket>& sockets() const override { return registry_.sockets(); }
  const std::vector<Device>& devices() const override { return registry_.devices(); }
  const std::vector<MetricDescriptor>& metrics() const override { return metric_list_; }

  std::optional<MetricDescriptor> describe(const std::string& key) const override {
    auto it = metric_index_.find(key);
    if (it == metric_index_.end()) return std::nullopt;
    return metric_list_[it->second];
  }

  std::vector<std::string> loadedPlugins() const override {
    std::vector<std::string> n;
    for (const auto& p : plugins_) n.push_back(p->name());
    return n;
  }

  std::optional<Entity> resolve(const std::string& sel) const override {
    auto make = [](const gpum_entity_id& id) {
      return Entity{id, EntityLabel(id)};
    };
    // g<gpu>.<part>
    if (sel.size() >= 4 && (sel[0] == 'g' || sel[0] == 'c') && sel.find('.') != std::string::npos &&
        std::isdigit(static_cast<unsigned char>(sel[1]))) {
      int gpu = -1, part = -1;
      if (std::sscanf(sel.c_str() + 1, "%d.%d", &gpu, &part) == 2 && gpu >= 0 && part >= 0) {
        const Device* d = registry_.gpuByIndex(static_cast<uint32_t>(gpu));
        if (!d) return std::nullopt;
        gpum_entity_id id;
        id.kind = GPUM_ENTITY_GPU_PARTITION;
        id.socket = d->id.socket;
        id.gpu = static_cast<uint32_t>(gpu);
        id.partition = part;
        return make(id);
      }
    }
    auto prefix = [&](const char* p) -> std::optional<std::string> {
      size_t n = std::strlen(p);
      if (sel.size() > n && sel.compare(0, n, p) == 0) return sel.substr(n);
      return std::nullopt;
    };
    if (auto v = prefix("gpu:")) {
      const Device* d = registry_.gpuByIndex(std::strtoul(v->c_str(), nullptr, 10));
      if (d) return make(d->id);
      return std::nullopt;
    }
    if (auto v = prefix("socket:")) {
      uint32_t s = std::strtoul(v->c_str(), nullptr, 10);
      for (const auto& sk : registry_.sockets())
        if (sk.index == s) {
          gpum_entity_id id;
          id.kind = GPUM_ENTITY_SOCKET;
          id.socket = s;
          id.gpu = 0;
          id.partition = -1;
          return make(id);
        }
      return std::nullopt;
    }
    if (auto v = prefix("bdf:")) {
      uint64_t bdf;
      if (ParseBdf(*v, &bdf)) {
        const Device* d = registry_.gpuByBdf(bdf);
        if (d) return make(d->id);
      }
      return std::nullopt;
    }
    if (auto v = prefix("uuid:")) {
      uint8_t u[16];
      if (ParseUuid(*v, u)) {
        const Device* d = registry_.gpuByUuid(u);
        if (d) return make(d->id);
      }
      return std::nullopt;
    }
    // bare number -> gpu index
    if (!sel.empty() && std::all_of(sel.begin(), sel.end(),
                                    [](char c) { return std::isdigit((unsigned char)c); })) {
      const Device* d = registry_.gpuByIndex(std::strtoul(sel.c_str(), nullptr, 10));
      if (d) return make(d->id);
    }
    return std::nullopt;
  }

  std::vector<Entity> entities(gpum_entity_kind kind) const override {
    std::vector<Entity> out;
    if (kind == GPUM_ENTITY_SOCKET) {
      for (const auto& s : registry_.sockets()) {
        gpum_entity_id id;
        id.kind = GPUM_ENTITY_SOCKET;
        id.socket = s.index;
        id.gpu = 0;
        id.partition = -1;
        out.push_back({id, EntityLabel(id)});
      }
    } else if (kind == GPUM_ENTITY_GPU) {
      for (const auto& d : registry_.devices()) out.push_back({d.id, EntityLabel(d.id)});
    } else {  // partitions
      for (const auto& d : registry_.devices()) {
        if (d.partitions.empty()) continue;
        for (int32_t p : d.partitions) {
          gpum_entity_id id;
          id.kind = GPUM_ENTITY_GPU_PARTITION;
          id.socket = d.id.socket;
          id.gpu = d.id.gpu;
          id.partition = p;
          out.push_back({id, EntityLabel(id)});
        }
      }
    }
    return out;
  }

  Sample read(const gpum_entity_id& e, const std::string& key) override {
    auto v = read(e, std::vector<std::string>{key});
    return v.empty() ? Sample{key, GPUM_ERR_INTERNAL, GPUM_TYPE_F64, {}, 0} : v[0];
  }

  std::vector<Sample> read(const gpum_entity_id& e, const std::vector<std::string>& keys) override {
    std::vector<std::pair<gpum_entity_id, std::string>> reqs;
    reqs.reserve(keys.size());
    for (const auto& k : keys) reqs.emplace_back(e, k);
    return readMany(reqs);
  }

  std::vector<Sample> readMany(
      const std::vector<std::pair<gpum_entity_id, std::string>>& reqs) override {
    std::vector<Sample> out(reqs.size());
    for (size_t i = 0; i < reqs.size(); ++i) {
      out[i].key = reqs[i].second;
      out[i].status = GPUM_ERR_NOT_FOUND;
    }

    // Group requests by (plugin, local index, partition) for one batched read
    // call per plugin.
    struct Key {
      std::string plugin;
      uint32_t local;
      int32_t partition;
      bool operator<(const Key& o) const {
        if (plugin != o.plugin) return plugin < o.plugin;
        if (local != o.local) return local < o.local;
        return partition < o.partition;
      }
    };
    struct Batched {
      std::vector<gpum_read_req> reqs;
      std::vector<size_t> out_idx;  // maps batch position -> out[] position
    };
    std::map<Key, Batched> batches;

    for (size_t i = 0; i < reqs.size(); ++i) {
      const auto& [ent, key] = reqs[i];
      auto mi = metric_index_.find(key);
      if (mi == metric_index_.end()) {
        out[i].status = GPUM_ERR_NOT_FOUND;
        continue;
      }
      const std::string& provider = metric_list_[mi->second].provider;
      out[i].type = metric_list_[mi->second].type;

      // find a handle for this entity served by the metric's provider
      auto handles = registry_.handlesFor(ent);
      const PluginHandle* chosen = nullptr;
      for (const auto& h : handles)
        if (h.plugin == provider) {
          chosen = &h;
          break;
        }
      if (!chosen) {
        out[i].status = GPUM_ERR_UNSUPPORTED;
        continue;
      }
      Key k{chosen->plugin, chosen->plugin_local_index, chosen->partition};
      auto& b = batches[k];
      gpum_read_req rr;
      rr.plugin_local_index = chosen->plugin_local_index;
      rr.partition = chosen->partition;
      rr.key = metric_list_[mi->second].key.c_str();
      b.reqs.push_back(rr);
      b.out_idx.push_back(i);
    }

    for (auto& [k, b] : batches) {
      LoadedPlugin* lp = byName(k.plugin);
      if (!lp) continue;
      std::vector<gpum_sample> samples;
      gpum_status st = lp->read(b.reqs, samples);
      for (size_t j = 0; j < b.out_idx.size(); ++j) {
        size_t oi = b.out_idx[j];
        if (st != GPUM_OK) {
          out[oi].status = st;
          continue;
        }
        const gpum_sample& s = samples[j];
        out[oi].status = s.status;
        out[oi].type = s.type;
        out[oi].timestamp_ns = s.timestamp_ns;
        if (s.status == GPUM_OK) out[oi].value = from_c_value(s);
      }
    }
    return out;
  }

 private:
  LoadedPlugin* byName(const std::string& n) {
    for (auto& p : plugins_)
      if (p->name() == n) return p.get();
    return nullptr;
  }

  void buildMetricRegistry() {
    // Order plugins by provider priority so conflicts go to the higher one.
    std::vector<LoadedPlugin*> ordered;
    for (const auto& name : opts_.provider_priority)
      for (auto& p : plugins_)
        if (p->name() == name) ordered.push_back(p.get());
    for (auto& p : plugins_)
      if (std::find(ordered.begin(), ordered.end(), p.get()) == ordered.end())
        ordered.push_back(p.get());

    for (LoadedPlugin* lp : ordered) {
      for (const auto& m : lp->metrics()) {
        std::string key = m.key;
        if (metric_index_.count(key)) continue;  // higher-priority provider won
        MetricDescriptor d;
        d.key = key;
        d.type = m.type;
        d.unit = m.unit;
        d.scope = m.scope;
        d.description = m.description;
        d.provider = lp->name();
        metric_index_[key] = metric_list_.size();
        metric_list_.push_back(std::move(d));
      }
    }
    std::sort(metric_list_.begin(), metric_list_.end(),
              [](const MetricDescriptor& a, const MetricDescriptor& b) { return a.key < b.key; });
    metric_index_.clear();
    for (size_t i = 0; i < metric_list_.size(); ++i) metric_index_[metric_list_[i].key] = i;
  }

  CollectorOptions opts_;
  std::vector<std::unique_ptr<LoadedPlugin>> plugins_;
  DeviceRegistry registry_;
  std::vector<MetricDescriptor> metric_list_;
  std::unordered_map<std::string, size_t> metric_index_;
  std::vector<std::string> load_errors_;
};

std::unique_ptr<Collector> Collector::Create(const CollectorOptions& opts, gpum_status* status) {
  auto c = std::make_unique<CollectorImpl>();
  gpum_status st = c->init(opts);
  if (status) *status = st;
  if (st != GPUM_OK) return nullptr;
  return c;
}

}  // namespace gpumetrics
