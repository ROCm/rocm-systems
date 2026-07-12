// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "device_registry.h"

#include <algorithm>
#include <cstring>

namespace gpumetrics {

namespace {
bool uuid_nonzero(const uint8_t u[16]) {
  for (int i = 0; i < 16; ++i)
    if (u[i]) return true;
  return false;
}
bool uuid_eq(const uint8_t a[16], const uint8_t b[16]) { return std::memcmp(a, b, 16) == 0; }

int priority_index(const std::vector<std::string>& prio, const std::string& name) {
  for (size_t i = 0; i < prio.size(); ++i)
    if (prio[i] == name) return static_cast<int>(i);
  return static_cast<int>(prio.size());
}
}  // namespace

bool SamePhysicalGpu(const gpum_device_identity& a, const gpum_device_identity& b) {
  // Compare on the highest-priority key both identities share, ordered by how
  // well each survives partitioning: masked BDF and oam_id are per physical GPU;
  // kfd/uuid/render are per partition on CPX but per GPU otherwise, so they are
  // the fallback for consumer GPUs. The top shared key decides.
  const uint64_t ma = gpum_bdf_masked(a.bdf), mb = gpum_bdf_masked(b.bdf);
  if (ma != 0 && mb != 0) return ma == mb;
  if (a.oam_id != GPUM_ID_UNKNOWN && b.oam_id != GPUM_ID_UNKNOWN) return a.oam_id == b.oam_id;
  if (a.kfd_node_id != 0 && b.kfd_node_id != 0) return a.kfd_node_id == b.kfd_node_id;
  if (uuid_nonzero(a.uuid) && uuid_nonzero(b.uuid)) return uuid_eq(a.uuid, b.uuid);
  if (a.drm_render_minor != 0 && b.drm_render_minor != 0)
    return a.drm_render_minor == b.drm_render_minor;
  return false;  // no shared key
}

DeviceRegistry DeviceRegistry::Build(const std::vector<PluginDeviceRef>& refs,
                                     const std::vector<std::string>& provider_priority) {
  DeviceRegistry reg;

  // For each ref, find or create the canonical GPU it matches (by
  // SamePhysicalGpu against existing merged identities). Partition refs
  // (partition_index >= 0) fold into their physical GPU.
  struct Canon {
    gpum_device_identity merged{};  // best keys seen
    std::string best_name;
    int best_name_prio = 1 << 30;
    uint32_t socket_hint = GPUM_SOCKET_UNKNOWN;
  };
  std::vector<Canon> canon;
  std::vector<size_t> ref_to_canon(refs.size(), SIZE_MAX);

  // Fold a ref's identity into the canonical GPU. Canonical BDF is always masked
  // (physical-GPU key). The whole-GPU handle is authoritative for per-GPU keys:
  // prefer its kfd/uuid over a partition's (per-partition on CPX).
  auto merge_identity = [](gpum_device_identity& dst, const gpum_device_identity& src) {
    if (gpum_bdf_masked(dst.bdf) == 0) dst.bdf = gpum_bdf_masked(src.bdf);
    if (dst.oam_id == GPUM_ID_UNKNOWN) dst.oam_id = src.oam_id;
    if (dst.socket_id == GPUM_SOCKET_UNKNOWN) dst.socket_id = src.socket_id;
    const bool src_is_whole = src.partition_index < 0;
    if (src_is_whole || dst.kfd_node_id == 0) {
      if (src.kfd_node_id != 0) dst.kfd_node_id = src.kfd_node_id;
      if (uuid_nonzero(src.uuid)) std::memcpy(dst.uuid, src.uuid, 16);
      if (src.drm_render_minor != 0) dst.drm_render_minor = src.drm_render_minor;
    }
  };

  for (size_t i = 0; i < refs.size(); ++i) {
    const auto& r = refs[i];
    size_t found = SIZE_MAX;
    for (size_t c = 0; c < canon.size(); ++c) {
      if (SamePhysicalGpu(canon[c].merged, r.identity)) {
        found = c;
        break;
      }
    }
    if (found == SIZE_MAX) {
      Canon c;
      c.merged = r.identity;
      c.merged.bdf = gpum_bdf_masked(r.identity.bdf);  // physical-GPU key
      c.merged.partition_index = -1;
      c.best_name = r.name;
      c.best_name_prio = priority_index(provider_priority, r.plugin);
      c.socket_hint = r.identity.socket_id;
      canon.push_back(c);
      found = canon.size() - 1;
    } else {
      merge_identity(canon[found].merged, r.identity);
      int p = priority_index(provider_priority, r.plugin);
      if (!r.name.empty() && p < canon[found].best_name_prio) {
        canon[found].best_name = r.name;
        canon[found].best_name_prio = p;
      }
      if (canon[found].socket_hint == GPUM_SOCKET_UNKNOWN)
        canon[found].socket_hint = r.identity.socket_id;
    }
    ref_to_canon[i] = found;
  }

  // Assign canonical GPU ordinals in a stable order: socket hint, then BDF.
  std::vector<size_t> order(canon.size());
  for (size_t i = 0; i < canon.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    uint32_t sa = canon[a].socket_hint, sb = canon[b].socket_hint;
    if (sa != sb) return sa < sb;
    return canon[a].merged.bdf < canon[b].merged.bdf;
  });
  std::vector<uint32_t> canon_to_ordinal(canon.size());
  for (uint32_t ord = 0; ord < order.size(); ++ord) canon_to_ordinal[order[ord]] = ord;

  reg.devices_.resize(canon.size());
  reg.routes_.resize(canon.size());

  // Socket assignment: group by socket hint; unknown hints get their own socket.
  std::map<uint32_t, uint32_t> hint_to_socket;
  uint32_t next_socket = 0;
  auto socket_for = [&](uint32_t hint, uint32_t gpu_ord) -> uint32_t {
    if (hint == GPUM_SOCKET_UNKNOWN) return next_socket++;  // unique
    auto it = hint_to_socket.find(hint);
    if (it != hint_to_socket.end()) return it->second;
    uint32_t s = next_socket++;
    hint_to_socket[hint] = s;
    return s;
  };

  // First pass: fill devices + socket ids (in ordinal order for determinism).
  std::vector<uint32_t> gpu_socket(canon.size());
  for (uint32_t ord = 0; ord < order.size(); ++ord) {
    size_t c = order[ord];
    uint32_t sock = socket_for(canon[c].socket_hint, ord);
    gpu_socket[ord] = sock;
    Device d;
    d.id.kind = GPUM_ENTITY_GPU;
    d.id.socket = sock;
    d.id.gpu = ord;
    d.id.partition = -1;
    d.name = canon[c].best_name;
    d.identity = canon[c].merged;
    d.identity.plugin_local_index = 0;  // meaningless at canonical level
    reg.devices_[ord] = std::move(d);
  }

  // Second pass: routing + partition discovery from refs.
  for (size_t i = 0; i < refs.size(); ++i) {
    const auto& r = refs[i];
    uint32_t ord = canon_to_ordinal[ref_to_canon[i]];
    auto& route = reg.routes_[ord];
    GpuRoute::PerPlugin* pp = nullptr;
    for (auto& e : route.plugins)
      if (e.plugin == r.plugin) {
        pp = &e;
        break;
      }
    if (!pp) {
      route.plugins.push_back({});
      pp = &route.plugins.back();
      pp->plugin = r.plugin;
    }
    if (r.identity.partition_index < 0) {
      pp->whole_local_index = r.identity.plugin_local_index;
      pp->has_whole = true;
    } else {
      pp->partition_local_index[r.identity.partition_index] = r.identity.plugin_local_index;
    }
    auto& dev = reg.devices_[ord];
    if (std::find(dev.providers.begin(), dev.providers.end(), r.plugin) == dev.providers.end())
      dev.providers.push_back(r.plugin);
    if (r.identity.partition_index >= 0) {
      if (std::find(dev.partitions.begin(), dev.partitions.end(), r.identity.partition_index) ==
          dev.partitions.end())
        dev.partitions.push_back(r.identity.partition_index);
    }
  }
  for (auto& d : reg.devices_) std::sort(d.partitions.begin(), d.partitions.end());

  // Build sockets.
  std::map<uint32_t, Socket> smap;
  for (const auto& d : reg.devices_) {
    auto& s = smap[d.id.socket];
    s.index = d.id.socket;
    s.gpus.push_back(d.id.gpu);
  }
  for (auto& [k, v] : smap) {
    v.name = "socket:" + std::to_string(k);
    reg.sockets_.push_back(v);
  }
  return reg;
}

std::vector<PluginHandle> DeviceRegistry::handlesFor(const gpum_entity_id& e) const {
  std::vector<PluginHandle> out;
  auto add_gpu = [&](uint32_t gpu, int32_t partition) {
    if (gpu >= routes_.size()) return;
    for (const auto& pp : routes_[gpu].plugins) {
      if (partition < 0) {
        if (pp.has_whole) out.push_back({pp.plugin, pp.whole_local_index, -1});
      } else {
        auto it = pp.partition_local_index.find(partition);
        if (it != pp.partition_local_index.end()) {
          out.push_back({pp.plugin, it->second, partition});
        } else if (pp.has_whole) {
          // plugin doesn't model this partition; serve from whole device with
          // the partition hint.
          out.push_back({pp.plugin, pp.whole_local_index, partition});
        }
      }
    }
  };

  switch (e.kind) {
    case GPUM_ENTITY_GPU:
      add_gpu(e.gpu, -1);
      break;
    case GPUM_ENTITY_GPU_PARTITION:
      add_gpu(e.gpu, e.partition);
      break;
    case GPUM_ENTITY_SOCKET:
      for (const auto& s : sockets_)
        if (s.index == e.socket)
          for (uint32_t g : s.gpus) add_gpu(g, -1);
      break;
  }
  return out;
}

const Device* DeviceRegistry::gpuByIndex(uint32_t gpu) const {
  if (gpu < devices_.size()) return &devices_[gpu];
  return nullptr;
}
const Device* DeviceRegistry::gpuByBdf(uint64_t bdf) const {
  for (const auto& d : devices_)
    if (d.identity.bdf == bdf && bdf != 0) return &d;
  return nullptr;
}
const Device* DeviceRegistry::gpuByUuid(const uint8_t uuid[16]) const {
  for (const auto& d : devices_)
    if (uuid_nonzero(d.identity.uuid) && uuid_eq(d.identity.uuid, uuid)) return &d;
  return nullptr;
}

}  // namespace gpumetrics
