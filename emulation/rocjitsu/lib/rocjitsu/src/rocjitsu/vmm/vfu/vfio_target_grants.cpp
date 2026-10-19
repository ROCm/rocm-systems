// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vmm/vfu/vfio_target_grants.h"

#include "rocjitsu/vmm/vfu/vfio_device_host.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rocjitsu {

namespace {

amdgpu::MemoryIngressPermit
retain_peer_effects(std::shared_ptr<std::vector<amdgpu::VmTransferEffectGate::Permit>> retained,
                    std::vector<std::weak_ptr<VfioQuiesceWake>> wakes) {
  // MemoryIngressPermit uses a copyable callback; keep the move-only gate
  // permits together until the whole peer effect has finished.
  return amdgpu::MemoryIngressPermit(
      [retained = std::move(retained), wakes = std::move(wakes)]() mutable {
        const bool held_effect = !retained->empty();
        retained.reset();
        if (held_effect)
          for (const auto &weak_wake : wakes)
            if (const auto wake = weak_wake.lock())
              wake->notify();
      });
}

} // namespace

VfioQuiesceWake::VfioQuiesceWake() : fd_(::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)) {}

VfioQuiesceWake::~VfioQuiesceWake() {
  if (fd_ >= 0)
    (void)::close(fd_);
}

void VfioQuiesceWake::notify() const noexcept {
  if (fd_ < 0)
    return;
  const uint64_t one = 1;
  (void)::write(fd_, &one, sizeof(one));
}

VfioTargetGrants::VfioTargetGrants(std::size_t host_count)
    : hosts_(host_count, nullptr), wakes_(host_count), gpu_ids_(host_count, 0),
      quiescing_(host_count, false), failed_peer_range_records_(host_count, false),
      admission_lifetime_(std::make_shared<AdmissionLifetime>()) {
  admission_lifetime_->owner = this;
}

VfioTargetGrants::~VfioTargetGrants() {
  const std::lock_guard lock(admission_lifetime_->mutex);
  admission_lifetime_->owner = nullptr;
}

bool VfioTargetGrants::bind(uint32_t ordinal, VfioDeviceHost &host, uint32_t gpu_id) {
  const std::lock_guard lock(mutex_);
  if (ordinal >= hosts_.size() || hosts_[ordinal] != nullptr)
    return false;
  hosts_[ordinal] = &host;
  wakes_[ordinal] = host.quiesce_wake();
  gpu_ids_[ordinal] = gpu_id;
  return true;
}

void VfioTargetGrants::publish_bar_window(
    uint32_t ordinal, std::optional<simdojo::PciBarAddressDecoder::Window> window) {
  struct RevokedGrant {
    IovaKey key;
    std::shared_ptr<amdgpu::VmTransferEffectGate> gate;
  };
  std::vector<RevokedGrant> revoked;
  {
    const std::lock_guard lock(mutex_);
    if (ordinal >= hosts_.size())
      return;
    if (window && window->function != ordinal)
      window.reset();
    const auto previous = address_decoder_.window(ordinal, GpuPciDevice::kVramBar);
    if (previous == window)
      return;
    address_decoder_.revoke(ordinal, GpuPciDevice::kVramBar);
    if (window && !address_decoder_.publish(*window))
      window.reset();
    for (const auto &[key, grant] : grants_) {
      if (grant.map.owner_gpu_ordinal != ordinal || grant.map.owner_bar != GpuPciDevice::kVramBar)
        continue;
      grant.effect_gate->close();
      revoked.push_back({key, grant.effect_gate});
    }
  }
  for (const RevokedGrant &entry : revoked)
    entry.gate->revoke();
  const std::lock_guard lock(mutex_);
  for (const RevokedGrant &entry : revoked) {
    const auto it = grants_.find(entry.key);
    if (it != grants_.end() && it->second.effect_gate == entry.gate)
      grants_.erase(it);
  }
}

std::optional<uint64_t> VfioTargetGrants::register_standard_map(uint32_t requester,
                                                                const simdojo::DmaRegion &region) {
  if (region.length == 0 || region.access == simdojo::DmaAccess::None ||
      region.guest_phys > UINT64_MAX - region.length)
    return std::nullopt;
  VfioDeviceHost *requester_host = nullptr;
  {
    const std::lock_guard lock(mutex_);
    if (requester >= hosts_.size())
      return std::nullopt;
    requester_host = hosts_[requester];
  }
  if (requester_host == nullptr)
    return std::nullopt;
  const auto session = requester_host->target_session_snapshot();
  if (session == nullptr || session->state() != simdojo::PciTransportSession::State::Open)
    return std::nullopt;
  MapRequest map;
  {
    const std::lock_guard lock(mutex_);
    if (hosts_[requester] != requester_host ||
        session->state() != simdojo::PciTransportSession::State::Open)
      return std::nullopt;
    const auto owner =
        address_decoder_.resolve(requester, region.guest_phys, region.length, region.access);
    if (!owner)
      return std::nullopt;
    if (owner->function >= hosts_.size())
      return std::nullopt;
    if (!requester_host->can_route_peer(gpu_ids_[owner->function])) {
      try {
        remember_peer_range_locked({requester, session->generation()}, region.guest_phys,
                                   region.length);
      } catch (...) {
        // The protocol cannot reject DMA_MAP. Deny all copies for this host
        // until its session retires if a rejected peer interval cannot be kept.
        failed_peer_range_records_[requester] = true;
      }
      return std::nullopt;
    }
    if (next_standard_map_id_ == UINT64_MAX)
      return std::nullopt;
    map = {.iova = region.guest_phys,
           .size = region.length,
           .map_id = next_standard_map_id_++,
           .owner_offset = owner->offset,
           .owner_gpu_ordinal = owner->function,
           .owner_bar = owner->bar,
           .rights = region.access,
           .decoded_owner_generation = owner->generation};
  }
  return register_map(requester, map);
}

void VfioTargetGrants::remember_peer_range_locked(SessionKey session, uint64_t iova,
                                                  uint64_t size) {
  assert(size != 0 && iova <= UINT64_MAX - size);
  auto &known = known_peer_ranges_[session];
  auto [range, inserted] = known.try_emplace(iova, iova + size);
  if (!inserted)
    range->second = std::max(range->second, iova + size);
  if (range != known.begin()) {
    auto prior = std::prev(range);
    if (prior->second >= range->first) {
      prior->second = std::max(prior->second, range->second);
      known.erase(range);
      range = prior;
    }
  }
  auto next = std::next(range);
  while (next != known.end() && next->first <= range->second) {
    range->second = std::max(range->second, next->second);
    next = known.erase(next);
  }
}

bool VfioTargetGrants::overlaps_known_peer_range_locked(SessionKey session, uint64_t begin,
                                                        uint64_t end) const {
  if (begin >= end)
    return false;
  const auto known = known_peer_ranges_.find(session);
  if (known == known_peer_ranges_.end())
    return false;
  const auto range = known->second.lower_bound(begin);
  return (range != known->second.end() && range->first < end) ||
         (range != known->second.begin() && std::prev(range)->second > begin);
}

void VfioTargetGrants::unregister_standard_map(uint32_t requester, uint64_t requester_generation,
                                               uint64_t iova, uint64_t size) {
  MapRequest map;
  {
    const std::lock_guard lock(mutex_);
    const auto it = grants_.find({requester, iova});
    if (it == grants_.end() || it->second.requester_generation != requester_generation ||
        it->second.map.size != size)
      return;
    map = it->second.map;
  }
  unregister_map(requester, requester_generation, map);
}

std::optional<uint64_t> VfioTargetGrants::register_map(uint32_t requester, const MapRequest &map) {
  if (map.size == 0 || map.map_id == 0 || map.iova > UINT64_MAX - map.size ||
      map.owner_offset > UINT64_MAX - map.size || map.owner_gpu_ordinal == requester ||
      map.owner_bar != GpuPciDevice::kVramBar || map.rights == simdojo::DmaAccess::None)
    return std::nullopt;

  VfioDeviceHost *requester_host = nullptr;
  VfioDeviceHost *owner_host = nullptr;
  {
    const std::lock_guard lock(mutex_);
    if (requester >= hosts_.size() || map.owner_gpu_ordinal >= hosts_.size())
      return std::nullopt;
    requester_host = hosts_[requester];
    owner_host = hosts_[map.owner_gpu_ordinal];
  }
  if (requester_host == nullptr || owner_host == nullptr)
    return std::nullopt;

  // The two snapshots may race reset, but their sessions are subsequently
  // acquired at access time. A reset closes them before draining operations.
  auto requester_session = requester_host->target_session_snapshot();
  auto owner_session = owner_host->target_session_snapshot();
  if (requester_session == nullptr || owner_session == nullptr ||
      requester_session->state() != simdojo::PciTransportSession::State::Open ||
      owner_session->state() != simdojo::PciTransportSession::State::Open ||
      owner_session->generation() != map.decoded_owner_generation)
    return std::nullopt;

  std::shared_ptr<std::byte> mapped_bytes;
  uint64_t backing_offset = 0;
  uint64_t backing_id = 0;
  for (const simdojo::BarSpec &bar : owner_host->device().bars()) {
    if (bar.index != static_cast<int>(map.owner_bar))
      continue;
    if (!bar.mem || bar.backing_fd < 0 || map.owner_offset + map.size > bar.size ||
        bar.fd_offset > UINT64_MAX - map.owner_offset - map.size)
      return std::nullopt;
    bool in_mmap_area = false;
    for (const simdojo::MmapArea &area : bar.mmap_areas) {
      if (map.owner_offset >= area.offset && map.owner_offset - area.offset <= area.length &&
          map.size <= area.length - (map.owner_offset - area.offset)) {
        in_mmap_area = true;
        break;
      }
    }
    if (!in_mmap_area)
      return std::nullopt;

    // A direct shared mapping has no recoverable short-write path. Require a
    // memfd whose size cannot change after this extent check, then retain the
    // mapping until UNMAP/reset and every admitted copy have finished.
    const int seals = ::fcntl(bar.backing_fd, F_GET_SEALS);
    constexpr int kStableSizeSeals = F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL;
    if (seals < 0 || (seals & kStableSizeSeals) != kStableSizeSeals)
      return std::nullopt;
    struct stat backing {};
    if (::fstat(bar.backing_fd, &backing) != 0 || backing.st_size < 0 ||
        bar.fd_offset + map.owner_offset + map.size > static_cast<uint64_t>(backing.st_size))
      return std::nullopt;

    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
      return std::nullopt;
    const uint64_t file_offset = bar.fd_offset + map.owner_offset;
    const uint64_t aligned_offset = file_offset - file_offset % static_cast<uint64_t>(page_size);
    const uint64_t delta = file_offset - aligned_offset;
    if (aligned_offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
        map.size > std::numeric_limits<std::size_t>::max() ||
        delta > std::numeric_limits<std::size_t>::max() - map.size ||
        delta + map.size > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()))
      return std::nullopt;
    const auto mapped_length = static_cast<std::size_t>(delta + map.size);
    void *base = ::mmap(nullptr, mapped_length, PROT_READ | PROT_WRITE, MAP_SHARED, bar.backing_fd,
                        static_cast<off_t>(aligned_offset));
    if (base == MAP_FAILED)
      return std::nullopt;
    auto mapping = std::shared_ptr<std::byte>(
        static_cast<std::byte *>(base),
        [mapped_length](std::byte *bytes) { (void)::munmap(bytes, mapped_length); });
    mapped_bytes = std::shared_ptr<std::byte>(mapping, mapping.get() + delta);
    backing_offset = file_offset;
    backing_id = bar.backing_id;
    break;
  }
  if (mapped_bytes == nullptr)
    return std::nullopt;

  const uint64_t requester_generation = requester_session->generation();
  const SessionKey session{requester, requester_generation};
  const IovaKey key{requester, map.iova};
  const std::lock_guard lock(mutex_);
  const auto owner = address_decoder_.resolve(requester, map.iova, map.size, map.rights);
  if (!owner || owner->function != map.owner_gpu_ordinal || owner->bar != map.owner_bar ||
      owner->offset != map.owner_offset || owner->generation != map.decoded_owner_generation)
    return std::nullopt;
  const auto highest = highest_map_ids_.find(session);
  const auto next = grants_.lower_bound(key);
  const bool overlaps_next = next != grants_.end() && next->first.first == requester &&
                             next->first.second < map.iova + map.size;
  const bool overlaps_prior = next != grants_.begin() && [&] {
    const auto prior = std::prev(next);
    return prior->first.first == requester &&
           prior->second.map.iova + prior->second.map.size > map.iova;
  }();
  if ((highest != highest_map_ids_.end() && map.map_id <= highest->second) || overlaps_next ||
      overlaps_prior || requester_session->state() != simdojo::PciTransportSession::State::Open ||
      owner_session->state() != simdojo::PciTransportSession::State::Open)
    return std::nullopt;
  const Grant grant{.map = map,
                    .requester_generation = requester_generation,
                    .owner_generation = owner_session->generation(),
                    .backing_offset = backing_offset,
                    .backing_id = backing_id,
                    .mapped_bytes = std::move(mapped_bytes),
                    .requester_session = requester_session,
                    .owner_session = owner_session,
                    .effect_gate = std::make_shared<amdgpu::VmTransferEffectGate>()};
  const auto [entry, inserted] = grants_.emplace(key, grant);
  if (!inserted)
    return std::nullopt;
  try {
    // Allocate the new node before changing any retained tombstones. If that
    // allocation fails, an old peer range must not fall through to guest RAM.
    remember_peer_range_locked(session, map.iova, map.size);
    if (highest != highest_map_ids_.end())
      highest->second = map.map_id;
    else
      highest_map_ids_.emplace(session, map.map_id);
  } catch (...) {
    grants_.erase(entry);
    failed_peer_range_records_[requester] = true;
    throw;
  }
  return requester_generation;
}

void VfioTargetGrants::unregister_map(uint32_t requester, uint64_t requester_generation,
                                      const MapRequest &map) {
  const std::lock_guard lock(mutex_);
  const auto it = grants_.find({requester, map.iova});
  if (it != grants_.end() && it->second.requester_generation == requester_generation &&
      it->second.map.size == map.size && it->second.map.map_id == map.map_id) {
    it->second.effect_gate->close();
    grants_.erase(it);
  }
}

void VfioTargetGrants::revoke_session(uint32_t ordinal, uint64_t generation) {
  const auto matches = [ordinal, generation](const auto &entry) noexcept {
    const Grant &grant = entry.second;
    return (entry.first.first == ordinal && grant.requester_generation == generation) ||
           (grant.map.owner_gpu_ordinal == ordinal && grant.owner_generation == generation);
  };
  {
    const std::lock_guard lock(mutex_);
    const auto window = address_decoder_.window(ordinal, GpuPciDevice::kVramBar);
    if (window && window->generation == generation)
      address_decoder_.revoke(ordinal, GpuPciDevice::kVramBar);
    // Close all matching gates before releasing the lock. Drain each grant
    // below without allocating a collection during session retirement.
    for (const auto &entry : grants_)
      if (matches(entry))
        entry.second.effect_gate->close();
    highest_map_ids_.erase({ordinal, generation});
    known_peer_ranges_.erase({ordinal, generation});
    failed_peer_range_records_[ordinal] = false;
  }
  for (;;) {
    std::shared_ptr<amdgpu::VmTransferEffectGate> gate;
    IovaKey key;
    {
      const std::lock_guard lock(mutex_);
      const auto it = std::find_if(grants_.begin(), grants_.end(), matches);
      if (it == grants_.end())
        break;
      key = it->first;
      gate = it->second.effect_gate;
    }
    // Keep the grant discoverable to another host retiring the same peer
    // relationship. Both detach paths must wait for its admitted effects.
    gate->revoke();
    {
      const std::lock_guard lock(mutex_);
      const auto it = grants_.find(key);
      if (it != grants_.end() && matches(*it) && it->second.effect_gate == gate)
        grants_.erase(it);
    }
  }
}

VfioTargetGrants::QuiesceGates VfioTargetGrants::begin_quiesce(uint32_t ordinal) {
  QuiesceGates gates;
  const std::lock_guard lock(mutex_);
  if (ordinal >= quiescing_.size())
    return gates;
  for (const auto &[key, grant] : grants_)
    if (key.first == ordinal || grant.map.owner_gpu_ordinal == ordinal)
      gates.push_back(grant.effect_gate);
  quiescing_[ordinal] = true;
  return gates;
}

bool VfioTargetGrants::drained(const QuiesceGates &gates) noexcept {
  return std::ranges::all_of(gates, [](const auto &gate) { return gate->drained(); });
}

void VfioTargetGrants::end_quiesce(uint32_t ordinal) {
  const std::lock_guard lock(mutex_);
  if (ordinal < quiescing_.size())
    quiescing_[ordinal] = false;
}

amdgpu::MemoryIngressPermit
VfioTargetGrants::retain_effect(std::optional<amdgpu::VmTransferEffectGate::Permit> permit,
                                uint32_t requester, uint32_t owner) {
  auto retained = std::make_shared<amdgpu::VmTransferEffectGate::Permit>(std::move(*permit));
  auto requester_wake = wakes_[requester];
  auto owner_wake = wakes_[owner];
  return amdgpu::MemoryIngressPermit(
      [retained = std::move(retained), requester_wake, owner_wake]() mutable {
        retained.reset();
        if (const auto wake = requester_wake.lock())
          wake->notify();
        if (const auto wake = owner_wake.lock())
          wake->notify();
      });
}

VfioTargetGrants::PeerPreparation VfioTargetGrants::prepare_peer(uint32_t requester,
                                                                 uint64_t requester_generation,
                                                                 uint64_t iova, std::size_t length,
                                                                 bool to_guest) {
  if (length == 0 || length > UINT64_MAX - iova)
    return {.kind = PeerKind::Rejected,
            .outcome = simdojo::DmaAccessOutcome::Malformed,
            .snapshot = {}};
  const std::lock_guard lock(mutex_);
  if (requester < failed_peer_range_records_.size() && failed_peer_range_records_[requester])
    return {
        .kind = PeerKind::Rejected, .outcome = simdojo::DmaAccessOutcome::Faulted, .snapshot = {}};
  auto it = grants_.lower_bound({requester, iova});
  if (it != grants_.begin()) {
    const auto prior = std::prev(it);
    if (prior->first.first == requester && prior->second.map.iova + prior->second.map.size > iova)
      it = prior;
  }
  if (it == grants_.end() || it->first.first != requester || it->first.second >= iova + length) {
    if (overlaps_known_peer_range_locked({requester, requester_generation}, iova, iova + length))
      return {.kind = PeerKind::Rejected,
              .outcome = simdojo::DmaAccessOutcome::Faulted,
              .snapshot = {}};
    return {};
  }
  const Grant &grant = it->second;
  const auto owner =
      address_decoder_.resolve(requester, grant.map.iova, grant.map.size, grant.map.rights);
  if (!owner || owner->function != grant.map.owner_gpu_ordinal ||
      owner->bar != grant.map.owner_bar || owner->offset != grant.map.owner_offset ||
      owner->generation != grant.map.decoded_owner_generation)
    return {.kind = PeerKind::Rejected,
            .outcome = simdojo::DmaAccessOutcome::Unavailable,
            .snapshot = {}};
  const bool contained =
      iova >= grant.map.iova && length <= grant.map.size - (iova - grant.map.iova);
  if (!contained || grant.requester_generation != requester_generation ||
      !simdojo::contains(grant.map.rights,
                         to_guest ? simdojo::DmaAccess::Write : simdojo::DmaAccess::Read))
    return {
        .kind = PeerKind::Rejected, .outcome = simdojo::DmaAccessOutcome::Faulted, .snapshot = {}};
  const auto owner_session = grant.owner_session.lock();
  if (!owner_session || owner_session->generation() != grant.owner_generation ||
      owner_session->state() != simdojo::PciTransportSession::State::Open)
    return {.kind = PeerKind::Rejected,
            .outcome = simdojo::DmaAccessOutcome::Unavailable,
            .snapshot = {}};
  const auto offset = static_cast<std::size_t>(iova - grant.map.iova);
  return {.kind = PeerKind::Ready,
          .outcome = simdojo::DmaAccessOutcome::Complete,
          .snapshot = {.map_iova = grant.map.iova,
                       .map_size = grant.map.size,
                       .iova = iova,
                       .map_id = grant.map.map_id,
                       .requester_generation = grant.requester_generation,
                       .owner_generation = grant.owner_generation,
                       .owner_offset = grant.map.owner_offset + offset,
                       .backing_offset = grant.backing_offset + offset,
                       .backing_id = grant.backing_id,
                       .requester_ordinal = requester,
                       .owner_ordinal = grant.map.owner_gpu_ordinal,
                       .owner_gpu_id = gpu_ids_[grant.map.owner_gpu_ordinal],
                       .length = length,
                       .mapped_bytes = std::shared_ptr<std::byte>(
                           grant.mapped_bytes, grant.mapped_bytes.get() + offset),
                       .effect_gate = grant.effect_gate}};
}

amdgpu::MemoryIngressAdmission VfioTargetGrants::admit_peer(const PeerSnapshot &snapshot) {
  amdgpu::MemoryIngressPermit cohort_release;
  std::vector<std::weak_ptr<VfioQuiesceWake>> wakes;
  {
    const std::lock_guard lock(mutex_);
    const auto it = grants_.find({snapshot.requester_ordinal, snapshot.map_iova});
    if (it == grants_.end())
      return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
    const Grant &grant = it->second;
    if (snapshot.iova < grant.map.iova || snapshot.length == 0)
      return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
    const uint64_t offset = snapshot.iova - grant.map.iova;
    if (offset >= grant.map.size || snapshot.length > grant.map.size - offset ||
        snapshot.map_size != grant.map.size || snapshot.map_id != grant.map.map_id ||
        snapshot.requester_generation != grant.requester_generation ||
        snapshot.owner_generation != grant.owner_generation ||
        snapshot.owner_ordinal != grant.map.owner_gpu_ordinal ||
        snapshot.owner_gpu_id != gpu_ids_[grant.map.owner_gpu_ordinal] ||
        snapshot.owner_offset != grant.map.owner_offset + offset ||
        snapshot.backing_offset != grant.backing_offset + offset ||
        snapshot.backing_id != grant.backing_id || snapshot.mapped_bytes == nullptr ||
        snapshot.mapped_bytes.get() != grant.mapped_bytes.get() + offset ||
        snapshot.mapped_bytes.owner_before(grant.mapped_bytes) ||
        grant.mapped_bytes.owner_before(snapshot.mapped_bytes) ||
        snapshot.effect_gate != grant.effect_gate)
      return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};

    struct Candidate {
      uint32_t requester;
      const Grant *grant;
    };
    std::vector<Candidate> selected;
    if (snapshot.backing_id == 0) {
      // Generic BARs have no canonical identity, so retain only the exact
      // grant as before rather than treating unrelated zero IDs as aliases.
      selected.push_back({snapshot.requester_ordinal, &grant});
    } else {
      constexpr uint64_t kLineSize = CanonicalLineFreshness::kLineSize;
      const uint64_t first_line = snapshot.backing_offset & ~(kLineSize - 1);
      const uint64_t last_line_byte =
          (snapshot.backing_offset + snapshot.length - 1) | (kLineSize - 1);
      for (const auto &[key, candidate] : grants_) {
        if (candidate.backing_id == snapshot.backing_id &&
            candidate.backing_offset <= last_line_byte &&
            candidate.backing_offset + candidate.map.size > first_line)
          selected.push_back({key.first, &candidate});
      }
    }
    if (selected.empty())
      return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};

    std::vector<bool> affected_hosts(hosts_.size(), false);
    for (const Candidate &candidate : selected) {
      const Grant &alias = *candidate.grant;
      const uint32_t owner = alias.map.owner_gpu_ordinal;
      if (quiescing_[candidate.requester] || quiescing_[owner])
        return {.outcome = amdgpu::VmAccessOutcome::Unavailable, .permit = {}};
      const auto owner_session = alias.owner_session.lock();
      if (!owner_session || owner_session->generation() != alias.owner_generation ||
          owner_session->state() != simdojo::PciTransportSession::State::Open)
        return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
      affected_hosts[candidate.requester] = true;
      affected_hosts[owner] = true;
    }

    auto retained = std::make_shared<std::vector<amdgpu::VmTransferEffectGate::Permit>>();
    retained->reserve(selected.size());
    wakes.reserve(hosts_.size());
    for (std::size_t ordinal = 0; ordinal < affected_hosts.size(); ++ordinal)
      if (affected_hosts[ordinal])
        wakes.push_back(wakes_[ordinal]);
    // Construct the release callback before entering any gate. An allocation
    // failure cannot strand an acquired permit without waking VFIO quiesce.
    cohort_release = retain_peer_effects(retained, std::move(wakes));
    for (const Candidate &candidate : selected) {
      auto permit = candidate.grant->effect_gate->try_acquire();
      if (!permit)
        return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
      retained->push_back(std::move(*permit));
    }
  }
  return {.outcome = amdgpu::VmAccessOutcome::Complete, .permit = std::move(cohort_release)};
}

std::function<amdgpu::MemoryIngressAdmission()>
VfioTargetGrants::capture_admission(PeerSnapshot snapshot) const {
  std::weak_ptr<AdmissionLifetime> weak_lifetime = admission_lifetime_;
  return [weak_lifetime, snapshot = std::move(snapshot)] {
    const auto lifetime = weak_lifetime.lock();
    if (!lifetime)
      return amdgpu::MemoryIngressAdmission{.outcome = amdgpu::VmAccessOutcome::Revoked,
                                            .permit = {}};
    const std::lock_guard lock(lifetime->mutex);
    return lifetime->owner != nullptr
               ? lifetime->owner->admit_peer(snapshot)
               : amdgpu::MemoryIngressAdmission{.outcome = amdgpu::VmAccessOutcome::Revoked,
                                                .permit = {}};
  };
}

VfioTargetGrants::PrepareResult VfioTargetGrants::prepare(uint32_t requester,
                                                          uint64_t requester_generation,
                                                          uint64_t iova, std::size_t length,
                                                          bool to_guest) {
  if (length > UINT64_MAX - iova)
    return {.outcome = simdojo::DmaAccessOutcome::Malformed, .segments = {}};
  const uint64_t end = iova + length;
  struct Candidate {
    Grant grant;
    amdgpu::MemoryIngressPermit effect_permit;
  };
  std::vector<Candidate> candidates;
  {
    const std::lock_guard lock(mutex_);
    if (requester < failed_peer_range_records_.size() && failed_peer_range_records_[requester])
      return {.outcome = simdojo::DmaAccessOutcome::Faulted, .segments = {}};
    const SessionKey session{requester, requester_generation};
    uint64_t uncovered_begin = iova;
    auto it = grants_.lower_bound({requester, iova});
    if (it != grants_.begin()) {
      const auto prior = std::prev(it);
      if (prior->first.first == requester && prior->second.map.iova + prior->second.map.size > iova)
        it = prior;
    }
    for (; it != grants_.end() && it->first.first == requester && it->first.second < end; ++it) {
      if (it->second.map.iova + it->second.map.size <= iova)
        continue;
      const Grant &grant = it->second;
      const uint64_t grant_begin = std::max(iova, grant.map.iova);
      if (uncovered_begin < grant_begin &&
          overlaps_known_peer_range_locked(session, uncovered_begin, grant_begin))
        return {.outcome = simdojo::DmaAccessOutcome::Faulted, .segments = {}};
      uncovered_begin = std::max(uncovered_begin, std::min(end, grant.map.iova + grant.map.size));
      if (quiescing_[requester] || quiescing_[grant.map.owner_gpu_ordinal])
        return {.outcome = simdojo::DmaAccessOutcome::Unavailable, .segments = {}};
      auto permit = grant.effect_gate->try_acquire();
      if (!permit)
        return {.outcome = simdojo::DmaAccessOutcome::Unavailable, .segments = {}};
      candidates.push_back(
          {grant, retain_effect(std::move(permit), requester, grant.map.owner_gpu_ordinal)});
    }
    if (uncovered_begin < end && overlaps_known_peer_range_locked(session, uncovered_begin, end))
      return {.outcome = simdojo::DmaAccessOutcome::Faulted, .segments = {}};
  }

  PrepareResult result;
  uint64_t prior_end = iova;
  for (Candidate &candidate : candidates) {
    const Grant &grant = candidate.grant;
    const uint64_t start = std::max(iova, grant.map.iova);
    const uint64_t stop = std::min(end, grant.map.iova + grant.map.size);
    if (start < prior_end || grant.requester_generation != requester_generation ||
        !simdojo::contains(grant.map.rights,
                           to_guest ? simdojo::DmaAccess::Write : simdojo::DmaAccess::Read))
      return {.outcome = simdojo::DmaAccessOutcome::Faulted, .segments = {}};
    auto owner_session = grant.owner_session.lock();
    if (owner_session == nullptr || owner_session->generation() != grant.owner_generation)
      return {.outcome = simdojo::DmaAccessOutcome::Unavailable, .segments = {}};
    auto lease = owner_session->acquire();
    if (!lease)
      return {.outcome = simdojo::DmaAccessOutcome::Unavailable, .segments = {}};
    result.segments.push_back(
        {.iova = start,
         .length = static_cast<std::size_t>(stop - start),
         .mapped_bytes = std::shared_ptr<std::byte>(
             grant.mapped_bytes,
             grant.mapped_bytes.get() + static_cast<std::size_t>(start - grant.map.iova)),
         .owner_lease = std::move(lease),
         .effect_permit = std::move(candidate.effect_permit)});
    prior_end = stop;
  }
  return result;
}

void VfioTargetGrants::PreparedSegment::copy(std::span<std::byte> bytes, bool to_guest) const {
  assert(bytes.size() == length && mapped_bytes != nullptr && owner_lease);
  if (to_guest)
    std::memmove(mapped_bytes.get(), bytes.data(), length);
  else
    std::memmove(bytes.data(), mapped_bytes.get(), length);
}

} // namespace rocjitsu
