// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/diagnostics/hazard_sites.h"

#include "rocjitsu/vm/plugins/kernel_dispatch_info.h"
#include "util/json.h"
#include "util/log.h"

#include <cstddef>
#include <format>
#include <string>
#include <utility>

namespace rocjitsu::diagnostics {
namespace {

size_t mix(size_t seed, uint64_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
}

} // namespace

size_t HazardSites::KeyHash::operator()(const HazardSiteKey &key) const noexcept {
  size_t value = key.dispatch;
  value = mix(value, key.producer_pc);
  value = mix(value, key.consumer_pc);
  return mix(value, (uint64_t{key.reg} << 16) | (uint64_t{key.reg_class} << 8) | key.checker);
}

HazardSites::HazardSites() : HazardSites(Settings{}) {}

HazardSites::HazardSites(Settings settings, uint32_t gpu)
    : settings_(std::move(settings)), gpu_(gpu), writer_(report_writer()) {}

HazardSites::~HazardSites() = default;

HazardSites::UnitCache *HazardSites::cache_for(const void *unit) {
  // Fibonacci hashing keeps page-aligned unit addresses from sharing a start slot.
  const size_t start = static_cast<size_t>(
      (reinterpret_cast<uintptr_t>(unit) * 0x9e3779b97f4a7c15ull) >> (64 - kUnitBits));
  if (UnitEntry *entries = units_.load(std::memory_order_acquire)) {
    for (size_t probe = 0; probe < kUnitEntries; ++probe) {
      UnitEntry &entry = entries[(start + probe) & (kUnitEntries - 1)];
      const void *owner = entry.unit.load(std::memory_order_acquire);
      if (owner == unit)
        return entry.cache.load(std::memory_order_relaxed);
      if (owner == nullptr)
        break;
    }
  }
  std::lock_guard lock(mutex_);
  if (!unit_storage_) {
    unit_storage_ = std::make_unique<UnitEntry[]>(kUnitEntries);
    units_.store(unit_storage_.get(), std::memory_order_release);
  }
  for (size_t probe = 0; probe < kUnitEntries; ++probe) {
    UnitEntry &entry = unit_storage_[(start + probe) & (kUnitEntries - 1)];
    const void *owner = entry.unit.load(std::memory_order_relaxed);
    if (owner == unit)
      return entry.cache.load(std::memory_order_relaxed);
    if (owner == nullptr) {
      caches_.push_back(std::make_unique<UnitCache>());
      entry.cache.store(caches_.back().get(), std::memory_order_relaxed);
      entry.unit.store(unit, std::memory_order_release);
      return caches_.back().get();
    }
  }
  return nullptr;
}

void HazardSites::register_dispatch(DispatchIdentity identity) {
  const uint32_t id = identity.dispatch_id;
  std::lock_guard lock(mutex_);
  identities_.insert_or_assign(id, std::move(identity));
}

DispatchIdentity HazardSites::find_dispatch(uint32_t dispatch_id) const {
  std::lock_guard lock(mutex_);
  auto found = identities_.find(dispatch_id);
  return found == identities_.end() ? DispatchIdentity{} : found->second;
}

namespace {

uint32_t slot_tag(const HazardSiteKey &key) {
  return 0x80000000u | (uint32_t{key.checker} << 24) | (uint32_t{key.reg_class} << 16) |
         (key.reg & 0xffffu);
}

size_t slot_group(const HazardSiteKey &key) {
  const uint64_t value = key.consumer_pc ^ (key.producer_pc << 7) ^ (uint64_t{key.reg} << 40);
  return static_cast<size_t>((value * 0x9e3779b97f4a7c15ull) >> 60);
}

} // namespace

HazardDisposition HazardSites::classify(const void *unit, const HazardSiteKey &key) {
  static_assert(UnitCache::kGroups == 16, "slot_group() yields four bits");
  UnitCache *cache = cache_for(unit);
  const uint32_t tag = slot_tag(key);
  const size_t group = slot_group(key);
  if (cache != nullptr) {
    UnitCache::Slot *ways = &cache->slots[group * UnitCache::kWays];
    for (size_t way = 0; way < UnitCache::kWays; ++way) {
      UnitCache::Slot &slot = ways[way];
      if (slot.tag.load(std::memory_order_relaxed) != tag ||
          slot.consumer_pc.load(std::memory_order_relaxed) != key.consumer_pc ||
          slot.producer_pc.load(std::memory_order_relaxed) != key.producer_pc ||
          slot.dispatch.load(std::memory_order_relaxed) != key.dispatch)
        continue;
      // Until its dispatch retires, a slot's count has a single writer: this unit.
      slot.pending.store(slot.pending.load(std::memory_order_relaxed) + 1,
                         std::memory_order_relaxed);
      return slot.suppressed.load(std::memory_order_relaxed) ? HazardDisposition::Suppressed
                                                             : HazardDisposition::Repeat;
    }
  }
  HazardDisposition disposition = HazardDisposition::New;
  bool announce = false;
  Site *site = nullptr;
  {
    std::lock_guard lock(mutex_);
    Dispatch &dispatch = dispatches_[key.dispatch];
    if (auto found = dispatch.sites.find(key); found != dispatch.sites.end()) {
      site = found->second.get();
      disposition = HazardDisposition::Repeat;
    } else if (dispatch.sites.size() >= settings_.max_sites_per_dispatch) {
      site = &dispatch.suppressed;
      announce = !std::exchange(dispatch.suppression_printed, true);
      disposition = HazardDisposition::Suppressed;
    } else {
      site = dispatch.sites.emplace(key, std::make_unique<Site>()).first->second.get();
    }
    site->occurrences.fetch_add(1, std::memory_order_relaxed);
  }
  if (announce)
    writer_.publish(
        [&] {
          util::Logger::warn(std::format("HAZARD_SUPPRESSED dispatch={} max_sites_per_dispatch={}",
                                         key.dispatch, settings_.max_sites_per_dispatch));
        },
        settings_.report_path,
        std::format("{{\"record\":\"suppressed\",\"gpu\":{},\"dispatch\":{},\"max_sites_per_"
                    "dispatch\":{}}}",
                    gpu_, key.dispatch, settings_.max_sites_per_dispatch));
  if (cache != nullptr)
    install(*cache, group, tag, key, site, disposition == HazardDisposition::Suppressed);
  return disposition;
}

void HazardSites::install(UnitCache &cache, size_t group, uint32_t tag, const HazardSiteKey &key,
                          Site *site, bool suppressed) {
  std::lock_guard lock(cache.mutex);
  UnitCache::Slot *ways = &cache.slots[group * UnitCache::kWays];
  UnitCache::Slot *slot = nullptr;
  for (size_t way = 0; way < UnitCache::kWays && slot == nullptr; ++way)
    if (ways[way].tag.load(std::memory_order_relaxed) == 0)
      slot = &ways[way];
  if (slot == nullptr) {
    slot = &ways[cache.victim[group]++ % UnitCache::kWays];
    flush(*slot);
  }
  slot->dispatch.store(key.dispatch, std::memory_order_relaxed);
  slot->producer_pc.store(key.producer_pc, std::memory_order_relaxed);
  slot->consumer_pc.store(key.consumer_pc, std::memory_order_relaxed);
  slot->site.store(site, std::memory_order_relaxed);
  slot->suppressed.store(suppressed, std::memory_order_relaxed);
  slot->pending.store(0, std::memory_order_relaxed);
  slot->tag.store(tag, std::memory_order_relaxed);
}

// A slot with a nonzero count belongs to a dispatch that has not retired, so its
// site is still alive; merge() zeroes and clears a dispatch's slots before then.
void HazardSites::flush(UnitCache::Slot &slot) {
  const uint64_t pending = slot.pending.exchange(0, std::memory_order_relaxed);
  if (pending != 0)
    slot.site.load(std::memory_order_relaxed)
        ->occurrences.fetch_add(pending, std::memory_order_relaxed);
}

void HazardSites::publish(const HazardRecord &record) const {
  std::string text = format_text(record, "HAZARD");
  // Logger::warn ends the line itself.
  text.pop_back();
  writer_.publish([&] { util::Logger::warn(text); }, settings_.report_path,
                  settings_.report_path.empty() ? std::string() : format_jsonl(record));
}

void HazardSites::merge(UnitCache &cache, uint32_t dispatch_id) {
  std::lock_guard lock(cache.mutex);
  for (UnitCache::Slot &slot : cache.slots) {
    if (slot.tag.load(std::memory_order_relaxed) == 0 ||
        slot.dispatch.load(std::memory_order_relaxed) != dispatch_id)
      continue;
    flush(slot);
    slot.tag.store(0, std::memory_order_relaxed);
  }
}

void HazardSites::finish_dispatch(uint32_t dispatch_id) {
  std::vector<UnitCache *> caches;
  {
    std::lock_guard lock(mutex_);
    if (!dispatches_.contains(dispatch_id)) {
      identities_.erase(dispatch_id);
      return;
    }
    caches.reserve(caches_.size());
    for (const auto &cache : caches_)
      caches.push_back(cache.get());
  }
  // Every wave of the dispatch has retired, so no unit adds to its slots any more.
  for (UnitCache *cache : caches)
    merge(*cache, dispatch_id);
  std::string summary;
  std::string summary_json;
  {
    std::lock_guard lock(mutex_);
    if (auto found = dispatches_.find(dispatch_id); found != dispatches_.end()) {
      const Dispatch &dispatch = found->second;
      uint64_t occurrences = 0;
      for (const auto &[key, site] : dispatch.sites)
        occurrences += site->occurrences.load(std::memory_order_relaxed);
      const auto identity = identities_.find(dispatch_id);
      const std::string kernel = identity == identities_.end() || identity->second.kernel.empty()
                                     ? std::string(kUnknownKernelIdentity)
                                     : identity->second.kernel;
      summary = std::format("HAZARD_SUMMARY dispatch={} kernel={} sites={} occurrences={}",
                            dispatch_id, kernel, dispatch.sites.size(), occurrences);
      summary_json = std::format("{{\"record\":\"summary\",\"gpu\":{},\"dispatch\":{},"
                                 "\"kernel\":\"{}\",\"sites\":{},\"occurrences\":{}",
                                 gpu_, dispatch_id, util::json_escape(kernel),
                                 dispatch.sites.size(), occurrences);
      if (const uint64_t suppressed =
              dispatch.suppressed.occurrences.load(std::memory_order_relaxed)) {
        summary += std::format(" suppressed={}", suppressed);
        summary_json += std::format(",\"suppressed\":{}", suppressed);
      }
      summary_json += '}';
      dispatches_.erase(found);
    }
    identities_.erase(dispatch_id);
  }
  if (!summary.empty())
    writer_.publish([&] { util::Logger::warn(summary); }, settings_.report_path, summary_json);
}

ReportWriter &report_writer() {
  // Never destroyed: a compute unit may report during static destruction.
  static auto *writer = new ReportWriter;
  return *writer;
}

namespace {

struct UnattachedSites {
  std::mutex mutex;
  std::unordered_map<const void *, std::unique_ptr<HazardSites>> sites;
};

UnattachedSites &unattached() {
  // Never destroyed: a compute unit may outlive static destruction.
  static auto *value = new UnattachedSites;
  return *value;
}

} // namespace

HazardSites &unattached_sites(const void *unit) {
  UnattachedSites &all = unattached();
  std::lock_guard lock(all.mutex);
  std::unique_ptr<HazardSites> &sites = all.sites[unit];
  if (!sites)
    sites = std::make_unique<HazardSites>();
  return *sites;
}

void release_unattached_sites(const void *unit) {
  std::unique_ptr<HazardSites> released;
  UnattachedSites &all = unattached();
  std::lock_guard lock(all.mutex);
  if (auto found = all.sites.find(unit); found != all.sites.end()) {
    released = std::move(found->second);
    all.sites.erase(found);
  }
}

} // namespace rocjitsu::diagnostics
