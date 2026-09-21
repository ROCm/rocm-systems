// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/bar_access_trace.h"

#include "util/log.h"

#include <algorithm>
#include <format>
#include <vector>

namespace rocjitsu {

BarAccessTrace::BarAccessTrace(const RegisterSymbols &symbols)
    : BarAccessTrace(symbols, Config{}) {}

BarAccessTrace::BarAccessTrace(const RegisterSymbols &symbols, Config config)
    : symbols_(symbols), config_(config) {}

uint64_t BarAccessTrace::make_key(int bar, uint64_t offset) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(bar)) << 56) |
         (offset & 0x00ffffffffffffffULL);
}

void BarAccessTrace::record_rejected(int bar, uint64_t offset, std::size_t width, bool write) {
  const uint64_t key = make_key(bar, offset);
  std::lock_guard lock(mutex_);
  Site &site = rejected_[key];
  site.bar = bar;
  site.offset = offset;
  site.width = width;
  if (write) {
    ++site.writes;
  } else {
    ++site.reads;
  }
}

void BarAccessTrace::record(int bar, uint64_t offset, std::size_t width, bool write, bool modeled) {
  const uint64_t key = make_key(bar, offset);

  std::unique_lock lock(mutex_);

  if (!modeled) {
    Site &site = unmodeled_[key];
    site.bar = bar;
    site.offset = offset;
    site.width = width;
    if (write) {
      ++site.writes;
    } else {
      ++site.reads;
    }
  }

  // A write means the driver acted on something it read, so any polling loop it
  // was in has made progress. Only an unbroken run of reads of one register is
  // evidence of a wait that will not end.
  if (write) {
    repeated_read_count_ = 0;
    repeated_read_warned_ = false;
    return;
  }

  if (repeated_read_count_ != 0 && repeated_read_key_ == key) {
    ++repeated_read_count_;
  } else {
    repeated_read_key_ = key;
    repeated_read_count_ = 1;
    repeated_read_warned_ = false;
  }

  if (repeated_read_count_ < config_.spin_threshold || repeated_read_warned_) {
    return;
  }
  repeated_read_warned_ = true;
  ++spin_warnings_;

  const std::string_view name = symbols_.lookup(bar, offset);
  const uint32_t count = repeated_read_count_;
  lock.unlock();

  util::Logger::warn(
      std::format("vfu: guest has read bar{} +{:#x} ({}) {} times with no intervening write; "
                  "the driver is likely spinning on a status bit this device never sets",
                  bar, offset, name.empty() ? "unnamed register" : name, count));
}

std::string BarAccessTrace::unmodeled_report() const {
  return render(unmodeled_, "unmodeled register(s)");
}

std::string BarAccessTrace::rejected_report() const {
  return render(rejected_, "refused access(es)");
}

std::string BarAccessTrace::render(const std::unordered_map<uint64_t, Site> &from,
                                   std::string_view heading) const {
  std::vector<Site> sites;
  {
    const std::lock_guard lock(mutex_);
    sites.reserve(from.size());
    for (const auto &[key, site] : from) {
      sites.push_back(site);
    }
  }
  if (sites.empty()) {
    return {};
  }

  std::ranges::sort(sites, [](const Site &lhs, const Site &rhs) {
    const uint64_t lhs_total = lhs.reads + lhs.writes;
    const uint64_t rhs_total = rhs.reads + rhs.writes;
    if (lhs_total != rhs_total) {
      return lhs_total > rhs_total;
    }
    if (lhs.bar != rhs.bar) {
      return lhs.bar < rhs.bar;
    }
    return lhs.offset < rhs.offset;
  });

  std::string report = std::format("vfu: {} {}, most used first:\n", sites.size(), heading);
  for (const Site &site : sites) {
    const std::string_view name = symbols_.lookup(site.bar, site.offset);
    report +=
        std::format("  bar{} +{:#010x} width={} reads={} writes={} {}\n", site.bar, site.offset,
                    site.width, site.reads, site.writes, name.empty() ? "?" : name);
  }
  return report;
}

uint64_t BarAccessTrace::spin_warnings() const {
  const std::lock_guard lock(mutex_);
  return spin_warnings_;
}

} // namespace rocjitsu
