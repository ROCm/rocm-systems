// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/statistics.h"

#include <array>
#include <cstdint>
#include <memory>

namespace rocjitsu {

/// @brief Optional cache read accounting, bound while the owning cache is quiescent.
/// @details Requests count incoming spans, while classifications count controller
/// chunks. L1 vector spans follow coalescing; scalar dword loads classify dword
/// chunks; L2 and scalar byte loads classify line chunks. A failed admission can
/// leave a request unclassified, and a classified miss can subsequently retry.
/// These observations describe the functional cache model, not silicon counters.
class CacheStatistics {
public:
  void bind(const std::shared_ptr<stats::Scope> &scope) {
    *this = CacheStatistics{};
    if (!scope)
      return;
    requests_ = scope->counter("cache.read_requests", "requests",
                               "Nonempty incoming read spans attempted, including retries");
    bytes_ = scope->counter("cache.read_bytes", "bytes", "Bytes in attempted incoming read spans");
    hits_ =
        scope->counter("cache.read_hit_chunks", "chunks", "Demand chunks served by resident bytes");
    misses_ =
        scope->counter("cache.read_miss_chunks", "chunks",
                       "Demand lookups without usable resident bytes, including retried misses");
    bypasses_ = scope->counter("cache.read_bypass_chunks", "chunks",
                               "Demand chunks bypassing cache lookup or forcing coherent refetch");
    unavailable_probes_ = scope->counter("cache.peer_probe_unavailable", "requests",
                                         "Peer probes refused before resident-byte lookup");
    constexpr std::array bounds{1., 2., 4., 8., 16., 32., 64., 128., 256., 512., 1024., 4096.};
    sizes_ =
        scope->histogram("cache.read_size", "bytes", "Attempted incoming read span sizes", bounds);
    observe_requests_ = requests_ || bytes_ || sizes_;
  }

  void request(uint64_t bytes) const noexcept {
    if (!observe_requests_ || bytes == 0)
      return;
    if (requests_)
      requests_->increment();
    if (bytes_)
      bytes_->add(bytes);
    if (sizes_)
      sizes_->observe(static_cast<double>(bytes));
  }

  void hit() const noexcept {
    if (hits_)
      hits_->increment();
  }

  void miss() const noexcept {
    if (misses_)
      misses_->increment();
  }

  void bypass(uint64_t chunks = 1) const noexcept {
    if (bypasses_)
      bypasses_->add(chunks);
  }

  void peer_probe_unavailable() const noexcept {
    if (unavailable_probes_)
      unavailable_probes_->increment();
  }

private:
  std::shared_ptr<stats::RegisteredCounter> requests_;
  std::shared_ptr<stats::RegisteredCounter> bytes_;
  std::shared_ptr<stats::RegisteredCounter> hits_;
  std::shared_ptr<stats::RegisteredCounter> misses_;
  std::shared_ptr<stats::RegisteredCounter> bypasses_;
  std::shared_ptr<stats::RegisteredCounter> unavailable_probes_;
  std::shared_ptr<stats::RegisteredHistogram> sizes_;
  bool observe_requests_ = false;
};

} // namespace rocjitsu
