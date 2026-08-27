// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/cache_des.h"

#include "rocjitsu/vm/timing/coalesce.h"
#include "rocjitsu/vm/timing/engine.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace rocjitsu::timing {
namespace {

/// @brief Round up to a power of two, never below one.
std::uint64_t round_up_pow2(std::uint64_t value) {
  return value <= 1 ? 1 : std::uint64_t{1} << line_shift_for(value);
}

/// @brief Cycles to move @p items at @p per_cycle, never fewer than one.
///
/// @details Never zero even for an empty request: a level that has been handed
/// a request has looked at it, and a rate high enough to round the work to
/// nothing would otherwise make an infinitely fast cache out of a rounding
/// mode.
std::uint64_t cycles_for(std::uint64_t items, double per_cycle) {
  if (!(per_cycle > 0.0))
    return std::max<std::uint64_t>(1, items);
  const double cycles = std::ceil(static_cast<double>(items) / per_cycle);
  if (!(cycles > 1.0))
    return 1;
  return static_cast<std::uint64_t>(cycles);
}

} // namespace

// -- TagArray ---------------------------------------------------------------

void TagArray::configure(std::uint64_t sets, std::uint64_t ways, std::uint64_t line_bytes) {
  line_bytes_ = round_up_pow2(line_bytes);
  line_shift_ = line_shift_for(line_bytes_);
  sets_ = round_up_pow2(std::max<std::uint64_t>(1, sets));
  ways_ = std::max<std::uint64_t>(1, ways);
  set_mask_ = sets_ - 1;
  tags_.assign(static_cast<std::size_t>(sets_ * ways_), kEmpty);
  stamps_.assign(static_cast<std::size_t>(sets_ * ways_), 0);
}

bool TagArray::access(std::uint64_t byte_address, std::uint64_t stamp) {
  if (ways_ == 0 || tags_.empty())
    return false;
  const std::uint64_t line = byte_address >> line_shift_;
  const std::size_t base = static_cast<std::size_t>((line & set_mask_) * ways_);
  std::size_t victim = base;
  std::uint64_t oldest = ~0ULL;
  for (std::uint64_t way = 0; way < ways_; ++way) {
    const std::size_t slot = base + static_cast<std::size_t>(way);
    if (tags_[slot] == line) {
      stamps_[slot] = stamp;
      return true;
    }
    // An empty way is always the victim; otherwise the least recently used one.
    const std::uint64_t age = tags_[slot] == kEmpty ? 0 : stamps_[slot];
    if (age < oldest) {
      oldest = age;
      victim = slot;
    }
  }
  tags_[victim] = line;
  stamps_[victim] = stamp;
  return false;
}

void TagArray::invalidate() { std::fill(tags_.begin(), tags_.end(), kEmpty); }

// -- CacheDes ---------------------------------------------------------------

CacheDes::CacheDes(std::string name, const simdojo::ClockDomain &domain, TimingEngine &engine,
                   const CacheTuning &tuning, std::vector<std::uint64_t> *line_pool)
    : TimedComponent(std::move(name), domain, engine), tuning_(tuning), line_pool_(line_pool) {
  tags_.configure(tuning_.sets, tuning_.ways, tuning_.line_bytes);
}

void CacheDes::invalidate() { tags_.invalidate(); }

std::uint64_t CacheDes::advance(std::uint64_t now) {
  if (inbox().empty())
    return 0;

  // Only what was already queued is served at this tick. A completion resumes
  // a wavefront, which can issue its next access straight back into this level,
  // and that arrival belongs to a later tick rather than to this drain. Each
  // request is copied out by value because such a re-entry can reallocate the
  // inbox underneath the loop; the inbox itself is kept, and its capacity with
  // it, because this is the hottest path in the plane and swapping it out would
  // cost a heap allocation on every wake.
  const std::size_t queued = inbox().size();

  const std::uint64_t line_bytes = tags_.line_bytes();
  const std::uint64_t line_shift = tags_.line_shift();

  for (std::size_t entry = 0; entry < queued; ++entry) {
    MemoryRequest request = inbox()[entry];

    // Re-derived per request, and only used before any callback runs: a
    // completion can resume a wavefront that appends to the pool, which
    // reallocates it and leaves this pointer dangling.
    std::uint64_t *lines = nullptr;
    if (line_pool_ != nullptr && request.line_count != 0 &&
        static_cast<std::size_t>(request.line_base) + request.line_count <= line_pool_->size())
      lines = line_pool_->data() + request.line_base;
    if (lines == nullptr)
      request.line_count = 0;

    // A level with wider lines than the one above it must not treat the finer
    // lines as separate work; see recoalesce_lines().
    if (lines != nullptr && request.line_bytes != line_bytes)
      request.line_count = recoalesce_lines(lines, request.line_count, line_shift);
    request.line_bytes = static_cast<std::uint32_t>(line_bytes);

    // A non-temporal access is documented not to allocate in a level configured
    // to honour that, and probing it anyway would invent a hit the hardware
    // does not have. The tag array is left entirely alone rather than probed
    // and then ignored, so the access also does not evict a line a temporal
    // access is still using.
    const bool probe = allocates_ || !request.non_temporal;

    // Partition the request's slice of the line pool in place: misses to the
    // front, hits behind them. The two halves are disjoint ranges of the pool,
    // so the miss half can travel downstream while the hit half is completed,
    // with no copy and no second allocation. This is the whole reason a request
    // names a pool range instead of carrying its own vector: a wave64 access is
    // up to sixty-four lines, and a hierarchy that allocated per split would
    // spend more time in the allocator than the emulator spends executing.
    std::uint32_t missed = 0;
    for (std::uint32_t index = 0; index < request.line_count; ++index) {
      const std::uint64_t address = lines[index];
      if (probe && tags_.access(address, stamp_++))
        continue;
      lines[index] = lines[missed];
      lines[missed] = address;
      ++missed;
    }
    const std::uint32_t hit_lines = request.line_count - missed;

    hits_.fetch_add(hit_lines, std::memory_order_relaxed);
    misses_.fetch_add(missed, std::memory_order_relaxed);
    // Every line the request presented crossed this level, whether it hit or
    // was filled through it. Charging only the hits would make a cache that
    // misses everything look like a cache that is doing nothing, when it is in
    // fact moving the most traffic it ever will.
    bytes_.fetch_add(static_cast<std::uint64_t>(request.line_count) * line_bytes,
                     std::memory_order_relaxed);

    // Occupancy is the whole request, not the hit part: every line costs a tag
    // lookup slot, and a fill occupies the data array on its way back up.
    const std::uint64_t done =
        reserve(now, cycles_for(request.line_count, tuning_.lines_per_cycle));

    // `hit_cycles` is documented as the delay from *request* to data, measured
    // at the requester, so it composes as a floor from the tick the request
    // entered the hierarchy and not as a term added at each level. Adding it
    // per level would charge a second-level hit the first level's latency as
    // well, when the config's second-level number already contains it. What is
    // added on top of the floor is congestion, and only congestion: `done` is
    // when this level actually finished, so a request that queued behind others
    // is late by exactly how long it waited. That is the honest form of the
    // queueing term a previous model tried to derive from utilisation, which
    // needed an elapsed time it did not have and stretched every early access
    // twentyfold.
    const std::uint64_t nominal = request.issued_tick + cycles_to_ticks(tuning_.hit_cycles);
    const std::uint64_t ready = std::max(nominal, done);

    // A request with no lines is still a request. The functional side sends
    // one for an access no lane participated in, and this level zeroes the
    // count for a request whose pool range does not exist. Either way it has to
    // be answered: a request that is never completed never retires the wait
    // counter it posted to, and the wavefront behind it waits forever.
    if (request.line_count == 0) {
      if (completion_)
        completion_(request, ready);
      continue;
    }

    if (hit_lines != 0 && completion_) {
      MemoryRequest hit = request;
      hit.line_base = request.line_base + missed;
      hit.line_count = hit_lines;
      completion_(hit, ready);
    }

    if (missed == 0)
      continue;

    MemoryRequest miss = request;
    miss.line_count = missed;
    miss.depth = request.depth == 0xFFu ? 0xFFu : static_cast<std::uint8_t>(request.depth + 1);

    if (downstream_) {
      // Handed over at the current tick rather than at `done`. A request
      // carries no arrival stamp -- `issued_tick` is its entry into the
      // hierarchy, which a completion needs to report the latency the
      // wavefront actually saw -- so the hand-off cannot be dated later without
      // destroying that. The downstream therefore starts up to this level's own
      // streaming time early, which is bounded by the request's line count over
      // `lines_per_cycle` and is small beside the round trip it is starting.
      downstream_(miss);
      continue;
    }

    // No downstream: this level is the bottom of the configured hierarchy, and
    // the lines it just missed have nowhere to go. Complete them here rather
    // than dropping them. A dropped request never retires its wait counter and
    // the wavefront that issued it waits forever, which turns a configuration
    // mistake into a hang; costing it as a hit is wrong by a latency and says
    // so in the miss counter.
    if (completion_)
      completion_(miss, ready);
  }

  inbox().erase(inbox().begin(), inbox().begin() + static_cast<std::ptrdiff_t>(queued));

  // Whatever a callback pushed back in has already re-armed this component
  // through deliver(), which waits for busy_until_ before it wakes, so there is
  // no self-wake to schedule here.
  return 0;
}

} // namespace rocjitsu::timing
