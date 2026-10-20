// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_race_checker.h"

#include "rocjitsu/vm/amdgpu/memory_diagnostic_report.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <format>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace rocjitsu::amdgpu {
namespace {

using Access = MemoryRaceChecker::Access;
using AffineLaneRange = MemoryRaceChecker::AffineLaneRange;
using Finding = MemoryRaceChecker::Finding;
using Kind = MemoryRaceChecker::Kind;
using LaneRange = MemoryRaceChecker::LaneRange;
using Space = MemoryRaceChecker::Space;
using Statistics = MemoryRaceChecker::Statistics;
using WaveKey = MemoryRaceChecker::WaveKey;
using CompletionToken = MemoryWaitScoreboard::CompletionToken;

constexpr uint32_t kNoIndex = std::numeric_limits<uint32_t>::max();
constexpr uint8_t kRead = 1;
constexpr uint8_t kWrite = 2;
constexpr uint8_t kAtomic = 4;
// Marks an LDS record whose operation was still in flight at a barrier.
constexpr uint8_t kInFlight = 8;
// Counter capacity bounds how many of a wave's accesses are truly in flight;
// past this many entries, completed ones are pruned before adding more.
constexpr size_t kPendingPruneThreshold = 64;
using StatisticsClock = std::chrono::steady_clock;

struct WordStatistics {
  uint64_t dword_candidates = 0;
  uint64_t dword_visits = 0;
  uint64_t dword_duplicates_skipped = 0;
  uint64_t affine_dword_runs = 0;
  uint64_t affine_dword_visits = 0;
};

uint64_t elapsed_ns(StatisticsClock::time_point started) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(StatisticsClock::now() - started)
          .count());
}

template <typename T> void hash_combine(size_t &seed, const T &value) {
  seed ^= std::hash<T>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

uint64_t mix(uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

uint64_t saturated_end(uint64_t address, uint32_t size) {
  return size > std::numeric_limits<uint64_t>::max() - address
             ? std::numeric_limits<uint64_t>::max()
             : address + size;
}

uint8_t flags_of(const Access &access) {
  return static_cast<uint8_t>((access.read ? kRead : 0) | (access.write ? kWrite : 0) |
                              (access.atomic ? kAtomic : 0));
}

void set_flags(Access &access, uint8_t flags) {
  access.read = flags & kRead;
  access.write = flags & kWrite;
  access.atomic = flags & kAtomic;
}

// Atomic pairs are ordered by the memory system; every other pair conflicts
// when at least one side writes.
bool conflicting(uint8_t prior, uint8_t incoming) {
  return ((prior | incoming) & kWrite) && !(prior & incoming & kAtomic);
}

Kind kind_of(uint8_t producer, uint8_t consumer) {
  if ((consumer & kWrite) && (producer & kWrite))
    return Kind::WAW;
  if ((consumer & kWrite) && (producer & kRead))
    return Kind::WAR;
  return Kind::RAW;
}

/// Bytes of the dword at `word` covered by [address, end). The caller
/// guarantees that the range touches the dword.
uint8_t word_mask(uint64_t word, uint64_t address, uint64_t end) {
  const uint32_t first = address > word ? static_cast<uint32_t>(address - word) : 0;
  const uint32_t last = end - word >= 4 ? 4 : static_cast<uint32_t>(end - word);
  return static_cast<uint8_t>((0xfu >> (4 - last)) & (0xfu << first));
}

int8_t compact_lane(int32_t lane) { return static_cast<int8_t>(std::clamp(lane, -1, 127)); }

Access consumer_of(const Access &instruction, const LaneRange &range) {
  Access consumer = instruction;
  consumer.address = range.address;
  consumer.size = range.size;
  consumer.lane = range.lane;
  return consumer;
}

template <typename Visit> void for_each_range(std::span<const LaneRange> lanes, Visit &&visit) {
  for (const LaneRange &range : lanes)
    visit(range);
}

template <typename Visit> void for_each_range(const AffineLaneRange &lanes, Visit &&visit) {
  for (uint32_t i = 0; i < lanes.count; ++i) {
    const LaneRange range{lanes.address + uint64_t{i} * lanes.stride, lanes.size,
                          lanes.first_lane + static_cast<int32_t>(i)};
    visit(range);
  }
}

template <typename Ranges> size_t range_count(const Ranges &lanes) { return lanes.size(); }
size_t range_count(const AffineLaneRange &lanes) { return lanes.count; }

/// Visit each dword a range sequence touches as (dword index, byte mask,
/// range). Repeated recent dword/mask pairs are skipped, preserving the first
/// lane attribution.
template <typename Ranges, typename Visit, typename Counters = Statistics>
void for_each_word_expanded(const Ranges &lanes, Visit &&visit, Counters *statistics = nullptr) {
  constexpr size_t kRecent = 64;
  std::array<uint64_t, kRecent> recent_index;
  std::array<uint8_t, kRecent> recent_mask;
  uint64_t aligned_run_begin = 0;
  uint64_t aligned_run_end = 0;
  bool aligned_run = false;
  const bool dedupe = range_count(lanes) > 1;
  if (dedupe)
    recent_index.fill(std::numeric_limits<uint64_t>::max());
  for_each_range(lanes, [&](const LaneRange &range) {
    if (!range.size)
      return;
    const uint64_t end = saturated_end(range.address, range.size);
    const uint64_t last = (end - 1) >> 2;
    uint64_t first = range.address >> 2;
    // Regular aligned lane ranges often overlap by several dwords. The recent
    // cache below would reject the covered prefix one dword at a time. Skip it
    // before expansion when every skipped dword is still guaranteed to occupy
    // the same direct-mapped cache slot, preserving the first lane attribution.
    if (dedupe && (range.address & 3) == 0 && (range.size & 3) == 0) {
      const uint64_t range_end = last + 1;
      const uint64_t cached_begin =
          aligned_run_end > kRecent ? aligned_run_end - kRecent : aligned_run_begin;
      if (aligned_run && first >= std::max(aligned_run_begin, cached_begin) &&
          first <= aligned_run_end) {
        const uint64_t duplicate_end = std::min(range_end, aligned_run_end);
        if (first < duplicate_end) {
          if (statistics)
            statistics->dword_duplicates_skipped += duplicate_end - first;
          first = duplicate_end;
        }
        aligned_run_end = std::max(aligned_run_end, range_end);
      } else {
        aligned_run_begin = first;
        aligned_run_end = range_end;
        aligned_run = true;
      }
      if (first == range_end)
        return;
    } else {
      aligned_run = false;
    }
    for (uint64_t index = first; index <= last; ++index) {
      if (statistics)
        ++statistics->dword_candidates;
      const uint8_t mask = word_mask(index << 2, range.address, end);
      if (dedupe) {
        const size_t slot = index & (kRecent - 1);
        if (recent_index[slot] == index && recent_mask[slot] == mask) {
          if (statistics)
            ++statistics->dword_duplicates_skipped;
          continue;
        }
        recent_index[slot] = index;
        recent_mask[slot] = mask;
      }
      if (statistics)
        ++statistics->dword_visits;
      visit(index, mask, range);
    }
  });
}

template <typename Visit, typename Counters = Statistics>
void for_each_word(std::span<const LaneRange> lanes, Visit &&visit,
                   Counters *statistics = nullptr) {
  for_each_word_expanded(lanes, std::forward<Visit>(visit), statistics);
}

/// Aligned affine accesses are represented as one or more dword runs. This
/// avoids rebuilding every lane range and preserves the expanded path's first
/// lane attribution and duplicate accounting exactly. Irregular byte shapes
/// retain the generic expansion above.
template <typename Visit, typename Counters = Statistics>
void for_each_word(const AffineLaneRange &lanes, Visit &&visit, Counters *statistics = nullptr) {
  if (!lanes.count || !lanes.size)
    return;

  const bool aligned =
      ((lanes.address | lanes.stride | static_cast<uint64_t>(lanes.size)) & 3) == 0;
  const bool extent_fits =
      lanes.count == 1 ||
      lanes.stride <= (std::numeric_limits<uint64_t>::max() - lanes.address) / (lanes.count - 1);
  const uint64_t last_address =
      extent_fits ? lanes.address + uint64_t{lanes.count - 1} * lanes.stride : 0;
  const bool ends_fit =
      extent_fits && lanes.size <= std::numeric_limits<uint64_t>::max() - last_address;
  if (!aligned || !ends_fit) {
    for_each_word_expanded(lanes, std::forward<Visit>(visit), statistics);
    return;
  }

  const uint64_t words_per_lane = lanes.size >> 2;
  const uint64_t stride_words = lanes.stride >> 2;
  const uint64_t first_word = lanes.address >> 2;
  // Overlapping ranges larger than the recent cache intentionally retain the
  // expanded path: its bounded dedupe behavior is part of provenance today.
  const bool compact_overlap = stride_words < words_per_lane && words_per_lane <= 64;
  const bool disjoint = stride_words >= words_per_lane;
  if (!compact_overlap && !disjoint) {
    for_each_word_expanded(lanes, std::forward<Visit>(visit), statistics);
    return;
  }

  if (stride_words <= words_per_lane) {
    const uint64_t total_words = uint64_t{lanes.count - 1} * stride_words + words_per_lane;
    if (statistics) {
      statistics->dword_candidates += total_words;
      statistics->dword_visits += total_words;
      statistics->affine_dword_runs += 1;
      statistics->affine_dword_visits += total_words;
      statistics->dword_duplicates_skipped += uint64_t{lanes.count} * words_per_lane - total_words;
    }
    for (uint64_t offset = 0; offset < total_words; ++offset) {
      uint64_t lane_index = 0;
      if (stride_words == words_per_lane) {
        lane_index = offset / words_per_lane;
      } else if (stride_words != 0 && offset >= words_per_lane) {
        lane_index = (offset - words_per_lane + stride_words) / stride_words;
      }
      const LaneRange range{lanes.address + lane_index * lanes.stride, lanes.size,
                            lanes.first_lane + static_cast<int32_t>(lane_index)};
      visit(first_word + offset, 0xf, range);
    }
    return;
  }

  if (statistics) {
    statistics->affine_dword_runs += lanes.count;
    statistics->affine_dword_visits += uint64_t{lanes.count} * words_per_lane;
  }
  for (uint32_t lane_index = 0; lane_index < lanes.count; ++lane_index) {
    const uint64_t lane_word = first_word + uint64_t{lane_index} * stride_words;
    const LaneRange range{lanes.address + uint64_t{lane_index} * lanes.stride, lanes.size,
                          lanes.first_lane + static_cast<int32_t>(lane_index)};
    for (uint64_t offset = 0; offset < words_per_lane; ++offset) {
      if (statistics) {
        ++statistics->dword_candidates;
        ++statistics->dword_visits;
      }
      visit(lane_word + offset, 0xf, range);
    }
  }
}

bool same_site(const Access &a, const Access &b) {
  return a.wave == b.wave && a.pc == b.pc && a.lane == b.lane && a.read == b.read &&
         a.write == b.write && a.atomic == b.atomic;
}

void coalesce_findings(std::vector<Finding> &findings) {
  std::ranges::sort(findings, [](const auto &a, const auto &b) {
    return std::tie(a.space, a.kind, a.producer.pc, a.consumer.pc, a.producer.lane, a.consumer.lane,
                    a.address) < std::tie(b.space, b.kind, b.producer.pc, b.consumer.pc,
                                          b.producer.lane, b.consumer.lane, b.address);
  });
  std::vector<Finding> merged;
  merged.reserve(findings.size());
  for (auto &finding : findings) {
    if (!merged.empty()) {
      auto &prior = merged.back();
      if (prior.space == finding.space && prior.kind == finding.kind &&
          prior.requires_barrier == finding.requires_barrier &&
          same_site(prior.producer, finding.producer) &&
          same_site(prior.consumer, finding.consumer) &&
          prior.address + prior.size == finding.address) {
        prior.size += finding.size;
        continue;
      }
    }
    merged.push_back(std::move(finding));
  }
  findings = std::move(merged);
}

/// One outstanding LDS instruction of a wave. Its lane ranges live in the
/// owning WaveSlot so that issuing an instruction does not allocate.
struct PendingOp {
  uint64_t pc = 0;
  uint64_t lo = 0;
  uint64_t hi = 0;
  uint32_t first = kNoIndex;
  uint32_t count = 0;
  uint32_t ordered_class = 0;
  uint8_t flags = 0;
  uint8_t completion_count = 0;
  std::array<CompletionToken, 2> completions{};
  AffineLaneRange affine{};

  bool is_affine() const { return first == kNoIndex; }
};

bool complete(const PendingOp &op, const MemoryWaitScoreboard &scoreboard) {
  for (uint8_t i = 0; i < op.completion_count; ++i)
    if (!scoreboard.completed(op.completions[i]))
      return false;
  return true;
}

struct WaveSlot {
  WaveKey key;
  bool active = true;
  // Sequence of the wave's newest LDS access; no history record is newer.
  uint64_t last_sequence = 0;
  // The wave's own scoreboard, valid while the wave is active. Barriers use it
  // to observe completions that happened after the wave's last LDS access.
  const MemoryWaitScoreboard *scoreboard = nullptr;
  std::vector<PendingOp> pending{};
  std::vector<LaneRange> ranges{};

  void reset() {
    pending.clear();
    ranges.clear();
    scoreboard = nullptr;
  }

  void prune(const MemoryWaitScoreboard &completions) {
    std::erase_if(pending, [&](const PendingOp &op) { return complete(op, completions); });
    uint32_t next = 0;
    for (PendingOp &op : pending) {
      if (op.is_affine())
        continue;
      if (op.first != next)
        std::copy_n(ranges.begin() + op.first, op.count, ranges.begin() + next);
      op.first = next;
      next += op.count;
    }
    ranges.resize(next);
  }

  Access access(const PendingOp &op) const {
    Access result{.wave = key,
                  .pc = op.pc,
                  .ordered_class = op.ordered_class,
                  .completions = op.completions,
                  .completion_count = op.completion_count};
    set_flags(result, op.flags);
    return result;
  }
};

/// One retained LDS access to one dword.
struct LdsRecord {
  uint64_t pc = 0;
  uint64_t sequence = 0;
  uint32_t next = kNoIndex;
  uint16_t wave = 0;
  int8_t lane = -1;
  uint8_t bits = 0; // access and kInFlight flags in the low nibble, byte mask in the high nibble

  uint8_t flags() const { return bits & 0xf; }
  uint8_t bytes() const { return bits >> 4; }
};
static_assert(sizeof(LdsRecord) == 24);

uint8_t pack(uint8_t flags, uint8_t bytes) { return static_cast<uint8_t>(flags | (bytes << 4)); }

/// Record-list heads for one workgroup's LDS dwords. A workgroup's LDS
/// allocation is contiguous, so a dense window covers it; addresses that would
/// stretch the window too far spill to a map.
class WordIndex {
public:
  uint32_t &head(uint64_t word) {
    if (word - base_ < heads_.size())
      return heads_[word - base_];
    return grow(word);
  }

  void clear() {
    for (const uint64_t word : touched)
      head(word) = kNoIndex;
    touched.clear();
  }

  /// Words whose record list is nonempty.
  std::vector<uint64_t> touched;

private:
  static constexpr uint64_t kChunk = 256;
  static constexpr uint64_t kMaxWindow = uint64_t{1} << 18;

  uint32_t &grow(uint64_t word) {
    const uint64_t chunk = word & ~(kChunk - 1);
    if (heads_.empty()) {
      base_ = chunk;
      heads_.assign(kChunk, kNoIndex);
      return heads_[word - base_];
    }
    // The window only widens, so a word rejected once is never inside it.
    const uint64_t base = std::min(base_, chunk);
    const uint64_t end = std::max(base_ + heads_.size(), chunk + kChunk);
    if (end - base > kMaxWindow)
      return overflow_.try_emplace(word, kNoIndex).first->second;
    if (base < base_) {
      heads_.insert(heads_.begin(), base_ - base, kNoIndex);
      base_ = base;
    }
    heads_.resize(end - base_, kNoIndex);
    return heads_[word - base_];
  }

  uint64_t base_ = 0;
  std::vector<uint32_t> heads_;
  std::unordered_map<uint64_t, uint32_t> overflow_;
};

// Dispatch ids are unique across the SoC and workgroup ids are grid-wide, so
// this pair names one workgroup, as in the CU's per-workgroup maps.
struct WorkgroupKey {
  uint32_t dispatch;
  uint32_t workgroup;
  bool operator==(const WorkgroupKey &) const = default;
};

struct WorkgroupKeyHash {
  size_t operator()(const WorkgroupKey &key) const {
    return mix(uint64_t{key.dispatch} << 32 | key.workgroup);
  }
};

WorkgroupKey workgroup_of(const WaveKey &key) { return {key.dispatch, key.workgroup}; }

/// Global shadow pages cover 256 dwords (1 KiB) of one dispatch's addresses.
constexpr uint32_t kPageShift = 8;
constexpr uint32_t kPageWords = 1u << kPageShift;

/// One retained global access to one dword by one workgroup.
struct GlobalRecord {
  uint32_t next = kNoIndex;
  uint32_t workgroup = 0;
  uint32_t site = kNoIndex;
  uint8_t flags = 0;
  uint8_t bytes = 0;
  uint16_t reserved = 0;
};
static_assert(sizeof(GlobalRecord) == 16);

/// Exact diagnostic provenance interned by a dispatch shadow.
struct GlobalSite {
  uint64_t pc = 0;
  uint32_t wave = 0;
  int8_t lane = -1;
  std::array<uint8_t, 3> reserved{};

  bool operator==(const GlobalSite &) const = default;
};
static_assert(sizeof(GlobalSite) == 16);

struct GlobalPage {
  GlobalPage() { heads.fill(kNoIndex); }
  std::array<uint32_t, kPageWords> heads;
};

struct PageKey {
  uint32_t dispatch;
  uint64_t page;
  bool operator==(const PageKey &) const = default;
};

struct PageKeyHash {
  size_t operator()(const PageKey &key) const { return mix(key.page ^ mix(key.dispatch)); }
};

} // namespace

struct MemoryRaceChecker::WorkgroupState {
  std::mutex mutex;
  std::vector<WaveSlot> waves;
  /// Vector clocks, row-major by observing wave: clocks[observer * n + source].
  std::vector<uint64_t> clocks;
  WordIndex words;
  std::vector<LdsRecord> records;
  uint32_t free_record = kNoIndex;
  // Keep optional P0 storage after the checker-hot fields so disabling
  // statistics preserves their pre-instrumentation layout.
  std::unique_ptr<Statistics> statistics;

  uint32_t find_wave(const WaveKey &key, Statistics *stats = nullptr) const {
    for (uint32_t i = 0; i < waves.size(); ++i) {
      if (stats)
        ++stats->wave_search_steps;
      if (waves[i].key.wave == key.wave && waves[i].key.generation == key.generation)
        return i;
    }
    return kNoIndex;
  }

  uint32_t wave_index(const WaveKey &key, Statistics *stats = nullptr) {
    if (const uint32_t found = find_wave(key, stats); found != kNoIndex)
      return found;
    const size_t n = waves.size();
    if (stats)
      ++stats->clock_matrix_growths;
    std::vector<uint64_t> grown((n + 1) * (n + 1), 0);
    for (size_t observer = 0; observer < n; ++observer)
      std::copy_n(clocks.begin() + observer * n, n, grown.begin() + observer * (n + 1));
    clocks = std::move(grown);
    waves.push_back({.key = key});
    return static_cast<uint32_t>(n);
  }

  void begin_waves(std::span<const WaveKey> members, std::vector<uint32_t> &indices,
                   Statistics *stats) {
    if (waves.empty()) {
      if (stats && !members.empty())
        ++stats->clock_matrix_growths;
      waves.reserve(members.size());
      clocks.assign(members.size() * members.size(), 0);
      for (const WaveKey &key : members) {
        indices.push_back(static_cast<uint32_t>(waves.size()));
        waves.push_back({.key = key});
      }
      return;
    }
    for (const WaveKey &key : members)
      indices.push_back(wave_index(key, stats));
  }

  uint64_t &clock(size_t observer, size_t source) {
    return clocks[observer * waves.size() + source];
  }

  uint32_t allocate(const LdsRecord &record, Statistics *stats = nullptr) {
    if (free_record == kNoIndex) {
      if (stats && records.size() == records.capacity())
        ++stats->record_vector_growths;
      records.push_back(record);
      return static_cast<uint32_t>(records.size() - 1);
    }
    const uint32_t index = free_record;
    free_record = records[index].next;
    records[index] = record;
    return index;
  }

  void release(uint32_t index) {
    records[index].next = free_record;
    free_record = index;
  }

  void clear_history(Statistics *stats = nullptr) {
    if (stats) {
      stats->cleanup_dwords += words.touched.size();
      stats->cleanup_records += records.size();
    }
    words.clear();
    records.clear();
    free_record = kNoIndex;
  }

  /// Drop records visible to every active wave. Ended waves never access LDS
  /// again, so they do not keep history alive.
  void prune_history(std::vector<uint64_t> &floor, Statistics *stats = nullptr) {
    const size_t n = waves.size();
    floor.assign(n, std::numeric_limits<uint64_t>::max());
    for (size_t observer = 0; observer < n; ++observer)
      if (waves[observer].active)
        for (size_t source = 0; source < n; ++source)
          floor[source] = std::min(floor[source], clock(observer, source));
    size_t kept = 0;
    for (const uint64_t word : words.touched) {
      uint32_t *link = &words.head(word);
      while (*link != kNoIndex) {
        if (stats)
          ++stats->lds_records_visited;
        LdsRecord &record = records[*link];
        if (floor[record.wave] >= record.sequence) {
          const uint32_t dead = *link;
          *link = record.next;
          release(dead);
        } else {
          link = &record.next;
        }
      }
      if (words.head(word) != kNoIndex)
        words.touched[kept++] = word;
    }
    words.touched.resize(kept);
  }

  Finding history_finding(const LdsRecord &record, const Access &instruction,
                          const LaneRange &range, uint8_t flags, uint64_t word,
                          uint8_t overlap) const {
    const uint32_t start = std::countr_zero(overlap);
    const uint32_t length = std::countr_one(static_cast<uint8_t>(overlap >> start));
    Access producer{.wave = waves[record.wave].key,
                    .pc = record.pc,
                    .address = word + start,
                    .size = length,
                    .lane = record.lane};
    set_flags(producer, record.flags());
    return {Space::LDS,
            kind_of(record.flags(), flags),
            producer,
            consumer_of(instruction, range),
            word + start,
            length,
            !(record.flags() & kInFlight)};
  }

  /// Check `self`'s new access against other waves' unsynchronized history,
  /// then record it as the newest access of its kind by `self`.
  template <bool Collect, typename Ranges>
  void record(uint32_t self, const Access &instruction, uint8_t flags, const Ranges &lanes,
              std::vector<Finding> &findings, Statistics *stats) {
    const uint64_t sequence = ++clock(self, self);
    waves[self].last_sequence = sequence;
    const uint64_t *seen = clocks.data() + size_t{self} * waves.size();
    for_each_word(
        lanes,
        [&](uint64_t index, uint8_t mask, const LaneRange &range) {
          uint32_t *link = &words.head(index);
          const bool was_empty = *link == kNoIndex;
          if constexpr (Collect) {
            if (was_empty)
              ++stats->empty_record_heads;
            else
              ++stats->nonempty_record_heads;
          }
          bool kept = false;
          while (*link != kNoIndex) {
            if constexpr (Collect)
              ++stats->lds_records_visited;
            LdsRecord &prior = records[*link];
            if (prior.wave == self) {
              // A newer same-wave access with the same flags and a covering mask
              // subsumes an older one: a wave not ordered after the older access
              // cannot be ordered after the newer one.
              if (prior.flags() == flags && (prior.bytes() & ~mask) == 0) {
                if (kept) {
                  const uint32_t dead = *link;
                  *link = prior.next;
                  release(dead);
                  continue;
                }
                prior.pc = instruction.pc;
                prior.sequence = sequence;
                prior.lane = compact_lane(range.lane);
                prior.bits = pack(flags, mask);
                kept = true;
                if constexpr (Collect)
                  ++stats->records_replaced;
              }
              link = &prior.next;
              continue;
            }
            const uint8_t overlap = prior.bytes() & mask;
            if (overlap && conflicting(prior.flags(), flags) && seen[prior.wave] < prior.sequence)
              findings.push_back(
                  history_finding(prior, instruction, range, flags, index << 2, overlap));
            link = &prior.next;
          }
          if (kept)
            return;
          const uint32_t added =
              allocate({instruction.pc, sequence, kNoIndex, static_cast<uint16_t>(self),
                        compact_lane(range.lane), pack(flags, mask)},
                       Collect ? stats : nullptr);
          if constexpr (Collect)
            ++stats->records_inserted;
          uint32_t &head = words.head(index);
          records[added].next = head;
          head = added;
          if (was_empty)
            words.touched.push_back(index);
        },
        Collect ? stats : nullptr);
  }

  /// Re-record `self`'s in-flight operation at `sequence`, which the other
  /// barrier members have not observed. It stays unordered with their later
  /// accesses until a barrier that follows its completion.
  template <bool Collect>
  void carry(uint32_t self, const WaveSlot &wave, const PendingOp &op, uint64_t sequence,
             Statistics *stats) {
    const uint8_t flags = op.flags | kInFlight;
    auto carry_ranges = [&](const auto &lanes) {
      for_each_word(
          lanes,
          [&](uint64_t index, uint8_t mask, const LaneRange &range) {
            if constexpr (Collect) {
              if (words.head(index) == kNoIndex)
                ++stats->empty_record_heads;
              else
                ++stats->nonempty_record_heads;
            }
            for (uint32_t i = words.head(index); i != kNoIndex; i = records[i].next) {
              if constexpr (Collect)
                ++stats->lds_records_visited;
              LdsRecord &prior = records[i];
              if (prior.wave == self && (prior.flags() & ~kInFlight) == op.flags &&
                  (prior.bytes() & ~mask) == 0) {
                prior.pc = op.pc;
                prior.sequence = sequence;
                prior.lane = compact_lane(range.lane);
                prior.bits = pack(flags, mask);
                if constexpr (Collect)
                  ++stats->records_replaced;
                return;
              }
            }
            const bool was_empty = words.head(index) == kNoIndex;
            const uint32_t added = allocate({op.pc, sequence, kNoIndex, static_cast<uint16_t>(self),
                                             compact_lane(range.lane), pack(flags, mask)},
                                            Collect ? stats : nullptr);
            if constexpr (Collect)
              ++stats->records_inserted;
            uint32_t &head = words.head(index);
            records[added].next = head;
            head = added;
            if (was_empty)
              words.touched.push_back(index);
          },
          Collect ? stats : nullptr);
    };
    if (op.is_affine())
      carry_ranges(op.affine);
    else
      carry_ranges(std::span(wave.ranges).subspan(op.first, op.count));
  }

  template <bool Collect>
  void barrier(std::span<const WaveKey> members, std::vector<uint32_t> &indices,
               std::vector<uint64_t> &merged, std::vector<uint8_t> &joined, Statistics *stats) {
    if constexpr (Collect) {
      ++stats->barrier_calls;
      stats->barrier_members += members.size();
      ++stats->barrier_scratch_vectors;
    }
    indices.clear();
    for (const WaveKey &key : members) {
      const uint32_t index = find_wave(key, Collect ? stats : nullptr);
      if (index == kNoIndex)
        continue;
      indices.push_back(index);
      WaveSlot &wave = waves[index];
      if (wave.scoreboard)
        wave.prune(*wave.scoreboard);
    }
    if (indices.empty())
      return;
    const size_t n = waves.size();
    if constexpr (Collect)
      stats->barrier_scratch_vectors += 2;
    merged.assign(n, 0);
    for (const uint32_t member : indices)
      for (size_t source = 0; source < n; ++source) {
        if constexpr (Collect)
          ++stats->barrier_clock_cells;
        merged[source] = std::max(merged[source], clock(member, source));
      }
    joined.assign(n, 0);
    for (const uint32_t member : indices) {
      if (joined[member])
        continue;
      joined[member] = 1;
      for (size_t source = 0; source < n; ++source) {
        if constexpr (Collect)
          ++stats->barrier_clock_cells;
        clock(member, source) = std::max(clock(member, source), merged[source]);
      }
      ++clock(member, member);
    }
    // History is fully visible when every active wave joined and every ended
    // wave's last access was already ordered before the barrier. Records still
    // needed by a nonparticipant survive named barriers.
    bool visible = true;
    for (size_t wave = 0; wave < n && visible; ++wave)
      visible = joined[wave] || (!waves[wave].active && merged[wave] >= waves[wave].last_sequence);
    if (visible)
      clear_history(Collect ? stats : nullptr);
    else
      prune_history(merged, Collect ? stats : nullptr);
    // A barrier orders only completed operations; multi-buffered async copies
    // legitimately stay in flight across the barrier that guards another buffer.
    for (size_t wave = 0; wave < n; ++wave) {
      WaveSlot &slot = waves[wave];
      if (!joined[wave] || slot.pending.empty())
        continue;
      const uint64_t sequence = clock(wave, wave);
      slot.last_sequence = sequence;
      for (const PendingOp &op : slot.pending) {
        if constexpr (Collect)
          ++stats->barrier_carried_ops;
        carry<Collect>(static_cast<uint32_t>(wave), slot, op, sequence, stats);
      }
    }
  }
};

struct MemoryRaceChecker::WorkgroupShard {
  std::mutex mutex;
  std::unordered_map<WorkgroupKey, std::shared_ptr<WorkgroupState>, WorkgroupKeyHash> states;
};

struct MemoryRaceChecker::GlobalBudget {
  explicit GlobalBudget(uint64_t initial) : limit(initial) {}

  bool reserve(uint64_t bytes) {
    uint64_t observed = current.load(std::memory_order_relaxed);
    for (;;) {
      const uint64_t maximum = limit.load(std::memory_order_relaxed);
      if (observed > maximum || bytes > maximum - observed)
        return false;
      if (current.compare_exchange_weak(observed, observed + bytes, std::memory_order_acq_rel,
                                        std::memory_order_relaxed)) {
        uint64_t high = peak.load(std::memory_order_relaxed);
        while (high < observed + bytes &&
               !peak.compare_exchange_weak(high, observed + bytes, std::memory_order_relaxed))
          ;
        epoch.fetch_add(1, std::memory_order_release);
        return true;
      }
    }
  }

  void release(uint64_t bytes) {
    if (!bytes)
      return;
    [[maybe_unused]] const uint64_t before = current.fetch_sub(bytes, std::memory_order_acq_rel);
    assert(before >= bytes);
    epoch.fetch_add(1, std::memory_order_release);
  }

  void set_limit(uint64_t bytes) {
    limit.store(bytes, std::memory_order_relaxed);
    epoch.fetch_add(1, std::memory_order_release);
  }

  std::atomic<uint64_t> limit;
  std::atomic<uint64_t> current{0};
  std::atomic<uint64_t> peak{0};
  std::atomic<uint64_t> epoch{1};
};

struct MemoryRaceChecker::GlobalShard {
  static constexpr uint32_t kPageSlabEntries = 16;
  static constexpr uint32_t kRecordSlabEntries = 1024;
  static constexpr uint32_t kSiteSlabEntries = 256;
  static constexpr uint32_t kInitialDirectoryEntries = 16;
  static constexpr size_t kSaturationEntries = 64;

  template <typename T, uint32_t Entries> struct SlabStore {
    struct Slab {
      std::unique_ptr<T[]> data;
      uint32_t size = 0;
      uint32_t capacity = 0;
    };

    std::vector<Slab> slabs;
    uint64_t entries = 0;

    bool has_space() const { return !slabs.empty() && slabs.back().size < slabs.back().capacity; }

    bool ensure(GlobalBudget &budget, uint64_t &charged, Statistics *stats,
                uint64_t Statistics::*allocation_counter) {
      if (has_space())
        return true;
      uint32_t capacity = Entries;
      const uint64_t maximum = budget.limit.load(std::memory_order_relaxed);
      const uint64_t used = budget.current.load(std::memory_order_relaxed);
      const uint64_t available = used < maximum ? maximum - used : 0;
      capacity = std::min<uint64_t>(capacity, available / sizeof(T));
      if (!capacity)
        capacity = 1;
      for (; capacity; capacity /= 2) {
        const uint64_t bytes = uint64_t{capacity} * sizeof(T);
        if (stats)
          ++stats->budget_reservation_attempts;
        if (!budget.reserve(bytes)) {
          if (stats)
            ++stats->budget_reservation_failures;
          continue;
        }
        try {
          auto data = std::make_unique<T[]>(capacity);
          slabs.push_back({std::move(data), 0, capacity});
        } catch (...) {
          budget.release(bytes);
          throw;
        }
        charged += bytes;
        if (stats) {
          ++(stats->*allocation_counter);
          stats->retained_bytes += bytes;
        }
        return true;
      }
      if (stats)
        stats->rejected_bytes += sizeof(T);
      return false;
    }

    uint32_t append(T value) {
      assert(!slabs.empty() && slabs.back().size < slabs.back().capacity);
      Slab &slab = slabs.back();
      const uint64_t index = uint64_t{slabs.size() - 1} * Entries + slab.size;
      if (index >= kNoIndex)
        return kNoIndex;
      slab.data[slab.size++] = std::move(value);
      ++entries;
      return static_cast<uint32_t>(index);
    }

    T &operator[](uint32_t index) {
      Slab &slab = slabs[index / Entries];
      assert(index % Entries < slab.size);
      return slab.data[index % Entries];
    }

    const T &operator[](uint32_t index) const {
      const Slab &slab = slabs[index / Entries];
      assert(index % Entries < slab.size);
      return slab.data[index % Entries];
    }
  };

  struct PageSlot {
    uint64_t key = 0;
    uint32_t page = kNoIndex;
    uint32_t reserved = 0;
  };
  static_assert(sizeof(PageSlot) == 16);

  struct SiteSlot {
    uint32_t hash = 0;
    uint32_t site = kNoIndex;
  };
  static_assert(sizeof(SiteSlot) == 8);

  struct DispatchShadow {
    explicit DispatchShadow(uint32_t id) : dispatch(id) {}

    uint32_t dispatch;
    uint64_t charged = 0;
    std::unique_ptr<PageSlot[]> page_directory;
    uint32_t page_directory_capacity = 0;
    uint32_t page_directory_size = 0;
    std::unique_ptr<SiteSlot[]> site_directory;
    uint32_t site_directory_capacity = 0;
    uint32_t site_directory_size = 0;
    GlobalSite recent_site;
    uint32_t recent_site_id = kNoIndex;
    SlabStore<GlobalPage, kPageSlabEntries> pages;
    SlabStore<GlobalRecord, kRecordSlabEntries> records;
    SlabStore<GlobalSite, kSiteSlabEntries> sites;

    bool site_insert_requires_growth() const {
      return !site_directory_capacity ||
             site_directory_size + 1 > site_directory_capacity * 3 / 4 || !sites.has_space();
    }

    bool page_insert_requires_growth() const {
      return !page_directory_capacity ||
             page_directory_size + 1 > page_directory_capacity * 3 / 4 || !pages.has_space();
    }

    template <typename Slot>
    bool allocate_directory(std::unique_ptr<Slot[]> &directory, uint32_t &capacity, uint32_t &size,
                            GlobalBudget &budget, Statistics *stats,
                            uint64_t Statistics::*growth_counter, const auto &rehash) {
      const uint32_t next =
          capacity ? (capacity > (std::numeric_limits<uint32_t>::max() / 2) ? 0 : capacity * 2)
                   : kInitialDirectoryEntries;
      if (!next)
        return false;
      const uint64_t bytes = uint64_t{next} * sizeof(Slot);
      if (stats)
        ++stats->budget_reservation_attempts;
      if (!budget.reserve(bytes)) {
        if (stats) {
          ++stats->budget_reservation_failures;
          stats->rejected_bytes += sizeof(Slot);
        }
        return false;
      }
      std::unique_ptr<Slot[]> grown;
      try {
        grown = std::make_unique<Slot[]>(next);
      } catch (...) {
        budget.release(bytes);
        throw;
      }
      rehash(grown.get(), next);
      const uint64_t old_bytes = uint64_t{capacity} * sizeof(Slot);
      directory.reset();
      if (old_bytes) {
        charged -= old_bytes;
        if (stats)
          stats->budget_bytes_released += old_bytes;
        budget.release(old_bytes);
      }
      directory = std::move(grown);
      capacity = next;
      charged += bytes;
      if (stats) {
        ++(stats->*growth_counter);
        stats->retained_bytes += bytes;
      }
      (void)size;
      return true;
    }

    bool ensure_page_directory(GlobalBudget &budget, Statistics *stats) {
      if (page_directory_capacity && page_directory_size + 1 <= page_directory_capacity * 3 / 4)
        return true;
      auto rehash = [&](PageSlot *grown, uint32_t capacity) {
        for (uint32_t i = 0; i < page_directory_capacity; ++i) {
          const PageSlot &old = page_directory[i];
          if (old.page == kNoIndex)
            continue;
          uint32_t slot = static_cast<uint32_t>(mix(old.key)) & (capacity - 1);
          while (grown[slot].page != kNoIndex)
            slot = (slot + 1) & (capacity - 1);
          grown[slot] = old;
        }
      };
      return allocate_directory(page_directory, page_directory_capacity, page_directory_size,
                                budget, stats, &Statistics::page_directory_growths, rehash);
    }

    GlobalPage *find_page(uint64_t key) {
      if (!page_directory_capacity)
        return nullptr;
      uint32_t slot = static_cast<uint32_t>(mix(key)) & (page_directory_capacity - 1);
      for (uint32_t probes = 0; probes < page_directory_capacity; ++probes) {
        const PageSlot &entry = page_directory[slot];
        if (entry.page == kNoIndex)
          return nullptr;
        if (entry.key == key)
          return &pages[entry.page];
        slot = (slot + 1) & (page_directory_capacity - 1);
      }
      return nullptr;
    }

    GlobalPage *create_page(uint64_t key, GlobalBudget &budget, Statistics *stats) {
      if (!ensure_page_directory(budget, stats))
        return nullptr;
      if (!pages.ensure(budget, charged, stats, &Statistics::page_slab_allocations))
        return nullptr;
      const uint32_t page = pages.append(GlobalPage{});
      if (page == kNoIndex)
        return nullptr;
      uint32_t slot = static_cast<uint32_t>(mix(key)) & (page_directory_capacity - 1);
      while (page_directory[slot].page != kNoIndex)
        slot = (slot + 1) & (page_directory_capacity - 1);
      page_directory[slot] = {.key = key, .page = page};
      ++page_directory_size;
      if (stats)
        ++stats->shadow_pages_created;
      return &pages[page];
    }

    static uint32_t site_hash(const GlobalSite &site) {
      return static_cast<uint32_t>(
          mix(site.pc ^ mix((uint64_t{site.wave} << 8) | static_cast<uint8_t>(site.lane))));
    }

    uint32_t find_site(const GlobalSite &site, uint32_t &hash, uint32_t &empty_slot,
                       Statistics *stats) {
      if (stats)
        ++stats->site_lookups;
      if (recent_site_id != kNoIndex && recent_site == site)
        return recent_site_id;
      if (!site_directory_capacity)
        return kNoIndex;
      hash = site_hash(site);
      uint32_t slot = static_cast<uint32_t>(hash) & (site_directory_capacity - 1);
      for (uint32_t probes = 0; probes < site_directory_capacity; ++probes) {
        const SiteSlot &entry = site_directory[slot];
        if (entry.site == kNoIndex) {
          empty_slot = slot;
          return kNoIndex;
        }
        if (entry.hash == hash && sites[entry.site] == site) {
          recent_site = site;
          recent_site_id = entry.site;
          return entry.site;
        }
        slot = (slot + 1) & (site_directory_capacity - 1);
      }
      return kNoIndex;
    }

    bool ensure_site_directory(GlobalBudget &budget, Statistics *stats) {
      if (site_directory_capacity && site_directory_size + 1 <= site_directory_capacity * 3 / 4)
        return true;
      auto rehash = [&](SiteSlot *grown, uint32_t capacity) {
        for (uint32_t i = 0; i < site_directory_capacity; ++i) {
          const SiteSlot &old = site_directory[i];
          if (old.site == kNoIndex)
            continue;
          uint32_t slot = static_cast<uint32_t>(old.hash) & (capacity - 1);
          while (grown[slot].site != kNoIndex)
            slot = (slot + 1) & (capacity - 1);
          grown[slot] = old;
        }
      };
      return allocate_directory(site_directory, site_directory_capacity, site_directory_size,
                                budget, stats, &Statistics::site_directory_growths, rehash);
    }

    uint32_t insert_site(const GlobalSite &site, uint32_t hash, uint32_t empty_slot,
                         GlobalBudget &budget, Statistics *stats) {
      const bool directory_growth =
          !site_directory_capacity || site_directory_size + 1 > site_directory_capacity * 3 / 4;
      if (!ensure_site_directory(budget, stats) ||
          !sites.ensure(budget, charged, stats, &Statistics::site_slab_allocations))
        return kNoIndex;
      const uint32_t index = sites.append(site);
      if (index == kNoIndex)
        return kNoIndex;
      uint32_t slot = empty_slot;
      if (directory_growth || slot == kNoIndex) {
        slot = static_cast<uint32_t>(hash) & (site_directory_capacity - 1);
        while (site_directory[slot].site != kNoIndex)
          slot = (slot + 1) & (site_directory_capacity - 1);
      }
      site_directory[slot] = {.hash = hash, .site = index};
      ++site_directory_size;
      recent_site = site;
      recent_site_id = index;
      if (stats)
        ++stats->sites_interned;
      return index;
    }
  };

  struct SaturationEntry {
    uint64_t page = 0;
    uint64_t epoch = 0;
    uint32_t dispatch = 0;
    bool occupied = false;
  };

  std::mutex mutex;
  std::vector<std::unique_ptr<DispatchShadow>> dispatches;
  std::array<SaturationEntry, kSaturationEntries> saturation{};

  DispatchShadow *find_dispatch(uint32_t dispatch) {
    const auto it =
        std::ranges::find(dispatches, dispatch, [](const auto &state) { return state->dispatch; });
    return it == dispatches.end() ? nullptr : it->get();
  }

  DispatchShadow &create_dispatch(uint32_t dispatch) {
    dispatches.push_back(std::make_unique<DispatchShadow>(dispatch));
    return *dispatches.back();
  }

  bool saturated(uint32_t dispatch, uint64_t page, const GlobalBudget &budget, Statistics *stats) {
    SaturationEntry &entry = saturation[PageKeyHash{}({dispatch, page}) & (kSaturationEntries - 1)];
    if (!entry.occupied || entry.dispatch != dispatch || entry.page != page)
      return false;
    const uint64_t current_epoch = budget.epoch.load(std::memory_order_acquire);
    if (entry.epoch == current_epoch) {
      if (stats)
        ++stats->saturation_cache_hits;
      return true;
    }
    entry.occupied = false;
    if (stats)
      ++stats->saturation_cache_invalidations;
    return false;
  }

  void mark_saturated(uint32_t dispatch, uint64_t page, const GlobalBudget &budget,
                      Statistics *stats) {
    saturation[PageKeyHash{}({dispatch, page}) & (kSaturationEntries - 1)] = {
        .page = page,
        .epoch = budget.epoch.load(std::memory_order_acquire),
        .dispatch = dispatch,
        .occupied = true};
    if (stats)
      ++stats->saturation_cache_inserts;
  }

  std::unique_ptr<DispatchShadow> take_dispatch(uint32_t dispatch, Statistics *stats) {
    for (SaturationEntry &entry : saturation) {
      if (entry.occupied && entry.dispatch == dispatch) {
        entry.occupied = false;
        if (stats)
          ++stats->saturation_cache_invalidations;
      }
    }
    const auto it =
        std::ranges::find(dispatches, dispatch, [](const auto &state) { return state->dispatch; });
    if (it == dispatches.end())
      return {};
    std::unique_ptr<DispatchShadow> released = std::move(*it);
    dispatches.erase(it);
    return released;
  }
};

struct MemoryRaceChecker::StatisticsState {
  // This mutex protects only completed_workgroups. Each map in the arrays is
  // protected by the correspondingly indexed workgroup/global shard mutex so
  // hot-path counter updates do not contend on one checker-wide lock.
  std::mutex mutex;
  std::unordered_map<uint32_t, Statistics> completed_workgroups;
  std::array<std::unordered_map<uint32_t, Statistics>, kWorkgroupShards> workgroup_shards;
  std::array<std::unordered_map<uint32_t, Statistics>, kGlobalShards> global_shards;
};

namespace {

std::string describe(const Access &access) {
  return std::format("{} at pc={:#x} wg={} wave={}{}",
                     access.atomic  ? "atomic"
                     : access.write ? "write"
                                    : "read",
                     access.pc, access.wave.workgroup, access.wave.wave,
                     access.lane < 0 ? std::string() : std::format(" lane={}", access.lane));
}

void log_finding(const Finding &finding) {
  const char *kind = finding.kind == Kind::RAW ? "RAW" : finding.kind == Kind::WAR ? "WAR" : "WAW";
  const char *remedy = finding.space == Space::Global
                           ? "Cross-workgroup synchronization or an atomic access is required"
                       : finding.requires_barrier ? "A barrier between the waves is required"
                                                  : "A completion wait is required";
  util::Logger::warn(std::format(
      "memory-race: dispatch={} {} {} on bytes [{:#x},{:#x}): {} is unordered with {}. {}; set "
      "top-level race_checking to \"off\" (rocjitsu --race-checking=off) to disable race "
      "checking.",
      finding.consumer.wave.dispatch, finding.space == Space::LDS ? "LDS" : "global", kind,
      finding.address, finding.address + finding.size, describe(finding.consumer),
      describe(finding.producer), remedy));
}

} // namespace

void MemoryRaceChecker::Statistics::add(const Statistics &other) {
  lds_accesses += other.lds_accesses;
  global_accesses += other.global_accesses;
  lane_ranges += other.lane_ranges;
  dword_candidates += other.dword_candidates;
  dword_visits += other.dword_visits;
  dword_duplicates_skipped += other.dword_duplicates_skipped;
  workgroup_lookups += other.workgroup_lookups;
  workgroup_creations += other.workgroup_creations;
  wave_search_steps += other.wave_search_steps;
  workgroup_map_lock_acquisitions += other.workgroup_map_lock_acquisitions;
  workgroup_map_lock_wait_ns += other.workgroup_map_lock_wait_ns;
  workgroup_state_lock_acquisitions += other.workgroup_state_lock_acquisitions;
  workgroup_state_lock_wait_ns += other.workgroup_state_lock_wait_ns;
  global_shard_lock_acquisitions += other.global_shard_lock_acquisitions;
  global_shard_lock_wait_ns += other.global_shard_lock_wait_ns;
  shadow_page_lookups += other.shadow_page_lookups;
  shadow_pages_created += other.shadow_pages_created;
  shadow_pages_rejected += other.shadow_pages_rejected;
  empty_record_heads += other.empty_record_heads;
  nonempty_record_heads += other.nonempty_record_heads;
  lds_records_visited += other.lds_records_visited;
  global_records_visited += other.global_records_visited;
  records_inserted += other.records_inserted;
  records_replaced += other.records_replaced;
  records_rejected += other.records_rejected;
  record_vector_growths += other.record_vector_growths;
  clock_matrix_growths += other.clock_matrix_growths;
  pending_vector_growths += other.pending_vector_growths;
  range_vector_growths += other.range_vector_growths;
  pending_candidates += other.pending_candidates;
  barrier_calls += other.barrier_calls;
  barrier_members += other.barrier_members;
  barrier_clock_cells += other.barrier_clock_cells;
  barrier_carried_ops += other.barrier_carried_ops;
  barrier_scratch_vectors += other.barrier_scratch_vectors;
  cleanup_workgroups += other.cleanup_workgroups;
  cleanup_dwords += other.cleanup_dwords;
  cleanup_pages += other.cleanup_pages;
  cleanup_records += other.cleanup_records;
  retained_bytes += other.retained_bytes;
  rejected_bytes += other.rejected_bytes;
  budget_reservation_attempts += other.budget_reservation_attempts;
  budget_reservation_failures += other.budget_reservation_failures;
  budget_bytes_released += other.budget_bytes_released;
  page_directory_growths += other.page_directory_growths;
  page_slab_allocations += other.page_slab_allocations;
  record_slab_allocations += other.record_slab_allocations;
  site_lookups += other.site_lookups;
  sites_interned += other.sites_interned;
  sites_reused += other.sites_reused;
  site_directory_growths += other.site_directory_growths;
  site_slab_allocations += other.site_slab_allocations;
  saturation_cache_hits += other.saturation_cache_hits;
  saturation_cache_inserts += other.saturation_cache_inserts;
  saturation_cache_invalidations += other.saturation_cache_invalidations;
  affine_accesses += other.affine_accesses;
  affine_lane_ranges += other.affine_lane_ranges;
  affine_dword_runs += other.affine_dword_runs;
  affine_dword_visits += other.affine_dword_visits;
  affine_pending_ops += other.affine_pending_ops;
  explicit_accesses += other.explicit_accesses;
}

bool MemoryRaceChecker::Statistics::empty() const {
  return lds_accesses == 0 && global_accesses == 0 && lane_ranges == 0 && dword_candidates == 0 &&
         dword_visits == 0 && dword_duplicates_skipped == 0 && workgroup_lookups == 0 &&
         workgroup_creations == 0 && wave_search_steps == 0 &&
         workgroup_map_lock_acquisitions == 0 && workgroup_map_lock_wait_ns == 0 &&
         workgroup_state_lock_acquisitions == 0 && workgroup_state_lock_wait_ns == 0 &&
         global_shard_lock_acquisitions == 0 && global_shard_lock_wait_ns == 0 &&
         shadow_page_lookups == 0 && shadow_pages_created == 0 && shadow_pages_rejected == 0 &&
         empty_record_heads == 0 && nonempty_record_heads == 0 && lds_records_visited == 0 &&
         global_records_visited == 0 && records_inserted == 0 && records_replaced == 0 &&
         records_rejected == 0 && record_vector_growths == 0 && clock_matrix_growths == 0 &&
         pending_vector_growths == 0 && range_vector_growths == 0 && pending_candidates == 0 &&
         barrier_calls == 0 && barrier_members == 0 && barrier_clock_cells == 0 &&
         barrier_carried_ops == 0 && barrier_scratch_vectors == 0 && cleanup_workgroups == 0 &&
         cleanup_dwords == 0 && cleanup_pages == 0 && cleanup_records == 0 && retained_bytes == 0 &&
         rejected_bytes == 0 && budget_reservation_attempts == 0 &&
         budget_reservation_failures == 0 && budget_bytes_released == 0 &&
         page_directory_growths == 0 && page_slab_allocations == 0 &&
         record_slab_allocations == 0 && site_lookups == 0 && sites_interned == 0 &&
         sites_reused == 0 && site_directory_growths == 0 && site_slab_allocations == 0 &&
         saturation_cache_hits == 0 && saturation_cache_inserts == 0 &&
         saturation_cache_invalidations == 0 && affine_accesses == 0 && affine_lane_ranges == 0 &&
         affine_dword_runs == 0 && affine_dword_visits == 0 && affine_pending_ops == 0 &&
         explicit_accesses == 0;
}

MemoryRaceChecker::MemoryRaceChecker(Reporter reporter)
    : workgroups_(std::make_unique<WorkgroupShard[]>(kWorkgroupShards)),
      global_budget_(std::make_unique<GlobalBudget>(kDefaultGlobalShadowBytes)),
      global_(std::make_unique<GlobalShard[]>(kGlobalShards)),
      statistics_(std::make_unique<StatisticsState>()),
      reporter_(
          std::make_shared<Reporter>(reporter ? std::move(reporter) : Reporter(log_finding))) {
  // The finding set is bounded. Reserving it before publication keeps allocator
  // work out of the report critical section.
  reported_.reserve(kMaxFindings);
}

MemoryRaceChecker::~MemoryRaceChecker() = default;

void MemoryRaceChecker::set_reporter(Reporter reporter) {
  auto replacement = std::make_shared<const Reporter>(std::move(reporter));
  std::shared_ptr<const Reporter> retired;
  {
    std::lock_guard lock(report_mutex_);
    retired = std::exchange(reporter_, std::move(replacement));
  }
}

void MemoryRaceChecker::set_report(std::shared_ptr<MemoryDiagnosticReport> report) {
  const auto attached = report;
  std::shared_ptr<MemoryDiagnosticReport> retired;
  {
    std::lock_guard lock(report_mutex_);
    retired = std::exchange(report_, std::move(report));
    statistics_enabled_.store(report_ != nullptr, std::memory_order_relaxed);
  }
  if (attached && coverage_degraded_.load(std::memory_order_relaxed))
    attached->set_coverage_degraded();
}

void MemoryRaceChecker::set_global_shadow_budget(uint64_t bytes) {
  global_budget_->set_limit(bytes);
}

uint64_t MemoryRaceChecker::global_shadow_budget() const {
  return global_budget_->limit.load(std::memory_order_relaxed);
}

size_t MemoryRaceChecker::FindingKeyHash::operator()(const FindingKey &key) const {
  size_t seed = 0;
  hash_combine(seed, static_cast<unsigned>(key.space));
  hash_combine(seed, static_cast<unsigned>(key.kind));
  hash_combine(seed, key.dispatch);
  hash_combine(seed, key.producer_pc);
  hash_combine(seed, key.consumer_pc);
  hash_combine(seed, key.address);
  return seed;
}

bool MemoryRaceChecker::WaveBinding::valid_for(const WaveKey &key) const {
  return state_ != nullptr && index_ != kNoIndex && key_ == key;
}

std::shared_ptr<MemoryRaceChecker::WorkgroupState>
MemoryRaceChecker::find_workgroup(const WaveKey &key, bool create) {
  const WorkgroupKey id = workgroup_of(key);
  const size_t shard_index = WorkgroupKeyHash{}(id) % kWorkgroupShards;
  auto &shard = workgroups_[shard_index];
  const bool collect = statistics_enabled_.load(std::memory_order_relaxed);
  StatisticsClock::time_point started;
  if (collect)
    started = StatisticsClock::now();
  std::lock_guard lock(shard.mutex);
  const uint64_t lock_wait_ns = collect ? elapsed_ns(started) : 0;
  Statistics *stats = nullptr;
  if (collect) {
    stats = &statistics_->workgroup_shards[shard_index][key.dispatch];
    ++stats->workgroup_map_lock_acquisitions;
    stats->workgroup_map_lock_wait_ns += lock_wait_ns;
    ++stats->workgroup_lookups;
  }
  if (const auto it = shard.states.find(id); it != shard.states.end())
    return it->second;
  if (!create)
    return nullptr;
  if (!tracking_.load(std::memory_order_relaxed))
    tracking_.store(true, std::memory_order_relaxed);
  auto state = std::make_shared<WorkgroupState>();
  shard.states.emplace(id, state);
  if (stats)
    ++stats->workgroup_creations;
  return state;
}

std::shared_ptr<MemoryRaceChecker::WorkgroupState>
MemoryRaceChecker::resolve_workgroup(const WaveKey &key, const WaveBinding *binding, bool create) {
  if (binding && binding->valid_for(key))
    return binding->state_;
  return find_workgroup(key, create);
}

uint32_t MemoryRaceChecker::resolve_wave_index(WorkgroupState &state, const WaveKey &key,
                                               const WaveBinding *binding, Statistics *statistics,
                                               bool create) {
  if (binding && binding->valid_for(key) && binding->state_.get() == &state &&
      binding->index_ < state.waves.size() && state.waves[binding->index_].key == key)
    return binding->index_;
  return create ? state.wave_index(key, statistics) : state.find_wave(key, statistics);
}

std::unique_lock<std::mutex> MemoryRaceChecker::lock_workgroup_state(WorkgroupState &state,
                                                                     Statistics **statistics) {
  const bool collect = statistics_enabled_.load(std::memory_order_relaxed);
  StatisticsClock::time_point started;
  if (collect)
    started = StatisticsClock::now();
  std::unique_lock lock(state.mutex);
  const uint64_t lock_wait_ns = collect ? elapsed_ns(started) : 0;
  if (collect) {
    if (!state.statistics)
      state.statistics = std::make_unique<Statistics>();
    ++state.statistics->workgroup_state_lock_acquisitions;
    state.statistics->workgroup_state_lock_wait_ns += lock_wait_ns;
  }
  if (statistics)
    *statistics = collect ? state.statistics.get() : nullptr;
  return lock;
}

void MemoryRaceChecker::merge_completed_statistics(uint32_t dispatch,
                                                   const Statistics &statistics) {
  if (!statistics_enabled_.load(std::memory_order_relaxed))
    return;
  std::lock_guard lock(statistics_->mutex);
  statistics_->completed_workgroups[dispatch].add(statistics);
}

std::vector<MemoryRaceChecker::WaveBinding>
MemoryRaceChecker::begin_workgroup(std::span<const WaveKey> members) {
  std::vector<WaveBinding> bindings;
  if (members.empty())
    return bindings;
  bindings.reserve(members.size());
  std::vector<uint32_t> indices;
  indices.reserve(members.size());
#ifndef NDEBUG
  const WorkgroupKey workgroup = workgroup_of(members.front());
  assert(std::ranges::all_of(members,
                             [&](const WaveKey &key) { return workgroup_of(key) == workgroup; }));
#endif
  const auto state = find_workgroup(members.front(), true);
  {
    Statistics *stats = nullptr;
    auto lock = lock_workgroup_state(*state, &stats);
    state->begin_waves(members, indices, stats);
    for (size_t i = 0; i < members.size(); ++i) {
      WaveSlot &wave = state->waves[indices[i]];
      wave.key = members[i];
      wave.reset();
      wave.active = true;
    }
  }
  for (size_t i = 0; i < members.size(); ++i) {
    WaveBinding binding;
    binding.state_ = state;
    binding.key_ = members[i];
    binding.index_ = indices[i];
    bindings.push_back(std::move(binding));
  }
  return bindings;
}

MemoryRaceChecker::WaveBinding MemoryRaceChecker::begin_wave(const WaveKey &key) {
  auto bindings = begin_workgroup(std::span(&key, 1));
  return bindings.empty() ? WaveBinding{} : std::move(bindings.front());
}

void MemoryRaceChecker::end_wave(const WaveKey &key, const WaveBinding *binding) {
  const auto state = resolve_workgroup(key, binding, false);
  if (!state)
    return;
  Statistics *stats = nullptr;
  auto lock = lock_workgroup_state(*state, &stats);
  const uint32_t index = resolve_wave_index(*state, key, binding, stats, false);
  if (index != kNoIndex) {
    state->waves[index].reset();
    state->waves[index].active = false;
  }
}

void MemoryRaceChecker::lds_access(const Access &instruction, std::span<const LaneRange> lanes,
                                   const MemoryWaitScoreboard &scoreboard,
                                   const WaveBinding *binding, LdsCheckMode mode) {
  lds_access_impl(instruction, {.explicit_lanes = lanes}, scoreboard, binding, mode);
}

void MemoryRaceChecker::lds_access(const Access &instruction, const AffineLaneRange &lanes,
                                   const MemoryWaitScoreboard &scoreboard,
                                   const WaveBinding *binding, LdsCheckMode mode) {
  lds_access_impl(instruction, {.affine = lanes, .is_affine = true}, scoreboard, binding, mode);
}

void MemoryRaceChecker::lds_access_impl(const Access &instruction, const AccessRanges &source,
                                        const MemoryWaitScoreboard &scoreboard,
                                        const WaveBinding *binding, LdsCheckMode mode) {
  if ((!instruction.read && !instruction.write) || (!mode.same_wave && !mode.cross_wave))
    return;
  const auto run = [&](const auto &lanes) {
    uint64_t lo = std::numeric_limits<uint64_t>::max();
    uint64_t hi = 0;
    uint32_t live = 0;
    for_each_range(lanes, [&](const LaneRange &range) {
      if (!range.size)
        return;
      lo = std::min(lo, range.address);
      hi = std::max(hi, saturated_end(range.address, range.size));
      ++live;
    });
    if (!live)
      return;

    const uint8_t flags = flags_of(instruction);
    const auto state = resolve_workgroup(instruction.wave, binding, true);
    std::vector<Finding> findings;
    {
      Statistics *stats = nullptr;
      auto lock = lock_workgroup_state(*state, &stats);
      const auto check = [&]<bool Collect>(Statistics *counters) {
        if constexpr (Collect) {
          ++counters->lds_accesses;
          counters->lane_ranges += live;
          if (source.is_affine) {
            ++counters->affine_accesses;
            counters->affine_lane_ranges += live;
          } else {
            ++counters->explicit_accesses;
          }
        }
        const uint32_t self = resolve_wave_index(*state, instruction.wave, binding, counters, true);
        WaveSlot &wave = state->waves[self];
        wave.scoreboard = &scoreboard;

        bool retired = false;
        if (mode.same_wave) {
          for (PendingOp &op : wave.pending) {
            if constexpr (Collect)
              ++counters->pending_candidates;
            if (!((op.flags | flags) & kWrite) || op.hi <= lo || hi <= op.lo)
              continue;
            // Ordinary DS operations in one ordered completion class observe
            // issue order and do not require an intervening wait.
            if (op.ordered_class && op.ordered_class == instruction.ordered_class)
              continue;
            if (complete(op, scoreboard)) {
              retired = true;
              continue;
            }
            const Access producer = wave.access(op);
            const auto compare = [&](const auto &prior_lanes) {
              for_each_range(prior_lanes, [&](const LaneRange &prior) {
                const uint64_t prior_end = saturated_end(prior.address, prior.size);
                if (prior_end <= lo || hi <= prior.address)
                  return;
                for_each_range(lanes, [&](const LaneRange &range) {
                  const uint64_t start = std::max(prior.address, range.address);
                  const uint64_t stop =
                      std::min(prior_end, saturated_end(range.address, range.size));
                  if (start >= stop)
                    return;
                  Access producer_lane = producer;
                  producer_lane.address = prior.address;
                  producer_lane.size = prior.size;
                  producer_lane.lane = prior.lane;
                  findings.push_back({Space::LDS, kind_of(op.flags, flags), producer_lane,
                                      consumer_of(instruction, range), start,
                                      static_cast<uint32_t>(stop - start), false});
                });
              });
            };
            if (op.is_affine())
              compare(op.affine);
            else
              compare(std::span(wave.ranges).subspan(op.first, op.count));
          }
        }
        if (retired || wave.pending.size() >= kPendingPruneThreshold)
          wave.prune(scoreboard);
        if (instruction.completion_count) {
          if constexpr (Collect) {
            if (wave.pending.size() == wave.pending.capacity())
              ++counters->pending_vector_growths;
          }
          PendingOp pending{.pc = instruction.pc,
                            .lo = lo,
                            .hi = hi,
                            .count = live,
                            .ordered_class = instruction.ordered_class,
                            .flags = flags,
                            .completion_count = instruction.completion_count,
                            .completions = instruction.completions};
          if (source.is_affine) {
            pending.affine = source.affine;
            if constexpr (Collect)
              ++counters->affine_pending_ops;
          } else {
            pending.first = static_cast<uint32_t>(wave.ranges.size());
            for_each_range(lanes, [&](const LaneRange &range) {
              if (range.size) {
                if constexpr (Collect) {
                  if (wave.ranges.size() == wave.ranges.capacity())
                    ++counters->range_vector_growths;
                }
                wave.ranges.push_back(range);
              }
            });
          }
          wave.pending.push_back(pending);
        }

        if (mode.cross_wave)
          state->record<Collect>(self, instruction, flags, lanes, findings, counters);
      };
      if (stats)
        check.template operator()<true>(stats);
      else
        check.template operator()<false>(nullptr);
    }
    if (!findings.empty())
      report(findings);
  };
  if (source.is_affine)
    run(source.affine);
  else
    run(source.explicit_lanes);
}

void MemoryRaceChecker::global_access(const Access &instruction, std::span<const LaneRange> lanes) {
  global_access_impl(instruction, {.explicit_lanes = lanes});
}

void MemoryRaceChecker::global_access(const Access &instruction, const AffineLaneRange &lanes) {
  global_access_impl(instruction, {.affine = lanes, .is_affine = true});
}

void MemoryRaceChecker::global_access_impl(const Access &instruction, const AccessRanges &source) {
  if (!instruction.read && !instruction.write)
    return;
  const bool collect = statistics_enabled_.load(std::memory_order_relaxed);
  const auto check = [&]<bool Collect>(const auto &lanes) {
    uint64_t live_ranges = 0;
    if constexpr (Collect)
      for_each_range(lanes, [&](const LaneRange &range) { live_ranges += range.size != 0; });
    if constexpr (Collect) {
      if (!live_ranges)
        return;
      if (!tracking_.load(std::memory_order_relaxed))
        tracking_.store(true, std::memory_order_relaxed);
    }
    WordStatistics word_statistics;
    Statistics *shard_statistics = nullptr;
    const uint8_t flags = flags_of(instruction);
    const uint32_t workgroup = instruction.wave.workgroup;
    std::vector<Finding> findings;
    bool degraded = false;
    GlobalShard *shard = nullptr;
    GlobalShard::DispatchShadow *shadow = nullptr;
    std::unique_lock<std::mutex> lock;
    PageKey key{instruction.wave.dispatch, std::numeric_limits<uint64_t>::max()};
    GlobalPage *page = nullptr;

    for_each_word(
        lanes,
        [&](uint64_t index, uint8_t mask, const LaneRange &range) {
          if ((index >> kPageShift) != key.page) {
            key.page = index >> kPageShift;
            const size_t owner_index = PageKeyHash{}(key) % kGlobalShards;
            GlobalShard &owner = global_[owner_index];
            if (&owner != shard) {
              // Hold one shard at a time: lanes visit shards in no fixed order.
              if (lock.owns_lock())
                lock.unlock();
              shard = &owner;
              StatisticsClock::time_point started;
              if constexpr (Collect)
                started = StatisticsClock::now();
              lock = std::unique_lock(owner.mutex);
              const uint64_t lock_wait_ns = Collect ? elapsed_ns(started) : 0;
              if constexpr (Collect) {
                shard_statistics =
                    &statistics_->global_shards[owner_index][instruction.wave.dispatch];
                ++shard_statistics->global_shard_lock_acquisitions;
                shard_statistics->global_shard_lock_wait_ns += lock_wait_ns;
              }
            }
            if (shard_statistics)
              ++shard_statistics->shadow_page_lookups;
            shadow = shard->find_dispatch(key.dispatch);
            page = shadow ? shadow->find_page(key.page) : nullptr;
          }
          const uint64_t word = index << 2;
          const size_t slot = index & (kPageWords - 1);
          if (shard_statistics) {
            if (page && page->heads[slot] != kNoIndex)
              ++shard_statistics->nonempty_record_heads;
            else
              ++shard_statistics->empty_record_heads;
          }
          uint32_t replacement = kNoIndex;
          // Bytes this workgroup already retains with the same access flags, and
          // bytes retained by one or by two distinct other workgroups. Two distinct
          // other owners suffice: any later conflicting workgroup differs from one.
          uint8_t covered = 0;
          uint8_t one = 0;
          uint8_t two = 0;
          std::array<uint32_t, 4> owners{};
          for (uint32_t i = page ? page->heads[slot] : kNoIndex; i != kNoIndex;) {
            const uint32_t current = i;
            if (shard_statistics)
              ++shard_statistics->global_records_visited;
            GlobalRecord &prior = shadow->records[current];
            i = prior.next;
            const uint8_t overlap = prior.bytes & mask;
            if (prior.workgroup == workgroup) {
              if (prior.flags != flags)
                continue;
              if (replacement == kNoIndex && (prior.bytes & ~mask) == 0)
                replacement = current;
              covered |= overlap;
              continue;
            }
            if (!overlap)
              continue;
            if (conflicting(prior.flags, flags)) {
              const GlobalSite &site = shadow->sites[prior.site];
              const uint32_t start = std::countr_zero(overlap);
              const uint32_t length = std::countr_one(static_cast<uint8_t>(overlap >> start));
              Access producer{.wave = {instruction.wave.process, instruction.wave.dispatch,
                                       prior.workgroup, site.wave, instruction.wave.generation},
                              .pc = site.pc,
                              .address = word + start,
                              .size = length,
                              .lane = site.lane};
              set_flags(producer, prior.flags);
              findings.push_back({Space::Global, kind_of(prior.flags, flags), producer,
                                  consumer_of(instruction, range), word + start, length, false});
            }
            if (prior.flags != flags)
              continue;
            for (uint8_t bits = overlap & ~two; bits; bits &= bits - 1) {
              const uint32_t byte = std::countr_zero(bits);
              const uint8_t bit = static_cast<uint8_t>(1u << byte);
              if (!(one & bit)) {
                one |= bit;
                owners[byte] = prior.workgroup;
              } else if (owners[byte] != prior.workgroup) {
                two |= bit;
              }
            }
          }

          const GlobalSite incoming_site{
              .pc = instruction.pc,
              .wave = instruction.wave.wave,
              .lane = compact_lane(range.lane),
          };
          const bool needs_retention = replacement != kNoIndex || (mask & ~covered & ~two);
          if (!needs_retention)
            return;

          if (!shadow) {
            shadow = &shard->create_dispatch(key.dispatch);
            tracking_.store(true, std::memory_order_relaxed);
          }
          uint32_t site_hash = 0;
          uint32_t site_slot = kNoIndex;
          uint32_t site = shadow->find_site(incoming_site, site_hash, site_slot, shard_statistics);
          if (site == kNoIndex) {
            if (shadow->site_insert_requires_growth() &&
                shard->saturated(key.dispatch, key.page, *global_budget_, shard_statistics)) {
              degraded = true;
              return;
            }
            if (!site_hash)
              site_hash = GlobalShard::DispatchShadow::site_hash(incoming_site);
            site = shadow->insert_site(incoming_site, site_hash, site_slot, *global_budget_,
                                       shard_statistics);
            if (site == kNoIndex) {
              shard->mark_saturated(key.dispatch, key.page, *global_budget_, shard_statistics);
              degraded = true;
              if (shard_statistics)
                ++shard_statistics->records_rejected;
              return;
            }
          } else if (shard_statistics) {
            ++shard_statistics->sites_reused;
          }

          if (replacement != kNoIndex) {
            GlobalRecord &prior = shadow->records[replacement];
            prior.site = site;
            prior.bytes = mask;
            if (shard_statistics)
              ++shard_statistics->records_replaced;
            return;
          }

          const bool record_growth = !shadow->records.has_space();
          if (record_growth) {
            if (shard->saturated(key.dispatch, key.page, *global_budget_, shard_statistics)) {
              degraded = true;
              return;
            }
            if (!shadow->records.ensure(*global_budget_, shadow->charged, shard_statistics,
                                        &Statistics::record_slab_allocations)) {
              shard->mark_saturated(key.dispatch, key.page, *global_budget_, shard_statistics);
              degraded = true;
              if (shard_statistics)
                ++shard_statistics->records_rejected;
              return;
            }
            if (shard_statistics)
              ++shard_statistics->record_vector_growths;
          }

          if (!page) {
            if (shadow->page_insert_requires_growth() &&
                shard->saturated(key.dispatch, key.page, *global_budget_, shard_statistics)) {
              degraded = true;
              return;
            }
            page = shadow->create_page(key.page, *global_budget_, shard_statistics);
            if (!page) {
              shard->mark_saturated(key.dispatch, key.page, *global_budget_, shard_statistics);
              degraded = true;
              if (shard_statistics)
                ++shard_statistics->shadow_pages_rejected;
              return;
            }
          }
          const uint32_t added = shadow->records.append({.next = page->heads[slot],
                                                         .workgroup = workgroup,
                                                         .site = site,
                                                         .flags = flags,
                                                         .bytes = mask});
          if (added == kNoIndex) {
            degraded = true;
            return;
          }
          page->heads[slot] = added;
          if (shard_statistics)
            ++shard_statistics->records_inserted;
        },
        Collect ? &word_statistics : nullptr);
    if constexpr (Collect) {
      if (shard_statistics) {
        ++shard_statistics->global_accesses;
        shard_statistics->lane_ranges += live_ranges;
        shard_statistics->dword_candidates += word_statistics.dword_candidates;
        shard_statistics->dword_visits += word_statistics.dword_visits;
        shard_statistics->dword_duplicates_skipped += word_statistics.dword_duplicates_skipped;
        shard_statistics->affine_dword_runs += word_statistics.affine_dword_runs;
        shard_statistics->affine_dword_visits += word_statistics.affine_dword_visits;
        if (source.is_affine) {
          ++shard_statistics->affine_accesses;
          shard_statistics->affine_lane_ranges += live_ranges;
        } else {
          ++shard_statistics->explicit_accesses;
        }
      }
    }
    if (lock.owns_lock())
      lock.unlock();
    if (degraded)
      mark_coverage_degraded();
    if (!findings.empty())
      report(findings);
  };
  if (collect) {
    if (source.is_affine)
      check.template operator()<true>(source.affine);
    else
      check.template operator()<true>(source.explicit_lanes);
  } else if (source.is_affine) {
    check.template operator()<false>(source.affine);
  } else {
    check.template operator()<false>(source.explicit_lanes);
  }
}

void MemoryRaceChecker::mark_coverage_degraded() {
  if (coverage_degraded_.load(std::memory_order_relaxed) ||
      coverage_degraded_.exchange(true, std::memory_order_relaxed))
    return;
  util::Logger::warn(std::format(
      "memory-race: global shadow budget of {} MiB reached. Later global accesses are checked "
      "but not retained, so some races can be missed; raise top-level "
      "race_checking_global_shadow_mb to retain more.",
      global_shadow_budget() >> 20));
  std::shared_ptr<MemoryDiagnosticReport> diagnostic_report;
  {
    std::lock_guard lock(report_mutex_);
    diagnostic_report = report_;
  }
  if (diagnostic_report)
    diagnostic_report->set_coverage_degraded();
}

void MemoryRaceChecker::retire_completed(const WaveKey &key, const MemoryWaitScoreboard &scoreboard,
                                         const WaveBinding *binding) {
  const auto state = resolve_workgroup(key, binding, false);
  if (!state)
    return;
  Statistics *stats = nullptr;
  auto lock = lock_workgroup_state(*state, &stats);
  const uint32_t index = resolve_wave_index(*state, key, binding, stats, false);
  if (index == kNoIndex)
    return;
  WaveSlot &wave = state->waves[index];
  wave.scoreboard = &scoreboard;
  if (!wave.pending.empty())
    wave.prune(scoreboard);
}

void MemoryRaceChecker::barrier(std::span<const WaveKey> members) {
  if (members.empty())
    return;
  const auto same_workgroup = [](const WaveKey &a, const WaveKey &b) {
    return workgroup_of(a) == workgroup_of(b);
  };
  if (std::ranges::all_of(members,
                          [&](const WaveKey &key) { return same_workgroup(key, members[0]); })) {
    barrier_workgroup(members);
  } else {
    // A cluster barrier can contain multiple workgroups. LDS histories are
    // resource-local, but clocks merge for the participating waves in each one.
    std::vector<WaveKey> sorted(members.begin(), members.end());
    std::ranges::stable_sort(
        sorted, {}, [](const WaveKey &key) { return std::tuple(key.dispatch, key.workgroup); });
    for (auto begin = sorted.begin(); begin != sorted.end();) {
      const auto end = std::find_if(
          begin, sorted.end(), [&](const WaveKey &key) { return !same_workgroup(key, *begin); });
      barrier_workgroup(std::span<const WaveKey>(begin, end));
      begin = end;
    }
  }
}

void MemoryRaceChecker::barrier_workgroup(std::span<const WaveKey> members) {
  const auto state = find_workgroup(members.front(), false);
  if (!state)
    return;
  // Full-workgroup barriers are the common case. Allocate their local clock
  // scratch before taking the state mutex; a named subset grows here only if
  // the state contains more waves than the participating span.
  std::vector<uint32_t> indices;
  std::vector<uint64_t> merged;
  std::vector<uint8_t> joined;
  indices.reserve(members.size());
  merged.reserve(members.size());
  joined.reserve(members.size());
  Statistics *stats = nullptr;
  auto lock = lock_workgroup_state(*state, &stats);
  if (stats)
    state->barrier<true>(members, indices, merged, joined, stats);
  else
    state->barrier<false>(members, indices, merged, joined, nullptr);
}

void MemoryRaceChecker::end_workgroup(uint32_t dispatch, uint32_t workgroup) {
  const WorkgroupKey id{dispatch, workgroup};
  auto &shard = workgroups_[WorkgroupKeyHash{}(id) % kWorkgroupShards];
  std::shared_ptr<WorkgroupState> released;
  {
    std::lock_guard lock(shard.mutex);
    if (const auto it = shard.states.find(id); it != shard.states.end()) {
      released = std::move(it->second);
      shard.states.erase(it);
    }
  }
  if (!released || !statistics_enabled_.load(std::memory_order_relaxed))
    return;
  Statistics completed;
  {
    Statistics *stats = nullptr;
    auto lock = lock_workgroup_state(*released, &stats);
    if (!stats)
      return;
    ++stats->cleanup_workgroups;
    stats->cleanup_dwords += released->words.touched.size();
    stats->cleanup_records += released->records.size();
    completed = *stats;
  }
  // The detached state no longer needs its mutex while dispatch-wide counters
  // are merged under the separate statistics lock.
  merge_completed_statistics(dispatch, completed);
}

void MemoryRaceChecker::end_dispatch(uint32_t dispatch, bool cleanup_workgroups,
                                     bool cleanup_global) {
  const bool collect = statistics_enabled_.load(std::memory_order_relaxed);
  if (!tracking_.load(std::memory_order_relaxed))
    return;
  DispatchStatistics dispatch_statistics;
  dispatch_statistics.dispatch = dispatch;
  Statistics completed_statistics;
  bool had_completed_statistics = false;
  {
    std::lock_guard lock(statistics_->mutex);
    if (const auto it = statistics_->completed_workgroups.find(dispatch);
        it != statistics_->completed_workgroups.end()) {
      if (collect) {
        completed_statistics = it->second;
        had_completed_statistics = true;
      }
      statistics_->completed_workgroups.erase(it);
    }
  }
  if (had_completed_statistics)
    dispatch_statistics.counters.add(completed_statistics);

  if (cleanup_workgroups) {
    // Most shards own at most this many workgroups from one dispatch. Keep those
    // detached pointers inline so cleanup does not allocate while the map is
    // locked; only unusually dense hash buckets spill.
    std::array<std::shared_ptr<WorkgroupState>, kWorkgroupShards> released_workgroups;
    size_t released_workgroup_count = 0;
    std::vector<std::shared_ptr<WorkgroupState>> released_workgroup_overflow;
    for (size_t i = 0; i < kWorkgroupShards; ++i) {
      for (size_t released = 0; released < released_workgroup_count; ++released)
        released_workgroups[released].reset();
      released_workgroup_count = 0;
      released_workgroup_overflow.clear();
      Statistics shard_statistics;
      bool had_shard_statistics = false;
      {
        std::lock_guard lock(workgroups_[i].mutex);
        auto &statistics = statistics_->workgroup_shards[i];
        if (const auto it = statistics.find(dispatch); it != statistics.end()) {
          if (collect) {
            shard_statistics = it->second;
            had_shard_statistics = true;
          }
          statistics.erase(it);
        }
        for (auto it = workgroups_[i].states.begin(); it != workgroups_[i].states.end();) {
          if (it->first.dispatch != dispatch) {
            ++it;
            continue;
          }
          if (released_workgroup_count < released_workgroups.size())
            released_workgroups[released_workgroup_count++] = std::move(it->second);
          else
            released_workgroup_overflow.push_back(std::move(it->second));
          it = workgroups_[i].states.erase(it);
        }
      }
      if (had_shard_statistics)
        dispatch_statistics.counters.add(shard_statistics);
      if (collect) {
        const auto add_state_statistics = [&](const auto &state) {
          Statistics state_statistics;
          bool had_state_statistics = false;
          {
            Statistics *stats = nullptr;
            auto lock = lock_workgroup_state(*state, &stats);
            if (stats) {
              ++stats->cleanup_workgroups;
              stats->cleanup_dwords += state->words.touched.size();
              stats->cleanup_records += state->records.size();
              state_statistics = *stats;
              had_state_statistics = true;
            }
          }
          if (had_state_statistics)
            dispatch_statistics.counters.add(state_statistics);
        };
        for (size_t released = 0; released < released_workgroup_count; ++released)
          add_state_statistics(released_workgroups[released]);
        for (const auto &state : released_workgroup_overflow)
          add_state_statistics(state);
      }
    }
    for (size_t released = 0; released < released_workgroup_count; ++released)
      released_workgroups[released].reset();
    released_workgroup_overflow.clear();
  }

  if (cleanup_global) {
    for (size_t i = 0; i < kGlobalShards; ++i) {
      Statistics shard_statistics;
      std::unique_ptr<GlobalShard::DispatchShadow> released;
      {
        std::lock_guard lock(global_[i].mutex);
        auto &statistics = statistics_->global_shards[i];
        if (const auto it = statistics.find(dispatch); it != statistics.end()) {
          if (collect)
            shard_statistics = it->second;
          statistics.erase(it);
        }
        released = global_[i].take_dispatch(dispatch, collect ? &shard_statistics : nullptr);
      }

      if (released) {
        const uint64_t charged = released->charged;
        dispatch_statistics.global_shadow_bytes_released += charged;
        if (collect) {
          shard_statistics.cleanup_pages += released->pages.entries;
          shard_statistics.cleanup_records += released->records.entries;
        }
        // Slab destruction can return substantial storage to the allocator. Do it
        // outside the shard lock, then make that capacity available to other shards.
        released.reset();
        global_budget_->release(charged);
        if (collect)
          shard_statistics.budget_bytes_released += charged;
      }
      if (!collect || shard_statistics.empty())
        continue;
      dispatch_statistics.counters.add(shard_statistics);
      dispatch_statistics.global_shards.push_back(
          {.shard = static_cast<uint32_t>(i), .counters = shard_statistics});
    }
  }
  dispatch_statistics.global_shadow_peak_bytes = global_shadow_peak_bytes();
  std::shared_ptr<MemoryDiagnosticReport> diagnostic_report;
  {
    std::lock_guard lock(report_mutex_);
    std::erase_if(reported_, [&](const FindingKey &key) { return key.dispatch == dispatch; });
    diagnostic_report = report_;
  }
  if (collect && diagnostic_report && !dispatch_statistics.counters.empty())
    diagnostic_report->add_race_statistics(std::move(dispatch_statistics));
}

size_t MemoryRaceChecker::tracked_workgroups() const {
  size_t count = 0;
  for (size_t i = 0; i < kWorkgroupShards; ++i) {
    std::lock_guard lock(workgroups_[i].mutex);
    count += workgroups_[i].states.size();
  }
  return count;
}

uint64_t MemoryRaceChecker::global_shadow_bytes() const {
  return global_budget_->current.load(std::memory_order_relaxed);
}

uint64_t MemoryRaceChecker::global_shadow_peak_bytes() const {
  return global_budget_->peak.load(std::memory_order_relaxed);
}

void MemoryRaceChecker::report(std::vector<Finding> &findings) {
  if (findings.empty())
    return;
  coalesce_findings(findings);
  std::vector<Finding> delivered;
  delivered.reserve(findings.size());
  std::shared_ptr<const Reporter> reporter;
  std::shared_ptr<MemoryDiagnosticReport> diagnostic_report;
  uint64_t suppressed = 0;
  bool warn_suppressed = false;
  {
    std::lock_guard lock(report_mutex_);
    reporter = reporter_;
    diagnostic_report = report_;
    for (Finding &finding : findings) {
      const FindingKey key{finding.space,       finding.kind,        finding.consumer.wave.dispatch,
                           finding.producer.pc, finding.consumer.pc, finding.address};
      if (reported_.contains(key))
        continue;
      ++finding_count_;
      if (reported_.size() >= kMaxFindings) {
        warn_suppressed |= ++suppressed_count_ == 1;
        ++suppressed;
        continue;
      }
      reported_.insert(key);
      delivered.push_back(std::move(finding));
    }
  }
  if (warn_suppressed)
    util::Logger::warn("memory-race: further diagnostics are suppressed");
  if (diagnostic_report) {
    if (suppressed)
      diagnostic_report->add_suppressed_races(suppressed);
    diagnostic_report->add_races(delivered);
  }
  if (reporter && *reporter)
    for (const Finding &finding : delivered)
      (*reporter)(finding);
}

} // namespace rocjitsu::amdgpu
