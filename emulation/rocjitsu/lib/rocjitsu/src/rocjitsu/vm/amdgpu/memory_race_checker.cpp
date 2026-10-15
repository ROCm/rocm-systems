// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_race_checker.h"

#include "rocjitsu/vm/amdgpu/memory_diagnostic_report.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <format>
#include <limits>
#include <tuple>
#include <unordered_map>

namespace rocjitsu::amdgpu {
namespace {

using Access = MemoryRaceChecker::Access;
using Finding = MemoryRaceChecker::Finding;
using Kind = MemoryRaceChecker::Kind;
using LaneRange = MemoryRaceChecker::LaneRange;
using Space = MemoryRaceChecker::Space;
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

/// Visit each dword a lane span touches as (dword index, byte mask, range).
/// Lanes repeating a recently visited dword and mask are skipped: they would
/// check and record exactly what the first lane did.
template <typename Visit> void for_each_word(std::span<const LaneRange> lanes, Visit &&visit) {
  constexpr size_t kRecent = 64;
  std::array<uint64_t, kRecent> recent_index;
  std::array<uint8_t, kRecent> recent_mask;
  const bool dedupe = lanes.size() > 1;
  if (dedupe)
    recent_index.fill(std::numeric_limits<uint64_t>::max());
  for (const LaneRange &range : lanes) {
    if (!range.size)
      continue;
    const uint64_t end = saturated_end(range.address, range.size);
    const uint64_t last = (end - 1) >> 2;
    for (uint64_t index = range.address >> 2; index <= last; ++index) {
      const uint8_t mask = word_mask(index << 2, range.address, end);
      if (dedupe) {
        const size_t slot = index & (kRecent - 1);
        if (recent_index[slot] == index && recent_mask[slot] == mask)
          continue;
        recent_index[slot] = index;
        recent_mask[slot] = mask;
      }
      visit(index, mask, range);
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
  uint32_t first = 0;
  uint32_t count = 0;
  uint32_t ordered_class = 0;
  uint8_t flags = 0;
  uint8_t completion_count = 0;
  std::array<CompletionToken, 2> completions{};
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
// Approximate per-page cost of the owning hash map node.
constexpr uint64_t kPageOverhead = 64;

/// One retained global access to one dword by one workgroup.
struct GlobalRecord {
  uint64_t pc = 0;
  uint32_t workgroup = 0;
  uint32_t wave = 0;
  uint32_t next = kNoIndex;
  int8_t lane = -1;
  uint8_t flags = 0;
  uint8_t bytes = 0;
};
static_assert(sizeof(GlobalRecord) == 24);

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

constexpr uint64_t kPageBytes = sizeof(GlobalPage) + kPageOverhead;

} // namespace

struct MemoryRaceChecker::WorkgroupState {
  std::mutex mutex;
  std::vector<WaveSlot> waves;
  /// Vector clocks, row-major by observing wave: clocks[observer * n + source].
  std::vector<uint64_t> clocks;
  WordIndex words;
  std::vector<LdsRecord> records;
  uint32_t free_record = kNoIndex;

  uint32_t find_wave(const WaveKey &key) const {
    for (uint32_t i = 0; i < waves.size(); ++i)
      if (waves[i].key.wave == key.wave && waves[i].key.generation == key.generation)
        return i;
    return kNoIndex;
  }

  uint32_t wave_index(const WaveKey &key) {
    if (const uint32_t found = find_wave(key); found != kNoIndex)
      return found;
    const size_t n = waves.size();
    std::vector<uint64_t> grown((n + 1) * (n + 1), 0);
    for (size_t observer = 0; observer < n; ++observer)
      std::copy_n(clocks.begin() + observer * n, n, grown.begin() + observer * (n + 1));
    clocks = std::move(grown);
    waves.push_back({.key = key});
    return static_cast<uint32_t>(n);
  }

  uint64_t &clock(size_t observer, size_t source) {
    return clocks[observer * waves.size() + source];
  }

  uint32_t allocate(const LdsRecord &record) {
    if (free_record == kNoIndex) {
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

  void clear_history() {
    words.clear();
    records.clear();
    free_record = kNoIndex;
  }

  /// Drop records visible to every active wave. Ended waves never access LDS
  /// again, so they do not keep history alive.
  void prune_history() {
    const size_t n = waves.size();
    std::vector<uint64_t> floor(n, std::numeric_limits<uint64_t>::max());
    for (size_t observer = 0; observer < n; ++observer)
      if (waves[observer].active)
        for (size_t source = 0; source < n; ++source)
          floor[source] = std::min(floor[source], clock(observer, source));
    size_t kept = 0;
    for (const uint64_t word : words.touched) {
      uint32_t *link = &words.head(word);
      while (*link != kNoIndex) {
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
  void record(uint32_t self, const Access &instruction, uint8_t flags,
              std::span<const LaneRange> lanes, std::vector<Finding> &findings) {
    const uint64_t sequence = ++clock(self, self);
    waves[self].last_sequence = sequence;
    const uint64_t *seen = clocks.data() + size_t{self} * waves.size();
    for_each_word(lanes, [&](uint64_t index, uint8_t mask, const LaneRange &range) {
      uint32_t *link = &words.head(index);
      const bool was_empty = *link == kNoIndex;
      bool kept = false;
      while (*link != kNoIndex) {
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
                    compact_lane(range.lane), pack(flags, mask)});
      uint32_t &head = words.head(index);
      records[added].next = head;
      head = added;
      if (was_empty)
        words.touched.push_back(index);
    });
  }

  /// Re-record `self`'s in-flight operation at `sequence`, which the other
  /// barrier members have not observed. It stays unordered with their later
  /// accesses until a barrier that follows its completion.
  void carry(uint32_t self, const WaveSlot &wave, const PendingOp &op, uint64_t sequence) {
    const uint8_t flags = op.flags | kInFlight;
    const auto lanes = std::span(wave.ranges).subspan(op.first, op.count);
    for_each_word(lanes, [&](uint64_t index, uint8_t mask, const LaneRange &range) {
      for (uint32_t i = words.head(index); i != kNoIndex; i = records[i].next) {
        LdsRecord &prior = records[i];
        if (prior.wave == self && (prior.flags() & ~kInFlight) == op.flags &&
            (prior.bytes() & ~mask) == 0) {
          prior.pc = op.pc;
          prior.sequence = sequence;
          prior.lane = compact_lane(range.lane);
          prior.bits = pack(flags, mask);
          return;
        }
      }
      const bool was_empty = words.head(index) == kNoIndex;
      const uint32_t added = allocate({op.pc, sequence, kNoIndex, static_cast<uint16_t>(self),
                                       compact_lane(range.lane), pack(flags, mask)});
      uint32_t &head = words.head(index);
      records[added].next = head;
      head = added;
      if (was_empty)
        words.touched.push_back(index);
    });
  }

  void barrier(std::span<const WaveKey> members) {
    std::vector<uint32_t> indices;
    indices.reserve(members.size());
    for (const WaveKey &key : members) {
      const uint32_t index = find_wave(key);
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
    std::vector<uint64_t> merged(n, 0);
    for (const uint32_t member : indices)
      for (size_t source = 0; source < n; ++source)
        merged[source] = std::max(merged[source], clock(member, source));
    std::vector<uint8_t> joined(n, 0);
    for (const uint32_t member : indices) {
      if (joined[member])
        continue;
      joined[member] = 1;
      for (size_t source = 0; source < n; ++source)
        clock(member, source) = std::max(clock(member, source), merged[source]);
      ++clock(member, member);
    }
    // History is fully visible when every active wave joined and every ended
    // wave's last access was already ordered before the barrier. Records still
    // needed by a nonparticipant survive named barriers.
    bool visible = true;
    for (size_t wave = 0; wave < n && visible; ++wave)
      visible = joined[wave] || (!waves[wave].active && merged[wave] >= waves[wave].last_sequence);
    if (visible)
      clear_history();
    else
      prune_history();
    // A barrier orders only completed operations; multi-buffered async copies
    // legitimately stay in flight across the barrier that guards another buffer.
    for (size_t wave = 0; wave < n; ++wave) {
      WaveSlot &slot = waves[wave];
      if (!joined[wave] || slot.pending.empty())
        continue;
      const uint64_t sequence = clock(wave, wave);
      slot.last_sequence = sequence;
      for (const PendingOp &op : slot.pending)
        carry(static_cast<uint32_t>(wave), slot, op, sequence);
    }
  }
};

struct MemoryRaceChecker::WorkgroupShard {
  std::mutex mutex;
  std::unordered_map<WorkgroupKey, std::shared_ptr<WorkgroupState>, WorkgroupKeyHash> states;
};

struct MemoryRaceChecker::GlobalShard {
  static constexpr size_t kMinRecords = 1024;

  std::mutex mutex;
  std::unordered_map<PageKey, std::unique_ptr<GlobalPage>, PageKeyHash> pages;
  std::vector<GlobalRecord> records;
  uint32_t free_record = kNoIndex;

  // Record storage costs its capacity, not just its live records.
  uint64_t bytes() const {
    return records.capacity() * sizeof(GlobalRecord) + pages.size() * kPageBytes;
  }

  GlobalPage *find(const PageKey &key) {
    const auto it = pages.find(key);
    return it == pages.end() ? nullptr : it->second.get();
  }

  GlobalPage *create(const PageKey &key, uint64_t budget) {
    if (bytes() + kPageBytes > budget)
      return nullptr;
    return pages.emplace(key, std::make_unique<GlobalPage>()).first->second.get();
  }

  bool insert(uint32_t &head, GlobalRecord record, uint64_t budget) {
    if (free_record == kNoIndex && records.size() == records.capacity()) {
      const uint64_t page_bytes = pages.size() * kPageBytes;
      const uint64_t fit =
          page_bytes < budget ? (budget - page_bytes) / sizeof(GlobalRecord) : uint64_t{0};
      if (fit <= records.size())
        return false;
      records.reserve(std::min<uint64_t>(fit, std::max(records.size() * 2, kMinRecords)));
    }
    record.next = head;
    if (free_record == kNoIndex) {
      head = static_cast<uint32_t>(records.size());
      records.push_back(record);
    } else {
      head = free_record;
      free_record = records[head].next;
      records[head] = record;
    }
    return true;
  }

  void clear() {
    pages.clear();
    std::vector<GlobalRecord>().swap(records);
    free_record = kNoIndex;
  }

  void erase_dispatch(uint32_t dispatch) {
    const auto matches = [&](const auto &entry) { return entry.first.dispatch == dispatch; };
    if (std::ranges::all_of(pages, matches)) {
      clear();
      return;
    }
    for (auto it = pages.begin(); it != pages.end();) {
      if (!matches(*it)) {
        ++it;
        continue;
      }
      for (uint32_t head : it->second->heads)
        while (head != kNoIndex) {
          const uint32_t dead = head;
          head = records[dead].next;
          records[dead].next = free_record;
          free_record = dead;
        }
      it = pages.erase(it);
    }
  }
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

MemoryRaceChecker::MemoryRaceChecker(Reporter reporter)
    : workgroups_(std::make_unique<WorkgroupShard[]>(kWorkgroupShards)),
      global_(std::make_unique<GlobalShard[]>(kGlobalShards)),
      reporter_(reporter ? std::move(reporter) : Reporter(log_finding)) {}

MemoryRaceChecker::~MemoryRaceChecker() = default;

void MemoryRaceChecker::set_reporter(Reporter reporter) {
  std::lock_guard lock(report_mutex_);
  reporter_ = std::move(reporter);
}

void MemoryRaceChecker::set_report(std::shared_ptr<MemoryDiagnosticReport> report) {
  std::lock_guard lock(report_mutex_);
  report_ = std::move(report);
  if (report_ && coverage_degraded_.load(std::memory_order_relaxed))
    report_->set_coverage_degraded();
}

void MemoryRaceChecker::set_global_shadow_budget(uint64_t bytes) {
  global_shadow_budget_.store(bytes);
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

std::shared_ptr<MemoryRaceChecker::WorkgroupState>
MemoryRaceChecker::find_workgroup(const WaveKey &key, bool create) {
  const WorkgroupKey id = workgroup_of(key);
  auto &shard = workgroups_[WorkgroupKeyHash{}(id) % kWorkgroupShards];
  std::lock_guard lock(shard.mutex);
  if (const auto it = shard.states.find(id); it != shard.states.end())
    return it->second;
  if (!create)
    return nullptr;
  if (!tracking_.load(std::memory_order_relaxed))
    tracking_.store(true, std::memory_order_relaxed);
  auto state = std::make_shared<WorkgroupState>();
  shard.states.emplace(id, state);
  return state;
}

void MemoryRaceChecker::begin_wave(const WaveKey &key) {
  const auto state = find_workgroup(key, true);
  std::lock_guard lock(state->mutex);
  WaveSlot &wave = state->waves[state->wave_index(key)];
  wave.reset();
  wave.active = true;
}

void MemoryRaceChecker::end_wave(const WaveKey &key) {
  const auto state = find_workgroup(key, false);
  if (!state)
    return;
  std::lock_guard lock(state->mutex);
  if (const uint32_t index = state->find_wave(key); index != kNoIndex) {
    state->waves[index].reset();
    state->waves[index].active = false;
  }
}

void MemoryRaceChecker::lds_access(const Access &instruction, std::span<const LaneRange> lanes,
                                   const MemoryWaitScoreboard &scoreboard) {
  if (!instruction.read && !instruction.write)
    return;
  uint64_t lo = std::numeric_limits<uint64_t>::max();
  uint64_t hi = 0;
  uint32_t live = 0;
  for (const LaneRange &range : lanes) {
    if (!range.size)
      continue;
    lo = std::min(lo, range.address);
    hi = std::max(hi, saturated_end(range.address, range.size));
    ++live;
  }
  if (!live)
    return;

  const uint8_t flags = flags_of(instruction);
  const auto state = find_workgroup(instruction.wave, true);
  std::vector<Finding> findings;
  {
    std::lock_guard lock(state->mutex);
    const uint32_t self = state->wave_index(instruction.wave);
    WaveSlot &wave = state->waves[self];
    wave.scoreboard = &scoreboard;

    bool retired = false;
    for (PendingOp &op : wave.pending) {
      if (!((op.flags | flags) & kWrite) || op.hi <= lo || hi <= op.lo)
        continue;
      // Ordinary DS operations in one ordered completion class observe issue
      // order and do not require an intervening wait.
      if (op.ordered_class && op.ordered_class == instruction.ordered_class)
        continue;
      if (complete(op, scoreboard)) {
        retired = true;
        continue;
      }
      const Access producer = wave.access(op);
      for (const LaneRange &prior : std::span(wave.ranges).subspan(op.first, op.count)) {
        const uint64_t prior_end = saturated_end(prior.address, prior.size);
        if (prior_end <= lo || hi <= prior.address)
          continue;
        for (const LaneRange &range : lanes) {
          const uint64_t start = std::max(prior.address, range.address);
          const uint64_t stop = std::min(prior_end, saturated_end(range.address, range.size));
          if (start >= stop)
            continue;
          Access producer_lane = producer;
          producer_lane.address = prior.address;
          producer_lane.size = prior.size;
          producer_lane.lane = prior.lane;
          findings.push_back({Space::LDS, kind_of(op.flags, flags), producer_lane,
                              consumer_of(instruction, range), start,
                              static_cast<uint32_t>(stop - start), false});
        }
      }
    }
    if (retired || wave.pending.size() >= kPendingPruneThreshold)
      wave.prune(scoreboard);
    if (instruction.completion_count) {
      wave.pending.push_back({.pc = instruction.pc,
                              .lo = lo,
                              .hi = hi,
                              .first = static_cast<uint32_t>(wave.ranges.size()),
                              .count = live,
                              .ordered_class = instruction.ordered_class,
                              .flags = flags,
                              .completion_count = instruction.completion_count,
                              .completions = instruction.completions});
      for (const LaneRange &range : lanes)
        if (range.size)
          wave.ranges.push_back(range);
    }

    state->record(self, instruction, flags, lanes, findings);
  }
  report(findings);
}

void MemoryRaceChecker::global_access(const Access &instruction, std::span<const LaneRange> lanes) {
  if (lanes.empty() || (!instruction.read && !instruction.write))
    return;
  const uint8_t flags = flags_of(instruction);
  const uint32_t workgroup = instruction.wave.workgroup;
  const uint64_t budget = global_shadow_budget_.load(std::memory_order_relaxed) / kGlobalShards;
  std::vector<Finding> findings;
  bool degraded = false;
  GlobalShard *shard = nullptr;
  std::unique_lock<std::mutex> lock;
  PageKey key{instruction.wave.dispatch, std::numeric_limits<uint64_t>::max()};
  GlobalPage *page = nullptr;

  for_each_word(lanes, [&](uint64_t index, uint8_t mask, const LaneRange &range) {
    if ((index >> kPageShift) != key.page) {
      key.page = index >> kPageShift;
      GlobalShard &owner = global_[PageKeyHash{}(key) % kGlobalShards];
      if (&owner != shard) {
        // Hold one shard at a time: lanes visit shards in no fixed order.
        if (lock.owns_lock())
          lock.unlock();
        shard = &owner;
        lock = std::unique_lock(owner.mutex);
      }
      page = shard->find(key);
    }
    const uint64_t word = index << 2;
    const size_t slot = index & (kPageWords - 1);
    bool updated = false;
    // Bytes this workgroup already retains with the same access flags, and
    // bytes retained by one or by two distinct other workgroups. Two distinct
    // other owners suffice: any later conflicting workgroup differs from one.
    uint8_t covered = 0;
    uint8_t one = 0;
    uint8_t two = 0;
    std::array<uint32_t, 4> owners{};
    for (uint32_t i = page ? page->heads[slot] : kNoIndex; i != kNoIndex;) {
      GlobalRecord &prior = shard->records[i];
      i = prior.next;
      const uint8_t overlap = prior.bytes & mask;
      if (prior.workgroup == workgroup) {
        if (prior.flags != flags)
          continue;
        if (!updated && (prior.bytes & ~mask) == 0) {
          // Keep the newest site without spending budget on the repeat.
          prior.pc = instruction.pc;
          prior.wave = instruction.wave.wave;
          prior.lane = compact_lane(range.lane);
          prior.bytes = mask;
          updated = true;
        }
        covered |= overlap;
        continue;
      }
      if (!overlap)
        continue;
      if (conflicting(prior.flags, flags)) {
        const uint32_t start = std::countr_zero(overlap);
        const uint32_t length = std::countr_one(static_cast<uint8_t>(overlap >> start));
        Access producer{.wave = {instruction.wave.process, instruction.wave.dispatch,
                                 prior.workgroup, prior.wave, instruction.wave.generation},
                        .pc = prior.pc,
                        .address = word + start,
                        .size = length,
                        .lane = prior.lane};
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
    if (updated || !(mask & ~covered & ~two))
      return;
    if (!page && (page = shard->create(key, budget)) && !tracking_.load(std::memory_order_relaxed))
      tracking_.store(true, std::memory_order_relaxed);
    if (!page || !shard->insert(page->heads[slot],
                                {instruction.pc, workgroup, instruction.wave.wave, kNoIndex,
                                 compact_lane(range.lane), flags, mask},
                                budget))
      degraded = true;
  });
  if (lock.owns_lock())
    lock.unlock();
  if (degraded)
    mark_coverage_degraded();
  report(findings);
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
  std::lock_guard lock(report_mutex_);
  if (report_)
    report_->set_coverage_degraded();
}

void MemoryRaceChecker::retire_completed(const WaveKey &key,
                                         const MemoryWaitScoreboard &scoreboard) {
  const auto state = find_workgroup(key, false);
  if (!state)
    return;
  std::lock_guard lock(state->mutex);
  const uint32_t index = state->find_wave(key);
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
  std::lock_guard lock(state->mutex);
  state->barrier(members);
}

void MemoryRaceChecker::end_workgroup(uint32_t dispatch, uint32_t workgroup) {
  const WorkgroupKey id{dispatch, workgroup};
  auto &shard = workgroups_[WorkgroupKeyHash{}(id) % kWorkgroupShards];
  std::shared_ptr<WorkgroupState> released;
  std::lock_guard lock(shard.mutex);
  if (const auto it = shard.states.find(id); it != shard.states.end()) {
    released = std::move(it->second);
    shard.states.erase(it);
  }
}

void MemoryRaceChecker::end_dispatch(uint32_t dispatch) {
  if (!tracking_.load(std::memory_order_relaxed))
    return;
  for (size_t i = 0; i < kWorkgroupShards; ++i) {
    std::lock_guard lock(workgroups_[i].mutex);
    std::erase_if(workgroups_[i].states,
                  [&](const auto &entry) { return entry.first.dispatch == dispatch; });
  }
  for (size_t i = 0; i < kGlobalShards; ++i) {
    std::lock_guard lock(global_[i].mutex);
    global_[i].erase_dispatch(dispatch);
  }
  std::lock_guard lock(report_mutex_);
  std::erase_if(reported_, [&](const FindingKey &key) { return key.dispatch == dispatch; });
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
  uint64_t bytes = 0;
  for (size_t i = 0; i < kGlobalShards; ++i) {
    std::lock_guard lock(global_[i].mutex);
    bytes += global_[i].bytes();
  }
  return bytes;
}

void MemoryRaceChecker::report(std::vector<Finding> &findings) {
  if (findings.empty())
    return;
  coalesce_findings(findings);
  for (Finding &finding : findings)
    report_one(std::move(finding));
}

void MemoryRaceChecker::report_one(Finding finding) {
  Reporter reporter;
  {
    std::lock_guard lock(report_mutex_);
    const FindingKey key{finding.space,       finding.kind,        finding.consumer.wave.dispatch,
                         finding.producer.pc, finding.consumer.pc, finding.address};
    if (reported_.contains(key))
      return;
    ++finding_count_;
    if (reported_.size() >= kMaxFindings) {
      if (++suppressed_count_ == 1)
        util::Logger::warn("memory-race: further diagnostics are suppressed");
      if (report_)
        report_->add_suppressed_race();
      return;
    }
    reported_.insert(key);
    if (report_)
      report_->add_race(finding);
    reporter = reporter_;
  }
  if (reporter)
    reporter(finding);
}

} // namespace rocjitsu::amdgpu
