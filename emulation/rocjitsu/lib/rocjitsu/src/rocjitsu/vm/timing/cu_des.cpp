// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/cu_des.h"

#include "rocjitsu/vm/timing/classify.h"
#include "rocjitsu/vm/timing/coalesce.h"
#include "rocjitsu/vm/timing/timing_plane.h"
#include "rocjitsu/vm/timing/vocabulary.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::timing {
namespace {

std::uint64_t ceil_div(std::uint64_t numerator, std::uint64_t denominator) {
  if (denominator == 0)
    return numerator;
  return (numerator + denominator - 1) / denominator;
}

std::uint64_t ceil_div_real(double numerator, double denominator) {
  if (!(denominator > 0.0))
    return static_cast<std::uint64_t>(numerator);
  const double quotient = numerator / denominator;
  if (!(quotient > 0.0))
    return 0;
  return static_cast<std::uint64_t>(std::ceil(quotient));
}

/// @brief What one memory instruction cost the wavefront that issued it.
struct AccessCost {
  /// @brief Cycles it occupies its issue port for.
  std::uint64_t issue_cycles = 0;
  /// @brief Cycles from issue until the data is readable.
  std::uint64_t latency_cycles = 0;
  /// @brief First-level misses it left outstanding, against the miss-status
  ///        register pool.
  std::uint32_t outstanding_lines = 0;
};

/// @brief Whether a unit processes the wavefront a SIMD's width at a time.
///
/// @details Keyed on the unit rather than the class because it is a property of
/// the hardware the instruction goes to. A wave wider than the SIMD occupies
/// its unit for several passes; a scalar unit sees one operation however wide
/// the wavefront is.
bool is_lane_parallel(FunctionalUnit unit) {
  switch (unit) {
  case FunctionalUnit::VectorAlu:
  case FunctionalUnit::Transcendental:
  case FunctionalUnit::MatrixMultiply:
  case FunctionalUnit::LocalDataShare:
  case FunctionalUnit::VectorMemory:
  case FunctionalUnit::Export:
    return true;
  case FunctionalUnit::None:
  case FunctionalUnit::ScalarAlu:
  case FunctionalUnit::ScalarMemory:
  case FunctionalUnit::Branch:
  case FunctionalUnit::Count:
    return false;
  }
  return false;
}

/// @brief The counter an operation posts to when the caller did not say.
///
/// @details Only a fallback. Which counter an operation lands on is a
/// per-target ISA decision and the functional compute unit reports the real
/// one; deriving it from the class would park a store's completion in a queue
/// no wait instruction on that target can name, and the wait would then cost
/// nothing.
WaitCounter counter_for_class(InstClass value) {
  switch (value) {
  case InstClass::VectorMemoryWrite:
    return WaitCounter::VectorStore;
  case InstClass::VectorMemoryRead:
  case InstClass::VectorMemoryAtomic:
    return WaitCounter::VectorLoad;
  case InstClass::LdsRead:
  case InstClass::LdsWrite:
  case InstClass::ScalarMemory:
    return WaitCounter::LgkmCombined;
  case InstClass::Export:
    return WaitCounter::Export;
  case InstClass::TensorMemory:
    return WaitCounter::Tensor;
  default:
    return WaitCounter::VectorLoad;
  }
}

std::size_t index_of(InstClass value) { return static_cast<std::size_t>(value); }
std::size_t index_of(FunctionalUnit value) { return static_cast<std::size_t>(value); }
std::size_t index_of(WaitCounter value) { return static_cast<std::size_t>(value); }
std::size_t index_of(MatrixType value) { return static_cast<std::size_t>(value); }
std::size_t index_of(RegisterFile value) { return static_cast<std::size_t>(value); }

std::uint64_t slot_count(const Tuning &tuning) {
  return std::max<std::uint64_t>(1, tuning.wave_slots_per_cu);
}

/// @brief Cycles from issue until @p value's destination registers hold their
///        result, for everything that is not a memory access.
///
/// @details Keyed on the class rather than the unit so the four figures the
/// config names stay one-to-one with the four pipes that have a published or
/// assumed depth. Anything not named falls to the vector figure, matching
/// unit_for_class(), which already puts an unclassified opcode on the vector
/// pipe: a model that costs an instruction on one unit and forwards it from
/// another is describing a machine nobody built.
std::uint64_t result_cycles(const Tuning &tuning, InstClass value) {
  switch (value) {
  case InstClass::ScalarAlu:
  case InstClass::Branch:
  case InstClass::Message:
    return tuning.scalar_alu_result_cycles;
  case InstClass::Transcendental:
    return tuning.transcendental_result_cycles;
  case InstClass::MatrixMultiply:
    return tuning.matrix_multiply_result_cycles;
  default:
    return tuning.vector_alu_result_cycles;
  }
}

/// @brief The most registers of @p file one wavefront can ever name.
///
/// @details The whole file: a wavefront that has its unit to itself is
/// allocated all of it, so this is the point past which an index is not a
/// register the part has. A name reaching beyond it is dropped rather than
/// tracked, which under-charges that one dependency and is the only choice that
/// cannot corrupt a neighbouring wavefront's scoreboard.
std::uint64_t file_limit(const Tuning &tuning, RegisterFile file) {
  switch (file) {
  case RegisterFile::Scalar:
    return std::max<std::uint64_t>(1, tuning.scalar_registers_per_cu);
  case RegisterFile::Vector:
  case RegisterFile::Accumulator:
    // The accumulator registers come out of the same physical file on the
    // targets that have them, so they share its bound.
    return std::max<std::uint64_t>(1, tuning.vector_registers_per_cu);
  case RegisterFile::Count:
    break;
  }
  return 1;
}

/// @brief The registers of @p file a wavefront holds when its unit is full.
///
/// @details Where the scoreboard starts. Every wave slot occupied means the
/// file divided that many ways, so this is what a wavefront names in the common
/// case and sizing for it keeps the scoreboard's footprint proportional to the
/// part rather than to its largest possible allocation.
std::uint64_t file_share(const Tuning &tuning, RegisterFile file) {
  return std::max<std::uint64_t>(1, file_limit(tuning, file) / slot_count(tuning));
}

/// @brief The cycle every source register named by @p ranges holds its value.
std::uint64_t sources_ready(const WaveTimeline &wave, const RegisterRanges &ranges) {
  std::uint64_t ready = 0;
  for (const RegisterRange &range : ranges) {
    const std::vector<std::uint64_t> &file = wave.register_ready[index_of(range.file)];
    // A register past what this wavefront has modelled has never been written
    // by it, so there is nothing to wait for and nothing to read out of bounds.
    const std::size_t last =
        std::min<std::size_t>(file.size(), std::size_t{range.index} + range.count);
    for (std::size_t reg = range.index; reg < last; ++reg)
      ready = std::max(ready, file[reg]);
  }
  return ready;
}

/// @brief Record that @p ranges hold their new value from @p ready_cycle on.
void write_destinations(WaveTimeline &wave, const Tuning &tuning, const RegisterRanges &ranges,
                        std::uint64_t ready_cycle) {
  for (const RegisterRange &range : ranges) {
    std::vector<std::uint64_t> &file = wave.register_ready[index_of(range.file)];
    const std::size_t needed = std::size_t{range.index} + range.count;
    const std::size_t limit = static_cast<std::size_t>(file_limit(tuning, range.file));
    // Grown geometrically and kept across wavefronts, so a kernel wanting more
    // registers than the full-occupancy share pays a handful of reallocations
    // per wave slot for the whole run rather than one per wavefront.
    if (needed > file.size() && file.size() < limit)
      file.resize(std::min(std::max(needed, file.size() * 2), limit), 0);
    const std::size_t last = std::min(file.size(), needed);
    for (std::size_t reg = range.index; reg < last; ++reg)
      file[reg] = ready_cycle;
  }
}

/// @brief The timeline of @p slot, folded into the configured slot count.
WaveTimeline &timeline_for(std::vector<WaveTimeline> &waves, const Tuning &tuning,
                           std::uint32_t slot) {
  return waves[static_cast<std::size_t>(slot % slot_count(tuning))];
}

/// @brief The per-unit totals of every wavefront this unit has already retired.
///
/// @details Kept in one extra WaveTimeline past the addressable slots, because
/// ComputeUnitDes::Occupancy carries the *reduced* issue figure rather than the
/// per-unit breakdown it is reduced from, and the breakdown has to survive
/// between wavefronts to be reduced correctly. Summing each wavefront's own
/// reduction instead would over-charge: a sum of per-unit maxima is never
/// smaller than the maximum of the per-unit sums, and on a kernel that
/// alternates between two ports it is nearly double.
WaveTimeline &retired_totals(std::vector<WaveTimeline> &waves) { return waves.back(); }

/// @brief Scratch for one access's line addresses.
///
/// @details TimingPlane::issue_memory takes a vector, and a vector built per
/// access would put a heap allocation on the hottest path the plane has -- one
/// per vector memory instruction executed, which is more than the emulator
/// spends executing them. Reused per thread instead, so the allocation happens
/// once and the capacity stays.
std::vector<std::uint64_t> &line_scratch() {
  static thread_local std::vector<std::uint64_t> scratch;
  scratch.clear();
  return scratch;
}

/// @brief Lanes the access actually presented to the memory path.
std::uint64_t participating_lanes(const RetiredInstruction &retired) {
  if (retired.active_lanes != 0)
    return retired.active_lanes;
  return std::max<std::uint32_t>(1, retired.wave_lanes);
}

/// @brief Whether the caller handed over a usable per-lane address list.
bool has_addresses(const RetiredInstruction &retired) {
  return retired.lane_addresses != nullptr && !retired.lane_addresses->empty();
}

/// @brief Whether this access touched nothing at all.
///
/// @details The test is the lane address list and *not* the execute mask. The
/// functional compute unit fills one entry per lane set in the memory lane
/// mask, which a predicated or out-of-range access can empty while EXEC is
/// still non-zero. Getting this wrong bills sixty-four divergent DRAM lines for
/// an access that moved no bytes: on one tall-skinny GEMM those accesses were
/// seventy-one per cent of the whole run's modelled cache lines, and the kernel
/// read 85 microseconds against a real 16.9. An access no lane participated in
/// still costs its issue slot, because predication does not make an instruction
/// free.
bool touched_nothing(const RetiredInstruction &retired) {
  return retired.addresses_known && !has_addresses(retired);
}

/// @brief Hand one already-coalesced line set to the plane.
/// @returns Cycles from issue until the access completes.
std::uint64_t submit(TimingPlane &plane, const std::vector<std::uint64_t> &lines,
                     std::uint64_t line_bytes, const RequestOrigin &origin,
                     std::uint64_t issued_tick, std::uint64_t period, bool is_load,
                     bool non_temporal, bool instruction_fetch, MemorySpace space) {
  MemoryRequest request;
  request.origin = origin;
  // The plane owns the line pool, so it is the one that knows where these lines
  // land in it. Only the count and the granularity are the wavefront's to say.
  request.line_base = 0;
  request.line_count = static_cast<std::uint32_t>(lines.size());
  request.line_bytes = static_cast<std::uint32_t>(line_bytes);
  request.is_load = is_load;
  request.non_temporal = non_temporal;
  request.instruction_fetch = instruction_fetch;
  request.space = space;
  request.issued_tick = issued_tick;

  const std::uint64_t completion = plane.issue_memory(request, lines);
  if (completion <= issued_tick || period == 0)
    return 0;
  return ceil_div(completion - issued_tick, period);
}

/// @brief Misses an access left in flight, reconstructed from what it cost.
///
/// @details The plane answers with a completion tick and nothing else, so a hit
/// and a miss are told apart by whether the access came back inside the
/// first-level hit latency. Every line of an access that went further is
/// counted, which over-counts a partial hit; the exact figure needs the plane
/// to report how many lines it could not serve.
std::uint32_t outstanding_from(const Tuning &tuning, std::uint64_t latency_cycles,
                               std::size_t lines) {
  if (latency_cycles <= tuning.l1_vector.hit_cycles)
    return 0;
  return static_cast<std::uint32_t>(lines);
}

/// @brief Cost one vector or tensor access from the addresses its lanes made.
AccessCost vector_access(const Tuning &tuning, TimingPlane &plane, std::uint32_t compute_unit,
                         const RetiredInstruction &retired, const RequestOrigin &origin,
                         std::uint64_t issued_tick, std::uint64_t period) {
  AccessCost cost;
  if (touched_nothing(retired)) {
    cost.issue_cycles = 1;
    cost.latency_cycles = tuning.l1_vector.hit_cycles;
    return cost;
  }

  const std::uint64_t lanes = participating_lanes(retired);
  // The address path is a throughput limit of its own: the wavefront walks
  // every active lane through it whether or not the access coalesces well.
  const std::uint64_t address_cycles =
      ceil_div_real(static_cast<double>(lanes), tuning.lane_addresses_per_cycle);
  const std::uint64_t shift = line_shift_for(tuning.l1_vector.line_bytes);
  const std::uint64_t line_bytes = std::uint64_t{1} << shift;

  std::vector<std::uint64_t> &lines = line_scratch();
  bool non_temporal = retired.non_temporal;
  if (retired.inst_class == InstClass::VectorMemoryAtomic && has_addresses(retired)) {
    // Atomics do not coalesce. Each one is a distinct read-modify-write at the
    // point of coherence, so a wavefront issuing sixty-four of them issues
    // sixty-four operations even when every lane names the same address -- and
    // naming the same address is the common case, because that is what a
    // reduction is. Folding them the way an ordinary access folds makes the
    // cross-workgroup half of every reduction free, which is why the
    // normalization kernels, all of which end in one, read a third fast.
    lines.clear();
    for (std::uint64_t address : *retired.lane_addresses)
      lines.push_back((address >> shift) << shift);
  } else if (has_addresses(retired)) {
    coalesce_lines(*retired.lane_addresses, shift, retired.bytes_per_lane, lines);
  } else {
    // The access happened and the addresses did not survive. Charge it as fully
    // divergent all the way to memory. Guessing cheap here is the failure that
    // looks exactly like accuracy: it costs nothing and silently deletes real
    // traffic. The synthetic addresses are made distinct per access and marked
    // non-temporal so they miss every level without leaving lines no kernel
    // ever touched resident in the tag arrays.
    non_temporal = true;
    const std::uint64_t base = (static_cast<std::uint64_t>(compute_unit) << 44) |
                               (static_cast<std::uint64_t>(origin.sequence) << 20);
    std::array<std::uint64_t, kMaxLanes> synthetic{};
    const std::size_t count = static_cast<std::size_t>(std::min<std::uint64_t>(lanes, kMaxLanes));
    for (std::size_t lane = 0; lane < count; ++lane)
      synthetic[lane] = base | (static_cast<std::uint64_t>(lane) << shift);
    coalesce_lines(std::span<const std::uint64_t>(synthetic.data(), count), shift,
                   retired.bytes_per_lane, lines);
  }

  cost.latency_cycles = submit(plane, lines, line_bytes, origin, issued_tick, period,
                               retired.is_load, non_temporal, false, MemorySpace::Global);
  cost.issue_cycles = std::max(address_cycles, ceil_div_real(static_cast<double>(lines.size()),
                                                             tuning.l1_vector.lines_per_cycle));
  cost.outstanding_lines = outstanding_from(tuning, cost.latency_cycles, lines.size());
  return cost;
}

/// @brief Cost one scalar (constant path) access.
AccessCost scalar_access(const Tuning &tuning, TimingPlane &plane,
                         const RetiredInstruction &retired, const RequestOrigin &origin,
                         std::uint64_t issued_tick, std::uint64_t period) {
  const std::uint64_t bytes = std::max<std::uint32_t>(1, retired.scalar_bytes);
  const std::uint64_t shift = line_shift_for(tuning.l1_scalar.line_bytes);
  const std::uint64_t line_bytes = std::uint64_t{1} << shift;
  const std::uint64_t mask = ~(line_bytes - 1);

  std::vector<std::uint64_t> &lines = line_scratch();
  const std::uint64_t first = retired.scalar_address & mask;
  const std::uint64_t last = (retired.scalar_address + bytes - 1) & mask;
  for (std::uint64_t address = first; address <= last; address += line_bytes)
    lines.push_back(address);

  AccessCost cost;
  cost.latency_cycles = submit(plane, lines, line_bytes, origin, issued_tick, period,
                               retired.is_load, retired.non_temporal, false, MemorySpace::Scalar);
  // One operation on the scalar unit however many lines it spans: the scalar
  // path has no lanes to walk.
  cost.issue_cycles = 1;
  cost.outstanding_lines = outstanding_from(tuning, cost.latency_cycles, lines.size());
  return cost;
}

/// @brief Cost one local data share access, conflicts included.
///
/// @details The local data share never leaves the compute unit, so this is the
/// one access the plane does not see.
AccessCost lds_access(const Tuning &tuning, const RetiredInstruction &retired) {
  AccessCost cost;
  if (touched_nothing(retired)) {
    cost.issue_cycles = 1;
    cost.latency_cycles = tuning.lds_latency_cycles;
    return cost;
  }

  const std::uint64_t banks = std::max<std::uint64_t>(1, tuning.lds_banks);
  if (!has_addresses(retired)) {
    // Charge the widest conflict the wavefront could have produced, and charge
    // it *here* rather than in global memory: losing the addresses does not
    // lose the space, and a blocked kernel stages tiles through the local data
    // share precisely to keep them off DRAM.
    const std::uint64_t phases = ceil_div(participating_lanes(retired),
                                          std::max<std::uint64_t>(1, tuning.lds_lanes_per_phase));
    cost.issue_cycles = std::max<std::uint64_t>(1, phases) * banks;
  } else {
    cost.issue_cycles = lds_conflict_cycles(*retired.lane_addresses, retired.bytes_per_lane, banks,
                                            tuning.lds_bank_bytes, tuning.lds_lanes_per_phase);
  }
  // The conflict passes happen after the access reaches the array, so they
  // extend the return rather than overlapping it.
  cost.latency_cycles = tuning.lds_latency_cycles + cost.issue_cycles;
  return cost;
}

/// @brief Cost fetching the instruction at @p program_counter.
/// @returns Zero when the line was already resident.
std::uint64_t fetch_cycles(const Tuning &tuning, TimingPlane &plane, std::uint64_t program_counter,
                           const RequestOrigin &origin, std::uint64_t issued_tick,
                           std::uint64_t period) {
  const std::uint64_t shift = line_shift_for(tuning.l1_instruction.line_bytes);
  std::vector<std::uint64_t> &lines = line_scratch();
  lines.push_back((program_counter >> shift) << shift);
  return submit(plane, lines, std::uint64_t{1} << shift, origin, issued_tick, period, true, false,
                true, MemorySpace::Global);
}

/// @brief Advance @p wave to @p completion, recording the wait as a stall.
void wait_until(WaveTimeline &wave, std::uint64_t completion) {
  if (completion <= wave.cycle)
    return;
  wave.stall_cycles += completion - wave.cycle;
  wave.cycle = completion;
}

/// @brief Retire misses that have already landed, freeing their registers.
///
/// @details A miss whose completion is behind the wavefront's own cycle has
/// released its miss-status register whether or not a wait named it, so this
/// runs on every instruction rather than only inside a wait. It never moves
/// time: an entry that has already completed cannot stall anything.
void release_landed_misses(WaveTimeline &wave) {
  while (!wave.misses.empty() && wave.misses.front().first <= wave.cycle) {
    wave.outstanding_lines -= std::min(wave.outstanding_lines, wave.misses.front().second);
    wave.misses.pop_front();
  }
}

} // namespace

void WaveTimeline::reset() {
  cycle = 0;
  stall_cycles = 0;
  unit_cycles.fill(0);
  for (auto &queue : outstanding)
    queue.clear();
  misses.clear();
  outstanding_lines = 0;
  // Zeroed rather than emptied: the storage was sized for this compute unit and
  // the next wavefront in this slot needs exactly as much of it.
  for (std::vector<std::uint64_t> &file : register_ready)
    std::fill(file.begin(), file.end(), 0);
  instructions = 0;
  class_counts.fill(0);
  fetch_exposed = false;
  last_unit = 0xFFFFFFFFu;
  live = false;
}

ComputeUnitDes::ComputeUnitDes(std::string name, const simdojo::ClockDomain &domain,
                               TimingEngine &engine, const Tuning &tuning, TimingPlane &plane,
                               std::uint32_t index)
    : TimedComponent(std::move(name), domain, engine), tuning_(tuning), plane_(plane),
      index_(index) {
  // One entry per addressable wave slot, plus the retired-totals accumulator
  // retired_totals() reads. See its comment for why that has to live somewhere.
  waves_.resize(static_cast<std::size_t>(slot_count(tuning)) + 1);
  for (WaveTimeline &wave : waves_) {
    for (std::size_t file = 0; file < kNumRegisterFiles; ++file) {
      const RegisterFile which = static_cast<RegisterFile>(file);
      wave.register_ready[file].assign(static_cast<std::size_t>(file_share(tuning, which)), 0);
    }
  }
}

std::uint64_t ComputeUnitDes::advance(std::uint64_t now) {
  (void)now;
  // A compute unit is a timed component because it belongs to a clock domain
  // and can be woken, not because it waits on an inbox: the plane answers what
  // an access cost at the point the access is made, and a wavefront's timeline
  // advances there. Anything delivered here anyway is dropped rather than left
  // to accumulate, and the unit schedules nothing of its own.
  inbox().clear();
  return 0;
}

void ComputeUnitDes::wave_begin(std::uint32_t slot) {
  WaveTimeline &wave = timeline_for(waves_, tuning_, slot);
  wave.reset();
  wave.live = true;
}

void ComputeUnitDes::instruction(std::uint32_t slot, const RetiredInstruction &retired) {
  WaveTimeline &wave = timeline_for(waves_, tuning_, slot);
  if (!wave.live) {
    // A wavefront can reach the unit before anyone announced it. Starting it
    // here keeps its work attributed rather than dropped.
    wave.reset();
    wave.live = true;
  }

  const InstClass inst_class = retired.inst_class;
  const FunctionalUnit unit = unit_for_class(inst_class);

  // -- Waits -----------------------------------------------------------------
  // A wait names a threshold per counter, not a count of operations to retire.
  // Draining until at most the threshold remains is what makes a partial wait
  // -- `s_waitcnt vmcnt(1)`, the shape every software-pipelined loop uses --
  // cost the right thing: the wavefront waits for the older load and keeps the
  // newer one in flight.
  //
  // Only a wait instruction's thresholds are read. RetiredInstruction's array
  // is value-initialised rather than filled with kUnconstrained, and a zero
  // threshold means "drain this counter completely", so honouring the array on
  // every instruction would make an ordinary v_add retire every load the
  // wavefront had in flight.
  if (inst_class == InstClass::WaitCounter) {
    for (std::size_t counter = 0; counter < kNumWaitCounters; ++counter) {
      const std::uint32_t threshold = retired.wait[counter];
      if (threshold == kUnconstrained)
        continue;
      std::deque<std::uint64_t> &queue = wave.outstanding[counter];
      while (queue.size() > threshold) {
        const std::uint64_t completion = queue.front();
        queue.pop_front();
        wait_until(wave, completion);
      }
    }
  }

  // -- Register dependencies -------------------------------------------------
  // A wavefront cannot issue an instruction whose sources are still being
  // written, however free the port it wants is. This is the term that keeps a
  // compute unit off peak issue throughput: without it the model describes a
  // machine where every result is readable the cycle after it is produced, and
  // it is wrong in proportion to instruction density, which is why the densest
  // categories were the ones reading fastest.
  wait_until(wave, sources_ready(wave, retired.reads));
  release_landed_misses(wave);

  // -- Instruction fetch -----------------------------------------------------
  // Charged before issue because that is where it happens: a wavefront cannot
  // issue an instruction it has not fetched. The fill's bandwidth is charged by
  // the plane on every miss and is always real; its *latency* is exposed only
  // once per wavefront, at the cold start. The fetch unit runs several lines
  // ahead of issue, so a miss found while the wavefront still has buffered
  // instructions costs bandwidth and no time. Exposing every miss instead took
  // the reference corpus from a median ratio of 0.65 to 2.04 on its own.
  const RequestOrigin origin{index_, slot, static_cast<std::uint32_t>(wave.instructions)};
  // The only clock a wavefront has is its own, which starts at zero. That is
  // the honest tick to hand over: deriving one from a device-wide elapsed time
  // is what made every early access read a saturated memory system and take a
  // twentyfold stretch.
  const std::uint64_t issued_tick = cycles_to_ticks(wave.cycle);
  if (const std::uint64_t fetch =
          fetch_cycles(tuning_, plane_, retired.pc, origin, issued_tick, period());
      fetch != 0 && !wave.fetch_exposed) {
    wave.fetch_exposed = true;
    wave.stall_cycles += fetch;
    wave.cycle += fetch;
  }

  // -- Issue occupancy -------------------------------------------------------
  // Everything the instruction writes is dated from here, once the wavefront
  // has waited out its sources and fetched the instruction.
  const std::uint64_t issue_cycle = wave.cycle;
  std::uint64_t result_ready = issue_cycle + result_cycles(tuning_, inst_class);
  const std::uint64_t lanes = std::max<std::uint32_t>(1, retired.wave_lanes);
  const std::uint64_t passes =
      is_lane_parallel(unit) ? ceil_div(lanes, std::max<std::uint64_t>(1, tuning_.simd_lanes)) : 1;
  std::uint64_t occupancy =
      tuning_.issue_cycles[index_of(inst_class)] * std::max<std::uint64_t>(1, passes);

  if (inst_class == InstClass::MatrixMultiply && retired.mnemonic != nullptr) {
    // The matrix shape already covers the whole wavefront, so a matrix
    // instruction is charged one pass rather than one per SIMD width, and the
    // rate is the one its *input* element type selects.
    if (const std::uint64_t macs = matrix_macs(*retired.mnemonic); macs != 0) {
      const std::uint64_t rate = std::max<std::uint64_t>(
          1, tuning_.matrix_macs_per_cycle[index_of(matrix_type_of(*retired.mnemonic))]);
      occupancy = ceil_div(macs, rate);
    }
  }

  // -- Memory ----------------------------------------------------------------
  if (retired.space != MemorySpace::None) {
    AccessCost cost;
    switch (retired.space) {
    case MemorySpace::Global:
    case MemorySpace::Tensor:
      cost = vector_access(tuning_, plane_, index_, retired, origin, issued_tick, period());
      break;
    case MemorySpace::Scalar:
      cost = scalar_access(tuning_, plane_, retired, origin, issued_tick, period());
      break;
    case MemorySpace::LocalDataShare:
      cost = lds_access(tuning_, retired);
      break;
    case MemorySpace::None:
      break;
    }
    occupancy = std::max(occupancy, cost.issue_cycles);

    // A compute unit has a finite number of miss-status registers, and a
    // wavefront that runs out cannot issue another miss until one frees. This
    // is the term that turns a divergent access from bandwidth limited into
    // latency limited.
    //
    // The cap is checked against what was *already* outstanding, never against
    // this instruction's own lines. One memory instruction's misses are
    // concurrent by construction -- that is what coalescing and miss merging
    // are for -- so charging a wavefront a full round trip for issuing a single
    // divergent access would serialise the one thing the hardware overlaps.
    if (cost.outstanding_lines != 0) {
      const std::uint32_t cap = static_cast<std::uint32_t>(std::min<std::uint64_t>(
          std::max<std::uint64_t>(1, tuning_.miss_status_registers_per_cu), ~std::uint32_t{0}));
      while (wave.outstanding_lines > cap && !wave.misses.empty()) {
        const std::pair<std::uint64_t, std::uint32_t> oldest = wave.misses.front();
        wave.misses.pop_front();
        wave.outstanding_lines -= std::min(wave.outstanding_lines, oldest.second);
        wait_until(wave, oldest.first);
      }
      wave.misses.emplace_back(wave.cycle + cost.latency_cycles, cost.outstanding_lines);
      wave.outstanding_lines += cost.outstanding_lines;
    }

    const WaitCounter counter = retired.wait_counter != WaitCounter::Count
                                    ? retired.wait_counter
                                    : counter_for_class(inst_class);
    const std::uint64_t completion = wave.cycle + cost.latency_cycles;
    wave.outstanding[index_of(counter)].push_back(completion);
    // An access's destination registers are not ready a fixed pipeline depth
    // after issue; they are ready when the access completes. That is what makes
    // a dependent load chain cost round trips instead of issue slots, and it is
    // the same cycle the wait counter carries, so a wavefront that does wait
    // for it -- which every compiler emits -- is charged once and not twice.
    result_ready = completion;
  }

  write_destinations(wave, tuning_, retired.writes, result_ready);

  // The unit is occupied for the whole of `occupancy`; the wavefront is held
  // only until it may issue again, which is the same thing only when its next
  // instruction wants the same unit. Separating the two is what stops a mixed
  // instruction stream being serialised on the critical path when the hardware
  // pipelines it.
  const std::uint32_t unit_index = static_cast<std::uint32_t>(index_of(unit));
  const std::uint64_t reissue =
      (tuning_.wavefront_issue_cycles == 0 || unit_index == wave.last_unit)
          ? occupancy
          : std::min(occupancy, tuning_.wavefront_issue_cycles);
  wave.last_unit = unit_index;
  wave.cycle += reissue;
  wave.unit_cycles[index_of(unit)] += occupancy;
  // Every instruction also occupies a front-end issue slot, whatever functional
  // unit it then goes to. A compute unit can start a bounded number of
  // instructions per cycle across all its units, so a kernel whose work is
  // spread thinly over many units is limited by the front end rather than by
  // any one of them, and a scalar or branch or wait instruction is not free
  // just because nothing else wanted the scalar unit that cycle.
  //
  // Leaving it out is an under-charge proportional to instruction count, which
  // is how it presented: the most instruction-dense categories read fastest,
  // normalization at 0.61 of measured and matrix kernels at 0.72, against
  // streaming kernels near 0.88 where the memory system sets the pace anyway.
  //
  // Charged per class, in sixteenths of a cycle: a matrix instruction and a
  // lane-parallel add do not occupy the same issue resources for the same
  // time, and one shared figure for all of them is what made the model's
  // accuracy a direct median-versus-maximum trade.
  wave.unit_cycles[index_of(FunctionalUnit::None)] +=
      tuning_.front_end_sixteenths[static_cast<std::size_t>(inst_class)];
  ++wave.instructions;
  wave.class_counts[static_cast<std::size_t>(inst_class)] += 1;
}

void ComputeUnitDes::barrier(const std::vector<std::uint32_t> &slots) {
  // Every wavefront in the group leaves the barrier at the cycle the slowest
  // one reached, so the barrier costs the spread. Knowable only for the group,
  // which is why it arrives as one.
  std::uint64_t latest = 0;
  for (const std::uint32_t slot : slots)
    latest = std::max(latest, timeline_for(waves_, tuning_, slot).cycle);
  for (const std::uint32_t slot : slots)
    wait_until(timeline_for(waves_, tuning_, slot), latest);
}

void ComputeUnitDes::wave_end(std::uint32_t slot) {
  WaveTimeline &wave = timeline_for(waves_, tuning_, slot);
  if (!wave.live)
    return;

  // A wavefront is not finished when its last instruction issues: anything
  // still in flight has to land before it retires, and the terminal instruction
  // waits for exactly that.
  std::uint64_t end = wave.cycle;
  for (const std::deque<std::uint64_t> &queue : wave.outstanding)
    for (const std::uint64_t completion : queue)
      end = std::max(end, completion);

  WaveTimeline &totals = retired_totals(waves_);
  for (std::size_t unit = 0; unit < kNumFunctionalUnits; ++unit)
    totals.unit_cycles[unit] += wave.unit_cycles[unit];
  for (std::size_t cls = 0; cls < kNumInstClasses; ++cls)
    occupancy_.class_counts[cls] += wave.class_counts[cls];

  // The work the unit has to retire is the busiest port's queue, and a port
  // that issues several operations per cycle drains its queue that much faster.
  std::uint64_t issue = 0;
  for (std::size_t unit = 0; unit < kNumFunctionalUnits; ++unit) {
    // The front end's queue is carried in sixteenths of a cycle, everything
    // else in whole ones.
    const double scale =
        unit == index_of(FunctionalUnit::None) ? static_cast<double>(kFrontEndScale) : 1.0;
    const double rate = std::max(1e-9, tuning_.ports[unit]) * scale;
    issue = std::max(issue, static_cast<std::uint64_t>(
                                std::ceil(static_cast<double>(totals.unit_cycles[unit]) / rate)));
  }
  occupancy_.issue_cycles = issue;
  occupancy_.unit_cycles = totals.unit_cycles;
  occupancy_.critical_path_cycles = std::max(occupancy_.critical_path_cycles, end);
  occupancy_.stall_cycles += wave.stall_cycles;
  occupancy_.worst_stall_cycles = std::max(occupancy_.worst_stall_cycles, wave.stall_cycles);
  occupancy_.instructions += wave.instructions;
  ++occupancy_.waves;

  wave.reset();
}

void ComputeUnitDes::reset_occupancy() {
  occupancy_ = Occupancy{};
  retired_totals(waves_).reset();
}

} // namespace rocjitsu::timing
