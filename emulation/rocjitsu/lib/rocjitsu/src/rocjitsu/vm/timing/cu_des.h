// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cu_des.h
/// @brief The timed compute unit: per-wavefront timelines over issue ports.
///
/// @details One of these per modelled compute unit. It is fed by the functional
/// compute unit through a direct call at instruction retire -- not through the
/// execution-plugin interface, which is for observers and would make timing one
/// tool among several rather than part of the machine.
///
/// Every wavefront gets a cycle counter. It advances as instructions issue and
/// stops when an `s_waitcnt` names a completion that has not arrived. That is
/// the whole difference from a bucket model: a stall is not work, so a model
/// with no time axis cannot represent one, and latency exposure is what decides
/// every kernel too small to saturate the part -- which is most of a realistic
/// corpus.
///
/// The compute unit's own duration is then the larger of two things: the work
/// it has to retire through its ports, and the longest single wavefront it
/// held. Below occupancy the second wins and the kernel is latency bound; above
/// it the first wins and the kernel is throughput bound. A unit holding more
/// wavefronts than it has room for runs them in rounds, and pays its longest
/// wavefront once per round.

#pragma once

#include "rocjitsu/vm/timing/cache_des.h"
#include "rocjitsu/vm/timing/timed_component.h"
#include "rocjitsu/vm/timing/tuning.h"
#include "rocjitsu/vm/timing/vocabulary.h"

#include <array>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace rocjitsu::timing {

class TimingPlane;

/// @brief The register files a dependency can name.
///
/// @details Only the three general-purpose files. The condition code, the exec
/// mask and the mode registers are deliberately absent: the scalar unit
/// forwards them within a single cycle, so tracking them would put a lookup on
/// the hottest path in the simulator for a stall the hardware does not have.
enum class RegisterFile : std::uint8_t {
  Scalar,      ///< Scalar general-purpose registers.
  Vector,      ///< Vector general-purpose registers.
  Accumulator, ///< Accumulator vector registers, where the target has them.
  Count,       ///< Number of files; not a file itself.
};

/// @brief Number of register files, for the per-file arrays a wavefront keeps.
inline constexpr std::size_t kNumRegisterFiles = static_cast<std::size_t>(RegisterFile::Count);

/// @brief A contiguous run of registers one operand names.
///
/// @details @ref count is in 32-bit lanes, the granularity the decoder reports
/// and the granularity a dependency is tracked at: a 64-bit address pair is a
/// count of two, and a matrix accumulator can be sixteen.
struct RegisterRange {
  RegisterFile file = RegisterFile::Vector;
  std::uint16_t index = 0;
  std::uint8_t count = 1;
};

/// @brief The register ranges one instruction names, in one direction.
///
/// @details Fixed capacity, not a std::vector, because this rides on the
/// per-instruction path of the whole simulator, where one heap allocation per
/// instruction costs more than executing the instruction does. The capacity is
/// the widest operand array rocjitsu's decoder has, so nothing an instruction
/// names is dropped for want of room.
struct RegisterRanges {
  /// @brief Widest source operand array the decoder exposes.
  static constexpr std::size_t kCapacity = 6;

  std::array<RegisterRange, kCapacity> entries{};
  std::uint8_t count = 0;

  /// @brief Append @p range, or drop it if the instruction named more ranges
  ///        than the decoder can hold.
  void push(const RegisterRange &range) {
    if (count < kCapacity)
      entries[count++] = range;
  }

  const RegisterRange *begin() const { return entries.data(); }
  const RegisterRange *end() const { return entries.data() + count; }
  bool empty() const { return count == 0; }
};

/// @brief What the functional compute unit reports for one instruction.
///
/// @details Deliberately a plain struct with no simulator types in it, so the
/// compute unit can be driven from a test with hand-built sequences. Most of
/// what is worth testing about a timing model is how stalls compose, and
/// building the exact sequence tests that far more precisely than hoping a
/// compiled kernel reaches the case.
struct RetiredInstruction {
  std::uint64_t pc = 0;
  InstClass inst_class = InstClass::Unknown;
  const std::string *mnemonic = nullptr;
  std::uint32_t active_lanes = 0;
  std::uint32_t wave_lanes = 64;
  /// @brief Thresholds an `s_waitcnt` waits down to; kUnconstrained for none.
  std::array<std::uint32_t, kNumWaitCounters> wait{};
  /// @brief Per-lane byte addresses, in lane order, for a vector access. Empty
  ///        when no lane participated, which is not the same as unrecoverable.
  const std::vector<std::uint64_t> *lane_addresses = nullptr;
  std::uint32_t bytes_per_lane = 4;
  MemorySpace space = MemorySpace::None;
  bool is_load = true;
  bool non_temporal = false;
  bool addresses_known = true;
  std::uint64_t scalar_address = 0;
  std::uint32_t scalar_bytes = 4;
  WaitCounter wait_counter = WaitCounter::Count;
  /// @brief Registers the instruction reads. Its issue cannot happen before the
  ///        last of them has been written.
  RegisterRanges reads;
  /// @brief Registers it writes, which are readable only once its result lands.
  RegisterRanges writes;
};

/// @brief One wavefront's modelled timeline.
struct WaveTimeline {
  std::uint64_t cycle = 0;
  std::uint64_t stall_cycles = 0;
  std::array<std::uint64_t, kNumFunctionalUnits> unit_cycles{};
  /// @brief Completion cycles of operations still outstanding, per counter, in
  ///        issue order. An `s_waitcnt N` drains until at most N remain.
  std::array<std::deque<std::uint64_t>, kNumWaitCounters> outstanding;
  std::deque<std::pair<std::uint64_t, std::uint32_t>> misses;
  std::uint32_t outstanding_lines = 0;
  /// @brief Cycle each architectural register becomes readable, per file.
  ///
  /// @details The scoreboard. A wavefront cannot issue an instruction whose
  /// sources are still being written, and without this the compute unit runs at
  /// peak issue throughput, which no pipeline does.
  ///
  /// Sized by the compute unit from the tuning and grown only as far as a
  /// wavefront actually names, so a kernel living in sixty-four vector
  /// registers does not pay for a file it never touches. reset() zeroes the
  /// entries and keeps the storage, because the next wavefront to land in this
  /// slot is nearly always the same kernel again.
  std::array<std::vector<std::uint64_t>, kNumRegisterFiles> register_ready;
  std::uint64_t instructions = 0;
  /// @brief Instructions retired per class, for the per-dispatch trace.
  ///
  /// @details Carried on the wavefront rather than counted centrally so that
  /// the count follows the wavefront's own accounting: a wavefront whose slot
  /// is reused mid-dispatch contributes what it retired and nothing more.
  std::array<std::uint64_t, kNumInstClasses> class_counts{};
  bool fetch_exposed = false;
  /// @brief Functional unit the previous instruction went to, so that two in a
  ///        row wanting the same one serialise and two wanting different ones
  ///        do not.
  std::uint32_t last_unit = 0xFFFFFFFFu;
  bool live = false;

  void reset();
};

/// @brief The timed compute unit.
class ComputeUnitDes final : public TimedComponent {
public:
  ComputeUnitDes(std::string name, const simdojo::ClockDomain &domain, TimingEngine &engine,
                 const Tuning &tuning, TimingPlane &plane, std::uint32_t index);

  std::uint64_t advance(std::uint64_t now) override;

  void wave_begin(std::uint32_t slot);
  void instruction(std::uint32_t slot, const RetiredInstruction &retired);
  void wave_end(std::uint32_t slot);
  /// @brief Every wavefront in a group leaves a barrier at the cycle the
  ///        slowest one reached. Knowable only for the group, which is why it
  ///        arrives as one.
  void barrier(const std::vector<std::uint32_t> &slots);

  /// @brief Cycles this unit was occupied for the dispatch just finished, and
  ///        the terms that produced it.
  struct Occupancy {
    std::uint64_t issue_cycles = 0;
    std::uint64_t critical_path_cycles = 0;
    std::uint64_t waves = 0;
    std::uint64_t instructions = 0;
    std::uint64_t stall_cycles = 0;
    /// @brief Stall cycles of the single worst wavefront, not the sum.
    ///
    /// @details The sum says how much waiting the unit did in total; this says
    /// how much of it one wavefront could not avoid, which is the part that
    /// other wavefronts have to cover if it is to be hidden at all.
    std::uint64_t worst_stall_cycles = 0;
    /// @brief Cycles this unit's wavefronts spent on each functional unit.
    ///
    /// @details Reported rather than folded away because issue_cycles is
    /// already the maximum over the ports and cannot be taken apart again. A
    /// trace that names which unit the issue term came from is the difference
    /// between knowing a dispatch is issue bound and knowing what to change.
    std::array<std::uint64_t, kNumFunctionalUnits> unit_cycles{};
    /// @brief Instructions retired per class.
    std::array<std::uint64_t, kNumInstClasses> class_counts{};
  };
  const Occupancy &occupancy() const { return occupancy_; }
  void reset_occupancy();

  std::uint32_t index() const { return index_; }

private:
  const Tuning &tuning_;
  TimingPlane &plane_;
  std::uint32_t index_ = 0;
  std::vector<WaveTimeline> waves_;
  Occupancy occupancy_;
};

} // namespace rocjitsu::timing
