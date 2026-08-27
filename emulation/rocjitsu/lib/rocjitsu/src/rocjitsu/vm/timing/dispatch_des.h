// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dispatch_des.h
/// @brief One dispatch's cost, and the bandwidth ledger it is charged against.
///
/// @details A dispatch is the unit at which the guest can observe time, so it
/// is the unit at which the plane composes one. The composition is three
/// questions asked of the same run, and the largest answer wins:
///
///   How much work has to go through the compute units' issue ports, and how
///   long was the longest single wavefront any of them held? The first is the
///   throughput bound, the second the latency bound, and which of them wins is
///   what separates a kernel that saturates the part from the much larger set
///   that does not.
///
///   How many bytes crossed each level of the memory hierarchy, and how long
///   did the busiest instance of the busiest level need to move them?
///
///   Could the command processor even place the grid that fast? A grid larger
///   than the placement rate can start in the time the work takes is placement
///   bound however little each workgroup does.
///
/// The bandwidth ledger is *per dispatch*, and that is not an implementation
/// detail. A device-wide ledger is wrong in both directions at once: a dispatch
/// that follows a heavy one inherits its bytes, and two dispatches on separate
/// streams each see the other's traffic as their own. Both were measured -- a
/// 236-instruction kernel costed at 52 microseconds because its predecessor had
/// moved 1.7 million instructions' worth of bytes, and a two-stream case costed
/// at seventeen times its real duration.

#pragma once

#include "rocjitsu/vm/timing/cu_des.h"
#include "rocjitsu/vm/timing/timed_component.h"
#include "rocjitsu/vm/timing/timing_plane.h"
#include "rocjitsu/vm/timing/tuning.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu::timing {

/// @brief Running byte accounting for one level of the hierarchy, per instance.
///
/// @details Instances are memory channels for the fabric, the memory-side cache
/// and the DRAM interface, dies for the second-level cache, and compute units
/// for the first. Keeping them separate is what lets the plane say that a
/// kernel hashing all its traffic onto one channel does not get the whole
/// part's bandwidth; an aggregate rate cannot represent that at all.
class BandwidthLevel {
public:
  /// @brief Size the level and split @p bytes_per_cycle_total across it.
  void configure(std::uint64_t instances, double bytes_per_cycle_total);

  /// @brief Charge @p bytes to the instance serving @p index.
  void charge(std::uint64_t index, std::uint64_t bytes);

  /// @brief Cycles the busiest instance is occupied for.
  std::uint64_t busiest_cycles() const;
  /// @brief Bytes charged across every instance.
  std::uint64_t total_bytes() const;
  void reset();

  std::uint64_t instances() const { return instances_; }

private:
  /// @brief One instance's byte count, on its own cache line so that compute
  ///        units charging concurrently do not share one.
  struct alignas(64) Instance {
    std::atomic<std::uint64_t> bytes{0};
  };

  std::uint64_t instances_ = 1;
  double bytes_per_cycle_per_instance_ = 1.0;
  std::vector<Instance> instance_;
};

/// @brief Bytes one dispatch moved, at every level of the hierarchy.
class BandwidthLedger {
public:
  BandwidthLedger() = default;
  BandwidthLedger(const BandwidthLedger &) = delete;
  BandwidthLedger &operator=(const BandwidthLedger &) = delete;

  void configure(const Tuning &tuning);
  void reset();

  void charge_l1(std::uint64_t compute_unit, std::uint64_t bytes) {
    l1_.charge(compute_unit, bytes);
  }
  void charge_l2(std::uint64_t xcd, std::uint64_t bytes) { l2_.charge(xcd, bytes); }
  void charge_fabric(std::uint64_t channel, std::uint64_t bytes) { fabric_.charge(channel, bytes); }
  void charge_mall(std::uint64_t channel, std::uint64_t bytes) { mall_.charge(channel, bytes); }
  void charge_dram(std::uint64_t channel, std::uint64_t bytes) { dram_.charge(channel, bytes); }
  /// @brief Charge a channel cycles that move no bytes.
  ///
  /// @details DRAM row activations. The bound below is bytes divided by a rate,
  /// which cannot express a cost that occupies a channel without transferring
  /// anything, and a row activation is exactly that. Without a place to put it,
  /// everything the timed channel modelled about row locality was computed and
  /// then dropped on the way to the answer.
  void charge_dram_cycles(std::uint64_t channel, std::uint64_t cycles);

  /// @brief Cycles the busiest resource anywhere in the hierarchy was occupied.
  std::uint64_t bound_cycles() const;
  /// @brief Which level that was, as a report name.
  const char *bound_level() const;

  std::uint64_t l1_cycles() const { return l1_.busiest_cycles(); }
  std::uint64_t l2_cycles() const { return l2_.busiest_cycles(); }
  std::uint64_t fabric_cycles() const { return fabric_.busiest_cycles(); }
  std::uint64_t mall_cycles() const { return mall_.busiest_cycles(); }
  std::uint64_t dram_cycles() const { return dram_.busiest_cycles(); }

  std::uint64_t l1_bytes() const { return l1_.total_bytes(); }
  std::uint64_t l2_bytes() const { return l2_.total_bytes(); }
  std::uint64_t fabric_bytes() const { return fabric_.total_bytes(); }
  std::uint64_t mall_bytes() const { return mall_.total_bytes(); }
  std::uint64_t dram_bytes() const { return dram_.total_bytes(); }

private:
  BandwidthLevel l1_;
  BandwidthLevel l2_;
  BandwidthLevel fabric_;
  BandwidthLevel mall_;
  BandwidthLevel dram_;
  /// @brief Cycles per channel that moved no bytes; see charge_dram_cycles().
  std::vector<std::uint64_t> dram_extra_;
};

/// @brief The timed dispatch: one kernel launch, from its packet to its signal.
class DispatchDes final : public TimedComponent {
public:
  /// @param units Every timed compute unit the plane owns. The dispatch reads
  ///        their occupancy at end() and clears it at begin(), which is why it
  ///        holds them rather than being told about them.
  DispatchDes(std::string name, const simdojo::ClockDomain &domain, TimingEngine &engine,
              const Tuning &tuning, std::vector<ComputeUnitDes *> units);

  std::uint64_t advance(std::uint64_t now) override;

  /// @brief Take the dispatch's shape and clear what the last one accumulated.
  ///
  /// @details Called when the packet is parsed, which is *not* when the kernel
  /// starts running: the command processor parses ahead. Nothing here may
  /// depend on execution having begun; see claim_acquire() for the part that
  /// does.
  void begin(const DispatchShape &shape);

  /// @brief Claim the launch acquire for the first wavefront that executes.
  /// @returns true exactly once per dispatch, for the caller that must then
  ///          invalidate every first-level cache and drop the second level.
  ///
  /// @details Split from begin() because the *moment* matters as much as the
  /// invalidate does. Driving it from the packet produced a kernel that
  /// alternated between 3.378 and 1.724 microseconds on identical work: the
  /// invalidate sometimes landed before the kernel's own execution and
  /// sometimes after, and when it landed after, the kernel ran with its
  /// predecessor's lines still resident.
  bool claim_acquire();

  /// @brief Clear what the previous dispatch left on the compute units.
  ///
  /// @details Called with the launch acquire, for the same reason: the moment
  /// a dispatch starts executing, not the moment its packet was parsed.
  void clear_units();

  /// @brief Compose the dispatch's duration from everything it accumulated.
  void end();

  /// @brief Every term end() composed, kept rather than discarded.
  ///
  /// @details The composed cycle count answers "how long", and nothing else.
  /// Which of the terms produced it, how far behind the others were, and what
  /// the compute units were actually doing are all questions a tuning run has
  /// to answer, and re-deriving them from the total is not possible. Recording
  /// them costs one struct copy per dispatch and turns a ten-minute corpus
  /// re-run into an offline recomputation.
  struct Terms {
    std::uint64_t issue = 0;
    std::uint64_t bandwidth = 0;
    std::uint64_t placement = 0;
    std::uint64_t latency = 0;
    std::uint64_t filling = 0;
    std::uint64_t fixed = 0;
    /// @brief Longest single wavefront's dependence chain, before hiding.
    std::uint64_t critical_path = 0;
    /// @brief Cycles the worst wavefront spent waiting on memory.
    std::uint64_t worst_stall = 0;
    /// @brief Round trip to the deepest level this dispatch's traffic reached.
    std::uint64_t fill = 0;
    /// @brief The straggler correction, kept apart from the latency it is
    ///        added to so that the modelled work stays legible beside it.
    std::uint64_t straggler = 0;
    std::uint64_t rounds = 0;
    std::uint64_t resident = 0;
    /// @brief The busiest compute unit's own numbers, not the device totals.
    ///
    /// @details end() takes a maximum over compute units, so a sum over them
    /// cannot reproduce what it did. Every wavefront in a dispatch runs the
    /// same kernel, so one unit's accounting is representative -- and it is the
    /// one the composition actually used.
    std::uint64_t unit_issue = 0;
    std::uint64_t unit_critical = 0;
    std::uint64_t unit_worst_stall = 0;
    std::uint64_t unit_waves = 0;
    std::array<std::uint64_t, kNumFunctionalUnits> unit_cycles{};
    /// @brief The same, for the compute unit whose exposed latency won.
    ///
    /// @details A separate maximum from the issue one and, on a grid too small
    /// to fill the machine, a different compute unit: one holding two
    /// workgroups issues the most and one holding a workgroup that missed
    /// everywhere waits the longest. Reusing the issue winner's chain
    /// understated the latency term by up to a third on exactly the small
    /// dispatches whose duration is all latency.
    std::uint64_t latency_issue = 0;
    std::uint64_t latency_critical = 0;
    std::uint64_t latency_worst_stall = 0;
    std::uint64_t latency_waves = 0;
    std::array<std::uint64_t, kNumFunctionalUnits> latency_unit_cycles{};
    std::array<std::uint64_t, kNumInstClasses> class_counts{};
    /// @brief Cycles each bandwidth level's busiest instance was occupied for,
    ///        so that a rate change can be re-evaluated against the field and
    ///        not only against the level that happened to win.
    std::uint64_t l1_cycles = 0;
    std::uint64_t l2_cycles = 0;
    std::uint64_t fabric_cycles = 0;
    std::uint64_t mall_cycles = 0;
    std::uint64_t dram_cycles = 0;
  };
  const Terms &terms() const { return terms_; }

  /// @brief Cycles the dispatch cost, valid after end().
  std::uint64_t cycles() const { return cycles_; }
  /// @brief Which term produced cycles(): "issue", "latency", "placement",
  ///        "launch", or the name of the bandwidth level that bound it.
  const char *bound_by() const { return bound_by_; }

  /// @brief Where this dispatch's traffic is charged. Handed to whatever walks
  ///        the hierarchy on its behalf.
  BandwidthLedger &ledger() { return ledger_; }
  const BandwidthLedger &ledger() const { return ledger_; }

  const DispatchShape &shape() const { return shape_; }
  /// @brief Wavefronts one compute unit holds at once for this dispatch.
  std::uint64_t resident_waves() const { return resident_waves_; }
  std::uint64_t waves() const { return waves_; }
  std::uint64_t instructions() const { return instructions_; }
  std::uint64_t stall_cycles() const { return stall_cycles_; }

private:
  const Tuning &tuning_;
  std::vector<ComputeUnitDes *> units_;
  DispatchShape shape_;
  BandwidthLedger ledger_;
  /// @brief Whether the launch acquire has been claimed. Atomic because the
  ///        first wavefront to execute may be on any partition thread.
  std::atomic<bool> acquired_{false};
  std::uint64_t resident_waves_ = 1;
  std::uint64_t cycles_ = 0;
  std::uint64_t waves_ = 0;
  std::uint64_t instructions_ = 0;
  std::uint64_t stall_cycles_ = 0;
  const char *bound_by_ = "launch";
  Terms terms_;
};

} // namespace rocjitsu::timing
