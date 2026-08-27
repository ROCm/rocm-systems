// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/dispatch_des.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

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

/// @brief Wavefronts one compute unit can hold at once, from what one of them
///        demands and what the unit has.
///
/// @details Three independent limits, and the tightest wins: wave slots,
/// register files, and group memory. Group memory is per workgroup rather than
/// per wavefront, so its limit is converted through the workgroup's wavefront
/// count before it can be compared with the others.
std::uint64_t resident_waves_for(const Tuning &tuning, const DispatchShape &shape) {
  std::uint64_t limit = std::max<std::uint64_t>(1, tuning.wave_slots_per_cu);
  if (shape.vector_registers_per_wave > 0 && tuning.vector_registers_per_cu > 0)
    limit = std::min(limit, std::max<std::uint64_t>(1, tuning.vector_registers_per_cu /
                                                           shape.vector_registers_per_wave));
  if (shape.scalar_registers_per_wave > 0 && tuning.scalar_registers_per_cu > 0)
    limit = std::min(limit, std::max<std::uint64_t>(1, tuning.scalar_registers_per_cu /
                                                           shape.scalar_registers_per_wave));
  if (shape.lds_bytes_per_workgroup > 0 && tuning.lds_bytes_per_cu > 0) {
    const std::uint64_t groups =
        std::max<std::uint64_t>(1, tuning.lds_bytes_per_cu / shape.lds_bytes_per_workgroup);
    limit = std::min(limit, groups * std::max<std::uint32_t>(1, shape.waves_per_workgroup));
  }
  return std::max<std::uint64_t>(1, limit);
}

} // namespace

// -- BandwidthLevel ----------------------------------------------------------

void BandwidthLevel::configure(std::uint64_t instances, double bytes_per_cycle_total) {
  instances_ = std::max<std::uint64_t>(1, instances);
  // Never zero: a level with no rate would divide a real byte count by nothing
  // and report the dispatch as infinitely slow rather than as misconfigured.
  bytes_per_cycle_per_instance_ =
      std::max(1e-9, bytes_per_cycle_total / static_cast<double>(instances_));
  instance_ = std::vector<Instance>(static_cast<std::size_t>(instances_));
}

void BandwidthLevel::charge(std::uint64_t index, std::uint64_t bytes) {
  if (instance_.empty())
    return;
  instance_[static_cast<std::size_t>(index % instances_)].bytes.fetch_add(
      bytes, std::memory_order_relaxed);
}

std::uint64_t BandwidthLevel::busiest_cycles() const {
  std::uint64_t worst = 0;
  for (const Instance &instance : instance_) {
    const std::uint64_t bytes = instance.bytes.load(std::memory_order_relaxed);
    worst =
        std::max(worst, ceil_div_real(static_cast<double>(bytes), bytes_per_cycle_per_instance_));
  }
  return worst;
}

std::uint64_t BandwidthLevel::total_bytes() const {
  std::uint64_t total = 0;
  for (const Instance &instance : instance_)
    total += instance.bytes.load(std::memory_order_relaxed);
  return total;
}

void BandwidthLevel::reset() {
  for (Instance &instance : instance_)
    instance.bytes.store(0, std::memory_order_relaxed);
}

// -- BandwidthLedger ---------------------------------------------------------

void BandwidthLedger::configure(const Tuning &tuning) {
  // The cache levels state their rate in lines per cycle per instance, so the
  // aggregate has to be reconstructed before it can be split again. Doing it
  // this way rather than storing a device-wide byte rate keeps one number in
  // the config -- the one a datasheet quotes -- instead of two that can
  // disagree.
  l1_.configure(tuning.compute_units, tuning.l1_vector.lines_per_cycle *
                                          static_cast<double>(tuning.l1_vector.line_bytes) *
                                          static_cast<double>(tuning.compute_units));
  l2_.configure(tuning.xcds, tuning.l2.lines_per_cycle * static_cast<double>(tuning.l2.line_bytes) *
                                 static_cast<double>(tuning.xcds));
  fabric_.configure(tuning.memory_channels, tuning.fabric_bytes_per_cycle);
  mall_.configure(tuning.memory_channels, tuning.mall_bytes_per_cycle);
  dram_.configure(tuning.memory_channels, tuning.dram_bytes_per_cycle);
}

void BandwidthLedger::reset() {
  l1_.reset();
  l2_.reset();
  fabric_.reset();
  mall_.reset();
  dram_.reset();
  std::fill(dram_extra_.begin(), dram_extra_.end(), 0);
}

void BandwidthLedger::charge_dram_cycles(std::uint64_t channel, std::uint64_t cycles) {
  if (dram_extra_.empty())
    dram_extra_.assign(static_cast<std::size_t>(std::max<std::uint64_t>(1, dram_.instances())), 0);
  dram_extra_[static_cast<std::size_t>(channel % dram_extra_.size())] += cycles;
}

std::uint64_t BandwidthLedger::bound_cycles() const {
  // The DRAM term is its transfer time plus the row activations that channel
  // had to perform, which occupy it without moving a byte.
  std::uint64_t dram = 0;
  for (std::size_t channel = 0; channel < dram_extra_.size(); ++channel)
    dram = std::max(dram, dram_extra_[channel]);
  dram += dram_.busiest_cycles();
  return std::max({l1_.busiest_cycles(), l2_.busiest_cycles(), fabric_.busiest_cycles(),
                   mall_.busiest_cycles(), dram});
}

const char *BandwidthLedger::bound_level() const {
  const std::uint64_t cycles[] = {l1_.busiest_cycles(), l2_.busiest_cycles(),
                                  fabric_.busiest_cycles(), mall_.busiest_cycles(),
                                  dram_.busiest_cycles()};
  static const char *const kNames[] = {"l1", "l2", "fabric", "mall", "dram"};
  std::size_t worst = 0;
  for (std::size_t level = 1; level < std::size(cycles); ++level)
    if (cycles[level] > cycles[worst])
      worst = level;
  return kNames[worst];
}

// -- DispatchDes -------------------------------------------------------------

DispatchDes::DispatchDes(std::string name, const simdojo::ClockDomain &domain, TimingEngine &engine,
                         const Tuning &tuning, std::vector<ComputeUnitDes *> units)
    : TimedComponent(std::move(name), domain, engine), tuning_(tuning), units_(std::move(units)) {
  ledger_.configure(tuning_);
}

std::uint64_t DispatchDes::advance(std::uint64_t now) {
  (void)now;
  // Like the compute unit, this component is on the engine so that it has a
  // clock domain and an identity in the component tree, not because it serves
  // an inbox: a dispatch's cost is composed from what its compute units and its
  // ledger accumulated, at the point the guest can observe it. Anything
  // delivered here is dropped rather than left to accumulate.
  inbox().clear();
  return 0;
}

void DispatchDes::begin(const DispatchShape &shape) {
  shape_ = shape;
  resident_waves_ = resident_waves_for(tuning_, shape_);
  ledger_.reset();
  acquired_.store(false, std::memory_order_relaxed);
  cycles_ = 0;
  waves_ = 0;
  instructions_ = 0;
  stall_cycles_ = 0;
  bound_by_ = "launch";
  // The compute units are deliberately NOT cleared here. begin() runs when the
  // packet is parsed and the command processor parses ahead, so clearing at
  // this point either wipes the dispatch that is still running or, when a
  // dispatch id is reused (they are allocated per command processor and do
  // repeat), does not run between two dispatches at all. The symptom of the
  // second was a kernel launched three times reporting 16, then 32, then 48
  // wavefronts. Clearing belongs with the launch acquire, at the moment the
  // first wavefront actually issues; see clear_units().
}

void DispatchDes::clear_units() {
  for (ComputeUnitDes *unit : units_)
    if (unit != nullptr)
      unit->reset_occupancy();
}

bool DispatchDes::claim_acquire() {
  bool expected = false;
  return acquired_.compare_exchange_strong(expected, true, std::memory_order_relaxed);
}

void DispatchDes::end() {
  std::uint64_t issue_bound = 0;
  std::uint64_t latency_bound = 0;
  waves_ = 0;
  instructions_ = 0;
  stall_cycles_ = 0;
  terms_ = Terms{};

  for (const ComputeUnitDes *unit : units_) {
    if (unit == nullptr)
      continue;
    const ComputeUnitDes::Occupancy &occupancy = unit->occupancy();
    if (occupancy.waves == 0)
      continue;
    waves_ += occupancy.waves;
    instructions_ += occupancy.instructions;
    stall_cycles_ += occupancy.stall_cycles;
    // Wavefronts resident together hide each other's stalls, but only as far as
    // they have work to issue while one of them is waiting. The unit's duration
    // is therefore the work it has to retire plus whatever of its longest
    // wavefront's critical path that work did not cover -- not the larger of
    // the two.
    //
    // Taking the larger is the classic memory-warp-parallelism form and it is
    // optimistic by construction: it assumes every resident wavefront always
    // has something ready the instant another stalls, which is the best case
    // and not the common one. It showed up as a shortfall that was flat in
    // every direction -- flat across launch counts, flat across kernel
    // durations, flat across categories -- because it is a property of the
    // composition rather than of any one modelled effect. Concurrency is
    // latency times throughput on both sides, so the honest reading is that
    // hiding is bounded by the issue work actually available to overlap.
    //
    // The two limits are still right. At high occupancy the hidden term
    // swallows the critical path and the unit is throughput bound; at an
    // occupancy of one nothing is hidden and it is the sum, which is what a
    // single wavefront running alone actually costs.
    const std::uint64_t rounds = ceil_div(occupancy.waves, resident_waves_);
    const std::uint64_t resident = std::min(occupancy.waves, resident_waves_);
    const std::uint64_t per_wave_issue =
        occupancy.waves != 0 ? occupancy.issue_cycles / occupancy.waves : 0;
    const std::uint64_t hidden = resident > 1 ? (resident - 1) * per_wave_issue : 0;
    // The wavefront's chain and the waiting inside it are scaled separately:
    // the chain is issue slots, which the machine parameters describe, and the
    // waiting is the memory system, whose exposure is calibrated.
    const std::uint64_t raw_stall = static_cast<std::uint64_t>(
        static_cast<double>(occupancy.worst_stall_cycles) * tuning_.stall_exposed_fraction);
    const std::uint64_t critical =
        (occupancy.critical_path_cycles > occupancy.worst_stall_cycles
             ? occupancy.critical_path_cycles - occupancy.worst_stall_cycles
             : 0) +
        raw_stall;
    std::uint64_t exposed = critical > hidden ? critical - hidden : 0;
    // Whatever the issue-work arithmetic says, a wavefront's memory stalls are
    // only hidden to the extent that other wavefronts' stalls are staggered
    // against them. They usually are not: wavefronts running the same code
    // reach the same load at the same time. This floors the exposure at the
    // worst wavefront's stalls divided by however many of them genuinely
    // overlap, which is a property of the memory system rather than of the
    // issue stream.
    const std::uint64_t overlap =
        std::max<std::uint64_t>(1, std::min(resident, tuning_.stall_overlap_wavefronts));
    exposed = std::max(exposed, raw_stall / overlap);
    issue_bound = std::max(issue_bound, occupancy.issue_cycles);
    if (exposed * rounds >= latency_bound) {
      latency_bound = exposed * rounds;
      terms_.latency_issue = occupancy.issue_cycles;
      terms_.latency_critical = critical;
      terms_.latency_worst_stall = raw_stall;
      terms_.latency_waves = occupancy.waves;
      terms_.latency_unit_cycles = occupancy.unit_cycles;
      terms_.rounds = rounds;
      terms_.resident = resident;
    }
    terms_.critical_path = std::max(terms_.critical_path, critical);
    terms_.worst_stall = std::max(terms_.worst_stall, occupancy.worst_stall_cycles);
    if (occupancy.issue_cycles >= terms_.unit_issue) {
      terms_.unit_issue = occupancy.issue_cycles;
      terms_.unit_critical = critical;
      terms_.unit_worst_stall = occupancy.worst_stall_cycles;
      terms_.unit_waves = occupancy.waves;
      terms_.unit_cycles = occupancy.unit_cycles;
    }
    for (std::size_t cls = 0; cls < kNumInstClasses; ++cls)
      terms_.class_counts[cls] += occupancy.class_counts[cls];
  }

  // Throughput and latency compose by addition, not by taking the larger. A
  // resource has to fill before it can drain: the first access of a streaming
  // kernel still waits a full round trip before any byte arrives, and only then
  // does the bandwidth term describe what follows. Taking the larger dropped
  // that fill entirely from every kernel the memory system bounded, which is
  // most of the ones the model was under-charging -- a one-megabyte copy came
  // out at 1.53 microseconds against a measured 2.36, and the 0.83 it was
  // missing is one round trip.
  //
  // The two limits are the check that this is the right shape. A kernel with
  // enough parallelism to hide its latency has an exposed term near zero and is
  // purely throughput bound; a kernel with none has a throughput term near zero
  // and is purely its own critical path.
  // One round trip to the deepest level the dispatch's traffic actually reached
  // is exposed no matter how much parallelism the kernel has. Occupancy hides
  // latency by giving a compute unit another wavefront's work to do while one
  // waits, and at kernel start there is no such work: every resident wavefront
  // is waiting on its own first access at once. A streaming kernel with eight
  // wavefronts per compute unit was being credited with hiding all of its
  // latency for exactly that reason, and came out at 0.65 of its measured
  // duration.
  //
  // Which level is read off the traffic rather than assumed, so a kernel whose
  // working set stayed in cache is not charged a memory round trip it never
  // took.
  std::uint64_t fill = tuning_.l1_vector.hit_cycles;
  if (ledger_.l2_bytes() != 0)
    fill = tuning_.l2.hit_cycles;
  if (ledger_.mall_bytes() != 0)
    fill = tuning_.mall.hit_cycles;
  if (ledger_.dram_bytes() != 0)
    fill = tuning_.dram_latency_cycles;
  fill = static_cast<std::uint64_t>(static_cast<double>(fill) * tuning_.fill_exposure_scale);
  terms_.fill = fill;
  latency_bound = static_cast<std::uint64_t>(static_cast<double>(latency_bound) *
                                             tuning_.latency_exposure_scale);
  if (waves_ != 0)
    latency_bound = std::max(latency_bound, fill);

  // A dispatch ends when its slowest wavefront ends, not when the average one
  // does. Every compute unit here runs the same instruction stream and so
  // finishes at the same modelled cycle, which makes the modelled maximum the
  // modelled mean and drops the straggler entirely. The expected maximum of
  // many independent noisy durations grows with the logarithm of how many
  // there are, and that is what the measurements show: a launch-dominated
  // kernel costs about the same extra per doubling of its wavefront count,
  // from four wavefronts to a thousand.
  if (tuning_.straggler_cycles != 0 && waves_ > 1)
    terms_.straggler = static_cast<std::uint64_t>(static_cast<double>(tuning_.straggler_cycles) *
                                                  std::log2(static_cast<double>(waves_)));
  latency_bound += terms_.straggler;

  const std::uint64_t placement =
      ceil_div_real(static_cast<double>(shape_.workgroup_count), tuning_.workgroups_per_cycle);
  const std::uint64_t bandwidth = ledger_.bound_cycles();
  std::uint64_t throughput = issue_bound;
  bound_by_ = "issue";
  if (bandwidth > throughput) {
    throughput = bandwidth;
    bound_by_ = ledger_.bound_level();
  }
  // A grid larger than the command processor can place in the time the work
  // takes is placement bound, however little each workgroup does.
  if (placement > throughput) {
    throughput = placement;
    bound_by_ = "placement";
  }
  // Filling the machine is serial and comes before steady state. The command
  // processor places workgroups one at a time, so until it has placed enough to
  // occupy every compute unit the part is not yet running at the rate the
  // throughput term describes. That ramp is additive, not an alternative: a
  // grid still has to be placed and then still has to do its work.
  //
  // This is what was missing from mid-size streaming kernels. Achieved
  // bandwidth grows with transfer size on hardware -- an eight megabyte copy
  // reaches 6.2 TB/s while a one megabyte copy of the same shape reaches 3.6 --
  // and a model with a constant rate and no ramp reads the small one fast. A
  // single-workgroup kernel has no ramp at all, which is why this does not
  // disturb the small latency-bound cases at the other end of the error.
  const std::uint64_t filling = static_cast<std::uint64_t>(
      tuning_.fill_ramp_scale *
      static_cast<double>(ceil_div_real(
          static_cast<double>(std::min<std::uint64_t>(
              shape_.workgroup_count, std::max<std::uint64_t>(1, tuning_.compute_units))),
          tuning_.workgroups_per_cycle)));

  std::uint64_t body = throughput + latency_bound + filling;
  if (latency_bound > throughput)
    bound_by_ = "latency";
  if (waves_ == 0) {
    // Nothing was seen executing. The dispatch still cost the machine its
    // launch, and reporting zero would be the one guess that is always in the
    // fast direction.
    body = 0;
    bound_by_ = "launch";
  }

  // Every dispatch is charged the full launch path. A kernel launched back to
  // back arguably should not be, since the command processor parses ahead and
  // part of its launch overlapped its predecessor. It is left unmodelled
  // because neither signal available here is trustworthy: gating on "another
  // dispatch is open" fires almost always, because a dispatch whose completion
  // the observer could not attribute stays open for the rest of the run, and it
  // moved the corpus from 17.1 to 20.1 per cent; gating on "another dispatch is
  // executing" almost never fires, because the emulator runs them one at a
  // time, and it moved nothing. The effect is real and the observation is not
  // there to detect it.
  terms_.issue = issue_bound;
  terms_.bandwidth = bandwidth;
  terms_.l1_cycles = ledger_.l1_cycles();
  terms_.l2_cycles = ledger_.l2_cycles();
  terms_.fabric_cycles = ledger_.fabric_cycles();
  terms_.mall_cycles = ledger_.mall_cycles();
  terms_.dram_cycles = ledger_.dram_cycles();
  terms_.placement = placement;
  // Recorded without the straggler, which is reported beside it: a reader has
  // to be able to tell the modelled work from the correction applied to it.
  terms_.latency = latency_bound - std::min(latency_bound, terms_.straggler);
  terms_.filling = filling;
  terms_.fixed = tuning_.dispatch_start_cycles + tuning_.launch_invalidate_cycles +
                 tuning_.dispatch_end_cycles;
  cycles_ = terms_.fixed + body;
}

} // namespace rocjitsu::timing
