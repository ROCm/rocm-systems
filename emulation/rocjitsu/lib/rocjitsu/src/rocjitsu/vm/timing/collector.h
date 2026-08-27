// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file collector.h
/// @brief The one place rocjitsu execution becomes the timing plane's vocabulary.
///
/// @details There is exactly one extraction layer and this is it. Turning a
/// compute unit's execution into a coherent per-wavefront stream is the fiddly
/// part of the feature: wait targets that stay set from an earlier instruction,
/// a terminator that never reaches the after hook, a program counter that means
/// something different after execution than a reader expects, dispatch ids that
/// collide across command processors. Every one of those is a silent
/// under-count if it is got wrong, so the derivation lives here once rather
/// than being re-derived per component.
///
/// This is deliberately not an ExecutionPlugin. Timing is part of the machine,
/// not an observer of it: the compute unit calls the collector directly, the
/// same way it already reads its own caches, and installing no plane costs a
/// single predictable not-taken branch on the issue path.

#pragma once

#include "rocjitsu/vm/plugins/kernel_dispatch_info.h"
#include "rocjitsu/vm/plugins/wavefront_state.h"
#include "rocjitsu/vm/timing/cu_des.h"
#include "rocjitsu/vm/timing/inst_class.h"
#include "rocjitsu/vm/timing/vocabulary.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocjitsu {
class Instruction;
namespace amdgpu {
class Wavefront;
} // namespace amdgpu
} // namespace rocjitsu

namespace rocjitsu::timing {

class TimingPlane;

/// @brief What the collector derives once per distinct program counter.
///
/// @details A kernel executes each instruction many times and re-deriving its
/// class from the mnemonic every time would dominate the collector's cost. The
/// mnemonic is stored rather than referenced because the decoded instruction
/// does not outlive the execution that produced it, while RetiredInstruction
/// hands the plane a pointer to it.
struct StaticInstruction {
  std::string mnemonic;
  InstClass inst_class = InstClass::Unknown;
  std::uint32_t size_bytes = 4;
  /// @brief Registers the instruction reads and writes.
  ///
  /// @details Derived from the decoded operands, which carry no wavefront state
  /// and so are the same on every execution of this address. Kept here for that
  /// reason: an operand walk per execution would cost more than the dependency
  /// it recovers is worth.
  RegisterRanges reads;
  RegisterRanges writes;
};

/// @brief Everything the collector keeps on one wavefront.
///
/// @details Lives in the wavefront's own plugin-state slot so the hot path
/// never looks a wavefront up in a map. The pending fields carry an instruction
/// from the before hook, which knows the program counter it issued at and the
/// lanes that were live going in, to the after hook, which knows what addresses
/// were produced.
struct CollectedWave final : rocjitsu::WavefrontState {
  /// @brief Modelled compute unit this wavefront contends on, and its slot.
  ///
  /// @details The plane folds both into its own geometry, so these are the
  /// placement rather than an index into anything the collector owns.
  std::uint32_t compute_unit = 0;
  std::uint32_t wave_slot = 0;
  std::uint32_t dispatch_id = 0;
  std::uint32_t queue_id = 0;

  /// @brief Program counter the pending instruction issued at.
  std::uint64_t pending_pc = 0;
  /// @brief Static properties of the pending instruction, owned by the
  ///        collector's per-pc cache, which outlives every wavefront.
  const StaticInstruction *pending_info = nullptr;
  /// @brief Lanes live *before* the pending instruction executed.
  ///
  /// @details Sampled in the before hook because an instruction may rewrite
  /// EXEC -- v_cmpx and the saveexec family do -- and the work the machine
  /// performed is the work of the lanes that were live going in. Sampling
  /// afterwards would cost a diverging branch nothing on the iteration that
  /// narrowed the mask.
  std::uint32_t pending_active_lanes = 0;
  bool has_pending = false;

  /// @brief Per-lane addresses of the instruction being reported.
  ///
  /// @details Owned by the wavefront and reused, so the sixty-four addresses a
  /// vector access produces cost no allocation after the first one.
  std::vector<std::uint64_t> lane_addresses;
};

/// @brief Turns rocjitsu execution into the timing plane's calls.
///
/// @details Borrows the plane; whoever owns both must destroy the collector
/// first, because every entry point here calls into it.
class TimingCollector {
public:
  /// @param plane The timed machine to drive. Borrowed; must outlive this.
  explicit TimingCollector(TimingPlane &plane);
  ~TimingCollector();

  TimingCollector(const TimingCollector &) = delete;
  TimingCollector &operator=(const TimingCollector &) = delete;

  /// @brief Wavefront plugin-state slot the collector keeps CollectedWave in.
  ///
  /// @details Reserved above the range ExecutionPluginGroup hands out, which it
  /// assigns from zero in load order. Fixed rather than negotiated because the
  /// plane is not a plugin and must not depend on how many are loaded, and
  /// because a slot decided at run time would put an indirection on the hottest
  /// path in the simulator.
  static constexpr std::uint32_t kWavefrontStateSlot = 16;

  /// @brief Give @p wf its state slot, before any wavefront can execute.
  ///
  /// @details Called by the compute unit when the collector is installed, for
  /// every slot it owns. Wavefront::plugin_state() reads the slot without
  /// bounds checking, so the vector behind it has to be grown before the hot
  /// path ever reads it, and growing it there would put an allocation on the
  /// issue path. Attaching here also means a wavefront that executes without
  /// ever having been dispatched through the collector reads back a live state
  /// rather than whatever the slot happened to alias.
  void attach(amdgpu::Wavefront &wf);

  /// @brief A dispatch packet was parsed. Announces the dispatch's shape.
  void dispatch_packet(const rocjitsu::KernelDispatchInfo &info);
  /// @brief Every workgroup of @p dispatch_id has retired.
  void dispatch_execution_end(std::uint32_t dispatch_id);

  /// @brief A wavefront started executing.
  void wave_dispatched(amdgpu::Wavefront &wf);
  /// @brief A wavefront terminated. Also where its terminator is reported.
  void wave_halted(amdgpu::Wavefront &wf);
  /// @brief A barrier released @p waves together.
  void barrier_resolved(std::span<amdgpu::Wavefront *> waves);

  /// @brief About to execute @p inst at @p pc.
  void before_execute(std::uint64_t pc, const Instruction &inst, amdgpu::Wavefront &wf);
  /// @brief Just executed @p inst, which issued at the pc @ref before_execute saw.
  void after_execute(std::uint64_t pc, const Instruction &inst, amdgpu::Wavefront &wf);

private:
  /// @brief Static properties for @p pc, derived on first sight.
  ///
  /// @details The cached mnemonic is checked against the instruction's: binary
  /// translation can place different code at an address seen earlier in the
  /// run, and a stale entry would silently cost the new code as the old.
  const StaticInstruction &static_info(std::uint64_t pc, const Instruction &inst);

  CollectedWave *wave_state(const amdgpu::Wavefront &wf) const;

  /// @brief Announce a dispatch the collector has not seen a packet for.
  ///
  /// @details Packets are parsed ahead of execution, so the normal order is
  /// announcement first and wavefronts later. A wavefront belonging to a
  /// dispatch nobody announced would otherwise have its work dropped, and the
  /// first entries of a report are exactly the kernels a run is usually about.
  /// Synthesising a shape-only announcement keeps it attributable, and a real
  /// packet arriving afterwards replaces it rather than being dropped as a
  /// duplicate.
  void ensure_dispatch_announced(const amdgpu::Wavefront &wf, std::uint32_t dispatch_id,
                                 std::uint32_t queue_id);

  /// @brief Complete and report the wavefront's pending instruction.
  ///
  /// @param inst The executing instruction, or null when it is no longer
  ///        reachable -- the terminal path, where the memory payload is moot
  ///        because the instruction that halted the wave is not a memory one.
  void emit_pending(CollectedWave &wave, const amdgpu::Wavefront &wf, const Instruction *inst);

  /// @brief Pack a dispatch's two-part identity into one integer.
  static std::uint64_t packed(std::uint32_t dispatch_id, std::uint32_t queue_id) {
    return (static_cast<std::uint64_t>(queue_id) << 32) | dispatch_id;
  }

  TimingPlane &plane_;
  /// @brief Sampled once. A disabled plane owns no compute units, so every
  ///        entry point has to return before it indexes them.
  bool enabled_ = false;

  /// @brief Serialises every call into the plane.
  ///
  /// @details The plane is one event loop over one component graph with one
  /// line pool, and the emulator executes wavefronts on as many threads as its
  /// partitioning gives it. Sharding per compute unit -- which is what an
  /// observer feeding independent per-unit models would do -- is wrong here for
  /// exactly the reason the plane is worth having: its compute units share a
  /// memory hierarchy, so two of them are never independent.
  std::mutex plane_mutex_;

  /// @brief Read on every instruction, written once per distinct program
  ///        counter, so concurrent wavefronts read it in parallel and only a
  ///        first sighting takes the exclusive lock.
  mutable std::shared_mutex info_mutex_;
  std::unordered_map<std::uint64_t, std::unique_ptr<StaticInstruction>> info_cache_;
  /// @brief Entries displaced from @ref info_cache_, kept alive for the run.
  ///
  /// @details An entry is only ever displaced when translated code reuses an
  /// address, and a wavefront can be holding the old one as its pending
  /// instruction at that moment. Nothing tracks those references, so the only
  /// safe lifetime is the collector's own.
  std::vector<std::unique_ptr<StaticInstruction>> retired_info_;

  /// @brief Guards the dispatch bookkeeping below.
  std::mutex dispatch_mutex_;
  /// @brief dispatch_id to queue_id, because the completion path carries only
  ///        the id and the plane keys on both halves.
  std::unordered_map<std::uint32_t, std::uint32_t> dispatch_queue_;
  /// @brief Dispatches already announced to the plane.
  std::unordered_set<std::uint64_t> announced_;
  /// @brief The subset of @ref announced_ synthesised from a wavefront rather
  ///        than read from a packet, and so carrying no kernel name and no
  ///        workgroup count. A real packet replaces one of these.
  std::unordered_set<std::uint64_t> synthesized_;
};

} // namespace rocjitsu::timing
