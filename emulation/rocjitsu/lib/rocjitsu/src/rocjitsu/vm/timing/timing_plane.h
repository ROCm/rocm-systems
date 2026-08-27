// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file timing_plane.h
/// @brief The timed machine: components, clocks, and the clock the guest reads.
///
/// @details Instantiated by the SoC when the architecture config carries a
/// `timing` block, and part of the machine rather than something loaded next to
/// it. It owns the clock domains, the compute units, the cache levels, the
/// fabric and the memory channels, wires them together, and runs them on its
/// own event loop over modelled hardware time.
///
/// The functional simulator drives it through three direct calls -- a dispatch
/// beginning, an instruction retiring, a dispatch ending. Those calls are what
/// a compute unit already knows and cost a null check when no plane is
/// installed, which is the same thing the execution-plugin group already costs
/// on the same path.
///
/// What comes out is one number: the tick the modelled machine has reached.
/// That is published into SimulatedClock, and from there it is what
/// hipEventElapsedTime subtracts, what the KFD clock-counters ioctl answers,
/// what a completion signal's timestamps carry, and what `s_memtime` returns
/// inside a kernel. A guest that times itself measures the modelled part rather
/// than the host it happens to be running on.

#pragma once

#include "rocjitsu/vm/timing/cache_des.h"
#include "rocjitsu/vm/timing/cu_des.h"
#include "rocjitsu/vm/timing/dram_des.h"
#include "rocjitsu/vm/timing/engine.h"
#include "rocjitsu/vm/timing/request.h"
#include "rocjitsu/vm/timing/time_source.h"
#include "rocjitsu/vm/timing/tuning.h"

#include "simdojo/sim/clock_domain.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocjitsu::timing {

class DispatchDes;

/// @brief What the command processor reports when a dispatch starts.
struct DispatchShape {
  std::uint32_t dispatch_id = 0;
  std::uint32_t queue_id = 0;
  std::string kernel_name;
  std::uint32_t workgroup_count = 0;
  std::uint32_t waves_per_workgroup = 0;
  std::uint32_t vector_registers_per_wave = 0;
  std::uint32_t scalar_registers_per_wave = 0;
  std::uint32_t lds_bytes_per_workgroup = 0;
  std::uint32_t wave_size = 64;
};

/// @brief The timed machine.
class TimingPlane final : public TimeSource {
public:
  explicit TimingPlane(Tuning tuning);
  ~TimingPlane() override;

  bool enabled() const { return tuning_.enabled; }

  // -- Driven by the functional simulator ----------------------------------
  void dispatch_begin(const DispatchShape &shape);
  /// @brief A wavefront of @p dispatch_id started on @p compute_unit.
  ///
  /// @details Also where the launch acquire is applied, because this is the
  /// first moment the dispatch is known to be executing. The packet-processed
  /// callback is not: the command processor parses the next packet while the
  /// current one is still running.
  void wave_begin(std::uint32_t compute_unit, std::uint32_t slot, std::uint32_t dispatch_id,
                  std::uint32_t queue_id);
  void instruction(std::uint32_t compute_unit, std::uint32_t slot,
                   const RetiredInstruction &retired);
  void wave_end(std::uint32_t compute_unit, std::uint32_t slot);
  void dispatch_end(std::uint32_t dispatch_id, std::uint32_t queue_id);
  ComputeUnitDes &compute_unit(std::uint32_t index);

  // -- Used by the compute units -------------------------------------------
  /// @brief Hand one memory instruction's lines to the hierarchy.
  /// @returns The tick the access completes.
  std::uint64_t issue_memory(const MemoryRequest &request, const std::vector<std::uint64_t> &lines);
  /// @brief Apply a dispatch's launch acquire, exactly once, at the moment its
  ///        first wavefront issues. Not when its packet is parsed: the command
  ///        processor parses ahead, so a packet-driven invalidate lands at an
  ///        unpredictable point relative to execution.
  void acquire_once(std::uint32_t dispatch_id, std::uint32_t queue_id);

  // -- TimeSource ----------------------------------------------------------
  std::uint64_t current_cycles() const override { return cycles_.load(std::memory_order_relaxed); }
  double clock_ghz() const override { return tuning_.clock_mhz / 1000.0; }

  void write_report(std::string &out) const;

private:
  void publish(std::uint64_t cycles);
  /// @brief Record what every component had moved, so a dispatch's own traffic
  ///        is the difference rather than the total.
  /// @brief Split a second-level miss across the channels its lines select.
  void forward_to_channels(const MemoryRequest &request);
  void snapshot_bandwidth(std::uint64_t key);
  void charge_bandwidth(DispatchDes &dispatch, std::uint64_t key);
  /// @brief Append one dispatch's terms to the trace, when one is open.
  ///
  /// @details Off unless ROCJITSU_TIMING_TRACE names a file. What it writes is
  /// every term the composition used, not the total it produced: a tuning
  /// change that only recombines those terms can then be evaluated against a
  /// recorded corpus instead of by re-running it, which is the difference
  /// between a ten-minute experiment and a one-second one.
  void trace_dispatch(const DispatchDes &dispatch);

  Tuning tuning_;
  simdojo::ClockDomain shader_;
  simdojo::ClockDomain fabric_;
  simdojo::ClockDomain memory_;
  TimingEngine engine_;

  std::vector<std::unique_ptr<ComputeUnitDes>> compute_units_;
  std::vector<std::unique_ptr<CacheDes>> l1_vector_;
  std::vector<std::unique_ptr<CacheDes>> l1_scalar_;
  std::vector<std::unique_ptr<CacheDes>> l1_instruction_;
  std::vector<std::unique_ptr<CacheDes>> l2_;
  std::vector<std::unique_ptr<ChannelDes>> fabric_channels_;
  std::vector<std::unique_ptr<CacheDes>> mall_;
  std::vector<std::unique_ptr<ChannelDes>> dram_;

  /// @brief Line addresses in flight, referenced by MemoryRequest::line_base.
  std::vector<std::uint64_t> line_pool_;
  /// @brief Tick each compute unit's current dispatch began, and which that is.
  std::vector<std::uint64_t> cu_base_;
  std::vector<std::uint64_t> cu_dispatch_;
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> bandwidth_base_;
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> activation_base_;
  /// @brief Open trace stream, or null when no trace was asked for.
  std::FILE *trace_ = nullptr;
  /// @brief The request issue_memory() is waiting for, and its answer.
  std::uint64_t sequence_ = 0;
  std::uint32_t awaited_ = 0;
  std::uint32_t awaited_outstanding_ = 0;
  std::vector<std::uint64_t> channel_scratch_;
  std::vector<std::uint64_t> request_scratch_;
  /// @brief Distinct DRAM rows one request touches on one channel.
  ///
  /// @details A tiny insertion-ordered set reused across requests rather than
  /// allocated per request, because this runs inside the per-instruction path.
  struct RowSet {
    std::vector<std::uint64_t> rows;
    void clear() { rows.clear(); }
    bool insert(std::uint64_t row) {
      for (std::uint64_t seen : rows)
        if (seen == row)
          return false;
      rows.push_back(row);
      return true;
    }
  };
  RowSet row_scratch_;
  bool answered_ = false;
  std::uint64_t completion_tick_ = 0;

  mutable std::mutex mutex_;
  std::unordered_map<std::uint64_t, std::unique_ptr<DispatchDes>> open_;
  std::vector<std::string> completed_;

  /// @brief Read from guest threads holding none of this plane's locks, so it
  ///        is published from a relaxed atomic and never allowed to retreat.
  std::atomic<std::uint64_t> cycles_{0};
};

} // namespace rocjitsu::timing
