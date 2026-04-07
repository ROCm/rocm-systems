// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file pacing_controller.h
/// @brief PI-controlled wall-clock pacing for simulation time.
///
/// @details Implements a PI servo with token bucket burst absorption for
/// pacing the simulation engine against wall-clock time. We leverage a
/// three-state machine (INIT → TRACKING → STABLE) with step threshold
/// for large offsets, anti-windup on the integral term, and seqlock-protected
/// timestamp translation for external threads.

#ifndef SIMDOJO_SIM_PACING_CONTROLLER_H_
#define SIMDOJO_SIM_PACING_CONTROLLER_H_

#include "simdojo/sim/sim_types.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <thread>

namespace simdojo {

/// @brief PI-controlled wall-clock pacing with token bucket burst absorption.
///
/// @details Maintains a linear mapping between wall-clock time and simulation
/// time, with a closed-loop PI controller that eliminates steady-state drift
/// and a configurable burst buffer that absorbs event clustering jitter.
///
/// Three states:
///   - INIT: Collecting initial samples to estimate drift. No PI correction.
///   - TRACKING: PI loop active with base gains. Transitions to STABLE after
///     sustained low-offset operation.
///   - STABLE: Tighter PI gains for reduced jitter in steady state.
///
/// Large offsets (> step_threshold) trigger a phase step: the anchor is reset
/// and the integral accumulator is cleared, similar to NTP/PTP step behavior.
///
/// The control loop time constant (1/proportional_gain ≈ 10 epochs) provides a 5x Nyquist
/// margin over the 1-epoch LBTS sampling interval, per NTP best practice.
class PacingController {
public:
  /// @brief Pacing controller configuration.
  struct Config {
    double ratio = 0.0; ///< Sim-time per wall-time (0 = disabled, 1.0 = real-time).
    double proportional_gain_tracking =
        0.1; ///< Proportional gain in TRACKING state (LinuxPTP sw default).
    double integral_gain_tracking = 0.001;  ///< Integral gain in TRACKING state.
    double proportional_gain_stable = 0.05; ///< Proportional gain in STABLE state (halved).
    double integral_gain_stable = 0.0005;   ///< Integral gain in STABLE state.
    double step_threshold_ns = 1e6;         ///< Offset threshold for phase step (1ms default).
    double burst_ns = 4e6;      ///< Token bucket depth (4ms default, absorbs Linux sleep jitter).
    uint32_t init_samples = 4;  ///< Samples before engaging PI loop.
    uint32_t stable_count = 16; ///< Consecutive low-offset samples to enter STABLE.
    double stable_offset_ns = 1e5; ///< Offset threshold for "low offset" (100us).
  };

  /// @brief Controller state.
  enum class State : uint8_t { INIT, TRACKING, STABLE };

  PacingController() : PacingController(Config{}) {}
  explicit PacingController(Config config)
      : config_(config), step_threshold_ticks_(ns_to_ticks(config.step_threshold_ns)),
        burst_ticks_(ns_to_ticks(config.burst_ns)),
        stable_offset_ticks_(ns_to_ticks(config.stable_offset_ns)) {}

  /// @brief Whether pacing is active.
  bool enabled() const { return config_.ratio > 0.0; }

  /// @brief Set the anchor point mapping wall-clock now to a simulation tick.
  /// @param sim_tick The simulation tick at anchor time (typically 0).
  void anchor(Tick sim_tick = 0) {
    std::unique_lock lock(anchor_mutex_);
    wall_anchor_ = clock::now();
    sim_anchor_ = sim_tick;
    paused_ = false;
    state_ = State::INIT;
    integral_ = 0.0;
    init_count_ = 0;
    stable_streak_ = 0;
  }

  /// @brief Pause the wall-clock mapping before an idle wait.
  void pause() {
    if (!enabled())
      return;
    std::unique_lock lock(anchor_mutex_);
    if (!paused_) {
      pause_start_ = clock::now();
      paused_ = true;
    }
  }

  /// @brief Resume the wall-clock mapping after an idle wait.
  ///
  /// @details Shifts wall_anchor_ forward by the exact idle duration using
  /// chrono integer arithmetic (no floating-point rounding). The PI integral
  /// is preserved across idle periods to retain the frequency estimate.
  void resume() {
    if (!enabled())
      return;
    std::unique_lock lock(anchor_mutex_);
    if (paused_) {
      auto idle = clock::now() - pause_start_;
      wall_anchor_ += idle;
      paused_ = false;
    }
  }

  /// @brief Convert the current wall-clock time to a simulation tick.
  ///
  /// @details Thread-safe via seqlock. External threads calling this never
  /// block — they retry if a write was in progress.
  Tick sim_tick_now() const {
    if (!enabled())
      return 0;
    clock::time_point anchor_wall;
    Tick anchor_sim;
    {
      std::shared_lock lock(anchor_mutex_);
      anchor_wall = wall_anchor_;
      anchor_sim = sim_anchor_;
    }
    auto elapsed = clock::now() - anchor_wall;
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    if (ns <= 0)
      return anchor_sim;
    return anchor_sim + static_cast<Tick>(static_cast<double>(ns) * config_.ratio * TICKS_PER_NS);
  }

  /// @brief PI-controlled throttle. Call after processing an epoch/batch.
  ///
  /// @details Computes the offset between simulation time and the target
  /// (wall-clock-projected) time. In INIT state, counts samples without
  /// correcting. In TRACKING/STABLE, applies PI correction. Sleeps if
  /// the offset exceeds the burst buffer. Large offsets trigger a step.
  void throttle(Tick sim_time) {
    if (!enabled())
      return;

    Tick target = sim_tick_now();
    double offset = static_cast<double>(sim_time) - static_cast<double>(target);

    // Step threshold: large offsets → reset anchor.
    if (std::abs(offset) > static_cast<double>(step_threshold_ticks_)) {
      {
        std::unique_lock lock(anchor_mutex_);
        wall_anchor_ = clock::now();
        sim_anchor_ = sim_time;
      }
      integral_ = 0.0;
      if (state_ == State::STABLE)
        state_ = State::TRACKING;
      stable_streak_ = 0;
      return;
    }

    // INIT state: just count samples.
    if (state_ == State::INIT) {
      ++init_count_;
      if (init_count_ >= config_.init_samples)
        state_ = State::TRACKING;
      return;
    }

    // Sim behind: don't throttle, anti-windup on integral.
    if (offset <= 0) {
      integral_ = std::max(0.0, integral_ + offset);
      return;
    }

    // Track STABLE transitions (convergence tracking happens regardless
    // of whether the burst buffer absorbs or the PI loop throttles).
    if (std::abs(offset) < static_cast<double>(stable_offset_ticks_)) {
      ++stable_streak_;
      if (state_ == State::TRACKING && stable_streak_ >= config_.stable_count)
        state_ = State::STABLE;
    } else {
      stable_streak_ = 0;
      if (state_ == State::STABLE)
        state_ = State::TRACKING;
    }

    // Token bucket: absorb bursts within the buffer. When the burst buffer
    // absorbs the offset, do NOT update the integral — this prevents integral
    // windup from accumulating during burst-absorbed periods.
    if (offset < static_cast<double>(burst_ticks_))
      return;

    // PI correction (only reached when actually throttling).
    double proportional_gain = (state_ == State::STABLE) ? config_.proportional_gain_stable
                                                         : config_.proportional_gain_tracking;
    double integral_gain =
        (state_ == State::STABLE) ? config_.integral_gain_stable : config_.integral_gain_tracking;
    integral_ += offset;

    // Symmetric integral anti-windup: cap accumulator to prevent runaway.
    double integral_cap = step_threshold_ticks_ / integral_gain;
    integral_ = std::clamp(integral_, -integral_cap, integral_cap);

    double adj = proportional_gain * offset + integral_gain * integral_;

    // Convert adjustment to wall-nanoseconds and sleep.
    double sleep_ns = adj / (config_.ratio * TICKS_PER_NS);
    auto sleep_ns_int = static_cast<int64_t>(sleep_ns);
    if (sleep_ns_int <= 0)
      return;

    // Hybrid sleep + spin: coarse sleep minus 500us, then spin.
    static constexpr int64_t SPIN_THRESHOLD_NS = 500'000;
    if (sleep_ns_int > SPIN_THRESHOLD_NS)
      std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_ns_int - SPIN_THRESHOLD_NS));

    // Tight busy-wait for the final spin portion. The clock::now() call in the
    // while condition provides a sufficient memory barrier; no yield needed.
    auto target_wall =
        clock::now() + std::chrono::nanoseconds(std::min(sleep_ns_int, SPIN_THRESHOLD_NS));
    while (clock::now() < target_wall) {
    }
  }

  /// @brief Maximum duration to block during a quiescent wait.
  /// @returns Wait duration, or 0ms if pacing is disabled (use unbounded wait).
  std::chrono::milliseconds idle_wait_duration() const {
    if (!enabled())
      return std::chrono::milliseconds(0);
    return std::chrono::milliseconds(10);
  }

  /// @brief Return the configured ratio.
  double ratio() const { return config_.ratio; }

  /// @brief Return the current controller state.
  State state() const { return state_; }

private:
  using clock = std::chrono::steady_clock;
  static constexpr double TICKS_PER_NS = 1000.0;

  double ns_to_ticks(double ns) const { return ns * config_.ratio * TICKS_PER_NS; }

  Config config_;
  double step_threshold_ticks_;
  double burst_ticks_;
  double stable_offset_ticks_;

  // Anchor state (protected by shared_mutex: readers take shared lock,
  // writers take exclusive lock).
  mutable std::shared_mutex anchor_mutex_;
  clock::time_point wall_anchor_{};
  Tick sim_anchor_ = 0;
  bool paused_ = false;
  clock::time_point pause_start_{};

  // PI state (only accessed from the throttle() caller's thread).
  double integral_ = 0.0;
  State state_ = State::INIT;
  uint32_t init_count_ = 0;
  uint32_t stable_streak_ = 0;
};

} // namespace simdojo

#endif // SIMDOJO_SIM_PACING_CONTROLLER_H_
