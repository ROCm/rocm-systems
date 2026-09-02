// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simdojo/sim/simulation.h"

#include "util/log.h"

#include <algorithm>
#include <cassert>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>

namespace simdojo {

namespace {

/// @brief Claims an engine for the duration of one execution call.
///
/// @details The engine's queue and partition tick belong to exactly one
/// driving loop at a time. A second loop entered from inside a handler pops
/// from the queue the outer frame is iterating and rewrites the tick
/// underneath it, which is not a misbehaviour that shows up as a wrong number
/// -- it is a run that silently simulated something other than what it
/// reports.
class ExecutionGuard {
public:
  /// @brief Claim the engine, or report that someone else holds it. The form
  /// run() needs, which reports a refusal rather than throwing.
  ExecutionGuard(std::atomic<bool> &executing, std::nothrow_t) : executing_(executing) {
    bool expected = false;
    owned_ = executing_.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
  }

  /// @throws std::logic_error if the engine is already executing.
  explicit ExecutionGuard(std::atomic<bool> &executing) : ExecutionGuard(executing, std::nothrow) {
    // The throw runs ~ExecutionGuard(), which is correct precisely because
    // owned_ is false: it releases nothing it did not claim.
    if (!owned_)
      throw std::logic_error(already_executing);
  }

  ~ExecutionGuard() {
    if (owned_)
      executing_.store(false, std::memory_order_release);
  }

  /// @retval true This guard claimed the engine.
  explicit operator bool() const { return owned_; }

  /// @brief The one wording both the throw and run()'s ExitStatus report.
  static constexpr const char *already_executing = "simulation engine is already executing";

  ExecutionGuard(const ExecutionGuard &) = delete;
  ExecutionGuard &operator=(const ExecutionGuard &) = delete;

private:
  std::atomic<bool> &executing_;
  bool owned_ = false;
};

std::string link_endpoints(const Link &link) {
  return link.src()->full_path() + " -> " + link.dst()->full_path();
}

} // namespace

void PartitionContext::drain_incoming() {
  for (auto &queue : incoming)
    queue->drain_into(event_queue);
}

SimulationEngine::SimulationEngine(Config config)
    : config_(config), pacer_(PacingController::Config{.ratio = config.wall_clock_ratio}) {}

SimulationEngine::~SimulationEngine() {
  if (created_)
    shutdown();
}

void SimulationEngine::create() {
  assert(!created_ && "create() called twice without shutdown()");
  if (config_.num_threads == 0)
    throw std::invalid_argument("SimulationEngine requires at least one worker thread");

  if (topology_.partitions().empty()) {
    if (config_.num_threads > 1) {
      throw std::invalid_argument(
          "multi-threaded SimulationEngine requires an explicit topology partition policy");
    }
    topology_.partition_balanced(1);
  }
  setup_partitions();

  done_.store(false, std::memory_order_release);
  startup_complete_.store(false, std::memory_order_release);
  startup_failed_.store(false, std::memory_order_release);
  components_shut_down_ = false;
  active_primaries_.store(0, std::memory_order_release);
  has_primaries_.store(false, std::memory_order_release);
  exit_status_ = {};
  exit_set_ = false;
  executing_.store(false, std::memory_order_release);
  current_time_.store(0, std::memory_order_release);
  global_lbts_.store(0, std::memory_order_release);

  initialize_components();
  created_ = true;
}

void SimulationEngine::shutdown() {
  if (!created_)
    return;

  // Signal done. Workers will see this after the next barrier and exit.
  done_.store(true, std::memory_order_release);

  // In single-threaded mode, wake idle CV. In multi-threaded mode,
  // threads will see done_ after their current barrier epoch completes.
  wake_partition(0);

  workers_.clear();
  barrier_.reset();
  // Capture, do not propagate: engine cleanup below MUST still run, and shutdown()
  // is reached from ~SimulationEngine(), from rj_vm_run() on the engine's own
  // background thread, and across the C API — none of which can absorb an
  // exception. A failing component hook is reported, not thrown.
  std::exception_ptr component_failure = shutdown_components();

  {
    // Exclusive: a foreign thread may be mid-wake or mid-async-schedule on these
    // vectors. Workers are already joined above, so nothing we join can be waiting
    // on this lock. Taken after shutdown_components() so component teardown, which
    // may still schedule, is not blocked.
    std::unique_lock<std::shared_mutex> lock(contexts_mutex_);
    contexts_.clear();
    async_queues_.clear();
  }
  created_ = false;
  report_component_failure(component_failure, "engine shutdown");
}

void SimulationEngine::setup_partitions() {
  const uint32_t num_threads = config_.num_threads;
  if (topology_.partitions().size() != num_threads) {
    throw std::invalid_argument(
        "SimulationEngine topology partition count must match the worker thread count");
  }

  // Validate cross-partition link constraints.
  for (auto &link : topology_.links()) {
    if (link->is_cross_partition()) {
      if (link->latency() == 0) {
        throw std::invalid_argument("SimulationEngine cross-partition link " +
                                    link_endpoints(*link) + " requires positive latency");
      }
      if (dynamic_cast<QueuedLink *>(link.get()) != nullptr) {
        throw std::invalid_argument("SimulationEngine QueuedLink " + link_endpoints(*link) +
                                    " must not cross partition boundaries");
      }
    }
  }

  contexts_.reserve(num_threads);
  for (uint32_t i = 0; i < num_threads; ++i)
    contexts_.push_back(std::make_unique<PartitionContext>(i, num_threads));

  async_queues_.reserve(num_threads);
  for (uint32_t i = 0; i < num_threads; ++i)
    async_queues_.push_back(std::make_unique<AsyncQueue>());

  // Set engine pointer on all components.
  for (auto &part : topology_.partitions()) {
    for (auto *comp : part.components)
      comp->set_engine(this);
  }
}

ExitStatus SimulationEngine::run() {
  assert(created_ && "run() called before create()");
  const uint32_t num_threads = config_.num_threads;

  // Reported rather than thrown: run() may execute on a background thread (the
  // LD_PRELOAD interposer's local VM) whose top-level lambda has no catch, so
  // an escaping exception would call std::terminate.
  ExecutionGuard execution(executing_, std::nothrow);
  if (!execution) {
    return ExitStatus(ExitReason::INTERRUPTED, current_time_.load(std::memory_order_acquire),
                      ExecutionGuard::already_executing, 1);
  }

  // A component startup() can throw, and for the same reason it must not escape
  // here: it would call std::terminate AND leave wait_until_started() blocked
  // forever (startup_complete_ never set). ensure_started() catches it, latches
  // readiness with a failure flag so waiters wake and can unwind, and reports
  // false rather than running the epoch loop against half-started components.
  // That also makes this create() generation terminal: partition/async event
  // queues, primary counters, and per-component state from the partial attempt
  // are left intact, so re-running startup on top of them would double-schedule
  // events and double-register primaries. Deliberately a RUNTIME guard with NO
  // paired assert: an assert would abort in assertion-enabled builds and return
  // in release ones, so the API's behaviour would depend on the build. Copied
  // under exit_mutex_ like the normal return below: a foreign thread can be
  // inside request_exit() assigning exit_status_, whose message is a std::string.
  if (!ensure_started(/*propagate_failure=*/false)) {
    std::lock_guard<std::mutex> lock(exit_mutex_);
    return exit_status_;
  }
  pacer_.anchor(current_time_.load(std::memory_order_acquire));

  if (num_threads == 1) {
    worker_loop(0);
  } else {
    barrier_ = std::make_unique<std::barrier<std::function<void()>>>(
        static_cast<std::ptrdiff_t>(num_threads),
        std::function<void()>([this]() { barrier_completion(); }));

    for (uint32_t i = 0; i < num_threads; ++i)
      workers_.emplace_back([this, i]() { worker_loop(i); });
    workers_.clear();
    barrier_.reset();
  }

  // Copy under the lock: a foreign thread can still be inside request_exit()
  // assigning exit_status_ as run() unwinds, and the copy reads its message string.
  std::lock_guard<std::mutex> lock(exit_mutex_);
  return exit_status_;
}

void SimulationEngine::fail_startup(std::string message) {
  // Records the status and marks the generation done in ONE critical section, so a
  // concurrent request_exit() cannot replace this failure with EXIT_REQUEST/code 0 and
  // report a generation whose components never started as a clean run.
  set_terminal_exit(ExitReason::INTERRUPTED, current_time_.load(std::memory_order_acquire),
                    std::move(message), /*code=*/1);
  // Unwind BEFORE publishing readiness, so a waiter that observes the failure also
  // observes a generation that is torn down rather than one still running component
  // shutdown hooks. Nothing here can escape past the latch: shutdown_components()
  // and report_component_failure() are both noexcept and catch each callback.
  report_component_failure(shutdown_components(), "startup unwind");
  latch_startup(/*failed=*/true);
}

void SimulationEngine::latch_startup_if_unlatched(bool failed) {
  if (startup_complete_.load(std::memory_order_acquire))
    return;
  if (failed) {
    // Same reasoning as fail_startup(): the flag latched here is run()/step()'s
    // terminal guard, so it needs a status behind it, recorded together with done_.
    set_terminal_exit(ExitReason::INTERRUPTED, current_time_.load(std::memory_order_acquire),
                      "engine thread returned without completing component startup",
                      /*code=*/1);
  }
  latch_startup(failed);
}

void SimulationEngine::report_component_failure(std::exception_ptr failure,
                                                const char *phase) noexcept {
  if (!failure)
    return;
  // Outer catch-all: this is the last step of a teardown that must not throw, and
  // the reporting path itself formats and allocates.
  try {
    try {
      std::rethrow_exception(failure);
    } catch (const std::exception &e) {
      util::Logger::warn("SimulationEngine: a component shutdown() threw during ", phase, ": ",
                         e.what());
    } catch (...) {
      util::Logger::warn("SimulationEngine: a component shutdown() threw a non-standard exception "
                         "during ",
                         phase);
    }
  } catch (...) {
    // Reporting itself failed; nothing further here is safe or useful.
  }
}

bool SimulationEngine::wait_until_started() const {
  while (!startup_complete_.load(std::memory_order_acquire))
    startup_complete_.wait(false, std::memory_order_acquire);
  return !startup_failed_.load(std::memory_order_acquire);
}

bool SimulationEngine::ensure_started(bool propagate_failure) {
  // startup_complete_ is the "this generation has been started" flag, latched by
  // a successful startup below, by fail_startup(), and by the thread-death
  // backstop. A generation latched by either of the latter two is terminal: its
  // partial event/primary/component state is still live, so re-running startup
  // would double-schedule events and double-register primaries. Report failure;
  // the caller must shutdown() + create() to retry.
  if (startup_complete_.load(std::memory_order_acquire))
    return !startup_failed();
  // A foreground caller gets the throw after fail_startup() has recorded the
  // same terminal ExitStatus run() would, published it to wait_until_started()
  // waiters, and unwound.
  try {
    startup_components();
  } catch (const std::exception &e) {
    fail_startup(std::string("component startup failed: ") + e.what());
    if (propagate_failure)
      throw;
  } catch (...) {
    // A non-std::exception throw would otherwise still call std::terminate on a
    // background run() thread. Latch failure and report it for those too.
    fail_startup("component startup failed with a non-standard exception");
    if (propagate_failure)
      throw;
  }
  if (startup_failed())
    return false;

  // Armed before readiness is latched: latching is what marks this generation
  // started, so a throw from the push would otherwise leave a permanently
  // "started" generation with its configured tick cap silently unarmed.
  if (config_.max_ticks > 0 && config_.num_threads == 1) {
    max_ticks_event_.set_handler([this](Tick ts, Message *) {
      set_exit(ExitReason::COMPLETED, ts, "max ticks reached");
      done_.store(true, std::memory_order_release);
    });
    contexts_[0]->event_queue.push(
        EventQueueEntry{config_.max_ticks, 0, &max_ticks_event_, nullptr});
  }

  // Anchored here rather than in run(), because all three entry points start the
  // engine through this one. An unanchored PacingController maps wall-now against
  // a default-constructed epoch, so on a paced engine driven only by step() or
  // run_bounded() every schedule_event_now() would be stamped at roughly the
  // machine's uptime. run() re-anchors on each of its own calls, which is what
  // lets `run_until_idle(); run();` resume without the pacer believing it is
  // already far behind.
  pacer_.anchor(current_time_.load(std::memory_order_acquire));

  // Publish readiness only after every component's startup() has run, so an
  // embedding that launched run() on a background thread (the LD_PRELOAD
  // interposer's local VM) does not expose a half-started device.
  latch_startup(/*failed=*/false);
  return true;
}

bool SimulationEngine::step() {
  assert(created_ && "step() called before create()");
  assert(config_.num_threads == 1 && "step() requires single-threaded mode");

  ExecutionGuard execution(executing_);

  // Checked, not asserted, for the reason spelled out in run_bounded(): a
  // shutdown() engine still reports itself started.
  if (contexts_.empty())
    return false;

  if (!ensure_started(/*propagate_failure=*/true))
    return false;

  if (done_.load(std::memory_order_acquire))
    return false;

  drain_async_events();

  PartitionContext &ctx = *contexts_[0];

  if (ctx.event_queue.empty()) {
    if (check_termination(TICK_MAX))
      return false;
    return true;
  }

  Tick step_tick = ctx.event_queue.next_event_time();
  while (!ctx.event_queue.empty() && ctx.event_queue.next_event_time() == step_tick) {
    auto entry = ctx.event_queue.pop();
    // Published before the handler runs, for the reason spelled out in
    // run_bounded(): request_exit() and schedule_event_now() read this clock
    // from inside the handler.
    current_time_.store(step_tick, std::memory_order_release);
    process_event(ctx, entry);
    if (done_.load(std::memory_order_acquire))
      return false;
  }

  current_time_.store(step_tick, std::memory_order_release);
  return true;
}

Tick SimulationEngine::run_bounded(const bool &done) {
  // Checked rather than asserted, and before anything touches contexts_[0]:
  // popping partition 0's queue while its own worker thread is running it is a
  // data race, and asserts are compiled out of the builds this ships as.
  if (config_.num_threads != 1)
    throw std::invalid_argument("run_bounded() requires a single-partition engine");

  ExecutionGuard execution(executing_);

  // Checked for the same reason the partition count is, and it is the likelier
  // mistake of the two: shutdown() clears contexts_ but leaves the startup latch
  // set, so ensure_started() below would report a started engine and the deref
  // would index an empty vector with no diagnostic.
  if (contexts_.empty())
    return current_time_.load(std::memory_order_acquire);

  if (!ensure_started(/*propagate_failure=*/true))
    return current_time_.load(std::memory_order_acquire);

  Tick reached = contexts_[0]->current_tick();
  for (;;) {
    // Every iteration, matching what the worker loop does at each of its tick
    // boundaries: this is the one driver whose single call spans many ticks, so
    // it is the one that would otherwise hand control back to its caller with
    // an async event -- a doorbell, a compute unit's own deferred work --
    // still invisible in the queue. It costs one acquire load per partition
    // when nothing is pending.
    drain_async_events();
    // Re-read rather than cached across handler calls: a handler is free to call
    // shutdown(), which destroys every PartitionContext, and a reference bound
    // once before the loop would be read again on the way out.
    if (contexts_.empty())
      break;
    PartitionContext &ctx = *contexts_[0];
    if (done || done_.load(std::memory_order_acquire) || ctx.event_queue.empty())
      break;

    auto entry = ctx.event_queue.pop();
    // Published before the handler runs, not after: request_exit() stamps its
    // tick from this, schedule_event_now() timestamps events from it, and a
    // handler may read global_time(). Publishing afterwards left all three
    // reading the previous event's tick, and an async event stamped in the past
    // sorts ahead of the queue head and rewinds the partition clock.
    current_time_.store(entry.timestamp, std::memory_order_release);
    process_event(ctx, entry);
    reached = entry.timestamp;
  }

  return reached;
}

Tick SimulationEngine::run_until_idle() {
  static constexpr bool never = false;
  return run_bounded(never);
}

uint64_t SimulationEngine::events_processed() const {
  // Shared, like every other foreign-thread reader of contexts_: shutdown()
  // clears the vector under the same lock held exclusively.
  std::shared_lock<std::shared_mutex> lock(contexts_mutex_);
  uint64_t total = 0;
  for (const auto &ctx : contexts_)
    total += ctx->events_processed();
  return total;
}

void SimulationEngine::worker_loop(PartitionID partition_id) {
  PartitionContext &ctx = *contexts_[partition_id];
  uint32_t num_threads = config_.num_threads;

  while (true) {
    if (done_.load(std::memory_order_acquire) && num_threads == 1)
      break; // Single-threaded: safe to exit immediately.

    if (num_threads == 1) {
      // Drain async events first (doorbells, external stimuli).
      drain_async_events();

      // Single-threaded: drain all events in timestamp order.
      // Drain async events at each tick boundary so that events from other
      // threads (e.g. doorbell poll threads) are merged promptly instead of
      // waiting for the main queue to empty — which may never happen when
      // CU work events continuously reschedule.
      //
      // Within a tick, defer pushes so that handler reschedules don't
      // interleave with pops (avoids O(N log N) heap churn per tick).
      Tick last_drained_tick = 0;
      while (!ctx.event_queue.empty()) {
        Tick next_tick = ctx.event_queue.next_event_time();
        if (next_tick > last_drained_tick) {
          last_drained_tick = next_tick;
          drain_async_events();
        }
        auto entry = ctx.event_queue.pop();
        process_event(ctx, entry);
        if (done_.load(std::memory_order_acquire))
          return;
      }

      // Queue drained now update global time for external observers.
      current_time_.store(ctx.event_queue.current_tick(), std::memory_order_release);

      // Throttle: PI-controlled pacing against wall clock.
      pacer_.throttle(ctx.event_queue.current_tick());

      drain_async_events();

      // If async events added new work, continue processing.
      if (!ctx.event_queue.empty())
        continue;

      // Quiescent, check termination.
      if (check_termination(TICK_MAX))
        break;

      // Primaries still active so wait for an async event. With wall-clock
      // sync, use a timed sleep+check loop so we can advance sim time
      // proportionally to elapsed wall time during idle periods. Without
      // pacing, use atomic::wait() for an efficient unbounded wait.
      pacer_.pause();
      {
        auto timeout = pacer_.idle_wait_duration();
        if (timeout.count() > 0) {
          // Timed idle wait: sleep+check loop (std::atomic has no wait_for).
          while (!ctx.idle_wakeup_.load(std::memory_order_acquire) &&
                 !done_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(timeout);
          }
        } else {
          // Unbounded idle wait with periodic done_ check. Using a short
          // sleep loop instead of atomic::wait avoids lost-notification races
          // between request_exit() and the wait entry point.
          using namespace std::chrono_literals;
          while (!ctx.idle_wakeup_.load(std::memory_order_acquire) &&
                 !done_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(1ms);
          }
        }
      }
      if (done_.load(std::memory_order_acquire))
        break;
      ctx.idle_wakeup_.store(false, std::memory_order_release);

      // After waking from idle, resume the pacing clock.
      pacer_.resume();

      drain_async_events();
    } else {
      // Multi-threaded barrier-based LBTS path.
      //
      // Each epoch:
      //   1. Drain incoming cross-partition events + async events
      //   2. Process events <= global_lbts
      //   3. Publish local_next (earliest possible future activity)
      //   4. Arrive at barrier (completion function computes new global_lbts)

      // Check done_ early. Use arrive_and_drop to permanently decrement the
      // barrier's expected count, then exit. This prevents deadlock when some
      // workers exit while others are still processing.
      if (done_.load(std::memory_order_acquire)) {
        ctx.local_next.store(TICK_MAX, std::memory_order_relaxed);
        barrier_->arrive_and_drop();
        return;
      }

      // Phase 1: Drain incoming + async.
      ctx.drain_incoming();
      drain_async_for_partition(ctx);

      // Phase 2: Process all events with timestamp <= global LBTS.
      Tick lbts = global_lbts_.load(std::memory_order_acquire);
      while (!ctx.event_queue.empty() && ctx.event_queue.next_event_time() <= lbts) {
        auto entry = ctx.event_queue.pop();
        process_event(ctx, entry);
        if (done_.load(std::memory_order_acquire)) {
          ctx.local_next.store(TICK_MAX, std::memory_order_relaxed);
          barrier_->arrive_and_drop();
          return;
        }
      }

      // Phase 3: Publish local_next.
      Tick next = std::min(ctx.event_queue.next_event_time(), ctx.local_min_outgoing);
      ctx.local_next.store(next, std::memory_order_relaxed);
      ctx.local_min_outgoing = TICK_MAX;

      // Phase 4: Barrier (completion function computes new global_lbts).
      barrier_->arrive_and_wait();
      if (done_.load(std::memory_order_acquire)) {
        // request_exit() can set done_ while peers released from the same phase
        // are already entering the next one. Contribute this worker's arrival
        // instead of abandoning that phase and stranding those peers.
        ctx.local_next.store(TICK_MAX, std::memory_order_relaxed);
        barrier_->arrive_and_drop();
        return;
      }
    }
  }
}

void SimulationEngine::barrier_completion() {
  // Compute global LBTS = min of all partitions' local_next.
  Tick new_lbts = TICK_MAX;
  for (auto &ctx : contexts_) {
    Tick ln = ctx->local_next.load(std::memory_order_relaxed);
    new_lbts = std::min(new_lbts, ln);
  }

  // Update global LBTS and current time.
  global_lbts_.store(new_lbts, std::memory_order_release);
  if (new_lbts != TICK_MAX)
    current_time_.store(new_lbts, std::memory_order_release);

  if (check_termination(new_lbts))
    return;
}

void SimulationEngine::process_event(PartitionContext &ctx, EventQueueEntry &entry) {
  ctx.event_queue.set_current_tick(entry.timestamp);

  if (entry.event->has_handler()) {
    entry.event->execute(entry.timestamp, entry.message.get());
    // After execute(), so a handler that threw is not counted as work done. A
    // relaxed load+store rather than fetch_add: the counter has exactly one
    // writer, and this is the hottest line in the simulator, where a
    // lock-prefixed read-modify-write costs an order of magnitude more.
    ctx.events_processed_.store(ctx.events_processed_.load(std::memory_order_relaxed) + 1,
                                std::memory_order_relaxed);
  }
}

void SimulationEngine::schedule_event(Event *event, Tick timestamp,
                                      std::unique_ptr<Message> message) {
  Component *target = event->target();
  assert(target != nullptr && "schedule_event: event has no target component");
  PartitionID pid = target->partition_id();
  assert(pid < contexts_.size() && "schedule_event: target partition ID out of range");
  contexts_[pid]->event_queue.push(EventQueueEntry{timestamp, 0, event, std::move(message)});
}

void SimulationEngine::send_cross_partition(PartitionID src_partition, PartitionID dst_partition,
                                            Event *event, Tick timestamp,
                                            std::unique_ptr<Message> message) {
  assert(src_partition < contexts_.size());
  assert(dst_partition < contexts_.size());

  PartitionContext &src_ctx = *contexts_[src_partition];
  PartitionContext &dst_ctx = *contexts_[dst_partition];

  // Deposit into the destination partition's incoming queue for this source.
  dst_ctx.incoming[src_partition]->push(EventQueueEntry{timestamp, 0, event, std::move(message)});

  // Update the source partition's min_outgoing (accessed only by owning thread).
  if (timestamp < src_ctx.local_min_outgoing)
    src_ctx.local_min_outgoing = timestamp;
}

void SimulationEngine::register_as_primary() {
  active_primaries_.fetch_add(1, std::memory_order_release);
  has_primaries_.store(true, std::memory_order_release);
}

void SimulationEngine::primary_release() {
  [[maybe_unused]] uint32_t prev = active_primaries_.fetch_sub(1, std::memory_order_release);
  assert(prev > 0 && "primary_release called without matching register_as_primary or retain");
  // Wake idle engine so it can check termination (e.g., doorbell monitor releasing primary).
  if (prev == 1)
    wake_partition(0);
}

void SimulationEngine::primary_retain() {
  active_primaries_.fetch_add(1, std::memory_order_release);
}

bool SimulationEngine::is_fully_quiescent() const {
  for (auto &ctx : contexts_) {
    // Check incoming queues for undelivered cross-partition messages.
    for (auto &q : ctx->incoming) {
      if (!q->empty())
        return false;
    }
    // Check for any processable local events.
    if (!ctx->event_queue.empty())
      return false;
  }
  return true;
}

bool SimulationEngine::all_primaries_done() const {
  return has_primaries_.load(std::memory_order_acquire) &&
         active_primaries_.load(std::memory_order_acquire) == 0;
}

bool SimulationEngine::check_termination(Tick lbts) {
  if (all_primaries_done()) {
    // All primaries have signaled done. Enter draining phase: continue
    // processing until the system is fully quiescent (all incoming queues
    // empty and all event queues drained). This ensures in-flight
    // cross-partition messages are fully delivered before termination.
    if (!is_fully_quiescent())
      return false;
    set_exit(ExitReason::COMPLETED, lbts, "all primaries completed");
    done_.store(true, std::memory_order_release);
    return true;
  }
  // Multi-threaded max_ticks: terminate when global LBTS reaches max_ticks.
  if (config_.max_ticks > 0 && config_.num_threads > 1) {
    if (lbts >= config_.max_ticks) {
      set_exit(ExitReason::COMPLETED, config_.max_ticks, "max ticks reached");
      done_.store(true, std::memory_order_release);
      return true;
    }
  }
  if (lbts == TICK_MAX) {
    // If primaries are still active, they are promising future work
    // (via async events). Don't terminate.
    if (active_primaries_.load(std::memory_order_acquire) > 0)
      return false;
    // If await_primaries is set (e.g., KFD driver mode), don't terminate on
    // quiescence until at least one primary has registered. This keeps the
    // engine alive while waiting for external stimuli (doorbells).
    if (config_.await_primaries && !has_primaries_.load(std::memory_order_acquire))
      return false;
    set_exit(ExitReason::COMPLETED, current_time_.load(std::memory_order_acquire),
             "all partitions quiescent");
    done_.store(true, std::memory_order_release);
    return true;
  }
  return false;
}

void SimulationEngine::wake_partition_locked(PartitionID pid) {
  if (config_.num_threads != 1 || pid >= contexts_.size())
    return;
  PartitionContext &ctx = *contexts_[pid];
  ctx.idle_wakeup_.store(true, std::memory_order_release);
  ctx.idle_wakeup_.notify_one();
}

void SimulationEngine::wake_partition(PartitionID pid) {
  std::shared_lock<std::shared_mutex> lock(contexts_mutex_);
  wake_partition_locked(pid);
}

void SimulationEngine::request_exit(std::string reason, int code) {
  Tick tick = current_time_.load(std::memory_order_acquire);
  {
    std::lock_guard<std::mutex> lock(exit_mutex_);
    // done_.store must be inside the lock: without it, a concurrent caller
    // could acquire the lock after we release it but before we store done_=true,
    // see done_=false, and overwrite exit_status_.
    if (!done_.load(std::memory_order_acquire)) {
      exit_status_ = ExitStatus(ExitReason::EXIT_REQUEST, tick, std::move(reason), code);
      exit_set_ = true;
    }
    done_.store(true, std::memory_order_release);
  }

  // In single-threaded mode, wake the idle wait. Guarded: the engine thread may be
  // inside shutdown() destroying the contexts while the host thread requests exit.
  wake_partition(0);
  // In multi-threaded mode, done_ is checked after each barrier epoch.
  // No explicit wake needed since threads will see it at the next barrier.
}

void SimulationEngine::schedule_event_async(Event *event, Tick timestamp,
                                            std::unique_ptr<Message> message) {
  Component *target = event->target();
  assert(target != nullptr && "schedule_event_async: event has no target component");
  PartitionID pid = target->partition_id();

  // Shared: keeps async_queues_ and contexts_ alive for the duration. Producers do
  // not exclude one another, so this stays off the critical path; only shutdown()
  // blocks here. Held across the wake so the context cannot be freed under us.
  std::shared_lock<std::shared_mutex> lock(contexts_mutex_);
  if (pid >= async_queues_.size()) {
    // shutdown() has cleared the queues, so the event has nowhere left to go. A
    // non-empty queue vector here instead means an out-of-range partition ID, which
    // is a caller bug. (Checked against async_queues_ rather than created_: that
    // flag is written outside the lock, so reading it here would itself race.)
    assert(async_queues_.empty() && "schedule_event_async: target partition ID out of range");
    return;
  }
  auto &aq = *async_queues_[pid];
  {
    std::lock_guard<std::mutex> qlock(aq.mutex);
    aq.events.push_back(EventQueueEntry{timestamp, 0, event, std::move(message)});
    aq.pending.store(true, std::memory_order_release);
  }

  // Wake the target partition so idle workers pick up the new event.
  wake_partition_locked(pid);
  // In multi-threaded mode, async events are drained at each barrier epoch.
}

void SimulationEngine::schedule_event_now(Event *event, std::unique_ptr<Message> message) {
  Tick timestamp =
      pacer_.enabled() ? pacer_.sim_tick_now() : current_time_.load(std::memory_order_acquire);
  schedule_event_async(event, timestamp, std::move(message));
}

void SimulationEngine::drain_async_events() {
  for (uint32_t i = 0; i < async_queues_.size(); ++i) {
    auto &aq = *async_queues_[i];
    if (!aq.pending.load(std::memory_order_acquire))
      continue;
    std::lock_guard<std::mutex> lock(aq.mutex);
    for (auto &e : aq.events)
      contexts_[i]->event_queue.push(std::move(e));
    aq.events.clear();
    aq.pending.store(false, std::memory_order_release);
  }
}

void SimulationEngine::set_exit(ExitReason reason, Tick tick, std::string message, int code) {
  std::lock_guard<std::mutex> lock(exit_mutex_);
  // First writer wins. Use an explicit flag rather than message.empty() to
  // avoid false negatives when the message string happens to be empty.
  if (!exit_set_) {
    exit_status_ = ExitStatus(reason, tick, std::move(message), code);
    exit_set_ = true;
  }
}

void SimulationEngine::set_terminal_exit(ExitReason reason, Tick tick, std::string message,
                                         int code) {
  std::lock_guard<std::mutex> lock(exit_mutex_);
  // Deliberately NOT first-writer-wins, unlike set_exit(). Only terminal failures
  // reach here, and a terminal failure outranks whatever was recorded before it: a
  // request_exit() that already stored its default code-0 EXIT_REQUEST -- including
  // one issued from inside a startup callback that then threw -- would otherwise
  // survive, and run() would hand back a success status for a generation whose
  // components never started. done_ is stored in this same critical section for the
  // mirror-image reason: request_exit() gates its own overwrite on done_ read under
  // this lock, so releasing before storing it would let a LATER request_exit()
  // replace this failure.
  exit_status_ = ExitStatus(reason, tick, std::move(message), code);
  exit_set_ = true;
  done_.store(true, std::memory_order_release);
}

void SimulationEngine::initialize_components() {
  for (auto &part : topology_.partitions()) {
    for (auto *comp : part.components)
      comp->initialize();
  }
}

void SimulationEngine::startup_components() {
  for (auto &part : topology_.partitions()) {
    for (auto *comp : part.components)
      comp->startup();
  }
}

std::exception_ptr SimulationEngine::shutdown_components() noexcept {
  // shutdown() pairs with initialize() (both run over every component), so it must
  // shut down every INITIALIZED component in reverse topology order — not only the
  // ones whose startup() ran. create() initializes all components, so a
  // create(); shutdown() with no run()/step() still releases their resources, and a
  // partial-startup unwind still cleans up every initialized component. Guarded so
  // it runs exactly once per create() generation: both the startup-failure unwind
  // and the later engine shutdown() reach here, and a component must not be shut
  // down twice. Reset by create().
  if (components_shut_down_)
    return {};
  components_shut_down_ = true;

  // Iterate in place rather than building a temporary vector: this runs on the
  // teardown path reached from ~SimulationEngine(), so a bad_alloc from that
  // temporary would skip EVERY component's cleanup and escape a noexcept
  // destructor. Reverse partition order, reverse component order within each.
  //
  // Each callback is isolated: Component::shutdown() is not noexcept, and the
  // generation is already marked shut down above, so letting one throw would
  // permanently skip cleanup for the components after it with no way to retry.
  std::exception_ptr first_failure;
  auto &partitions = topology_.partitions();
  for (auto part = partitions.rbegin(); part != partitions.rend(); ++part) {
    for (auto comp = part->components.rbegin(); comp != part->components.rend(); ++comp) {
      try {
        (*comp)->shutdown();
      } catch (...) {
        if (!first_failure)
          first_failure = std::current_exception();
      }
    }
  }
  return first_failure;
}

Tick SimulationEngine::compute_async_floor(const PartitionContext &ctx) const {
  if (config_.num_threads == 1)
    return 0;
  // Floor at the global LBTS, clamped to at least the partition's current tick.
  Tick floor = ctx.event_queue.current_tick();
  Tick lbts = global_lbts_.load(std::memory_order_acquire);
  if (lbts != TICK_MAX)
    floor = std::max(floor, lbts);
  return floor;
}

void SimulationEngine::drain_async_for_partition(PartitionContext &ctx) {
  auto &aq = *async_queues_[ctx.partition_id];
  if (!aq.pending.load(std::memory_order_acquire))
    return;
  std::lock_guard<std::mutex> lock(aq.mutex);
  Tick floor = compute_async_floor(ctx);
  for (auto &e : aq.events) {
    if (e.timestamp < floor)
      e.timestamp = floor;
    ctx.event_queue.push(std::move(e));
  }
  aq.events.clear();
  aq.pending.store(false, std::memory_order_release);
}

} // namespace simdojo
