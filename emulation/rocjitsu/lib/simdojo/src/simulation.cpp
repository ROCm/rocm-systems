// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simdojo/sim/simulation.h"

#include <algorithm>
#include <cassert>

namespace simdojo {

void PartitionContext::drain_incoming() {
  for (auto &queue : incoming)
    queue->drain_into(event_queue);
}

SimulationEngine::SimulationEngine(Config config)
    : config_(config), pacer_(PacingController::Config{.ratio = config.wall_clock_ratio}) {}

SimulationEngine::~SimulationEngine() {
  if (built_)
    shutdown();
}

void SimulationEngine::build() {
  assert(!built_ && "build() called twice without shutdown()");
  if (topology_.partitions().empty())
    topology_.partition(config_.num_threads);
  setup_partitions();

  done_.store(false, std::memory_order_release);
  active_primaries_.store(0, std::memory_order_release);
  has_primaries_.store(false, std::memory_order_release);
  exit_status_ = {};
  exit_set_ = false;
  current_time_.store(0, std::memory_order_release);
  global_lbts_.store(0, std::memory_order_release);

  initialize_components();
  built_ = true;
}

void SimulationEngine::shutdown() {
  if (!built_)
    return;

  // Signal done. Workers will see this after the next barrier and exit.
  done_.store(true, std::memory_order_release);

  // In single-threaded mode, wake idle CV. In multi-threaded mode,
  // threads will see done_ after their current barrier epoch completes.
  if (config_.num_threads == 1 && !contexts_.empty()) {
    auto &ctx = *contexts_[0];
    ctx.idle_wakeup_.store(true, std::memory_order_release);
    ctx.idle_wakeup_.notify_one();
  }

  workers_.clear();
  barrier_.reset();
  shutdown_components();

  running_ = false;
  contexts_.clear();
  async_queues_.clear();
  built_ = false;
}

void SimulationEngine::setup_partitions() {
  const uint32_t num_threads = config_.num_threads;
  assert(num_threads > 0);
  assert(topology_.partitions().size() == num_threads);

  contexts_.reserve(num_threads);
  for (uint32_t i = 0; i < num_threads; ++i)
    contexts_.push_back(std::make_unique<PartitionContext>(i, num_threads));

  async_queues_.reserve(num_threads);
  for (uint32_t i = 0; i < num_threads; ++i)
    async_queues_.push_back(std::make_unique<AsyncQueue>());

  // Validate cross-partition link constraints.
  for (auto &link : topology_.links()) {
    if (link->is_cross_partition()) {
      assert(link->latency() > 0 && "cross-partition links require positive latency for LBTS");
      assert(dynamic_cast<QueuedLink *>(link.get()) == nullptr &&
             "QueuedLinks must not cross partition boundaries (they bypass the LBTS protocol)");
    }
  }

  // Set engine pointer on all components.
  for (auto &part : topology_.partitions()) {
    for (auto *comp : part.components)
      comp->set_engine(this);
  }
}

ExitStatus SimulationEngine::run() {
  assert(built_ && "run() called before build()");
  const uint32_t num_threads = config_.num_threads;

  startup_components();
  running_ = true;
  pacer_.anchor(0);

  if (config_.max_ticks > 0 && num_threads == 1) {
    max_ticks_event_.set_handler([this](Tick ts, Message *) {
      set_exit(ExitReason::COMPLETED, ts, "max ticks reached");
      done_.store(true, std::memory_order_release);
    });
    contexts_[0]->event_queue.push(
        EventQueueEntry{config_.max_ticks, 0, &max_ticks_event_, nullptr});
  }

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

  running_ = false;
  return exit_status_;
}

bool SimulationEngine::step() {
  assert(built_ && "step() called before build()");
  assert(config_.num_threads == 1 && "step() requires single-threaded mode");

  if (!running_) {
    startup_components();
    running_ = true;

    if (config_.max_ticks > 0) {
      max_ticks_event_.set_handler([this](Tick ts, Message *) {
        set_exit(ExitReason::COMPLETED, ts, "max ticks reached");
        done_.store(true, std::memory_order_release);
      });
      contexts_[0]->event_queue.push(
          EventQueueEntry{config_.max_ticks, 0, &max_ticks_event_, nullptr});
    }
  }

  if (done_.load(std::memory_order_acquire))
    return false;

  drain_async_events();

  PartitionContext &ctx = *contexts_[0];

  if (ctx.event_queue.empty()) {
    if (check_termination(TICK_MAX)) {
      running_ = false;
      return false;
    }
    return true;
  }

  Tick step_tick = ctx.event_queue.next_event_time();
  while (!ctx.event_queue.empty() && ctx.event_queue.next_event_time() == step_tick) {
    auto entry = ctx.event_queue.pop();
    process_event(ctx, entry);
    if (done_.load(std::memory_order_acquire)) {
      current_time_.store(step_tick, std::memory_order_release);
      running_ = false;
      return false;
    }
  }

  current_time_.store(step_tick, std::memory_order_release);
  return true;
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
        // After arrive_and_wait, if done_ was set by the completion function,
        // ALL threads see it simultaneously and ALL exit.
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

  if (entry.event->has_handler())
    entry.event->execute(entry.timestamp, entry.message.get());
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
  if (prev == 1 && config_.num_threads == 1 && !contexts_.empty()) {
    auto &ctx = *contexts_[0];
    ctx.idle_wakeup_.store(true, std::memory_order_release);
    ctx.idle_wakeup_.notify_one();
  }
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

  // In single-threaded mode, wake the idle wait.
  if (config_.num_threads == 1 && !contexts_.empty()) {
    auto &ctx = *contexts_[0];
    ctx.idle_wakeup_.store(true, std::memory_order_release);
    ctx.idle_wakeup_.notify_one();
  }
  // In multi-threaded mode, done_ is checked after each barrier epoch.
  // No explicit wake needed since threads will see it at the next barrier.
}

void SimulationEngine::schedule_event_async(Event *event, Tick timestamp,
                                            std::unique_ptr<Message> message) {
  Component *target = event->target();
  assert(target != nullptr && "schedule_event_async: event has no target component");
  PartitionID pid = target->partition_id();
  assert(pid < async_queues_.size() && "schedule_event_async: target partition ID out of range");
  auto &aq = *async_queues_[pid];
  {
    std::lock_guard<std::mutex> lock(aq.mutex);
    aq.events.push_back(EventQueueEntry{timestamp, 0, event, std::move(message)});
  }

  // Wake the target partition so idle workers pick up the new event.
  if (config_.num_threads == 1 && pid < contexts_.size()) {
    PartitionContext &pctx = *contexts_[pid];
    pctx.idle_wakeup_.store(true, std::memory_order_release);
    pctx.idle_wakeup_.notify_one();
  }
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
    std::lock_guard<std::mutex> lock(aq.mutex);
    for (auto &e : aq.events)
      contexts_[i]->event_queue.push(std::move(e));
    aq.events.clear();
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

void SimulationEngine::shutdown_components() {
  for (auto &part : topology_.partitions()) {
    for (auto *comp : part.components)
      comp->shutdown();
  }
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
  std::lock_guard<std::mutex> lock(aq.mutex);
  if (aq.events.empty())
    return;
  Tick floor = compute_async_floor(ctx);
  for (auto &e : aq.events) {
    if (e.timestamp < floor)
      e.timestamp = floor;
    ctx.event_queue.push(std::move(e));
  }
  aq.events.clear();
}

} // namespace simdojo
