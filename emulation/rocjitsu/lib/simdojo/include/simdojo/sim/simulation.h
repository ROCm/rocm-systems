// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file simulation.h
/// @brief SimulationEngine and PartitionContext for single- and multi-threaded PDES execution.

#ifndef SIMDOJO_SIM_SIMULATION_H_
#define SIMDOJO_SIM_SIMULATION_H_

#include "simdojo/sim/event_queue.h"
#include "simdojo/sim/pacing_controller.h"
#include "simdojo/sim/topology.h"

#include <atomic>
#include <barrier>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace simdojo {

/// @brief Per-partition simulation context.
///
/// @details Each worker thread manages exactly one partition and holds the
/// partition's event queue, incoming message queues, and synchronization
/// state. In single-threaded mode only the event queue and async wakeup
/// fields are used. In multi-threaded mode, threads synchronize via a
/// global barrier; each partition publishes its local_next before the
/// barrier, and the barrier completion function computes the new global LBTS.
class alignas(64) PartitionContext {
public:
  explicit PartitionContext(PartitionID pid, uint32_t num_partitions) : partition_id(pid) {
    incoming.reserve(num_partitions);
    for (uint32_t i = 0; i < num_partitions; ++i)
      incoming.push_back(std::make_unique<CrossPartitionQueue>());
  }

  // Non-copyable, non-movable (atomics are not movable).
  PartitionContext(const PartitionContext &) = delete;
  PartitionContext &operator=(const PartitionContext &) = delete;
  PartitionContext(PartitionContext &&) = delete;
  PartitionContext &operator=(PartitionContext &&) = delete;

  /// @brief Return this partition's ID.
  PartitionID id() const { return partition_id; }

  /// @brief Return the partition's current processing tick.
  /// @returns The tick of the last event processed by this partition.
  Tick current_tick() const { return event_queue.current_tick(); }

  /// @brief Return the number of event handlers this partition has run.
  ///
  /// @details Counts handlers that ran to completion: an entry with no handler
  /// is not counted, and neither is one whose handler threw. This measures work
  /// done, which is what a model asserting that an idle component costs nothing
  /// needs. The engine's own max-ticks sentinel has a handler and so is counted
  /// -- an engine configured with max_ticks reports one event it did not model.
  /// Readable from any thread, so a test or a progress readout can sample it
  /// while the partition's own worker is still processing.
  /// @returns Cumulative count since create().
  uint64_t events_processed() const { return events_processed_.load(std::memory_order_relaxed); }

private:
  friend class SimulationEngine;

  PartitionID partition_id; ///< This partition's ID.
  EventQueue event_queue;   ///< Thread-local event priority queue.

  /// @brief Events this partition has executed.
  /// @details Written only by the owning worker thread; atomic because
  /// foreign threads read it. Relaxed on both sides: it carries no other
  /// state, and a reader that sees a slightly stale count is in the same
  /// position as one that read a moment earlier.
  std::atomic<uint64_t> events_processed_{0};

  /// @brief Min timestamp of outgoing cross-partition events in this epoch.
  /// @details Only accessed by the owning worker thread during event processing
  /// and at the end of each epoch.
  Tick local_min_outgoing = TICK_MAX;

  /// @brief Incoming cross-partition events. One CrossPartitionQueue per source partition.
  std::vector<std::unique_ptr<CrossPartitionQueue>> incoming;

  /// @brief Published before barrier, this is the earliest tick this partition could produce next.
  /// @details Set to min(event_queue.next_event_time(), local_min_outgoing) before
  /// arriving at the barrier. The barrier completion function reads all partitions'
  /// local_next values to compute the new global LBTS.
  std::atomic<Tick> local_next{TICK_MAX};

  /// @brief Single-threaded idle wait (atomic wait/notify for async wakeup).
  std::atomic<bool> idle_wakeup_{false};

  /// @brief Drain all incoming queues into the local event queue.
  void drain_incoming();
};

/// @brief The main simulation engine.
///
/// @details Owns the simulation topology and drives the full lifecycle: build,
/// run/step, and shutdown.
///
/// **Single-threaded mode** (`num_threads == 1`): drains events directly from
/// the priority queue in timestamp order. No LBTS computation, no barriers,
/// no epoch structure. When the queue empties, drains async buffers and either
/// terminates or blocks on a condition variable for external events.
///
/// **Multi-threaded mode** (`num_threads > 1`): runs a conservative
/// barrier-based Parallel Discrete Event Simulation (PDES) loop. Each epoch:
///   1. Drain incoming cross-partition events + async events
///   2. Process events up to global_lbts
///   3. Publish local_next (earliest possible future activity)
///   4. Arrive at barrier (completion function computes new global_lbts)
///
/// Components participate in the simulation by scheduling events during
/// startup(). Untimed (functional) components use the
/// Functional\<Base\> CRTP mixin, which self-schedules a timer callback and
/// re-enqueues after each advance(). If no events are ever scheduled and no
/// primaries are registered, the simulation terminates via quiescence. If
/// primaries are registered, the engine blocks until async events arrive or
/// all primaries signal completion.
///
/// Typical usage:
/// @code
///   SimulationEngine::Config config{.num_threads = 4};
///   SimulationEngine engine(config);
///   engine.topology().set_root(std::move(my_model));
///   engine.topology().partition_balanced(config.num_threads);
///   engine.create(); // validates the partition policy and initializes components
///   // (user enqueues work via CP)
///   auto exit = engine.run(); // starts up components, runs to completion
/// @endcode
class SimulationEngine {
public:
  /// @brief Configuration parameters for the simulation engine.
  struct Config {
    Tick max_ticks = 0;       ///< Simulation stops at this tick (0 = unlimited).
    uint32_t num_threads = 1; ///< Number of worker threads (one per partition).
    double wall_clock_ratio =
        0.0;                      ///< Sim-time per wall-time ratio (0 = disabled, 1.0 = real-time).
    bool await_primaries = false; ///< If true, don't terminate on quiescence until at least one
                                  ///< primary has registered. Used by the KFD driver to keep the
                                  ///< engine alive while waiting for external stimuli (doorbells).
  };

  /// @brief Construct with config. Caller populates topology(), then calls create().
  /// @param config Engine configuration parameters.
  explicit SimulationEngine(Config config);

  ~SimulationEngine();

  SimulationEngine(const SimulationEngine &) = delete;
  SimulationEngine &operator=(const SimulationEngine &) = delete;
  SimulationEngine(SimulationEngine &&) = delete;
  SimulationEngine &operator=(SimulationEngine &&) = delete;

  /// @brief Initialize the topology and prepare for execution.
  ///
  /// @details Required after the Config-only constructor once the topology is
  /// populated. Multi-threaded engines require the caller to choose an explicit
  /// topology partition policy before calling create(); single-threaded engines
  /// create their sole partition automatically. Also used to rebuild after
  /// shutdown(). Calls initialize_components() so that components can set up
  /// ports and handlers before run() or step() starts them. Event
  /// self-scheduling is deferred: the Clocked/Functional mixins enqueue their
  /// first event in startup() (invoked by the first run()/step() call), not
  /// here, so no events exist until execution begins.
  /// @throws std::invalid_argument if the worker count, partition count, or
  /// cross-partition link configuration is invalid.
  void create();

  /// @brief Tear down engine state (shutdown components, join workers).
  ///
  /// @details After shutdown(), the engine can be rebuilt with a new create() call.
  /// Called automatically by the destructor if still built.
  void shutdown();

  /// @brief Return whether the engine has been created and is ready for execution.
  /// @retval true Engine is created and ready for run() or step().
  /// @retval false Engine has not been created or has been shut down.
  bool is_created() const { return created_; }

  /// @brief Access the topology for model setup (add components, links, clock domains).
  ///
  /// @details Mutable access is allowed during setup and initialize(). After the
  /// simulation starts running, the topology is frozen and only const access is
  /// permitted (enforced by assertion in debug builds).
  Topology &topology() {
    assert(!executing_.load(std::memory_order_relaxed) &&
           "topology is read-only while the simulation is running");
    return topology_;
  }
  const Topology &topology() const { return topology_; }

  /// @brief Run the simulation to completion.
  ///
  /// @details Starts all components (initialization happens in create()),
  /// then enters the PDES epoch loop. Components are shut down on return.
  /// @returns An ExitStatus describing why the simulation stopped.
  ExitStatus run();

  /// @brief Run queued events until @p done becomes true, or the queue drains.
  ///
  /// @details The resumable middle ground between step(), which advances only
  /// to the next event time, and run(), which owns the whole execution
  /// lifecycle and returns only once the simulation has ended. Neither of
  /// those is "drain what is queued, stop on a predicate, stay resumable",
  /// which is what a model asked for one answer -- the tick a particular
  /// request completes -- needs.
  ///
  /// Quiescence here means the partition's event queue is empty and no async
  /// event is waiting to be merged into it. It is deliberately not run()'s
  /// termination contract: no primary-component check, no quiescence exit, and
  /// no wall-clock pacing or idle wait. A caller that wants those wants run().
  /// Two consequences the caller owns, because nothing here will latch them:
  /// an engine driven only this way never reaches check_termination(), so
  /// "all primaries completed" is never recorded and is_done() stays false
  /// however finished the model is -- a drive loop waiting on an external
  /// stimulus must sleep on its own rather than spin on this; and last_exit()
  /// stays default-constructed until max_ticks or a request_exit() sets it.
  ///
  /// A configured max_ticks is honoured, but note what it costs here: the
  /// sentinel is an ordinary queued entry at max_ticks, so the queue is not
  /// empty until then and "drain what is queued" becomes "run to the horizon".
  /// A bounded run on a max_ticks engine returns max_ticks, not the tick its
  /// own work finished at, and ends the shared simulation.
  ///
  /// Every other event in flight advances with it, which is the point: this is
  /// a bounded run of a shared simulation, not a private one per query.
  ///
  /// Resume with run_bounded() or step(), not with run(): run() has its own
  /// lifecycle and would not resume this one.
  ///
  /// **A caller that loops until its predicate flips must also check
  /// is_done()**, or it spins forever once the engine has exited -- on
  /// max_ticks, on any request_exit(), or after shutdown().
  ///
  /// Single-threaded engines only. Nesting a *different* engine inside a
  /// handler is the supported arrangement and is tested; re-entering this one,
  /// through any entry point, throws.
  /// @param done Predicate polled between events; the caller flips it. Held by
  ///        reference and read repeatedly, so it must outlive the call and be
  ///        something the handlers can actually change. Read without
  ///        synchronization, so the writer must be a handler on this engine's
  ///        own thread -- a foreign thread storing to it is a data race. Must
  ///        not be a bit-field: it cannot bind to the deleted rvalue overload
  ///        below, so it silently binds to a frozen copy instead.
  /// @returns The tick the engine reached.
  /// @throws std::logic_error if this engine is already executing.
  /// @throws std::invalid_argument if the engine has more than one partition.
  Tick run_bounded(const bool &done);

  /// @brief Refuse a temporary predicate.
  ///
  /// @details `run_bounded(flag.load())` would bind a frozen copy and turn a
  /// bounded run into an unbounded one, silently.
  Tick run_bounded(bool &&done) = delete;

  /// @brief Run every queued event, leaving the engine resumable.
  ///
  /// @details Stops early if the engine exits; see run_bounded().
  /// @returns The tick the engine reached, which is the previous one if there
  ///          was nothing to run.
  /// @throws Whatever run_bounded() throws.
  Tick run_until_idle();

  /// @brief Whether the engine has exited and will process no further events.
  ///
  /// @details Latched by max_ticks, by request_exit(), and by shutdown(). A
  /// drive loop around run_bounded() has to test it, because a finished engine
  /// is otherwise indistinguishable from one with nothing to do.
  /// @retval true The simulation has ended; see last_exit() for why.
  /// @retval false The engine will still process events.
  bool is_done() const { return done_.load(std::memory_order_acquire); }

  /// @brief Number of event handlers run since create(), across all partitions.
  ///
  /// @details Zero after shutdown(), because the counters live in the
  /// partition contexts and those are the generation. Safe to call from any
  /// thread: it takes the same lock shutdown() clears the contexts under.
  /// @returns Cumulative count of entries whose handler ran.
  uint64_t events_processed() const;

  /// @brief Block until run() (or step()'s first call) has completed component
  /// startup.
  ///
  /// @details A readiness boundary for embeddings that launch run() on a
  /// background thread and must not publish the simulated device until every
  /// component's startup() callback has completed — otherwise a caller can reach
  /// a half-started device, and short-lived processes can race component startup
  /// against C++ finalizers. Once observed, readiness stays latched until the next
  /// create().
  /// @returns true if startup completed normally; false if a component's startup()
  /// threw (run() caught it, latched readiness so this call wakes, and returned an
  /// error ExitStatus). A false return means the engine is NOT running and the
  /// embedding must unwind rather than publish the device.
  [[nodiscard]] bool wait_until_started() const;

  /// @brief Latch readiness only if nothing has latched it yet.
  /// @details Liveness backstop for an embedding that runs the engine on its own
  /// thread: wait_until_started() blocks until run()/step() latches, so ANY path
  /// that returns without reaching the latch (a caller-side validation failure
  /// before run() is entered, an engine that is never run at all) would strand the
  /// waiter forever. The owning thread calls this after its run() returns so
  /// readiness becomes a property of the THREAD's completion rather than of run()
  /// being reached. Must be called from that same thread, after run()/step()
  /// returns, so it cannot race the in-band latch.
  /// A failed latch also records the terminal ExitStatus. The flag it sets is what
  /// run()/step() use as their terminal guard, so latching without a status would
  /// leave the create() default in place and report a generation that never started
  /// as a clean run.
  void latch_startup_if_unlatched(bool failed);

  /// @brief Process all events at the next event time (single-threaded only).
  ///
  /// @details On the first call, starts all components (initialization
  /// happens in create()). Each subsequent call advances to the next event
  /// time and processes all events at that timestamp, then returns.
  /// Ticks without scheduled events are skipped. If the queue is empty
  /// but primaries are registered, returns true so the caller can poll
  /// for async events.
  /// @retval true Simulation can continue.
  /// @retval false Simulation is done (query last_exit() for details).
  bool step();

  /// @brief Request the simulation to stop.
  ///
  /// @details Thread-safe. Can be called from any thread (worker, external, signal
  /// handler via flag).
  /// @param reason Human-readable reason for stopping.
  /// @param code Exit code (0 = success).
  void request_exit(std::string reason, int code = 0);

  /// @brief Register the calling component as a primary (work-producing).
  ///
  /// @details Primary components participate in the end-of-simulation consensus
  /// protocol. The simulation ends gracefully when all registered primaries
  /// have called primary_release(). Must be called during initialize()
  /// or startup(), before the main simulation loop starts.
  void register_as_primary();

  /// @brief Release this primary (signal it is done producing work).
  ///
  /// @details Thread-safe. Decrements the active primary count. When all
  /// registered primaries have been released, the engine sets exit reason
  /// COMPLETED at the next epoch boundary.
  void primary_release();

  /// @brief Retain this primary (it has new work after a previous release).
  ///
  /// @details Thread-safe. Re-increments the active primary count, preventing
  /// termination until a subsequent primary_release().
  void primary_retain();

  /// @brief Return the exit event from the last completed run() or step().
  /// @returns Const reference to the stored exit event.
  const ExitStatus &last_exit() const { return exit_status_; }

  /// @brief Enqueue an event from any thread (thread-safe).
  ///
  /// @details The event is buffered and drained into the target partition's queue
  /// at the next safe point (epoch boundary or start of step).
  /// @param event Reusable event descriptor.
  /// @param timestamp Simulation tick at which the event fires.
  /// @param message Optional message payload (ownership transferred).
  void schedule_event_async(Event *event, Tick timestamp,
                            std::unique_ptr<Message> message = nullptr);

  /// @brief Enqueue an event from any thread at the current simulation time.
  ///
  /// @details Thread-safe. When wall-clock sync is enabled, translates the
  /// current wall time to a simulation tick. When disabled, uses the engine's
  /// current simulation time. Preferred over schedule_event_async() for
  /// external stimulus (driver doorbells, host submissions).
  /// @param event Reusable event descriptor.
  /// @param message Optional message payload (ownership transferred).
  void schedule_event_now(Event *event, std::unique_ptr<Message> message = nullptr);

  /// @brief Deposit an event into another partition's cross-partition inbox.
  /// @param src_partition Source partition ID (selects the incoming queue).
  /// @param dst_partition Destination partition ID.
  /// @param event Reusable event descriptor.
  /// @param timestamp Simulation tick at which the event fires.
  /// @param message Optional message payload (ownership transferred).
  void send_cross_partition(PartitionID src_partition, PartitionID dst_partition, Event *event,
                            Tick timestamp, std::unique_ptr<Message> message = nullptr);

  /// @brief Return the number of partition contexts.
  /// @returns Number of partitions.
  uint32_t num_contexts() const { return static_cast<uint32_t>(contexts_.size()); }

  /// @brief Access a partition context by partition ID (const).
  /// @param pid The partition to look up.
  /// @returns Const reference to the PartitionContext.
  const PartitionContext &context(PartitionID pid) const {
    assert(pid < contexts_.size() && "partition ID out of range");
    return *contexts_[pid];
  }

  /// @brief Access a partition context by partition ID.
  /// @param pid The partition to look up.
  /// @returns Mutable reference to the PartitionContext.
  PartitionContext &context(PartitionID pid) {
    assert(pid < contexts_.size() && "partition ID out of range");
    return *contexts_[pid];
  }

  /// @brief Return the current simulation time.
  /// @returns The latest processed simulation tick.
  Tick global_time() const { return current_time_.load(std::memory_order_acquire); }

  /// @brief Access the pacing controller.
  /// @returns Const reference to the PacingController.
  const PacingController &pacer() const { return pacer_; }

  /// @brief Enqueue an event into the target component's partition queue.
  ///
  /// @details Must only be called from the thread that owns the target partition.
  /// For cross-partition delivery, use send_cross_partition() instead.
  /// @param event Reusable event descriptor.
  /// @param timestamp Simulation tick at which the event fires.
  /// @param message Optional message payload (ownership transferred).
  void schedule_event(Event *event, Tick timestamp, std::unique_ptr<Message> message = nullptr);

private:
  /// @brief Worker loop executed by each partition thread.
  void worker_loop(PartitionID partition_id);

  /// @brief Process a single heap entry: execute its event handler if present.
  /// @param ctx The partition context that owns the event queue.
  /// @param entry The heap entry to process.
  void process_event(PartitionContext &ctx, EventQueueEntry &entry);

  /// @brief Barrier completion function, runs in the last-arriving thread.
  ///
  /// @details Computes the new global LBTS from all partitions' local_next values,
  /// updates current_time_, runs service callbacks, and checks termination.
  void barrier_completion();

  /// @brief Call initialize() on all components across all partitions.
  void initialize_components();

  /// @brief Call startup() on all components across all partitions.
  void startup_components();

  /// @brief Call shutdown() on every initialized component, in reverse order.
  /// @details Every callback runs even if an earlier one throws: the generation is
  /// marked shut down up front, so skipping the rest would strand them with no way
  /// to retry. Never throws — the first failure is RETURNED so the caller can finish
  /// its own cleanup before reporting it.
  /// @returns The first callback exception, or null if all succeeded.
  [[nodiscard]] std::exception_ptr shutdown_components() noexcept;

  /// @brief Log a captured component-cleanup failure. Never throws.
  static void report_component_failure(std::exception_ptr failure, const char *phase) noexcept;

  /// @brief Latch startup readiness and wake wait_until_started() observers.
  /// @details Stores @p failed BEFORE startup_complete_, then notifies — the order
  /// matters so a waiter that observes complete also observes the correct failed
  /// state. Centralizes the epilogue used by run()/step() on both the success and
  /// the throw paths so the ordering cannot drift. A startup throw is terminal for
  /// the create() generation (see startup_failed()), so this latches failure once and
  /// the caller must shutdown() + create() before a fresh startup can latch success.
  void latch_startup(bool failed) {
    startup_failed_.store(failed, std::memory_order_release);
    startup_complete_.store(true, std::memory_order_release);
    startup_complete_.notify_all();
  }

  /// @brief Common epilogue for a startup_components() throw in run()/step().
  /// @details Records the terminal failure -- startup_failed_, a failure ExitStatus,
  /// and done_ so a racing request_exit() cannot overwrite it -- unwinds the started
  /// components, and only then PUBLISHES readiness via latch_startup(). Publishing
  /// last is what makes a false return from wait_until_started() mean the generation
  /// is torn down rather than still running shutdown hooks; it is safe to leave the
  /// latch until the end because the unwind cannot escape, shutdown_components() and
  /// report_component_failure() both being noexcept and catching each callback
  /// individually. Both run() and step() share this so the recorded status and the
  /// ordering cannot drift apart.
  void fail_startup(std::string message);

  /// @brief Drain all async event buffers into their partition queues (single-threaded).
  void drain_async_events();

  /// @brief Drain async events for a single partition, flooring timestamps.
  ///
  /// @details In multi-threaded mode, floors each async event timestamp at
  /// the global LBTS to ensure that events are not placed before the
  /// already-committed safe time. In single-threaded mode, no flooring is needed.
  void drain_async_for_partition(PartitionContext &ctx);

  /// @brief Compute the safe minimum timestamp for async events in a partition.
  ///
  /// @details Returns global_lbts_ clamped to at least event_queue.current_tick().
  /// Returns 0 in single-threaded mode.
  Tick compute_async_floor(const PartitionContext &ctx) const;

  /// @brief Set the exit event (internal helper).
  void set_exit(ExitReason reason, Tick tick, std::string message, int code = 0);

  /// @brief Record a terminal FAILURE status and mark the generation done, atomically.
  /// @details For terminal failures only, and unlike set_exit() it is last-writer-wins:
  /// a startup failure outranks an ordinary exit request, including one already
  /// recorded by a request_exit() issued from inside a startup callback that then
  /// threw. Keeping first-writer-wins there would let run() report success for a
  /// generation whose components never started. done_ is stored in the same critical
  /// section because request_exit() gates its overwrite on reading it under this lock.
  void set_terminal_exit(ExitReason reason, Tick tick, std::string message, int code);

  /// @brief Check if all registered primaries have signaled OK to end.
  /// @retval true All primaries are done (and at least one is registered).
  /// @retval false No primaries registered, or some are still active.
  bool all_primaries_done() const;

  /// @brief Check if all partitions are fully quiescent.
  ///
  /// @details Returns true only when all incoming queues are empty and no
  /// partition has processable local events. Called from the barrier
  /// completion function where all threads are synchronized.
  bool is_fully_quiescent() const;

  /// @brief Check end-of-sim conditions (primaries, max_ticks, quiescence).
  /// Sets exit event and done_ flag if simulation should stop.
  /// @param lbts The current or newly computed LBTS value.
  /// @retval true Simulation should stop.
  /// @retval false Simulation continues.
  bool check_termination(Tick lbts);

  /// @brief Set up partition contexts, async queues, and engine pointers.
  void setup_partitions();

  /// @brief Start components and arm the max-ticks sentinel, once per create().
  ///
  /// @details Shared by run(), step(), and run_bounded(), so that whichever
  /// reaches the engine first starts it and the others find it started. It is
  /// keyed on startup_complete_, not on "executing right now": run() clears the
  /// latter on return, and a shared startup keyed on it would start every
  /// component a second time for `run_until_idle(); run();`, double-scheduling
  /// their first events and double-registering primaries. A second bool beside
  /// startup_complete_ could only drift from it, and would miss the thread-death
  /// backstop (latch_startup_if_unlatched).
  /// @param propagate_failure Whether to rethrow a component's startup()
  ///        exception. True for the foreground entry points; false for run(),
  ///        which may be on a background thread whose lambda has no catch and
  ///        which converts the failure into an ExitStatus instead.
  /// @retval true The engine is started and may process events.
  /// @retval false A startup() threw; this create() generation is terminal and
  ///         the caller must shutdown() + create() to retry.
  bool ensure_started(bool propagate_failure);

  /// @brief Wake an idle single-threaded worker so it re-checks its exit and event state.
  ///
  /// @details No-op outside single-threaded mode, or once the partition is gone.
  /// The caller must already hold @ref contexts_mutex_ (shared or exclusive).
  /// @param pid The partition to wake.
  void wake_partition_locked(PartitionID pid);

  /// @brief Wake an idle single-threaded worker, guarding against concurrent teardown.
  ///
  /// @details Takes a shared lock on @ref contexts_mutex_ so the context cannot be
  /// destroyed by shutdown() between the bounds check and the notify.
  /// @param pid The partition to wake.
  void wake_partition(PartitionID pid);

  Topology topology_; ///< The simulation topology.
  Config config_;     ///< Engine configuration.
  /// @brief Reusable event for the max-ticks exit sentinel.
  /// @details Scheduled at config_.max_ticks during init. When processed, triggers sim exit.
  Event max_ticks_event_{nullptr, EventType::SIM_EXIT};
  /// @brief Guards destruction of contexts_ and async_queues_ against foreign threads.
  ///
  /// @details The wake and async-scheduling paths run on threads the engine does not
  /// own (the host thread calling request_exit(), the doorbell monitor releasing a
  /// primary) and can overlap shutdown() clearing these vectors. Those readers take a
  /// shared lock; shutdown() takes it exclusively around the clear.
  mutable std::shared_mutex contexts_mutex_;
  std::vector<std::unique_ptr<PartitionContext>>
      contexts_;                      ///< Per-partition state (one per thread).
  std::vector<std::jthread> workers_; ///< Worker threads (multi-threaded mode).
  std::atomic<Tick> current_time_{0}; ///< Current simulation time (latest processed tick).
  PacingController pacer_;            ///< PI-controlled wall-clock pacing.
  std::atomic<bool> done_{false};     ///< Signals simulation completion.
  /// @brief Active primary component count.
  /// @details Incremented on registration, decremented when a primary releases.
  /// Simulation terminates when this reaches 0 (after at least one registration)
  /// and the system is fully quiescent.
  std::atomic<uint32_t> active_primaries_{0};
  std::atomic<bool> has_primaries_{false}; ///< Set on first register_as_primary().
  ExitStatus exit_status_;                 ///< Exit information from the last run/step.
  bool created_ = false; ///< Whether create() has completed (components initialized).
  /// @brief Set while any entry point is driving this engine's queue.
  ///
  /// @details A runtime guard rather than an assert, unlike this class's other
  /// preconditions. Re-entering -- run() or step() or run_bounded() from
  /// inside a handler -- has the inner loop popping from the very queue the
  /// outer process_event() frame is iterating and rewriting the partition tick
  /// underneath it. That corrupts a run rather than merely misbehaving, and a
  /// release build would do it in silence. Atomic because the entry points can
  /// be called from different threads: a host polling step() while run() drives
  /// the engine on a background thread is exactly the collision this refuses.
  /// It is also the flag topology() asserts on, so "an execution loop is in
  /// progress" has one definition that every entry point restores on unwind.
  std::atomic<bool> executing_{false};
  /// @brief Latched true once startup() has run for every component (or thrown);
  /// reset by create(). Read by wait_until_started() from an embedding thread.
  std::atomic<bool> startup_complete_{false};
  /// @brief Set true if a component's startup() threw; reset by create(). Lets
  /// wait_until_started() report failure instead of blocking forever, since a
  /// throwing startup latches startup_complete_ without the engine running.
  std::atomic<bool> startup_failed_{false};

  /// @brief True once a startup() throw made this create() generation terminal.
  /// @details Same flag wait_until_started() publishes, reused as run()/step()'s
  /// terminal guard rather than a second bool: both are set only by latch_startup()
  /// and cleared only by create(), so a duplicate could only ever drift. Reusing it
  /// also makes the thread-death backstop (latch_startup_if_unlatched) terminal for
  /// the generation, which a separate same-thread flag would have missed.
  ///
  /// Terminal rather than retryable because a partial attempt leaves the partition,
  /// async and cross-partition event queues, the primary-registration counters and
  /// per-component state intact: re-running startup on top of them would
  /// double-schedule events and double-register primaries. A caller that wants
  /// another attempt has to shutdown() and create() a fresh generation.
  bool startup_failed() const { return startup_failed_.load(std::memory_order_acquire); }

  /// @brief True once shutdown_components() has run for this generation, so each
  /// initialized component's shutdown() fires exactly once even if both the
  /// startup-failure unwind and the engine shutdown() path reach it. Reset by
  /// create().
  bool components_shut_down_ = false;

  /// @brief Global lower bound on time stamp, updated by barrier completion.
  std::atomic<Tick> global_lbts_{0};

  /// @brief Barrier for multi-threaded epoch synchronization.
  /// @details The completion function computes the new global LBTS from all
  /// partitions' local_next values.
  std::unique_ptr<std::barrier<std::function<void()>>> barrier_;

  /// @brief Per-partition async event buffer for cross-thread insertion.
  struct AsyncQueue {
    std::mutex mutex;
    std::vector<EventQueueEntry> events;
    std::atomic<bool> pending{false}; ///< Set by producer, checked before locking.
  };
  std::vector<std::unique_ptr<AsyncQueue>> async_queues_; ///< One per partition.

  std::mutex exit_mutex_; ///< Protects exit_status_ and exit_set_ (first-writer-wins).
  bool exit_set_ = false; ///< True once exit_status_ has been written (sentinel for set_exit).
};

} // namespace simdojo

#endif // SIMDOJO_SIM_SIMULATION_H_
