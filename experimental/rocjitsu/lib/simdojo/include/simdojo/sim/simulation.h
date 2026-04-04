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
#include <functional>
#include <memory>
#include <mutex>
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

private:
  friend class SimulationEngine;

  PartitionID partition_id; ///< This partition's ID.
  EventQueue event_queue;   ///< Thread-local event priority queue.

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
/// initialize() or startup(). Untimed (functional) components use the
/// Functional\<Base\> CRTP mixin, which self-schedules a timer callback and
/// re-enqueues after each step(). If no events are ever scheduled and no
/// primaries are registered, the simulation terminates via quiescence. If
/// primaries are registered, the engine blocks until async events arrive or
/// all primaries signal completion.
///
/// Typical usage:
/// @code
///   SimulationEngine engine(config);
///   engine.topology().set_root(std::move(my_model));
///   engine.build(); // partitions, initializes components
///   // (user enqueues work via CP)
///   auto exit = engine.run(); // starts up components, runs to completion
///   // OR: while (engine.step()) {} // starts up on first call, one tick per call
/// @endcode
class SimulationEngine {
public:
  /// @brief Configuration parameters for the simulation engine.
  struct Config {
    Tick max_ticks = 0;       ///< Simulation stops at this tick (0 = unlimited).
    uint32_t num_threads = 1; ///< Number of worker threads (one per partition).
    double wall_clock_ratio =
        0.0; ///< Sim-time per wall-time ratio (0 = disabled, 1.0 = real-time).
  };

  /// @brief Construct with config. Caller populates topology(), then calls build().
  /// @param config Engine configuration parameters.
  explicit SimulationEngine(Config config);

  ~SimulationEngine();

  SimulationEngine(const SimulationEngine &) = delete;
  SimulationEngine &operator=(const SimulationEngine &) = delete;
  SimulationEngine(SimulationEngine &&) = delete;
  SimulationEngine &operator=(SimulationEngine &&) = delete;

  /// @brief Partition the topology, initialize components, and prepare for execution.
  ///
  /// @details Required after the Config-only constructor once the topology is
  /// populated. Also used to rebuild after shutdown(). Calls
  /// initialize_components() so that components can set up ports and handlers
  /// before run() or step() starts them.
  void build();

  /// @brief Tear down engine state (shutdown components, join workers).
  ///
  /// @details After shutdown(), the engine can be rebuilt with a new build() call.
  /// Called automatically by the destructor if still built.
  void shutdown();

  /// @brief Return whether the engine has been built and is ready for execution.
  /// @retval true Engine is built and ready for run() or step().
  /// @retval false Engine has not been built or has been shut down.
  bool is_built() const { return built_; }

  /// @brief Access the topology for model setup (add components, links, clock domains).
  ///
  /// @details Mutable access is allowed during setup and initialize(). After the
  /// simulation starts running, the topology is frozen and only const access is
  /// permitted (enforced by assertion in debug builds).
  Topology &topology() {
    assert(!running_ && "topology is read-only while the simulation is running");
    return topology_;
  }
  const Topology &topology() const { return topology_; }

  /// @brief Run the simulation to completion.
  ///
  /// @details Starts all components (initialization happens in build()),
  /// then enters the PDES epoch loop. Components are shut down on return.
  /// @returns An ExitStatus describing why the simulation stopped.
  ExitStatus run();

  /// @brief Advance the simulation by one tick (single-threaded only).
  ///
  /// @details On the first call, starts all components (initialization
  /// happens in build()). Each subsequent call processes all events at the
  /// next timestamp, then returns. If the queue is empty but primaries are
  /// registered, returns true so the caller can poll for async events.
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

  /// @brief Call shutdown() on all components across all partitions.
  void shutdown_components();

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

  Topology topology_; ///< The simulation topology.
  Config config_;     ///< Engine configuration.
  /// @brief Reusable event for the max-ticks exit sentinel.
  /// @details Scheduled at config_.max_ticks during init. When processed, triggers sim exit.
  Event max_ticks_event_{nullptr, EventType::SIM_EXIT};
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
  bool built_ = false;   ///< Whether build() has completed (components initialized).
  bool running_ = false; ///< True while running; also guards step() first-call startup.

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
  };
  std::vector<std::unique_ptr<AsyncQueue>> async_queues_; ///< One per partition.

  std::mutex exit_mutex_; ///< Protects exit_status_ and exit_set_ (first-writer-wins).
  bool exit_set_ = false; ///< True once exit_status_ has been written (sentinel for set_exit).
};

} // namespace simdojo

#endif // SIMDOJO_SIM_SIMULATION_H_
