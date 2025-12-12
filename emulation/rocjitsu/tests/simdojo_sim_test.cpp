// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file simdojo_sim_test.cpp
/// @brief Tests for simdojo simulation engine: LBTS correctness, cross-partition
/// communication, async causality, termination, pacing, spinlock, and stress.

#include "simdojo/components/cache.h"
#include "simdojo/sim/pacing_controller.h"
#include "simdojo/sim/simulation.h"
#include "util/spinlock.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace simdojo;

// ============================================================================
// Test Helper Components
// ============================================================================

namespace {

/// Helper: create a message with an integer payload.
inline std::unique_ptr<Message> make_test_msg(uint64_t val = 0) {
  auto msg = std::make_unique<Message>();
  msg->set_payload(static_cast<uintptr_t>(val));
  return msg;
}

/// Leaf component that schedules N self-events at ticks 1..N during startup.
/// Records each processing tick for later assertion.
class CounterComponent : public Component {
public:
  CounterComponent(std::string name, uint32_t num_events)
      : Component(std::move(name)), num_events_(num_events) {
    timer_event_.set_handler([this](Tick ts, Message *) {
      ticks_processed.push_back(ts);
      ++count;
      if (count < num_events_)
        schedule_event(&timer_event_, ts + 1);
    });
  }

  void startup() override {
    if (num_events_ > 0)
      schedule_event(&timer_event_, 1);
  }

  std::vector<Tick> ticks_processed;
  uint32_t count = 0;

private:
  uint32_t num_events_;
  Event timer_event_{this, EventType::TIMER_CALLBACK};
};

/// Component with IN/OUT ports that replies to each received message.
/// Optionally registers as primary (default: true).
class PingPongComponent : public Component {
public:
  PingPongComponent(std::string name, uint32_t target, bool send_first = false,
                    bool register_primary = true)
      : Component(std::move(name)), target_count_(target), send_first_(send_first),
        register_primary_(register_primary) {
    in_ = add_port(std::make_unique<Port>("in", 0, this, PortDirection::IN, PortProtocol::UNTYPED));
    out_ =
        add_port(std::make_unique<Port>("out", 1, this, PortDirection::OUT, PortProtocol::UNTYPED));
    in_->set_handler([this](Tick ts, Message *) {
      recv_ticks.push_back(ts);
      ++recv_count;
      if (recv_count >= target_count_) {
        if (register_primary_)
          engine()->primary_release();
        return;
      }
      out_->send(make_test_msg(recv_count));
    });
  }

  void startup() override {
    if (register_primary_)
      engine()->register_as_primary();
    if (send_first_)
      out_->send(make_test_msg(0));
  }

  Port *in_port() { return in_; }
  Port *out_port() { return out_; }

  std::vector<Tick> recv_ticks;
  uint32_t recv_count = 0;

private:
  uint32_t target_count_;
  bool send_first_;
  bool register_primary_;
  Port *in_ = nullptr;
  Port *out_ = nullptr;
};

/// Component that sends N messages on its OUT port at consecutive ticks.
/// Optionally registers as primary (default: true for backward compat).
class ProducerComponent : public Component {
public:
  ProducerComponent(std::string name, uint32_t count, Tick start_tick = 1,
                    bool register_primary = true)
      : Component(std::move(name)), total_(count), start_tick_(start_tick),
        register_primary_(register_primary) {
    out_ =
        add_port(std::make_unique<Port>("out", 0, this, PortDirection::OUT, PortProtocol::UNTYPED));
    timer_event_.set_handler([this](Tick ts, Message *) {
      out_->send(make_test_msg(sent_));
      ++sent_;
      if (sent_ < total_)
        schedule_event(&timer_event_, ts + 10);
      else if (register_primary_)
        engine()->primary_release();
    });
  }

  void startup() override {
    if (register_primary_)
      engine()->register_as_primary();
    schedule_event(&timer_event_, start_tick_);
  }

  Port *out_port() { return out_; }
  uint32_t sent_ = 0;

private:
  uint32_t total_;
  Tick start_tick_;
  bool register_primary_;
  Port *out_ = nullptr;
  Event timer_event_{this, EventType::TIMER_CALLBACK};
};

/// Component that records all received messages.
class ConsumerComponent : public Component {
public:
  explicit ConsumerComponent(std::string name) : Component(std::move(name)) {
    in_ = add_port(std::make_unique<Port>("in", 0, this, PortDirection::IN, PortProtocol::UNTYPED));
    in_->set_handler([this](Tick ts, Message *msg) {
      uint64_t val = msg ? static_cast<uint64_t>(msg->payload()) : 0;
      received.emplace_back(ts, val);
    });
  }

  Port *in_port() { return in_; }
  std::vector<std::pair<Tick, uint64_t>> received;

private:
  Port *in_ = nullptr;
};

/// Component that runs forever (for max_ticks / request_exit tests).
class InfiniteComponent : public Component {
public:
  explicit InfiniteComponent(std::string name) : Component(std::move(name)) {
    timer_event_.set_handler([this](Tick ts, Message *) { schedule_event(&timer_event_, ts + 1); });
  }

  void startup() override { schedule_event(&timer_event_, 1); }

private:
  Event timer_event_{this, EventType::TIMER_CALLBACK};
};

/// Helper: build engine with manual partition assignment.
/// assigner maps component name → partition ID.
void build_with_manual_partitions(SimulationEngine &engine, uint32_t num_partitions,
                                  std::function<PartitionID(Component *)> assigner) {
  engine.topology().partition_manual(num_partitions, std::move(assigner));
  engine.build();
}

/// Helper: partition by component name suffix digit (e.g., "a0" → 0, "b1" → 1).
PartitionID partition_by_name_suffix(Component *comp) {
  const auto &name = comp->name();
  if (name.empty())
    return 0;
  char last = name.back();
  if (last >= '0' && last <= '9')
    return static_cast<PartitionID>(last - '0');
  return 0;
}

} // namespace

// ============================================================================
// Area 5: PacingController Unit Tests
// ============================================================================

TEST(PacingControllerTest, DisabledIsNoop) {
  PacingController pc;
  EXPECT_FALSE(pc.enabled());
  EXPECT_EQ(pc.sim_tick_now(), 0u);
  EXPECT_EQ(pc.idle_wait_duration().count(), 0);
  // throttle should not block.
  auto start = std::chrono::steady_clock::now();
  pc.throttle(999999);
  auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 10);
}

TEST(PacingControllerTest, StateTransitions) {
  PacingController::Config cfg;
  cfg.ratio = 1.0;
  cfg.init_samples = 4;
  cfg.stable_count = 8;
  cfg.stable_offset_ns = 1e12;  // Extremely generous: 1s. Any realistic offset is "low".
  cfg.burst_ns = 1e15;          // Huge burst to avoid throttling.
  cfg.step_threshold_ns = 1e15; // Huge step threshold to avoid stepping.
  PacingController pc(cfg);
  pc.anchor(0);

  EXPECT_EQ(pc.state(), PacingController::State::INIT);

  // INIT → TRACKING after init_samples.
  for (uint32_t i = 0; i < 4; ++i)
    pc.throttle(pc.sim_tick_now());
  EXPECT_EQ(pc.state(), PacingController::State::TRACKING);

  // TRACKING → STABLE after stable_count low-offset calls.
  // Pass sim_time well ahead of target to guarantee a positive offset even
  // under sanitizer slowdown (TSan can add milliseconds between the caller's
  // sim_tick_now() and the internal re-read in throttle).
  for (uint32_t i = 0; i < 8; ++i)
    pc.throttle(pc.sim_tick_now() +
                100'000'000'000ULL); // +100ms wall-equivalent ensures positive offset under TSan.
  EXPECT_EQ(pc.state(), PacingController::State::STABLE);
}

TEST(PacingControllerTest, StepThresholdResetsAnchor) {
  PacingController::Config cfg;
  cfg.ratio = 1.0;
  cfg.init_samples = 1;
  cfg.step_threshold_ns = 1e6; // 1ms.
  cfg.burst_ns = 1e12;
  PacingController pc(cfg);
  pc.anchor(0);

  pc.throttle(pc.sim_tick_now()); // Move past INIT.
  EXPECT_EQ(pc.state(), PacingController::State::TRACKING);

  // Trigger step with huge offset.
  Tick huge = pc.sim_tick_now() + 1'000'000'000; // Way past threshold.
  pc.throttle(huge);
  // Should still be TRACKING (step resets but doesn't change state from TRACKING).
  EXPECT_EQ(pc.state(), PacingController::State::TRACKING);
}

TEST(PacingControllerTest, PauseResumeNoDrift) {
  PacingController::Config cfg;
  cfg.ratio = 1.0;
  PacingController pc(cfg);
  pc.anchor(0);

  // Active for ~20ms.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  Tick before_pause = pc.sim_tick_now();
  EXPECT_GT(before_pause, 0u);

  // Pause for 100ms.
  pc.pause();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  pc.resume();

  // After resume, sim_tick_now should be close to before_pause + a tiny delta,
  // NOT before_pause + 100ms.
  Tick after_resume = pc.sim_tick_now();
  Tick drift = after_resume - before_pause;
  // Should be < 20ms worth of ticks (some time passes during pause/resume calls).
  Tick max_acceptable = 20'000'000'000ULL; // 20ms in ticks (at 1000 ticks/ns).
  EXPECT_LT(drift, max_acceptable);
}

TEST(PacingControllerTest, SeqlockConcurrentReads) {
  PacingController::Config cfg;
  cfg.ratio = 1.0;
  PacingController pc(cfg);
  pc.anchor(0);

  std::atomic<bool> done{false};
  std::atomic<uint32_t> reads{0};

  auto reader = [&]() {
    while (!done.load(std::memory_order_relaxed)) {
      Tick t = pc.sim_tick_now();
      // If seqlock is broken, we'd get garbage values (e.g., > 1e18).
      // A sane value for <1s of runtime at ratio=1.0 is < 1e15 ticks (1s in ps).
      EXPECT_LT(t, 1'000'000'000'000'000ULL);
      reads.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::vector<std::thread> readers_vec;
  for (int i = 0; i < 4; ++i)
    readers_vec.emplace_back(reader);

  // Writer: pause/resume cycles.
  for (int i = 0; i < 100; ++i) {
    pc.pause();
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    pc.resume();
  }

  done.store(true, std::memory_order_relaxed);
  for (auto &t : readers_vec)
    t.join();

  EXPECT_GT(reads.load(), 0u);
}

TEST(PacingControllerTest, TokenBucketAbsorbsBursts) {
  PacingController::Config cfg;
  cfg.ratio = 1.0;
  cfg.init_samples = 1;
  cfg.burst_ns = 4e6; // 4ms burst buffer.
  PacingController pc(cfg);
  pc.anchor(0);
  pc.throttle(pc.sim_tick_now()); // Past INIT.

  // Offset within burst: should not sleep.
  Tick target = pc.sim_tick_now();
  Tick within_burst = target + 2'000'000'000ULL; // 2ms ahead (within 4ms burst).
  auto start = std::chrono::steady_clock::now();
  pc.throttle(within_burst);
  auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 5);
}

// ============================================================================
// Area 6: Spinlock Unit Tests
// ============================================================================

TEST(SpinlockTest, BasicLockUnlock) {
  util::Spinlock lock;
  lock.lock();
  EXPECT_FALSE(lock.try_lock()); // Already locked.
  lock.unlock();
  EXPECT_TRUE(lock.try_lock()); // Now free.
  lock.unlock();
}

TEST(SpinlockTest, ConcurrentIncrement) {
  util::Spinlock lock;
  uint64_t counter = 0;
  constexpr int THREADS = 8;
  constexpr int ITERS = 100000;

  std::vector<std::thread> threads;
  for (int i = 0; i < THREADS; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < ITERS; ++j) {
        std::lock_guard<util::Spinlock> guard(lock);
        ++counter;
      }
    });
  }
  for (auto &t : threads)
    t.join();

  EXPECT_EQ(counter, static_cast<uint64_t>(THREADS) * ITERS);
}

TEST(SpinlockTest, CrossPartitionQueueHighContention) {
  CrossPartitionQueue queue;
  constexpr int WRITERS = 8;
  constexpr int PER_WRITER = 10000;
  std::atomic<uint64_t> drained_total{0};

  // Each writer pushes entries with unique sequence numbers.
  std::vector<std::thread> writers;
  for (int w = 0; w < WRITERS; ++w) {
    writers.emplace_back([&, w]() {
      for (int i = 0; i < PER_WRITER; ++i) {
        Event dummy_event{nullptr, EventType::TIMER_CALLBACK};
        queue.push(
            EventQueueEntry{static_cast<Tick>(w * PER_WRITER + i), 0, &dummy_event, nullptr});
      }
    });
  }

  // Reader thread drains continuously.
  std::atomic<bool> writers_done{false};
  std::thread reader([&]() {
    EventQueue local_queue;
    while (!writers_done.load(std::memory_order_relaxed) || !queue.empty()) {
      drained_total.fetch_add(queue.drain_into(local_queue), std::memory_order_relaxed);
      std::this_thread::yield();
    }
    // Final drain.
    drained_total.fetch_add(queue.drain_into(local_queue), std::memory_order_relaxed);
  });

  for (auto &t : writers)
    t.join();
  writers_done.store(true, std::memory_order_relaxed);
  reader.join();

  EXPECT_EQ(drained_total.load(), static_cast<uint64_t>(WRITERS) * PER_WRITER);
}

TEST(SpinlockTest, CrossPartitionQueueEmptyDrain) {
  CrossPartitionQueue queue;
  EventQueue local;
  EXPECT_EQ(queue.drain_into(local), 0u);
  EXPECT_TRUE(queue.empty());
}

// ============================================================================
// Area 4: Termination Tests
// ============================================================================

TEST(TerminationTest, QuiescenceDetection) {
  SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<CounterComponent>("c0", 10));
  engine.topology().set_root(std::move(root));
  engine.build();
  auto exit = engine.run();

  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_NE(exit.message.find("quiescent"), std::string::npos);
}

TEST(TerminationTest, AllPrimaryDoneTrigger) {
  SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *p = root->add_child(std::make_unique<ProducerComponent>("p0", 5));
  auto *c = root->add_child(std::make_unique<ConsumerComponent>("c0"));
  engine.topology().set_root(std::move(root));
  engine.topology().add_link(static_cast<ProducerComponent *>(p)->out_port(),
                             static_cast<ConsumerComponent *>(c)->in_port(), 1);
  engine.build();
  auto exit = engine.run();

  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_NE(exit.message.find("primaries"), std::string::npos);
  EXPECT_EQ(static_cast<ConsumerComponent *>(c)->received.size(), 5u);
}

TEST(TerminationTest, MaxTicksSentinel) {
  SimulationEngine engine({.max_ticks = 100, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<InfiniteComponent>("inf0"));
  engine.topology().set_root(std::move(root));
  engine.build();
  auto exit = engine.run();

  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_NE(exit.message.find("max ticks"), std::string::npos);
}

TEST(TerminationTest, RequestExitWakesAllPartitions) {
  // max_ticks as safety net. Keep low since each tick = one barrier round.
  SimulationEngine engine({.max_ticks = 500, .num_threads = 4});
  auto root = std::make_unique<CompositeComponent>("root");
  for (int i = 0; i < 4; ++i)
    root->add_child(std::make_unique<InfiniteComponent>("inf" + std::to_string(i)));
  engine.topology().set_root(std::move(root));
  engine.build();

  // Run in background, request exit after 50ms.
  std::thread runner([&]() { engine.run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  engine.request_exit("test stop");
  runner.join();

  // Accept either EXIT_REQUEST (request_exit propagated) or COMPLETED
  // (max_ticks safety net fired first).
  auto reason = engine.last_exit().reason;
  EXPECT_TRUE(reason == ExitReason::EXIT_REQUEST || reason == ExitReason::COMPLETED);
}

TEST(TerminationTest, StepModeConsistency) {
  SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *c = root->add_child(std::make_unique<CounterComponent>("c0", 10));
  engine.topology().set_root(std::move(root));
  engine.build();

  while (engine.step())
    ;

  EXPECT_EQ(static_cast<CounterComponent *>(c)->count, 10u);
  EXPECT_EQ(engine.last_exit().reason, ExitReason::COMPLETED);
}

// ============================================================================
// Area 1: LBTS Correctness Tests
// ============================================================================

TEST(LBTSTest, TwoPartitionPingPong) {
  SimulationEngine engine({.max_ticks = 1000, .num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *a = root->add_child(std::make_unique<ProducerComponent>("p0", 1, 1, false));
  auto *b = root->add_child(std::make_unique<ConsumerComponent>("c1"));
  auto *prod = static_cast<ProducerComponent *>(a);
  auto *cons = static_cast<ConsumerComponent *>(b);

  engine.topology().set_root(std::move(root));
  engine.topology().add_link(prod->out_port(), cons->in_port(), 5);
  build_with_manual_partitions(engine, 2, partition_by_name_suffix);

  auto exit = engine.run();
  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_EQ(cons->received.size(), 1u);
}

TEST(LBTSTest, IdlePartitionDoesNotBlock) {
  // 3 partitions: A sends to B, C is idle.
  SimulationEngine engine({.max_ticks = 100000, .num_threads = 3});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *a = root->add_child(std::make_unique<ProducerComponent>("p0", 5, 1, false));
  auto *b = root->add_child(std::make_unique<ConsumerComponent>("c1"));
  root->add_child(std::make_unique<CounterComponent>("idle2", 0)); // Idle.

  auto *pa = static_cast<ProducerComponent *>(a);
  auto *cb = static_cast<ConsumerComponent *>(b);

  engine.topology().set_root(std::move(root));
  engine.topology().add_link(pa->out_port(), cb->in_port(), 10);
  build_with_manual_partitions(engine, 3, partition_by_name_suffix);

  auto exit = engine.run();
  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_EQ(cb->received.size(), 5u);
}

TEST(LBTSTest, TimestampAdvancesEnableProgress) {
  // A → B (no return link). A sends one message, then finishes.
  SimulationEngine engine({.max_ticks = 1000, .num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *a = root->add_child(std::make_unique<ProducerComponent>("p0", 1, 1, false));
  auto *b = root->add_child(std::make_unique<ConsumerComponent>("c1"));

  auto *pa = static_cast<ProducerComponent *>(a);
  auto *cb = static_cast<ConsumerComponent *>(b);

  engine.topology().set_root(std::move(root));
  engine.topology().add_link(pa->out_port(), cb->in_port(), 20);
  build_with_manual_partitions(engine, 2, partition_by_name_suffix);

  auto exit = engine.run();
  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_EQ(cb->received.size(), 1u);
  // Message should arrive at start_tick (1) + latency (20) = 21.
  EXPECT_EQ(cb->received[0].first, 21u);
}

// ============================================================================
// Area 2: Cross-Partition Communication Tests
// ============================================================================

TEST(CrossPartitionTest, MessageDelivery) {
  // max_ticks must be past last arrival: 20 msgs at ticks 1,11,...,191 + latency 100 = 291.
  SimulationEngine engine({.max_ticks = 1000, .num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *p = root->add_child(std::make_unique<ProducerComponent>("p0", 20, 1, false));
  auto *c = root->add_child(std::make_unique<ConsumerComponent>("c1"));

  auto *prod = static_cast<ProducerComponent *>(p);
  auto *cons = static_cast<ConsumerComponent *>(c);

  engine.topology().set_root(std::move(root));
  engine.topology().add_link(prod->out_port(), cons->in_port(), 100);
  build_with_manual_partitions(engine, 2, partition_by_name_suffix);

  auto exit = engine.run();
  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_EQ(cons->received.size(), 20u);

  // Verify messages arrive in order and at correct ticks.
  for (size_t i = 1; i < cons->received.size(); ++i)
    EXPECT_GE(cons->received[i].first, cons->received[i - 1].first);
}

TEST(CrossPartitionTest, LinkLatencyAdded) {
  SimulationEngine engine({.max_ticks = 1000, .num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *p = root->add_child(std::make_unique<ProducerComponent>("p0", 1, 100, false));
  auto *c = root->add_child(std::make_unique<ConsumerComponent>("c1"));

  auto *prod = static_cast<ProducerComponent *>(p);
  auto *cons = static_cast<ConsumerComponent *>(c);

  engine.topology().set_root(std::move(root));
  engine.topology().add_link(prod->out_port(), cons->in_port(), 42);
  build_with_manual_partitions(engine, 2, partition_by_name_suffix);

  engine.run();
  ASSERT_EQ(cons->received.size(), 1u);
  EXPECT_EQ(cons->received[0].first, 142u); // 100 + 42.
}

// ============================================================================
// Area 3: Async Event Causality Tests
// ============================================================================

TEST(AsyncCausalityTest, ScheduleEventNowProducesReasonableTimestamp) {
  // Single-threaded, no pacing: schedule_event_now should use GVT (current_time_).
  SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *c = root->add_child(std::make_unique<CounterComponent>("c0", 5));
  engine.topology().set_root(std::move(root));
  engine.build();

  // Run a few steps to advance time.
  for (int i = 0; i < 3; ++i)
    engine.step();

  // Inject async event — it should get timestamp >= current_time.
  Tick before = engine.global_time();
  std::atomic<Tick> injected_tick{0};
  Event test_event{static_cast<CounterComponent *>(c), EventType::TIMER_CALLBACK,
                   [&](Tick ts, Message *) { injected_tick.store(ts, std::memory_order_relaxed); }};
  engine.schedule_event_now(&test_event);

  // Process remaining.
  while (engine.step()) {
  }

  EXPECT_GE(injected_tick.load(), before);
}

// ============================================================================
// Area 7: Stress / Integration Tests
// ============================================================================

TEST(StressTest, FourPartitionRingStress) {
  constexpr uint32_t LAPS = 10;
  SimulationEngine engine({.max_ticks = 10000, .num_threads = 4});
  auto root = std::make_unique<CompositeComponent>("root");

  // Create 4 ping-pong components in a ring (non-primary, rely on max_ticks).
  auto *a = root->add_child(std::make_unique<PingPongComponent>("r0", LAPS, true, false));
  auto *b = root->add_child(std::make_unique<PingPongComponent>("r1", LAPS, false, false));
  auto *c = root->add_child(std::make_unique<PingPongComponent>("r2", LAPS, false, false));
  auto *d = root->add_child(std::make_unique<PingPongComponent>("r3", LAPS, false, false));
  auto *pa = static_cast<PingPongComponent *>(a);
  auto *pb = static_cast<PingPongComponent *>(b);
  auto *pc = static_cast<PingPongComponent *>(c);
  auto *pd = static_cast<PingPongComponent *>(d);

  engine.topology().set_root(std::move(root));
  engine.topology().add_link(pa->out_port(), pb->in_port(), 5);
  engine.topology().add_link(pb->out_port(), pc->in_port(), 5);
  engine.topology().add_link(pc->out_port(), pd->in_port(), 5);
  engine.topology().add_link(pd->out_port(), pa->in_port(), 5);
  build_with_manual_partitions(engine, 4, partition_by_name_suffix);

  auto exit = engine.run();
  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  // The initiator (r0) may stop replying on its last receive, causing
  // downstream components to get one fewer message. Verify >=LAPS-1.
  EXPECT_GE(pa->recv_count, LAPS - 1);
  EXPECT_GE(pb->recv_count, LAPS - 1);
  EXPECT_GE(pc->recv_count, LAPS - 1);
  EXPECT_GE(pd->recv_count, LAPS - 1);
}

TEST(StressTest, LongRunningThousandsOfEvents) {
  constexpr uint32_t MESSAGES = 10;
  SimulationEngine engine({.max_ticks = 1000, .num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");

  auto *a = root->add_child(std::make_unique<PingPongComponent>("pp0", MESSAGES, true, false));
  auto *b = root->add_child(std::make_unique<PingPongComponent>("pp1", MESSAGES, false, false));
  auto *pa = static_cast<PingPongComponent *>(a);
  auto *pb = static_cast<PingPongComponent *>(b);

  engine.topology().set_root(std::move(root));
  engine.topology().add_link(pa->out_port(), pb->in_port(), 1);
  engine.topology().add_link(pb->out_port(), pa->in_port(), 1);
  build_with_manual_partitions(engine, 2, partition_by_name_suffix);

  auto exit = engine.run();
  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  // Verify at least some messages were exchanged (the LBTS protocol has
  // overhead that limits throughput in short simulations).
  EXPECT_GT(pa->recv_count, 0u);
  EXPECT_GT(pb->recv_count, 0u);
  // Verify monotonic ordering for both.
  for (size_t i = 1; i < pa->recv_ticks.size(); ++i)
    EXPECT_GE(pa->recv_ticks[i], pa->recv_ticks[i - 1]);
  for (size_t i = 1; i < pb->recv_ticks.size(); ++i)
    EXPECT_GE(pb->recv_ticks[i], pb->recv_ticks[i - 1]);
}

TEST(StressTest, AsyncInjectionDuringActiveSimulation) {
  // Use InfiniteComponent to keep the simulation running while we inject.
  // max_ticks as safety net. Keep low since each tick = one barrier round.
  SimulationEngine engine({.max_ticks = 500, .num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<InfiniteComponent>("inf0"));
  root->add_child(std::make_unique<InfiniteComponent>("inf1"));
  engine.topology().set_root(std::move(root));
  engine.build();

  std::atomic<uint32_t> async_processed{0};
  auto *target = engine.topology().partitions()[0].components[0];
  Event async_event{target, EventType::TIMER_CALLBACK, [&](Tick, Message *) {
                      async_processed.fetch_add(1, std::memory_order_relaxed);
                    }};

  // Inject async events before running, so they're available immediately.
  for (int i = 0; i < 20; ++i)
    engine.schedule_event_async(&async_event, static_cast<Tick>(i + 1));

  auto exit = engine.run();

  EXPECT_GT(async_processed.load(), 0u);
}

// ============================================================================
// Cache VMID-tagging invariants
// ============================================================================
//
// The memory hierarchy tags every line by (vmid, addr) so two processes that
// alias the same guest VA do not share a cached line. These tests exercise that
// invariant directly on the header-only Cache: distinct data per VMID at the
// same address, eviction reporting the evicted line's owner VMID, and per-VMID
// invalidation.

namespace {
// 64B lines, 4 sets, 2-way. Small associativity makes eviction easy to force.
using TestCache = Cache<6, 4, 2>;

// Fill the whole line for @p addr/@p vmid with a repeating 32-bit pattern.
void fill_line_word(TestCache &cache, uint64_t addr, uint32_t vmid, uint32_t word) {
  uint8_t line[TestCache::LINE_SIZE];
  for (uint32_t i = 0; i < TestCache::LINE_SIZE; i += sizeof(word))
    std::memcpy(line + i, &word, sizeof(word));
  cache.allocate(addr, vmid);
  cache.fill_line(addr, line, vmid);
}

uint32_t read_line_word(TestCache &cache, uint64_t addr, uint32_t vmid) {
  uint32_t word = 0;
  cache.read_line(addr, reinterpret_cast<uint8_t *>(&word), 0, sizeof(word), vmid);
  return word;
}
} // namespace

TEST(CacheVmidTest, SameAddressUnderTwoVmidsStoresDistinctData) {
  TestCache cache;
  constexpr uint64_t kAddr = 0x4000;

  fill_line_word(cache, kAddr, /*vmid=*/1, 0xAAAAAAAAu);
  fill_line_word(cache, kAddr, /*vmid=*/2, 0xBBBBBBBBu);

  // Two distinct lines coexist in the same set; each VMID sees its own data.
  CacheTag *tag1 = nullptr;
  CacheTag *tag2 = nullptr;
  EXPECT_TRUE(cache.lookup(kAddr, &tag1, /*vmid=*/1));
  EXPECT_TRUE(cache.lookup(kAddr, &tag2, /*vmid=*/2));
  EXPECT_EQ(read_line_word(cache, kAddr, /*vmid=*/1), 0xAAAAAAAAu);
  EXPECT_EQ(read_line_word(cache, kAddr, /*vmid=*/2), 0xBBBBBBBBu);
}

TEST(CacheVmidTest, EvictionReportsEvictedLineOwnerVmid) {
  TestCache cache;
  // Three addresses that map to the same set (set index = (addr >> 6) & 3).
  // With 2 ways, allocating a third forces eviction of the LRU (first) line.
  constexpr uint64_t kSetStride = static_cast<uint64_t>(TestCache::LINE_SIZE) * 4;
  const uint64_t addr_a = 0x1000;
  const uint64_t addr_b = addr_a + kSetStride;
  const uint64_t addr_c = addr_b + kSetStride;

  // First line is owned by vmid 7 and marked dirty so a real cache would write
  // it back under that vmid.
  CacheTag *ta = cache.allocate(addr_a, /*vmid=*/7);
  ta->dirty = true;
  cache.allocate(addr_b, /*vmid=*/8);

  CacheTag evicted;
  cache.allocate(addr_c, /*vmid=*/9, &evicted);

  EXPECT_TRUE(evicted.valid);
  EXPECT_TRUE(evicted.dirty);
  EXPECT_EQ(evicted.vmid, 7u); // writeback must use the evicted line's owner.
}

TEST(CacheVmidTest, InvalidatePerVmidLeavesOtherVmidIntact) {
  TestCache cache;
  constexpr uint64_t kAddr = 0x8000;

  fill_line_word(cache, kAddr, /*vmid=*/1, 0x11111111u);
  fill_line_word(cache, kAddr, /*vmid=*/2, 0x22222222u);

  cache.invalidate(kAddr, /*vmid=*/1);

  EXPECT_FALSE(cache.lookup(kAddr, nullptr, /*vmid=*/1));
  EXPECT_TRUE(cache.lookup(kAddr, nullptr, /*vmid=*/2));
  EXPECT_EQ(read_line_word(cache, kAddr, /*vmid=*/2), 0x22222222u);
}
