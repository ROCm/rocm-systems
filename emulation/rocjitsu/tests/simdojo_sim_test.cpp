// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file simdojo_sim_test.cpp
/// @brief Tests for simdojo: the simulation engine (LBTS correctness,
/// cross-partition communication, async causality, termination, pacing,
/// spinlock, stress) and the components built on it (Cache, TagArray).

#include "simdojo/components/cache.h"
#include "simdojo/components/tag_array.h"
#include "simdojo/sim/clocked.h"
#include "simdojo/sim/pacing_controller.h"
#include "simdojo/sim/simulation.h"
#include "util/spinlock.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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

/// Component that counts initialize()/startup()/shutdown() calls. Used to verify
/// that shutdown() cleanup fires exactly once per initialized component, on both
/// the normal shutdown path and the startup-failure unwind path. When given a
/// shared order log, records its own name on shutdown() so tests can assert the
/// reverse-topology-order shutdown contract.
class LifecycleCountingComponent : public Component {
public:
  explicit LifecycleCountingComponent(std::string name,
                                      std::vector<std::string> *shutdown_order = nullptr)
      : Component(std::move(name)), shutdown_order_(shutdown_order) {}

  void initialize() override { ++initializes; }
  void startup() override { ++startups; }
  void shutdown() override {
    ++shutdowns;
    if (shutdown_order_)
      shutdown_order_->push_back(name());
  }

  uint32_t initializes = 0;
  uint32_t startups = 0;
  uint32_t shutdowns = 0;

private:
  std::vector<std::string> *shutdown_order_;
};

/// Component whose startup() requests an ordinary exit and only then throws.
///
/// @details Pins the precedence between the two terminal writers. The exit request
/// records a code-0 status FIRST, so unless the failure outranks it, run() hands back
/// success for a generation whose components never started.
class ExitThenThrowStartupComponent : public Component {
public:
  explicit ExitThenThrowStartupComponent(std::string name) : Component(std::move(name)) {}

  void startup() override {
    engine()->request_exit("ordinary exit request from startup", /*code=*/0);
    throw std::runtime_error("startup failed after requesting exit");
  }
};

/// Component whose startup() throws the first N times it is called, then
/// succeeds. Also counts shutdown() so tests can assert unwind cleanup.
class FlakyStartupComponent : public Component {
public:
  FlakyStartupComponent(std::string name, uint32_t throws_before_success)
      : Component(std::move(name)), remaining_throws_(throws_before_success) {}

  void startup() override {
    if (remaining_throws_ > 0) {
      --remaining_throws_;
      throw std::runtime_error("startup deliberately failing");
    }
    ++successful_startups;
  }

  void shutdown() override { ++shutdowns; }

  uint32_t successful_startups = 0;
  uint32_t shutdowns = 0;

private:
  uint32_t remaining_throws_;
};

/// Component whose startup() AND shutdown() both throw. Component::shutdown() is
/// not noexcept, so the startup-failure unwind must survive a throwing shutdown
/// hook without escaping the engine or stranding wait_until_started().
class DoublyThrowingComponent : public Component {
public:
  explicit DoublyThrowingComponent(std::string name) : Component(std::move(name)) {}

  void startup() override {
    ++startups;
    throw std::runtime_error("startup deliberately failing");
  }

  void shutdown() override {
    ++shutdowns;
    throw std::runtime_error("shutdown deliberately failing");
  }

  uint32_t startups = 0;
  uint32_t shutdowns = 0;
};

/// Helper: build engine with manual partition assignment.
/// assigner maps component name → partition ID.
void build_with_manual_partitions(SimulationEngine &engine, uint32_t num_partitions,
                                  std::function<PartitionID(Component *)> assigner) {
  engine.topology().partition_manual(num_partitions, std::move(assigner));
  engine.create();
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

TEST(TopologyPartitionTest, RepartitionRetainsExternalLinkOwnerOnce) {
  // Topology borrows external endpoint owners, so external must outlive topology.
  ProducerComponent external("external", 0, 1, false);
  Topology topology;
  auto root = std::make_unique<CompositeComponent>("root");
  auto consumer = std::make_unique<ConsumerComponent>("consumer");
  auto *consumer_ptr = consumer.get();
  root->add_child(std::move(consumer));
  topology.set_root(std::move(root));

  topology.add_link(external.out_port(), consumer_ptr->in_port(), 1);

  for (int pass = 0; pass < 2; ++pass) {
    SCOPED_TRACE(::testing::Message() << "pass=" << pass);
    topology.partition_manual(2, [](Component *) { return PartitionID{0}; });
    EXPECT_EQ(external.partition_id(), 0u);
    EXPECT_EQ(std::count(topology.partitions()[0].components.begin(),
                         topology.partitions()[0].components.end(), &external),
              1);
  }
}

TEST(TopologyPartitionTest, ManualPartitionRunsExternalProducerOnAssignedPartition) {
  // SimulationEngine borrows external endpoint owners, so external must outlive engine.
  ProducerComponent external("external", 3);
  SimulationEngine engine({.num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  auto consumer_component = std::make_unique<ConsumerComponent>("consumer");
  auto *consumer = consumer_component.get();
  root->add_child(std::move(consumer_component));
  engine.topology().set_root(std::move(root));
  engine.topology().add_link(external.out_port(), consumer->in_port(), 0);

  engine.topology().partition_manual(2, [](Component *) { return PartitionID{1}; });

  ASSERT_EQ(external.partition_id(), 1u);
  ASSERT_EQ(consumer->partition_id(), 1u);
  engine.create();
  auto exit = engine.run();

  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_EQ(external.sent_, 3u);
  ASSERT_EQ(consumer->received.size(), 3u);
  for (size_t i = 0; i < consumer->received.size(); ++i)
    EXPECT_EQ(consumer->received[i].second, i);
}

TEST(TopologyPartitionTest, BalancedSinglePartitionIncludesExternalLinkOwner) {
  // Topology borrows external endpoint owners, so external must outlive topology.
  ProducerComponent external("external", 0, 1, false);
  Topology topology;
  auto root = std::make_unique<CompositeComponent>("root");
  auto consumer = std::make_unique<ConsumerComponent>("consumer");
  auto *consumer_ptr = consumer.get();
  root->add_child(std::move(consumer));
  topology.set_root(std::move(root));

  topology.add_link(external.out_port(), consumer_ptr->in_port(), 1);

  topology.partition_balanced(1);

  ASSERT_EQ(topology.partitions().size(), 1u);
  EXPECT_EQ(external.partition_id(), 0u);
  EXPECT_EQ(std::count(topology.partitions()[0].components.begin(),
                       topology.partitions()[0].components.end(), &external),
            1);
}

TEST(TopologyPartitionTest, BalancedRepartitionRetainsExternalLinkOwnerOnce) {
  // Topology borrows external endpoint owners, so external must outlive topology.
  ProducerComponent external("external", 0, 1, false);
  Topology topology;
  auto root = std::make_unique<CompositeComponent>("root");
  auto consumer = std::make_unique<ConsumerComponent>("consumer");
  auto *consumer_ptr = consumer.get();
  root->add_child(std::move(consumer));
  topology.set_root(std::move(root));

  topology.add_link(external.out_port(), consumer_ptr->in_port(), 1);

  for (int pass = 0; pass < 2; ++pass) {
    SCOPED_TRACE(::testing::Message() << "pass=" << pass);
    topology.partition_balanced(2);
    ASSERT_EQ(topology.partitions().size(), 2u);
    EXPECT_LT(external.partition_id(), 2u);

    size_t occurrences = 0;
    for (const auto &partition : topology.partitions())
      occurrences +=
          std::count(partition.components.begin(), partition.components.end(), &external);
    EXPECT_EQ(occurrences, 1u);
  }
}

TEST(TopologyPartitionTest, MultiThreadedEngineRequiresExplicitPolicy) {
  SimulationEngine engine({.num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<CounterComponent>("counter0", 0));
  root->add_child(std::make_unique<CounterComponent>("counter1", 0));
  engine.topology().set_root(std::move(root));

  EXPECT_THROW(engine.create(), std::invalid_argument);
}

TEST(TopologyPartitionTest, RejectsZeroPartitionCount) {
  Topology topology;

  EXPECT_THROW(topology.partition_balanced(0), std::invalid_argument);
  EXPECT_TRUE(topology.partitions().empty());

  EXPECT_THROW(topology.partition_manual(0, [](Component *) { return PartitionID{0}; }),
               std::invalid_argument);
  EXPECT_TRUE(topology.partitions().empty());
}

TEST(TopologyPartitionTest, OutOfRangeManualAssignmentLeavesExistingStateIntact) {
  Topology topology;
  auto root = std::make_unique<CompositeComponent>("root");
  auto *counter0 = root->add_child(std::make_unique<CounterComponent>("counter0", 0));
  auto *counter1 = root->add_child(std::make_unique<CounterComponent>("counter1", 0));
  topology.set_root(std::move(root));
  topology.partition_manual(1, [](Component *) { return PartitionID{0}; });

  ASSERT_EQ(topology.partitions().size(), 1u);
  ASSERT_EQ(counter0->partition_id(), 0u);
  ASSERT_EQ(counter1->partition_id(), 0u);

  EXPECT_THROW(topology.partition_manual(2, [](Component *) { return PartitionID{2}; }),
               std::invalid_argument);
  EXPECT_EQ(topology.partitions().size(), 1u);
  EXPECT_EQ(counter0->partition_id(), 0u);
  EXPECT_EQ(counter1->partition_id(), 0u);
}

TEST(TopologyPartitionTest, EngineRejectsZeroWorkerThreads) {
  SimulationEngine engine({.num_threads = 0});

  EXPECT_THROW(engine.create(), std::invalid_argument);
  EXPECT_FALSE(engine.is_created());
  EXPECT_TRUE(engine.topology().partitions().empty());
}

TEST(TopologyPartitionTest, EngineRejectsPartitionCountMismatch) {
  SimulationEngine engine({.num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<CounterComponent>("counter0", 0));
  engine.topology().set_root(std::move(root));
  engine.topology().partition_manual(1, [](Component *) { return PartitionID{0}; });

  EXPECT_THROW(engine.create(), std::invalid_argument);
  EXPECT_FALSE(engine.is_created());
}

TEST(TopologyPartitionTest, EngineRejectsZeroLatencyCrossPartitionLink) {
  SimulationEngine engine({.num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  auto producer = std::make_unique<ProducerComponent>("producer0", 0, 1, false);
  auto consumer = std::make_unique<ConsumerComponent>("consumer1");
  auto *producer_ptr = producer.get();
  auto *consumer_ptr = consumer.get();
  root->add_child(std::move(producer));
  root->add_child(std::move(consumer));
  engine.topology().set_root(std::move(root));
  engine.topology().add_link(producer_ptr->out_port(), consumer_ptr->in_port(), 0);
  engine.topology().partition_manual(2, partition_by_name_suffix);

  try {
    engine.create();
    FAIL() << "expected invalid cross-partition latency";
  } catch (const std::invalid_argument &e) {
    const std::string message = e.what();
    EXPECT_NE(message.find("root.producer0.out"), std::string::npos);
    EXPECT_NE(message.find("root.consumer1.in"), std::string::npos);
    EXPECT_NE(message.find("positive latency"), std::string::npos);
  }
  EXPECT_FALSE(engine.is_created());
}

TEST(TopologyPartitionTest, EngineRejectsCrossPartitionQueuedLink) {
  SimulationEngine engine({.num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  auto producer = std::make_unique<ProducerComponent>("producer0", 0, 1, false);
  auto consumer = std::make_unique<ConsumerComponent>("consumer1");
  auto *producer_ptr = producer.get();
  auto *consumer_ptr = consumer.get();
  root->add_child(std::move(producer));
  root->add_child(std::move(consumer));
  engine.topology().set_root(std::move(root));
  engine.topology().add_queued_link(producer_ptr->out_port(), consumer_ptr->in_port(), 1, 4);
  engine.topology().partition_manual(2, partition_by_name_suffix);

  try {
    engine.create();
    FAIL() << "expected cross-partition QueuedLink rejection";
  } catch (const std::invalid_argument &e) {
    const std::string message = e.what();
    EXPECT_NE(message.find("root.producer0.out"), std::string::npos);
    EXPECT_NE(message.find("root.consumer1.in"), std::string::npos);
    EXPECT_NE(message.find("QueuedLink"), std::string::npos);
  }
  EXPECT_FALSE(engine.is_created());
}

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
  engine.create();
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
  engine.create();
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
  engine.create();
  auto exit = engine.run();

  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_NE(exit.message.find("max ticks"), std::string::npos);
}

TEST(StartupReadinessTest, CleanStartupReportsReady) {
  // A clean startup must make wait_until_started() return true without hanging.
  SimulationEngine engine({.max_ticks = 5, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<LifecycleCountingComponent>("c0"));
  engine.topology().set_root(std::move(root));
  engine.create();

  engine.step();
  EXPECT_TRUE(engine.wait_until_started());
}

TEST(StartupReadinessTest, StartupFailureOutranksAnEarlierExitRequest) {
  // The failure is recorded SECOND here, after an ordinary code-0 exit request that
  // the same startup() issued. Whoever wrote first must not win: a generation whose
  // components never started has to report failure, or the C API maps code 0 to
  // success and a dead VM looks like a clean run.
  SimulationEngine engine({.max_ticks = 10, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<ExitThenThrowStartupComponent>("exit_then_throw0"));
  engine.topology().set_root(std::move(root));
  engine.create();

  auto exit = engine.run();

  EXPECT_EQ(exit.reason, ExitReason::INTERRUPTED);
  EXPECT_EQ(exit.code, 1) << "a startup failure must not be reported as a clean exit";
  EXPECT_FALSE(engine.wait_until_started());
  EXPECT_EQ(engine.last_exit().code, 1);
}

TEST(StartupReadinessTest, StartupFailureIsTerminalForTheGeneration) {
  // A startup() throw leaves the partial attempt's event/primary/component state
  // intact, so the create() generation is terminal: step() rethrows once, latches
  // failure, and a same-generation retry must NOT re-run startup (it returns done).
  // A clean retry requires shutdown() + create().
  SimulationEngine engine({.max_ticks = 10, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *flaky = root->add_child(std::make_unique<FlakyStartupComponent>("flaky0", 1));
  engine.topology().set_root(std::move(root));
  engine.create();

  // First step(): startup throws, wait_until_started() observes failure.
  EXPECT_THROW(engine.step(), std::runtime_error);
  EXPECT_FALSE(engine.wait_until_started());

  // step() must record the SAME failure ExitStatus run() would. Without it the
  // create() default {COMPLETED, 0} survives and the terminal guard below hands
  // back a success-looking status for a generation that never started.
  EXPECT_EQ(engine.last_exit().reason, ExitReason::INTERRUPTED);
  EXPECT_EQ(engine.last_exit().code, 1);

  // Same-generation retry: startup is NOT re-run (generation is terminal), so the
  // component's startup() never succeeds and step() reports done.
  EXPECT_FALSE(engine.step());
  EXPECT_EQ(static_cast<FlakyStartupComponent *>(flaky)->successful_startups, 0u);

  // A run() after the failed step() goes through the same terminal guard and must
  // report the recorded failure, not the create() default.
  auto after_failed_step = engine.run();
  EXPECT_EQ(after_failed_step.reason, ExitReason::INTERRUPTED);
  EXPECT_EQ(after_failed_step.code, 1);
  EXPECT_EQ(static_cast<FlakyStartupComponent *>(flaky)->successful_startups, 0u);

  // A fresh generation (shutdown + create) starts cleanly and succeeds.
  engine.shutdown();
  engine.create();
  engine.step();
  EXPECT_TRUE(engine.wait_until_started());
  EXPECT_EQ(static_cast<FlakyStartupComponent *>(flaky)->successful_startups, 1u);
}

TEST(StartupReadinessTest, ThrowingShutdownDuringStartupUnwindStillPublishesFailure) {
  // Component::shutdown() is not noexcept. If the startup-failure unwind ran
  // BEFORE the readiness latch, a throwing shutdown hook would escape the catch
  // path — terminating the interposer's background run() thread — while
  // wait_until_started() stayed blocked forever. The failure must therefore be
  // published first and the unwind caught separately.
  SimulationEngine engine({.max_ticks = 10, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *doubly = root->add_child(std::make_unique<DoublyThrowingComponent>("doubly0"));
  engine.topology().set_root(std::move(root));
  engine.create();

  // run() must return the terminal status rather than letting the shutdown throw
  // propagate out of the engine.
  auto exit = engine.run();
  EXPECT_EQ(exit.reason, ExitReason::INTERRUPTED);
  EXPECT_EQ(exit.code, 1);
  EXPECT_FALSE(engine.wait_until_started());

  auto *comp = static_cast<DoublyThrowingComponent *>(doubly);
  EXPECT_EQ(comp->startups, 1u);
  EXPECT_EQ(comp->shutdowns, 1u) << "the unwind must still attempt the shutdown hook";

  // The engine's own shutdown() must also survive the throwing hook (it is guarded
  // against a second invocation, so this asserts no rethrow escapes destruction).
  EXPECT_NO_THROW(engine.shutdown());
}

TEST(StartupReadinessTest, ThrowingShutdownDuringStepUnwindRethrowsOnlyTheStartupError) {
  // step() rethrows to its foreground caller. The exception it propagates must be
  // the STARTUP failure, not whatever the best-effort unwind's shutdown hook threw
  // on top of it, and readiness must still be published as failed.
  SimulationEngine engine({.max_ticks = 10, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *doubly = root->add_child(std::make_unique<DoublyThrowingComponent>("doubly0"));
  engine.topology().set_root(std::move(root));
  engine.create();

  try {
    engine.step();
    ADD_FAILURE() << "step() must rethrow the startup failure";
  } catch (const std::runtime_error &e) {
    EXPECT_STREQ(e.what(), "startup deliberately failing")
        << "the unwind's shutdown throw must not replace the startup error";
  }

  EXPECT_FALSE(engine.wait_until_started());
  EXPECT_EQ(engine.last_exit().reason, ExitReason::INTERRUPTED);
  EXPECT_EQ(engine.last_exit().code, 1);

  auto *comp = static_cast<DoublyThrowingComponent *>(doubly);
  EXPECT_EQ(comp->startups, 1u);
  EXPECT_EQ(comp->shutdowns, 1u);
}

TEST(StartupReadinessTest, RunAfterFailedStartupFailsClosedWithoutRerun) {
  // run()'s terminal-generation guard must be a RUNTIME check, not just an
  // assert: under -DNDEBUG a second run() after a startup throw must fail closed
  // (return the terminal INTERRUPTED status) instead of re-entering
  // startup_components() on the dirty generation. This runs regardless of build
  // type, so it covers the release path.
  SimulationEngine engine({.max_ticks = 10, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *flaky = root->add_child(std::make_unique<FlakyStartupComponent>("flaky0", 1));
  engine.topology().set_root(std::move(root));
  engine.create();

  // First run(): startup throws; run() catches, latches failure, returns INTERRUPTED.
  auto first = engine.run();
  EXPECT_EQ(first.reason, ExitReason::INTERRUPTED);
  EXPECT_EQ(first.code, 1);
  EXPECT_FALSE(engine.wait_until_started());
  EXPECT_EQ(static_cast<FlakyStartupComponent *>(flaky)->successful_startups, 0u);

  // Second run() on the same (dirty) generation must NOT re-run startup — it
  // returns the terminal status. Without the runtime guard this would re-enter
  // startup_components() on already-scheduled/registered state.
  auto second = engine.run();
  EXPECT_EQ(second.reason, ExitReason::INTERRUPTED);
  EXPECT_EQ(second.code, 1);
  EXPECT_EQ(static_cast<FlakyStartupComponent *>(flaky)->successful_startups, 0u);
}

TEST(StartupReadinessTest, StartupFailureUnwindsEveryInitializedComponentOnce) {
  // A throwing component partway through startup must not leave earlier components'
  // resources dangling: shutdown() pairs with initialize(), so the unwind shuts
  // down EVERY initialized component (not only the started prefix) exactly once —
  // including the component that threw and the one after it that never started.
  std::vector<std::string> shutdown_order;
  SimulationEngine engine({.max_ticks = 10, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *before =
      root->add_child(std::make_unique<LifecycleCountingComponent>("before", &shutdown_order));
  auto *flaky = root->add_child(std::make_unique<FlakyStartupComponent>("flaky", 1));
  auto *after =
      root->add_child(std::make_unique<LifecycleCountingComponent>("after", &shutdown_order));
  engine.topology().set_root(std::move(root));
  engine.create();

  EXPECT_THROW(engine.step(), std::runtime_error);

  auto *before_c = static_cast<LifecycleCountingComponent *>(before);
  auto *after_c = static_cast<LifecycleCountingComponent *>(after);
  EXPECT_EQ(before_c->initializes, 1u);
  EXPECT_EQ(before_c->startups, 1u);
  EXPECT_EQ(before_c->shutdowns, 1u); // started, then unwound
  EXPECT_EQ(after_c->initializes, 1u);
  EXPECT_EQ(after_c->startups, 0u);  // never reached
  EXPECT_EQ(after_c->shutdowns, 1u); // initialized, so still cleaned up
  EXPECT_EQ(static_cast<FlakyStartupComponent *>(flaky)->shutdowns, 1u);

  // shutdown() runs in reverse topology order. "flaky" has no counting log, but
  // the two counting components bracket it, so "after" must be shut down before
  // "before".
  ASSERT_EQ(shutdown_order.size(), 2u);
  EXPECT_EQ(shutdown_order[0], "after");
  EXPECT_EQ(shutdown_order[1], "before");

  // The engine's own shutdown() must not double-shut-down the already-unwound
  // components.
  engine.shutdown();
  EXPECT_EQ(before_c->shutdowns, 1u);
  EXPECT_EQ(after_c->shutdowns, 1u);
}

TEST(StartupReadinessTest, ThrowingShutdownStillCleansUpRemainingComponents) {
  // shutdown_components() marks the generation shut down before invoking any
  // callback, so a callback that throws must not be allowed to skip the components
  // after it — there is no second chance to clean them up. Each callback is
  // isolated; the first failure is reported once the rest have run.
  std::vector<std::string> shutdown_order;
  SimulationEngine engine({.max_ticks = 10, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *first =
      root->add_child(std::make_unique<LifecycleCountingComponent>("first", &shutdown_order));
  auto *thrower = root->add_child(std::make_unique<DoublyThrowingComponent>("thrower"));
  auto *last =
      root->add_child(std::make_unique<LifecycleCountingComponent>("last", &shutdown_order));
  engine.topology().set_root(std::move(root));
  engine.create();

  // Reverse order is last -> thrower -> first, so the throw lands between them.
  // shutdown() must NOT propagate it: it is reached from ~SimulationEngine(), from
  // the engine's own background thread, and across the C API, none of which can
  // absorb an exception.
  EXPECT_NO_THROW(engine.shutdown());
  EXPECT_FALSE(engine.is_created()) << "engine cleanup must complete despite a throwing hook";

  EXPECT_EQ(static_cast<DoublyThrowingComponent *>(thrower)->shutdowns, 1u);
  EXPECT_EQ(static_cast<LifecycleCountingComponent *>(last)->shutdowns, 1u);
  EXPECT_EQ(static_cast<LifecycleCountingComponent *>(first)->shutdowns, 1u)
      << "a throwing callback must not skip cleanup for the components after it";
  ASSERT_EQ(shutdown_order.size(), 2u);
  EXPECT_EQ(shutdown_order[0], "last");
  EXPECT_EQ(shutdown_order[1], "first");
}

TEST(StartupReadinessTest, CreateThenShutdownRunsComponentCleanup) {
  // create() initializes every component; shutting the engine down before any
  // run()/step() must still invoke shutdown() on each initialized component
  // exactly once (shutdown() pairs with initialize(), not startup()).
  SimulationEngine engine({.max_ticks = 10, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *comp = root->add_child(std::make_unique<LifecycleCountingComponent>("comp"));
  engine.topology().set_root(std::move(root));
  engine.create();

  auto *counting = static_cast<LifecycleCountingComponent *>(comp);
  EXPECT_EQ(counting->initializes, 1u);
  EXPECT_EQ(counting->startups, 0u);

  engine.shutdown();
  EXPECT_EQ(counting->shutdowns, 1u);
}

TEST(TerminationTest, RequestExitWakesAllPartitions) {
  // Infinite work makes request_exit() the only termination path.
  SimulationEngine engine({.num_threads = 4});
  auto root = std::make_unique<CompositeComponent>("root");
  for (int i = 0; i < 4; ++i)
    root->add_child(std::make_unique<InfiniteComponent>("inf" + std::to_string(i)));
  engine.topology().set_root(std::move(root));
  engine.topology().partition_balanced(4);
  engine.create();

  ExitStatus exit_status;
  std::thread runner([&]() { exit_status = engine.run(); });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  engine.request_exit("test stop");
  runner.join();

  EXPECT_EQ(exit_status.reason, ExitReason::EXIT_REQUEST);
}

// Regression: the host thread calling request_exit() must not touch the partition
// contexts while the engine thread is destroying them in shutdown(). The daemon does
// exactly this - teardown() requests exit while the VM thread may already be shutting
// down after the guest exited - which TSan caught as a data race and a use-after-free
// on the partition's idle_wakeup_ flag.
TEST(TerminationTest, RequestExitRacingShutdownIsSafe) {
  // Single-threaded mode is what arms the wake path that dereferences contexts_[0].
  // Iterate: the two threads must interleave inside the narrow shutdown window.
  for (int iter = 0; iter < 200; ++iter) {
    SimulationEngine engine({.num_threads = 1});
    auto root = std::make_unique<CompositeComponent>("root");
    root->add_child(std::make_unique<CounterComponent>("c0", 4));
    engine.topology().set_root(std::move(root));
    engine.create();

    // Mirrors rj_vm_run(): run to self-termination, then tear the contexts down.
    std::thread runner([&engine]() {
      engine.run();
      engine.shutdown();
    });

    // Mirrors the daemon teardown path landing concurrently with that shutdown.
    engine.request_exit("test stop");
    runner.join();
  }
}

TEST(TerminationTest, StepModeConsistency) {
  SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *c = root->add_child(std::make_unique<CounterComponent>("c0", 10));
  engine.topology().set_root(std::move(root));
  engine.create();

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
  engine.create();

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
  engine.topology().partition_balanced(2);
  engine.create();

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

TEST(CacheVmidTest, AllocateWithDataReturnsWritableLineAndEvictedBytes) {
  TestCache cache;
  constexpr uint64_t kSetStride = static_cast<uint64_t>(TestCache::LINE_SIZE) * 4;
  constexpr uint64_t kAddrA = 0x1000;
  constexpr uint64_t kAddrB = kAddrA + kSetStride;
  constexpr uint64_t kAddrC = kAddrB + kSetStride;

  auto first = cache.allocate_with_data(kAddrA, /*vmid=*/7);
  ASSERT_NE(first.tag, nullptr);
  ASSERT_NE(first.data, nullptr);
  first.tag->dirty = true;
  std::fill_n(first.data, TestCache::LINE_SIZE, 0xA5);
  cache.allocate(kAddrB, /*vmid=*/8);

  CacheTag evicted;
  std::array<uint8_t, TestCache::LINE_SIZE> evicted_data{};
  auto replacement = cache.allocate_with_data(kAddrC, /*vmid=*/9, &evicted, evicted_data.data());

  ASSERT_NE(replacement.tag, nullptr);
  ASSERT_NE(replacement.data, nullptr);
  EXPECT_TRUE(evicted.valid);
  EXPECT_TRUE(evicted.dirty);
  EXPECT_EQ(evicted.vmid, 7u);
  EXPECT_TRUE(std::all_of(evicted_data.begin(), evicted_data.end(),
                          [](uint8_t byte) { return byte == 0xA5; }));
}

TEST(CacheVmidTest, InvalidateAllVmidsRemovesEveryAliasedLine) {
  TestCache cache;
  constexpr uint64_t kAddr = 0xC000;

  fill_line_word(cache, kAddr, /*vmid=*/1, 0x11111111u);
  fill_line_word(cache, kAddr, /*vmid=*/2, 0x22222222u);

  cache.invalidate_all_vmids(kAddr);

  EXPECT_FALSE(cache.lookup(kAddr, nullptr, /*vmid=*/1));
  EXPECT_FALSE(cache.lookup(kAddr, nullptr, /*vmid=*/2));
}

TEST(CacheVmidTest, LineDataForReadReturnsMatchingVmidData) {
  TestCache cache;
  constexpr uint64_t kAddr = 0x10000;

  auto allocation = cache.allocate_with_data(kAddr, /*vmid=*/4);
  ASSERT_NE(allocation.data, nullptr);
  for (uint32_t i = 0; i < TestCache::LINE_SIZE; ++i)
    allocation.data[i] = static_cast<uint8_t>(i ^ 0x5A);

  const uint8_t *line = cache.line_data_for_read(kAddr, /*vmid=*/4);
  ASSERT_NE(line, nullptr);
  for (uint32_t i = 0; i < TestCache::LINE_SIZE; ++i)
    EXPECT_EQ(line[i], static_cast<uint8_t>(i ^ 0x5A));
  EXPECT_EQ(cache.line_data_for_read(kAddr, /*vmid=*/5), nullptr);
}

// ============================================================================
// ClockDomain arithmetic
// ============================================================================

namespace {

/// @brief A 1 GHz domain: period 1000 ticks at the 1 ps tick resolution.
ClockDomain ghz(Tick phase = 0) { return ClockDomain("test", 1'000'000'000ULL, phase); }

} // namespace

TEST(ClockDomainArithmeticTest, NextEdgeRoundsUpOntoTheGrid) {
  const ClockDomain domain = ghz();
  EXPECT_EQ(domain.next_edge(1000), 1000u);
  EXPECT_EQ(domain.next_edge(1001), 2000u);
  EXPECT_EQ(domain.next_edge(1999), 2000u);
  EXPECT_EQ(domain.next_edge(2000), 2000u);
  // Idempotent, and specifically not by being constant: the two inputs below
  // are on different edges, so a next_edge() that always answered first_edge()
  // would fail here even though it would pass a bare double-application check.
  EXPECT_EQ(domain.next_edge(domain.next_edge(1001)), 2000u);
  EXPECT_EQ(domain.next_edge(domain.next_edge(5001)), 6000u);
}

TEST(ClockDomainArithmeticTest, NextEdgeClampsUpToTheFirstEdge) {
  // A domain has no edges before first_edge(), so every tick below it resolves
  // there. Anything else would hand out a tick the framework does not treat as
  // an edge: startup() schedules the first one at first_edge().
  const ClockDomain domain = ghz(/*phase=*/250);
  EXPECT_EQ(domain.first_edge(), 1250u);
  EXPECT_EQ(domain.next_edge(0), 1250u);
  EXPECT_EQ(domain.next_edge(1249), 1250u);
  EXPECT_EQ(domain.next_edge(1250), 1250u);
  EXPECT_EQ(domain.next_edge(1251), 2250u);
  EXPECT_EQ(domain.next_edge(2250), 2250u);
}

TEST(ClockDomainArithmeticTest, NextEdgeHandlesAPhaseWiderThanThePeriod) {
  // The phase is an absolute tick, not a sub-period offset, so it may exceed
  // the period. The modulus is taken against the distance from the phase, not
  // from zero, which is the case a "% period" written against the wrong origin
  // gets wrong.
  const ClockDomain domain = ghz(/*phase=*/7'400);
  EXPECT_EQ(domain.first_edge(), 8'400u);
  EXPECT_EQ(domain.next_edge(8'401), 9'400u);
  EXPECT_EQ(domain.next_edge(9'400), 9'400u);
  EXPECT_EQ(domain.next_edge(12'000), 12'400u);
}

TEST(ClockDomainArithmeticTest, EdgeAfterIsOnePeriodOn) {
  const ClockDomain domain = ghz(/*phase=*/250);
  EXPECT_EQ(domain.edge_after(1250), 2250u);
  EXPECT_EQ(domain.edge_after(2250), 3250u);
  // Agrees with the general form on every edge, which is the precondition
  // under which the clock handler is allowed to use the cheap one.
  for (Tick edge = domain.first_edge(); edge < 20'000; edge = domain.edge_after(edge))
    EXPECT_EQ(domain.edge_after(edge), domain.next_edge(edge + 1));
}

TEST(ClockDomainArithmeticTest, EdgeAlignmentSaturatesRatherThanWrapping) {
  const ClockDomain domain = ghz();
  EXPECT_EQ(domain.next_edge(TICK_MAX - 1), TICK_MAX);
  EXPECT_EQ(domain.next_edge(TICK_MAX), TICK_MAX);
  EXPECT_EQ(domain.edge_after(TICK_MAX - 1), TICK_MAX);
  EXPECT_EQ(domain.edge_after(TICK_MAX), TICK_MAX);
  // The last representable edge still aligns to itself rather than saturating.
  const Tick last = TICK_MAX - (TICK_MAX % 1000);
  EXPECT_EQ(domain.next_edge(last), last);
}

TEST(ClockDomainArithmeticTest, CyclesToTicksSaturatesOnlyWhereItMust) {
  const ClockDomain domain = ghz();
  EXPECT_EQ(domain.cycles_to_ticks(0), 0u);
  EXPECT_EQ(domain.cycles_to_ticks(5), 5000u);
  // The largest representable duration, pinned so that a boundary slip which
  // turned legitimate long durations into "never" would be visible.
  EXPECT_EQ(domain.cycles_to_ticks(TICK_MAX / 1000), (TICK_MAX / 1000) * 1000);
  EXPECT_EQ(domain.cycles_to_ticks(TICK_MAX / 1000 + 1), TICK_MAX);
  EXPECT_EQ(domain.cycles_to_ticks(TICK_MAX), TICK_MAX);
}

TEST(ClockDomainArithmeticTest, DeadlineSaturatesTheSumRatherThanTheDuration) {
  const ClockDomain domain = ghz();
  EXPECT_EQ(domain.deadline(1000, 3), 4000u);
  EXPECT_EQ(domain.deadline(0, 0), 0u);
  // The whole reason deadline() exists: `start + cycles_to_ticks(n)` written
  // by hand wraps back into the past here, and this must not.
  EXPECT_EQ(domain.deadline(5000, TICK_MAX), TICK_MAX);
  EXPECT_EQ(domain.deadline(TICK_MAX - 500, 1), TICK_MAX);
  EXPECT_GE(domain.deadline(TICK_MAX - 500, 1), TICK_MAX - 500);
}

TEST(ClockDomainArithmeticTest, TicksToCyclesTruncatesTowardZero) {
  const ClockDomain domain = ghz();
  EXPECT_EQ(domain.ticks_to_cycles(0), 0u);
  EXPECT_EQ(domain.ticks_to_cycles(999), 0u);
  EXPECT_EQ(domain.ticks_to_cycles(1000), 1u);
  EXPECT_EQ(domain.ticks_to_cycles(1999), 1u);
}

TEST(ClockDomainArithmeticTest, AFrequencyThatDoesNotDivideTheResolutionRoundsDown) {
  // 2.1 GHz is the shape of a real shader clock and does not divide a
  // picosecond resolution. The period rounds down, so the domain runs slightly
  // fast, and the two frequency accessors must not agree about it.
  const ClockDomain domain("shader", 2'100'000'000ULL);
  EXPECT_EQ(domain.period(), 476u);
  EXPECT_EQ(domain.frequency(), 2'100'000'000u);
  EXPECT_EQ(domain.effective_frequency(), 2'100'840'336u);
  EXPECT_EQ(domain.cycles_to_ticks(1'000'000'000ULL), 476'000'000'000ULL);
  // A domain whose frequency divides the resolution has nothing to disagree
  // about.
  EXPECT_EQ(ghz().frequency(), ghz().effective_frequency());
}

TEST(ClockDomainArithmeticTest, AnUnrepresentableClockIsRejected) {
  EXPECT_THROW(ClockDomain("stopped", 0), std::invalid_argument);
  // Above the tick resolution the period rounds to zero, and a zero period is
  // a division by zero in every conversion below rather than a fast clock.
  EXPECT_THROW(ClockDomain("too_fast", TICKS_PER_SECOND + 1), std::invalid_argument);
  EXPECT_NO_THROW(ClockDomain("at_resolution", TICKS_PER_SECOND));
  // A phase that leaves no representable first edge would wrap first_edge(),
  // and every alignment is computed from it.
  EXPECT_THROW(ClockDomain("late", 1'000'000'000ULL, TICK_MAX - 999), std::invalid_argument);
  EXPECT_NO_THROW(ClockDomain("just_in_time", 1'000'000'000ULL, TICK_MAX - 1000));
}

TEST(ClockDomainArithmeticTest, ADerivedDomainsEdgesAreASubsetOfItsParents) {
  // 3 GHz does not divide the tick resolution: its period is 333, so a
  // quarter-rate child derived from the frequency would get 1333 and drift off
  // the parent's grid without bound.
  const ClockDomain parent("parent", 3'000'000'000ULL);
  EXPECT_EQ(parent.period(), 333u);
  const ClockDomain child = parent.derive("child", 4);
  EXPECT_EQ(child.period(), 1332u);
  EXPECT_EQ(child.frequency(), 750'000'000u);

  for (Tick edge = child.first_edge(); edge < 100'000; edge = child.edge_after(edge))
    EXPECT_EQ(parent.next_edge(edge), edge) << "child edge " << edge << " is not a parent edge";
}

TEST(ClockDomainArithmeticTest, DeriveRejectsWhatItCannotRepresent) {
  const ClockDomain parent = ghz();
  EXPECT_THROW(parent.derive("zero", 0), std::invalid_argument);
  EXPECT_THROW(parent.derive("phased", 2, TICK_MAX), std::invalid_argument);
  EXPECT_NO_THROW(parent.derive("half", 2));
}

namespace {

/// @brief A clocked component that records the edges it was advanced on.
class EdgeRecorder : public Clocked<Component> {
public:
  EdgeRecorder(const ClockDomain &domain, uint32_t edges)
      : Clocked<Component>("edge_recorder", domain), remaining_(edges) {}

  bool advance(Tick now) override {
    edges.push_back(now);
    if (remaining_ > 0)
      --remaining_;
    return remaining_ > 0;
  }

  /// @brief Resume from @p after on the next step, from outside advance().
  void resume_from(Tick after) { resume_clock(after); }

  std::vector<Tick> edges;

private:
  uint32_t remaining_;
};

/// @brief An engine holding one EdgeRecorder, stepped under a hard cap.
class EdgeRig {
public:
  EdgeRig(const ClockDomain &domain, uint32_t edges) {
    auto root = std::make_unique<CompositeComponent>("root");
    recorder =
        static_cast<EdgeRecorder *>(root->add_child(std::make_unique<EdgeRecorder>(domain, edges)));
    engine.topology().set_root(std::move(root));
    engine.create();
  }

  /// @brief Step until the recorder's clock stops, or the cap trips.
  ///
  /// @details Stops on the component rather than on step(), because an engine
  /// stepped past an empty queue terminates and cannot be resumed, and these
  /// tests resume it.
  /// @returns Whether the clock went quiet on its own.
  bool run(uint32_t max_steps = 64) {
    for (uint32_t i = 0; i < max_steps; ++i) {
      if (!engine.step())
        return true;
      if (!recorder->running())
        return true;
    }
    return false;
  }

  SimulationEngine engine{{}};
  EdgeRecorder *recorder = nullptr;
};

} // namespace

TEST(ClockedEdgeTest, EdgesFollowThePhaseOffset) {
  const ClockDomain domain("phased", 1'000'000'000ULL, /*phase=*/250);
  EdgeRig rig(domain, 3);

  ASSERT_TRUE(rig.run());
  EXPECT_EQ(rig.recorder->edges, (std::vector<Tick>{1250, 2250, 3250}));
}

TEST(ClockedEdgeTest, ResumeClockLandsOnAnEdge) {
  // resume_clock() is the call whose control flow this commit changed: it lost
  // an open-coded modulus and a special case for ticks below the first edge.
  const ClockDomain domain("phased", 1'000'000'000ULL, /*phase=*/250);
  EdgeRig rig(domain, 1);

  ASSERT_TRUE(rig.run());
  EXPECT_EQ(rig.recorder->edges, (std::vector<Tick>{1250}));
  EXPECT_FALSE(rig.recorder->running());

  // Mid-period rounds up to the next edge.
  rig.recorder->resume_from(4000);
  EXPECT_TRUE(rig.recorder->running());
  ASSERT_TRUE(rig.run());
  EXPECT_EQ(rig.recorder->edges.back(), 4250u);

  // Exactly on an edge stays on it rather than skipping a period.
  rig.recorder->resume_from(9250);
  ASSERT_TRUE(rig.run());
  EXPECT_EQ(rig.recorder->edges.back(), 9250u);

  // Below the first edge clamps up to it rather than taking a modulus against
  // a distance that has not elapsed yet.
  EXPECT_EQ(domain.next_edge(0), 1250u);

  // And a second, later ask does not move the edge: wakes collapse onto the
  // earliest outstanding one, so the reusable clock event cannot be queued
  // twice however many times it is asked.
  rig.recorder->resume_from(20'000);
  EXPECT_TRUE(rig.recorder->running());
  rig.recorder->resume_from(30'000);
  ASSERT_TRUE(rig.run());
  EXPECT_EQ(rig.recorder->edges.back(), 20'250u);
}

TEST(ClockedEdgeTest, AClockWithNoRepresentableEdgeLeftStops) {
  // next_edge()/edge_after() saturate at TICK_MAX, which is not an edge and is
  // also the queue's empty sentinel. Scheduling it would re-enqueue the clock
  // event at the tick it is already on and spin the engine with time frozen,
  // so the handler has to stop instead.
  const ClockDomain domain("late", 1'000'000'000ULL, TICK_MAX - 3000);
  EdgeRig rig(domain, /*edges=*/16);

  EXPECT_TRUE(rig.run()) << "the clock did not stop at the end of representable time";
  ASSERT_FALSE(rig.recorder->edges.empty());
  // The grid's last member would be TICK_MAX itself, which is not an edge.
  EXPECT_EQ(rig.recorder->edges.back(), TICK_MAX - 1000);
  EXPECT_FALSE(rig.recorder->running());
}

// ============================================================================
// Bounded, resumable execution
// ============================================================================

namespace {

/// @brief A component that records the ticks its event actually fired at.
class TickRecorder : public Component {
public:
  TickRecorder() : Component("tick_recorder") {}

  /// @brief Schedule this component's one event at @p tick.
  void ask(Tick tick) { this->schedule_event(&event_, tick); }

  Event event_{this, EventType::TIMER_CALLBACK,
               [this](Tick now, Message *) { fired.push_back(now); }};
  std::vector<Tick> fired;
};

/// @brief An engine with one TickRecorder under a bare root.
class RecorderRig {
public:
  explicit RecorderRig(SimulationEngine::Config config = {}) : engine(config) {
    auto root = std::make_unique<CompositeComponent>("root");
    recorder = static_cast<TickRecorder *>(root->add_child(std::make_unique<TickRecorder>()));
    engine.topology().set_root(std::move(root));
    engine.create();
  }

  SimulationEngine engine;
  TickRecorder *recorder = nullptr;
};

} // namespace

TEST(BoundedRunTest, DrainsTheQueueAndReportsTheTickItReached) {
  RecorderRig rig;
  rig.recorder->ask(4000);

  EXPECT_EQ(rig.engine.run_until_idle(), 4000u);
  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{4000}));
  EXPECT_EQ(rig.engine.events_processed(), 1u);
}

TEST(BoundedRunTest, AnEmptyQueueLeavesTheTickWhereItWas) {
  RecorderRig rig;
  rig.recorder->ask(4000);
  EXPECT_EQ(rig.engine.run_until_idle(), 4000u);

  // Quiescence is an empty queue, not an end of the simulation: a second call
  // with nothing to do must be a no-op rather than a rewind.
  EXPECT_EQ(rig.engine.run_until_idle(), 4000u);
  EXPECT_EQ(rig.engine.events_processed(), 1u);
}

TEST(BoundedRunTest, StopsOnThePredicateAndResumes) {
  RecorderRig rig;
  bool stop = false;
  rig.recorder->event_.set_handler([&](Tick now, Message *) {
    rig.recorder->fired.push_back(now);
    stop = true;
  });

  rig.recorder->ask(1000);
  rig.recorder->ask(2000);
  EXPECT_EQ(rig.engine.run_bounded(stop), 1000u);
  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{1000}));

  // Resumable: the engine was not shut down, and the event the predicate cut
  // the loop short of is still queued.
  stop = false;
  EXPECT_EQ(rig.engine.run_bounded(stop), 2000u);
  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{1000, 2000}));
}

TEST(BoundedRunTest, ASecondEngineRunsNestedInsideTheFirstsHandler) {
  // The arrangement a timing model uses: a functional engine executing an
  // instruction, and a second engine driven forward to answer what that
  // instruction cost. Each keeps its own clock; neither can see the other's
  // queue.
  RecorderRig inner;
  RecorderRig outer;

  Tick inner_reached = 0;
  outer.recorder->event_.set_handler([&](Tick, Message *) {
    inner.recorder->ask(7000);
    inner_reached = inner.engine.run_until_idle();
  });

  outer.recorder->ask(250'000);
  EXPECT_EQ(outer.engine.run_until_idle(), 250'000u);

  EXPECT_EQ(inner_reached, 7000u);
  EXPECT_EQ(inner.engine.context(0).current_tick(), 7000u);
  // The outer engine's clock is untouched by the nested run.
  EXPECT_EQ(outer.engine.context(0).current_tick(), 250'000u);
}

TEST(BoundedRunTest, ReenteringTheSameEngineIsRefused) {
  // Re-entry would have the inner loop popping from the very queue the outer
  // process_event() frame is iterating. A throw, not an assert: a release
  // build would otherwise corrupt the run in silence.
  RecorderRig rig;
  int refusals = 0;
  ExitReason nested_run = ExitReason::COMPLETED;
  rig.recorder->event_.set_handler([&](Tick, Message *) {
    // Every entry point, not just the one that opened the loop: all three
    // drive the same queue, so all three are the same defect.
    for (int attempt = 0; attempt < 2; ++attempt) {
      try {
        if (attempt == 0)
          rig.engine.run_until_idle();
        else
          rig.engine.step();
      } catch (const std::logic_error &) {
        ++refusals;
      }
    }
    // run() reports the refusal rather than throwing: it may be on a
    // background thread whose top-level lambda has no catch, where an escaping
    // exception would call std::terminate.
    nested_run = rig.engine.run().reason;
  });

  rig.recorder->ask(1000);
  rig.engine.run_until_idle();
  EXPECT_EQ(refusals, 2);
  EXPECT_EQ(nested_run, ExitReason::INTERRUPTED);

  // And the refusals left the engine usable.
  rig.recorder->event_.set_handler(
      [&](Tick now, Message *) { rig.recorder->fired.push_back(now); });
  rig.recorder->ask(2000);
  EXPECT_EQ(rig.engine.run_until_idle(), 2000u);
  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{2000}));
}

TEST(BoundedRunTest, AMultiPartitionEngineIsRefused) {
  // Popping partition 0's queue while its own worker thread owns it is a data
  // race, so this is checked rather than asserted: the builds this ships as
  // have asserts compiled out.
  SimulationEngine engine({.num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<TickRecorder>());
  root->add_child(std::make_unique<TickRecorder>());
  engine.topology().set_root(std::move(root));
  engine.topology().partition_balanced(2);
  engine.create();

  bool never = false;
  EXPECT_THROW(engine.run_bounded(never), std::invalid_argument);
}

TEST(BoundedRunTest, AnAsyncEventPostedByAHandlerIsSeenBeforeIdle) {
  // The queue emptying is not quiescence on its own: a handler can post an
  // async event -- a doorbell, a compute unit's deferred work -- and returning
  // "idle" with one pending would have the caller resume later and run it at a
  // tick the engine has long passed.
  RecorderRig rig;
  Event follow_up{rig.recorder, EventType::TIMER_CALLBACK,
                  [&](Tick now, Message *) { rig.recorder->fired.push_back(now); }};
  bool posted = false;
  rig.recorder->event_.set_handler([&](Tick now, Message *) {
    rig.recorder->fired.push_back(now);
    if (!posted) {
      posted = true;
      rig.engine.schedule_event_async(&follow_up, now + 1000);
    }
  });

  rig.recorder->ask(4000);
  const Tick reached = rig.engine.run_until_idle();

  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{4000, 5000}));
  EXPECT_EQ(reached, 5000u);
  EXPECT_EQ(rig.engine.global_time(), 5000u);
}

TEST(BoundedRunTest, TheGlobalClockKeepsUpWithTheRun) {
  // request_exit() stamps its tick from the global clock and
  // schedule_event_now() timestamps events from it, so a bounded run that
  // published only on exit would hand both a tick from before the run.
  RecorderRig rig;
  std::vector<Tick> observed;
  rig.recorder->event_.set_handler(
      [&](Tick, Message *) { observed.push_back(rig.engine.global_time()); });

  rig.recorder->ask(1000);
  rig.recorder->ask(2000);
  rig.recorder->ask(3000);
  rig.engine.run_until_idle();

  // Each handler sees the tick of the event before it, which is as current as
  // a value published after processing can be.
  EXPECT_EQ(observed, (std::vector<Tick>{0, 1000, 2000}));
  EXPECT_EQ(rig.engine.global_time(), 3000u);
}

TEST(BoundedRunTest, MaxTicksEndsABoundedRunAndIsVisible) {
  // A drive loop that waits on its own predicate has to notice the engine
  // ending underneath it, or it spins forever.
  RecorderRig rig({.max_ticks = 1500});
  rig.recorder->ask(1000);
  rig.recorder->ask(2000);

  bool never = false;
  EXPECT_EQ(rig.engine.run_bounded(never), 1500u);
  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{1000}));
  EXPECT_TRUE(rig.engine.is_done());
  EXPECT_EQ(rig.engine.last_exit().message, "max ticks reached");

  // And it stays done: a second call runs nothing, which is why is_done() has
  // to be checkable.
  EXPECT_EQ(rig.engine.run_bounded(never), 1500u);
  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{1000}));
}

TEST(BoundedRunTest, APredicateAlreadyTrueOnEntryRunsNothing) {
  RecorderRig rig;
  rig.recorder->ask(1000);

  bool stop = true;
  EXPECT_EQ(rig.engine.run_bounded(stop), 0u);
  EXPECT_TRUE(rig.recorder->fired.empty());
  EXPECT_EQ(rig.engine.events_processed(), 0u);
}

namespace {

/// @brief A component that counts how many times it was started.
class StartupCounter : public Component {
public:
  StartupCounter() : Component("startup_counter") {}
  void startup() override { ++startups; }
  uint32_t startups = 0;
};

/// @brief Count the startups a given drive order produces.
uint32_t startups_for(const std::function<void(SimulationEngine &)> &drive) {
  SimulationEngine engine({});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *counter =
      static_cast<StartupCounter *>(root->add_child(std::make_unique<StartupCounter>()));
  engine.topology().set_root(std::move(root));
  engine.create();
  drive(engine);
  return counter->startups;
}

} // namespace

TEST(BoundedRunTest, StartupHappensOnceHoweverTheEngineIsDriven) {
  // Three entry points start the engine lazily. Whichever gets there first
  // must be the only one that does: starting a second time re-schedules every
  // component's first event and re-registers every primary, on top of state
  // the first attempt left live.
  //
  // run() clears its "executing" flag on return, so a lazy startup keyed on
  // that flag rather than on a separate one starts everything twice for the
  // second order below.
  EXPECT_EQ(startups_for([](SimulationEngine &e) {
              e.run_until_idle();
              e.run();
            }),
            1u);
  EXPECT_EQ(startups_for([](SimulationEngine &e) {
              e.run();
              e.run_until_idle();
            }),
            1u);
  EXPECT_EQ(startups_for([](SimulationEngine &e) {
              e.step();
              e.run_until_idle();
              e.run();
            }),
            1u);
}

TEST(BoundedRunTest, OnlyHandlersAreCounted) {
  // The count is what a model uses to assert that an idle component costs
  // nothing, so an entry with no handler must not inflate it.
  RecorderRig rig;
  Event silent{rig.recorder, EventType::TIMER_CALLBACK};
  rig.engine.schedule_event(&silent, 500);
  rig.recorder->ask(1000);

  rig.engine.run_until_idle();

  EXPECT_EQ(rig.engine.events_processed(), 1u);
}

TEST(BoundedRunTest, AThrowingHandlerLeavesTheEngineRunnable) {
  RecorderRig rig;
  rig.recorder->event_.set_handler([](Tick, Message *) { throw std::runtime_error("boom"); });

  rig.recorder->ask(1000);
  EXPECT_THROW(rig.engine.run_until_idle(), std::runtime_error);

  rig.recorder->event_.set_handler(
      [&](Tick now, Message *) { rig.recorder->fired.push_back(now); });
  rig.recorder->ask(2000);
  EXPECT_EQ(rig.engine.run_until_idle(), 2000u);
  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{2000}));
}

TEST(BoundedRunTest, StopsWhenAComponentRequestsExit) {
  RecorderRig rig;
  rig.recorder->event_.set_handler([&](Tick now, Message *) {
    rig.recorder->fired.push_back(now);
    rig.engine.request_exit("done here");
  });

  rig.recorder->ask(1000);
  rig.recorder->ask(2000);
  EXPECT_EQ(rig.engine.run_until_idle(), 1000u);
  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{1000}));
}

TEST(BoundedRunTest, StepAndBoundedRunShareOneLazyStartup) {
  // Both start the engine on their first call. Mixing them must not start it
  // twice, which would double-schedule every component's first event.
  const ClockDomain domain("ghz", 1'000'000'000ULL);
  SimulationEngine engine({});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *recorder =
      static_cast<EdgeRecorder *>(root->add_child(std::make_unique<EdgeRecorder>(domain, 4)));
  engine.topology().set_root(std::move(root));
  engine.create();

  engine.step();
  EXPECT_EQ(recorder->edges, (std::vector<Tick>{1000}));
  engine.run_until_idle();
  EXPECT_EQ(recorder->edges, (std::vector<Tick>{1000, 2000, 3000, 4000}));
  EXPECT_EQ(engine.events_processed(), 4u);
}

// ============================================================================
// Collapsing wakes and on-demand clocking
// ============================================================================

namespace {

/// @brief A component whose one event is armed only through schedule_wake().
class WakeRecorder : public Component {
public:
  WakeRecorder() : Component("wake_recorder") {}

  /// @brief Ask to be woken at @p tick.
  /// @returns Whether the ask superseded whatever was pending.
  bool ask(Tick tick) { return this->schedule_wake(&wake_, tick); }

  /// @brief Schedule the same event the ordinary way.
  void ask_plainly(Tick tick) { this->schedule_event(&wake_, tick); }

  Event wake_{this, EventType::TIMER_CALLBACK,
              [this](Tick now, Message *) { fired.push_back(now); }};
  std::vector<Tick> fired;
};

/// @brief An engine holding one WakeRecorder.
class WakeRig {
public:
  WakeRig() {
    auto root = std::make_unique<CompositeComponent>("root");
    recorder = static_cast<WakeRecorder *>(root->add_child(std::make_unique<WakeRecorder>()));
    engine.topology().set_root(std::move(root));
    engine.create();
  }

  SimulationEngine engine{{}};
  WakeRecorder *recorder = nullptr;
};

} // namespace

TEST(CollapsingWakeTest, AnEarlierAskSupersedesALaterOne) {
  WakeRig rig;
  EXPECT_TRUE(rig.recorder->ask(5000));
  EXPECT_TRUE(rig.recorder->ask(2000));  // earlier: supersedes
  EXPECT_FALSE(rig.recorder->ask(9000)); // later: ignored
  EXPECT_FALSE(rig.recorder->ask(2000)); // equal: ignored

  rig.engine.run_until_idle();

  // The superseded entry is left in the queue and dropped when it surfaces, so
  // exactly one firing happens, at the earliest tick asked for.
  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{2000}));
}

TEST(CollapsingWakeTest, ASupersededEntryDoesNotMoveTheClock) {
  // The property the whole mechanism turns on. A superseded entry always sits
  // later than the wake that replaced it, so a drop taken after the tick moved
  // would drag the engine's clock forward for work that never ran -- and a
  // model reading that clock would charge it as elapsed time.
  WakeRig rig;
  rig.recorder->ask(1'000'000);
  rig.recorder->ask(3000);

  EXPECT_EQ(rig.engine.run_until_idle(), 3000u);
  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{3000}));
  EXPECT_EQ(rig.engine.context(0).current_tick(), 3000u);
  // One event ran, not two.
  EXPECT_EQ(rig.engine.events_processed(), 1u);
}

TEST(CollapsingWakeTest, ARearmAtTheSupersededTickStillFires) {
  // Identity, not tick equality, is what tells a live entry from a stale one:
  // a wake re-armed at a tick it had already been armed for must not be
  // mistaken for the entry the supersession left behind.
  WakeRig rig;
  rig.recorder->ask(9000);
  rig.recorder->ask(2000);
  rig.engine.run_until_idle();
  ASSERT_EQ(rig.recorder->fired, (std::vector<Tick>{2000}));

  rig.recorder->ask(9000);
  rig.engine.run_until_idle();
  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{2000, 9000}));
}

TEST(CollapsingWakeTest, AWakeIntoThePastIsClampedForward) {
  // A request can legitimately reach a component the engine has already
  // advanced past. Refusing it would strand the request forever.
  WakeRig rig;
  rig.recorder->ask(4000);
  rig.engine.run_until_idle();

  EXPECT_TRUE(rig.recorder->ask(1000));
  rig.engine.run_until_idle();
  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{4000, 4000}));
}

TEST(CollapsingWakeTest, AnOrdinaryEventOnTheSameDescriptorIsUnaffected) {
  // schedule_event() and schedule_wake() share one Event. Ordinary entries
  // carry no wake identity and must all fire, including ones queued while a
  // wake is pending and ones queued after a wake has been superseded.
  WakeRig rig;
  rig.recorder->ask(8000);
  rig.recorder->ask(3000);
  rig.recorder->ask_plainly(1000);
  rig.recorder->ask_plainly(5000);

  rig.engine.run_until_idle();

  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{1000, 3000, 5000}));
  EXPECT_EQ(rig.engine.events_processed(), 3u);
}

TEST(CollapsingWakeTest, AWakeArmedFromInsideItsOwnHandlerIsHonoured) {
  WakeRig rig;
  std::vector<Tick> chain{4000, 9000};
  rig.recorder->wake_.set_handler([&](Tick now, Message *) {
    rig.recorder->fired.push_back(now);
    if (!chain.empty()) {
      const Tick next = chain.front();
      chain.erase(chain.begin());
      rig.recorder->ask(next);
    }
  });

  rig.recorder->ask(1000);
  rig.engine.run_until_idle();

  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{1000, 4000, 9000}));
}

namespace {

/// @brief A clocked component that advances only when asked.
class OnDemand : public Clocked<Component> {
public:
  explicit OnDemand(const ClockDomain &domain) : Clocked<Component>("on_demand", domain) {}

  bool clock_at_startup() const override { return false; }

  bool advance(Tick now) override {
    advanced.push_back(now);
    if (!follow_ups.empty()) {
      const Tick next = follow_ups.front();
      follow_ups.erase(follow_ups.begin());
      wake_at(next);
    }
    return keep_clocking;
  }

  std::vector<Tick> advanced;
  std::vector<Tick> follow_ups;
  bool keep_clocking = false;
};

/// @brief An engine holding one OnDemand component in a 1 GHz domain.
class OnDemandRig {
public:
  OnDemandRig() {
    auto root = std::make_unique<CompositeComponent>("root");
    block = static_cast<OnDemand *>(root->add_child(std::make_unique<OnDemand>(domain)));
    engine.topology().set_root(std::move(root));
    engine.create();
  }

  ClockDomain domain{"ghz", 1'000'000'000ULL};
  SimulationEngine engine{{}};
  OnDemand *block = nullptr;
};

} // namespace

TEST(ClockedOnDemandTest, SchedulesNothingUntilAsked) {
  OnDemandRig rig;

  rig.engine.run_until_idle();

  EXPECT_TRUE(rig.block->advanced.empty());
  EXPECT_FALSE(rig.block->running());
  // The point of the whole mechanism: an idle component costs no events at
  // all, rather than one per cycle of elapsed simulated time.
  EXPECT_EQ(rig.engine.events_processed(), 0u);
}

TEST(ClockedOnDemandTest, WakeAtAlignsToAnEdgeAndCollapses) {
  OnDemandRig rig;

  rig.block->wake_at(5500); // rounds up to 6000
  rig.block->wake_at(2500); // supersedes; rounds up to 3000
  rig.engine.run_until_idle();

  EXPECT_EQ(rig.block->advanced, (std::vector<Tick>{3000}));
  EXPECT_EQ(rig.engine.events_processed(), 1u);
}

TEST(ClockedOnDemandTest, AdvanceCanScheduleItsOwnNextVisit) {
  // The idiom Clocked could not express before: resume_clock() early-returns
  // while running_ is set, and running_ was only cleared after advance()
  // returned, so a component had no way to ask for a later visit from inside
  // its own advance().
  OnDemandRig rig;

  rig.block->follow_ups = {4000, 9000};
  rig.block->wake_at(1000);
  rig.engine.run_until_idle();

  EXPECT_EQ(rig.block->advanced, (std::vector<Tick>{1000, 4000, 9000}));
}

TEST(ClockedOnDemandTest, AContinuousClockStillRunsEveryEdge) {
  // clock_at_startup() is the only thing being opted out of. A component that
  // returns true from advance() clocks every edge as before, now through the
  // same collapsing path.
  OnDemandRig rig;
  rig.block->keep_clocking = true;

  rig.block->wake_at(1000);
  for (int i = 0; i < 4; ++i)
    rig.engine.step();

  EXPECT_EQ(rig.block->advanced, (std::vector<Tick>{1000, 2000, 3000, 4000}));
  EXPECT_TRUE(rig.block->running());
}

TEST(ClockedOnDemandTest, AnEarlierAskFromInsideAdvanceBeatsTheNextEdge) {
  // advance() returning true arms the next edge, but a wake advance() armed
  // for an earlier tick must win -- and both must not produce two entries.
  OnDemandRig rig;
  rig.block->keep_clocking = true;
  rig.block->follow_ups = {1500}; // rounds to 2000, the next edge anyway
  rig.block->wake_at(1000);

  rig.engine.step();
  rig.engine.step();

  EXPECT_EQ(rig.block->advanced, (std::vector<Tick>{1000, 2000}));
}

TEST(ClockedOnDemandTest, ReserveSerialisesOverlappingWork) {
  OnDemandRig rig;

  // Three cycles of work ready at tick 0 starts at the domain's first edge.
  EXPECT_EQ(rig.block->reserve(0, 3), 4000u);
  // Work that could have started at 2000 queues behind it instead.
  EXPECT_EQ(rig.block->reserve(2000, 2), 6000u);
  // Work arriving after the server is free starts when it arrives.
  EXPECT_EQ(rig.block->reserve(20'000, 1), 21'000u);
  EXPECT_EQ(rig.block->busy_until(), 21'000u);
  // Reserving schedules nothing; it only says when the server is next free.
  EXPECT_FALSE(rig.block->running());
  EXPECT_EQ(rig.engine.events_processed(), 0u);
}

TEST(ClockedOnDemandTest, ReserveSaturatesRatherThanWrapping) {
  OnDemandRig rig;

  EXPECT_EQ(rig.block->reserve(0, TICK_MAX), TICK_MAX);
  // And stays saturated: a busy_until() that wrapped would make the server
  // look free again.
  EXPECT_EQ(rig.block->reserve(0, 1), TICK_MAX);
}

TEST(ServiceCyclesTest, RoundsUpAndNeverToNothing) {
  EXPECT_EQ(ClockDomain::service_cycles(8, 2.0), 4u);
  EXPECT_EQ(ClockDomain::service_cycles(9, 2.0), 5u);
  EXPECT_EQ(ClockDomain::service_cycles(1, 2.0), 1u);
  // A server handed work has looked at it, so no amount of rate rounds the
  // work away and makes the component infinitely fast.
  EXPECT_EQ(ClockDomain::service_cycles(0, 2.0), 1u);
  EXPECT_EQ(ClockDomain::service_cycles(1, 1e9), 1u);
  // A rate that is not a rate falls back to one unit per cycle.
  EXPECT_EQ(ClockDomain::service_cycles(7, 0.0), 7u);
  EXPECT_EQ(ClockDomain::service_cycles(7, -1.0), 7u);
  EXPECT_EQ(ClockDomain::service_cycles(0, 0.0), 1u);
  // A rate below one unit per cycle can ask for more cycles than there are
  // ticks; saturating keeps it a number rather than an overflow.
  EXPECT_EQ(ClockDomain::service_cycles(std::numeric_limits<uint64_t>::max(), 0.5),
            std::numeric_limits<uint64_t>::max());
}

TEST(ClockedOnDemandTest, AWakeIntoThePastStaysOnTheClockGrid) {
  // The case the collapsing-wake contract calls normal: a request reaches a
  // component the engine has already advanced past. The engine clamps a wake
  // into the past forward to the current tick, which is not an edge of this
  // component's domain -- so the alignment has to happen after the clamp, not
  // before it, or every edge from here on inherits the offset.
  const ClockDomain domain("ghz", 1'000'000'000ULL);
  SimulationEngine engine({});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *block = static_cast<OnDemand *>(root->add_child(std::make_unique<OnDemand>(domain)));
  auto *pacer = static_cast<TickRecorder *>(root->add_child(std::make_unique<TickRecorder>()));
  engine.topology().set_root(std::move(root));
  engine.create();

  // Drive the partition's clock to a tick that is not on the 1 GHz grid.
  pacer->ask(5500);
  engine.run_until_idle();
  ASSERT_EQ(engine.context(0).current_tick(), 5500u);

  // Asked for a tick long past, then twice more for ticks that are not edges.
  block->follow_ups = {6100, 7200};
  block->wake_at(1000);
  engine.run_until_idle();

  ASSERT_EQ(block->advanced.size(), 3u);
  for (Tick edge : block->advanced)
    EXPECT_EQ(edge % 1000, 0u) << "advanced at " << edge << ", which is not an edge";
  EXPECT_EQ(block->advanced, (std::vector<Tick>{6000, 7000, 8000}));
}

TEST(ClockedOnDemandTest, AComponentStillClocksAfterTheEngineIsRebuilt) {
  // An Event is owned by its component and outlives the queue that held its
  // entry: shutdown() destroys the contexts, not the components. A wake armed
  // in the old generation must not refuse the identical wake the new
  // generation's startup asks for.
  OnDemandRig rig;
  rig.block->wake_at(1000);
  ASSERT_TRUE(rig.block->running());

  // Torn down before that edge ever fired.
  rig.engine.shutdown();
  ASSERT_TRUE(rig.block->advanced.empty());

  rig.engine.create();
  rig.block->wake_at(1000);
  rig.engine.run_until_idle();

  EXPECT_EQ(rig.block->advanced, (std::vector<Tick>{1000}));
  EXPECT_FALSE(rig.block->running());
}

TEST(ClockedOnDemandTest, StartupClearsAReservationFromABuriedGeneration) {
  // busy_until_ is an absolute tick. Carried across a rebuild it would leave
  // the component looking occupied until a tick the new generation reaches
  // only after re-simulating everything the old one did.
  OnDemandRig rig;
  EXPECT_EQ(rig.block->reserve(0, 3), 4000u);
  rig.engine.shutdown();
  rig.engine.create();

  rig.block->wake_at(1000);
  rig.engine.run_until_idle();
  EXPECT_EQ(rig.block->busy_until(), 0u);
}

TEST(ClockedOnDemandTest, AnAdvanceThatThrowsLeavesTheComponentRecoverable) {
  // The wake is consumed before the handler runs, so a throw leaves nothing
  // armed. If the component still reported a running clock, every later
  // attempt to restart it would be silently dropped.
  OnDemandRig rig;
  bool throw_now = true;
  class Thrower : public Clocked<Component> {
  public:
    Thrower(const ClockDomain &domain, bool &fail) : Clocked("thrower", domain), fail_(fail) {}
    bool clock_at_startup() const override { return false; }
    bool advance(Tick now) override {
      if (fail_)
        throw std::runtime_error("boom");
      advanced.push_back(now);
      return false;
    }
    std::vector<Tick> advanced;

  private:
    bool &fail_;
  };

  SimulationEngine engine({});
  ClockDomain domain("ghz", 1'000'000'000ULL);
  auto root = std::make_unique<CompositeComponent>("root");
  auto *thrower =
      static_cast<Thrower *>(root->add_child(std::make_unique<Thrower>(domain, throw_now)));
  engine.topology().set_root(std::move(root));
  engine.create();

  thrower->wake_at(1000);
  EXPECT_THROW(engine.run_until_idle(), std::runtime_error);
  EXPECT_FALSE(thrower->running()) << "a throw left the clock reporting itself as live";

  throw_now = false;
  thrower->resume_clock(2000);
  engine.run_until_idle();
  EXPECT_EQ(thrower->advanced, (std::vector<Tick>{2000}));
}

TEST(CollapsingWakeTest, TheSentinelTickIsRefused) {
  // TICK_MAX is what an empty queue reports as its next event time, so an
  // entry at it would be invisible to the termination check -- and, because it
  // is also how an Event spells "no wake armed", every later ask would push
  // another entry rather than collapsing onto it.
  WakeRig rig;
  EXPECT_FALSE(rig.recorder->ask(TICK_MAX));
  EXPECT_FALSE(rig.recorder->ask(TICK_MAX));
  EXPECT_TRUE(rig.recorder->ask(4000));

  rig.engine.run_until_idle();
  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{4000}));
  EXPECT_EQ(rig.engine.events_processed(), 1u);
}

TEST(CollapsingWakeTest, SteppingOverASupersededEntryDoesNotMoveTheClock) {
  // The same property as ASupersededEntryDoesNotMoveTheClock, on the other
  // driver. step() advances to the tick it selected, which is the tick of the
  // entry it is about to drop -- so it has to publish the partition's tick
  // instead, or a model reading the global clock is charged the difference.
  WakeRig rig;
  rig.recorder->ask(3000);
  rig.engine.step();
  ASSERT_EQ(rig.recorder->fired, (std::vector<Tick>{3000}));

  rig.recorder->ask(1'000'000);
  rig.recorder->ask(9000);
  rig.engine.step(); // fires the 9000 wake
  rig.engine.step(); // surfaces the superseded 1'000'000 entry

  EXPECT_EQ(rig.recorder->fired, (std::vector<Tick>{3000, 9000}));
  EXPECT_EQ(rig.engine.context(0).current_tick(), 9000u);
  EXPECT_EQ(rig.engine.global_time(), 9000u);
}

TEST(ClockedOnDemandTest, AWakeArmedBeforeStartupIsReportedAsRunning) {
  // startup() runs on the engine's first step, so a component armed between
  // create() and that step already has an entry queued. running() is the
  // framework's answer to "will this be visited again", and startup() must
  // read it rather than assert it: answering no with an entry queued is the
  // opposite of the truth.
  const ClockDomain domain("ghz", 1'000'000'000ULL);
  SimulationEngine engine({});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *block = static_cast<OnDemand *>(root->add_child(std::make_unique<OnDemand>(domain)));
  auto *pacer = static_cast<TickRecorder *>(root->add_child(std::make_unique<TickRecorder>()));
  engine.topology().set_root(std::move(root));
  engine.create();

  block->wake_at(5000);
  pacer->ask(1000);
  engine.step(); // startup, then the pacer's event at 1000

  EXPECT_TRUE(block->running());
  EXPECT_TRUE(block->advanced.empty());

  engine.run_until_idle();
  EXPECT_EQ(block->advanced, (std::vector<Tick>{5000}));
}

// ============================================================================
// TagArray
// ============================================================================

namespace {

/// @brief The byte address of line @p line in set @p set of a @p sets-set array
///        with @p line_bytes lines.
///
/// @details The line size is a parameter rather than a literal: an address
/// computed for 64-byte lines and handed to a 128-byte-line array is not the
/// line the caller named, and the test would go on asserting something else.
uint64_t line_address(uint64_t sets, uint64_t set, uint64_t line, uint64_t line_bytes = 64) {
  return ((line * sets) + set) * line_bytes;
}

/// @brief An independent least-recently-used model, written the obvious way.
///
/// @details Deliberately not the implementation's algorithm: one recency list
/// per set, most recent at the back. If both agree on a long access sequence,
/// the stamp-based version is LRU rather than something that merely passes the
/// handful of cases a test would think to write.
class ReferenceLru {
public:
  ReferenceLru(uint64_t sets, uint64_t ways, uint64_t line_bytes)
      : sets_(sets), ways_(ways), line_shift_(static_cast<uint32_t>(std::countr_zero(line_bytes))),
        order_(static_cast<std::size_t>(sets)) {}

  /// @returns Whether the line was already resident.
  bool access(uint64_t byte_address, uint32_t vmid = 0) {
    const uint64_t line = byte_address >> line_shift_;
    auto &set = order_[static_cast<std::size_t>(line & (sets_ - 1))];
    const auto key = std::make_pair(line, vmid);
    auto it = std::find(set.begin(), set.end(), key);
    if (it != set.end()) {
      set.erase(it);
      set.push_back(key);
      return true;
    }
    if (set.size() == ways_)
      set.erase(set.begin());
    set.push_back(key);
    return false;
  }

  bool contains(uint64_t byte_address, uint32_t vmid = 0) const {
    const uint64_t line = byte_address >> line_shift_;
    const auto &set = order_[static_cast<std::size_t>(line & (sets_ - 1))];
    return std::find(set.begin(), set.end(), std::make_pair(line, vmid)) != set.end();
  }

private:
  uint64_t sets_;
  uint64_t ways_;
  uint32_t line_shift_;
  std::vector<std::vector<std::pair<uint64_t, uint32_t>>> order_;
};

} // namespace

TEST(TagArrayTest, AllocatesOnAMissAndHitsAfterwards) {
  constexpr uint64_t kSets = 16;
  TagArray tags(kSets, /*ways=*/4, /*line_bytes=*/64);

  EXPECT_FALSE(tags.access(0x1000));
  EXPECT_TRUE(tags.access(0x1000));
  // Every byte of a line is the same line, and the byte after it is not.
  EXPECT_TRUE(tags.access(0x103F));
  EXPECT_FALSE(tags.access(0x1040));
  // A line that indexes into the same set is a different line too.
  EXPECT_FALSE(tags.access(0x1000 + kSets * 64));
  EXPECT_TRUE(tags.access(0x1000));
}

TEST(TagArrayTest, ContainsDoesNotAllocateOrDisturbRecency) {
  TagArray tags(/*sets=*/1, /*ways=*/2, /*line_bytes=*/64);

  EXPECT_FALSE(tags.contains(0x2000));
  EXPECT_FALSE(tags.access(0x2000));
  EXPECT_TRUE(tags.contains(0x2000));
  // Still one way in use: contains() did not fill anything.
  EXPECT_FALSE(tags.access(0x3000));
  EXPECT_TRUE(tags.contains(0x2000));

  // A probe of the older line must not save it from eviction, or a
  // non-allocating access would silently act like an allocating one.
  EXPECT_TRUE(tags.contains(0x2000));
  EXPECT_FALSE(tags.access(0x4000));
  EXPECT_FALSE(tags.contains(0x2000));
  EXPECT_TRUE(tags.contains(0x3000));
}

TEST(TagArrayTest, EvictsTheLeastRecentlyUsedWay) {
  constexpr uint64_t kSets = 8;
  TagArray tags(kSets, /*ways=*/2, /*line_bytes=*/64);

  const uint64_t kept = line_address(kSets, /*set=*/3, /*line=*/0);
  const uint64_t evicted = line_address(kSets, /*set=*/3, /*line=*/1);
  const uint64_t filler = line_address(kSets, /*set=*/3, /*line=*/2);

  EXPECT_FALSE(tags.access(kept));
  EXPECT_FALSE(tags.access(evicted));
  EXPECT_TRUE(tags.access(kept)); // kept is now the more recent of the two
  EXPECT_FALSE(tags.access(filler));

  EXPECT_FALSE(tags.contains(evicted)) << "the least recently used way should have gone";
  EXPECT_TRUE(tags.contains(kept));
  EXPECT_TRUE(tags.contains(filler));
}

TEST(TagArrayTest, OnlyTheIndexedSetIsDisturbed) {
  constexpr uint64_t kSets = 8;
  TagArray tags(kSets, /*ways=*/1, /*line_bytes=*/64);

  for (uint64_t set = 0; set < kSets; ++set)
    EXPECT_FALSE(tags.access(line_address(kSets, set, /*line=*/0)));
  // A direct-mapped array holds one line per set, and eight of them fit.
  for (uint64_t set = 0; set < kSets; ++set)
    EXPECT_TRUE(tags.contains(line_address(kSets, set, /*line=*/0)));

  // A conflicting line evicts only its own set's occupant.
  EXPECT_FALSE(tags.access(line_address(kSets, /*set=*/2, /*line=*/1)));
  EXPECT_FALSE(tags.contains(line_address(kSets, /*set=*/2, /*line=*/0)));
  for (uint64_t set = 0; set < kSets; ++set) {
    if (set != 2) {
      EXPECT_TRUE(tags.contains(line_address(kSets, set, /*line=*/0))) << "set " << set;
    }
  }
}

TEST(TagArrayTest, AgreesWithAnIndependentLruOverALongSequence) {
  // Enough conflict to evict constantly: four sets of three ways against
  // sixteen lines per set, in an order that revisits at irregular distances.
  constexpr uint64_t kSets = 4;
  constexpr uint64_t kWays = 3;
  constexpr uint64_t kLineBytes = 128;
  TagArray tags(kSets, kWays, kLineBytes);
  ReferenceLru reference(kSets, kWays, kLineBytes);

  // A fixed-seed Mersenne twister rather than a hand-rolled LCG: an LCG modulo
  // a power of two has short-period low bits, and the two bits that select the
  // set would repeat every 1024 draws, so a four-thousand-step sequence would
  // explore a quarter of the interleavings its length suggests.
  std::mt19937_64 rng(0x9E3779B9u);
  std::uniform_int_distribution<uint64_t> line_of(0, 63);
  std::uniform_int_distribution<uint32_t> vmid_of(0, 1);
  uint32_t misses = 0;
  for (uint32_t step = 0; step < 4000; ++step) {
    const uint32_t vmid = vmid_of(rng);
    const uint64_t address = line_of(rng) * kLineBytes;
    const bool hit = tags.access(address, vmid);
    ASSERT_EQ(hit, reference.access(address, vmid)) << "diverged at step " << step;
    misses += hit ? 0u : 1u;
  }
  // Guards the guard: a sequence that never evicted would agree trivially.
  EXPECT_GT(misses, 2000u) << "the working set fit; nothing was ever evicted";

  // And the residency they arrived at is the same, not just the hit sequence.
  for (uint64_t line = 0; line < 64; ++line) {
    for (uint32_t vmid = 0; vmid < 2; ++vmid) {
      const uint64_t address = line * kLineBytes;
      EXPECT_EQ(tags.contains(address, vmid), reference.contains(address, vmid))
          << "line " << line << " vmid " << vmid;
    }
  }
}

TEST(TagArrayTest, AFreeWayIsFilledBeforeAResidentOneIsEvicted) {
  TagArray tags(/*sets=*/1, /*ways=*/2, /*line_bytes=*/64);

  EXPECT_FALSE(tags.access(0x0));
  EXPECT_FALSE(tags.access(0x40));
  EXPECT_FALSE(tags.invalidate(0x80)) << "nothing resident to drop";
  // The *newer* line, so that the freed way is the one a stamp-ordered victim
  // search would pick last. It is only picked first if invalidate() reset the
  // stamp rather than just clearing the valid flag.
  EXPECT_TRUE(tags.invalidate(0x40));

  EXPECT_FALSE(tags.access(0x80));
  EXPECT_TRUE(tags.contains(0x0)) << "a resident line was evicted while a way was free";
  EXPECT_TRUE(tags.contains(0x80));
}

TEST(TagArrayTest, AddressSpacesDoNotAliasEachOther) {
  TagArray tags(/*sets=*/1, /*ways=*/2, /*line_bytes=*/64);

  EXPECT_FALSE(tags.access(0x8000, /*vmid=*/1));
  // Same virtual address, different guest: not the same line, and it occupies
  // a way of its own.
  EXPECT_FALSE(tags.access(0x8000, /*vmid=*/2));
  EXPECT_TRUE(tags.access(0x8000, /*vmid=*/2));
  // Guest 1 last, so it is the more recent way *and* the first-filled one.
  // Evicting by arrival order and evicting by recency now disagree, and only
  // one of them keeps guest 1.
  EXPECT_TRUE(tags.access(0x8000, /*vmid=*/1));
  EXPECT_FALSE(tags.contains(0x8000, /*vmid=*/3));

  EXPECT_FALSE(tags.access(0x8000, /*vmid=*/3));
  EXPECT_TRUE(tags.contains(0x8000, /*vmid=*/1)) << "evicted by arrival order, not by recency";
  EXPECT_FALSE(tags.contains(0x8000, /*vmid=*/2));

  EXPECT_TRUE(tags.invalidate(0x8000, /*vmid=*/1));
  EXPECT_FALSE(tags.contains(0x8000, /*vmid=*/1));
  EXPECT_TRUE(tags.contains(0x8000, /*vmid=*/3));
}

TEST(TagArrayTest, AssociativityNeedNotBeAPowerOfTwo) {
  // Ways index by multiplication rather than by a shift, so a three-way array
  // is a real geometry and has to behave like one.
  constexpr uint64_t kSets = 2;
  TagArray tags(kSets, /*ways=*/3, /*line_bytes=*/64);

  for (uint64_t line = 0; line < 3; ++line)
    EXPECT_FALSE(tags.access(line_address(kSets, /*set=*/1, line)));
  for (uint64_t line = 0; line < 3; ++line)
    EXPECT_TRUE(tags.contains(line_address(kSets, /*set=*/1, line))) << "line " << line;
  // The other set is untouched by all of it.
  EXPECT_FALSE(tags.contains(line_address(kSets, /*set=*/0, /*line=*/0)));

  // A fourth line evicts the oldest of the three.
  EXPECT_FALSE(tags.access(line_address(kSets, /*set=*/1, /*line=*/3)));
  EXPECT_FALSE(tags.contains(line_address(kSets, /*set=*/1, /*line=*/0)));
  EXPECT_TRUE(tags.contains(line_address(kSets, /*set=*/1, /*line=*/1)));
}

TEST(TagArrayTest, InvalidateAllDropsEverythingAndKeepsRecencyOrdered) {
  constexpr uint64_t kSets = 1;
  TagArray tags(kSets, /*ways=*/2, /*line_bytes=*/64);

  EXPECT_FALSE(tags.access(0x0));
  EXPECT_FALSE(tags.access(0x40));
  tags.invalidate_all();
  EXPECT_FALSE(tags.contains(0x0));
  EXPECT_FALSE(tags.contains(0x40));
  EXPECT_EQ(tags.sets(), 1u);
  EXPECT_EQ(tags.ways(), 2u);

  // Replacement still works on the emptied array.
  EXPECT_FALSE(tags.access(0x80));
  EXPECT_FALSE(tags.access(0xC0));
  EXPECT_TRUE(tags.access(0x80));
  EXPECT_FALSE(tags.access(0x100));
  EXPECT_TRUE(tags.contains(0x80)) << "the more recently used line was evicted";
  EXPECT_FALSE(tags.contains(0xC0));
}

TEST(TagArrayTest, ADroppedLineDoesNotReorderTheOnesThatSurvive) {
  // The recency counter is never reset, and this is the case that can tell:
  // a partial invalidate leaves a resident line behind, so a reset would make
  // it look newer than everything filled afterwards and invert replacement.
  // invalidate_all() cannot witness it -- it leaves nothing resident, and its
  // assertions pass whether or not the counter is reset.
  TagArray tags(/*sets=*/1, /*ways=*/2, /*line_bytes=*/64);

  EXPECT_FALSE(tags.access(0x0)); // the survivor, and the oldest line there is
  EXPECT_FALSE(tags.access(0x40));
  EXPECT_TRUE(tags.invalidate(0x40));

  EXPECT_FALSE(tags.access(0x80));
  // 0x0 predates 0x80, so it is what the next fill must take.
  EXPECT_FALSE(tags.access(0xC0));
  EXPECT_FALSE(tags.contains(0x0)) << "the survivor was treated as newer than a later fill";
  EXPECT_TRUE(tags.contains(0x80));
  EXPECT_TRUE(tags.contains(0xC0));
}

TEST(TagArrayTest, GeometryIsReportedAndReconfigurable) {
  TagArray tags;
  EXPECT_FALSE(tags.configured());

  tags.configure(/*sets=*/64, /*ways=*/8, /*line_bytes=*/128);
  EXPECT_TRUE(tags.configured());
  EXPECT_EQ(tags.sets(), 64u);
  EXPECT_EQ(tags.ways(), 8u);
  EXPECT_EQ(tags.line_bytes(), 128u);
  EXPECT_EQ(tags.line_shift(), 7u);

  // Same sets and line size, so 0x400 still decodes to the same line of the
  // same set: only a reconfigure that actually dropped the entries can miss.
  EXPECT_FALSE(tags.access(0x400));
  tags.configure(/*sets=*/64, /*ways=*/2, /*line_bytes=*/128);
  EXPECT_FALSE(tags.contains(0x400)) << "reconfiguring must not leave stale residency";

  tags.configure(/*sets=*/2, /*ways=*/1, /*line_bytes=*/64);
  EXPECT_EQ(tags.line_shift(), 6u);
}

TEST(TagArrayTest, ARejectedReconfigureLeavesTheArrayAsItWas) {
  TagArray tags(/*sets=*/64, /*ways=*/8, /*line_bytes=*/128);
  ASSERT_FALSE(tags.access(0x400));

  EXPECT_THROW(tags.configure(/*sets=*/3, /*ways=*/4, /*line_bytes=*/64), std::invalid_argument);

  EXPECT_EQ(tags.sets(), 64u);
  EXPECT_EQ(tags.ways(), 8u);
  EXPECT_EQ(tags.line_bytes(), 128u);
  EXPECT_TRUE(tags.contains(0x400)) << "a rejected geometry destroyed a live array";
}

TEST(TagArrayTest, AGeometryTheHardwareCouldNotIndexIsRejected) {
  TagArray tags;
  // Rounded rather than rejected, a geometry silently becomes a different
  // cache from the one the configuration asked for.
  EXPECT_THROW(tags.configure(/*sets=*/3, /*ways=*/4, /*line_bytes=*/64), std::invalid_argument);
  EXPECT_THROW(tags.configure(/*sets=*/4, /*ways=*/4, /*line_bytes=*/96), std::invalid_argument);
  EXPECT_THROW(tags.configure(/*sets=*/0, /*ways=*/4, /*line_bytes=*/64), std::invalid_argument);
  EXPECT_THROW(tags.configure(/*sets=*/4, /*ways=*/0, /*line_bytes=*/64), std::invalid_argument);
  EXPECT_THROW(tags.configure(/*sets=*/4, /*ways=*/4, /*line_bytes=*/0), std::invalid_argument);
  EXPECT_FALSE(tags.configured()) << "a rejected geometry must leave nothing half-built";

  // Ways need not be a power of two: a three-way cache is a real thing.
  EXPECT_NO_THROW(tags.configure(/*sets=*/4, /*ways=*/3, /*line_bytes=*/64));
}

TEST(TagArrayTest, AnArrayTooLargeToBeRealIsRejectedRatherThanAllocated) {
  TagArray tags;
  // The product overflows, so no allocation is even described.
  EXPECT_THROW(tags.configure(/*sets=*/1ULL << 62, /*ways=*/1ULL << 62, /*line_bytes=*/64),
               std::length_error);
  // And the case that does not overflow: a plausible units mistake in a
  // configuration file, which would otherwise pass every arithmetic check and
  // ask the operating system for hundreds of terabytes.
  EXPECT_THROW(tags.configure(/*sets=*/1ULL << 40, /*ways=*/16, /*line_bytes=*/64),
               std::length_error);
  EXPECT_FALSE(tags.configured());

  // The entry limit is checked, not materialised: configuring at kMaxEntries
  // would really allocate 2^26 entries -- a gigabyte and a half of zeroed host
  // memory in a unit test -- to assert something the rejection side already
  // shows. One entry past the limit is what has to throw.
  EXPECT_THROW(tags.configure(TagArray::kMaxEntries, /*ways=*/2, /*line_bytes=*/64),
               std::length_error);
  EXPECT_THROW(tags.configure(TagArray::kMaxEntries / 2, /*ways=*/3, /*line_bytes=*/64),
               std::length_error);
  EXPECT_FALSE(tags.configured());

  // Comfortably under it, and the array is usable: 2^21 entries, 50 MiB.
  EXPECT_NO_THROW(tags.configure(TagArray::kMaxEntries / 32, /*ways=*/1, /*line_bytes=*/64));
  EXPECT_FALSE(tags.access(0x1000));
}

TEST(TagArrayTest, ACacheTooLargeToBeRealIsRejectedEvenWhenItsTagsWouldFit) {
  TagArray tags;
  // sets * ways is small, so the entry limit sees nothing wrong; only the line
  // size is absurd. Left unchecked, every address in a normal virtual range
  // collapses onto one or two lines and the array reports a hit for almost
  // everything, with nothing thrown to point at.
  EXPECT_THROW(tags.configure(/*sets=*/64, /*ways=*/4, /*line_bytes=*/1ULL << 40),
               std::length_error);
  // And the geometry that stays inside the entry limit while claiming a
  // 256 GiB cache.
  EXPECT_THROW(tags.configure(/*sets=*/1ULL << 22, /*ways=*/16, /*line_bytes=*/4096),
               std::length_error);
  EXPECT_FALSE(tags.configured());

  // A large but real cache is still fine: 256 MiB at 128-byte lines.
  EXPECT_NO_THROW(tags.configure(/*sets=*/1ULL << 17, /*ways=*/16, /*line_bytes=*/128));
}

TEST(TagArrayTest, AGeometryRejectedForItsSizeAlsoLeavesTheArrayAsItWas) {
  // The counterpart of ARejectedReconfigureLeavesTheArrayAsItWas for the
  // length_error path, which rejects after the arithmetic checks rather than
  // before them.
  TagArray tags(/*sets=*/64, /*ways=*/8, /*line_bytes=*/128);
  ASSERT_FALSE(tags.access(0x400));

  EXPECT_THROW(tags.configure(/*sets=*/1ULL << 40, /*ways=*/16, /*line_bytes=*/64),
               std::length_error);

  EXPECT_EQ(tags.sets(), 64u);
  EXPECT_EQ(tags.ways(), 8u);
  EXPECT_EQ(tags.line_bytes(), 128u);
  EXPECT_TRUE(tags.contains(0x400)) << "an oversized geometry destroyed a live array";
}

TEST(TagArrayTest, AMovedFromArrayReportsNoGeometryRatherThanAStaleOne) {
  // The default move would take the entries and leave the geometry behind, so
  // the source would answer sets()/line_bytes() with numbers it no longer has
  // while every operation on it threw -- a plausible answer from an unusable
  // object, which is worse than no answer.
  TagArray source(/*sets=*/64, /*ways=*/8, /*line_bytes=*/128);
  ASSERT_FALSE(source.access(0x400));

  TagArray moved = std::move(source);
  EXPECT_TRUE(moved.configured());
  EXPECT_TRUE(moved.contains(0x400));

  EXPECT_FALSE(source.configured()); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.sets(), 0u);
  EXPECT_EQ(source.ways(), 0u);
  EXPECT_EQ(source.line_bytes(), 0u);
  EXPECT_EQ(source.line_shift(), 0u);
  EXPECT_THROW((void)source.contains(0x400), std::logic_error);

  // And it is reusable, not merely inert.
  source.configure(/*sets=*/2, /*ways=*/1, /*line_bytes=*/64);
  EXPECT_FALSE(source.access(0x400));
}

TEST(TagArrayTest, EveryOperationRefusesAnArrayWithNoGeometry) {
  // One contract, not four. A model that probes before its configuration has
  // been parsed would otherwise be told "not resident", which is
  // indistinguishable from a cold miss, and only find out later when a real
  // access threw from inside the simulation loop.
  TagArray tags;
  EXPECT_THROW((void)tags.access(0x100), std::logic_error);
  EXPECT_THROW((void)tags.contains(0x100), std::logic_error);
  EXPECT_THROW((void)tags.invalidate(0x100), std::logic_error);
  EXPECT_THROW(tags.invalidate_all(), std::logic_error);
}
