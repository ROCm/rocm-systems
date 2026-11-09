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

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <time.h>
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

/// Leaf kept alive until an external producer finishes posting events.
class AsyncPrimaryComponent : public Component {
public:
  explicit AsyncPrimaryComponent(std::string name) : Component(std::move(name)) {}
  void startup() override { engine()->register_as_primary(); }
};

/// Active primary that performs one real event but deliberately remains registered.
class ScheduledPrimaryComponent : public Component {
public:
  ScheduledPrimaryComponent(std::string name, Tick event_tick)
      : Component(std::move(name)), event_tick_(event_tick) {
    timer_event_.set_handler([this](Tick tick, Message *) {
      handled = true;
      handled_tick = tick;
    });
  }

  void startup() override {
    engine()->register_as_primary();
    schedule_event(&timer_event_, event_tick_);
  }

  bool handled = false;
  Tick handled_tick = 0;

private:
  Tick event_tick_;
  Event timer_event_{this, EventType::TIMER_CALLBACK};
};

/// Primary that releases itself after processing one scheduled event.
class ReleasingPrimaryComponent : public Component {
public:
  ReleasingPrimaryComponent(std::string name, Tick event_tick)
      : Component(std::move(name)), event_tick_(event_tick) {
    timer_event_.set_handler([this](Tick tick, Message *) {
      handled_tick = tick;
      engine()->primary_release();
    });
  }

  void startup() override {
    engine()->register_as_primary();
    schedule_event(&timer_event_, event_tick_);
  }

  Tick handled_tick = 0;

private:
  Tick event_tick_;
  Event timer_event_{this, EventType::TIMER_CALLBACK};
};

/// Primary that posts async work and optionally releases after that work runs.
class AsyncPostingPrimaryComponent : public Component {
public:
  AsyncPostingPrimaryComponent(std::string name, Tick callback_tick, Tick late_tick,
                               bool release_after_async = false)
      : Component(std::move(name)), late_tick_(late_tick) {
    post_.set_handler([this, callback_tick, release_after_async](Tick tick, Message *) {
      engine()->schedule_event_async(&async_, callback_tick);
      if (!release_after_async)
        release(tick);
    });
    async_.set_handler([this, release_after_async](Tick tick, Message *) {
      async_tick = tick;
      if (release_after_async)
        schedule_event(&release_, tick + kReleaseDelayTicks);
    });
    release_.set_handler([this](Tick tick, Message *) { release(tick); });
    late_.set_handler([this](Tick, Message *) { late_ran = true; });
  }

  void startup() override {
    engine()->register_as_primary();
    schedule_event(&post_, 10);
    schedule_event(&late_, late_tick_);
  }

  Tick async_tick = 0;
  Tick release_tick = 0;
  bool late_ran = false;

private:
  static constexpr Tick kReleaseDelayTicks = 5;

  void release(Tick tick) {
    release_tick = tick;
    engine()->primary_release();
  }

  Tick late_tick_;
  Event post_{this, EventType::TIMER_CALLBACK};
  Event async_{this, EventType::TIMER_CALLBACK};
  Event release_{this, EventType::TIMER_CALLBACK};
  Event late_{this, EventType::TIMER_CALLBACK};
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

/// Injects a self-replenishing async stream from the first handler in a large
/// same-tick batch. This models repeated host submissions arriving while
/// already-scheduled CU quanta remain at that tick.
class SameTickAsyncFairnessComponent : public Component {
public:
  SameTickAsyncFairnessComponent(std::string name, uint32_t batch_size, uint32_t async_count)
      : Component(std::move(name)), batch_size_(batch_size), async_count_(async_count) {
    inject_event_.set_handler(
        [this](Tick, Message *) { engine()->schedule_event_now(&async_event_); });
    batch_event_.set_handler([this](Tick, Message *) {
      if (batch_events_processed_ == 0)
        async_events_before_first_batch_event_ = async_events_processed_;
      ++batch_events_processed_;
    });
    async_event_.set_handler([this](Tick tick, Message *) {
      if (async_events_processed_ == 0) {
        first_async_tick_ = tick;
        batch_events_before_first_async_ = batch_events_processed_;
      }
      ++async_events_processed_;
      if (async_events_processed_ < async_count_)
        engine()->schedule_event_now(&async_event_);
    });
  }

  void startup() override {
    schedule_event(&inject_event_, 1);
    for (uint32_t batch_index = 0; batch_index < batch_size_; ++batch_index)
      schedule_event(&batch_event_, 1);
  }

  Tick first_async_tick() const { return first_async_tick_; }
  uint32_t async_events_processed() const { return async_events_processed_; }
  uint32_t batch_events_processed() const { return batch_events_processed_; }
  uint32_t batch_events_before_first_async() const { return batch_events_before_first_async_; }
  uint32_t async_events_before_first_batch_event() const {
    return async_events_before_first_batch_event_;
  }

private:
  uint32_t batch_size_ = 0;
  uint32_t async_count_ = 0;
  uint32_t async_events_processed_ = 0;
  uint32_t batch_events_processed_ = 0;
  uint32_t batch_events_before_first_async_ = 0;
  uint32_t async_events_before_first_batch_event_ = 0;
  Tick first_async_tick_ = TICK_MAX;
  Event inject_event_{this, EventType::TIMER_CALLBACK};
  Event batch_event_{this, EventType::TIMER_CALLBACK};
  Event async_event_{this, EventType::TIMER_CALLBACK};
};

/// Models a level-triggered host poll that keeps requesting a retry while local
/// device work is already scheduled for the following tick.
class NextTickRetryFairnessComponent : public Component {
public:
  explicit NextTickRetryFairnessComponent(std::string name, uint32_t retry_count)
      : Component(std::move(name)), retry_count_(retry_count) {
    inject_event_.set_handler(
        [this](Tick, Message *) { engine()->schedule_event_next_tick(&retry_event_); });
    local_event_.set_handler([this](Tick tick, Message *) {
      local_tick_ = tick;
      retries_before_local_ = retries_processed_;
    });
    retry_event_.set_handler([this](Tick, Message *) {
      ++retries_processed_;
      if (retries_processed_ < retry_count_) {
        engine()->schedule_event_next_tick(&retry_event_);
      } else {
        engine()->primary_release();
      }
    });
  }

  void startup() override {
    engine()->register_as_primary();
    schedule_event(&inject_event_, 1);
    schedule_event(&local_event_, 2);
  }

  Tick local_tick() const { return local_tick_; }
  uint32_t retries_processed() const { return retries_processed_; }
  uint32_t retries_before_local() const { return retries_before_local_; }

private:
  uint32_t retry_count_ = 0;
  uint32_t retries_processed_ = 0;
  uint32_t retries_before_local_ = 0;
  Tick local_tick_ = TICK_MAX;
  Event inject_event_{this, EventType::TIMER_CALLBACK};
  Event local_event_{this, EventType::TIMER_CALLBACK};
  Event retry_event_{this, EventType::TIMER_CALLBACK};
};

/// Records the timestamp produced by a next-tick injection at the end of the
/// representable simulation timeline.
class NextTickSaturationComponent : public Component {
public:
  explicit NextTickSaturationComponent(std::string name) : Component(std::move(name)) {
    inject_event_.set_handler(
        [this](Tick, Message *) { engine()->schedule_event_next_tick(&observed_event_); });
    observed_event_.set_handler([this](Tick tick, Message *) { observed_tick_ = tick; });
  }

  void startup() override { schedule_event(&inject_event_, TICK_MAX - 1); }

  Tick observed_tick() const { return observed_tick_; }

private:
  Tick observed_tick_ = 0;
  Event inject_event_{this, EventType::TIMER_CALLBACK};
  Event observed_event_{this, EventType::TIMER_CALLBACK};
};

/// Component that counts initialize()/startup()/shutdown() calls. Used to verify
/// that shutdown() cleanup fires exactly once per initialized component, on both
/// the normal shutdown path and the initialization-failure unwind path. When given a
/// shared order log, records its own name on shutdown() so tests can assert the
/// reverse-topology-order shutdown contract.
class LifecycleCountingComponent : public Component {
public:
  explicit LifecycleCountingComponent(std::string name,
                                      std::vector<std::string> *shutdown_order = nullptr)
      : Component(std::move(name)), shutdown_order_(shutdown_order) {}

  util::Result initialize(util::DiagnosticEmitter = {}) override {
    ++initializes;
    return util::Result::success();
  }
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

class FailingInitializeComponent : public LifecycleCountingComponent {
public:
  using LifecycleCountingComponent::LifecycleCountingComponent;

  util::Result initialize(util::DiagnosticEmitter emit_error = {}) override {
    ++initializes;
    if (fail)
      return emit_error.emit() << "component initialization rejected";
    return util::Result::success();
  }

  bool fail = true;
};

/// Helper: build engine with manual partition assignment.
/// assigner maps component name → partition ID.
void build_with_manual_partitions(SimulationEngine &engine, uint32_t num_partitions,
                                  std::function<PartitionID(Component *)> assigner) {
  util::StringDiagnostic admission_diagnostic;
  ASSERT_TRUE(
      engine.topology()
          .partition_manual(num_partitions, std::move(assigner), admission_diagnostic.emitter())
          .succeeded())
      << admission_diagnostic.message();
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();
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

ExitStatus run_with_watchdog(SimulationEngine &engine) {
  ExitStatus exit;
  std::atomic<bool> finished{false};
  std::thread runner([&]() {
    exit = engine.run();
    finished.store(true, std::memory_order_release);
  });
  const bool started = engine.wait_until_started();
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!finished.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  const bool finished_on_its_own = finished.load(std::memory_order_acquire);
  if (!finished_on_its_own)
    engine.request_exit("termination regression timeout", 1);
  runner.join();
  EXPECT_TRUE(started);
  EXPECT_TRUE(finished_on_its_own);
  return exit;
}

} // namespace

TEST(TopologyAdmissionTest, MissingRootAndInvalidPathsAreDiagnosed) {
  Topology topology;
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(topology.resolve_component("missing", diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("no root"), std::string::npos);
  diagnostic("");
  EXPECT_TRUE(topology.resolve_port("missing.port", diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("no root"), std::string::npos);
  diagnostic("");
  EXPECT_TRUE(topology.wire_links({}, ExecMode::FUNCTIONAL, diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("no root"), std::string::npos);

  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<CounterComponent>("leaf", 0));
  topology.set_root(std::move(root));
  for (const auto &path : {"missing", "leaf.child"}) {
    diagnostic("");
    EXPECT_TRUE(topology.resolve_component(path, diagnostic.emitter()).failed());
    EXPECT_NE(diagnostic.message().find(path), std::string::npos);
  }
  diagnostic("");
  EXPECT_TRUE(topology.resolve_port("leaf", diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("needs '.'"), std::string::npos);
}

TEST(TopologyAdmissionTest, InvalidLaterLinkDoesNotCreatePortsOrLinks) {
  Topology topology;
  auto root = std::make_unique<CompositeComponent>("root");
  auto *leaf = root->add_child(std::make_unique<CounterComponent>("leaf", 0));
  topology.set_root(std::move(root));
  const auto ports_before = leaf->ports().size();
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(topology
                  .wire_links({{"leaf.new_out", "leaf.new_in"}, {"leaf.new_out", "missing.new_in"}},
                              ExecMode::FUNCTIONAL, diagnostic.emitter())
                  .failed());
  EXPECT_FALSE(diagnostic.message().empty());
  EXPECT_TRUE(topology.links().empty());
  EXPECT_EQ(leaf->ports().size(), ports_before);
}

TEST(TopologyAdmissionTest, EmptyAssignerPreservesPartitions) {
  util::StringDiagnostic admission_diagnostic;
  Topology topology;
  auto root = std::make_unique<CompositeComponent>("root");
  auto *leaf = root->add_child(std::make_unique<CounterComponent>("leaf", 0));
  topology.set_root(std::move(root));
  ASSERT_TRUE(topology.partition_balanced(1, admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(topology.partition_manual(2, {}, diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("assigner must be set"), std::string::npos);
  diagnostic("");
  EXPECT_TRUE(
      topology.partition_manual(
                  0, [](Component *) { return PartitionID{0}; }, diagnostic.emitter())
          .failed());
  EXPECT_NE(diagnostic.message().find("num_partitions must be nonzero"), std::string::npos);
  ASSERT_EQ(topology.partitions().size(), 1u);
  EXPECT_EQ(leaf->partition_id(), 0u);
}

TEST(TopologyAdmissionTest, ClockFailuresDoNotRegisterDomains) {
  Topology topology;
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(topology.add_clock_domain("bad", 0, 0, diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("positive"), std::string::npos);
  EXPECT_TRUE(topology.clock_domains().empty());
  diagnostic("");
  EXPECT_TRUE(
      topology.add_clock_domain("fast", TICKS_PER_SECOND + 1, 0, diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("must not exceed"), std::string::npos);
  EXPECT_TRUE(topology.clock_domains().empty());
  diagnostic("");
  auto limit = topology.add_clock_domain("limit", TICKS_PER_SECOND, 0, diagnostic.emitter());
  ASSERT_TRUE(limit.succeeded()) << diagnostic.message();
  ClockDomain *limit_domain = limit.value();
  EXPECT_EQ(limit_domain->period(), 1u);
  ASSERT_EQ(topology.clock_domains().size(), 1u);
  EXPECT_EQ(topology.clock_domains().front().get(), limit_domain);
  auto domain = ClockDomain::create("clock", 1, 0, diagnostic.emitter());
  ASSERT_TRUE(domain.succeeded()) << diagnostic.message();
  const ClockDomain &clock = domain.value();
  diagnostic("");
  EXPECT_TRUE(clock.derive("bad", 0, 0, diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("divisor"), std::string::npos);
  diagnostic("");
  EXPECT_TRUE(clock.derive("bad", 2, 0, diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("frequency"), std::string::npos);
}

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
    util::StringDiagnostic admission_diagnostic;
    SCOPED_TRACE(::testing::Message() << "pass=" << pass);
    ASSERT_TRUE(
        topology
            .partition_manual(
                2, [](Component *) { return PartitionID{0}; }, admission_diagnostic.emitter())
            .succeeded())
        << admission_diagnostic.message();
    EXPECT_EQ(external.partition_id(), 0u);
    EXPECT_EQ(std::ranges::count(topology.partitions()[0].components, &external), 1);
  }
}

TEST(TopologyPartitionTest, ManualPartitionRunsExternalProducerOnAssignedPartition) {
  util::StringDiagnostic admission_diagnostic;
  // SimulationEngine borrows external endpoint owners, so external must outlive engine.
  ProducerComponent external("external", 3);
  SimulationEngine engine({.num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  auto consumer_component = std::make_unique<ConsumerComponent>("consumer");
  auto *consumer = consumer_component.get();
  root->add_child(std::move(consumer_component));
  engine.topology().set_root(std::move(root));
  engine.topology().add_link(external.out_port(), consumer->in_port(), 0);

  ASSERT_TRUE(engine.topology()
                  .partition_manual(
                      2, [](Component *) { return PartitionID{1}; }, admission_diagnostic.emitter())
                  .succeeded())
      << admission_diagnostic.message();

  ASSERT_EQ(external.partition_id(), 1u);
  ASSERT_EQ(consumer->partition_id(), 1u);
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();
  auto exit = engine.run();

  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_EQ(external.sent_, 3u);
  ASSERT_EQ(consumer->received.size(), 3u);
  for (size_t i = 0; i < consumer->received.size(); ++i)
    EXPECT_EQ(consumer->received[i].second, i);
}

TEST(TopologyPartitionTest, BalancedSinglePartitionIncludesExternalLinkOwner) {
  util::StringDiagnostic admission_diagnostic;
  // Topology borrows external endpoint owners, so external must outlive topology.
  ProducerComponent external("external", 0, 1, false);
  Topology topology;
  auto root = std::make_unique<CompositeComponent>("root");
  auto consumer = std::make_unique<ConsumerComponent>("consumer");
  auto *consumer_ptr = consumer.get();
  root->add_child(std::move(consumer));
  topology.set_root(std::move(root));

  topology.add_link(external.out_port(), consumer_ptr->in_port(), 1);

  ASSERT_TRUE(topology.partition_balanced(1, admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();

  ASSERT_EQ(topology.partitions().size(), 1u);
  EXPECT_EQ(external.partition_id(), 0u);
  EXPECT_EQ(std::ranges::count(topology.partitions()[0].components, &external), 1);
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
    util::StringDiagnostic admission_diagnostic;
    SCOPED_TRACE(::testing::Message() << "pass=" << pass);
    ASSERT_TRUE(topology.partition_balanced(2, admission_diagnostic.emitter()).succeeded())
        << admission_diagnostic.message();
    ASSERT_EQ(topology.partitions().size(), 2u);
    EXPECT_LT(external.partition_id(), 2u);

    size_t occurrences = 0;
    for (const auto &partition : topology.partitions())
      occurrences += std::ranges::count(partition.components, &external);
    EXPECT_EQ(occurrences, 1u);
  }
}

TEST(TopologyPartitionTest, MultiThreadedEngineRequiresExplicitPolicy) {
  SimulationEngine engine({.num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<CounterComponent>("counter0", 0));
  root->add_child(std::make_unique<CounterComponent>("counter1", 0));
  engine.topology().set_root(std::move(root));

  EXPECT_TRUE(engine.create().failed());
}

TEST(TopologyPartitionTest, RejectsZeroPartitionCount) {
  Topology topology;

  EXPECT_TRUE(topology.partition_balanced(0).failed());
  EXPECT_TRUE(topology.partitions().empty());

  EXPECT_TRUE(topology.partition_manual(0, [](Component *) { return PartitionID{0}; }).failed());
  EXPECT_TRUE(topology.partitions().empty());
}

TEST(TopologyPartitionTest, OutOfRangeManualAssignmentLeavesExistingStateIntact) {
  util::StringDiagnostic admission_diagnostic;
  Topology topology;
  auto root = std::make_unique<CompositeComponent>("root");
  auto *counter0 = root->add_child(std::make_unique<CounterComponent>("counter0", 0));
  auto *counter1 = root->add_child(std::make_unique<CounterComponent>("counter1", 0));
  topology.set_root(std::move(root));
  ASSERT_TRUE(topology
                  .partition_manual(
                      1, [](Component *) { return PartitionID{0}; }, admission_diagnostic.emitter())
                  .succeeded())
      << admission_diagnostic.message();

  ASSERT_EQ(topology.partitions().size(), 1u);
  ASSERT_EQ(counter0->partition_id(), 0u);
  ASSERT_EQ(counter1->partition_id(), 0u);

  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(topology
                  .partition_manual(
                      2,
                      [](Component *component) {
                        return component->name() == "counter1" ? PartitionID{2} : PartitionID{1};
                      },
                      diagnostic.emitter())
                  .failed());
  EXPECT_NE(diagnostic.message().find("out-of-range"), std::string::npos);
  EXPECT_EQ(topology.partitions().size(), 1u);
  EXPECT_EQ(counter0->partition_id(), 0u);
  EXPECT_EQ(counter1->partition_id(), 0u);
}

TEST(TopologyPartitionTest, EngineRejectsZeroWorkerThreads) {
  SimulationEngine engine({.num_threads = 0});

  EXPECT_TRUE(engine.create().failed());
  EXPECT_FALSE(engine.is_created());
  EXPECT_TRUE(engine.topology().partitions().empty());
}

TEST(TopologyPartitionTest, EngineRejectsPartitionCountMismatch) {
  util::StringDiagnostic admission_diagnostic;
  SimulationEngine engine({.num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *counter = root->add_child(std::make_unique<CounterComponent>("counter0", 0));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.topology()
                  .partition_manual(
                      1, [](Component *) { return PartitionID{0}; }, admission_diagnostic.emitter())
                  .succeeded())
      << admission_diagnostic.message();

  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(engine.create(diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("partition count"), std::string::npos);
  EXPECT_FALSE(engine.is_created());
  EXPECT_EQ(counter->engine(), nullptr);
  ASSERT_TRUE(engine.topology()
                  .partition_manual(
                      2, [](Component *) { return PartitionID{0}; }, admission_diagnostic.emitter())
                  .succeeded())
      << admission_diagnostic.message();
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();
  EXPECT_EQ(counter->engine(), &engine);
}

TEST(TopologyPartitionTest, EngineRejectsZeroLatencyCrossPartitionLink) {
  util::StringDiagnostic admission_diagnostic;
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
  ASSERT_TRUE(engine.topology()
                  .partition_manual(2, partition_by_name_suffix, admission_diagnostic.emitter())
                  .succeeded())
      << admission_diagnostic.message();

  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(engine.create(diagnostic.emitter()).failed());
  const std::string &message = diagnostic.message();
  EXPECT_NE(message.find("root.producer0.out"), std::string::npos);
  EXPECT_NE(message.find("root.consumer1.in"), std::string::npos);
  EXPECT_NE(message.find("positive latency"), std::string::npos);
  EXPECT_EQ(producer_ptr->engine(), nullptr);
  EXPECT_EQ(consumer_ptr->engine(), nullptr);
  EXPECT_FALSE(engine.is_created());
}

TEST(TopologyPartitionTest, EngineRejectsCrossPartitionQueuedLink) {
  util::StringDiagnostic admission_diagnostic;
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
  ASSERT_TRUE(engine.topology()
                  .partition_manual(2, partition_by_name_suffix, admission_diagnostic.emitter())
                  .succeeded())
      << admission_diagnostic.message();

  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(engine.create(diagnostic.emitter()).failed());
  const std::string &message = diagnostic.message();
  EXPECT_NE(message.find("root.producer0.out"), std::string::npos);
  EXPECT_NE(message.find("root.consumer1.in"), std::string::npos);
  EXPECT_NE(message.find("QueuedLink"), std::string::npos);
  EXPECT_EQ(producer_ptr->engine(), nullptr);
  EXPECT_EQ(consumer_ptr->engine(), nullptr);
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
  util::StringDiagnostic admission_diagnostic;
  SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<CounterComponent>("c0", 10));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();
  const ExitStatus exit = engine.run();

  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_EQ(exit.cause, ExitCause::QUIESCENCE);
}

TEST(TerminationTest, AllPrimaryDoneTrigger) {
  util::StringDiagnostic admission_diagnostic;
  SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *p = root->add_child(std::make_unique<ProducerComponent>("p0", 5));
  auto *c = root->add_child(std::make_unique<ConsumerComponent>("c0"));
  engine.topology().set_root(std::move(root));
  engine.topology().add_link(static_cast<ProducerComponent *>(p)->out_port(),
                             static_cast<ConsumerComponent *>(c)->in_port(), 1);
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();
  const ExitStatus exit = engine.run();

  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_EQ(exit.cause, ExitCause::ALL_PRIMARIES_COMPLETED);
  EXPECT_EQ(static_cast<ConsumerComponent *>(c)->received.size(), 5u);
}

TEST(TerminationTest, MaxTicksSentinel) {
  util::StringDiagnostic admission_diagnostic;
  SimulationEngine engine({.max_ticks = 100, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<InfiniteComponent>("inf0"));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();
  const ExitStatus exit = engine.run();

  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_EQ(exit.cause, ExitCause::MAX_TICKS);
}

TEST(TerminationTest, MultiThreadedMaxTicksStopsAQuiescentPrimary) {
  SimulationEngine engine({.max_ticks = 100, .num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<AsyncPrimaryComponent>("primary0"));
  root->add_child(std::make_unique<AsyncPrimaryComponent>("primary1"));
  engine.topology().set_root(std::move(root));
  build_with_manual_partitions(engine, 2, partition_by_name_suffix);

  const ExitStatus exit = run_with_watchdog(engine);
  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_EQ(exit.cause, ExitCause::ACTIVE_PRIMARY_STALL);
  EXPECT_EQ(exit.tick, 0u);
}

TEST(TerminationTest, SingleThreadedMaxTicksStopsAQuiescentPrimary) {
  SimulationEngine engine({.max_ticks = 100, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<AsyncPrimaryComponent>("primary0"));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.create().succeeded());

  const ExitStatus exit = engine.run();

  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_EQ(exit.cause, ExitCause::ACTIVE_PRIMARY_STALL);
  EXPECT_EQ(exit.tick, 0u);
}

TEST(TerminationTest, StepModeMaxTicksStopsAQuiescentPrimary) {
  SimulationEngine engine({.max_ticks = 100, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<AsyncPrimaryComponent>("primary0"));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.create().succeeded());

  EXPECT_FALSE(engine.step());
  EXPECT_EQ(engine.last_exit().reason, ExitReason::COMPLETED);
  EXPECT_EQ(engine.last_exit().cause, ExitCause::ACTIVE_PRIMARY_STALL);
  EXPECT_EQ(engine.last_exit().tick, 0u);
}

TEST(TerminationTest, FinitePrimaryStallAppliesOnlyBeforeMaxTicks) {
  for (const Tick event_tick : {Tick{99}, Tick{100}}) {
    SCOPED_TRACE(::testing::Message() << "run event_tick=" << event_tick);
    SimulationEngine engine({.max_ticks = 100, .num_threads = 1});
    auto root = std::make_unique<CompositeComponent>("root");
    auto *component = static_cast<ScheduledPrimaryComponent *>(
        root->add_child(std::make_unique<ScheduledPrimaryComponent>("primary0", event_tick)));
    engine.topology().set_root(std::move(root));
    ASSERT_TRUE(engine.create().succeeded());

    const ExitStatus exit = engine.run();
    EXPECT_TRUE(component->handled);
    EXPECT_EQ(component->handled_tick, event_tick);
    EXPECT_EQ(exit.tick, event_tick);
    EXPECT_EQ(exit.cause,
              event_tick < 100 ? ExitCause::ACTIVE_PRIMARY_STALL : ExitCause::MAX_TICKS);
  }

  for (const Tick event_tick : {Tick{99}, Tick{100}}) {
    SCOPED_TRACE(::testing::Message() << "step event_tick=" << event_tick);
    SimulationEngine engine({.max_ticks = 100, .num_threads = 1});
    auto root = std::make_unique<CompositeComponent>("root");
    auto *component = static_cast<ScheduledPrimaryComponent *>(
        root->add_child(std::make_unique<ScheduledPrimaryComponent>("primary0", event_tick)));
    engine.topology().set_root(std::move(root));
    ASSERT_TRUE(engine.create().succeeded());

    while (engine.step()) {
    }
    EXPECT_TRUE(component->handled);
    EXPECT_EQ(component->handled_tick, event_tick);
    EXPECT_EQ(engine.last_exit().tick, event_tick);
    EXPECT_EQ(engine.last_exit().cause,
              event_tick < 100 ? ExitCause::ACTIVE_PRIMARY_STALL : ExitCause::MAX_TICKS);
  }
}

TEST(TerminationTest, FiniteRunsReportEarlyPrimaryCompletionAtTheLastProcessedTick) {
  for (const uint32_t partitions : {1u, 2u}) {
    SCOPED_TRACE(::testing::Message() << "run partitions=" << partitions);
    SimulationEngine engine({.max_ticks = 100, .num_threads = partitions});
    auto root = std::make_unique<CompositeComponent>("root");
    auto *component = static_cast<ReleasingPrimaryComponent *>(
        root->add_child(std::make_unique<ReleasingPrimaryComponent>("primary0", 10)));
    engine.topology().set_root(std::move(root));
    if (partitions == 1) {
      ASSERT_TRUE(engine.create().succeeded());
    } else {
      build_with_manual_partitions(engine, partitions, partition_by_name_suffix);
    }

    const ExitStatus exit = engine.run();
    EXPECT_EQ(component->handled_tick, 10u);
    EXPECT_EQ(exit.cause, ExitCause::ALL_PRIMARIES_COMPLETED);
    EXPECT_EQ(exit.tick, 10u);
  }

  SimulationEngine engine({.max_ticks = 100, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *component = static_cast<ReleasingPrimaryComponent *>(
      root->add_child(std::make_unique<ReleasingPrimaryComponent>("primary0", 10)));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.create().succeeded());
  while (engine.step()) {
  }
  EXPECT_EQ(component->handled_tick, 10u);
  EXPECT_EQ(engine.last_exit().cause, ExitCause::ALL_PRIMARIES_COMPLETED);
  EXPECT_EQ(engine.last_exit().tick, 10u);
}

TEST(TerminationTest, FiniteHorizonBoundsDrainAfterPrimaryRelease) {
  for (const uint32_t partitions : {1u, 2u}) {
    SCOPED_TRACE(partitions);
    SimulationEngine engine({.max_ticks = 100, .num_threads = partitions});
    auto root = std::make_unique<CompositeComponent>("root");
    auto producer = std::make_unique<ProducerComponent>("producer0", 1, 10);
    auto consumer = std::make_unique<ConsumerComponent>("consumer1");
    ProducerComponent *producer_ptr = producer.get();
    ConsumerComponent *consumer_ptr = consumer.get();
    root->add_child(std::move(producer));
    root->add_child(std::move(consumer));
    engine.topology().set_root(std::move(root));
    engine.topology().add_link(producer_ptr->out_port(), consumer_ptr->in_port(), 190);
    if (partitions == 1) {
      ASSERT_TRUE(engine.create().succeeded());
    } else {
      build_with_manual_partitions(engine, partitions, partition_by_name_suffix);
    }

    const ExitStatus exit = engine.run();
    EXPECT_EQ(producer_ptr->sent_, 1u);
    EXPECT_TRUE(consumer_ptr->received.empty());
    EXPECT_EQ(exit.cause, ExitCause::MAX_TICKS);
    EXPECT_EQ(exit.tick, 100u);
  }
}

TEST(TerminationTest, FiniteHorizonDrainsAsyncWorkBeforeStopping) {
  for (const uint32_t partitions : {1u, 2u}) {
    SCOPED_TRACE(partitions);
    constexpr Tick kHorizon = 100;
    for (const Tick callback_tick : {Tick{20}, kHorizon - 1, kHorizon, kHorizon + 1}) {
      SCOPED_TRACE(callback_tick);
      SimulationEngine engine({.max_ticks = kHorizon, .num_threads = partitions});
      auto root = std::make_unique<CompositeComponent>("root");
      auto primary = std::make_unique<AsyncPostingPrimaryComponent>("primary0", callback_tick, 200);
      AsyncPostingPrimaryComponent *primary_ptr = primary.get();
      root->add_child(std::move(primary));
      engine.topology().set_root(std::move(root));
      if (partitions == 1) {
        ASSERT_TRUE(engine.create().succeeded());
      } else {
        build_with_manual_partitions(engine, partitions, partition_by_name_suffix);
      }

      const ExitStatus exit = engine.run();
      // The single-threaded sentinel follows this same-tick callback.
      // The partitioned loop stops before processing the horizon epoch.
      const bool ran = callback_tick < kHorizon || (callback_tick == kHorizon && partitions == 1);
      EXPECT_EQ(primary_ptr->async_tick, ran ? callback_tick : 0u);
      EXPECT_FALSE(primary_ptr->late_ran);
      EXPECT_EQ(exit.cause, ExitCause::MAX_TICKS);
      EXPECT_EQ(exit.tick, kHorizon);
    }
  }
}

TEST(TerminationTest, TickLimitDoesNotDelayBufferedAsyncWork) {
  constexpr Tick kUnrelatedTick = 99;
  for (const uint32_t partitions : {1u, 2u}) {
    for (const Tick horizon : {Tick{0}, kUnrelatedTick, kUnrelatedTick + 1}) {
      SCOPED_TRACE(::testing::Message() << "partitions=" << partitions << " horizon=" << horizon);
      SimulationEngine engine({.max_ticks = horizon, .num_threads = partitions});
      auto root = std::make_unique<CompositeComponent>("root");
      auto primary = std::make_unique<AsyncPostingPrimaryComponent>("primary0", 20, kUnrelatedTick,
                                                                    /*release_after_async=*/true);
      AsyncPostingPrimaryComponent *primary_ptr = primary.get();
      root->add_child(std::move(primary));
      engine.topology().set_root(std::move(root));
      if (partitions == 1) {
        ASSERT_TRUE(engine.create().succeeded());
      } else {
        build_with_manual_partitions(engine, partitions, partition_by_name_suffix);
      }

      const ExitStatus exit = engine.run();
      EXPECT_EQ(primary_ptr->async_tick, 20u);
      EXPECT_EQ(primary_ptr->release_tick, 25u);
      EXPECT_EQ(exit.cause, horizon == kUnrelatedTick ? ExitCause::MAX_TICKS
                                                      : ExitCause::ALL_PRIMARIES_COMPLETED);
      EXPECT_EQ(exit.tick, kUnrelatedTick);
    }
  }
}

TEST(TerminationTest, FiniteHorizonStopsBeforeSaturatedEvent) {
  for (const uint32_t partitions : {1u, 2u}) {
    for (const bool release_primary : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "partitions=" << partitions << " release_primary=" << release_primary);
      SimulationEngine engine({.max_ticks = 100, .num_threads = partitions});
      auto root = std::make_unique<CompositeComponent>("root");
      if (release_primary) {
        root->add_child(std::make_unique<ReleasingPrimaryComponent>("primary0", 10));
      } else {
        root->add_child(std::make_unique<ScheduledPrimaryComponent>("primary0", 10));
      }
      auto late = std::make_unique<ProducerComponent>("late1", 1, TICK_MAX,
                                                      /*register_primary=*/false);
      ProducerComponent *late_ptr = late.get();
      root->add_child(std::move(late));
      engine.topology().set_root(std::move(root));
      if (partitions == 1) {
        ASSERT_TRUE(engine.create().succeeded());
      } else {
        build_with_manual_partitions(engine, partitions, partition_by_name_suffix);
      }

      const ExitStatus exit = run_with_watchdog(engine);
      EXPECT_EQ(late_ptr->sent_, 0u);
      EXPECT_EQ(exit.cause, ExitCause::MAX_TICKS);
      EXPECT_EQ(exit.tick, 100u);
    }
  }
}

TEST(TerminationTest, FiniteRunsReportEarlyQuiescenceWithoutPrimaries) {
  for (const uint32_t partitions : {1u, 2u}) {
    SCOPED_TRACE(::testing::Message() << "run partitions=" << partitions);
    SimulationEngine engine({.max_ticks = 100, .num_threads = partitions});
    auto root = std::make_unique<CompositeComponent>("root");
    root->add_child(std::make_unique<LifecycleCountingComponent>("component0"));
    engine.topology().set_root(std::move(root));
    if (partitions == 1) {
      ASSERT_TRUE(engine.create().succeeded());
    } else {
      ASSERT_TRUE(engine.topology().partition_balanced(partitions).succeeded());
      ASSERT_TRUE(engine.create().succeeded());
    }

    const ExitStatus exit = engine.run();
    EXPECT_EQ(exit.cause, ExitCause::QUIESCENCE);
    EXPECT_EQ(exit.tick, 0u);
  }

  SimulationEngine engine({.max_ticks = 100, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<LifecycleCountingComponent>("component0"));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.create().succeeded());
  while (engine.step()) {
  }
  EXPECT_EQ(engine.last_exit().cause, ExitCause::QUIESCENCE);
  EXPECT_EQ(engine.last_exit().tick, 0u);
}

TEST(TerminationTest, FiniteMaxTicksStillAwaitsTheFirstPrimary) {
  for (const uint32_t partitions : {1u, 2u}) {
    SCOPED_TRACE(partitions);
    SimulationEngine engine({.max_ticks = 100, .num_threads = partitions, .await_primaries = true});
    auto root = std::make_unique<CompositeComponent>("root");
    auto *component0 = static_cast<LifecycleCountingComponent *>(
        root->add_child(std::make_unique<LifecycleCountingComponent>("component0")));
    root->add_child(std::make_unique<LifecycleCountingComponent>("component1"));
    engine.topology().set_root(std::move(root));
    if (partitions > 1) {
      ASSERT_TRUE(engine.topology().partition_balanced(partitions).succeeded());
    }
    ASSERT_TRUE(engine.create().succeeded());

    ExitStatus exit;
    std::atomic<bool> finished{false};
    std::thread runner([&]() {
      exit = engine.run();
      finished.store(true, std::memory_order_release);
    });
    const bool started = engine.wait_until_started();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const bool stayed_alive_while_idle = !finished.load(std::memory_order_acquire);
    engine.request_exit("external stop", 7);

    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!finished.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    const bool stopped_on_request = finished.load(std::memory_order_acquire);
    Event rescue_event(component0, EventType::TIMER_CALLBACK, [](Tick, Message *) {});
    if (!stopped_on_request)
      engine.schedule_event_now(&rescue_event);
    runner.join();

    EXPECT_TRUE(started);
    EXPECT_TRUE(stayed_alive_while_idle);
    EXPECT_TRUE(stopped_on_request);
    EXPECT_EQ(exit.reason, ExitReason::EXIT_REQUEST);
    EXPECT_EQ(exit.code, 7);
    EXPECT_EQ(exit.message, "external stop");
  }
}

TEST(StartupReadinessTest, CleanStartupReportsReady) {
  util::StringDiagnostic admission_diagnostic;
  // A clean startup must make wait_until_started() return true without hanging.
  SimulationEngine engine({.max_ticks = 5, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<LifecycleCountingComponent>("c0"));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();

  engine.step();
  EXPECT_TRUE(engine.wait_until_started());
}

TEST(StartupReadinessTest, InitializationFailureUnwindsAttemptedPrefixAndAllowsFreshCreate) {
  util::StringDiagnostic diagnostic;
  std::vector<std::string> shutdown_order;
  SimulationEngine engine({.max_ticks = 5, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *before = static_cast<LifecycleCountingComponent *>(
      root->add_child(std::make_unique<LifecycleCountingComponent>("before", &shutdown_order)));
  auto *failing = static_cast<FailingInitializeComponent *>(
      root->add_child(std::make_unique<FailingInitializeComponent>("failing", &shutdown_order)));
  auto *after = static_cast<LifecycleCountingComponent *>(
      root->add_child(std::make_unique<LifecycleCountingComponent>("after", &shutdown_order)));
  engine.topology().set_root(std::move(root));

  EXPECT_TRUE(engine.create(diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("initialization rejected"), std::string::npos);
  EXPECT_FALSE(engine.is_created());
  EXPECT_FALSE(engine.wait_until_started());
  EXPECT_EQ(engine.last_exit().code, 1);
  EXPECT_EQ(before->initializes, 1u);
  EXPECT_EQ(failing->initializes, 1u);
  EXPECT_EQ(after->initializes, 0u);
  EXPECT_EQ(before->startups, 0u);
  EXPECT_EQ(failing->startups, 0u);
  EXPECT_EQ(before->shutdowns, 1u);
  EXPECT_EQ(failing->shutdowns, 1u);
  EXPECT_EQ(after->shutdowns, 0u);
  EXPECT_EQ(shutdown_order, (std::vector<std::string>{"failing", "before"}));
  engine.shutdown();
  EXPECT_EQ(before->shutdowns, 1u);
  EXPECT_EQ(failing->shutdowns, 1u);

  failing->fail = false;
  ASSERT_TRUE(engine.create(diagnostic.emitter()).succeeded()) << diagnostic.message();
  engine.step();
  EXPECT_TRUE(engine.wait_until_started());
  EXPECT_EQ(after->initializes, 1u);
  engine.shutdown();
  EXPECT_EQ(before->shutdowns, 2u);
  EXPECT_EQ(failing->shutdowns, 2u);
  EXPECT_EQ(after->shutdowns, 1u);
  EXPECT_EQ(shutdown_order,
            (std::vector<std::string>{"failing", "before", "after", "failing", "before"}));
}

TEST(StartupReadinessTest, CreateThenShutdownRunsComponentCleanup) {
  util::StringDiagnostic admission_diagnostic;
  // create() initializes every component; shutting the engine down before any
  // run()/step() must still invoke shutdown() on each initialized component
  // exactly once (shutdown() pairs with initialize(), not startup()).
  SimulationEngine engine({.max_ticks = 10, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *comp = root->add_child(std::make_unique<LifecycleCountingComponent>("comp"));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();

  auto *counting = static_cast<LifecycleCountingComponent *>(comp);
  EXPECT_EQ(counting->initializes, 1u);
  EXPECT_EQ(counting->startups, 0u);

  engine.shutdown();
  EXPECT_EQ(counting->shutdowns, 1u);
}

TEST(TerminationTest, RequestExitWakesAllPartitions) {
  util::StringDiagnostic admission_diagnostic;
  // Infinite work makes request_exit() the only termination path.
  SimulationEngine engine({.num_threads = 4});
  auto root = std::make_unique<CompositeComponent>("root");
  for (int i = 0; i < 4; ++i)
    root->add_child(std::make_unique<InfiniteComponent>("inf" + std::to_string(i)));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.topology().partition_balanced(4, admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();

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
  // Every worker mode dereferences context zero in its wake path. Iterate:
  // the two threads must interleave inside the narrow shutdown window.
  for (const uint32_t partitions : {1u, 2u, 4u}) {
    SCOPED_TRACE(partitions);
    for (int iter = 0; iter < 100; ++iter) {
      util::StringDiagnostic admission_diagnostic;
      SimulationEngine engine({.num_threads = partitions});
      auto root = std::make_unique<CompositeComponent>("root");
      root->add_child(std::make_unique<CounterComponent>("c0", 4));
      engine.topology().set_root(std::move(root));
      if (partitions > 1) {

        ASSERT_TRUE(engine.topology()
                        .partition_balanced(partitions, admission_diagnostic.emitter())
                        .succeeded())
            << admission_diagnostic.message();
      }
      ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
          << admission_diagnostic.message();

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
}

// All foreign-thread termination paths must wake an idle engine, including a
// barrier completion sleeping on behalf of the other partitions. No async event
// is posted after termination to rescue a missing publication.
TEST(TerminationTest, IdleExitAndLastPrimaryReleaseWakeEveryWorkerMode) {
  for (const uint32_t partitions : {1u, 2u, 4u}) {
    for (const bool release_primary : {false, true}) {
      for (int iteration = 0; iteration < 16; ++iteration) {
        util::StringDiagnostic admission_diagnostic;
        SCOPED_TRACE(::testing::Message() << "partitions=" << partitions << " primary="
                                          << release_primary << " iteration=" << iteration);
        SimulationEngine engine({.num_threads = partitions, .await_primaries = true});
        auto root = std::make_unique<CompositeComponent>("root");
        if (release_primary)
          root->add_child(std::make_unique<AsyncPrimaryComponent>("primary"));
        engine.topology().set_root(std::move(root));
        if (partitions > 1) {

          ASSERT_TRUE(engine.topology()
                          .partition_balanced(partitions, admission_diagnostic.emitter())
                          .succeeded())
              << admission_diagnostic.message();
        }
        ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
            << admission_diagnostic.message();

        ExitStatus exit_status;
        std::atomic<bool> finished{false};
        std::thread runner([&]() {
          exit_status = engine.run();
          finished.store(true, std::memory_order_release);
        });
        const bool started = engine.wait_until_started();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (release_primary)
          engine.primary_release();
        else
          engine.request_exit("idle exit", 7);

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!finished.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline)
          std::this_thread::yield();
        const bool finished_on_its_own = finished.load(std::memory_order_acquire);
        if (!finished_on_its_own)
          engine.request_exit("idle termination timeout", 1);
        runner.join();

        EXPECT_TRUE(started);
        EXPECT_TRUE(finished_on_its_own);
        EXPECT_EQ(exit_status.reason,
                  release_primary ? ExitReason::COMPLETED : ExitReason::EXIT_REQUEST);
        EXPECT_EQ(exit_status.code, release_primary ? 0 : 7);
      }
    }
  }
}

namespace {

/// Waits for a peer partition by re-arming its own event every tick, the way the
/// command processor re-checks a doorbell whose grid is still running on another
/// XCD. It schedules nothing during startup: the engine has to come up fully
/// quiescent so the first barrier publishes TICK_MAX, which is the state the
/// stall has to survive.
class PeerWaitComponent : public Component {
public:
  explicit PeerWaitComponent(std::string name) : Component(std::move(name)) {
    in_ = add_port(std::make_unique<Port>("in", 0, this, PortDirection::IN, PortProtocol::UNTYPED));
    out_ =
        add_port(std::make_unique<Port>("out", 1, this, PortDirection::OUT, PortProtocol::UNTYPED));
    in_->set_handler([this](Tick, Message *) { replied = true; });
    recheck_.set_handler([this](Tick ts, Message *) {
      ++rechecks;
      if (replied) {
        engine()->primary_release();
        return;
      }
      if (!sent_) {
        sent_ = true;
        out_->send(make_test_msg(1));
      }
      // The reply can only be delivered by a later epoch's incoming drain, so a
      // re-check that outlives the epoch it was armed in is the whole point.
      schedule_event(&recheck_, ts + 1);
    });
  }

  void startup() override { engine()->register_as_primary(); }

  Port *in_port() { return in_; }
  Port *out_port() { return out_; }
  Event *recheck_event() { return &recheck_; }

  bool replied = false;
  uint64_t rechecks = 0;

private:
  bool sent_ = false;
  Port *in_ = nullptr;
  Port *out_ = nullptr;
  Event recheck_{this, EventType::TIMER_CALLBACK};
};

} // namespace

// Regression: a partition that re-arms its own event while waiting on a peer must
// not be able to run the epoch loop by itself.
//
// An engine that comes up with nothing scheduled publishes TICK_MAX at its first
// barrier. TICK_MAX is the absence of a horizon, not an infinite one, but the
// epoch loop used to compare against it like any other bound -- so the first
// partition to receive async work processed events, and everything those handlers
// re-armed, without ever arriving at the next barrier. Its peers stayed parked in
// that barrier, the reply this partition was waiting for was never drained, and
// the run hung. Reproduces at any partition count; rocjitsu hit it on every
// multi-XCD config whose partition count was not exactly the XCD count.
TEST(TerminationTest, QuiescentEpochDoesNotStrandPeersAtTheBarrier) {
  util::StringDiagnostic admission_diagnostic;
  SimulationEngine engine({.num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *waiter = static_cast<PeerWaitComponent *>(
      root->add_child(std::make_unique<PeerWaitComponent>("waiter0")));
  // target=2 so the responder replies to the message it receives instead of
  // treating it as the last one; it is not a primary, so it never ends the run.
  auto *responder = static_cast<PingPongComponent *>(root->add_child(
      std::make_unique<PingPongComponent>("responder1", 2, false, /*register_primary=*/false)));
  engine.topology().set_root(std::move(root));
  engine.topology().add_link(waiter->out_port(), responder->in_port(), 1);
  engine.topology().add_link(responder->out_port(), waiter->in_port(), 1);
  ASSERT_TRUE(engine.topology()
                  .partition_manual(2, partition_by_name_suffix, admission_diagnostic.emitter())
                  .succeeded())
      << admission_diagnostic.message();
  ASSERT_EQ(waiter->partition_id(), 0u);
  ASSERT_EQ(responder->partition_id(), 1u);
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();

  ExitStatus exit_status;
  std::atomic<bool> finished{false};
  std::thread runner([&]() {
    exit_status = engine.run();
    finished.store(true, std::memory_order_release);
  });
  ASSERT_TRUE(engine.wait_until_started());

  // Let the engine settle into the quiescent state first. Nothing is scheduled,
  // so the first barrier publishes TICK_MAX within microseconds of startup; the
  // kick has to land after that, because an engine that never reaches a
  // TICK_MAX barrier does not exercise this at all.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Kick the waiter from a foreign thread, exactly as a doorbell write does.
  engine.schedule_event_now(waiter->recheck_event());

  // Bounded wait: a regression deadlocks the engine, and the test has to fail
  // rather than hang the suite. request_exit() unwinds the workers either way.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!finished.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  const bool finished_on_its_own = finished.load(std::memory_order_acquire);
  if (!finished_on_its_own)
    engine.request_exit("regression timeout");
  runner.join();

  EXPECT_TRUE(finished_on_its_own) << "engine did not terminate: the waiting partition never "
                                      "reached the barrier, so its peer's reply was never drained";
  EXPECT_EQ(exit_status.reason, ExitReason::COMPLETED);
  EXPECT_TRUE(waiter->replied);
  EXPECT_EQ(responder->recv_count, 1u);
}

#if defined(CLOCK_THREAD_CPUTIME_ID)
class IdleCpuProbe : public Component {
public:
  IdleCpuProbe(std::string name, std::atomic<uint64_t> &sample)
      : Component(std::move(name)), sample_(sample) {
    sample_event_.set_handler([this](Tick, Message *) {
      timespec cpu{};
      const uint64_t value = ::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu) == 0
                                 ? static_cast<uint64_t>(cpu.tv_sec) * 1'000'000'000 + cpu.tv_nsec
                                 : UINT64_MAX;
      sample_.store(value, std::memory_order_release);
    });
  }

  void startup() override { engine()->register_as_primary(); }
  Event *sample_event() { return &sample_event_; }

private:
  std::atomic<uint64_t> &sample_;
  Event sample_event_{this, EventType::TIMER_CALLBACK};
};
#endif

// Regression: a quiescent engine must idle, not spin, in every worker mode.
//
// The single worker waits directly; in multi-worker mode, barrier completion
// waits on behalf of all partitions. Removing either wait burns host CPU and
// can starve the guest process that must ring the next doorbell. Measure only
// worker CPU time so sanitizer and helper threads do not count against this
// engine's budget. Sampling events also finish worker startup before measuring.
TEST(TerminationTest, QuiescentEngineDoesNotSpinTheHost) {
#if !defined(CLOCK_THREAD_CPUTIME_ID)
  GTEST_SKIP() << "per-thread CPU clocks are unavailable";
#else
  for (const uint32_t partitions : {1u, 2u, 4u}) {
    util::StringDiagnostic admission_diagnostic;
    SCOPED_TRACE(partitions);
    std::vector<std::atomic<uint64_t>> samples(partitions);
    SimulationEngine engine({.num_threads = partitions});
    auto root = std::make_unique<CompositeComponent>("root");
    std::vector<IdleCpuProbe *> probes;
    for (uint32_t i = 0; i < partitions; ++i) {
      auto probe = std::make_unique<IdleCpuProbe>("idle" + std::to_string(i), samples[i]);
      probes.push_back(probe.get());
      root->add_child(std::move(probe));
    }
    engine.topology().set_root(std::move(root));
    ASSERT_TRUE(engine.topology()
                    .partition_balanced(partitions, admission_diagnostic.emitter())
                    .succeeded())
        << admission_diagnostic.message();
    ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
        << admission_diagnostic.message();
    for (uint32_t i = 0; i < partitions; ++i)
      ASSERT_EQ(std::ranges::count_if(
                    probes, [i](const IdleCpuProbe *probe) { return probe->partition_id() == i; }),
                1);

    std::thread runner([&]() { engine.run(); });
    const bool started = engine.wait_until_started();

    const auto sample_cpu = [&](uint64_t &total) {
      for (std::atomic<uint64_t> &sample : samples)
        sample.store(0, std::memory_order_relaxed);
      for (IdleCpuProbe *probe : probes)
        engine.schedule_event_now(probe->sample_event());
      const std::chrono::steady_clock::time_point deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (std::chrono::steady_clock::now() < deadline) {
        total = 0;
        bool ready = true;
        for (const std::atomic<uint64_t> &sample : samples) {
          const uint64_t value = sample.load(std::memory_order_acquire);
          if (value == UINT64_MAX)
            return false;
          ready &= value != 0;
          total += value;
        }
        if (ready)
          return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      return false;
    };
    uint64_t cpu_before = 0, cpu_after = 0;
    const bool sampled_before = started && sample_cpu(cpu_before);
    if (sampled_before)
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const bool sampled_after = sampled_before && sample_cpu(cpu_after);

    engine.request_exit("test stop");
    runner.join();

    ASSERT_TRUE(started);
    ASSERT_TRUE(sampled_before);
    ASSERT_TRUE(sampled_after);
    ASSERT_GE(cpu_after, cpu_before);
    const double cpu_used = static_cast<double>(cpu_after - cpu_before) / 1'000'000'000;
    // Even one spinning worker burns about 0.2s of CPU over this window; idle
    // workers burn milliseconds. Leave room for startup and scheduling noise.
    EXPECT_LT(cpu_used, 0.05) << "idle engine consumed " << cpu_used << "s of CPU in 0.2s";
  }
#endif
}

TEST(TerminationTest, StepModeConsistency) {
  util::StringDiagnostic admission_diagnostic;
  SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *c = root->add_child(std::make_unique<CounterComponent>("c0", 10));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();

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
  ASSERT_NO_FATAL_FAILURE(build_with_manual_partitions(engine, 2, partition_by_name_suffix));

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
  ASSERT_NO_FATAL_FAILURE(build_with_manual_partitions(engine, 3, partition_by_name_suffix));

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
  ASSERT_NO_FATAL_FAILURE(build_with_manual_partitions(engine, 2, partition_by_name_suffix));

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
  ASSERT_NO_FATAL_FAILURE(build_with_manual_partitions(engine, 2, partition_by_name_suffix));

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
  ASSERT_NO_FATAL_FAILURE(build_with_manual_partitions(engine, 2, partition_by_name_suffix));

  engine.run();
  ASSERT_EQ(cons->received.size(), 1u);
  EXPECT_EQ(cons->received[0].first, 142u); // 100 + 42.
}

// ============================================================================
// Area 3: Async Event Causality Tests
// ============================================================================

TEST(AsyncCausalityTest, IdlePeerUsesLastCommittedTickAsAsyncFloor) {
  for (const Tick horizon : {Tick{0}, Tick{100}}) {
    SCOPED_TRACE(horizon);
    // Repeat because the peer may drain before or after the posting handler.
    for (unsigned repetition = 0; repetition < 16; ++repetition) {
      SimulationEngine engine({.max_ticks = horizon, .num_threads = 2});
      auto root = std::make_unique<CompositeComponent>("root");
      Component *primary = root->add_child(std::make_unique<AsyncPrimaryComponent>("primary0"));
      Component *peer = root->add_child(std::make_unique<CompositeComponent>("peer1"));
      engine.topology().set_root(std::move(root));
      ASSERT_NO_FATAL_FAILURE(build_with_manual_partitions(engine, 2, partition_by_name_suffix));

      Tick callback_tick = TICK_MAX;
      Event callback(peer, EventType::TIMER_CALLBACK,
                     [&](Tick tick, Message *) { callback_tick = tick; });
      Event post(primary, EventType::TIMER_CALLBACK, [&](Tick, Message *) {
        engine.schedule_event_async(&callback, 5);
        engine.primary_release();
      });
      engine.schedule_event_async(&post, 10);

      const ExitStatus exit = engine.run();
      EXPECT_EQ(callback_tick, 10u);
      EXPECT_EQ(exit.cause, ExitCause::ALL_PRIMARIES_COMPLETED);
      EXPECT_EQ(exit.tick, 10u);
    }
  }
}

TEST(AsyncCausalityTest, ScheduleEventNowProducesReasonableTimestamp) {
  util::StringDiagnostic admission_diagnostic;
  // Single-threaded, no pacing: schedule_event_now should use GVT (current_time_).
  SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *c = root->add_child(std::make_unique<CounterComponent>("c0", 5));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();

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

// Regression: draining a handler's async self-post can leave its wake flag set.
// Clearing that stale notification must acquire any concurrent producer's
// publication, or the worker can sleep with an event stranded in the async queue.
TEST(AsyncCausalityTest, StaleWakeDoesNotHideConcurrentPublication) {
  util::StringDiagnostic admission_diagnostic;
  constexpr uint32_t kEvents = 2000;
  SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *target = root->add_child(std::make_unique<AsyncPrimaryComponent>("target"));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();

  std::atomic<uint32_t> processed{0};
  Event noop{target, EventType::TIMER_CALLBACK, [](Tick, Message *) {}};
  Event test_event{target, EventType::TIMER_CALLBACK, [&](Tick, Message *) {
                     // Self-posting leaves a stale wake flag after the no-op is
                     // drained, so idle entry can race the next publication.
                     engine.schedule_event_now(&noop);
                     if (processed.fetch_add(1, std::memory_order_release) + 1 == kEvents)
                       engine.primary_release();
                   }};
  ExitStatus exit_status;
  std::thread runner([&]() { exit_status = engine.run(); });
  const bool started = engine.wait_until_started();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  uint32_t sent = 0;
  if (started) {
    // Only one producer event may be outstanding: a later publication must not
    // rescue a lost wake.
    while (sent < kEvents && std::chrono::steady_clock::now() < deadline) {
      engine.schedule_event_now(&test_event);
      ++sent;
      while (processed.load(std::memory_order_acquire) < sent &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    }
  }
  const bool completed = processed.load(std::memory_order_acquire) == kEvents;
  if (!completed)
    engine.request_exit("async notification timeout", /*code=*/1);
  runner.join();
  EXPECT_TRUE(started);
  EXPECT_TRUE(completed) << "sent " << sent << ", processed " << processed.load();
  EXPECT_EQ(exit_status.reason, ExitReason::COMPLETED);
}

// Successive quiescent barrier completions can run on different workers, but
// they share one notification consumer. Exercise publications to every partition
// while self-posts leave stale notifications; a later producer cannot rescue a
// stranded event because at most one producer event is outstanding.
TEST(AsyncCausalityTest, QuiescentPublicationsToEveryPartitionMakeProgress) {
  constexpr uint32_t kEvents = 2000;
  for (const uint32_t partitions : {2u, 4u}) {
    util::StringDiagnostic admission_diagnostic;
    SCOPED_TRACE(partitions);
    SimulationEngine engine({.num_threads = partitions});
    auto root = std::make_unique<CompositeComponent>("root");
    std::vector<Component *> targets;
    for (uint32_t i = 0; i < partitions; ++i)
      targets.push_back(
          root->add_child(std::make_unique<AsyncPrimaryComponent>("target" + std::to_string(i))));
    engine.topology().set_root(std::move(root));
    ASSERT_TRUE(
        engine.topology()
            .partition_manual(partitions, partition_by_name_suffix, admission_diagnostic.emitter())
            .succeeded())
        << admission_diagnostic.message();
    ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
        << admission_diagnostic.message();

    std::atomic<uint32_t> processed{0};
    std::vector<std::unique_ptr<Event>> noops;
    std::vector<std::unique_ptr<Event>> events;
    for (uint32_t i = 0; i < partitions; ++i) {
      noops.push_back(
          std::make_unique<Event>(targets[i], EventType::TIMER_CALLBACK, [](Tick, Message *) {}));
      events.push_back(
          std::make_unique<Event>(targets[i], EventType::TIMER_CALLBACK, [&, i](Tick, Message *) {
            engine.schedule_event_now(noops[i].get());
            processed.fetch_add(1, std::memory_order_release);
          }));
    }
    ExitStatus exit_status;
    std::thread runner([&]() { exit_status = engine.run(); });
    const bool started = engine.wait_until_started();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    uint32_t sent = 0;
    if (started) {
      while (sent < kEvents && std::chrono::steady_clock::now() < deadline) {
        engine.schedule_event_now(events[sent % partitions].get());
        ++sent;
        while (processed.load(std::memory_order_acquire) < sent &&
               std::chrono::steady_clock::now() < deadline)
          std::this_thread::yield();
      }
    }
    const bool completed = processed.load(std::memory_order_acquire) == kEvents;
    if (completed) {
      for (uint32_t i = 0; i < partitions; ++i)
        engine.primary_release();
    } else {
      engine.request_exit("multi-partition notification timeout", 1);
    }
    runner.join();
    EXPECT_TRUE(started);
    EXPECT_TRUE(completed) << "sent " << sent << ", processed " << processed.load();
    EXPECT_EQ(exit_status.reason, ExitReason::COMPLETED);
  }
}

TEST(AsyncCausalityTest, SustainedSameTickAsyncAndExistingBatchMakeBoundedProgress) {
  util::StringDiagnostic admission_diagnostic;
  constexpr uint32_t kBatchSize = 1024;
  constexpr uint32_t kAsyncCount = 32;
  SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *component = static_cast<SameTickAsyncFairnessComponent *>(
      root->add_child(std::make_unique<SameTickAsyncFairnessComponent>("same-tick-fairness",
                                                                       kBatchSize, kAsyncCount)));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();

  ExitStatus exit = engine.run();

  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_EQ(component->async_events_processed(), kAsyncCount);
  EXPECT_EQ(component->batch_events_processed(), kBatchSize);
  EXPECT_EQ(component->first_async_tick(), 1u)
      << "schedule_event_now moved simulation time backward";
  EXPECT_EQ(component->batch_events_before_first_async(), 0u)
      << "an async event injected by the first handler waited behind the whole same-tick batch";
  EXPECT_LE(component->async_events_before_first_batch_event(),
            EventQueue::kMaxConsecutiveAsyncEvents)
      << "a self-replenishing async stream starved pre-existing same-tick local work";
}

TEST(AsyncCausalityTest, StepUsesSameBoundedAsyncArbitration) {
  util::StringDiagnostic admission_diagnostic;
  constexpr uint32_t kBatchSize = 32;
  constexpr uint32_t kAsyncCount = EventQueue::kMaxConsecutiveAsyncEvents + 1;
  SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *component = static_cast<SameTickAsyncFairnessComponent *>(root->add_child(
      std::make_unique<SameTickAsyncFairnessComponent>("step-fairness", kBatchSize, kAsyncCount)));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();

  ASSERT_TRUE(engine.step());

  EXPECT_EQ(component->async_events_processed(), kAsyncCount);
  EXPECT_EQ(component->batch_events_processed(), kBatchSize);
  EXPECT_EQ(component->batch_events_before_first_async(), 0u);
  EXPECT_LE(component->async_events_before_first_batch_event(),
            EventQueue::kMaxConsecutiveAsyncEvents);
}

TEST(AsyncCausalityTest, NextTickRetryDoesNotStarveScheduledDeviceWork) {
  constexpr uint32_t kRetryCount = 32;
  for (uint32_t num_threads : {1u, 2u}) {
    util::StringDiagnostic admission_diagnostic;
    SCOPED_TRACE(::testing::Message() << "num_threads=" << num_threads);
    SimulationEngine engine({.num_threads = num_threads});
    auto root = std::make_unique<CompositeComponent>("root");
    auto *component = static_cast<NextTickRetryFairnessComponent *>(root->add_child(
        std::make_unique<NextTickRetryFairnessComponent>("next-tick-retry", kRetryCount)));
    engine.topology().set_root(std::move(root));
    if (num_threads > 1) {

      ASSERT_TRUE(engine.topology()
                      .partition_manual(
                          num_threads, [](Component *) { return PartitionID{0}; },
                          admission_diagnostic.emitter())
                      .succeeded())
          << admission_diagnostic.message();
    }
    ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
        << admission_diagnostic.message();

    ExitStatus exit = engine.run();

    EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
    EXPECT_EQ(component->retries_processed(), kRetryCount);
    EXPECT_EQ(component->local_tick(), 2u);
    EXPECT_LE(component->retries_before_local(), EventQueue::kMaxConsecutiveAsyncEvents)
        << "a level-triggered retry stream starved device work at the following tick";
  }
}

TEST(AsyncCausalityTest, NextTickTimestampSaturatesAtTickMax) {
  util::StringDiagnostic admission_diagnostic;
  SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *component = static_cast<NextTickSaturationComponent *>(
      root->add_child(std::make_unique<NextTickSaturationComponent>("next-tick-saturation")));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();

  ExitStatus exit = engine.run();

  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  EXPECT_EQ(component->observed_tick(), TICK_MAX);
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
  ASSERT_NO_FATAL_FAILURE(build_with_manual_partitions(engine, 4, partition_by_name_suffix));

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
  ASSERT_NO_FATAL_FAILURE(build_with_manual_partitions(engine, 2, partition_by_name_suffix));

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
  util::StringDiagnostic admission_diagnostic;
  // Use InfiniteComponent to keep the simulation running while we inject.
  // max_ticks as safety net. Keep low since each tick = one barrier round.
  SimulationEngine engine({.max_ticks = 500, .num_threads = 2});
  auto root = std::make_unique<CompositeComponent>("root");
  root->add_child(std::make_unique<InfiniteComponent>("inf0"));
  root->add_child(std::make_unique<InfiniteComponent>("inf1"));
  engine.topology().set_root(std::move(root));
  ASSERT_TRUE(engine.topology().partition_balanced(2, admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();
  ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
      << admission_diagnostic.message();

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
// Cache storage, LRU replacement, maintenance, and VMID invariants
// ============================================================================
//
// Exercise storage lifetime, replacement order, maintenance behavior, and VMID isolation
// directly on the header-only Cache.

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

template <uint32_t Ways> void check_lru_way_indices() {
  LRUPolicy<2, Ways> policy;
  for (uint32_t way = 0; way < Ways; ++way) {
    EXPECT_EQ(policy.victim(0), way);
    policy.access(0, way);
    EXPECT_EQ(policy.victim(1), 0u);
  }
  // Copy a non-default order so a fresh policy cannot pass as a copied one.
  policy.access(0, 0);
  LRUPolicy<2, Ways> copied = policy;
  for (uint32_t way = 0; way < Ways; ++way) {
    const uint32_t victim = (way + 1) % Ways;
    EXPECT_EQ(copied.victim(0), victim);
    copied.access(0, victim);
    EXPECT_EQ(policy.victim(0), Ways > 1 ? 1u : 0u);
    EXPECT_EQ(policy.victim(1), 0u);
    EXPECT_EQ(copied.victim(1), 0u);
  }
}

template <uint32_t LineSizeBits> void check_valid_byte_range_updates() {
  using RangeCache = Cache<LineSizeBits, 1, 1>;
  constexpr std::array<std::array<uint64_t, 2>, 6> initial_masks = {{
      {0, 0},
      {~uint64_t{0}, ~uint64_t{0}},
      {0, ~uint64_t{0}},
      {~uint64_t{0}, 0},
      {0xA5A5A5A5A5A5A5A5, 0x5A5A5A5A5A5A5A5A},
      {0x0123456789ABCDEF, 0xFEDCBA9876543210},
  }};
  for (const auto &initial : initial_masks) {
    std::array<bool, 128> initial_bytes{};
    for (uint32_t byte = 0; byte < initial_bytes.size(); ++byte)
      initial_bytes[byte] = (initial[byte / 64] >> (byte % 64)) & 1;
    for (uint32_t offset = 0; offset <= RangeCache::LINE_SIZE; ++offset) {
      for (uint32_t size = 0; size <= RangeCache::LINE_SIZE - offset; ++size) {
        CacheTag tag;
        tag.valid_bytes = initial;
        auto expected_bytes = initial_bytes;
        bool expected_valid = true;
        for (uint32_t byte = offset; byte < offset + size; ++byte) {
          expected_valid &= expected_bytes[byte];
          expected_bytes[byte] = true;
        }
        ASSERT_EQ(RangeCache::bytes_valid(tag, offset, size), expected_valid)
            << "line=" << RangeCache::LINE_SIZE << " offset=" << offset << " size=" << size;
        ASSERT_EQ(tag.valid_bytes, initial);
        RangeCache::mark_valid_bytes(tag, offset, size);
        // Inspect the full two-word mask, including bits beyond smaller line
        // geometries: marking this range must leave those bits unchanged.
        for (uint32_t byte = 0; byte < expected_bytes.size(); ++byte) {
          const bool observed = (tag.valid_bytes[byte / 64] >> (byte % 64)) & 1;
          ASSERT_EQ(observed, expected_bytes[byte])
              << "line=" << RangeCache::LINE_SIZE << " offset=" << offset << " size=" << size
              << " byte=" << byte;
        }
        ASSERT_TRUE(RangeCache::bytes_valid(tag, offset, size));
      }
    }
  }
}

template <uint32_t LineSizeBits> void check_valid_byte_queries_with_one_missing_byte() {
  using RangeCache = Cache<LineSizeBits, 1, 1>;
  for (uint32_t missing = 0; missing < RangeCache::LINE_SIZE; ++missing) {
    CacheTag tag;
    tag.valid_bytes[missing / 64] &= ~(uint64_t{1} << (missing % 64));
    const auto initial = tag.valid_bytes;
    for (uint32_t offset = 0; offset <= RangeCache::LINE_SIZE; ++offset) {
      for (uint32_t size = 0; size <= RangeCache::LINE_SIZE - offset; ++size) {
        const bool expected_valid = missing < offset || missing >= offset + size;
        ASSERT_EQ(RangeCache::bytes_valid(tag, offset, size), expected_valid)
            << "line=" << RangeCache::LINE_SIZE << " offset=" << offset << " size=" << size
            << " missing=" << missing;
      }
    }
    ASSERT_EQ(tag.valid_bytes, initial);
  }
}
} // namespace

TEST(CacheLruTest, WayIndicesCoverByteBoundary) {
  check_lru_way_indices<1>();
  check_lru_way_indices<16>();
  check_lru_way_indices<256>();
  check_lru_way_indices<257>();
}

TEST(CacheStorageTest, InitialBytesAreZeroAndInvalidationRetainsData) {
  TestCache cache;
  for (uint32_t set = 0; set < 4; ++set) {
    for (uint32_t way = 0; way < 2; ++way) {
      const uint64_t addr = (set + way * 4) * TestCache::LINE_SIZE;
      CacheTag evicted;
      evicted.valid = true;
      TestCache::Allocation allocation = cache.allocate_with_data(addr, /*vmid=*/0, &evicted);
      EXPECT_FALSE(evicted.valid);
      ASSERT_NE(allocation.data, nullptr);
      for (uint32_t i = 0; i < TestCache::LINE_SIZE; ++i)
        EXPECT_EQ(allocation.data[i], 0);
      std::ranges::fill_n(allocation.data, TestCache::LINE_SIZE, 0xA5);
    }
  }
  cache.invalidate_all();
  TestCache invalid_copy(cache);
  for (auto *retained : {&cache, &invalid_copy}) {
    TestCache::Allocation allocation = retained->allocate_with_data(0);
    for (uint32_t i = 0; i < TestCache::LINE_SIZE; ++i)
      EXPECT_EQ(allocation.data[i], 0xA5);
  }
}

TEST(CacheStorageTest, CopyIsIndependentAndMoveRetainsContents) {
  TestCache original;
  fill_line_word(original, 0x4000, 7, 0x12345678);
  TestCache copied(original);
  EXPECT_EQ(read_line_word(copied, 0x4000, 7), 0x12345678u);
  const uint32_t replacement = 0x87654321;
  copied.write_line(0x4000, reinterpret_cast<const uint8_t *>(&replacement), 0, sizeof(replacement),
                    7);
  EXPECT_EQ(read_line_word(original, 0x4000, 7), 0x12345678u);

  TestCache assigned;
  auto *assigned_bytes = assigned.allocate_with_data(0x4000, 7).data;
  assigned = copied;
  EXPECT_EQ(read_line_word(assigned, 0x4000, 7), replacement);
  EXPECT_EQ(assigned.line_data_for_read(0x4000, 7), assigned_bytes);
  TestCache moved(std::move(assigned));
  EXPECT_EQ(read_line_word(moved, 0x4000, 7), replacement);
  TestCache move_assigned;
  move_assigned = std::move(moved);
  EXPECT_EQ(read_line_word(move_assigned, 0x4000, 7), replacement);
  // Assignment must also restore a moved-from cache to usable storage.
  assigned = original;
  EXPECT_EQ(read_line_word(assigned, 0x4000, 7), 0x12345678u);
}

TEST(CacheStorageTest, PartialLineTracksValidBytes) {
  TestCache cache;
  constexpr uint64_t kAddr = 0x5000;
  constexpr uint32_t kOffset = 61;
  constexpr std::array<uint8_t, 3> kBytes = {0x12, 0x34, 0x56};

  CacheTag *tag = cache.allocate(kAddr, /*vmid=*/7);
  TestCache::clear_valid_bytes(*tag);
  EXPECT_FALSE(TestCache::bytes_valid(*tag, kOffset, kBytes.size()));

  cache.write_line(kAddr, kBytes.data(), kOffset, kBytes.size(), /*vmid=*/7);
  EXPECT_TRUE(TestCache::bytes_valid(*tag, kOffset, kBytes.size()));
  EXPECT_FALSE(TestCache::bytes_valid(*tag, kOffset - 1, kBytes.size() + 1));

  std::array<uint8_t, kBytes.size()> observed{};
  cache.read_line(kAddr, observed.data(), kOffset, observed.size(), /*vmid=*/7);
  EXPECT_EQ(observed, kBytes);
}

TEST(CacheStorageTest, ValidByteRangesMatchIndependentByteOracle) {
  check_valid_byte_range_updates<0>();
  check_valid_byte_range_updates<1>();
  check_valid_byte_range_updates<2>();
  check_valid_byte_range_updates<3>();
  check_valid_byte_range_updates<4>();
  check_valid_byte_range_updates<5>();
  check_valid_byte_range_updates<6>();
  check_valid_byte_range_updates<7>();
}

TEST(CacheStorageTest, ValidByteQueriesDetectEveryMissingByte) {
  check_valid_byte_queries_with_one_missing_byte<0>();
  check_valid_byte_queries_with_one_missing_byte<1>();
  check_valid_byte_queries_with_one_missing_byte<2>();
  check_valid_byte_queries_with_one_missing_byte<3>();
  check_valid_byte_queries_with_one_missing_byte<4>();
  check_valid_byte_queries_with_one_missing_byte<5>();
  check_valid_byte_queries_with_one_missing_byte<6>();
  check_valid_byte_queries_with_one_missing_byte<7>();
}

TEST(CacheMaintenanceTest, FullInvalidationClearsRetainedMetadataAfterRefill) {
  TestCache cache;
  for (uint32_t round = 0; round < 4; ++round) {
    const uint64_t addr = round * TestCache::LINE_SIZE;
    auto *tag = cache.allocate(addr, round + 1);
    tag->dirty = true;
    tag->coherence = CoherenceState::MODIFIED;
    tag->coherence_epoch = 17;

    cache.invalidate_all();
    EXPECT_FALSE(cache.lookup(addr, nullptr, round + 1));
    EXPECT_FALSE(tag->valid);
    EXPECT_FALSE(tag->dirty);
    EXPECT_EQ(tag->coherence, CoherenceState::INVALID);
    EXPECT_EQ(tag->coherence_epoch, 0u);
    cache.invalidate_all();
  }
}

TEST(CacheMaintenanceTest, FullInvalidationVisitsSparseAndBoundarySets) {
  Cache<6, 32, 2> cache;
  constexpr std::array<uint32_t, 5> sets{0, 7, 8, 24, 31};
  std::array<CacheTag *, sets.size()> retained{};
  for (uint32_t round = 0; round != 3; ++round) {
    for (size_t index = 0; index != sets.size(); ++index) {
      const uint64_t address = uint64_t{sets[index]} * 64;
      auto *tag = cache.allocate(address, round);
      tag->dirty = true;
      tag->coherence_epoch = 17;
      tag->coherence = CoherenceState::MODIFIED;
      retained[index] = tag;
      if (index % 2 == 0)
        cache.invalidate(address, round);
    }
    cache.invalidate_all();
    for (size_t index = 0; index != sets.size(); ++index) {
      EXPECT_FALSE(cache.lookup(uint64_t{sets[index]} * 64, nullptr, round));
      EXPECT_FALSE(retained[index]->valid);
      EXPECT_FALSE(retained[index]->dirty);
      EXPECT_EQ(retained[index]->coherence_epoch, 0u);
      EXPECT_EQ(retained[index]->coherence, CoherenceState::INVALID);
    }
    cache.invalidate_all();
  }
}

TEST(CacheMaintenanceTest, DirtyWalkPreservesOrderAcrossReplacementAndInvalidation) {
  TestCache cache;
  constexpr uint64_t kLineSize = TestCache::LINE_SIZE;
  auto fill_dirty = [&](uint64_t addr, uint32_t vmid) {
    fill_line_word(cache, addr, vmid, vmid);
    CacheTag *tag = nullptr;
    ASSERT_TRUE(cache.lookup(addr, &tag, vmid));
    tag->dirty = true;
  };
  // Set three is visited after set one regardless of allocation order.
  fill_dirty(3 * kLineSize, 3);
  fill_dirty(kLineSize, 1);
  fill_dirty(5 * kLineSize, 5);
  // Replace set one's LRU way without allocating into an invalid way.
  fill_dirty(9 * kLineSize, 9);

  auto dirty_addresses = [&] {
    std::vector<uint64_t> addresses;
    cache.for_each_dirty([&](CacheTag &tag, uint64_t addr, uint8_t *data) {
      addresses.push_back(addr);
      uint32_t word = 0;
      std::memcpy(&word, data, sizeof(word));
      EXPECT_EQ(tag.vmid, word);
    });
    return addresses;
  };
  EXPECT_EQ(dirty_addresses(),
            (std::vector<uint64_t>{9 * kLineSize, 5 * kLineSize, 3 * kLineSize}));
  cache.invalidate(5 * kLineSize, 5);
  EXPECT_EQ(dirty_addresses(), (std::vector<uint64_t>{9 * kLineSize, 3 * kLineSize}));
  cache.invalidate_all();
  EXPECT_TRUE(dirty_addresses().empty());
  fill_dirty(kLineSize, 11);
  EXPECT_EQ(dirty_addresses(), (std::vector<uint64_t>{kLineSize}));
}

TEST(CacheMaintenanceTest, CopiesAndMovesRetainDirtyAndInvalidationState) {
  TestCache original;
  auto allocation = original.allocate_with_data(2 * TestCache::LINE_SIZE, 7);
  allocation.data[0] = 42;
  allocation.tag->dirty = true;
  TestCache copied(original);
  TestCache assigned;
  assigned.allocate(0)->dirty = true;
  assigned = original;
  TestCache moved(std::move(copied));
  original.invalidate_all();

  for (auto *cache : {&assigned, &moved}) {
    uint32_t count = 0;
    cache->for_each_dirty([&](CacheTag &tag, uint64_t addr, uint8_t *data) {
      ++count;
      EXPECT_EQ(addr, 2u * TestCache::LINE_SIZE);
      EXPECT_EQ(tag.vmid, 7u);
      EXPECT_EQ(data[0], 42);
    });
    EXPECT_EQ(count, 1u);
    cache->invalidate_all();
    EXPECT_FALSE(cache->lookup(2 * TestCache::LINE_SIZE, nullptr, 7));
    cache->for_each_dirty([](CacheTag &, uint64_t, uint8_t *) { ADD_FAILURE(); });
  }
}

// The memory hierarchy tags every line by (vmid, addr) so two processes that
// alias the same guest VA do not share a cached line. These tests exercise that
// invariant directly on the header-only Cache: distinct data per VMID at the
// same address, eviction reporting the evicted line's owner VMID, and per-VMID
// invalidation.

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
  std::ranges::fill_n(first.data, TestCache::LINE_SIZE, 0xA5);
  cache.allocate(kAddrB, /*vmid=*/8);

  CacheTag evicted;
  std::array<uint8_t, TestCache::LINE_SIZE> evicted_data{};
  auto replacement = cache.allocate_with_data(kAddrC, /*vmid=*/9, &evicted, evicted_data.data());

  ASSERT_NE(replacement.tag, nullptr);
  ASSERT_NE(replacement.data, nullptr);
  EXPECT_TRUE(evicted.valid);
  EXPECT_TRUE(evicted.dirty);
  EXPECT_EQ(evicted.vmid, 7u);
  EXPECT_TRUE(std::ranges::all_of(evicted_data, [](uint8_t byte) { return byte == 0xA5; }));
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
