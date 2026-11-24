/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "suites/functional/signal_event_exhaustion.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "common/base_rocr_utils.h"
#include "gtest/gtest.h"
#include "hsa/amd_hsa_signal.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

namespace {

// KFD_SIGNAL_EVENT_LIMIT is 4096; allow for events the runtime and other
// tests already hold, and stop well before an unbounded allocation.
constexpr int kMaxSignalsToExhaust = 16384;
constexpr int kSamples = 200;
constexpr int kRounds = 3;
constexpr auto kInterval = std::chrono::microseconds(1000);
constexpr auto kObserveTimeout = std::chrono::seconds(1);

int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

uint32_t EventId(hsa_signal_t s) {
  // The low handle bits carry hints; the rest is the amd_signal_t address.
  return reinterpret_cast<const amd_signal_t*>(static_cast<uintptr_t>(s.handle & ~uint64_t(0x3F)))
      ->event_id;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double Percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(p * v.size()))];
}

// Async handler: records the first time it sees the completion, then re-arms
// the signal itself. It stays registered (returns true), so if it left the
// value at 0 it would run again on every pass of the async-events loop until
// the test thread re-armed it.
bool OnCompletion(hsa_signal_value_t, void* arg) {
  auto* t = static_cast<SignalEventExhaustionTest::Target*>(arg);
  int64_t pending = 0;
  t->seen_ns.compare_exchange_strong(pending, NowNs());
  if (t->stop.load()) return false;
  hsa_signal_store_screlease(t->signal, 1);
  return true;
}

}  // namespace

SignalEventExhaustionTest::SignalEventExhaustionTest()
    : TestBase(), queue_(nullptr), first_event_id_(0), timestamp_frequency_(0) {
  set_num_iteration(1);
  set_title("RocR Signal Event Exhaustion Test");
  set_description(
      "Uses up the process's KFD signal events, then checks that async handlers and "
      "blocked waiters on signals created afterwards are woken promptly while the "
      "process stays idle.");
}

SignalEventExhaustionTest::~SignalEventExhaustionTest() {}

void SignalEventExhaustionTest::SetUp(void) {
  TestBase::SetUp();
  if (test_skipped_) return;
  ASSERT_EQ(HSA_STATUS_SUCCESS, rocrtst::SetDefaultAgents(this));
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &timestamp_frequency_));
  ASSERT_GT(timestamp_frequency_, 0u);

  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_signal_create(1, 0, nullptr, 0, &helper_.signal));
  first_event_id_ = EventId(helper_.signal);
  if (first_event_id_ == 0) {
    SkipAfterInit("interrupt signals have no KFD event (interrupts disabled)");
    return;
  }
  // Queue creation itself needs KFD events, so create the queue for the GPU
  // completion test before they are used up.
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_queue_create(*gpu_device1(), 64, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
                             UINT32_MAX, UINT32_MAX, &queue_));
  if (!ExhaustEvents()) {
    // An environment precondition (e.g. a KFD with a larger event limit), not a product bug.
    SkipAfterInit("KFD signal events were not exhausted within the signal limit");
    return;
  }
  // The helper keeps a signal with its own event in the async batch next to the
  // post-exhaustion ones, as in a real process. It is only completed in Close().
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_amd_signal_async_handler(helper_.signal, HSA_SIGNAL_CONDITION_LT, 1, OnCompletion,
                                         &helper_));
  handlers_.push_back(&helper_);
}

void SignalEventExhaustionTest::Run(void) { TestBase::Run(); }

void SignalEventExhaustionTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void SignalEventExhaustionTest::DisplayResults(void) const { TestBase::DisplayResults(); }

void SignalEventExhaustionTest::SkipAfterInit(const char* reason) {
  ReleaseResources();
  TestBase::Close();
  test_skipped_ = true;
  // The bundled gtest predates GTEST_SKIP(); report the skip like the other
  // rocrtst tests that set test_skipped_.
  std::cout << "[  SKIPPED ] " << reason << std::endl;
}

void SignalEventExhaustionTest::Close() {
  ReleaseResources();
  TestBase::Close();
}

void SignalEventExhaustionTest::ReleaseResources() {
  // Let every handler return false once, so none is left registered on a
  // signal that is about to be destroyed.
  for (Target* t : handlers_) {
    t->stop.store(true);
    t->seen_ns.store(0);
    hsa_signal_store_screlease(t->signal, 0);
    const auto deadline = std::chrono::steady_clock::now() + kObserveTimeout;
    while (t->seen_ns.load() == 0 && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    EXPECT_NE(0, t->seen_ns.load()) << "async handler did not run on shutdown";
  }
  handlers_.clear();
  if (queue_ != nullptr) {
    EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_queue_destroy(queue_));
    queue_ = nullptr;
  }
  for (auto& t : targets_) EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_signal_destroy(t->signal));
  targets_.clear();
  for (hsa_signal_t s : held_) EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_signal_destroy(s));
  held_.clear();
  if (helper_.signal.handle != 0) {
    EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_signal_destroy(helper_.signal));
    helper_.signal.handle = 0;
  }
}

bool SignalEventExhaustionTest::ExhaustEvents() {
  for (int i = 0; i < kMaxSignalsToExhaust; ++i) {
    hsa_signal_t s;
    if (hsa_amd_signal_create(1, 0, nullptr, 0, &s) != HSA_STATUS_SUCCESS) return false;
    held_.push_back(s);
    const uint32_t id = EventId(s);
    if (id == 0) return true;
    if (held_.size() > 1 && id == EventId(held_[held_.size() - 2])) return true;
  }
  return false;
}

SignalEventExhaustionTest::Target* SignalEventExhaustionTest::NewHandlerTarget() {
  targets_.emplace_back(new Target());
  Target* t = targets_.back().get();
  EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_amd_signal_create(1, 0, nullptr, 0, &t->signal));
  EXPECT_EQ(HSA_STATUS_SUCCESS,
            hsa_amd_signal_async_handler(t->signal, HSA_SIGNAL_CONDITION_LT, 1, OnCompletion, t));
  handlers_.push_back(t);
  if (verbosity() > 0) {
    std::cout << "  events: first=" << first_event_id_ << " exhausted after " << held_.size()
              << " signals, target event_id=" << EventId(t->signal) << std::endl;
  }
  return t;
}

double SignalEventExhaustionTest::Complete(Target* t, bool gpu) {
  t->seen_ns.store(0);
  t->start_ns.store(NowNs());
  if (gpu) {
    // A barrier packet with only a completion signal: the CP decrements the
    // signal (1 -> 0) and raises the end-of-pipe interrupt for its event.
    const uint64_t index = hsa_queue_add_write_index_scacq_screl(queue_, 1);
    while (index - hsa_queue_load_read_index_scacquire(queue_) >= queue_->size)
      std::this_thread::yield();
    auto* pkt =
        reinterpret_cast<hsa_barrier_and_packet_t*>(queue_->base_address) + (index % queue_->size);
    memset(reinterpret_cast<uint8_t*>(pkt) + 4, 0, sizeof(*pkt) - 4);
    pkt->completion_signal = t->signal;
    const uint32_t header = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
        (1 << HSA_PACKET_HEADER_BARRIER) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n(reinterpret_cast<uint32_t*>(pkt), header, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(queue_->doorbell_signal, index);
  } else {
    hsa_signal_store_screlease(t->signal, 0);
  }
  const auto deadline = std::chrono::steady_clock::now() + kObserveTimeout;
  int64_t seen;
  while ((seen = t->seen_ns.load()) == 0) {
    if (std::chrono::steady_clock::now() > deadline) return -1;
    std::this_thread::yield();
  }
  return (seen - t->start_ns.load()) / 1e3;
}

double SignalEventExhaustionTest::ProcessCpuPercent(int ms) {
  auto cpu_s = [] {
    rusage r{};
    getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6 + r.ru_stime.tv_sec +
        r.ru_stime.tv_usec * 1e-6;
  };
  const double c0 = cpu_s();
  const auto w0 = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
  return 100 * (cpu_s() - c0) / wall;
}

/*
 * Test Name: TestAsyncHandlerLatency
 * Scope: Functional / Performance regression
 *
 * Purpose: A CPU store to a signal created after the KFD events ran out must
 * reach its async handler without waiting out a polling nap. With no event,
 * the async-events thread either spins (sub-microsecond, one full core) or
 * polls with 20-200 us naps (~200 us median); with an event the handler runs
 * after one interrupt wake-up (~5-15 us on an idle desktop).
 *
 * Expected Results: Every completion reaches the handler within 1 s, and the
 * median latency of at least one of 3 rounds of 200 completions, 1 ms apart,
 * is below 100 us.
 */
void SignalEventExhaustionTest::TestAsyncHandlerLatency(void) {
  Target* t = NewHandlerTarget();
  double best = 0;
  for (int round = 0; round < kRounds; ++round) {
    std::vector<double> us;
    for (int i = 0; i < kSamples; ++i) {
      std::this_thread::sleep_for(kInterval);
      const double lat = Complete(t, false);
      ASSERT_GE(lat, 0) << "completion " << i << " never reached the async handler";
      us.push_back(lat);
    }
    const double median = Median(us);
    if (verbosity() > 0) {
      std::cout << "  round " << round << ": handler latency p50=" << median
                << "us p99=" << Percentile(us, 0.99) << "us max=" << Percentile(us, 1.0) << "us"
                << std::endl;
    }
    best = (round == 0) ? median : std::min(best, median);
    if (best < 100) break;
  }
  EXPECT_LT(best, 100) << "async handler on a post-exhaustion signal: best median " << best
                       << " us in " << kRounds << " rounds";
}

/*
 * Test Name: TestAsyncHandlerIdleCpu
 * Scope: Functional / Performance regression
 *
 * Purpose: While an async handler on a post-exhaustion signal is pending and
 * nothing completes, the async-events thread must sleep. A spinning loop
 * costs one full core per process.
 *
 * Expected Results: Process CPU over 500 ms is below 20% in at least one of 3
 * measurements, and the handler still runs when the signal completes.
 */
void SignalEventExhaustionTest::TestAsyncHandlerIdleCpu(void) {
  Target* t = NewHandlerTarget();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double best = 0;
  for (int round = 0; round < kRounds; ++round) {
    const double cpu = ProcessCpuPercent(500);
    if (verbosity() > 0)
      std::cout << "  round " << round << ": idle process CPU " << cpu << "%" << std::endl;
    best = (round == 0) ? cpu : std::min(best, cpu);
    if (best < 20) break;
  }
  EXPECT_LT(best, 20) << "idle process CPU with a pending post-exhaustion async handler";
  EXPECT_GE(Complete(t, false), 0) << "completion never reached the async handler";
}

/*
 * Test Name: TestBlockedWaiters
 * Scope: Functional / Performance regression
 *
 * Purpose: Threads blocked in hsa_signal_wait_scacquire(HSA_WAIT_STATE_BLOCKED)
 * on post-exhaustion signals must sleep, and a completion of one of them, or
 * of an async-handled signal, must wake the right waiter even when the signals
 * share a KFD event (no lost wake-ups).
 *
 * Expected Results: 200 completions, spread at random over 4 waiters and one
 * async handler, are all observed within 1 s; process CPU while all of them
 * wait is below 20% in at least one of 3 measurements.
 */
void SignalEventExhaustionTest::TestBlockedWaiters(void) {
  constexpr int kWaiters = 4;
  Target* handler = NewHandlerTarget();
  std::vector<Target*> waiters;
  for (int i = 0; i < kWaiters; ++i) {
    targets_.emplace_back(new Target());
    waiters.push_back(targets_.back().get());
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_signal_create(1, 0, nullptr, 0, &waiters.back()->signal));
  }
  // Bounded waits: longer than the 1 s observe timeout, so a lost wake-up is
  // still reported as a missed completion, but a waiter that never wakes up
  // cannot hang join() below.
  const uint64_t wait_ticks = 2 * timestamp_frequency_;
  std::vector<std::thread> threads;
  for (Target* w : waiters) {
    threads.emplace_back([w, wait_ticks] {
      while (true) {
        const hsa_signal_value_t v = hsa_signal_wait_scacquire(
            w->signal, HSA_SIGNAL_CONDITION_LT, 1, wait_ticks, HSA_WAIT_STATE_BLOCKED);
        if (w->stop.load()) return;
        if (v >= 1) continue;  // timed out
        int64_t pending = 0;
        w->seen_ns.compare_exchange_strong(pending, NowNs());
        hsa_signal_store_screlease(w->signal, 1);
      }
    });
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double best = 0;
  for (int round = 0; round < kRounds; ++round) {
    const double cpu = ProcessCpuPercent(500);
    if (verbosity() > 0)
      std::cout << "  round " << round << ": idle process CPU with " << kWaiters
                << " blocked waiters " << cpu << "%" << std::endl;
    best = (round == 0) ? cpu : std::min(best, cpu);
    if (best < 20) break;
  }

  std::mt19937 rng(12345);
  std::vector<double> waiter_us, handler_us;
  int lost = 0;
  for (int i = 0; i < kSamples; ++i) {
    std::this_thread::sleep_for(kInterval);
    const int pick = static_cast<int>(rng() % (kWaiters + 1));
    Target* t = (pick == kWaiters) ? handler : waiters[pick];
    const double lat = Complete(t, false);
    if (lat < 0) {
      ++lost;
      hsa_signal_store_screlease(t->signal, 1);
      continue;
    }
    (t == handler ? handler_us : waiter_us).push_back(lat);
  }

  for (Target* w : waiters) {
    w->stop.store(true);
    hsa_signal_store_screlease(w->signal, -1);
  }
  for (auto& th : threads) th.join();

  if (verbosity() > 0 && !waiter_us.empty() && !handler_us.empty()) {
    std::cout << "  waiter latency p50=" << Median(waiter_us)
              << "us p99=" << Percentile(waiter_us, 0.99)
              << "us, handler latency p50=" << Median(handler_us)
              << "us p99=" << Percentile(handler_us, 0.99) << "us" << std::endl;
  }
  EXPECT_EQ(0, lost) << "completions not observed within 1 s";
  EXPECT_LT(best, 20) << "idle process CPU with " << kWaiters
                      << " waiters blocked on post-exhaustion signals";
}

/*
 * Test Name: TestGpuCompletion
 * Scope: Functional
 *
 * Purpose: A post-exhaustion signal used as the completion signal of an AQL
 * packet must reach its async handler: the CP end-of-pipe interrupt has to
 * name an event the async-events thread is waiting on.
 *
 * Expected Results: 200 barrier-packet completions all reach the handler
 * within 1 s.
 */
void SignalEventExhaustionTest::TestGpuCompletion(void) {
  Target* t = NewHandlerTarget();
  std::vector<double> us;
  for (int i = 0; i < kSamples; ++i) {
    std::this_thread::sleep_for(kInterval);
    const double lat = Complete(t, true);
    ASSERT_GE(lat, 0) << "GPU completion " << i << " never reached the async handler";
    us.push_back(lat);
  }
  if (verbosity() > 0) {
    std::cout << "  GPU completion -> handler p50=" << Median(us)
              << "us p99=" << Percentile(us, 0.99) << "us" << std::endl;
  }
}
