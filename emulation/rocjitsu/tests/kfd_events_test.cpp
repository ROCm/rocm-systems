// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/events.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <tuple>
#include <vector>

namespace {

class KfdEventsTest : public ::testing::Test {
protected:
  rocjitsu::EventState events;

  uint32_t create(bool auto_reset = true, uint32_t type = KFD_IOC_EVENT_SIGNAL) {
    kfd_ioctl_create_event_args args{};
    args.event_type = type;
    args.auto_reset = auto_reset;
    EXPECT_EQ(events.create_event(&args, 1), 0);
    return args.event_id;
  }

  void signal(uint32_t id, int source = 0) {
    if (source == 0) {
      kfd_ioctl_set_event_args args{};
      args.event_id = id;
      EXPECT_EQ(events.set_event(&args), 0);
    } else {
      events.signal_interrupt(source == 1 ? id : 0);
    }
  }

  void reset(uint32_t id) {
    kfd_ioctl_reset_event_args args{};
    args.event_id = id;
    EXPECT_EQ(events.reset_event(&args), 0);
  }

  bool registered(uint32_t id, size_t count = 1) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
      if (events.waiter_count(id) == count)
        return true;
      std::this_thread::yield();
    }
    return false;
  }

  struct Wait {
    rocjitsu::EventState &events;
    std::vector<kfd_event_data> data;
    kfd_ioctl_wait_events_args args{};
    int rc = -1;
    std::atomic<bool> done{false};
    std::thread thread;

    Wait(rocjitsu::EventState &events, std::initializer_list<uint32_t> ids, uint64_t age = 0,
         bool all = true, bool poll = false)
        : events(events), data(ids.size()) {
      size_t i = 0;
      for (uint32_t id : ids) {
        data[i].event_id = id;
        data[i++].signal_event_data.last_event_age = age;
      }
      args.events_ptr = reinterpret_cast<uint64_t>(data.data());
      args.num_events = data.size();
      args.wait_for_all = all;
      args.timeout = poll ? 0 : 5000;
      if (poll)
        rc = events.wait_events(&args);
      else
        thread = std::thread([this] {
          rc = this->events.wait_events(&args);
          done.store(true);
        });
    }

    ~Wait() {
      if (thread.joinable()) {
        events.begin_wait_cancel();
        thread.join();
      }
    }

    void complete() {
      if (thread.joinable())
        thread.join();
      EXPECT_EQ(rc, 0);
      EXPECT_EQ(args.wait_result, KFD_IOC_WAIT_RESULT_COMPLETE);
    }

    bool finished() {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!done.load() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      return done.load();
    }
  };

  void expect_unsignaled(uint32_t id) {
    Wait poll(events, {id}, 0, true, true);
    EXPECT_EQ(poll.rc, 0);
    EXPECT_EQ(poll.args.wait_result, KFD_IOC_WAIT_RESULT_TIMEOUT);
  }
};

class KfdEventOrderingTest : public KfdEventsTest,
                             public ::testing::WithParamInterface<std::tuple<int, uint64_t, bool>> {
};

TEST_P(KfdEventOrderingTest, AutoResetSignalReachesWaiter) {
  const auto [source, age, signal_first] = GetParam();
  const auto id = create();
  if (signal_first)
    signal(id, source);
  Wait wait(events, {id}, age);
  if (!signal_first) {
    ASSERT_TRUE(registered(id));
    signal(id, source);
  }
  wait.complete();
  expect_unsignaled(id);
}

INSTANTIATE_TEST_SUITE_P(SetInterruptBroadcast, KfdEventOrderingTest,
                         ::testing::Combine(::testing::Values(0, 1, 2),
                                            ::testing::Values(uint64_t{0}, uint64_t{1}),
                                            ::testing::Bool()));

TEST_F(KfdEventsTest, AutoResetSignalReachesEveryRegisteredWaiter) {
  const auto id = create();
  Wait legacy(events, {id});
  Wait aged(events, {id}, 1);
  Wait second_legacy(events, {id});
  ASSERT_TRUE(registered(id, 3));
  signal(id);
  legacy.complete();
  aged.complete();
  second_legacy.complete();
  expect_unsignaled(id);
}

TEST_F(KfdEventsTest, WaitAllRemembersSignalsWithoutExposingThemToNewWaiters) {
  const auto first = create();
  const auto second = create();
  Wait wait(events, {first, second});
  ASSERT_TRUE(registered(first));
  ASSERT_TRUE(registered(second));
  signal(first);
  expect_unsignaled(first);
  reset(first);
  signal(second);
  wait.complete();
  expect_unsignaled(first);
  expect_unsignaled(second);
}

TEST_F(KfdEventsTest, WaitAllConsumesInitiallySignaledAutoResetEvent) {
  const auto first = create();
  const auto second = create();
  signal(first);
  Wait wait(events, {first, second});
  ASSERT_TRUE(registered(second));
  expect_unsignaled(first);
  signal(second);
  wait.complete();
}

TEST_F(KfdEventsTest, ManualResetDoesNotRevokeRegisteredWaiterActivation) {
  const auto first = create(false);
  const auto second = create(false);
  Wait wait(events, {first, second});
  ASSERT_TRUE(registered(first));
  signal(first);
  reset(first);
  expect_unsignaled(first);
  signal(second);
  wait.complete();
  Wait poll(events, {second}, 0, true, true);
  poll.complete();
  reset(second);
  expect_unsignaled(second);
}

TEST_F(KfdEventsTest, CompletingWaitAllDoesNotConsumeALaterPendingSignal) {
  const auto first = create();
  const auto second = create();
  signal(first);
  Wait wait(events, {first, second});
  ASSERT_TRUE(registered(second));
  signal(first);
  signal(second);
  wait.complete();
  Wait later(events, {first}, 0, true, true);
  later.complete();
  expect_unsignaled(first);
}

TEST_F(KfdEventsTest, TimeoutConsumesPartialAutoResetActivation) {
  const auto first = create();
  const auto second = create();
  signal(first);
  Wait wait(events, {first, second}, 0, true, true);
  EXPECT_EQ(wait.args.wait_result, KFD_IOC_WAIT_RESULT_TIMEOUT);
  expect_unsignaled(first);
  EXPECT_EQ(events.waiter_count(first), 0u);
  EXPECT_EQ(events.waiter_count(second), 0u);
}

TEST_F(KfdEventsTest, AutoResetMemoryFaultReachesAllParkedWaiters) {
  const auto id = create(true, KFD_IOC_EVENT_MEMORY);
  Wait first(events, {id});
  Wait second(events, {id});
  ASSERT_TRUE(registered(id, 2));
  ASSERT_TRUE(events.signal_memory_fault({.va = 0x1234000, .gpu_id = 7}));
  first.complete();
  second.complete();
  EXPECT_EQ(first.data[0].memory_exception_data.va, 0x1234000u);
  EXPECT_EQ(second.data[0].memory_exception_data.gpu_id, 7u);
  expect_unsignaled(id);
}

TEST_F(KfdEventsTest, WaitAnyCompletesWhenOnlyOneEventSignals) {
  const auto first = create();
  const auto second = create();
  Wait wait(events, {first, second}, 0, false);
  ASSERT_TRUE(registered(second));
  signal(second);
  wait.complete();
  expect_unsignaled(first);
  expect_unsignaled(second);
}

TEST_F(KfdEventsTest, LegacyWaitKeepsAgeDisabled) {
  const auto id = create();
  signal(id);
  Wait wait(events, {id}, 0, true, true);
  wait.complete();
  EXPECT_EQ(wait.data[0].signal_event_data.last_event_age, 0u);
}

TEST_F(KfdEventsTest, AgeBasedWaitObservesSignalConsumedByLegacyWaiter) {
  const auto id = create();
  signal(id);
  Wait legacy(events, {id}, 0, true, true);
  legacy.complete();
  Wait aged(events, {id}, 1, true, true);
  aged.complete();
  EXPECT_EQ(aged.data[0].signal_event_data.last_event_age, 2u);
  Wait unchanged(events, {id}, 2, true, true);
  EXPECT_EQ(unchanged.args.wait_result, KFD_IOC_WAIT_RESULT_TIMEOUT);
}

TEST_F(KfdEventsTest, TimedOutWaitAllPreservesAgesForRetry) {
  const auto first = create();
  const auto second = create();
  signal(first);
  Wait wait(events, {first, second}, 1, true, true);
  EXPECT_EQ(wait.args.wait_result, KFD_IOC_WAIT_RESULT_TIMEOUT);
  EXPECT_EQ(wait.data[0].signal_event_data.last_event_age, 1u);
  EXPECT_EQ(wait.data[1].signal_event_data.last_event_age, 1u);
  signal(second);
  EXPECT_EQ(events.wait_events(&wait.args), 0);
  EXPECT_EQ(wait.args.wait_result, KFD_IOC_WAIT_RESULT_COMPLETE);
  EXPECT_EQ(wait.data[0].signal_event_data.last_event_age, 2u);
  EXPECT_EQ(wait.data[1].signal_event_data.last_event_age, 2u);
}

TEST_F(KfdEventsTest, TimedOutWaitAllDoesNotCopyMemoryFault) {
  const auto fault = create(true, KFD_IOC_EVENT_MEMORY);
  const auto signal = create();
  ASSERT_TRUE(events.signal_memory_fault({.va = 0x1234000, .gpu_id = 7}));
  Wait wait(events, {fault, signal}, 0, true, true);
  EXPECT_EQ(wait.args.wait_result, KFD_IOC_WAIT_RESULT_TIMEOUT);
  EXPECT_EQ(wait.data[0].memory_exception_data.va, 0u);
  EXPECT_EQ(wait.data[0].memory_exception_data.gpu_id, 0u);
}

TEST_F(KfdEventsTest, DestroyedEventFailsRegisteredWait) {
  const auto id = create();
  Wait wait(events, {id});
  ASSERT_TRUE(registered(id));
  kfd_ioctl_destroy_event_args destroy{};
  destroy.event_id = id;
  EXPECT_EQ(events.destroy_event(&destroy), 0);
  wait.thread.join();
  EXPECT_EQ(wait.rc, 0);
  EXPECT_EQ(wait.args.wait_result, KFD_IOC_WAIT_RESULT_FAIL);
}

TEST_F(KfdEventsTest, DestroyingInitiallySatisfiedEventWakesWaitAll) {
  const auto first = create();
  const auto second = create();
  signal(first);
  Wait wait(events, {first, second});
  ASSERT_TRUE(registered(second));
  kfd_ioctl_destroy_event_args destroy{};
  destroy.event_id = first;
  EXPECT_EQ(events.destroy_event(&destroy), 0);
  ASSERT_TRUE(wait.finished()) << "destruction must wake wait-all before its timeout";
  wait.thread.join();
  EXPECT_EQ(wait.rc, 0);
  EXPECT_EQ(wait.args.wait_result, KFD_IOC_WAIT_RESULT_FAIL);
}

} // namespace
