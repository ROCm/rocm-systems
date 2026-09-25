// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <thread>

namespace rocjitsu::amdgpu {

class GpuQueueRegistryTestAccess {
public:
  static void fail_next_allocation(GpuQueueRegistry &registry) {
    std::lock_guard lock(registry.mutex_);
    registry.fail_next_registry_allocation_ = true;
  }
};

namespace {

using namespace std::chrono_literals;

class IdentityTranslator final : public AddressSpaceTranslator {
public:
  VmTranslationResult translate(uint64_t address, std::size_t size,
                                VmAccessKind /*access*/) const override {
    if (size == 0)
      return {.outcome = VmAccessOutcome::Malformed, .translation = {}};
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation = {.domain = VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = size,
                        .mtype = Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }
};

class NullPhysicalMemory final : public PhysicalMemoryAccess {
public:
  VmAccessOutcome read(VmMemoryDomain, uint64_t, std::span<std::byte> bytes) override {
    std::ranges::fill(bytes, std::byte{0});
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome write(VmMemoryDomain, uint64_t, std::span<const std::byte>) override {
    return VmAccessOutcome::Complete;
  }
};

struct BindingControl {
  std::atomic<QueuePrepareCloseStatus> prepare_status{QueuePrepareCloseStatus::Ready};
  std::atomic<QueueSubmissionStatus> submit_status{QueueSubmissionStatus::Accepted};
  std::atomic<int> destructions{0};
  std::atomic<bool> throw_submit{false};
  std::atomic<bool> throw_reconfigure{false};
  std::mutex mutex;
  std::condition_variable condition;
  bool submit_entered = false;
  bool submit_released = true;
  bool reconfigure_entered = false;
  bool reconfigure_released = true;

  bool wait_for_submit(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex);
    return condition.wait_for(lock, timeout, [this] { return submit_entered; });
  }

  bool wait_for_reconfigure(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex);
    return condition.wait_for(lock, timeout, [this] { return reconfigure_entered; });
  }

  void release_submit() {
    std::lock_guard lock(mutex);
    submit_released = true;
    condition.notify_all();
  }

  void release_reconfigure() {
    std::lock_guard lock(mutex);
    reconfigure_released = true;
    condition.notify_all();
  }
};

class TestQueueBinding final : public QueueBinding {
public:
  explicit TestQueueBinding(std::shared_ptr<BindingControl> control)
      : control_(std::move(control)) {}
  ~TestQueueBinding() override { ++control_->destructions; }

  QueuePrepareCloseStatus prepare_close() noexcept override {
    return control_->prepare_status.load();
  }

  QueueReconfigureStatus reconfigure(const QueueReconfigureRequest &) override {
    if (control_->throw_reconfigure.load())
      throw std::runtime_error("reconfigure failure");
    std::unique_lock lock(control_->mutex);
    control_->reconfigure_entered = true;
    control_->condition.notify_all();
    control_->condition.wait(lock, [this] { return control_->reconfigure_released; });
    return QueueReconfigureStatus::Applied;
  }

  QueueSubmissionStatus submit_producer(uint64_t) override {
    if (control_->throw_submit.load())
      throw std::runtime_error("submit failure");
    std::unique_lock lock(control_->mutex);
    control_->submit_entered = true;
    control_->condition.notify_all();
    control_->condition.wait(lock, [this] { return control_->submit_released; });
    return control_->submit_status.load();
  }

private:
  std::shared_ptr<BindingControl> control_;
};

struct FactoryControl {
  enum class Result { Bind, Reject, Throw };

  explicit FactoryControl(std::shared_ptr<BindingControl> binding_control)
      : binding(std::move(binding_control)) {}

  std::shared_ptr<BindingControl> binding;
  std::atomic<int> creates{0};
  Result result = Result::Bind;
  std::mutex mutex;
  std::condition_variable condition;
  bool block = false;
  bool entered = false;
  bool released = false;

  bool wait_for_entry(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex);
    return condition.wait_for(lock, timeout, [this] { return entered; });
  }

  void release() {
    std::lock_guard lock(mutex);
    released = true;
    condition.notify_all();
  }
};

class TestQueueFactory final : public QueueBindingFactory {
public:
  explicit TestQueueFactory(std::shared_ptr<FactoryControl> control)
      : control_(std::move(control)) {}

  QueueBindingCreateResult create_binding(const QueueRegistrationRequest &) override {
    ++control_->creates;
    {
      std::unique_lock lock(control_->mutex);
      control_->entered = true;
      control_->condition.notify_all();
      if (control_->block)
        control_->condition.wait(lock, [this] { return control_->released; });
    }
    if (control_->result == FactoryControl::Result::Throw)
      throw std::runtime_error("factory failure");
    if (control_->result == FactoryControl::Result::Reject)
      return {};
    return {.status = QueueBindingCreateStatus::Bound,
            .binding = std::make_unique<TestQueueBinding>(control_->binding)};
  }

private:
  std::shared_ptr<FactoryControl> control_;
};

template <typename Predicate>
bool wait_until(Predicate predicate, std::chrono::milliseconds timeout = 2s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::yield();
  }
  return true;
}

class GpuQueueRegistryTest : public ::testing::Test {
protected:
  static constexpr uint32_t kProcessId = 7;

  void SetUp() override {
    address_space_ = gpu_vm_.register_translated(kProcessId, std::make_shared<IdentityTranslator>(),
                                                 std::make_shared<NullPhysicalMemory>());
    ASSERT_TRUE(address_space_);
  }

  QueueRegistrationRequest request(std::shared_ptr<QueueBindingFactory> factory,
                                   uint32_t queue_id = 1) const {
    QueueRegistrationRequest result;
    result.identity = {
        .address_space = address_space_, .process_id = kProcessId, .queue_id = queue_id};
    result.binding_factory = std::move(factory);
    return result;
  }

  std::shared_ptr<FactoryControl>
  factory_control(std::shared_ptr<BindingControl> binding = std::make_shared<BindingControl>()) {
    return std::make_shared<FactoryControl>(std::move(binding));
  }

  GpuVm gpu_vm_;
  GpuQueueRegistry registry_{gpu_vm_};
  AddressSpaceHandle address_space_;
};

TEST_F(GpuQueueRegistryTest, RegistrationFailuresRollBackAddressSpaceRetention) {
  auto rejected = factory_control();
  rejected->result = FactoryControl::Result::Reject;
  EXPECT_FALSE(registry_.register_queue(request(std::make_shared<TestQueueFactory>(rejected))));
  ASSERT_TRUE(gpu_vm_.lookup(address_space_));
  EXPECT_EQ(gpu_vm_.lookup(address_space_)->queue_references, 0u);

  auto throwing = factory_control();
  throwing->result = FactoryControl::Result::Throw;
  EXPECT_THROW(
      (void)registry_.register_queue(request(std::make_shared<TestQueueFactory>(throwing))),
      std::runtime_error);
  ASSERT_TRUE(gpu_vm_.lookup(address_space_));
  EXPECT_EQ(gpu_vm_.lookup(address_space_)->queue_references, 0u);

  auto allocation = factory_control();
  GpuQueueRegistryTestAccess::fail_next_allocation(registry_);
  EXPECT_THROW(
      (void)registry_.register_queue(request(std::make_shared<TestQueueFactory>(allocation))),
      std::bad_alloc);
  ASSERT_TRUE(gpu_vm_.lookup(address_space_));
  EXPECT_EQ(gpu_vm_.lookup(address_space_)->queue_references, 0u);
  EXPECT_EQ(allocation->binding->destructions, 1);
  EXPECT_EQ(registry_.active_queues(), 0u);
}

TEST_F(GpuQueueRegistryTest, DuplicateIdentityIsRejectedBeforeCreatingAnotherBinding) {
  auto first = factory_control();
  const QueueHandle handle =
      registry_.register_queue(request(std::make_shared<TestQueueFactory>(first)));
  ASSERT_TRUE(handle);

  auto duplicate = factory_control();
  EXPECT_FALSE(registry_.register_queue(request(std::make_shared<TestQueueFactory>(duplicate))));
  EXPECT_EQ(duplicate->creates, 0);
  ASSERT_TRUE(gpu_vm_.lookup(address_space_));
  EXPECT_EQ(gpu_vm_.lookup(address_space_)->queue_references, 1u);
  EXPECT_TRUE(registry_.unregister_queue(handle, QueueCloseMode::ForceCancel));
}

TEST_F(GpuQueueRegistryTest, AddressSpaceHandleDoesNotConflatePasidAndVmid) {
  auto control = factory_control();
  QueueRegistrationRequest registration = request(std::make_shared<TestQueueFactory>(control));
  registration.identity.process_id = kProcessId + 1;

  const QueueHandle handle = registry_.register_queue(registration);

  ASSERT_TRUE(handle);
  EXPECT_EQ(control->creates, 1);
  EXPECT_TRUE(registry_.unregister_queue(handle, QueueCloseMode::ForceCancel));
}

TEST_F(GpuQueueRegistryTest, SubmissionResultIsTrueOnlyWhenAccepted) {
  auto control = std::make_shared<BindingControl>();
  const QueueHandle handle = registry_.register_queue(
      request(std::make_shared<TestQueueFactory>(factory_control(control))));
  ASSERT_TRUE(handle);

  EXPECT_TRUE(registry_.submit_producer(handle, 1));
  control->submit_status = QueueSubmissionStatus::Retry;
  const QueueSubmissionResult retry = registry_.submit_producer(handle, 2);
  EXPECT_TRUE(retry.found);
  EXPECT_FALSE(retry);
  control->submit_status = QueueSubmissionStatus::Faulted;
  const QueueSubmissionResult faulted = registry_.submit_producer(handle, 3);
  EXPECT_TRUE(faulted.found);
  EXPECT_FALSE(faulted);

  EXPECT_TRUE(registry_.unregister_queue(handle, QueueCloseMode::ForceCancel));
}

TEST_F(GpuQueueRegistryTest, ClosedHandlesStayStaleAfterSlotReuse) {
  auto first = factory_control();
  const QueueHandle stale =
      registry_.register_queue(request(std::make_shared<TestQueueFactory>(first)));
  ASSERT_TRUE(stale);
  EXPECT_TRUE(registry_.unregister_queue(stale, QueueCloseMode::ForceCancel));

  auto replacement = factory_control();
  const QueueHandle current =
      registry_.register_queue(request(std::make_shared<TestQueueFactory>(replacement)));
  ASSERT_TRUE(current);
  EXPECT_EQ(current.slot, stale.slot);
  EXPECT_NE(current.generation, stale.generation);
  EXPECT_FALSE(registry_.submit_producer(stale, 1).found);
  EXPECT_TRUE(registry_.submit_producer(current, 1).found);
  EXPECT_TRUE(registry_.unregister_queue(current, QueueCloseMode::ForceCancel));
}

TEST_F(GpuQueueRegistryTest, GracefulClosePreservesBusyAndFaultedBindingsForForcedRemoval) {
  auto control = std::make_shared<BindingControl>();
  auto factory = factory_control(control);
  const QueueHandle handle =
      registry_.register_queue(request(std::make_shared<TestQueueFactory>(factory)));
  ASSERT_TRUE(handle);

  control->prepare_status = QueuePrepareCloseStatus::Busy;
  const QueueCloseResult busy = registry_.unregister_queue(handle);
  EXPECT_TRUE(busy.found);
  EXPECT_EQ(busy.status, QueueCloseStatus::Busy);
  EXPECT_TRUE(registry_.contains(handle));

  control->prepare_status = QueuePrepareCloseStatus::Faulted;
  const QueueCloseResult faulted = registry_.unregister_queue(handle);
  EXPECT_TRUE(faulted.found);
  EXPECT_EQ(faulted.status, QueueCloseStatus::Faulted);
  EXPECT_TRUE(registry_.contains(handle));

  EXPECT_TRUE(registry_.unregister_queue(handle, QueueCloseMode::ForceCancel));
  EXPECT_EQ(control->destructions, 1);
}

TEST_F(GpuQueueRegistryTest, CloseWaitsForAnAdmittedProducerSubmission) {
  auto control = std::make_shared<BindingControl>();
  control->submit_released = false;
  auto factory = factory_control(control);
  const QueueHandle handle =
      registry_.register_queue(request(std::make_shared<TestQueueFactory>(factory)));
  ASSERT_TRUE(handle);

  auto submit =
      std::async(std::launch::async, [&] { return registry_.submit_producer(handle, 17); });
  const bool entered = control->wait_for_submit(2s);
  if (!entered)
    control->release_submit();
  ASSERT_TRUE(entered);

  auto close = std::async(std::launch::async, [&] {
    return registry_.unregister_queue(handle, QueueCloseMode::ForceCancel);
  });
  const bool closing = wait_until([&] { return !registry_.contains(handle); });
  const auto before_release = close.wait_for(50ms);
  const QueueSubmissionResult rejected = registry_.submit_producer(handle, 18);
  control->release_submit();

  EXPECT_TRUE(closing);
  EXPECT_EQ(before_release, std::future_status::timeout);
  EXPECT_FALSE(rejected.found);
  EXPECT_EQ(submit.get().status, QueueSubmissionStatus::Accepted);
  EXPECT_TRUE(close.get());
}

TEST_F(GpuQueueRegistryTest, CloseWaitsForAnAdmittedReconfiguration) {
  auto control = std::make_shared<BindingControl>();
  control->reconfigure_released = false;
  auto factory = factory_control(control);
  const QueueHandle handle =
      registry_.register_queue(request(std::make_shared<TestQueueFactory>(factory)));
  ASSERT_TRUE(handle);

  auto reconfigure = std::async(std::launch::async, [&] {
    return registry_.reconfigure_queue(
        handle,
        {.ring_base_address = 0x1000, .ring_size_bytes = 4096, .scheduling_percentage = 100});
  });
  const bool entered = control->wait_for_reconfigure(2s);
  if (!entered)
    control->release_reconfigure();
  ASSERT_TRUE(entered);

  auto close = std::async(std::launch::async, [&] {
    return registry_.unregister_queue(handle, QueueCloseMode::ForceCancel);
  });
  const bool closing = wait_until([&] { return !registry_.contains(handle); });
  const auto before_release = close.wait_for(50ms);
  control->release_reconfigure();

  EXPECT_TRUE(closing);
  EXPECT_EQ(before_release, std::future_status::timeout);
  EXPECT_EQ(reconfigure.get().status, QueueReconfigureStatus::Applied);
  EXPECT_TRUE(close.get());
}

TEST_F(GpuQueueRegistryTest, ThrowingOperationsReleaseTheirLeases) {
  auto control = std::make_shared<BindingControl>();
  control->throw_submit = true;
  control->throw_reconfigure = true;
  auto factory = factory_control(control);
  const QueueHandle handle =
      registry_.register_queue(request(std::make_shared<TestQueueFactory>(factory)));
  ASSERT_TRUE(handle);

  EXPECT_THROW((void)registry_.submit_producer(handle, 1), std::runtime_error);
  EXPECT_THROW((void)registry_.reconfigure_queue(handle, {}), std::runtime_error);
  EXPECT_TRUE(registry_.unregister_queue(handle, QueueCloseMode::ForceCancel));
}

TEST_F(GpuQueueRegistryTest, CloseAllFencesAndRollsBackAnInFlightRegistration) {
  auto blocked = factory_control();
  blocked->block = true;
  auto registration = std::async(std::launch::async, [&] {
    return registry_.register_queue(request(std::make_shared<TestQueueFactory>(blocked)));
  });
  const bool factory_entered = blocked->wait_for_entry(2s);
  if (!factory_entered)
    blocked->release();
  ASSERT_TRUE(factory_entered);

  const uint64_t initial_epoch = registry_.lifecycle_epoch();
  auto close_all = std::async(std::launch::async, [&] { registry_.close_all(); });
  const bool admission_closed =
      wait_until([&] { return !registry_.accepting_registrations_for_test(); });

  auto rejected = factory_control();
  const QueueHandle rejected_handle =
      registry_.register_queue(request(std::make_shared<TestQueueFactory>(rejected), 2));
  blocked->release();

  EXPECT_TRUE(admission_closed);
  EXPECT_FALSE(rejected_handle);
  EXPECT_EQ(rejected->creates, 0);
  EXPECT_FALSE(registration.get());
  close_all.get();
  EXPECT_TRUE(registry_.accepting_registrations_for_test());
  EXPECT_GT(registry_.lifecycle_epoch(), initial_epoch);
  EXPECT_EQ(registry_.active_queues(), 0u);
  ASSERT_TRUE(gpu_vm_.lookup(address_space_));
  EXPECT_EQ(gpu_vm_.lookup(address_space_)->queue_references, 0u);
}

TEST(InterruptSubscriptionTest, ResetDrainsAnInFlightCallbackAndRevokesTheSink) {
  std::mutex mutex;
  std::condition_variable condition;
  bool entered = false;
  bool released = false;
  std::atomic<int> calls{0};
  InterruptSubscription subscription([&](uint32_t, uint32_t) {
    ++calls;
    std::unique_lock lock(mutex);
    entered = true;
    condition.notify_all();
    condition.wait(lock, [&] { return released; });
  });
  const InterruptSink sink = subscription.sink();

  auto delivery = std::async(std::launch::async, [&] { sink.deliver(7, 11); });
  bool callback_entered = false;
  {
    std::unique_lock lock(mutex);
    callback_entered = condition.wait_for(lock, 2s, [&] { return entered; });
  }
  if (!callback_entered) {
    std::lock_guard lock(mutex);
    released = true;
    condition.notify_all();
  }
  ASSERT_TRUE(callback_entered);

  auto reset = std::async(std::launch::async, [&] { subscription.reset(); });
  const auto before_release = reset.wait_for(50ms);
  {
    std::lock_guard lock(mutex);
    released = true;
    condition.notify_all();
  }
  delivery.get();
  reset.get();
  sink.deliver(7, 12);

  EXPECT_EQ(before_release, std::future_status::timeout);
  EXPECT_EQ(calls, 1);
  EXPECT_FALSE(subscription);
}

TEST(InterruptSubscriptionTest, ConcurrentCallbacksCanResetTheSameSubscription) {
  struct Harness {
    std::barrier<> callbacks_ready{2};
    std::mutex mutex;
    std::condition_variable condition;
    int completed = 0;
    InterruptSubscription subscription;
  };

  auto *harness = new Harness;
  harness->subscription = InterruptSubscription([harness](uint32_t, uint32_t) {
    harness->callbacks_ready.arrive_and_wait();
    harness->subscription.reset();
    {
      std::lock_guard lock(harness->mutex);
      ++harness->completed;
      harness->condition.notify_all();
    }
  });
  const InterruptSink sink = harness->subscription.sink();
  std::thread first([sink] { sink.deliver(7, 1); });
  std::thread second([sink] { sink.deliver(7, 2); });

  bool completed = false;
  {
    std::unique_lock lock(harness->mutex);
    completed =
        harness->condition.wait_for(lock, 2s, [harness] { return harness->completed == 2; });
  }
  if (!completed) {
    first.detach();
    second.detach();
    ADD_FAILURE() << "concurrent reset deadlocked";
    return; // Deliberately retain the harness used by the blocked callbacks.
  }

  first.join();
  second.join();
  EXPECT_FALSE(harness->subscription);
  delete harness;
}

} // namespace
} // namespace rocjitsu::amdgpu
