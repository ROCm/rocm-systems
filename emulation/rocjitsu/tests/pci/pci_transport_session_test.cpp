// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/ip_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/physical_memory_access.h"

#include "simdojo/components/pci_device.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <semaphore>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace {

using SessionState = simdojo::PciTransportSession::State;
using OperationLease = simdojo::PciTransportSession::OperationLease;
using BorrowedLease = simdojo::PciTransportSession::BorrowedLease;

/// @brief How long a step that should finish at once may take before a test
/// reports a hang.
/// @details Never reached by a passing run, so it is sized for a sanitizer build
/// on a loaded host rather than for speed.
constexpr std::chrono::seconds kHangDeadline{30};

constexpr uint64_t kGuestAddress = 0x1000;
constexpr uint32_t kInterruptVector = 3;

/// @brief One helper thread that a test waits for with a deadline.
///
/// @details The session defects these tests look for show up as a thread that
/// never returns. Joining such a thread would hang the test binary without a
/// message, so a thread that misses the deadline is detached and the test fails
/// instead. A detached thread can outlive the test body, which is why the work
/// must own everything it touches: capture `std::shared_ptr`s by value, never
/// references to the test's locals.
///
/// The work is not expected to throw. An exception that escapes it ends the
/// test binary, which is reported, rather than being kept in a future nobody
/// reads and passing for work that finished.
class BoundedTask {
public:
  template <typename Work> explicit BoundedTask(Work work) {
    std::promise<void> finished;
    finished_ = finished.get_future();
    thread_ = std::thread([work = std::move(work), finished = std::move(finished)]() mutable {
      work();
      finished.set_value();
    });
  }
  BoundedTask(const BoundedTask &) = delete;
  BoundedTask &operator=(const BoundedTask &) = delete;

  ~BoundedTask() {
    if (!missed_deadline_ && finished_before_deadline()) {
      thread_.join();
    } else {
      thread_.detach();
    }
  }

  /// @brief Whether the work has already returned, without waiting for it.
  [[nodiscard]] bool has_finished() const {
    return finished_.wait_for(std::chrono::seconds{0}) == std::future_status::ready;
  }

  /// @brief Wait for the work to return, giving up at @ref kHangDeadline.
  [[nodiscard]] bool finished_before_deadline() {
    if (finished_.wait_for(kHangDeadline) == std::future_status::ready) {
      return true;
    }
    missed_deadline_ = true;
    return false;
  }

private:
  std::thread thread_;
  std::future<void> finished_;
  bool missed_deadline_ = false;
};

/// @brief Transport endpoints that count the operations reaching them.
/// @details The counts are atomic because several tests drive one transport
/// from many threads at once.
class CountingTransport final : public simdojo::DmaEngine, public simdojo::IrqSink {
public:
  bool read(uint64_t guest_phys, std::span<std::byte> dst) override {
    return read_outcome(guest_phys, dst) == simdojo::DmaAccessOutcome::Complete;
  }
  bool write(uint64_t guest_phys, std::span<const std::byte> src) override {
    return write_outcome(guest_phys, src) == simdojo::DmaAccessOutcome::Complete;
  }
  simdojo::DmaAccessOutcome read_outcome(uint64_t, std::span<std::byte> dst) override {
    reads_.fetch_add(1);
    if (hold_next_read_.exchange(false)) {
      read_entered_.release();
      read_may_resume_.acquire();
    }
    std::ranges::fill(dst, std::byte{0x5a});
    return simdojo::DmaAccessOutcome::Complete;
  }
  simdojo::DmaAccessOutcome write_outcome(uint64_t, std::span<const std::byte>) override {
    writes_.fetch_add(1);
    return simdojo::DmaAccessOutcome::Complete;
  }
  bool trigger(uint32_t) override {
    triggers_.fetch_add(1);
    return true;
  }

  [[nodiscard]] uint64_t reads() const { return reads_.load(); }
  [[nodiscard]] uint64_t writes() const { return writes_.load(); }
  [[nodiscard]] uint64_t triggers() const { return triggers_.load(); }

  /// @brief Make the next read wait inside the transport until @ref resume_read.
  /// @details Only the next one, so a later read that a defect lets through
  /// completes and fails its expectation rather than blocking the test.
  void hold_next_read() { hold_next_read_.store(true); }

  /// @brief Wait until a read is parked inside the transport.
  [[nodiscard]] bool read_held_before_deadline() {
    return read_entered_.try_acquire_for(kHangDeadline);
  }

  void resume_read() {
    hold_next_read_.store(false);
    read_may_resume_.release();
  }

private:
  std::atomic<uint64_t> reads_{0};
  std::atomic<uint64_t> writes_{0};
  std::atomic<uint64_t> triggers_{0};
  std::atomic<bool> hold_next_read_{false};
  std::binary_semaphore read_entered_{0};
  std::binary_semaphore read_may_resume_{0};
};

/// @brief The smallest PCI function that can publish a transport session.
class LeaseTestDevice final : public simdojo::PciDevice {
public:
  LeaseTestDevice() : PciDevice("lease-test", {}) {}

  [[nodiscard]] std::vector<simdojo::BarSpec> bars() const override { return {}; }
  [[nodiscard]] int64_t bar_access(int, std::span<std::byte>, uint64_t, bool) override {
    return -1;
  }
  void dma_map(const simdojo::DmaRegion &) override {}
  void dma_unmap(const simdojo::DmaRegion &) override {}

  [[nodiscard]] std::shared_ptr<simdojo::PciTransportSession> capture_session() const {
    return transport_session();
  }
};

/// @brief Which halves of a transport a test attaches.
enum class SuppliedEndpoints : uint8_t { Both, DmaOnly, IrqOnly };

/// @brief A device attached to counting endpoints, together with its session.
///
/// @details Always owned through a `std::shared_ptr` that every helper thread
/// captures by value, so a thread a test stopped waiting for cannot outlive the
/// device or the endpoints its lease points at.
class AttachedSession {
public:
  explicit AttachedSession(SuppliedEndpoints supplied = SuppliedEndpoints::Both)
      : endpoints_{.irq = supplied != SuppliedEndpoints::DmaOnly ? &transport_ : nullptr,
                   .dma = supplied != SuppliedEndpoints::IrqOnly ? &transport_ : nullptr} {
    if (device_.attach_transport(&endpoints_)) {
      session_ = device_.capture_session();
    }
  }

  [[nodiscard]] const std::shared_ptr<simdojo::PciTransportSession> &session() const {
    return session_;
  }
  [[nodiscard]] CountingTransport &transport() { return transport_; }
  [[nodiscard]] const simdojo::PciTransport &endpoints() const { return endpoints_; }
  [[nodiscard]] uint64_t active_generation() const {
    return device_.transport_session_generation();
  }

  /// @brief Detach the way a transport owner does, which drains the session.
  [[nodiscard]] bool detach() { return device_.detach_transport(&endpoints_); }

private:
  CountingTransport transport_;
  simdojo::PciTransport endpoints_;
  LeaseTestDevice device_;
  std::shared_ptr<simdojo::PciTransportSession> session_;
};

/// @brief Admission through acquire(): the lease retains its session.
class OwningAdmission {
public:
  using Lease = OperationLease;
  [[nodiscard]] static Lease admit(simdojo::PciTransportSession &session) {
    return session.acquire();
  }
};

/// @brief Admission through borrow(): the caller keeps the session alive.
class BorrowedAdmission {
public:
  using Lease = BorrowedLease;
  [[nodiscard]] static Lease admit(simdojo::PciTransportSession &session) {
    return session.borrow();
  }
};

/// @brief Run wait_until_drained() on its own thread.
[[nodiscard]] BoundedTask start_drain(std::shared_ptr<simdojo::PciTransportSession> session) {
  return BoundedTask([session = std::move(session)] { session->wait_until_drained(); });
}

/// @brief Wait, without sleeping, until @p session has stopped admitting.
/// @details Used after starting a drain on another thread: the drain closes
/// admission before it waits for leases, so a session that still reads Closing
/// afterwards has a drain parked behind a live lease.
[[nodiscard]] bool admission_closed_before_deadline(const simdojo::PciTransportSession &session) {
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + kHangDeadline;
  while (session.state() == SessionState::Open) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

/// @brief Wait, without sleeping, until @p count has reached @p target.
[[nodiscard]] bool reaches_before_deadline(const std::atomic<uint32_t> &count, uint32_t target) {
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + kHangDeadline;
  while (count.load() < target) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

/// @brief Detach on a helper thread and report whether it completed.
/// @details Detaching drains the session again. That returns at once only if
/// every admission, granted or refused, was counted back out.
[[nodiscard]] bool detaches_before_deadline(const std::shared_ptr<AttachedSession> &attached) {
  auto detached = std::make_shared<std::atomic<bool>>(false);
  BoundedTask detaching_thread([attached, detached] { detached->store(attached->detach()); });
  return detaching_thread.finished_before_deadline() && detached->load();
}

/// @brief Check everything an empty lease promises: no endpoints, no effect.
template <typename LeaseType> void expect_empty_lease(const LeaseType &lease) {
  std::array<std::byte, 4> bytes{};
  EXPECT_FALSE(lease);
  EXPECT_EQ(lease.dma(), nullptr);
  EXPECT_EQ(lease.irq(), nullptr);
  EXPECT_EQ(lease.read(kGuestAddress, bytes), simdojo::DmaAccessOutcome::Unavailable);
  EXPECT_EQ(lease.write(kGuestAddress, bytes), simdojo::DmaAccessOutcome::Unavailable);
  EXPECT_EQ(lease.atomic_load(kGuestAddress, sizeof(uint32_t)).outcome,
            simdojo::DmaAccessOutcome::Unavailable);
  EXPECT_EQ(lease.atomic_store(kGuestAddress, sizeof(uint32_t), 0),
            simdojo::DmaAccessOutcome::Unavailable);
  EXPECT_EQ(lease.compare_exchange(kGuestAddress, sizeof(uint32_t), 0, 1).outcome,
            simdojo::DmaAccessOutcome::Unavailable);
  EXPECT_FALSE(lease.trigger(kInterruptVector));
}

/// @brief Move-construct a lease from @p source.
/// @details The source is taken by reference and left with the caller, because
/// what a moved-from lease still does is part of what is being tested.
template <typename LeaseType> [[nodiscard]] LeaseType move_construct_from(LeaseType &source) {
  return LeaseType(std::move(source));
}

/// @brief Move-assign @p source over @p target, leaving the source with the
/// caller for the same reason as @ref move_construct_from.
template <typename LeaseType> void move_assign(LeaseType &target, LeaseType &source) {
  target = std::move(source);
}

template <typename Admission> class PciTransportSessionLease : public ::testing::Test {};
using LeaseAdmissions = ::testing::Types<OwningAdmission, BorrowedAdmission>;
TYPED_TEST_SUITE(PciTransportSessionLease, LeaseAdmissions);

TYPED_TEST(PciTransportSessionLease, AdmitsWhileOpenAndReachesTheAttachedEndpoints) {
  auto attached = std::make_shared<AttachedSession>();
  const std::shared_ptr<simdojo::PciTransportSession> session = attached->session();
  ASSERT_NE(session, nullptr);
  const uint64_t generation = session->generation();
  EXPECT_NE(generation, 0u);
  EXPECT_EQ(generation, attached->active_generation());

  {
    typename TypeParam::Lease first = TypeParam::admit(*session);
    typename TypeParam::Lease second = TypeParam::admit(*session);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second) << "an open session admits more than one operation at a time";

    std::array<std::byte, 4> bytes{};
    EXPECT_EQ(first.dma(), attached->endpoints().dma);
    EXPECT_EQ(first.irq(), attached->endpoints().irq);
    EXPECT_EQ(second.dma(), attached->endpoints().dma);
    EXPECT_EQ(second.irq(), attached->endpoints().irq);
    EXPECT_EQ(first.read(kGuestAddress, bytes), simdojo::DmaAccessOutcome::Complete);
    EXPECT_EQ(second.read(kGuestAddress, bytes), simdojo::DmaAccessOutcome::Complete);
    EXPECT_EQ(first.write(kGuestAddress, bytes), simdojo::DmaAccessOutcome::Complete);
    EXPECT_TRUE(second.trigger(kInterruptVector));
    EXPECT_EQ(attached->transport().reads(), 2u);
    EXPECT_EQ(attached->transport().writes(), 1u);
    EXPECT_EQ(attached->transport().triggers(), 1u);
    EXPECT_EQ(session->state(), SessionState::Open);
    EXPECT_EQ(session->generation(), generation);
  }

  ASSERT_TRUE(detaches_before_deadline(attached))
      << "detach never returned although every lease had been released";
  EXPECT_EQ(session->state(), SessionState::Revoked);
  EXPECT_EQ(session->generation(), generation);
}

TYPED_TEST(PciTransportSessionLease, ExposesOnlyTheEndpointsTheTransportSupplies) {
  std::array<std::byte, 4> bytes{};
  {
    auto attached = std::make_shared<AttachedSession>(SuppliedEndpoints::DmaOnly);
    ASSERT_NE(attached->session(), nullptr);
    {
      const typename TypeParam::Lease lease = TypeParam::admit(*attached->session());
      ASSERT_TRUE(lease) << "a transport without an interrupt sink still admits operations";
      EXPECT_EQ(lease.irq(), nullptr);
      EXPECT_FALSE(lease.trigger(kInterruptVector));
      EXPECT_EQ(lease.dma(), attached->endpoints().dma);
      EXPECT_EQ(lease.read(kGuestAddress, bytes), simdojo::DmaAccessOutcome::Complete);
    }
    EXPECT_TRUE(detaches_before_deadline(attached));
  }
  {
    auto attached = std::make_shared<AttachedSession>(SuppliedEndpoints::IrqOnly);
    ASSERT_NE(attached->session(), nullptr);
    {
      const typename TypeParam::Lease lease = TypeParam::admit(*attached->session());
      ASSERT_TRUE(lease) << "a transport without a DMA engine still admits operations";
      EXPECT_EQ(lease.dma(), nullptr);
      EXPECT_EQ(lease.read(kGuestAddress, bytes), simdojo::DmaAccessOutcome::Unavailable);
      EXPECT_EQ(lease.write(kGuestAddress, bytes), simdojo::DmaAccessOutcome::Unavailable);
      EXPECT_EQ(lease.irq(), attached->endpoints().irq);
      EXPECT_TRUE(lease.trigger(kInterruptVector));
    }
    EXPECT_EQ(attached->transport().reads(), 0u);
    EXPECT_EQ(attached->transport().writes(), 0u);
    EXPECT_TRUE(detaches_before_deadline(attached));
  }
}

TYPED_TEST(PciTransportSessionLease, IsRefusedOnceRevocationBeginsAndLeavesNothingToDrain) {
  auto attached = std::make_shared<AttachedSession>();
  const std::shared_ptr<simdojo::PciTransportSession> session = attached->session();
  ASSERT_NE(session, nullptr);
  const uint64_t generation = session->generation();

  EXPECT_TRUE(session->begin_revoke());
  EXPECT_FALSE(session->begin_revoke()) << "only the first call closes admission";
  EXPECT_EQ(session->state(), SessionState::Closing);
  expect_empty_lease(TypeParam::admit(*session));

  BoundedTask drain = start_drain(session);
  ASSERT_TRUE(drain.finished_before_deadline())
      << "wait_until_drained() never returned although the only admission was refused: "
         "the refusal left an operation counted";
  EXPECT_EQ(session->state(), SessionState::Revoked);

  expect_empty_lease(TypeParam::admit(*session));
  EXPECT_FALSE(session->begin_revoke());
  EXPECT_EQ(session->generation(), generation);
  EXPECT_EQ(attached->transport().reads(), 0u);
  EXPECT_EQ(attached->transport().writes(), 0u);
  EXPECT_EQ(attached->transport().triggers(), 0u);

  // A second drain, here through detach, also sees the refusal made after the
  // first one returned.
  EXPECT_TRUE(detaches_before_deadline(attached))
      << "an admission refused by a revoked session left an operation counted";
}

TYPED_TEST(PciTransportSessionLease, RefusalsFromManyThreadsLeaveNothingToDrain) {
  constexpr uint32_t kRefusingThreads = 4;
  constexpr uint32_t kAttemptsPerThread = 100;
  auto attached = std::make_shared<AttachedSession>();
  const std::shared_ptr<simdojo::PciTransportSession> session = attached->session();
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(session->begin_revoke());

  auto wrongly_admitted = std::make_shared<std::atomic<uint32_t>>(0);
  {
    std::deque<BoundedTask> refused_threads;
    for (uint32_t thread_index = 0; thread_index < kRefusingThreads; ++thread_index) {
      refused_threads.emplace_back([session, wrongly_admitted] {
        for (uint32_t attempt = 0; attempt < kAttemptsPerThread; ++attempt) {
          if (TypeParam::admit(*session)) {
            wrongly_admitted->fetch_add(1);
          }
        }
      });
    }
    for (BoundedTask &refused_thread : refused_threads) {
      ASSERT_TRUE(refused_thread.finished_before_deadline())
          << "an admission attempt against a closing session never returned";
    }
  }
  EXPECT_EQ(wrongly_admitted->load(), 0u);

  BoundedTask drain = start_drain(session);
  ASSERT_TRUE(drain.finished_before_deadline())
      << "wait_until_drained() never returned although every admission was refused: "
         "a refusal left an operation counted";
  EXPECT_EQ(session->state(), SessionState::Revoked);
  EXPECT_TRUE(detaches_before_deadline(attached));
}

TYPED_TEST(PciTransportSessionLease, AdmittedBeforeRevocationStaysUsableAndHoldsOffTheDrain) {
  auto attached = std::make_shared<AttachedSession>();
  const std::shared_ptr<simdojo::PciTransportSession> session = attached->session();
  ASSERT_NE(session, nullptr);
  const uint64_t generation = session->generation();
  typename TypeParam::Lease lease = TypeParam::admit(*session);
  ASSERT_TRUE(lease);

  BoundedTask drain = start_drain(session);
  EXPECT_TRUE(admission_closed_before_deadline(*session))
      << "wait_until_drained() never closed admission";
  EXPECT_EQ(session->state(), SessionState::Closing)
      << "the drain completed while an admitted lease was still held";
  EXPECT_FALSE(drain.has_finished())
      << "wait_until_drained() returned while an admitted lease was still held";
  expect_empty_lease(TypeParam::admit(*session));

  std::array<std::byte, 4> bytes{};
  EXPECT_TRUE(lease);
  EXPECT_EQ(lease.dma(), attached->endpoints().dma);
  EXPECT_EQ(lease.irq(), attached->endpoints().irq);
  EXPECT_EQ(lease.read(kGuestAddress, bytes), simdojo::DmaAccessOutcome::Complete);
  EXPECT_EQ(lease.write(kGuestAddress, bytes), simdojo::DmaAccessOutcome::Complete);
  EXPECT_TRUE(lease.trigger(kInterruptVector));
  EXPECT_EQ(attached->transport().reads(), 1u);
  EXPECT_EQ(attached->transport().writes(), 1u);
  EXPECT_EQ(attached->transport().triggers(), 1u);
  EXPECT_EQ(session->generation(), generation);

  lease = {};
  ASSERT_TRUE(drain.finished_before_deadline())
      << "wait_until_drained() did not return after the only lease was released";
  EXPECT_EQ(session->state(), SessionState::Revoked);
  EXPECT_EQ(session->generation(), generation);
  EXPECT_TRUE(detaches_before_deadline(attached));
}

TYPED_TEST(PciTransportSessionLease, MoveCarriesExactlyOneAdmission) {
  auto attached = std::make_shared<AttachedSession>();
  const std::shared_ptr<simdojo::PciTransportSession> session = attached->session();
  ASSERT_NE(session, nullptr);
  typename TypeParam::Lease first = TypeParam::admit(*session);
  typename TypeParam::Lease second = TypeParam::admit(*session);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);

  typename TypeParam::Lease survivor = move_construct_from(first);
  expect_empty_lease(first);
  EXPECT_TRUE(survivor);
  EXPECT_EQ(survivor.dma(), attached->endpoints().dma);
  EXPECT_EQ(survivor.irq(), attached->endpoints().irq);

  // Assigning over a live lease gives back the admission it held, so exactly
  // one of the two admissions above is outstanding from here on.
  move_assign(survivor, second);
  expect_empty_lease(second);
  EXPECT_TRUE(survivor);
  EXPECT_EQ(survivor.dma(), attached->endpoints().dma);
  EXPECT_EQ(survivor.irq(), attached->endpoints().irq);

  BoundedTask drain = start_drain(session);
  EXPECT_TRUE(admission_closed_before_deadline(*session))
      << "wait_until_drained() never closed admission";
  first = {};
  second = {};
  EXPECT_EQ(session->state(), SessionState::Closing)
      << "a moved-from lease released the admission it had handed on";
  EXPECT_FALSE(drain.has_finished())
      << "wait_until_drained() returned while the moved-to lease was still held";

  std::array<std::byte, 4> bytes{};
  EXPECT_EQ(survivor.read(kGuestAddress, bytes), simdojo::DmaAccessOutcome::Complete);

  survivor = {};
  ASSERT_TRUE(drain.finished_before_deadline())
      << "wait_until_drained() did not return after the moved-to lease was released: a move "
         "duplicated or lost an admission";
  EXPECT_EQ(session->state(), SessionState::Revoked);
  EXPECT_TRUE(detaches_before_deadline(attached));
}

TEST(PciTransportSession, OnlyTheOwningLeaseRetainsTheSession) {
  auto attached = std::make_shared<AttachedSession>();
  const std::shared_ptr<simdojo::PciTransportSession> session = attached->session();
  ASSERT_NE(session, nullptr);
  const long references_before = session.use_count();
  {
    const BorrowedLease borrowed = session->borrow();
    ASSERT_TRUE(borrowed);
    EXPECT_EQ(session.use_count(), references_before)
        << "a borrowed lease must leave the session's shared reference count alone";
    const OperationLease owning = session->acquire();
    ASSERT_TRUE(owning);
    EXPECT_EQ(session.use_count(), references_before + 1)
        << "an owning lease keeps the session alive on its own";
  }
  EXPECT_EQ(session.use_count(), references_before);
  EXPECT_TRUE(detaches_before_deadline(attached));
}

/// @brief Leases handed from the threads that admitted them to the threads
/// that release them.
///
/// @details Each slot holds one owning and one borrowed lease, admitted
/// together on one thread. A test passes the object from thread to thread
/// through joins, so the slots need no lock; only the count of releases is read
/// while another thread may be changing it.
class HandedOverLeases {
public:
  HandedOverLeases(std::shared_ptr<simdojo::PciTransportSession> session, std::size_t slots)
      : session_(std::move(session)), owning_(slots), borrowed_(slots) {}

  /// @brief Admit both leases of @p slot on the calling thread.
  void admit(std::size_t slot) {
    owning_[slot] = session_->acquire();
    borrowed_[slot] = session_->borrow();
  }

  /// @brief Whether both leases of @p slot were admitted.
  [[nodiscard]] bool admitted(std::size_t slot) const {
    return static_cast<bool>(owning_[slot]) && static_cast<bool>(borrowed_[slot]);
  }

  /// @brief Release, on the calling thread, every lease but the last borrowed
  /// one.
  void release_all_but_the_last() {
    for (OperationLease &lease : owning_) {
      release(lease);
    }
    for (BorrowedLease &lease : std::span(borrowed_).first(borrowed_.size() - 1)) {
      release(lease);
    }
  }

  /// @brief The one lease @ref release_all_but_the_last leaves held.
  [[nodiscard]] const BorrowedLease &last() const { return borrowed_.back(); }

  /// @brief Release that last lease on the calling thread.
  void release_the_last() { release(borrowed_.back()); }

  /// @brief How many leases have been released, or are about to be.
  /// @details Counted before each release, so a drain that returns only after
  /// the last release must see every one of them.
  [[nodiscard]] uint32_t released() const { return released_.load(); }

private:
  template <typename LeaseType> void release(LeaseType &lease) {
    released_.fetch_add(1);
    lease = {};
  }

  // Declared first so that it is destroyed last: a borrowed lease must not
  // outlive the session it was admitted by.
  std::shared_ptr<simdojo::PciTransportSession> session_;
  std::vector<OperationLease> owning_;
  std::vector<BorrowedLease> borrowed_;
  std::atomic<uint32_t> released_{0};
};

TEST(PciTransportSession, DrainWaitsForLeasesAdmittedAndReleasedOnDifferentThreads) {
  // The session keeps a fixed set of admission counters (64 when this was
  // written) and gives each thread one of them. More admitting threads than
  // counters makes some of them share, and none of these leases is released by
  // the thread that admitted it. The threads run one after another, so no more
  // than a few are alive at once.
  constexpr uint32_t kAdmittingThreads = 72;
  auto attached = std::make_shared<AttachedSession>();
  const std::shared_ptr<simdojo::PciTransportSession> session = attached->session();
  ASSERT_NE(session, nullptr);

  auto leases = std::make_shared<HandedOverLeases>(session, kAdmittingThreads);
  for (uint32_t lease_index = 0; lease_index < kAdmittingThreads; ++lease_index) {
    BoundedTask admitting_thread([leases, lease_index] { leases->admit(lease_index); });
    ASSERT_TRUE(admitting_thread.finished_before_deadline())
        << "admission on an open session never returned";
    ASSERT_TRUE(leases->admitted(lease_index));
  }

  auto leases_released_when_drained = std::make_shared<std::atomic<uint32_t>>(0);
  BoundedTask drain([session, leases, leases_released_when_drained] {
    session->wait_until_drained();
    leases_released_when_drained->store(leases->released());
  });
  EXPECT_TRUE(admission_closed_before_deadline(*session))
      << "wait_until_drained() never closed admission";
  EXPECT_FALSE(session->acquire()) << "admitted after the drain had started";
  EXPECT_FALSE(session->borrow()) << "admitted after the drain had started";

  {
    BoundedTask releasing_thread([leases] { leases->release_all_but_the_last(); });
    EXPECT_TRUE(releasing_thread.finished_before_deadline())
        << "releasing a lease on a closing session never returned";
  }
  EXPECT_EQ(session->state(), SessionState::Closing)
      << "the drain completed with one lease still held";
  EXPECT_FALSE(drain.has_finished()) << "wait_until_drained() returned with one lease still held";
  std::array<std::byte, 4> bytes{};
  EXPECT_EQ(leases->last().read(kGuestAddress, bytes), simdojo::DmaAccessOutcome::Complete)
      << "the last lease lost its endpoints before it was released";

  leases->release_the_last();
  ASSERT_TRUE(drain.finished_before_deadline())
      << "wait_until_drained() did not return after the last lease was released on a thread "
         "other than the one that admitted it";
  EXPECT_EQ(leases_released_when_drained->load(), 2 * kAdmittingThreads)
      << "wait_until_drained() returned before every lease had been released";
  EXPECT_EQ(session->state(), SessionState::Revoked);
  EXPECT_TRUE(detaches_before_deadline(attached));
}

// A drain adds up every admission counter, and one it overlooked would let it
// return under a lease counted there. Each thread here is new, so each is given
// the next counter in turn, and each admits the only leases of a session of its
// own: a drain that overlooked that thread's counter has nothing else to wait
// for. More threads than there are counters (64 when this was written) puts a
// lease on every one of them.
//
// The check does not depend on timing. A drain that waits leaves the session
// closing; one that finds nothing to wait for closes and revokes it in a single
// step, so the session is never seen closing at all.
TEST(PciTransportSession, DrainWaitsForALeaseAdmittedOnAnyThread) {
  constexpr uint32_t kAdmittingThreads = 72;
  for (uint32_t thread_index = 0; thread_index < kAdmittingThreads; ++thread_index) {
    SCOPED_TRACE(testing::Message() << "admitting thread " << thread_index);
    auto attached = std::make_shared<AttachedSession>();
    const std::shared_ptr<simdojo::PciTransportSession> session = attached->session();
    ASSERT_NE(session, nullptr);
    auto leases = std::make_shared<HandedOverLeases>(session, 1);
    {
      BoundedTask admitting_thread([leases] { leases->admit(0); });
      ASSERT_TRUE(admitting_thread.finished_before_deadline())
          << "admission on an open session never returned";
    }
    ASSERT_TRUE(leases->admitted(0));

    BoundedTask drain = start_drain(session);
    ASSERT_TRUE(admission_closed_before_deadline(*session))
        << "wait_until_drained() never closed admission";
    EXPECT_EQ(session->state(), SessionState::Closing)
        << "the drain did not wait for leases admitted on another thread";

    leases->release_all_but_the_last();
    leases->release_the_last();
    ASSERT_TRUE(drain.finished_before_deadline())
        << "wait_until_drained() did not return after the leases were released";
    EXPECT_EQ(session->state(), SessionState::Revoked);
    ASSERT_TRUE(detaches_before_deadline(attached));
  }
}

TEST(PciTransportSession, ConcurrentDrainsAllReturnAfterTheLastRelease) {
  auto attached = std::make_shared<AttachedSession>();
  const std::shared_ptr<simdojo::PciTransportSession> session = attached->session();
  ASSERT_NE(session, nullptr);
  OperationLease lease = session->acquire();
  ASSERT_TRUE(lease);

  // Counted as each thread is about to wait. A drain that only starts after the
  // lease has gone finds nothing to wait for and says nothing about wakeups, so
  // the release below is held back until both have got this far. That makes it
  // likely that both are waiting by then; nothing can make it certain.
  auto drains_started = std::make_shared<std::atomic<uint32_t>>(0);
  const auto counted_drain = [session, drains_started] {
    drains_started->fetch_add(1);
    session->wait_until_drained();
  };
  BoundedTask first_drain(counted_drain);
  BoundedTask second_drain(counted_drain);
  EXPECT_TRUE(reaches_before_deadline(*drains_started, 2)) << "a drain thread never started";
  EXPECT_TRUE(admission_closed_before_deadline(*session))
      << "wait_until_drained() never closed admission";
  EXPECT_EQ(session->state(), SessionState::Closing);
  EXPECT_FALSE(first_drain.has_finished());
  EXPECT_FALSE(second_drain.has_finished());

  lease = {};
  ASSERT_TRUE(first_drain.finished_before_deadline())
      << "one of two concurrent drains missed the release that ended the wait";
  ASSERT_TRUE(second_drain.finished_before_deadline())
      << "one of two concurrent drains missed the release that ended the wait";
  EXPECT_EQ(session->state(), SessionState::Revoked);
  EXPECT_TRUE(detaches_before_deadline(attached));
}

/// @brief What the admitting threads and the revoking thread of one round saw.
/// @details Every thread of a round reports here while the others are still
/// running, so each count is atomic.
class RevocationRaceCounters {
public:
  /// @brief Count one admitted lease, and whatever was wrong with admitting it.
  /// @param[in] after_drain_returned The drain had already returned.
  /// @param[in] after_refusal The session had already refused this thread.
  /// @param[in] usable The lease reached the attached endpoints.
  void count_admission(bool after_drain_returned, bool after_refusal, bool usable) {
    admitted_.fetch_add(1);
    if (after_drain_returned) {
      admitted_after_drain_returned_.fetch_add(1);
    }
    if (after_refusal) {
      admitted_after_refusal_.fetch_add(1);
    }
    if (!usable) {
      unusable_leases_.fetch_add(1);
    }
  }
  void count_refusal() { refused_.fetch_add(1); }
  void note_drain_returned() { drain_returned_.store(true); }
  /// @brief Tell the threads still looping to stop, after one of them hung.
  void give_up() { gave_up_.store(true); }

  [[nodiscard]] uint64_t admitted() const { return admitted_.load(); }
  [[nodiscard]] uint64_t refused() const { return refused_.load(); }
  [[nodiscard]] uint64_t admitted_after_drain_returned() const {
    return admitted_after_drain_returned_.load();
  }
  [[nodiscard]] uint64_t admitted_after_refusal() const { return admitted_after_refusal_.load(); }
  [[nodiscard]] uint64_t unusable_leases() const { return unusable_leases_.load(); }
  [[nodiscard]] bool drain_returned() const { return drain_returned_.load(); }
  [[nodiscard]] bool gave_up() const { return gave_up_.load(); }

private:
  std::atomic<uint64_t> admitted_{0};
  std::atomic<uint64_t> refused_{0};
  std::atomic<uint64_t> admitted_after_drain_returned_{0};
  std::atomic<uint64_t> admitted_after_refusal_{0};
  std::atomic<uint64_t> unusable_leases_{0};
  std::atomic<bool> drain_returned_{false};
  std::atomic<bool> gave_up_{false};
};

/// @brief Admit, use and release leases until the session has refused
/// @p refusals_to_confirm admissions from this thread.
template <typename Admission>
void admit_until_refused(const AttachedSession &attached, RevocationRaceCounters &counters,
                         uint32_t refusals_to_confirm) {
  simdojo::PciTransportSession &session = *attached.session();
  std::array<std::byte, 4> bytes{};
  uint32_t refusals = 0;
  while (refusals < refusals_to_confirm && !counters.gave_up()) {
    const bool drain_had_returned = counters.drain_returned();
    const typename Admission::Lease lease = Admission::admit(session);
    if (!lease) {
      ++refusals;
      counters.count_refusal();
      continue;
    }
    const bool usable = lease.dma() == attached.endpoints().dma &&
                        lease.irq() == attached.endpoints().irq &&
                        lease.read(kGuestAddress, bytes) == simdojo::DmaAccessOutcome::Complete;
    // The drain is looked at a second time while the lease is still held, and
    // a drain may not return past a held lease.
    counters.count_admission(drain_had_returned || counters.drain_returned(), refusals != 0,
                             usable);
  }
}

TEST(PciTransportSession, NoAdmissionSucceedsOnceARacingDrainHasReturned) {
  constexpr uint32_t kRounds = 12;
  constexpr uint32_t kAdmittingThreads = 6;
  constexpr uint64_t kAdmissionsBeforeRevocation = 200;
  constexpr uint32_t kRefusalsToConfirm = 50;

  for (uint32_t round = 0; round < kRounds; ++round) {
    SCOPED_TRACE(testing::Message() << "round " << round);
    auto attached = std::make_shared<AttachedSession>();
    const std::shared_ptr<simdojo::PciTransportSession> session = attached->session();
    ASSERT_NE(session, nullptr);
    auto counters = std::make_shared<RevocationRaceCounters>();
    auto start = std::make_shared<std::barrier<>>(kAdmittingThreads + 1);

    std::deque<BoundedTask> threads;
    for (uint32_t thread_index = 0; thread_index < kAdmittingThreads; ++thread_index) {
      const bool owning = thread_index % 2 == 0;
      threads.emplace_back([attached, counters, start, owning] {
        start->arrive_and_wait();
        if (owning) {
          admit_until_refused<OwningAdmission>(*attached, *counters, kRefusalsToConfirm);
        } else {
          admit_until_refused<BorrowedAdmission>(*attached, *counters, kRefusalsToConfirm);
        }
      });
    }
    // Alternate rounds enter revocation the way a transport owner does, with
    // begin_revoke() first, and through wait_until_drained() alone.
    const bool revoke_first = round % 2 == 0;
    threads.emplace_back([session, counters, start, revoke_first] {
      start->arrive_and_wait();
      while (counters->admitted() < kAdmissionsBeforeRevocation && !counters->gave_up()) {
        std::this_thread::yield();
      }
      if (revoke_first) {
        (void)session->begin_revoke();
      }
      session->wait_until_drained();
      counters->note_drain_returned();
    });

    bool every_thread_finished = true;
    for (BoundedTask &racing_thread : threads) {
      if (racing_thread.finished_before_deadline()) {
        continue;
      }
      every_thread_finished = false;
      // Lets the threads still looping stop, so one hang is reported once.
      counters->give_up();
    }
    ASSERT_TRUE(every_thread_finished)
        << "deadlock or lost wakeup: a thread racing admission against revocation never "
           "finished (admitted "
        << counters->admitted() << ", refused " << counters->refused() << ", drain returned "
        << counters->drain_returned() << ")";

    EXPECT_EQ(counters->admitted_after_drain_returned(), 0u)
        << "a lease was admitted, or still held, after wait_until_drained() had returned";
    EXPECT_EQ(counters->admitted_after_refusal(), 0u)
        << "a thread was admitted after the session had already refused it";
    EXPECT_EQ(counters->unusable_leases(), 0u)
        << "an admitted lease did not reach the attached endpoints";
    EXPECT_GE(counters->admitted(), kAdmissionsBeforeRevocation);
    EXPECT_EQ(counters->refused(), uint64_t{kAdmittingThreads} * kRefusalsToConfirm);
    EXPECT_EQ(attached->transport().reads(), counters->admitted());
    EXPECT_EQ(session->state(), SessionState::Revoked);
    EXPECT_FALSE(session->acquire());
    EXPECT_FALSE(session->borrow());
    ASSERT_TRUE(detaches_before_deadline(attached))
        << "admissions racing revocation left an operation counted";
  }
}

/// @brief Device memory that offers a transport session and nothing else.
class SessionOnlyMemory final : public rocjitsu::PciMemoryAccess {
public:
  explicit SessionOnlyMemory(std::shared_ptr<simdojo::PciTransportSession> session)
      : session_(std::move(session)) {}

  [[nodiscard]] std::shared_ptr<simdojo::PciTransportSession>
  capture_transport_session() const override {
    return session_;
  }
  bool read_vram(uint64_t, std::span<std::byte>) override { return false; }
  bool write_vram(uint64_t, std::span<const std::byte>) override { return false; }
  rocjitsu::amdgpu::AtomicLoadResult atomic_load_vram(uint64_t, uint32_t) override {
    return {.outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted};
  }
  rocjitsu::amdgpu::VmAccessOutcome atomic_store_vram(uint64_t, uint32_t, uint64_t) override {
    return rocjitsu::amdgpu::VmAccessOutcome::Faulted;
  }
  rocjitsu::amdgpu::AtomicCompareExchangeResult compare_exchange_vram(uint64_t, uint32_t, uint64_t,
                                                                      uint64_t) override {
    return {.outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted};
  }
  bool read_register(uint64_t, uint32_t &) override { return false; }
  bool write_register(uint64_t, uint32_t) override { return false; }
  bool deliver_interrupt(const rocjitsu::InterruptEntry &) override { return false; }

private:
  std::shared_ptr<simdojo::PciTransportSession> session_;
};

/// @brief A VM backing adapter together with the device memory it refers to.
class PhysicalAccessUnderTest {
public:
  explicit PhysicalAccessUnderTest(std::shared_ptr<simdojo::PciTransportSession> session)
      : memory_(std::move(session)), physical_(memory_) {}

  [[nodiscard]] rocjitsu::PciPhysicalMemoryAccess &physical() { return physical_; }

private:
  SessionOnlyMemory memory_;
  rocjitsu::PciPhysicalMemoryAccess physical_;
};

TEST(PciTransportSession, GuestMemoryReadInFlightHoldsOffTheDrain) {
  auto attached = std::make_shared<AttachedSession>();
  const std::shared_ptr<simdojo::PciTransportSession> session = attached->session();
  ASSERT_NE(session, nullptr);
  auto backing = std::make_shared<PhysicalAccessUnderTest>(session);
  auto read_outcome = std::make_shared<std::atomic<rocjitsu::amdgpu::VmAccessOutcome>>(
      rocjitsu::amdgpu::VmAccessOutcome::Faulted);

  attached->transport().hold_next_read();
  BoundedTask guest_read([attached, backing, read_outcome] {
    std::array<std::byte, 4> bytes{};
    read_outcome->store(
        backing->physical().read(rocjitsu::amdgpu::VmMemoryDomain::System, kGuestAddress, bytes));
  });
  const bool read_reached_transport = attached->transport().read_held_before_deadline();
  // Without a read parked there the hold is still armed, and it would park
  // whichever read came next: a late one from the helper thread, or the one
  // this thread makes below.
  if (!read_reached_transport) {
    attached->transport().resume_read();
  }
  ASSERT_TRUE(read_reached_transport) << "the guest memory read never reached the transport";

  BoundedTask drain = start_drain(session);
  EXPECT_TRUE(admission_closed_before_deadline(*session))
      << "wait_until_drained() never closed admission";
  EXPECT_EQ(session->state(), SessionState::Closing)
      << "the drain completed while a guest memory read was still inside the transport";
  EXPECT_FALSE(drain.has_finished())
      << "wait_until_drained() returned while a guest memory read was still inside the transport";
  std::array<std::byte, 4> bytes{};
  EXPECT_EQ(
      backing->physical().read(rocjitsu::amdgpu::VmMemoryDomain::System, kGuestAddress, bytes),
      rocjitsu::amdgpu::VmAccessOutcome::Unavailable)
      << "a guest memory read was admitted after revocation had begun";

  attached->transport().resume_read();
  ASSERT_TRUE(guest_read.finished_before_deadline())
      << "the guest memory read never returned after the transport resumed it";
  EXPECT_EQ(read_outcome->load(), rocjitsu::amdgpu::VmAccessOutcome::Complete)
      << "revocation cut short a guest memory read that had already been admitted";
  ASSERT_TRUE(drain.finished_before_deadline())
      << "wait_until_drained() did not return after the guest memory read finished";
  EXPECT_EQ(session->state(), SessionState::Revoked);
  EXPECT_EQ(attached->transport().reads(), 1u);
  EXPECT_TRUE(detaches_before_deadline(attached));
}

} // namespace
