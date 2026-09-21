// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file interrupt_sink.h
/// @brief Revocable ownership for frontend-specific GPU interrupt delivery.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>

namespace rocjitsu::amdgpu {

class InterruptSubscription;

/// @brief Copyable route carried by a queue and every dispatch derived from it.
/// @details Delivery is a no-op after the owning subscription is revoked. The
/// route keeps only the control state alive; revocation clears the callback so
/// queue records cannot retain their frontend.
class InterruptSink {
public:
  InterruptSink() = default;

  void deliver(uint32_t process_id, uint32_t event_id) const;
  explicit operator bool() const { return state_ != nullptr; }

private:
  class State;
  friend class InterruptSubscription;

  explicit InterruptSink(std::shared_ptr<State> state) : state_(std::move(state)) {}

  std::shared_ptr<State> state_;
};

/// @brief Move-only ownership of one frontend interrupt callback.
/// @details Reset revokes new calls and drains every callback running on other
/// threads. Reset from inside the callback is also supported: the current call
/// owns a local callback copy and is excluded from its own drain wait.
class InterruptSubscription {
public:
  using Callback = std::function<void(uint32_t process_id, uint32_t event_id)>;

  InterruptSubscription() = default;
  explicit InterruptSubscription(Callback callback);
  ~InterruptSubscription();
  InterruptSubscription(InterruptSubscription &&other) noexcept;
  InterruptSubscription &operator=(InterruptSubscription &&other) noexcept;
  InterruptSubscription(const InterruptSubscription &) = delete;
  InterruptSubscription &operator=(const InterruptSubscription &) = delete;

  [[nodiscard]] InterruptSink sink() const {
    return InterruptSink(state_.load(std::memory_order_acquire));
  }
  void reset();
  explicit operator bool() const { return state_.load(std::memory_order_acquire) != nullptr; }

private:
  std::atomic<std::shared_ptr<InterruptSink::State>> state_;
};

class InterruptSink::State {
public:
  explicit State(InterruptSubscription::Callback callback) : callback_(std::move(callback)) {}

  void deliver(uint32_t process_id, uint32_t event_id) {
    InterruptSubscription::Callback callback;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!active_)
        return;
      callback = callback_;
      ++callbacks_in_flight_;
    }
    CallbackLease lease(this);
    callback(process_id, event_id);
  }

  void revoke_and_drain() {
    const uint64_t callbacks_on_this_thread = CallbackLease::count(this);
    std::unique_lock<std::mutex> lock(mutex_);
    active_ = false;
    idle_.wait(lock, [this, callbacks_on_this_thread] {
      return callbacks_in_flight_ <= callbacks_on_this_thread;
    });
    callback_ = {};
  }

private:
  class CallbackLease {
  public:
    explicit CallbackLease(State *state) : state_(state), previous_(current_) { current_ = this; }
    ~CallbackLease() {
      {
        std::lock_guard<std::mutex> lock(state_->mutex_);
        --state_->callbacks_in_flight_;
        state_->idle_.notify_all();
      }
      current_ = previous_;
    }
    CallbackLease(const CallbackLease &) = delete;
    CallbackLease &operator=(const CallbackLease &) = delete;

    [[nodiscard]] static uint64_t count(const State *state) {
      uint64_t count = 0;
      for (const CallbackLease *lease = current_; lease != nullptr; lease = lease->previous_)
        count += lease->state_ == state;
      return count;
    }

  private:
    State *state_ = nullptr;
    CallbackLease *previous_ = nullptr;
    inline static thread_local CallbackLease *current_ = nullptr;
  };

  std::mutex mutex_;
  std::condition_variable idle_;
  InterruptSubscription::Callback callback_;
  uint64_t callbacks_in_flight_ = 0;
  bool active_ = true;
};

inline void InterruptSink::deliver(uint32_t process_id, uint32_t event_id) const {
  if (state_)
    state_->deliver(process_id, event_id);
}

inline InterruptSubscription::InterruptSubscription(Callback callback) {
  if (callback)
    state_.store(std::make_shared<InterruptSink::State>(std::move(callback)),
                 std::memory_order_release);
}

inline InterruptSubscription::~InterruptSubscription() { reset(); }

inline InterruptSubscription::InterruptSubscription(InterruptSubscription &&other) noexcept
    : state_(other.state_.exchange(nullptr, std::memory_order_acq_rel)) {}

inline InterruptSubscription &
InterruptSubscription::operator=(InterruptSubscription &&other) noexcept {
  if (this == &other)
    return *this;
  reset();
  state_.store(other.state_.exchange(nullptr, std::memory_order_acq_rel),
               std::memory_order_release);
  return *this;
}

inline void InterruptSubscription::reset() {
  std::shared_ptr<InterruptSink::State> state = state_.exchange(nullptr, std::memory_order_acq_rel);
  if (!state)
    return;
  state->revoke_and_drain();
}

} // namespace rocjitsu::amdgpu
