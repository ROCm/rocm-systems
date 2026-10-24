// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/gpu_alias_ingress.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rocjitsu {

std::optional<amdgpu::MemoryIngressPermit>
GpuAliasIngressGate::try_acquire(std::span<const Key> aliases) noexcept {
  return try_acquire_impl(aliases, nullptr, nullptr);
}

std::optional<amdgpu::MemoryIngressPermit>
GpuAliasIngressGate::try_acquire_for_drained_owner(std::span<const Key> aliases,
                                                   const DrainAuthority &authority) noexcept {
  return try_acquire_impl(aliases, &authority, nullptr);
}

void GpuAliasIngressGate::EffectWindow::reset() noexcept {
  auto state = std::exchange(state_, {});
  auto gate = std::exchange(gate_, {});
  if (gate && state)
    gate->release_effect_window(state);
}

std::optional<GpuAliasIngressGate::EffectWindow>
GpuAliasIngressGate::try_freeze_for_effect(std::span<const Key> aliases) noexcept {
  if (aliases.empty())
    return std::nullopt;
  try {
    auto state = std::make_shared<EffectWindow::State>();
    state->keys.assign(aliases.begin(), aliases.end());
    std::sort(state->keys.begin(), state->keys.end());
    state->keys.erase(std::unique(state->keys.begin(), state->keys.end()), state->keys.end());
    std::vector<Key> inserted;
    inserted.reserve(state->keys.size());
    auto self = shared_from_this();
    {
      std::lock_guard lock(mutex_);
      if (single_close_claim_ || batch_close_claim_ || effect_window_claim_)
        return std::nullopt;
      for (const Key &key : state->keys) {
        const auto it = entries_.find(key);
        if (it != entries_.end() && (!it->second.open || it->second.active != 0))
          return std::nullopt;
      }
      try {
        for (const Key &key : state->keys)
          if (entries_.try_emplace(key).second)
            inserted.push_back(key);
      } catch (...) {
        for (const Key &key : inserted)
          entries_.erase(key);
        return std::nullopt;
      }
      for (const Key &key : state->keys)
        entries_.find(key)->second.open = false;
      effect_window_claim_ = state;
    }
    return EffectWindow(std::move(self), std::move(state));
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<amdgpu::MemoryIngressPermit>
GpuAliasIngressGate::try_acquire_for_effect(std::span<const Key> aliases,
                                            const EffectWindow &window) noexcept {
  if (aliases.empty())
    return std::nullopt;
  return try_acquire_impl(aliases, nullptr, &window);
}

bool GpuAliasIngressGate::drained_owner_covers(std::span<const Key> aliases,
                                               const DrainAuthority &authority) const noexcept {
  if (aliases.empty())
    return false;
  std::lock_guard lock(mutex_);
  return std::ranges::all_of(
      aliases, [&](const Key &key) { return drained_owner_covers_key_locked(authority, key); });
}

std::optional<amdgpu::MemoryIngressPermit>
GpuAliasIngressGate::try_acquire_impl(std::span<const Key> aliases, const DrainAuthority *authority,
                                      const EffectWindow *window) noexcept {
  try {
    std::vector<Key> keys(aliases.begin(), aliases.end());
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    if (keys.empty())
      return amdgpu::MemoryIngressPermit{};

    // Prepare the release closure before counting anything. If allocation
    // fails, admission declines without leaving an outstanding count behind.
    std::function<void()> release = [self = shared_from_this(), keys] { self->release(keys); };
    std::vector<Key> inserted;
    inserted.reserve(keys.size());
    {
      std::lock_guard lock(mutex_);
      if (window && !valid_effect_window_locked(*window))
        return std::nullopt;
      for (const Key &key : keys) {
        const auto it = entries_.find(key);
        if (window) {
          if (it == entries_.end() || it->second.open ||
              !std::binary_search(window->state_->keys.begin(), window->state_->keys.end(), key) ||
              it->second.active == SIZE_MAX)
            return std::nullopt;
          continue;
        }
        const bool drained_owner =
            authority != nullptr && drained_owner_covers_key_locked(*authority, key);
        if (it != entries_.end() &&
            ((!it->second.open && !drained_owner) || it->second.active == SIZE_MAX))
          return std::nullopt;
      }
      try {
        for (const Key &key : keys) {
          const bool was_inserted = entries_.try_emplace(key).second;
          if (was_inserted)
            inserted.push_back(key);
        }
      } catch (...) {
        for (const Key &key : inserted)
          entries_.erase(key);
        throw;
      }
      for (const Key &key : keys) {
        Entry &entry = entries_.find(key)->second;
        if (entry.batch && entry.active == 0)
          ++entry.batch->active_generations;
        ++entry.active;
      }
    }
    return amdgpu::MemoryIngressPermit(std::move(release));
  } catch (...) {
    return std::nullopt;
  }
}

bool GpuAliasIngressGate::valid_effect_window_locked(const EffectWindow &window) const {
  return window.gate_.get() == this && window.state_ && effect_window_claim_ == window.state_ &&
         window.state_->active;
}

void GpuAliasIngressGate::finish_effect_window_if_quiescent_locked() noexcept {
  const auto &state = effect_window_claim_;
  if (!state || !state->release_requested)
    return;
  for (const Key &key : state->keys) {
    const auto it = entries_.find(key);
    assert(it != entries_.end() && !it->second.open);
    if (it == entries_.end() || it->second.active != 0)
      return;
  }
  for (const Key &key : state->keys)
    entries_.erase(key);
  effect_window_claim_.reset();
}

void GpuAliasIngressGate::release_effect_window(
    const std::shared_ptr<EffectWindow::State> &state) noexcept {
  std::lock_guard lock(mutex_);
  if (effect_window_claim_ != state)
    return;
  state->active = false;
  state->release_requested = true;
  finish_effect_window_if_quiescent_locked();
}

bool GpuAliasIngressGate::valid_drained_owner_locked(const DrainAuthority &authority) const {
  if (authority.gate_.get() != this || !authority.state_ || !authority.state_->active)
    return false;
  if (authority.key_) {
    const auto it = entries_.find(*authority.key_);
    return single_close_claim_ == authority.key_ && !batch_close_claim_ && it != entries_.end() &&
           !it->second.open && it->second.drain_pending && !it->second.retryable &&
           !it->second.batch && !it->second.drain;
  }
  const auto batch = authority.batch_.lock();
  return batch && batch_close_claim_ == batch && !single_close_claim_ && batch->drain_pending &&
         !batch->retryable && !batch->drain;
}

bool GpuAliasIngressGate::drained_owner_covers_key_locked(const DrainAuthority &authority,
                                                          Key key) const {
  if (!valid_drained_owner_locked(authority))
    return false;
  if (authority.key_)
    return *authority.key_ == key;
  const auto batch = authority.batch_.lock();
  const auto it = entries_.find(key);
  return batch && std::binary_search(batch->keys.begin(), batch->keys.end(), key) &&
         it != entries_.end() && !it->second.open && it->second.batch == batch;
}

void GpuAliasIngressGate::set_drain_authority_active(const DrainAuthority &authority,
                                                     bool active) noexcept {
  std::lock_guard lock(mutex_);
  authority.state_->active = active;
}

std::optional<bool>
GpuAliasIngressGate::final_writeback_done(const DrainAuthority &authority) const {
  std::lock_guard lock(mutex_);
  if (!valid_drained_owner_locked(authority))
    return std::nullopt;
  if (authority.key_)
    return entries_.find(*authority.key_)->second.final_writeback_done;
  return authority.batch_.lock()->final_writeback_done;
}

bool GpuAliasIngressGate::mark_final_writeback_done(const DrainAuthority &authority) {
  std::lock_guard lock(mutex_);
  if (!valid_drained_owner_locked(authority))
    return false;
  if (authority.key_)
    entries_.find(*authority.key_)->second.final_writeback_done = true;
  else
    authority.batch_.lock()->final_writeback_done = true;
  return true;
}

std::function<void()> GpuAliasIngressGate::make_drain_continuation(Key alias,
                                                                   DrainedOwner drained) {
  auto self = shared_from_this();
  auto state = std::make_shared<DrainAuthority::State>();
  return [self = std::move(self), alias, drained = std::move(drained), state = std::move(state)] {
    const DrainAuthority authority(self, alias, state);
    self->set_drain_authority_active(authority, true);
    try {
      drained(authority);
    } catch (...) {
      self->set_drain_authority_active(authority, false);
      self->fail_drain(alias);
      throw;
    }
    self->set_drain_authority_active(authority, false);
    self->finish_drain(alias);
  };
}

std::function<void()>
GpuAliasIngressGate::make_batch_drain_continuation(std::shared_ptr<Batch> batch,
                                                   DrainedOwner drained) {
  auto state = std::make_shared<DrainAuthority::State>();
  return [self = shared_from_this(), weak_batch = std::weak_ptr<Batch>(batch),
          drained = std::move(drained), state = std::move(state)] {
    const auto batch = weak_batch.lock();
    if (!batch)
      throw std::runtime_error("alias drain batch was discarded before maintenance");
    const DrainAuthority authority(self, batch, state);
    self->set_drain_authority_active(authority, true);
    try {
      drained(authority);
    } catch (...) {
      self->set_drain_authority_active(authority, false);
      self->fail_batch(batch);
      throw;
    }
    self->set_drain_authority_active(authority, false);
    self->finish_batch(batch);
  };
}

bool GpuAliasIngressGate::close(Key alias, OwnerPost owner_post, std::function<void()> drained) {
  if (!drained)
    return false;
  return close_with_drain_authority(
      alias, std::move(owner_post),
      [drained = std::move(drained)](const DrainAuthority &) { drained(); });
}

bool GpuAliasIngressGate::close_with_drain_authority(Key alias, OwnerPost owner_post,
                                                     DrainedOwner drained) {
  if (!owner_post || !drained)
    return false;
  std::function<void()> continuation;
  try {
    continuation = make_drain_continuation(alias, std::move(drained));
  } catch (...) {
    return false;
  }
  std::optional<Drain> ready;
  {
    std::lock_guard lock(mutex_);
    if (single_close_claim_ || batch_close_claim_ || effect_window_claim_)
      return false;
    Entry &entry = entries_[alias];
    if (!entry.open)
      return false;
    single_close_claim_ = alias;
    entry.open = false;
    entry.drain_pending = true;
    entry.retryable = false;
    if (entry.active == 0)
      ready = Drain{std::move(owner_post), std::move(continuation)};
    else
      entry.drain = Drain{std::move(owner_post), std::move(continuation)};
  }
  if (ready) {
    try {
      ready->owner_post(std::move(ready->continuation));
    } catch (...) {
      fail_drain(alias);
      return false;
    }
  }
  return true;
}

bool GpuAliasIngressGate::retry_closed(Key alias, OwnerPost owner_post,
                                       std::function<void()> drained) {
  if (!drained)
    return false;
  return retry_closed_with_drain_authority(
      alias, std::move(owner_post),
      [drained = std::move(drained)](const DrainAuthority &) { drained(); });
}

bool GpuAliasIngressGate::retry_closed_with_drain_authority(Key alias, OwnerPost owner_post,
                                                            DrainedOwner drained) {
  if (!owner_post || !drained)
    return false;
  std::function<void()> continuation;
  try {
    continuation = make_drain_continuation(alias, std::move(drained));
  } catch (...) {
    return false;
  }
  {
    std::lock_guard lock(mutex_);
    const auto it = entries_.find(alias);
    if (single_close_claim_ != alias || batch_close_claim_ || it == entries_.end() ||
        it->second.open || it->second.batch || it->second.active != 0 ||
        !it->second.drain_pending || !it->second.retryable || it->second.drain)
      return false;
    // A queued retry is not itself eligible for a second retry.
    it->second.retryable = false;
  }
  try {
    owner_post(std::move(continuation));
  } catch (...) {
    fail_drain(alias);
    return false;
  }
  return true;
}

bool GpuAliasIngressGate::close_batch(std::span<const Key> aliases, OwnerPost owner_post,
                                      std::function<void()> drained) {
  if (!drained)
    return false;
  return close_batch_with_drain_authority(
      aliases, std::move(owner_post),
      [drained = std::move(drained)](const DrainAuthority &) { drained(); });
}

bool GpuAliasIngressGate::close_batch_with_drain_authority(std::span<const Key> aliases,
                                                           OwnerPost owner_post,
                                                           DrainedOwner drained) {
  if (!owner_post || !drained || aliases.empty())
    return false;
  std::shared_ptr<Batch> batch;
  std::vector<Key> inserted;
  try {
    batch = std::make_shared<Batch>();
    batch->keys.assign(aliases.begin(), aliases.end());
    std::sort(batch->keys.begin(), batch->keys.end());
    batch->keys.erase(std::unique(batch->keys.begin(), batch->keys.end()), batch->keys.end());
    inserted.reserve(batch->keys.size());
    batch->drain.emplace(
        Drain{std::move(owner_post), make_batch_drain_continuation(batch, std::move(drained))});
  } catch (...) {
    return false;
  }

  std::optional<Drain> ready;
  {
    std::lock_guard lock(mutex_);
    // A failed single-generation UNMAP may be adopted by FREE for the same
    // allocation. No unrelated close may overlap this batch's maintenance.
    if (batch_close_claim_ || effect_window_claim_ ||
        (single_close_claim_ &&
         std::ranges::find(batch->keys, *single_close_claim_) == batch->keys.end()))
      return false;
    for (const Key &key : batch->keys) {
      const auto it = entries_.find(key);
      if (it != entries_.end() &&
          (!it->second.open &&
           (it->second.batch || it->second.active != 0 || !it->second.drain_pending ||
            !it->second.retryable || it->second.drain)))
        return false;
    }
    try {
      for (const Key &key : batch->keys)
        if (entries_.try_emplace(key).second)
          inserted.push_back(key);
    } catch (...) {
      for (const Key &key : inserted)
        entries_.erase(key);
      return false;
    }
    single_close_claim_.reset();
    batch_close_claim_ = batch;
    for (const Key &key : batch->keys) {
      Entry &entry = entries_.find(key)->second;
      if (entry.active != 0)
        ++batch->active_generations;
      entry.open = false;
      entry.drain_pending = false;
      entry.retryable = false;
      entry.drain.reset();
      entry.batch = batch;
    }
    if (batch->active_generations == 0) {
      ready = std::move(batch->drain);
      batch->drain.reset();
    }
  }
  if (ready) {
    try {
      ready->owner_post(std::move(ready->continuation));
    } catch (...) {
      fail_batch(batch);
      return false;
    }
  }
  return true;
}

bool GpuAliasIngressGate::retry_closed_batch(std::span<const Key> aliases, OwnerPost owner_post,
                                             std::function<void()> drained) {
  if (!drained)
    return false;
  return retry_closed_batch_with_drain_authority(
      aliases, std::move(owner_post),
      [drained = std::move(drained)](const DrainAuthority &) { drained(); });
}

bool GpuAliasIngressGate::retry_closed_batch_with_drain_authority(std::span<const Key> aliases,
                                                                  OwnerPost owner_post,
                                                                  DrainedOwner drained) {
  if (!owner_post || !drained || aliases.empty())
    return false;
  std::vector<Key> keys;
  std::shared_ptr<Batch> batch;
  try {
    keys.assign(aliases.begin(), aliases.end());
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::lock_guard lock(mutex_);
    const auto first = entries_.find(keys.front());
    if (first == entries_.end() || !first->second.batch)
      return false;
    batch = first->second.batch;
    if (batch_close_claim_ != batch || single_close_claim_ || batch->keys != keys ||
        !batch->drain_pending || !batch->retryable || batch->active_generations != 0 ||
        batch->drain)
      return false;
    for (const Key &key : keys) {
      const auto it = entries_.find(key);
      if (it == entries_.end() || it->second.batch != batch || it->second.active != 0)
        return false;
    }
  } catch (...) {
    return false;
  }
  std::function<void()> continuation;
  try {
    continuation = make_batch_drain_continuation(batch, std::move(drained));
  } catch (...) {
    return false;
  }
  {
    std::lock_guard lock(mutex_);
    if (batch_close_claim_ != batch || single_close_claim_ || !batch->drain_pending ||
        !batch->retryable || batch->active_generations != 0)
      return false;
    for (const Key &key : keys) {
      const auto it = entries_.find(key);
      if (it == entries_.end() || it->second.batch != batch || it->second.active != 0)
        return false;
    }
    batch->retryable = false;
  }
  try {
    owner_post(std::move(continuation));
  } catch (...) {
    fail_batch(batch);
    return false;
  }
  return true;
}

bool GpuAliasIngressGate::reopen_batch(std::span<const Key> aliases) {
  if (aliases.empty())
    return false;
  std::lock_guard lock(mutex_);
  const auto first = entries_.find(aliases.front());
  if (first == entries_.end() || !first->second.batch)
    return false;
  const auto &batch = first->second.batch;
  if (batch_close_claim_ != batch || single_close_claim_ || aliases.size() != batch->keys.size() ||
      batch->drain_pending || batch->active_generations != 0)
    return false;
  for (size_t index = 0; index < aliases.size(); ++index) {
    const Key &key = aliases[index];
    for (size_t previous = 0; previous < index; ++previous)
      if (aliases[previous] == key)
        return false;
    const auto it = entries_.find(key);
    if (it == entries_.end() || it->second.batch != batch || it->second.active != 0)
      return false;
  }
  for (const Key &key : aliases)
    entries_.erase(key);
  batch_close_claim_.reset();
  return true;
}

bool GpuAliasIngressGate::reopen(Key alias) {
  std::lock_guard lock(mutex_);
  const auto it = entries_.find(alias);
  if (single_close_claim_ != alias || batch_close_claim_ || it == entries_.end() ||
      it->second.open || it->second.batch || it->second.active != 0 || it->second.drain_pending)
    return false;
  entries_.erase(it);
  single_close_claim_.reset();
  return true;
}

bool GpuAliasIngressGate::closed(Key alias) const {
  std::lock_guard lock(mutex_);
  const auto it = entries_.find(alias);
  return it != entries_.end() && !it->second.open;
}

bool GpuAliasIngressGate::close_claimed() const {
  std::lock_guard lock(mutex_);
  return single_close_claim_.has_value() || batch_close_claim_ != nullptr;
}

bool GpuAliasIngressGate::close_claimed_by_process(uint64_t process_incarnation) const {
  std::lock_guard lock(mutex_);
  if (single_close_claim_)
    return single_close_claim_->process_incarnation == process_incarnation;
  return batch_close_claim_ && !batch_close_claim_->keys.empty() &&
         std::ranges::all_of(batch_close_claim_->keys, [&](const Key &key) {
           return key.process_incarnation == process_incarnation;
         });
}

bool GpuAliasIngressGate::discard_failed(Key alias) {
  std::lock_guard lock(mutex_);
  const auto it = entries_.find(alias);
  if (it != entries_.end() && it->second.batch) {
    const auto batch = it->second.batch;
    if (batch_close_claim_ != batch || single_close_claim_ || !batch->drain_pending ||
        !batch->retryable || batch->active_generations != 0)
      return false;
    for (const Key &key : batch->keys) {
      const auto member = entries_.find(key);
      if (member == entries_.end() || member->second.batch != batch || member->second.active != 0)
        return false;
    }
    for (const Key &key : batch->keys)
      entries_.erase(key);
    batch_close_claim_.reset();
    return true;
  }
  if (single_close_claim_ != alias || batch_close_claim_ || it == entries_.end() ||
      it->second.open || it->second.active != 0 || !it->second.drain_pending ||
      !it->second.retryable || it->second.drain)
    return false;
  entries_.erase(it);
  single_close_claim_.reset();
  return true;
}

void GpuAliasIngressGate::wait_for_process_quiescent(uint64_t process_incarnation) noexcept {
  std::unique_lock lock(mutex_);
  quiescent_.wait(lock, [&] {
    const Key first{.process_incarnation = process_incarnation};
    for (auto it = entries_.lower_bound(first);
         it != entries_.end() && it->first.process_incarnation == process_incarnation; ++it)
      if (it->second.active != 0)
        return false;
    return true;
  });
}

void GpuAliasIngressGate::release(std::span<const Key> aliases) noexcept {
  for (const Key &key : aliases) {
    std::optional<Drain> ready;
    std::shared_ptr<Batch> batch;
    bool became_inactive = false;
    {
      std::lock_guard lock(mutex_);
      auto it = entries_.find(key);
      assert(it != entries_.end() && it->second.active != 0);
      if (--it->second.active == 0) {
        became_inactive = true;
        if (it->second.open) {
          entries_.erase(it);
        } else if (it->second.batch) {
          batch = it->second.batch;
          assert(batch->active_generations != 0);
          if (--batch->active_generations == 0) {
            ready = std::move(batch->drain);
            batch->drain.reset();
          }
        } else if (it->second.drain) {
          ready = std::move(it->second.drain);
          it->second.drain.reset();
        } else if (effect_window_claim_ && effect_window_claim_->release_requested) {
          finish_effect_window_if_quiescent_locked();
        }
      }
    }
    if (became_inactive)
      quiescent_.notify_all();
    if (ready) {
      try {
        ready->owner_post(std::move(ready->continuation));
      } catch (...) {
        if (batch)
          fail_batch(batch);
        else
          fail_drain(key);
      }
    }
  }
}

void GpuAliasIngressGate::fail_drain(Key alias) noexcept {
  std::lock_guard lock(mutex_);
  const auto it = entries_.find(alias);
  if (it != entries_.end() && !it->second.open && it->second.drain_pending)
    it->second.retryable = true;
}

void GpuAliasIngressGate::finish_drain(Key alias) noexcept {
  std::lock_guard lock(mutex_);
  const auto it = entries_.find(alias);
  if (it != entries_.end()) {
    it->second.drain_pending = false;
    it->second.retryable = false;
  }
}

void GpuAliasIngressGate::fail_batch(const std::shared_ptr<Batch> &batch) noexcept {
  std::lock_guard lock(mutex_);
  if (batch->drain_pending)
    batch->retryable = true;
}

void GpuAliasIngressGate::finish_batch(const std::shared_ptr<Batch> &batch) noexcept {
  std::lock_guard lock(mutex_);
  batch->drain_pending = false;
  batch->retryable = false;
}
} // namespace rocjitsu
