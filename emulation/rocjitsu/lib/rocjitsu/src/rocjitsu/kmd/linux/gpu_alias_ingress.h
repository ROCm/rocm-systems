// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_alias_ingress.h
/// @brief Counted cache ingress for canonical KFD alias generations.

#pragma once

#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include <compare>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu {

/// @brief An independent barrier for CU and SDMA cache ingress.
/// @details Admission of all aliases touched by one operation is atomic. A
/// closer supplies an owner-thread poster; draining schedules its continuation
/// through that poster and never executes maintenance on a releasing CU thread.
/// Only one retirement close, close batch, or live effect window may own the
/// gate at once. This includes a failed drain awaiting retry and an effect
/// window awaiting its last bypass permit. Group cache maintenance invalidates
/// every local domain epoch, so a second closed dirty alias cannot be left
/// unresolved.
class GpuAliasIngressGate : public std::enable_shared_from_this<GpuAliasIngressGate> {
  struct Batch;

public:
  class Key {
  public:
    uint64_t process_incarnation = 0;
    uint32_t gpu_id = 0;
    uint64_t gpu_va = 0;
    uint64_t backing_id = 0;
    uint64_t admission_generation = 0;
    auto operator<=>(const Key &) const = default;
  };

  using OwnerPost = std::function<void(std::function<void()>)>;

  /// @brief Authority for one exact closed generation or closed batch while its
  /// drained owner continuation is running. A retained copy expires on return.
  class DrainAuthority {
  public:
    DrainAuthority(const DrainAuthority &) = default;
    DrainAuthority &operator=(const DrainAuthority &) = default;

  private:
    friend class GpuAliasIngressGate;
    struct State {
      bool active = false; // Protected by the owning gate's mutex.
    };
    DrainAuthority(std::shared_ptr<GpuAliasIngressGate> gate, Key key, std::shared_ptr<State> state)
        : gate_(std::move(gate)), key_(key), state_(std::move(state)) {}
    DrainAuthority(std::shared_ptr<GpuAliasIngressGate> gate, std::shared_ptr<Batch> batch,
                   std::shared_ptr<State> state)
        : gate_(std::move(gate)), batch_(std::move(batch)), state_(std::move(state)) {}
    std::shared_ptr<GpuAliasIngressGate> gate_;
    std::optional<Key> key_;
    std::weak_ptr<Batch> batch_;
    std::shared_ptr<State> state_;
  };
  using DrainedOwner = std::function<void(const DrainAuthority &)>;

  /// @brief Exclusive, reversible admission freeze for one live effect.
  /// @details Destruction restores admission after any effect bypass permits
  /// have left. A moved-from or released window carries no bypass authority.
  class EffectWindow {
  public:
    EffectWindow() = default;
    EffectWindow(const EffectWindow &) = delete;
    EffectWindow &operator=(const EffectWindow &) = delete;
    EffectWindow(EffectWindow &&other) noexcept
        : gate_(std::exchange(other.gate_, {})), state_(std::exchange(other.state_, {})) {}
    EffectWindow &operator=(EffectWindow &&other) noexcept {
      if (this != &other) {
        reset();
        gate_ = std::exchange(other.gate_, {});
        state_ = std::exchange(other.state_, {});
      }
      return *this;
    }
    ~EffectWindow() { reset(); }

    void reset() noexcept;
    explicit operator bool() const noexcept { return state_ != nullptr; }

  private:
    friend class GpuAliasIngressGate;
    struct State {
      std::vector<Key> keys;
      bool active = true; // Protected by the owning gate's mutex.
      bool release_requested = false;
    };
    EffectWindow(std::shared_ptr<GpuAliasIngressGate> gate, std::shared_ptr<State> state)
        : gate_(std::move(gate)), state_(std::move(state)) {}
    std::shared_ptr<GpuAliasIngressGate> gate_;
    std::shared_ptr<State> state_;
  };

  static Key key(const KfdProcess::GpuAliasRecord &alias) {
    return {.process_incarnation = alias.process_incarnation,
            .gpu_id = alias.gpu_id,
            .gpu_va = alias.gpu_va,
            .backing_id = alias.backing_id,
            .admission_generation = alias.admission_generation};
  }

  /// @brief Atomically admit ordinary ingress for every requested alias generation.
  /// @details A closed generation denies the entire request. A successful
  /// permit keeps every admitted generation counted until its release.
  [[nodiscard]] std::optional<amdgpu::MemoryIngressPermit>
  try_acquire(std::span<const Key> aliases) noexcept;
  /// @brief Count every touched alias atomically during drained maintenance.
  /// @details Only the authority's exact closed key or batch keys bypass
  /// ordinary denial; all other keys follow try_acquire().
  [[nodiscard]] std::optional<amdgpu::MemoryIngressPermit>
  try_acquire_for_drained_owner(std::span<const Key> aliases,
                                const DrainAuthority &authority) noexcept;
  /// @brief Whether every key is an exact member of this active drained close.
  /// @details A drained-owner permit may also count unrelated open keys through
  /// ordinary ingress; those keys must not authorize controlled canonical
  /// writeback.
  [[nodiscard]] bool drained_owner_covers(std::span<const Key> aliases,
                                          const DrainAuthority &authority) const noexcept;
  /// @brief Freeze selected open, idle generations atomically for a live effect.
  /// @details A retirement close or another effect window excludes this claim.
  /// Ordinary ingress to the selected keys is denied until the window and all
  /// its bypass permits have left. Unrelated ingress remains available.
  [[nodiscard]] std::optional<EffectWindow>
  try_freeze_for_effect(std::span<const Key> aliases) noexcept;
  /// @brief Admit exact selected aliases for the effect's dirty writeback.
  /// @details Every requested key must belong to the live window. This bypass
  /// cannot admit unrelated aliases or survive release of the window.
  [[nodiscard]] std::optional<amdgpu::MemoryIngressPermit>
  try_acquire_for_effect(std::span<const Key> aliases, const EffectWindow &window) noexcept;
  /// @brief Close one generation and post @p drained when all permits leave.
  /// @pre @p owner_post queues work to the gate owner's thread; it must not run
  ///      the continuation inline on the releasing thread. It either accepts
  ///      one callback and returns, or throws without retaining that callback.
  [[nodiscard]] bool close(Key alias, OwnerPost owner_post, std::function<void()> drained);
  /// @brief Close one generation and give its drained owner exact authority.
  /// @details The callback runs through @p owner_post on the gate owner's
  /// thread, never inline on a releasing thread. The authority covers only
  /// this closed generation and expires when the callback returns, including
  /// any copies retained by the callback.
  /// @pre @p owner_post obeys the same contract as close().
  [[nodiscard]] bool close_with_drain_authority(Key alias, OwnerPost owner_post,
                                                DrainedOwner drained);
  /// @brief Retry maintenance for a closed generation after its posted work failed.
  /// @details A continuation that is merely queued is not retryable. Failed
  /// posting or a throwing callback leaves the generation closed until the
  /// owner supplies another continuation through this method.
  [[nodiscard]] bool retry_closed(Key alias, OwnerPost owner_post, std::function<void()> drained);
  /// @brief Retry a closed generation with fresh drained-owner authority.
  /// @details The callback runs on the owner's thread and receives authority
  /// only for this closed generation. Copied authority expires on return.
  /// @pre @p owner_post obeys the same contract as close().
  [[nodiscard]] bool retry_closed_with_drain_authority(Key alias, OwnerPost owner_post,
                                                       DrainedOwner drained);
  /// @brief Remember completed final writeback until the exact generation or
  /// batch is reopened or discarded, including failed grant publication.
  [[nodiscard]] std::optional<bool> final_writeback_done(const DrainAuthority &authority) const;
  /// @brief Mark final writeback complete for the exact closed generation or batch.
  /// @details Returns false for expired or unrelated authority. Completion
  /// remains recorded across failed grant publication until reopen or discard.
  [[nodiscard]] bool mark_final_writeback_done(const DrainAuthority &authority);
  /// @brief Atomically close every generation and post one owner continuation
  /// after every permit leaves every generation.
  /// @details A failed single-generation UNMAP may leave an inactive, retryable
  /// closed generation. Closing a batch adopts it together with open aliases.
  /// @pre @p owner_post obeys the same owner-thread, no-inline contract as close().
  [[nodiscard]] bool close_batch(std::span<const Key> aliases, OwnerPost owner_post,
                                 std::function<void()> drained);
  /// @brief Close a batch and give its drained owner authority for that batch.
  /// @details The callback runs through @p owner_post on the gate owner's
  /// thread, never inline on a releasing thread. Authority covers exactly
  /// the closed batch and all retained copies expire when the callback returns.
  /// @pre @p owner_post obeys the same contract as close().
  [[nodiscard]] bool close_batch_with_drain_authority(std::span<const Key> aliases,
                                                      OwnerPost owner_post, DrainedOwner drained);
  /// @brief Retry one failed batch without admitting new ingress.
  [[nodiscard]] bool retry_closed_batch(std::span<const Key> aliases, OwnerPost owner_post,
                                        std::function<void()> drained);
  /// @brief Retry a closed batch with fresh authority for its exact members.
  /// @details The callback runs on the owner's thread. Retained copies of the
  /// authority expire when it returns.
  /// @pre @p owner_post obeys the same contract as close().
  [[nodiscard]] bool retry_closed_batch_with_drain_authority(std::span<const Key> aliases,
                                                             OwnerPost owner_post,
                                                             DrainedOwner drained);
  /// @brief Reopen all generations only after their shared maintenance finished.
  [[nodiscard]] bool reopen_batch(std::span<const Key> aliases);
  /// @brief Reopen after the owner has finished maintenance.
  [[nodiscard]] bool reopen(Key alias);
  /// @brief Whether a generation is closed to new ingress.
  [[nodiscard]] bool closed(Key alias) const;
  /// @brief Whether a retryable retirement owns the exclusive close claim.
  /// @details A live effect instead owns the alias publication writer lock, so
  /// retirement waits for it to finish rather than returning EAGAIN.
  [[nodiscard]] bool close_claimed() const;
  /// @brief Whether the gate-wide claim belongs to this exact process lifetime.
  [[nodiscard]] bool close_claimed_by_process(uint64_t process_incarnation) const;
  /// @brief Forget failed maintenance after its grants have been unpublished.
  /// @pre The owner has retired this exact generation and, for a batch, every
  ///      batch member generation from the live grant index.
  [[nodiscard]] bool discard_failed(Key alias);
  /// @brief Wait for every admitted alias permit of one process lifetime.
  /// @details Closing the process table prevents a late admission from taking
  /// effect, but its gate permit may still briefly exist while it rechecks the
  /// table. This wait observes active counts without allocating gate state.
  void wait_for_process_quiescent(uint64_t process_incarnation) noexcept;
  [[nodiscard]] size_t tracked_generation_count() const {
    std::lock_guard lock(mutex_);
    return entries_.size();
  }

private:
  struct Drain {
    OwnerPost owner_post;
    std::function<void()> continuation;
  };
  struct Batch {
    std::vector<Key> keys;
    std::optional<Drain> drain;
    size_t active_generations = 0;
    bool drain_pending = true;
    bool retryable = false;
    bool final_writeback_done = false;
  };
  struct Entry {
    bool open = true;
    size_t active = 0;
    bool drain_pending = false;
    bool retryable = false;
    bool final_writeback_done = false;
    std::optional<Drain> drain;
    std::shared_ptr<Batch> batch;
  };

  [[nodiscard]] std::function<void()> make_drain_continuation(Key alias, DrainedOwner drained);
  [[nodiscard]] std::optional<amdgpu::MemoryIngressPermit>
  try_acquire_impl(std::span<const Key> aliases, const DrainAuthority *authority,
                   const EffectWindow *window) noexcept;
  [[nodiscard]] bool valid_drained_owner_locked(const DrainAuthority &authority) const;
  [[nodiscard]] bool drained_owner_covers_key_locked(const DrainAuthority &authority,
                                                     Key key) const;
  [[nodiscard]] bool valid_effect_window_locked(const EffectWindow &window) const;
  void release_effect_window(const std::shared_ptr<EffectWindow::State> &state) noexcept;
  void finish_effect_window_if_quiescent_locked() noexcept;
  void set_drain_authority_active(const DrainAuthority &authority, bool active) noexcept;
  [[nodiscard]] std::function<void()> make_batch_drain_continuation(std::shared_ptr<Batch> batch,
                                                                    DrainedOwner drained);
  void release(std::span<const Key> aliases) noexcept;
  void fail_drain(Key alias) noexcept;
  void finish_drain(Key alias) noexcept;
  void fail_batch(const std::shared_ptr<Batch> &batch) noexcept;
  void finish_batch(const std::shared_ptr<Batch> &batch) noexcept;
  mutable std::mutex mutex_;
  std::condition_variable quiescent_;
  std::map<Key, Entry> entries_;
  // Exactly one of these owns the global claim until reopen/discard or effect release.
  std::optional<Key> single_close_claim_;
  std::shared_ptr<Batch> batch_close_claim_;
  std::shared_ptr<EffectWindow::State> effect_window_claim_;
};

} // namespace rocjitsu
