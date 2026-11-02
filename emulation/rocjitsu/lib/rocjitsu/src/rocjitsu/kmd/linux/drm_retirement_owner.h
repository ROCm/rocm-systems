// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <memory>
#include <mutex>

namespace rocjitsu {

struct DrmFileState;
class LinuxKfd;

/// @brief Retires local DRM files while their runtime's GPU services are still alive.
/// @details File deleters enqueue ownership; application calls drain with a runtime pin,
/// and the runtime deleter finishes retirement after stopping its producers.
class DrmRetirementOwner {
public:
  /// @brief Retire queues and GEM resources.
  /// @details Called without the owner mutex, before deleting the file. Cleanup
  /// must not throw or retain the file.
  using Cleanup = std::function<void(DrmFileState &, LinuxKfd &)>;

  /// @brief A detached batch owns its runtime pin until cleanup and file deletion end.
  /// @details It also owns its cleanup callback, so it can outlive the retirement owner.
  class Batch {
  public:
    Batch();
    ~Batch();
    Batch(Batch &&other) noexcept;
    Batch &operator=(Batch &&other) noexcept;
    Batch(const Batch &) = delete;
    Batch &operator=(const Batch &) = delete;
    explicit operator bool() const { return files_ != nullptr; }

  private:
    friend class DrmRetirementOwner;
    void retire() noexcept;

    std::shared_ptr<LinuxKfd> backend_;
    std::shared_ptr<const Cleanup> cleanup_;
    std::unique_ptr<DrmFileState> files_;
  };

  explicit DrmRetirementOwner(Cleanup cleanup);
  ~DrmRetirementOwner();
  DrmRetirementOwner(const DrmRetirementOwner &) = delete;
  DrmRetirementOwner &operator=(const DrmRetirementOwner &) = delete;

  /// @brief Bind once, before publishing any files.
  /// @details The weak pointer must share the runtime's control block, as the interposer's
  /// simulated driver alias does.
  void bind_backend(std::weak_ptr<LinuxKfd> backend);
  /// @brief Enqueue a file for retirement.
  /// @details Final release cannot allocate, including during allocation-failure unwind.
  void enqueue(std::unique_ptr<DrmFileState> file) noexcept;

  /// @brief Detach files while retaining their runtime.
  /// @details Leave files queued if the runtime pin cannot be acquired or teardown began.
  [[nodiscard]] Batch take_batch();
  void drain();

  /// @brief Stop application detachment before stopping and joining the engine.
  /// @details Runtime-deleter path only. No application batch can remain when
  /// that deleter runs.
  void begin_teardown();
  /// @brief Finish retirement after engine join and process close release their file owners.
  /// @details The caller owns the runtime's destruction sequence, keeping backend alive.
  void finish_teardown(LinuxKfd &backend);

private:
  enum class Phase { Running, Closing, Closed };

  const std::shared_ptr<const Cleanup> cleanup_;
  std::mutex mutex_;
  Phase phase_ = Phase::Running;
  std::weak_ptr<LinuxKfd> backend_;
  std::unique_ptr<DrmFileState> pending_;
  DrmFileState *pending_tail_ = nullptr;
};

} // namespace rocjitsu
