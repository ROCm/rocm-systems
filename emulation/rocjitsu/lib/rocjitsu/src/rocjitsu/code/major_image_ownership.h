// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace rocjitsu::major_image_ownership {

/// Coarse owner roles used by ConSan's image-proportional admission model.
enum class OwnerKind : uint8_t {
  InputImage,
  Parser,
  ResultImage,
  PatcherImage,
  ReplacementBytes,
  TransactionImage,
  CompositeImage,
  CompactIndex,
  DescriptorProbe,
  Count,
};

/// Transform phases whose live owners are bounded independently.
enum class Phase : uint8_t {
  None,
  IncrementalPatch,
  CompositeIncrementalPatch,
  FinalValidation,
  Count,
};

inline constexpr size_t kOwnerKindCount = static_cast<size_t>(OwnerKind::Count);
inline constexpr size_t kMeasuredPhaseCount = static_cast<size_t>(Phase::Count) - 1u;

struct PhaseMeasurement {
  uint64_t peak_bytes = 0;
  std::array<uint64_t, kOwnerKindCount> bytes_at_peak{};
  uint64_t observed_owner_mask = 0;
};

struct Measurement {
  std::array<PhaseMeasurement, kMeasuredPhaseCount> phases{};
  uint64_t overall_peak_bytes = 0;
  bool overflowed = false;
  bool bookkeeping_error = false;

  [[nodiscard]] const PhaseMeasurement &phase(Phase value) const {
    assert(value > Phase::None && value < Phase::Count);
    return phases[static_cast<size_t>(value) - 1u];
  }
};

using MeasureBytes = uint64_t (*)(const void *) noexcept;

/// Enable ownership observation for the current thread.
///
/// Production execution has only an inactive thread-local pointer check at
/// ownership boundaries. The allocation-backed observer exists only while a
/// test or debug diagnostic explicitly creates this scope. Nested scopes join
/// the active thread measurement so they see the same live owners and cannot
/// leave registrations in a shadow state.
class ScopedMeasurement {
public:
  ScopedMeasurement();
  ScopedMeasurement(const ScopedMeasurement &) = delete;
  ScopedMeasurement &operator=(const ScopedMeasurement &) = delete;
  ~ScopedMeasurement();

  [[nodiscard]] Measurement snapshot() const;

private:
  void *state_ = nullptr;
  bool owns_state_ = false;
};

class ScopedPhase {
public:
  explicit ScopedPhase(Phase phase, bool only_if_none = false);
  ScopedPhase(const ScopedPhase &) = delete;
  ScopedPhase &operator=(const ScopedPhase &) = delete;
  ~ScopedPhase();

private:
  Phase previous_ = Phase::None;
  bool changed_ = false;
};

/// Register one fixed-size or dynamically sized major-image owner.
class ScopedOwner {
public:
  ScopedOwner(OwnerKind kind, uint64_t bytes) noexcept;
  ScopedOwner(OwnerKind kind, const void *shared_storage_identity, uint64_t bytes) noexcept;
  ScopedOwner(OwnerKind kind, const void *context, MeasureBytes measure) noexcept;

  template <typename T, typename Allocator>
  ScopedOwner(OwnerKind kind, const std::vector<T, Allocator> &storage) noexcept
      : ScopedOwner(kind, &storage, &measure_vector<T, Allocator>) {}

  ScopedOwner(const ScopedOwner &) = delete;
  ScopedOwner &operator=(const ScopedOwner &) = delete;
  ~ScopedOwner();

  /// Re-sample dynamic owners after an allocation without another owner event.
  void checkpoint() const noexcept;

private:
  template <typename T, typename Allocator>
  [[nodiscard]] static uint64_t measure_vector(const void *context) noexcept {
    const auto &storage = *static_cast<const std::vector<T, Allocator> *>(context);
    if (storage.capacity() > std::numeric_limits<uint64_t>::max() / sizeof(T))
      return std::numeric_limits<uint64_t>::max();
    return static_cast<uint64_t>(storage.capacity()) * sizeof(T);
  }

  bool registered_ = false;
};

/// Register an owner whose lifetime is managed by another object.
bool register_owner(const void *identity, OwnerKind kind, uint64_t bytes) noexcept;
bool register_owner(const void *identity, OwnerKind kind, const void *shared_storage_identity,
                    uint64_t bytes) noexcept;
bool register_owner(const void *identity, OwnerKind kind, const void *context,
                    MeasureBytes measure) noexcept;
void add_owner_bytes(const void *identity, uint64_t bytes) noexcept;
void transfer_owner(const void *old_identity, const void *new_identity) noexcept;
void transfer_owner(const void *old_identity, const void *new_identity,
                    const void *new_context) noexcept;
void unregister_owner(const void *identity) noexcept;
void checkpoint() noexcept;

} // namespace rocjitsu::major_image_ownership
