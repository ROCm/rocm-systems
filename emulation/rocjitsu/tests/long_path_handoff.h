// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace rocjitsu::test {

/// @brief Resolved host KFD gpu_id every oversized runtime handoff in the test suite carries.
///
/// @details The runtime config handoff has two consumers with two independent readers: the
/// KFD interposer's raw read loop and the HSA hook's std::ifstream reader. A handoff longer
/// than 4095 bytes must resolve to the same host GPU through both, so the tests that pin that
/// pairing -- GuestKfdConfigTest.ReadsRuntimeHandoffLargerThan4095Bytes and
/// ConfigLoaderTest.ReadsRuntimeHandoffLargerThan4095Bytes -- read the id from here instead of
/// each spelling its own. A value only one of them agreed to would let the two readers drift
/// apart without any test noticing.
inline constexpr uint32_t kOversizedHandoffHostGpuId = 50148;

/// @brief Verdict from building the oversized config handoff: ok, skip, or fail.
/// @details Three states rather than testing::AssertionResult's two, because not every
/// way of not producing the handoff is a defect. Some preconditions depend on the
/// caller's runtime directory rather than on the handoff reader under test: a directory
/// already longer than the path being built says nothing about whether the reader copes
/// with a handoff past 4095 bytes, so it must skip rather than go red. Every state carries
/// a reason, so a skip is still reported and never silently swallowed. GTEST_SKIP expands
/// to a bare `return` and cannot be issued from a function that returns a value, so the
/// verdict travels back to the TEST body, which issues the skip or the failure itself.
class LongPathHandoff {
public:
  enum class Status { kOk, kSkip, kFail };

  /// @brief The oversized handoff was built and installed.
  static LongPathHandoff ok() { return LongPathHandoff(Status::kOk, {}); }
  /// @brief This environment cannot host the path, so the reader is untestable here.
  static LongPathHandoff skip(std::string reason) {
    return LongPathHandoff(Status::kSkip, std::move(reason));
  }
  /// @brief A genuine defect in the helper's arithmetic or in the run itself.
  static LongPathHandoff fail(std::string reason) {
    return LongPathHandoff(Status::kFail, std::move(reason));
  }

  [[nodiscard]] Status status() const { return status_; }
  [[nodiscard]] const std::string &reason() const { return reason_; }

private:
  LongPathHandoff(Status status, std::string reason)
      : status_(status), reason_(std::move(reason)) {}

  Status status_;
  std::string reason_;
};

inline constexpr std::string_view kLongConfigFileName = "config.json";
inline constexpr size_t kMaxComponentLength = 200; // stays well under NAME_MAX
inline constexpr size_t kTargetPathLength = 4093;
inline constexpr size_t kTargetParentLength = kTargetPathLength - kLongConfigFileName.size() - 1;

/// @brief Smallest handoff the built payload must reach to be past a single 4096-byte read.
inline constexpr size_t kMinHandoffPayloadSize = 4096;

/// @brief The handoff directory as a path to build beneath, with any trailing separator
/// dropped: the padding arithmetic charges one separator per appended component, so a
/// directory that already ends in one would be charged twice. A bare root keeps its
/// separator, since dropping it would leave a relative path.
inline std::string normalized_handoff_parent(const std::filesystem::path &dir) {
  std::string parent = dir.string();
  while (parent.size() > 1 && parent.back() == '/')
    parent.pop_back();
  return parent;
}

/// @brief Bytes a normalized parent contributes to a path built beneath it. A bare root is the
/// separator the first component would otherwise have to pay for, so it contributes none.
inline size_t handoff_parent_length(const std::string &parent) {
  return parent == "/" ? 0 : parent.size();
}

/// @brief kSkip when @p dir is too deep to host the oversized handoff.
/// @details Split out of install_oversized_handoff() so a test can consult it before it
/// installs anything. A directory too deep for the >4095-byte path is also too deep for the
/// ordinary config an install writes first, so a test that only checked at install time would
/// already have gone red for the very same environmental reason, and the skip would be
/// unreachable.
inline LongPathHandoff long_path_handoff_supported(const std::filesystem::path &dir) {
  const size_t parent_length = handoff_parent_length(normalized_handoff_parent(dir));

  if (parent_length > kTargetParentLength) {
    std::ostringstream reason;
    reason << "handoff directory " << dir << " contributes " << parent_length
           << " bytes, already past the " << kTargetParentLength << "-byte parent to build";
    return LongPathHandoff::skip(reason.str());
  }
  if (kTargetParentLength - parent_length == 1) {
    std::ostringstream reason;
    reason << "handoff directory " << dir << " contributes " << parent_length
           << " bytes, one short of " << kTargetParentLength
           << ": no component fits in a separator plus one byte";
    return LongPathHandoff::skip(reason.str());
  }
  return LongPathHandoff::ok();
}

/// @brief Write @p config_json at a path just under PATH_MAX and point @p dir's handoff at it.
///
/// @details The handoff the reader parses -- the config path, a newline, and
/// kOversizedHandoffHostGpuId -- then exceeds 4095 bytes and cannot be consumed by a single
/// fixed 4096-byte read. Shared by both consumers' tests so the bytes each one is handed are
/// built by the same arithmetic; a second copy of this could drift and quietly stop being
/// oversized on one side.
///
/// The padding is planned up front rather than grown greedily. Each appended component costs
/// one separator plus at least one character, so a greedy loop can land on a remainder it
/// cannot spend: exactly on the target (where subtracting the separator underflows size_t and
/// asks std::string for a SIZE_MAX-long component) or one byte short of it (where no legal
/// component fits). Sizing the components from the byte budget keeps every intermediate length
/// in range. The remaining unspendable cases -- a starting directory at or past the budget, or
/// one byte short of it -- are the environmental ones long_path_handoff_supported() turns into
/// kSkip. The self-checks on the built path length and on the handoff size, and every write
/// failure, stay kFail, because those are defects in the helper or the run rather than in where
/// it runs.
inline LongPathHandoff install_oversized_handoff(const std::filesystem::path &dir,
                                                 std::string_view config_json) {
  // Re-checked here rather than trusted from the caller, so the helper stays correct
  // standalone and the budget arithmetic below cannot underflow.
  if (LongPathHandoff supported = long_path_handoff_supported(dir);
      supported.status() != LongPathHandoff::Status::kOk)
    return supported;

  try {
    const std::string parent = normalized_handoff_parent(dir);
    const size_t parent_length = handoff_parent_length(parent);
    const size_t pad_budget = kTargetParentLength - parent_length;

    std::filesystem::path config_dir(parent);
    if (pad_budget > 0) {
      // Fewest components that keep each within kMaxComponentLength, with the
      // characters left after the separators spread as evenly as possible.
      const size_t components = (pad_budget + kMaxComponentLength) / (kMaxComponentLength + 1);
      const size_t chars = pad_budget - components;
      for (size_t i = 0; i < components; ++i)
        config_dir /= std::string(chars / components + (i < chars % components ? 1 : 0), 'a');
    }

    // Self-check on the sizing arithmetic above, deliberately before anything touches
    // the filesystem: a miscomputed length must surface here as kFail, not as the
    // ENAMETOOLONG the catch below would forgive as an environment limit.
    const std::filesystem::path config_path = config_dir / kLongConfigFileName;
    if (config_path.string().size() != kTargetPathLength) {
      std::ostringstream reason;
      reason << "built a " << config_path.string().size() << "-byte config path from a "
             << parent_length << "-byte handoff directory, expected " << kTargetPathLength;
      return LongPathHandoff::fail(reason.str());
    }

    std::filesystem::create_directories(config_dir);
    std::ofstream config(config_path);
    config << config_json;
    config.close();
    if (!config)
      return LongPathHandoff::fail("failed to write config to the long path");

    std::ostringstream payload;
    payload << config_path.string() << '\n' << kOversizedHandoffHostGpuId << '\n';
    // The point of the whole exercise, so it is checked rather than assumed: a handoff that
    // fits in one 4096-byte read would still pass both tests while proving nothing.
    if (payload.str().size() < kMinHandoffPayloadSize) {
      std::ostringstream reason;
      reason << "built a " << payload.str().size() << "-byte handoff, expected at least "
             << kMinHandoffPayloadSize;
      return LongPathHandoff::fail(reason.str());
    }

    std::filesystem::create_directories(dir);
    std::ofstream handoff(dir / "config_path", std::ios::trunc);
    handoff << payload.str();
    if (!handoff.good())
      return LongPathHandoff::fail("failed to rewrite the config_path handoff");
    return LongPathHandoff::ok();
  } catch (const std::filesystem::filesystem_error &error) {
    // The built path is already proven to be the intended length, so ENAMETOOLONG
    // here means this filesystem refuses a path the kernel itself allows -- a shorter
    // internal limit, or name-expanding encryption -- which is a property of where the
    // test runs. Every other code (EACCES, ENOSPC) is a real failure.
    std::ostringstream reason;
    reason << "filesystem error building the " << kTargetPathLength
           << "-byte path: " << error.what();
    if (error.code() == std::errc::filename_too_long)
      return LongPathHandoff::skip(reason.str());
    return LongPathHandoff::fail(reason.str());
  }
}

} // namespace rocjitsu::test
