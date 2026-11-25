// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"
#include "util/result.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::test {

struct ExpectedReplacement {
  uint64_t offset;
  std::vector<uint8_t> expected;
  std::vector<uint8_t> replacement;
};

enum class FactsRoute { Cold, Queried, Fork, WarmFork, WarmQueriedFork, WarmDirect };

inline const char *facts_route_name(FactsRoute route) {
  switch (route) {
  case FactsRoute::Cold:
    return "cold";
  case FactsRoute::Queried:
    return "queried";
  case FactsRoute::Fork:
    return "fork";
  case FactsRoute::WarmFork:
    return "warm fork";
  case FactsRoute::WarmQueriedFork:
    return "warm queried fork";
  case FactsRoute::WarmDirect:
    return "warm direct";
  }
  return "unknown facts route";
}

// Keep the seed and its optional child alive through validation and emission.
class FactsRouteFixture {
public:
  FactsRouteFixture(const AmdGpuCodeObject &source, rocjitsu::IsaFamily arch)
      : seed_(source, arch) {}

  testing::AssertionResult prepare(FactsRoute route, uint64_t warm_anchor = 0) {
    if (route == FactsRoute::WarmFork || route == FactsRoute::WarmQueriedFork ||
        route == FactsRoute::WarmDirect) {
      seed_.add_point_by_offset(warm_anchor);
      const Instrumentor::ValidationResult warmed = seed_.validate_points();
      if (!warmed.errors.empty()) {
        return testing::AssertionFailure() << warmed.errors.front();
      }
      if (warmed.sites.size() != 1) {
        return testing::AssertionFailure()
               << "warm validation produced " << warmed.sites.size() << " sites instead of one";
      }
    }
    if (route == FactsRoute::Queried || route == FactsRoute::WarmQueriedFork ||
        route == FactsRoute::WarmDirect) {
      TextEditError error;
      queried_facts_ = seed_.instruction_boundaries(error);
      if (queried_facts_ == nullptr) {
        return testing::AssertionFailure() << error.message;
      }
    }
    if (route == FactsRoute::Fork || route == FactsRoute::WarmFork ||
        route == FactsRoute::WarmQueriedFork) {
      TextEditError error;
      util::FailureOr<std::unique_ptr<Instrumentor>> fork = seed_.fork(error);
      if (fork.failed()) {
        return testing::AssertionFailure() << error.message;
      }
      child_ = std::move(fork).value();
    }
    return testing::AssertionSuccess();
  }

  Instrumentor &seed() { return seed_; }
  Instrumentor &batch() { return child_ ? *child_ : seed_; }
  const InstructionBoundaries *queried_facts() const { return queried_facts_; }

private:
  Instrumentor seed_;
  std::unique_ptr<Instrumentor> child_;
  const InstructionBoundaries *queried_facts_ = nullptr;
};

inline testing::AssertionResult source_image_matches(const AmdGpuCodeObject &source,
                                                     std::span<const uint8_t> pristine) {
  if (source.image_size() != pristine.size()) {
    return testing::AssertionFailure()
           << "source image size changed from " << pristine.size() << " to " << source.image_size();
  }
  if (!pristine.empty() &&
      std::memcmp(source.image_data(), pristine.data(), pristine.size()) != 0) {
    return testing::AssertionFailure() << "source image bytes changed";
  }
  return testing::AssertionSuccess();
}

// Compute the displacement independently of the production planner/range helper.
inline std::optional<uint32_t> expected_replacement_branch(uint64_t from, uint64_t to,
                                                           rocjitsu::IsaFamily arch) {
  if (from > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - 4 ||
      to > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }
  const int64_t distance = static_cast<int64_t>(to) - static_cast<int64_t>(from) - 4;
  if (distance % 4 != 0 || distance / 4 < std::numeric_limits<int16_t>::min() ||
      distance / 4 > std::numeric_limits<int16_t>::max()) {
    return std::nullopt;
  }
  const util::FailureOr<uint32_t> branch = build_s_branch(0, arch);
  if (branch.failed()) {
    return std::nullopt;
  }
  return branch.value() | static_cast<uint16_t>(distance / 4);
}

inline void expect_replacement_guest(const InstrumentedCodeObjectDebug &result,
                                     uint64_t source_offset, uint32_t original_size,
                                     std::span<const uint8_t> replacement, rocjitsu::IsaFamily arch,
                                     bool has_probe) {
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  ASSERT_EQ(result.sites.size(), 1u);
  ASSERT_EQ(result.patches.size(), 1u);
  const InstrumentedSite &site = result.sites.front();
  EXPECT_EQ(site.kind, InstrumentationKind::ReplaceWithInst);
  EXPECT_EQ(site.original_offset, source_offset);
  EXPECT_EQ(site.original_size, original_size);
  EXPECT_EQ(site.guest_size, replacement.size());
  EXPECT_EQ(site.instrumentation_offset + site.instrumentation_size, site.guest_offset);
  if (has_probe) {
    EXPECT_GT(site.instrumentation_size, 0u);
  } else {
    EXPECT_EQ(site.instrumentation_size, 0u);
  }
  const InstrumentationPatch &patch = result.patches.front();
  EXPECT_EQ(patch.is_probe_call, has_probe);
  EXPECT_EQ(patch.trampoline_offset, site.instrumentation_offset);
  const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(candidate.is_valid());
  ASSERT_EQ(candidate.text_sections().size(), 1u);
  const Section &text = *candidate.text_sections().front();
  ASSERT_LE(source_offset, text.size());
  ASSERT_LE(sizeof(uint32_t), text.size() - source_offset);
  ASSERT_LE(site.guest_offset, text.size());
  ASSERT_LE(replacement.size(), text.size() - site.guest_offset);
  const uint64_t return_offset = site.guest_offset + replacement.size();
  ASSERT_LE(sizeof(uint32_t), text.size() - return_offset);
  if (!replacement.empty()) {
    EXPECT_EQ(std::memcmp(text.data() + site.guest_offset, replacement.data(), replacement.size()),
              0);
  }
  const std::optional<uint32_t> entry_branch =
      expected_replacement_branch(source_offset, site.instrumentation_offset, arch);
  ASSERT_TRUE(entry_branch);
  uint32_t actual = 0;
  std::memcpy(&actual, text.data() + source_offset, sizeof(actual));
  EXPECT_EQ(actual, *entry_branch);
  const std::optional<uint32_t> return_branch =
      expected_replacement_branch(return_offset, source_offset + original_size, arch);
  ASSERT_TRUE(return_branch);
  std::memcpy(&actual, text.data() + return_offset, sizeof(actual));
  EXPECT_EQ(actual, *return_branch);
  ASSERT_TRUE(patch.return_target);
  EXPECT_EQ(*patch.return_target, source_offset + original_size);
}

// This oracle constructs source splices and replacement caves without consuming
// emitted site ranges, plans or candidate bytes. The shared patcher supplies only
// ELF layout/metadata updates; its writer has separate regression coverage.
inline std::optional<std::vector<uint8_t>>
expected_replacement_image(const AmdGpuCodeObject &source,
                           std::span<const ExpectedReplacement> edits, rocjitsu::IsaFamily arch) {
  if (!source.is_valid() || source.text_sections().size() != 1) {
    return std::nullopt;
  }
  const Section &text = *source.text_sections().front();
  const auto *data = reinterpret_cast<const uint8_t *>(text.data());
  std::vector<uint8_t> expected_text(data, data + text.size());
  const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, arch);
  if (maybe_nop.failed()) {
    return std::nullopt;
  }
  const uint32_t nop = maybe_nop.value();
  std::unique_ptr<Decoder> decoder =
      source.target_id() == ROCJITSU_CODE_TARGET_INVALID
          ? Decoder::create(arch)
          : Decoder::create(default_isa_target_registry(), source.target_id());
  if (!decoder) {
    return std::nullopt;
  }
  for (const ExpectedReplacement &edit : edits) {
    if (edit.expected.empty() || edit.expected.size() % 4 != 0 || edit.offset > text.size() ||
        edit.expected.size() > text.size() - edit.offset || edit.replacement.size() % 4 != 0) {
      return std::nullopt;
    }
    const uint64_t trampoline_offset = expected_text.size();
    const std::optional<uint32_t> splice =
        expected_replacement_branch(edit.offset, trampoline_offset, arch);
    if (!splice) {
      return std::nullopt;
    }
    std::memcpy(expected_text.data() + edit.offset, &*splice, sizeof(*splice));
    for (size_t padding = 4; padding < edit.expected.size(); padding += 4) {
      std::memcpy(expected_text.data() + edit.offset + padding, &nop, sizeof(nop));
    }
    expected_text.insert(expected_text.end(), edit.replacement.begin(), edit.replacement.end());
    std::vector<uint32_t> words(edit.replacement.size() / 4);
    if (!words.empty()) {
      std::memcpy(words.data(), edit.replacement.data(), edit.replacement.size());
    }
    size_t cursor = 0;
    bool terminal = false;
    while (cursor < words.size()) {
      const DecodeResult decoded =
          decoder->decode_window(std::span(words).subspan(cursor), trampoline_offset + cursor * 4);
      if (decoded.failed() || decoded.value()->size() == 0 || decoded.value()->size() % 4 != 0 ||
          static_cast<size_t>(decoded.value()->size()) / 4 > words.size() - cursor) {
        return std::nullopt;
      }
      terminal = (decoded.value()->flags() & PROGRAM_TERMINATOR) != 0;
      cursor += static_cast<size_t>(decoded.value()->size()) / 4;
    }
    if (!terminal) {
      const std::optional<uint32_t> continuation = expected_replacement_branch(
          expected_text.size(), edit.offset + edit.expected.size(), arch);
      if (!continuation) {
        return std::nullopt;
      }
      const auto *bytes = reinterpret_cast<const uint8_t *>(&*continuation);
      expected_text.insert(expected_text.end(), bytes, bytes + sizeof(*continuation));
    }
  }
  CodeObjectPatcher expected_patcher(source);
  if (!expected_patcher.replace_text(expected_text)) {
    return std::nullopt;
  }
  return expected_patcher.emit();
}

} // namespace rocjitsu::test
