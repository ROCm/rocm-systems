// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/major_image_ownership.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace rocjitsu {
namespace {

using major_image_ownership::Measurement;
using major_image_ownership::OwnerKind;
using major_image_ownership::Phase;
using major_image_ownership::ScopedMeasurement;
using major_image_ownership::ScopedOwner;
using major_image_ownership::ScopedPhase;

TEST(MajorImageOwnership, IsInertWithoutMeasurement) {
  const ScopedPhase phase(Phase::IncrementalPatch);
  const ScopedOwner owner(OwnerKind::Parser, 100);
  owner.checkpoint();
}

TEST(MajorImageOwnership, DeduplicatesAliasedInputsButAddsDistinctBuffers) {
  uint8_t first_storage = 0;
  uint8_t second_storage = 0;
  ScopedMeasurement measurement;
  const ScopedPhase phase(Phase::IncrementalPatch);
  const ScopedOwner outer_input(OwnerKind::InputImage, &first_storage, 100);
  const ScopedOwner nested_input(OwnerKind::InputImage, &first_storage, 120);
  const ScopedOwner distinct_input(OwnerKind::InputImage, &second_storage, 80);
  const ScopedOwner parser(OwnerKind::Parser, 5);

  const Measurement observed = measurement.snapshot();
  const auto &incremental = observed.phase(Phase::IncrementalPatch);
  EXPECT_FALSE(observed.overflowed);
  EXPECT_FALSE(observed.bookkeeping_error);
  EXPECT_EQ(incremental.peak_bytes, 205u);
  EXPECT_EQ(incremental.bytes_at_peak[static_cast<size_t>(OwnerKind::InputImage)], 200u);
}

TEST(MajorImageOwnership, NestedMeasurementSharesOwnersAndCleanupState) {
  ScopedMeasurement outer_measurement;
  const ScopedPhase phase(Phase::IncrementalPatch);
  std::optional<ScopedOwner> owner(std::in_place, OwnerKind::Parser, 123);
  {
    ScopedMeasurement inner_measurement;
    EXPECT_EQ(inner_measurement.snapshot().phase(Phase::IncrementalPatch).peak_bytes, 123u);
    owner.reset();
    major_image_ownership::checkpoint();
    EXPECT_FALSE(inner_measurement.snapshot().bookkeeping_error);
  }
  const Measurement observed = outer_measurement.snapshot();
  EXPECT_FALSE(observed.bookkeeping_error);
  EXPECT_EQ(observed.phase(Phase::IncrementalPatch).peak_bytes, 123u);
}

TEST(MajorImageOwnership, SaturatesOverflowWithinOneKind) {
  ScopedMeasurement measurement;
  const ScopedPhase phase(Phase::IncrementalPatch);
  const ScopedOwner huge(OwnerKind::Parser, std::numeric_limits<uint64_t>::max());
  const ScopedOwner one(OwnerKind::Parser, 1);

  const Measurement observed = measurement.snapshot();
  EXPECT_TRUE(observed.overflowed);
  EXPECT_FALSE(observed.bookkeeping_error);
  EXPECT_EQ(observed.phase(Phase::IncrementalPatch).peak_bytes,
            std::numeric_limits<uint64_t>::max());
}

TEST(MajorImageOwnership, SaturatesOverflowAcrossKinds) {
  ScopedMeasurement measurement;
  const ScopedPhase phase(Phase::IncrementalPatch);
  const ScopedOwner huge(OwnerKind::Parser, std::numeric_limits<uint64_t>::max());
  const ScopedOwner one(OwnerKind::PatcherImage, 1);

  const Measurement observed = measurement.snapshot();
  EXPECT_TRUE(observed.overflowed);
  EXPECT_FALSE(observed.bookkeeping_error);
  EXPECT_EQ(observed.phase(Phase::IncrementalPatch).peak_bytes,
            std::numeric_limits<uint64_t>::max());
}

TEST(MajorImageOwnership, ReportsDuplicateRegistrationInReleaseBuilds) {
  ScopedMeasurement measurement;
  const ScopedPhase phase(Phase::IncrementalPatch);
  uint8_t identity = 0;
  EXPECT_TRUE(major_image_ownership::register_owner(&identity, OwnerKind::Parser, 100));
  EXPECT_FALSE(major_image_ownership::register_owner(&identity, OwnerKind::Parser, 5000));

  const Measurement observed = measurement.snapshot();
  EXPECT_TRUE(observed.bookkeeping_error);
  EXPECT_EQ(observed.phase(Phase::IncrementalPatch).peak_bytes, 100u);
  major_image_ownership::unregister_owner(&identity);
}

TEST(MajorImageOwnership, ReportsTransferWithoutRegisteredSource) {
  ScopedMeasurement measurement;
  uint8_t old_identity = 0;
  uint8_t new_identity = 0;
  major_image_ownership::transfer_owner(&old_identity, &new_identity);
  EXPECT_TRUE(measurement.snapshot().bookkeeping_error);
}

TEST(MajorImageOwnership, ExistingPhaseWinsWhenNestedPhaseIsConditional) {
  ScopedMeasurement measurement;
  const ScopedPhase outer(Phase::CompositeIncrementalPatch);
  {
    const ScopedPhase inner(Phase::FinalValidation, /*only_if_none=*/true);
    const ScopedOwner parser(OwnerKind::Parser, 7);
  }

  const Measurement observed = measurement.snapshot();
  EXPECT_FALSE(observed.bookkeeping_error);
  EXPECT_EQ(observed.phase(Phase::CompositeIncrementalPatch).peak_bytes, 7u);
  EXPECT_EQ(observed.phase(Phase::FinalValidation).peak_bytes, 0u);
}

TEST(MajorImageOwnership, CheckpointResamplesDynamicVectorCapacity) {
  ScopedMeasurement measurement;
  const ScopedPhase phase(Phase::IncrementalPatch);
  std::vector<uint8_t> bytes;
  const ScopedOwner owner(OwnerKind::ReplacementBytes, bytes);
  bytes.reserve(256);
  owner.checkpoint();

  const Measurement observed = measurement.snapshot();
  EXPECT_FALSE(observed.bookkeeping_error);
  EXPECT_GE(observed.phase(Phase::IncrementalPatch)
                .bytes_at_peak[static_cast<size_t>(OwnerKind::ReplacementBytes)],
            256u);
}

} // namespace
} // namespace rocjitsu
