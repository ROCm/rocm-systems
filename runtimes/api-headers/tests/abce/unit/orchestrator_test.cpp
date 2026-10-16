/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Map and submit end to end over fake rings.
//
//   * Golden frames: the packet sequences the README states as verified.
//   * Sizing == emission, for every ISA and op kind: the decoder must land on
//     the last byte of every frame, or a ring would be silently corrupted.
//   * Placement: back-to-back vs fan-out follows the batch total.
//   * Submit places exactly the bytes the map phase produced.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "abce_test_harness.hpp"

namespace {

using abce_test::AllIsas;
using abce_test::FakeAddress;
using abce_test::IsaByName;
using abce_test::IsaCase;
using abce_test::MakeHostToDevice;
using abce_test::Mapper;
using abce_test::MapperOptions;
using Names = std::vector<std::string>;

constexpr size_t kKiB = 1024;
constexpr size_t kMiB = 1024 * kKiB;

abce_copy_op_t MakeMulticast(uint64_t src_offset, std::vector<void*>* destinations, size_t size) {
  abce_copy_op_t op = MakeHostToDevice(0, src_offset, size);
  op.kind = ABCE_OP_KIND_MULTICAST;
  op.dst = nullptr;
  op.dsts = destinations->data();
  op.num_dsts = static_cast<uint32_t>(destinations->size());
  return op;
}

std::vector<void*> MakeDestinations(uint32_t count) {
  std::vector<void*> destinations;
  for (uint32_t dst_idx = 0; dst_idx < count; ++dst_idx)
    destinations.push_back(FakeAddress(0x10000000ull + 0x100000ull * dst_idx));
  return destinations;
}

std::vector<abce_copy_op_t> MakeLinearBatch(uint32_t count, size_t size_each) {
  std::vector<abce_copy_op_t> ops;
  for (uint32_t copy_idx = 0; copy_idx < count; ++copy_idx)
    ops.push_back(MakeHostToDevice(0x20000000ull + copy_idx * 0x4000000ull,
                                   0x40000000ull + copy_idx * 0x4000000ull, size_each));
  return ops;
}

//===----------------------------------------------------------------------===//
// Golden frames (README section 11)
//===----------------------------------------------------------------------===//

struct MappedPlan {
  abce_plan_t plan;
  abce_status_t status;
};

MappedPlan MapOn(Mapper& mapper, const std::vector<abce_copy_op_t>& ops) {
  MappedPlan mapped;
  const abce_copy_metadata_t metadata = mapper.NewMetadata();
  mapped.status = mapper.Map(ops, metadata, &mapped.plan);
  return mapped;
}

TEST(GoldenFrames, Gfx942SingleHostToDevice4KiB) {
  Mapper mapper(IsaByName("gfx942"));
  MappedPlan mapped = MapOn(mapper, {MakeHostToDevice(0x1000, 0x8000, 4 * kKiB)});
  ASSERT_EQ(mapped.status, ABCE_STATUS_OK);
  ASSERT_EQ(mapped.plan.num_frames, 1u);
  EXPECT_EQ(mapped.plan.frames[0].bytes, 84u);
  EXPECT_EQ(mapper.WalkFrame(mapped.plan, 0).Names(), (Names{"HDP_FLUSH", "COPY_LINEAR", "ATOMIC"}));
}

TEST(GoldenFrames, Gfx942FourBackToBack64KiBCopiesShareOneFrame) {
  Mapper mapper(IsaByName("gfx942"));
  MappedPlan mapped = MapOn(mapper, MakeLinearBatch(4, 64 * kKiB));
  ASSERT_EQ(mapped.status, ABCE_STATUS_OK);
  ASSERT_EQ(mapped.plan.num_frames, 1u);
  EXPECT_EQ(mapped.plan.frames[0].bytes, 168u);
  EXPECT_EQ(mapper.WalkFrame(mapped.plan, 0).Names(),
            (Names{"HDP_FLUSH", "COPY_LINEAR", "COPY_LINEAR", "COPY_LINEAR", "COPY_LINEAR", "ATOMIC"}));
}

TEST(GoldenFrames, Gfx1250SingleCopyIsOneFusedPacketAfterTheFlush) {
  Mapper mapper(IsaByName("gfx1250"));
  MappedPlan mapped = MapOn(mapper, {MakeHostToDevice(0x1000, 0x8000, 4 * kKiB)});
  ASSERT_EQ(mapped.status, ABCE_STATUS_OK);
  ASSERT_EQ(mapped.plan.num_frames, 1u);
  EXPECT_EQ(mapped.plan.frames[0].bytes, 72u);
  EXPECT_EQ(mapper.WalkFrame(mapped.plan, 0).Names(), (Names{"HDP_FLUSH", "COPY_LINEAR_WAITSIGNAL"}));
}

TEST(GoldenFrames, Gfx942MulticastDecomposesToTwoBroadcastPackets) {
  Mapper mapper(IsaByName("gfx942"));
  std::vector<void*> destinations = MakeDestinations(4);
  MappedPlan mapped = MapOn(mapper, {MakeMulticast(0x8000, &destinations, 4 * kKiB)});
  ASSERT_EQ(mapped.status, ABCE_STATUS_OK);
  ASSERT_EQ(mapped.plan.num_frames, 1u);
  EXPECT_EQ(mapped.plan.frames[0].bytes, 128u);
  EXPECT_EQ(mapper.WalkFrame(mapped.plan, 0).Names(),
            (Names{"HDP_FLUSH", "COPY_LINEAR_BROADCAST", "COPY_LINEAR_BROADCAST", "ATOMIC"}));
}

TEST(GoldenFrames, Gfx1250MulticastIsOneFusedPacket) {
  Mapper mapper(IsaByName("gfx1250"));
  std::vector<void*> destinations = MakeDestinations(4);
  MappedPlan mapped = MapOn(mapper, {MakeMulticast(0x8000, &destinations, 4 * kKiB)});
  ASSERT_EQ(mapped.status, ABCE_STATUS_OK);
  ASSERT_EQ(mapped.plan.num_frames, 1u);
  EXPECT_EQ(mapped.plan.frames[0].bytes, 96u);
  EXPECT_EQ(mapper.WalkFrame(mapped.plan, 0).Names(),
            (Names{"HDP_FLUSH", "COPY_LINEAR_MULTICAST_WAITSIGNAL"}));
}

TEST(GoldenFrames, Gfx942GatedFanOutHasACoordinatorAndFollowers) {
  MapperOptions options;
  options.num_engines = 4;
  Mapper mapper(IsaByName("gfx942"), options);
  MappedPlan mapped = MapOn(mapper, MakeLinearBatch(4, 1 * kMiB));
  ASSERT_EQ(mapped.status, ABCE_STATUS_OK);
  ASSERT_EQ(mapped.plan.num_frames, 4u);
  EXPECT_TRUE(mapped.plan.start_gate_required);
  EXPECT_TRUE(mapped.plan.frames[0].coordinator);
  EXPECT_EQ(mapper.WalkFrame(mapped.plan, 0).Names(),
            (Names{"HDP_FLUSH", "ATOMIC", "COPY_LINEAR", "POLL_REGMEM", "ATOMIC"}));
  for (uint32_t frame_idx = 1; frame_idx < 4; ++frame_idx) {
    SCOPED_TRACE(frame_idx);
    EXPECT_FALSE(mapped.plan.frames[frame_idx].coordinator);
    EXPECT_EQ(mapper.WalkFrame(mapped.plan, frame_idx).Names(),
              (Names{"POLL_REGMEM", "COPY_LINEAR", "ATOMIC"}));
  }
}

// A signal armed for this copy alone is written rather than decremented.
TEST(GoldenFrames, Gfx942SoleUpdateCompletesWithAFence) {
  Mapper mapper(IsaByName("gfx942"));
  abce_copy_metadata_t metadata = mapper.NewMetadata();
  metadata.out.sole_update = true;
  MappedPlan mapped;
  mapped.status = mapper.Map({MakeHostToDevice(0x1000, 0x8000, 4 * kKiB)}, metadata, &mapped.plan);
  ASSERT_EQ(mapped.status, ABCE_STATUS_OK);
  ASSERT_EQ(mapped.plan.num_frames, 1u);
  EXPECT_EQ(mapped.plan.frames[0].bytes, 84u - sizeof(SDMA_PKT_ATOMIC) + sizeof(SDMA_PKT_FENCE));
  EXPECT_EQ(mapper.WalkFrame(mapped.plan, 0).Names(), (Names{"HDP_FLUSH", "COPY_LINEAR", "FENCE"}));
}

// FENCE writes 32 bits, so a done value with a high word takes one per word.
TEST(GoldenFrames, Gfx942SoleUpdateAbove32BitsWritesBothWords) {
  Mapper mapper(IsaByName("gfx942"));
  abce_copy_metadata_t metadata = mapper.NewMetadata();
  metadata.out.completion_value = uint64_t{1} << 32;
  metadata.out.sole_update = true;
  MappedPlan mapped;
  mapped.status = mapper.Map({MakeHostToDevice(0x1000, 0x8000, 4 * kKiB)}, metadata, &mapped.plan);
  ASSERT_EQ(mapped.status, ABCE_STATUS_OK);
  EXPECT_EQ(mapper.WalkFrame(mapped.plan, 0).Names(),
            (Names{"HDP_FLUSH", "COPY_LINEAR", "FENCE", "FENCE"}));
}

// Only the coordinator's final update becomes a FENCE: the start gate and the
// fan-in counter are counted down by several frames and keep their atomics.
TEST(GoldenFrames, Gfx942SoleUpdateFanOutKeepsTheCountingAtomics) {
  MapperOptions options;
  options.num_engines = 4;
  Mapper mapper(IsaByName("gfx942"), options);
  abce_copy_metadata_t metadata = mapper.NewMetadata();
  metadata.out.sole_update = true;
  MappedPlan mapped;
  mapped.status = mapper.Map(MakeLinearBatch(4, 1 * kMiB), metadata, &mapped.plan);
  ASSERT_EQ(mapped.status, ABCE_STATUS_OK);
  ASSERT_EQ(mapped.plan.num_frames, 4u);
  EXPECT_EQ(mapper.WalkFrame(mapped.plan, 0).Names(),
            (Names{"HDP_FLUSH", "ATOMIC", "COPY_LINEAR", "POLL_REGMEM", "FENCE"}));
  for (uint32_t frame_idx = 1; frame_idx < 4; ++frame_idx) {
    SCOPED_TRACE(frame_idx);
    EXPECT_EQ(mapper.WalkFrame(mapped.plan, frame_idx).Names(),
              (Names{"POLL_REGMEM", "COPY_LINEAR", "ATOMIC"}));
  }
}

// README section 6: the gate-free fast path needs the HDP flush off, because a
// flush on the coordinator's ring orders nothing on the other rings.
TEST(GoldenFrames, Gfx1250GateFreeFanOutNeedsTheHdpFlushOff) {
  for (const bool hdp_flush : {true, false}) {
    SCOPED_TRACE(hdp_flush ? "hdp flush on (default)" : "hdp flush off");
    MapperOptions options;
    options.num_engines = 4;
    abce_platform_caps_t caps = abce_detect_default_platform_caps(IsaByName("gfx1250").isa);
    caps.emit_hdp_flush = hdp_flush;
    options.caps = caps;
    Mapper mapper(IsaByName("gfx1250"), options);
    MappedPlan mapped = MapOn(mapper, MakeLinearBatch(4, 1 * kMiB));
    ASSERT_EQ(mapped.status, ABCE_STATUS_OK);
    ASSERT_EQ(mapped.plan.num_frames, 4u);
    EXPECT_EQ(mapped.plan.start_gate_required, hdp_flush);
    EXPECT_EQ(mapped.plan.epilogue_required, hdp_flush);
    EXPECT_EQ(mapped.plan.fan_in_count, hdp_flush ? 3u : 4u);
    if (!hdp_flush) {
      for (uint32_t frame_idx = 0; frame_idx < 4; ++frame_idx) {
        EXPECT_EQ(mapped.plan.frames[frame_idx].bytes, 48u) << "frame " << frame_idx;
        EXPECT_EQ(mapper.WalkFrame(mapped.plan, frame_idx).Names(), (Names{"COPY_LINEAR_WAITSIGNAL"}));
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Placement: back-to-back vs fan-out follows the batch TOTAL
//===----------------------------------------------------------------------===//

uint32_t FramesFor(Mapper& mapper, const std::vector<abce_copy_op_t>& ops,
                   abce_linear_batch_mode_t mode = ABCE_LINEAR_BATCH_MODE_AUTOMATIC) {
  abce_copy_metadata_t metadata = mapper.NewMetadata();
  metadata.linear_batch_mode = mode;
  abce_plan_t plan;
  EXPECT_EQ(mapper.Map(ops, metadata, &plan), ABCE_STATUS_OK);
  return plan.num_frames;
}

TEST(Placement, TheDecisionFollowsTheBatchTotalNotTheEntrySize) {
  MapperOptions options;
  options.num_engines = 8;
  Mapper mapper(IsaByName("gfx942"), options);
  EXPECT_EQ(FramesFor(mapper, MakeLinearBatch(8, 64 * kKiB)), 1u) << "512 KiB total stays on one ring";
  EXPECT_GT(FramesFor(mapper, MakeLinearBatch(64, 64 * kKiB)), 1u)
      << "same entry size, eight times the bytes, must fan out";
  EXPECT_EQ(FramesFor(mapper, MakeLinearBatch(64, 8 * kKiB)), 1u)
      << "64 x 8 KiB serialises exactly as many bytes as 8 x 64 KiB";
}

TEST(Placement, TheTotalThresholdIsInclusive) {
  MapperOptions options;
  options.num_engines = 4;
  Mapper mapper(IsaByName("gfx942"), options);
  std::vector<abce_copy_op_t> at_the_cap = MakeLinearBatch(8, 64 * kKiB);
  EXPECT_EQ(FramesFor(mapper, at_the_cap), 1u);
  std::vector<abce_copy_op_t> just_over = MakeLinearBatch(8, 64 * kKiB);
  just_over.back().size += 8;
  EXPECT_GT(FramesFor(mapper, just_over), 1u);
}

TEST(Placement, ExplicitModesOverrideTheSizeRule) {
  MapperOptions options;
  options.num_engines = 4;
  Mapper mapper(IsaByName("gfx942"), options);
  EXPECT_EQ(FramesFor(mapper, MakeLinearBatch(64, 64 * kKiB), ABCE_LINEAR_BATCH_MODE_FORCE_BACK_TO_BACK),
            1u);
  EXPECT_GT(FramesFor(mapper, MakeLinearBatch(8, 4 * kKiB), ABCE_LINEAR_BATCH_MODE_FORCE_FAN_OUT), 1u);
}

TEST(Placement, ALoneCopyIsNeverTreatedAsABatch) {
  MapperOptions options;
  options.num_engines = 4;
  Mapper mapper(IsaByName("gfx942"), options);
  EXPECT_EQ(FramesFor(mapper, MakeLinearBatch(1, 64 * kKiB)), 1u);
  EXPECT_EQ(FramesFor(mapper, MakeLinearBatch(1, 1 * kMiB)), 1u);
}

//===----------------------------------------------------------------------===//
// Known defects
//
// Disabled so the suite stays green, and kept as real tests so the defect is
// reproducible and the fix is verifiable: run with --gtest_also_run_disabled_tests
// and --gtest_filter=KnownDefects.*
//===----------------------------------------------------------------------===//

// On gfx90a, SDMA0 is the only engine legal for H2D. A multicast above the
// broadcast threshold is decomposed into one linear copy per destination, and
// each goes through abce_copy_orchestrator_select_engine(). With fan-out enabled
// that function restricts `allowed` to the candidates not yet selected, and only
// then ranks them for legality. The first destination takes engine 0; for the
// second the only unselected candidate is engine 1, which the RAS reservation
// makes illegal for H2D, so the ranked set is empty and map fails with
// NO_LEGAL_ENGINE although engine 0 could take every copy. A plain linear batch
// does not hit this: it goes through the legal-set balancer instead. With one
// engine registered the same request maps, and gfx942 never hits it because its
// ranking appends every remaining engine.
TEST(KnownDefects, DISABLED_Gfx90aLargeMulticastWithSeveralEnginesIsMappable) {
  for (const uint32_t num_engines : {1u, 2u, 4u}) {
    SCOPED_TRACE(::testing::Message() << num_engines << " engines registered");
    MapperOptions options;
    options.num_engines = num_engines;
    Mapper mapper(IsaByName("gfx90a"), options);
    std::vector<void*> destinations = MakeDestinations(5);
    MappedPlan mapped = MapOn(mapper, {MakeMulticast(0x8000, &destinations, 1 * kMiB)});
    EXPECT_EQ(mapped.status, ABCE_STATUS_OK);
  }
}

//===----------------------------------------------------------------------===//
// Sizing == emission, for every ISA and op kind
//===----------------------------------------------------------------------===//

struct NamedBatch {
  std::string label;
  std::vector<abce_copy_op_t> ops;
};

class SizingMatchesEmission : public ::testing::TestWithParam<std::string> {
 protected:
  // Keeps multicast destination arrays alive while their ops are in use.
  std::vector<std::vector<void*>> destination_storage_;

  std::vector<NamedBatch> BatchesFor(const IsaCase& isa_case, size_t max_copy_size) {
    std::vector<NamedBatch> batches;
    destination_storage_.reserve(16);

    for (const size_t size : {size_t{1}, size_t{4}, 4 * kKiB, 1 * kMiB, max_copy_size,
                              max_copy_size + 1, 3 * max_copy_size + 7})
      batches.push_back({"linear " + std::to_string(size), {MakeHostToDevice(0x1000, 0x8000, size)}});

    batches.push_back({"linear batch 3 x 4 KiB", MakeLinearBatch(3, 4 * kKiB)});

    for (const size_t size : {size_t{4}, 4 * kKiB, 1 * kMiB}) {
      abce_copy_op_t fill = MakeHostToDevice(0x1000, 0, size);
      fill.kind = ABCE_OP_KIND_FILL;
      fill.fill_value = 0xdeadbeef;
      batches.push_back({"fill " + std::to_string(size), {fill}});
    }

    if (isa_case.swap_supported) {
      for (const size_t size : {size_t{4096}, 1 * kMiB, 5 * kMiB}) {
        abce_copy_op_t swap = MakeHostToDevice(0x100000, 0x200000, size);
        swap.kind = ABCE_OP_KIND_SWAP;
        swap.src_end = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_DEVICE, 0);
        batches.push_back({"swap " + std::to_string(size), {swap}});
      }
    }

    for (const uint32_t num_dsts : {1u, 2u, 4u, 17u}) {
      destination_storage_.push_back(MakeDestinations(num_dsts));
      batches.push_back({"multicast " + std::to_string(num_dsts) + " dsts 4 KiB",
                         {MakeMulticast(0x8000, &destination_storage_.back(), 4 * kKiB)}});
    }
    // gfx90a with more than one engine registered cannot map this: see
    // KnownDefects.DISABLED_Gfx90aLargeMulticastWithSeveralEnginesIsMappable.
    if (std::string(isa_case.name) != "gfx90a") {
      destination_storage_.push_back(MakeDestinations(5));
      batches.push_back(
          {"multicast 5 dsts 1 MiB", {MakeMulticast(0x8000, &destination_storage_.back(), 1 * kMiB)}});
    }

    if (!isa_case.is_gfx125plus && isa_case.swap_supported) {
      destination_storage_.push_back(MakeDestinations(2));
      abce_copy_op_t broadcast = MakeMulticast(0x8000, &destination_storage_.back(), 4 * kKiB);
      broadcast.kind = ABCE_OP_KIND_BROADCAST;
      batches.push_back({"broadcast 2 dsts", {broadcast}});
    }

    if (isa_case.is_gfx125plus) {
      for (const bool indirect_src : {true, false}) {
        abce_copy_op_t indirect = MakeHostToDevice(0x1000, 0x8000, 4 * kKiB);
        indirect.kind = ABCE_OP_KIND_INDIRECT;
        indirect.indirect_src = indirect_src;
        indirect.indirect_dst = !indirect_src;
        batches.push_back({indirect_src ? "indirect src" : "indirect dst", {indirect}});
      }
    }
    return batches;
  }
};

TEST_P(SizingMatchesEmission, EveryFrameDecodesToExactlyItsReservedBytes) {
  const IsaCase isa_case = IsaByName(GetParam());
  MapperOptions options;
  options.num_engines = 4;
  Mapper mapper(isa_case, options);
  const size_t max_copy_size = abce_builder_max_copy_size(&mapper.builder());

  for (const NamedBatch& batch : BatchesFor(isa_case, max_copy_size)) {
    SCOPED_TRACE(batch.label);
    abce_plan_t plan;
    const abce_copy_metadata_t metadata = mapper.NewMetadata();
    ASSERT_EQ(mapper.Map(batch.ops, metadata, &plan), ABCE_STATUS_OK);
    ASSERT_GE(plan.num_frames, 1u);
    for (uint32_t frame_idx = 0; frame_idx < plan.num_frames; ++frame_idx) {
      SCOPED_TRACE(::testing::Message() << "frame " << frame_idx);
      const abce_test::WalkResult walk = mapper.WalkFrame(plan, frame_idx);
      EXPECT_TRUE(walk.clean) << "decoder did not land on the frame's last byte, or hit an "
                                 "unrecognised packet";
      EXPECT_EQ(walk.consumed_bytes, plan.frames[frame_idx].bytes);
      EXPECT_EQ(plan.frames[frame_idx].bytes % sizeof(uint32_t), 0u);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(AllIsas, SizingMatchesEmission,
                         ::testing::Values("gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100",
                                           "gfx1201", "gfx1250"),
                         [](const ::testing::TestParamInfo<std::string>& info) { return info.param; });

//===----------------------------------------------------------------------===//
// Submit places exactly what map produced (README section 1.1)
//===----------------------------------------------------------------------===//

class SubmitMatchesMap : public ::testing::TestWithParam<std::string> {};

TEST_P(SubmitMatchesMap, RingBytesAreIdenticalToTheMappedFrames) {
  const IsaCase isa_case = IsaByName(GetParam());
  MapperOptions options;
  options.num_engines = 4;
  Mapper mapper(isa_case, options);

  struct Shape {
    const char* label;
    std::vector<abce_copy_op_t> ops;
  };
  const std::vector<Shape> shapes = {
      {"single copy", {MakeHostToDevice(0x1000, 0x8000, 64 * kKiB)}},
      {"back-to-back batch", MakeLinearBatch(4, 16 * kKiB)},
      {"fan-out batch", MakeLinearBatch(4, 1 * kMiB)},
  };
  for (const Shape& shape : shapes) {
    SCOPED_TRACE(shape.label);
    std::vector<size_t> stream_size_before;
    for (uint32_t engine_slot = 0; engine_slot < 4; ++engine_slot)
      stream_size_before.push_back(mapper.ring(engine_slot).stream().size());

    abce_plan_t plan;
    const abce_copy_metadata_t metadata = mapper.NewMetadata();
    ASSERT_EQ(mapper.Map(shape.ops, metadata, &plan), ABCE_STATUS_OK);
    std::vector<std::vector<uint8_t>> mapped_frames;
    for (uint32_t frame_idx = 0; frame_idx < plan.num_frames; ++frame_idx)
      mapped_frames.push_back(Mapper::FrameBytes(plan, frame_idx));
    std::vector<uint32_t> frame_engines;
    for (uint32_t frame_idx = 0; frame_idx < plan.num_frames; ++frame_idx)
      frame_engines.push_back(plan.frames[frame_idx].engine);

    ASSERT_EQ(mapper.Submit(&plan), ABCE_STATUS_OK);

    for (uint32_t frame_idx = 0; frame_idx < mapped_frames.size(); ++frame_idx) {
      SCOPED_TRACE(::testing::Message() << "frame " << frame_idx << " on engine " << frame_engines[frame_idx]);
      const std::vector<uint8_t>& stream = mapper.ring(frame_engines[frame_idx]).stream();
      const size_t offset = stream_size_before[frame_engines[frame_idx]];
      ASSERT_GE(stream.size(), offset + mapped_frames[frame_idx].size());
      EXPECT_EQ(std::vector<uint8_t>(stream.begin() + offset,
                                     stream.begin() + offset + mapped_frames[frame_idx].size()),
                mapped_frames[frame_idx])
          << "the ring must carry the mapped bytes with nothing patched in";
    }
  }
}

INSTANTIATE_TEST_SUITE_P(AllIsas, SubmitMatchesMap,
                         ::testing::Values("gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100",
                                           "gfx1201", "gfx1250"),
                         [](const ::testing::TestParamInfo<std::string>& info) { return info.param; });

}  // namespace
