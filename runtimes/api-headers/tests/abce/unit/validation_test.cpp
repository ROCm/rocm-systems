/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Map-phase request validation: every status the README says map can return,
// and the failed_op attribution that goes with it.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "abce_test_harness.hpp"

namespace {

using abce_test::FakeAddress;
using abce_test::IsaByName;
using abce_test::MakeHostToDevice;
using abce_test::Mapper;
using abce_test::MapperOptions;

struct MapOutcome {
  abce_status_t status;
  uint32_t failed_op;
};

MapOutcome MapOps(const char* isa_name, const std::vector<abce_copy_op_t>& ops,
                  const MapperOptions& options = MapperOptions()) {
  Mapper mapper(IsaByName(isa_name), options);
  const abce_copy_metadata_t metadata = mapper.NewMetadata();
  abce_plan_t plan;
  const abce_status_t status = mapper.Map(ops, metadata, &plan);
  return {status, plan.failed_op};
}

abce_copy_op_t MakeSwap(uint64_t dst_offset, uint64_t src_offset, size_t size) {
  abce_copy_op_t op = MakeHostToDevice(dst_offset, src_offset, size);
  op.kind = ABCE_OP_KIND_SWAP;
  op.src_end = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_DEVICE, 0);
  return op;
}

//===----------------------------------------------------------------------===//
// Batch-level arguments
//===----------------------------------------------------------------------===//

TEST(Validation, EmptyBatchIsInvalidAndNotAttributedToAnOp) {
  Mapper mapper(IsaByName("gfx942"));
  const abce_copy_metadata_t metadata = mapper.NewMetadata();
  abce_plan_t plan;
  EXPECT_EQ(abce_copy_orchestrator_map_copy(mapper.orchestrator(), nullptr, 0, &metadata, &plan),
            ABCE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(plan.failed_op, UINT32_MAX);
  EXPECT_FALSE(plan.valid);
}

TEST(Validation, MissingCompletionSignalIsInvalid) {
  Mapper mapper(IsaByName("gfx942"));
  abce_copy_metadata_t metadata = mapper.NewMetadata();
  metadata.out.value = nullptr;
  abce_plan_t plan;
  EXPECT_EQ(mapper.Map({MakeHostToDevice(0x1000, 0, 4096)}, metadata, &plan),
            ABCE_STATUS_INVALID_ARGUMENT);
  EXPECT_FALSE(plan.valid);
}

TEST(Validation, BatchAboveTheEntryLimitIsInvalid) {
  const std::vector<abce_copy_op_t> ops(ABCE_MAX_BATCH_ENTRIES + 1, MakeHostToDevice(0x1000, 0, 4));
  EXPECT_EQ(MapOps("gfx942", ops).status, ABCE_STATUS_INVALID_ARGUMENT);
}

TEST(Validation, DependencyCountWithoutDependenciesIsInvalid) {
  Mapper mapper(IsaByName("gfx942"));
  abce_copy_metadata_t metadata = mapper.NewMetadata();
  metadata.num_deps = 1;
  metadata.deps = nullptr;
  abce_plan_t plan;
  EXPECT_EQ(mapper.Map({MakeHostToDevice(0x1000, 0, 4096)}, metadata, &plan),
            ABCE_STATUS_INVALID_ARGUMENT);
}

//===----------------------------------------------------------------------===//
// Per-op arguments, attributed to the offending op
//===----------------------------------------------------------------------===//

TEST(Validation, HostToHostIsRejectedAndNamesTheOp) {
  abce_copy_op_t host_to_host = MakeHostToDevice(0x1000, 0, 4096);
  host_to_host.dst_end = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_HOST, ABCE_ANY_DEVICE);
  const MapOutcome outcome =
      MapOps("gfx942", {MakeHostToDevice(0x1000, 0, 4096), host_to_host});
  EXPECT_EQ(outcome.status, ABCE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(outcome.failed_op, 1u);
}

TEST(Validation, ZeroSizeIsRejected) {
  EXPECT_EQ(MapOps("gfx942", {MakeHostToDevice(0x1000, 0, 0)}).status, ABCE_STATUS_INVALID_ARGUMENT);
}

TEST(Validation, NullEndpointIsRejected) {
  abce_copy_op_t op = MakeHostToDevice(0x1000, 0, 4096);
  op.src = nullptr;
  EXPECT_EQ(MapOps("gfx942", {op}).status, ABCE_STATUS_INVALID_ARGUMENT);
}

TEST(Validation, FillSizeMustBeAMultipleOfADword) {
  abce_copy_op_t fill = MakeHostToDevice(0x1000, 0, 4097);
  fill.kind = ABCE_OP_KIND_FILL;
  EXPECT_EQ(MapOps("gfx942", {fill}).status, ABCE_STATUS_INVALID_ARGUMENT);
  fill.size = 4096;
  EXPECT_EQ(MapOps("gfx942", {fill}).status, ABCE_STATUS_OK);
}

TEST(Validation, BroadcastNeedsExactlyTwoDestinations) {
  void* two[2] = {FakeAddress(0x1000), FakeAddress(0x2000)};
  void* three[3] = {FakeAddress(0x1000), FakeAddress(0x2000), FakeAddress(0x3000)};
  abce_copy_op_t broadcast = MakeHostToDevice(0, 0x8000, 4096);
  broadcast.kind = ABCE_OP_KIND_BROADCAST;
  broadcast.dst = nullptr;
  broadcast.dsts = two;
  broadcast.num_dsts = 2;
  EXPECT_EQ(MapOps("gfx942", {broadcast}).status, ABCE_STATUS_OK);
  broadcast.dsts = three;
  broadcast.num_dsts = 3;
  EXPECT_EQ(MapOps("gfx942", {broadcast}).status, ABCE_STATUS_INVALID_ARGUMENT);
}

TEST(Validation, MulticastRejectsANullDestinationEntry) {
  void* destinations[2] = {FakeAddress(0x1000), nullptr};
  abce_copy_op_t multicast = MakeHostToDevice(0, 0x8000, 4096);
  multicast.kind = ABCE_OP_KIND_MULTICAST;
  multicast.dst = nullptr;
  multicast.dsts = destinations;
  multicast.num_dsts = 2;
  EXPECT_EQ(MapOps("gfx1250", {multicast}).status, ABCE_STATUS_INVALID_ARGUMENT);
}

TEST(Validation, RectWithoutGeometryIsRejected) {
  abce_copy_op_t rect = MakeHostToDevice(0x1000, 0, 4096);
  rect.kind = ABCE_OP_KIND_COPY_RECT;
  rect.rect = nullptr;
  EXPECT_EQ(MapOps("gfx942", {rect}).status, ABCE_STATUS_INVALID_ARGUMENT);
}

// The packet-count limit is UINT32_MAX, so at a ~1 GiB packet it takes a 2^62
// byte copy to reach it. See TEST_PLAN.md, "Findings": a request just below that
// still maps, and builds a frame of hundreds of MiB.
TEST(Validation, AnOpNeedingMoreThanFourBillionPacketsIsOutOfRange) {
  const abce_copy_op_t huge = MakeHostToDevice(0x1000, 0, size_t{1} << 62);
  const MapOutcome outcome = MapOps("gfx942", {huge});
  EXPECT_EQ(outcome.status, ABCE_STATUS_OUT_OF_RANGE);
  EXPECT_EQ(outcome.failed_op, 0u);
}

// A frame at or above the ring size can never fit, whatever the engine frees, so
// submit refuses it outright instead of waiting forever. Map has no ring size to
// check against, so it succeeds.
TEST(Validation, AFrameLargerThanTheRingMapsButFailsAtSubmit) {
  MapperOptions options;
  options.ring_options.size_bytes = 1u << 16;
  Mapper mapper(IsaByName("gfx942"), options);
  const abce_copy_metadata_t metadata = mapper.NewMetadata();
  abce_plan_t plan;
  // ~3000 packets of ~1 GiB, 28 bytes each: a frame of roughly 86 KB.
  const size_t size_needing_more_than_a_ring_of_packets = size_t{3} << 40;
  ASSERT_EQ(mapper.Map({MakeHostToDevice(0x1000, 0, size_needing_more_than_a_ring_of_packets)},
                       metadata, &plan),
            ABCE_STATUS_OK);
  ASSERT_GE(plan.frames[0].bytes, options.ring_options.size_bytes);
  EXPECT_EQ(mapper.Submit(&plan), ABCE_STATUS_OUT_OF_RANGE);
  EXPECT_FALSE(plan.submitted);
  EXPECT_EQ(mapper.ring(0).write_index(), 0u) << "a refused submit must not move the ring";
}

//===----------------------------------------------------------------------===//
// Hardware capability gates
//===----------------------------------------------------------------------===//

TEST(Validation, IndirectCopyNeedsGfx125Plus) {
  abce_copy_op_t indirect = MakeHostToDevice(0x1000, 0x8000, 4096);
  indirect.kind = ABCE_OP_KIND_INDIRECT;
  indirect.indirect_src = true;
  for (const abce_test::IsaCase& isa_case : abce_test::AllIsas()) {
    SCOPED_TRACE(isa_case.name);
    EXPECT_EQ(MapOps(isa_case.name, {indirect}).status,
              isa_case.is_gfx125plus ? ABCE_STATUS_OK : ABCE_STATUS_UNIMPLEMENTED);
  }
}

// ROCr db3394dbb2d: swap addresses must be aligned, 64 bytes on gfx94x/95x and
// 32 on gfx1250, and swap is rejected on parts without the packet. The expected
// values here come from the ISA table in the harness, not from the builder
// helpers the validation uses, so the two can disagree.
TEST(Validation, SwapIsGatedOnTheHardwareAndOnAlignment) {
  for (const abce_test::IsaCase& isa_case : abce_test::AllIsas()) {
    SCOPED_TRACE(isa_case.name);
    const uint64_t required_alignment = isa_case.is_gfx125plus ? 32 : 64;
    if (!isa_case.swap_supported) {
      EXPECT_EQ(MapOps(isa_case.name, {MakeSwap(0x10000, 0x20000, 4096)}).status,
                ABCE_STATUS_UNIMPLEMENTED);
      continue;
    }
    EXPECT_EQ(MapOps(isa_case.name, {MakeSwap(0x10000, 0x20000, 4096)}).status, ABCE_STATUS_OK);
    EXPECT_EQ(MapOps(isa_case.name, {MakeSwap(0x10000 + required_alignment,
                                               0x20000 + required_alignment, 4096)})
                  .status,
              ABCE_STATUS_OK);
    EXPECT_EQ(MapOps(isa_case.name, {MakeSwap(0x10000 + 4, 0x20000, 4096)}).status,
              ABCE_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(MapOps(isa_case.name, {MakeSwap(0x10000, 0x20000 + 4, 4096)}).status,
              ABCE_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(MapOps(isa_case.name, {MakeSwap(0x10000 + required_alignment / 2, 0x20000, 4096)}).status,
              ABCE_STATUS_INVALID_ARGUMENT)
        << "aligned to half the requirement is still misaligned";
    // Linear copies have no such constraint.
    EXPECT_EQ(MapOps(isa_case.name, {MakeHostToDevice(0x10004, 0x20004, 4096)}).status,
              ABCE_STATUS_OK);
  }
}

//===----------------------------------------------------------------------===//
// Engine availability and fan-out requirements
//===----------------------------------------------------------------------===//

TEST(Validation, EngineMaskExcludingEveryRegisteredEngineIsNoEngine) {
  Mapper mapper(IsaByName("gfx942"));
  abce_copy_metadata_t metadata = mapper.NewMetadata();
  metadata.engine_mask = 1ull << 5;  // only engine 0 is registered
  abce_plan_t plan;
  EXPECT_EQ(mapper.Map({MakeHostToDevice(0x1000, 0, 4096)}, metadata, &plan), ABCE_STATUS_NO_ENGINE);
}

TEST(Validation, FanOutNeedsACoordinationWord) {
  MapperOptions options;
  options.num_engines = 4;
  Mapper mapper(IsaByName("gfx942"), options);
  const abce_copy_metadata_t without_scratch = mapper.NewMetadata(/*with_coordination_scratch=*/false);
  std::vector<abce_copy_op_t> ops;
  for (uint64_t copy_idx = 0; copy_idx < 4; ++copy_idx)
    ops.push_back(MakeHostToDevice(0x100000 * (copy_idx + 1), 0x80000000 + 0x100000 * copy_idx, 1u << 20));

  abce_plan_t plan;
  EXPECT_EQ(mapper.Map(ops, without_scratch, &plan), ABCE_STATUS_MISSING_COORDINATION_SCRATCH);
  EXPECT_FALSE(plan.valid);

  const abce_copy_metadata_t with_scratch = mapper.NewMetadata();
  EXPECT_EQ(mapper.Map(ops, with_scratch, &plan), ABCE_STATUS_OK);
  EXPECT_TRUE(plan.valid);
  EXPECT_GT(plan.num_frames, 1u) << "the batch must actually fan out for this test to mean anything";
}

//===----------------------------------------------------------------------===//
// Plan lifecycle
//===----------------------------------------------------------------------===//

TEST(Validation, ASuccessfulPlanSubmitsOnceAndOnlyOnce) {
  Mapper mapper(IsaByName("gfx942"));
  const abce_copy_metadata_t metadata = mapper.NewMetadata();
  abce_plan_t plan;
  ASSERT_EQ(mapper.Map({MakeHostToDevice(0x1000, 0, 4096)}, metadata, &plan), ABCE_STATUS_OK);
  EXPECT_TRUE(plan.valid);
  EXPECT_FALSE(plan.submitted);
  EXPECT_EQ(mapper.Submit(&plan), ABCE_STATUS_OK);
  EXPECT_TRUE(plan.submitted);
  EXPECT_EQ(mapper.Submit(&plan), ABCE_STATUS_INVALID_PLAN);
}

TEST(Validation, APlanThatNeverMappedCannotBeSubmitted) {
  Mapper mapper(IsaByName("gfx942"));
  abce_plan_t plan;
  abce_plan_initialize(&plan);
  EXPECT_EQ(mapper.Submit(&plan), ABCE_STATUS_INVALID_PLAN);
}

TEST(Validation, AFailedMapLeavesTheRingsUntouched) {
  Mapper mapper(IsaByName("gfx942"));
  const abce_copy_metadata_t metadata = mapper.NewMetadata();
  abce_plan_t plan;
  ASSERT_NE(mapper.Map({MakeHostToDevice(0x1000, 0, 0)}, metadata, &plan), ABCE_STATUS_OK);
  EXPECT_EQ(mapper.Submit(&plan), ABCE_STATUS_INVALID_PLAN);
  EXPECT_EQ(mapper.ring(0).write_index(), 0u);
  EXPECT_TRUE(mapper.ring(0).doorbell_history().empty());
}

}  // namespace
