/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Graph Node Kind
 * @{
 * @ingroup HRRTest
 * Replay skips a kind-specific graph call when the recorded node is an empty
 * node. Creating the graph needs a HIP runtime; the skipped call does not
 * launch work.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <cstdint>
#include <string>

hipError_t playback_hipGraphKernelNodeCopyAttributes(PlaybackContext& ctx,
                                                     const uint8_t* payload);
hipError_t playback_hipGraphMemsetNodeSetParams(PlaybackContext& ctx,
                                                const uint8_t* payload);

/**
 * Test Description
 * ----------------
 *   - An empty node is recorded under one handle. CopyAttributes and
 *     MemsetNodeSetParams both skip it and say why, instead of downcasting it.
 */
TEST_CASE("Unit_HRR_GraphNodeKind_EmptyNodeIsSkipped", "[hrr]") {
  hipGraph_t graph{};
  REQUIRE(hipGraphCreate(&graph, 0) == hipSuccess);
  hipGraphNode_t empty{};
  REQUIRE(hipGraphAddEmptyNode(&empty, graph, nullptr, 0) == hipSuccess);

  PlaybackContext ctx;
  constexpr uint64_t rec = 0x1000;
  ctx.record_graph_node(rec, empty);

  hrr_test::ExactRecord<hrr_args_hipGraphKernelNodeCopyAttributes> copy;
  copy.get()->hSrc = rec;
  copy.get()->hDst = rec;
  hipError_t r = hipErrorUnknown;
  std::string err = hrr_test::capture_stderr(
      [&] { r = playback_hipGraphKernelNodeCopyAttributes(ctx, copy.bytes()); });
  CHECK(r == hipSuccess);
  CHECK(err.find("not a kernel node") != std::string::npos);

  hrr_test::ExactRecord<hrr_args_hipGraphMemsetNodeSetParams> memset_rec;
  memset_rec.get()->node = rec;
  memset_rec.get()->pNodeParams_present = 1;
  r = hipErrorUnknown;
  err = hrr_test::capture_stderr(
      [&] { r = playback_hipGraphMemsetNodeSetParams(ctx, memset_rec.bytes()); });
  CHECK(r == hipSuccess);
  CHECK(err.find("not a memset node") != std::string::npos);

  REQUIRE(hipGraphDestroy(graph) == hipSuccess);
}

/**
 * End doxygen group HRRTest.
 * @}
 */
