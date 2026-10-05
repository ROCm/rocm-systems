/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Graph Dependency Bounds
 * @{
 * @ingroup HRRTest
 * Tests that the hand-written graph-node handlers (hip_playback.cpp) do not
 * read past a record whose dependency count is larger than the record holds
 *
 *
 * The handlers are linked in and called directly with a record built here, held
 * in a heap buffer of exactly sizeof(record) so that AddressSanitizer reports
 * any read past it. The target is built with AddressSanitizer by default
 * (HRR_HANDLER_TESTS_ASAN): without the bound the 1000-dependency case copies
 * 8000 bytes out of a 128-byte field and the sanitizer aborts the run. No case
 * reaches a HIP call, so no GPU is needed.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"
#include "hrr_payload_bounds.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

// Defined in hip_playback.cpp; not declared in the public header.
hipError_t playback_hipGraphAddMemAllocNode(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipGraphAddMemcpyNodeToSymbol(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipGraphAddMemcpyNodeFromSymbol(PlaybackContext& ctx, const uint8_t* pl);

namespace {

using hrr_test::capture_stderr;
using hrr_test::ExactRecord;

constexpr uint64_t kGraph = 0x1000;

}  // namespace

/**
 * Test Description
 * ----------------
 *   - The capacity helper: 128 bytes of 8-byte nodes hold 16, and a count above
 *     that does not fit, however large.
 */
TEST_CASE("Unit_HRR_GraphDeps_InlineCapacity", "[hrr]") {
  CHECK(hrr::inline_capacity(128, 8) == 16);
  CHECK(hrr::inline_count_fits(0, 128, 8));
  CHECK(hrr::inline_count_fits(16, 128, 8));
  CHECK_FALSE(hrr::inline_count_fits(17, 128, 8));
  CHECK_FALSE(hrr::inline_count_fits(UINT32_MAX, 128, 8));
  CHECK_FALSE(hrr::inline_count_fits(1, 0, 8));
}

/**
 * Test Description
 * ----------------
 *   - hipGraphAddMemAllocNode with a dependency count of 17, 1000 and
 *     UINT32_MAX against a 16-slot field is refused: it returns without a HIP
 *     call, the graph is marked incomplete, and the message names the API and
 *     the event index. Under AddressSanitizer nothing is read past the record.
 */
TEST_CASE("Unit_HRR_GraphDeps_MemAllocNodeRefusesOversizeCount", "[hrr]") {
  for (uint32_t n : {17u, 1000u, UINT32_MAX}) {
    PlaybackContext ctx;
    ctx.record_graph(kGraph, reinterpret_cast<hipGraph_t>(0x1));

    ExactRecord<hrr_args_hipGraphAddMemAllocNode> rec;
    auto* a = rec.get();
    a->graph = kGraph;
    a->pNodeParams_present = 1;
    a->pDependencies_present = 1;
    a->pDependencies_n = n;
    std::memset(a->pDependencies_bytes, 0xAB, sizeof(a->pDependencies_bytes));

    hrr_dispatch_index = 42;
    hipError_t r = hipSuccess;
    const std::string err = capture_stderr(
        [&] { r = playback_hipGraphAddMemAllocNode(ctx, rec.bytes()); });

    INFO("count " << n << " stderr: " << err);
    CHECK(r == hipSuccess);
    CHECK(ctx.graph_is_incomplete(kGraph));
    CHECK(err.find("hipGraphAddMemAllocNode") != std::string::npos);
    CHECK(err.find("event 42") != std::string::npos);
    CHECK(err.find(std::to_string(n)) != std::string::npos);
  }
}

/**
 * Test Description
 * ----------------
 *   - The same refusal in the two symbol-copy handlers, which read their
 *     dependencies through a different helper.
 */
TEST_CASE("Unit_HRR_GraphDeps_SymbolCopyNodesRefuseOversizeCount", "[hrr]") {
  constexpr uint64_t kSym = 0x2000, kDev = 0x3000;
  char symbol_storage[64];
  char dev_storage[64];

  auto make_ctx = [&](PlaybackContext& ctx) {
    ctx.record_graph(kGraph, reinterpret_cast<hipGraph_t>(0x1));
    ctx.record_symbol(kSym, "sym", symbol_storage, sizeof(symbol_storage));
    ctx.record_alloc(kDev, dev_storage, sizeof(dev_storage));
  };

  {  // hipGraphAddMemcpyNodeToSymbol: device source, no blob
    PlaybackContext ctx;
    make_ctx(ctx);
    ExactRecord<hrr_args_hipGraphAddMemcpyNodeToSymbol> rec;
    auto* a = rec.get();
    a->graph = kGraph;
    a->symbol = kSym;
    a->src = kDev;
    a->kind = hipMemcpyDeviceToDevice;
    a->count = 8;
    a->pDependencies_present = 1;
    a->pDependencies_n = 1000;
    std::memset(a->pDependencies_bytes, 0xCD, sizeof(a->pDependencies_bytes));

    hrr_dispatch_index = 7;
    hipError_t r = hipErrorUnknown;
    const std::string err = capture_stderr(
        [&] { r = playback_hipGraphAddMemcpyNodeToSymbol(ctx, rec.bytes()); });
    INFO(err);
    CHECK(r == hipSuccess);
    CHECK(ctx.graph_is_incomplete(kGraph));
    CHECK(err.find("hipGraphAddMemcpyNodeToSymbol") != std::string::npos);
    CHECK(err.find("event 7") != std::string::npos);
  }

  {  // hipGraphAddMemcpyNodeFromSymbol: the destination is device memory
    PlaybackContext ctx;
    make_ctx(ctx);
    ExactRecord<hrr_args_hipGraphAddMemcpyNodeFromSymbol> rec;
    auto* a = rec.get();
    a->graph = kGraph;
    a->symbol = kSym;
    a->dst = kDev;
    a->kind = hipMemcpyDeviceToDevice;
    a->count = 8;
    a->pDependencies_present = 1;
    a->pDependencies_n = 1000;
    std::memset(a->pDependencies_bytes, 0xCD, sizeof(a->pDependencies_bytes));

    hrr_dispatch_index = 9;
    hipError_t r = hipErrorUnknown;
    const std::string err = capture_stderr(
        [&] { r = playback_hipGraphAddMemcpyNodeFromSymbol(ctx, rec.bytes()); });
    INFO(err);
    CHECK(r == hipSuccess);
    CHECK(ctx.graph_is_incomplete(kGraph));
    CHECK(err.find("hipGraphAddMemcpyNodeFromSymbol") != std::string::npos);
    CHECK(err.find("event 9") != std::string::npos);
  }
}

/** @} */
