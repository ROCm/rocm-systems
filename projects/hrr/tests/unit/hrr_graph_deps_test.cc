/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Graph Dependency Bounds
 * @{
 * @ingroup HRRTest
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

TEST_CASE("Unit_HRR_GraphDeps_InlineCapacity", "[hrr]") {
  CHECK(hrr::inline_capacity(128, 8) == 16);
  CHECK(hrr::inline_count_fits(0, 128, 8));
  CHECK(hrr::inline_count_fits(16, 128, 8));
  CHECK_FALSE(hrr::inline_count_fits(17, 128, 8));
  CHECK_FALSE(hrr::inline_count_fits(UINT32_MAX, 128, 8));
  CHECK_FALSE(hrr::inline_count_fits(1, 0, 8));
}

TEST_CASE("Unit_HRR_GraphDeps_TruncatedDependencyListIsRefused", "[hrr]") {
  CHECK(hrr::deps_complete(0, 0));
  CHECK(hrr::deps_complete(16, 16));
  CHECK_FALSE(hrr::deps_complete(20, 16));

  PlaybackContext ctx;
  ctx.record_graph(kGraph, reinterpret_cast<hipGraph_t>(0x1));

  ExactRecord<hrr_args_hipGraphAddMemAllocNode> rec;
  auto* a = rec.get();
  a->graph = kGraph;
  a->pNodeParams_present = 1;
  a->pDependencies_present = 1;
  a->pDependencies_n = 16;
  a->numDependencies = 20;
  std::memset(a->pDependencies_bytes, 0, sizeof(a->pDependencies_bytes));

  hrr_dispatch_seq = 7;
  hipError_t r = hipErrorUnknown;
  const std::string err = capture_stderr(
      [&] { r = playback_hipGraphAddMemAllocNode(ctx, rec.bytes()); });

  INFO("stderr: " << err);
  CHECK(r == hipSuccess);
  CHECK(ctx.graph_is_incomplete(kGraph));
  CHECK(err.find("hipGraphAddMemAllocNode") != std::string::npos);
  CHECK(err.find("20 dependencies") != std::string::npos);
  CHECK(err.find("holds 16") != std::string::npos);
}

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

    hrr_dispatch_seq = 42;
    hipError_t r = hipSuccess;
    const std::string err = capture_stderr(
        [&] { r = playback_hipGraphAddMemAllocNode(ctx, rec.bytes()); });

    INFO("count " << n << " stderr: " << err);
    CHECK(r == hipSuccess);
    CHECK(ctx.graph_is_incomplete(kGraph));
    CHECK(err.find("hipGraphAddMemAllocNode") != std::string::npos);
    CHECK(err.find("sequence number 42") != std::string::npos);
    CHECK(err.find(std::to_string(n)) != std::string::npos);
  }
}

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

    hrr_dispatch_seq = 7;
    hipError_t r = hipErrorUnknown;
    const std::string err = capture_stderr(
        [&] { r = playback_hipGraphAddMemcpyNodeToSymbol(ctx, rec.bytes()); });
    INFO(err);
    CHECK(r == hipSuccess);
    CHECK(ctx.graph_is_incomplete(kGraph));
    CHECK(err.find("hipGraphAddMemcpyNodeToSymbol") != std::string::npos);
    CHECK(err.find("sequence number 7") != std::string::npos);
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

    hrr_dispatch_seq = 9;
    hipError_t r = hipErrorUnknown;
    const std::string err = capture_stderr(
        [&] { r = playback_hipGraphAddMemcpyNodeFromSymbol(ctx, rec.bytes()); });
    INFO(err);
    CHECK(r == hipSuccess);
    CHECK(ctx.graph_is_incomplete(kGraph));
    CHECK(err.find("hipGraphAddMemcpyNodeFromSymbol") != std::string::npos);
    CHECK(err.find("sequence number 9") != std::string::npos);
  }
}

TEST_CASE("Unit_HRR_GraphDeps_SymbolCopyNodesRefuseTruncatedList", "[hrr]") {
  constexpr uint64_t kSym = 0x2000, kDev = 0x3000;
  char symbol_storage[64];
  char dev_storage[64];

  auto make_ctx = [&](PlaybackContext& ctx) {
    ctx.record_graph(kGraph, reinterpret_cast<hipGraph_t>(0x1));
    ctx.record_symbol(kSym, "sym", symbol_storage, sizeof(symbol_storage));
    ctx.record_alloc(kDev, dev_storage, sizeof(dev_storage));
  };

  {  // hipGraphAddMemcpyNodeToSymbol
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
    a->pDependencies_n = 16;
    a->numDependencies = 20;
    std::memset(a->pDependencies_bytes, 0, sizeof(a->pDependencies_bytes));

    hrr_dispatch_seq = 11;
    hipError_t r = hipErrorUnknown;
    const std::string err = capture_stderr(
        [&] { r = playback_hipGraphAddMemcpyNodeToSymbol(ctx, rec.bytes()); });
    INFO(err);
    CHECK(r == hipSuccess);
    CHECK(ctx.graph_is_incomplete(kGraph));
    CHECK(err.find("hipGraphAddMemcpyNodeToSymbol") != std::string::npos);
    CHECK(err.find("sequence number 11") != std::string::npos);
    CHECK(err.find("20 dependencies") != std::string::npos);
    CHECK(err.find("holds 16") != std::string::npos);
  }

  {  // hipGraphAddMemcpyNodeFromSymbol
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
    a->pDependencies_n = 16;
    a->numDependencies = 20;
    std::memset(a->pDependencies_bytes, 0, sizeof(a->pDependencies_bytes));

    hrr_dispatch_seq = 12;
    hipError_t r = hipErrorUnknown;
    const std::string err = capture_stderr(
        [&] { r = playback_hipGraphAddMemcpyNodeFromSymbol(ctx, rec.bytes()); });
    INFO(err);
    CHECK(r == hipSuccess);
    CHECK(ctx.graph_is_incomplete(kGraph));
    CHECK(err.find("hipGraphAddMemcpyNodeFromSymbol") != std::string::npos);
    CHECK(err.find("sequence number 12") != std::string::npos);
    CHECK(err.find("20 dependencies") != std::string::npos);
    CHECK(err.find("holds 16") != std::string::npos);
  }
}

/** @} */
