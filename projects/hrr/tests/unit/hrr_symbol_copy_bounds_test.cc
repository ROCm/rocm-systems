/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Symbol Copy Bounds
 * @{
 * @ingroup HRRTest
 * Tests that the six graph symbol-copy handlers (hip_playback.cpp) refuse a
 * record whose offset and count do not fit the symbol's registered size.
 *
 * The offset and count come from the archive. Each handler adds the offset to
 * the symbol's live address and forwards the sum to a 1D memcpy node call, and
 * HIP does not bound that copy against the symbol. The handlers are linked in
 * and called with crafted records and a symbol registry seeded here; the three
 * 1D HIP calls are stand-ins that count their calls, so no GPU is needed and a
 * test can show HIP was never reached.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <cstdint>
#include <cstring>
#include <string>

// Defined in hip_playback.cpp; not declared in the public header.
hipError_t playback_hipGraphAddMemcpyNodeToSymbol(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipGraphAddMemcpyNodeFromSymbol(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipGraphMemcpyNodeSetParamsToSymbol(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipGraphMemcpyNodeSetParamsFromSymbol(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipGraphExecMemcpyNodeSetParamsToSymbol(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipGraphExecMemcpyNodeSetParamsFromSymbol(PlaybackContext& ctx, const uint8_t* pl);

namespace {

using hrr_test::capture_stderr;
using hrr_test::ExactRecord;

// What the stand-ins below saw.
struct Seen {
  int calls = 0;
  const void* dst = nullptr;
  const void* src = nullptr;
  size_t count = 0;
};
Seen g_seen;

constexpr uint64_t kGraph = 0x1000, kNode = 0x1100, kExec = 0x1200;
constexpr uint64_t kSym = 0x2000, kDev = 0x3000;
constexpr size_t kSymSize = 64;

char g_symbol[256];  // the live global; larger than registered, to spot a stray write
char g_dev[256];

// A registry holding a kSymSize-byte symbol, a device allocation for the other
// operand, and the graph, node and exec the handlers translate.
struct Fixture {
  PlaybackContext ctx;
  explicit Fixture(size_t symbol_size = kSymSize) {
    ctx.record_graph(kGraph, reinterpret_cast<hipGraph_t>(0x1));
    ctx.record_graph_node(kNode, reinterpret_cast<hipGraphNode_t>(0x2));
    ctx.record_graph_exec(kExec, reinterpret_cast<hipGraphExec_t>(0x3));
    ctx.record_symbol(kSym, "sym", g_symbol, symbol_size);
    ctx.record_alloc(kDev, g_dev, sizeof(g_dev));
    g_seen = Seen{};
  }
};

// One entry per handler: fills a record from (offset, count) and runs it.
struct Handler {
  const char* name;
  hipError_t (*run)(PlaybackContext&, uint64_t offset, uint64_t count);
  bool dst_is_symbol;  // the copy writes the symbol, so HIP sees symbol+offset as dst
};

template <typename Rec>
void set_common(Rec* a, uint64_t offset, uint64_t count) {
  a->symbol = kSym;
  a->offset = offset;
  a->count = count;
  a->kind = hipMemcpyDeviceToDevice;
}

hipError_t run_add_to(PlaybackContext& ctx, uint64_t off, uint64_t n) {
  ExactRecord<hrr_args_hipGraphAddMemcpyNodeToSymbol> rec;
  set_common(rec.get(), off, n);
  rec.get()->graph = kGraph;
  rec.get()->src = kDev;
  return playback_hipGraphAddMemcpyNodeToSymbol(ctx, rec.bytes());
}
hipError_t run_add_from(PlaybackContext& ctx, uint64_t off, uint64_t n) {
  ExactRecord<hrr_args_hipGraphAddMemcpyNodeFromSymbol> rec;
  set_common(rec.get(), off, n);
  rec.get()->graph = kGraph;
  rec.get()->dst = kDev;
  return playback_hipGraphAddMemcpyNodeFromSymbol(ctx, rec.bytes());
}
hipError_t run_set_to(PlaybackContext& ctx, uint64_t off, uint64_t n) {
  ExactRecord<hrr_args_hipGraphMemcpyNodeSetParamsToSymbol> rec;
  set_common(rec.get(), off, n);
  rec.get()->node = kNode;
  rec.get()->src = kDev;
  return playback_hipGraphMemcpyNodeSetParamsToSymbol(ctx, rec.bytes());
}
hipError_t run_set_from(PlaybackContext& ctx, uint64_t off, uint64_t n) {
  ExactRecord<hrr_args_hipGraphMemcpyNodeSetParamsFromSymbol> rec;
  set_common(rec.get(), off, n);
  rec.get()->node = kNode;
  rec.get()->dst = kDev;
  return playback_hipGraphMemcpyNodeSetParamsFromSymbol(ctx, rec.bytes());
}
hipError_t run_exec_to(PlaybackContext& ctx, uint64_t off, uint64_t n) {
  ExactRecord<hrr_args_hipGraphExecMemcpyNodeSetParamsToSymbol> rec;
  set_common(rec.get(), off, n);
  rec.get()->hGraphExec = kExec;
  rec.get()->node = kNode;
  rec.get()->src = kDev;
  return playback_hipGraphExecMemcpyNodeSetParamsToSymbol(ctx, rec.bytes());
}
hipError_t run_exec_from(PlaybackContext& ctx, uint64_t off, uint64_t n) {
  ExactRecord<hrr_args_hipGraphExecMemcpyNodeSetParamsFromSymbol> rec;
  set_common(rec.get(), off, n);
  rec.get()->hGraphExec = kExec;
  rec.get()->node = kNode;
  rec.get()->dst = kDev;
  return playback_hipGraphExecMemcpyNodeSetParamsFromSymbol(ctx, rec.bytes());
}

const Handler kHandlers[] = {
    {"hipGraphAddMemcpyNodeToSymbol", run_add_to, true},
    {"hipGraphAddMemcpyNodeFromSymbol", run_add_from, false},
    {"hipGraphMemcpyNodeSetParamsToSymbol", run_set_to, true},
    {"hipGraphMemcpyNodeSetParamsFromSymbol", run_set_from, false},
    {"hipGraphExecMemcpyNodeSetParamsToSymbol", run_exec_to, true},
    {"hipGraphExecMemcpyNodeSetParamsFromSymbol", run_exec_from, false},
};

}  // namespace

// Stand-ins for the HIP calls. The test executable defines them, so the
// handlers' calls bind here instead of to the HIP runtime.
extern "C" hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* node, hipGraph_t, const hipGraphNode_t*,
                                              size_t, void* dst, const void* src, size_t count,
                                              hipMemcpyKind) {
  ++g_seen.calls;
  g_seen.dst = dst;
  g_seen.src = src;
  g_seen.count = count;
  if (node) *node = reinterpret_cast<hipGraphNode_t>(0x4);
  return hipSuccess;
}

extern "C" hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t, void* dst, const void* src,
                                                    size_t count, hipMemcpyKind) {
  ++g_seen.calls;
  g_seen.dst = dst;
  g_seen.src = src;
  g_seen.count = count;
  return hipSuccess;
}

extern "C" hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t, hipGraphNode_t, void* dst,
                                                        const void* src, size_t count,
                                                        hipMemcpyKind) {
  ++g_seen.calls;
  g_seen.dst = dst;
  g_seen.src = src;
  g_seen.count = count;
  return hipSuccess;
}

/**
 * Test Description
 * ----------------
 *   - Control: a copy that fits the symbol, including one that ends exactly at
 *     its end, reaches HIP once, aimed at the live symbol plus the offset. This
 *     shows the refusals below are the bound and not a setup failure.
 */
TEST_CASE("Unit_HRR_SymbolCopyBounds_InRangeCopyReachesHip", "[hrr]") {
  struct Fit { uint64_t off, n; };
  for (const Handler& h : kHandlers) {
    for (const Fit f : {Fit{0, 1}, Fit{8, 16}, Fit{0, kSymSize}, Fit{kSymSize - 1, 1},
                        Fit{kSymSize, 0}}) {
      Fixture fx;
      hipError_t r = hipErrorUnknown;
      const std::string err = capture_stderr([&] { r = h.run(fx.ctx, f.off, f.n); });
      INFO(h.name << " offset " << f.off << " count " << f.n << " stderr: " << err);
      CHECK(r == hipSuccess);
      REQUIRE(g_seen.calls == 1);
      CHECK(g_seen.count == f.n);
      const void* sym_side = h.dst_is_symbol ? g_seen.dst : g_seen.src;
      CHECK(sym_side == static_cast<const void*>(g_symbol + f.off));
    }
  }
}

/**
 * Test Description
 * ----------------
 *   - A record whose offset is at or past the symbol's size, or whose offset
 *     plus count passes it, is refused with hipErrorInvalidValue and HIP is
 *     never called. The message names the API, the offset and the registered
 *     size.
 */
TEST_CASE("Unit_HRR_SymbolCopyBounds_OutOfRangeIsRefusedBeforeHip", "[hrr]") {
  struct Bad { uint64_t off, n; };
  for (const Handler& h : kHandlers) {
    for (const Bad b : {Bad{kSymSize, 1}, Bad{kSymSize + 1, 1}, Bad{4096, 1}, Bad{0, kSymSize + 1},
                        Bad{kSymSize - 1, 2}, Bad{1, kSymSize}}) {
      Fixture fx;
      hipError_t r = hipSuccess;
      const std::string err = capture_stderr([&] { r = h.run(fx.ctx, b.off, b.n); });
      INFO(h.name << " offset " << b.off << " count " << b.n << " stderr: " << err);
      CHECK(r == hipErrorInvalidValue);
      CHECK(g_seen.calls == 0);
      CHECK(err.find(h.name) != std::string::npos);
      CHECK(err.find("does not fit") != std::string::npos);
      CHECK(err.find("64 bytes registered") != std::string::npos);
    }
  }
}

/**
 * Test Description
 * ----------------
 *   - Offsets and counts chosen so that offset + count wraps past UINT64_MAX
 *     to a small number are refused: the sum is never formed. So is a count of
 *     UINT64_MAX at offset 0.
 */
TEST_CASE("Unit_HRR_SymbolCopyBounds_WrappingSumIsRefused", "[hrr]") {
  struct Bad { uint64_t off, n; };
  for (const Handler& h : kHandlers) {
    for (const Bad b : {Bad{UINT64_MAX - 3, 8}, Bad{UINT64_MAX, 1}, Bad{8, UINT64_MAX - 3},
                        Bad{0, UINT64_MAX}, Bad{UINT64_MAX, UINT64_MAX}}) {
      Fixture fx;
      hipError_t r = hipSuccess;
      const std::string err = capture_stderr([&] { r = h.run(fx.ctx, b.off, b.n); });
      INFO(h.name << " offset " << b.off << " count " << b.n << " stderr: " << err);
      CHECK(r == hipErrorInvalidValue);
      CHECK(g_seen.calls == 0);
    }
  }
}

/** @} */
