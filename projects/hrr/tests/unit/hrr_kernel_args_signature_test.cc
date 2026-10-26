/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Kernel Argument Signature
 * @{
 * @ingroup HRRTest
 * Handler-level tests that replay checks a launch record's arguments against
 * the signature of the kernel it resolved before it launches: the handlers are
 * called through the dispatch table with crafted records, and the code object's
 * signature is a stand-in (PlaybackContext::kernel_param_sizes). Every refusal
 * returns before the launch, so no GPU is needed. A launch that matches goes on
 * to HIP and is covered by the integration suite.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint64_t kSeq = 4242;
// Never dereferenced: the signature comes from the stand-in, and every launch
// in this file is refused before HIP sees the function.
hipFunction_t fake_func() { return reinterpret_cast<hipFunction_t>(uintptr_t{0x1000}); }

using Sizes = std::vector<size_t>;

struct Payload {
  std::vector<uint8_t> b;
  void put(const void* d, size_t n) {
    b.insert(b.end(), static_cast<const uint8_t*>(d), static_cast<const uint8_t*>(d) + n);
  }
  // One argument record: value_kind 0 (by value), size, then `present` bytes.
  void arg(uint16_t size, size_t present) {
    const uint8_t kind = 0;
    put(&kind, 1);
    put(&size, 2);
    b.insert(b.end(), present, 0xab);
  }
  void args(const Sizes& sizes) {
    for (size_t s : sizes) arg(static_cast<uint16_t>(s), s);
  }
  void finish(hrr_api_id_t id) {
    auto* h = reinterpret_cast<hrr_event_header*>(b.data());
    h->event_type = static_cast<uint16_t>(id);
    h->sequence_id = kSeq;
    h->payload_length = static_cast<uint32_t>(b.size());
  }
};

// A hipModuleLaunchKernel event that declares `declared` arguments and carries
// the records `fill` writes.
template <typename Fill>
Payload launch(const char* name, uint16_t declared, Fill&& fill) {
  Payload records;
  fill(records);
  Payload p;
  p.b = hrr_test::make_launch_event(kSeq, name, declared, /*num_snapshots=*/0, records.b);
  return p;
}

// A context whose kernel "k" resolves to a function whose code object declares
// parameters of the given sizes.
void declare(PlaybackContext& ctx, Sizes declared) {
  ctx.func_cache["k"] = fake_func();
  ctx.kernel_param_sizes = [declared](hipFunction_t f, std::vector<size_t>& out) {
    REQUIRE(f == fake_func());
    out = declared;
    return true;
  };
}

hipError_t run_launch(PlaybackContext& ctx, const Payload& p, std::string* err) {
  hipError_t r = hipSuccess;
  *err = hrr_test::capture_stderr([&] {
    hrr_dispatch_seq = kSeq;
    r = hrr_playback_dispatch[HRR_API_HIPMODULELAUNCHKERNEL](ctx, p.b.data());
  });
  return r;
}

void check_refused(PlaybackContext& ctx, const Payload& p, const char* what) {
  std::string err;
  CHECK(run_launch(ctx, p, &err) == hipErrorInvalidValue);
  CHECK(err.find("launch refused") != std::string::npos);
  CHECK(err.find("seq=4242") != std::string::npos);
  CHECK(err.find(what) != std::string::npos);
  CHECK(ctx.kernels_launched.load() == 0);
}

}  // namespace

/**
 * Test Description
 * ----------------
 *   - The comparison accepts an argument list that covers the signature: the
 *     same sizes, and also extra arguments or wider ones, which CLR never
 *     reads. It refuses a short list and a narrower argument, saying which.
 */
TEST_CASE("Unit_HRR_KernelArgsSignature_Match", "[hrr]") {
  std::string why;
  CHECK(hrr_kernel_args_match({8, 4}, {8, 4}, &why));
  CHECK(hrr_kernel_args_match({}, {}, &why));
  CHECK(hrr_kernel_args_match({8, 4, 4}, {8, 4}, &why));  // extra argument
  CHECK(hrr_kernel_args_match({4}, {}, &why));            // kernel declares none
  CHECK(hrr_kernel_args_match({16, 4}, {8, 4}, &why));    // wider argument
  CHECK_FALSE(hrr_kernel_args_match({8}, {8, 4}, &why));
  CHECK(why.find("carries 1 arguments, the kernel declares 2") != std::string::npos);
  CHECK_FALSE(hrr_kernel_args_match({8, 4}, {8, 8}, &why));
  CHECK(why.find("argument 1 is 4 bytes, the kernel declares 8") != std::string::npos);
  CHECK_FALSE(hrr_kernel_args_match({8, 2}, {8, 4}, nullptr));
}

/**
 * Test Description
 * ----------------
 *   - A launch record with fewer arguments than the code object's kernel
 *     declares is refused before the launch, naming the sequence.
 */
TEST_CASE("Unit_HRR_KernelArgsSignature_TooFewArguments", "[hrr]") {
  PlaybackContext ctx;
  declare(ctx, {8, 8, 4});
  auto p = launch("k", 2, [](Payload& p) { p.args({8, 8}); });
  check_refused(ctx, p, "carries 2 arguments, the kernel declares 3");
}

/**
 * Test Description
 * ----------------
 *   - An argument narrower than its parameter is refused before the launch,
 *     with the argument's index. Extra and wider arguments are not refused:
 *     they go on to HIP, which the integration suite covers on a GPU.
 */
TEST_CASE("Unit_HRR_KernelArgsSignature_NarrowArgument", "[hrr]") {
  PlaybackContext ctx;
  declare(ctx, {8, 16});
  auto narrow = launch("k", 2, [](Payload& p) { p.args({8, 4}); });
  check_refused(ctx, narrow, "argument 1 is 4 bytes, the kernel declares 16");
}

/**
 * Test Description
 * ----------------
 *   - A payload that ends before the declared arguments do is refused even
 *     when the arguments that did decode would match, whether it ends between
 *     two records or inside one.
 */
TEST_CASE("Unit_HRR_KernelArgsSignature_TruncatedPayload", "[hrr]") {
  PlaybackContext ctx;
  declare(ctx, {8});
  // Declares two arguments, carries one whole one; the first matches the
  // one-parameter signature, so only the truncation distinguishes it.
  auto between = launch("k", 2, [](Payload& p) { p.args({8}); });
  // The launch-attribute tail follows, so cut the event short instead of
  // letting it read as the second argument.
  between.b.resize(between.b.size() - 8);
  between.finish(HRR_API_HIPMODULELAUNCHKERNEL);
  check_refused(ctx, between, "payload ends before the arguments");

  auto inside = launch("k", 1, [](Payload& p) { p.arg(8, 3); });
  inside.b.resize(inside.b.size() - 8);
  inside.finish(HRR_API_HIPMODULELAUNCHKERNEL);
  check_refused(ctx, inside, "payload ends before the arguments");
}

/**
 * Test Description
 * ----------------
 *   - A kernel whose signature cannot be read is refused rather than launched
 *     unchecked.
 */
TEST_CASE("Unit_HRR_KernelArgsSignature_UnreadableSignature", "[hrr]") {
  PlaybackContext ctx;
  ctx.func_cache["k"] = fake_func();
  ctx.kernel_param_sizes = [](hipFunction_t, std::vector<size_t>&) { return false; };
  auto p = launch("k", 1, [](Payload& p) { p.args({8}); });
  check_refused(ctx, p, "signature cannot be read");
}

/**
 * Test Description
 * ----------------
 *   - A graph kernel node whose arguments do not match the signature is not
 *     added: the node is skipped and its graph is marked incomplete.
 */
TEST_CASE("Unit_HRR_KernelArgsSignature_GraphKernelNode", "[hrr]") {
  PlaybackContext ctx;
  declare(ctx, {8, 8});
  ctx.graph_map[0x77] = reinterpret_cast<hipGraph_t>(uintptr_t{0x2000});

  Payload p;
  p.b.assign(sizeof(hrr_args_hipGraphAddKernelNode), 0);
  auto* a = reinterpret_cast<hrr_args_hipGraphAddKernelNode*>(p.b.data());
  a->graph = 0x77;
  a->pNodeParams_present = 1;
  const char* name = "k";
  const uint16_t name_len = 1, num_args = 1;
  const uint64_t co_hash[2] = {0, 0};
  p.put(&name_len, 2);
  p.put(name, 1);
  p.put(co_hash, 16);
  p.put(&num_args, 2);
  p.args({8});
  p.finish(HRR_API_HIPGRAPHADDKERNELNODE);

  hipError_t r = hipErrorUnknown;
  std::string err = hrr_test::capture_stderr([&] {
    hrr_dispatch_seq = kSeq;
    r = hrr_playback_dispatch[HRR_API_HIPGRAPHADDKERNELNODE](ctx, p.b.data());
  });
  CHECK(r == hipSuccess);
  CHECK(err.find("launch refused") != std::string::npos);
  CHECK(err.find("seq=4242") != std::string::npos);
  CHECK(err.find("carries 1 arguments, the kernel declares 2") != std::string::npos);
  CHECK(ctx.incomplete_graphs.count(0x77) == 1);
}

/**
 * Test Description
 * ----------------
 *   - A set-params record whose arguments do not cover the signature fails with
 *     hipErrorInvalidValue, as a refused launch does, instead of returning
 *     hipSuccess and leaving the node on its old arguments. Covers
 *     hipGraphKernelNodeSetParams and hipGraphExecKernelNodeSetParams.
 */
TEST_CASE("Unit_HRR_KernelArgsSignature_GraphSetParams", "[hrr]") {
  PlaybackContext ctx;
  declare(ctx, {8, 8, 4});
  ctx.graph_node_map[0x88] = reinterpret_cast<hipGraphNode_t>(uintptr_t{0x3000});
  ctx.graph_exec_map[0x99] = reinterpret_cast<hipGraphExec_t>(uintptr_t{0x4000});

  // Two arguments for a three-parameter kernel.
  auto tail = [](Payload& p) {
    const char* name = "k";
    const uint16_t name_len = 1, num_args = 2;
    const uint64_t co_hash[2] = {0, 0};
    p.put(&name_len, 2);
    p.put(name, 1);
    p.put(co_hash, 16);
    p.put(&num_args, 2);
    p.args({8, 8});
  };
  auto run = [&](Payload& p, hrr_api_id_t id) {
    hipError_t r = hipSuccess;
    std::string err = hrr_test::capture_stderr([&] {
      hrr_dispatch_seq = kSeq;
      r = hrr_playback_dispatch[id](ctx, p.b.data());
    });
    CHECK(r == hipErrorInvalidValue);
    CHECK(err.find("launch refused") != std::string::npos);
    CHECK(err.find("carries 2 arguments, the kernel declares 3") != std::string::npos);
  };

  Payload node;
  node.b.assign(sizeof(hrr_args_hipGraphKernelNodeSetParams), 0);
  auto* n = reinterpret_cast<hrr_args_hipGraphKernelNodeSetParams*>(node.b.data());
  n->node = 0x88;
  n->pNodeParams_present = 1;
  tail(node);
  node.finish(HRR_API_HIPGRAPHKERNELNODESETPARAMS);
  run(node, HRR_API_HIPGRAPHKERNELNODESETPARAMS);

  Payload exec;
  exec.b.assign(sizeof(hrr_args_hipGraphExecKernelNodeSetParams), 0);
  auto* e = reinterpret_cast<hrr_args_hipGraphExecKernelNodeSetParams*>(exec.b.data());
  e->hGraphExec = 0x99;
  e->node = 0x88;
  e->pNodeParams_present = 1;
  tail(exec);
  exec.finish(HRR_API_HIPGRAPHEXECKERNELNODESETPARAMS);
  run(exec, HRR_API_HIPGRAPHEXECKERNELNODESETPARAMS);
}

/** @} */
