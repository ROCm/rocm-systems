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
  // One argument record: value_kind 0 (by value), size, then `size` bytes.
  void arg(uint16_t size) {
    const uint8_t kind = 0;
    put(&kind, 1);
    put(&size, 2);
    b.insert(b.end(), size, 0xab);
  }
  void args(const Sizes& sizes) {
    for (size_t s : sizes) arg(static_cast<uint16_t>(s));
  }
  void finish(hrr_api_id_t id) {
    auto* h = reinterpret_cast<hrr_event_header*>(b.data());
    h->event_type = static_cast<uint16_t>(id);
    h->sequence_id = kSeq;
    h->payload_length = static_cast<uint32_t>(b.size());
  }
};

// A hipModuleLaunchKernel event for kernel "k" that declares and carries one
// argument per size.
Payload launch(const Sizes& sizes) {
  Payload records;
  records.args(sizes);
  Payload p;
  p.b = hrr_test::make_launch_event(kSeq, "k", static_cast<uint16_t>(sizes.size()),
                                    /*num_snapshots=*/0, records.b);
  return p;
}

// A context whose kernel "k" resolves to a function whose code object declares
// parameters of the given sizes.
void declare(PlaybackContext& ctx, Sizes declared) {
  ctx.func_cache["k"] = fake_func();
  ctx.kernel_param_sizes = [declared](hipFunction_t f, std::vector<size_t>& out) {
    REQUIRE(f == fake_func());
    out = declared;
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
 *   - A launch record with fewer arguments than the code object's kernel
 *     declares is refused before the launch, naming the sequence.
 */
TEST_CASE("Unit_HRR_KernelArgsSignature_TooFewArguments", "[hrr]") {
  PlaybackContext ctx;
  declare(ctx, {8, 8, 4});
  auto p = launch({8, 8});
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
  auto narrow = launch({8, 4});
  check_refused(ctx, narrow, "argument 1 is 4 bytes, the kernel declares 16");
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

/**
 * Test Description
 * ----------------
 *   - Every decoded argument is followed by 8 zeroed bytes past its recorded
 *     size, so CLR's 8-byte read of a pointer parameter stays inside the
 *     allocation for a 4-byte argument and for an empty one (whose data() would
 *     otherwise be null). kernelParams[i] points at that storage.
 */
TEST_CASE("Unit_HRR_KernelArgsSignature_ArgumentsCarryReadPadding", "[hrr]") {
  PlaybackContext ctx;
  Payload records;
  records.args({4, 0, 16});
  std::vector<void*> ptrs;
  std::vector<std::vector<uint8_t>> storage;
  const uint8_t* begin = records.b.data();
  hrr_decode_kernel_args(ctx, begin, begin + records.b.size(), 3, ptrs, storage);

  const Sizes recorded = {4, 0, 16};
  REQUIRE(storage.size() == recorded.size());
  REQUIRE(ptrs.size() == recorded.size());
  for (size_t i = 0; i < recorded.size(); i++) {
    INFO("argument " << i);
    REQUIRE(storage[i].size() == recorded[i] + 8);
    CHECK(ptrs[i] == storage[i].data());
    for (size_t b = 0; b < recorded[i]; b++) CHECK(storage[i][b] == 0xab);
    for (size_t b = recorded[i]; b < recorded[i] + 8; b++) CHECK(storage[i][b] == 0);
  }
}

/**
 * Test Description
 * ----------------
 *   - A kernel set-params record with no parameters, as capture writes it for a
 *     call the runtime refused with hipErrorInvalidValue, replays as that
 *     recorded failure: hipSuccess, one note. The same record with a recorded
 *     success still fails with hipErrorInvalidValue. Covers
 *     hipGraphKernelNodeSetParams and hipGraphExecKernelNodeSetParams.
 */
TEST_CASE("Unit_HRR_KernelArgsSignature_SetParamsWithoutParameters", "[hrr]") {
  const int32_t refused = static_cast<int32_t>(hipErrorInvalidValue);
  auto run = [](hrr_api_id_t id, int32_t recorded, std::string* err) {
    PlaybackContext ctx;
    ctx.graph_node_map[0x88] = reinterpret_cast<hipGraphNode_t>(uintptr_t{0x3000});
    ctx.graph_exec_map[0x99] = reinterpret_cast<hipGraphExec_t>(uintptr_t{0x4000});
    Payload p;
    if (id == HRR_API_HIPGRAPHKERNELNODESETPARAMS) {
      p.b.assign(sizeof(hrr_args_hipGraphKernelNodeSetParams), 0);
      auto* a = reinterpret_cast<hrr_args_hipGraphKernelNodeSetParams*>(p.b.data());
      a->ret = recorded;
      a->node = 0x88;
    } else {
      p.b.assign(sizeof(hrr_args_hipGraphExecKernelNodeSetParams), 0);
      auto* a = reinterpret_cast<hrr_args_hipGraphExecKernelNodeSetParams*>(p.b.data());
      a->ret = recorded;
      a->hGraphExec = 0x99;
      a->node = 0x88;
    }
    p.finish(id);
    hipError_t r = hipErrorUnknown;
    *err = hrr_test::capture_stderr([&] {
      hrr_dispatch_seq = kSeq;
      r = hrr_playback_dispatch[id](ctx, p.b.data());
    });
    return r;
  };

  for (hrr_api_id_t id : {HRR_API_HIPGRAPHKERNELNODESETPARAMS,
                          HRR_API_HIPGRAPHEXECKERNELNODESETPARAMS}) {
    INFO("api id " << id);
    std::string err;
    CHECK(run(id, refused, &err) == hipSuccess);
    CHECK(err.find("what it returned at capture") != std::string::npos);
    CHECK(run(id, 0, &err) == hipErrorInvalidValue);
    CHECK(err.find("what it returned at capture") == std::string::npos);
  }
}

/** @} */
