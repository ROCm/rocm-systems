// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Benchmark for vector_scalar_add kernel dispatch via XRT (old instruction-buffer API).
// Measures combined: sync input -> dispatch -> wait -> sync output.

#include <benchmark/benchmark.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include "xrt/experimental/xrt_kernel.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

static std::string g_xclbin_path = STRINGIFY(DEFAULT_XCLBIN_PATH);
static std::string g_insts_path = STRINGIFY(DEFAULT_INSTS_PATH);

static constexpr int N = 1024;

static std::vector<uint32_t> load_instr_binary(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("Cannot open instruction file: " + path);

  auto size = f.tellg();
  f.seekg(0);
  std::vector<uint32_t> instr(size / sizeof(uint32_t));
  f.read(reinterpret_cast<char*>(instr.data()), size);
  return instr;
}

static void VectorScalarAddXRT(benchmark::State& state) {
  // Load instructions
  std::vector<uint32_t> instr_v = load_instr_binary(g_insts_path);

  // Open device and load xclbin
  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(g_xclbin_path);

  // Find kernel by name prefix "MLIR_AIE"
  auto xkernels = xclbin.get_kernels();
  auto xkernel_it = std::find_if(xkernels.begin(), xkernels.end(), [](const auto& k) {
    return k.get_name().rfind("MLIR_AIE", 0) == 0;
  });
  if (xkernel_it == xkernels.end()) {
    state.SkipWithError("Kernel MLIR_AIE not found in xclbin");
    return;
  }
  auto kernel_name = xkernel_it->get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernel_name);

  const std::int32_t num_dispatches = state.range(0);

  // Allocate buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE,
                          kernel.group_id(1));

  std::vector<xrt::bo> bo_ins;
  std::vector<xrt::bo> bo_outs;
  bo_ins.reserve(num_dispatches);
  bo_outs.reserve(num_dispatches);
  for (std::int32_t i = 0; i < num_dispatches; ++i) {
    bo_ins.emplace_back(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    bo_outs.emplace_back(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  }

  // Copy instructions into BO
  void* buf_instr = bo_instr.map<void*>();
  std::memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  // Initialize inputs and zero outputs
  for (std::int32_t i = 0; i < num_dispatches; ++i) {
    // Initialize input: [1, 2, ..., 1024]
    auto* buf_in = bo_ins[i].map<int32_t*>();
    std::iota(buf_in, buf_in + N, 1);

    // Initialize output: all zeros
    auto* buf_out = bo_outs[i].map<int32_t*>();
    std::fill_n(buf_out, N, 0);
  }

  // Sync instructions (constant, only once)
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Benchmark loop: sync in -> dispatch -> wait -> sync out (all dispatches)
  for (auto _ : state) {
    for (std::int32_t i = 0; i < num_dispatches; ++i) {
      bo_ins[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);

      const unsigned int opcode = 3;
      auto run = kernel(opcode, bo_instr, instr_v.size(), bo_ins[i], bo_outs[i]);
      run.wait();

      bo_outs[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }

    benchmark::ClobberMemory();
  }
}

static void VectorScalarAddXRTRunlist(benchmark::State& state) {
  // Load instructions
  std::vector<uint32_t> instr_v = load_instr_binary(g_insts_path);

  // Open device and load xclbin
  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(g_xclbin_path);

  // Find kernel by name prefix "MLIR_AIE"
  auto xkernels = xclbin.get_kernels();
  auto xkernel_it = std::find_if(xkernels.begin(), xkernels.end(), [](const auto& k) {
    return k.get_name().rfind("MLIR_AIE", 0) == 0;
  });
  if (xkernel_it == xkernels.end()) {
    state.SkipWithError("Kernel MLIR_AIE not found in xclbin");
    return;
  }
  auto kernel_name = xkernel_it->get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernel_name);

  const std::int32_t num_dispatches = state.range(0);

  // Allocate buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE,
                          kernel.group_id(1));

  std::vector<xrt::bo> bo_ins;
  std::vector<xrt::bo> bo_outs;
  bo_ins.reserve(num_dispatches);
  bo_outs.reserve(num_dispatches);
  for (std::int32_t i = 0; i < num_dispatches; ++i) {
    bo_ins.emplace_back(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    bo_outs.emplace_back(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  }

  // Copy instructions into BO
  void* buf_instr = bo_instr.map<void*>();
  std::memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  // Initialize inputs and zero outputs
  for (std::int32_t i = 0; i < num_dispatches; ++i) {
    // Initialize input: [1, 2, ..., 1024]
    auto* buf_in = bo_ins[i].map<int32_t*>();
    std::iota(buf_in, buf_in + N, 1);

    // Initialize output: all zeros
    auto* buf_out = bo_outs[i].map<int32_t*>();
    std::fill_n(buf_out, N, 0);
  }

  // Sync instructions (constant, only once)
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Benchmark loop: sync inputs -> build runlist -> execute -> wait -> sync outputs
  for (auto _ : state) {
    xrt::runlist runlist(context);

    for (std::int32_t i = 0; i < num_dispatches; ++i) {
      const unsigned int opcode = 3;
      xrt::run run(kernel);
      run.set_arg(0, opcode);
      run.set_arg(1, bo_instr);
      run.set_arg(2, instr_v.size());
      run.set_arg(3, bo_ins[i]);
      run.set_arg(4, bo_outs[i]);
      runlist.add(std::move(run));
      bo_ins[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    runlist.execute();
    runlist.wait();

    for (std::int32_t i = 0; i < num_dispatches; ++i) {
      bo_outs[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }

    benchmark::ClobberMemory();
  }
}

BENCHMARK(VectorScalarAddXRT)->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1, 32);
BENCHMARK(VectorScalarAddXRTRunlist)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(1, 32);
