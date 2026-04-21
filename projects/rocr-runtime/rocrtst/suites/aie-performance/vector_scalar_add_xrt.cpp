// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Benchmark for vector_scalar_add kernel dispatch via XRT (old instruction-buffer API).
// Measures combined: sync input -> dispatch -> wait -> sync output.

#include <benchmark/benchmark.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
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

  // Allocate buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE,
                          kernel.group_id(1));
  auto bo_in = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  // Dummy BOs for unused kernel arguments
  auto bo_tmp = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_ctrlpkts = xrt::bo(device, 8, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  auto bo_trace = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  // Copy instructions into BO
  void* buf_instr = bo_instr.map<void*>();
  std::memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  // Initialize input: [1, 2, 3, ..., 1024]
  auto* buf_in = bo_in.map<int32_t*>();
  std::iota(buf_in, buf_in + N, 1);

  // Zero output
  auto* buf_out = bo_out.map<int32_t*>();
  std::memset(buf_out, 0, N * sizeof(int32_t));

  // Sync instructions (constant, only once)
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Benchmark loop: sync in -> dispatch -> wait -> sync out
  for (auto _ : state) {
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    unsigned int opcode = 3;
    auto run =
        kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_out, bo_tmp, bo_ctrlpkts, bo_trace);
    run.wait();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    benchmark::ClobberMemory();
  }
}

BENCHMARK(VectorScalarAddXRT)->Unit(benchmark::kMicrosecond);
