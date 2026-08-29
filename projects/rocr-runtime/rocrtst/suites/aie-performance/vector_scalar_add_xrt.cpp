// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Benchmark for vector_scalar_add kernel dispatch via XRT.

#include <benchmark/benchmark.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_kernel.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

static std::string g_xclbin_path = STRINGIFY(DEFAULT_XCLBIN_PATH);
static std::string g_insts_path = STRINGIFY(DEFAULT_INSTS_PATH);

static constexpr int N = 1024;

namespace {

// Load the control-code binary produced by aiecc.
std::vector<uint32_t> load_instr_binary(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("Cannot open instruction file: " + path);

  auto size = f.tellg();
  f.seekg(0);
  std::vector<uint32_t> instr(size / sizeof(uint32_t));
  f.read(reinterpret_cast<char*>(instr.data()), size);
  return instr;
}

// The xclbin names the kernel MLIR_AIE<suffix>; match on the prefix.
std::string find_kernel_name(const xrt::xclbin& xclbin) {
  auto xkernels = xclbin.get_kernels();
  auto it = std::find_if(xkernels.begin(), xkernels.end(),
                         [](const auto& k) { return k.get_name().rfind("MLIR_AIE", 0) == 0; });
  if (it == xkernels.end()) throw std::runtime_error("Kernel MLIR_AIE not found in xclbin");
  return it->get_name();
}

// The xclbin has to be registered with the device before a hardware context can
// be created from its UUID; do both in one step so it fits a member initializer.
xrt::uuid register_xclbin(xrt::device& device, const xrt::xclbin& xclbin) {
  device.register_xclbin(xclbin);
  return xclbin.get_uuid();
}

// Per-benchmark fixture: load the xclbin, open the device, and allocate the
// instruction buffer plus one input and one output buffer per dispatch.
struct XrtHarness {
  std::vector<uint32_t> instr_v;
  xrt::device device;
  xrt::xclbin xclbin;
  xrt::hw_context context;
  xrt::kernel kernel;
  xrt::bo bo_instr;
  std::vector<xrt::bo> bo_ins;
  std::vector<xrt::bo> bo_outs;

  explicit XrtHarness(std::int32_t num_dispatches)
      : instr_v(load_instr_binary(g_insts_path)),
        device(0),
        xclbin(g_xclbin_path),
        context(device, register_xclbin(device, xclbin)),
        kernel(context, find_kernel_name(xclbin)),
        bo_instr(device, instr_v.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE,
                 kernel.group_id(1)) {
    // Copy the control code in and sync it once; it is constant across dispatches.
    void* buf_instr = bo_instr.map<void*>();
    std::memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    bo_ins.reserve(num_dispatches);
    bo_outs.reserve(num_dispatches);
    for (std::int32_t i = 0; i < num_dispatches; ++i) {
      bo_ins.emplace_back(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
      bo_outs.emplace_back(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

      // Initialize input: [1, 2, ..., 1024]
      auto* buf_in = bo_ins[i].map<int32_t*>();
      std::iota(buf_in, buf_in + N, 1);

      // Initialize output: all zeros. The mapping is cached, so the write has
      // to be flushed before the first dispatch -- otherwise the dirty lines
      // write back over the DMA'd result.
      auto* buf_out = bo_outs[i].map<int32_t*>();
      std::fill_n(buf_out, N, 0);
      bo_outs[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
  }

  // Verify the kernel actually ran; otherwise the numbers are timing a no-op
  // dispatch. Input is [1..N], so element i must come back as i + 2.
  bool verify() const {
    for (const auto& bo_out : bo_outs) {
      const auto* buf_out = bo_out.map<const int32_t*>();
      for (int i = 0; i < N; ++i) {
        if (buf_out[i] != i + 2) return false;
      }
    }
    return true;
  }
};

}  // namespace

static void VectorScalarAddXRTKernel(benchmark::State& state) {
  const std::int32_t num_dispatches = state.range(0);
  try {
    XrtHarness h(num_dispatches);

    // Benchmark loop: sync in -> dispatch -> wait -> sync out (all dispatches)
    std::vector<xrt::run> runs(num_dispatches);
    for (auto _ : state) {
      for (std::int32_t i = 0; i < num_dispatches; ++i) {
        h.bo_ins[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        const unsigned int opcode = 3;
        runs[i] = h.kernel(opcode, h.bo_instr, h.instr_v.size(), h.bo_ins[i], h.bo_outs[i]);
      }

      for (std::int32_t i = 0; i < num_dispatches; ++i) {
        runs[i].wait();
        h.bo_outs[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      }

      benchmark::ClobberMemory();
    }

    if (!h.verify()) state.SkipWithError("Incorrect kernel output");
  } catch (const std::exception& e) {
    // Google Benchmark does not catch, so without this a missing artifact or a
    // failed dispatch terminates the whole run instead of failing one case.
    state.SkipWithError(e.what());
  }
}

static void VectorScalarAddXRTRunlist(benchmark::State& state) {
  const std::int32_t num_dispatches = state.range(0);
  try {
    XrtHarness h(num_dispatches);

    // Bind once outside the loop; the runlist is re-executed each iteration.
    xrt::runlist runlist(h.context);
    for (std::int32_t i = 0; i < num_dispatches; ++i) {
      const unsigned int opcode = 3;
      xrt::run run(h.kernel);
      run.set_arg(0, opcode);
      run.set_arg(1, h.bo_instr);
      run.set_arg(2, h.instr_v.size());
      run.set_arg(3, h.bo_ins[i]);
      run.set_arg(4, h.bo_outs[i]);
      runlist.add(std::move(run));
    }

    // Benchmark loop: sync inputs -> execute -> wait -> sync outputs
    for (auto _ : state) {
      for (std::int32_t i = 0; i < num_dispatches; ++i) {
        h.bo_ins[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
      }

      runlist.execute();
      runlist.wait();

      for (std::int32_t i = 0; i < num_dispatches; ++i) {
        h.bo_outs[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      }

      benchmark::ClobberMemory();
    }

    if (!h.verify()) state.SkipWithError("Incorrect kernel output");
  } catch (const std::exception& e) {
    state.SkipWithError(e.what());
  }
}

BENCHMARK(VectorScalarAddXRTKernel)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(1, 32);
BENCHMARK(VectorScalarAddXRTRunlist)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(1, 32);
