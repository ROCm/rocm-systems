// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Scaling test: measures simulation wall-clock time for vector_add, matmul_tiled,
// and matmul_mfma across 1..8 shared dispatch-pool threads. Outputs CSV to stdout.

#include "aql_queue.h"
#include "scaling_thread_counts.h"
#include "test_paths.h"

#include "embedded_schema.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"

#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAS_DEVICE_KERNELS

using namespace rocjitsu;

static const std::string CONFIG_PATH = test::config_path("gfx950_mi355x.json");
using test::kernel_path;

static constexpr uint32_t TOTAL_XCDS = 8;
static constexpr uint32_t CUS_PER_XCD = 36; // 4 SEs x 9 physical CUs
static constexpr uint32_t TOTAL_CUS = TOTAL_XCDS * CUS_PER_XCD;
static constexpr uint32_t WF_SIZE = 64;

static constexpr uint64_t KD_ADDR = 0x10000;
static constexpr uint64_t DATA_ADDR = 0x100000;
static constexpr uint64_t DATA_ALIGNMENT = 4096;

uint64_t align_up(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

std::vector<float> expected_result(const std::vector<float> &a, const std::vector<float> &b,
                                   uint32_t n, bool is_vector_add) {
  std::vector<float> expected(a.size(), 0.0f);
  if (is_vector_add) {
    for (size_t i = 0; i < a.size(); ++i)
      expected[i] = a[i] + b[i];
    return expected;
  }

  for (uint32_t row = 0; row < n; ++row)
    for (uint32_t col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < n; ++k)
        sum += a[static_cast<size_t>(row) * n + k] * b[static_cast<size_t>(k) * n + col];
      expected[static_cast<size_t>(row) * n + col] = sum;
    }
  return expected;
}

void validate_result(const char *kernel_name, const std::vector<float> &actual,
                     const std::vector<float> &expected) {
  size_t mismatches = 0;
  size_t first_mismatch = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float tolerance = 1e-3f * std::fabs(expected[i]) + 1e-6f;
    if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance) {
      if (mismatches == 0)
        first_mismatch = i;
      ++mismatches;
    }
  }
  if (mismatches == 0)
    return;

  std::ostringstream message;
  message << kernel_name << " produced " << mismatches << " incorrect values; first mismatch at "
          << first_mismatch << ": got " << actual[first_mismatch] << ", expected "
          << expected[first_mismatch];
  throw std::runtime_error(message.str());
}

double run_kernel(const char *kernel_name, uint32_t N, uint32_t total_wgs, uint32_t num_threads) {
  Executable exec(kernel_path(kernel_name));
  if (!exec.is_valid())
    throw std::runtime_error(std::string("failed to load kernel: ") + kernel_name);
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  if (!co)
    throw std::runtime_error(std::string("missing gfx950 code object: ") + kernel_name);
  if (total_wgs % TOTAL_XCDS != 0)
    throw std::runtime_error(std::string(kernel_name) + " workgroups do not divide across XCDs");
  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  loaded.engine_config.num_threads = 1;
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  // Sweep the host-wide shared dispatch-pool thread count; one engine partition.
  soc->set_dispatch_threads(num_threads);
  engine->create();

  co->load_to_memory(memory, KD_ADDR);
  uint64_t kernel_object = KD_ADDR + co->kernel_descriptor_offset(kernel_name);
  if (kernel_object == KD_ADDR)
    throw std::runtime_error(std::string("missing kernel descriptor: ") + kernel_name);

  // Setup data.
  size_t elems = static_cast<size_t>(N) * N;
  bool is_vector_add = (std::string(kernel_name) == "vector_add");
  if (is_vector_add)
    elems = static_cast<size_t>(total_wgs) * WF_SIZE;

  size_t data_bytes = elems * sizeof(float);
  std::vector<float> A(elems), B(elems);
  for (size_t i = 0; i < elems; ++i) {
    A[i] = static_cast<float>(i % 17) * 0.1f;
    B[i] = static_cast<float>(i % 13) * 0.1f;
  }
  const std::vector<float> expected = expected_result(A, B, N, is_vector_add);

  // Keep buffers disjoint for the largest workload. The vector-add sweep uses
  // 2.25 MiB per buffer, so fixed addresses spaced 1 MiB apart would overlap.
  const uint64_t a_addr = DATA_ADDR;
  const uint64_t b_addr = align_up(a_addr + data_bytes, DATA_ALIGNMENT);
  const uint64_t c_addr = align_up(b_addr + data_bytes, DATA_ALIGNMENT);
  const uint64_t kernarg_addr = align_up(c_addr + data_bytes, DATA_ALIGNMENT);

  memory->load_image(reinterpret_cast<const uint8_t *>(A.data()), data_bytes, a_addr);
  memory->load_image(reinterpret_cast<const uint8_t *>(B.data()), data_bytes, b_addr);
  std::vector<float> zeros(elems, 0.0f);
  memory->load_image(reinterpret_cast<const uint8_t *>(zeros.data()), data_bytes, c_addr);

  uint32_t kernarg_N = is_vector_add ? static_cast<uint32_t>(elems) : N;
  struct {
    uint64_t A, B, C;
    uint32_t N;
  } args = {a_addr, b_addr, c_addr, kernarg_N};
  memory->load_image(reinterpret_cast<const uint8_t *>(&args), sizeof(args), kernarg_addr);

  // Dispatch across all XCDs via AQL queues.
  uint32_t wgs_per_xcd = total_wgs / TOTAL_XCDS;
  for (uint32_t xi = 0; xi < TOTAL_XCDS; ++xi) {
    auto *cp = soc->xcd(xi)->command_processor();
    cp->set_workgroup_id_offset(xi * wgs_per_xcd);
    uint64_t ring = 0xF0000000ULL + xi * 0x100000ULL;
    test::AqlQueue queue(memory, cp, ring, 4096, ring + 0x10000, ring + 0x10008, ring + 0x10010);
    queue.dispatch(kernel_object, wgs_per_xcd * WF_SIZE, WF_SIZE, kernarg_addr);
  }

  // Time only simulation execution; cache maintenance and validation are setup/reporting work.
  auto start = std::chrono::steady_clock::now();
  engine->run();
  auto end = std::chrono::steady_clock::now();

  soc->flush_all();
  std::vector<float> actual(elems);
  memory->read_block(c_addr, std::span<uint8_t>(reinterpret_cast<uint8_t *>(actual.data()),
                                                actual.size() * sizeof(float)));
  validate_result(kernel_name, actual, expected);

  return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char **argv) {
  // Thread counts come from argv when given, otherwise sweep 1..TOTAL_XCDS.
  const auto thread_counts = test::parse_thread_counts(
      std::span<const char *const>(argv + 1, static_cast<size_t>(argc - 1)), TOTAL_XCDS);
  if (!thread_counts) {
    std::cerr << "usage: " << argv[0] << " [thread-count ...]   (each 1.." << TOTAL_XCDS
              << "; none means sweep them all)\n";
    return 2;
  }

  struct Kernel {
    const char *name;
    uint32_t N;
    uint32_t total_wgs;
  };
  Kernel kernels[] = {
      {"vector_add", 0, TOTAL_CUS * 32}, // N unused for vector_add
      {"matmul_tiled", 256, (256 * 256) / WF_SIZE},
      {"matmul_mfma", 128, (128 / 4) * (128 / 4)},
  };

  std::cout << "dispatch_threads";
  for (auto &k : kernels)
    std::cout << "," << k.name;
  std::cout << "\n";

  constexpr int RUNS = 3;

  try {
    for (uint32_t t : *thread_counts) {
      std::vector<double> medians;
      medians.reserve(std::size(kernels));
      for (auto &k : kernels) {
        std::vector<double> times;
        times.reserve(RUNS);
        for (int r = 0; r < RUNS; ++r)
          times.push_back(run_kernel(k.name, k.N, k.total_wgs, t));
        std::sort(times.begin(), times.end());
        medians.push_back(times[RUNS / 2]);
      }
      std::cout << t;
      for (double median : medians)
        std::cout << "," << median;
      std::cout << "\n";
      std::cout.flush();
    }
  } catch (const std::exception &error) {
    std::cerr << "scaling test failed: " << error.what() << "\n";
    return 1;
  }
  return 0;
}

#else
int main() {
  std::cerr << "Device kernels not available. Build with HAS_DEVICE_KERNELS.\n";
  return 1;
}
#endif
