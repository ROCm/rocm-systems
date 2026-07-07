// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Scaling test: measures simulation wall-clock time for vector_add, matmul_tiled,
// and matmul_mfma across 1..8 threads (one per XCD). Outputs CSV to stdout.

#include "aql_queue.h"
#include "test_paths.h"

#include "embedded_schema.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/partitioning.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef HAS_DEVICE_KERNELS

using namespace rocjitsu;

static const std::string CONFIG_PATH = test::config_path("gfx950_cdna4.json");
using test::kernel_path;

static constexpr uint32_t TOTAL_XCDS = 8;
static constexpr uint32_t CUS_PER_XCD = 32;
static constexpr uint32_t TOTAL_CUS = TOTAL_XCDS * CUS_PER_XCD;
static constexpr uint32_t WF_SIZE = 64;
static constexpr uint32_t DEFAULT_CPU_DISPATCH_THREADS = 32;

static constexpr uint64_t KD_ADDR = 0x10000;
static constexpr uint64_t A_ADDR = 0x100000;
static constexpr uint64_t B_ADDR = 0x200000;
static constexpr uint64_t C_ADDR = 0x300000;
static constexpr uint64_t KERNARG_ADDR = 0x400000;

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;

enum class ScalingMode {
  DispatchPool,
  Partitioned,
};

struct Options {
  ScalingMode mode = ScalingMode::DispatchPool;
  uint32_t runs = 1;
  uint32_t partitions = 0;
  uint32_t dispatch_threads = 0;
};

uint32_t parse_u32_arg(const char *arg, const char *prefix) {
  return static_cast<uint32_t>(std::strtoul(arg + std::strlen(prefix), nullptr, 10));
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode=partitioned") == 0) {
      options.mode = ScalingMode::Partitioned;
    } else if (std::strcmp(argv[i], "--mode=dispatch-pool") == 0) {
      options.mode = ScalingMode::DispatchPool;
    } else if (std::strncmp(argv[i], "--runs=", 7) == 0) {
      options.runs = std::max(parse_u32_arg(argv[i], "--runs="), 1u);
    } else if (std::strncmp(argv[i], "--partitions=", 13) == 0) {
      options.partitions = std::clamp(parse_u32_arg(argv[i], "--partitions="), 1u, TOTAL_XCDS);
    } else if (std::strncmp(argv[i], "--dispatch-threads=", 19) == 0) {
      options.dispatch_threads = std::max(parse_u32_arg(argv[i], "--dispatch-threads="), 1u);
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      std::exit(2);
    }
  }
  return options;
}

KD read_kd(const CodeObject &co) {
  for (const auto *sec : co.rodata_sections())
    if (sec->size() >= sizeof(KD)) {
      KD kd;
      std::memcpy(&kd, sec->data(), sizeof(kd));
      return kd;
    }
  return {};
}

double run_kernel(const char *kernel_name, uint32_t N, uint32_t total_wgs, uint32_t partitions,
                  uint32_t dispatch_threads, ScalingMode mode) {
  Executable exec(kernel_path(kernel_name));
  if (!exec.is_valid())
    return -1;
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  if (!co)
    return -1;
  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  uint32_t engine_threads = mode == ScalingMode::Partitioned ? partitions : 1;
  loaded.engine_config.num_threads = engine_threads;
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  if (mode == ScalingMode::Partitioned)
    amdgpu::partition_topology_by_xcds(engine->topology(), soc, partitions);
  soc->set_dispatch_threads(dispatch_threads, engine_threads);
  engine->build();

  co->load_to_memory(memory, KD_ADDR);
  uint64_t kernel_object = KD_ADDR + co->kernel_descriptor_offset(kernel_name);
  if (kernel_object == KD_ADDR)
    return -1;

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

  memory->load_image(reinterpret_cast<const uint8_t *>(A.data()), data_bytes, A_ADDR);
  memory->load_image(reinterpret_cast<const uint8_t *>(B.data()), data_bytes, B_ADDR);
  std::vector<float> zeros(elems, 0.0f);
  memory->load_image(reinterpret_cast<const uint8_t *>(zeros.data()), data_bytes, C_ADDR);

  uint32_t kernarg_N = is_vector_add ? static_cast<uint32_t>(elems) : N;
  struct {
    uint64_t A, B, C;
    uint32_t N;
  } args = {A_ADDR, B_ADDR, C_ADDR, kernarg_N};
  memory->load_image(reinterpret_cast<const uint8_t *>(&args), sizeof(args), KERNARG_ADDR);

  // Dispatch across all XCDs via AQL queues.
  uint32_t wgs_per_xcd = total_wgs / TOTAL_XCDS;
  for (uint32_t xi = 0; xi < TOTAL_XCDS; ++xi) {
    auto *cp = soc->xcd(xi)->command_processor();
    cp->set_workgroup_id_offset(xi * wgs_per_xcd);
    uint64_t ring = 0xF0000000ULL + xi * 0x100000ULL;
    test::AqlQueue queue(memory, cp, ring, 4096, ring + 0x10000, ring + 0x10008, ring + 0x10010);
    queue.dispatch(kernel_object, wgs_per_xcd * WF_SIZE, WF_SIZE, KERNARG_ADDR);
  }

  // Time the simulation.
  auto start = std::chrono::steady_clock::now();
  engine->run();
  soc->flush_all();
  auto end = std::chrono::steady_clock::now();

  return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char **argv) {
  Options options = parse_options(argc, argv);
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

  if (options.mode == ScalingMode::Partitioned)
    std::cout << "partitions,dispatch_threads";
  else
    std::cout << "threads";
  for (auto &k : kernels)
    std::cout << "," << k.name;
  std::cout << "\n";

  uint32_t begin = options.partitions ? options.partitions : 1;
  uint32_t end = options.partitions ? options.partitions : TOTAL_XCDS;
  for (uint32_t t = begin; t <= end; ++t) {
    uint32_t partitions = options.mode == ScalingMode::Partitioned ? t : 1;
    uint32_t dispatch_threads =
        options.dispatch_threads
            ? options.dispatch_threads
            : (options.mode == ScalingMode::Partitioned ? DEFAULT_CPU_DISPATCH_THREADS : t);
    if (options.mode == ScalingMode::Partitioned)
      std::cout << partitions << "," << dispatch_threads;
    else
      std::cout << t;
    for (auto &k : kernels) {
      // Take the median of the requested runs.
      std::vector<double> times;
      for (uint32_t r = 0; r < options.runs; ++r) {
        double ms =
            run_kernel(k.name, k.N, k.total_wgs, partitions, dispatch_threads, options.mode);
        times.push_back(ms);
      }
      std::sort(times.begin(), times.end());
      std::cout << "," << times[times.size() / 2];
    }
    std::cout << "\n";
    std::cout.flush();
  }
  return 0;
}

#else
int main() {
  std::cerr << "Device kernels not available. Build with HAS_DEVICE_KERNELS.\n";
  return 1;
}
#endif
