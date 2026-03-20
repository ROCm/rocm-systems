// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Scaling test: measures simulation wall-clock time for vector_add, matmul_tiled,
// and matmul_mfma across 1..8 threads (one per XCD). Outputs CSV to stdout.

#include "rocjitsu/code/executable.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef HAS_DEVICE_KERNELS

using namespace rocjitsu;

static const std::string kSchemaPath = std::string(SCHEMA_DIR) + "/simulation_config.fbs";
static const std::string kConfigPath = std::string(CONFIG_DIR) + "/amdgpu_cdna4.json";
static std::string kernel_path(const char *name) {
  return std::string(KERNEL_DIR) + "/" + name + ".o";
}

static constexpr uint32_t TOTAL_XCDS = 8;
static constexpr uint32_t CUS_PER_XCD = 32;
static constexpr uint32_t TOTAL_CUS = TOTAL_XCDS * CUS_PER_XCD;
static constexpr uint32_t WF_SIZE = 64;

static constexpr uint64_t CODE_ADDR = 0x1000;
static constexpr uint64_t A_ADDR = 0x100000;
static constexpr uint64_t B_ADDR = 0x200000;
static constexpr uint64_t C_ADDR = 0x300000;
static constexpr uint64_t KERNARG_ADDR = 0x400000;

struct KernelDescriptor {
  uint32_t group_segment_fixed_size;
  uint32_t private_segment_fixed_size;
  uint32_t kernarg_size;
  uint8_t reserved0[4];
  int64_t kernel_code_entry_byte_offset;
  uint8_t reserved1[20];
  uint32_t compute_pgm_rsrc3;
  uint32_t compute_pgm_rsrc1;
  uint32_t compute_pgm_rsrc2;
  uint16_t kernel_code_properties;
  uint16_t kernarg_preload_spec;
  uint8_t reserved2[4];
};

KernelDescriptor read_kd(const CodeObject &co) {
  for (const auto *sec : co.rodata_sections())
    if (sec->size() >= sizeof(KernelDescriptor)) {
      KernelDescriptor kd;
      std::memcpy(&kd, sec->data(), sizeof(kd));
      return kd;
    }
  return {};
}

double run_kernel(const char *kernel_name, uint32_t N, uint32_t num_threads) {
  Executable exec(kernel_path(kernel_name));
  if (!exec.is_valid())
    return -1;
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  if (!co)
    return -1;
  auto kd = read_kd(*co);

  auto loaded = config::load_config(kConfigPath, kSchemaPath);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  loaded.engine_config.num_threads = num_threads;
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());

  if (num_threads > 1) {
    std::unordered_map<simdojo::Component *, simdojo::PartitionID> xcd_map;
    for (uint32_t i = 0; i < soc->num_xcds(); ++i)
      xcd_map[soc->xcd(i)] = i % num_threads;
    engine->topology().partition_manual(
        num_threads, [&](simdojo::Component *c) -> simdojo::PartitionID {
          for (auto *p = static_cast<simdojo::Component *>(c); p != nullptr;
               p = static_cast<simdojo::Component *>(p->parent())) {
            auto it = xcd_map.find(p);
            if (it != xcd_map.end())
              return it->second;
          }
          return 0;
        });
  }
  engine->build();

  const auto *text = co->text_sections()[0];
  memory->load_image(reinterpret_cast<const uint8_t *>(text->data()), text->size(), CODE_ADDR);

  // Setup data.
  size_t elems = static_cast<size_t>(N) * N;
  bool is_vector_add = (std::string(kernel_name) == "vector_add");
  if (is_vector_add)
    elems = TOTAL_CUS * WF_SIZE;

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

  // Dispatch across all XCDs.
  uint32_t wgs_per_xcd = TOTAL_CUS / TOTAL_XCDS;
  amdgpu::DispatchPacket pkt;
  pkt.kernel_entry_pc = CODE_ADDR;
  pkt.wfs_per_workgroup = 1;
  pkt.sgprs_per_wf = 104;
  pkt.vgprs_per_wf = 256;
  pkt.kernarg_addr = KERNARG_ADDR;
  pkt.num_user_sgprs = (kd.compute_pgm_rsrc2 >> 1) & 0x1F;

  for (uint32_t xi = 0; xi < TOTAL_XCDS; ++xi) {
    pkt.workgroup_count = wgs_per_xcd;
    pkt.workgroup_id_offset = xi * wgs_per_xcd;
    soc->xcd(xi)->command_processor()->enqueue(pkt);
  }

  // Time the simulation.
  auto start = std::chrono::steady_clock::now();
  engine->run();
  soc->flush_all();
  auto end = std::chrono::steady_clock::now();

  return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
  struct Kernel {
    const char *name;
    uint32_t N;
  };
  Kernel kernels[] = {
      {"vector_add", 0}, // N unused for vector_add
      {"matmul_tiled", 128},
      {"matmul_mfma", 64},
  };

  std::cout << "threads";
  for (auto &k : kernels)
    std::cout << "," << k.name;
  std::cout << "\n";

  constexpr int RUNS = 3;

  for (uint32_t t = 1; t <= TOTAL_XCDS; ++t) {
    std::cout << t;
    for (auto &k : kernels) {
      // Take the median of RUNS.
      std::vector<double> times;
      for (int r = 0; r < RUNS; ++r) {
        double ms = run_kernel(k.name, k.N, t);
        times.push_back(ms);
      }
      std::sort(times.begin(), times.end());
      std::cout << "," << times[RUNS / 2];
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
