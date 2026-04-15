// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file Multi-XCD vector addition stress test with golden reference validation.
///
/// Loads a compiled vector_add.hip kernel and dispatches 256 workgroups across
/// all 8 XCDs (CDNA4 topology), one workgroup per CU. Each wavefront of 64
/// threads computes C[gid] = A[gid] + B[gid]. Results are compared against a
/// CPU golden reference.

#include "rocjitsu/code/executable.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef HAS_DEVICE_KERNELS

namespace {

using namespace rocjitsu;

const std::string kSchemaPath = std::string(SCHEMA_DIR) + "/simulation_config.fbs";
const std::string kConfigPath = std::string(CONFIG_DIR) + "/amdgpu_cdna4.json";

std::string kernel_path(const char *name) { return std::string(KERNEL_DIR) + "/" + name + ".o"; }

constexpr uint32_t TOTAL_XCDS = 8;
constexpr uint32_t CUS_PER_XCD = 32; // 4 SEs x 8 CUs
constexpr uint32_t TOTAL_CUS = TOTAL_XCDS * CUS_PER_XCD;
constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t N = TOTAL_CUS * WF_SIZE; // 16384 elements, one WG per CU

constexpr uint64_t CODE_ADDR = 0x1000;
constexpr uint64_t A_ADDR = 0x100000;
constexpr uint64_t B_ADDR = 0x200000;
constexpr uint64_t C_ADDR = 0x300000;
constexpr uint64_t KERNARG_ADDR = 0x400000;

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
static_assert(sizeof(KernelDescriptor) == 64);

KernelDescriptor read_kernel_descriptor(const CodeObject &co) {
  for (const auto *sec : co.rodata_sections())
    if (sec->size() >= sizeof(KernelDescriptor)) {
      KernelDescriptor kd;
      std::memcpy(&kd, sec->data(), sizeof(kd));
      return kd;
    }
  ADD_FAILURE() << "No .rodata section with kernel descriptor found";
  return {};
}

TEST(VectorAddStressTest, AllCUsGoldenReference) {
  // Load the compiled vector_add kernel.
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load vector_add.o";
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto kd = read_kernel_descriptor(*co);
  ASSERT_NE(kd.kernel_code_entry_byte_offset, 0);

  // Build the simulation engine with CDNA4 topology.
  auto loaded = config::load_config(kConfigPath, kSchemaPath);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  engine->build();

  // Load kernel code into GPU memory.
  const auto *text = co->text_sections()[0];
  memory->load_image(reinterpret_cast<const uint8_t *>(text->data()), text->size(), CODE_ADDR);

  // Generate input vectors.
  size_t vec_bytes = N * sizeof(float);
  std::vector<float> A(N), B(N), C_expected(N);
  for (uint32_t i = 0; i < N; ++i) {
    A[i] = static_cast<float>(i % 97) * 0.1f;
    B[i] = static_cast<float>(i % 61) * 0.2f;
    C_expected[i] = A[i] + B[i];
  }

  memory->load_image(reinterpret_cast<const uint8_t *>(A.data()), vec_bytes, A_ADDR);
  memory->load_image(reinterpret_cast<const uint8_t *>(B.data()), vec_bytes, B_ADDR);
  std::vector<float> zeros(N, 0.0f);
  memory->load_image(reinterpret_cast<const uint8_t *>(zeros.data()), vec_bytes, C_ADDR);

  // Write kernel arguments: (A*, B*, C*, N).
  struct {
    uint64_t A, B, C;
    uint32_t N;
  } args = {A_ADDR, B_ADDR, C_ADDR, N};
  memory->load_image(reinterpret_cast<const uint8_t *>(&args), sizeof(args), KERNARG_ADDR);

  // Dispatch across all XCDs.
  uint32_t wgs_per_xcd = TOTAL_CUS / TOTAL_XCDS;
  amdgpu::DispatchPacket pkt;
  pkt.kernel_entry_pc = CODE_ADDR;
  pkt.wfs_per_workgroup = 1;
  pkt.sgprs_per_wf = 104;
  pkt.vgprs_per_wf = 256;
  pkt.kernarg_addr = KERNARG_ADDR;
  pkt.num_user_sgprs = (kd.compute_pgm_rsrc2 >> 1) & 0x1F; // bits[5:1] = USER_SGPR

  for (uint32_t xi = 0; xi < TOTAL_XCDS; ++xi) {
    pkt.workgroup_count = wgs_per_xcd;
    pkt.workgroup_id_offset = xi * wgs_per_xcd;
    soc->xcd(xi)->command_processor()->enqueue(pkt);
  }

  // Engine drives all CPs and CUs to completion.
  engine->run();
  soc->flush_all();

  // Read back results and compare against golden reference.
  unsigned mismatches = 0;
  for (uint32_t i = 0; i < N; ++i) {
    float actual = std::bit_cast<float>(memory->read32(C_ADDR + i * sizeof(float)));
    if (std::abs(actual - C_expected[i]) > 1e-6f) {
      if (mismatches < 10)
        ADD_FAILURE() << "Mismatch at C[" << i << "]: GPU=" << actual << " CPU=" << C_expected[i];
      ++mismatches;
    }
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " elements differ (showing first 10)";
}

// Multi-threaded version: one worker thread per XCD (8 threads).
// Exercises the barrier-based LBTS protocol with real GPU kernel execution.
// Multi-threaded: exercises barrier-based LBTS with real GPU kernel execution.
TEST(VectorAddStressTest, AllCUsGoldenReference_MultiThreaded) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load vector_add.o";
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto kd = read_kernel_descriptor(*co);
  ASSERT_NE(kd.kernel_code_entry_byte_offset, 0);

  auto loaded = config::load_config(kConfigPath, kSchemaPath);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  loaded.engine_config.num_threads = TOTAL_XCDS;
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());

  // Partition by XCD so each XCD's components stay on one thread.
  std::unordered_map<simdojo::Component *, simdojo::PartitionID> xcd_map;
  for (uint32_t i = 0; i < soc->num_xcds(); ++i)
    xcd_map[soc->xcd(i)] = i;
  engine->topology().partition_manual(
      TOTAL_XCDS, [&](simdojo::Component *c) -> simdojo::PartitionID {
        for (auto *p = static_cast<simdojo::Component *>(c); p != nullptr;
             p = static_cast<simdojo::Component *>(p->parent())) {
          auto it = xcd_map.find(p);
          if (it != xcd_map.end())
            return it->second;
        }
        return 0;
      });
  engine->build();

  const auto *text = co->text_sections()[0];
  memory->load_image(reinterpret_cast<const uint8_t *>(text->data()), text->size(), CODE_ADDR);

  size_t vec_bytes = N * sizeof(float);
  std::vector<float> A(N), B(N), C_expected(N);
  for (uint32_t i = 0; i < N; ++i) {
    A[i] = static_cast<float>(i % 97) * 0.1f;
    B[i] = static_cast<float>(i % 61) * 0.2f;
    C_expected[i] = A[i] + B[i];
  }

  memory->load_image(reinterpret_cast<const uint8_t *>(A.data()), vec_bytes, A_ADDR);
  memory->load_image(reinterpret_cast<const uint8_t *>(B.data()), vec_bytes, B_ADDR);
  std::vector<float> zeros(N, 0.0f);
  memory->load_image(reinterpret_cast<const uint8_t *>(zeros.data()), vec_bytes, C_ADDR);

  struct {
    uint64_t A, B, C;
    uint32_t N;
  } args = {A_ADDR, B_ADDR, C_ADDR, N};
  memory->load_image(reinterpret_cast<const uint8_t *>(&args), sizeof(args), KERNARG_ADDR);

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

  engine->run();
  soc->flush_all();

  unsigned mismatches = 0;
  for (uint32_t i = 0; i < N; ++i) {
    float actual = std::bit_cast<float>(memory->read32(C_ADDR + i * sizeof(float)));
    if (std::abs(actual - C_expected[i]) > 1e-6f) {
      if (mismatches < 10)
        ADD_FAILURE() << "Mismatch at C[" << i << "]: GPU=" << actual << " CPU=" << C_expected[i];
      ++mismatches;
    }
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " elements differ (showing first 10)";
}

TEST(VectorAddCodeObjectTest, LoadsAndDecodes) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load vector_add.o";
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);
  ASSERT_FALSE(co->text_sections().empty());

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  const auto *text = co->text_sections()[0];
  const auto *data = reinterpret_cast<const uint32_t *>(text->data());
  auto inst = decoder->decode(data);
  EXPECT_NE(inst, nullptr) << "Failed to decode first instruction";
}

} // namespace

#endif // HAS_DEVICE_KERNELS
