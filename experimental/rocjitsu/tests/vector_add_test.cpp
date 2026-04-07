// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file Multi-XCD vector addition stress test with golden reference validation.
///
/// Loads a compiled vector_add.hip kernel and dispatches 256 workgroups across
/// all 8 XCDs (CDNA4 topology), one workgroup per CU. Each wavefront of 64
/// threads computes C[gid] = A[gid] + B[gid]. Results are compared against a
/// CPU golden reference.

#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/kernel_metadata.h"
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

/// Write a kernel argument to the kernarg segment.
/// Returns false if the argument is not found in the map.
bool write_kernarg(amdgpu::GpuMemory *memory, uint64_t kernarg_addr,
                   const KernelArgMap &args, const std::string &name,
                   uint32_t value) {
  auto it = args.find(name);
  if (it == args.end()) {
    return false;
  }
  uint64_t addr = kernarg_addr + it->second.offset;
  if (it->second.size == 2) {
    // uint16: read-modify-write to preserve adjacent half.
    uint64_t aligned = addr & ~3ULL;
    uint32_t dword = memory->read32(aligned);
    uint32_t shift = (addr & 2) * 8;
    dword &= ~(0xFFFFu << shift);
    dword |= (value & 0xFFFF) << shift;
    memory->write32(aligned, dword);
  } else {
    memory->write32(addr, value);
  }
  return true;
}

/// Write HIP implicit kernel arguments using offsets from the code object.
/// Returns false if any required implicit argument is missing.
bool write_implicit_args(amdgpu::GpuMemory *memory, uint64_t kernarg_addr,
                         const KernelArgMap &args,
                         uint32_t block_count_x, uint32_t group_size_x) {
  bool ok = true;
  ok &= write_kernarg(memory, kernarg_addr, args, "hidden_block_count_x", block_count_x);
  ok &= write_kernarg(memory, kernarg_addr, args, "hidden_block_count_y", 1);
  ok &= write_kernarg(memory, kernarg_addr, args, "hidden_block_count_z", 1);
  ok &= write_kernarg(memory, kernarg_addr, args, "hidden_group_size_x", group_size_x);
  ok &= write_kernarg(memory, kernarg_addr, args, "hidden_group_size_y", 1);
  ok &= write_kernarg(memory, kernarg_addr, args, "hidden_group_size_z", 1);
  return ok;
}

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

  // Write kernel arguments: (A*, B*, C*, N) + implicit args.
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
                                                           //
  auto arg_meta = parseKernelArgs(co->image_data(), co->size());
  ASSERT_TRUE(arg_meta.has_value()) << "Failed to parse kernel arg metadata";
  ASSERT_TRUE(write_implicit_args(memory, KERNARG_ADDR, *arg_meta, TOTAL_CUS, WF_SIZE));

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
  auto arg_meta = parseKernelArgs(co->image_data(), co->size());
  ASSERT_TRUE(arg_meta.has_value()) << "Failed to parse kernel arg metadata";
  ASSERT_TRUE(write_implicit_args(memory, KERNARG_ADDR, *arg_meta, TOTAL_CUS, WF_SIZE));

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

// 2 wavefronts per workgroup: verifies that workitem IDs are correctly
// offset per wavefront (v0 = wf_index * wf_size + lane).
TEST(VectorAddStressTest, TwoWavefrontsPerWorkgroup) {
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
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  engine->build();

  const auto *text = co->text_sections()[0];
  memory->load_image(reinterpret_cast<const uint8_t *>(text->data()), text->size(), CODE_ADDR);

  constexpr uint32_t WFS_PER_WG = 2;
  constexpr uint32_t THREADS_PER_WG = WFS_PER_WG * WF_SIZE;
  constexpr uint32_t TOTAL_N = TOTAL_CUS * THREADS_PER_WG;

  size_t vec_bytes = TOTAL_N * sizeof(float);
  std::vector<float> A(TOTAL_N), B(TOTAL_N), C_expected(TOTAL_N);
  for (uint32_t i = 0; i < TOTAL_N; ++i) {
    A[i] = static_cast<float>(i % 97) * 0.1f;
    B[i] = static_cast<float>(i % 61) * 0.2f;
    C_expected[i] = A[i] + B[i];
  }

  memory->load_image(reinterpret_cast<const uint8_t *>(A.data()), vec_bytes, A_ADDR);
  memory->load_image(reinterpret_cast<const uint8_t *>(B.data()), vec_bytes, B_ADDR);
  std::vector<float> zeros(TOTAL_N, 0.0f);
  memory->load_image(reinterpret_cast<const uint8_t *>(zeros.data()), vec_bytes, C_ADDR);

  struct {
    uint64_t A, B, C;
    uint32_t N;
  } args = {A_ADDR, B_ADDR, C_ADDR, TOTAL_N};
  memory->load_image(reinterpret_cast<const uint8_t *>(&args), sizeof(args), KERNARG_ADDR);
  // Implicit args start after explicit args (28 bytes, padded to 32).
  auto arg_meta = parseKernelArgs(co->image_data(), co->size());
  ASSERT_TRUE(arg_meta.has_value()) << "Failed to parse kernel arg metadata";
  ASSERT_TRUE(write_implicit_args(memory, KERNARG_ADDR, *arg_meta, TOTAL_CUS, THREADS_PER_WG));

  uint32_t wgs_per_xcd = TOTAL_CUS / TOTAL_XCDS;
  amdgpu::DispatchPacket pkt;
  pkt.kernel_entry_pc = CODE_ADDR;
  pkt.wfs_per_workgroup = WFS_PER_WG;
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
  for (uint32_t i = 0; i < TOTAL_N; ++i) {
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

TEST(VectorAddCodeObjectTest, KernelMetadataContainsImplicitArgs) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto args = parseKernelArgs(co->image_data(), co->size());
  ASSERT_TRUE(args.has_value()) << "Failed to parse kernel metadata";

  // Explicit args: 3 global buffers (A, B, C) and 1 scalar (N).
  EXPECT_TRUE(args->count("global_buffer"));
  EXPECT_TRUE(args->count("by_value"));

  // Implicit args needed for blockDim.x / gridDim.x.
  ASSERT_TRUE(args->count("hidden_block_count_x"));
  ASSERT_TRUE(args->count("hidden_group_size_x"));
  EXPECT_EQ(args->at("hidden_block_count_x").size, 4u);
  EXPECT_EQ(args->at("hidden_group_size_x").size, 2u);

  // group_size offset must come after the explicit args.
  EXPECT_GT(args->at("hidden_group_size_x").offset, args->at("by_value").offset);
}

} // namespace

#endif // HAS_DEVICE_KERNELS
