// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aql_queue.h"

#include "embedded_schema.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

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

const std::string CONFIG_PATH = std::string(CONFIG_DIR) + "/amdgpu_cdna4.json";

std::string kernel_path(const char *name) { return std::string(KERNEL_DIR) + "/" + name + ".o"; }

constexpr uint32_t TOTAL_XCDS = 8;
constexpr uint32_t CUS_PER_XCD = 32; // 4 SEs × 8 CUs
constexpr uint32_t TOTAL_CUS = TOTAL_XCDS * CUS_PER_XCD;

// AMDGPU kernel descriptor (HSA code object v3, 64 bytes).

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;

// CPU golden reference.

void cpu_matmul(const float *A, const float *B, float *C, unsigned N) {
  for (unsigned i = 0; i < N; ++i)
    for (unsigned j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (unsigned k = 0; k < N; ++k)
        sum += A[i * N + k] * B[k * N + j];
      C[i * N + j] = sum;
    }
}

// Helpers.

std::vector<std::unique_ptr<Instruction>> decode_all(const CodeObject &co) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<std::unique_ptr<Instruction>> insts;

  for (const auto *sec : co.text_sections()) {
    const auto *data = reinterpret_cast<const uint32_t *>(sec->data());
    size_t words = sec->size() / sizeof(uint32_t);
    size_t pc = 0;
    while (pc < words) {
      std::unique_ptr<Instruction> inst(decoder->decode(&data[pc]));
      EXPECT_NE(inst, nullptr);
      ++pc;
      if (inst && inst->size() == 8)
        ++pc;
      if (inst)
        insts.push_back(std::move(inst));
    }
  }
  return insts;
}

KD read_kernel_descriptor(const CodeObject &co) {
  for (const auto *sec : co.rodata_sections())
    if (sec->size() >= sizeof(KD)) {
      KD kd;
      std::memcpy(&kd, sec->data(), sizeof(kd));
      return kd;
    }
  ADD_FAILURE() << "No .rodata section with kernel descriptor found";
  return {};
}

// Memory layout.
constexpr uint64_t KD_ADDR = 0x10000;
constexpr uint64_t A_ADDR = 0x100000;
constexpr uint64_t B_ADDR = 0x200000;
constexpr uint64_t C_ADDR = 0x300000;
constexpr uint64_t KERNARG_ADDR = 0x400000;

/// Fixture that loads a device kernel, places it and matrix data in simulator
/// GPU memory, and dispatches wavefronts via the engine-driven path.
struct KernelExecFixture {
  std::unique_ptr<simdojo::SimulationEngine> engine;
  SoC *soc = nullptr;
  amdgpu::GpuMemory *gpu_mem = nullptr;
  uint64_t kernel_object = 0;

  void setup(const char *kernel_name, uint32_t num_threads = 1) {
    auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
    soc = loaded.soc();
    gpu_mem = loaded.memory();
    loaded.engine_config.num_threads = num_threads;
    engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine->topology().set_root(loaded.take_root());
    loaded.wire_links(engine->topology());
    if (num_threads > 1)
      partition_by_xcd(num_threads);
    else
      engine->build();

    Executable exec(kernel_path(kernel_name));
    ASSERT_TRUE(exec.is_valid());
    ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
    auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
    ASSERT_NE(co, nullptr);

    // Load code object to GPU memory respecting ELF segment VA mapping.
    co->load_to_memory(mem(), KD_ADDR);
    kernel_object = KD_ADDR + co->kernel_descriptor_offset(kernel_name);
    ASSERT_NE(kernel_object, KD_ADDR) << "Kernel descriptor symbol not found";
  }

  /// Partition so that each XCD's components stay in the same partition.
  /// Components not under any XCD (SoC, VM, IODs) go to partition 0.
  void partition_by_xcd(uint32_t num_partitions) {
    // Build a map from Xcd pointer to partition ID.
    std::unordered_map<simdojo::Component *, simdojo::PartitionID> xcd_map;
    for (uint32_t i = 0; i < soc->num_xcds(); ++i)
      xcd_map[soc->xcd(i)] = i % num_partitions;

    engine->topology().partition_manual(
        num_partitions, [&](simdojo::Component *c) -> simdojo::PartitionID {
          // Walk up the parent chain looking for an XCD ancestor.
          for (auto *p = static_cast<simdojo::Component *>(c); p != nullptr;
               p = static_cast<simdojo::Component *>(p->parent())) {
            auto it = xcd_map.find(p);
            if (it != xcd_map.end())
              return it->second;
          }
          return 0; // Top-level components → partition 0.
        });
    engine->build();
  }

  amdgpu::GpuMemory *mem() { return gpu_mem; }
  amdgpu::CommandProcessor *cp(uint32_t xcd = 0) { return soc->xcd(xcd)->command_processor(); }

  void load_matrices(const float *A, const float *B, unsigned N) {
    size_t mat_bytes = static_cast<size_t>(N) * N * sizeof(float);
    mem()->load_image(reinterpret_cast<const uint8_t *>(A), mat_bytes, A_ADDR);
    mem()->load_image(reinterpret_cast<const uint8_t *>(B), mat_bytes, B_ADDR);
    std::vector<float> zeros(static_cast<size_t>(N) * N, 0.0f);
    mem()->load_image(reinterpret_cast<const uint8_t *>(zeros.data()), mat_bytes, C_ADDR);
  }

  void write_kernargs(unsigned N) {
    struct {
      uint64_t A, B, C;
      uint32_t N;
    } args = {A_ADDR, B_ADDR, C_ADDR, N};
    mem()->load_image(reinterpret_cast<const uint8_t *>(&args), sizeof(args), KERNARG_ADDR);
  }

  /// Dispatch workgroups across all XCDs via AQL queues.
  /// @param total_wgs Total number of workgroups to dispatch.
  void dispatch_all_xcds(uint32_t total_wgs) {
    uint32_t num_xcds = soc->num_xcds();
    uint32_t wgs_per_xcd = total_wgs / num_xcds;
    ASSERT_EQ(wgs_per_xcd * num_xcds, total_wgs) << "WGs must divide evenly across XCDs";

    for (uint32_t xi = 0; xi < num_xcds; ++xi) {
      cp(xi)->set_workgroup_id_offset(xi * wgs_per_xcd);
      uint64_t ring = 0xF0000000ULL + xi * 0x100000ULL;
      test::AqlQueue queue(mem(), cp(xi), ring, 4096, ring + 0x10000, ring + 0x10008,
                           ring + 0x10010);
      queue.dispatch(kernel_object, wgs_per_xcd * 64, 64, KERNARG_ADDR);
    }
  }

  /// Run the simulation engine to completion and flush all caches.
  void run() {
    engine->run();
    soc->flush_all();
  }

  /// Read back an NxN output matrix from GPU memory.
  void read_result(float *C_out, unsigned N) {
    for (unsigned i = 0; i < N * N; ++i)
      C_out[i] = std::bit_cast<float>(mem()->read32(C_ADDR + i * sizeof(float)));
  }
};

TEST(MatmulCodeObjectTest, NaiveLoadsAndDecodes) {
  Executable exec(kernel_path("matmul_naive"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load matmul_naive.o";
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);
  ASSERT_FALSE(co->text_sections().empty()) << "No .text sections in naive kernel";

  auto insts = decode_all(*co);
  EXPECT_GT(insts.size(), 0u);
}

TEST(MatmulCodeObjectTest, MfmaLoadsAndDecodes) {
  Executable exec(kernel_path("matmul_mfma"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load matmul_mfma.o";
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);
  ASSERT_FALSE(co->text_sections().empty()) << "No .text sections in MFMA kernel";

  auto insts = decode_all(*co);
  EXPECT_GT(insts.size(), 0u);

  bool has_mfma = false;
  for (const auto &inst : insts)
    if (inst->mnemonic().starts_with("v_mfma_")) {
      has_mfma = true;
      break;
    }
  EXPECT_TRUE(has_mfma) << "MFMA kernel does not contain any v_mfma instructions";
}

TEST(MatmulCodeObjectTest, NaiveUsesVALU) {
  Executable exec(kernel_path("matmul_naive"));
  ASSERT_TRUE(exec.is_valid());
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto insts = decode_all(*co);
  bool has_vmul = false;
  bool has_vadd = false;
  for (const auto &inst : insts) {
    const auto &m = inst->mnemonic();
    if (m.starts_with("v_mul_") || m.starts_with("v_fmac_") || m.starts_with("v_fma_"))
      has_vmul = true;
    if (m.starts_with("v_add_"))
      has_vadd = true;
  }
  EXPECT_TRUE(has_vmul) << "Naive kernel should contain vector multiply/FMA";
  EXPECT_TRUE(has_vadd) << "Naive kernel should contain vector add";
}

TEST(MatmulCodeObjectTest, MfmaHasMultipleMfmaInstructions) {
  Executable exec(kernel_path("matmul_mfma"));
  ASSERT_TRUE(exec.is_valid());
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto insts = decode_all(*co);
  size_t mfma_count = 0;
  for (const auto &inst : insts)
    if (inst->mnemonic().starts_with("v_mfma_"))
      ++mfma_count;
  EXPECT_GE(mfma_count, 1u) << "Expected at least 1 MFMA instruction";
}

TEST(MatmulCodeObjectTest, KernelDescriptorParsesCorrectly) {
  Executable exec(kernel_path("matmul_naive"));
  ASSERT_TRUE(exec.is_valid());
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto kd = read_kernel_descriptor(*co);
  EXPECT_GT(kd.kernel_code_entry_byte_offset, 0) << "Entry offset should be positive";
  EXPECT_EQ(kd.kernarg_size, 28u) << "naive kernel takes (float*,float*,float*,uint)";
  EXPECT_GE((kd.compute_pgm_rsrc2 >> 1) & 0x1F, 2u) << "Need at least 2 user SGPRs for kernarg ptr";
}

TEST(MatmulCodeObjectTest, TiledLoadsAndDecodes) {
  Executable exec(kernel_path("matmul_tiled"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load matmul_tiled.o";
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);
  ASSERT_FALSE(co->text_sections().empty()) << "No .text sections in tiled kernel";

  auto insts = decode_all(*co);
  EXPECT_GT(insts.size(), 0u);
}

TEST(MatmulCodeObjectTest, TiledKernelDescriptor) {
  Executable exec(kernel_path("matmul_tiled"));
  ASSERT_TRUE(exec.is_valid());
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto kd = read_kernel_descriptor(*co);
  EXPECT_GT(kd.kernel_code_entry_byte_offset, 0) << "Entry offset should be positive";
  EXPECT_EQ(kd.kernarg_size, 28u) << "tiled kernel takes (float*,float*,float*,uint)";
  EXPECT_GE((kd.compute_pgm_rsrc2 >> 1) & 0x1F, 2u) << "Need at least 2 user SGPRs for kernarg ptr";
}

// Each test dispatches one workgroup per CU across all 8 XCDs (256 CUs total).

constexpr unsigned STRESS_N = 128;

/// Helper to generate input matrices, compute CPU reference, run GPU, and compare.
void run_matmul_stress(const char *kernel_name, unsigned N, uint32_t num_threads = 1) {
  size_t mat_elems = static_cast<size_t>(N) * N;

  // For MFMA with 4x4 tiles: total_wgs = (N/4)^2.
  // For tiled with 64-element blocks: total_wgs = ceil(N*N / 64).
  // With N=128: tiled = 256 WGs, MFMA (4x4 tiles) = 1024 WGs.
  // With N=64:  tiled = 64 WGs, MFMA (4x4 tiles) = 256 WGs.
  // We pick N so that the number of WGs equals TOTAL_CUS for exact 1:1 mapping.

  // Generate input matrices.
  std::vector<float> A(mat_elems);
  std::vector<float> B(mat_elems);
  for (size_t i = 0; i < mat_elems; ++i) {
    A[i] = static_cast<float>(i % 17) * 0.1f;
    B[i] = static_cast<float>(i % 13) * 0.1f;
  }

  // CPU golden reference.
  std::vector<float> C_expected(mat_elems);
  cpu_matmul(A.data(), B.data(), C_expected.data(), N);

  KernelExecFixture fx;
  fx.setup(kernel_name, num_threads);
  fx.load_matrices(A.data(), B.data(), N);
  fx.write_kernargs(N);

  fx.dispatch_all_xcds(TOTAL_CUS);
  fx.run();

  // Read back results and compare against golden reference.
  std::vector<float> C_gpu(mat_elems);
  fx.read_result(C_gpu.data(), N);

  unsigned mismatches = 0;
  for (size_t i = 0; i < mat_elems; ++i) {
    if (std::abs(C_gpu[i] - C_expected[i]) > 1e-3f * std::abs(C_expected[i]) + 1e-6f) {
      if (mismatches < 10) {
        unsigned row = static_cast<unsigned>(i) / N;
        unsigned col = static_cast<unsigned>(i) % N;
        ADD_FAILURE() << "Mismatch at C[" << row << "][" << col << "]: GPU=" << C_gpu[i]
                      << " CPU=" << C_expected[i];
      }
      ++mismatches;
    }
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " elements differ (showing first 10)";
}

// Tiled (VALU) stress test: N=128, 256 WGs of 64 threads each.
// global_id = blockIdx.x * 64 + threadIdx.x → covers all 128*128 = 16384 elements.
TEST(MatmulStressTest, TiledAllCUs) { run_matmul_stress("matmul_tiled", STRESS_N); }

// MFMA stress test: N=64, 256 WGs of 4x4 MFMA tiles.
// (64/4)^2 = 16*16 = 256 tiles, one per CU.
constexpr unsigned MFMA_N = 64;
TEST(MatmulStressTest, MfmaAllCUs) { run_matmul_stress("matmul_mfma", MFMA_N); }

// Multi-threaded versions: one worker thread per XCD (8 threads).
// Exercises the barrier-based LBTS protocol with real GPU kernel execution.
// Multi-threaded: exercises barrier-based LBTS with real GPU kernel execution.
TEST(MatmulStressTest, TiledAllCUs_MultiThreaded) {
  run_matmul_stress("matmul_tiled", STRESS_N, TOTAL_XCDS);
}

TEST(MatmulStressTest, MfmaAllCUs_MultiThreaded) {
  run_matmul_stress("matmul_mfma", MFMA_N, TOTAL_XCDS);
}

// Topology-only stress test: verify that the CDNA4 topology builds correctly
// and all 256 CUs can dispatch and halt wavefronts (engine-driven, no kernel execution).

TEST(MatmulStressTest, Cdna4TopologyDispatchAndHalt) {
  constexpr uint32_t total_wgs = TOTAL_CUS;
  constexpr uint32_t SOPP_S_ENDPGM = 0xBF810000;

  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  engine->build();

  // Write a kernel descriptor + s_endpgm to GPU memory.
  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd{};
  kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  ((256 / 8) - 1)); // CDNA4 VGPR granularity is 8
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  ((104 / 8) - 1));
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);

  memory->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), KD_ADDR);
  memory->write32(KD_ADDR + sizeof(kernel_descriptor_t), SOPP_S_ENDPGM);

  // Dispatch workgroups across all XCDs via AQL queues.
  uint32_t wgs_per_xcd = total_wgs / TOTAL_XCDS;
  for (uint32_t xi = 0; xi < TOTAL_XCDS; ++xi) {
    auto *cp = soc->xcd(xi)->command_processor();
    cp->set_workgroup_id_offset(xi * wgs_per_xcd);
    uint64_t ring = 0xF0000000ULL + xi * 0x100000ULL;
    test::AqlQueue queue(memory, cp, ring, 4096, ring + 0x10000, ring + 0x10008, ring + 0x10010);
    queue.dispatch(KD_ADDR, wgs_per_xcd * 64, 64);
  }

  engine->run();

  // Verify all wavefronts halted.
  for (uint32_t xi = 0; xi < TOTAL_XCDS; ++xi) {
    auto *xcd = soc->xcd(xi);
    for (uint32_t si = 0; si < xcd->num_shader_engines(); ++si) {
      auto *se = xcd->shader_engine(si);
      for (uint32_t ci = 0; ci < se->num_compute_units(); ++ci) {
        auto *unit = se->compute_unit(ci);
        EXPECT_FALSE(unit->has_active_wfs()) << unit->name() << " still active";
      }
    }
  }
}

// Multi-threaded topology-only: 1 thread per XCD, dispatch s_endpgm to all CUs.
TEST(MatmulStressTest, Cdna4TopologyDispatchAndHalt_MultiThreaded) {
  constexpr uint32_t total_wgs = TOTAL_CUS;
  constexpr uint32_t SOPP_S_ENDPGM = 0xBF810000;

  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
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

  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd{};
  kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  ((256 / 8) - 1)); // CDNA4 VGPR granularity is 8
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  ((104 / 8) - 1));
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);

  memory->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), KD_ADDR);
  memory->write32(KD_ADDR + sizeof(kernel_descriptor_t), SOPP_S_ENDPGM);

  uint32_t wgs_per_xcd = total_wgs / TOTAL_XCDS;
  for (uint32_t xi = 0; xi < TOTAL_XCDS; ++xi) {
    auto *cp = soc->xcd(xi)->command_processor();
    cp->set_workgroup_id_offset(xi * wgs_per_xcd);
    uint64_t ring = 0xF0000000ULL + xi * 0x100000ULL;
    test::AqlQueue queue(memory, cp, ring, 4096, ring + 0x10000, ring + 0x10008, ring + 0x10010);
    queue.dispatch(KD_ADDR, wgs_per_xcd * 64, 64);
  }

  engine->run();

  for (uint32_t xi = 0; xi < TOTAL_XCDS; ++xi) {
    auto *xcd = soc->xcd(xi);
    for (uint32_t si = 0; si < xcd->num_shader_engines(); ++si) {
      auto *se = xcd->shader_engine(si);
      for (uint32_t ci = 0; ci < se->num_compute_units(); ++ci) {
        auto *unit = se->compute_unit(ci);
        EXPECT_FALSE(unit->has_active_wfs()) << unit->name() << " still active";
      }
    }
  }
}

} // namespace

#endif // HAS_DEVICE_KERNELS
