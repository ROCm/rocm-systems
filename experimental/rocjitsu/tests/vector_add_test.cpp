// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file Multi-XCD vector addition stress test with golden reference validation.
///
/// Loads a compiled vector_add.hip kernel and dispatches 256 workgroups across
/// all 8 XCDs (CDNA4 topology), one workgroup per CU. Each wavefront of 64
/// threads computes C[gid] = A[gid] + B[gid]. Results are compared against a
/// CPU golden reference.

#include "aql_queue.h"

#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/kernel_metadata.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP
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

const std::string SCHEMA_PATH = std::string(SCHEMA_DIR) + "/simulation_config.fbs";
const std::string CONFIG_PATH = std::string(CONFIG_DIR) + "/amdgpu_cdna4.json";

std::string kernel_path(const char *name) { return std::string(KERNEL_DIR) + "/" + name + ".o"; }

constexpr uint32_t TOTAL_XCDS = 8;
constexpr uint32_t CUS_PER_XCD = 32; // 4 SEs x 8 CUs
constexpr uint32_t TOTAL_CUS = TOTAL_XCDS * CUS_PER_XCD;
constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t N = TOTAL_CUS * WF_SIZE; // 16384 elements, one WG per CU

constexpr uint64_t KD_ADDR = 0x10000;
constexpr uint64_t A_ADDR = 0x100000;
constexpr uint64_t B_ADDR = 0x200000;
constexpr uint64_t C_ADDR = 0x300000;
constexpr uint64_t KERNARG_ADDR = 0x400000;

/// Write a single hidden kernel argument by value_kind, using the parsed
/// metadata to determine the offset and size.
/// Returns false if the argument is not found in the metadata.
bool write_hidden_arg(amdgpu::GpuMemory *memory, uint64_t kernarg_addr,
                      const KernelArgs &args, const std::string &kind,
                      uint32_t value) {
  auto idx = args.getIndexFromValueKind(kind);
  if (!idx) {
    return false;
  }
  uint64_t addr = kernarg_addr + args.getOffsetFromIndex(*idx);
  uint32_t sz = args.getSizeFromIndex(*idx);
  if (sz == 2) {
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

/// Write implicit dispatch dimensions (blockDim, gridDim) into the kernarg
/// segment, using offsets parsed from the code object metadata.
/// Returns false if any required hidden argument is missing.
bool write_dispatch_dims(amdgpu::GpuMemory *memory, uint64_t kernarg_addr,
                         const KernelArgs &args, uint32_t total_workgroups,
                         uint32_t group_size_x) {
  bool ok = true;
  ok &= write_hidden_arg(memory, kernarg_addr, args, "hidden_block_count_x", total_workgroups);
  ok &= write_hidden_arg(memory, kernarg_addr, args, "hidden_block_count_y", 1);
  ok &= write_hidden_arg(memory, kernarg_addr, args, "hidden_block_count_z", 1);
  ok &= write_hidden_arg(memory, kernarg_addr, args, "hidden_group_size_x", group_size_x);
  ok &= write_hidden_arg(memory, kernarg_addr, args, "hidden_group_size_y", 1);
  ok &= write_hidden_arg(memory, kernarg_addr, args, "hidden_group_size_z", 1);
  return ok;
}

/// Run vector_add with golden reference validation.
/// @param num_threads  Simulation threads (1 = single, >1 = multi with XCD partitioning).
/// @param wfs_per_wg   Wavefronts per workgroup (1 or 2).
void run_vector_add_stress(uint32_t num_threads, uint32_t wfs_per_wg) {
  uint32_t group_size = wfs_per_wg * WF_SIZE;
  uint32_t total_n = TOTAL_CUS * group_size;

  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load vector_add.o";
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto loaded = config::load_config(CONFIG_PATH, SCHEMA_PATH);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  loaded.engine_config.num_threads = num_threads;
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());

  if (num_threads > 1) {
    std::unordered_map<simdojo::Component *, simdojo::PartitionID> xcd_map;
    for (uint32_t i = 0; i < soc->num_xcds(); ++i) {
      xcd_map[soc->xcd(i)] = i;
    }
    engine->topology().partition_manual(
        num_threads, [&](simdojo::Component *c) -> simdojo::PartitionID {
          for (auto *p = static_cast<simdojo::Component *>(c); p != nullptr;
               p = static_cast<simdojo::Component *>(p->parent())) {
            auto it = xcd_map.find(p);
            if (it != xcd_map.end()) {
              return it->second;
            }
          }
          return 0;
        });
  }
  engine->build();

  co->load_to_memory(memory, KD_ADDR);
  uint64_t kernel_object = KD_ADDR + co->kernel_descriptor_offset("vector_add");
  ASSERT_NE(kernel_object, KD_ADDR);

  size_t vec_bytes = total_n * sizeof(float);
  std::vector<float> a(total_n), b(total_n), c_expected(total_n);
  for (uint32_t i = 0; i < total_n; ++i) {
    a[i] = static_cast<float>(i % 97) * 0.1f;
    b[i] = static_cast<float>(i % 61) * 0.2f;
    c_expected[i] = a[i] + b[i];
  }

  memory->load_image(reinterpret_cast<const uint8_t *>(a.data()), vec_bytes, A_ADDR);
  memory->load_image(reinterpret_cast<const uint8_t *>(b.data()), vec_bytes, B_ADDR);
  std::vector<float> zeros(total_n, 0.0f);
  memory->load_image(reinterpret_cast<const uint8_t *>(zeros.data()), vec_bytes, C_ADDR);

  struct { uint64_t a, b, c; uint32_t n; } args = {A_ADDR, B_ADDR, C_ADDR, total_n};
  memory->load_image(reinterpret_cast<const uint8_t *>(&args), sizeof(args), KERNARG_ADDR);

  auto parsed = KernelArgs::parse(
      reinterpret_cast<const uint8_t *>(co->image_data()), co->image_size(),
      "vector_add");
  ASSERT_TRUE(std::holds_alternative<KernelArgs>(parsed))
      << std::get<std::string>(parsed);
  ASSERT_TRUE(write_dispatch_dims(memory, KERNARG_ADDR,
                                  std::get<KernelArgs>(parsed), TOTAL_CUS, group_size));

  uint32_t wgs_per_xcd = TOTAL_CUS / TOTAL_XCDS;
  for (uint32_t xi = 0; xi < TOTAL_XCDS; ++xi) {
    auto *cp = soc->xcd(xi)->command_processor();
    cp->set_workgroup_id_offset(xi * wgs_per_xcd);
    uint64_t ring = 0xF0000000ULL + xi * 0x100000ULL;
    test::AqlQueue queue(memory, cp, ring, 4096, ring + 0x10000, ring + 0x10008, ring + 0x10010);
    queue.dispatch(kernel_object, wgs_per_xcd * group_size, group_size, KERNARG_ADDR);
  }

  engine->run();
  soc->flush_all();

  unsigned mismatches = 0;
  for (uint32_t i = 0; i < total_n; ++i) {
    float actual = std::bit_cast<float>(memory->read32(C_ADDR + i * sizeof(float)));
    if (std::abs(actual - c_expected[i]) > 1e-6f) {
      if (mismatches < 10) {
        ADD_FAILURE() << "Mismatch at C[" << i << "]: GPU=" << actual << " CPU=" << c_expected[i];
      }
      ++mismatches;
    }
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " elements differ (showing first 10)";
}

TEST(VectorAddStressTest, AllCUsGoldenReference) { run_vector_add_stress(1, 1); }

// Multi-threaded: exercises barrier-based LBTS with real GPU kernel execution.
TEST(VectorAddStressTest, AllCUsGoldenReference_MultiThreaded) {
  run_vector_add_stress(TOTAL_XCDS, 1);
}

// 2 wavefronts per workgroup: verifies workitem IDs and hidden_group_size_x.
TEST(VectorAddStressTest, TwoWavefrontsPerWorkgroup) { run_vector_add_stress(1, 2); }

TEST(VectorAddCodeObjectTest, LoadsAndDecodes) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load vector_add.o";
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);
  ASSERT_FALSE(co->text_sections().empty());

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  const auto *text = co->text_sections()[0];
  const auto *data = reinterpret_cast<const uint32_t *>(text->data());
  std::unique_ptr<Instruction> inst(decoder->decode(data));
  EXPECT_NE(inst, nullptr) << "Failed to decode first instruction";
}

TEST(VectorAddCodeObjectTest, KernelMetadataParses) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto parsed = KernelArgs::parse(
      reinterpret_cast<const uint8_t *>(co->image_data()), co->image_size(),
      "vector_add");
  ASSERT_TRUE(std::holds_alternative<KernelArgs>(parsed))
      << std::get<std::string>(parsed);
  auto &args = std::get<KernelArgs>(parsed);

  // vector_add(float* A, float* B, float* C, unsigned N).
  EXPECT_GE(args.size(), 4u);
  EXPECT_EQ(args.countValueKind("global_buffer"), 3u);
  EXPECT_GE(args.countValueKind("by_value"), 1u);

  // Non-unique kind returns nullopt.
  EXPECT_FALSE(args.getIndexFromValueKind("global_buffer").has_value());

  // Unique kind returns valid index.
  auto by_val_idx = args.getIndexFromValueKind("by_value");
  ASSERT_TRUE(by_val_idx.has_value());
  EXPECT_EQ(args.getOffsetFromIndex(*by_val_idx), 24u);
  EXPECT_EQ(args.getSizeFromIndex(*by_val_idx), 4u);
  EXPECT_EQ(args.getValueKindFromIndex(*by_val_idx), "by_value");

  // Missing kind returns nullopt.
  EXPECT_FALSE(args.getIndexFromValueKind("nonexistent").has_value());
  EXPECT_EQ(args.countValueKind("nonexistent"), 0u);

  // Access by index.
  EXPECT_EQ(args.getOffsetFromIndex(0), 0u);
  EXPECT_EQ(args.getOffsetFromIndex(1), 8u);
  EXPECT_EQ(args.getOffsetFromIndex(2), 16u);
  EXPECT_EQ(args.getOffsetFromIndex(3), 24u);

  // Address space present for global_buffer args.
  EXPECT_EQ(args.getAddressSpaceFromIndex(0), "global");

  // The kernel uses blockDim.x, so hidden_group_size_x should be present.
  auto group_idx = args.getIndexFromValueKind("hidden_group_size_x");
  ASSERT_TRUE(group_idx.has_value()) << "hidden_group_size_x not found";
  EXPECT_EQ(args.getSizeFromIndex(*group_idx), 2u);

  auto block_idx = args.getIndexFromValueKind("hidden_block_count_x");
  ASSERT_TRUE(block_idx.has_value()) << "hidden_block_count_x not found";
  EXPECT_EQ(args.getSizeFromIndex(*block_idx), 4u);
}

// Verify kernel name selection from a code object containing two kernels.
// add_one(int*, unsigned) and add_two(int*, unsigned, int) differ by one
// argument, so parsing each by name must yield different arg counts.
// A nonexistent name must return an error that includes the requested name.
TEST(KernelMetadataTest, SelectsKernelByName) {
  Executable exec(kernel_path("two_kernels"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load " << kernel_path("two_kernels");
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto *elf = reinterpret_cast<const uint8_t *>(co->image_data());
  size_t elf_size = co->image_size();

  auto r1 = KernelArgs::parse(elf, elf_size, "add_one");
  ASSERT_TRUE(std::holds_alternative<KernelArgs>(r1))
      << std::get<std::string>(r1);
  auto &k1 = std::get<KernelArgs>(r1);

  auto r2 = KernelArgs::parse(elf, elf_size, "add_two");
  ASSERT_TRUE(std::holds_alternative<KernelArgs>(r2))
      << std::get<std::string>(r2);
  auto &k2 = std::get<KernelArgs>(r2);
  EXPECT_EQ(k2.size(), k1.size() + 1);

  // Nonexistent kernel returns an error mentioning the name.
  auto r3 = KernelArgs::parse(elf, elf_size, "no_such_kernel");
  ASSERT_TRUE(std::holds_alternative<std::string>(r3));
  EXPECT_NE(std::get<std::string>(r3).find("no_such_kernel"), std::string::npos);
}

} // namespace

#endif // HAS_DEVICE_KERNELS

namespace {

using namespace rocjitsu;

// Verify that KernelArgs::parse returns a descriptive error string (not a
// KernelArgs) when given invalid input. The exact error message is checked
// to ensure diagnostics are actionable.
TEST(KernelMetadataTest, EmptyInputReturnsError) {
  auto result = KernelArgs::parse(nullptr, 0, "any");
  ASSERT_TRUE(std::holds_alternative<std::string>(result));
  EXPECT_EQ(std::get<std::string>(result), "ELF too small (0 bytes)");
}

} // namespace
