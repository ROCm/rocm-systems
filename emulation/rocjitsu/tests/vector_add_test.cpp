// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file Multi-XCD vector addition stress test with golden reference validation.
///
/// Loads a compiled vector_add.hip kernel and dispatches 288 workgroups across
/// all 8 XCDs (CDNA4 physical topology), one workgroup per CU. Each wavefront of 64
/// threads computes C[gid] = A[gid] + B[gid]. Results are compared against a
/// CPU golden reference.

#include "aql_queue.h"
#include "decode_test_util.h"
#include "test_paths.h"

#include "embedded_schema.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/partitioning.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef HAS_DEVICE_KERNELS

namespace {

using namespace rocjitsu;

const std::string CONFIG_PATH = test::config_path("gfx950_mi355x.json");
using test::kernel_path;

constexpr uint32_t TOTAL_XCDS = 8;
constexpr uint32_t CUS_PER_XCD = 36; // 4 SEs x 9 physical CUs
constexpr uint32_t TOTAL_CUS = TOTAL_XCDS * CUS_PER_XCD;
constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t N = TOTAL_CUS * WF_SIZE; // 18432 elements, one WG per physical CU

constexpr uint64_t KD_ADDR = 0x10000;
constexpr uint64_t A_ADDR = 0x100000;
constexpr uint64_t B_ADDR = 0x200000;
constexpr uint64_t C_ADDR = 0x300000;
constexpr uint64_t KERNARG_ADDR = 0x400000;
constexpr uint64_t SIGNAL_ADDR = 0x500000;
// amd_signal_t::value sits 8 bytes into the signal object.
constexpr uint32_t SIGNAL_VALUE_OFFSET = 8;

struct VectorAddRunResult {
  std::vector<uint32_t> output_words;
  std::vector<float> expected;
};

void run_vector_add(uint32_t dispatch_threads, VectorAddRunResult &result) {
  // Load the compiled vector_add kernel.
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load vector_add.o";
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  // Build the simulation engine with CDNA4 topology.
  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  soc->set_dispatch_threads(dispatch_threads);
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  engine->create();

  // Load code into GPU memory. Place .rodata (kernel descriptor) and .text (code)
  // at their virtual addresses relative to a base. The .kd symbol value matches
  // the rodata vaddr, and kernel_code_entry_byte_offset bridges to .text vaddr.
  uint64_t kd_offset = co->kernel_descriptor_offset("vector_add");
  ASSERT_NE(kd_offset, 0u) << "Kernel descriptor symbol not found";
  for (const auto *sec : co->rodata_sections())
    memory->load_image(reinterpret_cast<const uint8_t *>(sec->data()), sec->size(),
                       KD_ADDR + sec->vaddr());
  for (const auto *sec : co->text_sections())
    memory->load_image(reinterpret_cast<const uint8_t *>(sec->data()), sec->size(),
                       KD_ADDR + sec->vaddr());
  uint64_t kernel_object = KD_ADDR + kd_offset;

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

  // Dispatch across all XCDs via AQL queues.
  uint32_t wgs_per_xcd = TOTAL_CUS / TOTAL_XCDS;
  for (uint32_t xi = 0; xi < TOTAL_XCDS; ++xi) {
    auto *cp = soc->xcd(xi)->command_processor();
    cp->set_workgroup_id_offset(xi * wgs_per_xcd);
    uint64_t ring = 0xF0000000ULL + xi * 0x100000ULL;
    test::AqlQueue queue(memory, cp, ring, 4096, ring + 0x10000, ring + 0x10008, ring + 0x10010);
    queue.dispatch(kernel_object, wgs_per_xcd * WF_SIZE, WF_SIZE, KERNARG_ADDR);
  }

  // Engine drives all CPs and CUs to completion.
  engine->run();
  soc->flush_all();

  std::vector<uint32_t> output_words(N);
  for (uint32_t i = 0; i < N; ++i)
    output_words[i] = memory->read32(C_ADDR + i * sizeof(float));

  result.output_words = std::move(output_words);
  result.expected = std::move(C_expected);
}

void expect_vector_add_matches_golden(const std::vector<uint32_t> &output_words,
                                      const std::vector<float> &expected) {
  ASSERT_EQ(output_words.size(), expected.size());

  unsigned mismatches = 0;
  for (size_t i = 0; i < output_words.size(); ++i) {
    float actual = std::bit_cast<float>(output_words[i]);
    if (std::abs(actual - expected[i]) > 1e-6f) {
      if (mismatches < 10)
        ADD_FAILURE() << "Mismatch at C[" << i << "]: GPU=" << actual << " CPU=" << expected[i];
      ++mismatches;
    }
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " elements differ (showing first 10)";
}

void expect_same_output_words(const std::vector<uint32_t> &expected_words,
                              const std::vector<uint32_t> &actual_words) {
  ASSERT_EQ(actual_words.size(), expected_words.size());

  unsigned mismatches = 0;
  for (size_t i = 0; i < actual_words.size(); ++i) {
    if (actual_words[i] == expected_words[i])
      continue;
    if (mismatches < 10) {
      ADD_FAILURE() << "Output word mismatch at C[" << i
                    << "]: dispatch_threads=1 word=" << expected_words[i]
                    << " threaded word=" << actual_words[i];
    }
    ++mismatches;
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " output words differ (showing first 10)";
}

TEST(VectorAddStressTest, AllCUsGoldenReference) {
  VectorAddRunResult result;
  ASSERT_NO_FATAL_FAILURE(run_vector_add(1, result));
  expect_vector_add_matches_golden(result.output_words, result.expected);
}

TEST(VectorAddStressTest, CpuDispatchThreadsPreserveOutputBytes) {
  VectorAddRunResult serial;
  VectorAddRunResult threaded;
  ASSERT_NO_FATAL_FAILURE(run_vector_add(1, serial));
  ASSERT_NO_FATAL_FAILURE(run_vector_add(TOTAL_XCDS, threaded));

  expect_vector_add_matches_golden(serial.output_words, serial.expected);
  expect_vector_add_matches_golden(threaded.output_words, threaded.expected);
  expect_same_output_words(serial.output_words, threaded.output_words);
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

  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  loaded.engine_config.num_threads = TOTAL_XCDS;
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());

  ASSERT_TRUE(amdgpu::partition_topology_by_xcds(engine->topology(), soc, TOTAL_XCDS));
  engine->create();

  co->load_to_memory(memory, KD_ADDR);
  uint64_t kernel_object = KD_ADDR + co->kernel_descriptor_offset("vector_add");
  ASSERT_NE(kernel_object, KD_ADDR);

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
  for (uint32_t xi = 0; xi < TOTAL_XCDS; ++xi) {
    auto *cp = soc->xcd(xi)->command_processor();
    cp->set_workgroup_id_offset(xi * wgs_per_xcd);
    uint64_t ring = 0xF0000000ULL + xi * 0x100000ULL;
    test::AqlQueue queue(memory, cp, ring, 4096, ring + 0x10000, ring + 0x10008, ring + 0x10010);
    queue.dispatch(kernel_object, wgs_per_xcd * WF_SIZE, WF_SIZE, KERNARG_ADDR);
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
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, data));
  EXPECT_NE(inst, nullptr) << "Failed to decode first instruction";
}

// One queue, one dispatch, spread over all 8 XCDs by fan-out rather than by the
// caller splitting the grid by hand. Proves the parts that a workgroup histogram
// cannot: that a shard's workgroup ids address the right slice of the grid, that
// each XCD's caches are published before the dispatch retires, and that the
// completion signal fires once, after the last workgroup anywhere on the device.
void run_fanout_golden_reference(uint32_t num_threads) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load vector_add.o";
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  loaded.engine_config.num_threads = num_threads;
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  if (num_threads > 1) {
    ASSERT_TRUE(amdgpu::partition_topology_by_xcds(engine->topology(), soc, num_threads));
  }
  engine->create();

  co->load_to_memory(memory, KD_ADDR);
  uint64_t kernel_object = KD_ADDR + co->kernel_descriptor_offset("vector_add");
  ASSERT_NE(kernel_object, KD_ADDR);

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

  auto *cp = soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  auto queue = test::make_fanout_queue(memory, cp);
  queue->dispatch(kernel_object, N, WF_SIZE, KERNARG_ADDR);

  engine->run();
  soc->flush_all();

  auto counts = soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(counts.size(), TOTAL_XCDS);
  for (uint32_t xi = 0; xi < TOTAL_XCDS; ++xi)
    EXPECT_EQ(counts[xi], TOTAL_CUS / TOTAL_XCDS) << "xcd" << xi;

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

// Every fan-out test above writes its inputs before any cache is filled, and
// AqlQueue::dispatch leaves the packet's acquire scope at NONE, so none of them
// depends on the peer-side invalidate at all: they would pass with the fence
// removed. This primes each XCD's L2 with stale inputs first, so a peer that
// skips the invalidate computes from data that is provably out of date.
//
// @param acquire_scope HSA_FENCE_SCOPE_AGENT to fence, HSA_FENCE_SCOPE_NONE for
// the control that shows the fence is what produced the result.
// @returns Number of elements that disagree with the golden reference, or nullopt
// if the run never happened. Deliberately not a count: zero mismatches is the
// PASSING result for the agent-scope case, so returning it when a prerequisite is
// missing would report a dispatch that never ran as a clean one.
std::optional<unsigned> run_fanout_with_primed_peer_caches(uint32_t acquire_scope) {
  Executable exec(kernel_path("vector_add"));
  EXPECT_TRUE(exec.is_valid()) << "Failed to load vector_add.o";
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  EXPECT_NE(co, nullptr);
  if (!co)
    return std::nullopt;

  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  engine->create();

  co->load_to_memory(memory, KD_ADDR);
  uint64_t kernel_object = KD_ADDR + co->kernel_descriptor_offset("vector_add");

  const size_t vec_bytes = N * sizeof(float);
  std::vector<float> stale(N, -1.0f);
  std::vector<float> A(N), B(N), C_expected(N);
  for (uint32_t i = 0; i < N; ++i) {
    A[i] = static_cast<float>(i % 97) * 0.1f;
    B[i] = static_cast<float>(i % 61) * 0.2f;
    C_expected[i] = A[i] + B[i];
  }

  // 1. Stale inputs reach the backing store.
  memory->load_image(reinterpret_cast<const uint8_t *>(stale.data()), vec_bytes, A_ADDR);
  memory->load_image(reinterpret_cast<const uint8_t *>(stale.data()), vec_bytes, B_ADDR);
  std::vector<float> zeros(N, 0.0f);
  memory->load_image(reinterpret_cast<const uint8_t *>(zeros.data()), vec_bytes, C_ADDR);

  // 2. Pull them through every XCD's L2, so each one now holds the stale lines.
  std::vector<uint8_t> sink(vec_bytes);
  for (uint32_t xi = 0; xi < TOTAL_XCDS; ++xi) {
    auto *l2 = soc->xcd(xi)->l2_cache();
    EXPECT_NE(l2, nullptr) << "xcd" << xi << " has no L2 to prime";
    if (!l2)
      return std::nullopt;
    l2->read(A_ADDR, sink.data(), static_cast<uint32_t>(vec_bytes));
    l2->read(B_ADDR, sink.data(), static_cast<uint32_t>(vec_bytes));
  }

  // 3. Update the backing store behind those caches. Every XCD is now stale.
  memory->load_image(reinterpret_cast<const uint8_t *>(A.data()), vec_bytes, A_ADDR);
  memory->load_image(reinterpret_cast<const uint8_t *>(B.data()), vec_bytes, B_ADDR);

  struct {
    uint64_t A, B, C;
    uint32_t N;
  } args = {A_ADDR, B_ADDR, C_ADDR, N};
  memory->load_image(reinterpret_cast<const uint8_t *>(&args), sizeof(args), KERNARG_ADDR);

  auto *cp = soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  EXPECT_NE(cp, nullptr);
  if (!cp)
    return std::nullopt;
  auto queue = test::make_fanout_queue(memory, cp);

  // 4. One fanned-out dispatch carrying the requested acquire scope. Only the
  // owner would invalidate if the fence did not travel with each shard.
  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header =
      HSA_PACKET_TYPE_KERNEL_DISPATCH | (acquire_scope << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
  pkt.setup = 1;
  pkt.workgroup_size_x = WF_SIZE;
  pkt.workgroup_size_y = 1;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = N;
  pkt.grid_size_y = 1;
  pkt.grid_size_z = 1;
  pkt.kernel_object = kernel_object;
  pkt.kernarg_address = reinterpret_cast<void *>(KERNARG_ADDR);
  queue->submit(pkt);

  engine->run();
  soc->flush_all();

  unsigned mismatches = 0;
  for (uint32_t i = 0; i < N; ++i) {
    float actual = std::bit_cast<float>(memory->read32(C_ADDR + i * sizeof(float)));
    if (std::abs(actual - C_expected[i]) > 1e-6f)
      ++mismatches;
  }
  return mismatches;
}

// With an agent-scope acquire, every shard invalidates its own XCD's caches on
// its own thread and reads the updated inputs.
TEST(VectorAddStressTest, FanoutAgentAcquireReachesEveryXcdsCaches) {
  auto mismatches = run_fanout_with_primed_peer_caches(HSA_FENCE_SCOPE_AGENT);
  ASSERT_TRUE(mismatches.has_value()) << "the primed fan-out dispatch never ran";
  EXPECT_EQ(*mismatches, 0u)
      << "a peer XCD computed from inputs it had cached before they were updated";
}

// The control. Without the fence the stale lines survive, so this must NOT come
// out clean -- otherwise the test above proves nothing about the acquire path
// and the priming is what is broken.
TEST(VectorAddStressTest, FanoutWithoutAcquireKeepsStalePeerCaches) {
  auto mismatches = run_fanout_with_primed_peer_caches(HSA_FENCE_SCOPE_NONE);
  ASSERT_TRUE(mismatches.has_value()) << "the primed fan-out dispatch never ran";
  EXPECT_GT(*mismatches, 0u)
      << "priming did not make any XCD stale, so the acquire test is vacuous";
}

// Watches a fanned-out dispatch's completion signal from inside the run.
//
// The signal is the only thing a host waits on, so "fires exactly once" is not
// the whole contract: it must also not fire while any XCD still has workgroups
// to run, or a host that woke on it would read a half-written buffer. Checking
// the value after engine->run() cannot tell those apart, so sample it as the
// grid retires instead.
class SignalOrderPlugin : public ExecutionPlugin {
public:
  SignalOrderPlugin(amdgpu::GpuMemory *memory, uint64_t initial_value, uint32_t total_wgs,
                    const std::vector<float> &expected)
      : ExecutionPlugin("fanout-signal-order"), memory_(memory), initial_value_(initial_value),
        total_wgs_(total_wgs), expected_(expected) {}

  void onAmdgpuWorkgroupCompleted(uint32_t, uint32_t) override {
    ++wgs_completed_;
    if (memory_->read64(SIGNAL_ADDR + SIGNAL_VALUE_OFFSET) == initial_value_)
      return;
    // The signal moved. Record the first time we saw that, and whether the whole
    // grid's results had reached memory by then.
    if (signal_moved_after_wgs_ != 0)
      return;
    signal_moved_after_wgs_ = wgs_completed_;
    results_visible_when_signal_moved_ = 0;
    for (uint32_t i = 0; i < expected_.size(); ++i) {
      float actual = std::bit_cast<float>(memory_->read32(C_ADDR + i * sizeof(float)));
      if (std::abs(actual - expected_[i]) <= 1e-6f)
        ++results_visible_when_signal_moved_;
    }
  }

  uint32_t wgs_completed() const { return wgs_completed_; }
  /// Workgroups that had retired the first time the signal was seen changed, or
  /// 0 if it never changed while the grid was still running.
  uint32_t signal_moved_after_wgs() const { return signal_moved_after_wgs_; }
  uint32_t results_visible_when_signal_moved() const { return results_visible_when_signal_moved_; }
  uint32_t total_wgs() const { return total_wgs_; }

private:
  amdgpu::GpuMemory *memory_;
  uint64_t initial_value_;
  uint32_t total_wgs_;
  const std::vector<float> &expected_;
  uint32_t wgs_completed_ = 0;
  uint32_t signal_moved_after_wgs_ = 0;
  uint32_t results_visible_when_signal_moved_ = 0;
};

// A real completion signal on a fanned-out dispatch, watched while the grid
// retires rather than only after it.
//
// Every other fan-out test leaves completion_signal at zero, so none of them
// enters the signalling path at all. The one that does check the value reads it
// after engine->run(), which cannot distinguish one final decrement from a
// decrement issued as soon as the owning XCD finished its own eighth of the
// grid: both leave the same value behind. This runs the real vector_add kernel,
// so each shard writes results a host could actually read, and samples the
// signal on every workgroup retirement. An implementation that signalled at its
// own share's retirement would be caught by the 252 of 288 samples that follow.
//
// @param num_threads 1 for the single drain loop, TOTAL_XCDS for one engine
// thread per XCD -- the case where an XCD can genuinely still be running while
// the owner's share is done.
void run_fanout_signal_ordering(uint32_t num_threads) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load vector_add.o";
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto loaded = config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  loaded.engine_config.num_threads = num_threads;
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  if (num_threads > 1) {
    ASSERT_TRUE(amdgpu::partition_topology_by_xcds(engine->topology(), soc, num_threads));
  }
  engine->create();

  co->load_to_memory(memory, KD_ADDR);
  const uint64_t kernel_object = KD_ADDR + co->kernel_descriptor_offset("vector_add");
  ASSERT_NE(kernel_object, KD_ADDR);

  const size_t vec_bytes = N * sizeof(float);
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

  // Not 1, so a signal written rather than decremented is also visible.
  constexpr uint64_t kInitialSignal = 5;
  memory->write64(SIGNAL_ADDR + SIGNAL_VALUE_OFFSET, kInitialSignal);

  constexpr uint32_t kTotalWgs = N / WF_SIZE;
  auto plugin = std::make_unique<SignalOrderPlugin>(memory, kInitialSignal, kTotalWgs, C_expected);
  auto *watch = plugin.get();
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(group->add(std::move(plugin)));
  soc->set_plugin_group(group);

  auto *cp = soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  auto queue = test::make_fanout_queue(memory, cp);

  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH |
               (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
  pkt.setup = 1;
  pkt.workgroup_size_x = WF_SIZE;
  pkt.workgroup_size_y = 1;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = N;
  pkt.grid_size_y = 1;
  pkt.grid_size_z = 1;
  pkt.kernel_object = kernel_object;
  pkt.kernarg_address = reinterpret_cast<void *>(KERNARG_ADDR);
  pkt.completion_signal.handle = SIGNAL_ADDR;
  queue->submit(pkt);

  engine->run();
  soc->flush_all();

  // The grid really was spread, or none of this says anything about fan-out.
  auto counts = soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(counts.size(), TOTAL_XCDS);
  for (uint32_t xi = 0; xi < TOTAL_XCDS; ++xi)
    ASSERT_EQ(counts[xi], TOTAL_CUS / TOTAL_XCDS) << "xcd" << xi;
  // ...and every workgroup was sampled, or the watch above saw nothing.
  ASSERT_EQ(watch->wgs_completed(), kTotalWgs);

  // The ordering claim. Seeing the signal move at all while workgroups were still
  // retiring means a host could have woken on it early.
  if (watch->signal_moved_after_wgs() != 0) {
    EXPECT_EQ(watch->signal_moved_after_wgs(), watch->total_wgs())
        << "the completion signal moved with " << (kTotalWgs - watch->signal_moved_after_wgs())
        << " workgroups of the grid still outstanding";
    EXPECT_EQ(watch->results_visible_when_signal_moved(), N)
        << "the completion signal moved before every XCD's results reached memory";
  }

  // ...and it moved exactly once, not once per share.
  EXPECT_EQ(memory->read64(SIGNAL_ADDR + SIGNAL_VALUE_OFFSET), kInitialSignal - 1)
      << "a fanned-out dispatch must decrement its completion signal once, not once per XCD";

  unsigned mismatches = 0;
  for (uint32_t i = 0; i < N; ++i) {
    float actual = std::bit_cast<float>(memory->read32(C_ADDR + i * sizeof(float)));
    if (std::abs(actual - C_expected[i]) > 1e-6f)
      ++mismatches;
  }
  EXPECT_EQ(mismatches, 0u) << "the shards did not write the results the signal stands for";
}

TEST(VectorAddStressTest, FanoutHoldsTheSignalUntilTheWholeGridRetires) {
  run_fanout_signal_ordering(/*num_threads=*/1);
}

TEST(VectorAddStressTest, FanoutHoldsTheSignalUntilTheWholeGridRetires_MultiThreaded) {
  run_fanout_signal_ordering(/*num_threads=*/TOTAL_XCDS);
}

TEST(VectorAddStressTest, FanoutSingleQueueGoldenReference) {
  run_fanout_golden_reference(/*num_threads=*/1);
}

// The same dispatch with one engine thread per XCD. Here the owning XCD hands
// shards to command processors on other partitions and the last XCD to finish
// wakes the owner to fire the completion signal, so this is the case where the
// cross-partition handoff and the flush-before-publish ordering actually matter.
TEST(VectorAddStressTest, FanoutSingleQueueGoldenReference_MultiThreaded) {
  run_fanout_golden_reference(/*num_threads=*/TOTAL_XCDS);
}

} // namespace

#endif // HAS_DEVICE_KERNELS
