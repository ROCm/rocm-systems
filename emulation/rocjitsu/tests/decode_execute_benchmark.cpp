// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file decode_execute_benchmark.cpp
/// @brief Performance benchmark for decode-execute pipeline throughput.
///
/// Measures millions of simulated instructions per second (MIPS) for the
/// decode → construct → execute → destroy cycle.  Reports wall-clock time,
/// average ns/instruction, and separate decode vs execute timing.
///
/// This benchmark is NOT a pass/fail gate — it reports numbers for tracking
/// performance regressions across changesets.

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/except.h"

#include "rocjitsu/isa/arch/amdgpu/cdna4/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/test_encodings.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string_view>
#include <vector>

namespace {

using namespace rocjitsu;
using Clock = std::chrono::steady_clock;

/// @brief Common test encoding entry (matches all ISA test_encodings.h).
struct TestEncEntry {
  std::string_view mnemonic;
  std::array<uint32_t, 2> words;
};

enum class InstCategory : uint8_t { SCALAR_ALU, VECTOR_ALU, MEMORY };

struct CorpusEntry {
  std::string_view mnemonic;
  std::array<uint32_t, 2> words;
  InstCategory category;
};

/// @brief Classify a mnemonic into a benchmark category.
inline InstCategory classify_mnemonic(std::string_view mn) {
  if (mn.starts_with("s_load") || mn.starts_with("s_store") || mn.starts_with("s_buffer") ||
      mn.starts_with("s_atomic") || mn.starts_with("buffer_") || mn.starts_with("tbuffer_") ||
      mn.starts_with("flat_") || mn.starts_with("global_") || mn.starts_with("scratch_") ||
      mn.starts_with("ds_") || mn.starts_with("image_"))
    return InstCategory::MEMORY;
  if (mn.starts_with("v_"))
    return InstCategory::VECTOR_ALU;
  return InstCategory::SCALAR_ALU;
}

/// @brief Check if a mnemonic should be skipped entirely (control flow,
/// system instructions that halt/hang on zeroed state).
inline bool should_skip(std::string_view mn) {
  static constexpr std::string_view SKIP[] = {
      "s_endpgm",     "s_branch",      "s_cbranch",  "s_setpc",     "s_swappc",        "s_call",
      "s_waitcnt",    "s_wait_",       "s_barrier",  "s_trap",      "s_sleep",         "s_sethalt",
      "s_sendmsg",    "s_nop",         "s_getpc",    "s_getreg",    "s_setreg",        "s_rfe",
      "s_icache_inv", "s_sendmsghalt", "v_readlane", "v_writelane", "v_readfirstlane", "v_mfma_",
      "v_permlane",   "v_smfma_",      "v_wmma_",    "v_swmmac_",
  };
  for (auto p : SKIP)
    if (mn.substr(0, p.size()) == p)
      return true;
  return false;
}

/// @brief Build a corpus of decodable instructions from test data.
/// Only filters by skip list and decodability — does not try execute.
std::vector<CorpusEntry> build_corpus(const TestEncEntry *encodings, size_t num_encodings,
                                      Decoder &decoder) {
  std::vector<CorpusEntry> corpus;

  for (size_t i = 0; i < num_encodings; ++i) {
    const auto &te = encodings[i];
    if (should_skip(te.mnemonic))
      continue;

    auto cat = classify_mnemonic(te.mnemonic);

    // Try decode — skip if the synthesized encoding doesn't decode.
    try {
      auto *inst = decoder.decode(te.words.data());
      if (inst) {
        corpus.push_back({te.mnemonic, te.words, cat});
        delete inst;
      }
    } catch (...) {
    }
  }
  return corpus;
}

/// @brief Report timing for a category subset of the corpus.
struct TimingResult {
  size_t count = 0;
  double decode_ns_per_inst = 0;
  double full_ns_per_inst = 0;
  double mips = 0;
};

/// @brief Run the benchmark for a given ISA.
void run_benchmark(rj_code_arch_t arch, std::string_view arch_name, const TestEncEntry *encodings,
                   size_t num_encodings) {
  // Set up CU + wavefront.
  amdgpu::GpuMemory gpu_mem(std::string(arch_name) + "_bench_mem");
  amdgpu::L2Cache l2(std::string(arch_name) + "_bench_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create(std::string(arch_name), cfg, &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto decoder = Decoder::create(arch);
  ASSERT_NE(decoder, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);

  // Build corpus (decode-only filtering, no execute warmup).
  auto corpus = build_corpus(encodings, num_encodings, *decoder);
  cu->reset_all_wf();
  wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);

  // Count per category.
  size_t n_scalar = 0, n_vector = 0, n_memory = 0;
  for (const auto &e : corpus) {
    switch (e.category) {
    case InstCategory::SCALAR_ALU:
      ++n_scalar;
      break;
    case InstCategory::VECTOR_ALU:
      ++n_vector;
      break;
    case InstCategory::MEMORY:
      ++n_memory;
      break;
    }
  }

  if (corpus.empty()) {
    std::printf("  %.*s: empty corpus, skipping\n", static_cast<int>(arch_name.size()),
                arch_name.data());
    return;
  }

  constexpr int ITERATIONS = 200;
  const size_t corpus_size = corpus.size();
  const size_t total_instructions = corpus_size * ITERATIONS;

  // Filter to non-memory for execute benchmark (memory ops crash on zeroed).
  std::vector<const CorpusEntry *> executable;
  for (const auto &e : corpus)
    if (e.category != InstCategory::MEMORY)
      executable.push_back(&e);

  const size_t exec_total = executable.size() * ITERATIONS;

  // --- Measure decode only (all categories) ---
  auto decode_start = Clock::now();
  for (int iter = 0; iter < ITERATIONS; ++iter) {
    for (const auto &entry : corpus) {
      auto *inst = decoder->decode(entry.words.data());
      delete inst;
    }
  }
  auto decode_end = Clock::now();
  auto decode_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(decode_end - decode_start).count();

  // --- Measure decode + execute (non-memory only) ---
  auto full_start = Clock::now();
  for (int iter = 0; iter < ITERATIONS; ++iter) {
    for (const auto *entry : executable) {
      auto *inst = decoder->decode(entry->words.data());
      if (inst) {
        try {
          cu->execute_instruction(inst, *wf);
        } catch (...) {
        }
        delete inst;
      }
    }
  }
  auto full_end = Clock::now();
  auto full_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(full_end - full_start).count();

  // Compute metrics.
  double decode_ns_per = static_cast<double>(decode_ns) / static_cast<double>(total_instructions);
  double full_ns_per =
      exec_total > 0 ? static_cast<double>(full_ns) / static_cast<double>(exec_total) : 0;
  double execute_ns_per = full_ns_per - decode_ns_per;
  double mips_decode =
      static_cast<double>(total_instructions) / (static_cast<double>(decode_ns) / 1e9) / 1e6;
  double mips_full =
      exec_total > 0 ? static_cast<double>(exec_total) / (static_cast<double>(full_ns) / 1e9) / 1e6
                     : 0;

  std::printf("\n  === %.*s DECODE-EXECUTE BENCHMARK ===\n"
              "  Corpus: %zu instructions (%zu scalar, %zu vector, %zu memory)\n"
              "  Iterations: %d (total: %zu decode, %zu decode+execute)\n"
              "  \n"
              "  Decode only:  %.1f ns/inst  (%.2f MIPS)\n"
              "  Decode+Exec:  %.1f ns/inst  (%.2f MIPS)\n"
              "  Execute only: %.1f ns/inst  (estimated)\n"
              "  Wall clock:   %.1f ms (decode), %.1f ms (decode+exec)\n",
              static_cast<int>(arch_name.size()), arch_name.data(), corpus_size, n_scalar, n_vector,
              n_memory, ITERATIONS, total_instructions, exec_total, decode_ns_per, mips_decode,
              full_ns_per, mips_full, execute_ns_per, static_cast<double>(decode_ns) / 1e6,
              static_cast<double>(full_ns) / 1e6);

  EXPECT_GT(mips_decode, 0.1) << "Decode throughput too low for " << arch_name;
}

// --- Benchmark tests ---

TEST(DecodeExecuteBenchmark, Cdna4) {
  run_benchmark(ROCJITSU_CODE_ARCH_CDNA4, "cdna4",
                reinterpret_cast<const TestEncEntry *>(cdna4::test_data::ENCODINGS),
                cdna4::test_data::NUM_ENCODINGS);
}

TEST(DecodeExecuteBenchmark, Rdna4) {
  run_benchmark(ROCJITSU_CODE_ARCH_RDNA4, "rdna4",
                reinterpret_cast<const TestEncEntry *>(rdna4::test_data::ENCODINGS),
                rdna4::test_data::NUM_ENCODINGS);
}

TEST(DecodeExecuteBenchmark, Gfx1250) {
  run_benchmark(ROCJITSU_CODE_ARCH_GFX1250, "gfx1250",
                reinterpret_cast<const TestEncEntry *>(gfx1250::test_data::ENCODINGS),
                gfx1250::test_data::NUM_ENCODINGS);
}

} // namespace
