// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Benchmark operand layouts, deterministic inputs, timing, and workload registration.

#pragma once

#include "benchmark/sampling.h"
#include "common/cpu.h"
#include "rj_arith/rj_arith.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu::arith::bench {

// Stable storage: rows for different CPU/semantics configurations share inputs.
// Eight VGPRs per operand cover every matrix layout, including gfx11 replication.
// The constructor initializes corresponding lane groups identically for gfx11.
class Operands {
public:
  uint32_t data[5][8][64] = {};
  uint32_t initial_d[8][64] = {};
  uint32_t *d[8];
  const uint32_t *a[8], *b[8], *c[8], *index[8];
  uint32_t status = 0, condition32 = 0, literal = 0;
  uint64_t condition64 = 0;
  int wave = 32;
  int accumulator_registers = 0;
  bool floating_point = false;

  // Create mnemonic-appropriate operands, optionally with one quiet NaN input element.
  explicit Operands(const std::string &instruction, bool one_nan_lane = false);
  Operands(const Operands &) = delete;
  Operands &operator=(const Operands &) = delete;

  // Restore destination data and clear status/condition outputs before a measured batch.
  void reset() {
    std::memcpy(data[3], initial_d, sizeof(initial_d));
    status = condition32 = 0;
    condition64 = 0;
  }

  // Restore the accumulator registers to their initial values.
  void reset_accumulator() {
    for (int reg = 0; reg < accumulator_registers; ++reg) {
      std::memcpy(d[reg], initial_d[reg], wave * sizeof(uint32_t));
    }
  }
};

// Call is a small lambda containing a direct public API call. Its timing loop
// is instantiated here so no std::function/function-pointer dispatch is paid
// per instruction. Type erasure happens only once per benchmark repetition.
template <typename Call>
double measure(Call call, Operands &r, uint64_t flags, double min_ms, uint64_t &iterations) {
  r.reset();
  for (;;) {
    int status = RJ_ARITH_SUCCESS;
    auto start = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < iterations; ++i) {
      // FMAC/DOT2ACC/SWMMAC read old D. Restore it to avoid drifting to Inf or
      // measuring a converged accumulator. This copy is included in their cost.
      r.reset_accumulator();
      status |= call(r, flags);
      // Compiler-only barrier: keep repeated calls observable under LTO too.
      std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    auto elapsed = std::chrono::steady_clock::now() - start;
    if (status != RJ_ARITH_SUCCESS) {
      throw std::runtime_error("API failed during measurement");
    }
    double ms = std::chrono::duration<double, std::milli>(elapsed).count();
    if (ms >= min_ms && ms > 0) {
      return (ms / iterations) * 1e6;
    }
    if (iterations > std::numeric_limits<uint64_t>::max() / 2) {
      throw std::runtime_error("Iteration count overflow");
    }
    iterations *= 2;
  }
}

// Return the cached host-supported CPU levels for benchmark registration.
const std::vector<uint64_t> &cpu_levels();

// Register supported CPU/semantics combinations and finite/one-NaN variants for an instruction.
template <typename Call>
void add(std::vector<Benchmark> &cases, const char *category, const char *instruction, Call call) {
  for (bool one_nan_lane : {false, true}) {
    auto operands = std::make_shared<Operands>(instruction, one_nan_lane);
    if (one_nan_lane && !operands->floating_point) {
      break;
    }
    const uint64_t modes[] = {RJ_ARITH_SEMANTICS_LOOSE,
                              RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                              RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL};
    const char *names[] = {"loose", "bit_exact_empirical_no_fp_status", "bit_exact_empirical"};
    for (int mode = 0; mode < 3; ++mode) {
      for (uint64_t cpu : cpu_levels()) {
        Benchmark row;
        row.non_finite_inputs = one_nan_lane ? "one NaN lane" : "";
        row.category = category;
        row.instruction = instruction;
        row.wave = operands->wave;
        if (operands->accumulator_registers != 0) {
          row.variants.push_back("accumulator reset");
        }
        row.semantics = names[mode];
        row.cpu = rj_arith_cpu_name(cpu);
        uint64_t flags = modes[mode] | cpu;
        operands->reset();
        row.api_status = call(*operands, flags);
        if (row.api_status == RJ_ARITH_SUCCESS) {
          row.sample = [operands, flags, call, iterations = uint64_t(128)](double min_ms) mutable {
            return measure(call, *operands, flags, min_ms, iterations);
          };
        } else if (row.api_status != RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS &&
                   row.api_status != RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS) {
          throw std::runtime_error(std::string(instruction) + ": invalid benchmark request (" +
                                   std::to_string(row.api_status) + ")");
        }
        cases.push_back(std::move(row));
      }
    }
  }
}

// Append WMMA instruction workloads and supported CPU/semantics variants.
void add_wmma(std::vector<Benchmark> &cases);

// Append DOT instruction workloads and supported CPU/semantics variants.
void add_dot(std::vector<Benchmark> &cases);

// Append FP instruction workloads and supported CPU/semantics variants.
void add_fp(std::vector<Benchmark> &cases);

// Append MATH instruction workloads and supported CPU/semantics variants.
void add_math(std::vector<Benchmark> &cases);

} // namespace rocjitsu::arith::bench
