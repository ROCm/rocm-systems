// SPDX-License-Identifier: MIT

// Benchmark operand layouts, deterministic inputs, timing, and workload registration.

#include "benchmark/workloads.h"

#include <random>

namespace rocjitsu::arith::bench {

// Return the cached host-supported CPU levels for benchmark registration.
const std::vector<uint64_t> &cpu_levels() {
  static const auto levels = rocjitsu::arith::test::supported_cpu_levels();
  return levels;
}

namespace {

// Extract recognized element-format tokens from an instruction mnemonic, in order.
std::vector<std::string> element_types(const std::string &name) {
  std::vector<std::string> types;
  for (int begin = 0; begin < int(name.size());) {
    auto position = name.find('_', begin);
    int end = position == std::string::npos ? int(name.size()) : int(position);
    std::string token = name.substr(begin, end - begin);
    if (token == "f16" || token == "bf16" || token == "f32" || token == "f64" || token == "fp8" ||
        token == "bf8" || token == "i32" || token == "u32" || token == "iu8" || token == "iu4" ||
        token == "u8" || token == "u4") {
      types.push_back(token);
    }
    begin = end + 1;
  }
  return types;
}

// Construct a deterministic finite operand word for the named element format and register
// position.
uint32_t word(const std::string &type, uint32_t random, int reg) {
  // Positive finite fractional operands exercise rounding without exceptional
  // fallbacks. Each packed element has its own fraction. Arbitrary-value and
  // exception correctness belongs to the instruction unit tests.
  if (type == "f64") {
    uint64_t bits = 0x3fe0000000000000ULL | (uint64_t(random) << 20) | (random >> 12);
    return uint32_t(bits >> (32 * (reg % 2)));
  }
  if (type == "f32") {
    return 0x3f000000U | (random & 0x7fffffU);
  }
  if (type == "f16") {
    return 0x38003800U | (random & 0x03ff03ffU);
  }
  if (type == "bf16") {
    return 0x3f003f00U | (random & 0x007f007fU);
  }
  if (type == "fp8") {
    return 0x30303030U | (random & 0x07070707U);
  }
  if (type == "bf8") {
    return 0x38383838U | (random & 0x03030303U);
  }
  return 0x01010101U | (random & 0x0f0f0f0fU);
}

} // namespace

// Create reproducible operand storage for the mnemonic; optionally replace one floating-point
// input element with a quiet NaN.
Operands::Operands(const std::string &instruction, bool one_nan_lane) {
  wave = instruction.find("_w64") == std::string::npos ? 32 : 64;
  auto types = element_types(instruction);
  if (types.empty()) {
    throw std::runtime_error("Missing operand type: " + instruction);
  }
  bool matrix = instruction.find("wmma") != std::string::npos;
  bool dot = instruction.find("dot") != std::string::npos;
  std::string output = types.front();
  std::string left = (matrix || dot) && types.size() > 1 ? types[1] : output;
  std::string right = (matrix || dot) && types.size() > 2 ? types[2] : left;
  if (instruction.find("frexp_exp") != std::string::npos) {
    left = right = types.back();
  }
  // Mixed FMA defaults to FP32 inputs even for its half-output variants.
  if (instruction.find("fma_mix") != std::string::npos) {
    left = right = "f32";
  }
  std::string formats[] = {left, right, output, output};
  if (instruction.find("fma_mix") != std::string::npos) {
    formats[2] = "f32";
  }
  floating_point = left == "f16" || left == "bf16" || left == "f32" || left == "f64" ||
                   left == "fp8" || left == "bf8";
  std::minstd_rand random;
  for (int operand = 0; operand < 4; ++operand) {
    for (int reg = 0; reg < 8; ++reg) {
      for (int lane = 0; lane < 16; ++lane) {
        uint32_t raw = (uint32_t(random()) >> 15) | ((uint32_t(random()) >> 15) << 16);
        uint32_t value = word(formats[operand], raw, reg);
        for (int group = 0; group < 4; ++group) {
          data[operand][reg][lane + 16 * group] = value;
        }
      }
    }
  }
  if (instruction.find("ldexp") != std::string::npos ||
      instruction.find("trig_preop") != std::string::npos) {
    for (int lane = 0; lane < 64; ++lane) {
      data[1][0][lane] = lane % 4;
    }
  }
  for (int reg = 0; reg < 8; ++reg) {
    a[reg] = data[0][reg];
    b[reg] = data[1][reg];
    c[reg] = data[2][reg];
    d[reg] = data[3][reg];
    index[reg] = data[4][reg];
    // Sparse metadata: select positions 0 and 1 in every group of four,
    // independently in both index keys (packed two-bit indices 0,1,0,1,...).
    for (int lane = 0; lane < 64; ++lane) {
      data[4][reg][lane] = 0x44444444U;
    }
  }
  literal = output == "f16" ? 0x3a01U : 0x3f400001U;
  if (instruction.find("swmmac") != std::string::npos) {
    accumulator_registers = 256 / wave / (output == "f16" || output == "bf16" ? 2 : 1);
  } else if (instruction.find("fmac") != std::string::npos ||
             instruction.find("dot2acc") != std::string::npos) {
    accumulator_registers = output == "f64" ? 2 : 1;
  }
  if (one_nan_lane && floating_point) {
    // Change one input element, away from SIMD boundaries. Gfx11 WMMA requires
    // replicated A/B inputs: use C there to keep just one physical NaN lane.
    // Other matrix instructions use one A element (which can affect many outputs).
    bool gfx11 = instruction.find("v_wmma_") == 0 &&
                 instruction.find("gfx12") == std::string::npos &&
                 (left == "f16" || left == "bf16");
    int operand = gfx11 ? 2 : 0;
    const auto &type = formats[operand];
    auto &value = data[operand][0][5];
    if (type == "f64") {
      value = 0;
      data[operand][1][5] = 0x7ff80000U;
    } else if (type == "f32") {
      value = 0x7fc00000U;
    } else if (type == "f16" || type == "bf16") {
      value = (value & 0xffff0000U) | (type == "f16" ? 0x7e00U : 0x7fc0U);
    } else {
      // FP8 E4M3 has a single NaN encoding; BF8 E5M2 uses a quiet NaN.
      value = (value & 0xffffff00U) | (type == "fp8" ? 0x7fU : 0x7eU);
    }
  }
  std::memcpy(initial_d, data[3], sizeof(initial_d));
}

} // namespace rocjitsu::arith::bench
