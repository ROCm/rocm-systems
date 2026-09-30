// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "fp_math_provider_probe.h"

#include "rocjitsu/isa/arch/amdgpu/shared/fp_math_provider.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/amdgpu_exp.h"
#include "util/amdgpu_log.h"
#include "util/simd_test_hooks.h"

#include <array>
#include <cfenv>
#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu::fp_math;
using Clock = std::chrono::steady_clock;
constexpr uint32_t kSentinel = 0xcafef00du;
constexpr unsigned kSgprs = 106, kVgprs = 256;

int error(rj_test_fp_math_result *out, const char *message) noexcept {
  std::snprintf(out->error, sizeof(out->error), "%s", message);
  return -1;
}

std::array<uint32_t, 64> inputs(Operation operation) {
  // Retained gfx1201 witnesses and boundaries, followed by deterministic bits.
  constexpr std::array<uint32_t, 16> exp = {0x33800000, 0xb3800000, 0x42ffffff, 0x43000000,
                                            0xc2fc0000, 0xc2fc0001, 0x3f0567ec, 0xc114ed44,
                                            0x3e000090, 0x3ec0001a, 0x7fa12345, 0xff800000,
                                            0x80000001, 0x7f800000, 0x00000000, 0x80000000};
  constexpr std::array<uint32_t, 16> log = {0x3f7bffff, 0x3f7c0000, 0x3f7e0000, 0x3f7fffff,
                                            0x3f800000, 0x3f800001, 0x3f810000, 0x3f820000,
                                            0x3f840000, 0x3fffffff, 0x7fa12345, 0xff800000,
                                            0x80000001, 0x7f800000, 0x00000000, 0x80000000};
  std::array<uint32_t, 64> result{};
  const auto &witnesses = operation == Operation::Exp ? exp : log;
  for (unsigned lane = 0; lane < result.size(); ++lane)
    result[lane] =
        lane < witnesses.size() ? witnesses[lane] : 0x3e000000u + ((lane * 0x10123u) & 0x01ffffffu);
  return result;
}

struct Fixture {
  amdgpu::GpuMemory memory{"fp_shared_probe_mem"};
  amdgpu::L2Cache l2{"fp_shared_probe_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wave;
  // CU creation activates a thread-local decoder pool. An instruction here
  // must not depend on that pool surviving another nested fixture's teardown.
  // Destruction order keeps this guard alive until instruction is destroyed.
  std::optional<Instruction::ScopedHeapAllocation> heap_allocation;
  std::unique_ptr<Instruction> instruction;
  unsigned destination;

  Fixture(Operation operation, unsigned scenario, unsigned wave_size,
          rj_code_target_id_t target = ROCJITSU_CODE_TARGET_GFX1201)
      : wave(nullptr), destination(scenario == 4 || scenario == 5 ? 0 : 2) {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = ROCJITSU_CODE_ARCH_RDNA4;
    config.target = target;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = kSgprs;
    config.vgprs_per_wf = kVgprs;
    config.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("fp_shared_probe_cu", config, &memory, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
    if (!cu || !decoder || !(wave = cu->dispatch_wf(0, 0, kSgprs, kVgprs, wave_size)))
      throw std::runtime_error("cannot create gfx1201 FP probe wave");
    heap_allocation.emplace();
    const unsigned opcode = operation == Operation::Exp ? 37 : 39;
    const unsigned source = scenario == 6 ? 0 : scenario == 7 ? 242 : 256;
    const uint32_t words[] = {0x7e000000u | (destination << 17) | (opcode << 9) | source, 0, 0, 0};
    auto decoded = decoder->decode(words);
    if (decoded.failed())
      throw std::runtime_error("cannot decode gfx1201 VOP1 EXP/LOG");
    instruction = std::move(decoded.value());
    if (!instruction || !instruction->execute)
      throw std::runtime_error("decoded EXP/LOG has no execute callback");
    wave->set_mode_raw(0);
  }

  void seed(const std::array<uint32_t, 64> &input, uint64_t exec) {
    const unsigned base = wave->vgpr_alloc().base;
    for (unsigned lane = 0; lane < wave->wf_size(); ++lane) {
      cu->write_vgpr(base, lane, input[lane]);
      if (destination != 0)
        cu->write_vgpr(base + destination, lane, kSentinel);
    }
    cu->write_sgpr(wave->sgpr_alloc().base, 0x3f0567ecu);
    wave->set_exec(exec);
  }

  void execute() {
    if (!cu->execute_instruction(instruction.get(), *wave).succeeded())
      throw std::runtime_error("decoded shared-image EXP/LOG execution failed");
  }

  uint32_t result(unsigned lane) const {
    return cu->read_vgpr(wave->vgpr_alloc().base + destination, lane);
  }
};

struct SavedEnvironment {
  fenv_t environment{};
  SavedEnvironment() { std::fegetenv(&environment); }
  ~SavedEnvironment() { std::fesetenv(&environment); }
};

} // namespace

extern "C" RJ_API_EXPORT int rj_test_fp_math_probe(uint32_t operation, uint32_t scenario,
                                                   uint32_t wave_size, uint32_t iterations,
                                                   rj_test_fp_math_result *out) noexcept {
  if (!out)
    return -1;
  *out = {};
  if (operation > 1 || scenario > 7 || (wave_size != 32 && wave_size != 64) ||
      (iterations != 0 && scenario != 0))
    return error(out, "invalid FP probe operation/scenario/wave size/iterations");
  try {
    const Operation op = operation == 0 ? Operation::Exp : Operation::Log;
    const auto start_init = Clock::now();
    const auto &provider = fp_provider();
    const FpQualificationKey key{ROCJITSU_CODE_TARGET_GFX1201,
                                 op,
                                 InstructionForm::Vop1,
                                 F32Format::F32,
                                 F32Format::F32,
                                 0,
                                 0,
                                 0,
                                 0,
                                 false};
    const auto *kernel = provider.unary_f32(key);
    out->tier = static_cast<uint32_t>(provider.selected_tier());
    out->kind = kernel ? static_cast<uint32_t>(kernel->kind) : 0;
    out->qualified_model = kernel != nullptr || util::force_scalar();
    out->wave_callback_addr = kernel ? reinterpret_cast<uintptr_t>(kernel->execute) : 0;
    Fixture fixture(op, scenario, wave_size);
    out->first_init_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_init).count());
    out->instruction_callback_addr = reinterpret_cast<uintptr_t>(fixture.instruction->execute);
    const uint64_t full = wave_size == 32 ? 0xffffffffu : ~uint64_t{0};
    out->exec = (scenario == 1 || scenario == 5 || scenario >= 6) ? full & 0xf0f0000fff00ff0full
                : scenario == 2                                   ? full & 0x8000000100000081ull
                : scenario == 3                                   ? 0
                                                                  : full;
    const auto input = inputs(op);
    fixture.seed(input, out->exec);

    // Establish hostile controls and sticky flags before actual instruction execution.
    SavedEnvironment saved;
    std::fesetround(FE_DOWNWARD);
    std::feraiseexcept(FE_DIVBYZERO);
    const int rounding = std::fegetround(), exceptions = std::fetestexcept(FE_ALL_EXCEPT);
#if defined(__x86_64__) || defined(__i386__)
    const unsigned mxcsr = _mm_getcsr();
#endif
    fixture.execute();
    out->fenv_preserved =
        std::fegetround() == rounding && std::fetestexcept(FE_ALL_EXCEPT) == exceptions;
#if defined(__x86_64__) || defined(__i386__)
    out->fenv_preserved &= _mm_getcsr() == mxcsr;
#endif
    out->output_count = wave_size;
    for (unsigned lane = 0; lane < wave_size; ++lane) {
      out->output_words[lane] = fixture.result(lane);
      const uint32_t source = scenario == 6   ? 0x3f0567ecu
                              : scenario == 7 ? 0x3f800000u
                                              : input[lane];
      out->expected_words[lane] = (out->exec & (uint64_t{1} << lane))
                                      ? (op == Operation::Exp ? util::detail::exp::evaluate(source)
                                                              : util::detail::log::evaluate(source))
                                      : (fixture.destination == 0 ? input[lane] : kSentinel);
    }
    if (!out->qualified_model) {
      // Auto deliberately retains legacy math, not the explicit hardware model.
      Fixture legacy(op, scenario, wave_size, ROCJITSU_CODE_TARGET_GFX1200);
      legacy.seed(input, out->exec);
      std::fesetround(FE_DOWNWARD);
      legacy.execute();
      for (unsigned lane = 0; lane < wave_size; ++lane)
        out->expected_words[lane] = legacy.result(lane);
    }
    for (unsigned lane = 0; lane < wave_size; ++lane)
      out->mismatches += out->output_words[lane] != out->expected_words[lane];
    for (unsigned warmup = 0; warmup < (iterations ? 25u : 0u); ++warmup)
      fixture.execute();
    const auto start = Clock::now();
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
      fixture.execute();
      ++out->executed;
    }
    out->elapsed_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    return 0;
  } catch (const std::exception &exception) {
    return error(out, exception.what());
  } catch (...) {
    return error(out, "unknown FP probe exception");
  }
}
