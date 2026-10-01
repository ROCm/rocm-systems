// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_wmma_provider_probe.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop3p.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <utility>

namespace {

using namespace rocjitsu;
using Clock = std::chrono::steady_clock;

constexpr uint32_t kWaveSize = 32;
constexpr uint32_t kA = 0;
constexpr uint32_t kB = 32;
constexpr uint32_t kDst = 64;
constexpr uint32_t kC = 96;
constexpr uint32_t kInputRegs = 16;
constexpr uint32_t kAccRegs = 8;

constexpr std::array<uint16_t, 5> kOpcodes = {
    cdna5::kVWmmaF3216x16x32F16Vop3p, cdna5::kVWmmaF1616x16x32F16Vop3p,
    cdna5::kVWmmaF3216x16x32Bf16Vop3p, cdna5::kVWmmaBf1616x16x32Bf16Vop3p,
    cdna5::kVWmmaBf16f3216x16x32Bf16Vop3p};

int error(rj_test_cdna5_wmma_result *out, const char *message) noexcept {
  std::snprintf(out->error, sizeof(out->error), "%s", message);
  return -1;
}

uint32_t packed_input(uint32_t form, uint32_t reg, uint32_t lane, bool nonfinite) {
  constexpr std::array<uint16_t, 4> f16 = {0x3c00, 0x3800, 0xbc00, 0x0000};
  constexpr std::array<uint16_t, 4> bf16 = {0x3f80, 0x3f00, 0xbf80, 0x0000};
  constexpr std::array<uint16_t, 4> f16_special = {0x7c00, 0xfc00, 0x7e00, 0x8000};
  constexpr std::array<uint16_t, 4> bf16_special = {0x7f80, 0xff80, 0x7fc0, 0x8000};
  const auto &values =
      form < 2 ? (nonfinite ? f16_special : f16) : (nonfinite ? bf16_special : bf16);
  const uint16_t lo = values[(reg * 3 + lane) & 3];
  const uint16_t hi = values[(reg * 3 + lane + 2) & 3];
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

uint32_t accumulator(uint32_t form, uint32_t reg, uint32_t lane) {
  if (form == 1 || form == 3)
    return packed_input(form, reg + 1, lane, false);
  constexpr std::array<uint32_t, 4> f32 = {0x3f800000, 0x3f000000, 0xbf800000, 0x00000000};
  return f32[(reg + lane * 3) & 3];
}

void seed_registers(amdgpu::ComputeUnitCore &cu, uint32_t base, uint32_t form, uint32_t scenario) {
  // Distinct signed payloads expose host FMA operand-order differences that
  // canonical NaNs cannot detect. Cover A/B/C priority and signaling inputs.
  if (scenario >= 9) {
    const bool f16 = form < 2;
    const uint16_t one = f16 ? 0x3c00u : 0x3f80u;
    const uint16_t aq = f16 ? 0x7e11u : 0x7fc1u;
    const uint16_t as = f16 ? 0x7c11u : 0x7f81u;
    const uint16_t bq = f16 ? 0xfe22u : 0xffc2u;
    const uint16_t bs = f16 ? 0xfc22u : 0xff82u;
    const uint16_t a = scenario == 9    ? aq
                       : scenario == 10 ? as
                       : scenario == 13 ? aq | 0x8000u
                                        : one;
    const uint16_t b = scenario == 11 || scenario == 13 ? bs : scenario == 12 ? one : bq;
    const bool signaling_c = scenario >= 12;
    const uint16_t c16 =
        form == 1 ? (signaling_c ? 0x7c33u : 0x7e33u) : (signaling_c ? 0x7f83u : 0x7fc3u);
    const uint32_t c = form == 1 || form == 3 ? uint32_t{c16} | (uint32_t{c16} << 16)
                                              : (signaling_c ? 0x7f833333u : 0x7fc33333u);
    for (uint32_t reg = 0; reg < kInputRegs; ++reg)
      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        cu.write_vgpr(base + kA + reg, lane, uint32_t{a} | (uint32_t{a} << 16));
        cu.write_vgpr(base + kB + reg, lane, uint32_t{b} | (uint32_t{b} << 16));
      }
    for (uint32_t reg = 0; reg < kAccRegs; ++reg)
      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        cu.write_vgpr(base + kC + reg, lane, c);
        cu.write_vgpr(base + kDst + reg, lane, 0);
      }
    return;
  }
  for (uint32_t reg = 0; reg < kInputRegs; ++reg)
    for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
      cu.write_vgpr(base + kA + reg, lane, packed_input(form, reg, lane, scenario == 4));
      cu.write_vgpr(base + kB + reg, lane, packed_input(form, reg + 5, lane, scenario == 4));
    }
  for (uint32_t reg = 0; reg < kAccRegs; ++reg)
    for (uint32_t lane = 0; lane < kWaveSize; ++lane)
      cu.write_vgpr(base + kC + reg, lane, accumulator(form, reg, lane));
  // Inline +1.0 C must not pass if an implementation accidentally reads D as
  // its accumulator. Seed D with +2.0; keep the timed case's zero-initialized D.
  const uint32_t dst_initial =
      scenario == 8 ? (form == 0 || form == 2 ? 0x40000000u : 0x40004000u) : 0u;
  if (scenario != 1 && scenario != 2 && scenario != 3)
    for (uint32_t reg = 0; reg < kAccRegs; ++reg)
      for (uint32_t lane = 0; lane < kWaveSize; ++lane)
        cu.write_vgpr(base + kDst + reg, lane, dst_initial);
}

template <typename T> const cdna5::Operand *decoded_src2(Instruction *inst) {
  const auto *wmma = dynamic_cast<const T *>(inst);
  return wmma ? &wmma->src2 : nullptr;
}

const cdna5::Operand *decoded_src2(Instruction *inst, uint32_t form) {
  switch (form) {
  case 0:
    return decoded_src2<cdna5::VWmmaF3216x16x32F16Vop3p>(inst);
  case 1:
    return decoded_src2<cdna5::VWmmaF1616x16x32F16Vop3p>(inst);
  case 2:
    return decoded_src2<cdna5::VWmmaF3216x16x32Bf16Vop3p>(inst);
  case 3:
    return decoded_src2<cdna5::VWmmaBf1616x16x32Bf16Vop3p>(inst);
  case 4:
    return decoded_src2<cdna5::VWmmaBf16f3216x16x32Bf16Vop3p>(inst);
  default:
    return nullptr;
  }
}

class VgprObservationPlugin final : public ExecutionPlugin {
public:
  VgprObservationPlugin(const amdgpu::Wavefront &wf, rj_test_cdna5_wmma_result &result)
      : ExecutionPlugin("cdna5_wmma_provider_observer"), wf_(&wf), result_(result) {}

  void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t reg, uint64_t lanes,
                             uint8_t bytes) override {
    record(wf, reg, lanes, bytes, result_.read_lanes, result_.read_bytes);
  }

  void onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *wf, uint32_t reg, uint64_t lanes,
                              uint8_t bytes) override {
    record(wf, reg, lanes, bytes, result_.write_lanes, result_.write_bytes);
  }

  bool valid() const { return valid_; }

private:
  void record(const amdgpu::Wavefront *wf, uint32_t reg, uint64_t lanes, uint8_t bytes,
              uint64_t *recorded_lanes, uint8_t *recorded_bytes) {
    const uint32_t base = wf_->vgpr_alloc().base;
    if (wf != wf_ || reg < base || reg - base >= RJ_TEST_WMMA_MAX_VGPRS) {
      valid_ = false;
      return;
    }
    recorded_lanes[reg - base] |= lanes;
    recorded_bytes[reg - base] |= bytes;
  }

  const amdgpu::Wavefront *wf_;
  rj_test_cdna5_wmma_result &result_;
  bool valid_ = true;
};

int run_probe(uint32_t form, uint32_t scenario, uint32_t iterations, bool observe,
              rj_test_cdna5_wmma_result *out) noexcept {
  if (!out)
    return -1;
  std::memset(out, 0, sizeof(*out));
  if (form >= kOpcodes.size() || scenario > 14)
    return error(out, "invalid form or scenario");
  if (scenario >= 5 && scenario <= 7 && (form == 1 || form == 3))
    return error(out, "C modifier is not applicable to packed-output form");
  if (scenario != 0 && iterations != 0)
    return error(out, "only independent-operand scenario can be timed");

  try {
    amdgpu::GpuMemory memory("cdna5_wmma_provider_probe_mem");
    amdgpu::L2Cache l2("cdna5_wmma_provider_probe_l2");
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 256;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("cdna5_wmma_provider_probe", cfg, &memory, &l2);
    if (!cu)
      return error(out, "failed to create CDNA5 compute unit");
    auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
    if (!wf)
      return error(out, "failed to dispatch wavefront");

    const uint32_t dst = scenario == 1 ? kA : scenario == 2 ? kB : scenario == 3 ? kC : kDst;
    seed_registers(*cu, wf->vgpr_alloc().base, form, scenario);
    const auto words = cdna5::build_vop3p(
        kOpcodes[form], {.vdst = static_cast<uint8_t>(dst),
                         .neg_hi = static_cast<uint8_t>(scenario == 6 || scenario == 7 ? 4 : 0),
                         .src0 = static_cast<uint16_t>(256 + kA),
                         .src1 = static_cast<uint16_t>(256 + kB),
                         .src2 = static_cast<uint16_t>(scenario == 8 ? 242 : 256 + kC),
                         .neg = static_cast<uint8_t>(scenario == 5 || scenario == 7 ? 4 : 0)});

    const auto decode_start = Clock::now();
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    if (!decoder)
      return error(out, "failed to create CDNA5 decoder");
    auto decoded = decoder->decode(words.data());
    if (decoded.failed())
      return error(out, "failed to decode CDNA5 WMMA instruction");
    std::unique_ptr<Instruction> inst = std::move(decoded.value());
    const auto decode_end = Clock::now();
    out->first_decode_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(decode_end - decode_start).count());
    if (!inst || !inst->execute)
      return error(out, "decoded instruction has no execute callback");
    if (scenario == 8) {
      const auto *src2 = decoded_src2(inst.get(), form);
      if (!src2 || src2->encoding_value() != 242 || src2->const_value() != 0x3f800000u)
        return error(out, "inline +1.0 C did not decode as selector 242");
    }
    out->callback_addr = reinterpret_cast<uintptr_t>(inst->execute);

    std::shared_ptr<ExecutionPluginGroup> observation_group;
    VgprObservationPlugin *observer = nullptr;
    if (observe) {
      if (scenario == 0)
        for (uint32_t reg = 0; reg < kAccRegs; ++reg)
          for (uint32_t lane = 0; lane < kWaveSize; ++lane)
            cu->write_vgpr(wf->vgpr_alloc().base + dst + reg, lane,
                           0xDEAD0000u | (reg << 8) | lane);
      observation_group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
      auto plugin = std::make_unique<VgprObservationPlugin>(*wf, *out);
      observer = plugin.get();
      if (!observation_group->add(std::move(plugin)))
        return error(out, "failed to add WMMA observation plugin");
      cu->set_plugin_group(observation_group);
      observation_group->onInit();
    }
    if (!cu->execute_instruction(inst.get(), *wf).succeeded())
      return error(out, "first WMMA execution failed");
    if (observer) {
      if (!observer->valid())
        return error(out, "observed VGPR access is outside the probe wavefront");
      // Output inspection uses the VM's raw read API and must not become an
      // instruction observation. Timing also keeps the original observer-free path.
      cu->set_plugin_group(nullptr);
      observation_group->onShutdown();
    }
    const uint32_t output_regs = form == 0 || form == 2 ? 8 : 4;
    out->output_count = output_regs * kWaveSize;
    for (uint32_t reg = 0; reg < output_regs; ++reg)
      for (uint32_t lane = 0; lane < kWaveSize; ++lane)
        out->output_words[reg * kWaveSize + lane] =
            cu->read_vgpr(wf->vgpr_alloc().base + dst + reg, lane);

    if (iterations == 0)
      return 0;
    for (int i = 0; i < 25; ++i)
      if (!cu->execute_instruction(inst.get(), *wf).succeeded())
        return error(out, "WMMA warmup execution failed");
    const auto start = Clock::now();
    for (uint32_t i = 0; i < iterations; ++i) {
      if (!cu->execute_instruction(inst.get(), *wf).succeeded())
        return error(out, "timed WMMA execution failed");
      ++out->executed;
    }
    const auto end = Clock::now();
    out->elapsed_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    return 0;
  } catch (const std::exception &e) {
    return error(out, e.what());
  } catch (...) {
    return error(out, "unknown WMMA probe exception");
  }
}

} // namespace

extern "C" RJ_API_EXPORT int rj_test_cdna5_wmma_probe(uint32_t form, uint32_t scenario,
                                                      uint32_t iterations,
                                                      rj_test_cdna5_wmma_result *out) noexcept {
  return run_probe(form, scenario, iterations, false, out);
}

extern "C" RJ_API_EXPORT int rj_test_cdna5_wmma_observe(uint32_t form, uint32_t scenario,
                                                        rj_test_cdna5_wmma_result *out) noexcept {
  return run_probe(form, scenario, 0, true, out);
}
