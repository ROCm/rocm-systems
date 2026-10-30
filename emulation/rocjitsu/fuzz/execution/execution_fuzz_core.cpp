// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "execution_fuzz_core.h"

#include "embedded_schema.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/arch/amdgpu/async_mma_policy.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/isa/target_registry.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/gws_device.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "rocjitsu/vm/soc.h"
#include "util/diagnostic.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <exception>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace rocjitsu::execution_fuzz {
namespace {

constexpr uint64_t kCodeBase = 0x10000;
constexpr uint64_t kDataBase = 0x20000;

struct TargetInfo {
  std::string_view name;
  rj_code_arch_t arch;
  rj_code_target_id_t id;
};

constexpr std::array<TargetInfo, static_cast<size_t>(FuzzTarget::Count)> kTargets{{
    {"gfx908", ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_TARGET_INVALID},
    {"gfx90a", ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_TARGET_GFX90A},
    {"gfx942", ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_TARGET_GFX942},
    {"gfx950", ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_TARGET_GFX950},
    {"gfx1010", ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_TARGET_INVALID},
    {"gfx1030", ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_TARGET_INVALID},
    {"gfx1100", ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_TARGET_INVALID},
    {"gfx1151", ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_TARGET_INVALID},
    {"gfx1200", ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1200},
    {"gfx1201", ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201},
    {"gfx1250", ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_TARGET_GFX1250},
    {"gfx1251", ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_TARGET_GFX1251},
}};

const TargetInfo &target_info(FuzzTarget target) {
  const size_t index = static_cast<size_t>(target);
  if (index >= kTargets.size())
    throw std::invalid_argument("invalid execution fuzz target");
  return kTargets[index];
}

class BoundedFuzzAddressSpace final : public amdgpu::AddressSpaceTranslator {
public:
  BoundedFuzzAddressSpace(uint64_t code_size, uint64_t data_size)
      : code_end_(kCodeBase + code_size), data_end_(kDataBase + data_size) {}

  amdgpu::VmTranslationResult translate(uint64_t address, std::size_t size,
                                        amdgpu::VmAccessKind) const override {
    const auto in_range = [&](uint64_t begin, uint64_t end) {
      return size != 0 && address >= begin && address < end && size <= end - address;
    };
    uint64_t end = 0;
    if (in_range(kCodeBase, code_end_))
      end = code_end_;
    else if (in_range(kDataBase, data_end_))
      end = data_end_;
    else
      return {.outcome = amdgpu::VmAccessOutcome::Faulted, .translation = {}};

    return {
        .outcome = amdgpu::VmAccessOutcome::Complete,
        .translation = {.domain = amdgpu::VmMemoryDomain::Local,
                        .address = address,
                        .contiguous_bytes = end - address,
                        .mtype = amdgpu::Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }

private:
  uint64_t code_end_ = 0;
  uint64_t data_end_ = 0;
};

uint64_t next_random(uint64_t &state) {
  state += 0x9e3779b97f4a7c15ULL;
  uint64_t mixed = state;
  mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ULL;
  mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebULL;
  return mixed ^ (mixed >> 31);
}

uint32_t initial_value(uint64_t &seed) {
  constexpr uint32_t edges[] = {0, 1, UINT32_MAX, 0x80000000, 0x7fffffff, 31, 32, 63};
  const uint64_t bits = next_random(seed);
  return (bits & 1) ? static_cast<uint32_t>(bits >> 32) : edges[(bits >> 1) % std::size(edges)];
}

std::string fuzz_config(FuzzTarget target) {
  const TargetInfo &info = target_info(target);
  std::string json = R"({"num_threads":1,"async_helper_threads":0,"vm":{"arch":")";
  json += config::arch_to_string(info.arch);
  json += '"';
  if (info.id != ROCJITSU_CODE_TARGET_INVALID) {
    json += R"(,"target":")";
    json += info.name;
    json += '"';
  }
  json +=
      R"(},"wait_checking":"off","topology":{"root":{"name":"soc","type":"soc","children":[{"name":"vram","type":"gpu_memory"},{"name":"xcd0","type":"xcd","children":[{"name":"l2","type":"l2_cache"},{"name":"cp","type":"command_processor"},{"name":"se0","type":"shader_engine","children":[{"name":"cu0","type":"compute_unit","config":[{"key":"num_wf_slots","value":"1"},{"key":"sgprs_per_wf","value":"128"},{"key":"vgprs_per_wf","value":")";
  json += std::to_string(isa_properties(info.arch).max_addressable_vgprs_per_wf);
  json += R"("},{"key":"lds_size_kb","value":")";
  json += std::to_string(arch_lds_bytes(info.arch) / 1024);
  json += R"("}]}]}]}]},"links":[]}})";
  return json;
}

class ExecutionCounter final : public ExecutionPlugin {
public:
  explicit ExecutionCounter(RunStats &stats)
      : ExecutionPlugin("execution_fuzz_raw_counter"), stats_(stats) {}

  bool observes_after_execute_instruction() const override { return false; }
  bool observes_async_instruction_issued() const override { return true; }
  bool observes_memory_instruction_routing() const override { return false; }
  bool observes_vgpr_reads() const override { return false; }
  bool observes_vgpr_writes() const override { return false; }
  bool observes_scalar_register_writes() const override { return false; }
  bool observes_sgpr_reads() const override { return false; }
  bool observes_memory_routing() const override { return false; }
  bool observes_tensor_dma_memory_access() const override { return false; }
  bool supports_async_instructions() const override { return true; }
  void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &,
                                        amdgpu::Wavefront &) override {
    ++stats_.executed;
  }
  void onAmdgpuAsyncInstructionIssued(uint64_t, const Instruction &, amdgpu::Wavefront &) override {
    ++stats_.executed;
    ++stats_.async_executed;
  }
  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &) override { ++stats_.waves_halted; }

private:
  RunStats &stats_;
};

template <typename Encodings>
std::vector<std::vector<uint8_t>> make_instruction_seeds(FuzzTarget target,
                                                         const Encodings &encodings) {
  std::vector<std::vector<uint8_t>> result;
  const TargetInfo &info = target_info(target);
  auto decoder = info.id == ROCJITSU_CODE_TARGET_INVALID
                     ? Decoder::create(default_isa_target_registry(), info.arch)
                     : Decoder::create(default_isa_target_registry(), info.id);
  if (!decoder)
    throw std::logic_error(std::format("{} decoder is unavailable", info.name));
  for (const auto &encoding : encodings) {
    // Keep generated campaigns moving forward; arbitrary raw campaigns still
    // cover control flow and termination encodings.
    if (encoding.mnemonic.starts_with("s_branch") || encoding.mnemonic.starts_with("s_cbranch") ||
        encoding.mnemonic.starts_with("s_endpgm") || encoding.mnemonic.starts_with("s_setpc") ||
        encoding.mnemonic.starts_with("s_swappc") || encoding.mnemonic.starts_with("s_rfe") ||
        encoding.mnemonic.starts_with("s_trap"))
      continue;
    std::array<uint32_t, 4> words = {encoding.words[0], encoding.words[1], 0, 0};
    util::StringDiagnostic rejection;
    DecodeResult decoded = decoder->decode(words.data(), rejection.emitter());
    if (decoded.failed())
      continue;
    const int size = decoded.value()->size();
    if (size <= 0 || size > 16 || size % 4 != 0)
      continue;
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    for (int byte = 0; byte < size; ++byte)
      bytes[byte] = static_cast<uint8_t>(words[byte / 4] >> (8 * (byte % 4)));
    result.push_back(std::move(bytes));
  }
  return result;
}

const std::vector<std::vector<uint8_t>> &instruction_seeds(FuzzTarget target) {
  static const std::array<std::vector<std::vector<uint8_t>>, static_cast<size_t>(FuzzTarget::Count)>
      seeds = {
          make_instruction_seeds(FuzzTarget::Gfx908, cdna1::test_data::ENCODINGS),
          make_instruction_seeds(FuzzTarget::Gfx90a, cdna2::test_data::ENCODINGS),
          make_instruction_seeds(FuzzTarget::Gfx942, cdna3::test_data::ENCODINGS),
          make_instruction_seeds(FuzzTarget::Gfx950, cdna4::test_data::ENCODINGS),
          make_instruction_seeds(FuzzTarget::Gfx1010, rdna1::test_data::ENCODINGS),
          make_instruction_seeds(FuzzTarget::Gfx1030, rdna2::test_data::ENCODINGS),
          make_instruction_seeds(FuzzTarget::Gfx1100, rdna3::test_data::ENCODINGS),
          make_instruction_seeds(FuzzTarget::Gfx1151, rdna3_5::test_data::ENCODINGS),
          make_instruction_seeds(FuzzTarget::Gfx1200, rdna4::test_data::ENCODINGS),
          make_instruction_seeds(FuzzTarget::Gfx1201, rdna4::test_data::ENCODINGS),
          make_instruction_seeds(FuzzTarget::Gfx1250, cdna5::test_data::ENCODINGS),
          make_instruction_seeds(FuzzTarget::Gfx1251, cdna5::test_data::ENCODINGS),
      };
  return seeds[static_cast<size_t>(target)];
}

uint32_t terminating_instruction(FuzzTarget target) {
  switch (target_info(target).arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::build_sopp(cdna1::kSEndpgmSopp)[0];
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::build_sopp(cdna2::kSEndpgmSopp)[0];
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_sopp(cdna3::kSEndpgmSopp)[0];
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::build_sopp(cdna4::kSEndpgmSopp)[0];
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::build_sopp(rdna1::kSEndpgmSopp)[0];
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::build_sopp(rdna2::kSEndpgmSopp)[0];
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::build_sopp(rdna3::kSEndpgmSopp)[0];
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::build_sopp(rdna3_5::kSEndpgmSopp)[0];
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::build_sopp(rdna4::kSEndpgmSopp)[0];
  case ROCJITSU_CODE_ARCH_CDNA5:
    return cdna5::build_sopp(cdna5::kSEndpgmSopp)[0];
  default:
    throw std::logic_error("execution fuzz target has no terminating instruction");
  }
}

} // namespace

std::optional<FuzzTarget> parse_target(std::string_view name) {
  for (size_t index = 0; index < kTargets.size(); ++index)
    if (kTargets[index].name == name)
      return static_cast<FuzzTarget>(index);
  return std::nullopt;
}

std::string_view target_name(FuzzTarget target) { return target_info(target).name; }

bool targets_cover_amdgpu_execution_registry() {
  for (const IsaTargetDescriptor &descriptor : default_isa_target_registry().targets()) {
    if (!descriptor.supports_execution)
      continue;
    if (descriptor.architecture_id == ROCJITSU_CODE_ARCH_RV32I ||
        descriptor.architecture_id == ROCJITSU_CODE_ARCH_RV64I)
      continue;
    if (descriptor.gpu_targets.empty()) {
      const bool covered = std::ranges::any_of(kTargets, [&](const TargetInfo &target) {
        return target.arch == descriptor.architecture_id;
      });
      if (!covered)
        return false;
      continue;
    }
    for (const IsaGpuTargetDescription &gpu : descriptor.gpu_targets) {
      if (!gpu.capabilities.execution_implemented)
        continue;
      const bool covered = std::ranges::any_of(
          kTargets, [&](const TargetInfo &target) { return target.id == gpu.public_id; });
      if (!covered)
        return false;
    }
  }
  return true;
}

std::optional<FuzzTarget> indexed_target(uint8_t index) {
  if (index >= kTargets.size())
    return std::nullopt;
  return static_cast<FuzzTarget>(index);
}

namespace {
RunStats run_input_once(
    std::span<const uint8_t> input, std::span<const uint8_t> instructions,
    FuzzTarget resolved_target,
    const std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources> &async_resources) {
  std::vector<uint8_t> code(instructions.begin(), instructions.end());
  while (code.size() % sizeof(uint32_t) != 0)
    code.push_back(0);
  const uint32_t end = terminating_instruction(resolved_target);
  for (unsigned byte = 0; byte < 4; ++byte)
    code.push_back(static_cast<uint8_t>(end >> (8 * byte)));
  code.resize(code.size() + 16);

  // Keep a mapped data island near the program. Address-bearing initial state
  // points into this region while other registers retain high-entropy values.
  std::array<uint8_t, 64 * 1024> data{};
  uint64_t seed = 0x4f1bbcdc6762c73bULL;
  for (uint8_t byte : input)
    seed = std::rotl(seed ^ byte, 11) * 0x9e3779b97f4a7c15ULL;
  seed ^= static_cast<uint64_t>(resolved_target) * 0xd6e8feb86659fd93ULL;
  for (auto &byte : data)
    byte = static_cast<uint8_t>(next_random(seed));

  util::StringDiagnostic diagnostic;
  auto maybe_loaded = config::load_config_from_string(fuzz_config(resolved_target), kEmbeddedSchema,
                                                      diagnostic.emitter(), 1, async_resources);
  if (maybe_loaded.failed())
    throw std::runtime_error("execution fuzz config failed: " + diagnostic.message());
  auto loaded = std::move(maybe_loaded).value();
  auto *memory = loaded.memory();
  auto *soc = loaded.soc();
  auto *cu = soc->xcd(0)->shader_engine(0)->compute_unit(0);
  memory->load_image(code.data(), code.size(), kCodeBase);
  memory->load_image(data.data(), data.size(), kDataBase);
  amdgpu::GpuVm gpu_vm;
  auto translator =
      std::make_shared<BoundedFuzzAddressSpace>(code.size(), static_cast<uint64_t>(data.size()));
  auto physical_memory = std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(*memory);
  const auto address_space =
      gpu_vm.register_address_space(1, translator, physical_memory, [](uint64_t, auto) {});
  if (!address_space)
    throw std::runtime_error("execution fuzz address-space registration failed");
  cu->set_gpu_vm(&gpu_vm);
  RunStats stats;
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  if (!group->add(std::make_unique<ExecutionCounter>(stats)))
    throw std::logic_error("cannot attach execution fuzz counter");
  cu->set_plugin_group(group);

  const TargetInfo &info = target_info(resolved_target);
  const uint32_t vgprs_per_wf = isa_properties(info.arch).max_addressable_vgprs_per_wf;
  auto *wf = cu->dispatch_wf(0, kCodeBase, 128, vgprs_per_wf);
  if (!wf)
    throw std::runtime_error("execution fuzz wave allocation failed");
  wf->set_process_id(1);
  // Arbitrary streams may contain a GWS barrier without a preceding GWS INIT.
  // A compute unit without a CP or engine rejects such a barrier by throwing.
  // Arm every process-scoped resource at zero so fuzzing starts from a defined
  // state and can exercise GWS instructions. This deliberately trades detection
  // of missing guest setup for that coverage; a direct compute-unit test still
  // covers the rejection.
  for (uint32_t rid = 0; rid < amdgpu::GwsDevice::kResourceCount; ++rid)
    cu->gws_init(*wf, rid, 0);
  wf->set_address_space(address_space);
  wf->set_lds_size(arch_lds_bytes(info.arch));
  const uint64_t initial_exec = next_random(seed) | 1;
  const uint8_t control = input.size() >= kHeaderSize ? input[1] : 0;
  const bool full_exec = control & (kAsyncSelectBit | kFullExecMaskBit);
  wf->set_exec_raw(full_exec ? (wf->wf_size() == 64 ? ~uint64_t{0} : 0xffffffffu) : initial_exec);
  wf->set_vcc_raw(next_random(seed));
  wf->write_scc(next_random(seed) & 1);
  // Seed eight complete 64-bit address pairs; each high half is zero.
  for (uint32_t reg = 0; reg < 128; ++reg) {
    const uint32_t value =
        reg < 16 ? (reg % 2 == 0 ? static_cast<uint32_t>(kDataBase + (reg / 2) * 256) : 0)
                 : initial_value(seed);
    wf->debug_write_sgpr(reg, value);
  }
  for (uint32_t reg = 0; reg < vgprs_per_wf; ++reg) {
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      const uint32_t value =
          reg < 16
              ? (reg % 2 == 0 ? static_cast<uint32_t>(kDataBase + (reg / 2) * 256 + lane * 4) : 0)
              : initial_value(seed);
      wf->debug_write_vgpr(reg, lane, value);
    }
  }

  constexpr uint32_t kStepBudget = 256;
  uint32_t steps = 0;
  while (steps++ < kStepBudget && cu->has_active_wfs())
    cu->step();
  if (cu->has_active_wfs())
    ++stats.budget_exhausted;
  else if (stats.waves_halted != 0)
    ++stats.inputs_halted;
  else
    ++stats.inputs_stopped;
  return stats;
}
} // namespace

RunStats run_input(std::span<const uint8_t> input, std::optional<FuzzTarget> target,
                   std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources> async_resources,
                   AsyncMode async_mode, std::optional<unsigned> issuers) {
  if (input.empty() || input.size() > kMaxInputSize)
    throw std::invalid_argument("execution fuzz input must contain 1..1024 bytes");
  if (issuers && (*issuers == 0 || *issuers > 8))
    throw std::invalid_argument("execution fuzz issuers must be 1..8");
  if (async_mode == AsyncMode::Enabled && !async_resources)
    throw std::invalid_argument("enabled asynchronous execution requires helper resources");

  const std::optional<FuzzTarget> selected = target ? target : indexed_target(input.front());
  if (!selected)
    return RunStats{.inputs = 1};
  const FuzzTarget resolved_target = *selected;
  const uint8_t control = input.size() >= kHeaderSize ? input[1] : 0;
  const std::span<const uint8_t> instructions = input.subspan(std::min(kHeaderSize, input.size()));
  const auto arch = target_info(resolved_target).arch;
  const bool async_capable = amdgpu::async_mma_policy::supported(arch);
  const bool async_selected = async_mode == AsyncMode::Enabled ||
                              (async_mode == AsyncMode::Auto && (control & kAsyncSelectBit));
  const bool async_enabled = async_capable && async_selected && async_resources;
  if (!async_enabled)
    async_resources.reset();
  const unsigned issuer_count =
      issuers.value_or((control & kConcurrentIssuerMask) == kConcurrentIssuerMask ? 2 : 1);

  std::vector<RunStats> results(issuer_count);
  std::vector<std::exception_ptr> failures(issuer_count);
  std::vector<std::jthread> workers;
  workers.reserve(issuer_count > 1 ? issuer_count : 0);
  const auto execute = [&](unsigned issuer) {
    try {
      results[issuer] = run_input_once(input, instructions, resolved_target, async_resources);
    } catch (...) {
      failures[issuer] = std::current_exception();
    }
  };
  if (issuer_count == 1) {
    execute(0);
  } else {
    for (unsigned issuer = 0; issuer != issuer_count; ++issuer)
      workers.emplace_back(execute, issuer);
    for (auto &worker : workers)
      worker.join();
  }
  for (const auto &failure : failures)
    if (failure)
      std::rethrow_exception(failure);

  RunStats total;
  bool halted = false;
  bool stopped = false;
  for (const RunStats &result : results) {
    total.executed += result.executed;
    total.async_executed += result.async_executed;
    total.waves_halted += result.waves_halted;
    total.budget_exhausted += result.budget_exhausted;
    halted |= result.inputs_halted != 0;
    stopped |= result.inputs_stopped != 0;
  }
  total.async_capable_inputs = async_capable;
  total.async_enabled_inputs = async_enabled;
  total.concurrent_inputs = issuer_count > 1;
  total.issuer_executions = issuer_count;
  total.rounds = 1;
  total.inputs = 1;
  total.inputs_executed = total.executed != 0;
  total.budget_exhausted = total.budget_exhausted != 0;
  if (total.budget_exhausted == 0 && halted)
    total.inputs_halted = 1;
  else if (total.budget_exhausted == 0 && stopped)
    total.inputs_stopped = 1;
  return total;
}

std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources>
make_async_execution_resources(unsigned helpers) {
  if (helpers == 0 || helpers > 128)
    throw std::invalid_argument("async helper count must be 1..128");
  return std::make_shared<amdgpu::matrix_coexecution::ExecutionResources>(helpers);
}

std::vector<uint8_t> random_input(uint64_t seed, size_t size, std::optional<FuzzTarget> target) {
  if (size == 0 || size > kMaxInputSize)
    throw std::invalid_argument("execution fuzz input size must be 1..1024");
  // The final size is known. Fill in place to avoid GCC 14/15's -O3 false
  // positive -Wfree-nonheap-object in vector's reserve/push_back reallocation path.
  std::vector<uint8_t> result(size);
  size_t cursor = std::min(size, kHeaderSize);
  const uint64_t mode_seed = seed;
  const FuzzTarget resolved_target =
      target.value_or(static_cast<FuzzTarget>(static_cast<uint8_t>(seed) % kTargets.size()));
  result[0] = static_cast<uint8_t>(resolved_target);
  constexpr std::array first_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p,
                         {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3});
  constexpr std::array second_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p,
                         {.vdst = 96, .src0 = 256, .src1 = 288, .src2 = 352, .opsel_hi = 3});
  constexpr size_t wmma_prefix_size = (first_wmma.size() + second_wmma.size()) * sizeof(uint32_t);
  const bool async_wmma_bias = target_info(resolved_target).arch == ROCJITSU_CODE_ARCH_CDNA5 &&
                               (seed & 15) == 1 && size - cursor >= wmma_prefix_size;
  size_t protected_prefix = cursor;
  if (async_wmma_bias) {
    auto append = [&](const auto &words) {
      for (uint32_t word : words)
        for (unsigned byte = 0; byte < sizeof(word); ++byte)
          result[cursor++] = static_cast<uint8_t>(word >> (8 * byte));
    };
    append(first_wmma);
    append(second_wmma);
    assert(cursor - protected_prefix == wmma_prefix_size);
    protected_prefix = cursor;
  }
  // Preserve fully arbitrary coverage while biasing most cases toward long
  // sequences that have already crossed the decoder boundary.
  if ((seed & 7) == 0) {
    while (cursor < size)
      result[cursor++] = static_cast<uint8_t>(next_random(seed));
  } else {
    const auto &seeds = instruction_seeds(resolved_target);
    while (cursor < size) {
      const auto &instruction = seeds[next_random(seed) % seeds.size()];
      const size_t count = std::min(instruction.size(), size - cursor);
      std::copy_n(instruction.begin(), count, result.begin() + cursor);
      cursor += count;
    }
    // One sparse mutation per input explores operand and opcode neighborhoods
    // without turning every first instruction into a decode rejection.
    if (protected_prefix < result.size()) {
      const size_t byte = protected_prefix + next_random(seed) % (result.size() - protected_prefix);
      result[byte] ^= static_cast<uint8_t>(uint8_t{1} << (next_random(seed) & 7));
    }
  }
  if (result.size() >= kHeaderSize) {
    uint8_t mode = static_cast<uint8_t>((mode_seed >> 4) & kAsyncSelectBit);
    if (async_wmma_bias)
      mode |= kFullExecMaskBit;
    if (((mode_seed >> 5) & 3) == 3)
      mode |= kConcurrentIssuerMask;
    result[1] = mode;
  }
  return result;
}

} // namespace rocjitsu::execution_fuzz
