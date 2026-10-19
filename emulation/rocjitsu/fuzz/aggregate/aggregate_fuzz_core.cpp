// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aggregate_fuzz_core.h"

#include "code_object_fuzz_core.h"
#include "decode_fuzz_core.h"
#include "execution_fuzz_core.h"

#include "rocjitsu/isa/decoder.h"

#include <array>
#include <memory>
#include <optional>
#include <stdexcept>

namespace rocjitsu::aggregate_fuzz {
namespace {

constexpr std::array kTargets = {
    TargetDescriptor{FuzzTarget::Decode, "decode"},
    TargetDescriptor{FuzzTarget::Execution, "execution"},
    TargetDescriptor{FuzzTarget::CodeObject, "code-object"},
};

uint64_t run_decode(std::span<const uint8_t> payload) {
  if (payload.size() != decode_fuzz::kWindowSize)
    return 0;
  thread_local std::unique_ptr<Decoder> decoder = decode_fuzz::create_decoder("gfx1250");
  return decode_fuzz::decode_window(*decoder, payload).valid;
}

uint64_t run_execution(std::span<const uint8_t> payload) {
  if (payload.empty() || payload.size() > execution_fuzz::kMaxInputSize)
    return 0;
  thread_local std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources> resources =
      execution_fuzz::make_async_execution_resources(1);
  const execution_fuzz::RunStats stats =
      execution_fuzz::run_input(payload, std::nullopt, resources);
  return stats.executed + stats.async_executed;
}

uint64_t run_code_object(std::span<const uint8_t> payload) {
  const code_object_fuzz::RunStats stats = code_object_fuzz::run_input(payload);
  return stats.code_objects_valid + stats.executables_valid;
}

} // namespace

std::span<const TargetDescriptor> targets() { return kTargets; }

std::optional<FuzzTarget> selected_target(uint8_t selector) {
  for (const TargetDescriptor &descriptor : kTargets) {
    if (static_cast<uint8_t>(descriptor.id) == selector)
      return descriptor.id;
  }
  return std::nullopt;
}

std::optional<DispatchResult> run_input(std::span<const uint8_t> input) {
  if (input.empty() || input.size() > kMaxInputSize)
    throw std::invalid_argument("aggregate fuzz input has an invalid size");
  const std::optional<FuzzTarget> target = selected_target(input.front());
  if (!target)
    return std::nullopt;
  const std::span<const uint8_t> payload = input.subspan(1);
  switch (*target) {
  case FuzzTarget::Decode:
    return DispatchResult{*target, run_decode(payload)};
  case FuzzTarget::Execution:
    return DispatchResult{*target, run_execution(payload)};
  case FuzzTarget::CodeObject:
    return DispatchResult{*target, run_code_object(payload)};
  }
  throw std::logic_error("aggregate fuzz target registry is inconsistent");
}

std::vector<std::vector<uint8_t>> seed_corpus() {
  std::vector<std::vector<uint8_t>> seeds;

  std::vector<uint8_t> decode_seed(1 + decode_fuzz::kWindowSize, 0);
  decode_seed[0] = static_cast<uint8_t>(FuzzTarget::Decode);
  constexpr uint32_t kSNop = 0xBF800000u;
  for (unsigned byte = 0; byte < sizeof(kSNop); ++byte)
    decode_seed[1 + byte] = static_cast<uint8_t>(kSNop >> (8 * byte));
  seeds.push_back(std::move(decode_seed));

  std::vector<uint8_t> execution_seed =
      execution_fuzz::random_input(17, 32, execution_fuzz::FuzzTarget::Gfx1250);
  execution_seed.insert(execution_seed.begin(), static_cast<uint8_t>(FuzzTarget::Execution));
  seeds.push_back(std::move(execution_seed));

  for (std::vector<uint8_t> code_object_seed : code_object_fuzz::seed_corpus()) {
    code_object_seed.insert(code_object_seed.begin(), static_cast<uint8_t>(FuzzTarget::CodeObject));
    seeds.push_back(std::move(code_object_seed));
  }
  return seeds;
}

} // namespace rocjitsu::aggregate_fuzz
