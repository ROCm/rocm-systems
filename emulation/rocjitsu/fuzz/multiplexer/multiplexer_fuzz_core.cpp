// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "multiplexer_fuzz_core.h"

#include "code_object_fuzz_core.h"
#include "concurrent_dispatch_fuzz_core.h"
#include "decode_fuzz_core.h"
#include "execution_fuzz_core.h"

#include "rocjitsu/isa/decoder.h"

#include <array>
#include <memory>
#include <optional>

namespace rocjitsu::multiplexer_fuzz {
namespace {

constexpr std::array kTargets = {
    TargetDescriptor{FuzzTarget::Decode, "decode"},
    TargetDescriptor{FuzzTarget::Execution, "execution"},
    TargetDescriptor{FuzzTarget::CodeObject, "code-object"},
    TargetDescriptor{FuzzTarget::ConcurrentDispatch, "concurrent-dispatch"},
};

util::FailureOr<uint64_t> run_decode(std::span<const uint8_t> payload,
                                     util::DiagnosticEmitter diagnostic) {
  if (payload.size() != decode_fuzz::kWindowSize)
    return 0;
  thread_local util::FailureOr<std::unique_ptr<Decoder>> decoder =
      decode_fuzz::create_decoder("gfx1250");
  if (decoder.failed())
    return diagnostic.emit() << "multiplexer decode harness failed";
  const util::FailureOr<decode_fuzz::DecodeRecord> record =
      decode_fuzz::decode_window(*decoder.value(), payload);
  if (record.failed())
    return diagnostic.emit() << "multiplexer decode harness failed";
  return record.value().valid;
}

util::FailureOr<uint64_t> run_execution(std::span<const uint8_t> payload,
                                        util::DiagnosticEmitter diagnostic) {
  if (payload.empty() || payload.size() > execution_fuzz::kMaxInputSize)
    return 0;
  thread_local util::FailureOr<std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources>>
      resources = execution_fuzz::make_async_execution_resources(1);
  if (resources.failed())
    return diagnostic.emit() << "multiplexer execution resources failed";
  const util::FailureOr<execution_fuzz::RunStats> stats =
      execution_fuzz::run_input(payload, std::nullopt, resources.value(),
                                execution_fuzz::AsyncMode::Auto, std::nullopt, diagnostic);
  if (stats.failed())
    return util::Result::failure();
  return stats.value().executed;
}

util::FailureOr<uint64_t> run_code_object(std::span<const uint8_t> payload,
                                          util::DiagnosticEmitter diagnostic) {
  static_assert(kMaxInputSize - 1 <= code_object_fuzz::kMaxInputSize);
  const util::FailureOr<code_object_fuzz::RunStats> result =
      code_object_fuzz::run_input(payload, diagnostic);
  if (result.failed())
    return util::Result::failure();
  const code_object_fuzz::RunStats &stats = result.value();
  return stats.code_objects_valid + stats.executables_valid;
}

uint64_t run_concurrent_dispatch(std::span<const uint8_t> payload) {
  if (payload.empty() || payload.size() > concurrent_dispatch_fuzz::kMaxInputSize)
    return 0;
  return concurrent_dispatch_fuzz::run_input(payload).instructions;
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

util::FailureOr<std::optional<DispatchResult>> run_input(std::span<const uint8_t> input,
                                                         util::DiagnosticEmitter diagnostic) {
  if (input.empty() || input.size() > kMaxInputSize)
    return diagnostic.emit() << "multiplexer fuzz input has an invalid size";
  const std::optional<FuzzTarget> target = selected_target(input.front());
  if (!target)
    return std::nullopt;
  const std::span<const uint8_t> payload = input.subspan(1);
  util::FailureOr<uint64_t> work_units = util::Result::failure();
  switch (*target) {
  case FuzzTarget::Decode:
    work_units = run_decode(payload, diagnostic);
    break;
  case FuzzTarget::Execution:
    work_units = run_execution(payload, diagnostic);
    break;
  case FuzzTarget::CodeObject:
    work_units = run_code_object(payload, diagnostic);
    break;
  case FuzzTarget::ConcurrentDispatch:
    work_units = run_concurrent_dispatch(payload);
    break;
  }
  if (work_units.failed())
    return util::Result::failure();
  return std::optional<DispatchResult>{{*target, work_units.value()}};
}

util::FailureOr<std::vector<std::vector<uint8_t>>> seed_corpus(util::DiagnosticEmitter diagnostic) {
  std::vector<std::vector<uint8_t>> seeds;

  std::vector<uint8_t> decode_seed(1 + decode_fuzz::kWindowSize, 0);
  decode_seed[0] = static_cast<uint8_t>(FuzzTarget::Decode);
  constexpr uint32_t kSNop = 0xBF800000u;
  for (unsigned byte = 0; byte < sizeof(kSNop); ++byte)
    decode_seed[1 + byte] = static_cast<uint8_t>(kSNop >> (8 * byte));
  seeds.push_back(std::move(decode_seed));

  util::FailureOr<std::vector<uint8_t>> maybe_execution_seed =
      execution_fuzz::random_input(17, 32, execution_fuzz::FuzzTarget::Gfx1250, diagnostic);
  if (maybe_execution_seed.failed())
    return util::Result::failure();
  std::vector<uint8_t> execution_seed = std::move(maybe_execution_seed).value();
  execution_seed.insert(execution_seed.begin(), static_cast<uint8_t>(FuzzTarget::Execution));
  seeds.push_back(std::move(execution_seed));

  util::FailureOr<std::vector<uint8_t>> maybe_kernel_seed = execution_fuzz::generated_kernel_input(
      0, execution_fuzz::kGeneratedKernelSeedSize, execution_fuzz::FuzzTarget::Gfx1250, diagnostic);
  if (maybe_kernel_seed.failed())
    return util::Result::failure();
  std::vector<uint8_t> kernel_seed = std::move(maybe_kernel_seed).value();
  kernel_seed.insert(kernel_seed.begin(), static_cast<uint8_t>(FuzzTarget::Execution));
  seeds.push_back(std::move(kernel_seed));

  for (std::vector<uint8_t> code_object_seed : code_object_fuzz::seed_corpus()) {
    code_object_seed.insert(code_object_seed.begin(), static_cast<uint8_t>(FuzzTarget::CodeObject));
    seeds.push_back(std::move(code_object_seed));
  }
  std::vector<uint8_t> concurrent_seed =
      concurrent_dispatch_fuzz::seed_corpus()[static_cast<std::size_t>(
          execution_fuzz::FuzzTarget::Gfx950)];
  concurrent_seed.insert(concurrent_seed.begin(),
                         static_cast<uint8_t>(FuzzTarget::ConcurrentDispatch));
  seeds.push_back(std::move(concurrent_seed));
  return seeds;
}

} // namespace rocjitsu::multiplexer_fuzz
