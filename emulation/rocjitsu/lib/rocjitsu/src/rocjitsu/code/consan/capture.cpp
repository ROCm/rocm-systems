// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/capture.h"
#include "rocjitsu/code/consan/encoding.h"

#include <algorithm>
#include <cstring>
#include <map>

namespace rocjitsu::consan {
namespace {

// Identify the capture header, each execution frame, and the closing End trailer.
constexpr std::array<uint8_t, 8> kMagic{'R', 'J', 'C', 'S', 'C', 'A', 'P', '0'};
constexpr std::array<uint8_t, 8> kExecutionMagic{'R', 'J', 'C', 'S', 'E', 'X', 'E', 'C'};
constexpr std::array<uint8_t, 8> kEndMagic{'R', 'J', 'C', 'S', 'E', 'N', 'D', '0'};
constexpr uint32_t kHeaderBytes = 64;
constexpr uint32_t kFrameBytes = 48;
constexpr uint32_t kExecutionBytes = 144;
constexpr uint32_t kEndBytes = 64;

Result reject(ContractError *error, ContractErrorCode code, const char *message) {
  if (error != nullptr)
    *error = {code, message};
  return Result::failure();
}

template <size_t Count>
bool matches(std::span<const uint8_t> bytes, const std::array<uint8_t, Count> &expected) {
  return bytes.size() >= Count && std::equal(expected.begin(), expected.end(), bytes.begin());
}

bool all_zero(std::span<const uint8_t> bytes) {
  return std::ranges::all_of(bytes, [](uint8_t value) { return value == 0; });
}

bool valid_header(const CaptureHeader &header) {
  return header.execution_conditions == kSupportedExecutionConditions &&
         !all_zero(header.run_identity);
}

Result validate_execution(const Manifest &manifest, const CapturedExecution &execution,
                          ContractError *error, const CaptureLimits &limits) {
  const auto layout = plan_launch(manifest, execution.dimensions, execution.group_bytes,
                                  execution.private_bytes, error, limits.contract);
  if (layout.failed())
    return Result::failure();
  if (execution.arguments.size() > limits.argument_bytes)
    return reject(error, ContractErrorCode::ResourceLimit, "capture argument budget exceeded");
  if (execution.serial == 0 || execution.storage_slot == 0 || execution.storage_generation == 0 ||
      execution.completion != kCompletionVisibleForwarded ||
      execution.arguments.size() != manifest.entry.original_kernarg_bytes ||
      execution.report.size() != layout.value().report_bytes ||
      execution.group_bytes != layout.value().group_bytes ||
      execution.private_bytes != layout.value().private_bytes)
    return reject(error, ContractErrorCode::InvalidFacts,
                  "invalid execution ownership, extent or completion");
  const auto &context = execution.context;
  const auto address = wire::read_u64(context, wire::kContextReportPointer);
  if (wire::read_u64(context, wire::kContextSerial) != execution.serial ||
      wire::read_u32(context, wire::kContextEntry) != manifest.entry.id ||
      !all_zero(std::span(context).subspan(12, 4)) || !address || *address == 0 ||
      (*address % 64) != 0 || execution.report.size() > UINT64_MAX - *address ||
      wire::read_u64(context, wire::kContextReportBytes) != execution.report.size() ||
      wire::read_u32(context, wire::kContextGroupBytes) != execution.group_bytes ||
      wire::read_u32(context, wire::kContextPrivateBytes) != execution.private_bytes)
    return reject(error, ContractErrorCode::IdentityMismatch,
                  "context does not match its owning execution");
  for (size_t index = 0; index < 3; ++index) {
    if (wire::read_u32(context, wire::kContextGrid + index * 4) !=
            execution.dimensions.grid[index] ||
        wire::read_u32(context, wire::kContextWorkgroup + index * 4) !=
            execution.dimensions.workgroup[index])
      return reject(error, ContractErrorCode::IdentityMismatch,
                    "context geometry differs from envelope");
  }
  return Result::success();
}

std::vector<uint8_t> encode_execution(const CapturedExecution &execution) {
  std::vector<uint8_t> payload(kExecutionBytes + execution.arguments.size() +
                               execution.report.size());
  wire::write_u64(payload, 0, execution.serial);
  for (size_t index = 0; index < 3; ++index) {
    wire::write_u32(payload, 8 + index * 4, execution.dimensions.grid[index]);
    wire::write_u32(payload, 20 + index * 4, execution.dimensions.workgroup[index]);
  }
  wire::write_u32(payload, 32, execution.group_bytes);
  wire::write_u32(payload, 36, execution.private_bytes);
  wire::write_u32(payload, 40, execution.storage_slot);
  wire::write_u32(payload, 44, execution.storage_generation);
  wire::write_u32(payload, 48, execution.completion);
  wire::write_u32(payload, 56, static_cast<uint32_t>(execution.arguments.size()));
  wire::write_u64(payload, 64, execution.report.size());
  std::copy(execution.context.begin(), execution.context.end(), payload.begin() + 72);
  std::copy(execution.arguments.begin(), execution.arguments.end(),
            payload.begin() + kExecutionBytes);
  std::copy(execution.report.begin(), execution.report.end(),
            payload.begin() + kExecutionBytes + execution.arguments.size());
  std::vector<uint8_t> frame(kFrameBytes + payload.size());
  std::copy(kExecutionMagic.begin(), kExecutionMagic.end(), frame.begin());
  wire::write_u64(frame, 8, payload.size());
  const Digest checksum = digest(payload);
  std::copy(checksum.begin(), checksum.end(), frame.begin() + 16);
  std::copy(payload.begin(), payload.end(), frame.begin() + kFrameBytes);
  return frame;
}

FailureOr<CapturedExecution> decode_execution(std::span<const uint8_t> payload,
                                              const Manifest &manifest, ContractError *error,
                                              const CaptureLimits &limits) {
  if (payload.size() < kExecutionBytes || !all_zero(payload.subspan(52, 4)) ||
      !all_zero(payload.subspan(60, 4)) || !all_zero(payload.subspan(136, 8)))
    return reject(error, ContractErrorCode::InvalidEncoding,
                  "missing execution fields or reserved bytes");
  const uint32_t argument_bytes = *wire::read_u32(payload, 56);
  const uint64_t report_bytes = *wire::read_u64(payload, 64);
  if (argument_bytes > limits.argument_bytes || report_bytes > limits.contract.report_bytes)
    return reject(error, ContractErrorCode::ResourceLimit, "execution payload exceeds budget");
  if (argument_bytes > payload.size() - kExecutionBytes ||
      report_bytes != payload.size() - kExecutionBytes - argument_bytes)
    return reject(error, ContractErrorCode::InvalidEncoding, "execution payload extent mismatch");
  CapturedExecution execution;
  execution.serial = *wire::read_u64(payload, 0);
  for (size_t index = 0; index < 3; ++index) {
    execution.dimensions.grid[index] = *wire::read_u32(payload, 8 + index * 4);
    execution.dimensions.workgroup[index] = *wire::read_u32(payload, 20 + index * 4);
  }
  execution.group_bytes = *wire::read_u32(payload, 32);
  execution.private_bytes = *wire::read_u32(payload, 36);
  execution.storage_slot = *wire::read_u32(payload, 40);
  execution.storage_generation = *wire::read_u32(payload, 44);
  execution.completion = *wire::read_u32(payload, 48);
  std::copy_n(payload.begin() + 72, execution.context.size(), execution.context.begin());
  execution.arguments.assign(payload.begin() + kExecutionBytes,
                             payload.begin() + kExecutionBytes + argument_bytes);
  execution.report.assign(payload.begin() + kExecutionBytes + argument_bytes, payload.end());
  if (validate_execution(manifest, execution, error, limits).failed())
    return Result::failure();
  return execution;
}

} // namespace

FailureOr<std::vector<uint8_t>> CaptureEncoder::start(const CaptureHeader &header,
                                                      ContractError *error) {
  if (error != nullptr)
    *error = {};
  if (started_ || !valid_header(header))
    return reject(error, ContractErrorCode::InvalidFacts,
                  "capture already started or invalid run execution conditions/identity");
  const auto facts = encode_manifest(header.manifest, error);
  if (facts.failed())
    return Result::failure();
  std::vector<uint8_t> prefix(kHeaderBytes + facts.value().size());
  std::copy(kMagic.begin(), kMagic.end(), prefix.begin());
  std::copy(kProducerContract.begin(), kProducerContract.end(), prefix.begin() + 8);
  std::copy(header.run_identity.begin(), header.run_identity.end(), prefix.begin() + 24);
  wire::write_u64(prefix, 40, facts.value().size());
  wire::write_u32(prefix, 48, header.execution_conditions);
  std::copy(facts.value().begin(), facts.value().end(), prefix.begin() + kHeaderBytes);
  header_ = header;
  hash_.update(prefix.data(), prefix.size());
  bytes_ = prefix.size();
  started_ = true;
  return prefix;
}

FailureOr<std::vector<uint8_t>> CaptureEncoder::append(const CapturedExecution &execution,
                                                       ContractError *error) {
  const CaptureLimits limits;
  if (!started_ || finished_ || execution.serial != count_ + 1)
    return reject(error, ContractErrorCode::InvalidFacts,
                  "capture not open or serial is not its next execution");
  if (count_ >= limits.executions)
    return reject(error, ContractErrorCode::ResourceLimit, "capture execution budget exceeded");
  if (validate_execution(header_.manifest, execution, error, limits).failed())
    return Result::failure();
  const auto generation = generations_.find(execution.storage_slot);
  if (generation != generations_.end() && execution.storage_generation <= generation->second)
    return reject(error, ContractErrorCode::InvalidFacts,
                  "storage generation was already captured");
  const uint64_t frame_bytes =
      kFrameBytes + kExecutionBytes + execution.arguments.size() + execution.report.size();
  if (frame_bytes + kEndBytes > limits.file_bytes - bytes_)
    return reject(error, ContractErrorCode::ResourceLimit, "capture file budget exceeded");
  std::vector<uint8_t> frame = encode_execution(execution);
  hash_.update(frame.data(), frame.size());
  generations_[execution.storage_slot] = execution.storage_generation;
  bytes_ += frame.size();
  ++count_;
  return frame;
}

FailureOr<std::vector<uint8_t>> CaptureEncoder::finish(ContractError *error) {
  if (!started_ || finished_)
    return reject(error, ContractErrorCode::InvalidFacts, "capture is not open for End");
  std::vector<uint8_t> end(kEndBytes);
  std::copy(kEndMagic.begin(), kEndMagic.end(), end.begin());
  wire::write_u64(end, 8, count_);
  wire::write_u32(end, 16, header_.execution_conditions);
  hash_.update(end.data(), 32);
  const Digest checksum = hash_.finish();
  std::copy(checksum.begin(), checksum.end(), end.begin() + 32);
  finished_ = true;
  return end;
}

FailureOr<Capture> decode_capture(std::span<const uint8_t> bytes, ContractError *error,
                                  const CaptureLimits &limits) {
  if (error != nullptr)
    *error = {};
  if (bytes.size() > limits.file_bytes)
    return reject(error, ContractErrorCode::ResourceLimit, "capture file budget exceeded");
  if (bytes.size() < kHeaderBytes + kEndBytes || !matches(bytes, kMagic) ||
      !matches(bytes.subspan(bytes.size() - kEndBytes), kEndMagic))
    return reject(error, ContractErrorCode::InvalidEncoding,
                  "capture header or complete End is missing");
  if (!matches(bytes.subspan(8), kProducerContract))
    return reject(error, ContractErrorCode::IncompatibleContract,
                  "capture producer contract mismatch");
  const Digest checksum = digest(bytes.first(bytes.size() - 32));
  if (!std::equal(checksum.begin(), checksum.end(), bytes.end() - 32))
    return reject(error, ContractErrorCode::InvalidEncoding,
                  "capture complete-file checksum mismatch");
  if (!all_zero(bytes.subspan(52, 12)))
    return reject(error, ContractErrorCode::InvalidEncoding, "unknown capture header fields");
  const uint64_t facts_bytes = *wire::read_u64(bytes, 40);
  if (facts_bytes > limits.contract.metadata_bytes)
    return reject(error, ContractErrorCode::ResourceLimit, "capture facts budget exceeded");
  if (facts_bytes > bytes.size() - kHeaderBytes - kEndBytes)
    return reject(error, ContractErrorCode::InvalidEncoding, "capture facts extent is truncated");
  const auto manifest =
      decode_manifest(bytes.subspan(kHeaderBytes, facts_bytes), error, limits.contract);
  if (manifest.failed())
    return Result::failure();
  Capture capture;
  capture.header.manifest = manifest.value();
  std::copy_n(bytes.begin() + 24, capture.header.run_identity.size(),
              capture.header.run_identity.begin());
  capture.header.execution_conditions = *wire::read_u32(bytes, 48);
  if (!valid_header(capture.header))
    return reject(error, ContractErrorCode::InvalidFacts,
                  "unsupported run execution conditions or missing identity");
  const auto end = bytes.last(kEndBytes);
  const uint64_t count = *wire::read_u64(end, 8);
  if (count > limits.executions)
    return reject(error, ContractErrorCode::ResourceLimit,
                  "capture execution count exceeds budget");
  if (*wire::read_u32(end, 16) != capture.header.execution_conditions ||
      !all_zero(end.subspan(20, 12)))
    return reject(error, ContractErrorCode::InvalidFacts,
                  "End does not close these execution conditions/scope");
  size_t offset = kHeaderBytes + facts_bytes;
  std::map<uint32_t, uint32_t> generations;
  while (offset < bytes.size() - kEndBytes) {
    if (capture.executions.size() >= count || bytes.size() - kEndBytes - offset < kFrameBytes ||
        !matches(bytes.subspan(offset), kExecutionMagic))
      return reject(error, ContractErrorCode::InvalidEncoding, "missing or excess execution frame");
    const uint64_t payload_bytes = *wire::read_u64(bytes, offset + 8);
    if (payload_bytes > bytes.size() - kEndBytes - offset - kFrameBytes)
      return reject(error, ContractErrorCode::InvalidEncoding, "truncated execution frame");
    const auto payload = bytes.subspan(offset + kFrameBytes, payload_bytes);
    const Digest frame_hash = digest(payload);
    if (!std::equal(frame_hash.begin(), frame_hash.end(), bytes.begin() + offset + 16))
      return reject(error, ContractErrorCode::InvalidEncoding, "execution frame checksum mismatch");
    auto execution = decode_execution(payload, capture.header.manifest, error, limits);
    if (execution.failed())
      return Result::failure();
    if (execution.value().serial != capture.executions.size() + 1 ||
        execution.value().storage_generation <= generations[execution.value().storage_slot])
      return reject(error, ContractErrorCode::InvalidFacts,
                    "duplicate execution or storage generation");
    generations[execution.value().storage_slot] = execution.value().storage_generation;
    capture.executions.push_back(std::move(execution.value()));
    offset += kFrameBytes + payload_bytes;
  }
  if (capture.executions.size() != count || offset != bytes.size() - kEndBytes)
    return reject(error, ContractErrorCode::InvalidEncoding, "End membership count is incomplete");
  return capture;
}

} // namespace rocjitsu::consan
