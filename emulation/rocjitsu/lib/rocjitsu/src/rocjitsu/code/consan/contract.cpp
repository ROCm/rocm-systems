// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/contract.h"
#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/consan/target.h"

#include "util/sha256.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <utility>

namespace rocjitsu::consan {
namespace {

// "RJCSFACT" identifies serialized ConSan facts so readers can reject unrelated
// input before decoding the payload.
constexpr std::array<uint8_t, 8> kFactsMagic = {'R', 'J', 'C', 'S', 'F', 'A', 'C', 'T'};
constexpr size_t kEnvelopeBytes = 8 + 16 + 8 + 32;
constexpr size_t kSiteWireBytes = 68;
// Manifest prefix (144), entry fields including the name length (76), and site
// count (4). Name bytes and per-site records are accounted for separately.
constexpr size_t kFixedPayloadBytes = 144 + 76 + 4;

Result reject(ContractError *error, ContractErrorCode code, std::string_view message) {
  if (error != nullptr)
    *error = {.code = code, .message = std::string(message)};
  return Result::failure();
}

void clear_error(ContractError *error) {
  if (error != nullptr)
    *error = {};
}

bool fits(uint64_t offset, uint64_t extent, uint64_t bound) {
  return offset <= bound && extent <= bound - offset;
}

class Encoder {
public:
  void u32(uint32_t value) {
    for (uint32_t shift = 0; shift != 32; shift += 8)
      bytes_.push_back(static_cast<uint8_t>(value >> shift));
  }
  void u64(uint64_t value) {
    u32(static_cast<uint32_t>(value));
    u32(static_cast<uint32_t>(value >> 32));
  }
  void bytes(std::span<const uint8_t> value) {
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }
  void string(std::string_view value) {
    u32(static_cast<uint32_t>(value.size()));
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }
  std::vector<uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<uint8_t> bytes_;
};

class Decoder {
public:
  explicit Decoder(std::span<const uint8_t> bytes) : bytes_(bytes) {}

  bool u32(uint32_t &value) {
    if (remaining() < 4)
      return false;
    value = 0;
    for (uint32_t shift = 0; shift != 32; shift += 8)
      value |= uint32_t{bytes_[offset_++]} << shift;
    return true;
  }
  bool u64(uint64_t &value) {
    uint32_t low = 0;
    uint32_t high = 0;
    if (!u32(low) || !u32(high))
      return false;
    value = uint64_t{low} | (uint64_t{high} << 32);
    return true;
  }
  bool bytes(std::span<uint8_t> value) {
    if (value.size() > remaining())
      return false;
    std::copy_n(bytes_.begin() + offset_, value.size(), value.begin());
    offset_ += value.size();
    return true;
  }
  bool string(std::string &value, uint32_t limit) {
    uint32_t size = 0;
    if (!u32(size) || size > limit || size > remaining())
      return false;
    value.assign(reinterpret_cast<const char *>(bytes_.data() + offset_), size);
    offset_ += size;
    return true;
  }
  size_t remaining() const { return bytes_.size() - offset_; }

private:
  std::span<const uint8_t> bytes_;
  size_t offset_ = 0;
};

void encode_identity(Encoder &encoder, const ImageIdentity &identity) {
  encoder.bytes(identity.sha256);
  encoder.u64(identity.bytes);
}

bool decode_identity(Decoder &decoder, ImageIdentity &identity) {
  return decoder.bytes(identity.sha256) && decoder.u64(identity.bytes);
}

void encode_entry(Encoder &encoder, const EntryFacts &entry) {
  encoder.u32(entry.id);
  encoder.string(entry.name);
  encoder.u64(entry.original_descriptor_offset);
  encoder.u64(entry.prepared_descriptor_offset);
  encoder.u64(entry.original_entry_offset);
  encoder.u64(entry.prepared_entry_offset);
  encoder.u32(entry.original_kernarg_bytes);
  encoder.u32(entry.kernarg_alignment);
  encoder.u32(entry.original_group_bytes);
  encoder.u32(entry.prepared_group_bytes);
  encoder.u32(entry.original_private_bytes);
  encoder.u32(entry.prepared_private_bytes);
  encoder.u32(entry.prepared_sgprs);
  encoder.u32(entry.prepared_vgprs);
  encoder.u32(entry.maximum_workgroup_size);
}

bool decode_entry(Decoder &decoder, EntryFacts &entry, uint32_t string_limit) {
  return decoder.u32(entry.id) && decoder.string(entry.name, string_limit) &&
         decoder.u64(entry.original_descriptor_offset) &&
         decoder.u64(entry.prepared_descriptor_offset) &&
         decoder.u64(entry.original_entry_offset) && decoder.u64(entry.prepared_entry_offset) &&
         decoder.u32(entry.original_kernarg_bytes) && decoder.u32(entry.kernarg_alignment) &&
         decoder.u32(entry.original_group_bytes) && decoder.u32(entry.prepared_group_bytes) &&
         decoder.u32(entry.original_private_bytes) && decoder.u32(entry.prepared_private_bytes) &&
         decoder.u32(entry.prepared_sgprs) && decoder.u32(entry.prepared_vgprs) &&
         decoder.u32(entry.maximum_workgroup_size);
}

void encode_site(Encoder &encoder, const SiteFacts &site) {
  encoder.u32(site.id);
  encoder.u32(site.owner_entry);
  encoder.u32(static_cast<uint32_t>(site.analysis));
  encoder.u32(static_cast<uint32_t>(site.record));
  encoder.u32(site.range_id);
  encoder.u32(static_cast<uint32_t>(site.access));
  encoder.u32(site.width);
  encoder.u64(site.original_offset);
  encoder.u32(site.original_size);
  encoder.u64(site.prepared_guest_offset);
  encoder.u32(site.prepared_guest_size);
  encoder.u64(site.probe_offset);
  encoder.u32(site.probe_size);
  encoder.u32(site.first_slot);
}

bool decode_site(Decoder &decoder, SiteFacts &site) {
  uint32_t analysis = 0;
  uint32_t record = 0;
  uint32_t access = 0;
  if (!decoder.u32(site.id) || !decoder.u32(site.owner_entry) || !decoder.u32(analysis) ||
      !decoder.u32(record) || !decoder.u32(site.range_id) || !decoder.u32(access) ||
      !decoder.u32(site.width) || !decoder.u64(site.original_offset) ||
      !decoder.u32(site.original_size) || !decoder.u64(site.prepared_guest_offset) ||
      !decoder.u32(site.prepared_guest_size) || !decoder.u64(site.probe_offset) ||
      !decoder.u32(site.probe_size) || !decoder.u32(site.first_slot))
    return false;
  site.analysis = static_cast<AnalysisKind>(analysis);
  site.record = static_cast<RecordKind>(record);
  site.access = static_cast<AccessKind>(access);
  return true;
}

} // namespace

Digest digest(std::span<const uint8_t> bytes) {
  util::Sha256 hash;
  hash.update(bytes.data(), bytes.size());
  return hash.finish();
}

Digest mvp_options_digest() {
  util::Sha256 hash;
  hash.update(kMvpOptions);
  return hash.finish();
}

Result validate_manifest(const Manifest &manifest, ContractError *error,
                         const ContractLimits &limits) {
  clear_error(error);
  if (manifest.original.bytes > limits.image_bytes ||
      manifest.prepared.bytes > limits.image_bytes || manifest.sites.size() > limits.sites ||
      manifest.entry.name.size() > limits.string_bytes)
    return reject(error, ContractErrorCode::ResourceLimit, "image, site or name budget exceeded");

  const uint64_t encoded_bytes = kEnvelopeBytes + kFixedPayloadBytes + manifest.entry.name.size() +
                                 uint64_t{manifest.sites.size()} * kSiteWireBytes;
  if (encoded_bytes > limits.metadata_bytes)
    return reject(error, ContractErrorCode::ResourceLimit, "static facts exceed metadata budget");

  const auto *target = target_profile(manifest.target);
  if (manifest.original.bytes == 0 || manifest.prepared.bytes == 0 || target == nullptr ||
      !supports_wave(*target, manifest.wave_size) || manifest.code_object_abi != 5 ||
      manifest.scope != ScopeKind::NativeLdsNoSynchronization ||
      manifest.options != mvp_options_digest())
    return reject(error, ContractErrorCode::InvalidFacts,
                  "empty image or unsupported target, ABI, scope or options");

  const EntryFacts &entry = manifest.entry;
  const uint64_t wrapper_bytes = ((uint64_t{entry.original_kernarg_bytes} + 7) & ~uint64_t{7}) + 16;
  if (wrapper_bytes > std::numeric_limits<uint32_t>::max())
    return reject(error, ContractErrorCode::ResourceLimit, "wrapper exceeds descriptor size field");
  if (entry.id != 0 || entry.name.empty() || entry.name.find('\0') != std::string::npos ||
      !std::has_single_bit(entry.kernarg_alignment) || entry.kernarg_alignment < 8 ||
      entry.original_kernarg_bytes == 0 || entry.original_private_bytes != 0 ||
      entry.prepared_private_bytes != 0 ||
      entry.prepared_group_bytes < entry.original_group_bytes || entry.prepared_group_bytes < 4 ||
      entry.prepared_group_bytes > 65536 || entry.prepared_sgprs == 0 ||
      entry.prepared_sgprs > 106 || entry.prepared_vgprs == 0 || entry.prepared_vgprs > 256 ||
      (entry.prepared_vgprs % 4) != 0 || entry.maximum_workgroup_size == 0 ||
      entry.maximum_workgroup_size > 1024)
    return reject(error, ContractErrorCode::InvalidFacts, "invalid entry ABI or resources");

  if (!fits(entry.original_descriptor_offset, 64, manifest.original.bytes) ||
      !fits(entry.prepared_descriptor_offset, 64, manifest.prepared.bytes) ||
      manifest.original_text_bytes == 0 || manifest.original_text_bytes > manifest.original.bytes ||
      manifest.prepared_text_bytes == 0 || manifest.prepared_text_bytes > manifest.prepared.bytes ||
      !fits(entry.original_entry_offset, 4, manifest.original_text_bytes) ||
      !fits(entry.prepared_entry_offset, 4, manifest.prepared_text_bytes) ||
      (entry.original_entry_offset % 4) != 0 || (entry.prepared_entry_offset % 4) != 0 ||
      manifest.sites.empty())
    return reject(error, ContractErrorCode::InvalidFacts,
                  "entry mapping or text extent is missing");

  uint64_t previous_original_end = 0;
  std::vector<std::pair<uint64_t, uint64_t>> guest_ranges;
  guest_ranges.reserve(manifest.sites.size());
  for (size_t index = 0; index < manifest.sites.size(); ++index) {
    const SiteFacts &site = manifest.sites[index];
    if (site.id != index || site.owner_entry != entry.id ||
        site.analysis != AnalysisKind::LdsRaceDetection || site.record != RecordKind::Access ||
        site.range_id != 0 ||
        (site.access != AccessKind::Read && site.access != AccessKind::Write) || site.width != 4 ||
        site.original_size != 8 || site.prepared_guest_size != 8 || site.probe_size == 0 ||
        (site.probe_size % 4) != 0 || (site.original_offset % 4) != 0 ||
        (site.prepared_guest_offset % 4) != 0 || (site.probe_offset % 4) != 0 ||
        site.original_offset < previous_original_end ||
        !fits(site.original_offset, site.original_size, manifest.original_text_bytes) ||
        !fits(site.prepared_guest_offset, site.prepared_guest_size, manifest.prepared_text_bytes) ||
        !fits(site.probe_offset, site.probe_size, manifest.prepared_text_bytes) ||
        uint64_t{site.first_slot} != uint64_t{index} * kBanksPerSite)
      return reject(error, ContractErrorCode::InvalidFacts,
                    "invalid site identity, mapping or bank");
    previous_original_end = site.original_offset + site.original_size;
    guest_ranges.emplace_back(site.prepared_guest_offset,
                              site.prepared_guest_offset + site.prepared_guest_size);
  }
  std::sort(guest_ranges.begin(), guest_ranges.end());
  for (size_t index = 1; index < guest_ranges.size(); ++index) {
    if (guest_ranges[index].first < guest_ranges[index - 1].second)
      return reject(error, ContractErrorCode::InvalidFacts,
                    "duplicate or overlapping guest mapping");
  }
  return Result::success();
}

Result match_images(const Manifest &manifest, std::span<const uint8_t> original,
                    std::span<const uint8_t> prepared, ContractError *error,
                    const ContractLimits &limits) {
  if (validate_manifest(manifest, error, limits).failed())
    return Result::failure();
  if (original.size() != manifest.original.bytes || prepared.size() != manifest.prepared.bytes ||
      digest(original) != manifest.original.sha256 || digest(prepared) != manifest.prepared.sha256)
    return reject(error, ContractErrorCode::IdentityMismatch,
                  "exact code-object identity mismatch");
  return Result::success();
}

FailureOr<std::vector<uint8_t>> encode_manifest(const Manifest &manifest, ContractError *error,
                                                const ContractLimits &limits) {
  if (validate_manifest(manifest, error, limits).failed())
    return Result::failure();
  Encoder payload;
  encode_identity(payload, manifest.original);
  encode_identity(payload, manifest.prepared);
  payload.bytes(manifest.options);
  payload.u32(manifest.target);
  payload.u32(manifest.wave_size);
  payload.u32(manifest.code_object_abi);
  payload.u32(static_cast<uint32_t>(manifest.scope));
  payload.u64(manifest.original_text_bytes);
  payload.u64(manifest.prepared_text_bytes);
  encode_entry(payload, manifest.entry);
  payload.u32(static_cast<uint32_t>(manifest.sites.size()));
  for (const SiteFacts &site : manifest.sites)
    encode_site(payload, site);
  const std::vector<uint8_t> contents = payload.take();

  Encoder envelope;
  envelope.bytes(kFactsMagic);
  envelope.bytes(kProducerContract);
  envelope.u64(contents.size());
  envelope.bytes(digest(contents));
  envelope.bytes(contents);
  return envelope.take();
}

FailureOr<Manifest> decode_manifest(std::span<const uint8_t> bytes, ContractError *error,
                                    const ContractLimits &limits) {
  clear_error(error);
  if (bytes.size() > limits.metadata_bytes)
    return reject(error, ContractErrorCode::ResourceLimit, "static facts exceed metadata budget");
  Decoder envelope(bytes);
  std::array<uint8_t, 8> magic{};
  std::array<uint8_t, 16> contract{};
  uint64_t payload_size = 0;
  Digest checksum{};
  if (!envelope.bytes(magic) || !envelope.bytes(contract) || !envelope.u64(payload_size) ||
      !envelope.bytes(checksum) || magic != kFactsMagic || payload_size != envelope.remaining())
    return reject(error, ContractErrorCode::InvalidEncoding, "invalid static-facts framing");
  if (contract != kProducerContract)
    return reject(error, ContractErrorCode::IncompatibleContract, "producer contract mismatch");
  const std::span<const uint8_t> payload = bytes.subspan(kEnvelopeBytes);
  if (digest(payload) != checksum)
    return reject(error, ContractErrorCode::IdentityMismatch, "static-facts digest mismatch");

  Decoder decoder(payload);
  Manifest manifest;
  uint32_t scope = 0;
  uint32_t count = 0;
  if (!decode_identity(decoder, manifest.original) ||
      !decode_identity(decoder, manifest.prepared) || !decoder.bytes(manifest.options) ||
      !decoder.u32(manifest.target) || !decoder.u32(manifest.wave_size) ||
      !decoder.u32(manifest.code_object_abi) || !decoder.u32(scope) ||
      !decoder.u64(manifest.original_text_bytes) || !decoder.u64(manifest.prepared_text_bytes) ||
      !decode_entry(decoder, manifest.entry, limits.string_bytes) || !decoder.u32(count))
    return reject(error, ContractErrorCode::InvalidEncoding, "truncated or oversized static facts");
  manifest.scope = static_cast<ScopeKind>(scope);
  if (count > limits.sites)
    return reject(error, ContractErrorCode::ResourceLimit, "site count exceeds budget");
  if (uint64_t{count} * kSiteWireBytes != decoder.remaining())
    return reject(error, ContractErrorCode::InvalidEncoding, "site count disagrees with extent");
  manifest.sites.resize(count);
  for (SiteFacts &site : manifest.sites) {
    if (!decode_site(decoder, site))
      return reject(error, ContractErrorCode::InvalidEncoding, "truncated site facts");
  }
  if (validate_manifest(manifest, error, limits).failed())
    return Result::failure();
  return manifest;
}

FailureOr<LaunchLayout> plan_launch(const Manifest &manifest, const LaunchDimensions &dimensions,
                                    uint32_t packet_group_bytes, uint32_t packet_private_bytes,
                                    ContractError *error, const ContractLimits &limits) {
  if (validate_manifest(manifest, error, limits).failed())
    return Result::failure();
  if (dimensions.grid != dimensions.workgroup || dimensions.workgroup[0] == 0 ||
      dimensions.workgroup[0] > manifest.entry.maximum_workgroup_size ||
      dimensions.workgroup[1] != 1 || dimensions.workgroup[2] != 1 || packet_private_bytes != 0)
    return reject(error, ContractErrorCode::UnsupportedLaunch,
                  "native LDS capture requires one bounded 1D workgroup and no private segment");

  const uint64_t row_stride =
      kRowHeaderBytes + uint64_t{manifest.sites.size()} * kBanksPerSite * kAccessRecordBytes;
  const uint32_t waves = (dimensions.workgroup[0] + manifest.wave_size - 1) / manifest.wave_size;
  const uint64_t report_bytes = kReportHeaderBytes + waves * row_stride;
  const uint64_t original_pointer_offset =
      (uint64_t{manifest.entry.original_kernarg_bytes} + 7) & ~uint64_t{7};
  const uint64_t wrapper_bytes = original_pointer_offset + 16;
  if (row_stride > std::numeric_limits<uint32_t>::max() || report_bytes > limits.report_bytes ||
      wrapper_bytes > std::numeric_limits<uint32_t>::max())
    return reject(error, ContractErrorCode::ResourceLimit,
                  "report or wrapper extent exceeds budget");
  const uint32_t group_bytes = std::max(packet_group_bytes, manifest.entry.prepared_group_bytes);
  if (group_bytes > 65536)
    return reject(error, ContractErrorCode::UnsupportedLaunch, "LDS extent exceeds subset limit");
  return LaunchLayout{.wrapper_bytes = static_cast<uint32_t>(wrapper_bytes),
                      .original_pointer_offset = static_cast<uint32_t>(original_pointer_offset),
                      .context_pointer_offset = static_cast<uint32_t>(original_pointer_offset + 8),
                      .row_stride = static_cast<uint32_t>(row_stride),
                      .report_bytes = report_bytes,
                      .group_bytes = group_bytes,
                      .private_bytes = 0,
                      .wave_count = waves};
}

FailureOr<TransportImage> initialize_transport(const Manifest &manifest,
                                               const LaunchDimensions &dimensions,
                                               uint32_t group_bytes, uint32_t private_bytes,
                                               uint64_t serial, uint64_t report_address,
                                               ContractError *error) {
  const auto layout = plan_launch(manifest, dimensions, group_bytes, private_bytes, error);
  if (layout.failed())
    return Result::failure();
  if (serial == 0 || report_address == 0 || (report_address % 64) != 0 ||
      layout.value().report_bytes > std::numeric_limits<uint64_t>::max() - report_address)
    return reject(error, ContractErrorCode::InvalidFacts,
                  "invalid serial or report address extent");
  TransportImage image;
  image.report.resize(layout.value().report_bytes);
  wire::write_u64(image.context, wire::kContextSerial, serial);
  wire::write_u32(image.context, wire::kContextEntry, manifest.entry.id);
  wire::write_u64(image.context, wire::kContextReportPointer, report_address);
  wire::write_u64(image.context, wire::kContextReportBytes, image.report.size());
  for (size_t dimension = 0; dimension < 3; ++dimension) {
    wire::write_u32(image.context, wire::kContextGrid + dimension * 4, dimensions.grid[dimension]);
    wire::write_u32(image.context, wire::kContextWorkgroup + dimension * 4,
                    dimensions.workgroup[dimension]);
  }
  wire::write_u32(image.context, wire::kContextGroupBytes, layout.value().group_bytes);
  wire::write_u32(image.context, wire::kContextPrivateBytes, layout.value().private_bytes);
  wire::write_u64(image.report, wire::kReportSerial, serial);
  wire::write_u32(image.report, wire::kReportEntry, manifest.entry.id);
  wire::write_u32(image.report, wire::kReportRows, layout.value().wave_count);
  wire::write_u32(image.report, wire::kReportRowStride, layout.value().row_stride);
  wire::write_u32(image.report, wire::kReportSites, static_cast<uint32_t>(manifest.sites.size()));
  for (uint32_t wave = 0; wave < layout.value().wave_count; ++wave) {
    const size_t row = kReportHeaderBytes + size_t{wave} * layout.value().row_stride;
    wire::write_u64(image.report, row + wire::kRowSerial, serial);
    wire::write_u32(image.report, row + wire::kRowEntry, manifest.entry.id);
    wire::write_u32(image.report, row + wire::kRowWave, wave);
  }
  return image;
}

} // namespace rocjitsu::consan
