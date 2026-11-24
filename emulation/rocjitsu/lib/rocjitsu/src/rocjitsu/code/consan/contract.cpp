// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/contract.h"
#include "rocjitsu/code/consan/contract_internal.h"
#include "rocjitsu/code/consan/target.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"

#include "util/bit.h"
#include "util/sha256.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <utility>

namespace rocjitsu::consan {

namespace wire {

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
  encoder.u32(entry.workgroup_id_scalar);
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
         decoder.u32(entry.maximum_workgroup_size) && decoder.u32(entry.workgroup_id_scalar);
}

} // namespace wire

namespace {

// "RJCSFACT" identifies serialized ConSan facts so readers can reject unrelated
// input before decoding the payload.
constexpr std::array<uint8_t, 8> kFactsMagic = {'R', 'J', 'C', 'S', 'F', 'A', 'C', 'T'};
constexpr size_t kEnvelopeBytes = 8 + 16 + 8 + 32;
constexpr size_t kSiteWireBytes = 68;
constexpr size_t kBarrierWireBytes = 56;
// Manifest prefix (144), entry fields including the name length (80), and site
// count (4). Name bytes and per-site records are accounted for separately.
constexpr size_t kFixedPayloadBytes = 144 + 80 + 4;

void encode_site(wire::Encoder &encoder, const SiteFacts &site) {
  encoder.u32(site.id);
  encoder.u32(site.owner_entry);
  encoder.u32(static_cast<uint32_t>(site.analysis));
  encoder.u32(static_cast<uint32_t>(site.record));
  // The original single-range encoding wrote zero in this word.
  encoder.u32((static_cast<uint32_t>(site.lds_ranges.shape) << 16) |
              (uint32_t{site.lds_ranges.offset1} << 8) | site.lds_ranges.offset0);
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

bool decode_site(wire::Decoder &decoder, SiteFacts &site) {
  uint32_t analysis = 0;
  uint32_t record = 0;
  uint32_t ranges = 0;
  uint32_t access = 0;
  if (!decoder.u32(site.id) || !decoder.u32(site.owner_entry) || !decoder.u32(analysis) ||
      !decoder.u32(record) || !decoder.u32(ranges) || !decoder.u32(access) ||
      !decoder.u32(site.width) || !decoder.u64(site.original_offset) ||
      !decoder.u32(site.original_size) || !decoder.u64(site.prepared_guest_offset) ||
      !decoder.u32(site.prepared_guest_size) || !decoder.u64(site.probe_offset) ||
      !decoder.u32(site.probe_size) || !decoder.u32(site.first_slot)) {
    return false;
  }
  site.lds_ranges = {.shape = static_cast<LdsRangeShape>(ranges >> 16),
                     .offset0 = static_cast<uint8_t>(ranges),
                     .offset1 = static_cast<uint8_t>(ranges >> 8)};
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

Result validate_manifest(const Manifest &manifest, util::DiagnosticEmitter diagnostic,
                         const ContractLimits &limits) {
  if (manifest.original.bytes > limits.image_bytes ||
      manifest.prepared.bytes > limits.image_bytes || manifest.sites.size() > limits.sites ||
      manifest.entry.name.size() > limits.string_bytes ||
      manifest.barriers.size() > kSyncCapacity) {
    return diagnostic.emit() << "image, site or name budget exceeded";
  }

  const uint64_t encoded_bytes = kEnvelopeBytes + kFixedPayloadBytes + manifest.entry.name.size() +
                                 uint64_t{manifest.sites.size()} * kSiteWireBytes +
                                 (4 + manifest.barriers.size() * kBarrierWireBytes);
  if (encoded_bytes > limits.metadata_bytes) {
    return diagnostic.emit() << "static facts exceed metadata budget";
  }

  // Older capture profiles share this wire layout. Only admit facts their producer supported.
  constexpr Digest kPriorWidths = {
      0x8e, 0x4e, 0x17, 0xc7, 0x9d, 0x00, 0x15, 0x6a, 0x93, 0x8a, 0xeb,
      0xa2, 0x32, 0x06, 0x3d, 0xb0, 0xa7, 0x24, 0x2c, 0x81, 0x8f, 0xe1,
      0x9f, 0xa8, 0x3e, 0x6c, 0x37, 0x78, 0xd4, 0x3c, 0xff, 0x17,
  };
  constexpr Digest kB96Widths = {
      0x0f, 0xd8, 0x00, 0xf0, 0xff, 0xf8, 0x77, 0xb0, 0xf0, 0x51, 0x04,
      0x56, 0xfe, 0x42, 0x8d, 0x4c, 0xab, 0x4e, 0xe3, 0x50, 0x3b, 0xb6,
      0x11, 0xe3, 0x11, 0xda, 0xd5, 0x94, 0x33, 0x0b, 0xd5, 0xf5,
  };
  constexpr Digest kOrdinaryLds = {
      0x0c, 0x9d, 0x4a, 0x7a, 0x78, 0x6b, 0xd8, 0x4a, 0xe6, 0x60, 0xc3,
      0xff, 0xa2, 0xcf, 0xe2, 0xf3, 0x2b, 0xb4, 0x29, 0xc7, 0x23, 0x4b,
      0x13, 0x17, 0x87, 0x26, 0x0f, 0x4d, 0xfa, 0x19, 0xe6, 0x15,
  };
  const bool admits_atomics = manifest.options == mvp_options_digest();
  const bool admits_bytes = admits_atomics || manifest.options == kOrdinaryLds;
  const bool admits_b96 = admits_bytes || manifest.options == kB96Widths;
  const TargetProfile *target = target_profile(manifest.target);
  if (manifest.original.bytes == 0 || manifest.prepared.bytes == 0 || target == nullptr ||
      !supports_wave(*target, manifest.wave_size) || manifest.code_object_abi != 5 ||
      (manifest.scope != ScopeKind::NativeLdsNoSynchronization &&
       manifest.scope != ScopeKind::NativeLdsWorkgroupBarrier &&
       manifest.scope != ScopeKind::NativeLdsWorkgroupBarrierLoop) ||
      (!admits_b96 && manifest.options != kPriorWidths)) {
    return diagnostic.emit() << "empty image or unsupported target, ABI, scope or options";
  }

  const EntryFacts &entry = manifest.entry;
  if (!make_kernarg_extension_layout(entry.original_kernarg_bytes, kContextPointerPayloads)) {
    return diagnostic.emit() << "wrapper exceeds descriptor size field";
  }
  if (entry.id != 0 || entry.name.empty() || entry.name.find('\0') != std::string::npos ||
      !std::has_single_bit(entry.kernarg_alignment) || entry.kernarg_alignment < 8 ||
      entry.original_kernarg_bytes == 0 || entry.original_private_bytes != 0 ||
      entry.prepared_private_bytes != 0 ||
      entry.prepared_group_bytes < entry.original_group_bytes || entry.prepared_group_bytes < 4 ||
      entry.prepared_group_bytes > 65536 || entry.prepared_sgprs == 0 ||
      entry.prepared_sgprs > 106 || entry.prepared_vgprs == 0 || entry.prepared_vgprs > 256 ||
      (entry.prepared_vgprs % 4) != 0 || entry.maximum_workgroup_size == 0 ||
      entry.maximum_workgroup_size > 1024 ||
      (entry.workgroup_id_scalar != UINT32_MAX &&
       (isa_properties(target->arch).uses_ttmp_workgroup_ids
            ? entry.workgroup_id_scalar != kWorkgroupIdTtmp9Selector
            : (entry.workgroup_id_scalar < 2 || entry.workgroup_id_scalar > 16 ||
               entry.workgroup_id_scalar >= entry.prepared_sgprs)))) {
    return diagnostic.emit() << "invalid entry ABI or resources";
  }

  if (!util::is_range_in_bounds(entry.original_descriptor_offset, 64, manifest.original.bytes) ||
      !util::is_range_in_bounds(entry.prepared_descriptor_offset, 64, manifest.prepared.bytes) ||
      manifest.original_text_bytes == 0 || manifest.original_text_bytes > manifest.original.bytes ||
      manifest.prepared_text_bytes == 0 || manifest.prepared_text_bytes > manifest.prepared.bytes ||
      !util::is_range_in_bounds(entry.original_entry_offset, 4, manifest.original_text_bytes) ||
      !util::is_range_in_bounds(entry.prepared_entry_offset, 4, manifest.prepared_text_bytes) ||
      (entry.original_entry_offset % 4) != 0 || (entry.prepared_entry_offset % 4) != 0 ||
      manifest.sites.empty()) {
    return diagnostic.emit() << "entry mapping or text extent is missing";
  }

  if ((manifest.scope != ScopeKind::NativeLdsNoSynchronization) != !manifest.barriers.empty()) {
    return diagnostic.emit() << "barrier scope and facts disagree";
  }
  if (!manifest.barriers.empty() && entry.maximum_workgroup_size <= manifest.wave_size) {
    return diagnostic.emit()
           << "barrier profile requires a maximum workgroup size permitting two waves";
  }
  std::vector<std::pair<uint64_t, uint64_t>> prepared_ranges;
  prepared_ranges.reserve(2 * manifest.sites.size() + 2 * manifest.barriers.size());
  uint64_t previous_barrier_end = 0;
  const bool split_barrier = uses_split_workgroup_barrier(target->arch);
  for (const BarrierFacts &barrier : manifest.barriers) {
    if (barrier.original_drain < previous_barrier_end || barrier.identity != UINT32_MAX ||
        !util::is_range_in_bounds(barrier.original_drain, split_barrier ? 12 : 8,
                                  manifest.original_text_bytes) ||
        barrier.original_drain % 4 != 0 || barrier.original_signal != barrier.original_drain + 4 ||
        (split_barrier ? barrier.original_wait < barrier.original_signal + 4
                       : barrier.original_wait != barrier.original_signal) ||
        barrier.original_wait % 4 != 0 ||
        !util::is_range_in_bounds(barrier.original_wait, 4, manifest.original_text_bytes) ||
        !util::is_range_in_bounds(barrier.prepared_signal, 4, manifest.prepared_text_bytes) ||
        barrier.prepared_signal % 4 != 0 || barrier.prepared_wait % 4 != 0 ||
        (split_barrier ? barrier.prepared_signal == barrier.prepared_wait
                       : barrier.prepared_signal != barrier.prepared_wait) ||
        !util::is_range_in_bounds(barrier.prepared_wait, 4, manifest.prepared_text_bytes) ||
        barrier.completion_probe != barrier.prepared_wait + 4 || barrier.probe_size == 0 ||
        barrier.probe_size % 4 != 0 ||
        !util::is_range_in_bounds(barrier.completion_probe, barrier.probe_size,
                                  manifest.prepared_text_bytes)) {
      return diagnostic.emit() << "invalid fixed barrier qualification or mapping";
    }
    previous_barrier_end = barrier.original_wait + 4;
    for (const SiteFacts &site : manifest.sites) {
      if (site.original_offset + site.original_size > barrier.original_drain &&
          site.original_offset < barrier.original_wait + 4) {
        return diagnostic.emit() << "access overlaps original barrier group";
      }
    }
    // Distinct source operations cannot share prepared instruction bytes.
    // Include each wait's completion callback so partial overlaps fail too.
    if (split_barrier) {
      prepared_ranges.emplace_back(barrier.prepared_signal, barrier.prepared_signal + 4);
    }
    prepared_ranges.emplace_back(barrier.prepared_wait,
                                 barrier.completion_probe + barrier.probe_size);
  }

  uint64_t previous_original_end = 0;
  for (size_t index = 0; index < manifest.sites.size(); ++index) {
    const SiteFacts &site = manifest.sites[index];
    const LdsRangeRecipe &ranges = site.lds_ranges;
    bool valid_ranges = ranges.offset0 == 0 && ranges.offset1 == 0;
    if (is_dual(ranges.shape)) {
      valid_ranges = admits_bytes && (site.width == 4 || site.width == 8);
    } else if (is_transpose(ranges.shape)) {
      const bool cdna4 = target->arch == IsaFamily::Cdna4 && manifest.wave_size == 64;
      const bool cdna5 = target->arch == IsaFamily::Cdna5 && manifest.wave_size == 32;
      const uint32_t width = ranges.shape == LdsRangeShape::Transpose6             ? 12
                             : ranges.shape == LdsRangeShape::Transpose16 && cdna5 ? 16
                                                                                   : 8;
      valid_ranges = valid_ranges && admits_bytes && (cdna4 || cdna5) &&
                     site.access == AccessKind::Read && site.width == width;
    } else {
      valid_ranges = valid_ranges && ranges.shape == LdsRangeShape::Single;
    }
    const bool valid_access = site.access == AccessKind::Read || site.access == AccessKind::Write ||
                              (site.access == AccessKind::Atomic && admits_atomics &&
                               ranges.shape == LdsRangeShape::Single && site.width == 4);
    if (site.id != index || site.owner_entry != entry.id ||
        site.analysis != AnalysisKind::LdsRaceDetection || site.record != RecordKind::Access ||
        !valid_ranges || !valid_access ||
        (site.width != 2 && site.width != 4 && site.width != 8 && site.width != 16 &&
         (!admits_b96 || site.width != 12) && (!admits_bytes || site.width != 1)) ||
        site.original_size != 8 || site.prepared_guest_size != 8 || site.probe_size == 0 ||
        (site.probe_size % 4) != 0 || (site.original_offset % 4) != 0 ||
        (site.prepared_guest_offset % 4) != 0 || (site.probe_offset % 4) != 0 ||
        site.original_offset < previous_original_end ||
        !util::is_range_in_bounds(site.original_offset, site.original_size,
                                  manifest.original_text_bytes) ||
        !util::is_range_in_bounds(site.prepared_guest_offset, site.prepared_guest_size,
                                  manifest.prepared_text_bytes) ||
        !util::is_range_in_bounds(site.probe_offset, site.probe_size,
                                  manifest.prepared_text_bytes) ||
        uint64_t{site.first_slot} != uint64_t{index} * kAddressGroupsPerSite) {
      return diagnostic.emit() << "invalid site identity, mapping or address group";
    }
    previous_original_end = site.original_offset + site.original_size;
    prepared_ranges.emplace_back(site.prepared_guest_offset,
                                 site.prepared_guest_offset + site.prepared_guest_size);
    prepared_ranges.emplace_back(site.probe_offset, site.probe_offset + site.probe_size);
  }
  std::sort(prepared_ranges.begin(), prepared_ranges.end());
  for (size_t index = 1; index < prepared_ranges.size(); ++index) {
    if (prepared_ranges[index].first < prepared_ranges[index - 1].second) {
      return diagnostic.emit() << "duplicate or overlapping prepared mapping";
    }
  }
  return Result::success();
}

Result match_images(const Manifest &manifest, std::span<const uint8_t> original,
                    std::span<const uint8_t> prepared, util::DiagnosticEmitter diagnostic,
                    const ContractLimits &limits) {
  if (validate_manifest(manifest, diagnostic, limits).failed()) {
    return Result::failure();
  }
  if (original.size() != manifest.original.bytes || prepared.size() != manifest.prepared.bytes ||
      digest(original) != manifest.original.sha256 ||
      digest(prepared) != manifest.prepared.sha256) {
    return diagnostic.emit() << "exact code-object identity mismatch";
  }
  return Result::success();
}

FailureOr<std::vector<uint8_t>> encode_manifest(const Manifest &manifest,
                                                util::DiagnosticEmitter diagnostic,
                                                const ContractLimits &limits) {
  if (validate_manifest(manifest, diagnostic, limits).failed()) {
    return Result::failure();
  }
  wire::Encoder payload;
  wire::encode_identity(payload, manifest.original);
  wire::encode_identity(payload, manifest.prepared);
  payload.bytes(manifest.options);
  payload.u32(manifest.target);
  payload.u32(manifest.wave_size);
  payload.u32(manifest.code_object_abi);
  payload.u32(static_cast<uint32_t>(manifest.scope));
  payload.u64(manifest.original_text_bytes);
  payload.u64(manifest.prepared_text_bytes);
  wire::encode_entry(payload, manifest.entry);
  payload.u32(static_cast<uint32_t>(manifest.sites.size()));
  for (const SiteFacts &site : manifest.sites) {
    encode_site(payload, site);
  }
  payload.u32(static_cast<uint32_t>(manifest.barriers.size()));
  for (const BarrierFacts &barrier : manifest.barriers) {
    payload.u64(barrier.original_drain);
    payload.u64(barrier.original_signal);
    payload.u64(barrier.original_wait);
    payload.u64(barrier.prepared_signal);
    payload.u64(barrier.prepared_wait);
    payload.u64(barrier.completion_probe);
    payload.u32(barrier.probe_size);
    payload.u32(barrier.identity);
  }
  const std::vector<uint8_t> contents = payload.take();

  wire::Encoder envelope;
  envelope.bytes(kFactsMagic);
  envelope.bytes(kProducerContract);
  envelope.u64(contents.size());
  envelope.bytes(digest(contents));
  envelope.bytes(contents);
  return envelope.take();
}

FailureOr<Manifest> decode_manifest(std::span<const uint8_t> bytes,
                                    util::DiagnosticEmitter diagnostic,
                                    const ContractLimits &limits) {
  if (bytes.size() > limits.metadata_bytes) {
    return diagnostic.emit() << "static facts exceed metadata budget";
  }
  wire::Decoder envelope(bytes);
  std::array<uint8_t, 8> magic{};
  std::array<uint8_t, 16> contract{};
  uint64_t payload_size = 0;
  Digest checksum{};
  if (!envelope.bytes(magic) || !envelope.bytes(contract) || !envelope.u64(payload_size) ||
      !envelope.bytes(checksum) || magic != kFactsMagic || payload_size != envelope.remaining()) {
    return diagnostic.emit() << "invalid static-facts framing";
  }
  if (contract != kProducerContract) {
    return diagnostic.emit() << "producer contract mismatch";
  }
  const std::span<const uint8_t> payload = bytes.subspan(kEnvelopeBytes);
  if (digest(payload) != checksum) {
    return diagnostic.emit() << "static-facts digest mismatch";
  }

  wire::Decoder decoder(payload);
  Manifest manifest;
  uint32_t scope = 0;
  uint32_t count = 0;
  if (!wire::decode_identity(decoder, manifest.original) ||
      !wire::decode_identity(decoder, manifest.prepared) || !decoder.bytes(manifest.options) ||
      !decoder.u32(manifest.target) || !decoder.u32(manifest.wave_size) ||
      !decoder.u32(manifest.code_object_abi) || !decoder.u32(scope) ||
      !decoder.u64(manifest.original_text_bytes) || !decoder.u64(manifest.prepared_text_bytes) ||
      !wire::decode_entry(decoder, manifest.entry, limits.string_bytes) || !decoder.u32(count)) {
    return diagnostic.emit() << "truncated or oversized static facts";
  }
  manifest.scope = static_cast<ScopeKind>(scope);
  if (count > limits.sites) {
    return diagnostic.emit() << "site count exceeds budget";
  }
  if (uint64_t{count} * kSiteWireBytes + 4 > decoder.remaining()) {
    return diagnostic.emit() << "site count disagrees with extent";
  }
  manifest.sites.resize(count);
  for (SiteFacts &site : manifest.sites) {
    if (!decode_site(decoder, site)) {
      return diagnostic.emit() << "truncated site facts";
    }
  }
  uint32_t groups = 0;
  if (!decoder.u32(groups) || groups > kSyncCapacity ||
      uint64_t{groups} * kBarrierWireBytes != decoder.remaining()) {
    return diagnostic.emit() << "barrier count disagrees with extent";
  }
  for (uint32_t group = 0; group < groups; ++group) {
    BarrierFacts barrier;
    if (!decoder.u64(barrier.original_drain) || !decoder.u64(barrier.original_signal) ||
        !decoder.u64(barrier.original_wait) || !decoder.u64(barrier.prepared_signal) ||
        !decoder.u64(barrier.prepared_wait) || !decoder.u64(barrier.completion_probe) ||
        !decoder.u32(barrier.probe_size) || !decoder.u32(barrier.identity)) {
      return diagnostic.emit() << "truncated barrier facts";
    }
    manifest.barriers.push_back(barrier);
  }
  if (validate_manifest(manifest, diagnostic, limits).failed()) {
    return Result::failure();
  }
  return manifest;
}

FailureOr<LaunchLayout> plan_launch(const Manifest &manifest, const LaunchDimensions &dimensions,
                                    uint32_t packet_group_bytes, uint32_t packet_private_bytes,
                                    util::DiagnosticEmitter diagnostic,
                                    const ContractLimits &limits) {
  if (validate_manifest(manifest, diagnostic, limits).failed()) {
    return Result::failure();
  }
  if (dimensions.workgroup[0] == 0 || dimensions.grid[0] == 0 ||
      dimensions.workgroup[0] > manifest.entry.maximum_workgroup_size ||
      dimensions.workgroup[1] != 1 || dimensions.workgroup[2] != 1 || dimensions.grid[1] != 1 ||
      dimensions.grid[2] != 1 || packet_private_bytes != 0 ||
      dimensions.grid[0] % dimensions.workgroup[0] != 0) {
    return diagnostic.emit()
           << "capture requires complete bounded 1D workgroups and no private segment";
  }
  const uint32_t workgroups = dimensions.grid[0] / dimensions.workgroup[0];
  if (workgroups > kWorkgroupCapacity ||
      (workgroups > 1 && manifest.entry.workgroup_id_scalar == UINT32_MAX)) {
    return diagnostic.emit() << "multi-workgroup profile requires at most 16 groups and original "
                                "workgroup-X input";
  }
  const uint64_t row_stride = report_row_bytes(manifest.sites.size(), !manifest.barriers.empty());
  const uint32_t waves_per_group =
      (dimensions.workgroup[0] + manifest.wave_size - 1) / manifest.wave_size;
  if (!manifest.barriers.empty() && waves_per_group < 2) {
    return diagnostic.emit() << "barrier profile requires at least two waves per workgroup";
  }
  const uint64_t rows = uint64_t{workgroups} * waves_per_group;
  if (rows > std::numeric_limits<uint32_t>::max() ||
      row_stride > (std::numeric_limits<uint64_t>::max() - kReportHeaderBytes) / rows) {
    return diagnostic.emit() << "report extent arithmetic overflow";
  }
  const uint64_t report_bytes = kReportHeaderBytes + rows * row_stride;
  const std::optional<KernargExtensionLayout> kernarg =
      make_kernarg_extension_layout(manifest.entry.original_kernarg_bytes, kContextPointerPayloads);
  // row_stride is a 32-bit wire field. The report_bytes cap is a separate bound
  // of this capture profile; the context extent and probe offsets remain 64-bit.
  if (row_stride > std::numeric_limits<uint32_t>::max() ||
      report_bytes > std::numeric_limits<uint32_t>::max() || report_bytes > limits.report_bytes ||
      !kernarg) {
    return diagnostic.emit() << "report or wrapper extent exceeds budget";
  }
  const uint32_t group_bytes = std::max(packet_group_bytes, manifest.entry.prepared_group_bytes);
  if (group_bytes > 65536) {
    return diagnostic.emit() << "LDS extent exceeds subset limit";
  }
  return LaunchLayout{.wrapper_bytes = kernarg->wrapper_size,
                      .original_pointer_offset = kernarg->original_kernarg_pointer_offset,
                      .context_pointer_offset = kernarg->payload_offsets[0],
                      .row_stride = static_cast<uint32_t>(row_stride),
                      .report_bytes = report_bytes,
                      .group_bytes = group_bytes,
                      .private_bytes = 0,
                      .wave_count = static_cast<uint32_t>(rows),
                      .workgroup_count = workgroups,
                      .waves_per_workgroup = waves_per_group};
}

FailureOr<TransportImage> initialize_transport(const Manifest &manifest,
                                               const LaunchDimensions &dimensions,
                                               uint32_t group_bytes, uint32_t private_bytes,
                                               uint64_t serial, uint64_t report_address,
                                               util::DiagnosticEmitter diagnostic) {
  const FailureOr<LaunchLayout> layout =
      plan_launch(manifest, dimensions, group_bytes, private_bytes, diagnostic);
  if (layout.failed()) {
    return Result::failure();
  }
  if (serial == 0 || report_address == 0 || (report_address % 64) != 0 ||
      layout.value().report_bytes > std::numeric_limits<uint64_t>::max() - report_address) {
    return diagnostic.emit() << "invalid serial or report address extent";
  }
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
    wire::write_u32(image.report, row + wire::kRowWave, wave % layout.value().waves_per_workgroup);
    wire::write_u32(image.report, row + wire::kRowWorkgroupX,
                    wave / layout.value().waves_per_workgroup);
  }
  return image;
}

} // namespace rocjitsu::consan
