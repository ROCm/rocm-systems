// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/result.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::consan {

using Digest = std::array<uint8_t, 32>;

// Fixed compatibility token: UUIDv4-shaped bytes, chosen arbitrarily once and
// checked in (a replacement can use Python uuid.uuid4().bytes). This identifies
// the shared interpretation of facts, context, records and captures, not an
// image hash, execution or automatically changing build ID. Change it when
// those byte layouts or their meaning become incompatible. Comments, refactors
// and internal API changes alone do not require a new token; no negotiation is
// performed between producers and consumers.
inline constexpr std::array<uint8_t, 16> kProducerContract = {
    0x9a, 0x62, 0xb4, 0x31, 0x37, 0xe0, 0x4d, 0x6c, 0xae, 0x59, 0x71, 0x38, 0x02, 0x96, 0xf8, 0x42};
// Canonical hash input for the manifest's options identity. Changing any byte
// changes the identity required by artifact and capture admission.
inline constexpr std::string_view kMvpOptions =
    "native-lds-b32;no-sync;one-workgroup-1d;mask64;address-groups-per-site=4;collision=on";
// Software record slots for distinct effective LDS address groups per wave/site.
// Remaining lanes after these slots are filled cause one lost occurrence;
// this reservation count is independent of the GPU's hardware LDS banks.
inline constexpr uint32_t kAddressGroupsPerSite = 4;
inline constexpr uint32_t kReportHeaderBytes = 64;
inline constexpr uint32_t kRowHeaderBytes = 64;
inline constexpr uint32_t kAccessRecordBytes = 64;
inline constexpr uint32_t kContextBytes = 64;
inline constexpr std::array<KernargExtensionPayloadLayout, 1> kContextPointerPayloads = {
    {{.size = sizeof(uint64_t), .alignment = alignof(uint64_t)}}};

enum class AnalysisKind : uint32_t { Invalid = 0, LdsRaceDetection = 1 };
enum class RecordKind : uint32_t { Access = 1 };
enum class AccessKind : uint32_t { Read = 1, Write = 2 };
enum class ScopeKind : uint32_t { NativeLdsNoSynchronization = 1 };

/// @brief GPU wire offsets; consumers never serialize native C++ struct layout.
namespace wire {
inline constexpr uint32_t kContextSerial = 0;
inline constexpr uint32_t kContextEntry = 8;
inline constexpr uint32_t kContextReportPointer = 16;
inline constexpr uint32_t kContextReportBytes = 24;
inline constexpr uint32_t kContextGrid = 32;
inline constexpr uint32_t kContextWorkgroup = 44;
inline constexpr uint32_t kContextGroupBytes = 56;
inline constexpr uint32_t kContextPrivateBytes = 60;

inline constexpr uint32_t kReportSerial = 0;
inline constexpr uint32_t kReportEntry = 8;
inline constexpr uint32_t kReportRows = 12;
inline constexpr uint32_t kReportRowStride = 16;
inline constexpr uint32_t kReportSites = 20;
inline constexpr uint32_t kReportReservedBegin = 24;

inline constexpr uint32_t kRowSerial = 0;
inline constexpr uint32_t kRowEntry = 8;
inline constexpr uint32_t kRowWave = 12;
inline constexpr uint32_t kRowStatus = 16;
inline constexpr uint32_t kRowLoss = 20;
inline constexpr uint32_t kRowFinalSequence = 24;
inline constexpr uint32_t kRowReservedBegin = 28;
inline constexpr uint32_t kRowEntered = 1;
inline constexpr uint32_t kRowExited = 2;
inline constexpr uint32_t kRowExhausted = 4;

inline constexpr uint32_t kRecordCommit = 0;
inline constexpr uint32_t kRecordAnalysis = 4;
inline constexpr uint32_t kRecordKind = 8;
inline constexpr uint32_t kRecordSite = 12;
inline constexpr uint32_t kRecordRange = 16;
inline constexpr uint32_t kRecordSequence = 20;
inline constexpr uint32_t kRecordLdsOffset = 24;
inline constexpr uint32_t kRecordWidth = 28;
inline constexpr uint32_t kRecordGuestMask = 32;
inline constexpr uint32_t kRecordFlags = 40;
inline constexpr uint32_t kRecordReservedBegin = 44;
inline constexpr uint32_t kAccessCommitted = 0x43534143;
inline constexpr uint32_t kExactGuestMask = 1;
} // namespace wire

enum class ContractErrorCode {
  None,
  InvalidEncoding,
  IncompatibleContract,
  IdentityMismatch,
  InvalidFacts,
  UnsupportedLaunch,
  ResourceLimit,
};

struct ContractError {
  ContractErrorCode code = ContractErrorCode::None;
  std::string message;
};

/// @brief Initial software budgets, checked before allocating untrusted extents.
/// @details These conservative experimental caps are not measured production
/// thresholds, hardware limits, or the limits of the integer wire fields.
/// Callers may lower them; validation/planning reports ResourceLimit on excess.
struct ContractLimits {
  // Bounds facts parsing/storage, with headroom over 65536 68-byte site maps.
  uint64_t metadata_bytes = 16 * 1024 * 1024;
  // Bounds owning original/prepared image copies and hashing work, also used
  // by the compiler's default input budget. Not a code-object ABI size limit.
  uint64_t image_bytes = 256 * 1024 * 1024;
  // Matches the compiler's initial instruction-analysis cap; one physical
  // access creates at most one site. This is not the u32 site-ID limit.
  uint32_t sites = 65536;
  // Bounds owned entry names and diagnostics, independent of filesystem paths.
  uint32_t string_bytes = 4096;
  // Bounds each launch's report allocation/copy, not the GPU's LDS capacity.
  uint64_t report_bytes = 64 * 1024 * 1024;
};

class ImageIdentity {
public:
  Digest sha256{};
  uint64_t bytes = 0;
  bool operator==(const ImageIdentity &) const = default;
};

/// @brief One physical native LDS instruction, owned by the sole MVP entry.
/// @details Offsets are .text-relative. A zero mapping size means missing,
/// whereas offset zero is a valid location. Both checks share this reservation.
class SiteFacts {
public:
  uint32_t id = 0;
  uint32_t owner_entry = 0;
  AnalysisKind analysis = AnalysisKind::Invalid;
  RecordKind record = RecordKind::Access;
  uint32_t range_id = 0;
  AccessKind access = AccessKind::Read;
  uint32_t width = 4;
  uint64_t original_offset = 0;
  uint32_t original_size = 0;
  uint64_t prepared_guest_offset = 0;
  uint32_t prepared_guest_size = 0;
  uint64_t probe_offset = 0;
  uint32_t probe_size = 0;
  uint32_t first_slot = 0;
  bool operator==(const SiteFacts &) const = default;
};

class EntryFacts {
public:
  uint32_t id = 0;
  std::string name;
  uint64_t original_descriptor_offset = 0;
  uint64_t prepared_descriptor_offset = 0;
  uint64_t original_entry_offset = 0;
  uint64_t prepared_entry_offset = 0;
  uint32_t original_kernarg_bytes = 0;
  uint32_t kernarg_alignment = 0;
  uint32_t original_group_bytes = 0;
  uint32_t prepared_group_bytes = 0;
  uint32_t original_private_bytes = 0;
  uint32_t prepared_private_bytes = 0;
  uint32_t prepared_sgprs = 0;
  uint32_t prepared_vgprs = 0;
  uint32_t maximum_workgroup_size = 1024;
  bool operator==(const EntryFacts &) const = default;
};

/// @brief Owning static handoff shared by AOT, runtime and offline readers.
/// @details This subset admits one entry, dynamically sized wave rows, native LDS
/// b32 Read/Write, no relevant synchronization, no private frame or segment
/// addends. The compiler must establish the scope from original instructions.
/// Codec validation establishes structural consistency, not ISA safety.
class Manifest {
public:
  ImageIdentity original;
  ImageIdentity prepared;
  Digest options{};
  uint32_t target = 0;
  uint32_t wave_size = 0;
  uint32_t code_object_abi = 0;
  ScopeKind scope = ScopeKind::NativeLdsNoSynchronization;
  uint64_t original_text_bytes = 0;
  uint64_t prepared_text_bytes = 0;
  EntryFacts entry;
  std::vector<SiteFacts> sites;
  bool operator==(const Manifest &) const = default;
};

struct LaunchDimensions {
  std::array<uint32_t, 3> grid{};
  std::array<uint32_t, 3> workgroup{};
};

class LaunchLayout {
public:
  uint32_t wrapper_bytes = 0;
  uint32_t original_pointer_offset = 0;
  uint32_t context_pointer_offset = 0;
  uint32_t row_stride = 0;
  uint64_t report_bytes = 0;
  uint32_t group_bytes = 0;
  uint32_t private_bytes = 0;
  uint32_t wave_count = 0;
  bool operator==(const LaunchLayout &) const = default;
};

struct TransportImage {
  std::array<uint8_t, kContextBytes> context{};
  std::vector<uint8_t> report;
};

/// @brief Initialize parent identities and bindings; GPU alone publishes row status.
[[nodiscard]] FailureOr<TransportImage>
initialize_transport(const Manifest &manifest, const LaunchDimensions &dimensions,
                     uint32_t group_bytes, uint32_t private_bytes, uint64_t serial,
                     uint64_t report_address, ContractError *error = nullptr);

[[nodiscard]] Digest digest(std::span<const uint8_t> bytes);
[[nodiscard]] Digest mvp_options_digest();

[[nodiscard]] Result validate_manifest(const Manifest &manifest, ContractError *error = nullptr,
                                       const ContractLimits &limits = {});

/// @brief Check exact image bytes separately from static-facts-only decoding.
/// @details Offline captures own facts, but need not retain either code image.
[[nodiscard]] Result match_images(const Manifest &manifest, std::span<const uint8_t> original,
                                  std::span<const uint8_t> prepared, ContractError *error = nullptr,
                                  const ContractLimits &limits = {});

/// @brief Encode canonical, checksummed, explicitly little-endian static facts.
[[nodiscard]] FailureOr<std::vector<uint8_t>> encode_manifest(const Manifest &manifest,
                                                              ContractError *error = nullptr,
                                                              const ContractLimits &limits = {});

/// @brief Decode bounded, owning facts; reject trailing bytes and absent mappings.
[[nodiscard]] FailureOr<Manifest> decode_manifest(std::span<const uint8_t> bytes,
                                                  ContractError *error = nullptr,
                                                  const ContractLimits &limits = {});

/// @brief Check actual launch shape and derive the single canonical storage layout.
/// @details One one-dimensional workgroup, bounded by kernel facts and budgets.
/// Segment minima preserve the original packet extents. No dynamic private
/// frame is admitted in MVP.
[[nodiscard]] FailureOr<LaunchLayout>
plan_launch(const Manifest &manifest, const LaunchDimensions &dimensions,
            uint32_t packet_group_bytes, uint32_t packet_private_bytes,
            ContractError *error = nullptr, const ContractLimits &limits = {});

} // namespace rocjitsu::consan
