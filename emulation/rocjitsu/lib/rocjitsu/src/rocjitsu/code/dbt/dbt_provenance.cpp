// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/dbt_provenance.h"

#include <array>
#include <cstring>

namespace rocjitsu {
namespace {

constexpr std::array<uint8_t, 8> kMagic = {'R', 'J', 'P', 'R', 'O', 'V', '1', '\0'};
constexpr uint32_t kVersion = 1;

/// @brief Stable on-disk encoding of ProcessorRevision.
///
/// @details Pinned separately from the enum's declaration order so reordering
/// ProcessorRevision cannot silently change an emitted note's meaning.
enum class RevisionWire : uint32_t {
  kUnspecified = 0,
  kGfx1250A0 = 1,
  kGfx1250B0 = 2,
};

struct ProvenanceHeader {
  std::array<uint8_t, 8> magic{};
  uint32_t version = 0;
  uint32_t input_revision = 0;  ///< RevisionWire
  uint32_t output_revision = 0; ///< RevisionWire
  uint32_t reserved = 0;
};

static_assert(sizeof(ProvenanceHeader) == 24);

[[nodiscard]] RevisionWire to_wire(ProcessorRevision revision) {
  switch (revision) {
  case ProcessorRevision::Gfx1250A0:
    return RevisionWire::kGfx1250A0;
  case ProcessorRevision::Gfx1250B0:
    return RevisionWire::kGfx1250B0;
  case ProcessorRevision::Unspecified:
    return RevisionWire::kUnspecified;
  }
  return RevisionWire::kUnspecified;
}

[[nodiscard]] bool from_wire(uint32_t raw, ProcessorRevision &revision) {
  switch (static_cast<RevisionWire>(raw)) {
  case RevisionWire::kGfx1250A0:
    revision = ProcessorRevision::Gfx1250A0;
    return true;
  case RevisionWire::kGfx1250B0:
    revision = ProcessorRevision::Gfx1250B0;
    return true;
  case RevisionWire::kUnspecified:
    // A well-formed note never attests an Unspecified revision (serialize
    // refuses to emit one), so decoding one is treated as malformed.
    return false;
  }
  return false;
}

} // namespace

std::vector<uint8_t> serialize_dbt_provenance(const DbtProvenance &provenance) {
  if (provenance.input_revision == ProcessorRevision::Unspecified ||
      provenance.output_revision == ProcessorRevision::Unspecified)
    return {};

  ProvenanceHeader header{};
  header.magic = kMagic;
  header.version = kVersion;
  header.input_revision = static_cast<uint32_t>(to_wire(provenance.input_revision));
  header.output_revision = static_cast<uint32_t>(to_wire(provenance.output_revision));

  std::vector<uint8_t> out(sizeof(header));
  std::memcpy(out.data(), &header, sizeof(header));
  return out;
}

std::optional<DbtProvenance> parse_dbt_provenance(std::span<const uint8_t> bytes) {
  ProvenanceHeader header{};
  if (bytes.size() < sizeof(header))
    return std::nullopt;
  std::memcpy(&header, bytes.data(), sizeof(header));
  if (header.magic != kMagic || header.version != kVersion || header.reserved != 0)
    return std::nullopt;

  DbtProvenance provenance{};
  if (!from_wire(header.input_revision, provenance.input_revision) ||
      !from_wire(header.output_revision, provenance.output_revision))
    return std::nullopt;
  return provenance;
}

} // namespace rocjitsu
