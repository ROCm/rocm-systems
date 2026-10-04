// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/options.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace addc::cper
{

// --- GUID --------------------------------------------------------------------

/// An EFI GUID with its four components stored in canonical (host) form.
/// Binary wire layout: d1 LE32, d2 LE16, d3 LE16, d4[0..7] as-is.
struct Guid
{
    uint32_t d1{};
    uint16_t d2{};
    uint16_t d3{};
    std::array<uint8_t, 8> d4{};

    constexpr Guid() noexcept = default;

    constexpr Guid(uint32_t d1_, uint16_t d2_, uint16_t d3_,
                   std::array<uint8_t, 8> d4_) noexcept :
        d1{d1_}, d2{d2_}, d3{d3_}, d4{d4_}
    {}

    /// Parse from 16 raw bytes in EFI wire order (d1 LE, d2 LE, d3 LE, d4 raw).
    [[nodiscard]] static Guid parse(
        std::span<const uint8_t, 16> bytes) noexcept;

    /// Produce the canonical lowercase hyphenated GUID string:
    ///   xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    [[nodiscard]] std::string to_string() const;

    constexpr bool operator==(const Guid&) const noexcept = default;
};

/// Serialises as a plain JSON string (lowercase GUID).
void to_json(nlohmann::json& j, const Guid& g);

// --- Timestamp ---------------------------------------------------------------

/// BCD-encoded EFI timestamp (8 bytes).
struct Timestamp
{
    uint8_t seconds{};
    uint8_t minutes{};
    uint8_t hours{};
    uint8_t flag{}; ///< bit 0 = is_precise
    uint8_t day{};
    uint8_t month{};
    uint8_t year{};
    uint8_t century{};

    [[nodiscard]] bool is_precise() const noexcept
    {
        return (flag & 0x1u) != 0u;
    }

    /// Decode the timestamp fields into an ISO 8601 string:
    /// "YYYY-MM-DDTHH:MM:SS+00:00".
    ///
    /// NOTE: The UEFI CPER spec mandates BCD-encoded fields (e.g. 0x24 = year
    /// 2024). AMD CPER records do not comply - fields are raw binary integers
    /// (e.g. 0x18 = 24, not 0x24). All bytes are therefore used as-is.
    [[nodiscard]] std::string decode_timestamp() const;
};

// --- Record-level flags
// -------------------------------------------------------

/// Raw header flags - serialised as { "name": string, "value": uint } for
/// compatibility with libcper JSON consumers.
struct RecordFlags
{
    uint32_t raw{};
};

void to_json(nlohmann::json& j, const RecordFlags& f);

// --- Record header -----------------------------------------------------------

struct RecordHeader
{
    uint8_t revision_major{};
    uint8_t revision_minor{};
    uint16_t section_count{};
    uint32_t error_severity{};
    uint32_t validation_bits{}; ///< Controls conditional field emission.
    uint32_t record_length{};
    Timestamp timestamp{};
    Guid platform_id{};
    Guid partition_id{};
    Guid creator_id{};
    Guid notification_type{};
    uint64_t record_id{};
    RecordFlags flags{};
    uint64_t persistence_info{};
};

/// Serialise the record header using the libcper-compatible JSON schema.
void to_json(nlohmann::json& j, const RecordHeader& h);

// --- Section-level flags
// ------------------------------------------------------

/// Decomposed bit-flags from EFI_ERROR_SECTION_DESCRIPTOR.SectionFlags.
/// Serialised as a JSON object with 8 named boolean fields (alphabetically
/// ordered after assignment) for compatibility with libcper JSON consumers.
struct SectionFlags
{
    bool primary{};
    bool containment_warning{};
    bool reset{};
    bool error_threshold_exceeded{};
    bool resource_not_accessible{};
    bool latent_error{};
    bool propagated{};
    bool overflow{};
};

void to_json(nlohmann::json& j, const SectionFlags& f);

// --- Section descriptor
// -------------------------------------------------------

struct SectionDescriptor
{
    uint32_t section_offset{};
    uint32_t section_length{};
    uint8_t revision_major{};
    uint8_t revision_minor{};
    uint8_t sec_valid_mask{}; ///< bit0=fruID valid, bit1=fruText valid.
    SectionFlags flags{};
    Guid section_type{};
    Guid fru_id{};
    uint32_t severity{};
    std::string fru_text{};
};

/// Produces the JSON form of a section descriptor.
/// NOTE: "sectionType"."type" is set to "Unknown" here; the parser patches it
///       with the direct classifier's type name after construction.
void to_json(nlohmann::json& j, const SectionDescriptor& d);

// --- Parse error -------------------------------------------------------------

enum class ParseError
{
    TooSmall,
    InvalidSignature,
    InvalidSectionCount,
    InvalidRecordLength,
    InvalidSectionLength,
    SectionOutOfBounds,
    RecoveryRequired,
    InvalidSectionBody,
};

/// Structured location for a fatal parse error.  offset identifies the byte
/// in the caller's CPER buffer that contains the rejected field, or the first
/// missing byte when the input is truncated.
struct ParseFailure
{
    ParseError code = ParseError::TooSmall;
    std::size_t offset{};
    std::optional<std::size_t> section_index;
};

[[nodiscard]] std::string_view parse_error_message(ParseError error) noexcept;

struct ParseRepair
{
    std::string code;
    std::string field;
    std::optional<std::size_t> section_index;
    uint64_t original_value{};
    uint64_t substituted_value{};
    std::string reason;
};

void to_json(nlohmann::json& j, const ParseRepair& repair);

/// Representation used for binary fields produced by section parsers. Public
/// JSON IR uses Base64 strings; the typed analysis path keeps bytes native so
/// it does not encode data merely to decode it again.
enum class BinaryRepresentation : uint8_t
{
    Base64,
    Native,
};

struct ParseContext
{
    addc::ParseMode mode = addc::ParseMode::Tolerant;
    BinaryRepresentation binary_representation = BinaryRepresentation::Base64;
    std::size_t section_index{};
    std::vector<ParseRepair>* repairs = nullptr;

    /// Record a tolerated substitution. Returns false in strict mode so the
    /// caller can reject the malformed section instead.
    bool repair(std::string code, std::string field, uint64_t original_value,
                uint64_t substituted_value, std::string reason) const;
};

// --- CPER record -------------------------------------------------------------

struct CperRecord
{
    RecordHeader header{};
    std::vector<SectionDescriptor> descriptors{};
    std::vector<std::vector<uint8_t>> section_bodies{};
    std::vector<ParseRepair> repairs{};

    /// Parse a raw CPER binary buffer.
    /// Returns std::nullopt on failure; optionally writes a ParseError code.
    [[nodiscard]] static std::optional<CperRecord> parse(
        std::span<const uint8_t> buf, ParseError* error_out = nullptr);

    [[nodiscard]] static std::optional<CperRecord> parse(
        std::span<const uint8_t> buf, const addc::DecodeOptions& options,
        ParseError* error_out = nullptr);

    /// Detailed form used by product facades.  Compatibility overloads above
    /// retain the original ParseError-only contract.
    [[nodiscard]] static std::optional<CperRecord> parse_detailed(
        std::span<const uint8_t> buf, const addc::DecodeOptions& options,
        ParseFailure* failure_out = nullptr);
};

} // namespace addc::cper
