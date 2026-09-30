// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/cper/record.hpp"
#include "addc/detail/ascii.hpp"
#include "addc/detail/format.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <string_view>

namespace addc::cper
{

namespace
{

// --- Portable little-endian reads --------------------------------------------

constexpr uint16_t readU16Le(std::span<const uint8_t> d, std::size_t o) noexcept
{
    return static_cast<uint16_t>(d[o]) |
           static_cast<uint16_t>(static_cast<uint16_t>(d[o + 1U]) << 8U);
}

constexpr uint32_t readU32Le(std::span<const uint8_t> d, std::size_t o) noexcept
{
    return static_cast<uint32_t>(d[o]) |
           (static_cast<uint32_t>(d[o + 1U]) << 8U) |
           (static_cast<uint32_t>(d[o + 2U]) << 16U) |
           (static_cast<uint32_t>(d[o + 3U]) << 24U);
}

constexpr uint64_t readU64Le(std::span<const uint8_t> d, std::size_t o) noexcept
{
    uint64_t v = 0;
    for (std::size_t i = 0; i < 8U; ++i)
    {
        v |= static_cast<uint64_t>(d[o + i]) << (8U * i);
    }
    return v;
}

// --- GUID binary parse -------------------------------------------------------

Guid parseGuid(std::span<const uint8_t> d, std::size_t o) noexcept
{
    std::array<uint8_t, 8> d4{};
    std::copy_n(d.data() + o + 8U, 8U, d4.begin());
    return Guid{readU32Le(d, o), readU16Le(d, o + 4U), readU16Le(d, o + 6U),
                d4};
}

// --- Severity / flag name lookups
// ---------------------------------------------

constexpr std::string_view severityToString(uint32_t code) noexcept
{
    // Names used by the libcper-compatible JSON schema.
    switch (code)
    {
        case 0:
            return "Recoverable";
        case 1:
            return "Fatal";
        case 2:
            return "Corrected";
        case 3:
            return "Informational";
        default:
            return "Unknown";
    }
}

constexpr std::string_view recordFlagsName(uint32_t flags) noexcept
{
    // integer_to_readable_pair with keys {1,2,4}
    switch (flags)
    {
        case 1:
            return "HW_ERROR_FLAGS_RECOVERED";
        case 2:
            return "HW_ERROR_FLAGS_PREVERR";
        case 4:
            return "HW_ERROR_FLAGS_SIMULATED";
        default:
            return "Unknown";
    }
}

// --- Notification-type GUID -> human-readable name
// ----------------------------

// Notification GUIDs are defined by the UEFI CPER specification. The readable
// names preserve compatibility with existing libcper JSON consumers.
std::string_view notificationTypeName(const Guid& g) noexcept
{
    struct Entry
    {
        Guid guid;
        std::string_view name;
    };
    // Non-constexpr: MSVC does not allow constexpr functions with static
    // locals.
    static const std::array<Entry, 11> kTable{{
        {Guid{0x2DCE8BB1U,
              0xBDD7U,
              0x450eU,
              {0xB9U, 0xADU, 0x9CU, 0xF4U, 0xEBU, 0xD4U, 0xF8U, 0x90U}},
         "CMC"},
        {Guid{0x4E292F96U,
              0xD843U,
              0x4a55U,
              {0xA8U, 0xC2U, 0xD4U, 0x81U, 0xF2U, 0x7EU, 0xBEU, 0xEEU}},
         "CPE"},
        {Guid{0xE8F56FFEU,
              0x919CU,
              0x4cc5U,
              {0xBAU, 0x88U, 0x65U, 0xABU, 0xE1U, 0x49U, 0x13U, 0xBBU}},
         "MCE"},
        {Guid{0xCF93C01FU,
              0x1A16U,
              0x4dfcU,
              {0xB8U, 0xBCU, 0x9CU, 0x4DU, 0xAFU, 0x67U, 0xC1U, 0x04U}},
         "PCIe"},
        {Guid{0xCC5263E8U,
              0x9308U,
              0x454aU,
              {0x89U, 0xD0U, 0x34U, 0x0BU, 0xD3U, 0x9BU, 0xC9U, 0x8EU}},
         "INIT"},
        {Guid{0x5BAD89FFU,
              0xB7E6U,
              0x42c9U,
              {0x81U, 0x4AU, 0xCFU, 0x24U, 0x85U, 0xD6U, 0xE9U, 0x8AU}},
         "NMI"},
        {Guid{0x3D61A466U,
              0xAB40U,
              0x409aU,
              {0xA6U, 0x98U, 0xF3U, 0x62U, 0xD4U, 0x64U, 0xB3U, 0x8FU}},
         "Boot"},
        {Guid{0x667DD791U,
              0xC6B3U,
              0x4c27U,
              {0x8AU, 0x6BU, 0x0FU, 0x8EU, 0x72U, 0x2DU, 0xEBU, 0x41U}},
         "DMAr"},
        {Guid{0x9A78788AU,
              0xBBE8U,
              0x11E4U,
              {0x80U, 0x9EU, 0x67U, 0x61U, 0x1EU, 0x5DU, 0x46U, 0xB0U}},
         "SEA"},
        {Guid{0x5C284C81U,
              0xB0AEU,
              0x4E87U,
              {0xA3U, 0x22U, 0xB0U, 0x4CU, 0x85U, 0x62U, 0x43U, 0x23U}},
         "SEI"},
        {Guid{0x09A9D5ACU,
              0x5204U,
              0x4214U,
              {0x96U, 0xE5U, 0x94U, 0x99U, 0x2EU, 0x75U, 0x2BU, 0xCDU}},
         "PEI"},
    }};
    for (const auto& e : kTable)
    {
        if (g == e.guid)
        {
            return e.name;
        }
    }
    return "Unknown";
}

} // anonymous namespace

// --- Guid --------------------------------------------------------------------

Guid Guid::parse(std::span<const uint8_t, 16> bytes) noexcept
{
    std::array<uint8_t, 8> d4{};
    std::copy_n(bytes.data() + 8U, 8U, d4.begin());
    return Guid{readU32Le(bytes, 0U), readU16Le(bytes, 4U),
                readU16Le(bytes, 6U), d4};
}

std::string Guid::to_string() const
{
    return addc::format(
        "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        d1, d2, d3, d4[0], d4[1], d4[2], d4[3], d4[4], d4[5], d4[6], d4[7]);
}

void to_json(nlohmann::json& j, const Guid& g)
{
    j = g.to_string();
}

// --- Timestamp ---------------------------------------------------------------

// NOTE: The UEFI CPER spec requires BCD-encoded timestamp fields.
// AMD CPER records are non-compliant: fields are raw binary integers.
// e.g. a year byte of 0x18 means 24, not the BCD value 18.
// Fields are therefore used as-is without BCD decoding.
std::string Timestamp::decode_timestamp() const
{
    const int year_full =
        (static_cast<int>(century) - 1) * 100 + static_cast<int>(year);
    return addc::format("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}+00:00", year_full,
                        static_cast<int>(month), static_cast<int>(day),
                        static_cast<int>(hours), static_cast<int>(minutes),
                        static_cast<int>(seconds));
}

// --- RecordFlags -------------------------------------------------------------

void to_json(nlohmann::json& j, const RecordFlags& f)
{
    j = {
        {"name", recordFlagsName(f.raw)},
        {"value", static_cast<uint64_t>(f.raw)},
    };
}

// --- SectionFlags ------------------------------------------------------------

void to_json(nlohmann::json& j, const SectionFlags& f)
{
    // Insert in bit-order (0->7); nlohmann::json sorts keys alphabetically,
    // which matches the libcper sample output.
    j = {
        {"containmentWarning", f.containment_warning},
        {"errorThresholdExceeded", f.error_threshold_exceeded},
        {"latentError", f.latent_error},
        {"overflow", f.overflow},
        {"primary", f.primary},
        {"propagated", f.propagated},
        {"reset", f.reset},
        {"resourceNotAccessible", f.resource_not_accessible},
    };
}

// --- SectionDescriptor -------------------------------------------------------

void to_json(nlohmann::json& j, const SectionDescriptor& d)
{
    j["sectionOffset"] = static_cast<uint64_t>(d.section_offset);
    j["sectionLength"] = static_cast<uint64_t>(d.section_length);
    j["revision"] = {{"major", d.revision_major}, {"minor", d.revision_minor}};
    j["flags"] = d.flags;

    // sectionType.type is intentionally "Unknown" here; parser.cpp patches it.
    j["sectionType"] = {
        {"data", d.section_type.to_string()},
        {"type", "Unknown"},
    };

    if ((d.sec_valid_mask & 0x1U) != 0U)
    {
        j["fruID"] = d.fru_id.to_string();
    }
    if ((d.sec_valid_mask & 0x2U) != 0U)
    {
        j["fruText"] = d.fru_text;
    }

    j["severity"] = {
        {"code", static_cast<uint64_t>(d.severity)},
        {"name", severityToString(d.severity)},
    };
}

void to_json(nlohmann::json& j, const ParseRepair& repair)
{
    j = {
        {"code", repair.code},
        {"field", repair.field},
        {"original_value", repair.original_value},
        {"substituted_value", repair.substituted_value},
        {"reason", repair.reason},
    };
    if (repair.section_index)
    {
        j["section_index"] = *repair.section_index;
    }
}

bool ParseContext::repair(std::string code, std::string field,
                          uint64_t original_value, uint64_t substituted_value,
                          std::string reason) const
{
    if (mode == addc::ParseMode::Strict)
    {
        return false;
    }
    if (repairs != nullptr)
    {
        repairs->push_back(ParseRepair{
            .code = std::move(code),
            .field = std::move(field),
            .section_index = section_index,
            .original_value = original_value,
            .substituted_value = substituted_value,
            .reason = std::move(reason),
        });
    }
    return true;
}

// --- RecordHeader ------------------------------------------------------------

void to_json(nlohmann::json& j, const RecordHeader& h)
{
    j["revision"] = {{"major", h.revision_major}, {"minor", h.revision_minor}};
    j["sectionCount"] = static_cast<int64_t>(h.section_count);
    j["severity"] = {
        {"code", static_cast<uint64_t>(h.error_severity)},
        {"name", severityToString(h.error_severity)},
    };
    j["recordLength"] = static_cast<uint64_t>(h.record_length);

    // Conditional timestamp (ValidationBits bit 1).
    if ((h.validation_bits & 0x2U) != 0U)
    {
        j["timestamp"] = h.timestamp.decode_timestamp();
        j["timestampIsPrecise"] = h.timestamp.is_precise();
    }

    // Conditional platformID (ValidationBits bit 0).
    if ((h.validation_bits & 0x1U) != 0U)
    {
        j["platformID"] = h.platform_id.to_string();
    }

    // Conditional partitionID (ValidationBits bit 2).
    if ((h.validation_bits & 0x4U) != 0U)
    {
        j["partitionID"] = h.partition_id.to_string();
    }

    j["creatorID"] = h.creator_id.to_string();

    j["notificationType"] = {
        {"guid", h.notification_type.to_string()},
        {"type", notificationTypeName(h.notification_type)},
    };

    j["recordID"] = h.record_id;
    j["flags"] = h.flags;
    j["persistenceInfo"] = h.persistence_info;
}

// --- CperRecord::parse
// --------------------------------------------------------

// Wire-format constants (all sizes in bytes, #pragma pack(push,1) equivalent).
static constexpr std::size_t kHeaderSize = 128U;
static constexpr std::size_t kDescriptorSize = 72U;

// EFI_COMMON_ERROR_RECORD_HEADER field offsets:
static constexpr std::size_t kOffSigStart = 0U;
static constexpr std::size_t kOffRevision = 4U;
static constexpr std::size_t kOffSigEnd = 6U;
static constexpr std::size_t kOffSectionCount = 10U;
static constexpr std::size_t kOffSeverity = 12U;
static constexpr std::size_t kOffValidBits = 16U;
static constexpr std::size_t kOffRecordLength = 20U;
static constexpr std::size_t kOffTimestamp = 24U;
static constexpr std::size_t kOffPlatformId = 32U;
static constexpr std::size_t kOffPartitionId = 48U;
static constexpr std::size_t kOffCreatorId = 64U;
static constexpr std::size_t kOffNotifType = 80U;
static constexpr std::size_t kOffRecordId = 96U;
static constexpr std::size_t kOffFlags = 104U;
static constexpr std::size_t kOffPersistInfo = 108U;

// EFI_ERROR_SECTION_DESCRIPTOR field offsets (relative to descriptor base):
static constexpr std::size_t kDOffSectionOffset = 0U;
static constexpr std::size_t kDOffSectionLength = 4U;
static constexpr std::size_t kDOffRevision = 8U;
static constexpr std::size_t kDOffSecValidMask = 10U;
static constexpr std::size_t kDOffSectionFlags = 12U;
static constexpr std::size_t kDOffSectionType = 16U;
static constexpr std::size_t kDOffFruId = 32U;
static constexpr std::size_t kDOffSeverity = 48U;
static constexpr std::size_t kDOffFruString = 52U;
static constexpr std::size_t kFruStringLen = 20U;

std::optional<CperRecord> CperRecord::parse(std::span<const uint8_t> buf,
                                            ParseError* error_out)
{
    return parse(buf, addc::DecodeOptions{}, error_out);
}

std::optional<CperRecord> CperRecord::parse(std::span<const uint8_t> buf,
                                            const addc::DecodeOptions& options,
                                            ParseError* error_out)
{
    ParseFailure failure;
    auto result = parse_detailed(buf, options, &failure);
    if (!result && (error_out != nullptr))
    {
        *error_out = failure.code;
    }
    return result;
}

std::string_view parse_error_message(ParseError error) noexcept
{
    switch (error)
    {
        case ParseError::TooSmall:
            return "CPER record is truncated";
        case ParseError::InvalidSignature:
            return "CPER signature is invalid";
        case ParseError::InvalidSectionCount:
            return "CPER section count is invalid";
        case ParseError::InvalidRecordLength:
            return "CPER record length is invalid";
        case ParseError::InvalidSectionLength:
            return "CPER section length is invalid";
        case ParseError::SectionOutOfBounds:
            return "CPER section is outside the record bounds";
        case ParseError::RecoveryRequired:
            return "CPER record requires tolerant recovery";
        case ParseError::InvalidSectionBody:
            return "CPER section body is invalid";
    }
    return "CPER parse failed";
}

std::optional<CperRecord> CperRecord::parse_detailed(
    std::span<const uint8_t> buf, const addc::DecodeOptions& options,
    ParseFailure* failure_out)
{
    auto fail = [&](ParseError error, std::size_t offset,
                    std::optional<std::size_t> section_index =
                        std::nullopt) -> std::optional<CperRecord> {
        if (failure_out)
        {
            *failure_out = ParseFailure{
                .code = error,
                .offset = offset,
                .section_index = section_index,
            };
        }
        return std::nullopt;
    };

    // -- Validate record header ----------------------------------------------
    if (buf.size() < kHeaderSize)
    {
        return fail(ParseError::TooSmall, buf.size());
    }

    constexpr uint32_t kSigStart = 0x52455043U; // "CPER"
    constexpr uint32_t kSigEnd = 0xFFFFFFFFU;

    if (readU32Le(buf, kOffSigStart) != kSigStart)
    {
        return fail(ParseError::InvalidSignature, kOffSigStart);
    }
    if (readU32Le(buf, kOffSigEnd) != kSigEnd)
    {
        return fail(ParseError::InvalidSignature, kOffSigEnd);
    }

    uint16_t section_count = readU16Le(buf, kOffSectionCount);
    if (section_count == 0U)
    {
        return fail(ParseError::InvalidSectionCount, kOffSectionCount);
    }

    // Enough bytes for all descriptors?
    const std::size_t descriptors_end =
        kHeaderSize + static_cast<std::size_t>(section_count) * kDescriptorSize;
    if (buf.size() < descriptors_end)
    {
        return fail(ParseError::TooSmall, buf.size());
    }

    // -- Parse record header ------------------------------------------------
    RecordHeader hdr{};
    const uint16_t revision = readU16Le(buf, kOffRevision);
    hdr.revision_major = static_cast<uint8_t>((revision >> 8U) & 0xFFU);
    hdr.revision_minor = static_cast<uint8_t>(revision & 0xFFU);
    hdr.section_count = section_count;
    hdr.error_severity = readU32Le(buf, kOffSeverity);
    hdr.validation_bits = readU32Le(buf, kOffValidBits);
    hdr.record_length = readU32Le(buf, kOffRecordLength);

    {
        auto& ts = hdr.timestamp;
        ts.seconds = buf[kOffTimestamp + 0U];
        ts.minutes = buf[kOffTimestamp + 1U];
        ts.hours = buf[kOffTimestamp + 2U];
        ts.flag = buf[kOffTimestamp + 3U];
        ts.day = buf[kOffTimestamp + 4U];
        ts.month = buf[kOffTimestamp + 5U];
        ts.year = buf[kOffTimestamp + 6U];
        ts.century = buf[kOffTimestamp + 7U];
    }

    hdr.platform_id = parseGuid(buf, kOffPlatformId);
    hdr.partition_id = parseGuid(buf, kOffPartitionId);
    hdr.creator_id = parseGuid(buf, kOffCreatorId);
    hdr.notification_type = parseGuid(buf, kOffNotifType);
    hdr.record_id = readU64Le(buf, kOffRecordId);
    hdr.flags.raw = readU32Le(buf, kOffFlags);
    hdr.persistence_info = readU64Le(buf, kOffPersistInfo);

    // -- Parse section descriptors -----------------------------------------
    CperRecord record{};
    record.header = hdr;
    record.descriptors.reserve(section_count);
    record.section_bodies.reserve(section_count);

    // Track the next sequential offset to recover from malformed records where
    // two section descriptors list the same sectionOffset (overlapping
    // descriptors).
    std::size_t logical_record_size = hdr.record_length;
    if (logical_record_size < descriptors_end ||
        logical_record_size > buf.size())
    {
        if (options.parse_mode == addc::ParseMode::Strict)
        {
            return fail(ParseError::InvalidRecordLength, kOffRecordLength);
        }
        record.repairs.push_back(ParseRepair{
            .code = "record_length_substituted",
            .field = "record_length",
            .section_index = std::nullopt,
            .original_value = hdr.record_length,
            .substituted_value = static_cast<uint64_t>(buf.size()),
            .reason =
                "declared record length does not contain the available record",
        });
        logical_record_size = buf.size();
    }

    std::size_t next_expected_offset = descriptors_end;

    for (uint16_t i = 0U; i < section_count; ++i)
    {
        const std::size_t desc_base =
            kHeaderSize + static_cast<std::size_t>(i) * kDescriptorSize;

        SectionDescriptor desc{};
        desc.section_offset = readU32Le(buf, desc_base + kDOffSectionOffset);
        desc.section_length = readU32Le(buf, desc_base + kDOffSectionLength);

        if (desc.section_length == 0U)
        {
            return fail(ParseError::InvalidSectionLength,
                        desc_base + kDOffSectionLength, i);
        }

        const uint16_t drev = readU16Le(buf, desc_base + kDOffRevision);
        desc.revision_major = static_cast<uint8_t>((drev >> 8U) & 0xFFU);
        desc.revision_minor = static_cast<uint8_t>(drev & 0xFFU);
        desc.sec_valid_mask = buf[desc_base + kDOffSecValidMask];

        const uint32_t raw_flags =
            readU32Le(buf, desc_base + kDOffSectionFlags);
        desc.flags.primary = (raw_flags & (1U << 0U)) != 0U;
        desc.flags.containment_warning = (raw_flags & (1U << 1U)) != 0U;
        desc.flags.reset = (raw_flags & (1U << 2U)) != 0U;
        desc.flags.error_threshold_exceeded = (raw_flags & (1U << 3U)) != 0U;
        desc.flags.resource_not_accessible = (raw_flags & (1U << 4U)) != 0U;
        desc.flags.latent_error = (raw_flags & (1U << 5U)) != 0U;
        desc.flags.propagated = (raw_flags & (1U << 6U)) != 0U;
        desc.flags.overflow = (raw_flags & (1U << 7U)) != 0U;

        desc.section_type = parseGuid(buf, desc_base + kDOffSectionType);
        desc.fru_id = parseGuid(buf, desc_base + kDOffFruId);
        desc.severity = readU32Le(buf, desc_base + kDOffSeverity);

        if ((desc.sec_valid_mask & 0x2U) != 0U)
        {
            desc.fru_text = addc::detail::read_ascii_field(
                buf, desc_base + kDOffFruString, kFruStringLen);
        }

        // -- Copy section body ---------------------------------------------
        const std::size_t body_offset_raw =
            static_cast<std::size_t>(desc.section_offset);
        const std::size_t body_len =
            static_cast<std::size_t>(desc.section_length);

        // Recovery for malformed records: some CPER records list the same
        // sectionOffset for multiple sections. Only a physically bounded
        // sequential substitution is safe; otherwise the record is not
        // repairable.
        std::size_t body_offset;
        if (body_offset_raw < next_expected_offset)
        {
            if (options.parse_mode == addc::ParseMode::Strict)
            {
                return fail(ParseError::RecoveryRequired,
                            desc_base + kDOffSectionOffset, i);
            }
            const bool seq_fits = next_expected_offset <= buf.size() &&
                                  body_len <= buf.size() - next_expected_offset;
            if (!seq_fits)
            {
                return fail(ParseError::SectionOutOfBounds,
                            desc_base + kDOffSectionOffset, i);
            }
            body_offset = next_expected_offset;
            record.repairs.push_back(ParseRepair{
                .code = "section_offset_substituted",
                .field = "section_offset",
                .section_index = i,
                .original_value = body_offset_raw,
                .substituted_value = body_offset,
                .reason = "declared section overlaps preceding record data",
            });
        }
        else
        {
            body_offset = body_offset_raw;
        }

        if (body_offset > buf.size() || body_len > buf.size() - body_offset)
        {
            return fail(ParseError::SectionOutOfBounds,
                        desc_base + kDOffSectionOffset, i);
        }

        if (body_offset > logical_record_size ||
            body_len > logical_record_size - body_offset)
        {
            if (options.parse_mode == addc::ParseMode::Strict)
            {
                return fail(ParseError::SectionOutOfBounds,
                            desc_base + kDOffSectionOffset, i);
            }
            record.repairs.push_back(ParseRepair{
                .code = "record_length_extended",
                .field = "record_length",
                .section_index = i,
                .original_value = logical_record_size,
                .substituted_value = static_cast<uint64_t>(buf.size()),
                .reason = "section is physically present beyond the declared "
                          "record length",
            });
            logical_record_size = buf.size();
        }

        std::vector<uint8_t> body(body_len);
        std::copy_n(buf.data() + body_offset, body_len, body.begin());

        next_expected_offset = body_offset + body_len;

        record.descriptors.push_back(std::move(desc));
        record.section_bodies.push_back(std::move(body));
    }

    return record;
}

} // namespace addc::cper
