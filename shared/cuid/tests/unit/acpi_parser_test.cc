// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "unit/acpi_parser_test.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "src/acpi_parser.h"

namespace {

// Append the raw bytes of a packed firmware struct to a table image.
template <typename T>
void append_struct(std::vector<uint8_t>& table, const T& value) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
  table.insert(table.end(), bytes, bytes + sizeof(T));
}

// Patch the header length field and recompute the ACPI checksum so the table
// passes validation. Must be called after all entries have been appended.
void finalize(std::vector<uint8_t>& table) {
  const uint32_t length = static_cast<uint32_t>(table.size());
  std::memcpy(table.data() + offsetof(AcpiTableHeader, length), &length, sizeof(length));

  table[offsetof(AcpiTableHeader, checksum)] = 0;
  uint8_t sum = 0;
  for (uint8_t b : table) {
    sum = static_cast<uint8_t>(sum + b);
  }
  table[offsetof(AcpiTableHeader, checksum)] = static_cast<uint8_t>(0u - sum);
}

// A MADT image with a valid signature/length/checksum and no entries.
std::vector<uint8_t> make_empty_madt() {
  MadtHeader madt{};
  std::memcpy(madt.header.signature, "APIC", 4);
  madt.header.revision = 5;

  std::vector<uint8_t> table;
  append_struct(table, madt);
  return table;
}

MadtLocalApic make_local_apic(uint8_t uid, uint8_t apic_id, bool enabled) {
  MadtLocalApic e{};
  e.header.type = 0;
  e.header.length = sizeof(MadtLocalApic);
  e.acpi_processor_uid = uid;
  e.apic_id = apic_id;
  e.flags = enabled ? 1u : 0u;
  return e;
}

MadtLocalX2Apic make_local_x2apic(uint32_t uid, uint32_t x2apic_id, bool enabled) {
  MadtLocalX2Apic e{};
  e.header.type = 9;
  e.header.length = sizeof(MadtLocalX2Apic);
  e.x2apic_id = x2apic_id;
  e.flags = enabled ? 1u : 0u;
  e.acpi_processor_uid = uid;
  return e;
}

}  // namespace

TestAcpiMadtParse::TestAcpiMadtParse() {
  SetTitle("ACPI MADT Parse");
  SetDescription(
      "Parse synthetic MADT tables and verify that malformed/truncated tables are "
      "rejected without reading past the end of the buffer.");
}

void TestAcpiMadtParse::SetUp() {}

void TestAcpiMadtParse::Run() {
  std::vector<AcpiCpuInfo> cpus;

  // Happy path: one Local APIC entry and one x2APIC entry, plus an unknown
  // entry type that must be skipped rather than reported.
  {
    std::vector<uint8_t> table = make_empty_madt();
    append_struct(table, make_local_apic(/*uid=*/1, /*apic_id=*/0x11, /*enabled=*/true));
    // Unknown structure type 4 (Local APIC NMI), 6 bytes long.
    const uint8_t nmi[6] = {4, 6, 0, 0, 0, 0};
    table.insert(table.end(), nmi, nmi + sizeof(nmi));
    append_struct(table, make_local_x2apic(/*uid=*/2, /*x2apic_id=*/0x2222, /*enabled=*/false));
    finalize(table);

    ASSERT_EQ(AcpiParser::parse_madt_buffer(table.data(), table.size(), cpus),
              AMDCUID_STATUS_SUCCESS);
    ASSERT_EQ(cpus.size(), 2u);

    EXPECT_EQ(cpus[0].apic_id, 0x11u);
    EXPECT_EQ(cpus[0].processor_uid, 1u);
    EXPECT_TRUE(cpus[0].enabled);
    EXPECT_FALSE(cpus[0].is_x2apic);

    EXPECT_EQ(cpus[1].apic_id, 0x2222u);
    EXPECT_EQ(cpus[1].processor_uid, 2u);
    EXPECT_FALSE(cpus[1].enabled);
    EXPECT_TRUE(cpus[1].is_x2apic);
  }

  // Regression: a table whose final byte begins an entry. The old walker
  // tested only `entry < end` and then read entry->length, one byte past the
  // buffer. It must now be rejected.
  {
    std::vector<uint8_t> table = make_empty_madt();
    append_struct(table, make_local_apic(1, 0x11, true));
    table.push_back(0);  // dangling entry-header type byte, no length byte
    finalize(table);

    EXPECT_EQ(AcpiParser::parse_madt_buffer(table.data(), table.size(), cpus),
              AMDCUID_STATUS_INVALID_FORMAT);
  }

  // An entry whose declared length runs past the end of the table.
  {
    std::vector<uint8_t> table = make_empty_madt();
    MadtLocalApic e = make_local_apic(1, 0x11, true);
    e.header.length = sizeof(MadtLocalApic) + 32;  // lies about its size
    append_struct(table, e);
    finalize(table);

    EXPECT_EQ(AcpiParser::parse_madt_buffer(table.data(), table.size(), cpus),
              AMDCUID_STATUS_INVALID_FORMAT);
  }

  // A zero-length entry would never advance the walker; it must be rejected
  // rather than spun on.
  {
    std::vector<uint8_t> table = make_empty_madt();
    const uint8_t zero_len[4] = {0, 0, 0, 0};
    table.insert(table.end(), zero_len, zero_len + sizeof(zero_len));
    finalize(table);

    EXPECT_EQ(AcpiParser::parse_madt_buffer(table.data(), table.size(), cpus),
              AMDCUID_STATUS_INVALID_FORMAT);
  }

  // A Local APIC entry that declares a length smaller than the structure it
  // claims to be must be skipped, not decoded from adjacent bytes.
  {
    std::vector<uint8_t> table = make_empty_madt();
    const uint8_t short_apic[4] = {0, 4, 7, 7};  // type 0, length 4 (< 8)
    table.insert(table.end(), short_apic, short_apic + sizeof(short_apic));
    finalize(table);

    // No usable CPU entries were found.
    EXPECT_EQ(AcpiParser::parse_madt_buffer(table.data(), table.size(), cpus),
              AMDCUID_STATUS_DEVICE_NOT_FOUND);
    EXPECT_TRUE(cpus.empty());
  }

  // Degenerate inputs must not dereference anything.
  {
    EXPECT_EQ(AcpiParser::parse_madt_buffer(nullptr, 0, cpus), AMDCUID_STATUS_INVALID_FORMAT);

    std::vector<uint8_t> truncated = make_empty_madt();
    truncated.resize(sizeof(AcpiTableHeader));  // shorter than a MadtHeader
    EXPECT_EQ(AcpiParser::parse_madt_buffer(truncated.data(), truncated.size(), cpus),
              AMDCUID_STATUS_INVALID_FORMAT);
  }

  // Wrong signature and bad checksum are both rejected.
  {
    std::vector<uint8_t> table = make_empty_madt();
    append_struct(table, make_local_apic(1, 0x11, true));
    finalize(table);

    std::vector<uint8_t> wrong_sig = table;
    wrong_sig[0] = 'X';
    EXPECT_EQ(AcpiParser::parse_madt_buffer(wrong_sig.data(), wrong_sig.size(), cpus),
              AMDCUID_STATUS_INVALID_FORMAT);

    std::vector<uint8_t> bad_sum = table;
    bad_sum[offsetof(AcpiTableHeader, checksum)] ^= 0xFF;
    EXPECT_EQ(AcpiParser::parse_madt_buffer(bad_sum.data(), bad_sum.size(), cpus),
              AMDCUID_STATUS_INVALID_FORMAT);

    // Header length disagreeing with the actual buffer size is rejected too.
    std::vector<uint8_t> short_buf = table;
    short_buf.pop_back();
    EXPECT_EQ(AcpiParser::parse_madt_buffer(short_buf.data(), short_buf.size(), cpus),
              AMDCUID_STATUS_INVALID_FORMAT);
  }
}
