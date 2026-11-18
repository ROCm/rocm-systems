// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_TEST_UNIT_ACPI_PARSER_TEST_H_
#define CUID_TEST_UNIT_ACPI_PARSER_TEST_H_

#include "test_base.h"

// Parses synthetic MADT images through AcpiParser::parse_madt_buffer. No root
// and no real /sys/firmware/acpi/tables/APIC required. Covers the happy path
// (Local APIC + x2APIC entries) and the malformed cases that previously read
// past the end of the table buffer.
class TestAcpiMadtParse : public TestBase {
 public:
  TestAcpiMadtParse();
  void SetUp() override;
  void Run() override;
};

#endif  // CUID_TEST_UNIT_ACPI_PARSER_TEST_H_
