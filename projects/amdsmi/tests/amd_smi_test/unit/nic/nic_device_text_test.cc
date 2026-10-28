// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Text read from a device, a driver or firmware reaches the debug log and the CLI. A byte outside
 * printable ASCII, such as an escape sequence or a line break, must not get that far.
 */

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "nic_unit_fixture.h"
#include "smi_nic_text.h"
#include "smi_sysfs.h"

namespace {

constexpr auto kPathTemplate = "/tmp/amdsmi_nic_text_XXXXXX";

class SysfsFile_t {
 public:
  explicit SysfsFile_t(const std::string& content) {
    auto path = std::string{kPathTemplate};
    const auto fd = mkstemp(path.data());
    if (fd >= 0) {
      close(fd);
      m_path = path;
      std::ofstream{m_path} << content;
    }
  }
  ~SysfsFile_t() {
    if (!m_path.empty()) {
      unlink(m_path.c_str());
    }
  }

  SysfsFile_t(const SysfsFile_t&) = delete;
  SysfsFile_t& operator=(const SysfsFile_t&) = delete;

  auto path() const -> const std::string& { return m_path; }

 private:
  std::string m_path;
};

TEST_F(NicUnit, SysfsLineLosesControlBytesInASingleToken) {
  const SysfsFile_t file{"1.0\x1b[31m-rc\n"};
  auto value = SmiSysfsReader::SysfsValue{};

  ASSERT_EQ(SmiSysfsReader::readLine(file.path(), value), SmiSysfsReader::SysfsStatus::Success);

  EXPECT_EQ(std::get<std::string>(value), "1.0?[31m-rc");
}

TEST_F(NicUnit, SysfsLineLosesControlBytesInATextLine) {
  const SysfsFile_t file{"fw 1.0\x07 build\n"};
  auto value = SmiSysfsReader::SysfsValue{};

  ASSERT_EQ(SmiSysfsReader::readLine(file.path(), value), SmiSysfsReader::SysfsStatus::Success);

  EXPECT_EQ(std::get<std::string>(value), "fw 1.0? build");
}

TEST_F(NicUnit, SysfsReadAllLosesControlBytesInEachValue) {
  const SysfsFile_t file{"plain\nfw\x1b[0m 2\n"};
  auto values = std::vector<SmiSysfsReader::SysfsValue>{};

  ASSERT_EQ(SmiSysfsReader::readAll(file.path(), values), SmiSysfsReader::SysfsStatus::Success);

  ASSERT_EQ(values.size(), size_t{2});
  EXPECT_EQ(std::get<std::string>(values[0]), "plain");
  EXPECT_EQ(std::get<std::string>(values[1]), "fw?[0m 2");
}

TEST_F(NicUnit, SysfsPrintableTextIsReadUnchanged) {
  const SysfsFile_t file{"fw-1.2.3_rc4/x86\n"};
  auto value = SmiSysfsReader::SysfsValue{};

  ASSERT_EQ(SmiSysfsReader::readLine(file.path(), value), SmiSysfsReader::SysfsStatus::Success);

  EXPECT_EQ(std::get<std::string>(value), "fw-1.2.3_rc4/x86");
}

TEST_F(NicUnit, SysfsNumberTooLargeForAnIntIsStillPrintableText) {
  const SysfsFile_t file{"0xffffffffffffffffffff\x1b\n"};
  auto value = SmiSysfsReader::SysfsValue{};

  ASSERT_EQ(SmiSysfsReader::readLine(file.path(), value), SmiSysfsReader::SysfsStatus::Success);

  EXPECT_EQ(std::get<std::string>(value), "0xffffffffffffffffffff?");
}

TEST_F(NicUnit, SysfsNegativeNumberIsNotANumber) {
  EXPECT_EQ(parse_sysfs_uint(SmiSysfsReader::SysfsValue{std::string{"-1"}}), std::nullopt);
}

// The PCI core writes the link speed as text with a unit.
TEST_F(NicUnit, SysfsNumberFollowedByAUnitIsTheNumber) {
  EXPECT_EQ(parse_sysfs_uint(SmiSysfsReader::SysfsValue{std::string{"16.0 GT/s PCIe"}}),
            uint64_t{16});
}

TEST_F(NicUnit, SysfsDecimalAndHexNumbersAreParsed) {
  EXPECT_EQ(parse_sysfs_uint(SmiSysfsReader::SysfsValue{std::string{"10"}}), uint64_t{10});
  EXPECT_EQ(parse_sysfs_uint(SmiSysfsReader::SysfsValue{std::string{"0x10"}}), uint64_t{16});
  EXPECT_EQ(parse_sysfs_uint(SmiSysfsReader::SysfsValue{5}), uint64_t{5});
}

// ethtool_drvinfo text fields are not guaranteed NUL-terminated.
TEST_F(NicUnit, FixedTextFieldWithoutATerminatorIsReadWithinItsSize) {
  constexpr auto kFieldSize = size_t{4};
  const char field[kFieldSize] = {'f', 'w', '\x1b', '9'};

  EXPECT_EQ(amd::smi::nic::text::printable_field(field), "fw?9");
}

TEST_F(NicUnit, FixedTextFieldStopsAtItsTerminator) {
  const char field[8] = {'v', '1', '\0', 'x', 'x', 'x', 'x', 'x'};

  EXPECT_EQ(amd::smi::nic::text::printable_field(field), "v1");
}

}  // namespace
