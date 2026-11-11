// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Coverage for the ESMI -> AMD SMI status mapping. The enumerators are read
// from the vendored e_smi.h, so a status added upstream fails this test and is
// named in the output until esmi_status_map is updated. No GPU required.

#if ENABLE_ESMI_LIB

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_common.h"

namespace {

struct EsmiEnumerator {
  std::string name;
  int value;
};

// Parses the esmi_status_t enumerators in declaration order, honouring explicit
// "= N" values. Aliases such as ESMI_INITIALIZED are kept so they are checked too.
std::vector<EsmiEnumerator> ParseEsmiStatusEnumerators(const std::string& text) {
  std::vector<EsmiEnumerator> result;
  const auto end = text.find("} esmi_status_t;");
  if (end == std::string::npos) return result;
  const auto start = text.rfind("typedef enum {", end);
  if (start == std::string::npos) return result;

  static const std::regex kEntry(R"(^\s*(ESMI_[A-Z0-9_]+)\s*(=\s*(\d+))?\s*,)");
  std::istringstream lines(text.substr(start, end - start));
  std::string line;
  int next_value = 0;
  while (std::getline(lines, line)) {
    std::smatch match;
    if (!std::regex_search(line, match, kEntry)) continue;
    const int value = match[3].matched ? std::stoi(match[3].str()) : next_value;
    result.push_back({match[1].str(), value});
    next_value = value + 1;
  }
  return result;
}

std::string ReadEsmiStatusHeader() {
  // Source-tree location relative to this file. Installed tests do not ship the
  // header, so the caller skips when it cannot be read.
  const std::filesystem::path header = std::filesystem::path(__FILE__).parent_path() /
                                       "../../../../esmi_ib_library/include/e_smi/e_smi.h";
  std::ifstream in(header);
  if (!in) return {};
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

}  // namespace

TEST(SystemUnit, EsmiStatusMapEveryEsmiStatusIsMapped) {
  const std::string header = ReadEsmiStatusHeader();
  if (header.empty()) GTEST_SKIP() << "e_smi.h not found next to the source tree";

  const auto enumerators = ParseEsmiStatusEnumerators(header);
  ASSERT_FALSE(enumerators.empty()) << "No esmi_status_t enumerators parsed from e_smi.h";

  std::string missing;
  for (const auto& e : enumerators) {
    if (amd::smi::esmi_status_map.find(static_cast<esmi_status_t>(e.value)) ==
        amd::smi::esmi_status_map.end()) {
      missing += (missing.empty() ? "" : ", ") + e.name;
    }
  }
  EXPECT_TRUE(missing.empty()) << "esmi_status_t values missing from esmi_status_map: " << missing;
}

TEST(SystemUnit, EsmiStatusMapOnlyEsmiSuccessMapsToSuccess) {
  const std::string header = ReadEsmiStatusHeader();
  if (header.empty()) GTEST_SKIP() << "e_smi.h not found next to the source tree";

  for (const auto& e : ParseEsmiStatusEnumerators(header)) {
    const auto mapped = amd::smi::esmi_to_amdsmi_status(static_cast<esmi_status_t>(e.value));
    if (e.value == ESMI_SUCCESS) {
      EXPECT_EQ(mapped, AMDSMI_STATUS_SUCCESS) << e.name;
    } else {
      EXPECT_NE(mapped, AMDSMI_STATUS_SUCCESS) << e.name << " must not report success";
    }
  }
}

TEST(SystemUnit, EsmiStatusMapUnknownValueReturnsMapError) {
  EXPECT_EQ(amd::smi::esmi_to_amdsmi_status(static_cast<esmi_status_t>(0x7FFF)),
            AMDSMI_STATUS_MAP_ERROR);
}

TEST(SystemUnit, EsmiStatusMapErrMsgLooksUpByMappedStatus) {
  const char* msg = nullptr;
  EXPECT_EQ(amdsmi_get_esmi_err_msg(AMDSMI_STATUS_INVAL, &msg), AMDSMI_STATUS_SUCCESS);
  ASSERT_NE(msg, nullptr);
  EXPECT_STREQ(msg, esmi_get_err_msg(ESMI_INVALID_INPUT));
}

TEST(SystemUnit, EsmiStatusMapErrMsgForSharedStatusUsesFirstEsmiCode) {
  // AMDSMI_STATUS_BUSY is shared by ESMI_DEV_BUSY and ESMI_SMU_BUSY. The first
  // match in esmi_status_map order is reported.
  const char* msg = nullptr;
  EXPECT_EQ(amdsmi_get_esmi_err_msg(AMDSMI_STATUS_BUSY, &msg), AMDSMI_STATUS_SUCCESS);
  ASSERT_NE(msg, nullptr);
  EXPECT_STREQ(msg, esmi_get_err_msg(ESMI_DEV_BUSY));
}

TEST(SystemUnit, EsmiStatusMapErrMsgCoversCodesMissingFromVendoredEsmi) {
  // The vendored esmi_get_err_msg() has no case for this, so it returns
  // "Unknown error". The AMD SMI helper supplies the text instead.
  EXPECT_STRNE(amd::smi::esmi_status_message(ESMI_HSMP_TIMEOUT), "Unknown error");

  const char* msg = nullptr;
  EXPECT_EQ(amdsmi_get_esmi_err_msg(AMDSMI_STATUS_HSMP_TIMEOUT, &msg), AMDSMI_STATUS_SUCCESS);
  ASSERT_NE(msg, nullptr);
  EXPECT_STREQ(msg, amd::smi::esmi_status_message(ESMI_HSMP_TIMEOUT));
}

TEST(SystemUnit, EsmiStatusMapErrMsgUnmappedStatusReturnsMapError) {
  const char* msg = nullptr;
  EXPECT_EQ(amdsmi_get_esmi_err_msg(static_cast<amdsmi_status_t>(0x7FFF), &msg),
            AMDSMI_STATUS_MAP_ERROR);
  EXPECT_EQ(msg, nullptr);
}

#endif  // ENABLE_ESMI_LIB
