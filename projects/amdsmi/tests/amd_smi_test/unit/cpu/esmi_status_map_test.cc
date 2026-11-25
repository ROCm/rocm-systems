// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Coverage for the ESMI -> AMD SMI status mapping. The enumerators are read
// from the vendored e_smi.h, so a status added upstream fails this test and is
// named in the output until esmi_status_map is updated. No GPU required.

#if ENABLE_ESMI_LIB

#include <gtest/gtest.h>

#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_common.h"
#include "amd_smi/impl/amd_smi_system.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "esmi_status_enum_text.h"
#include "test_fixture_utils.h"

namespace {

struct EsmiEnumerator {
  std::string name;
  int value;
};

// Parses the esmi_status_t enumerators in declaration order, honouring explicit
// "= N" values. Aliases such as ESMI_INITIALIZED are kept so they are checked too.
// Fails the test (and returns nothing) when an entry does not use a plain
// "NAME" or "NAME = <decimal>" form, so an unsupported initializer cannot be skipped silently.
std::vector<EsmiEnumerator> ParseEsmiStatusEnumerators(const std::string& text) {
  std::vector<EsmiEnumerator> result;
  const auto end = text.find("} esmi_status_t;");
  if (end == std::string::npos) return result;
  const auto start = text.rfind("typedef enum {", end);
  if (start == std::string::npos) return result;

  std::string body = text.substr(start + std::string("typedef enum {").size(),
                                 end - start - std::string("typedef enum {").size());
  body = std::regex_replace(body, std::regex(R"(/\*[\s\S]*?\*/)"), " ");
  body = std::regex_replace(body, std::regex(R"(//[^\n]*)"), " ");

  static const std::regex kEntry(R"(^\s*(ESMI_[A-Z0-9_]+)\s*(=\s*(\d+))?\s*$)");
  std::istringstream entries(body);
  std::string entry;
  int next_value = 0;
  while (std::getline(entries, entry, ',')) {
    if (entry.find_first_not_of(" \t\r\n") == std::string::npos) continue;  // trailing comma
    std::smatch match;
    if (!std::regex_match(entry, match, kEntry)) {
      ADD_FAILURE() << "Unrecognized esmi_status_t entry, update the parser: '" << entry << "'";
      return {};
    }
    const int value = match[3].matched ? std::stoi(match[3].str()) : next_value;
    result.push_back({match[1].str(), value});
    next_value = value + 1;
  }
  return result;
}

// The esmi_status_t enum text, embedded at build time from the e_smi.h the library was built
// against (empty if that header was not available).
std::string ReadEsmiStatusHeader() { return kEsmiStatusEnumText; }

}  // namespace

TEST(CpuUnit, EsmiStatusMapEveryEsmiStatusIsMapped) {
  const std::string header = ReadEsmiStatusHeader();
  if (header.empty()) GTEST_SKIP() << "esmi_status_t text was not embedded at build time";

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

TEST(CpuUnit, EsmiStatusMapOnlyEsmiSuccessMapsToSuccess) {
  const std::string header = ReadEsmiStatusHeader();
  if (header.empty()) GTEST_SKIP() << "esmi_status_t text was not embedded at build time";

  for (const auto& e : ParseEsmiStatusEnumerators(header)) {
    const auto mapped = amd::smi::esmi_to_amdsmi_status(static_cast<esmi_status_t>(e.value));
    if (e.value == ESMI_SUCCESS) {
      EXPECT_EQ(mapped, AMDSMI_STATUS_SUCCESS) << e.name;
    } else {
      EXPECT_NE(mapped, AMDSMI_STATUS_SUCCESS) << e.name << " must not report success";
    }
  }
}

TEST(CpuUnit, EsmiStatusMapNewMappingsAreExplicit) {
  using amd::smi::esmi_to_amdsmi_status;
  EXPECT_EQ(esmi_to_amdsmi_status(ESMI_PRE_REQ_NOT_SAT), AMDSMI_STATUS_API_FAILED);
  EXPECT_EQ(esmi_to_amdsmi_status(ESMI_SMU_BUSY), AMDSMI_STATUS_BUSY);
  EXPECT_EQ(esmi_to_amdsmi_status(ESMI_METRICS_TABLE_NOT_SUPPORTED), AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(CpuUnit, EsmiStatusMapUnknownValueReturnsMapError) {
  // 31 is not an esmi_status_t enumerator in the pinned e_smi.h and has no map entry.
  const auto unmapped = static_cast<esmi_status_t>(31);
  ASSERT_EQ(amd::smi::esmi_status_map.count(unmapped), 0u);
  EXPECT_EQ(amd::smi::esmi_to_amdsmi_status(unmapped), AMDSMI_STATUS_MAP_ERROR);
}

TEST(CpuUnit, PublicCpuGetterReturnsMapErrorForUnmappedEsmiStatusAndKeepsOutput) {
  // Make the getter see an in-range ESMI status that has no map entry. The old behavior returned
  // AMDSMI_STATUS_SUCCESS here and left the output unset; it must now be MAP_ERROR with the
  // caller's output untouched.
  constexpr int kUnmapped = 31;
  ASSERT_EQ(amd::smi::esmi_status_map.count(static_cast<esmi_status_t>(kUnmapped)), 0u);

  amd::smi::amdsmi_library_init_ref_acquire();  // the getter checks the library is initialized
  int dummy = 0;  // the getter only rejects a null handle
  const auto handle = reinterpret_cast<amdsmi_processor_handle>(&dummy);
  constexpr uint32_t kSentinel = 0xDEADBEEF;
  uint32_t proto_ver = kSentinel;
  amdsmi_status_t status;
  {
    amd::smi::testing::ScopedOverride<int> guard(
        [] { return amd::smi::test_hooks::EsmiStatusOverride(); },
        [](const int& v) { amd::smi::test_hooks::SetEsmiStatusOverride(v); }, kUnmapped);
    status = amdsmi_get_cpu_hsmp_proto_ver(handle, &proto_ver);
  }
  amd::smi::amdsmi_library_init_ref_release();

  EXPECT_EQ(status, AMDSMI_STATUS_MAP_ERROR);
  EXPECT_EQ(proto_ver, kSentinel);
}

TEST(CpuUnit, EsmiStatusMapErrMsgLooksUpByMappedStatus) {
  const char* msg = nullptr;
  EXPECT_EQ(amdsmi_get_esmi_err_msg(AMDSMI_STATUS_INVAL, &msg), AMDSMI_STATUS_SUCCESS);
  ASSERT_NE(msg, nullptr);
  EXPECT_STREQ(msg, esmi_get_err_msg(ESMI_INVALID_INPUT));
}

TEST(CpuUnit, EsmiStatusMapErrMsgForSharedStatusUsesChosenEsmiCode) {
  // AMDSMI_STATUS_BUSY (ESMI_DEV_BUSY, ESMI_SMU_BUSY) and AMDSMI_STATUS_NOT_SUPPORTED
  // (ESMI_NOT_SUPPORTED, ESMI_METRICS_TABLE_NOT_SUPPORTED) each cover two ESMI codes. The message
  // is chosen explicitly, not by map order: the generic ESMI_DEV_BUSY / ESMI_NOT_SUPPORTED text.
  const char* msg = nullptr;
  EXPECT_EQ(amdsmi_get_esmi_err_msg(AMDSMI_STATUS_BUSY, &msg), AMDSMI_STATUS_SUCCESS);
  ASSERT_NE(msg, nullptr);
  EXPECT_STREQ(msg, esmi_get_err_msg(ESMI_DEV_BUSY));

  msg = nullptr;
  EXPECT_EQ(amdsmi_get_esmi_err_msg(AMDSMI_STATUS_NOT_SUPPORTED, &msg), AMDSMI_STATUS_SUCCESS);
  ASSERT_NE(msg, nullptr);
  EXPECT_STREQ(msg, esmi_get_err_msg(ESMI_NOT_SUPPORTED));
}

TEST(CpuUnit, EsmiStatusMapErrMsgUnmappedStatusesReturnMapErrorAndKeepOutput) {
  // Every status defined in amdsmi_status_t. Update this list when a status is added to amdsmi.h.
  const std::vector<amdsmi_status_t> all_statuses = {
      AMDSMI_STATUS_SUCCESS,           AMDSMI_STATUS_INVAL,
      AMDSMI_STATUS_NOT_SUPPORTED,     AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
      AMDSMI_STATUS_FAIL_LOAD_MODULE,  AMDSMI_STATUS_FAIL_LOAD_SYMBOL,
      AMDSMI_STATUS_DRM_ERROR,         AMDSMI_STATUS_API_FAILED,
      AMDSMI_STATUS_TIMEOUT,           AMDSMI_STATUS_RETRY,
      AMDSMI_STATUS_NO_PERM,           AMDSMI_STATUS_INTERRUPT,
      AMDSMI_STATUS_IO,                AMDSMI_STATUS_ADDRESS_FAULT,
      AMDSMI_STATUS_FILE_ERROR,        AMDSMI_STATUS_OUT_OF_RESOURCES,
      AMDSMI_STATUS_INTERNAL_EXCEPTION, AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS,
      AMDSMI_STATUS_INIT_ERROR,        AMDSMI_STATUS_REFCOUNT_OVERFLOW,
      AMDSMI_STATUS_DIRECTORY_NOT_FOUND, AMDSMI_STATUS_IPC_ERROR,
      AMDSMI_STATUS_BUSY,              AMDSMI_STATUS_NOT_FOUND,
      AMDSMI_STATUS_NOT_INIT,          AMDSMI_STATUS_NO_SLOT,
      AMDSMI_STATUS_DRIVER_NOT_LOADED, AMDSMI_STATUS_MORE_DATA,
      AMDSMI_STATUS_NO_DATA,           AMDSMI_STATUS_INSUFFICIENT_SIZE,
      AMDSMI_STATUS_UNEXPECTED_SIZE,   AMDSMI_STATUS_UNEXPECTED_DATA,
      AMDSMI_STATUS_NON_AMD_CPU,       AMDSMI_STATUS_NO_ENERGY_DRV,
      AMDSMI_STATUS_NO_MSR_DRV,        AMDSMI_STATUS_NO_HSMP_DRV,
      AMDSMI_STATUS_NO_HSMP_SUP,       AMDSMI_STATUS_NO_HSMP_MSG_SUP,
      AMDSMI_STATUS_HSMP_TIMEOUT,      AMDSMI_STATUS_NO_DRV,
      AMDSMI_STATUS_FILE_NOT_FOUND,    AMDSMI_STATUS_ARG_PTR_NULL,
      AMDSMI_STATUS_AMDGPU_RESTART_ERR, AMDSMI_STATUS_SETTING_UNAVAILABLE,
      AMDSMI_STATUS_CORRUPTED_EEPROM,  AMDSMI_STATUS_MAP_ERROR,
      AMDSMI_STATUS_UNKNOWN_ERROR};

  std::set<amdsmi_status_t> mapped;
  for (const auto& entry : amd::smi::esmi_status_map) mapped.insert(entry.second);

  size_t checked = 0;
  for (const auto status : all_statuses) {
    if (mapped.count(status) != 0) continue;
    ++checked;
    const char* const kSentinel = "sentinel";
    const char* msg = kSentinel;
    EXPECT_EQ(amdsmi_get_esmi_err_msg(status, &msg), AMDSMI_STATUS_MAP_ERROR)
        << "status=" << static_cast<uint32_t>(status);
    EXPECT_EQ(msg, kSentinel) << "output changed for status=" << static_cast<uint32_t>(status);
  }
  EXPECT_GT(checked, 0u);
}

TEST(CpuUnit, CpuModelNameReturnsAmdSmiStatusNotEsmiStatus) {
  // get_cpu_model_name() reads /proc/cpuinfo and reports AMD SMI statuses directly. A socket
  // that does not exist must yield NO_DATA (or FILE_ERROR without /proc), never MAP_ERROR,
  // which is what routing the status through esmi_to_amdsmi_status() produced.
  std::string name;
  const auto status = amd::smi::AMDSmiSystem::getInstance().get_cpu_model_name(0xFFFFFFFFu, &name);
  EXPECT_TRUE(status == AMDSMI_STATUS_NO_DATA || status == AMDSMI_STATUS_FILE_ERROR)
      << "status=" << status;
  EXPECT_NE(status, AMDSMI_STATUS_MAP_ERROR);
}

#endif  // ENABLE_ESMI_LIB
