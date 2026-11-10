// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * The helpers behind the NIC statistics getters: how many entries reach the caller's array and
 * which status says so, and which statistics scopes are accepted.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "amd_smi/impl/amd_smi_nic_stats.h"
#include "nic_unit_fixture.h"

namespace {

using amd::smi::copy_nic_fw_info;
using amd::smi::copy_nic_stats;
using amd::smi::get_nic_stats;
using amd::smi::was_nic_stat_scope_resolved;

constexpr auto kUnsetValue = uint64_t{0xDEADBEEF};
constexpr auto kFirstStatValue = uint64_t{100};
constexpr auto kUnknownScope = int{2};

// The info struct holds hundreds of KB, so it lives on the heap.
auto make_info(uint32_t count) -> std::unique_ptr<smi_nic_stat_info_t> {
  auto info = std::make_unique<smi_nic_stat_info_t>();
  *info = {};
  info->count = count;
  for (auto i = uint32_t{0}; i < count; ++i) {
    std::snprintf(info->stats[i].name, sizeof(info->stats[i].name), "stat_%u", i);
    info->stats[i].value = (kFirstStatValue + i);
  }
  return info;
}

auto make_out(uint32_t capacity) -> std::vector<amdsmi_nic_stat_t> {
  auto out = std::vector<amdsmi_nic_stat_t>(capacity);
  for (auto& stat : out) {
    stat = {};
    stat.value = kUnsetValue;
  }
  return out;
}

// The shim's firmware struct holds 64 versions of about 0.5 KB each, so it lives on the heap.
auto make_fw_info(
    const std::vector<std::tuple<smi_nic_fw_version_type_t, const char*, const char*>>& versions)
    -> std::unique_ptr<smi_nic_fw_info_t> {
  auto info = std::make_unique<smi_nic_fw_info_t>();
  *info = {};
  info->count = static_cast<uint32_t>(versions.size());
  for (auto i = size_t{0}; i < versions.size(); ++i) {
    const auto& [type, name, version] = versions[i];
    info->versions[i].type = type;
    std::snprintf(info->versions[i].name, sizeof(info->versions[i].name), "%s", name);
    std::snprintf(info->versions[i].version, sizeof(info->versions[i].version), "%s", version);
  }
  return info;
}

TEST_F(NicUnit, FwInfoCopyKeepsTheNameAndVersionOfEachEntry) {
  const auto shim = make_fw_info({{SMI_NIC_FW_VERSION_TYPE_FIXED, "board.id", "A"},
                                  {SMI_NIC_FW_VERSION_TYPE_RUNNING, "fw.version", "1.0"}});
  auto info = amdsmi_nic_fw_info_t{};

  copy_nic_fw_info(*shim, &info);

  EXPECT_EQ(info.num_fw, 2U);
  EXPECT_STREQ(info.fw[0].name, "board.id");
  EXPECT_STREQ(info.fw[0].version, "A");
  EXPECT_STREQ(info.fw[1].name, "fw.version");
  EXPECT_STREQ(info.fw[1].version, "1.0");
}

TEST_F(NicUnit, FwInfoCopyShowsTheKindOfAVersionThatRepeatsAName) {
  const auto shim = make_fw_info({{SMI_NIC_FW_VERSION_TYPE_RUNNING, "fw.version", "1.0"},
                                  {SMI_NIC_FW_VERSION_TYPE_STORED, "fw.version", "2.0"},
                                  {SMI_NIC_FW_VERSION_TYPE_FIXED, "fw.version", "0.9"}});
  auto info = amdsmi_nic_fw_info_t{};

  copy_nic_fw_info(*shim, &info);

  EXPECT_EQ(info.num_fw, 3U);
  EXPECT_STREQ(info.fw[0].name, "fw.version");
  EXPECT_STREQ(info.fw[1].name, "fw.version (stored)");
  EXPECT_STREQ(info.fw[2].name, "fw.version (fixed)");
  EXPECT_STREQ(info.fw[1].version, "2.0");
}

TEST_F(NicUnit, FwInfoCopyKeepsAtMostTheFirstMaxEntries) {
  auto versions = std::vector<std::tuple<smi_nic_fw_version_type_t, const char*, const char*>>{};
  auto storage = std::vector<std::string>{};
  for (auto i = 0; i < AMDSMI_MAX_NIC_FW + 4; ++i) {
    storage.push_back("fw_" + std::to_string(i));
  }
  for (const auto& name : storage) {
    versions.emplace_back(SMI_NIC_FW_VERSION_TYPE_RUNNING, name.c_str(), "1");
  }
  const auto shim = make_fw_info(versions);
  auto info = amdsmi_nic_fw_info_t{};

  copy_nic_fw_info(*shim, &info);

  EXPECT_EQ(info.num_fw, static_cast<uint32_t>(AMDSMI_MAX_NIC_FW));
  EXPECT_STREQ(info.fw[AMDSMI_MAX_NIC_FW - 1].name, "fw_15");
}

TEST_F(NicUnit, FwInfoCopyOfNoVersionsIsEmpty) {
  const auto shim = make_fw_info({});
  auto info = amdsmi_nic_fw_info_t{};
  info.num_fw = 5;

  copy_nic_fw_info(*shim, &info);

  EXPECT_EQ(info.num_fw, 0U);
  EXPECT_STREQ(info.fw[0].name, "");
}

TEST_F(NicUnit, StatsCopyReportsSuccessWhenTheArrayHoldsEveryEntry) {
  const auto info = make_info(3);
  auto out = make_out(8);
  auto num_stats = uint32_t{8};

  EXPECT_EQ(copy_nic_stats(*info, &num_stats, out.data()), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(num_stats, 3U);
  EXPECT_EQ(out[2].value, 102U);
  EXPECT_STREQ(out[2].name, "stat_2");
  EXPECT_EQ(out[3].value, kUnsetValue);
}

TEST_F(NicUnit, StatsCopyReportsSuccessForAnExactFit) {
  const auto info = make_info(3);
  auto out = make_out(3);
  auto num_stats = uint32_t{3};

  EXPECT_EQ(copy_nic_stats(*info, &num_stats, out.data()), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(num_stats, 3U);
}

TEST_F(NicUnit, StatsCopyIntoAShortArrayReportsInsufficientSize) {
  const auto info = make_info(5);
  auto out = make_out(8);
  auto num_stats = uint32_t{2};

  EXPECT_EQ(copy_nic_stats(*info, &num_stats, out.data()), AMDSMI_STATUS_INSUFFICIENT_SIZE);
  EXPECT_EQ(num_stats, 5U);
  EXPECT_EQ(out[0].value, 100U);
  EXPECT_EQ(out[1].value, 101U);
  EXPECT_EQ(out[2].value, kUnsetValue);
}

TEST_F(NicUnit, StatScopeMapsTheTwoPublicValues) {
  auto nic_scope = SMI_NIC_STAT_SCOPE_EXTENDED;

  EXPECT_TRUE(was_nic_stat_scope_resolved(AMDSMI_NIC_STAT_SCOPE_DEFAULT, &nic_scope));
  EXPECT_EQ(nic_scope, SMI_NIC_STAT_SCOPE_DEFAULT);
  EXPECT_TRUE(was_nic_stat_scope_resolved(AMDSMI_NIC_STAT_SCOPE_EXTENDED, &nic_scope));
  EXPECT_EQ(nic_scope, SMI_NIC_STAT_SCOPE_EXTENDED);
}

TEST_F(NicUnit, StatScopeRejectsAValueOutsideTheEnum) {
  auto nic_scope = SMI_NIC_STAT_SCOPE_EXTENDED;

  EXPECT_FALSE(was_nic_stat_scope_resolved(kUnknownScope, &nic_scope));
  EXPECT_EQ(nic_scope, SMI_NIC_STAT_SCOPE_EXTENDED);
  EXPECT_FALSE(was_nic_stat_scope_resolved(AMDSMI_NIC_STAT_SCOPE_DEFAULT, nullptr));
}

TEST_F(NicUnit, StatsTwoCallWithoutAnArrayAsksOnlyTheCount) {
  auto num_list_calls = int{0};
  auto num_stats = uint32_t{0};

  const auto status = get_nic_stats(
      [](uint32_t* available) {
        *available = 7;
        return AMDSMI_STATUS_SUCCESS;
      },
      [&num_list_calls](smi_nic_stat_info_t*) {
        ++num_list_calls;
        return AMDSMI_STATUS_SUCCESS;
      },
      &num_stats, nullptr);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(num_stats, 7U);
  EXPECT_EQ(num_list_calls, 0);
}

// A fill call must not also run the count: each one is a full read of the device.
TEST_F(NicUnit, StatsTwoCallWithAnArraySkipsTheCount) {
  auto num_count_calls = int{0};
  auto out = make_out(8);
  auto num_stats = uint32_t{8};

  const auto status = get_nic_stats(
      [&num_count_calls](uint32_t*) {
        ++num_count_calls;
        return AMDSMI_STATUS_SUCCESS;
      },
      [](smi_nic_stat_info_t* info) {
        *info = *make_info(3);
        return AMDSMI_STATUS_SUCCESS;
      },
      &num_stats, out.data());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(num_stats, 3U);
  EXPECT_EQ(out[2].value, 102U);
  EXPECT_STREQ(out[2].name, "stat_2");
  EXPECT_EQ(num_count_calls, 0);
}

TEST_F(NicUnit, StatsTwoCallReturnsTheCountError) {
  auto num_stats = uint32_t{55};

  const auto status = get_nic_stats([](uint32_t*) { return AMDSMI_STATUS_API_FAILED; },
                                    [](smi_nic_stat_info_t*) { return AMDSMI_STATUS_SUCCESS; },
                                    &num_stats, nullptr);

  EXPECT_EQ(status, AMDSMI_STATUS_API_FAILED);
  EXPECT_EQ(num_stats, 55U);
}

TEST_F(NicUnit, StatsTwoCallReturnsTheListError) {
  auto out = make_out(4);
  auto num_stats = uint32_t{4};

  const auto status = get_nic_stats([](uint32_t*) { return AMDSMI_STATUS_SUCCESS; },
                                    [](smi_nic_stat_info_t*) { return AMDSMI_STATUS_NOT_FOUND; },
                                    &num_stats, out.data());

  EXPECT_EQ(status, AMDSMI_STATUS_NOT_FOUND);
  EXPECT_EQ(out[0].value, kUnsetValue);
  EXPECT_EQ(num_stats, 4U);
}

TEST_F(NicUnit, StatsTwoCallWithAShortArrayReportsInsufficientSize) {
  auto out = make_out(2);
  auto num_stats = uint32_t{2};

  const auto status = get_nic_stats([](uint32_t*) { return AMDSMI_STATUS_SUCCESS; },
                                    [](smi_nic_stat_info_t* info) {
                                      *info = *make_info(3);
                                      return AMDSMI_STATUS_SUCCESS;
                                    },
                                    &num_stats, out.data());

  EXPECT_EQ(status, AMDSMI_STATUS_INSUFFICIENT_SIZE);
  EXPECT_EQ(num_stats, 3U);
}

}  // namespace
