// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Unit tests for the helpers behind the NIC statistics getters: how many entries reach the caller's
 * array and what status says so, and which statistics scopes are accepted. No hardware, no root.
 */

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_nic_stats.h"

namespace {

using amd::smi::copy_nic_stats;
using amd::smi::was_nic_stat_scope_resolved;

auto g_tests_run = int{0};
auto g_tests_failed = int{0};

auto check(const std::string& name, bool was_passed) -> void {
  ++g_tests_run;
  if (!was_passed) {
    ++g_tests_failed;
  }
  std::cout << (was_passed ? "  PASS: " : "  FAIL: ") << name << "\n";
}

// The info struct holds hundreds of KB, so it lives on the heap.
auto make_info(uint32_t count) -> std::unique_ptr<smi_nic_stat_info_t> {
  auto info = std::make_unique<smi_nic_stat_info_t>();
  *info = {};
  info->count = count;
  for (auto i = uint32_t{0}; i < count; ++i) {
    std::snprintf(info->stats[i].name, sizeof(info->stats[i].name), "stat_%u", i);
    info->stats[i].value = 100 + i;
  }
  return info;
}

constexpr auto kUnsetValue = uint64_t{0xDEADBEEF};

auto make_out(uint32_t capacity) -> std::vector<amdsmi_nic_stat_t> {
  auto out = std::vector<amdsmi_nic_stat_t>(capacity);
  for (auto& stat : out) {
    stat = {};
    stat.value = kUnsetValue;
  }
  return out;
}

auto test_copy_with_room_reports_success() -> void {
  std::cout << "\nCopy: an array that holds every entry\n";
  const auto info = make_info(3);
  auto out = make_out(8);
  auto num_stats = uint32_t{8};

  const auto status = copy_nic_stats(*info, &num_stats, out.data());

  check("status is SUCCESS", (status == AMDSMI_STATUS_SUCCESS));
  check("num_stats is the entry count", (num_stats == 3));
  check("last entry is copied", (out[2].value == 102) && (std::string(out[2].name) == "stat_2"));
  check("entries past the count are untouched", (out[3].value == kUnsetValue));
}

auto test_copy_exact_fit_reports_success() -> void {
  std::cout << "\nCopy: an array exactly as large as the entry count\n";
  const auto info = make_info(3);
  auto out = make_out(3);
  auto num_stats = uint32_t{3};

  check("status is SUCCESS",
        (copy_nic_stats(*info, &num_stats, out.data()) == AMDSMI_STATUS_SUCCESS));
  check("num_stats is the entry count", (num_stats == 3));
}

auto test_copy_into_short_array_says_so() -> void {
  std::cout << "\nCopy: an array too small for the entries\n";
  const auto info = make_info(5);
  auto out = make_out(8);
  auto num_stats = uint32_t{2};

  const auto status = copy_nic_stats(*info, &num_stats, out.data());

  check("status is INSUFFICIENT_SIZE", (status == AMDSMI_STATUS_INSUFFICIENT_SIZE));
  check("num_stats is the count the caller needs", (num_stats == 5));
  check("the entries that fit are copied", (out[0].value == 100) && (out[1].value == 101));
  check("nothing is written past the caller's size", (out[2].value == kUnsetValue));
}

auto test_scope_accepts_the_two_public_values() -> void {
  std::cout << "\nScope: public values map onto the NIC library's\n";
  auto nic_scope = SMI_NIC_STAT_SCOPE_EXTENDED;

  check("DEFAULT is accepted",
        was_nic_stat_scope_resolved(AMDSMI_NIC_STAT_SCOPE_DEFAULT, &nic_scope));
  check("DEFAULT maps to DEFAULT", (nic_scope == SMI_NIC_STAT_SCOPE_DEFAULT));
  check("EXTENDED is accepted",
        was_nic_stat_scope_resolved(AMDSMI_NIC_STAT_SCOPE_EXTENDED, &nic_scope));
  check("EXTENDED maps to EXTENDED", (nic_scope == SMI_NIC_STAT_SCOPE_EXTENDED));
}

auto test_scope_rejects_values_outside_the_enum() -> void {
  std::cout << "\nScope: a value outside the enum is rejected, not read as DEFAULT\n";
  auto nic_scope = SMI_NIC_STAT_SCOPE_EXTENDED;
  constexpr auto kUnknownScope = int{2};

  check("an unknown value is rejected", !was_nic_stat_scope_resolved(kUnknownScope, &nic_scope));
  check("a rejected value leaves the output alone", (nic_scope == SMI_NIC_STAT_SCOPE_EXTENDED));
  check("a null output is rejected",
        !was_nic_stat_scope_resolved(AMDSMI_NIC_STAT_SCOPE_DEFAULT, nullptr));
}

}  // namespace

int main() {
  test_copy_with_room_reports_success();
  test_copy_exact_fit_reports_success();
  test_copy_into_short_array_says_so();
  test_scope_accepts_the_two_public_values();
  test_scope_rejects_values_outside_the_enum();

  std::cout << "\n" << g_tests_run << " tests run, " << g_tests_failed << " failed\n";
  return (g_tests_failed == 0) ? 0 : 1;
}
