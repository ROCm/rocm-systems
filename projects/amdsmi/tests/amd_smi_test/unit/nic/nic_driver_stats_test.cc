// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Reading `ethtool -S` over ioctl takes three requests: the count, the names, the values. A driver
 * whose count changes between them must not overflow our buffers, and must not report names and
 * values that were never read. A fake driver models both kinds of kernel: one that rejects an
 * input count that no longer matches (2025 and later), and an older one that answers with its
 * current count whatever the caller allocated.
 */

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

#include "nic_unit_fixture.h"
#include "smi_nic_driver_stats.h"

namespace {

namespace ioctl_internal = amd::smi::nic::transport::ioctl_internal;
using amd::smi::nic::transport::Result;
using amd::smi::nic::transport::VendorStatistics;

constexpr auto kNameTemplate = "stat_";
constexpr auto kValueBase = uint64_t{1000};
constexpr auto kMaxAttempts = int{3};

class FakeDriver_t {
 public:
  FakeDriver_t(uint32_t count, bool is_input_checked)
      : m_count(count), m_is_input_checked(is_input_checked) {}

  // The count grows by one right after the next count query, as a reconfiguration would.
  auto grow_after_next_query() -> void { m_is_growing_once = true; }
  auto grow_after_every_query() -> void { m_is_growing_always = true; }
  auto fail_strings_with(int error) -> void { m_strings_error = error; }
  auto name_stats_with(std::string name_template) -> void {
    m_name_template = std::move(name_template);
  }

  auto ioctls() -> ioctl_internal::DriverStatsIoctls_t {
    auto ioctls = ioctl_internal::DriverStatsIoctls_t{};
    ioctls.get_count = [this]() -> Result<uint32_t> { return query_count(); };
    ioctls.get_strings = [this](ethtool_gstrings* strings, size_t capacity) {
      return answer_strings(strings, capacity);
    };
    ioctls.get_stats = [this](ethtool_stats* stats, size_t capacity) {
      return answer_stats(stats, capacity);
    };
    return ioctls;
  }

  auto num_count_queries() const -> int { return m_num_count_queries; }
  // True if the driver would have written past the end of a caller's buffer.
  auto has_overflowed() const -> bool { return m_has_overflowed; }

 private:
  auto query_count() -> Result<uint32_t> {
    ++m_num_count_queries;
    const auto reported = m_count;
    if (m_is_growing_always || m_is_growing_once) {
      ++m_count;
      m_is_growing_once = false;
    }
    return {true, reported, 0};
  }

  auto answer_strings(ethtool_gstrings* strings, size_t capacity) -> int {
    if (m_strings_error != 0) {
      errno = m_strings_error;
      return -1;
    }
    if (m_is_input_checked && (strings->len != 0) && (strings->len != m_count)) {
      strings->len = 0;
      return 0;
    }
    strings->len = m_count;
    m_has_overflowed = m_has_overflowed || (m_count > capacity);
    for (auto i = uint32_t{0}; (i < m_count) && (i < capacity); ++i) {
      const auto name = m_name_template + std::to_string(i);
      std::memcpy(&strings->data[i * ETH_GSTRING_LEN], name.c_str(), name.size() + 1);
    }
    return 0;
  }

  auto answer_stats(ethtool_stats* stats, size_t capacity) -> int {
    if (m_is_input_checked && (stats->n_stats != 0) && (stats->n_stats != m_count)) {
      stats->n_stats = 0;
      return 0;
    }
    stats->n_stats = m_count;
    m_has_overflowed = m_has_overflowed || (m_count > capacity);
    for (auto i = uint32_t{0}; (i < m_count) && (i < capacity); ++i) {
      stats->data[i] = kValueBase + i;
    }
    return 0;
  }

  uint32_t m_count;
  bool m_is_input_checked;
  bool m_is_growing_once = false;
  bool m_is_growing_always = false;
  int m_strings_error = 0;
  std::string m_name_template = kNameTemplate;
  int m_num_count_queries = 0;
  bool m_has_overflowed = false;
};

auto expect_entries(const VendorStatistics& stats, uint32_t count) -> void {
  ASSERT_EQ(stats.names.size(), size_t{count});
  ASSERT_EQ(stats.values.size(), size_t{count});
  for (auto i = uint32_t{0}; i < count; ++i) {
    EXPECT_EQ(stats.names[i], std::string{kNameTemplate} + std::to_string(i));
    EXPECT_EQ(stats.values[i], kValueBase + i);
  }
}

TEST_F(NicUnit, DriverStatsSteadyCountReadsEveryEntry) {
  auto driver = FakeDriver_t{3, /*is_input_checked=*/true};

  const auto result = ioctl_internal::read_driver_statistics(driver.ioctls());

  ASSERT_TRUE(result.success);
  expect_entries(result.value, 3);
  EXPECT_EQ(driver.num_count_queries(), 1);
}

TEST_F(NicUnit, DriverStatsZeroCountIsEmptySuccess) {
  auto driver = FakeDriver_t{0, /*is_input_checked=*/true};

  const auto result = ioctl_internal::read_driver_statistics(driver.ioctls());

  ASSERT_TRUE(result.success);
  EXPECT_TRUE(result.value.names.empty());
}

TEST_F(NicUnit, DriverStatsCountChangeIsRetriedOnAKernelThatChecksTheInput) {
  auto driver = FakeDriver_t{3, /*is_input_checked=*/true};
  driver.grow_after_next_query();

  const auto result = ioctl_internal::read_driver_statistics(driver.ioctls());

  ASSERT_TRUE(result.success);
  expect_entries(result.value, 4);
  EXPECT_EQ(driver.num_count_queries(), 2);
}

TEST_F(NicUnit, DriverStatsGrowthStaysInsideTheBufferOnAnOlderKernel) {
  auto driver = FakeDriver_t{3, /*is_input_checked=*/false};
  driver.grow_after_next_query();

  const auto result = ioctl_internal::read_driver_statistics(driver.ioctls());

  EXPECT_FALSE(driver.has_overflowed());
  ASSERT_TRUE(result.success);
  expect_entries(result.value, 4);
}

TEST_F(NicUnit, DriverStatsCountThatNeverSettlesFails) {
  auto driver = FakeDriver_t{3, /*is_input_checked=*/true};
  driver.grow_after_every_query();

  const auto result = ioctl_internal::read_driver_statistics(driver.ioctls());

  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.error_code, EAGAIN);
  EXPECT_EQ(driver.num_count_queries(), kMaxAttempts);
}

TEST_F(NicUnit, DriverStatsRequestFailureKeepsItsErrno) {
  auto driver = FakeDriver_t{3, /*is_input_checked=*/true};
  driver.fail_strings_with(EPERM);

  const auto result = ioctl_internal::read_driver_statistics(driver.ioctls());

  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.error_code, EPERM);
}

// A driver names its counters; a control byte in one must not reach the output.
TEST_F(NicUnit, DriverStatsNamesLoseControlBytes) {
  auto driver = FakeDriver_t{2, /*is_input_checked=*/true};
  driver.name_stats_with("rx\x1b[31m_");

  const auto result = ioctl_internal::read_driver_statistics(driver.ioctls());

  ASSERT_TRUE(result.success);
  ASSERT_EQ(result.value.names.size(), size_t{2});
  EXPECT_EQ(result.value.names[0], "rx?[31m_0");
  EXPECT_EQ(result.value.names[1], "rx?[31m_1");
}

}  // namespace
