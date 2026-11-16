// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Decoding of an ethtool netlink FEC reply (ETHTOOL_MSG_FEC_GET with statistics). The kernel sends
 * one ETHTOOL_A_FEC_STAT_HIST attribute per histogram bin, so a decoder that keeps a single
 * attribute per id loses every bin but one. Built without libnl-3 the decoder reports ENOTSUP and
 * these tests skip.
 */

#include <gtest/gtest.h>
#include <linux/genetlink.h>
#include <linux/netlink.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "nic_unit_fixture.h"
#include "smi_nic_testing.h"
#include "smi_nic_transport.h"

namespace {

using amd::smi::nic::transport::FecStatistics_t;
using amd::smi::nic::transport::Result;

using Bytes_t = std::vector<uint8_t>;
using Counters_t = std::map<std::string, uint64_t>;

// Kernel UAPI numbers, spelled out so the tests do not depend on the build host's headers.
constexpr uint16_t kNlaNested = 0x8000;
constexpr size_t kNlaAlign = 4;
constexpr size_t kNlaHeaderBytes = 4;
constexpr uint16_t kEthtoolFamilyId = 0x16;
constexpr uint8_t kFecGetReply = 0x1e;
constexpr uint8_t kEthtoolGenlVersion = 1;
constexpr uint16_t kFecReplyStats = 5;
constexpr uint16_t kFecStatCorrected = 2;
constexpr uint16_t kFecStatUncorrected = 3;
constexpr uint16_t kFecStatCorrectedBits = 4;
constexpr uint16_t kFecStatHist = 5;
constexpr uint16_t kFecHistBinLow = 2;
constexpr uint16_t kFecHistBinHigh = 3;
constexpr uint16_t kFecHistBinVal = 4;
constexpr uint32_t kHistogramBins = 16;
constexpr size_t kNarrowValueBytes = sizeof(uint32_t);
constexpr size_t kWideValueBytes = sizeof(uint64_t);

constexpr char kCorrectedBlocks[] = "corrected_blocks";
constexpr char kUncorrectableBlocks[] = "uncorrectable_blocks";
constexpr char kCorrectedBits[] = "corrected_bits";
constexpr char kBinNamePrefix[] = "fec_symbols_err_";

// The unmodified reply of an ionic port on a 6.16 vendor kernel that reports the histogram, taken
// with strace -e trace=recvmsg around "ethtool -I --show-fec". The netlink and generic netlink
// headers are rebuilt from the strace decode; the attributes are the captured bytes.
constexpr uint64_t kCapturedCorrectedBlocks = 35;
constexpr uint64_t kCapturedCorrectedBits = 28;
constexpr uint64_t kCapturedBin0 = 71336920;
constexpr uint64_t kCapturedBin1 = 28;
constexpr uint8_t kCapturedFecReply[] = {
    0x50, 0x02, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x15, 0x99, 0x28, 0x00,
    0x1e, 0x01, 0x00, 0x00, 0x1c, 0x00, 0x01, 0x80, 0x08, 0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x0f, 0x00, 0x02, 0x00, 0x65, 0x6e, 0x50, 0x31, 0x70, 0x33, 0x73, 0x30, 0x66, 0x33, 0x00, 0x00,
    0x28, 0x00, 0x02, 0x80, 0x04, 0x00, 0x01, 0x00, 0x08, 0x00, 0x02, 0x00, 0x79, 0x00, 0x00, 0x00,
    0x18, 0x00, 0x03, 0x80, 0x14, 0x00, 0x01, 0x80, 0x08, 0x00, 0x01, 0x00, 0x32, 0x00, 0x00, 0x00,
    0x07, 0x00, 0x02, 0x00, 0x52, 0x53, 0x00, 0x00, 0x05, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x08, 0x00, 0x04, 0x00, 0x32, 0x00, 0x00, 0x00, 0xe8, 0x01, 0x05, 0x80, 0x0c, 0x00, 0x02, 0x00,
    0x23, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x04, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x1c, 0x00, 0x05, 0x80, 0x08, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00, 0xd8, 0x83, 0x40, 0x04, 0x1c, 0x00, 0x05, 0x80,
    0x08, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x08, 0x00, 0x04, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x05, 0x80, 0x08, 0x00, 0x02, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00, 0x02, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x05, 0x80, 0x08, 0x00, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x08, 0x00, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x1c, 0x00, 0x05, 0x80, 0x08, 0x00, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x05, 0x80,
    0x08, 0x00, 0x02, 0x00, 0x05, 0x00, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00, 0x05, 0x00, 0x00, 0x00,
    0x08, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x05, 0x80, 0x08, 0x00, 0x02, 0x00,
    0x06, 0x00, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00, 0x06, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x05, 0x80, 0x08, 0x00, 0x02, 0x00, 0x07, 0x00, 0x00, 0x00,
    0x08, 0x00, 0x03, 0x00, 0x07, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x1c, 0x00, 0x05, 0x80, 0x08, 0x00, 0x02, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00,
    0x08, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x05, 0x80,
    0x08, 0x00, 0x02, 0x00, 0x09, 0x00, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00, 0x09, 0x00, 0x00, 0x00,
    0x08, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x05, 0x80, 0x08, 0x00, 0x02, 0x00,
    0x0a, 0x00, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x05, 0x80, 0x08, 0x00, 0x02, 0x00, 0x0b, 0x00, 0x00, 0x00,
    0x08, 0x00, 0x03, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x1c, 0x00, 0x05, 0x80, 0x08, 0x00, 0x02, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x05, 0x80,
    0x08, 0x00, 0x02, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00, 0x0d, 0x00, 0x00, 0x00,
    0x08, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x05, 0x80, 0x08, 0x00, 0x02, 0x00,
    0x0e, 0x00, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x05, 0x80, 0x08, 0x00, 0x02, 0x00, 0x0f, 0x00, 0x00, 0x00,
    0x08, 0x00, 0x03, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
};

auto append_bytes(Bytes_t& out, const void* data, size_t size) -> void {
  const auto* first = static_cast<const uint8_t*>(data);
  out.insert(out.end(), first, (first + size));
}

auto make_attr(uint16_t type, const Bytes_t& payload) -> Bytes_t {
  auto attr = Bytes_t{};
  const auto length = static_cast<uint16_t>(kNlaHeaderBytes + payload.size());
  append_bytes(attr, &length, sizeof(length));
  append_bytes(attr, &type, sizeof(type));
  attr.insert(attr.end(), payload.begin(), payload.end());
  attr.resize((((attr.size() + kNlaAlign) - 1) / kNlaAlign) * kNlaAlign, 0);
  return attr;
}

auto make_nested(uint16_t type, const std::vector<Bytes_t>& children) -> Bytes_t {
  auto payload = Bytes_t{};
  for (const auto& child : children) {
    payload.insert(payload.end(), child.begin(), child.end());
  }
  return make_attr(static_cast<uint16_t>(type | kNlaNested), payload);
}

auto make_u32_attr(uint16_t type, uint32_t value) -> Bytes_t {
  auto payload = Bytes_t{};
  append_bytes(payload, &value, sizeof(value));
  return make_attr(type, payload);
}

// The kernel sends each total as an array of u64 whose first element is the port total.
auto make_total_attr(uint16_t type, uint64_t total) -> Bytes_t {
  auto payload = Bytes_t{};
  append_bytes(payload, &total, sizeof(total));
  return make_attr(type, payload);
}

// A histogram value is a kernel "uint": four bytes when it fits, eight otherwise.
auto make_hist_bin(uint32_t low, uint32_t high, uint64_t value, size_t value_bytes) -> Bytes_t {
  auto value_payload = Bytes_t{};
  if (value_bytes == kNarrowValueBytes) {
    const auto narrow = static_cast<uint32_t>(value);
    append_bytes(value_payload, &narrow, sizeof(narrow));
  } else {
    append_bytes(value_payload, &value, sizeof(value));
  }
  return make_nested(kFecStatHist,
                     {make_u32_attr(kFecHistBinLow, low), make_u32_attr(kFecHistBinHigh, high),
                      make_attr(kFecHistBinVal, value_payload)});
}

auto make_hist_with(const std::vector<Bytes_t>& members) -> Bytes_t {
  return make_nested(kFecStatHist, members);
}

auto make_counters(uint64_t corrected, uint64_t uncorrected, uint64_t bits)
    -> std::vector<Bytes_t> {
  return {make_total_attr(kFecStatCorrected, corrected),
          make_total_attr(kFecStatUncorrected, uncorrected),
          make_total_attr(kFecStatCorrectedBits, bits)};
}

auto make_fec_reply(const Bytes_t& body) -> Bytes_t {
  auto netlink = nlmsghdr{};
  netlink.nlmsg_len = static_cast<uint32_t>(NLMSG_HDRLEN + GENL_HDRLEN + body.size());
  netlink.nlmsg_type = kEthtoolFamilyId;
  auto generic = genlmsghdr{};
  generic.cmd = kFecGetReply;
  generic.version = kEthtoolGenlVersion;

  auto reply = Bytes_t{};
  append_bytes(reply, &netlink, sizeof(netlink));
  append_bytes(reply, &generic, sizeof(generic));
  reply.insert(reply.end(), body.begin(), body.end());
  return reply;
}

auto make_stats_reply(const std::vector<Bytes_t>& stats_children) -> Bytes_t {
  return make_fec_reply(make_nested(kFecReplyStats, stats_children));
}

auto decode(const Bytes_t& reply) -> Result<FecStatistics_t> {
  return smi_nic_fec_stats_from_reply_for_testing(reply.data(), reply.size());
}

auto is_netlink_unavailable(const Result<FecStatistics_t>& result) -> bool {
  return ((!result.success) && (result.error_code == ENOTSUP));
}

// A repeated name or a names/values length mismatch is a decoder bug a map would hide.
auto to_counters(const FecStatistics_t& stats) -> Counters_t {
  auto counters = Counters_t{};
  if (stats.names.size() != stats.values.size()) {
    ADD_FAILURE() << "names and values differ in length";
    return counters;
  }
  for (size_t i = 0; i < stats.names.size(); ++i) {
    if (!counters.emplace(stats.names[i], stats.values[i]).second) {
      ADD_FAILURE() << "name reported twice: " << stats.names[i];
    }
  }
  return counters;
}

auto make_expected_counters(uint64_t corrected, uint64_t uncorrected, uint64_t bits) -> Counters_t {
  return Counters_t{
      {kCorrectedBlocks, corrected}, {kUncorrectableBlocks, uncorrected}, {kCorrectedBits, bits}};
}

auto bin_name(uint32_t bin) -> std::string {
  return (std::string{kBinNamePrefix} + std::to_string(bin));
}

TEST_F(NicUnit, FecReplyCapturedFromHardwareYieldsCountersAndEveryHistogramBin) {
  const auto reply = Bytes_t(std::begin(kCapturedFecReply), std::end(kCapturedFecReply));

  const auto result = decode(reply);
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  ASSERT_TRUE(result.success);
  auto expected = Counters_t{{kCorrectedBlocks, kCapturedCorrectedBlocks},
                             {kUncorrectableBlocks, 0},
                             {kCorrectedBits, kCapturedCorrectedBits}};
  for (uint32_t bin = 0; bin < kHistogramBins; ++bin) {
    expected[bin_name(bin)] = 0;
  }
  expected[bin_name(0)] = kCapturedBin0;
  expected[bin_name(1)] = kCapturedBin1;
  EXPECT_EQ(to_counters(result.value), expected);
  EXPECT_EQ(result.value.names.size(), expected.size());
}

TEST_F(NicUnit, FecReplyWithoutHistogramYieldsOnlyTheThreeCounters) {
  const auto reply = make_stats_reply(make_counters(7, 3, 5));

  const auto result = decode(reply);
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  ASSERT_TRUE(result.success);
  EXPECT_EQ(to_counters(result.value), make_expected_counters(7, 3, 5));
}

// Each bin carries a distinct value and the bins arrive in reverse, so a decoder that names a bin
// by its position in the reply swaps them.
TEST_F(NicUnit, FecHistogramBinsAreNamedByBinIndexNotByPosition) {
  auto children = make_counters(0, 0, 0);
  children.push_back(make_hist_bin(3, 3, 300, kNarrowValueBytes));
  children.push_back(make_hist_bin(2, 2, 200, kNarrowValueBytes));
  children.push_back(make_hist_bin(1, 1, 100, kNarrowValueBytes));
  children.push_back(make_hist_bin(0, 0, 9, kNarrowValueBytes));

  const auto result = decode(make_stats_reply(children));
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  ASSERT_TRUE(result.success);
  auto expected = make_expected_counters(0, 0, 0);
  expected[bin_name(0)] = 9;
  expected[bin_name(1)] = 100;
  expected[bin_name(2)] = 200;
  expected[bin_name(3)] = 300;
  EXPECT_EQ(to_counters(result.value), expected);
}

TEST_F(NicUnit, FecHistogramBinValueMayBeEightBytesWide) {
  constexpr uint64_t kBeyondThirtyTwoBits = 0x100000002ULL;
  auto children = make_counters(0, 0, 0);
  children.push_back(make_hist_bin(0, 0, kBeyondThirtyTwoBits, kWideValueBytes));

  const auto result = decode(make_stats_reply(children));
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  ASSERT_TRUE(result.success);
  auto expected = make_expected_counters(0, 0, 0);
  expected[bin_name(0)] = kBeyondThirtyTwoBits;
  EXPECT_EQ(to_counters(result.value), expected);
}

// A bin that spans several symbol counts has no single fec_symbols_err_<n> name.
TEST_F(NicUnit, FecHistogramBinSpanningSeveralSymbolCountsIsSkipped) {
  auto children = make_counters(0, 0, 0);
  children.push_back(make_hist_bin(0, 3, 40, kNarrowValueBytes));
  children.push_back(make_hist_bin(4, 4, 50, kNarrowValueBytes));

  const auto result = decode(make_stats_reply(children));
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  ASSERT_TRUE(result.success);
  auto expected = make_expected_counters(0, 0, 0);
  expected[bin_name(4)] = 50;
  EXPECT_EQ(to_counters(result.value), expected);
}

TEST_F(NicUnit, FecReplyWithoutStatisticsAttributeReportsNoData) {
  const auto result = decode(make_fec_reply({}));
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.error_code, ENODATA);
}

TEST_F(NicUnit, FecReplyWithEmptyStatisticsReportsNoCounters) {
  const auto result = decode(make_stats_reply({}));
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  ASSERT_TRUE(result.success);
  EXPECT_TRUE(result.value.names.empty());
}

// The decoder names whatever bin index it is given; the vendor table decides which names surface.
TEST_F(NicUnit, FecHistogramBinBeyondTheSixteenKnownBinsIsStillNamedByIndex) {
  constexpr uint32_t kBeyondKnownBins = 20;
  auto children = make_counters(0, 0, 0);
  children.push_back(make_hist_bin(kBeyondKnownBins, kBeyondKnownBins, 6, kNarrowValueBytes));

  const auto result = decode(make_stats_reply(children));
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  ASSERT_TRUE(result.success);
  auto expected = make_expected_counters(0, 0, 0);
  expected[bin_name(kBeyondKnownBins)] = 6;
  EXPECT_EQ(to_counters(result.value), expected);
}

TEST_F(NicUnit, FecHistogramBinWithAnInvertedRangeIsSkipped) {
  auto children = make_counters(0, 0, 0);
  children.push_back(make_hist_bin(3, 0, 40, kNarrowValueBytes));
  children.push_back(make_hist_bin(4, 4, 50, kNarrowValueBytes));

  const auto result = decode(make_stats_reply(children));
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  ASSERT_TRUE(result.success);
  auto expected = make_expected_counters(0, 0, 0);
  expected[bin_name(4)] = 50;
  EXPECT_EQ(to_counters(result.value), expected);
}

// A bin lacking one of its three members is skipped, and the valid bin after it still decodes.
TEST_F(NicUnit, FecHistogramBinWithoutLowIsSkipped) {
  auto children = make_counters(0, 0, 0);
  children.push_back(
      make_hist_with({make_u32_attr(kFecHistBinHigh, 0), make_u32_attr(kFecHistBinVal, 11)}));
  children.push_back(make_hist_bin(1, 1, 12, kNarrowValueBytes));

  const auto result = decode(make_stats_reply(children));
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  ASSERT_TRUE(result.success);
  auto expected = make_expected_counters(0, 0, 0);
  expected[bin_name(1)] = 12;
  EXPECT_EQ(to_counters(result.value), expected);
}

TEST_F(NicUnit, FecHistogramBinWithoutHighIsSkipped) {
  auto children = make_counters(0, 0, 0);
  children.push_back(
      make_hist_with({make_u32_attr(kFecHistBinLow, 0), make_u32_attr(kFecHistBinVal, 11)}));
  children.push_back(make_hist_bin(1, 1, 12, kNarrowValueBytes));

  const auto result = decode(make_stats_reply(children));
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  ASSERT_TRUE(result.success);
  auto expected = make_expected_counters(0, 0, 0);
  expected[bin_name(1)] = 12;
  EXPECT_EQ(to_counters(result.value), expected);
}

TEST_F(NicUnit, FecHistogramBinWithoutValueIsSkipped) {
  auto children = make_counters(0, 0, 0);
  children.push_back(
      make_hist_with({make_u32_attr(kFecHistBinLow, 0), make_u32_attr(kFecHistBinHigh, 0)}));
  children.push_back(make_hist_bin(1, 1, 12, kNarrowValueBytes));

  const auto result = decode(make_stats_reply(children));
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  ASSERT_TRUE(result.success);
  auto expected = make_expected_counters(0, 0, 0);
  expected[bin_name(1)] = 12;
  EXPECT_EQ(to_counters(result.value), expected);
}

// The kernel integer is four or eight bytes; any other width carries no value to report.
TEST_F(NicUnit, FecHistogramBinValueOfUnsupportedWidthIsSkipped) {
  auto children = make_counters(0, 0, 0);
  uint32_t bin = 1;
  for (const auto width : {size_t{0}, size_t{2}, size_t{6}, size_t{12}}) {
    children.push_back(
        make_hist_with({make_u32_attr(kFecHistBinLow, bin), make_u32_attr(kFecHistBinHigh, bin),
                        make_attr(kFecHistBinVal, Bytes_t(width, 0x7f))}));
    ++bin;
  }
  children.push_back(make_hist_bin(0, 0, 7, kNarrowValueBytes));

  const auto result = decode(make_stats_reply(children));
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  ASSERT_TRUE(result.success);
  auto expected = make_expected_counters(0, 0, 0);
  expected[bin_name(0)] = 7;
  EXPECT_EQ(to_counters(result.value), expected);
}

// A two-byte bin index is shorter than the four-byte read the old decoder assumed.
TEST_F(NicUnit, FecHistogramBinWithAShortIndexIsSkipped) {
  auto children = make_counters(0, 0, 0);
  children.push_back(
      make_hist_with({make_attr(kFecHistBinLow, Bytes_t(2, 0)), make_u32_attr(kFecHistBinHigh, 0),
                      make_u32_attr(kFecHistBinVal, 11)}));
  children.push_back(make_hist_bin(1, 1, 12, kNarrowValueBytes));

  const auto result = decode(make_stats_reply(children));
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  ASSERT_TRUE(result.success);
  auto expected = make_expected_counters(0, 0, 0);
  expected[bin_name(1)] = 12;
  EXPECT_EQ(to_counters(result.value), expected);
}

// A future statistics attribute whose payload looks like a bin must not be read as one.
TEST_F(NicUnit, FecStatisticsSiblingAttributeIsNotReadAsAHistogramBin) {
  constexpr uint16_t kUnrelatedStatType = 6;
  auto children = make_counters(0, 0, 0);
  children.push_back(make_nested(
      kUnrelatedStatType, {make_u32_attr(kFecHistBinLow, 2), make_u32_attr(kFecHistBinHigh, 2),
                           make_u32_attr(kFecHistBinVal, 99)}));

  const auto result = decode(make_stats_reply(children));
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  ASSERT_TRUE(result.success);
  EXPECT_EQ(to_counters(result.value), make_expected_counters(0, 0, 0));
}

TEST_F(NicUnit, FecReplyShorterThanTheNetlinkHeaderIsRejected) {
  // Shorter than the length field itself, so reading it unchecked is an out-of-bounds read.
  const auto reply = Bytes_t(sizeof(uint16_t), 0);

  const auto result = decode(reply);
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.error_code, EINVAL);
}

TEST_F(NicUnit, FecReplyClaimingMoreBytesThanItHasIsRejected) {
  auto reply = make_stats_reply(make_counters(1, 2, 3));
  const auto claimed = static_cast<uint32_t>(reply.size() + kNlaAlign);
  std::memcpy(reply.data(), &claimed, sizeof(claimed));

  const auto result = decode(reply);
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.error_code, EINVAL);
}

TEST_F(NicUnit, FecReplyShorterThanItsGenericNetlinkHeaderIsRejected) {
  auto reply = make_stats_reply(make_counters(1, 2, 3));
  const auto only_the_netlink_header = static_cast<uint32_t>(NLMSG_HDRLEN);
  std::memcpy(reply.data(), &only_the_netlink_header, sizeof(only_the_netlink_header));

  const auto result = decode(reply);
  if (is_netlink_unavailable(result)) {
    GTEST_SKIP() << "built without libnl-3";
  }

  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.error_code, EINVAL);
}

}  // namespace
