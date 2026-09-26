// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/nic/perfetto_policy.hpp"
#include "library/pmc/collectors/nic/types.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>

namespace rocprofsys::pmc::collectors::nic::testing
{
namespace
{
using counter_track = perfetto_counter_track<metrics>;

enabled_metrics
make_enabled(std::uint32_t value)
{
    return enabled_metrics{ .value = value };
}

// perfetto_counter_track<metrics> keeps process-wide static storage that has no
// reset hook, so track indices for a given device index only ever grow. Each
// test that registers tracks therefore uses a device index of its own.
constexpr size_t SETUP_ALL_DEVICE          = 100;
constexpr size_t SETUP_SUBSET_DEVICE       = 101;
constexpr size_t SETUP_NONE_DEVICE         = 102;
constexpr size_t SETUP_DISTINCT_IDX_DEVICE = 103;
constexpr size_t SETUP_ACCUMULATE_DEVICE   = 104;
constexpr size_t SETUP_RESOLVE_DEVICE      = 105;

class NicPerfettoPolicyTest : public ::testing::Test
{
protected:
    // perfetto_policy::tracks is `static inline`, so it persists across tests.
    void SetUp() override { perfetto_policy::tracks.clear(); }
    void TearDown() override { perfetto_policy::tracks.clear(); }
};

// --------------------------------------------------------------------------
// Default track table
// --------------------------------------------------------------------------

TEST_F(NicPerfettoPolicyTest, DefaultTracks_CoverEveryMetricBit)
{
    const auto& tracks = make_default_nic_tracks();

    EXPECT_EQ(tracks.size(), NIC_METRICS_COUNT);

    std::uint32_t combined = 0;
    for(const auto& [bit_value, description] : tracks)
        combined |= bit_value;

    EXPECT_EQ(combined, ALL_NIC_METRICS);
}

TEST_F(NicPerfettoPolicyTest, DefaultTracks_KeysAreSingleBits)
{
    for(const auto& [bit_value, description] : make_default_nic_tracks())
    {
        SCOPED_TRACE(description.track_name);
        EXPECT_NE(bit_value, 0u);
        // A power of two: exactly one metric bit per row.
        EXPECT_EQ(bit_value & (bit_value - 1), 0u);
    }
}

TEST_F(NicPerfettoPolicyTest, DefaultTracks_HaveExpectedLabelsAndUnits)
{
    const std::map<std::uint32_t, std::pair<std::string, std::string>> expected = {
        { RX_RDMA_UCAST_BYTES_VALUE, { "RX RDMA BYTES", "bytes" } },
        { TX_RDMA_UCAST_BYTES_VALUE, { "TX RDMA BYTES", "bytes" } },
        { RX_RDMA_UCAST_PKTS_VALUE, { "RX RDMA PACKETS", "packets" } },
        { TX_RDMA_UCAST_PKTS_VALUE, { "TX RDMA PACKETS", "packets" } },
        { RX_RDMA_CNP_PKTS_VALUE, { "RX CNP PACKETS", "packets" } },
        { TX_RDMA_CNP_PKTS_VALUE, { "TX CNP PACKETS", "packets" } },
        { TX_RDMA_ACK_TIMEOUT_VALUE, { "TX ACK TIMEOUT", "timeouts" } },
        { RESP_TX_PKT_SEQ_ERR_VALUE, { "RESP TX PKT SEQ ERR", "errors" } },
        { REQ_RX_PKT_SEQ_ERR_VALUE, { "REQ RX PKT SEQ ERR", "errors" } },
        { REQ_RX_IMPL_NAK_SEQ_ERR_VALUE, { "REQ RX IMPL NAK SEQ ERR", "errors" } },
    };

    const auto& tracks = make_default_nic_tracks();
    ASSERT_EQ(tracks.size(), expected.size());

    for(const auto& [bit_value, description] : tracks)
    {
        SCOPED_TRACE(description.track_name);
        auto it = expected.find(bit_value);
        ASSERT_NE(it, expected.end());
        EXPECT_EQ(std::string{ description.track_name }, it->second.first);
        EXPECT_EQ(std::string{ description.units }, it->second.second);
    }
}

TEST_F(NicPerfettoPolicyTest, DefaultTracks_LabelsAreUnique)
{
    std::set<std::string> names{};
    for(const auto& [bit_value, description] : make_default_nic_tracks())
        EXPECT_TRUE(names.emplace(description.track_name).second)
            << "duplicate track name: " << description.track_name;
}

TEST_F(NicPerfettoPolicyTest, DefaultTracks_TableIsBuiltOnce)
{
    // setup_counter_tracks() is called once per device; the table must not be
    // rebuilt on each call.
    EXPECT_EQ(&make_default_nic_tracks(), &make_default_nic_tracks());
}

// --------------------------------------------------------------------------
// Bit constants vs. the enabled_metrics bitfield layout
// --------------------------------------------------------------------------

TEST_F(NicPerfettoPolicyTest, MetricBitConstants_MatchBitfieldLayout)
{
    // Guards the invariant called out in perfetto_policy.hpp: the *_VALUE
    // constants and the enabled_metrics bitfield are parallel lists that must
    // stay in sync. Setting a field must produce exactly its constant.
    struct bit_case
    {
        const char* name;
        void (*set)(enabled_metrics&);
        std::uint32_t expected;
    };

    const bit_case cases[] = {
        { .name = "rx_rdma_ucast_bytes",
          .set  = [](enabled_metrics& m) { m.bits.rx_rdma_ucast_bytes = 1; },
          .expected        = RX_RDMA_UCAST_BYTES_VALUE },
        { .name = "tx_rdma_ucast_bytes",
          .set  = [](enabled_metrics& m) { m.bits.tx_rdma_ucast_bytes = 1; },
          .expected        = TX_RDMA_UCAST_BYTES_VALUE },
        { .name = "rx_rdma_ucast_pkts",
          .set  = [](enabled_metrics& m) { m.bits.rx_rdma_ucast_pkts = 1; },
          .expected        = RX_RDMA_UCAST_PKTS_VALUE },
        { .name = "tx_rdma_ucast_pkts",
          .set  = [](enabled_metrics& m) { m.bits.tx_rdma_ucast_pkts = 1; },
          .expected        = TX_RDMA_UCAST_PKTS_VALUE },
        { .name = "rx_rdma_cnp_pkts",
          .set  = [](enabled_metrics& m) { m.bits.rx_rdma_cnp_pkts = 1; },
          .expected        = RX_RDMA_CNP_PKTS_VALUE },
        { .name = "tx_rdma_cnp_pkts",
          .set  = [](enabled_metrics& m) { m.bits.tx_rdma_cnp_pkts = 1; },
          .expected        = TX_RDMA_CNP_PKTS_VALUE },
        { .name = "tx_rdma_ack_timeout",
          .set  = [](enabled_metrics& m) { m.bits.tx_rdma_ack_timeout = 1; },
          .expected        = TX_RDMA_ACK_TIMEOUT_VALUE },
        { .name = "resp_tx_pkt_seq_err",
          .set  = [](enabled_metrics& m) { m.bits.resp_tx_pkt_seq_err = 1; },
          .expected        = RESP_TX_PKT_SEQ_ERR_VALUE },
        { .name = "req_rx_pkt_seq_err",
          .set  = [](enabled_metrics& m) { m.bits.req_rx_pkt_seq_err = 1; },
          .expected        = REQ_RX_PKT_SEQ_ERR_VALUE },
        { .name = "req_rx_impl_nak_seq_err",
          .set  = [](enabled_metrics& m) { m.bits.req_rx_impl_nak_seq_err = 1; },
          .expected        = REQ_RX_IMPL_NAK_SEQ_ERR_VALUE },
    };

    for(const auto& tc : cases)
    {
        SCOPED_TRACE(tc.name);
        enabled_metrics em{};
        em.value = 0;
        tc.set(em);
        EXPECT_EQ(em.value, tc.expected);
    }
}

// --------------------------------------------------------------------------
// resolve_nic_track
// --------------------------------------------------------------------------

TEST_F(NicPerfettoPolicyTest, ResolveTrack_ReturnsIndex_WhenEnabledAndRegistered)
{
    const std::map<std::uint32_t, size_t> device_tracks = {
        { RX_RDMA_UCAST_BYTES_VALUE, 3 },
        { TX_RDMA_UCAST_BYTES_VALUE, 7 },
    };

    const auto enabled =
        make_enabled(RX_RDMA_UCAST_BYTES_VALUE | TX_RDMA_UCAST_BYTES_VALUE);

    EXPECT_EQ(resolve_nic_track(enabled, RX_RDMA_UCAST_BYTES_VALUE, device_tracks), 3);
    EXPECT_EQ(resolve_nic_track(enabled, TX_RDMA_UCAST_BYTES_VALUE, device_tracks), 7);
}

TEST_F(NicPerfettoPolicyTest, ResolveTrack_ReturnsZeroIndex_NotSentinel)
{
    // Index 0 is a valid track; it must not be confused with "no track".
    const std::map<std::uint32_t, size_t> device_tracks = {
        { RX_RDMA_UCAST_BYTES_VALUE, 0 },
    };

    EXPECT_EQ(resolve_nic_track(make_enabled(RX_RDMA_UCAST_BYTES_VALUE),
                                RX_RDMA_UCAST_BYTES_VALUE, device_tracks),
              0);
}

TEST_F(NicPerfettoPolicyTest, ResolveTrack_ReturnsNegative_WhenMetricDisabled)
{
    // Track exists, but the metric is not enabled.
    const std::map<std::uint32_t, size_t> device_tracks = {
        { RX_RDMA_UCAST_BYTES_VALUE, 3 },
    };

    EXPECT_LT(resolve_nic_track(make_enabled(TX_RDMA_UCAST_BYTES_VALUE),
                                RX_RDMA_UCAST_BYTES_VALUE, device_tracks),
              0);
    EXPECT_LT(
        resolve_nic_track(make_enabled(0), RX_RDMA_UCAST_BYTES_VALUE, device_tracks), 0);
}

TEST_F(NicPerfettoPolicyTest, ResolveTrack_ReturnsNegative_WhenTrackMissing)
{
    // Metric is enabled, but no track was registered for this device.
    const std::map<std::uint32_t, size_t> device_tracks = {
        { TX_RDMA_UCAST_BYTES_VALUE, 1 },
    };

    EXPECT_LT(resolve_nic_track(make_enabled(ALL_NIC_METRICS), RX_RDMA_UCAST_BYTES_VALUE,
                                device_tracks),
              0);
    EXPECT_LT(
        resolve_nic_track(make_enabled(ALL_NIC_METRICS), RX_RDMA_UCAST_BYTES_VALUE, {}),
        0);
}

// --------------------------------------------------------------------------
// setup_counter_tracks
// --------------------------------------------------------------------------

TEST_F(NicPerfettoPolicyTest, SetupTracks_RegistersEveryEnabledMetric)
{
    perfetto_policy::setup_counter_tracks(SETUP_ALL_DEVICE, "enp226s0",
                                          make_enabled(ALL_NIC_METRICS));

    const auto& device_tracks = perfetto_policy::tracks.at(SETUP_ALL_DEVICE);
    EXPECT_EQ(device_tracks.size(), NIC_METRICS_COUNT);

    for(const auto& [bit_value, description] : make_default_nic_tracks())
    {
        SCOPED_TRACE(description.track_name);
        EXPECT_EQ(device_tracks.count(bit_value), 1u);
    }
}

TEST_F(NicPerfettoPolicyTest, SetupTracks_RegistersOnlyEnabledSubset)
{
    const auto enabled = make_enabled(RX_RDMA_UCAST_BYTES_VALUE | RX_RDMA_CNP_PKTS_VALUE |
                                      REQ_RX_IMPL_NAK_SEQ_ERR_VALUE);

    perfetto_policy::setup_counter_tracks(SETUP_SUBSET_DEVICE, "enp226s0", enabled);

    const auto& device_tracks = perfetto_policy::tracks.at(SETUP_SUBSET_DEVICE);
    ASSERT_EQ(device_tracks.size(), 3u);
    EXPECT_EQ(device_tracks.count(RX_RDMA_UCAST_BYTES_VALUE), 1u);
    EXPECT_EQ(device_tracks.count(RX_RDMA_CNP_PKTS_VALUE), 1u);
    EXPECT_EQ(device_tracks.count(REQ_RX_IMPL_NAK_SEQ_ERR_VALUE), 1u);

    // Disabled metrics must not get a track.
    EXPECT_EQ(device_tracks.count(TX_RDMA_UCAST_BYTES_VALUE), 0u);
    EXPECT_EQ(device_tracks.count(TX_RDMA_ACK_TIMEOUT_VALUE), 0u);
}

TEST_F(NicPerfettoPolicyTest, SetupTracks_RegistersNothing_WhenNoMetricsEnabled)
{
    perfetto_policy::setup_counter_tracks(SETUP_NONE_DEVICE, "enp226s0", make_enabled(0));

    // The device entry is created, but holds no tracks.
    ASSERT_EQ(perfetto_policy::tracks.count(SETUP_NONE_DEVICE), 1u);
    EXPECT_TRUE(perfetto_policy::tracks.at(SETUP_NONE_DEVICE).empty());
}

TEST_F(NicPerfettoPolicyTest, SetupTracks_AssignsDistinctIndices)
{
    const auto before = counter_track::size(SETUP_DISTINCT_IDX_DEVICE);

    perfetto_policy::setup_counter_tracks(SETUP_DISTINCT_IDX_DEVICE, "enp226s0",
                                          make_enabled(ALL_NIC_METRICS));

    const auto& device_tracks = perfetto_policy::tracks.at(SETUP_DISTINCT_IDX_DEVICE);

    std::set<size_t> indices{};
    for(const auto& [bit_value, track_index] : device_tracks)
        EXPECT_TRUE(indices.emplace(track_index).second)
            << "duplicate track index: " << track_index;

    EXPECT_EQ(indices.size(), NIC_METRICS_COUNT);
    EXPECT_EQ(counter_track::size(SETUP_DISTINCT_IDX_DEVICE), before + NIC_METRICS_COUNT);

    // Every stored index must address a real counter track.
    for(auto idx : indices)
        EXPECT_TRUE(counter_track::exists(SETUP_DISTINCT_IDX_DEVICE,
                                          static_cast<std::int64_t>(idx)));
}

TEST_F(NicPerfettoPolicyTest, SetupTracks_KeepsDevicesSeparate)
{
    perfetto_policy::setup_counter_tracks(SETUP_ACCUMULATE_DEVICE, "enp226s0",
                                          make_enabled(RX_RDMA_UCAST_BYTES_VALUE));
    perfetto_policy::setup_counter_tracks(
        SETUP_ACCUMULATE_DEVICE + 1, "enp227s0",
        make_enabled(TX_RDMA_UCAST_BYTES_VALUE | TX_RDMA_CNP_PKTS_VALUE));

    EXPECT_EQ(perfetto_policy::tracks.at(SETUP_ACCUMULATE_DEVICE).size(), 1u);
    EXPECT_EQ(perfetto_policy::tracks.at(SETUP_ACCUMULATE_DEVICE + 1).size(), 2u);

    EXPECT_EQ(perfetto_policy::tracks.at(SETUP_ACCUMULATE_DEVICE)
                  .count(TX_RDMA_UCAST_BYTES_VALUE),
              0u);
    EXPECT_EQ(perfetto_policy::tracks.at(SETUP_ACCUMULATE_DEVICE + 1)
                  .count(RX_RDMA_UCAST_BYTES_VALUE),
              0u);
}

TEST_F(NicPerfettoPolicyTest, SetupTracks_FeedResolveTrack)
{
    // The two halves of the refactor have to agree: what setup registers is
    // what post_process_device resolves.
    const auto enabled =
        make_enabled(RX_RDMA_UCAST_BYTES_VALUE | TX_RDMA_ACK_TIMEOUT_VALUE);

    perfetto_policy::setup_counter_tracks(SETUP_RESOLVE_DEVICE, "enp226s0", enabled);

    const auto& device_tracks = perfetto_policy::tracks.at(SETUP_RESOLVE_DEVICE);

    EXPECT_GE(resolve_nic_track(enabled, RX_RDMA_UCAST_BYTES_VALUE, device_tracks), 0);
    EXPECT_GE(resolve_nic_track(enabled, TX_RDMA_ACK_TIMEOUT_VALUE, device_tracks), 0);
    EXPECT_LT(resolve_nic_track(enabled, RX_RDMA_CNP_PKTS_VALUE, device_tracks), 0);
}

// NOTE: emit_nic_counter() has no unit test on purpose. TRACE_COUNTER expands
// to CallIfCategoryEnabled(<lambda>), so the track argument is only evaluated
// while a tracing session is active. With no session the whole body is inert:
// deleting the `track_index < 0` guard still passes every assertion this file
// could make. Covering it needs a live perfetto session plus trace parsing,
// which belongs in an integration test rather than here.

}  // namespace
}  // namespace rocprofsys::pmc::collectors::nic::testing
