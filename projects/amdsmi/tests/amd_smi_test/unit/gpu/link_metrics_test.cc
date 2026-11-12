// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

#include "link_metrics_internal.h"
#include "xgmi_port_map_internal.h"

namespace {

struct LinkMetricsInput {
  LinkMetricsInput() {
    std::fill_n(mapping, AMDSMI_MAX_NUM_XGMI_LINKS, UINT16_MAX);
    std::fill_n(metrics.xgmi_read_data_acc, AMDSMI_MAX_NUM_XGMI_LINKS, UINT64_MAX);
    std::fill_n(metrics.xgmi_write_data_acc, AMDSMI_MAX_NUM_XGMI_LINKS, UINT64_MAX);
    metrics.xgmi_link_speed = 36;
    metrics.xgmi_link_width = 16;
  }

  void AddLink(uint32_t slot) {
    mapping[slot] = static_cast<uint16_t>(100 + slot);
    metrics.xgmi_read_data_acc[slot] = (uint64_t{1} << 40) + slot;
    metrics.xgmi_write_data_acc[slot] = (uint64_t{1} << 48) + slot;
    ++count;
  }

  amdsmi_status_t Query(amdsmi_link_metrics_t* result) const {
    return amd::smi::detail::populate_link_metrics(metrics, mapping, count, map_status, result,
                                                   [](uint16_t node) {
                                                     amdsmi_bdf_t bdf{};
                                                     bdf.bus_number = static_cast<uint8_t>(node);
                                                     return bdf;
                                                   });
  }

  amdsmi_gpu_metrics_t metrics{};
  uint16_t mapping[AMDSMI_MAX_NUM_XGMI_LINKS];
  uint32_t count = 0;
  amdsmi_status_t map_status = AMDSMI_STATUS_SUCCESS;
};

}  // namespace

TEST(GpuUnit, LinkMetricsCompactsSparsePhysicalSlots) {
  LinkMetricsInput input;
  for (uint32_t slot = 1; slot <= 7; ++slot) input.AddLink(slot);
  amdsmi_link_metrics_t result{};

  ASSERT_EQ(input.Query(&result), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(result.num_links, 7u);
  for (uint32_t index = 0; index < result.num_links; ++index) {
    const uint32_t slot = index + 1;
    EXPECT_EQ(result.links[index].bdf.bus_number, input.mapping[slot]) << index;
    EXPECT_EQ(result.links[index].read, input.metrics.xgmi_read_data_acc[slot]) << index;
    EXPECT_EQ(result.links[index].write, input.metrics.xgmi_write_data_acc[slot]) << index;
    EXPECT_EQ(result.links[index].bit_rate, 36u) << index;
    EXPECT_EQ(result.links[index].max_bandwidth, 576u) << index;
    EXPECT_EQ(result.links[index].link_type, AMDSMI_LINK_TYPE_XGMI) << index;
  }
}

TEST(GpuUnit, LinkMetricsPreservesMapFailure) {
  LinkMetricsInput input;
  input.AddLink(7);
  for (const auto status : {AMDSMI_STATUS_UNEXPECTED_SIZE, AMDSMI_STATUS_UNEXPECTED_DATA}) {
    input.map_status = status;
    amdsmi_link_metrics_t result{};
    result.num_links = 17;
    EXPECT_EQ(input.Query(&result), status);
    EXPECT_EQ(result.num_links, 0u);
  }
}

TEST(GpuUnit, LinkMetricsKeepsCountersWithoutPortMap) {
  LinkMetricsInput input;
  input.AddLink(2);
  input.AddLink(5);
  for (const auto status : {AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_FILE_ERROR}) {
    input.map_status = status;
    amdsmi_link_metrics_t result{};
    result.num_links = 17;
    ASSERT_EQ(input.Query(&result), AMDSMI_STATUS_SUCCESS);
    ASSERT_EQ(result.num_links, 2u);
    EXPECT_EQ(result.links[0].bdf.as_uint, UINT64_MAX);
    EXPECT_EQ(result.links[1].bdf.as_uint, UINT64_MAX);
    EXPECT_EQ(result.links[0].read, input.metrics.xgmi_read_data_acc[2]);
    EXPECT_EQ(result.links[1].write, input.metrics.xgmi_write_data_acc[5]);
  }
}

TEST(GpuUnit, LinkMetricsWithoutPortMapOrCountersIsEmpty) {
  LinkMetricsInput input;
  input.map_status = AMDSMI_STATUS_NOT_SUPPORTED;
  amdsmi_link_metrics_t result{};
  result.num_links = 17;
  ASSERT_EQ(input.Query(&result), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(result.num_links, 0u);
}

TEST(GpuUnit, LinkMetricsKeepsUnmappedPhysicalCounters) {
  LinkMetricsInput input;
  input.metrics.xgmi_read_data_acc[0] = 0;
  input.metrics.xgmi_write_data_acc[0] = 0;
  for (uint32_t slot = 1; slot <= 7; ++slot) input.AddLink(slot);
  amdsmi_link_metrics_t result{};

  ASSERT_EQ(input.Query(&result), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(result.num_links, 8u);
  EXPECT_EQ(result.links[0].bdf.as_uint, UINT64_MAX);
  EXPECT_EQ(result.links[0].read, 0u);
  EXPECT_EQ(result.links[0].write, 0u);
  for (uint32_t slot = 1; slot <= 7; ++slot) {
    EXPECT_EQ(result.links[slot].bdf.bus_number, input.mapping[slot]);
    EXPECT_EQ(result.links[slot].read, input.metrics.xgmi_read_data_acc[slot]);
    EXPECT_EQ(result.links[slot].write, input.metrics.xgmi_write_data_acc[slot]);
  }
}

TEST(GpuUnit, LinkMetricsKeepsOneSidedCountersWithoutMapping) {
  LinkMetricsInput input;
  input.metrics.xgmi_write_data_acc[1] = UINT64_MAX - 1;
  input.metrics.xgmi_read_data_acc[AMDSMI_MAX_NUM_XGMI_LINKS - 1] = UINT64_MAX - 2;
  amdsmi_link_metrics_t result{};

  ASSERT_EQ(input.Query(&result), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(result.num_links, 2u);
  EXPECT_EQ(result.links[0].bdf.as_uint, UINT64_MAX);
  EXPECT_EQ(result.links[0].read, UINT64_MAX);
  EXPECT_EQ(result.links[0].write, UINT64_MAX - 1);
  EXPECT_EQ(result.links[1].read, UINT64_MAX - 2);
  EXPECT_EQ(result.links[1].write, UINT64_MAX);
}

TEST(GpuUnit, LinkMetricsHandlesEmptyDenseAndMaximumCapacity) {
  for (const uint32_t count : {0u, 3u, static_cast<uint32_t>(AMDSMI_MAX_NUM_XGMI_LINKS)}) {
    LinkMetricsInput input;
    for (uint32_t slot = 0; slot < count; ++slot) input.AddLink(slot);
    struct {
      amdsmi_link_metrics_t result{};
      uint64_t canary = UINT64_MAX;
    } output;
    output.result.num_links = UINT32_MAX;

    ASSERT_EQ(input.Query(&output.result), AMDSMI_STATUS_SUCCESS);
    ASSERT_EQ(output.result.num_links, count);
    for (uint32_t slot = 0; slot < count; ++slot) {
      EXPECT_EQ(output.result.links[slot].bdf.bus_number, input.mapping[slot]);
      EXPECT_EQ(output.result.links[slot].read, input.metrics.xgmi_read_data_acc[slot]);
      EXPECT_EQ(output.result.links[slot].write, input.metrics.xgmi_write_data_acc[slot]);
    }
    EXPECT_EQ(output.canary, UINT64_MAX);
  }
}

TEST(GpuUnit, LinkMetricsPreservesUnavailableFieldsForMappedSlots) {
  LinkMetricsInput input;
  input.AddLink(AMDSMI_MAX_NUM_XGMI_LINKS - 1);
  std::fill_n(input.metrics.xgmi_read_data_acc, AMDSMI_MAX_NUM_XGMI_LINKS, UINT64_MAX);
  std::fill_n(input.metrics.xgmi_write_data_acc, AMDSMI_MAX_NUM_XGMI_LINKS, UINT64_MAX);
  input.metrics.xgmi_link_speed = UINT16_MAX;
  input.metrics.xgmi_link_width = UINT16_MAX;
  amdsmi_link_metrics_t result{};

  ASSERT_EQ(input.Query(&result), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(result.num_links, 1u);
  EXPECT_EQ(result.links[0].read, UINT64_MAX);
  EXPECT_EQ(result.links[0].write, UINT64_MAX);
  EXPECT_EQ(result.links[0].bit_rate, UINT32_MAX);
  EXPECT_EQ(result.links[0].max_bandwidth, UINT32_MAX);
}

TEST(GpuUnit, LinkMetricsRejectsOverreportedMapCount) {
  LinkMetricsInput input;
  input.AddLink(1);
  input.count = AMDSMI_MAX_NUM_XGMI_LINKS + 1;
  amdsmi_link_metrics_t result{};

  EXPECT_EQ(input.Query(&result), AMDSMI_STATUS_UNEXPECTED_SIZE);
  EXPECT_EQ(result.num_links, 0u);
}

TEST(GpuUnit, LinkMetricsPortMapRejectsDuplicateSourceSlots) {
  uint16_t mapping[RSMI_MAX_NUM_XGMI_LINKS];
  std::fill_n(mapping, RSMI_MAX_NUM_XGMI_LINKS, UINT16_MAX);
  uint32_t count = 0;

  EXPECT_EQ(amd::smi::detail::parse_xgmi_port_map("0:1 -> 2:0\n0:1 -> 3:0\n", &count, mapping),
            RSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, LinkMetricsPortMapRejectsOutOfRangeSourceSlot) {
  uint16_t mapping[RSMI_MAX_NUM_XGMI_LINKS + 1];
  std::fill_n(mapping, RSMI_MAX_NUM_XGMI_LINKS + 1, UINT16_MAX);
  uint32_t count = 0;

  EXPECT_EQ(amd::smi::detail::parse_xgmi_port_map("0:8 -> 2:0\n", &count, mapping),
            RSMI_STATUS_UNEXPECTED_SIZE);
  EXPECT_EQ(mapping[RSMI_MAX_NUM_XGMI_LINKS], UINT16_MAX);
  EXPECT_EQ(count, 0u);
}

TEST(GpuUnit, LinkMetricsPortMapRejectsMalformedRecordsWithoutThrowing) {
  for (const auto* record :
       {"0:1 -> 65536:0", "0:1 -> 65535:0", "0:bad -> 2:0", "0:99999999999999999999 -> 2:0",
        "0:1 -> -2:0", "0:1x -> 2:0", "broken", "0:1 xx 2:0"}) {
    uint16_t mapping[RSMI_MAX_NUM_XGMI_LINKS];
    std::fill_n(mapping, RSMI_MAX_NUM_XGMI_LINKS, UINT16_MAX);
    uint32_t count = 0;
    SCOPED_TRACE(record);
    EXPECT_NO_THROW(EXPECT_EQ(amd::smi::detail::parse_xgmi_port_map(record, &count, mapping),
                              RSMI_STATUS_UNEXPECTED_DATA));
    EXPECT_EQ(count, 0u);
    for (const auto node : mapping) EXPECT_EQ(node, UINT16_MAX);
  }
}

TEST(GpuUnit, LinkMetricsPreservesFullBandwidthFieldWidth) {
  LinkMetricsInput input;
  input.AddLink(1);
  input.metrics.xgmi_link_speed = UINT16_MAX - 1;
  input.metrics.xgmi_link_width = UINT16_MAX - 1;
  amdsmi_link_metrics_t result{};

  ASSERT_EQ(input.Query(&result), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(result.num_links, 1u);
  EXPECT_EQ(result.links[0].bit_rate, 65534u);
  EXPECT_EQ(result.links[0].max_bandwidth, uint32_t{65534} * 65534u);
}

TEST(GpuUnit, LinkMetricsParsesSparseMapInPhysicalOrder) {
  for (const bool unmapped_counters_available : {false, true}) {
    LinkMetricsInput input;
    for (uint32_t slot = 1; slot <= 7; ++slot) input.AddLink(slot);
    if (unmapped_counters_available) {
      input.metrics.xgmi_read_data_acc[0] = 0;
      input.metrics.xgmi_write_data_acc[0] = 0;
    }
    ASSERT_EQ(amd::smi::detail::parse_xgmi_port_map(
                  "01:04 -> 02:04\n01:06 -> 03:04\n01:03 -> 04:03\n01:01 -> 05:01\n"
                  "01:05 -> 06:05\n01:07 -> 07:07\n01:02 -> 08:02",
                  &input.count, input.mapping),
              RSMI_STATUS_SUCCESS);
    ASSERT_EQ(input.count, 7u);
    amdsmi_link_metrics_t result{};

    ASSERT_EQ(input.Query(&result), AMDSMI_STATUS_SUCCESS);
    const uint32_t offset = unmapped_counters_available ? 1 : 0;
    ASSERT_EQ(result.num_links, 7u + offset);
    const uint16_t nodes[] = {5, 8, 4, 2, 6, 3, 7};
    for (uint32_t index = 0; index < 7; ++index) {
      EXPECT_EQ(result.links[index + offset].bdf.bus_number, nodes[index]);
      EXPECT_EQ(result.links[index + offset].read, input.metrics.xgmi_read_data_acc[index + 1]);
      EXPECT_EQ(result.links[index + offset].write, input.metrics.xgmi_write_data_acc[index + 1]);
    }
  }
}

TEST(GpuUnit, LinkMetricsPortMapHandlesEmptyDenseAndMaximumCapacity) {
  for (const uint32_t count : {0u, 3u, static_cast<uint32_t>(RSMI_MAX_NUM_XGMI_LINKS)}) {
    std::string contents;
    for (uint32_t remaining = count; remaining > 0; --remaining) {
      contents +=
          "0:" + std::to_string(remaining - 1) + " -> " + std::to_string(remaining + 10) + ":0\n";
    }
    uint16_t mapping[RSMI_MAX_NUM_XGMI_LINKS] = {};
    uint32_t parsed_count = UINT32_MAX;
    ASSERT_EQ(amd::smi::detail::parse_xgmi_port_map(contents, &parsed_count, mapping),
              RSMI_STATUS_SUCCESS);
    EXPECT_EQ(parsed_count, count);
    for (uint32_t slot = 0; slot < RSMI_MAX_NUM_XGMI_LINKS; ++slot) {
      EXPECT_EQ(mapping[slot], slot < count ? slot + 11 : UINT16_MAX);
    }
  }
}

TEST(GpuUnit, LinkMetricsPreservesUnresolvedBdfAndUnavailableWidth) {
  LinkMetricsInput input;
  input.AddLink(7);
  input.metrics.xgmi_link_width = UINT16_MAX;
  amdsmi_link_metrics_t result{};

  ASSERT_EQ(amd::smi::detail::populate_link_metrics(input.metrics, input.mapping, input.count,
                                                    input.map_status, &result,
                                                    [](uint16_t) {
                                                      amdsmi_bdf_t bdf{};
                                                      bdf.as_uint = UINT64_MAX;
                                                      return bdf;
                                                    }),
            AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(result.num_links, 1u);
  EXPECT_EQ(result.links[0].bdf.as_uint, UINT64_MAX);
  EXPECT_EQ(result.links[0].bit_rate, 36u);
  EXPECT_EQ(result.links[0].max_bandwidth, UINT32_MAX);
  EXPECT_EQ(result.links[0].read, input.metrics.xgmi_read_data_acc[7]);
  EXPECT_EQ(result.links[0].write, input.metrics.xgmi_write_data_acc[7]);
}
