// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "fabric_write.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>

#include "amd_smi/amdsmi.h"

TestFabricWrite::TestFabricWrite() : TestBase() {
  set_title("AMDSMI Fabric Write Test");
  set_description(
      "Tests amdsmi_set_gpu_fabric_ppod_config(), amdsmi_set_gpu_fabric_vpod_config(), "
      "and amdsmi_set_gpu_fabric_station_config() for input validation and hw paths, "
      "along with read->set+commit->read round-trips with matching getters");
}

TestFabricWrite::~TestFabricWrite() {}

void TestFabricWrite::SetUp() { TestBase::SetUp(); }

void TestFabricWrite::DisplayTestInfo() { TestBase::DisplayTestInfo(); }

void TestFabricWrite::DisplayResults() const { TestBase::DisplayResults(); }

void TestFabricWrite::Close() { TestBase::Close(); }

/**
 *  Helpers to make minimal fabric config structs
 *      - Used to test input validation and hardware paths
 *      - All fields are set to 0 or false, except for the mask and commit bit
 *      - The mask is set to the minimal required bits for the test
 *      - The commit bit is set to false
 *      - The struct is returned
 */
static auto make_minimal_ppod_config() -> amdsmi_fabric_ppod_config_t {
  amdsmi_fabric_ppod_config_t ppod_config = {};
  ppod_config.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  ppod_config.mask = AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID;
  ppod_config.commit = false;
  ppod_config.data.accelerator_id = 0;
  return ppod_config;
}

static auto make_minimal_vpod_config() -> amdsmi_fabric_vpod_config_t {
  amdsmi_fabric_vpod_config_t vpod_config = {};
  vpod_config.version = AMDSMI_FABRIC_VPOD_CONFIG_V1;
  vpod_config.mask = AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID;
  vpod_config.commit = false;
  vpod_config.data.vpod_id = 1;  // 0 is rejected as an invalid pod ID before the hw-capability gate
  return vpod_config;
}

static auto make_minimal_station_config() -> amdsmi_fabric_station_config_t {
  amdsmi_fabric_station_config_t station_config = {};
  station_config.version = AMDSMI_FABRIC_STATION_CONFIG_V1;
  station_config.mask = AMDSMI_FABRIC_DF_FIELD_STATION_FLAGS;
  station_config.commit = false;
  station_config.data.station_flags = 0;
  return station_config;
}

/**
 *  Run the test
 */
void TestFabricWrite::Run() {
  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) { std::cout << "** SetUp Failed for this test. Skipping.**" << "\n"; }
    return;
  }

  if (num_monitor_devs() == 0) {
    IF_VERB(STANDARD) { std::cout << "\tNo GPU devices found. Skipping test." << "\n"; }
    GTEST_SKIP() << "No GPU devices found";
  }
  auto device = processor_handles_[0];

  /**
   *    IFoE capability gate: fabric-less systems report NOT_SUPPORTED from
   *    amdsmi_get_gpu_fabric_info(); skip the whole test rather than exercise the
   *    hardware paths below (matches the Fabric Read test)
   *    Probes device 0 only, assuming a homogeneous fabric config across devices;
   *    a mixed host (some devices fabric-capable, some not) would skip on device 0.
   */
  {
    auto probe = amdsmi_fabric_info_t{};
    probe.fabric_version = AMDSMI_FABRIC_INFO_VERSION_2;
    if (amdsmi_get_gpu_fabric_info(device, &probe) == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**Fabric (IFoE) not supported on this system; skipping test" << "\n";
      }
      GTEST_SKIP() << "Fabric (IFoE) not supported on this system";
    }
  }

  /**
   *    Null handle rejection (no device required)
   */
  {
    auto ppod = make_minimal_ppod_config();
    ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(nullptr, &ppod), AMDSMI_STATUS_INVAL);

    auto vpod = make_minimal_vpod_config();
    ASSERT_EQ(amdsmi_set_gpu_fabric_vpod_config(nullptr, &vpod), AMDSMI_STATUS_INVAL);

    auto station = make_minimal_station_config();
    ASSERT_EQ(amdsmi_set_gpu_fabric_station_config(nullptr, &station), AMDSMI_STATUS_INVAL);

    amdsmi_fabric_info_t info_rd = {};
    ASSERT_EQ(amdsmi_get_gpu_fabric_info(nullptr, &info_rd), AMDSMI_STATUS_INVAL);
  }

  /**
   *    Null struct pointer rejection
   */
  {
    ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(device, nullptr), AMDSMI_STATUS_INVAL);
    ASSERT_EQ(amdsmi_set_gpu_fabric_vpod_config(device, nullptr), AMDSMI_STATUS_INVAL);
    ASSERT_EQ(amdsmi_set_gpu_fabric_station_config(device, nullptr), AMDSMI_STATUS_INVAL);

    ASSERT_EQ(amdsmi_get_gpu_fabric_info(device, nullptr), AMDSMI_STATUS_INVAL);
  }

  /**
   *    Version mismatch rejection
   */
  {
    auto ppod = make_minimal_ppod_config();
    ppod.version = 0;
    ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(device, &ppod), AMDSMI_STATUS_INVAL);

    auto vpod = make_minimal_vpod_config();
    vpod.version = 0;
    ASSERT_EQ(amdsmi_set_gpu_fabric_vpod_config(device, &vpod), AMDSMI_STATUS_INVAL);

    auto station = make_minimal_station_config();
    station.version = 0;
    ASSERT_EQ(amdsmi_set_gpu_fabric_station_config(device, &station), AMDSMI_STATUS_INVAL);
  }

  /**
   *    No-op request rejection
   */
  {
    auto ppod = make_minimal_ppod_config();
    ppod.mask = 0;
    ppod.commit = false;
    ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(device, &ppod), AMDSMI_STATUS_INVAL);

    auto vpod = make_minimal_vpod_config();
    vpod.mask = 0;
    vpod.commit = false;
    ASSERT_EQ(amdsmi_set_gpu_fabric_vpod_config(device, &vpod), AMDSMI_STATUS_INVAL);

    auto station = make_minimal_station_config();
    station.mask = 0;
    station.commit = false;
    ASSERT_EQ(amdsmi_set_gpu_fabric_station_config(device, &station), AMDSMI_STATUS_INVAL);
  }

  /**
   *    Invalid mask bits rejection
   */
  {
    auto ppod = make_minimal_ppod_config();
    ppod.mask = 0xFFFFFFFF;
    ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(device, &ppod), AMDSMI_STATUS_INVAL);

    auto vpod = make_minimal_vpod_config();
    vpod.mask = 0xFFFFFFFF;
    ASSERT_EQ(amdsmi_set_gpu_fabric_vpod_config(device, &vpod), AMDSMI_STATUS_INVAL);

    auto station = make_minimal_station_config();
    station.mask = 0xFFFFFFFF;
    ASSERT_EQ(amdsmi_set_gpu_fabric_station_config(device, &station), AMDSMI_STATUS_INVAL);
  }

  /**
   *    Ppod: LOCAL_ACCELS with count=0 rejected
   */
  {
    auto ppod = make_minimal_ppod_config();
    ppod.mask = AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS;
    ppod.data.local_accelerator_count = 0;
    ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(device, &ppod), AMDSMI_STATUS_INVAL);
  }

  /**
   *    Ppod: LOCAL_ACCELS with count > max rejected
   */
  {
    auto ppod = make_minimal_ppod_config();
    ppod.mask = AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS;
    ppod.data.local_accelerator_count = (AMDSMI_FABRIC_MAX_LOCAL_GPUS + 1);
    ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(device, &ppod), AMDSMI_STATUS_INVAL);
  }

  /**
   *    Vpod: invalid addr_mode rejected before any sysfs write
   */
  {
    auto vpod = make_minimal_vpod_config();
    vpod.mask = (AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID | AMDSMI_FABRIC_VPOD_FIELD_ADDR_MODE);
    vpod.data.addr_mode = AMDSMI_FABRIC_NPA_ADDRESS_MODE_UNKNOWN;
    ASSERT_EQ(amdsmi_set_gpu_fabric_vpod_config(device, &vpod), AMDSMI_STATUS_INVAL);
  }

  /**
   *    Hw path: accepts valid requests or reports NOT_SUPPORTED
   *      - commit=false, so this reaches the support gate without writing sysfs;
   *        the round-trip block below covers the write path
   */
  for (auto dv_ind = uint32_t(0); dv_ind < num_monitor_devs(); ++dv_ind) {
    auto dev = processor_handles_[dv_ind];
    PrintDeviceHeader(dev);

    {
      auto ppod = make_minimal_ppod_config();
      auto status_code = amdsmi_set_gpu_fabric_ppod_config(dev, &ppod);
      if (status_code == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**amdsmi_set_gpu_fabric_ppod_config() not supported on this system"
                    << "\n";
        }
      } else {
        ASSERT_EQ(status_code, AMDSMI_STATUS_SUCCESS);
      }
    }

    {
      auto vpod = make_minimal_vpod_config();
      auto status_code = amdsmi_set_gpu_fabric_vpod_config(dev, &vpod);
      if (status_code == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**amdsmi_set_gpu_fabric_vpod_config() not supported on this system"
                    << "\n";
        }
      } else {
        ASSERT_EQ(status_code, AMDSMI_STATUS_SUCCESS);
      }
    }

    {
      auto station = make_minimal_station_config();
      auto status_code = amdsmi_set_gpu_fabric_station_config(dev, &station);
      if (status_code == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**amdsmi_set_gpu_fabric_station_config() not supported on this system"
                    << "\n";
        }
      } else {
        ASSERT_EQ(status_code, AMDSMI_STATUS_SUCCESS);
      }
    }
  }

  /**
   *    Round-trip: (set+commit) -> read back via amdsmi_get_gpu_fabric_info()
   *      - The per-plane getters were removed; amdsmi_get_gpu_fabric_info() is the sole reader
   *      - A baseline read establishes whether fabric is present (NOT_SUPPORTED skips the plane)
   *      - Only fields with a non-sentinel baseline are exercised: those can be restored exactly,
   *        so the device is left as found. Unconfigured fields (baseline == UINT*_MAX) are skipped
   *        rather than written, since a sentinel cannot be restored and committing would leave the
   *        device permanently mutated
   *      - round_trips_verified guards a silent pass when no field could be exercised
   */
  auto round_trips_verified = 0;
  for (auto dv_ind = uint32_t(0); dv_ind < num_monitor_devs(); ++dv_ind) {
    auto dev = processor_handles_[dv_ind];
    PrintDeviceHeader(dev);

    auto baseline_info = amdsmi_fabric_info_t{};
    baseline_info.fabric_version = AMDSMI_FABRIC_INFO_VERSION_2;
    auto baseline_status = amdsmi_get_gpu_fabric_info(dev, &baseline_info);
    if (baseline_status == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**Fabric round-trip not supported on this system" << "\n";
      }
      continue;
    }
    ASSERT_TRUE((baseline_status == AMDSMI_STATUS_SUCCESS) ||
                (baseline_status == AMDSMI_STATUS_NO_DATA));

    const auto& baseline_v2 = baseline_info.fabric_info.v2;

    /**
     *  Ppod: accelerator_id round-trip
     */
    {
      const auto baseline_accel_id = baseline_v2.ppod.accelerator_id;
      if (baseline_accel_id == std::numeric_limits<uint32_t>::max()) {
        IF_VERB(STANDARD) {
          std::cout << "\t**Ppod accelerator_id unconfigured; skipping round-trip "
                       "to avoid mutating hardware"
                    << "\n";
        }
      } else {
        auto wr = make_minimal_ppod_config();
        wr.mask = AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID;
        wr.commit = true;
        wr.data.accelerator_id = 7;
        auto set_status = amdsmi_set_gpu_fabric_ppod_config(dev, &wr);
        if (set_status != AMDSMI_STATUS_NOT_SUPPORTED) {
          ASSERT_EQ(set_status, AMDSMI_STATUS_SUCCESS);
          ++round_trips_verified;

          auto post = amdsmi_fabric_info_t{};
          post.fabric_version = AMDSMI_FABRIC_INFO_VERSION_2;
          auto rd = amdsmi_get_gpu_fabric_info(dev, &post);
          ASSERT_EQ(rd, AMDSMI_STATUS_SUCCESS);
          ASSERT_EQ(post.fabric_info.v2.ppod.accelerator_id, 7u);

          auto restore = make_minimal_ppod_config();
          restore.mask = AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID;
          restore.commit = true;
          restore.data.accelerator_id = baseline_accel_id;
          ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(dev, &restore), AMDSMI_STATUS_SUCCESS);
        }
      }
    }

    /**
     *  Vpod: vpod_id round-trip
     */
    {
      const auto baseline_vpod_id = baseline_v2.vpod.vpod_id;
      if (baseline_vpod_id == std::numeric_limits<uint32_t>::max()) {
        IF_VERB(STANDARD) {
          std::cout << "\t**Vpod vpod_id unconfigured; skipping round-trip "
                       "to avoid mutating hardware"
                    << "\n";
        }
      } else {
        auto wr = make_minimal_vpod_config();
        wr.mask = AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID;
        wr.commit = true;
        wr.data.vpod_id = 3;
        auto set_status = amdsmi_set_gpu_fabric_vpod_config(dev, &wr);
        if (set_status != AMDSMI_STATUS_NOT_SUPPORTED) {
          ASSERT_EQ(set_status, AMDSMI_STATUS_SUCCESS);
          ++round_trips_verified;

          auto post = amdsmi_fabric_info_t{};
          post.fabric_version = AMDSMI_FABRIC_INFO_VERSION_2;
          auto rd = amdsmi_get_gpu_fabric_info(dev, &post);
          ASSERT_EQ(rd, AMDSMI_STATUS_SUCCESS);
          ASSERT_EQ(post.fabric_info.v2.vpod.vpod_id, 3u);

          auto restore = make_minimal_vpod_config();
          restore.mask = AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID;
          restore.commit = true;
          restore.data.vpod_id = baseline_vpod_id;
          ASSERT_EQ(amdsmi_set_gpu_fabric_vpod_config(dev, &restore), AMDSMI_STATUS_SUCCESS);
        }
      }
    }

    /**
     *  Vpod: vpod_active_accelerators list round-trip
     *      - Unused slots must hold UINT32_MAX; writing two IDs must read back exactly
     *        two, with the next slot still at the sentinel (guards trailing slots being
     *        serialized as accelerator ID 0)
     */
    {
      const auto baseline_head = baseline_v2.vpod.vpod_active_accelerators[0];
      if (baseline_head == std::numeric_limits<uint32_t>::max()) {
        IF_VERB(STANDARD) {
          std::cout << "\t**Vpod vpod_active_accelerators unconfigured; skipping round-trip "
                       "to avoid mutating hardware"
                    << "\n";
        }
      } else {
        auto wr = make_minimal_vpod_config();
        wr.mask = AMDSMI_FABRIC_VPOD_FIELD_VPOD_ACTIVE_ACCELS;
        wr.commit = true;
        std::fill(std::begin(wr.data.vpod_active_accelerators),
                  std::end(wr.data.vpod_active_accelerators), std::numeric_limits<uint32_t>::max());
        wr.data.vpod_active_accelerators[0] = 2;
        wr.data.vpod_active_accelerators[1] = 5;
        auto set_status = amdsmi_set_gpu_fabric_vpod_config(dev, &wr);
        if (set_status != AMDSMI_STATUS_NOT_SUPPORTED) {
          ASSERT_EQ(set_status, AMDSMI_STATUS_SUCCESS);
          ++round_trips_verified;

          auto post = amdsmi_fabric_info_t{};
          post.fabric_version = AMDSMI_FABRIC_INFO_VERSION_2;
          auto rd = amdsmi_get_gpu_fabric_info(dev, &post);
          ASSERT_EQ(rd, AMDSMI_STATUS_SUCCESS);
          const auto& accels = post.fabric_info.v2.vpod.vpod_active_accelerators;
          ASSERT_EQ(accels[0], 2u);
          ASSERT_EQ(accels[1], 5u);
          ASSERT_EQ(accels[2], std::numeric_limits<uint32_t>::max());

          auto restore = make_minimal_vpod_config();
          restore.mask = AMDSMI_FABRIC_VPOD_FIELD_VPOD_ACTIVE_ACCELS;
          restore.commit = true;
          std::copy(std::begin(baseline_v2.vpod.vpod_active_accelerators),
                    std::end(baseline_v2.vpod.vpod_active_accelerators),
                    std::begin(restore.data.vpod_active_accelerators));
          ASSERT_EQ(amdsmi_set_gpu_fabric_vpod_config(dev, &restore), AMDSMI_STATUS_SUCCESS);
        }
      }
    }

    /**
     *  Station: station_flags round-trip
     */
    {
      const auto baseline_station_flags = baseline_v2.station.station_flags;
      if (baseline_station_flags == std::numeric_limits<uint32_t>::max()) {
        IF_VERB(STANDARD) {
          std::cout << "\t**Station station_flags unconfigured; skipping round-trip "
                       "to avoid mutating hardware"
                    << "\n";
        }
      } else {
        auto wr = make_minimal_station_config();
        wr.mask = AMDSMI_FABRIC_DF_FIELD_STATION_FLAGS;
        wr.commit = true;
        wr.data.station_flags = 1;
        auto set_status = amdsmi_set_gpu_fabric_station_config(dev, &wr);
        if (set_status != AMDSMI_STATUS_NOT_SUPPORTED) {
          ASSERT_EQ(set_status, AMDSMI_STATUS_SUCCESS);
          ++round_trips_verified;

          auto post = amdsmi_fabric_info_t{};
          post.fabric_version = AMDSMI_FABRIC_INFO_VERSION_2;
          auto rd = amdsmi_get_gpu_fabric_info(dev, &post);
          ASSERT_EQ(rd, AMDSMI_STATUS_SUCCESS);
          ASSERT_EQ(post.fabric_info.v2.station.station_flags, 1u);

          auto restore = make_minimal_station_config();
          restore.mask = AMDSMI_FABRIC_DF_FIELD_STATION_FLAGS;
          restore.commit = true;
          restore.data.station_flags = baseline_station_flags;
          ASSERT_EQ(amdsmi_set_gpu_fabric_station_config(dev, &restore), AMDSMI_STATUS_SUCCESS);
        }
      }
    }
  }

  if (round_trips_verified == 0) {
    IF_VERB(STANDARD) {
      std::cout << "\t**No fabric field had a restorable baseline with a supported setter; "
                   "round-trip verified nothing"
                << "\n";
    }
    GTEST_SKIP() << "Fabric round-trip exercised no field (setters unsupported or all "
                    "fields unconfigured)";
  }
}
