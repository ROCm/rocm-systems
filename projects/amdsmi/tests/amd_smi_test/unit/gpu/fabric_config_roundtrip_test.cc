// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Hardware-free round-trip of the fabric config serialize/parse paths. The
// apply_*_at / query_*_at cores take a resolved sysfs root + support flag instead
// of a live device, so the real WriteRequestBuilder/FieldReader logic is exercised
// against a temp directory: no hardware, and the commit file lands in that temp
// directory rather than on a driver. Round-trip cases set commit=true because a
// non-committing request writes nothing.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

#include "amd_smi/impl/amd_smi_fabric_ualink.h"

namespace amd::smi {

namespace {

namespace fs = std::filesystem;

constexpr auto kPpodAllFields = static_cast<uint32_t>(
    AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID | AMDSMI_FABRIC_PPOD_FIELD_PPOD_ID |
    AMDSMI_FABRIC_PPOD_FIELD_PPOD_SIZE | AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS |
    AMDSMI_FABRIC_PPOD_FIELD_BANDWIDTH | AMDSMI_FABRIC_PPOD_FIELD_LATENCY);

constexpr auto kVpodAllFields = static_cast<uint32_t>(
    AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID | AMDSMI_FABRIC_VPOD_FIELD_VPOD_SIZE |
    AMDSMI_FABRIC_VPOD_FIELD_VPOD_ACTIVE_ACCELS | AMDSMI_FABRIC_VPOD_FIELD_ADDR_MODE);

constexpr auto kStationAllFields = static_cast<uint32_t>(AMDSMI_FABRIC_DF_FIELD_STATION_FLAGS |
                                                         AMDSMI_FABRIC_DF_FIELD_NUM_STATIONS |
                                                         AMDSMI_FABRIC_DF_FIELD_LANE_EN_BITMAP);

constexpr auto kU32Sentinel = std::numeric_limits<uint32_t>::max();

// RAII sysfs root: creates the setup/config/stations subtree in a fresh temp dir
// and removes it on destruction. A plain helper (not a GTest fixture) so these
// tests can register under the shared GpuUnit suite required by test-design.md.
struct TempRoot {
  std::string path;

  TempRoot() {
    auto tmpl = std::string("/tmp/amdsmi_fabric_rt_XXXXXX");
    if (::mkdtemp(tmpl.data()) == nullptr) {
      return;
    }
    path = tmpl;
    for (const auto subdir :
         {kUALOE_UALINK_SETUP_SUBDIR, kUALOE_UALINK_CONFIG_SUBDIR, kUALOE_UALINK_STATIONS_SUBDIR}) {
      auto ec = std::error_code{};
      fs::create_directories(fs::path(path) / std::string(subdir), ec);
    }
  }

  ~TempRoot() {
    if (path.empty()) {
      return;
    }
    auto ec = std::error_code{};
    fs::remove_all(path, ec);
  }
};

TEST(GpuUnit, FabricConfigPpodAllFieldsRoundTrip) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_ppod_config_t{};
  in.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  in.mask = kPpodAllFields;
  in.commit = true;
  in.data.accelerator_id = 42;
  for (auto idx = 0; idx < AMDSMI_MAX_UUID_ELEMENTS; ++idx) {
    in.data.ppod_id[idx] = static_cast<uint8_t>(0x10 + idx);
  }
  in.data.ppod_size = 8;
  in.data.local_accelerators[0] = 100;
  in.data.local_accelerators[1] = 101;
  in.data.local_accelerators[2] = 102;
  in.data.local_accelerator_count = 3;
  in.data.bandwidth = 1234;
  in.data.latency = 56;

  ASSERT_EQ(fabric_ualink::apply_ppod_config_at(root.path, true, in), AMDSMI_STATUS_SUCCESS);

  auto out = amdsmi_fabric_ppod_config_t{};
  out.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  out.mask = kPpodAllFields;
  ASSERT_EQ(fabric_ualink::query_ppod_config_at(root.path, true, out), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(out.mask, kPpodAllFields);
  EXPECT_EQ(out.data.accelerator_id, 42u);
  for (auto idx = 0; idx < AMDSMI_MAX_UUID_ELEMENTS; ++idx) {
    EXPECT_EQ(out.data.ppod_id[idx], static_cast<uint8_t>(0x10 + idx)) << "ppod_id[" << idx << "]";
  }
  EXPECT_EQ(out.data.ppod_size, 8u);
  EXPECT_EQ(out.data.local_accelerator_count, 3u);
  EXPECT_EQ(out.data.local_accelerators[0], 100u);
  EXPECT_EQ(out.data.local_accelerators[1], 101u);
  EXPECT_EQ(out.data.local_accelerators[2], 102u);
  EXPECT_EQ(out.data.bandwidth, 1234u);
  EXPECT_EQ(out.data.latency, 56u);
}

// local_accelerator_count < list length: the serializer must stop at the count and
// the reader must not resurrect a phantom tail from the write-side array.
TEST(GpuUnit, FabricConfigPpodLocalAccelsStopAtCount) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_ppod_config_t{};
  in.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  in.mask = AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS;
  in.commit = true;
  for (auto idx = 0; idx < AMDSMI_FABRIC_MAX_LOCAL_GPUS; ++idx) {
    in.data.local_accelerators[idx] = static_cast<uint32_t>(200 + idx);
  }
  in.data.local_accelerator_count = 3;  // only 200, 201, 202 are valid

  ASSERT_EQ(fabric_ualink::apply_ppod_config_at(root.path, true, in), AMDSMI_STATUS_SUCCESS);

  auto out = amdsmi_fabric_ppod_config_t{};
  out.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  out.mask = AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS;
  ASSERT_EQ(fabric_ualink::query_ppod_config_at(root.path, true, out), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(out.data.local_accelerator_count, 3u);
  EXPECT_EQ(out.data.local_accelerators[0], 200u);
  EXPECT_EQ(out.data.local_accelerators[1], 201u);
  EXPECT_EQ(out.data.local_accelerators[2], 202u);
  EXPECT_EQ(out.data.local_accelerators[3], kU32Sentinel);
}

// A field left out of the write mask must read back at its guard sentinel, and the
// read mask must report only the field that was actually persisted.
TEST(GpuUnit, FabricConfigPpodPartialWriteLeavesUnwrittenAtSentinel) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_ppod_config_t{};
  in.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  in.mask = AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID;
  in.commit = true;
  in.data.accelerator_id = 77;
  ASSERT_EQ(fabric_ualink::apply_ppod_config_at(root.path, true, in), AMDSMI_STATUS_SUCCESS);

  auto out = amdsmi_fabric_ppod_config_t{};
  out.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  out.mask = kPpodAllFields;
  ASSERT_EQ(fabric_ualink::query_ppod_config_at(root.path, true, out), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(out.mask, static_cast<uint32_t>(AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID));
  EXPECT_EQ(out.data.accelerator_id, 77u);
  EXPECT_EQ(out.data.ppod_size, kU32Sentinel);
  EXPECT_EQ(out.data.bandwidth, kU32Sentinel);
  EXPECT_EQ(out.data.latency, kU32Sentinel);
}

// A 0 in the active-accelerator list is not a real ID and must terminate the
// serialized run; slots past it stay at the read sentinel.
TEST(GpuUnit, FabricConfigVpodRoundTripAccelIdZeroPreserved) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_vpod_config_t{};
  in.version = AMDSMI_FABRIC_VPOD_CONFIG_V1;
  in.mask = kVpodAllFields;
  in.commit = true;
  in.data.vpod_id = 7;
  in.data.vpod_size = 4;
  // Contract: unused slots hold the UNSET sentinel; the run ends at the first one.
  for (auto& id : in.data.vpod_active_accelerators) {
    id = kU32Sentinel;
  }
  in.data.vpod_active_accelerators[0] = 3;
  in.data.vpod_active_accelerators[1] = 0;  // 0 is a valid accelerator ID, not a terminator
  in.data.vpod_active_accelerators[2] = 5;
  in.data.addr_mode = AMDSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_ALIASING;

  ASSERT_EQ(fabric_ualink::apply_vpod_config_at(root.path, true, in), AMDSMI_STATUS_SUCCESS);

  auto out = amdsmi_fabric_vpod_config_t{};
  out.version = AMDSMI_FABRIC_VPOD_CONFIG_V1;
  out.mask = kVpodAllFields;
  ASSERT_EQ(fabric_ualink::query_vpod_config_at(root.path, true, out), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(out.mask, kVpodAllFields);
  EXPECT_EQ(out.data.vpod_id, 7u);
  EXPECT_EQ(out.data.vpod_size, 4u);
  EXPECT_EQ(out.data.vpod_active_accelerators[0], 3u);
  EXPECT_EQ(out.data.vpod_active_accelerators[1], 0u);
  EXPECT_EQ(out.data.vpod_active_accelerators[2], 5u);
  EXPECT_EQ(out.data.vpod_active_accelerators[3], kU32Sentinel);
  EXPECT_EQ(out.data.addr_mode, AMDSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_ALIASING);
}

TEST(GpuUnit, FabricConfigStationRoundTripFullBitmap) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_station_config_t{};
  in.version = AMDSMI_FABRIC_STATION_CONFIG_V1;
  in.mask = kStationAllFields;
  in.commit = true;
  in.data.station_flags = 0xABCD;
  in.data.num_stations = 12;
  for (auto idx = 0; idx < AMDSMI_FABRIC_MAX_BITMAP_SIZE; ++idx) {
    in.data.lane_en_bitmap[idx] = static_cast<uint8_t>((idx * 7) & 0xFF);
  }

  ASSERT_EQ(fabric_ualink::apply_station_config_at(root.path, true, in), AMDSMI_STATUS_SUCCESS);

  auto out = amdsmi_fabric_station_config_t{};
  out.version = AMDSMI_FABRIC_STATION_CONFIG_V1;
  out.mask = kStationAllFields;
  ASSERT_EQ(fabric_ualink::query_station_config_at(root.path, true, out), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(out.mask, kStationAllFields);
  EXPECT_EQ(out.data.station_flags, 0xABCDu);
  EXPECT_EQ(out.data.num_stations, static_cast<uint8_t>(12));
  for (auto idx = 0; idx < AMDSMI_FABRIC_MAX_BITMAP_SIZE; ++idx) {
    EXPECT_EQ(out.data.lane_en_bitmap[idx], static_cast<uint8_t>((idx * 7) & 0xFF))
        << "lane_en_bitmap[" << idx << "]";
  }
}

// commit=true writes the subtree commit file after the masked fields.
TEST(GpuUnit, FabricConfigPpodCommitWritesCommitFile) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_ppod_config_t{};
  in.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  in.mask = AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID;
  in.commit = true;
  in.data.accelerator_id = 3;
  ASSERT_EQ(fabric_ualink::apply_ppod_config_at(root.path, true, in), AMDSMI_STATUS_SUCCESS);

  const auto commit_path = fs::path(root.path) / std::string(kUALOE_UALINK_SETUP_SUBDIR) /
                           std::string(kUALOE_UALINK_COMMIT_FILE);
  ASSERT_TRUE(fs::exists(commit_path));
  auto stream = std::ifstream(commit_path);
  auto contents = std::string();
  std::getline(stream, contents);
  EXPECT_EQ(contents, "1");
}

// commit=false validates and writes nothing: the driver ignores the staging files
// unless a commit follows, so leaving values there would only feed the next commit.
TEST(GpuUnit, FabricConfigNoCommitWritesNothing) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_ppod_config_t{};
  in.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  in.mask = AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID;
  in.commit = false;
  in.data.accelerator_id = 3;
  ASSERT_EQ(fabric_ualink::apply_ppod_config_at(root.path, true, in), AMDSMI_STATUS_SUCCESS);

  const auto setup_dir = fs::path(root.path) / std::string(kUALOE_UALINK_SETUP_SUBDIR);
  EXPECT_FALSE(fs::exists(setup_dir / std::string(kUALOE_ACCEL_ID)));
  EXPECT_FALSE(fs::exists(setup_dir / std::string(kUALOE_UALINK_COMMIT_FILE)));
}

// device_supports_ualink=false is rejected even when the sysfs tree exists.
TEST(GpuUnit, FabricConfigApplyUnsupportedDeviceNotSupported) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_ppod_config_t{};
  in.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  in.mask = AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID;
  in.data.accelerator_id = 1;
  EXPECT_EQ(fabric_ualink::apply_ppod_config_at(root.path, false, in), AMDSMI_STATUS_NOT_SUPPORTED);
}

// Input validation precedes the support gate: a bad version is INVAL regardless of
// device support, preserving the validation-before-hardware-gate contract.
TEST(GpuUnit, FabricConfigApplyValidationRunsBeforeSupportGate) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_ppod_config_t{};
  in.version = 0xBADBADu;  // wrong version
  in.mask = AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID;
  EXPECT_EQ(fabric_ualink::apply_ppod_config_at(root.path, false, in), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, FabricConfigApplyLocalAccelCountOutOfRangeInval) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_ppod_config_t{};
  in.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  in.mask = AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS;
  in.data.local_accelerator_count = static_cast<uint32_t>(AMDSMI_FABRIC_MAX_LOCAL_GPUS + 1);
  EXPECT_EQ(fabric_ualink::apply_ppod_config_at(root.path, true, in), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, FabricConfigApplyAccelIdOutOfRangeInval) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_ppod_config_t{};
  in.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  in.mask = AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID;
  in.data.accelerator_id = 1024;  // valid range is 0..1023
  EXPECT_EQ(fabric_ualink::apply_ppod_config_at(root.path, true, in), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, FabricConfigApplyLocalAccelIdOutOfRangeInval) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_ppod_config_t{};
  in.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  in.mask = AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS;
  in.data.local_accelerator_count = 1;
  in.data.local_accelerators[0] = 1024;  // valid range is 0..1023
  EXPECT_EQ(fabric_ualink::apply_ppod_config_at(root.path, true, in), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, FabricConfigApplyVpodActiveAccelOutOfRangeInval) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_vpod_config_t{};
  in.version = AMDSMI_FABRIC_VPOD_CONFIG_V1;
  in.mask = AMDSMI_FABRIC_VPOD_FIELD_VPOD_ACTIVE_ACCELS;
  in.data.vpod_active_accelerators[0] = 1024;  // valid range is 0..1023
  in.data.vpod_active_accelerators[1] = kU32Sentinel;
  EXPECT_EQ(fabric_ualink::apply_vpod_config_at(root.path, true, in), AMDSMI_STATUS_INVAL);

  // A value stranded past the first sentinel would be silently dropped by the
  // serializer; reject the request instead of truncating the caller's list.
  auto gap = amdsmi_fabric_vpod_config_t{};
  gap.version = AMDSMI_FABRIC_VPOD_CONFIG_V1;
  gap.mask = AMDSMI_FABRIC_VPOD_FIELD_VPOD_ACTIVE_ACCELS;
  for (auto& id : gap.data.vpod_active_accelerators) {
    id = kU32Sentinel;
  }
  gap.data.vpod_active_accelerators[0] = 3;
  gap.data.vpod_active_accelerators[1] = kU32Sentinel;  // terminator
  gap.data.vpod_active_accelerators[2] = 5;             // stranded past the terminator
  EXPECT_EQ(fabric_ualink::apply_vpod_config_at(root.path, true, gap), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, FabricConfigApplyVpodIdZeroInval) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_vpod_config_t{};
  in.version = AMDSMI_FABRIC_VPOD_CONFIG_V1;
  in.mask = AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID;
  in.data.vpod_id = 0;  // 0 is not a valid pod ID
  EXPECT_EQ(fabric_ualink::apply_vpod_config_at(root.path, true, in), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, FabricConfigApplyPpodIdZeroInval) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_ppod_config_t{};
  in.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  in.mask = AMDSMI_FABRIC_PPOD_FIELD_PPOD_ID;
  // ppod_id is all-zero by value-init; 0 is not a valid pod ID.
  EXPECT_EQ(fabric_ualink::apply_ppod_config_at(root.path, true, in), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, FabricConfigApplyPpodIdUnsetSentinelInval) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_ppod_config_t{};
  in.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  in.mask = AMDSMI_FABRIC_PPOD_FIELD_PPOD_ID;
  // All-0x99 is the unset-readback sentinel; echoing it back is rejected on write.
  for (auto& byte : in.data.ppod_id) {
    byte = 0x99;
  }
  EXPECT_EQ(fabric_ualink::apply_ppod_config_at(root.path, true, in), AMDSMI_STATUS_INVAL);
}

// Missing write subtree under the root reports NOT_SUPPORTED rather than crashing.
TEST(GpuUnit, FabricConfigQueryMissingSubdirNotSupported) {
  auto tmpl = std::string("/tmp/amdsmi_fabric_empty_XXXXXX");
  ASSERT_NE(::mkdtemp(tmpl.data()), nullptr) << "mkdtemp failed";
  const auto empty_root = std::string(tmpl);

  auto cfg = amdsmi_fabric_ppod_config_t{};
  cfg.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  cfg.mask = AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID;
  EXPECT_EQ(fabric_ualink::query_ppod_config_at(empty_root, true, cfg),
            AMDSMI_STATUS_NOT_SUPPORTED);

  auto ec = std::error_code{};
  fs::remove_all(empty_root, ec);
}

// A commit with nothing freshly staged would flush a prior partial write's residue.
TEST(GpuUnit, FabricConfigBareCommitIsRejected) {
  auto root = TempRoot{};
  ASSERT_FALSE(root.path.empty()) << "mkdtemp failed";

  auto in = amdsmi_fabric_ppod_config_t{};
  in.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  in.mask = 0;
  in.commit = true;
  EXPECT_EQ(fabric_ualink::apply_ppod_config_at(root.path, true, in), AMDSMI_STATUS_INVAL);
}

// The unified getter reads the flat surface, which the driver syncs on commit. Fields must
// come back even when the setup/config/stations write subtrees are absent.
TEST(GpuUnit, FabricConfigQueryFlatSurfaceWithoutWriteSubtrees) {
  auto tmpl = std::string("/tmp/amdsmi_fabric_flat_XXXXXX");
  ASSERT_NE(::mkdtemp(tmpl.data()), nullptr) << "mkdtemp failed";
  const auto flat_root = std::string(tmpl);

  {
    auto accel_id_file = std::ofstream(fs::path(flat_root) / std::string(kUALOE_ACCEL_ID));
    accel_id_file << "42\n";
    auto ppod_size_file = std::ofstream(fs::path(flat_root) / std::string(kUALOE_PPOD_SIZE));
    ppod_size_file << "8\n";
  }

  auto cfg = amdsmi_fabric_ppod_config_t{};
  cfg.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  cfg.mask = (AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID | AMDSMI_FABRIC_PPOD_FIELD_PPOD_SIZE |
              AMDSMI_FABRIC_PPOD_FIELD_BANDWIDTH);
  EXPECT_EQ(fabric_ualink::query_ppod_config_at(flat_root, true, cfg, kUALOE_UALINK_FLAT_SUBDIR),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(cfg.mask, static_cast<uint32_t>(AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID |
                                            AMDSMI_FABRIC_PPOD_FIELD_PPOD_SIZE));
  EXPECT_EQ(cfg.data.accelerator_id, 42u);
  EXPECT_EQ(cfg.data.ppod_size, 8u);
  EXPECT_EQ(cfg.data.bandwidth, kU32Sentinel);

  auto ec = std::error_code{};
  fs::remove_all(flat_root, ec);
}

}  // namespace

}  // namespace amd::smi
