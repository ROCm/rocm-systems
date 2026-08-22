/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Hardware-free guard on the amdsmi_fabric_info_t versioned-union contract. Binaries
// already linked against the version 1 layout allocated only through the v1 union member,
// so both the field offsets and the write bound below are load-bearing: breaking either
// corrupts caller memory rather than producing a compile error on their side.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"

namespace amd::smi {

namespace {

// The shipped version 1 struct, byte for byte
constexpr auto kV1Size = std::size_t{244};

// Where the union starts: sizeof(amdsmi_bdf_t) + sizeof(fabric_version)
constexpr auto kUnionOffset = std::size_t{12};

// One past the last byte a version 1 caller allocated for the payload
constexpr auto kV1WriteBound = (kUnionOffset + kV1Size);

// What a version 1 caller allocated in total: kV1WriteBound plus reserved[15]
constexpr auto kV1CallerSize = std::size_t{320};

constexpr auto kGuardByte = std::uint8_t{0xCC};

static_assert(sizeof(amdsmi_fabric_info_v1_t) == kV1Size, "version 1 layout is frozen");
static_assert(offsetof(amdsmi_fabric_info_t, fabric_info) == kUnionOffset,
              "union offset is frozen");

static_assert(offsetof(amdsmi_fabric_info_v1_t, accelerator_id) == 0);
static_assert(offsetof(amdsmi_fabric_info_v1_t, fabric_type) == 4);
static_assert(offsetof(amdsmi_fabric_info_v1_t, bandwidth) == 8);
static_assert(offsetof(amdsmi_fabric_info_v1_t, latency) == 12);
static_assert(offsetof(amdsmi_fabric_info_v1_t, ppod_id) == 16);
static_assert(offsetof(amdsmi_fabric_info_v1_t, ppod_size) == 32);
static_assert(offsetof(amdsmi_fabric_info_v1_t, vpod_id) == 36);
static_assert(offsetof(amdsmi_fabric_info_v1_t, vpod_size) == 40);
static_assert(offsetof(amdsmi_fabric_info_v1_t, vpod_active_accelerators) == 44);
static_assert(offsetof(amdsmi_fabric_info_v1_t, local_accelerators) == 172);
static_assert(offsetof(amdsmi_fabric_info_v1_t, addr_mode) == 236);
static_assert(offsetof(amdsmi_fabric_info_v1_t, accel_state) == 240);

// Populate every field with a distinguishable value so a dropped or mis-mapped
// member shows up as a mismatch rather than as a coincidental zero
auto make_populated_v2() -> amdsmi_fabric_info_v2_t {
  auto source = amdsmi_fabric_info_v2_t{};
  source.fabric_type = AMDSMI_FABRIC_TYPE_UALINK;
  source.accel_state = AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE;

  source.ppod.accelerator_id = 11;
  for (auto i = std::size_t{0}; i < AMDSMI_MAX_UUID_ELEMENTS; ++i) {
    source.ppod.ppod_id[i] = static_cast<std::uint8_t>(i + 1);
  }
  source.ppod.ppod_size = 22;
  for (auto i = std::size_t{0}; i < AMDSMI_FABRIC_MAX_LOCAL_GPUS; ++i) {
    source.ppod.local_accelerators[i] = static_cast<std::uint32_t>(100 + i);
  }
  source.ppod.local_accelerator_count = AMDSMI_FABRIC_MAX_LOCAL_GPUS;
  source.ppod.bandwidth = 33;
  source.ppod.latency = 44;

  source.vpod.vpod_id = 55;
  source.vpod.vpod_size = 66;
  for (auto i = std::size_t{0}; i < AMDSMI_FABRIC_ACTIVE_ACCELERATORS_BITMAP_SIZE; ++i) {
    source.vpod.vpod_active_accelerators[i] = static_cast<std::uint32_t>(200 + i);
  }
  source.vpod.addr_mode = AMDSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_ALIASING;

  source.station.station_flags = 77;
  source.station.num_stations = 8;
  for (auto i = std::size_t{0}; i < AMDSMI_FABRIC_MAX_BITMAP_SIZE; ++i) {
    source.station.lane_en_bitmap[i] = static_cast<std::uint8_t>(0xA0 + (i % 16));
  }

  source.ppod_mask = 0x0000003Fu;
  source.vpod_mask = 0x0000000Fu;
  source.station_mask = 0x00000007u;

  return source;
}

// Stands in for a caller compiled against the version 1 header: only kV1CallerSize bytes
// are theirs, and the rest of the buffer catches an out-of-bounds write
class LegacyCallerBuffer {
 public:
  LegacyCallerBuffer() { storage_.fill(kGuardByte); }

  auto info() -> amdsmi_fabric_info_t& {
    return *reinterpret_cast<amdsmi_fabric_info_t*>(storage_.data());
  }

  auto is_untouched_from(std::size_t offset) const -> bool {
    for (auto i = offset; i < storage_.size(); ++i) {
      if (storage_[i] != kGuardByte) {
        return false;
      }
    }
    return true;
  }

 private:
  alignas(amdsmi_fabric_info_t) std::array<std::uint8_t, sizeof(amdsmi_fabric_info_t)> storage_{};
};

}  // namespace

TEST(GpuUnit, FabricInfoLayoutVersionOneRequestWritesNothingPastTheVersionOneMember) {
  auto buffer = LegacyCallerBuffer{};
  buffer.info().fabric_version = AMDSMI_FABRIC_INFO_VERSION_1;

  gpu_device::details::publish_fabric_info(amdsmi_bdf_t{}, make_populated_v2(),
                                           AMDSMI_FABRIC_INFO_VERSION_1, buffer.info());

  EXPECT_TRUE(buffer.is_untouched_from(kV1WriteBound));
  EXPECT_LE(kV1WriteBound, kV1CallerSize);
}

TEST(GpuUnit, FabricInfoLayoutUnrecognizedVersionDegradesToVersionOne) {
  const auto unrecognized_versions = std::array<std::uint32_t, 5>{
      1u, 2u, 3u, std::numeric_limits<std::uint32_t>::max(), 0xFAB10003u};

  for (const auto requested : unrecognized_versions) {
    auto buffer = LegacyCallerBuffer{};
    buffer.info().fabric_version = requested;

    gpu_device::details::publish_fabric_info(amdsmi_bdf_t{}, make_populated_v2(), requested,
                                             buffer.info());

    EXPECT_TRUE(buffer.is_untouched_from(kV1WriteBound)) << "requested version " << requested;
    EXPECT_NE(buffer.info().fabric_version, AMDSMI_FABRIC_INFO_VERSION_2)
        << "requested version " << requested;
  }
}

TEST(GpuUnit, FabricInfoLayoutVersionTwoRequestFillsTheNestedMember) {
  const auto source = make_populated_v2();

  auto destination = amdsmi_fabric_info_t{};
  gpu_device::details::publish_fabric_info(amdsmi_bdf_t{}, source, AMDSMI_FABRIC_INFO_VERSION_2,
                                           destination);

  EXPECT_EQ(destination.fabric_version, static_cast<std::uint32_t>(AMDSMI_FABRIC_INFO_VERSION_2));

  // Both operands descend from the same value-initialized aggregate, so the padding bytes agree
  // and a byte compare is exactly what "copied verbatim" means here
  // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison)
  EXPECT_EQ(std::memcmp(&destination.fabric_info.v2, &source, sizeof(source)), 0);
}

TEST(GpuUnit, FabricInfoLayoutFlattenMapsEveryVersionOneField) {
  const auto source = make_populated_v2();

  auto destination = amdsmi_fabric_info_v1_t{};
  gpu_device::details::flatten_v2_to_v1(source, destination);

  EXPECT_EQ(destination.accelerator_id, source.ppod.accelerator_id);
  EXPECT_EQ(destination.fabric_type, source.fabric_type);
  EXPECT_EQ(destination.bandwidth, source.ppod.bandwidth);
  EXPECT_EQ(destination.latency, source.ppod.latency);
  EXPECT_EQ(std::memcmp(destination.ppod_id, source.ppod.ppod_id, sizeof(destination.ppod_id)), 0);
  EXPECT_EQ(destination.ppod_size, source.ppod.ppod_size);
  EXPECT_EQ(destination.vpod_id, source.vpod.vpod_id);
  EXPECT_EQ(destination.vpod_size, source.vpod.vpod_size);
  EXPECT_EQ(std::memcmp(destination.vpod_active_accelerators, source.vpod.vpod_active_accelerators,
                        sizeof(destination.vpod_active_accelerators)),
            0);
  EXPECT_EQ(std::memcmp(destination.local_accelerators, source.ppod.local_accelerators,
                        sizeof(destination.local_accelerators)),
            0);
  EXPECT_EQ(destination.addr_mode, source.vpod.addr_mode);
  EXPECT_EQ(destination.accel_state, source.accel_state);
}

}  // namespace amd::smi
