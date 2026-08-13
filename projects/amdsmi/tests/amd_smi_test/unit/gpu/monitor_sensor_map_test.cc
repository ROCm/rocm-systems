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

// TODO(SPLIT-FROM-AILITOOLS-297): slightly related, coverage allowed us
// to proc the behavior during the coverage tooling ticket

// AILITOOLS-299: Monitor's sensor lookups call std::map::at() without checking
// that the key exists, so an out-of-range sensor type escapes as
// std::out_of_range and every rsmi_* entry point converts it (via TRY/CATCH ->
// handleException) into RSMI_STATUS_INTERNAL_EXCEPTION / AMDSMI_STATUS_INTERNAL_EXCEPTION
// instead of a meaningful NOT_SUPPORTED.
//
// These are characterization tests: as written they assert the CURRENT
// (defective) behavior so the bug is pinned down and reproducible without a GPU.
// When the .count() guards are restored in rocm_smi_monitor.cc, flip
// kSensorMapGuardsLanded to true and the same tests assert the fixed behavior.
// Leaving the flag false while the guards are in place makes these tests FAIL,
// which is the intended signal that the fix landed.
//
// Driver-free: a temp directory stands in for an amdgpu hwmon node.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_common.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_monitor.h"
#include "rocm_smi/rocm_smi_utils.h"

namespace {

// Set to true in the same change that restores the guards.
constexpr bool kSensorMapGuardsLanded = true;  // This is how the code responded before the fixes
                                               // Release builds allowed try/catches to
                                               // return AMDSMI_STATUS_INTERNAL_EXCEPTION
                                               // instead of properly handling

// amdsmi_temperature_type_t pads its group sentinels to round numbers (249),
// while rsmi_temperature_type_t aliases BASEBOARD_LAST to the last real member
// (BASEBOARD_IBC, 222). amdsmi_get_temp_metric forwards the value unchanged, so
// 249 reaches the rsmi layer as a type that was never seeded into the map.
constexpr auto kPaddedBaseboardLast =
    static_cast<rsmi_temperature_type_t>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_LAST);

// index_temp_type_map_ is only populated for file indices inside the four
// group ranges, leaving a hole between GENERAL_LAST (7) and GPUBOARD_NODE_FIRST (100).
constexpr uint64_t kUnmappedTempFileIndex = RSMI_TEMP_TYPE_GENERAL_LAST + 1;

// Mirrors the range predicate in setTempSensorLabelMap(). Kept as an independent
// copy on purpose: if the production ranges change, this test should notice.
bool IsSeededTempFileIndex(uint64_t i) {
  return (i >= 1 && i <= RSMI_TEMP_TYPE_GENERAL_LAST) ||
         (i >= RSMI_TEMP_TYPE_GPUBOARD_NODE_FIRST && i <= RSMI_TEMP_TYPE_GPUBOARD_NODE_LAST) ||
         (i >= RSMI_TEMP_TYPE_GPUBOARD_VR_FIRST && i <= RSMI_TEMP_TYPE_GPUBOARD_LAST) ||
         (i >= RSMI_TEMP_TYPE_BASEBOARD_FIRST && i <= RSMI_TEMP_TYPE_BASEBOARD_LAST);
}

// The only labels kTempSensorNameMap recognizes, and the files the fixture writes.
const std::map<uint64_t, rsmi_temperature_type_t> kLabeledTempFileIndices = {
    {1, RSMI_TEMP_TYPE_EDGE},
    {2, RSMI_TEMP_TYPE_JUNCTION},
    {3, RSMI_TEMP_TYPE_MEMORY},
};

// Stands in for an amdgpu hwmon node. Deliberately mirrors real hardware:
// temp1..temp3 carry labels, and only in0 does -- which is what leaves
// VDDBOARD out of volt_type_index_map_ entirely.
class FakeHwmonNode {
 public:
  FakeHwmonNode() {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    dir_ =
        std::filesystem::temp_directory_path() / ("amdsmi_hwmon_" + std::to_string(getpid()) + "_" +
                                                  (info != nullptr ? info->name() : "anon"));
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
    if (!std::filesystem::create_directories(dir_, ec)) {
      return;
    }
    ok_ = Write("name", "amdgpu") && Write("temp1_label", "edge") &&
          Write("temp2_label", "junction") && Write("temp3_label", "mem") &&
          Write("in0_label", "vddgfx");
    if (!ok_) {
      return;
    }
    mon_ = std::make_unique<amd::smi::Monitor>(dir_.string(), &env_vars_);
    ok_ = (mon_->setTempSensorLabelMap() == 0) && (mon_->setVoltSensorLabelMap() == 0);
  }

  ~FakeHwmonNode() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }

  FakeHwmonNode(const FakeHwmonNode&) = delete;
  FakeHwmonNode& operator=(const FakeHwmonNode&) = delete;

  bool ok() const { return ok_; }
  amd::smi::Monitor* monitor() const { return mon_.get(); }

 private:
  bool Write(const std::string& name, const std::string& contents) {
    std::ofstream file(dir_ / name);
    if (!file.is_open()) {
      return false;
    }
    file << contents << "\n";
    return file.good();
  }

  std::filesystem::path dir_;
  ::RocmSMI_env_vars env_vars_{};
  std::unique_ptr<amd::smi::Monitor> mon_;
  bool ok_ = false;
};

// The root cause, asserted directly: the two enums disagree on BASEBOARD_LAST.
// Aligning them is an alternative fix, and would fail here on purpose.
TEST(GpuUnit, MonitorTempSentinelsDivergeBetweenAmdsmiAndRsmi) {
  EXPECT_EQ(static_cast<uint32_t>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_LAST), 249U);
  EXPECT_GT(static_cast<uint32_t>(AMDSMI_TEMPERATURE_TYPE_BASEBOARD_LAST),
            static_cast<uint32_t>(RSMI_TEMP_TYPE_LAST))
      << "amdsmi pads BASEBOARD_LAST past the highest type rsmi ever seeds";
}

TEST(GpuUnit, MonitorGetTempSensorIndexPaddedSentinelThrows) {
  FakeHwmonNode node;
  ASSERT_TRUE(node.ok());

  if (kSensorMapGuardsLanded) {
    EXPECT_EQ(node.monitor()->getTempSensorIndex(kPaddedBaseboardLast),
              static_cast<uint32_t>(RSMI_TEMP_TYPE_INVALID));
  } else {
    EXPECT_THROW(static_cast<void>(node.monitor()->getTempSensorIndex(kPaddedBaseboardLast)),
                 std::out_of_range);
  }
}

// Control: BASEBOARD_IBC (222) IS seeded, so it already returns the INVALID
// sentinel rather than throwing. This is why the CLI reports NOT_SUPPORTED for
// every real baseboard type and INTERNAL_EXCEPTION only for *_LAST.
TEST(GpuUnit, MonitorGetTempSensorIndexSeededTypeReturnsInvalid) {
  FakeHwmonNode node;
  ASSERT_TRUE(node.ok());

  EXPECT_EQ(node.monitor()->getTempSensorIndex(RSMI_TEMP_TYPE_BASEBOARD_IBC),
            static_cast<uint32_t>(RSMI_TEMP_TYPE_INVALID));
  EXPECT_EQ(node.monitor()->getTempSensorIndex(RSMI_TEMP_TYPE_EDGE), 1U);
}

TEST(GpuUnit, MonitorGetTempSensorEnumUnmappedFileIndexThrows) {
  FakeHwmonNode node;
  ASSERT_TRUE(node.ok());

  if (kSensorMapGuardsLanded) {
    EXPECT_EQ(node.monitor()->getTempSensorEnum(kUnmappedTempFileIndex), RSMI_TEMP_TYPE_INVALID);
  } else {
    EXPECT_THROW((void)node.monitor()->getTempSensorEnum(kUnmappedTempFileIndex),
                 std::out_of_range);
  }
}

// Sweeps every temperature file index the library can ask about: labelled ones
// resolve to their type, seeded-but-unlabelled ones resolve to INVALID, and the
// holes between the four groups are where the unguarded .at() escapes.
TEST(GpuUnit, MonitorGetTempSensorEnumCoversEveryFileIndex) {
  FakeHwmonNode node;
  ASSERT_TRUE(node.ok());

  std::size_t labeled = 0;
  std::size_t seeded_unlabeled = 0;
  std::size_t gaps = 0;

  for (uint64_t i = 1; i <= static_cast<uint64_t>(RSMI_TEMP_TYPE_LAST) + 1; ++i) {
    SCOPED_TRACE("temp file index " + std::to_string(i));
    const auto labeled_it = kLabeledTempFileIndices.find(i);

    if (IsSeededTempFileIndex(i)) {
      if (labeled_it != kLabeledTempFileIndices.end()) {
        ++labeled;
        EXPECT_EQ(node.monitor()->getTempSensorEnum(i), labeled_it->second);
      } else {
        ++seeded_unlabeled;
        EXPECT_EQ(node.monitor()->getTempSensorEnum(i), RSMI_TEMP_TYPE_INVALID);
      }
      continue;
    }

    ++gaps;
    if (kSensorMapGuardsLanded) {
      EXPECT_EQ(node.monitor()->getTempSensorEnum(i), RSMI_TEMP_TYPE_INVALID);
    } else {
      EXPECT_THROW(static_cast<void>(node.monitor()->getTempSensorEnum(i)), std::out_of_range);
    }
  }

  EXPECT_EQ(labeled, kLabeledTempFileIndices.size());
  EXPECT_GT(seeded_unlabeled, 0U);
  EXPECT_GT(gaps, 0U) << "no unmapped indices means the bug surface disappeared";
}

// volt_type_index_map_ is never pre-seeded: a type is a key only when its
// in<N>_label file exists. With only in0_label present, VDDBOARD is absent.
TEST(GpuUnit, MonitorGetVoltSensorIndexUnlabeledTypeThrows) {
  FakeHwmonNode node;
  ASSERT_TRUE(node.ok());

  if (kSensorMapGuardsLanded) {
    EXPECT_EQ(node.monitor()->getVoltSensorIndex(RSMI_VOLT_TYPE_VDDBOARD),
              static_cast<uint32_t>(RSMI_VOLT_TYPE_INVALID));
  } else {
    EXPECT_THROW((void)node.monitor()->getVoltSensorIndex(RSMI_VOLT_TYPE_VDDBOARD),
                 std::out_of_range);
  }
}

TEST(GpuUnit, MonitorGetVoltSensorIndexLabeledTypeReturnsIndex) {
  FakeHwmonNode node;
  ASSERT_TRUE(node.ok());

  EXPECT_EQ(node.monitor()->getVoltSensorIndex(RSMI_VOLT_TYPE_VDDGFX), 0U);
}

// getVoltSensorEnum already carries the guard the other three lack; it is the
// precedent for the fix and must keep working.
TEST(GpuUnit, MonitorGetVoltSensorEnumUnknownIndexIsAlreadyGuarded) {
  FakeHwmonNode node;
  ASSERT_TRUE(node.ok());

  EXPECT_NO_THROW({ EXPECT_EQ(node.monitor()->getVoltSensorEnum(1), RSMI_VOLT_TYPE_INVALID); });
}

// End-to-end status translation, mirroring the TRY/CATCH wrapper that every
// rsmi_* entry point uses. This is the AMDSMI_STATUS_INTERNAL_EXCEPTION (16)
// the CLI and Python suites report, reproduced with no GPU present.
TEST(GpuUnit, MonitorMapOutOfRangeSurfacesAsInternalException) {
  FakeHwmonNode node;
  ASSERT_TRUE(node.ok());

  auto caught_status = [&](auto&& call) {
    rsmi_status_t ret = RSMI_STATUS_SUCCESS;
    try {
      call();
    } catch (...) {
      ret = amd::smi::handleException();
    }
    return ret;
  };

  const rsmi_status_t temp_index_ret =
      caught_status([&] { (void)node.monitor()->getTempSensorIndex(kPaddedBaseboardLast); });
  const rsmi_status_t temp_enum_ret =
      caught_status([&] { (void)node.monitor()->getTempSensorEnum(kUnmappedTempFileIndex); });
  const rsmi_status_t volt_index_ret =
      caught_status([&] { (void)node.monitor()->getVoltSensorIndex(RSMI_VOLT_TYPE_VDDBOARD); });

  if (kSensorMapGuardsLanded) {
    EXPECT_EQ(temp_index_ret, RSMI_STATUS_SUCCESS);
    EXPECT_EQ(temp_enum_ret, RSMI_STATUS_SUCCESS);
    EXPECT_EQ(volt_index_ret, RSMI_STATUS_SUCCESS);
  } else {
    EXPECT_EQ(temp_index_ret, RSMI_STATUS_INTERNAL_EXCEPTION);
    EXPECT_EQ(temp_enum_ret, RSMI_STATUS_INTERNAL_EXCEPTION);
    EXPECT_EQ(volt_index_ret, RSMI_STATUS_INTERNAL_EXCEPTION);

    EXPECT_EQ(amd::smi::rsmi_to_amdsmi_status(temp_index_ret), AMDSMI_STATUS_INTERNAL_EXCEPTION);
    EXPECT_EQ(amd::smi::rsmi_to_amdsmi_status(temp_enum_ret), AMDSMI_STATUS_INTERNAL_EXCEPTION);
    EXPECT_EQ(amd::smi::rsmi_to_amdsmi_status(volt_index_ret), AMDSMI_STATUS_INTERNAL_EXCEPTION);
  }
}

}  // namespace
