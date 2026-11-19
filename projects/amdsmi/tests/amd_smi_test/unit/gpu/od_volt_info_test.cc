// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The overdrive getters copy rocm_smi's structs into the public ones a field at a time: rocm_smi's
// ranges have no reserved words, so its structs are smaller than the public ones. A fake
// pp_od_clk_voltage drives the real public API here, so no GPU is needed.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "rocm_smi/rocm_smi.h"
#include "test_fixture_utils.h"

namespace {

constexpr uint64_t kMHz = 1000000;
constexpr uint64_t kUnavailable = std::numeric_limits<uint64_t>::max();
constexpr unsigned char kPoison = 0xA5;

// An MI350X: gfx and fabric clock ranges, no memory clock range and no OD_RANGE.
constexpr char kMi350xOdClkVoltage[] =
    "OD_SCLK:\n"
    "0: 500Mhz\n"
    "1: 2200Mhz\n"
    "OD_FCLK:\n"
    "0: 1250Mhz\n"
    "1: 1500Mhz\n";

// Navi 1x, in the driver's format (navi10_ppt.c), with OD_RANGE limits.
constexpr char kNavi1xOdClkVoltage[] =
    "OD_SCLK:\n"
    "0: 800Mhz\n"
    "1: 2150Mhz\n"
    "OD_MCLK:\n"
    "1: 875MHz\n"
    "OD_VDDC_CURVE:\n"
    "0: 800MHz 711mV\n"
    "1: 1475MHz 812mV\n"
    "2: 2150MHz 1203mV\n"
    "OD_RANGE:\n"
    "SCLK:     800Mhz       2150Mhz\n"
    "MCLK:     625Mhz        950Mhz\n"
    "VDDC_CURVE_SCLK[0]:     800Mhz       2150Mhz\n"
    "VDDC_CURVE_VOLT[0]:     750mV        1200mV\n"
    "VDDC_CURVE_SCLK[1]:     800Mhz       2150Mhz\n"
    "VDDC_CURVE_VOLT[1]:     750mV        1200mV\n"
    "VDDC_CURVE_SCLK[2]:     800Mhz       2150Mhz\n"
    "VDDC_CURVE_VOLT[2]:     750mV        1200mV\n";

// A GPU whose only sysfs file is pp_od_clk_voltage, or that has no overdrive at all when given
// nullptr, registered with RocmSMI and AMDSmiSystem for the life of the object so the public API
// can reach it.
class FakeOverDriveGpu {
 public:
  explicit FakeOverDriveGpu(const char* od_clk_voltage)
      : device_(tree_.root() + "/card0"),
        gpu_device_(device_.index(), "card0", amdsmi_bdf_t{}, drm_),
        registration_(&gpu_device_) {
    if (od_clk_voltage != nullptr) {
      tree_.WriteFile("card0/device/pp_od_clk_voltage", od_clk_voltage);
    }
  }

  amdsmi_processor_handle handle() const { return registration_.handle(); }
  uint32_t rsmi_index() const { return device_.index(); }

 private:
  ScopedAmdsmiInit init_;
  FakeSysfsTree tree_;
  ScopedRocmSmiDevice device_;
  amd::smi::AMDSmiDrm drm_;
  amd::smi::AMDSmiGPUDevice gpu_device_;
  amd::smi::testing::ScopedProcessorRegistration registration_;
};

void ExpectRange(const amdsmi_range_t& range, const rsmi_range_t& want, const char* name) {
  SCOPED_TRACE(name);
  EXPECT_EQ(range.lower_bound, want.lower_bound);
  EXPECT_EQ(range.upper_bound, want.upper_bound);
  EXPECT_EQ(range.reserved[0], 0u);
  EXPECT_EQ(range.reserved[1], 0u);
}

// Reads the GPU through the public API and through rocm_smi, and checks that every public field
// holds rocm_smi's value for it. Returns the public struct.
amdsmi_od_volt_freq_data_t ReadAndCompare(const FakeOverDriveGpu& gpu) {
  rsmi_od_volt_freq_data_t want{};
  EXPECT_EQ(rsmi_dev_od_volt_info_get(gpu.rsmi_index(), &want), RSMI_STATUS_SUCCESS);

  amdsmi_od_volt_freq_data_t odv;
  std::memset(&odv, kPoison, sizeof(odv));
  EXPECT_EQ(amdsmi_get_gpu_od_volt_info(gpu.handle(), &odv), AMDSMI_STATUS_SUCCESS);

  ExpectRange(odv.curr_sclk_range, want.curr_sclk_range, "curr_sclk_range");
  ExpectRange(odv.curr_mclk_range, want.curr_mclk_range, "curr_mclk_range");
  ExpectRange(odv.curr_fclk_range, want.curr_fclk_range, "curr_fclk_range");
  ExpectRange(odv.sclk_freq_limits, want.sclk_freq_limits, "sclk_freq_limits");
  ExpectRange(odv.mclk_freq_limits, want.mclk_freq_limits, "mclk_freq_limits");
  ExpectRange(odv.fclk_freq_limits, want.fclk_freq_limits, "fclk_freq_limits");
  for (uint32_t i = 0; i < AMDSMI_NUM_VOLTAGE_CURVE_POINTS; ++i) {
    EXPECT_EQ(odv.curve.vc_points[i].frequency, want.curve.vc_points[i].frequency) << i;
    EXPECT_EQ(odv.curve.vc_points[i].voltage, want.curve.vc_points[i].voltage) << i;
  }
  EXPECT_EQ(odv.num_regions, want.num_regions);
  return odv;
}

}  // namespace

// Each clock range lands in its own field: an MI350X has no memory clock range to report.
TEST(GpuUnit, OdVoltInfoKeepsEachClockRangeInItsField) {
  FakeOverDriveGpu gpu(kMi350xOdClkVoltage);
  const amdsmi_od_volt_freq_data_t odv = ReadAndCompare(gpu);

  EXPECT_EQ(odv.curr_sclk_range.lower_bound, 500 * kMHz);
  EXPECT_EQ(odv.curr_sclk_range.upper_bound, 2200 * kMHz);
  EXPECT_EQ(odv.curr_mclk_range.lower_bound, kUnavailable);
  EXPECT_EQ(odv.curr_mclk_range.upper_bound, kUnavailable);
  EXPECT_EQ(odv.curr_fclk_range.lower_bound, 1250 * kMHz);
  EXPECT_EQ(odv.curr_fclk_range.upper_bound, 1500 * kMHz);
  EXPECT_EQ(odv.num_regions, 0u);
}

// The OD_RANGE clock limits land in the limit fields.
TEST(GpuUnit, OdVoltInfoReportsClockLimits) {
  FakeOverDriveGpu gpu(kNavi1xOdClkVoltage);
  const amdsmi_od_volt_freq_data_t odv = ReadAndCompare(gpu);

  EXPECT_EQ(odv.sclk_freq_limits.lower_bound, 800 * kMHz);
  EXPECT_EQ(odv.sclk_freq_limits.upper_bound, 2150 * kMHz);
  EXPECT_EQ(odv.mclk_freq_limits.lower_bound, 625 * kMHz);
  EXPECT_EQ(odv.mclk_freq_limits.upper_bound, 950 * kMHz);
}

// A null pointer still reaches rocm_smi, which tells a GPU without overdrive from bad input.
TEST(GpuUnit, OdVoltInfoNullPointerStatusDependsOnSupport) {
  {
    FakeOverDriveGpu gpu(kMi350xOdClkVoltage);
    EXPECT_EQ(amdsmi_get_gpu_od_volt_info(gpu.handle(), nullptr), AMDSMI_STATUS_INVAL);
  }
  FakeOverDriveGpu gpu(nullptr);
  EXPECT_EQ(amdsmi_get_gpu_od_volt_info(gpu.handle(), nullptr), AMDSMI_STATUS_NOT_SUPPORTED);
}

// rocm_smi reports no curve regions on any GPU, so nothing is copied into the caller's buffer,
// whatever count the caller passes.
TEST(GpuUnit, OdVoltCurveRegionsLeavesBufferAloneWhenNoneAreReported) {
  FakeOverDriveGpu gpu(kNavi1xOdClkVoltage);

  for (const uint32_t asked : {2u, std::numeric_limits<uint32_t>::max()}) {
    SCOPED_TRACE(asked);
    amdsmi_freq_volt_region_t regions[2];
    std::memset(regions, kPoison, sizeof(regions));
    uint32_t num_regions = asked;
    EXPECT_EQ(amdsmi_get_gpu_od_volt_curve_regions(gpu.handle(), &num_regions, regions),
              AMDSMI_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(num_regions, 0u);

    const auto* bytes = reinterpret_cast<const unsigned char*>(regions);
    for (size_t i = 0; i < sizeof(regions); ++i) {
      ASSERT_EQ(bytes[i], kPoison) << "byte " << i << " was written";
    }
  }
}

// rocm_smi still checks the arguments: a null count or buffer asks whether the call is supported
// (INVAL with overdrive, NOT_SUPPORTED without), and a count of 0 is bad input. A call that fails
// before rocm_smi sets a count leaves the caller's count alone.
TEST(GpuUnit, OdVoltCurveRegionsArgumentStatusComesFromRocmSmi) {
  for (const char* od_clk_voltage : {kNavi1xOdClkVoltage, static_cast<const char*>(nullptr)}) {
    SCOPED_TRACE(od_clk_voltage != nullptr ? "overdrive" : "no overdrive");
    const amdsmi_status_t null_status =
        od_clk_voltage != nullptr ? AMDSMI_STATUS_INVAL : AMDSMI_STATUS_NOT_SUPPORTED;
    FakeOverDriveGpu gpu(od_clk_voltage);
    amdsmi_freq_volt_region_t region{};
    uint32_t count = 100;

    EXPECT_EQ(amdsmi_get_gpu_od_volt_curve_regions(gpu.handle(), nullptr, &region), null_status);
    EXPECT_EQ(amdsmi_get_gpu_od_volt_curve_regions(gpu.handle(), &count, nullptr), null_status);
    EXPECT_EQ(count, 100u);
    count = 0;
    EXPECT_EQ(amdsmi_get_gpu_od_volt_curve_regions(gpu.handle(), &count, &region),
              AMDSMI_STATUS_INVAL);
  }
}
