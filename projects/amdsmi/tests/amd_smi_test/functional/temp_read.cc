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

#include <cstdint>

#include <iostream>
#include <string>
#include <map>

#include <gtest/gtest.h>
#include "amd_smi/amdsmi.h"
#include "temp_read.h"


static const std::map<uint32_t, std::string> kTempSensorNameMap = {
    {AMDSMI_TEMPERATURE_TYPE_VRAM, "Memory"},
    {AMDSMI_TEMPERATURE_TYPE_HOTSPOT, "Hotspot"},
    {AMDSMI_TEMPERATURE_TYPE_JUNCTION, "Junction"},
    {AMDSMI_TEMPERATURE_TYPE_EDGE, "Edge"},
    {AMDSMI_TEMPERATURE_TYPE_HBM_0, "HBM_0"},
    {AMDSMI_TEMPERATURE_TYPE_HBM_1, "HBM_1"},
    {AMDSMI_TEMPERATURE_TYPE_HBM_2, "HBM_2"},
    {AMDSMI_TEMPERATURE_TYPE_HBM_3, "HBM_3"},
    {AMDSMI_TEMPERATURE_TYPE_PLX, "PLX"},
    
    // GPU Board Node Temperature Types (100-149)
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_RETIMER_X, "GPU Board Node Retimer X"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_IBC, "GPU Board Node OAM X IBC"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_IBC_2, "GPU Board Node OAM X IBC 2"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_VDD18_VR, "GPU Board Node OAM X VDD18 VR"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_B_VR, "GPU Board Node OAM X 04 HBM B VR"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_D_VR, "GPU Board Node OAM X 04 HBM D VR"},
    
    // GPU Board VR Temperature Types (150-199)
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD0, "GPU Board VDDCR VDD0"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD1, "GPU Board VDDCR VDD1"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD2, "GPU Board VDDCR VDD2"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD3, "GPU Board VDDCR VDD3"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOC_A, "GPU Board VDDCR SOC A"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOC_C, "GPU Board VDDCR SOC C"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOCIO_A, "GPU Board VDDCR SOCIO A"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOCIO_C, "GPU Board VDDCR SOCIO C"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDD_085_HBM, "GPU Board VDD 085 HBM"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_11_HBM_B, "GPU Board VDDCR 11 HBM B"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_11_HBM_D, "GPU Board VDDCR 11 HBM D"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDD_USR, "GPU Board VDD USR"},
    {AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_11_E32, "GPU Board VDDIO 11 E32"},
    
    // Baseboard System Temperature Types (200+)
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA, "Baseboard UBB FPGA"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FRONT, "Baseboard UBB Front"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_BACK, "Baseboard UBB Back"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_OAM7, "Baseboard UBB OAM7"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_IBC, "Baseboard UBB IBC"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_UFPGA, "Baseboard UBB UFPGA"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_OAM1, "Baseboard UBB OAM1"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_0_1_HSC, "Baseboard OAM 0-1 HSC"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_2_3_HSC, "Baseboard OAM 2-3 HSC"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_4_5_HSC, "Baseboard OAM 4-5 HSC"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_6_7_HSC, "Baseboard OAM 6-7 HSC"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA_0V72_VR, "Baseboard UBB FPGA 0V72 VR"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA_3V3_VR, "Baseboard UBB FPGA 3V3 VR"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_0_1_2_3_1V2_VR, "Baseboard Retimer 0-1-2-3 1V2 VR"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_4_5_6_7_1V2_VR, "Baseboard Retimer 4-5-6-7 1V2 VR"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_0_1_0V9_VR, "Baseboard Retimer 0-1 0V9 VR"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_4_5_0V9_VR, "Baseboard Retimer 4-5 0V9 VR"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_2_3_0V9_VR, "Baseboard Retimer 2-3 0V9 VR"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_6_7_0V9_VR, "Baseboard Retimer 6-7 0V9 VR"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_0_1_2_3_3V3_VR, "Baseboard OAM 0-1-2-3 3V3 VR"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_4_5_6_7_3V3_VR, "Baseboard OAM 4-5-6-7 3V3 VR"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_IBC_HSC, "Baseboard IBC HSC"},
    {AMDSMI_TEMPERATURE_TYPE_BASEBOARD_IBC, "Baseboard IBC"}
};
TestTempRead::TestTempRead() : TestBase() {
  set_title("AMDSMI Temp Read Test");
  set_description("The Temperature Read tests verifies that the temperature "
                   "monitors can be read properly.");
}

TestTempRead::~TestTempRead(void) {
}

void TestTempRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestTempRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestTempRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestTempRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestTempRead::Run(void) {
  amdsmi_status_t err;
  int64_t val_i64;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  uint32_t type(0);
  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      PrintDeviceHeader(processor_handles_[i]);

      auto print_temp_metric = [&](amdsmi_temperature_metric_t met,
                                                          std::string label) {
        err =  amdsmi_get_temp_metric(processor_handles_[i], static_cast<amdsmi_temperature_type_t>(type), met, &val_i64);

        if (err != AMDSMI_STATUS_SUCCESS) {
          if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
            IF_VERB(STANDARD) {
              std::cout << "\t**" << label << ": " <<
                                 "Not supported on this machine" << std::endl;
            }

            // Verify api support checking functionality is working
            err =  amdsmi_get_temp_metric(processor_handles_[i],  static_cast<amdsmi_temperature_type_t>(type), met, nullptr);
            ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
            return;
          } else {
            CHK_ERR_ASRT(err)
          }
        }
        // Verify api support checking functionality is working
        err =  amdsmi_get_temp_metric(processor_handles_[i],  static_cast<amdsmi_temperature_type_t>(type), met, nullptr);
        ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

        IF_VERB(STANDARD) {
          std::cout << "\t**" << label << ": " << val_i64 << "C" << std::endl;
        }
      };
      for (type = AMDSMI_TEMPERATURE_TYPE_FIRST; type <= AMDSMI_TEMPERATURE_TYPE__MAX; ++type) {
        if (kTempSensorNameMap.find(type) == kTempSensorNameMap.end()) {
          continue;
        }
        IF_VERB(STANDARD) {
          std::cout << "\t** **********" << kTempSensorNameMap.at(type) <<
                                        " Temperatures **********" << std::endl;
        }
        print_temp_metric(AMDSMI_TEMP_CURRENT, "Current Temp.");
        print_temp_metric(AMDSMI_TEMP_MAX, "Temperature max value");
        print_temp_metric(AMDSMI_TEMP_MIN, "Temperature min value");
        print_temp_metric(AMDSMI_TEMP_MAX_HYST,
                                  "Temperature hysteresis value for max limit");
        print_temp_metric(AMDSMI_TEMP_MIN_HYST,
                                  "Temperature hysteresis value for min limit");
        print_temp_metric(AMDSMI_TEMP_CRITICAL, "Temperature critical max value");
        print_temp_metric(AMDSMI_TEMP_CRITICAL_HYST,
                             "Temperature hysteresis value for critical limit");
        print_temp_metric(AMDSMI_TEMP_EMERGENCY,
                                             "Temperature emergency max value");
        print_temp_metric(AMDSMI_TEMP_EMERGENCY_HYST,
                            "Temperature hysteresis value for emergency limit");
        print_temp_metric(AMDSMI_TEMP_CRIT_MIN, "Temperature critical min value");
        print_temp_metric(AMDSMI_TEMP_CRIT_MIN_HYST,
                         "Temperature hysteresis value for critical min value");
        print_temp_metric(AMDSMI_TEMP_OFFSET, "Temperature offset");
        print_temp_metric(AMDSMI_TEMP_LOWEST, "Historical minimum temperature");
        print_temp_metric(AMDSMI_TEMP_HIGHEST, "Historical maximum temperature");
      }
    }
  }  // x
}
