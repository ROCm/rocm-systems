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

#ifndef ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_BOARD_TEMP_H_
#define ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_BOARD_TEMP_H_

#include "rocm_smi/rocm_smi.h"


// Headers from the driver
namespace amd::smi {
enum amdgpu_vr_temp {
        AMDGPU_VDDCR_VDD0_TEMP,
        AMDGPU_VDDCR_VDD1_TEMP,
        AMDGPU_VDDCR_VDD2_TEMP,
        AMDGPU_VDDCR_VDD3_TEMP,
        AMDGPU_VDDCR_SOC_A_TEMP,
        AMDGPU_VDDCR_SOC_C_TEMP,
        AMDGPU_VDDCR_SOCIO_A_TEMP,
        AMDGPU_VDDCR_SOCIO_C_TEMP,
        AMDGPU_VDD_085_HBM_TEMP,
        AMDGPU_VDDCR_11_HBM_B_TEMP,
        AMDGPU_VDDCR_11_HBM_D_TEMP,
        AMDGPU_VDD_USR_TEMP,
        AMDGPU_VDDIO_11_E32_TEMP,
        AMDGPU_VR_MAX_TEMP_ENTRIES,
};

enum amdgpu_system_temp {
        AMDGPU_UBB_FPGA_TEMP,
        AMDGPU_UBB_FRONT_TEMP,
        AMDGPU_UBB_BACK_TEMP,
        AMDGPU_UBB_OAM7_TEMP,
        AMDGPU_UBB_IBC_TEMP,
        AMDGPU_UBB_UFPGA_TEMP,
        AMDGPU_UBB_OAM1_TEMP,
        AMDGPU_OAM_0_1_HSC_TEMP,
        AMDGPU_OAM_2_3_HSC_TEMP,
        AMDGPU_OAM_4_5_HSC_TEMP,
        AMDGPU_OAM_6_7_HSC_TEMP,
        AMDGPU_UBB_FPGA_0V72_VR_TEMP,
        AMDGPU_UBB_FPGA_3V3_VR_TEMP,
        AMDGPU_RETIMER_0_1_2_3_1V2_VR_TEMP,
        AMDGPU_RETIMER_4_5_6_7_1V2_VR_TEMP,
        AMDGPU_RETIMER_0_1_0V9_VR_TEMP,
        AMDGPU_RETIMER_4_5_0V9_VR_TEMP,
        AMDGPU_RETIMER_2_3_0V9_VR_TEMP,
        AMDGPU_RETIMER_6_7_0V9_VR_TEMP,
        AMDGPU_OAM_0_1_2_3_3V3_VR_TEMP,
        AMDGPU_OAM_4_5_6_7_3V3_VR_TEMP,
        AMDGPU_IBC_HSC_TEMP,
        AMDGPU_IBC_TEMP,
        AMDGPU_SYSTEM_MAX_TEMP_ENTRIES = 32,
};

enum amdgpu_node_temp {
        AMDGPU_RETIMER_X_TEMP,
        AMDGPU_OAM_X_IBC_TEMP,
        AMDGPU_OAM_X_IBC_2_TEMP,
        AMDGPU_OAM_X_VDD18_VR_TEMP,
        AMDGPU_OAM_X_04_HBM_B_VR_TEMP,
        AMDGPU_OAM_X_04_HBM_D_VR_TEMP,
        AMDGPU_NODE_MAX_TEMP_ENTRIES = 12,
};

struct amdgpu_gpuboard_temp_metrics_v1_0 {
        struct metrics_table_header_t common_header;
        uint16_t label_version;
        uint16_t node_id;
        uint64_t accumulation_counter;
        /* Encoded temperature in Celcius, 24:31 is sensor id 0:23 is temp value */
        uint32_t node_temp[AMDGPU_NODE_MAX_TEMP_ENTRIES];
        uint32_t vr_temp[AMDGPU_VR_MAX_TEMP_ENTRIES];
};

struct amdgpu_baseboard_temp_metrics_v1_0 {
        struct metrics_table_header_t common_header;
        uint16_t label_version;
        uint16_t node_id;
        uint64_t accumulation_counter;
        /* Encoded temperature in Celcius, 24:31 is sensor id 0:23 is temp value */
        uint32_t system_temp[AMDGPU_SYSTEM_MAX_TEMP_ENTRIES];
};



rsmi_status_t read_gpuboard_temp_metrics(const char* filename, amdgpu_gpuboard_temp_metrics_v1_0& metrics);
rsmi_status_t read_baseboard_temp_metrics(const char* filename, amdgpu_baseboard_temp_metrics_v1_0& metrics);

rsmi_status_t get_baseboard_temp_value(const amdgpu_baseboard_temp_metrics_v1_0& metrics,
                                       rsmi_temperature_type_t temperature_type,
                                       int64_t* value);

rsmi_status_t get_gpuboard_temp_value(const amdgpu_gpuboard_temp_metrics_v1_0& metrics,
                                      rsmi_temperature_type_t temperature_type,
                                      int64_t* value);
}
#endif  // ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_BOARD_TEMP_H_
