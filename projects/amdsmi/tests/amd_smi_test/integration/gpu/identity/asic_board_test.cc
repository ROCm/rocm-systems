// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

AMDSMI_INTEGRATION_GPU_STRUCT_GETTER(GetDriverInfo, amdsmi_get_gpu_driver_info,
                                     amdsmi_driver_info_t)
AMDSMI_INTEGRATION_GPU_STRUCT_GETTER(GetAsicInfo, amdsmi_get_gpu_asic_info, amdsmi_asic_info_t)
AMDSMI_INTEGRATION_GPU_STRUCT_GETTER(GetKfdInfo, amdsmi_get_gpu_kfd_info, amdsmi_kfd_info_t)
AMDSMI_INTEGRATION_GPU_STRUCT_GETTER(GetVramInfo, amdsmi_get_gpu_vram_info, amdsmi_vram_info_t)
AMDSMI_INTEGRATION_GPU_STRUCT_GETTER(GetBoardInfo, amdsmi_get_gpu_board_info, amdsmi_board_info_t)
AMDSMI_INTEGRATION_GPU_STRUCT_GETTER(GetVbiosInfo, amdsmi_get_gpu_vbios_info, amdsmi_vbios_info_t)
AMDSMI_INTEGRATION_GPU_STRUCT_GETTER(GetFwInfo, amdsmi_get_fw_info, amdsmi_fw_info_t)
