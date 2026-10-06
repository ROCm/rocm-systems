// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <string.h>

#include "mock.h"

amdsmi_status_t amdsmi_get_gpu_device_uuid(amdsmi_processor_handle h, unsigned int* size,
                                           char* out) {
  amdsmi_status_t status = mock_begin(__func__, h);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (!size || !out || *size != AMDSMI_GPU_UUID_SIZE || !mock_is_zero(out, *size))
    return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (mock_mode(__func__) == 1) {
    memset(out, 'U', *size);
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  if (mock_mode(__func__) == 2) {
    *size = AMDSMI_GPU_UUID_SIZE + 1;
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  if (mock_mode(__func__) == 3) {
    memset(out, 'U', *size);
    *size = 0;
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  strcpy(out, "00000000-0000-0000-0000-000000000001");
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_gpu_asic_info(amdsmi_processor_handle h, amdsmi_asic_info_t* out) {
  amdsmi_status_t status = mock_begin(__func__, h);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (mock_mode(__func__) == 1) {
    memset(out->market_name, 'M', sizeof(out->market_name));
    memset(out->vendor_name, 'V', sizeof(out->vendor_name));
    memset(out->asic_serial, 'S', sizeof(out->asic_serial));
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  strcpy(out->market_name, "test-gpu");
  out->vendor_id = 0x1002;
  strcpy(out->vendor_name, "AMD");
  out->subvendor_id = 0x1002;
  out->device_id = (UINT64_C(1) << 40) + 0x1234;
  out->rev_id = UINT32_MAX;
  strcpy(out->asic_serial, "asic-serial");
  out->oam_id = 7;
  out->num_of_compute_units = 304;
  out->target_graphics_version = (UINT64_C(1) << 40) + 950;
  out->subsystem_id = 0x42;
  out->flags = (UINT64_C(1) << 63) + 1;
  out->physical_acc_id = 9;
  out->chip_rev_id = 0x91;
  out->external_rev_id = 0x92;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_gpu_driver_info(amdsmi_processor_handle h, amdsmi_driver_info_t* out) {
  amdsmi_status_t status = mock_begin(__func__, h);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (mock_mode(__func__) == 1) {
    memset(out->driver_version, 'V', sizeof(out->driver_version));
    memset(out->driver_date, 'D', sizeof(out->driver_date));
    memset(out->driver_name, 'N', sizeof(out->driver_name));
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  strcpy(out->driver_version, "6.16");
  strcpy(out->driver_date, "20260925");
  strcpy(out->driver_name, "amdgpu");
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_gpu_board_info(amdsmi_processor_handle h, amdsmi_board_info_t* out) {
  amdsmi_status_t status = mock_begin(__func__, h);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (mock_mode(__func__) == 1) {
    memset(out->model_number, 'M', sizeof(out->model_number));
    memset(out->product_serial, 'S', sizeof(out->product_serial));
    memset(out->fru_id, 'F', sizeof(out->fru_id));
    memset(out->product_name, 'P', sizeof(out->product_name));
    memset(out->manufacturer_name, 'A', sizeof(out->manufacturer_name));
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  strcpy(out->model_number, "model");
  strcpy(out->product_serial, "serial");
  strcpy(out->fru_id, "fru");
  strcpy(out->product_name, "board");
  strcpy(out->manufacturer_name, "AMD");
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_fw_info(amdsmi_processor_handle h, amdsmi_fw_info_t* out) {
  amdsmi_status_t status = mock_begin(__func__, h);
  uint32_t mode = mock_mode(__func__);
  if (status != AMDSMI_STATUS_SUCCESS && mode != 4) return mock_finish(status);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (mode == 1) {
    out->num_fw_info = 0;
    out->fw_info_list[0].fw_id = (amdsmi_fw_block_t)0x1234;
    out->fw_info_list[0].fw_version = UINT64_MAX;
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  if (mode == 2) {
    out->num_fw_info = AMDSMI_FW_ID__MAX;
    for (uint32_t i = 0; i < AMDSMI_FW_ID__MAX; ++i) {
      out->fw_info_list[i].fw_id = (amdsmi_fw_block_t)i;
      out->fw_info_list[i].fw_version = i;
    }
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  if (mode == 3) {
    out->num_fw_info = AMDSMI_FW_ID__MAX + 1;
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  out->num_fw_info = 2;
  out->fw_info_list[0].fw_id = AMDSMI_FW_ID_SMU;
  out->fw_info_list[0].fw_version = (UINT64_C(1) << 40) + 3;
  out->fw_info_list[1].fw_id = (amdsmi_fw_block_t)0x1234;
  out->fw_info_list[1].fw_version = UINT64_MAX;
  return mock_finish(status);
}

amdsmi_status_t amdsmi_get_gpu_vbios_info(amdsmi_processor_handle h, amdsmi_vbios_info_t* out) {
  amdsmi_status_t status = mock_begin(__func__, h);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (mock_mode(__func__) == 1) {
    memset(out->name, 'N', sizeof(out->name));
    memset(out->build_date, 'B', sizeof(out->build_date));
    memset(out->part_number, 'P', sizeof(out->part_number));
    memset(out->version, 'V', sizeof(out->version));
    memset(out->boot_firmware, 'F', sizeof(out->boot_firmware));
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  strcpy(out->name, "vbios");
  strcpy(out->build_date, "2026-09-25");
  strcpy(out->part_number, "part");
  strcpy(out->version, "1");
  strcpy(out->boot_firmware, "ubl");
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}
