// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mock.h"

amdsmi_status_t amdsmi_get_gpu_ecc_enabled(amdsmi_processor_handle h, uint64_t* out) {
  amdsmi_status_t status = mock_begin(__func__, h);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  *out = AMDSMI_GPU_BLOCK_UMC | AMDSMI_GPU_BLOCK_UCIE_PCS | AMDSMI_GPU_BLOCK_RESERVED |
         (UINT64_C(1) << 62);
  if (mock_mode(__func__) == 1) *out = 0;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_gpu_ecc_count(amdsmi_processor_handle h, amdsmi_gpu_block_t block,
                                         amdsmi_error_count_t* out) {
  amdsmi_status_t status = mock_begin(__func__, h);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (block != AMDSMI_GPU_BLOCK_UCIE_PCS) return mock_finish(AMDSMI_STATUS_INVAL);
  out->correctable_count = UINT64_C(1) << 40;
  out->uncorrectable_count = UINT64_C(1) << 41;
  out->deferred_count = UINT64_MAX;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_gpu_total_ecc_count(amdsmi_processor_handle h,
                                               amdsmi_error_count_t* out) {
  amdsmi_status_t status = mock_begin(__func__, h);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  out->correctable_count += UINT64_C(1) << 40;
  out->uncorrectable_count += UINT64_C(1) << 41;
  out->deferred_count += 7;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_gpu_ras_block_features_enabled(amdsmi_processor_handle h,
                                                          amdsmi_gpu_block_t block,
                                                          amdsmi_ras_err_state_t* out) {
  amdsmi_status_t status = mock_begin(__func__, h);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (block != AMDSMI_GPU_BLOCK_UCIE_PCS) return mock_finish(AMDSMI_STATUS_INVAL);
  *out = AMDSMI_RAS_ERR_STATE_ENABLED;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_gpu_ras_feature_info(amdsmi_processor_handle h,
                                                amdsmi_ras_feature_t* out) {
  amdsmi_status_t status = mock_begin(__func__, h);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  out->ras_eeprom_version = 0x102;
  out->ecc_correction_schema_flag = 0xf;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}
