/*
 * MIT License
 *
 * Copyright (c) 2020 Open Compute Project
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <assert.h>
#include <dirent.h>
#include <sstream>
#include <cstring>
#include <iostream>
#include <regex>  // NOLINT
#include <map>

#include "rocm_smi/rocm_smi_common.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_counters.h"
#include "rocm_smi/rocm_smi_kfd.h"
#include "rocm_smi/rocm_smi.h"

#include "oam/oam_mapi.h"
#include "oam/amd_oam.h"

static const std::map<int, const char *> err_map = {
  { AMDOAM_STATUS_INVALID_ARGS, "Invalid arguments" },
  { AMDOAM_STATUS_NOT_SUPPORTED, "Feature not supported" },
  { AMDOAM_STATUS_FILE_ERROR, "Problem accessing a file" },
  { AMDOAM_STATUS_PERMISSION, "Permission denied" },
  { AMDOAM_STATUS_OUT_OF_RESOURCES, "Not enough memory or other resource" },
  { AMDOAM_STATUS_INTERNAL_EXCEPTION, "An internal exception was caught" },
  { AMDOAM_STATUS_INPUT_OUT_OF_BOUNDS,
                  "The provided input is out of allowable or safe range" },
  { AMDOAM_STATUS_INIT_ERROR, "AMDOAM is not initialized or init failed" },
  { AMDOAM_STATUS_ERROR, "Generic error" },
  { AMDOAM_STATUS_NOT_FOUND, "An item was searched for but not found" }
};

#define TRY try {
#define CATCH } catch (...) {return handleRSMIException();}

static bool rsmi_initialized;

static int rsmi_status_to_amdoam_errorcode(rsmi_status_t status) {
  if (status > RSMI_STATUS_INIT_ERROR)
    return -AMDOAM_STATUS_ERROR;
  else
    return -1 * static_cast<int>(status);
}

static int handleRSMIException() {
  rsmi_status_t ret = amd::smi::handleException();
  return rsmi_status_to_amdoam_errorcode(ret);
}

int amdoam_get_error_description(int code, const char **description) {
  if (description == nullptr)
    return -AMDOAM_STATUS_INVALID_ARGS;

  auto search = err_map.find(code);
  if (search == err_map.end())
    return -AMDOAM_STATUS_NOT_FOUND;

  *description = search->second;
  return AMDOAM_STATUS_SUCCESS;
}

int amdoam_init(void) {
  TRY

  rsmi_status_t status = rsmi_init(0);

  if (status != RSMI_STATUS_SUCCESS)
    return rsmi_status_to_amdoam_errorcode(status);

  rsmi_initialized = true;
  return AMDOAM_STATUS_SUCCESS;

  CATCH
}

int amdoam_free(void) {
  rsmi_status_t status = rsmi_shut_down();

  if (status != RSMI_STATUS_SUCCESS)
    return rsmi_status_to_amdoam_errorcode(status);

  return AMDOAM_STATUS_SUCCESS;
}

int amdoam_discover_devices(uint32_t *device_count) {
  rsmi_status_t status;

  if (device_count == nullptr) {
    return -AMDOAM_STATUS_INVALID_ARGS;
  }

  status =  rsmi_num_monitor_devices(device_count);
  if (status != RSMI_STATUS_SUCCESS)
    return rsmi_status_to_amdoam_errorcode(status);

  return AMDOAM_STATUS_SUCCESS;
}

int amdoam_get_pci_properties(uint32_t device_id, oam_pci_info_t *pci_info) {
  uint64_t bdfid;

  TRY
  if (pci_info == nullptr) {
    return -AMDOAM_STATUS_INVALID_ARGS;
  }

  rsmi_status_t status = rsmi_dev_pci_id_get(device_id, &bdfid);
  if (status != RSMI_STATUS_SUCCESS)
    return rsmi_status_to_amdoam_errorcode(status);

  pci_info->domain = (uint16_t)(bdfid >> 32) & 0xffff;
  pci_info->bus = (bdfid >> 8) & 0xff;
  pci_info->device = (bdfid >> 3) & 0x1f;
  pci_info->function = bdfid & 0x7;
  CATCH

  return  AMDOAM_STATUS_SUCCESS;
}

int amdoam_get_dev_properties(uint32_t num_devices,
                              oam_dev_properties_t *devices) {
  const size_t buf_size = 32;
  char buf[buf_size] = "";
  uint32_t dev_inx;
  oam_dev_properties_t *dev = devices;

TRY
  if (devices == nullptr)
    return -AMDOAM_STATUS_INVALID_ARGS;
  if (!rsmi_initialized)
    return -AMDOAM_STATUS_INIT_ERROR;

  for (dev_inx = 0; dev_inx < num_devices; dev_inx++) {
    dev->device_id = dev_inx;
    /* If fails to get any following properties, it's not treated as a deal
     * breaker. Variable not filled means that property is not available on
     * this device or AMD doesn't support that property.
     */
    rsmi_dev_vendor_name_get(dev_inx, dev->device_vendor, DEVICE_VENDOR_LEN);
    rsmi_dev_name_get(dev_inx, dev->device_name, DEVICE_NAME_LEN);
    rsmi_dev_vbios_version_get(dev_inx, buf, buf_size);
    if (std::strlen(buf) > 0) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-truncation"
      std::strncpy(dev->sku_name, &buf[4], 6);
      std::strncpy(dev->board_name, buf, 12);
#pragma GCC diagnostic pop
    }
    rsmi_dev_serial_number_get(dev_inx, dev->board_serial_number,
                             BOARD_SERIAL_NUM_LEN);
    ++dev;
  }
CATCH

  return AMDOAM_STATUS_SUCCESS;
}

static uint32_t
get_num_sensors(std::string hwmon_path, std::string fn_reg) {
  uint32_t sensor_max = 0;
  std::string fn_reg_ex = "\\b" + fn_reg + "([0-9]+)([^ ]*)";
  std::string fn;
  std::smatch m;
  int32_t temp = 0;
  std::string s1("in");
  std::regex re(fn_reg_ex);
  auto hwmon_dir = opendir(hwmon_path.c_str());
  assert(hwmon_dir != nullptr);
  auto dentry = readdir(hwmon_dir);
  while (dentry != nullptr) {
    fn = dentry->d_name;
    if (std::regex_search(fn, m, re)) {
      std::string output = std::regex_replace(
        fn,
        std::regex("[^0-9]*([0-9]+).*"),
        std::string("$1"));
      temp = stoi(output);

      assert(temp >= 0);

      if (s1.compare(fn_reg) == 0)
        ++temp;
      if (static_cast<uint32_t>(temp) > sensor_max)
        sensor_max = static_cast<uint32_t>(temp);
    }
    dentry = readdir(hwmon_dir);
  }

  closedir(hwmon_dir);
  return sensor_max;
}


int amdoam_get_sensors_count(uint32_t device_id,
                             oam_sensor_count_t *sensor_count) {
  uint32_t dv_ind = device_id;

  TRY
  if (sensor_count == nullptr)
    return -AMDOAM_STATUS_INVALID_ARGS;
  GET_DEV_FROM_INDX
  assert(dev->monitor() != nullptr);
  std::string hwmon_path = dev->monitor()->path();
  sensor_count->num_temperature_sensors = get_num_sensors(hwmon_path, "temp");
  sensor_count->num_fans = get_num_sensors(hwmon_path, "fan");
  sensor_count->num_voltage_sensors = get_num_sensors(hwmon_path, "in");
  sensor_count->num_power_sensors = get_num_sensors(hwmon_path, "power");
  sensor_count->num_current_sensors = get_num_sensors(hwmon_path, "current");
  CATCH

  return AMDOAM_STATUS_SUCCESS;
}

int amdoam_get_sensors_info(uint32_t device_id, oam_sensor_type_t type,
                 uint32_t num_sensors, oam_sensor_info_t sensor_info[]) {
  uint32_t dv_ind = device_id;
  std::string val_str;
  uint32_t i;
  rsmi_status_t status;

  TRY
  if ((sensor_info == nullptr) || (type >= OAM_SENSOR_TYPE_UNKNOWN))
    return -AMDOAM_STATUS_INVALID_ARGS;
  GET_DEV_FROM_INDX
  assert(dev->monitor() != nullptr);
  switch (type) {
    case OAM_SENSOR_TYPE_POWER:
      for (i = 0; i < num_sensors; i++) {
        snprintf(sensor_info[i].sensor_name, OAM_SENSOR_NAME_MAX,
                                 "POWER_SENSOR_%u", i+1);
        sensor_info[i].sensor_type = type;
        status = rsmi_dev_power_ave_get(device_id, i,
                            reinterpret_cast<uint64_t*>(&sensor_info[i].value));
        if (status != RSMI_STATUS_SUCCESS)
          return rsmi_status_to_amdoam_errorcode(status);
      }
      break;

    case OAM_SENSOR_TYPE_VOLTAGE:
      for (i = 0; i < num_sensors; i++) {
        snprintf(sensor_info[i].sensor_name, OAM_SENSOR_NAME_MAX,
                                  "VOLTAGE_SENSOR_%u", i);
        sensor_info[i].sensor_type = type;
        status = rsmi_dev_volt_metric_get(device_id, RSMI_VOLT_TYPE_VDDGFX,
                          RSMI_VOLT_CURRENT, &sensor_info[i].value);
        if (status != RSMI_STATUS_SUCCESS)
          return rsmi_status_to_amdoam_errorcode(status);
      }
      break;

    case OAM_SENSOR_TYPE_TEMP:
      for (i = 0; i < num_sensors; i++) {
        snprintf(sensor_info[i].sensor_name, OAM_SENSOR_NAME_MAX,
                                   "TEMP_SENSOR_%u", i+1);
        sensor_info[i].sensor_type = type;
        status = rsmi_dev_temp_metric_get(device_id, i, RSMI_TEMP_CURRENT,
                                         &sensor_info[i].value);
        if (status != RSMI_STATUS_SUCCESS)
          return rsmi_status_to_amdoam_errorcode(status);
      }
      break;

    case OAM_SENSOR_TYPE_FAN_SPEED:
      for (i = 0; i < num_sensors; i++) {
        snprintf(sensor_info[i].sensor_name, OAM_SENSOR_NAME_MAX,
                                   "FAN_SENSOR_%u", i+1);
        sensor_info[i].sensor_type = type;
        status = rsmi_dev_fan_speed_get(device_id, i, &sensor_info[i].value);
        if (status != RSMI_STATUS_SUCCESS)
          return rsmi_status_to_amdoam_errorcode(status);
        }
      break;

    default:
      return -AMDOAM_STATUS_NOT_SUPPORTED;
  }
  CATCH

  return AMDOAM_STATUS_SUCCESS;
}

// TODO(x): This function doesn't work for OAM. It's just a version
// of rsmi_dev_ecc_count_get(), which has similar functionality.
// The purpose here is just to drive refactoring; e.g., making macros
// available and previously static functions global.
int
get_device_error_count(oam_dev_handle_t *handle,
                                          oam_dev_error_count_t *count) {
  std::vector<std::string> val_vec;
  rsmi_status_t ret;

  TRY
  // TODO(x): replace with final code...
  // Below, we are just returning errors for RSMI_GPU_BLOCK_GFX as a
  // placeholder
  (void)handle;  // Just ignore for now

  rsmi_gpu_block_t block = RSMI_GPU_BLOCK_GFX;

  // The macro CHK_SUPPORT_VAR assumes the existence of a device index variable
  // "dv_ind". Presumably, the device index will come from the "handle"
  // pointer. Since I don't know how that will be implemented, for now we
  // will just make up a device index:
  uint32_t dv_ind = 0;
  CHK_SUPPORT_VAR(count, block)

  amd::smi::DevInfoTypes type;
  switch (block) {
    case RSMI_GPU_BLOCK_UMC:
      type = amd::smi::kDevErrCntUMC;
      break;

    case RSMI_GPU_BLOCK_SDMA:
      type = amd::smi::kDevErrCntSDMA;
      break;

    case RSMI_GPU_BLOCK_GFX:
      type = amd::smi::kDevErrCntGFX;
      break;

    case RSMI_GPU_BLOCK_MMHUB:
      type = amd::smi::kDevErrCntMMHUB;
      break;

    case RSMI_GPU_BLOCK_PCIE_BIF:
      type = amd::smi::kDevErrCntPCIEBIF;
      break;

    case RSMI_GPU_BLOCK_HDP:
      type = amd::smi::kDevErrCntHDP;
      break;

    case RSMI_GPU_BLOCK_XGMI_WAFL:
      type = amd::smi::kDevErrCntXGMIWAFL;
      break;

    default:
      return RSMI_STATUS_NOT_SUPPORTED;
  }

  DEVICE_MUTEX

  ret = GetDevValueVec(type, dv_ind, &val_vec);

  if (ret == RSMI_STATUS_FILE_ERROR) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  if (ret != RSMI_STATUS_SUCCESS) {
    return static_cast<int>(ret);
  }

  assert(val_vec.size() == 2);

  std::string junk;
  std::istringstream fs1(val_vec[0]);

  fs1 >> junk;
  assert(junk == "ue:");
  fs1 >> count->total_error_count;

  std::istringstream fs2(val_vec[1]);

  fs2 >> junk;
  assert(junk == "ce:");
  fs2 >> count->total_error_count;

  return static_cast<int>(ret);
  CATCH
}
