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

#include <sstream>

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

#define TRY try {
#define CATCH } catch (...) {return handleRSMIException();}


static int handleRSMIException() {
  rsmi_status_t ret;
  ret = amd::smi::handleException();

  // TODO(x): convert RSMI return to OAM return
  // For now, just return int equiv.
  return static_cast<int>(ret);
}

int amdoam_init(oam_mapi_version_t version) {
  TRY

  // TODO(x): handle version argument
  (void)version;

  rsmi_status_t ret = rsmi_init(0);

  return 0;
  CATCH
}

int amdoam_free(void) {
  rsmi_status_t ret = rsmi_shut_down();

  // TODO(x) convert rsmi return to oam return val
  return static_cast<int>(ret);
}


int amdoam_discover_devices(int *device_count) {
  uint32_t dv_cnt;

  if (device_count == nullptr) {
    return -1;  // TODO(x): return appropriate OAM code
  }

  rsmi_status_t ret =  rsmi_num_monitor_devices(&dv_cnt);

  *device_count = static_cast<int>(dv_cnt);

  // TODO(x) convert rsmi return to oam return val
  return static_cast<int>(ret);
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

    default:
      return RSMI_STATUS_NOT_SUPPORTED;
  }

  DEVICE_MUTEX

  ret = GetDevValueVec(type, dv_ind, &val_vec);

  if (ret == RSMI_STATUS_FILE_ERROR) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
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

  return ret;
  CATCH
}

