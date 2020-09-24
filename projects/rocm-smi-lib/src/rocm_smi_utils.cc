/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2018, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */
#include <assert.h>
#include <errno.h>
#include <sys/stat.h>

#include <fstream>
#include <string>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_device.h"

namespace amd {
namespace smi {

// Return 0 if same file, 1 if not, and -1 for error
int SameFile(const std::string fileA, const std::string fileB) {
  struct stat aStat;
  struct stat bStat;
  int ret;

  ret = stat(fileA.c_str(), &aStat);
  if (ret) {
      return -1;
  }

  ret = stat(fileB.c_str(), &bStat);
  if (ret) {
      return -1;
  }

  if (aStat.st_dev != bStat.st_dev) {
      return 1;
  }

  if (aStat.st_ino != bStat.st_ino) {
      return 1;
  }

  return 0;
}

bool FileExists(char const *filename) {
  struct stat buf;
  return (stat(filename, &buf) == 0);
}

int isRegularFile(std::string fname, bool *is_reg) {
  struct stat file_stat;
  int ret;

  assert(is_reg != nullptr);

  ret = stat(fname.c_str(), &file_stat);
  if (ret) {
    return errno;
  }
  *is_reg = S_ISREG(file_stat.st_mode);
  return 0;
}

int WriteSysfsStr(std::string path, std::string val) {
  std::ofstream fs;
  int ret = 0;

  fs.open(path);
  if (!fs.is_open()) {
    ret = errno;
    errno = 0;
    return ret;
  }

  fs << val;
  fs.close();
  return ret;
}

int ReadSysfsStr(std::string path, std::string *retStr) {
  std::stringstream ss;
  int ret = 0;

  assert(retStr != nullptr);

  std::ifstream fs;
  fs.open(path);

  if (!fs.is_open()) {
    ret = errno;
    errno = 0;
    return ret;
  }
  ss << fs.rdbuf();
  fs.close();

  *retStr = ss.str();

  retStr->erase(std::remove(retStr->begin(), retStr->end(), '\n'),
                                                               retStr->end());
  return ret;
}

bool IsInteger(const std::string & n_str) {
  if (n_str.empty() || ((!isdigit(n_str[0])) && (n_str[0] != '-')
                                                      && (n_str[0] != '+'))) {
    return false;
  }

  char * tmp;
  strtol(n_str.c_str(), &tmp, 10);

  return (*tmp == 0);
}

rsmi_status_t handleException() {
  try {
    throw;
  } catch (const std::bad_alloc& e) {
    debug_print("RSMI exception: BadAlloc\n");
    return RSMI_STATUS_OUT_OF_RESOURCES;
  } catch (const amd::smi::rsmi_exception& e) {
    debug_print("Exception caught: %s.\n", e.what());
    return e.error_code();
  } catch (const std::exception& e) {
    debug_print("Exception caught: %s\n", e.what());
    return RSMI_STATUS_INTERNAL_EXCEPTION;
  } catch (const std::nested_exception& e) {
    debug_print("Callback threw.\n");
    return RSMI_STATUS_INTERNAL_EXCEPTION;
  } catch (...) {
    debug_print("Unknown exception caught.\n");
    return RSMI_STATUS_INTERNAL_EXCEPTION;
  }
}

pthread_mutex_t *GetMutex(uint32_t dv_ind) {
  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();

  if (dv_ind >= smi.monitor_devices().size()) {
    return nullptr;
  }
  std::shared_ptr<amd::smi::Device> dev = smi.monitor_devices()[dv_ind];
  assert(dev != nullptr);

  return dev->mutex();
}

rsmi_status_t GetDevValueVec(amd::smi::DevInfoTypes type,
                         uint32_t dv_ind, std::vector<std::string> *val_vec) {
  assert(val_vec != nullptr);
  if (val_vec == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  GET_DEV_FROM_INDX

  int ret = dev->readDevInfo(type, val_vec);
  return ErrnoToRsmiStatus(ret);
}

rsmi_status_t GetDevBinaryVec(amd::smi::DevInfoTypes type,
                         uint32_t dv_ind, std::vector<unsigned char> *val_vec) {
  assert(val_vec != nullptr);
  if (val_vec == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  GET_DEV_FROM_INDX

  int ret = dev->readDevInfo(type, val_vec);
  return ErrnoToRsmiStatus(ret);
}

rsmi_status_t ErrnoToRsmiStatus(uint32_t err) {
  switch (err) {
    case 0:      return RSMI_STATUS_SUCCESS;
    case ESRCH:  return RSMI_STATUS_NOT_FOUND;
    case EACCES: return RSMI_STATUS_PERMISSION;
    case EPERM:
    case ENOENT: return RSMI_STATUS_NOT_SUPPORTED;
    case EBADF:
    case EISDIR: return RSMI_STATUS_FILE_ERROR;
    case EINTR:  return RSMI_STATUS_INTERRUPT;
    case EIO:    return RSMI_STATUS_UNEXPECTED_SIZE;
    case ENXIO:  return RSMI_STATUS_UNEXPECTED_DATA;
    case EBUSY:  return RSMI_STATUS_BUSY;
    default:     return RSMI_STATUS_UNKNOWN_ERROR;
  }
}

}  // namespace smi
}  // namespace amd
