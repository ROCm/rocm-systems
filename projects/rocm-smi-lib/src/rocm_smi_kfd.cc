/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2019, Advanced Micro Devices, Inc.
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
#include <sys/stat.h>
#include <dirent.h>

#include <algorithm>
#include <string>
#include <map>
#include <fstream>
#include <cstdint>
#include <iostream>
#include <sstream>

#include "rocm_smi/rocm_smi_kfd.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_utils.h"

namespace amd {
namespace smi {

static const char *kKFDProcPathRoot = "/sys/class/kfd/kfd/proc";

// Sysfs file names
static const char *kKFDPasidFName = "pasid";

static bool is_number(const std::string &s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

int GetProcessInfo(rsmi_process_info_t *procs, uint32_t num_allocated,
                                                  uint32_t *num_procs_found) {
  assert(num_procs_found != nullptr);

  *num_procs_found = 0;
  errno = 0;
  auto proc_dir = opendir(kKFDProcPathRoot);

  if (proc_dir == nullptr) {
    perror("Unable to open process directory");
    return errno;
  }
  auto dentry = readdir(proc_dir);

  std::string prod_id_str;
  std::string tmp;

  while (dentry != nullptr) {
    if (dentry->d_name[0] == '.') {
      dentry = readdir(proc_dir);
      continue;
    }

    prod_id_str = dentry->d_name;
    assert(is_number(prod_id_str) && "Unexpected file name in kfd/proc dir");
    if (!is_number(prod_id_str)) {
      continue;
    }
    if (procs && *num_procs_found < num_allocated) {
      int err;
      std::string tmp;

      procs[*num_procs_found].process_id = std::stoi(prod_id_str);

      std::string pasid_str_path = kKFDProcPathRoot;
      pasid_str_path += "/";
      pasid_str_path += prod_id_str;
      pasid_str_path += "/";
      pasid_str_path += kKFDPasidFName;

      err = ReadSysfsStr(pasid_str_path, &tmp);
      if (err) {
        return err;
      }
      assert(is_number(tmp) && "Unexpected value in pasid file");
      procs[*num_procs_found].pasid = std::stoi(tmp);
    }
    ++(*num_procs_found);

    dentry = readdir(proc_dir);
  }

  errno = 0;
  if (closedir(proc_dir)) {
    return errno;
  }
  return 0;
}

int GetProcessInfoForPID(uint32_t pid, rsmi_process_info_t *proc) {
  assert(proc != nullptr);
  int err;
  std::string tmp;

  std::string proc_str_path = kKFDProcPathRoot;
  proc_str_path += "/";
  proc_str_path +=  std::to_string(pid);

  if (!FileExists(proc_str_path.c_str())) {
    return ESRCH;
  }
  proc->process_id = pid;

  std::string pasid_str_path = proc_str_path;
  pasid_str_path += "/";
  pasid_str_path += kKFDPasidFName;

  err = ReadSysfsStr(pasid_str_path, &tmp);
  if (err) {
    return err;
  }
  assert(is_number(tmp) && "Unexpected value in pasid file");
  proc->pasid = std::stoi(tmp);

  return 0;
}

}  // namespace smi
}  // namespace amd
