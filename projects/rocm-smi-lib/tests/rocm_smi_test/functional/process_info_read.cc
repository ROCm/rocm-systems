/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
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

#include <stdint.h>
#include <stddef.h>

#include <iostream>
#include <string>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/process_info_read.h"
#include "rocm_smi_test/test_common.h"

TestProcInfoRead::TestProcInfoRead() : TestBase() {
  set_title("RSMI Process Info Read Test");
  set_description("This test verifies that process information such as the "
                             "process ID, PASID, etc. can be read properly.");
}

TestProcInfoRead::~TestProcInfoRead(void) {
}

void TestProcInfoRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestProcInfoRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestProcInfoRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestProcInfoRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static void dumpProcess(rsmi_process_info_t *p) {
  assert(p != nullptr);
  std::cout << "\t** ProcessID: " << p->process_id << " ";
  std::cout << "PASID: " << p->pasid << " ";
  std::cout << std::endl;
}
void TestProcInfoRead::Run(void) {
  rsmi_status_t err;
  uint32_t num_proc_found;
  uint32_t val_ui32;
  rsmi_process_info_t *procs = nullptr;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  uint32_t num_devices;
  err = rsmi_num_monitor_devices(&num_devices);
  CHK_ERR_ASRT(err)

  err = rsmi_compute_process_info_get(nullptr, &num_proc_found);
  if (err != RSMI_STATUS_SUCCESS) {
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**Process info. read: Not supported on this machine"
                                                                 << std::endl;
        return;
      }
    } else {
      CHK_ERR_ASRT(err)
    }
  } else {
    IF_VERB(STANDARD) {
      std::cout << "\t** "  << std::dec << num_proc_found <<
                                          " GPU processes found" << std::endl;
    }
  }

  if (num_proc_found == 0) {
    return;
  }

  procs = new rsmi_process_info_t[num_proc_found];

  val_ui32 = num_proc_found;
  err = rsmi_compute_process_info_get(procs, &val_ui32);
  if (err != RSMI_STATUS_SUCCESS) {
    if (err == RSMI_STATUS_INSUFFICIENT_SIZE) {
      IF_VERB(STANDARD) {
        std::cout << "\t** " << val_ui32 <<
         " processes were read, but more became available that were unread."
                                                                 << std::endl;
        for (uint32_t i = 0; i < val_ui32; ++i) {
          dumpProcess(&procs[i]);
        }

        return;
      }
    } else {
      CHK_ERR_ASRT(err)
    }
  } else {
    IF_VERB(STANDARD) {
      std::cout << "\t** Processes currently using GPU: " << std::endl;
      for (uint32_t i = 0; i < val_ui32; ++i) {
        dumpProcess(&procs[i]);
      }
    }
  }

  // Reset to the number we actually read
  num_proc_found = val_ui32;
  if (num_proc_found) {
    // Allocate the max we expect to get
    uint32_t *dev_inds = new uint32_t[num_devices];
    uint32_t amt_allocd = num_devices;

    for (uint32_t j = 0; j < num_proc_found; j++) {
      err = rsmi_compute_process_gpus_get(procs[j].process_id, dev_inds,
                                                                 &amt_allocd);
      if (err == RSMI_STATUS_NOT_FOUND) {
        std::cout << "\t** Process " << procs[j].process_id <<
                                                     " is no longer present.";
        continue;
      } else {
        CHK_ERR_ASRT(err);
        ASSERT_LE(amt_allocd, num_devices);
      }
      std::cout << "\t** Process " << procs[j].process_id <<
                                           " is using devices with indices: ";
      uint32_t i;
      if (amt_allocd > 0) {
        for (i = 0; i < amt_allocd - 1; ++i) {
          std::cout << dev_inds[i] << ", ";
        }
        std::cout << dev_inds[i];
      }
      std::cout << std::endl;

      // Get details of the resource by the process on a specific device
      // if any process running on devices
      for (i = 0; i < amt_allocd; ++i) {
        rsmi_process_info_t proc_info;
        err = rsmi_compute_process_info_by_device_get(
            procs[j].process_id, dev_inds[i], &proc_info);
        CHK_ERR_ASRT(err)
        ASSERT_EQ(proc_info.process_id, procs[j].process_id);
        ASSERT_EQ(proc_info.pasid, procs[j].pasid);
        std::cout << "\t** Process ID: " << procs[j].process_id
              << " on device " << dev_inds[i] << " VRAM Usage: "
              << proc_info.vram_usage << " SDMA Usage: " << proc_info.sdma_usage
              << " Compute Unit Usage: " << proc_info.cu_occupancy << std::endl;
      }
      // Reset amt_allocd back to the amount acutally allocated
      amt_allocd = num_devices;
    }

    delete []dev_inds;

    rsmi_process_info_t proc_info;
    for (uint32_t j = 0; j < num_proc_found; j++) {
      memset(&proc_info, 0x0, sizeof(rsmi_process_info_t));
      err = rsmi_compute_process_info_by_pid_get(procs[j].process_id,
                                                                  &proc_info);
      if (err == RSMI_STATUS_NOT_FOUND) {
        std::cout <<
         "\t** WARNING: rsmi_compute_process_info_get() found process " <<
           procs[j].process_id << ", but subsequently, "
                       "rsmi_compute_process_info_by_pid_get() did not"
                                      " find this same process." << std::endl;
      } else {
        CHK_ERR_ASRT(err)
        ASSERT_EQ(proc_info.process_id, procs[j].process_id);
        ASSERT_EQ(proc_info.pasid, procs[j].pasid);
        std::cout << "\t** Process ID: " <<
            procs[j].process_id << " VRAM Usage: " <<
                                   proc_info.vram_usage <<
                                   " SDMA Usage: " <<
                                   proc_info.sdma_usage <<
                                   " Compute Unit Usage: " <<
                                   proc_info.cu_occupancy <<
                                   std::endl;
      }
    }
  }
  if (num_proc_found > 1) {
    rsmi_process_info_t tmp_proc;
    val_ui32 = 1;
    err = rsmi_compute_process_info_get(&tmp_proc, &val_ui32);

    if (err != RSMI_STATUS_INSUFFICIENT_SIZE) {
      std::cout << "Expected rsmi_compute_process_info_get() to tell us"
        " there are more processes available, but instead go return code " <<
                                                              err << std::endl;
    }
  }
  delete []procs;
}
