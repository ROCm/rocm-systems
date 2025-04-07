/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2020, Advanced Micro Devices, Inc.
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
#include <vector>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/hw_topology_read.h"
#include "rocm_smi_test/test_common.h"

typedef struct {
  std::string type;
  uint64_t hops;
  uint64_t weight;
  bool accessible;
} gpu_link_t;

TestHWTopologyRead::TestHWTopologyRead() : TestBase() {
  set_title("RSMI Hardware Topology Read Test");
  set_description(
      "This test verifies that Hardware Topology can be read properly.");
}

TestHWTopologyRead::~TestHWTopologyRead(void) {
}

void TestHWTopologyRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestHWTopologyRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestHWTopologyRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestHWTopologyRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other cleanup
  TestBase::Close();
}

void TestHWTopologyRead::Run(void) {
  rsmi_status_t err;
  uint32_t i, j;

  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) {
      std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    }
    return;
  }

  uint32_t num_devices;
  err = rsmi_num_monitor_devices(&num_devices);
  CHK_ERR_ASRT(err)

  // gpu_link_t gpu_links[num_devices][num_devices];
  std::vector<std::vector<gpu_link_t>> gpu_links(num_devices,
                                        std::vector<gpu_link_t>(num_devices));
  // uint32_t numa_numbers[num_devices];
  std::vector<uint32_t> numa_numbers(num_devices);

  for (uint32_t dv_ind = 0; dv_ind < num_devices; ++dv_ind) {
    err = rsmi_topo_get_numa_node_number(dv_ind, &numa_numbers[dv_ind]);
    if (err != RSMI_STATUS_SUCCESS) {
      if (err == RSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout <<
           "\t**Numa Node Number. read: Not supported on this machine" <<
                                                                    std::endl;
          return;
        }
      } else {
        CHK_ERR_ASRT(err)
      }
    }
  }

  for (uint32_t dv_ind_src = 0; dv_ind_src < num_devices; dv_ind_src++) {
    for (uint32_t dv_ind_dst = 0; dv_ind_dst < num_devices; dv_ind_dst++) {
      if (dv_ind_src == dv_ind_dst) {
        gpu_links[dv_ind_src][dv_ind_dst].type = "X";
        gpu_links[dv_ind_src][dv_ind_dst].hops = 0;
        gpu_links[dv_ind_src][dv_ind_dst].weight = 0;
        gpu_links[dv_ind_src][dv_ind_dst].accessible = true;
      } else {
        RSMI_IO_LINK_TYPE type;
        err = rsmi_topo_get_link_type(dv_ind_src, dv_ind_dst,
                &gpu_links[dv_ind_src][dv_ind_dst].hops, &type);
        if (err != RSMI_STATUS_SUCCESS) {
          if (err == RSMI_STATUS_NOT_SUPPORTED) {
            IF_VERB(STANDARD) {
              std::cout <<
                  "\t**Link Type. read: Not supported on this machine"
                                                                 << std::endl;
              return;
            }
          } else {
            CHK_ERR_ASRT(err)
          }
        } else {
          switch (type) {
            case RSMI_IOLINK_TYPE_PCIEXPRESS:
              gpu_links[dv_ind_src][dv_ind_dst].type = "PCIE";
              break;

            case RSMI_IOLINK_TYPE_XGMI:
              gpu_links[dv_ind_src][dv_ind_dst].type = "XGMI";
              break;

            default:
              gpu_links[dv_ind_src][dv_ind_dst].type = "XXXX";
              IF_VERB(STANDARD) {
                std::cout << "\t**Invalid IO LINK type. type=" << type <<
                                                                    std::endl;
              }
          }
        }
        err = rsmi_topo_get_link_weight(dv_ind_src, dv_ind_dst,
                                   &gpu_links[dv_ind_src][dv_ind_dst].weight);
        if (err != RSMI_STATUS_SUCCESS) {
          if (err == RSMI_STATUS_NOT_SUPPORTED) {
            IF_VERB(STANDARD) {
              std::cout <<
                      "\t**Link Weight. read: Not supported on this machine"
                                                                 << std::endl;
              return;
            }
          } else {
            CHK_ERR_ASRT(err)
          }
        }
        err = rsmi_is_P2P_accessible(dv_ind_src, dv_ind_dst,
                                   &gpu_links[dv_ind_src][dv_ind_dst].accessible);
        if (err != RSMI_STATUS_SUCCESS) {
          if (err == RSMI_STATUS_NOT_SUPPORTED) {
            IF_VERB(STANDARD) {
              std::cout <<
                      "\t**P2P Access. check: Not supported on this machine"
                                                                 << std::endl;
              return;
            }
          } else {
            CHK_ERR_ASRT(err)
          }
        }
      }
    }
  }

  IF_NVERB(STANDARD) {
    return;
  }

  std::cout << "**NUMA node number of GPUs**" << std::endl;
  std::cout << std::setw(12) << std::left <<"GPU#";
  std::cout <<"NUMA node number";
  std::cout << std::endl;
  for (i = 0; i < num_devices; ++i) {
    std::cout << std::setw(12) << std::left << i;
    std::cout << numa_numbers[i];
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::cout << std::endl;

  std::string tmp;
  std::cout << "**Type between two GPUs**" << std::endl;
  std::cout << "      ";
  for (i = 0; i < num_devices; ++i) {
    tmp = "GPU" + std::to_string(i);
    std::cout << std::setw(12) << std::left << tmp;
  }
  std::cout << std::endl;

  for (i = 0; i < num_devices; i++) {
    tmp = "GPU" + std::to_string(i);
    std::cout << std::setw(6) << std::left << tmp;
    for (j = 0; j < num_devices; j++) {
      if (i == j) {
        std::cout << std::setw(12) << std::left << "X";
      } else {
        std::cout << std::setw(12) << std::left << gpu_links[i][j].type;
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "**Hops between two GPUs**" << std::endl;
  std::cout << "      ";
  for (i = 0; i < num_devices; ++i) {
    tmp = "GPU" + std::to_string(i);
    std::cout << std::setw(12) << std::left << tmp;
  }
  std::cout << std::endl;

  for (i = 0; i < num_devices; i++) {
    tmp = "GPU" + std::to_string(i);
    std::cout << std::setw(6) << std::left << tmp;
    for (j = 0; j < num_devices; j++) {
      if (i == j) {
        std::cout << std::setw(12) << std::left << "X";
      } else {
        std::cout << std::setw(12) << std::left << gpu_links[i][j].hops;
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "**Weight between two GPUs**" << std::endl;
  std::cout << "      ";
  for (i = 0; i < num_devices; ++i) {
    tmp = "GPU" + std::to_string(i);
    std::cout << std::setw(12) << std::left << tmp;
  }
  std::cout << std::endl;

  for (i = 0; i < num_devices; i++) {
    tmp = "GPU" + std::to_string(i);
    std::cout << std::setw(6) << std::left << tmp;
    for (j = 0; j < num_devices; j++) {
      if (i == j) {
        std::cout << std::setw(12) << std::left << "X";
      } else {
        std::cout << std::setw(12) << std::left << gpu_links[i][j].weight;
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::cout << "**Access between two GPUs**" << std::endl;
  std::cout << "      ";
  for (i = 0; i < num_devices; ++i) {
    tmp = "GPU" + std::to_string(i);
    std::cout << std::setw(12) << std::left << tmp;
  }
  std::cout << std::endl;
  for (i = 0; i < num_devices; i++) {
    tmp = "GPU" + std::to_string(i);
    std::cout << std::setw(6) << std::left << tmp;
    for (j = 0; j < num_devices; j++) {
      std::cout << std::boolalpha;
      std::cout << std::setw(12) << std::left << gpu_links[i][j].accessible;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
