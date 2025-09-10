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

#include <cinttypes>
#include <cstdint>

#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "amd_smi/amdsmi.h"
#include "hw_topology_read.h"

typedef struct {
  std::string type;
  uint64_t hops;
  uint64_t weight;
  bool accessible;
  amdsmi_p2p_capability_t cap;
} gpu_link_t;

TestHWTopologyRead::TestHWTopologyRead() : TestBase() {
  set_title("AMDSMI Hardware Topology Read Test");
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
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other cleanup
  TestBase::Close();
}

void TestHWTopologyRead::Run(void) {
  amdsmi_status_t err;
  uint32_t i, j;

  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) {
      std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    }
    return;
  }

  uint32_t num_devices = num_monitor_devs();

  // gpu_link_t gpu_links[num_devices][num_devices];
  std::vector<std::vector<gpu_link_t>> gpu_links(num_devices,
                                        std::vector<gpu_link_t>(num_devices));
  // uint32_t numa_numbers[num_devices];
  std::vector<uint32_t> numa_numbers(num_devices);

  for (uint32_t dv_ind = 0; dv_ind < num_devices; ++dv_ind) {
    amdsmi_processor_handle dev_handle = processor_handles_[dv_ind];
    err = amdsmi_topo_get_numa_node_number(dev_handle, &numa_numbers[dv_ind]);
    if (err != AMDSMI_STATUS_SUCCESS) {
      if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
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
        gpu_links[dv_ind_src][dv_ind_dst].cap =
          {UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX};
      } else {
        amdsmi_link_type_t type;
        err = amdsmi_topo_get_link_type(processor_handles_[dv_ind_src],
                processor_handles_[dv_ind_dst],
                &gpu_links[dv_ind_src][dv_ind_dst].hops, &type);
        if (err != AMDSMI_STATUS_SUCCESS) {
          if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
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
            case AMDSMI_LINK_TYPE_PCIE:
              gpu_links[dv_ind_src][dv_ind_dst].type = "PCIE";
              break;

            case AMDSMI_LINK_TYPE_XGMI:
              gpu_links[dv_ind_src][dv_ind_dst].type = "XGMI";
              break;

            default:
              gpu_links[dv_ind_src][dv_ind_dst].type = "XXXX";
              IF_VERB(STANDARD) {
                std::cout << "\t**Invalid LINK type. type=" << type << std::endl;
              }
          }
        }
        err = amdsmi_topo_get_p2p_status(processor_handles_[dv_ind_src],
                processor_handles_[dv_ind_dst],
                &type, &gpu_links[dv_ind_src][dv_ind_dst].cap);
        if (err != AMDSMI_STATUS_SUCCESS) {
          if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
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
            case AMDSMI_LINK_TYPE_PCIE:
            case AMDSMI_LINK_TYPE_XGMI:
              // Do nothing, the type is printed by the previous test for amdsmi_topo_get_link_type
              break;
            default:
              gpu_links[dv_ind_src][dv_ind_dst].type = "XXXX";
              IF_VERB(STANDARD) {
                std::cout << "\t**Invalid LINK type. type=" << type << std::endl;
              }
          }
        }
        err = amdsmi_topo_get_link_weight(processor_handles_[dv_ind_src],
                    processor_handles_[dv_ind_dst],
                                   &gpu_links[dv_ind_src][dv_ind_dst].weight);
        if (err != AMDSMI_STATUS_SUCCESS) {
          if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
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
        err = amdsmi_is_P2P_accessible(processor_handles_[dv_ind_src],
                    processor_handles_[dv_ind_dst],
                    &gpu_links[dv_ind_src][dv_ind_dst].accessible);
        if (err != AMDSMI_STATUS_SUCCESS) {
          if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
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

  std::cout << "**Cache coherency between two GPUs**" << std::endl;
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
        continue;
      }

      if (gpu_links[i][j].cap.is_iolink_coherent == UINT8_MAX) {
        std::cout << std::setw(12) << std::left << "N/A";
        continue;
      }

      std::cout << std::setw(12) << std::left
                << (gpu_links[i][j].cap.is_iolink_coherent ? "C" : "NC");
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "**Atomics between two GPUs**" << std::endl;
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
        continue;
      }

      if (gpu_links[i][j].cap.is_iolink_atomics_64bit == UINT8_MAX ||
          gpu_links[i][j].cap.is_iolink_atomics_32bit == UINT8_MAX) {
        std::cout << std::setw(12) << std::left << "N/A";
        continue;
      }

      tmp = gpu_links[i][j].cap.is_iolink_atomics_64bit ? "64" : "";
      if (gpu_links[i][j].cap.is_iolink_atomics_32bit) {
        if (!tmp.empty()) {
          tmp += ",";
        }
        tmp += "32";
      }
      std::cout << std::setw(12) << std::left << (tmp.empty() ? "N/A" : tmp);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "**DMA between two GPUs**" << std::endl;
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
        continue;
      }

      if (gpu_links[i][j].cap.is_iolink_dma == UINT8_MAX) {
        std::cout << std::setw(12) << std::left << "N/A";
        continue;
      }

      std::cout << std::boolalpha;
      std::cout << std::setw(12) << std::left
                << static_cast<bool>(gpu_links[i][j].cap.is_iolink_dma);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "**BI-Directional between two GPUs**" << std::endl;
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
        continue;
      }

      if (gpu_links[i][j].cap.is_iolink_dma == UINT8_MAX) {
        std::cout << std::setw(12) << std::left << "N/A";
        continue;
      }

      std::cout << std::boolalpha;
      std::cout << std::setw(12) << std::left
                << static_cast<bool>(gpu_links[i][j].cap.is_iolink_bi_directional);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  char *topology_link_type_str[] = {
      "AMDSMI_LINK_TYPE_INTERNAL",
      "AMDSMI_LINK_TYPE_XGMI",
      "AMDSMI_LINK_TYPE_PCIE",
      "AMDSMI_LINK_TYPE_NOT_APPLICABLE",
      "AMDSMI_LINK_TYPE_UNKNOWN",
  };

  auto ret(amdsmi_status_t::AMDSMI_STATUS_SUCCESS);
  for (uint32_t dv_ind_src = 0; dv_ind_src < num_devices; dv_ind_src++) {
    std::cout <<"** Nearest GPUs for GPU" << dv_ind_src << " **" << "\n";
    for (uint32_t topo_link_type = AMDSMI_LINK_TYPE_INTERNAL; topo_link_type <= AMDSMI_LINK_TYPE_UNKNOWN; topo_link_type++) {


      /*
       *  Note:   We should get AMDSMI_STATUS_INVAL for the first call with amdsmi_topology_nearest_t = nullptr
       */
      ret = amdsmi_get_link_topology_nearest(processor_handles_[dv_ind_src],
                                             static_cast<amdsmi_link_type_t>(topo_link_type),
                                             nullptr);
      ASSERT_EQ(ret, amdsmi_status_t::AMDSMI_STATUS_INVAL);


      /*
       *
       */
      auto topology_nearest_info = amdsmi_topology_nearest_t();
      ret = amdsmi_get_link_topology_nearest(processor_handles_[dv_ind_src],
                                             static_cast<amdsmi_link_type_t>(topo_link_type),
                                             &topology_nearest_info);
      if (ret != amdsmi_status_t::AMDSMI_STATUS_SUCCESS) {
        continue;
      }

      std::cout <<"Nearest GPUs found for Link Type: " << topology_link_type_str[topo_link_type] << "\n";
      if (topology_nearest_info.count > 0) {
        for (uint32_t k = 0; k < topology_nearest_info.count; k++) {
          amdsmi_bdf_t bdf = {};
          ret = amdsmi_get_gpu_device_bdf(topology_nearest_info.processor_list[k], &bdf);
          if (ret != AMDSMI_STATUS_SUCCESS) {
            continue;
          }

          printf("\tGPU BDF %04" PRIx64 ":%02" PRIx32 ":%02" PRIx32 ".%" PRIu32 "\n",
            static_cast<uint64_t>(bdf.domain_number),
            static_cast<uint32_t>(bdf.bus_number),
            static_cast<uint32_t>(bdf.device_number),
            static_cast<uint32_t>(bdf.function_number));
        }
      }
      else {
        std::cout << "\tNot found" << "\n";
      }
    }
    std::cout << "\n";
  }
}
