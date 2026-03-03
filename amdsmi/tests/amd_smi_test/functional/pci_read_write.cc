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

#include <cstdint>

#include <iostream>
#include <bitset>
#include <string>
#include <algorithm>

#include <gtest/gtest.h>
#include "amd_smi/amdsmi.h"
#include "pci_read_write.h"


TestPciReadWrite::TestPciReadWrite() : TestBase() {
  set_title("AMDSMI PCIe Bandwidth Read/Write Test");
  set_description("The PCIe Bandwidth tests verify that the PCIe bandwidth "
                             "settings can be read and controlled properly.");
}

TestPciReadWrite::~TestPciReadWrite(void) {
}

void TestPciReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPciReadWrite::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestPciReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPciReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestPciReadWrite::Run(void) {
  amdsmi_status_t ret;
  amdsmi_pcie_bandwidth_t bw;
  uint32_t freq_bitmask;
  uint64_t sent, received, max_pkt_sz, u64int;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    ret =  amdsmi_get_gpu_pci_replay_counter(processor_handles_[dv_ind], &u64int);

     if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
        std::cout <<
            "\t** amdsmi_get_gpu_pci_replay_counter() is not supported"
            " on this machine" << std::endl;

        // Verify api support checking functionality is working
        ret =  amdsmi_get_gpu_pci_replay_counter(processor_handles_[dv_ind], nullptr);
        ASSERT_EQ(ret, AMDSMI_STATUS_NOT_SUPPORTED);
      } else {
        CHK_ERR_ASRT(ret)
        IF_VERB(STANDARD) {
          std::cout << "\tPCIe Replay Counter: " << u64int << std::endl;
        }
        // Verify api support checking functionality is working
        ret =  amdsmi_get_gpu_pci_replay_counter(processor_handles_[dv_ind], nullptr);
        ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);
      }

    ret = amdsmi_get_gpu_pci_throughput(processor_handles_[dv_ind], &sent, &received, &max_pkt_sz);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "WARNING: Current PCIe throughput is not detected. "
        "pcie_bw sysfs file is no longer supported on this device. "
        "Aborting test." << std::endl;

      // We don't need to verify api support checking functionality is working
      // as the user may choose to have any of the input parameters as 0.
      return;
    }
    CHK_ERR_ASRT(ret)

    IF_VERB(STANDARD) {
      std::cout << "\tPCIe Throughput (1 sec.): " << std::endl;
      std::cout << "\t\tSent: " << sent << " bytes" << std::endl;
      std::cout << "\t\tReceived: " << received << " bytes" << std::endl;
      std::cout << "\t\tMax Packet Size: " << max_pkt_sz << " bytes" <<
                                                                    std::endl;
      std::cout << std::endl;
    }

    ret = amdsmi_get_gpu_pci_bandwidth(processor_handles_[dv_ind], &bw);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "WARNING: Current PCIe bandwidth is not detected. "
        "pp_dpm_pcie sysfs file is no longer supported on this device. "
         "Aborting test." << std::endl;
      // Verify api support checking functionality is working
      ret = amdsmi_get_gpu_pci_bandwidth(processor_handles_[dv_ind], nullptr);
      ASSERT_EQ(ret, AMDSMI_STATUS_NOT_SUPPORTED);

      return;
    } else if (ret == AMDSMI_STATUS_NOT_YET_IMPLEMENTED) {
      std::cout << "TEST FAILURE: Current PCIe bandwidth is not implemented.\n";
      return;
    }
    CHK_ERR_ASRT(ret)

    IF_VERB(STANDARD) {
      std::cout << "\tInitial PCIe BW index is " << bw.transfer_rate.current <<
                                                                    std::endl;
    }
    // Verify api support checking functionality is working
    // NOTE:  We expect AMDSMI_STATUS_NOT_SUPPORTED, if rsmi_pcie_bandwidth_t* is NULL
    ret = amdsmi_get_gpu_pci_bandwidth(processor_handles_[dv_ind], nullptr);
    if (ret != amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED) {
        ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);
    }
    else {
        auto status_string("");
        amdsmi_status_code_to_string(ret, &status_string);
        std::cout << "\t\t** amdsmi_get_gpu_pci_bandwidth(): " << status_string << "\n";
    }

    // First set the bitmask to all supported bandwidths
    freq_bitmask = ~(~0u << bw.transfer_rate.num_supported);

    // Then, set the bitmask to all bandwidths besides the initial BW
    freq_bitmask ^= (1 << bw.transfer_rate.current);

    std::string freq_bm_str =
               std::bitset<AMDSMI_MAX_NUM_FREQUENCIES>(freq_bitmask).to_string();

    freq_bm_str.erase(0, std::min(freq_bm_str.find_first_not_of('0'),
                                                       freq_bm_str.size()-1));

    IF_VERB(STANDARD) {
    std::cout << "\tSetting bandwidth mask to " << "0b" << freq_bm_str <<
                                                            " ..." << std::endl;
    }
    ret =  amdsmi_set_gpu_pci_bandwidth(processor_handles_[dv_ind], freq_bitmask);
    if (ret != amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED) {
        CHK_ERR_ASRT(ret)
    }
    else {
        auto status_string("");
        amdsmi_status_code_to_string(ret, &status_string);
        std::cout << "\t\t** amdsmi_set_gpu_pci_bandwidth(): " << status_string << "\n";
    }

    ret = amdsmi_get_gpu_pci_bandwidth(processor_handles_[dv_ind], &bw);
    CHK_ERR_ASRT(ret)

    IF_VERB(STANDARD) {
      std::cout << "\tBandwidth is now index " << bw.transfer_rate.current <<
                                                                      std::endl;
      std::cout << "\tResetting mask to all bandwidths." << std::endl;
    }
    ret =  amdsmi_set_gpu_pci_bandwidth(processor_handles_[dv_ind], 0xFFFFFFFF);
    if (ret != amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED) {
        CHK_ERR_ASRT(ret)
    }
    else {
        auto status_string("");
        amdsmi_status_code_to_string(ret, &status_string);
        std::cout << "\t\t** amdsmi_set_gpu_pci_bandwidth(): " << status_string << "\n";
    }

    ret =  amdsmi_set_gpu_perf_level(processor_handles_[dv_ind], AMDSMI_DEV_PERF_LEVEL_AUTO);
    CHK_ERR_ASRT(ret)
  }
}
