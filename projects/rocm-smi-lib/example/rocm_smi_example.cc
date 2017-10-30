/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
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

#include <vector>
#include <iostream>
#include "rocm_smi/rocm_smi.h"

// Call-back function to append to a vector of Devices
static bool GetMonitorDevices(const std::shared_ptr<amd::smi::Device> &d,
                                                                    void *p) {
  std::string val_str;

  assert(p != nullptr);

  std::vector<std::shared_ptr<amd::smi::Device>> *device_list =
    reinterpret_cast<std::vector<std::shared_ptr<amd::smi::Device>> *>(p);

  if (d->monitor() != nullptr) {
    device_list->push_back(d);
  }
  return false;
}

int main() {
  amd::smi::RocmSMI hw;
  std::vector<std::shared_ptr<amd::smi::Device>> monitor_devices;

  // DiscoverDevices() will seach for devices and monitors and update internal
  // data structures.
  hw.DiscoverDevices();

  // IterateSMIDevices will iterate through all the known devices and apply
  // the provided call-back to each device found.
  hw.IterateSMIDevices(GetMonitorDevices,
                                  reinterpret_cast<void *>(&monitor_devices));

  std::string val_str;
  std::vector<std::string> val_vec;
  uint32_t value;
  uint32_t value2;
  int ret;

  // Iterate through the list of devices and print out information related to
  // that device.
  for (auto dev : monitor_devices) {
    dev->readDevInfo(amd::smi::kDevDevID, &val_str);
    std::cout << "\t**Device ID:" << val_str << std::endl;

    dev->readDevInfo(amd::smi::kDevPerfLevel, &val_str);
    std::cout << "\t**Performance Level:" << val_str << std::endl;

    dev->readDevInfo(amd::smi::kDevOverDriveLevel, &val_str);
    std::cout << "\t**OverDrive Level:" << val_str << std::endl;

    dev->readDevInfo(amd::smi::kDevGPUMClk, &val_vec);
    std::cout << "\t**Supported GPU Memory clock frequencies:" << std::endl;
    for (auto vs : val_vec) {
      std::cout << "\t**  " << vs << std::endl;
    }
    val_vec.clear();

    dev->readDevInfo(amd::smi::kDevGPUSClk, &val_vec);
    std::cout << "\t**Supported GPU clock frequencies:" << val_str << std::endl;
    for (auto vs : val_vec) {
      std::cout << "\t**  " << vs << std::endl;
    }
    val_vec.clear();

    // See if there is a monitor associated with the current device, and if so,
    // print out the associated monitor information.
    if (dev->monitor() != nullptr) {
      ret = dev->monitor()->readMonitor(amd::smi::kMonName, &val_str);
      std::cout << "\t**Monitor name: ";

      if (ret != -1) {
        std::cout << val_str << std::endl;
      } else {
        std::cout << "Not available" << std::endl;
      }

      std::cout << "\t**Temperature: ";
      ret = dev->monitor()->readMonitor(amd::smi::kMonTemp, &value);
      if (ret != -1) {
        std::cout << static_cast<float>(value)/1000.0 << "C" << std::endl;
      } else {
        std::cout << "Not available" << std::endl;
      }

      std::cout << "\t**Current Fan Speed: ";

      ret = dev->monitor()->readMonitor(amd::smi::kMonMaxFanSpeed, &value);
      if (ret == 0) {
        ret = dev->monitor()->readMonitor(amd::smi::kMonFanSpeed, &value2);
        if (ret != -1) {
          std::cout.setf(std::ios::dec, std::ios::basefield);
          std::cout << value2/static_cast<float>(value) * 100 << "% (" <<
                                   value2 << "/" << value << ")" << std::endl;
        } else {
          std::cout << "Not available" << std::endl;
        }
      }
    }
    std::cout << "\t=======" << std::endl;
  }
  return 0;
}
