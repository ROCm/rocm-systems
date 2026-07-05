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

#include "fabric_read.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>

#include "amd_smi/amdsmi.h"

// Category mask covering all telemetry categories
static constexpr uint32_t kAllCategories =
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_UALOE | AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_SWITCH |
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_CRYPTO | AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_PFC |
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_NETPORT |
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_DERIVED_UALOE |
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_DERIVED_NETPORT;

TestFabricRead::TestFabricRead() : TestBase() {
  set_title("AMDSMI Fabric (UALoE) Read Test");
  set_description(
      "This test verifies that fabric device info and telemetry data "
      "can be read properly via amdsmi_get_gpu_fabric_info(), "
      "amdsmi_alloc_fabric_telemetry(), amdsmi_get_fabric_telemetry_data(), "
      "amdsmi_free_fabric_telemetry(), and amdsmi_fabric_telem_id_to_string().");
}

TestFabricRead::~TestFabricRead(void) {}

void TestFabricRead::SetUp(void) {
  TestBase::SetUp();
  return;
}

void TestFabricRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestFabricRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestFabricRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestFabricRead::Run(void) {
  amdsmi_status_t err;

  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) { std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl; }
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    auto device = processor_handles_[dv_ind];
    PrintDeviceHeader(device);

    // Without a UALoE session the telemetry APIs must report NOT_SUPPORTED,
    // not NOT_INIT.
    {
      amdsmi_fabric_telemetry_t* probe = nullptr;
      err = amdsmi_alloc_fabric_telemetry(device, kAllCategories, &probe);
      ASSERT_NE(err, AMDSMI_STATUS_NOT_INIT);
      if (err == AMDSMI_STATUS_SUCCESS) {
        ASSERT_NE(probe, nullptr);
        err = amdsmi_get_fabric_telemetry_data(device, probe);
        ASSERT_NE(err, AMDSMI_STATUS_NOT_INIT);
        err = amdsmi_free_fabric_telemetry(device, probe);
        ASSERT_NE(err, AMDSMI_STATUS_NOT_INIT);
      } else {
        amdsmi_fabric_telemetry_t dummy = {};
        err = amdsmi_get_fabric_telemetry_data(device, &dummy);
        ASSERT_NE(err, AMDSMI_STATUS_NOT_INIT);
        err = amdsmi_free_fabric_telemetry(device, &dummy);
        ASSERT_NE(err, AMDSMI_STATUS_NOT_INIT);
      }
    }

    // ── amdsmi_get_gpu_fabric_info ─────────────────────────────────────────
    IF_VERB(STANDARD) { std::cout << "\t** Testing amdsmi_get_gpu_fabric_info()" << std::endl; }

    amdsmi_fabric_info_t fabric_info = {};
    err = amdsmi_get_gpu_fabric_info(device, &fabric_info);

    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_gpu_fabric_info() is not supported"
                     " on this system"
                  << std::endl;
      }
      continue;
    } else if (err != AMDSMI_STATUS_SUCCESS && err != AMDSMI_STATUS_NO_DATA) {
      CHK_ERR_ASRT(err)
    } else {
      IF_VERB(STANDARD) {
        if (err == AMDSMI_STATUS_NO_DATA) {
          std::cout << "\t**amdsmi_get_gpu_fabric_info() returned NO_DATA "
                       "(no UALoE sysfs content); BDF may still be valid"
                    << std::endl;
        }
        const auto& v1 = fabric_info.fabric_info.fabric_version.v1;
        std::cout << "\t\tversion:        " << fabric_info.fabric_info.version << "\n"
                  << "\t\taccelerator_id: " << v1.accelerator_id << "\n"
                  << "\t\tfabric_type:    " << v1.fabric_type << "\n"
                  << "\t\tbandwidth:      " << v1.bandwidth << " Mb/s" << "\n"
                  << "\t\tlatency:        " << v1.latency << " ns" << "\n"
                  << "\t\tvpod_id:        " << v1.vpod_id << "\n"
                  << "\t\tvpod_size:      " << v1.vpod_size << "\n"
                  << "\t\tppod_size:      " << v1.ppod_size << "\n"
                  << "\t\taddr_mode:      " << v1.addr_mode << "\n"
                  << "\t\taccel_state:    " << v1.accel_state << "\n";
      }
    }

    // Null-pointer validation
    err = amdsmi_get_gpu_fabric_info(device, nullptr);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    // ── amdsmi_alloc_fabric_telemetry / amdsmi_get_fabric_telemetry_data ──
    IF_VERB(STANDARD) {
      std::cout << "\t** Testing amdsmi_alloc_fabric_telemetry() / "
                   "amdsmi_get_fabric_telemetry_data() / "
                   "amdsmi_free_fabric_telemetry()"
                << std::endl;
    }

    amdsmi_fabric_telemetry_t* tel = nullptr;
    err = amdsmi_alloc_fabric_telemetry(device, kAllCategories, &tel);

    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_alloc_fabric_telemetry() is not supported"
                     " on this machine"
                  << std::endl;
      }
      continue;
    }
    CHK_ERR_ASRT(err)
    ASSERT_NE(tel, nullptr);

    err = amdsmi_get_fabric_telemetry_data(device, tel);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_fabric_telemetry_data() is not supported"
                     " on this machine"
                  << std::endl;
      }
      // Still free the allocated buffer before continuing
      amdsmi_free_fabric_telemetry(device, tel);
      continue;
    }
    CHK_ERR_ASRT(err)

    IF_VERB(STANDARD) {
      // Walk datasets and print a sample of telemetry items
      for (uint32_t cat = 0; cat < AMDSMI_FABRIC_TELEMETRY_CATEGORY_MAX; ++cat) {
        if (!tel->datasets[cat]) continue;
        const auto& ds = *tel->datasets[cat];
        std::cout << "\t\tcategory[" << cat << "]"
                  << "  instances=" << ds.instance_count << "  gen_count=" << ds.generation_count
                  << "\n";

        for (uint32_t inst = 0; inst < ds.instance_count; ++inst) {
          const auto& in = ds.instances[inst];
          std::cout << "\t\t  instance[" << inst << "] " << in.name.text
                    << "  logical_idx=" << in.logical_idx << "  items=" << in.item_count << "\n";

          // ── amdsmi_fabric_telem_id_to_string ─────────────────────────────
          for (uint32_t item = 0; item < in.item_count; ++item) {
            const auto& it = in.items[item];
            const char* name = amdsmi_fabric_telem_id_to_string(it.id);
            std::cout << "\t\t    [" << item << "] id=0x" << std::hex << it.id << std::dec
                      << "  name=" << (name ? name : "NULL") << "  value=" << it.value << "\n";

            // amdsmi_fabric_telem_id_to_string must return a non-null string
            // for any valid id obtained from the driver
            ASSERT_NE(name, nullptr);
          }
        }
      }
    }

    // ── amdsmi_free_fabric_telemetry ──────────────────────────────────────
    err = amdsmi_free_fabric_telemetry(device, tel);
    CHK_ERR_ASRT(err)

    // Null-pointer validation for alloc/free
    err = amdsmi_alloc_fabric_telemetry(device, kAllCategories, nullptr);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    err = amdsmi_free_fabric_telemetry(device, nullptr);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
  }
}
