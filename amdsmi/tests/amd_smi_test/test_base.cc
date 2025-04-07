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

#include <gtest/gtest.h>
#include <cassert>
#include <limits>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "test_base.h"
#include "test_common.h"

static const int kOutputLineLength = 80;
static const char kLabelDelimiter[] = "####";
static const char kDescriptionLabel[] = "TEST DESCRIPTION";
static const char kTitleLabel[] = "TEST NAME";
static const char kRunLabel[] = "TEST EXECUTION";
static const char kCloseLabel[] = "TEST CLEAN UP";
static const char kResultsLabel[] = "TEST RESULTS";

// This one is used outside this file
const char kSetupLabel[] = "TEST SETUP";

TestBase::TestBase() : setup_failed_(false) {
}
TestBase::~TestBase() = default;

void TestBase::MakeHeaderStr(const char *inStr,
                                   std::string *outStr) const {
  assert(outStr != nullptr);
  assert(inStr != nullptr);
  outStr->clear();
  IF_VERB(STANDARD) {
    *outStr = kLabelDelimiter;
    *outStr += " ";
    *outStr += inStr;
    *outStr += " ";
    *outStr += kLabelDelimiter;
  }
}

void TestBase::SetUp(void) {
  SetUp(AMDSMI_INIT_AMD_GPUS);
}

void TestBase::SetUp(uint64_t init_flags) {
  std::string label;
  amdsmi_status_t err;

  IF_VERB(STANDARD) {
    MakeHeaderStr(kSetupLabel, &label);
    printf("\n\t%s\n", label.c_str());
  }

  if (init_flags) {
    err = amdsmi_init(init_flags);
  } else {
    err = amdsmi_init(init_options());
  }

  if (err != AMDSMI_STATUS_SUCCESS) {
    setup_failed_ = true;
  }
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);


  err = amdsmi_get_socket_handles(&socket_count_, nullptr);
  if (err != AMDSMI_STATUS_SUCCESS) {
    setup_failed_ = true;
  }
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);

  // allocate memory
  sockets_.resize(socket_count_);
  err = amdsmi_get_socket_handles(&socket_count_, &sockets_[0]);
  if (err != AMDSMI_STATUS_SUCCESS) {
    setup_failed_ = true;
  }
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);

  // collect devices from sockets
  num_monitor_devs_ = 0;
  for (uint32_t i=0; i < socket_count_; i++) {
    // Get all devices of the socket
    uint32_t device_count = 0;
    err = amdsmi_get_processor_handles(sockets_[i],
            &device_count, nullptr);
    if (err != AMDSMI_STATUS_SUCCESS) {
      setup_failed_ = true;
    }
    ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);

    std::vector<amdsmi_processor_handle> processor_handles(device_count);
    err = amdsmi_get_processor_handles(sockets_[i],
            &device_count, &processor_handles[0]);
    if (err != AMDSMI_STATUS_SUCCESS) {
      setup_failed_ = true;
    }
    ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
    // store the device handles for following tests
    for (uint32_t j=0; j < device_count; j++) {
      if (num_monitor_devs_ >= MAX_MONITOR_DEVICES) {
        setup_failed_ = true;
        ASSERT_EQ(AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS, AMDSMI_STATUS_SUCCESS);
      }
      processor_handles_[num_monitor_devs_] = processor_handles[j];
      num_monitor_devs_++;
    }
  }

  if (num_monitor_devs_ == 0) {
    IF_VERB(STANDARD) {
      std::cout << "No monitor devices found on this machine." << std::endl;
      std::cout << "No AMD SMI tests can be run." << std::endl;
    }
  }
}

void TestBase::PrintDeviceHeader(amdsmi_processor_handle dv_ind) {
  amdsmi_status_t err;
  uint16_t val_ui16;
  uint32_t val_ui32;
  amdsmi_asic_info_t info;

  err = smi_amdgpu_get_device_count(&val_ui32);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
    std::cout << "\t**Total Devices: " << val_ui32 << std::endl;
  }

  err = smi_amdgpu_get_device_index(dv_ind, &val_ui32);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
    std::cout << "\t**AMD SMI Device index: " << val_ui32 << std::endl;
  }

  IF_VERB(STANDARD) {
    std::cout << "\t**Device handle: " << dv_ind << std::endl;
  }
  err = amdsmi_get_gpu_id(dv_ind, &val_ui16);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
    std::cout << "\t**Device ID: 0x" << std::hex << val_ui16 << std::endl;
  }

  err = amdsmi_get_gpu_revision(dv_ind, &val_ui16);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
      std::cout << "\t**Device Revision ID: 0x" << std::hex <<
          val_ui16 << std::endl;
  }

  amdsmi_board_info_t board_info;
  err = amdsmi_get_gpu_board_info(dv_ind, &board_info);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
    std::cout << "\t**Device name: " << board_info.product_name  << std::endl;

    err = amdsmi_get_gpu_asic_info(dv_ind, &info);
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << "\t**Device Vendor ID: 0x" << std::hex <<
          info.vendor_id << std::endl;
    }
  }

  amdsmi_asic_info_t asic_info;
  err = amdsmi_get_gpu_asic_info(dv_ind, &asic_info);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
    std::cout << "\t**Market name: " << asic_info.market_name  << std::endl;
    std::cout << "\t**ASIC serial: 0x" << std::hex << asic_info.asic_serial  << std::endl;
    std::cout << "\t**Target GFX Version: gfx" << asic_info.target_graphics_version  << std::endl;
  }

  err = amdsmi_get_gpu_subsystem_id(dv_ind, &val_ui16);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
    std::cout << "\t**Subsystem ID: 0x" << std::hex << val_ui16 << std::endl;
    std::cout << "\t**Subsystem Vendor ID: 0x" << std::hex
          << info.subvendor_id << std::endl;
  }

  std::cout << std::setbase(10);
}
void TestBase::Run(void) {
  std::string label;
  IF_VERB(STANDARD) {
    MakeHeaderStr(kRunLabel, &label);
    printf("\n\t%s\n", label.c_str());
  }
  ASSERT_TRUE(!setup_failed_);
}

void TestBase::Close(void) {
  std::string label;
  IF_VERB(STANDARD) {
    MakeHeaderStr(kCloseLabel, &label);
    printf("\n\t%s\n", label.c_str());
  }
  amdsmi_status_t err = amdsmi_shut_down();
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
}

void TestBase::DisplayResults(void) const {
  std::string label;
  IF_VERB(STANDARD) {
    MakeHeaderStr(kResultsLabel, &label);
    printf("\n\t%s\n", label.c_str());
  }
}

void TestBase::DisplayTestInfo(void) {
  IF_VERB(STANDARD) {
    printf("#########################################"
                                  "######################################\n");

    std::string label;
    MakeHeaderStr(kTitleLabel, &label);
    printf("\n\t%s\n%s\n", label.c_str(), title().c_str());

    if (verbosity() >= VERBOSE_STANDARD) {
      MakeHeaderStr(kDescriptionLabel, &label);
      printf("\n\t%s\n%s\n", label.c_str(), description().c_str());
    }
  }
}

void TestBase::set_description(std::string d) {
  int le = kOutputLineLength - 4;

  description_ = d;
  size_t endlptr;

  for (size_t i = le; i < description_.size(); i += le) {
    endlptr = description_.find_last_of(' ', i);
    description_.replace(endlptr, 1, "\n");
    i = endlptr;
  }
}

TestBase::AcceleratorProfileConfig TestBase::getAvailableProfileConfigs(
                                      uint32_t device_index,
                                      amdsmi_accelerator_partition_profile_t current_profile,
                                      amdsmi_accelerator_partition_profile_config_t config,
                                      bool isVerbose) {
  AcceleratorProfileConfig profile_config = {};
  profile_config.number_of_profiles = config.num_profiles;
  profile_config.original_profile_type = current_profile.profile_type;
  profile_config.original_profile_index = current_profile.profile_index;
  profile_config.original_profile_type_str =
    partition_types_map.at(current_profile.profile_type);
  profile_config.available_profiles = std::vector<amdsmi_accelerator_partition_type_t>(
    config.num_profiles);
  profile_config.available_profile_str = std::vector<std::string>(config.num_profiles);
  profile_config.available_profile_indices = std::vector<uint32_t>(config.num_profiles);
  for (uint32_t i = 0; i < config.num_profiles; i++) {
    std::string profile_type_str = "N/A";
    profile_config.available_profiles[i] = config.profiles[i].profile_type;
    profile_config.available_profile_str[i].clear();
    profile_config.available_profile_str[i] =
      partition_types_map.at(config.profiles[i].profile_type);
    profile_config.available_profile_indices[i] = config.profiles[i].profile_index;
  }

  if (isVerbose) {
    const uint32_t kMAX_UINT32 = std::numeric_limits<uint32_t>::max();
    std::cout << "\t**[Device #" << device_index << "] Profile Configs: ";
    std::cout << "\n\t\t**Original Profile Index: "
              << (profile_config.original_profile_index == kMAX_UINT32 ?
                  "N/A" : std::to_string(profile_config.original_profile_index))
              << "\n\t\t**Original Profile Type: "
              << profile_config.original_profile_type_str
              << "\n\t\t**Original profile: " << profile_config.original_profile_type
              << " (" << accelerator_types_map.at(profile_config.original_profile_type) << ")"
              << "\n\t\t**Number of Profiles: " << profile_config.number_of_profiles
              << "\n\t\t**Available_profiles: ";
  }
  std::string available_profiles_str = "N/A\n";
  for (uint32_t j = 0; j < profile_config.number_of_profiles; j++) {
    if (available_profiles_str == "N/A\n") {
      available_profiles_str.clear();
    }

    if (j + 1 >= profile_config.number_of_profiles) {
      available_profiles_str += ("\n\t\t\tProfile[profile_index: "
        + std::to_string(profile_config.available_profile_indices[j])
        + "]: " + profile_config.available_profile_str[j] + "\n");
    } else {
      available_profiles_str += ("\n\t\t\tProfile[profile_index: "
        + std::to_string(profile_config.available_profile_indices[j])
        + "]: " + profile_config.available_profile_str[j] + ", ");
    }
  }
  if (isVerbose) {
    std::cout << available_profiles_str;
  }
  return profile_config;
}

void TestBase::waitForUserInput() {
  for (;;) {
    std::cout << "\n\t**Press any key to continue**" << std::endl;
    int input = std::cin.get();
    if (input == EOF) {
      std::cout << "EOF detected. Exiting." << std::endl;
      return;
    }
    char input_char = static_cast<char>(input);
    std::cout << "User entered: " << input_char << std::endl;
    if (input_char == '\n') {
      return;
    }
  }
}

uint32_t TestBase::promptNumDevicesToTest(uint32_t current_num_devices) {
  uint32_t return_value = 0;
  std::cout << "**How many devices would you like to test? (0 to skip): ";
  std::string devices_to_test = "";
  do {
    int input = std::cin.get();
    if (input == EOF) {
      std::cout << "EOF detected. Exiting." << std::endl;
      return 0;
    }
    char input_char = static_cast<char>(input);
    if (input_char == '\n') {
      break;
    }
    if (input_char >= '0' && input_char <= '9') {
      devices_to_test += input_char;
    } else {
      std::cout << "Invalid input. Please enter a number between 0 and "
                << current_num_devices << std::endl;
    }
  } while (true);

  return_value = std::stoi(devices_to_test);
  if (return_value > current_num_devices) {
    std::cout << "Invalid input. Please enter a number between 0 and "
              << current_num_devices << std::endl;
    return 0;
  }
  return return_value;
}

std::string TestBase::getResourceType(amdsmi_accelerator_partition_resource_type_t resource_type) {
  std::string resource_type_str = "";
  switch (resource_type) {
    case AMDSMI_ACCELERATOR_XCC:
      resource_type_str = "XCC";
      break;
    case AMDSMI_ACCELERATOR_ENCODER:
      resource_type_str = "ENCODER";
      break;
    case AMDSMI_ACCELERATOR_DECODER:
      resource_type_str = "DECODER";
      break;
    case AMDSMI_ACCELERATOR_DMA:
      resource_type_str = "DMA";
      break;
    case AMDSMI_ACCELERATOR_JPEG:
      resource_type_str = "JPEG";
      break;
    case AMDSMI_ACCELERATOR_MAX:
      resource_type_str = "MAX";
      break;
    default:
      resource_type_str = "N/A";
      break;
  }
  return resource_type_str;
}

