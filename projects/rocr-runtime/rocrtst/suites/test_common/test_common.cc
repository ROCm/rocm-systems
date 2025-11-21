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
#include <stdint.h>
#include <getopt.h>

#include <iostream>
#include <string>
#include <sstream>

#include "suites/test_common/test_base.h"
#include "suites/test_common/test_common.h"
#include "amd_smi/amdsmi.h"

static const struct option long_options[] = {
  {"iterations", required_argument, nullptr, 'i'},
  {"verbose", required_argument, nullptr, 'v'},
  {"monitor_verbose", required_argument, nullptr, 'm'},

  {nullptr, 0, nullptr, 0}
};
static const char* short_options = "i:v:m:r";

static void PrintHelp(void) {
  std::cout <<
     "Optional RocRTst Arguments:\n"
     "--iterations, -i <number of iterations to execute>; override default, "
         "which varies for each test\n"
     "--rocrtst_help, -r print this help message\n"
     "--verbosity, -v <verbosity level>\n"
     "  Verbosity levels:\n"
     "   0    -- minimal; just summary information\n"
     "   1    -- intermediate; show intermediate values such as intermediate "
                  "perf. data\n"
     "   2    -- progress; show progress displays\n"
     "   >= 3 -- more debug output\n"
     "--monitor_verbosity, -m <monitor verbosity level>\n"
     "  Monitor Verbosity levels:\n"
     "   0    -- don't read or print out any GPU monitor information;\n"
     "   1    -- print out all available monitor information before the first "
                 "test and after each test\n"
     "   >= 2 -- print out even more monitor information (test specific)\n";
}

uint32_t ProcessCmdline(RocrTstGlobals* test, int arg_cnt, char** arg_list) {
  int a;
  int ind = -1;

  assert(test != nullptr);

  while (true) {
    a = getopt_long(arg_cnt, arg_list, short_options, long_options, &ind);

    if (a == -1) {
      break;
    }

    switch (a) {
      case 'i':
        test->num_iterations = std::stoi(optarg);
        break;

      case 'v':
        test->verbosity = std::stoi(optarg);
        break;

      case 'm':
        test->monitor_verbosity = std::stoi(optarg);
        break;

      case 'r':
        PrintHelp();
        return 1;

      default:
        PrintHelp();
        return 1;
    }
  }
  return 0;
}

template<typename T>
static std::string IntegerToString(T intVal, bool hex = true) {
  std::stringstream stream;

  if (hex) {
    stream << "0x" << std::hex << intVal;
  } else {
    stream << std::dec << intVal;
  }
  return stream.str();
}

int DumpMonitorInfo() {
  int ret = 0;
  uint64_t value_u64;
  uint16_t value_u16;
  uint32_t value_u32;
  int64_t value_i64;
  std::string val_str;
  std::vector<std::string> val_vec;
  amdsmi_status_t amdsmi_ret;
  int dump_ret = 0;

  auto print_attr_label =
      [&](std::string attrib) -> bool {
          std::cout << "\t** " << attrib;
          if (ret == -1) {
            std::cout << "not available" << std::endl;
            return false;
          }
          return true;
  };

  auto delim = "\t***********************************";

  std::cout << "\t***** Hardware monitor values *****" << std::endl;
  std::cout << delim << std::endl;
  std::cout.setf(std::ios::dec, std::ios::basefield);

  // Get socket handles
  uint32_t socket_count = AMDSMI_MAX_DEVICES;
  amdsmi_socket_handle socket_handles[AMDSMI_MAX_DEVICES];
  amdsmi_ret = amdsmi_get_socket_handles(&socket_count, socket_handles);
  if (amdsmi_ret != AMDSMI_STATUS_SUCCESS) {
      std::cout << "Failed to get socket count. Error: " << 
                                                      amdsmi_ret << std::endl;
      amdsmi_shut_down();
      return 1;
  }

  uint32_t socket_processors = AMDSMI_MAX_DEVICES;
  uint32_t total_num_processors = 0;

  amdsmi_processor_handle processor_handles[AMDSMI_MAX_DEVICES];
  amdsmi_processor_handle socket_processor_handles[AMDSMI_MAX_DEVICES];

  // Collect devices from sockets
  for (uint32_t socket_idx = 0; socket_idx < socket_count; ++socket_idx) {
    amdsmi_ret = amdsmi_get_processor_handles(socket_handles[socket_idx], 
      &socket_processors, socket_processor_handles);
    if (amdsmi_ret != AMDSMI_STATUS_SUCCESS) {
        std::cout << "amdsmi_get_processor_handles() for socket " << 
                        socket_idx << " returned " << amdsmi_ret << std::endl;
        amdsmi_shut_down();
        return 1;
    }

    for (uint32_t i = 0; i < socket_processors && 
                        total_num_processors + i < AMDSMI_MAX_DEVICES; ++i) {
      processor_handles[total_num_processors + i] = socket_processor_handles[i];
    }
    total_num_processors += socket_processors;
  }

  // Filter for GPU processors
  uint32_t gpu_count = 0;
  for (uint32_t i = 0; i < total_num_processors; ++i) {
      processor_type_t processor_type;
      amdsmi_ret = amdsmi_get_processor_type(processor_handles[i], 
                                                              &processor_type);
      if (amdsmi_ret == AMDSMI_STATUS_SUCCESS && 
                              processor_type == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
          gpu_count++;
      }
  }

  for (uint32_t dindx = 0; dindx < gpu_count; ++dindx) {
    auto print_frequencies = [&](amdsmi_frequencies_t *freqs, 
                                                            std::string label) {
      if (amdsmi_ret != AMDSMI_STATUS_SUCCESS) {
        std::cout << "get frequency call  returned " << amdsmi_ret << std::endl;
        dump_ret = 1;
        return;
      }

      if (print_attr_label(label)) {
        for (uint32_t i = 0; i < freqs->num_supported; ++i) {
          std::cout << "\t**  " << i << ": " <<
                                         freqs->frequency[i]/1000000 << "Mhz";
          if (i == freqs->current) {
            std::cout << " *";
          }

          std::cout << std::endl;
        }
      }
    };
    auto print_val_str = [&](std::string val, std::string label) {
      std::cout << "\t** " << label;
      if (ret != AMDSMI_STATUS_SUCCESS) {
        std::cout << "not available; amdsmi call returned" << amdsmi_ret;
        dump_ret = 1;
      } else {
        std::cout << val;
      }
      std::cout << std:: endl;
    };

    amdsmi_ret = amdsmi_get_gpu_id(processor_handles[dindx], &value_u16);
    print_val_str(IntegerToString(value_u16), "Device ID: ");

    amdsmi_dev_perf_level_t perf;
    std::string perf_str;
    amdsmi_ret = amdsmi_get_gpu_perf_level(processor_handles[dindx], &perf);
    switch (perf) {
      case AMDSMI_DEV_PERF_LEVEL_AUTO:
        perf_str = "auto";
        break;
      default:
        perf_str = "unknown";
    }
    print_val_str(perf_str, "Performance Level: ");

    uint32_t overdrive_level;
    amdsmi_ret = amdsmi_get_gpu_overdrive_level(processor_handles[dindx], 
                                                            &overdrive_level);

    print_val_str(IntegerToString(value_u32, false) + "%", "OverDrive Level: ");

    amdsmi_frequencies_t freqs;
    amdsmi_ret = amdsmi_get_clk_freq(processor_handles[dindx], 
                                                AMDSMI_CLK_TYPE_SYS, &freqs);

    print_frequencies(&freqs, "Supported GPU clock frequencies:\n");

    amdsmi_ret = amdsmi_get_clk_freq(processor_handles[dindx], 
                                                AMDSMI_CLK_TYPE_MEM, &freqs);
    print_frequencies(&freqs, "Supported GPU Memory clock frequencies:\n");

    amdsmi_board_info_t board_info;
    amdsmi_get_gpu_board_info(processor_handles[dindx], &board_info);
    print_val_str(board_info.product_name, "Monitor name: ");
    
    amdsmi_ret = amdsmi_get_temp_metric(processor_handles[dindx], 
                AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, &value_i64);
    print_val_str(IntegerToString(value_i64/1000, false) + "C",
                                                            "Temperature: ");

    amdsmi_ret = amdsmi_get_gpu_fan_speed(processor_handles[dindx], 
                                                                0, &value_i64);
    if (ret != AMDSMI_STATUS_SUCCESS) {
        std::cout << "not available; amdsmi call returned" << amdsmi_ret;
        dump_ret = 1;
    }
    amdsmi_ret = amdsmi_get_gpu_fan_speed_max(processor_handles[dindx], 
                                                                0, &value_u64);
    if (ret != AMDSMI_STATUS_SUCCESS) {
        std::cout << "not available; amdsmi call returned" << amdsmi_ret;
        dump_ret = 1;
    }
    if (print_attr_label("Current Fan Speed: ")) {
      std::cout << static_cast<float>(value_i64)/value_u64 * 100 << "% (" <<
          value_i64 << "/" << value_u64 << ")" << std::endl;
    }

    std::cout << "\t=======" << std::endl;
  }
  std::cout << delim << std::endl;
  return dump_ret;
}
