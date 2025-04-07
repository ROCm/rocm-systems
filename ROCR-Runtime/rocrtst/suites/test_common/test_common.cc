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
#include "rocm_smi/rocm_smi.h"

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
  rsmi_status_t rsmi_ret;
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

  uint32_t num_mon_devices;
  rsmi_ret = rsmi_num_monitor_devices(&num_mon_devices);
  if (rsmi_ret != RSMI_STATUS_SUCCESS) {
    std::cout << "rsmi_num_monitor_device() returned" << rsmi_ret << std::endl;
    return 1;
  }

  for (uint32_t dindx = 0; dindx < num_mon_devices; ++dindx) {
    auto print_frequencies = [&](rsmi_frequencies *freqs, std::string label) {
      if (rsmi_ret != RSMI_STATUS_SUCCESS) {
        std::cout << "get frequency call  returned " << rsmi_ret << std::endl;
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
      if (ret != RSMI_STATUS_SUCCESS) {
        std::cout << "not available; rsmi call returned" << rsmi_ret;
        dump_ret = 1;
      } else {
        std::cout << val;
      }
      std::cout << std:: endl;
    };

    rsmi_ret = rsmi_dev_id_get(dindx, &value_u16);
    print_val_str(IntegerToString(value_u16), "Device ID: ");

    rsmi_dev_perf_level perf;
    std::string perf_str;
    rsmi_ret = rsmi_dev_perf_level_get(dindx, &perf);
    switch (perf) {
      case RSMI_DEV_PERF_LEVEL_AUTO:
        perf_str = "auto";
        break;
      default:
        perf_str = "unknown";
    }
    print_val_str(perf_str, "Performance Level: ");

    rsmi_ret = rsmi_dev_overdrive_level_get(dindx, &value_u32);

    print_val_str(IntegerToString(value_u32, false) + "%", "OverDrive Level: ");

    rsmi_frequencies freqs;
    rsmi_ret = rsmi_dev_gpu_clk_freq_get(dindx, RSMI_CLK_TYPE_SYS, &freqs);

    print_frequencies(&freqs, "Supported GPU clock frequencies:\n");

    rsmi_ret = rsmi_dev_gpu_clk_freq_get(dindx, RSMI_CLK_TYPE_MEM, &freqs);
    print_frequencies(&freqs, "Supported GPU Memory clock frequencies:\n");


    char mon_name[32];
    rsmi_ret = rsmi_dev_name_get(dindx, mon_name, 32);
    print_val_str(mon_name, "Monitor name: ");
    rsmi_ret = rsmi_dev_temp_metric_get(dindx, 0,
                                              RSMI_TEMP_CURRENT, &value_i64);
    print_val_str(IntegerToString(value_i64/1000, false) + "C",
                                                            "Temperature: ");

    rsmi_ret = rsmi_dev_fan_speed_get(dindx, 0, &value_i64);
    if (ret != RSMI_STATUS_SUCCESS) {
      std::cout << "not available; rsmi call returned" << rsmi_ret;
      dump_ret = 1;
    }
    rsmi_ret = rsmi_dev_fan_speed_max_get(dindx, 0, &value_u64);
    if (ret != RSMI_STATUS_SUCCESS) {
      std::cout << "not available; rsmi call returned" << rsmi_ret;
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
