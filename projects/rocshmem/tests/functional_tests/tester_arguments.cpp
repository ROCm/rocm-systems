/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "tester_arguments.hpp"

#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <rocshmem/rocshmem.hpp>

#include "tester.hpp"

using namespace rocshmem;

TesterArguments::TesterArguments(int argc, char *argv[]) {
  if (argc > 0 && argv[0] != nullptr) {
    executable_name = argv[0];
  }
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-t") {
      i++;
      num_threads = atoi(argv[i]);
    } else if (arg == "-w") {
      i++;
      num_wgs = atoi(argv[i]);
    } else if (arg == "-s") {
      i++;
      max_msg_size = atoll(argv[i]);
    } else if (arg == "-a") {
      i++;
      std::string a_arg = argv[i];

      // If it's a pure integer, use it directly; otherwise look up by name
      bool is_number = !a_arg.empty() && std::all_of(a_arg.begin(), a_arg.end(), ::isdigit);
      if (is_number) {
        algorithm = std::stoul(a_arg);
      } else {
        // Build map from lowercased CamelName to enum value.
        // Generated automatically from ROCSHMEM_FOREACH_TEST_TYPE so it stays in
        // sync with the enum without a separate hand-maintained list.
        auto to_lower = [](std::string s) {
          std::transform(s.begin(), s.end(), s.begin(), ::tolower);
          return s;
        };
        static const std::unordered_map<std::string, unsigned> name_to_algo = []() {
          std::unordered_map<std::string, unsigned> m;
#define _ADD_TEST_ENTRY(name, val) \
          { std::string k = #name; \
            std::transform(k.begin(), k.end(), k.begin(), ::tolower); \
            m[k] = val; }
          ROCSHMEM_FOREACH_TEST_TYPE(_ADD_TEST_ENTRY)
#undef _ADD_TEST_ENTRY
          return m;
        }();

        auto it = name_to_algo.find(to_lower(a_arg));
        if (it == name_to_algo.end()) {
          std::cerr << "Unknown test name: " << a_arg << "\n";
          show_usage(argv[0]);
          exit(-1);
        }
        algorithm = it->second;
      }
    } else if (arg == "-v") {
      i++;
      max_volume_size = atoi(argv[i]);
    } else if (arg == "-z") {
      i++;
      wg_size = atoi(argv[i]);
    } else if (arg == "-c") {
      i++;
      coal_coef = atoi(argv[i]);
    } else if (arg == "-o") {
      i++;
      op_type = atoi(argv[i]);
    } else if (arg == "-ta") {
      i++;
      thread_access = atoi(argv[i]);
    } else if (arg == "-x") {
      i++;
      shmem_context = atoi(argv[i]);
    } else if (arg == "-m") {
      int atomics_addr_mode = atoi(argv[i]);
      if(atomics_addr_mode >= static_cast<int>(AddrMode::PerGrid) &&
         atomics_addr_mode <= static_cast<int>(AddrMode::PerBlock)) {
         addr_mode = static_cast<AddrMode>(atomics_addr_mode);
      }
      i++;
    } else if (arg == "-n") {
      i++;
      loop = atoi(argv[i]);
      loop_large = loop;
    } else if (arg == "-nloop") {
      i++;
      loop = atoi(argv[i]);
    } else if (arg == "-nlarge") {
      i++;
      loop_large = atoi(argv[i]);
    } else if (arg == "-nskip") {
      i++;
      skip = atoi(argv[i]);
    } else if (arg == "-noverif") {
      verif = false;
    } else if (arg == "-localbuftype") {
      i++;

      if (argc < i + 1) {
        fprintf(stderr, "Invalid arguments for -localbuftype.\n");
        exit(-1);
      }

      if (std::string(argv[i]) == "heap") {
        local_buf_type = USER_BUF_TYPE_HEAP;
      } else if (std::string(argv[i]) == "host") {
        local_buf_type = USER_BUF_TYPE_HOST;
      } else if (std::string(argv[i]) == "device") {
        local_buf_type = USER_BUF_TYPE_DEVICE;
      } else if (std::string(argv[i]) == "fine") {
        local_buf_type = USER_BUF_TYPE_FINE;
      } else if (std::string(argv[i]) == "uncached") {
        local_buf_type = USER_BUF_TYPE_UNCACHED;
      } else if (std::string(argv[i]) == "managed") {
        local_buf_type = USER_BUF_TYPE_MANAGED;
      } else {
        fprintf(stderr, "Invalid local buffer type. "
                        "Please use one of [heap, host, device, fine, uncached, managed]. "
                        "Defaulting to heap\n");
        local_buf_type = USER_BUF_TYPE_HEAP;
      }
    } else {
      show_usage(argv[0]);
      exit(-1);
    }
  }

  TestType type = (TestType)algorithm;

  switch (type) {
    case AMO_FAddTestType:
    case AMO_AddTestType:
    case AMO_SetTestType:
    case AMO_SwapTestType:
    case AMO_FetchAndTestType:
    case AMO_AndTestType:
    case AMO_FetchOrTestType:
    case AMO_OrTestType:
    case AMO_FetchXorTestType:
    case AMO_XorTestType:
    case AMO_FCswapTestType:
    case AMO_CswapTestType:
    case AMO_FIncTestType:
    case AMO_IncTestType:
    case AMO_FetchTestType:
    case BarrierAllTestType:
    case WAVEBarrierAllTestType:
    case WGBarrierAllTestType:
    case TeamBarrierTestType:
    case TeamWAVEBarrierTestType:
    case TeamWGBarrierTestType:
    case BarrierAllOnStreamTestType:
    case SyncAllTestType:
    case WAVESyncAllTestType:
    case WGSyncAllTestType:
    case TeamSyncTestType:
    case SignalWaitUntilOnStreamTestType:
      min_msg_size = 8;
      max_msg_size = 8;
      break;
    case PingPongTestType:
    case PingAllTestType:
    case ShmemPtrTestType:
      min_msg_size = 4;
      max_msg_size = 4;
      break;
    case RandomAccessTestType:
    case TeamAlltoallmemOnStreamTestType:
    case TeamBroadcastmemOnStreamTestType:
      min_msg_size = 4;
      break;
    case TeamFCollectTestType:
    case TeamAllToAllTestType:
    case TeamAllToAllvTestType:
    case TeamBroadcastTestType:
      min_msg_size = 8;
      break;
    case TeamCtxInfraTestType:
    case TeamCtxInfraSingleTestType:
    case TeamCtxInfraBlockTestType:
    case TeamCtxInfraOddEvenTestType:
    case TeamCtxSubsetParentInfraTestType:
      max_msg_size = min_msg_size;
      break;
    case FenceOrderPutWaveNbiChunksTestType:
      min_msg_size = 16;  // must be >= STRESS_NUM_CHUNKS for chunk_size >= 1
      break;
    case PutNBIMRTestType:
      min_msg_size = max_msg_size;
      break;
    case PTestType:
    case GTestType:
      min_msg_size = 1;
      max_msg_size = 1;
      break;
    case FloodPutTestType:
    case FloodPutNBITestType:
    case FloodPTestType:
    case FloodGetTestType:
    case FloodGetNBITestType:
    case FloodGTestType:
    case FloodAddTestType:
    case FloodFAddTestType:
    case FloodWaitAmoTestType:
      min_msg_size = max_msg_size = 8;
      break;
    default:
      break;
  }
}

void TesterArguments::show_usage(std::string executable_name) {
  std::cout << "Usage: " << executable_name << std::endl;
  std::cout << "\t-t <number of rocshmem service threads>\n";
  std::cout << "\t-w <number of workgroups>\n";
  std::cout << "\t-s <maximum message size (in bytes)>\n";
  std::cout << "\t-v <maximum per origin volume (in bytes)>\n";
  std::cout << "\t-a <algorithm number or test name to test>\n";
  std::cout << "\t-z <WorkGroup Size>\n";
  std::cout << "\t-c <Coalescing Coefficient>\n";
  std::cout << "\t-o <Operation type for the random_access test>\n";
  std::cout << "\t-ta <Number of Thread Accessing the communication>\n";
  std::cout << "\t-x <shmem context>\n";
  std::cout << "\t-m Atomics Address mode\n";
  std::cout << "\t-n Set both loop and loop_large count\n";
  std::cout << "\t-nloop Set loop count\n";
  std::cout << "\t-nlarge Set loop_large count\n";
  std::cout << "\t-nskip Set skip/warmup count\n";
  std::cout << "\t-noverif disable buffer verification\n";
}

void TesterArguments::get_arguments() {
  numprocs = rocshmem_n_pes();
  myid = rocshmem_my_pe();

  TestType type = (TestType)algorithm;
  // Check if test requires exactly 2 PEs
  // Tests that support arbitrary number of PEs are excluded
  bool requires_two_pes = true;
  switch (type) {
    // Collective/barrier tests - support any number of PEs
    case BarrierAllTestType:
    case WAVEBarrierAllTestType:
    case WGBarrierAllTestType:
    case SyncAllTestType:
    case WAVESyncAllTestType:
    case WGSyncAllTestType:
    case TeamSyncTestType:
    case TeamWAVESyncTestType:
    case TeamWGSyncTestType:
    case TeamAllToAllTestType:
    case TeamAllToAllvTestType:
    case TeamFCollectTestType:
    case TeamReductionTestType:
    case TeamBroadcastTestType:
    case PingAllTestType:
    case TeamBarrierTestType:
    case TeamWAVEBarrierTestType:
    case TeamWGBarrierTestType:
    case TeamCtxInfraBlockTestType:
    case TeamCtxInfraOddEvenTestType:
    case TeamCtxSubsetParentInfraTestType:
    // On-stream tests - support any number of PEs
    case TeamAlltoallmemOnStreamTestType:
    case BarrierAllOnStreamTestType:
    case QuietOnStreamTestType:
    case SyncAllOnStreamTestType:
    case TeamBroadcastmemOnStreamTestType:
    case GetmemOnStreamTestType:
    case PutmemOnStreamTestType:
    case PutmemSignalOnStreamTestType:
    case SignalWaitUntilOnStreamTestType:
    case FloodPutTestType:
    case FloodPutNBITestType:
    case FloodPTestType:
    case FloodGetTestType:
    case FloodGetNBITestType:
    case FloodGTestType:
    case FloodAddTestType:
    case FloodFAddTestType:
    case FloodWaitAmoTestType:
    case DeviceBitcodeTestType:
    case TeamCtxSharedInfraTestType:
    case FenceOrderFanoutTestType:
      requires_two_pes = false;
      break;
    default:
      break;
  }

  if (requires_two_pes && numprocs != 2) {
    if (myid == 0) {
      std::cerr << "This test requires exactly two processes, we have "
                << numprocs << "\n";
    }
    exit(-1);
  }
}
