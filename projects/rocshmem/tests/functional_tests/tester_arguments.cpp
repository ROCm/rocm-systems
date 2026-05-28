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
#include <rocshmem/rocshmem.hpp>

#include "tester.hpp"

using namespace rocshmem;

TesterArguments::TesterArguments(int argc, char *argv[]) {
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
      algorithm = atoi(argv[i]);
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
    case SyncAllTestType:
    case WAVESyncAllTestType:
    case WGSyncAllTestType:
    case TeamSyncTestType:
      min_msg_size = 8;
      max_msg_size = 8;
      break;
    case PingPongTestType:
    case ShmemPtrTestType:
      min_msg_size = 4;
      max_msg_size = 4;
      break;
    case RandomAccessTestType:
      min_msg_size = 4;
      break;
    case TeamFCollectTestType:
    case TeamAllToAllTestType:
    case TeamBroadcastTestType:
      min_msg_size = 8;
      break;
    case TeamCtxInfraTestType:
    case TeamCtxInfraTestSingleType:
    case TeamCtxInfraTestBlockType:
    case TeamCtxInfraTestOddEvenType:
      max_msg_size = min_msg_size;
      break;
    case PutNBIMRTestType:
      min_msg_size = max_msg_size;
      break;
    case PTestType:
    case GTestType:
      min_msg_size = 1;
      max_msg_size = 1;
    default:
      break;
  }
}

void TesterArguments::show_usage(std::string executable_name) {
  std::cout << "Usage: " << executable_name << std::endl;
  std::cout << "\t-t <number of rocshmem service threads>\n";
  std::cout << "\t-w <number of workgroups>\n";
  std::cout << "\t-s <maximum message size (in bytes)>\n";
  std::cout << "\t-a <algorithm number to test>\n";
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
}

void TesterArguments::get_arguments() {
  numprocs = rocshmem_n_pes();
  myid = rocshmem_my_pe();

  TestType type = (TestType)algorithm;
  if ((type != BarrierAllTestType) && (type != WAVEBarrierAllTestType) &&
      (type != WGBarrierAllTestType) && (type != SyncAllTestType) &&
      (type != WAVESyncAllTestType) && (type != WGSyncAllTestType) &&
      (type != TeamSyncTestType) && (type != TeamWAVESyncTestType) &&
      (type != TeamWGSyncTestType) && (type != TeamAllToAllTestType) &&
      (type != TeamFCollectTestType) && (type != TeamReductionTestType) &&
      (type != TeamBroadcastTestType) && (type != PingAllTestType) &&
      (type != TeamBarrierTestType) && (type != TeamWAVEBarrierTestType) &&
      (type != TeamWGBarrierTestType) && (type != TeamCtxInfraTestBlockType) &&
      (type != TeamCtxInfraTestOddEvenType)) {
    if (numprocs != 2) {
      if (myid == 0) {
        std::cerr << "This test requires exactly two processes, we have "
                  << numprocs << "\n";
      }
      exit(-1);
    }
  }
}
