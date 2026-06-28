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

#include "tester.hpp"

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <rocshmem/rocshmem.hpp>
#include <vector>

#include "amo_bitwise_tester.hpp"
#include "host_rma_tester.hpp"
#include "amo_extended_tester.hpp"
#include "amo_standard_tester.hpp"
#include "default_ctx_primitive_tester.hpp"
#include "barrier_all_tester.hpp"
#include "barrier_all_on_stream_tester.hpp"
#include "quiet_on_stream_tester.hpp"
#include "empty_tester.hpp"
#include "getmem_on_stream_tester.hpp"
#include "putmem_on_stream_tester.hpp"
#include "putmem_signal_on_stream_tester.hpp"
#include "signal_wait_until_on_stream_tester.hpp"
#include "ping_all_tester.hpp"
#include "ping_pong_tester.hpp"
#include "primitive_tester.hpp"
#include "random_access_tester.hpp"
#include "shmem_ptr_tester.hpp"
#include "signaling_operations_tester.hpp"
#include "sync_all_tester.hpp"
#include "team_sync_tester.hpp"
#include "team_alltoall_tester.hpp"
#include "team_alltoallv_tester.hpp"
#include "team_alltoallmem_on_stream_tester.hpp"
#include "team_broadcastmem_on_stream_tester.hpp"
#include "team_barrier_tester.hpp"
#include "team_broadcast_tester.hpp"
#include "team_ctx_infra_tester.hpp"
#include "team_ctx_primitive_tester.hpp"
#include "team_fcollect_tester.hpp"
#include "team_reduction_tester.hpp"
#include "wavefront_primitives.hpp"
#include "workgroup_primitives.hpp"
#include "flood_tester.hpp"
#include "flood_amo_tester.hpp"
#include "hipmodule_init_tester.hpp"
#include "device_bitcode_tester.hpp"
#include "library_info_tester.hpp"
#include "fence_ordering_tester.hpp"
#include "tile_rma_tester.hpp"
#include "tile_broadcast_tester.hpp"
#include "tile_allgather_tester.hpp"
#include "reduce_on_stream_tester.hpp"
#include "host_ctx_create_tester.hpp"
#include "team_split_2d_tester.hpp"
#include "host_team_sync_barrier_tester.hpp"

#include "backend_bc.hpp"
extern Backend* backend;

Tester::Tester(TesterArguments args) : args(args) {
  _type = (TestType)args.algorithm;
  _shmem_context = args.shmem_context;
  CHECK_HIP(hipGetDevice(&device_id));
  CHECK_HIP(hipGetDeviceProperties(&deviceProps, device_id));
  wf_size = deviceProps.warpSize;
  num_warps = (args.wg_size - 1) / wf_size + 1;
  CHECK_HIP(hipStreamCreate(&stream));
  CHECK_HIP(hipEventCreate(&start_event));
  CHECK_HIP(hipEventCreate(&stop_event));
  CHECK_HIP(hipDeviceGetAttribute(&wall_clk_rate,
    hipDeviceAttributeWallClockRate, device_id));
  num_timers = args.num_wgs;
  switch (_type) {
    case WAVEGetTestType:
    case WAVEGetNBITestType:
    case WAVEPutTestType:
    case WAVEPutNBITestType:
      num_timers = args.num_wgs * num_warps;
      break;
    default:
      break;
  }
  CHECK_HIP(hipMalloc((void**)&timer, sizeof(long long int) * num_timers));
  CHECK_HIP(hipMalloc((void**)&start_time, sizeof(long long int) * num_timers));
  CHECK_HIP(hipMalloc((void**)&end_time, sizeof(long long int) * num_timers));
  CHECK_HIP(hipHostMalloc((void**)&verification_error, sizeof(bool)));
  *verification_error = false;

  batch_size = (args.batch > 0) ? args.batch : args.loop;

  max_msg_size = args.max_msg_size;
  if (args.max_volume_size) {
    switch (_type) {
      case GetTestType:
      case GetNBITestType:
      case PutTestType:
      case PutNBITestType:
      case PutSignalTestType:
      case PutSignalNBITestType:
      case DefaultCTXGetTestType:
      case DefaultCTXGetNBITestType:
      case DefaultCTXPutTestType:
      case DefaultCTXPutNBITestType:
      case DefaultCTXPTestType:
      case DefaultCTXGTestType:
        max_msg_size = args.max_volume_size / args.num_wgs / args.wg_size;
        break;
      case WAVEGetTestType:
      case WAVEGetNBITestType:
      case WAVEPutTestType:
      case WAVEPutNBITestType:
      case WAVEPutSignalTestType:
      case WAVEPutSignalNBITestType:
        max_msg_size = args.max_volume_size / args.num_wgs / num_warps;
        break;
      case WGGetTestType:
      case WGGetNBITestType:
      case WGPutTestType:
      case WGPutNBITestType:
      case WGPutSignalTestType:
      case WGPutSignalNBITestType:
        max_msg_size = args.max_volume_size / args.num_wgs;
        break;
      case TeamBroadcastTestType:
      case TeamReductionTestType:
      case TeamFCollectTestType:
      case CollectTestType:
      case TeamAllToAllTestType:
      case TeamAllToAllvTestType:
      case TeamAlltoallmemOnStreamTestType:
        max_msg_size = args.max_volume_size / args.num_wgs / args.numprocs;
        break;
      default:
        break;
    }
    if (max_msg_size == 0) {
      if (args.myid == 0) {
        std::cerr << "Requested communication volume is smaller than what is required to send at least 1 byte per operation, adjust -w, -z, and -v to match, or remove -v.";
      }
      exit(-1);
    }
  }
}

Tester::~Tester() {
  CHECK_HIP(hipFree(end_time));
  CHECK_HIP(hipFree(start_time));
  CHECK_HIP(hipFree(timer));
  CHECK_HIP(hipEventDestroy(stop_event));
  CHECK_HIP(hipEventDestroy(start_event));
  CHECK_HIP(hipStreamDestroy(stream));
  CHECK_HIP(hipFree(verification_error));
}

std::vector<Tester*> Tester::create(TesterArguments args) {
  int rank = args.myid;
  std::vector<Tester*> testers;
  std::string test_name;

  BackendType backend_type = rocshmem_query_backend_type();
  TestType type = (TestType)args.algorithm;

  switch (type) {
    case InitTestType:
      test_name = "Init";
      testers.push_back(new EmptyTester(args));
      break;
    case GetTestType:
      test_name = "Blocking Gets";
      testers.push_back(new PrimitiveTester(args));
      break;
    case GetNBITestType:
      test_name = "Non-Blocking Gets";
      testers.push_back(new PrimitiveTester(args));
      break;
    case PutTestType:
      test_name = "Blocking Puts";
      testers.push_back(new PrimitiveTester(args));
      break;
    case PutNBITestType:
      test_name = "Non-Blocking Puts";
      testers.push_back(new PrimitiveTester(args));
      break;
    case DefaultCTXGetTestType:
      test_name = "Default context Blocking Gets";
      testers.push_back(new DefaultCTXPrimitiveTester(args));
      break;
    case DefaultCTXGetNBITestType:
      test_name = "Default context Non-Blocking Gets";
      testers.push_back(new DefaultCTXPrimitiveTester(args));
      break;
    case DefaultCTXPutTestType:
      test_name = "Default context Blocking Puts";
      testers.push_back(new DefaultCTXPrimitiveTester(args));
      break;
    case DefaultCTXPutNBITestType:
      test_name = "Default context Non-Blocking Puts";
      testers.push_back(new DefaultCTXPrimitiveTester(args));
      break;
    case TeamCtxInfraTestType:
      test_name = "Team Ctx Infra test";
      testers.push_back(new TeamCtxInfraTester(args));
      break;
    case TeamCtxInfraSingleTestType:
      test_name = "Team Ctx Infra Single test";
      args.team_type = ROCSHMEM_TEST_TEAM_SINGLE;
      testers.push_back(new TeamCtxInfraTester(args));
      break;
    case TeamCtxInfraBlockTestType:
      test_name = "Team Ctx Infra Block test";
      args.team_type = ROCSHMEM_TEST_TEAM_BLOCK;
      testers.push_back(new TeamCtxInfraTester(args));
      break;
    case TeamCtxInfraOddEvenTestType:
      test_name = "Team Ctx Infra Odd-Even test";
      args.team_type = ROCSHMEM_TEST_TEAM_ODDEVEN;
      testers.push_back(new TeamCtxInfraTester(args));
      break;
    case TeamCtxSharedInfraTestType:
      test_name = "Team Ctx Infra Shared test";
      args.team_type = ROCSHMEM_TEST_TEAM_SHARED;
      testers.push_back(new TeamCtxInfraTester(args));
      break;
    case TeamCtxSubsetParentInfraTestType:
      test_name = "Team Ctx Infra Subset Parent test";
      args.team_type = ROCSHMEM_TEST_TEAM_SUBSET_PARENT;
      testers.push_back(new TeamCtxInfraTester(args));
      break;
    case TeamCtxGetTestType:
      test_name = "Blocking Team Ctx Gets";
      testers.push_back(new TeamCtxPrimitiveTester(args));
      break;
    case TeamCtxGetNBITestType:
      test_name = "Non-Blocking Team Ctx Gets";
      testers.push_back(new TeamCtxPrimitiveTester(args));
      break;
    case TeamCtxPutTestType:
      test_name = "Blocking Team Ctx Puts";
      testers.push_back(new TeamCtxPrimitiveTester(args));
      break;
    case TeamCtxPutNBITestType:
      test_name = "Non-Blocking Team Ctx Puts";
      testers.push_back(new TeamCtxPrimitiveTester(args));
      break;
    case PTestType:
      test_name = "P Test";
      testers.push_back(new PrimitiveTester(args));
      break;
    case GTestType:
      test_name = "G Test";
      testers.push_back(new PrimitiveTester(args));
      break;
    case TeamReductionTestType:
      test_name = "All-to-All Team-based Reduction";
      testers.push_back(new TeamReductionTester<float, ROCSHMEM_SUM>(
          args,
          [](float& f1, float& f2) {
            f1 = 1;
            f2 = 1;
          },
          [](float v, float n_pes) {
            return (v == n_pes)
                       ? std::make_pair(true, "")
                       : std::make_pair(false, "Got " + std::to_string(v) +
                                                   ", Expect " +
                                                   std::to_string(n_pes));
          }));
      break;
    case TeamBroadcastTestType:
      test_name = "Team Broadcast Test";
      testers.push_back(new TeamBroadcastTester<int64_t>(args));
      testers.push_back(new TeamBroadcastTester<int>(args));
      testers.push_back(new TeamBroadcastTester<long long>(args));
      testers.push_back(new TeamBroadcastTester<float>(args));
      testers.push_back(new TeamBroadcastTester<double>(args));
      testers.push_back(new TeamBroadcastTester<char>(args));
      testers.push_back(new TeamBroadcastTester<unsigned char>(args));
      break;
    case TeamAllToAllTestType:
      test_name = "Alltoall Test";
      testers.push_back(new TeamAlltoallTester<float>(args));
      break;
    case TeamAllToAllvTestType:
      test_name = "Alltoallv Test";
      testers.push_back(new TeamAlltoallvTester<float>(args));
      break;
    case TeamAlltoallmemOnStreamTestType:
      test_name = "Alltoallmem_On_Stream";
      testers.push_back(new TeamAlltoallmemOnStreamTester(args));
      break;
    case BarrierAllOnStreamTestType:
      test_name = "Barrier_All_On_Stream";
      testers.push_back(new BarrierAllOnStreamTester(args));
      break;
    case QuietOnStreamTestType:
      test_name = "Quiet_On_Stream";
      testers.push_back(new QuietOnStreamTester(args));
      break;
    case SyncAllOnStreamTestType:
      test_name = "Sync_All_On_Stream";
      testers.push_back(new BarrierAllOnStreamTester(args, SYNC_ALL_OP));
      break;
    case TeamBroadcastmemOnStreamTestType:
      test_name = "Broadcastmem_On_Stream";
      testers.push_back(new TeamBroadcastmemOnStreamTester(args));
      break;
    case GetmemOnStreamTestType:
      test_name = "Getmem_On_Stream";
      testers.push_back(new GetmemOnStreamTester(args));
      break;
    case PutmemOnStreamTestType:
      test_name = "Putmem_On_Stream";
      testers.push_back(new PutmemOnStreamTester(args));
      break;
    case HostPutmemTestType:
      test_name = "Host_Putmem";
      if (BackendType::IPC_BACKEND == backend_type)
        testers.push_back(new HostRmaTester(args));
      break;
    case HostGetmemTestType:
      test_name = "Host_Getmem";
      if (BackendType::IPC_BACKEND == backend_type)
        testers.push_back(new HostRmaTester(args));
      break;
    case HostAmoFAddTestType:
      test_name = "Host_Amo_FAdd";
      if (BackendType::IPC_BACKEND == backend_type)
        testers.push_back(new HostRmaTester(args));
      break;
    case HostAmoFCswapTestType:
      test_name = "Host_Amo_FCswap";
      if (BackendType::IPC_BACKEND == backend_type)
        testers.push_back(new HostRmaTester(args));
      break;
    case HostCtxPutmemTestType:
      test_name = "Host_Ctx_Putmem";
      if (BackendType::IPC_BACKEND == backend_type)
        testers.push_back(new HostRmaTester(args));
      break;
    case HostCtxGetmemTestType:
      test_name = "Host_Ctx_Getmem";
      if (BackendType::IPC_BACKEND == backend_type)
        testers.push_back(new HostRmaTester(args));
      break;
    case HostIntAmoFAddTestType:
      test_name = "Host_Int_Amo_FAdd";
      if (BackendType::IPC_BACKEND == backend_type)
        testers.push_back(new HostRmaTester(args));
      break;
    case HostIntAmoFCswapTestType:
      test_name = "Host_Int_Amo_FCswap";
      if (BackendType::IPC_BACKEND == backend_type)
        testers.push_back(new HostRmaTester(args));
      break;
    case HostAmoAllPesTestType:
      test_name = "Host_Amo_AllPes";
      if (BackendType::IPC_BACKEND == backend_type)
        testers.push_back(new HostRmaTester(args));
      break;
    case HostAmoSelfTestType:
      test_name = "Host_Amo_Self";
      if (BackendType::IPC_BACKEND == backend_type)
        testers.push_back(new HostRmaTester(args));
      break;
    case HostAmoAddTestType:
      test_name = "Host_Amo_Add";
      if (BackendType::IPC_BACKEND == backend_type)
        testers.push_back(new HostRmaTester(args));
      break;
    case PutmemSignalOnStreamTestType:
      test_name = "Putmem_Signal_On_Stream";
      testers.push_back(new PutmemSignalOnStreamTester(args));
      break;
    case SignalWaitUntilOnStreamTestType:
      test_name = "Signal_Wait_Until_On_Stream";
      testers.push_back(new SignalWaitUntilOnStreamTester(args));
      break;
    case TeamFCollectTestType:
      test_name = "Fcollect Test";
      testers.push_back(new TeamFcollectTester<int64_t>(args));
      testers.push_back(new TeamFcollectTester<int>(args));
      testers.push_back(new TeamFcollectTester<long long>(args));
      testers.push_back(new TeamFcollectTester<float>(args));
      testers.push_back(new TeamFcollectTester<double>(args));
      testers.push_back(new TeamFcollectTester<char>(args));
      testers.push_back(new TeamFcollectTester<unsigned char>(args));
      break;
    case AMO_FAddTestType:
      test_name = "AMO Fetch_Add";
      testers.push_back(new AMOStandardTester<long long>(args));
      testers.push_back(new AMOStandardTester<long>(args));
      if (BackendType::GDA_BACKEND != backend_type) // not implemented for GDA
        testers.push_back(new AMOStandardTester<int>(args));
      break;
    case AMO_FIncTestType:
      test_name = "AMO Fetch_Inc";
      testers.push_back(new AMOStandardTester<long long>(args));
      testers.push_back(new AMOStandardTester<long>(args));
      if (BackendType::GDA_BACKEND != backend_type) // not implemented for GDA
        testers.push_back(new AMOStandardTester<int>(args));
      break;
    case AMO_FetchTestType:
      test_name = "AMO Fetch";
      testers.push_back(new AMOExtendedTester<long long>(args));
      testers.push_back(new AMOExtendedTester<long>(args));
      if (BackendType::GDA_BACKEND != backend_type) // not implemented for GDA
        testers.push_back(new AMOExtendedTester<int>(args));
      break;
    case AMO_FCswapTestType:
      test_name = "AMO Fetch_CSWAP";
      testers.push_back(new AMOStandardTester<long long>(args));
      testers.push_back(new AMOStandardTester<long>(args));
      if (BackendType::GDA_BACKEND != backend_type) // not implemented for GDA
        testers.push_back(new AMOStandardTester<int>(args));
      break;
    case AMO_AddTestType:
      test_name = "AMO Add";
      testers.push_back(new AMOStandardTester<long long>(args));
      testers.push_back(new AMOStandardTester<long>(args));
      if (BackendType::GDA_BACKEND != backend_type) // not implemented for GDA
        testers.push_back(new AMOStandardTester<int>(args));
      break;
    case AMO_SetTestType:
      test_name = "AMO Set";
      testers.push_back(new AMOExtendedTester<long long>(args));
      testers.push_back(new AMOExtendedTester<long>(args));
      if (BackendType::GDA_BACKEND != backend_type) // not implemented for GDA
        testers.push_back(new AMOExtendedTester<int>(args));
      break;
    case AMO_SwapTestType:
      test_name = "AMO Swap";
      testers.push_back(new AMOExtendedTester<long long>(args));
      testers.push_back(new AMOExtendedTester<long>(args));
      if (BackendType::GDA_BACKEND != backend_type) // not implemented for GDA
        testers.push_back(new AMOExtendedTester<int>(args));
      break;
    case AMO_FetchAndTestType:
      test_name = "AMO Fetch And";
      testers.push_back(new AMOBitwiseTester<unsigned long long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned long>(args));
      if (BackendType::GDA_BACKEND != backend_type) // not implemented for GDA
        testers.push_back(new AMOBitwiseTester<unsigned int>(args));
      break;
    case AMO_AndTestType:
      test_name = "AMO And";
      testers.push_back(new AMOBitwiseTester<unsigned long long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned long>(args));
      if (BackendType::GDA_BACKEND != backend_type) // not implemented for GDA
        testers.push_back(new AMOBitwiseTester<unsigned int>(args));
      break;
    case AMO_FetchOrTestType:
      test_name = "AMO Fetch Or";
      testers.push_back(new AMOBitwiseTester<unsigned long long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned long>(args));
      if (BackendType::GDA_BACKEND != backend_type) // not implemented for GDA
        testers.push_back(new AMOBitwiseTester<unsigned int>(args));
      break;
    case AMO_OrTestType:
      test_name = "AMO Or";
      testers.push_back(new AMOBitwiseTester<unsigned long long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned long>(args));
      if (BackendType::GDA_BACKEND != backend_type) // not implemented for GDA
        testers.push_back(new AMOBitwiseTester<unsigned int>(args));
      break;
    case AMO_FetchXorTestType:
      test_name = "AMO Fetch Xor";
      testers.push_back(new AMOBitwiseTester<unsigned long long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned long>(args));
      if (BackendType::GDA_BACKEND != backend_type) // not implemented for GDA
        testers.push_back(new AMOBitwiseTester<unsigned int>(args));
      break;
    case AMO_XorTestType:
      test_name = "AMO Xor";
      testers.push_back(new AMOBitwiseTester<unsigned long long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned long>(args));
      if (BackendType::GDA_BACKEND != backend_type) // not implemented for GDA
        testers.push_back(new AMOBitwiseTester<unsigned int>(args));
      break;
    case AMO_IncTestType:
      test_name = "AMO Inc";
      testers.push_back(new AMOStandardTester<long long>(args));
      testers.push_back(new AMOStandardTester<long>(args));
      if (BackendType::GDA_BACKEND != backend_type) // not implemented for GDA
        testers.push_back(new AMOStandardTester<int>(args));
      break;
    case PingPongTestType:
      test_name = "PingPong";
      testers.push_back(new PingPongTester(args));
      break;
    case PingAllTestType:
      test_name = "PingAll";
      testers.push_back(new PingAllTester(args));
      break;
    case BarrierAllTestType:
      test_name = "Barrier_All";
      testers.push_back(new BarrierAllTester(args));
      break;
    case WAVEBarrierAllTestType:
      test_name = "WAVE Barrier_All";
      testers.push_back(new BarrierAllTester(args));
      break;
    case WGBarrierAllTestType:
      test_name = "WG Barrier_All";
      testers.push_back(new BarrierAllTester(args));
      break;
    case TeamBarrierTestType:
      test_name = "Team Barrier Test";
      testers.push_back(new TeamBarrierTester(args));
      break;
    case TeamWAVEBarrierTestType:
      test_name = "Team WAVE Barrier Test";
      testers.push_back(new TeamBarrierTester(args));
      break;
    case TeamWGBarrierTestType:
      test_name = "Team WG Barrier Test";
      testers.push_back(new TeamBarrierTester(args));
      break;
    case SyncAllTestType:
      test_name = "SyncAll";
      testers.push_back(new SyncAllTester(args));
      break;
    case WAVESyncAllTestType:
      test_name = "WAVE SyncAll";
      testers.push_back(new SyncAllTester(args));
      break;
    case WGSyncAllTestType:
      test_name = "WG SyncAll";
      testers.push_back(new SyncAllTester(args));
      break;
    case TeamSyncTestType:
      test_name = "Team Sync";
      testers.push_back(new TeamSyncTester(args));
      break;
    case TeamWAVESyncTestType:
      test_name = "Team WAVE Sync";
      testers.push_back(new TeamSyncTester(args));
      break;
    case TeamWGSyncTestType:
      test_name = "Team WG Sync";
      testers.push_back(new TeamSyncTester(args));
      break;
    case RandomAccessTestType:
      test_name = "Random_Access";
      testers.push_back(new RandomAccessTester(args));
      break;
    case ShmemPtrTestType:
      test_name = "Shmem_Ptr";
      testers.push_back(new ShmemPtrTester(args));
      break;
    case WGGetTestType:
      test_name = "Blocking WG level Gets";
      testers.push_back(new WorkGroupPrimitiveTester(args));
      break;
    case WGGetNBITestType:
      test_name = "Non-Blocking WG level Gets";
      testers.push_back(new WorkGroupPrimitiveTester(args));
      break;
    case WGPutTestType:
      test_name = "Blocking WG level Puts";
      testers.push_back(new WorkGroupPrimitiveTester(args));
      break;
    case WGPutNBITestType:
      test_name = "Non-Blocking WG level Puts";
      testers.push_back(new WorkGroupPrimitiveTester(args));
      break;
    case WAVEGetTestType:
      test_name = "Blocking WAVE level Gets";
      testers.push_back(new WaveFrontPrimitiveTester(args));
      break;
    case WAVEGetNBITestType:
      test_name = "Non-Blocking WAVE level Gets";
      testers.push_back(new WaveFrontPrimitiveTester(args));
      break;
    case WAVEPutTestType:
      test_name = "Blocking WAVE level Puts";
      testers.push_back(new WaveFrontPrimitiveTester(args));
      break;
    case WAVEPutNBITestType:
      test_name = "Non-Blocking WAVE level Puts";
      testers.push_back(new WaveFrontPrimitiveTester(args));
      break;
    case PutSignalTestType:
      test_name = "Putmem Signal";
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      break;
    case WGPutSignalTestType:
      test_name = "WG Putmem Signal";
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      break;
    case WAVEPutSignalTestType:
      test_name = "Wave Putmem Signal";
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      break;
    case PutSignalNBITestType:
      test_name = "Non-Blocking Putmem Signal";
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      break;
    case WGPutSignalNBITestType:
      test_name = "Non-Blocking WG Putmem Signal";
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      break;
    case WAVEPutSignalNBITestType:
      test_name = "Non-Blocking Wave Putmem Signal";
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      break;
    case SignalFetchTestType:
      test_name = "Signal Fetch";
      testers.push_back(new SignalingOperationsTester(args));
      break;
    case WGSignalFetchTestType:
      test_name = "WG Signal Fetch";
      testers.push_back(new SignalingOperationsTester(args));
      break;
    case WAVESignalFetchTestType:
      test_name = "Wave Signal Fetch";
      testers.push_back(new SignalingOperationsTester(args));
      break;
    case FloodPutTestType:
      test_name = "Flood Put (multidirectional)";
      testers.push_back(new FloodTester(args));
      break;
    case FloodPutNBITestType:
      test_name = "Flood Non-Blocking Put (multidirectional)";
      testers.push_back(new FloodTester(args));
      break;
    case FloodPTestType:
      test_name = "Flood P (multidirectional)";
      testers.push_back(new FloodTester(args));
      break;
    case FloodGetTestType:
      test_name = "Flood Get (multidirectional)";
      testers.push_back(new FloodTester(args));
      break;
    case FloodGetNBITestType:
      test_name = "Flood Non-Blocking Get (multidirectional)";
      testers.push_back(new FloodTester(args));
      break;
    case FloodGTestType:
      test_name = "Flood G (multidirectional)";
      testers.push_back(new FloodTester(args));
      break;
    case HipModuleInitTestType:
      test_name = "HIP Module Init Test";
      testers.push_back(new HipModuleInitTester(args));
      break;
    case FloodAddTestType:
      test_name = "Flood Add (multidirectional)";
      testers.push_back(new FloodAmoTester(args));
      break;
    case FloodFAddTestType:
      test_name = "Flood FAdd (multidirectional)";
      testers.push_back(new FloodAmoTester(args));
      break;
    case FloodWaitAmoTestType:
      test_name = "Flood WaitAdd (multidirectional)";
      testers.push_back(new FloodAmoTester(args));
      break;
    case DeviceBitcodeTestType:
      test_name = "Device Bitcode Test";
      testers.push_back(new DeviceBitcodeTester(args));
      break;
    case LibraryInfoTestType:
      test_name = "Library Info Test";
      testers.push_back(new LibraryInfoTester(args));
      break;
    case FenceOrderPutWaveSignalTestType:
      test_name = "Fence PutWaveSignal Ordering";
      testers.push_back(new FenceOrderingTester(args));
      break;
    case FenceOrderPutLargeSmallTestType:
      test_name = "Fence PutLargeSmall Ordering";
      testers.push_back(new FenceOrderingTester(args));
      break;
    case FenceOrderFanoutTestType:
      test_name = "Fence Fanout Ordering";
      testers.push_back(new FenceOrderingTester(args));
      break;
    case FenceOrderPutWaveNbiChunksTestType:
      test_name = "Fence PutWaveNbiChunks Ordering";
      testers.push_back(new FenceOrderingTester(args));
      break;
    case TilePutContiguousTestType:
      test_name = "Tile Put Contiguous";
      testers.push_back(new TileRMATester(args));
      break;
    case TilePutRowMajorTestType:
      test_name = "Tile Put Row-Major";
      testers.push_back(new TileRMATester(args));
      break;
    case TilePutColumnMajorTestType:
      test_name = "Tile Put Column-Major";
      testers.push_back(new TileRMATester(args));
      break;
    case TilePutArbitraryTestType:
      test_name = "Tile Put Arbitrary Strides";
      testers.push_back(new TileRMATester(args));
      break;
    case TilePutWaveContiguousTestType:
      test_name = "Tile Put Wave-Collective Contiguous";
      testers.push_back(new TileRMATester(args));
      break;
    case TilePutWGContiguousTestType:
      test_name = "Tile Put Workgroup-Collective Contiguous";
      testers.push_back(new TileRMATester(args));
      break;
    case TileGetContiguousTestType:
      test_name = "Tile Get Contiguous";
      testers.push_back(new TileRMATester(args));
      break;
    case TileGetWGContiguousTestType:
      test_name = "Tile Get Workgroup-Collective Contiguous";
      testers.push_back(new TileRMATester(args));
      break;
    case TilePut1DTestType:
      test_name = "Tile Put 1D Tensor";
      testers.push_back(new TileRMATester(args));
      break;
    case TileGet1DTestType:
      test_name = "Tile Get 1D Tensor";
      testers.push_back(new TileRMATester(args));
      break;
    case TileGetWaveContiguousTestType:
      test_name = "Tile Get Wave-Collective Contiguous";
      testers.push_back(new TileRMATester(args));
      break;
    case TileGetRowMajorTestType:
      test_name = "Tile Get Row-Major";
      testers.push_back(new TileRMATester(args));
      break;
    case TileGetColumnMajorTestType:
      test_name = "Tile Get Column-Major";
      testers.push_back(new TileRMATester(args));
      break;
    case TileGetArbitraryTestType:
      test_name = "Tile Get Arbitrary Strides";
      testers.push_back(new TileRMATester(args));
      break;
    case HostTeamSyncBarrierTestType:
      test_name = "Host Team Sync/Barrier";
      testers.push_back(new HostTeamSyncBarrierTester(args));
      break;
    case ReduceOnStreamTestType:
      test_name = "Reduce On Stream";
      testers.push_back(new ReduceOnStreamTester(args));
      break;
    case HostCtxCreateTestType:
      test_name = "Host CTX Create";
      testers.push_back(new HostCtxCreateTester(args));
    case TeamSplit2DTestType:
      test_name = "Team Split 2D";
      testers.push_back(new TeamSplit2DTester(args));
      break;
    case TileBroadcastTestType:
      test_name = "Tile Broadcast";
      testers.push_back(new TileBroadcastTester(args));
      break;
    case TileBroadcastWaveTestType:
      test_name = "Tile Broadcast Wave-Collective";
      testers.push_back(new TileBroadcastTester(args));
      break;
    case TileBroadcastWGTestType:
      test_name = "Tile Broadcast Workgroup-Collective";
      testers.push_back(new TileBroadcastTester(args));
      break;
    case TileAllgatherTestType:
      test_name = "Tile Allgather";
      testers.push_back(new TileAllgatherTester(args));
      break;
    case TileAllgatherWaveTestType:
      test_name = "Tile Allgather Wave-Collective";
      testers.push_back(new TileAllgatherTester(args));
      break;
    case TileAllgatherWGTestType:
      test_name = "Tile Allgather Workgroup-Collective";
      testers.push_back(new TileAllgatherTester(args));
      break;
    default:
      test_name = "Empty";
      break;
  }

  if (rank == 0) {
    const char* backend_str =
        (backend_type == BackendType::IPC_BACKEND) ? "ipc" :
        (backend_type == BackendType::RO_BACKEND)  ? "ro"  : "gda";
    std::cout << "### Creating Test:\t" << test_name
              << "\tB=" << backend_str
              << " PE=" << args.numprocs
              << " W=" << args.num_wgs
              << " Z=" << args.wg_size
              << " ###" << std::endl;
  }

  return testers;
}

void Tester::execute() {
  if (_type == InitTestType) return;

  num_loops = args.loop;

  /**
   * Some tests loop through data sizes in powers of 2 and report the
   * results for those ranges.
   */
  for (size_t size = args.min_msg_size; size <= max_msg_size;
       size <<= 1) {
    /**
     * Restricts the number of iterations of really large messages.
     */
    if (size > args.large_message_size) num_loops = args.loop_large;

    // Reset after num_loops is set so subclasses can size their
    // buffers to the actual iteration count for this message size.
    resetBuffers(size);

    barrier();

    preLaunchKernel();

    /**
     * This conditional launches the HIP kernel.
     *
     * Some tests may only launch a single kernel. These kernels will
     * be kicked off by the initiator (denoted by the args.myid check).
     *
     * Other tests will initiate of both sides and launch from both
     * rocshmem pes.
     */
    if (peLaunchesKernel()) {
      memset(timer, 0, sizeof(uint64_t) * args.num_wgs);

      const dim3 blockSize(args.wg_size, 1, 1);
      const dim3 gridSize(args.num_wgs, 1, 1);

      CHECK_HIP(hipEventRecord(start_event, stream));

      launchKernel(gridSize, blockSize, num_loops, size);

      CHECK_HIP(hipEventRecord(stop_event, stream));

      hipError_t err = hipStreamSynchronize(stream);
      if (err != hipSuccess) {
        printf("error = %d \n", err);
      }
    }

    barrier();

    postLaunchKernel();

    // data validation
    if (args.verif)
      verifyResults(size);

    barrier();

    if (_type != TeamCtxInfraTestType       &&
        _type != TeamCtxInfraSingleTestType &&
        _type != TeamCtxInfraBlockTestType  &&
        _type != TeamCtxInfraOddEvenTestType &&
        _type != TeamCtxSharedInfraTestType &&
        _type != TeamCtxSubsetParentInfraTestType &&
        _type != HostCtxCreateTestType &&
        _type != TeamSplit2DTestType  ) {
      print(size);
    }
  }
}

bool Tester::peLaunchesKernel() {
  /**
   * The PE assigned 0 is always active in these tests.
   */
  bool is_launcher = (args.myid == 0);

  /**
   * Some test types are active on both sides.
   */
  switch (_type) {
    case ReduceOnStreamTestType:
    case TeamReductionTestType:
    case TeamBroadcastTestType:
    case TeamCtxInfraTestType:
    case TeamCtxInfraSingleTestType:
    case TeamCtxInfraBlockTestType:
    case TeamCtxInfraOddEvenTestType:
    case TeamCtxSharedInfraTestType:
    case TeamCtxSubsetParentInfraTestType:
    case TeamAllToAllTestType:
    case TeamAllToAllvTestType:
    case TeamFCollectTestType:
    case PingPongTestType:
    case BarrierAllTestType:
    case WAVEBarrierAllTestType:
    case WGBarrierAllTestType:
    case TeamSyncTestType:
    case TeamWAVESyncTestType:
    case TeamWGSyncTestType:
    case SyncAllTestType:
    case WAVESyncAllTestType:
    case WGSyncAllTestType:
    case RandomAccessTestType:
    case PingAllTestType:
    case TeamBarrierTestType:
    case TeamWAVEBarrierTestType:
    case TeamWGBarrierTestType:
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
    case HipModuleInitTestType:
    case FloodAddTestType:
    case FloodFAddTestType:
    case FloodWaitAmoTestType:
    case DeviceBitcodeTestType:
    case FenceOrderPutWaveSignalTestType:
    case FenceOrderPutLargeSmallTestType:
    case FenceOrderFanoutTestType:
    case FenceOrderPutWaveNbiChunksTestType:
    case TileBroadcastTestType:
    case TileBroadcastWaveTestType:
    case TileBroadcastWGTestType:
    case TileAllgatherTestType:
    case TileAllgatherWaveTestType:
    case TileAllgatherWGTestType:
      is_launcher = true;
      break;
    case HostPutmemTestType:
    case HostGetmemTestType:
    case HostAmoFAddTestType:
    case HostAmoFCswapTestType:
    case HostCtxPutmemTestType:
    case HostCtxGetmemTestType:
    case HostIntAmoFAddTestType:
    case HostIntAmoFCswapTestType:
    case HostAmoAllPesTestType:
    case HostAmoSelfTestType:
      is_launcher = true;
      break;
    default:
      break;
  }

  return is_launcher;
}

void Tester::print(uint64_t size) {
  if (args.myid != 0 || !_print_results) {
    return;
  }

  /**
   * Calculate total amount of data transferred
   */
  size_t total_size = size_factor * size * num_timed_msgs;
  size_t volume = total_size / num_loops;

  [[maybe_unused]] double timer_avg = timerAvgInMicroseconds();
  double time_us = gpuCyclesToMicroseconds(max_end_time - min_start_time);
  double time_s = time_us / 1e6;

  double latency = time_us / num_loops / rtt_factor;

  double msg_rate = num_timed_msgs / time_s;

  double bandwidth_gbs =
      static_cast<double>(bw_factor * total_size) / time_s / pow(2, 30);

  float total_kern_time_ms;
  CHECK_HIP(hipEventElapsedTime(&total_kern_time_ms, start_event, stop_event));
  [[maybe_unused]] float total_kern_time_s = total_kern_time_ms / 1000;

  int field_width = 20;
  int float_precision = 2;

  if (_print_header) {
    printf("%-*s%-*s%-*s%*s%*s%*s",
           15, "# Volume (B)",
           15, "Msg Size (B)",
           15, "# of timed Msgs",
           field_width, "Latency (us)",
           field_width, "Bandwidth (GB/s)",
           field_width + 1, "Msg Rate (Msg/s)\n");
    _print_header = 0;
  }

  printf("%-*lu%-*lu%-*zu%*.*f%*.*f%*.*f\n",
         15, volume,
         15, size,
         15, num_timed_msgs,
         field_width, float_precision, latency,
         field_width, float_precision, bandwidth_gbs,
         field_width, float_precision, msg_rate);

  fflush(stdout);
}

void flush_hdp() {
  int hip_dev_id{};
  unsigned int* hdp_flush_ptr_{nullptr};
  CHECK_HIP(hipGetDevice(&hip_dev_id));
  CHECK_HIP(hipDeviceGetAttribute(reinterpret_cast<int*>(&hdp_flush_ptr_),
                        hipDeviceAttributeHdpMemFlushCntl, hip_dev_id));
  if (hdp_flush_ptr_ != nullptr) {
    __atomic_store_n(hdp_flush_ptr_, 0x1, __ATOMIC_SEQ_CST);
  }
}

void Tester::barrier() {
  rocshmem_barrier_all();
#if defined USE_HDP_FLUSH
  flush_hdp();
#endif
}

double Tester::gpuCyclesToMicroseconds(long long int cycles) {
  return static_cast<double>(cycles) /
         (static_cast<double>(wall_clk_rate) * 1e-3);
}

double Tester::timerAvgInMicroseconds() {
  double sum = 0;
  min_start_time = LLONG_MAX;
  max_end_time = 0;

  for (uint32_t i = 0; i < num_timers; i++) {
    timer[i] = end_time[i] - start_time[i];
    sum += gpuCyclesToMicroseconds(timer[i]);
    min_start_time = (start_time[i] < min_start_time)
                     ? start_time[i]
                     : min_start_time;
    max_end_time = (end_time[i] > max_end_time)
                     ? end_time[i]
                     : max_end_time;
  }

  return sum / num_timers;
}

void* Tester::alloc_test_buffer(size_t size, enum UserBufType user_buf_type) {
  void *buffer;
  int err = ROCSHMEM_SUCCESS;

  switch (user_buf_type) {
    case USER_BUF_TYPE_HOST:
      CHECK_HIP(hipHostMalloc(&buffer, size));
      break;
    case USER_BUF_TYPE_DEVICE:
      CHECK_HIP(hipMalloc(&buffer, size));
      break;
    case USER_BUF_TYPE_FINE:
      CHECK_HIP(hipExtMallocWithFlags(&buffer, size, hipDeviceMallocFinegrained));
      break;
    case USER_BUF_TYPE_UNCACHED:
#ifdef HAVE_DEVICE_MALLOC_UNCACHED
      CHECK_HIP(hipExtMallocWithFlags(&buffer, size, hipDeviceMallocUncached));
#else
      std::cerr << "hipDeviceMallocUncached is unsupported. Please use another local memory type"
                << std::endl;
      exit(-1);
#endif
      break;
    case USER_BUF_TYPE_MANAGED:
      CHECK_HIP(hipMallocManaged(&buffer, size, hipMemAttachGlobal));
      break;
    case USER_BUF_TYPE_HEAP:
    default:
      buffer  = rocshmem_malloc(size);
      if (buffer == nullptr) {
        std::cerr << "Error allocating memory from symmetric heap" << std::endl;
        std::cerr << "buffer: " << (uintptr_t) buffer << std::endl;
        exit(-1);
      }
      return buffer;
  }

  err = rocshmem_buffer_register(buffer, size);

  if (ROCSHMEM_SUCCESS != err) {
    return nullptr;
  }

  return buffer;
}

void Tester::free_test_buffer(void *buffer, enum UserBufType user_buf_type) {
  int err = ROCSHMEM_SUCCESS;

  switch (user_buf_type) {
    case USER_BUF_TYPE_HOST:
      err = rocshmem_buffer_unregister(buffer);
      CHECK_HIP(hipHostFree(buffer));
      break;
    case USER_BUF_TYPE_DEVICE:
    case USER_BUF_TYPE_FINE:
    case USER_BUF_TYPE_UNCACHED:
    case USER_BUF_TYPE_MANAGED:
      err = rocshmem_buffer_unregister(buffer);
      CHECK_HIP(hipFree(buffer));
      break;
    case USER_BUF_TYPE_HEAP:
    default:
      rocshmem_free(buffer);
      break;
  }

  if (ROCSHMEM_SUCCESS != err) {
    fprintf(stderr, "Deregistration Error");
  }
}
