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
#include "type_lists.hpp"

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
#include "typed_rma_tester.hpp"
#include "team_fcollect_tester.hpp"
#include "fcollect_wave_tester.hpp"
#include "team_reduction_tester.hpp"
#include "team_reduce_scatter_tester.hpp"
#include "reduce_wave_tester.hpp"
#include "team_reduce_scatter_wave_tester.hpp"
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
#include "tile_reduce_tester.hpp"
#include "reduce_on_stream_tester.hpp"
#include "host_ctx_create_tester.hpp"
#include "team_split_2d_tester.hpp"
#include "host_team_sync_barrier_tester.hpp"
#include "broadcast_wave_tester.hpp"
#include "alltoall_wave_tester.hpp"
#if defined(USE_GDA)
#include "qp_ping_pong_tester.hpp"
#include "qp_put_nbi_tester.hpp"
#endif
#if defined(USE_SDMA)
#include "sdma_ping_pong_tester.hpp"
#include "sdma_put_nbi_tester.hpp"
#endif

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
    case BroadcastWaveTestType:
    case AllToAllWaveTestType:
    case FcollectWaveTestType:
    case ReduceWaveTestType:
    case TeamReduceScatterWaveTestType:
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
      case ReduceWaveTestType:
        max_msg_size = args.max_volume_size / args.num_wgs / num_warps;
        break;
      case WGGetTestType:
      case WGGetNBITestType:
      case WGPutTestType:
      case WGPutNBITestType:
      case WGPutSignalTestType:
      case WGPutSignalNBITestType:
      case QpPutNbiTestType:
      case SdmaPutNbiTestType:
        max_msg_size = args.max_volume_size / args.num_wgs;
        break;
      case PingPongTestType:
      case QpPingPongTestType:
      case SdmaPingPongTestType:
        if (args.op_type == 2) {
          max_msg_size = args.max_volume_size / args.num_wgs;
        }
        break;
      case TeamBroadcastTestType:
      case BroadcastWaveTestType:
      case TeamReductionTestType:
      case TeamReduceScatterTestType:
      case TeamReduceScatterWaveTestType:
      case TeamFCollectTestType:
      case FcollectWaveTestType:
      case CollectTestType:
      case TeamAllToAllTestType:
      case TeamAllToAllvTestType:
      case TeamAlltoallmemOnStreamTestType:
      case AllToAllWaveTestType:
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

/**
 * Queue a tester for deferred construction.
 *
 * `args` is captured by value at the point of the push, so cases that tweak
 * args (team_type, wg_size) before queuing keep those tweaks.  Variadic so
 * template arguments containing commas pass through intact.
 */
#define PUSH_TESTER(...) \
  testers.push_back([args]() -> Tester* { return new __VA_ARGS__; })

std::vector<TesterFactory> Tester::create(TesterArguments args) {
  int rank = args.myid;
  std::vector<TesterFactory> testers;
  std::string test_name;

  BackendType backend_type = rocshmem_query_backend_type();
  TestType type = (TestType)args.algorithm;

  if (args.num_wf > 0) {
    int device_id;
    hipDeviceProp_t props;
    CHECK_HIP(hipGetDevice(&device_id));
    CHECK_HIP(hipGetDeviceProperties(&props, device_id));
    args.wg_size = props.warpSize * args.num_wf;
  }

  switch (type) {
    case InitTestType:
      test_name = "Init";
      PUSH_TESTER(EmptyTester(args));
      break;
    case GetTestType:
      test_name = "Blocking Gets";
      PUSH_TESTER(PrimitiveTester(args));
      // PrimitiveTester goes through the byte-oriented getmem path; the typed
      // entry points come along when the caller selects an element type.
      #define PUSH_TYPED_RMA(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(TypedRMATester<T>(args));
      ROCSHMEM_RMA_TYPES_ALWAYS(PUSH_TYPED_RMA)
      if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
        ROCSHMEM_RMA_TYPES_NONRO(PUSH_TYPED_RMA)
      }
      #undef PUSH_TYPED_RMA
      break;
    case GetNBITestType:
      test_name = "Non-Blocking Gets";
      PUSH_TESTER(PrimitiveTester(args));
      #define PUSH_TYPED_RMA(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(TypedRMATester<T>(args));
      ROCSHMEM_RMA_TYPES_ALWAYS(PUSH_TYPED_RMA)
      if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
        ROCSHMEM_RMA_TYPES_NONRO(PUSH_TYPED_RMA)
      }
      #undef PUSH_TYPED_RMA
      break;
    case PutTestType:
      test_name = "Blocking Puts";
      PUSH_TESTER(PrimitiveTester(args));
      #define PUSH_TYPED_RMA(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(TypedRMATester<T>(args));
      ROCSHMEM_RMA_TYPES_ALWAYS(PUSH_TYPED_RMA)
      if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
        ROCSHMEM_RMA_TYPES_NONRO(PUSH_TYPED_RMA)
      }
      #undef PUSH_TYPED_RMA
      break;
    case PutNBITestType:
      test_name = "Non-Blocking Puts";
      PUSH_TESTER(PrimitiveTester(args));
      #define PUSH_TYPED_RMA(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(TypedRMATester<T>(args));
      ROCSHMEM_RMA_TYPES_ALWAYS(PUSH_TYPED_RMA)
      if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
        ROCSHMEM_RMA_TYPES_NONRO(PUSH_TYPED_RMA)
      }
      #undef PUSH_TYPED_RMA
      break;
    case DefaultCTXGetTestType:
      test_name = "Default context Blocking Gets";
      PUSH_TESTER(DefaultCTXPrimitiveTester(args));
      break;
    case DefaultCTXGetNBITestType:
      test_name = "Default context Non-Blocking Gets";
      PUSH_TESTER(DefaultCTXPrimitiveTester(args));
      break;
    case DefaultCTXPutTestType:
      test_name = "Default context Blocking Puts";
      PUSH_TESTER(DefaultCTXPrimitiveTester(args));
      break;
    case DefaultCTXPutNBITestType:
      test_name = "Default context Non-Blocking Puts";
      PUSH_TESTER(DefaultCTXPrimitiveTester(args));
      break;
    case TeamCtxInfraTestType:
      test_name = "Team Ctx Infra test";
      PUSH_TESTER(TeamCtxInfraTester(args));
      break;
    case TeamCtxInfraSingleTestType:
      test_name = "Team Ctx Infra Single test";
      args.team_type = ROCSHMEM_TEST_TEAM_SINGLE;
      PUSH_TESTER(TeamCtxInfraTester(args));
      break;
    case TeamCtxInfraBlockTestType:
      test_name = "Team Ctx Infra Block test";
      args.team_type = ROCSHMEM_TEST_TEAM_BLOCK;
      PUSH_TESTER(TeamCtxInfraTester(args));
      break;
    case TeamCtxInfraOddEvenTestType:
      test_name = "Team Ctx Infra Odd-Even test";
      args.team_type = ROCSHMEM_TEST_TEAM_ODDEVEN;
      PUSH_TESTER(TeamCtxInfraTester(args));
      break;
    case TeamCtxSharedInfraTestType:
      test_name = "Team Ctx Infra Shared test";
      args.team_type = ROCSHMEM_TEST_TEAM_SHARED;
      PUSH_TESTER(TeamCtxInfraTester(args));
      break;
    case TeamCtxSubsetParentInfraTestType:
      test_name = "Team Ctx Infra Subset Parent test";
      args.team_type = ROCSHMEM_TEST_TEAM_SUBSET_PARENT;
      PUSH_TESTER(TeamCtxInfraTester(args));
      break;
    case TeamCtxGetTestType:
      test_name = "Blocking Team Ctx Gets";
      PUSH_TESTER(TeamCtxPrimitiveTester(args));
      break;
    case TeamCtxGetNBITestType:
      test_name = "Non-Blocking Team Ctx Gets";
      PUSH_TESTER(TeamCtxPrimitiveTester(args));
      break;
    case TeamCtxPutTestType:
      test_name = "Blocking Team Ctx Puts";
      PUSH_TESTER(TeamCtxPrimitiveTester(args));
      break;
    case TeamCtxPutNBITestType:
      test_name = "Non-Blocking Team Ctx Puts";
      PUSH_TESTER(TeamCtxPrimitiveTester(args));
      break;
    case PTestType:
      test_name = "P Test";
      PUSH_TESTER(PrimitiveTester(args));
      #define PUSH_TYPED_RMA(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(TypedRMATester<T>(args));
      ROCSHMEM_RMA_TYPES_ALWAYS(PUSH_TYPED_RMA)
      if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
        ROCSHMEM_RMA_TYPES_NONRO(PUSH_TYPED_RMA)
      }
      #undef PUSH_TYPED_RMA
      break;
    case GTestType:
      test_name = "G Test";
      PUSH_TESTER(PrimitiveTester(args));
      #define PUSH_TYPED_RMA(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(TypedRMATester<T>(args));
      ROCSHMEM_RMA_TYPES_ALWAYS(PUSH_TYPED_RMA)
      if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
        ROCSHMEM_RMA_TYPES_NONRO(PUSH_TYPED_RMA)
      }
      #undef PUSH_TYPED_RMA
      break;
    case TeamReductionTestType:
      test_name = "All-to-All Team-based Reduction";
      if (args.type_coverage == TypeCoverage::Minimal) {
        ROCSHMEM_PUSH_REDUCTION_FLOAT(TeamReductionTester, float, args, testers)
      } else {
        ROCSHMEM_PUSH_REDUCTION_ALWAYS(TeamReductionTester, args, testers)
        if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
          ROCSHMEM_PUSH_REDUCTION_NONRO(TeamReductionTester, args, testers)
        }
      }
      break;
    case TeamReduceScatterTestType:
      test_name = "Team-based Reduce-Scatter";
      if (args.type_coverage == TypeCoverage::Minimal) {
        ROCSHMEM_PUSH_REDUCTION_FLOAT(TeamReduceScatterTester, float, args, testers)
      } else {
        ROCSHMEM_PUSH_REDUCTION_ALWAYS(TeamReduceScatterTester, args, testers)
        if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
          ROCSHMEM_PUSH_REDUCTION_NONRO(TeamReduceScatterTester, args, testers)
        }
      }
      break;
    case ReduceWaveTestType:
      test_name = "Wave-level Reduction";
      if (args.type_coverage == TypeCoverage::Minimal) {
        ROCSHMEM_PUSH_REDUCTION_FLOAT(ReduceWaveTester, float, args, testers)
      } else {
        ROCSHMEM_PUSH_REDUCTION_ALWAYS(ReduceWaveTester, args, testers)
        if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
          ROCSHMEM_PUSH_REDUCTION_NONRO(ReduceWaveTester, args, testers)
        }
      }
      break;
    case TeamReduceScatterWaveTestType:
      test_name = "Team-based Reduce-Scatter Wave";
      if (args.type_coverage == TypeCoverage::Minimal) {
        ROCSHMEM_PUSH_REDUCTION_FLOAT(TeamReduceScatterWaveTester, float, args, testers)
      } else {
        ROCSHMEM_PUSH_REDUCTION_ALWAYS(TeamReduceScatterWaveTester, args, testers)
        if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
          ROCSHMEM_PUSH_REDUCTION_NONRO(TeamReduceScatterWaveTester, args, testers)
        }
      }
      break;
    case TeamBroadcastTestType: {
      test_name = "Team Broadcast Test";
      // Full mode: push all 13 compiled types via the full list.
      // Minimal mode: push the original 7-type subset.
      // Custom mode: push any requested type from the full compiled set.
      #define PUSH_BCAST(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(TeamBroadcastTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(TeamBroadcastTester<int64_t>(args));
        PUSH_TESTER(TeamBroadcastTester<int>(args));
        PUSH_TESTER(TeamBroadcastTester<long long>(args));
        PUSH_TESTER(TeamBroadcastTester<float>(args));
        PUSH_TESTER(TeamBroadcastTester<double>(args));
        PUSH_TESTER(TeamBroadcastTester<char>(args));
        PUSH_TESTER(TeamBroadcastTester<unsigned char>(args));
      } else {
        ROCSHMEM_COLL_TYPES_ALWAYS(PUSH_BCAST)
        if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
          ROCSHMEM_COLL_TYPES_NONRO(PUSH_BCAST)
        }
      }
      #undef PUSH_BCAST
      break;
    }
    case BroadcastWaveTestType: {
      test_name = "Broadcast Wave Test";
      #define PUSH_BWAVE(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(BroadcastWaveTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(BroadcastWaveTester<int>(args));
        PUSH_TESTER(BroadcastWaveTester<long long>(args));
        PUSH_TESTER(BroadcastWaveTester<float>(args));
        PUSH_TESTER(BroadcastWaveTester<double>(args));
      } else {
        ROCSHMEM_COLL_TYPES_ALWAYS(PUSH_BWAVE)
        if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
          ROCSHMEM_COLL_TYPES_NONRO(PUSH_BWAVE)
        }
      }
      #undef PUSH_BWAVE
      break;
    }
    case TeamAllToAllTestType: {
      test_name = "Alltoall Test";
      #define PUSH_A2A(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(TeamAlltoallTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(TeamAlltoallTester<float>(args));
      } else {
        ROCSHMEM_COLL_TYPES_ALWAYS(PUSH_A2A)
        if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
          ROCSHMEM_COLL_TYPES_NONRO(PUSH_A2A)
        }
      }
      #undef PUSH_A2A
      break;
    }
    case TeamAllToAllvTestType: {
      test_name = "Alltoallv Test";
      #define PUSH_A2AV(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(TeamAlltoallvTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(TeamAlltoallvTester<float>(args));
      } else {
        ROCSHMEM_COLL_TYPES_ALWAYS(PUSH_A2AV)
        if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
          ROCSHMEM_COLL_TYPES_NONRO(PUSH_A2AV)
        }
      }
      #undef PUSH_A2AV
      break;
    }
    case TeamAlltoallmemOnStreamTestType:
      test_name = "Alltoallmem_On_Stream";
      PUSH_TESTER(TeamAlltoallmemOnStreamTester(args));
      break;
    case AllToAllWaveTestType: {
      test_name = "AllToAll Wave Test";
      #define PUSH_A2AWAVE(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(AlltoallWaveTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(AlltoallWaveTester<float>(args));
        PUSH_TESTER(AlltoallWaveTester<char>(args));
        PUSH_TESTER(AlltoallWaveTester<int>(args));
      } else {
        ROCSHMEM_COLL_TYPES_ALWAYS(PUSH_A2AWAVE)
        if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
          ROCSHMEM_COLL_TYPES_NONRO(PUSH_A2AWAVE)
        }
      }
      #undef PUSH_A2AWAVE
      break;
    }
    case BarrierAllOnStreamTestType:
      test_name = "Barrier_All_On_Stream";
      PUSH_TESTER(BarrierAllOnStreamTester(args));
      break;
    case QuietOnStreamTestType:
      test_name = "Quiet_On_Stream";
      PUSH_TESTER(QuietOnStreamTester(args));
      break;
    case SyncAllOnStreamTestType:
      test_name = "Sync_All_On_Stream";
      PUSH_TESTER(BarrierAllOnStreamTester(args, SYNC_ALL_OP));
      break;
    case TeamBroadcastmemOnStreamTestType:
      test_name = "Broadcastmem_On_Stream";
      PUSH_TESTER(TeamBroadcastmemOnStreamTester(args));
      break;
    case GetmemOnStreamTestType:
      test_name = "Getmem_On_Stream";
      PUSH_TESTER(GetmemOnStreamTester(args));
      break;
    case PutmemOnStreamTestType:
      test_name = "Putmem_On_Stream";
      PUSH_TESTER(PutmemOnStreamTester(args));
      break;
    case HostPutmemTestType:
      test_name = "Host_Putmem";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostGetmemTestType:
      test_name = "Host_Getmem";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostAmoFAddTestType:
      test_name = "Host_Amo_FAdd";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostAmoFCswapTestType:
      test_name = "Host_Amo_FCswap";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostCtxPutmemTestType:
      test_name = "Host_Ctx_Putmem";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostCtxGetmemTestType:
      test_name = "Host_Ctx_Getmem";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostIntAmoFAddTestType:
      test_name = "Host_Int_Amo_FAdd";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostIntAmoFCswapTestType:
      test_name = "Host_Int_Amo_FCswap";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostAmoAllPesTestType:
      test_name = "Host_Amo_AllPes";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostAmoSelfTestType:
      test_name = "Host_Amo_Self";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostAmoAddTestType:
      test_name = "Host_Amo_Add";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostWaitUntilTestType:
      test_name = "Host_Wait_Until";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostTestTestType:
      test_name = "Host_Test";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostWaitUntilAllTestType:
      test_name = "Host_Wait_Until_All";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostWaitUntilAnyTestType:
      test_name = "Host_Wait_Until_Any";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostWaitUntilSomeTestType:
      test_name = "Host_Wait_Until_Some";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostWaitUntilAllVectorTestType:
      test_name = "Host_Wait_Until_All_Vector";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostWaitUntilAnyVectorTestType:
      test_name = "Host_Wait_Until_Any_Vector";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostWaitUntilSomeVectorTestType:
      test_name = "Host_Wait_Until_Some_Vector";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostWaitUntilAllStatusTestType:
      test_name = "Host_Wait_Until_All_Status";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostWaitUntilAnyStatusTestType:
      test_name = "Host_Wait_Until_Any_Status";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case HostWaitUntilSomeStatusTestType:
      test_name = "Host_Wait_Until_Some_Status";
      if (BackendType::IPC_BACKEND == backend_type)
        PUSH_TESTER(HostRmaTester(args));
      break;
    case PutmemSignalOnStreamTestType:
      test_name = "Putmem_Signal_On_Stream";
      PUSH_TESTER(PutmemSignalOnStreamTester(args));
      break;
    case SignalWaitUntilOnStreamTestType:
      test_name = "Signal_Wait_Until_On_Stream";
      PUSH_TESTER(SignalWaitUntilOnStreamTester(args));
      break;
    case TeamFCollectTestType: {
      test_name = "Fcollect Test";
      #define PUSH_FCOL(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(TeamFcollectTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(TeamFcollectTester<int64_t>(args));
        PUSH_TESTER(TeamFcollectTester<int>(args));
        PUSH_TESTER(TeamFcollectTester<long long>(args));
        PUSH_TESTER(TeamFcollectTester<float>(args));
        PUSH_TESTER(TeamFcollectTester<double>(args));
        PUSH_TESTER(TeamFcollectTester<char>(args));
        PUSH_TESTER(TeamFcollectTester<unsigned char>(args));
      } else {
        ROCSHMEM_COLL_TYPES_ALWAYS(PUSH_FCOL)
        if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
          ROCSHMEM_COLL_TYPES_NONRO(PUSH_FCOL)
        }
      }
      #undef PUSH_FCOL
      break;
    }
    case FcollectWaveTestType: {
      test_name = "Fcollect Wave Test";
      #define PUSH_FWAVE(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(FcollectWaveTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(FcollectWaveTester<int64_t>(args));
        PUSH_TESTER(FcollectWaveTester<int>(args));
        PUSH_TESTER(FcollectWaveTester<long long>(args));
        PUSH_TESTER(FcollectWaveTester<float>(args));
        PUSH_TESTER(FcollectWaveTester<double>(args));
        PUSH_TESTER(FcollectWaveTester<char>(args));
        PUSH_TESTER(FcollectWaveTester<unsigned char>(args));
      } else {
        ROCSHMEM_COLL_TYPES_ALWAYS(PUSH_FWAVE)
        if (BackendType::RO_BACKEND != backend_type) { // no half/bfloat16 on RO
          ROCSHMEM_COLL_TYPES_NONRO(PUSH_FWAVE)
        }
      }
      #undef PUSH_FWAVE
      break;
    }
    case AMO_FAddTestType: {
      test_name = "AMO Fetch_Add";
      #define PUSH_AMO_STD(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(AMOStandardTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(AMOStandardTester<long long>(args));
        PUSH_TESTER(AMOStandardTester<long>(args));
        if (BackendType::GDA_BACKEND != backend_type) // GDA is 64-bit AMO only
          PUSH_TESTER(AMOStandardTester<int>(args));
      } else {
        ROCSHMEM_AMO_STD_TYPES_ALWAYS(PUSH_AMO_STD)
        if (BackendType::GDA_BACKEND != backend_type) { // GDA is 64-bit AMO only
          ROCSHMEM_AMO_STD_TYPES_NONGDA(PUSH_AMO_STD)
        }
      }
      #undef PUSH_AMO_STD
      break;
    }
    case AMO_FIncTestType: {
      test_name = "AMO Fetch_Inc";
      #define PUSH_AMO_STD(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(AMOStandardTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(AMOStandardTester<long long>(args));
        PUSH_TESTER(AMOStandardTester<long>(args));
        if (BackendType::GDA_BACKEND != backend_type) // GDA is 64-bit AMO only
          PUSH_TESTER(AMOStandardTester<int>(args));
      } else {
        ROCSHMEM_AMO_STD_TYPES_ALWAYS(PUSH_AMO_STD)
        if (BackendType::GDA_BACKEND != backend_type) { // GDA is 64-bit AMO only
          ROCSHMEM_AMO_STD_TYPES_NONGDA(PUSH_AMO_STD)
        }
      }
      #undef PUSH_AMO_STD
      break;
    }
    case AMO_FetchTestType: {
      test_name = "AMO Fetch";
      #define PUSH_AMO_EXT(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(AMOExtendedTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(AMOExtendedTester<long long>(args));
        PUSH_TESTER(AMOExtendedTester<long>(args));
        if (BackendType::GDA_BACKEND != backend_type) // GDA is 64-bit AMO only
          PUSH_TESTER(AMOExtendedTester<int>(args));
      } else {
        ROCSHMEM_AMO_EXT_TYPES_ALWAYS(PUSH_AMO_EXT)
        if (BackendType::GDA_BACKEND != backend_type) { // GDA is 64-bit AMO only
          ROCSHMEM_AMO_EXT_TYPES_NONGDA(PUSH_AMO_EXT)
        }
      }
      #undef PUSH_AMO_EXT
      break;
    }
    case AMO_FCswapTestType: {
      test_name = "AMO Fetch_CSWAP";
      #define PUSH_AMO_STD(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(AMOStandardTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(AMOStandardTester<long long>(args));
        PUSH_TESTER(AMOStandardTester<long>(args));
        if (BackendType::GDA_BACKEND != backend_type) // GDA is 64-bit AMO only
          PUSH_TESTER(AMOStandardTester<int>(args));
      } else {
        ROCSHMEM_AMO_STD_TYPES_ALWAYS(PUSH_AMO_STD)
        if (BackendType::GDA_BACKEND != backend_type) { // GDA is 64-bit AMO only
          ROCSHMEM_AMO_STD_TYPES_NONGDA(PUSH_AMO_STD)
        }
      }
      #undef PUSH_AMO_STD
      break;
    }
    case AMO_AddTestType: {
      test_name = "AMO Add";
      #define PUSH_AMO_STD(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(AMOStandardTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(AMOStandardTester<long long>(args));
        PUSH_TESTER(AMOStandardTester<long>(args));
        if (BackendType::GDA_BACKEND != backend_type) // GDA is 64-bit AMO only
          PUSH_TESTER(AMOStandardTester<int>(args));
      } else {
        ROCSHMEM_AMO_STD_TYPES_ALWAYS(PUSH_AMO_STD)
        if (BackendType::GDA_BACKEND != backend_type) { // GDA is 64-bit AMO only
          ROCSHMEM_AMO_STD_TYPES_NONGDA(PUSH_AMO_STD)
        }
      }
      #undef PUSH_AMO_STD
      break;
    }
    case AMO_SetTestType: {
      test_name = "AMO Set";
      #define PUSH_AMO_EXT(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(AMOExtendedTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(AMOExtendedTester<long long>(args));
        PUSH_TESTER(AMOExtendedTester<long>(args));
        if (BackendType::GDA_BACKEND != backend_type) // GDA is 64-bit AMO only
          PUSH_TESTER(AMOExtendedTester<int>(args));
      } else {
        ROCSHMEM_AMO_EXT_TYPES_ALWAYS(PUSH_AMO_EXT)
        if (BackendType::GDA_BACKEND != backend_type) { // GDA is 64-bit AMO only
          ROCSHMEM_AMO_EXT_TYPES_NONGDA(PUSH_AMO_EXT)
        }
      }
      #undef PUSH_AMO_EXT
      break;
    }
    case AMO_SwapTestType: {
      test_name = "AMO Swap";
      #define PUSH_AMO_EXT(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(AMOExtendedTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(AMOExtendedTester<long long>(args));
        PUSH_TESTER(AMOExtendedTester<long>(args));
        if (BackendType::GDA_BACKEND != backend_type) // GDA is 64-bit AMO only
          PUSH_TESTER(AMOExtendedTester<int>(args));
      } else {
        ROCSHMEM_AMO_EXT_TYPES_ALWAYS(PUSH_AMO_EXT)
        if (BackendType::GDA_BACKEND != backend_type) { // GDA is 64-bit AMO only
          ROCSHMEM_AMO_EXT_TYPES_NONGDA(PUSH_AMO_EXT)
        }
      }
      #undef PUSH_AMO_EXT
      break;
    }
    case AMO_FetchAndTestType: {
      test_name = "AMO Fetch And";
      #define PUSH_AMO_BIT(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(AMOBitwiseTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(AMOBitwiseTester<unsigned long long>(args));
        PUSH_TESTER(AMOBitwiseTester<unsigned long>(args));
        if (BackendType::GDA_BACKEND != backend_type) // GDA is 64-bit AMO only
          PUSH_TESTER(AMOBitwiseTester<unsigned int>(args));
      } else {
        ROCSHMEM_AMO_BIT_TYPES_ALWAYS(PUSH_AMO_BIT)
        if (BackendType::GDA_BACKEND != backend_type) { // GDA is 64-bit AMO only
          ROCSHMEM_AMO_BIT_TYPES_NONGDA(PUSH_AMO_BIT)
        }
      }
      #undef PUSH_AMO_BIT
      break;
    }
    case AMO_AndTestType: {
      test_name = "AMO And";
      #define PUSH_AMO_BIT(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(AMOBitwiseTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(AMOBitwiseTester<unsigned long long>(args));
        PUSH_TESTER(AMOBitwiseTester<unsigned long>(args));
        if (BackendType::GDA_BACKEND != backend_type) // GDA is 64-bit AMO only
          PUSH_TESTER(AMOBitwiseTester<unsigned int>(args));
      } else {
        ROCSHMEM_AMO_BIT_TYPES_ALWAYS(PUSH_AMO_BIT)
        if (BackendType::GDA_BACKEND != backend_type) { // GDA is 64-bit AMO only
          ROCSHMEM_AMO_BIT_TYPES_NONGDA(PUSH_AMO_BIT)
        }
      }
      #undef PUSH_AMO_BIT
      break;
    }
    case AMO_FetchOrTestType: {
      test_name = "AMO Fetch Or";
      #define PUSH_AMO_BIT(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(AMOBitwiseTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(AMOBitwiseTester<unsigned long long>(args));
        PUSH_TESTER(AMOBitwiseTester<unsigned long>(args));
        if (BackendType::GDA_BACKEND != backend_type) // GDA is 64-bit AMO only
          PUSH_TESTER(AMOBitwiseTester<unsigned int>(args));
      } else {
        ROCSHMEM_AMO_BIT_TYPES_ALWAYS(PUSH_AMO_BIT)
        if (BackendType::GDA_BACKEND != backend_type) { // GDA is 64-bit AMO only
          ROCSHMEM_AMO_BIT_TYPES_NONGDA(PUSH_AMO_BIT)
        }
      }
      #undef PUSH_AMO_BIT
      break;
    }
    case AMO_OrTestType: {
      test_name = "AMO Or";
      #define PUSH_AMO_BIT(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(AMOBitwiseTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(AMOBitwiseTester<unsigned long long>(args));
        PUSH_TESTER(AMOBitwiseTester<unsigned long>(args));
        if (BackendType::GDA_BACKEND != backend_type) // GDA is 64-bit AMO only
          PUSH_TESTER(AMOBitwiseTester<unsigned int>(args));
      } else {
        ROCSHMEM_AMO_BIT_TYPES_ALWAYS(PUSH_AMO_BIT)
        if (BackendType::GDA_BACKEND != backend_type) { // GDA is 64-bit AMO only
          ROCSHMEM_AMO_BIT_TYPES_NONGDA(PUSH_AMO_BIT)
        }
      }
      #undef PUSH_AMO_BIT
      break;
    }
    case AMO_FetchXorTestType: {
      test_name = "AMO Fetch Xor";
      #define PUSH_AMO_BIT(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(AMOBitwiseTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(AMOBitwiseTester<unsigned long long>(args));
        PUSH_TESTER(AMOBitwiseTester<unsigned long>(args));
        if (BackendType::GDA_BACKEND != backend_type) // GDA is 64-bit AMO only
          PUSH_TESTER(AMOBitwiseTester<unsigned int>(args));
      } else {
        ROCSHMEM_AMO_BIT_TYPES_ALWAYS(PUSH_AMO_BIT)
        if (BackendType::GDA_BACKEND != backend_type) { // GDA is 64-bit AMO only
          ROCSHMEM_AMO_BIT_TYPES_NONGDA(PUSH_AMO_BIT)
        }
      }
      #undef PUSH_AMO_BIT
      break;
    }
    case AMO_XorTestType: {
      test_name = "AMO Xor";
      #define PUSH_AMO_BIT(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(AMOBitwiseTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(AMOBitwiseTester<unsigned long long>(args));
        PUSH_TESTER(AMOBitwiseTester<unsigned long>(args));
        if (BackendType::GDA_BACKEND != backend_type) // GDA is 64-bit AMO only
          PUSH_TESTER(AMOBitwiseTester<unsigned int>(args));
      } else {
        ROCSHMEM_AMO_BIT_TYPES_ALWAYS(PUSH_AMO_BIT)
        if (BackendType::GDA_BACKEND != backend_type) { // GDA is 64-bit AMO only
          ROCSHMEM_AMO_BIT_TYPES_NONGDA(PUSH_AMO_BIT)
        }
      }
      #undef PUSH_AMO_BIT
      break;
    }
    case AMO_IncTestType: {
      test_name = "AMO Inc";
      #define PUSH_AMO_STD(T, name) \
        if (args.type_coverage == TypeCoverage::Full || args.type_enabled(name)) \
          PUSH_TESTER(AMOStandardTester<T>(args));
      if (args.type_coverage == TypeCoverage::Minimal) {
        PUSH_TESTER(AMOStandardTester<long long>(args));
        PUSH_TESTER(AMOStandardTester<long>(args));
        if (BackendType::GDA_BACKEND != backend_type) // GDA is 64-bit AMO only
          PUSH_TESTER(AMOStandardTester<int>(args));
      } else {
        ROCSHMEM_AMO_STD_TYPES_ALWAYS(PUSH_AMO_STD)
        if (BackendType::GDA_BACKEND != backend_type) { // GDA is 64-bit AMO only
          ROCSHMEM_AMO_STD_TYPES_NONGDA(PUSH_AMO_STD)
        }
      }
      #undef PUSH_AMO_STD
      break;
    }
    case PingPongTestType:
      test_name = (args.num_wgs > 1) ? "PingPong (W>1: bidir BW)"
                                     : "PingPong";
      PUSH_TESTER(PingPongTester(args));
      break;
    case PingAllTestType:
      test_name = "PingAll";
      PUSH_TESTER(PingAllTester(args));
      break;
    case BarrierAllTestType:
      test_name = "Barrier_All";
      PUSH_TESTER(BarrierAllTester(args));
      break;
    case WAVEBarrierAllTestType:
      test_name = "WAVE Barrier_All";
      PUSH_TESTER(BarrierAllTester(args));
      break;
    case WGBarrierAllTestType:
      test_name = "WG Barrier_All";
      PUSH_TESTER(BarrierAllTester(args));
      break;
    case TeamBarrierTestType:
      test_name = "Team Barrier Test";
      PUSH_TESTER(TeamBarrierTester(args));
      break;
    case TeamWAVEBarrierTestType:
      test_name = "Team WAVE Barrier Test";
      PUSH_TESTER(TeamBarrierTester(args));
      break;
    case TeamWGBarrierTestType:
      test_name = "Team WG Barrier Test";
      PUSH_TESTER(TeamBarrierTester(args));
      break;
    case SyncAllTestType:
      test_name = "SyncAll";
      PUSH_TESTER(SyncAllTester(args));
      break;
    case WAVESyncAllTestType:
      test_name = "WAVE SyncAll";
      PUSH_TESTER(SyncAllTester(args));
      break;
    case WGSyncAllTestType:
      test_name = "WG SyncAll";
      PUSH_TESTER(SyncAllTester(args));
      break;
    case TeamSyncTestType:
      test_name = "Team Sync";
      PUSH_TESTER(TeamSyncTester(args));
      break;
    case TeamWAVESyncTestType:
      test_name = "Team WAVE Sync";
      PUSH_TESTER(TeamSyncTester(args));
      break;
    case TeamWGSyncTestType:
      test_name = "Team WG Sync";
      PUSH_TESTER(TeamSyncTester(args));
      break;
    case RandomAccessTestType:
      test_name = "Random_Access";
      PUSH_TESTER(RandomAccessTester(args));
      break;
    case ShmemPtrTestType:
      test_name = "Shmem_Ptr";
      PUSH_TESTER(ShmemPtrTester(args));
      break;
    case WGGetTestType:
      test_name = "Blocking WG level Gets";
      PUSH_TESTER(WorkGroupPrimitiveTester(args));
      break;
    case WGGetNBITestType:
      test_name = "Non-Blocking WG level Gets";
      PUSH_TESTER(WorkGroupPrimitiveTester(args));
      break;
    case WGPutTestType:
      test_name = "Blocking WG level Puts";
      PUSH_TESTER(WorkGroupPrimitiveTester(args));
      break;
    case WGPutNBITestType:
      test_name = "Non-Blocking WG level Puts";
      PUSH_TESTER(WorkGroupPrimitiveTester(args));
      break;
    case WAVEGetTestType:
      test_name = "Blocking WAVE level Gets";
      PUSH_TESTER(WaveFrontPrimitiveTester(args));
      break;
    case WAVEGetNBITestType:
      test_name = "Non-Blocking WAVE level Gets";
      PUSH_TESTER(WaveFrontPrimitiveTester(args));
      break;
    case WAVEPutTestType:
      test_name = "Blocking WAVE level Puts";
      PUSH_TESTER(WaveFrontPrimitiveTester(args));
      break;
    case WAVEPutNBITestType:
      test_name = "Non-Blocking WAVE level Puts";
      PUSH_TESTER(WaveFrontPrimitiveTester(args));
      break;
    case PutSignalTestType:
      test_name = "Putmem Signal";
      PUSH_TESTER(SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      PUSH_TESTER(SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      break;
    case WGPutSignalTestType:
      test_name = "WG Putmem Signal";
      PUSH_TESTER(SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      PUSH_TESTER(SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      break;
    case WAVEPutSignalTestType:
      test_name = "Wave Putmem Signal";
      PUSH_TESTER(SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      PUSH_TESTER(SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      break;
    case PutSignalNBITestType:
      test_name = "Non-Blocking Putmem Signal";
      PUSH_TESTER(SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      PUSH_TESTER(SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      break;
    case WGPutSignalNBITestType:
      test_name = "Non-Blocking WG Putmem Signal";
      PUSH_TESTER(SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      PUSH_TESTER(SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      break;
    case WAVEPutSignalNBITestType:
      test_name = "Non-Blocking Wave Putmem Signal";
      PUSH_TESTER(SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      PUSH_TESTER(SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      break;
    case SignalFetchTestType:
      test_name = "Signal Fetch";
      PUSH_TESTER(SignalingOperationsTester(args));
      break;
    case WGSignalFetchTestType:
      test_name = "WG Signal Fetch";
      PUSH_TESTER(SignalingOperationsTester(args));
      break;
    case WAVESignalFetchTestType:
      test_name = "Wave Signal Fetch";
      PUSH_TESTER(SignalingOperationsTester(args));
      break;
    case FloodPutTestType:
      test_name = "Flood Put (multidirectional)";
      PUSH_TESTER(FloodTester(args));
      break;
    case FloodPutNBITestType:
      test_name = "Flood Non-Blocking Put (multidirectional)";
      PUSH_TESTER(FloodTester(args));
      break;
    case FloodPTestType:
      test_name = "Flood P (multidirectional)";
      PUSH_TESTER(FloodTester(args));
      break;
    case FloodGetTestType:
      test_name = "Flood Get (multidirectional)";
      PUSH_TESTER(FloodTester(args));
      break;
    case FloodGetNBITestType:
      test_name = "Flood Non-Blocking Get (multidirectional)";
      PUSH_TESTER(FloodTester(args));
      break;
    case FloodGTestType:
      test_name = "Flood G (multidirectional)";
      PUSH_TESTER(FloodTester(args));
      break;
    case HipModuleInitTestType:
      test_name = "HIP Module Init Test";
      PUSH_TESTER(HipModuleInitTester(args));
      break;
    case FloodAddTestType:
      test_name = "Flood Add (multidirectional)";
      PUSH_TESTER(FloodAmoTester(args));
      break;
    case FloodFAddTestType:
      test_name = "Flood FAdd (multidirectional)";
      PUSH_TESTER(FloodAmoTester(args));
      break;
    case FloodWaitAmoTestType:
      test_name = "Flood WaitAdd (multidirectional)";
      PUSH_TESTER(FloodAmoTester(args));
      break;
    case DeviceBitcodeTestType:
      test_name = "Device Bitcode Test";
      PUSH_TESTER(DeviceBitcodeTester(args));
      break;
    case LibraryInfoTestType:
      test_name = "Library Info Test";
      PUSH_TESTER(LibraryInfoTester(args));
      break;
    case FenceOrderPutWaveSignalTestType:
      test_name = "Fence PutWaveSignal Ordering";
      PUSH_TESTER(FenceOrderingTester(args));
      break;
    case FenceOrderPutLargeSmallTestType:
      test_name = "Fence PutLargeSmall Ordering";
      PUSH_TESTER(FenceOrderingTester(args));
      break;
    case FenceOrderFanoutTestType:
      test_name = "Fence Fanout Ordering";
      PUSH_TESTER(FenceOrderingTester(args));
      break;
    case FenceOrderPutWaveNbiChunksTestType:
      test_name = "Fence PutWaveNbiChunks Ordering";
      PUSH_TESTER(FenceOrderingTester(args));
      break;
    case TilePutContiguousTestType:
      test_name = "Tile Put Contiguous";
      PUSH_TESTER(TileRMATester(args));
      break;
    case TilePutRowMajorTestType:
      test_name = "Tile Put Row-Major";
      PUSH_TESTER(TileRMATester(args));
      break;
    case TilePutColumnMajorTestType:
      test_name = "Tile Put Column-Major";
      PUSH_TESTER(TileRMATester(args));
      break;
    case TilePutArbitraryTestType:
      test_name = "Tile Put Arbitrary Strides";
      PUSH_TESTER(TileRMATester(args));
      break;
    case TilePutWaveContiguousTestType:
      test_name = "Tile Put Wave-Collective Contiguous";
      PUSH_TESTER(TileRMATester(args));
      break;
    case TilePutWGContiguousTestType:
      test_name = "Tile Put Workgroup-Collective Contiguous";
      PUSH_TESTER(TileRMATester(args));
      break;
    case TileGetContiguousTestType:
      test_name = "Tile Get Contiguous";
      PUSH_TESTER(TileRMATester(args));
      break;
    case TileGetWGContiguousTestType:
      test_name = "Tile Get Workgroup-Collective Contiguous";
      PUSH_TESTER(TileRMATester(args));
      break;
    case TilePut1DTestType:
      test_name = "Tile Put 1D Tensor";
      PUSH_TESTER(TileRMATester(args));
      break;
    case TileGet1DTestType:
      test_name = "Tile Get 1D Tensor";
      PUSH_TESTER(TileRMATester(args));
      break;
    case TileGetWaveContiguousTestType:
      test_name = "Tile Get Wave-Collective Contiguous";
      PUSH_TESTER(TileRMATester(args));
      break;
    case TileGetRowMajorTestType:
      test_name = "Tile Get Row-Major";
      PUSH_TESTER(TileRMATester(args));
      break;
    case TileGetColumnMajorTestType:
      test_name = "Tile Get Column-Major";
      PUSH_TESTER(TileRMATester(args));
      break;
    case TileGetArbitraryTestType:
      test_name = "Tile Get Arbitrary Strides";
      PUSH_TESTER(TileRMATester(args));
      break;
    case HostTeamSyncBarrierTestType:
      test_name = "Host Team Sync/Barrier";
      PUSH_TESTER(HostTeamSyncBarrierTester(args));
      break;
    case ReduceOnStreamTestType:
      test_name = "Reduce On Stream";
      PUSH_TESTER(ReduceOnStreamTester<int>(args));
      break;
    case HostCtxCreateTestType:
      test_name = "Host CTX Create";
      PUSH_TESTER(HostCtxCreateTester(args));
      break;
    case TeamSplit2DTestType:
      test_name = "Team Split 2D";
      PUSH_TESTER(TeamSplit2DTester(args));
      break;
    case TileBroadcastTestType:
      test_name = "Tile Broadcast";
      PUSH_TESTER(TileBroadcastTester(args));
      break;
    case TileBroadcastWaveTestType:
      test_name = "Tile Broadcast Wave-Collective";
      PUSH_TESTER(TileBroadcastTester(args));
      break;
    case TileBroadcastWGTestType:
      test_name = "Tile Broadcast Workgroup-Collective";
      PUSH_TESTER(TileBroadcastTester(args));
      break;
    case TileAllgatherTestType:
      test_name = "Tile Allgather";
      PUSH_TESTER(TileAllgatherTester(args));
      break;
    case TileAllgatherWaveTestType:
      test_name = "Tile Allgather Wave-Collective";
      PUSH_TESTER(TileAllgatherTester(args));
      break;
    case TileAllgatherWGTestType:
      test_name = "Tile Allgather Workgroup-Collective";
      PUSH_TESTER(TileAllgatherTester(args));
      break;
    case TileReduceTestType:
      test_name = "Tile Reduce";
      PUSH_TESTER(TileReduceTester(args));
      break;
    case TileReduceWaveTestType:
      test_name = "Tile Reduce Wave-Collective";
      PUSH_TESTER(TileReduceTester(args));
      break;
    case TileReduceWGTestType:
      test_name = "Tile Reduce Workgroup-Collective";
      PUSH_TESTER(TileReduceTester(args));
      break;
#if defined(USE_GDA)
    case QpPingPongTestType:
      test_name = (args.num_wgs > 1) ? "QP-Direct PingPong (W>1: bidir BW)"
                                     : "QP-Direct PingPong";
      PUSH_TESTER(QpPingPongTester(args));
      break;
    case QpPutNbiTestType:
      test_name = "QP-Direct Put NBI";
      PUSH_TESTER(QpPutNbiTester(args));
      break;
#endif
#if defined(USE_SDMA)
    case SdmaPingPongTestType:
      test_name = (args.num_wgs > 1) ? "SDMA-Direct PingPong (W>1: bidir BW)"
                                     : "SDMA-Direct PingPong";
      PUSH_TESTER(SdmaPingPongTester(args));
      break;
    case SdmaPutNbiTestType:
      test_name = "SDMA-Direct Put NBI";
      PUSH_TESTER(SdmaPutNbiTester(args));
      break;
#endif
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

#undef PUSH_TESTER

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
    case TeamReduceScatterTestType:
    case ReduceWaveTestType:
    case TeamReduceScatterWaveTestType:
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
    case FcollectWaveTestType:
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
    case BroadcastWaveTestType:
    case AllToAllWaveTestType:
    case TileReduceTestType:
    case TileReduceWaveTestType:
    case TileReduceWGTestType:
    case QpPingPongTestType:
    case SdmaPingPongTestType:
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
    case HostWaitUntilTestType:
    case HostTestTestType:
    case HostWaitUntilAllTestType:
    case HostWaitUntilAnyTestType:
    case HostWaitUntilSomeTestType:
    case HostWaitUntilAllVectorTestType:
    case HostWaitUntilAnyVectorTestType:
    case HostWaitUntilSomeVectorTestType:
    case HostWaitUntilAllStatusTestType:
    case HostWaitUntilAnyStatusTestType:
    case HostWaitUntilSomeStatusTestType:
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
    const std::string tname = typeName();
    std::string type_header = "";
    if (!tname.empty()) {
      type_header = "   Type: " + tname;
    }
    printf("%-*s%-*s%-*s%*s%*s%*s%s\n",
           15, "# Volume (B)",
           15, "Msg Size (B)",
           15, "# of timed Msgs",
           field_width, "Latency (us)",
           field_width, "Bandwidth (GB/s)",
           field_width + 1, "Msg Rate (Msg/s)",
           type_header.c_str());
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
        std::cerr << "Error allocating memory from symmetric heap: requested "
                  << size << " bytes" << std::endl;
        std::cerr << "Raise ROCSHMEM_HEAP_SIZE, or lower -v / -w / -z / -b"
                  << std::endl;
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
