/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "TestBed.hpp"

namespace RcclUnitTesting
{
  // Large-message gather exercises the asymmetric P2P channel scaling path
  // in addP2pToPlan(). With nChannelsMax > nChannelsMin (e.g. 16-GPU DPX+NPS2),
  // gather should use nChannelsMax since only the root rank is the traffic hub.
  TEST(P2pChannelScaling, GatherLargeMessage)
  {
    TestBed testBed;

    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {16 * 1024 * 1024};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  // Large-message scatter exercises the same asymmetric path from the
  // send direction (root fans out to all peers).
  TEST(P2pChannelScaling, ScatterLargeMessage)
  {
    TestBed testBed;

    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollScatter};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {16 * 1024 * 1024};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  // Large-message alltoall is symmetric (all ranks send and receive),
  // so it must NOT get the boosted channel count.
  TEST(P2pChannelScaling, AllToAllLargeMessage)
  {
    TestBed testBed;

    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAlltoAll};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {16 * 1024 * 1024};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  // Large-message alltoallv is symmetric, same as alltoall.
  // AllToAllV requires explicit sendcounts/recvcounts/displacements.
  TEST(P2pChannelScaling, AllToAllVLargeMessage)
  {
    TestBed testBed;

    bool const inPlace       = false;
    bool const useManagedMem = false;
    bool const useHipGraph   = false;

    OptionalColArgs options;

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder));

      int const chunkSize = 65536;
      std::vector<size_t> numInputElements(totalRanks, 0);
      std::vector<size_t> numOutputElements(totalRanks, 0);

      for (int s = 0; s < totalRanks; ++s)
      for (int r = 0; r < totalRanks; ++r)
      {
        int numElements = (1 + s + r) * chunkSize;
        options.sendcounts[s * totalRanks + r] = numElements;
        options.recvcounts[r * totalRanks + s] = numElements;
      }
      for (int s = 0; s < totalRanks; ++s)
      {
        int totalSend = 0, totalRecv = 0;
        for (int r = 0; r < totalRanks; ++r)
        {
          options.sdispls[s * totalRanks + r] = totalSend;
          options.rdispls[s * totalRanks + r] = totalRecv;
          totalSend += options.sendcounts[s * totalRanks + r];
          totalRecv += options.recvcounts[s * totalRanks + r];
        }
        numInputElements[s]  = totalSend;
        numOutputElements[s] = totalRecv;
      }

      for (int rank = 0; rank < totalRanks; ++rank)
      {
        testBed.SetCollectiveArgs(ncclCollAlltoAllv,
                                  ncclFloat32,
                                  numInputElements[rank],
                                  numOutputElements[rank],
                                  options,
                                  -1,
                                  0,
                                  rank);
      }
      testBed.AllocateMem(inPlace, useManagedMem);
      testBed.PrepareData();
      testBed.ExecuteCollectives({}, useHipGraph);
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  // Non-root gather to exercise asymmetric detection from a different root.
  TEST(P2pChannelScaling, GatherNonZeroRoot)
  {
    TestBed testBed;

    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclBfloat16};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {1};
    std::vector<int>            const numElements     = {16 * 1024 * 1024};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }
}
