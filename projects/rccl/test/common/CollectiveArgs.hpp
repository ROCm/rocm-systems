/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once
#include "PtrUnion.hpp"
#include "PrepDataFuncs.hpp"
#include "rccl/rccl.h"

namespace RcclUnitTesting
{
  // Enumeration of all collective functions currently supported
  typedef enum
  {
    ncclCollBroadcast = 0,
    ncclCollReduce,
    ncclCollAllGather,
    ncclCollReduceScatter,
    ncclCollAllReduce,
    ncclCollGather,
    ncclCollScatter,
    ncclCollAllToAll,
    ncclCollAllToAllv,
    ncclCollSend,
    ncclCollRecv,
    ncclNumFuncs
  } ncclFunc_t;

  char const ncclFuncNames[ncclNumFuncs][32] =
  {
    "Broadcast",
    "Reduce",
    "AllGather",
    "ReduceScatter",
    "AllReduce",
    "Gather",
    "Scatter",
    "AllToAll",
    "AllToAllv",
    "Send",
    "Recv"
  };

  char const ncclDataTypeNames[ncclNumTypes][32] =
  {
    "ncclInt8",
    "ncclUint8",
    "ncclInt32",
    "ncclUint32",
    "ncclInt64",
    "ncclUint64",
    "ncclFloat16",
    "ncclFloat32",
    "ncclFloat64",
    "ncclBfloat16",
    "ncclFloat8e4m3",
    "ncclFloat8e5m2"
  };

  char const ncclRedOpNames[ncclNumOps][32] =
  {
    "sum",
    "prod",
    "max",
    "min",
    "avg"
  };

  class CollectiveArgs;

  #define MAX_RANKS 32
  struct ScalarTransport
  {
    char ptr[MAX_RANKS * sizeof(double)];
  };

  struct OptionalColArgs
  {
    ncclRedOp_t     redOp = ncclSum;
    int             root = 0;               // Used as "peer" for Send/Recv
    ScalarTransport scalarTransport;        // Used for custom reduction operators
    int             scalarMode = -1;        // -1 if scalar not used

    // allToAllv args
    size_t          sendcounts[MAX_RANKS*MAX_RANKS];
    size_t          sdispls[MAX_RANKS*MAX_RANKS];
    size_t          recvcounts[MAX_RANKS*MAX_RANKS];
    size_t          rdispls[MAX_RANKS*MAX_RANKS];
  };

  // Function pointer for functions that operate on CollectiveArgs
  // e.g. For filling input / computing expected results
  typedef ErrCode (*CollFuncPtr)(CollectiveArgs &);

  class CollectiveArgs
  {
  public:
    // Arguments to execute
    int             globalRank;
    int             totalRanks;
    int             deviceId;
    ncclFunc_t      funcType;
    ncclDataType_t  dataType;
    size_t          numInputElements;
    size_t          numOutputElements;
    PtrUnion        localScalar;
    int             streamIdx;
    OptionalColArgs options;

    // Data
    PtrUnion       inputGpu;
    PtrUnion       outputGpu;
    PtrUnion       outputCpu;
    PtrUnion       expected;
    bool           inPlace;
    bool           useManagedMem;
    bool           userRegistered;
    void*          commRegHandle;
    size_t         numInputBytesAllocated;
    size_t         numOutputBytesAllocated;
    size_t         numInputElementsAllocated;
    size_t         numOutputElementsAllocated;

    // Set collective arguments
    ErrCode SetArgs(int             const globalRank,
                    int             const totalRanks,
                    int             const deviceId,
                    ncclFunc_t      const funcType,
                    ncclDataType_t  const dataType,
                    size_t          const numInputElements,
                    size_t          const numOutputElements,
                    int             const streamIdx,
                    OptionalColArgs const &optionalArgs = {});

    // Allocates GPU memory for input/output and CPU memory for expected
    // When inPlace is true, input and output share the same memory
    ErrCode AllocateMem(bool   const inPlace,
                        bool   const useManagedMem,
                        bool   const userRegistered);

    // Execute the provided data preparation function to fill input and compute expected results
    ErrCode PrepareData(CollFuncPtr const prepareDataFunc);

    // Compare outputs to expected values
    ErrCode ValidateResults();

    // Deallocate memory
    ErrCode DeallocateMem();

    // Provide a description for the current collective arguments
    std::string GetDescription() const;

    // Returns the number of inputs/outputs based on collective function type
    static void GetNumElementsForFuncType(ncclFunc_t const funcType,
                                          int        const N,
                                          int        const totalRanks,
                                          int*             numInputElements,
                                          int*             numOutputElements);

    // Returns true if collective function performs reduction
    static bool UsesReduce(ncclFunc_t const funcType);

    // Returns true if collective function utilizes a root rank
    static bool UsesRoot(ncclFunc_t const funcType);
  };
}
