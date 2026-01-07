 /*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once
#include "ErrCode.hpp"

namespace RcclUnitTesting
{
  class CollectiveArgs;

  // Checks that enough memory has been allocated
  ErrCode CheckAllocation(CollectiveArgs const& collArgs);

  // Default PrepareData functions
  // PrepareData functions are responsible for setting up input / expected for the given collArgs
  ErrCode DefaultPrepareDataFunc(CollectiveArgs &collArgs);
  ErrCode DefaultPrepData_Broadcast(CollectiveArgs &collArgs);
  ErrCode DefaultPrepData_Reduce(CollectiveArgs &collArgs, bool const isAllReduce);
  ErrCode DefaultPrepData_Gather(CollectiveArgs &collArgs, bool const isAllGather);
  ErrCode DefaultPrepData_ReduceScatter(CollectiveArgs &collArgs);
  ErrCode DefaultPrepData_Scatter(CollectiveArgs &collArgs);
  ErrCode DefaultPrepData_AllToAll(CollectiveArgs &collArgs);
  ErrCode DefaultPrepData_AllToAllv(CollectiveArgs &collArgs);
  ErrCode DefaultPrepData_Send(CollectiveArgs &collArgs);
  ErrCode DefaultPrepData_Recv(CollectiveArgs &collArgs);
}
