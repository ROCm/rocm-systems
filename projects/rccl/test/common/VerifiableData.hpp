/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef VERIFIABLE_DATA_HPP
#define VERIFIABLE_DATA_HPP
#include "ErrCode.hpp"

namespace RcclUnitTesting
{
  class CollectiveArgs;

  // FP8 reductions whose input and single expected result come from the verifiable
  // generator (seeded, exactly reducible values, checked on the GPU; see VerifiableFp8.hpp)
  bool UseVerifiableData(CollectiveArgs const& collArgs);

  // Fills this rank's input and clears the output
  ErrCode VerifiablePrepData(CollectiveArgs& collArgs);

  // Compares outputGpu against the generated expected values on the GPU
  ErrCode VerifiableValidate(CollectiveArgs& collArgs);
}

#endif // VERIFIABLE_DATA_HPP
