/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef COUNTERS_HPP_
#define COUNTERS_HPP_

#include "top.hpp"

namespace amd {

/*! \addtogroup Runtime
 *  @{
 *
 *  \addtogroup Devicecounter
 *  @{
 */

/*! \class Counter
 *
 *  \brief The container class for the performance counters
 */
class Counter : public RuntimeObject {
 public:
  //! RTTI internal implementation
  virtual ObjectType objectType() const { return ObjectTypeCounter; }
};

/*@}*/
/*@}*/  // namespace amd
}  // namespace amd

#endif  // COUNTERS_HPP_
