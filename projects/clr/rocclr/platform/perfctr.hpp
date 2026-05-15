/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef PERFCTR_HPP_
#define PERFCTR_HPP_

#include "top.hpp"
#include "device/device.hpp"
#include "amdocl/cl_profile_amd.h"

namespace amd {

/*! \addtogroup Runtime
 *  @{
 *
 *  \addtogroup Perfcounter
 *  @{
 */

/*! \class PerfCounter
 *
 *  \brief The container class for the performance counters
 */
class PerfCounter : public RuntimeObject {
 public:
  typedef std::unordered_map<cl_perfcounter_property, ulong> Properties;

  //! Constructor of the performance counter object
  PerfCounter(const Device& device,    //!< device object
              Properties& properties)  //!< a list of properties
      : properties_(properties), deviceCounter_(NULL), device_(device) {}

  //! Get the performance counter's result
  const Device& device() const { return device_; }

  //! Get the properties
  const Properties& properties() const { return properties_; }

  //! Get the device performance counter
  const device::PerfCounter* getDeviceCounter() const { return deviceCounter_; }

  device::PerfCounter* getDeviceCounter() { return deviceCounter_; }

  //! Set the device performance counter
  void setDeviceCounter(device::PerfCounter* counter) { deviceCounter_ = counter; }

  //! RTTI internal implementation
  virtual ObjectType objectType() const { return ObjectTypePerfCounter; }

 protected:
  //! Destructor for PerfCounter class
  ~PerfCounter() { delete deviceCounter_; }

  Properties properties_;               //!< the perf counter properties
  device::PerfCounter* deviceCounter_;  //!< device performance counter
  const Device& device_;                //!< the device object
};

/*@}*/
/*@}*/  // namespace amd
}  // namespace amd

#endif  // PERFCTR_HPP_
