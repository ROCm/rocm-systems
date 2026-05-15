/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

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
