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

#ifndef THREAD_TRACE_HPP_
#define THREAD_TRACE_HPP_

#include "top.hpp"
#include "device/device.hpp"
#include "amdocl/cl_thread_trace_amd.h"


namespace amd {

#define THREAD_TRACE_BUFFER_DEFAULT_SIZE 4096

/*! \addtogroup Runtime
 *  @{
 *
 *  \addtogroup Threadtrace
 *  @{
 */

/*! \class ThreadTrace
 *
 *  \brief The container class for the thread traces
 */
class ThreadTrace : public RuntimeObject {
 public:
  enum State { Undefined, MemoryBound, Begin, End, Pause };
  typedef struct ThreadTraceConfigRec {
    size_t configSize_;   // structure size
    size_t cu_;           // target compute unit [cu]
    size_t sh_;           // target shader array [sh],that contains target cu
    size_t simdMask_;     // bitmask to enable or disable target tokens for different SIMDs
    size_t vmIdMask_;     // virtual memory [vm] IDs to capture
    size_t tokenMask_;    // bitmask indicating which trace token IDs will be included in the trace
    size_t regMask_;      // bitmask indicating which register types should be included in the trace
    size_t instMask_;     // types of instruction scheduling updates which should be recorded
    size_t randomSeed_;   // linear feedback shift register [LFSR] seed
    size_t userData_;     // user data ,which is written as payload
    size_t captureMode_;  // indicator for the way how THREAD_TRACE_START / STOP events affect token
                          // collection
    bool isUserData_;     // indicator if user_data is set
    bool isWrapped_;  // indicator if the memory buffer should be wrapped around instead of stopping
                      // at the end
    // default thread trace configuration/s initializator
    ThreadTraceConfigRec()
        : configSize_(0),
          cu_(0),
          sh_(0),
          simdMask_(0xF),
          vmIdMask_(CL_THREAD_TRACE_VM_ID_MASK_SINGLE),
          tokenMask_(CL_THREAD_TRACE_TOKEN_MASK_ALL_SI),
          regMask_(CL_THREAD_TRACE_REG_MASK_ALL_SI),
          instMask_(CL_THREAD_TRACE_INST_MASK_ALL),
          randomSeed_(0xFFF),
          userData_(0),
          captureMode_(CL_THREAD_TRACE_CAPTURE_ALL),
          isUserData_(false),
          isWrapped_(false) {
      configSize_ = sizeof(struct ThreadTraceConfigRec);
    }
  } ThreadTraceConfig;

  //! Constructor of the thread trace object
  ThreadTrace(const Device& device)  //!< device object
      : deviceThreadTrace_(NULL), device_(device), state_(Undefined) {}

  //! Get the thread trace's associated device
  const Device& device() const { return device_; }

  //! Get the shader engines number for thread trace`s associated device
  const size_t deviceSeNumThreadTrace() const { return device_.info().numberOfShaderEngines; }

  //! Get the device thread trace
  device::ThreadTrace* getDeviceThreadTrace() { return deviceThreadTrace_; }

  //! Set the device thread trace
  void setDeviceThreadTrace(device::ThreadTrace* threadTrace) { deviceThreadTrace_ = threadTrace; }

  void setState(State state) { state_ = state; }
  State getState() { return state_; }

  void setCU(unsigned int cu) { threadTraceConfig_.cu_ = cu; }

  void setSH(unsigned int sh) { threadTraceConfig_.sh_ = sh; }

  void setSIMD(unsigned int simdMask) { threadTraceConfig_.simdMask_ = simdMask; }

  void setUserData(unsigned int userData) {
    threadTraceConfig_.isUserData_ = true;
    threadTraceConfig_.userData_ = userData;
  }

  void setTokenMask(unsigned int tokenMask) { threadTraceConfig_.tokenMask_ = tokenMask; }

  void setRegMask(unsigned int regMask) { threadTraceConfig_.regMask_ = regMask; }

  void setVmIdMask(unsigned int vmIdMask) { threadTraceConfig_.vmIdMask_ = vmIdMask; }

  void setInstMask(unsigned int instMask) { threadTraceConfig_.instMask_ = instMask; }

  void setRandomSeed(unsigned int randomSeed) { threadTraceConfig_.randomSeed_ = randomSeed; }

  void setCaptureMode(unsigned int captureMode) { threadTraceConfig_.captureMode_ = captureMode; }

  void setIsWrapped(bool isWrapped) { threadTraceConfig_.isWrapped_ = isWrapped; }

  const ThreadTraceConfig& threadTraceConfig() const { return threadTraceConfig_; }

  //! RTTI internal implementation
  virtual ObjectType objectType() const { return ObjectTypeThreadTrace; }

 protected:
  //! Destructor for ThreadTrace class
  ~ThreadTrace() { delete deviceThreadTrace_; }

  device::ThreadTrace* deviceThreadTrace_;  //!< device thread trace object
  const Device& device_;                    //!< the device object
  State state_;
  ThreadTraceConfig threadTraceConfig_;
};

/*@}*/
/*@}*/  // namespace amd
}  // namespace amd

#endif  // THREAD_TRACE_HPP_
