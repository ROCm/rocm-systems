/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

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

#pragma once

#include "device/pal/palmemory.hpp"

/*! \addtogroup GPU GPU Device Implementation
 *  @{
 */
#ifndef isinf
#ifdef _MSC_VER
#define isinf(X) (!_finite(X) && !_isnan(X))
#else  //!_MSC_VER
#define isinf(X) (std::isinf(X))
#endif  //!_MSC_VER
#endif  // isinf

#ifndef isnan
#ifdef _MSC_VER
#define isnan(X) (_isnan(X))
#else  //!_MSC_VER
#define isnan(X) (std::isnan(X))
#endif  //!_MSC_VER
#endif  // isnan

#ifndef copysign
#ifdef _MSC_VER
#define copysign(X, Y) (_copysign(X, Y))
#endif  //_MSC_VER
#endif  // copysign

//! GPU Device Implementation
namespace amd::pal {

class Kernel;
class VirtualGPU;
class Memory;

class PrintfDbg : public amd::HeapObject {
 public:
  //! Debug buffer size per workitem
  static constexpr uint WorkitemDebugSize = 4096;

  //! Default constructor
  PrintfDbg(Device& device, FILE* file = NULL);

  //! Destructor
  ~PrintfDbg();

  //! Creates the PrintfDbg object
  bool create();

  //! Initializes the debug buffer before kernel's execution
  bool init(VirtualGPU& gpu,          //!< Virtual GPU object
            bool printfEnabled,       //!< checks for printf
            const amd::NDRange& size  //!< Kernel's workload
  );

  //! Prints the kernel's debug informaiton from the buffer
  bool output(VirtualGPU& gpu,                                   //!< Virtual GPU object
              bool printfEnabled,                                //!< checks for printf
              const amd::NDRange& size,                          //!< Kernel's workload
              const std::vector<device::PrintfInfo>& printfInfo  //!< printf info
  );

  //! Debug buffer size per workitem
  size_t wiDbgSize() const { return wiDbgSize_; }

  //! Returns debug buffer object
  Memory* dbgBuffer() const { return dbgBuffer_; }

 protected:
  Memory* dbgBuffer_;    //!< Buffer to hold debug output
  FILE* dbgFile_;        //!< Debug file
  Device& gpuDevice_;    //!< GPU device object
  Memory* xferBufRead_;  //!< Transfer buffer for the dump read

  //! Gets GPU device object
  Device& dev() const { return gpuDevice_; }

  //! Allocates the debug buffer
  bool allocate(bool realloc = false  //!< If TRUE then reallocate the debug memory
  );

  //! Returns TRUE if a float value has to be printed
  bool checkFloat(const std::string& fmt  //!< Format string
  ) const;

  //! Returns TRUE if a string value has to be printed
  bool checkString(const std::string& fmt  //!< Format string
  ) const;

  //! Finds the specifier in the format string
  int checkVectorSpecifier(const std::string& fmt,  //!< Format string
                           size_t startPos,         //!< Start position for processing
                           size_t& curPos           //!< End position for processing
  ) const;

  //! Outputs an argument
  size_t outputArgument(const std::string& fmt,  //!< Format strint
                        bool printFloat,         //!< Argument is a float value
                        size_t size,             //!< Argument's size
                        const void* argument     //!< Argument's location
  ) const;

  //! Displays the PrintfDbg
  void outputDbgBuffer(const device::PrintfInfo& info,  //!< printf info
                       const uint32_t* workitemData,    //!< The PrintfDbg dump buffer
                       size_t& i                        //!< index to the data in the buffer
  ) const;

 private:
  //! Disable copy constructor
  PrintfDbg(const PrintfDbg&);

  //! Disable assignment
  PrintfDbg& operator=(const PrintfDbg&);

  //! Returns the pointer to the workitem data block
  bool clearWorkitems(VirtualGPU& gpu,  //!< Virtual GPU object
                      size_t idxStart,  //!< Workitem global index start
                      size_t number     //!< Number of workitems to clear
  ) const;

  //! Returns the pointer to the workitem data block
  uint32_t* mapWorkitem(VirtualGPU& gpu,  //!< Virtual GPU object
                        size_t idx,       //!< Workitem global index
                        bool* realloc     //!< Returns TRUE if workitem reached the buffer limit
  );

  //! Unamp the staged buffer
  void unmapWorkitem(VirtualGPU& gpu,              //!< Virtual GPU object
                     const uint32_t* workitemData  //!< The PrintfDbg dump buffer
  ) const;

  size_t wiDbgSize_;     //!< Workitem debug size
  Memory initCntValue_;  //!< Initialized count value
};
class PrintfDbgHSA : public PrintfDbg {
 public:
  //! Default constructor
  PrintfDbgHSA(Device& device, FILE* file = NULL) : PrintfDbg(device, file) {}

  //! Initializes the debug buffer before kernel's execution
  bool init(VirtualGPU& gpu,    //!< Virtual GPU object
            bool printfEnabled  //!< checks for printf
  );

  //! Prints the kernel's debug informaiton from the buffer
  bool output(VirtualGPU& gpu,                                   //!< Virtual GPU object
              bool printfEnabled,                                //!< checks for printf
              const std::vector<device::PrintfInfo>& printfInfo  //!< printf info
  );

 private:
  //! Disable copy constructor
  PrintfDbgHSA(const PrintfDbgHSA&);

  //! Disable assignment
  PrintfDbgHSA& operator=(const PrintfDbgHSA&);
};

/*@}*/  // namespace amd::pal
}  // namespace amd::pal
