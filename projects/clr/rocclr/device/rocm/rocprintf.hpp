/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

/*! \addtogroup GPU GPU Device Implementation
 *  @{
 */

#ifndef copysign
#ifdef _MSC_VER
#define copysign(X, Y) (_copysign(X, Y))
#endif  //_MSC_VER
#endif  // copysign

//! GPU Device Implementation
namespace amd::roc {

class Kernel;
class VirtualGPU;
class Device;

class PrintfDbg : public amd::HeapObject {
 public:
  //! Debug buffer size per workitem
  static constexpr uint WorkitemDebugSize = 4096;

  //! constructor
  PrintfDbg(Device& device, FILE* file = nullptr);

  //! Destructor
  ~PrintfDbg();

  //! Initializes the debug buffer before kernel's execution
  bool init(bool printfEnabled  //!< checks for printf
  );

  //! Prints the kernel's debug informaiton from the buffer
  bool output(VirtualGPU& gpu,
              bool printfEnabled,                                //!< checks for printf
              const std::vector<device::PrintfInfo>& printfInfo  //!< printf info
  );

  //! Returns debug buffer object
  address dbgBuffer() const { return dbgBuffer_; }

 protected:
  address dbgBuffer_;      //!< Buffer to hold debug output
  size_t dbgBuffer_size_;  //!< Size of the debugger buffer
  FILE* dbgFile_;          //!< Debug file
  Device& gpuDevice_;      //!< GPU device object

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
};

/*@}*/  // namespace amd::roc
}  // namespace amd::roc
