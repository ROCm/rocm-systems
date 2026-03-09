/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

/*! \addtogroup GPU GPU Device Implementation
 *  @{
 */

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
