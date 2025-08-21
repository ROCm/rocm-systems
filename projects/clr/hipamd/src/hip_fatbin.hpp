/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_FAT_BINARY_HPP
#define HIP_FAT_BINARY_HPP

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip_internal.hpp"
#include "platform/program.hpp"

// Forward declaration for Unique FD
struct UniqueFD;

namespace hip {

// Fat Binary Info
class FatBinaryInfo {
 public:
  FatBinaryInfo(const char* fname, const void* image);
  ~FatBinaryInfo();

  hipError_t ExtractFatBinaryUsingCOMGR(const std::vector<hip::Device*>& devices);
  hipError_t AddDevProgram(hip::Device* device, const void* binary_image, size_t binary_size,
                           size_t binary_offset);
  hipError_t BuildProgram(const int device_id);

  // Device Id bounds check
  inline void DeviceIdCheck(const int device_id) const {
    guarantee(device_id >= 0, "Invalid DeviceId less than 0");
    guarantee(static_cast<size_t>(device_id) < dev_programs_.size(),
              "Invalid DeviceId, greater than no of device programs!");
  }

  // Getter Methods
  amd::Program* GetProgram(int device_id) {
    DeviceIdCheck(device_id);
    return dev_programs_[device_id];
  }

  hipModule_t Module(int device_id) const {
    DeviceIdCheck(device_id);
    return reinterpret_cast<hipModule_t>(as_cl(dev_programs_[device_id]));
  }

  hipError_t GetModule(int device_id, hipModule_t* hmod) const {
    DeviceIdCheck(device_id);
    *hmod = reinterpret_cast<hipModule_t>(as_cl(dev_programs_[device_id]));
    return hipSuccess;
  }

  //! Returns the lock for this fatbinary access
  amd::Monitor& FatBinaryLock() { return fb_lock_; }

 private:
  void ReleaseImageAndFile();

  std::string fname_;  //!< File name
  size_t foffset_;     //!< File Offset where the fat binary is present.

  // Even when file is passed image will be mmapped till ~desctructor.
  const void* image_;  //!< Image
  bool image_mapped_;  //!< flag to detect if image is mapped

  // Only used for FBs where image is directly passed
  std::string uri_;  //!< Uniform resource indicator

  std::vector<amd::Program*> dev_programs_;  //!< Program info per Device

  std::shared_ptr<UniqueFD> ufd_;  //!< Unique file descriptor
  amd::Monitor fb_lock_{true};     //!< Lock for the fat binary access
};

};  // namespace hip

#endif  // HIP_FAT_BINARY_HPP
