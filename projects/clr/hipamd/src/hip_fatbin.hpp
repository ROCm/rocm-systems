/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_FAT_BINARY_HPP
#define HIP_FAT_BINARY_HPP

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip_internal.hpp"
#include "platform/program.hpp"

#include <optional>

namespace hip {

// Fat Binary Info
class FatBinaryInfo {
 public:
  // Parameters for kpack'd (split device code) binaries
  struct KpackParams {
    const void* metadata;      //!< Msgpack metadata from .rocm_kpack_ref section
    std::string binary_path;   //!< Path to the host binary
    uint64_t bundle_index;     //!< Bundle index for multi-TU binaries (0-based)
  };

  FatBinaryInfo(const char* fname, const void* image);
  // Constructor for kpack'd (split device code) binaries
  explicit FatBinaryInfo(KpackParams kpack_params);
  ~FatBinaryInfo();

  hipError_t ExtractFatBinaryUsingCOMGR(const std::vector<hip::Device*>& devices);
  hipError_t ExtractKpackBinary(const std::vector<hip::Device*>& devices);
  hipError_t AddDevProgram(hip::Device* device, const void* binary_image, size_t binary_size,
                           amd::Os::FileDesc fdesc);

  // Translates a SPIR-V code object blob to a device ELF executable via COMGR,
  // targeting the given device's ISA. On success returns hipSuccess and stores a
  // newly heap-allocated (new char[]) executable in *out_co / *out_size, tracked
  // in heap_code_objects_ and owned by this object. Used to lower portable
  // amdgcnspirv device code (from a fat binary or a kpack archive) to the
  // runner's real GPU ISA at load time.
  hipError_t TranslateSpirvToExecutable(hip::Device* device, const void* spirv_blob,
                                        size_t spirv_size, char** out_co, size_t* out_size);
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
  std::recursive_mutex& FatBinaryLock() { return fb_lock_; }

 private:
  void ReleaseImageAndFile();

  std::string fname_;  //!< File name
  size_t foffset_;     //!< File Offset where the fat binary is present.

  // When loaded from a file, image_ is the mmap address; fd is closed once
  // ExtractFatBinaryUsingCOMGR has dup'd it for every per-device handoff.
  const void* image_;  //!< Image
  size_t image_size_;  //!< Mapped image size (only valid when image_mapped_ is true)
  bool image_mapped_;  //!< flag to detect if image is mapped

  // Only used for FBs where image is directly passed
  std::string uri_;  //!< Uniform resource indicator

  // Kpack parameters for split device code binaries (nullopt for normal fat binaries)
  std::optional<KpackParams> kpack_params_;

  std::vector<amd::Program*> dev_programs_;  //!< Program info per Device

  std::recursive_mutex fb_lock_;  //!< Lock for the fat binary access

  // Tracked code-object allocations, split by how they must be freed. kpack
  // loader buffers are freed via kpack_free_code_object(); heap buffers created
  // with new char[] (COMGR-translated executables, and per-device COMGR fat-bin
  // outputs) are freed via delete[]. Keeping these separate avoids a
  // free/delete mismatch when a kpack binary yields a COMGR-translated object.
  std::unordered_set<const void*> kpack_code_objects_;  //!< kpack_free_code_object()
  std::unordered_set<const void*> heap_code_objects_;   //!< delete[]
};

};  // namespace hip

#endif  // HIP_FAT_BINARY_HPP
