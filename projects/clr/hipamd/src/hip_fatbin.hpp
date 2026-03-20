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

// Forward declaration for Unique FD
struct ManagedUniqueFD;

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

  // Due to self-managed ownership, this class cannot be copy constructed or assigned.
  FatBinaryInfo(const FatBinaryInfo&) = delete;
  FatBinaryInfo& operator=(const FatBinaryInfo&) = delete;

  hipError_t ExtractFatBinaryUsingCOMGR(const std::vector<hip::Device*>& devices);
  hipError_t ExtractKpackBinary(const std::vector<hip::Device*>& devices);
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
  std::recursive_mutex& FatBinaryLock() { return fb_lock_; }

 private:
  std::string fname_;  //!< File name
  size_t foffset_;     //!< File Offset where the fat binary is present.

  // Helper class that ensures that mapped images get unmapped.
  struct ManagedImage {
    ManagedImage() {}
    ManagedImage(const void* image) : image_(image) {}
    ~ManagedImage();

    ManagedImage(const ManagedImage&) = delete;
    ManagedImage& operator=(const ManagedImage&) = delete;

    amd::Os::FileDesc getFileDescOrDefault() const;

    const void* getImage() const { return image_; }

    hipError_t map(const std::string& fname, size_t foffset);

   private:
    std::shared_ptr<ManagedUniqueFD> ufd_ = nullptr;  //!< Unique file descriptor
    const void* image_ = nullptr;                     //!< image
  } managed_image_;  //!< Managed file descriptor and mapping for fat binary file

  // Only used for FBs where image is directly passed
  std::string uri_;  //!< Uniform resource indicator

  // Kpack parameters for split device code binaries (nullopt for normal fat binaries)
  std::optional<KpackParams> kpack_params_;

  std::vector<amd::Program*> dev_programs_;  //!< Program info per Device

  std::recursive_mutex fb_lock_;                          //!< Lock for the fat binary access
  std::unordered_set<const void*> code_obj_allocations_;  //!< Track allocations for code objects
};

};  // namespace hip

#endif  // HIP_FAT_BINARY_HPP
