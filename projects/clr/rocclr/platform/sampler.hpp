/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef SAMPLER_HPP_
#define SAMPLER_HPP_

#include "top.hpp"
#include "platform/object.hpp"
#include "device/device.hpp"

namespace amd {

//! Abstraction layer sampler class
class Sampler : public RuntimeObject {
 public:
  typedef std::unordered_map<Device const*, device::Sampler*> DeviceSamplers;

  //! \note the sampler states must match the compiler's defines.
  //! See amd_ocl_sys_predef.c
  enum State {
    StateNormalizedCoordsFalse = 0x00,
    StateNormalizedCoordsTrue = 0x01,
    StateNormalizedCoordsMask = (StateNormalizedCoordsFalse | StateNormalizedCoordsTrue),
    StateFilterNearest = 0x10,
    StateFilterLinear = 0x20,
    StateFilterMask = (StateFilterNearest | StateFilterLinear)
  };

 private:
  Context& context_;               //!< OpenCL context associated with this sampler
  uint32_t state_;                 //!< Sampler state
  uint mipFilter_;                 //!< mip filter
  float minLod_;                   //!< min level of detail
  float maxLod_;                   //!< max level of detail
  DeviceSamplers deviceSamplers_;  //!< Container for the device samplers
  uint addressMode_[3];            //!< address modes in X, Y and Z
  //! Maximum anisotropy ratio. 0 and 1 both mean isotropic filtering. Kept as a
  //! separate member rather than packed into state_, because state_'s bit layout
  //! is contractual with the compiler (see the note above the State enum).
  uint32_t maxAnisotropy_;

 public:
  Sampler(Context& context,    //!< context for OCL
          bool normCoords,     //!< normalized coordinates
          uint addrMode,       //!< adressing mode
          uint filterMode,     //!< filter mode
          uint mipFilterMode,  //!< mip filter mode
          float minLod,        //!< min level of detail
          float maxLod,        //!< max level of detail
          uint32_t maxAnisotropy = 0  //!< max anisotropy ratio, 0/1 = isotropic
          )
      : context_(context),
        mipFilter_(mipFilterMode),
        minLod_(minLod),
        maxLod_(maxLod),
        maxAnisotropy_(maxAnisotropy) {  // Packs the sampler state into uint32_t for kernel execution
    state_ = 0;
    for (int i = 0; i < 3; i++) addressMode_[i] = addrMode;

    // Set normalized state
    if (normCoords) {
      state_ |= StateNormalizedCoordsTrue;
    } else {
      state_ |= StateNormalizedCoordsFalse;
    }

    // Program the sampler filter mode
    if (filterMode == CL_FILTER_LINEAR) {
      state_ |= StateFilterLinear;
    } else {
      state_ |= StateFilterNearest;
    }
  }

  Sampler(Context& context,        //!< context for Hip
          bool normCoords,         //!< normalized coordinates
          const uint addrMode[3],  //!< adressing modes in X, Y and Z directions
          uint filterMode,         //!< filter mode
          uint mipFilterMode,      //!< mip filter mode
          float minLod,            //!< min level of detail
          float maxLod,            //!< max level of detail
          uint32_t maxAnisotropy = 0  //!< max anisotropy ratio, 0/1 = isotropic
          )
      : Sampler(context, normCoords, addrMode[0], filterMode, mipFilterMode, minLod, maxLod,
                maxAnisotropy) {
    addressMode_[1] = addrMode[1];
    addressMode_[2] = addrMode[2];
  }

  virtual ~Sampler() {
    for (const auto& it : deviceSamplers_) {
      delete it.second;
    }
  }

  bool create() {
    for (uint i = 0; i < context_.devices().size(); ++i) {
      device::Sampler* sampler = NULL;
      Device* dev = context_.devices()[i];
      if (!dev->createSampler(*this, &sampler)) {
        ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_RESOURCE,
                 "Sampler creation failed for device: 0x%x \n", dev);
        return false;
      }
      deviceSamplers_[dev] = sampler;
    }
    return true;
  }

  device::Sampler* getDeviceSampler(const Device& dev) const {
    auto it = deviceSamplers_.find(&dev);
    if (it != deviceSamplers_.end()) {
      return it->second;
    }
    return NULL;
  }

  //! Accessor functions
  Context& context() const { return context_; }
  uint32_t state() const { return state_; }
  uint mipFilter() const { return mipFilter_; }
  float minLod() const { return minLod_; }
  float maxLod() const { return maxLod_; }
  uint32_t maxAnisotropy() const { return maxAnisotropy_; }
  const uint* addessMode() const { return addressMode_; }
  bool normalizedCoords() const { return (state_ & StateNormalizedCoordsTrue) ? true : false; }

  uint inline addressingMode(const int index = 0) const { return addressMode_[index]; }

  uint filterMode() const {
    return ((state_ & StateFilterMask) == StateFilterNearest) ? CL_FILTER_NEAREST
                                                              : CL_FILTER_LINEAR;
  }

  //! RTTI internal implementation
  virtual ObjectType objectType() const { return ObjectTypeSampler; }
};

}  // namespace amd

#endif /*SAMPLER_HPP_*/
