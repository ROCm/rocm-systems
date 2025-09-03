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

 public:
  Sampler(Context& context,    //!< context for OCL
          bool normCoords,     //!< normalized coordinates
          uint addrMode,       //!< adressing mode
          uint filterMode,     //!< filter mode
          uint mipFilterMode,  //!< mip filter mode
          float minLod,        //!< min level of detail
          float maxLod         //!< max level of detail
          )
      : context_(context),
        mipFilter_(mipFilterMode),
        minLod_(minLod),
        maxLod_(maxLod) {  // Packs the sampler state into uint32_t for kernel execution
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
          float maxLod             //!< max level of detail
          )
      : Sampler(context, normCoords, addrMode[0], filterMode, mipFilterMode, minLod, maxLod) {
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
        DevLogPrintfError("Sampler creation failed for device: 0x%x \n", dev);
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
