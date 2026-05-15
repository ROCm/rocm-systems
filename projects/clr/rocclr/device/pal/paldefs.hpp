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

#include "top.hpp"
#include "pal.h"
#include "palGpuMemory.h"
#include "palImage.h"
#include "palFormatInfo.h"
#include "util/palSysMemory.h"

//
/// Memory Object Type
//
enum PalGpuMemoryType {
  PAL_DEPTH_BUFFER = 0,  ///< Depth Buffer
  PAL_BUFFER,            ///< Pure buffer
  PAL_TEXTURE_3D,        ///< 3D texture
  PAL_TEXTURE_2D,        ///< 2D texture
  PAL_TEXTURE_1D,        ///< 1D texture
  PAL_TEXTURE_1D_ARRAY,  ///< 1D Array texture
  PAL_TEXTURE_2D_ARRAY,  ///< 2D Array texture
  PAL_TEXTURE_BUFFER,    ///< "buffer" texture inside VBO
};

//! Engine types
enum EngineType { MainEngine = 0, SdmaEngine, AllEngines };

struct GpuEvent {
  static constexpr uint32_t InvalidID = ((1 << 30) - 1);

  struct {
    uint32_t id_ : 30;       ///< Actual event id
    uint32_t modified_ : 1;  ///< Resource associated with the event was modified
    uint32_t engineId_ : 1;  ///< Type of the id
  };
  //! GPU event default constructor
  GpuEvent() : id_(InvalidID), modified_(false), engineId_(MainEngine) {}
  //! GPU event constructor
  GpuEvent(uint evt) : id_(evt), modified_(false), engineId_(MainEngine) {}

  //! Returns true if the current event is valid
  bool isValid() const { return (id_ != InvalidID) ? true : false; }

  //! Set invalid event id
  void invalidate() { id_ = InvalidID; }

  // Overwrite default assign operator to preserve modified_ field
  GpuEvent& operator=(const GpuEvent& evt) {
    id_ = evt.id_;
    engineId_ = evt.engineId_;
    return *this;
  }
};

/*! \addtogroup PAL
 *  @{
 */

//! PAL Device Implementation

namespace amd::pal {

//! Maximum number of the supported global atomic counters
static constexpr uint MaxAtomicCounters = 8;
//! Maximum number of the supported samplers
static constexpr uint MaxSamplers = 16;
//! Maximum number of supported read images
static constexpr uint MaxReadImage = 128;
//! Maximum number of supported write images
static constexpr uint MaxWriteImage = 8;
//! Maximum number of supported read/write images for OCL20
static constexpr uint MaxReadWriteImage = 64;
//! Maximum number of supported constant arguments
static constexpr uint MaxConstArguments = 8;
//! Maximum number of supported kernel UAV arguments
static constexpr uint MaxUavArguments = 1024;
//! Maximum number of pixels for a 1D image created from a buffer
static constexpr size_t MaxImageBufferSize = (1ull << 32) - 1;
//! Maximum number of pixels for a 1D image created from a buffer
static constexpr size_t MaxImageArraySize = 2048;

//! Maximum number of supported constant buffers
static constexpr uint MaxConstBuffers = MaxConstArguments + 8;

//! Maximum number of constant buffers for arguments
static constexpr uint MaxConstBuffersArguments = 2;

//! Alignment restriction for the pinned memory
static constexpr size_t PinnedMemoryAlignment = 4 * Ki;

//! HSA path specific defines for images
static constexpr uint HsaImageObjectSize = 48;
static constexpr uint HsaImageObjectAlignment = 16;
static constexpr uint HsaSamplerObjectSize = 32;
static constexpr uint HsaSamplerObjectAlignment = 16;

//! HSA path specific defines for images
static constexpr uint DeviceQueueMaskSize = 32;

// Supported OpenCL versions
enum OclVersion {
  OpenCL10 = 0x10,
  OpenCL11 = 0x11,
  OpenCL12 = 0x12,
  OpenCL20 = 0x20,
  OpenCL21 = 0x21,
  OpenCL22 = 0x22,
};

struct MemoryFormat {
  cl_image_format clFormat_;        //!< CL image format
  Pal::ChNumFormat palFormat_;      //!< PAL image format
  Pal::ChannelMapping palChannel_;  //!< PAL channel mapping
};

static constexpr MemoryFormat MemoryFormatMap[] = {
    // R
    {{CL_R, CL_UNORM_INT8},
     Pal::ChNumFormat::X8_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_UNORM_INT16},
     Pal::ChNumFormat::X16_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},

    {{CL_R, CL_SNORM_INT8},
     Pal::ChNumFormat::X8_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_SNORM_INT16},
     Pal::ChNumFormat::X16_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},

    {{CL_R, CL_SIGNED_INT8},
     Pal::ChNumFormat::X8_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_SIGNED_INT16},
     Pal::ChNumFormat::X16_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_SIGNED_INT32},
     Pal::ChNumFormat::X32_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_UNSIGNED_INT8},
     Pal::ChNumFormat::X8_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_UNSIGNED_INT16},
     Pal::ChNumFormat::X16_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_UNSIGNED_INT32},
     Pal::ChNumFormat::X32_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},

    {{CL_R, CL_HALF_FLOAT},
     Pal::ChNumFormat::X16_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_FLOAT},
     Pal::ChNumFormat::X32_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},

    // A
    {{CL_A, CL_UNORM_INT8},
     Pal::ChNumFormat::X8_Unorm,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_UNORM_INT16},
     Pal::ChNumFormat::X16_Unorm,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},

    {{CL_A, CL_SNORM_INT8},
     Pal::ChNumFormat::X8_Snorm,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_SNORM_INT16},
     Pal::ChNumFormat::X16_Snorm,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},

    {{CL_A, CL_SIGNED_INT8},
     Pal::ChNumFormat::X8_Sint,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_SIGNED_INT16},
     Pal::ChNumFormat::X16_Sint,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_SIGNED_INT32},
     Pal::ChNumFormat::X32_Sint,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_UNSIGNED_INT8},
     Pal::ChNumFormat::X8_Uint,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_UNSIGNED_INT16},
     Pal::ChNumFormat::X16_Uint,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_UNSIGNED_INT32},
     Pal::ChNumFormat::X32_Uint,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},

    {{CL_A, CL_HALF_FLOAT},
     Pal::ChNumFormat::X16_Float,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_FLOAT},
     Pal::ChNumFormat::X32_Float,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},

    // RG
    {{CL_RG, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_UNORM_INT16},
     Pal::ChNumFormat::X16Y16_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},

    {{CL_RG, CL_SNORM_INT8},
     Pal::ChNumFormat::X8Y8_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_SNORM_INT16},
     Pal::ChNumFormat::X16Y16_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},

    {{CL_RG, CL_SIGNED_INT8},
     Pal::ChNumFormat::X8Y8_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_SIGNED_INT16},
     Pal::ChNumFormat::X16Y16_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_SIGNED_INT32},
     Pal::ChNumFormat::X32Y32_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_UNSIGNED_INT8},
     Pal::ChNumFormat::X8Y8_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_UNSIGNED_INT16},
     Pal::ChNumFormat::X16Y16_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_UNSIGNED_INT32},
     Pal::ChNumFormat::X32Y32_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},

    {{CL_RG, CL_HALF_FLOAT},
     Pal::ChNumFormat::X16Y16_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_FLOAT},
     Pal::ChNumFormat::X32Y32_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},

    // RGB
    {{CL_RGB, CL_UNORM_INT_101010},
     Pal::ChNumFormat::X10Y10Z10W2_Unorm,
     {Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::One}},
    // RGBA
    // Note: special case for GL interops, OCL spec doesn't support 101010 format
    // for GL interop and GL uses real RGB channel order
    {{CL_RGBA, CL_UNORM_INT_101010},
     Pal::ChNumFormat::X10Y10Z10W2_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::One}},

    // RGBA
    {{CL_RGBA, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_UNORM_INT16},
     Pal::ChNumFormat::X16Y16Z16W16_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},

    {{CL_RGBA, CL_SNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_SNORM_INT16},
     Pal::ChNumFormat::X16Y16Z16W16_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},

    {{CL_RGBA, CL_SIGNED_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_SIGNED_INT16},
     Pal::ChNumFormat::X16Y16Z16W16_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_SIGNED_INT32},
     Pal::ChNumFormat::X32Y32Z32W32_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_UNSIGNED_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_UNSIGNED_INT16},
     Pal::ChNumFormat::X16Y16Z16W16_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_UNSIGNED_INT32},
     Pal::ChNumFormat::X32Y32Z32W32_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},

    {{CL_RGBA, CL_HALF_FLOAT},
     Pal::ChNumFormat::X16Y16Z16W16_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_FLOAT},
     Pal::ChNumFormat::X32Y32Z32W32_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},

    // ARGB
    {{CL_ARGB, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Unorm,
     {Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W,
      Pal::ChannelSwizzle::X}},
    {{CL_ARGB, CL_SNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Snorm,
     {Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W,
      Pal::ChannelSwizzle::X}},
    {{CL_ARGB, CL_SIGNED_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Sint,
     {Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W,
      Pal::ChannelSwizzle::X}},
    {{CL_ARGB, CL_UNSIGNED_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Uint,
     {Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W,
      Pal::ChannelSwizzle::X}},

    // BGRA
    {{CL_BGRA, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Unorm,
     {Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::W}},
    {{CL_BGRA, CL_SNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Snorm,
     {Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::W}},
    {{CL_BGRA, CL_SIGNED_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Sint,
     {Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::W}},
    {{CL_BGRA, CL_UNSIGNED_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Uint,
     {Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::W}},

    // LUMINANCE
    {{CL_LUMINANCE, CL_SNORM_INT8},
     Pal::ChNumFormat::X8_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::One}},
    {{CL_LUMINANCE, CL_SNORM_INT16},
     Pal::ChNumFormat::X16_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::One}},
    {{CL_LUMINANCE, CL_UNORM_INT8},
     Pal::ChNumFormat::X8_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::One}},
    {{CL_LUMINANCE, CL_UNORM_INT16},
     Pal::ChNumFormat::X16_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::One}},
    {{CL_LUMINANCE, CL_HALF_FLOAT},
     Pal::ChNumFormat::X16_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::One}},
    {{CL_LUMINANCE, CL_FLOAT},
     Pal::ChNumFormat::X32_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::One}},

    // INTENSITY
    {{CL_INTENSITY, CL_SNORM_INT8},
     Pal::ChNumFormat::X8_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},
    {{CL_INTENSITY, CL_SNORM_INT16},
     Pal::ChNumFormat::X16_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},
    {{CL_INTENSITY, CL_UNORM_INT8},
     Pal::ChNumFormat::X8_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},
    {{CL_INTENSITY, CL_UNORM_INT16},
     Pal::ChNumFormat::X16_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},
    {{CL_INTENSITY, CL_HALF_FLOAT},
     Pal::ChNumFormat::X16_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},
    {{CL_INTENSITY, CL_FLOAT},
     Pal::ChNumFormat::X32_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},

    // sRBGA
    {{CL_sRGBA, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Srgb,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},

    // sRBG
    {{CL_sRGB, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Srgb,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::One}},

    // sRBGx
    {{CL_sRGBx, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Srgb,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::One}},

    // sBGRA
    {{CL_sBGRA, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Srgb,
     {Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::W}},

    // DEPTH
    {{CL_DEPTH, CL_FLOAT},
     Pal::ChNumFormat::X32_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},

    {{CL_DEPTH, CL_UNORM_INT16},
     Pal::ChNumFormat::X16_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},

    {{CL_DEPTH_STENCIL, CL_UNORM_INT24},
     Pal::ChNumFormat::X32_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},
    {{CL_DEPTH_STENCIL, CL_FLOAT},
     Pal::ChNumFormat::X32_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}}};

}  // namespace amd::pal
