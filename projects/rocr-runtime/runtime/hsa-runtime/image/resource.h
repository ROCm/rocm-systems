////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_EXT_IMAGE_RESOURCE_H
#define HSA_RUNTIME_EXT_IMAGE_RESOURCE_H

#include <stdint.h>

#include <cstring>

#include "inc/hsa.h"
#include "inc/hsa_ext_image.h"
#include "addrlib/inc/addrinterface.h"

#include "util.h"

#define HSA_IMAGE_OBJECT_SIZE_DWORD 12
#define HSA_IMAGE_OBJECT_ALIGNMENT 16

#define HSA_SAMPLER_OBJECT_SIZE_DWORD 8
#define HSA_SAMPLER_OBJECT_ALIGNMENT 16

#define GEOMETRY_COUNT 8
#define ORDER_COUNT 20
#define TYPE_COUNT 16
#define RO HSA_EXT_IMAGE_CAPABILITY_READ_ONLY
#define ROWO \
  (HSA_EXT_IMAGE_CAPABILITY_READ_ONLY | HSA_EXT_IMAGE_CAPABILITY_WRITE_ONLY)
#define RW                                                                    \
  (HSA_EXT_IMAGE_CAPABILITY_READ_ONLY | HSA_EXT_IMAGE_CAPABILITY_WRITE_ONLY | \
  HSA_EXT_IMAGE_CAPABILITY_READ_WRITE)

namespace rocr {
namespace image {

typedef struct metadata_amd_s {
    uint32_t version; // Must be 1
    uint32_t vendorID; // AMD | CZ
    uint32_t words[8];
    uint32_t mip_offsets[0]; //Mip level offset bits [39:8] for each level (if any)
} metadata_amd_t;

/// @brief Structure to represent image access component.
typedef struct Swizzle {
  uint8_t x;
  uint8_t y;
  uint8_t z;
  uint8_t w;
} Swizzle;

/// @brief Structure to contain the property of an image with a particular
/// format and geometry.
typedef struct ImageProperty {
  uint8_t cap;           // hsa_ext_image_format_capability_t mask.
  uint8_t element_size;  // size per pixel in bytes.
  uint8_t data_format;   // device specific channel ordering.
  uint8_t data_type;     // device specific channel type.
} ImageProperty;

/// @brief Structure to represent an HSA image object.
typedef struct Image {
protected:
  Image()
      : data(nullptr),
        row_pitch(0),
        slice_pitch(0) {
    component.handle = 0;
    permission = HSA_ACCESS_PERMISSION_RO;
    std::memset(srd, 0, sizeof(srd));
    std::memset(&desc, 0, sizeof(desc));
    tile_mode = LINEAR;
  }

  virtual ~Image() {}

 public:
  typedef enum TileMode {
    LINEAR,
    TILED
  } TileMode;

  /// @brief Create an Image.
  static Image* Create(hsa_agent_t agent);

  /// @brief Destroy an Image.
  static void Destroy(const Image* image);

  /// @brief Convert from vendor representation to HSA handle.
  uint64_t Convert() const { return reinterpret_cast<uint64_t>(srd); }

  /// @brief Convert from HSA handle to vendor representation.
  static Image* Convert(uint64_t handle) {
    // Compute offset manually to avoid offsetof warning with virtual destructor
    Image* dummy = nullptr;
    const ptrdiff_t srd_offset =
        reinterpret_cast<const char*>(&dummy->srd) - reinterpret_cast<const char*>(dummy);
    return reinterpret_cast<Image*>(handle - srd_offset);
  }

  // Vendor specific image object.
  __ALIGNED__(
      HSA_IMAGE_OBJECT_ALIGNMENT) uint32_t srd[HSA_IMAGE_OBJECT_SIZE_DWORD];

  void const printSRD() const {
    char hexStr[200];
    size_t hexStrLen = 0;
    for (int i = 0; i < sizeof(srd) / sizeof(srd[0]); i++)
      hexStrLen += sprintf(&hexStr[hexStrLen], "0x%08x ", srd[i]);

    printf("\nSRD:%s\n\n", hexStr);
  }

  // HSA component of the image object.
  hsa_agent_t component;

  // HSA image descriptor of the image object.
  hsa_ext_image_descriptor_t desc;

  // HSA image access permission of the image object.
  hsa_access_permission_t permission;

  // Backing storage of the image object.
  void* data;

  // Device specific row pitch of the image object in size.
  size_t row_pitch;

  // Device specific slice pitch of the image object in size.
  size_t slice_pitch;

  // Device specific tile mode
  TileMode tile_mode;
} Image;

/// @brief Structure to represent an HSA sampler object.
typedef struct Sampler {
private:
  Sampler() {
    component.handle = 0;
    std::memset(srd, 0, sizeof(srd));
    std::memset(&desc, 0, sizeof(desc));
  }

  ~Sampler() {}

public:
  /// @brief Create a Sampler.
  static Sampler* Create(hsa_agent_t agent);

  /// @brief Destroy a Sampler.
  static void Destroy(const Sampler* sampler);

  /// @brief Convert from vendor representation to HSA handle.
  uint64_t Convert() { return reinterpret_cast<uint64_t>(srd); }

  /// @brief Convert from HSA handle to vendor representation.
  static Sampler* Convert(uint64_t handle) {
    return reinterpret_cast<Sampler*>(handle - offsetof(Sampler, srd));
  }

  // Vendor specific sampler object.
  __ALIGNED__(HSA_SAMPLER_OBJECT_ALIGNMENT)
  uint32_t srd[HSA_SAMPLER_OBJECT_SIZE_DWORD];

  // HSA component of the sampler object.
  hsa_agent_t component;

  // HSA sampler descriptor of the image object.
  hsa_ext_sampler_descriptor_v2_t desc;
} Sampler;

/// @brief Structure representing a mipmapped image array.
typedef struct MipmappedArray : public Image {
private:
  MipmappedArray()
      : size(0),
        num_levels(0),
        flags(0) {
    component.handle = 0;
    std::memset(srd, 0, sizeof(srd));
    std::memset(&desc, 0, sizeof(desc));
    permission = HSA_ACCESS_PERMISSION_RO;
    std::memset(&addr_output, 0, sizeof(addr_output));
    tile_mode = LINEAR;
  }

public:
 ~MipmappedArray() {}

 /// @brief Create a MipmappedArray.
 /// Only internal metadata is allocated; image data must be provided by the user.
 static MipmappedArray* Create(hsa_agent_t agent);

 /// @brief Destroy a MipmappedArray.
 static void Destroy(const MipmappedArray* array);

 /// @brief Convert from vendor representation to HSA handle.
 uint64_t Convert() const { return reinterpret_cast<uint64_t>(srd); }

 /// @brief Convert from HSA handle to vendor representation.
 static MipmappedArray* Convert(uint64_t handle) {
   // Compute offset manually to avoid offsetof warning with virtual destructor
   MipmappedArray* dummy = nullptr;
   const ptrdiff_t srd_offset =
       reinterpret_cast<const char*>(&dummy->srd) - reinterpret_cast<const char*>(dummy);
   return reinterpret_cast<MipmappedArray*>(handle - srd_offset);
 }

  // Total size of the allocated memory.
  size_t size;

  // Number of mipmap levels.
  uint32_t num_levels;

  // Reserved
  uint32_t flags;

  // Cached surface info.
  union {
    ADDR_COMPUTE_SURFACE_INFO_OUTPUT addr1;   // Pre-GFX9 versions
    ADDR2_COMPUTE_SURFACE_INFO_OUTPUT addr2;  // GFX9 and later
    ADDR3_COMPUTE_SURFACE_INFO_OUTPUT addr3;  // GFX10 and later
  } addr_output;

} MipmappedArray;

}  // namespace image
}  // namespace rocr
#endif  // HSA_RUNTIME_EXT_IMAGE_RESOURCE_H
