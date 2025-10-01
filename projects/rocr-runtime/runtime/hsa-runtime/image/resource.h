/*

Copyright © 2025 Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

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
    uint32_t version;
    uint32_t vendorID;
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
  Image() {
    component.handle = 0;
    permission = HSA_ACCESS_PERMISSION_RO;
    data = NULL;
    std::memset(srd, 0, sizeof(srd));
    std::memset(&desc, 0, sizeof(desc));
    row_pitch = slice_pitch = 0;
    tile_mode = LINEAR;
  }

  ~Image() {}

public:
  typedef enum TileMode {
    LINEAR,
    TILED
  } TileMode;

  /// @brief Create an Image.
  static Image* Create(hsa_agent_t agent);

  /// @brief Destroy an Image.
  static void Destroy(const Image* image);

  uint64_t Convert() const { return reinterpret_cast<uint64_t>(srd); }

  static Image* Convert(uint64_t handle) {
    return reinterpret_cast<Image*>(handle - offsetof(Image, srd));
  }

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

//Represents an HSA sampler object.
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

  uint64_t Convert() { return reinterpret_cast<uint64_t>(srd); }

  static Sampler* Convert(uint64_t handle) {
    return reinterpret_cast<Sampler*>(handle - offsetof(Sampler, srd));
  }

  __ALIGNED__(HSA_SAMPLER_OBJECT_ALIGNMENT)
  uint32_t srd[HSA_SAMPLER_OBJECT_SIZE_DWORD];

  // HSA component of the sampler object.
  hsa_agent_t component;

  // HSA sampler descriptor of the image object.
  hsa_ext_sampler_descriptor_v2_t desc;
} Sampler;

typedef struct MipmappedArray : public Image {
private:
  MipmappedArray() {
    component.handle = 0;
    data = NULL;
    size = 0;
    std::memset(srd, 0, sizeof(srd));
    std::memset(&desc, 0, sizeof(desc));
    permission = HSA_ACCESS_PERMISSION_RO;
    num_levels = 0;
    levels_allocated = 0;
    flags = 0;
    addr_handle = nullptr;
    std::memset(&addr_output, 0, sizeof(addr_output));
    row_pitch = slice_pitch = 0;
    tile_mode = LINEAR;
  }

~MipmappedArray() {
  if (addr_output.pMipInfo)
  {
    delete[] addr_output.pMipInfo;
    addr_output.pMipInfo = nullptr;
  }
}

public:
  static MipmappedArray* Create(hsa_agent_t agent, size_t expected_surf_size_bytes);

  static void Destroy(const MipmappedArray* array);

  using Image::Convert;

  static MipmappedArray* Convert(uint64_t handle) {
    Image* base_image = Image::Convert(handle);
    return static_cast<MipmappedArray*>(base_image);
  }

  size_t size;
  uint32_t num_levels;
  uint32_t levels_allocated;  // Number of levels actually allocated

  uint32_t flags;

  // Address library handle.
  ADDR_HANDLE addr_handle;

  ADDR2_COMPUTE_SURFACE_INFO_OUTPUT addr_output;
} MipmappedArray;

}  // namespace image
}  // namespace rocr
#endif  // HSA_RUNTIME_EXT_IMAGE_RESOURCE_H
