// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Keep the large AddrLib3 ABI behind one fixed-size surface-layout operation.

#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "addrinterface.h"
#include "addrlib.h"

struct RocAddr3Input {
  uint32_t family;
  uint32_t chip_revision;
  uint32_t gb_addr_config;
  uint32_t bits_per_pixel;
  uint32_t width;
  uint32_t height;
  uint32_t slices;
  uint32_t mip_levels;
  uint32_t resource_type;
  uint32_t linear;
  uint32_t pitch_in_elements;
};

struct RocAddr3Mip {
  uint64_t offset;
  uint32_t pitch;
  uint32_t height;
  uint32_t depth;
};

struct RocAddr3Output {
  uint64_t size;
  uint64_t slice_size;
  uint32_t alignment;
  uint32_t pitch;
  uint32_t height;
  uint32_t swizzle_mode;
  uint32_t first_mip_in_tail;
  RocAddr3Mip mips[32];
};

namespace {

void* Allocate(const rocr::ADDR_ALLOCSYSMEM_INPUT* input) {
  return std::malloc(input->sizeInBytes);
}

ADDR_E_RETURNCODE Free(const rocr::ADDR_FREESYSMEM_INPUT* input) {
  std::free(input->pVirtAddr);
  return ADDR_OK;
}

struct AddrOwner {
  rocr::ADDR_HANDLE handle = nullptr;
  ~AddrOwner() {
    if (handle != nullptr) rocr::AddrDestroy(handle);
  }
};

bool Create(const RocAddr3Input& request, AddrOwner& owner) {
  // AddrLib asserts on unknown families and some invalid revisions. Reject
  // those at this checked boundary before entering the vendor implementation.
  switch (request.family) {
    case 0x90: case 0x92: case 0x95: case 0x97:
      if (request.chip_revision == 0 || request.chip_revision >= 255) return false;
      break;
    case 0x8d: case 0x8e: case 0x8f: case 0x91:
    case 0x94: case 0x96: case 0x98: case 0x9a:
      break;
    default:
      return false;
  }
  rocr::ADDR_CREATE_INPUT create = {};
  create.size = sizeof(create);
  create.chipEngine = CIASICIDGFXENGINE_ARCTICISLAND;
  create.chipFamily = request.family;
  create.chipRevision = request.chip_revision;
  create.callbacks.allocSysMem = Allocate;
  create.callbacks.freeSysMem = Free;
  create.createFlags.useTileIndex = 1;
  create.regValue.gbAddrConfig = request.gb_addr_config;
  rocr::ADDR_CREATE_OUTPUT created = {};
  created.size = sizeof(created);
  const auto status = rocr::AddrCreate(&create, &created);
  owner.handle = created.hLib;
  return status == ADDR_OK && owner.handle != nullptr;
}

bool ValidInput(const RocAddr3Input* request) {
  return request != nullptr && request->mip_levels != 0 &&
         request->mip_levels <= (request->family == 0x98 ? 32u : 16u) && request->width != 0 &&
         request->height != 0 && request->slices != 0 &&
         request->bits_per_pixel >= 8 && request->bits_per_pixel <= 128 &&
         (request->bits_per_pixel & (request->bits_per_pixel - 1)) == 0 &&
         request->resource_type <= ADDR_RSRC_TEX_3D;
}

bool ValidLinearPitch(const RocAddr3Input& request) {
  // AddrLib ignores custom pitches for tiled or mipmapped surfaces.
  if (request.linear == 0 || request.pitch_in_elements == 0 ||
      request.mip_levels != 1) return true;

  // Invalid custom pitches can reach assertions in AddrLib's final layout
  // checks. Validate its row minimum and alignment before entering native code.
  const uint64_t element_bytes = request.bits_per_pixel / 8;
  const uint64_t row_bytes = request.pitch_in_elements * element_bytes;
  const uint64_t minimum_row_bytes =
      ((request.width * element_bytes + 255) / 256) * 256;
  const uint64_t row_alignment = request.family == 0x98 ? 128 : 256;
  if (row_bytes < minimum_row_bytes || row_bytes % row_alignment != 0) {
    return false;
  }
  // GFX12 permits 128-byte rows, but array and 3D slices still require full
  // 256-byte blocks. Reduce before multiplying to avoid a slice-size overflow.
  return request.family != 0x98 ||
         (request.resource_type != ADDR_RSRC_TEX_3D && request.slices == 1) ||
         ((row_bytes % 256) * request.height) % 256 == 0;
}

// GFX9/10/11 share AddrLib2's public surface ABI. The selected family still
// creates the exact hardware implementation in AddrCreate.
bool ComputeLegacy(const RocAddr3Input& request, rocr::ADDR_HANDLE handle,
                   RocAddr3Output& result) {
  rocr::ADDR2_COMPUTE_SURFACE_INFO_INPUT surface = {};
  surface.size = sizeof(surface);
  surface.flags.texture = 1;
  surface.flags.view3dAs2dArray = request.resource_type == ADDR_RSRC_TEX_3D;
  surface.resourceType = static_cast<AddrResourceType>(request.resource_type);
  surface.format = ADDR_FMT_INVALID;
  surface.bpp = request.bits_per_pixel;
  surface.width = request.width;
  surface.height = request.height;
  surface.numSlices = request.slices;
  surface.numMipLevels = request.mip_levels;
  surface.numSamples = 1;
  surface.numFrags = 1;
  surface.pitchInElement = request.pitch_in_elements;
  if (request.linear != 0) {
    surface.swizzleMode = ADDR_SW_LINEAR;
  } else {
    rocr::ADDR2_GET_PREFERRED_SURF_SETTING_INPUT input = {};
    input.size = sizeof(input);
    input.flags = surface.flags;
    input.resourceType = surface.resourceType;
    input.format = surface.format;
    input.bpp = surface.bpp;
    input.width = surface.width;
    input.height = surface.height;
    input.numSlices = surface.numSlices;
    input.numMipLevels = surface.numMipLevels;
    input.numSamples = surface.numSamples;
    input.numFrags = surface.numFrags;
    input.resourceLoction = ADDR_RSRC_LOC_UNDEF;
    rocr::ADDR2_GET_PREFERRED_SURF_SETTING_OUTPUT preferred = {};
    preferred.size = sizeof(preferred);
    if (rocr::Addr2GetPreferredSurfaceSetting(handle, &input, &preferred) != ADDR_OK) return false;
    surface.swizzleMode = preferred.swizzleMode;
  }
  std::array<rocr::ADDR2_MIP_INFO, 32> mips = {};
  rocr::ADDR2_COMPUTE_SURFACE_INFO_OUTPUT output = {};
  output.size = sizeof(output);
  output.pMipInfo = mips.data();
  if (rocr::Addr2ComputeSurfaceInfo(handle, &surface, &output) != ADDR_OK ||
      output.surfSize == 0 || output.baseAlign == 0) return false;
  RocAddr3Output value = {};
  value.size = output.surfSize;
  value.slice_size = output.sliceSize;
  value.alignment = output.baseAlign;
  value.pitch = output.pitch;
  value.height = output.height;
  value.swizzle_mode = surface.swizzleMode;
  value.first_mip_in_tail = output.firstMipIdInTail;
  for (uint32_t mip = 0; mip < request.mip_levels; ++mip) {
    value.mips[mip] = {mips[mip].offset, mips[mip].pitch,
                      mips[mip].height, mips[mip].depth};
  }
  result = value;
  return true;
}

}  // namespace

extern "C" int rocaddr3_compute(const RocAddr3Input* request,
                                  RocAddr3Output* result) noexcept {
  if (!ValidInput(request) || result == nullptr) {
    return 0;
  }
  if (!ValidLinearPitch(*request)) return 0;
  try {
    AddrOwner owner;
    if (!Create(*request, owner)) return 0;
    if (request->family != 0x98) {
      return ComputeLegacy(*request, owner.handle, *result) ? 1 : 0;
    }

    rocr::ADDR3_COMPUTE_SURFACE_INFO_INPUT surface = {};
    surface.size = sizeof(surface);
    surface.flags.texture = 1;
    surface.flags.view3dAs2dArray =
        request->resource_type == ADDR_RSRC_TEX_3D ? 1 : 0;
    surface.resourceType =
        static_cast<AddrResourceType>(request->resource_type);
    surface.format = ADDR_FMT_INVALID;
    surface.bpp = request->bits_per_pixel;
    surface.width = request->width;
    surface.height = request->height;
    surface.numSlices = request->slices;
    surface.numMipLevels = request->mip_levels;
    surface.numSamples = 1;
    surface.pitchInElement = request->pitch_in_elements;

    rocr::ADDR3_GET_POSSIBLE_SWIZZLE_MODE_INPUT possible = {};
    possible.size = sizeof(possible);
    possible.flags = surface.flags;
    possible.resourceType = surface.resourceType;
    possible.bpp = surface.bpp;
    possible.width = surface.width;
    possible.height = surface.height;
    possible.numSlices = surface.numSlices;
    possible.numMipLevels = surface.numMipLevels;
    possible.numSamples = 1;
    possible.maxAlign = 256 * 1024;
    rocr::ADDR3_GET_POSSIBLE_SWIZZLE_MODE_OUTPUT modes = {};
    modes.size = sizeof(modes);
    if (rocr::Addr3GetPossibleSwizzleModes(owner.handle, &possible, &modes) !=
        ADDR_OK) {
      return 0;
    }
    // ROCr excludes linear storage for native multidimensional and mipmapped
    // textures. Choose its last mode within a 2:1 size bound.
    if (request->linear == 0 && (request->height > 1 || request->mip_levels > 1)) {
      modes.validModes.swLinear = 0;
    }
    uint64_t smallest = std::numeric_limits<uint64_t>::max();
    int chosen = -1;
    for (int mode = ADDR3_LINEAR; mode < ADDR3_MAX_TYPE; ++mode) {
      if (request->linear != 0 && mode != ADDR3_LINEAR) continue;
      if ((modes.validModes.value & (1u << mode)) == 0) continue;
      surface.swizzleMode = static_cast<Addr3SwizzleMode>(mode);
      rocr::ADDR3_COMPUTE_SURFACE_INFO_OUTPUT candidate = {};
      candidate.size = sizeof(candidate);
      if (rocr::Addr3ComputeSurfaceInfo(owner.handle, &surface, &candidate) !=
          ADDR_OK || candidate.surfSize == 0) {
        continue;
      }
      if (chosen < 0 ||
          candidate.surfSize <= smallest ||
          candidate.surfSize - smallest <= smallest) {
        smallest = candidate.surfSize;
        chosen = mode;
      }
    }
    if (chosen < 0) return 0;

    surface.swizzleMode = static_cast<Addr3SwizzleMode>(chosen);
    std::array<rocr::ADDR3_MIP_INFO, 32> mip_info = {};
    rocr::ADDR3_COMPUTE_SURFACE_INFO_OUTPUT computed = {};
    computed.size = sizeof(computed);
    computed.pMipInfo = mip_info.data();
    if (rocr::Addr3ComputeSurfaceInfo(owner.handle, &surface, &computed) !=
        ADDR_OK || computed.surfSize == 0 || computed.baseAlign == 0) {
      return 0;
    }
    RocAddr3Output value = {};
    value.size = computed.surfSize;
    value.slice_size = computed.sliceSize;
    value.alignment = computed.baseAlign;
    value.pitch = computed.pitch;
    value.height = computed.height;
    value.swizzle_mode = chosen;
    value.first_mip_in_tail = computed.firstMipIdInTail;
    for (uint32_t index = 0; index < request->mip_levels; ++index) {
      value.mips[index] = {mip_info[index].offset, mip_info[index].pitch,
                           mip_info[index].height, mip_info[index].depth};
    }
    *result = value;
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" void* rocaddr3_mapper_create(const RocAddr3Input* request) noexcept {
  if (!ValidInput(request)) {
    return nullptr;
  }
  try {
    AddrOwner owner;
    if (!Create(*request, owner)) return nullptr;
    void* handle = owner.handle;
    owner.handle = nullptr;
    return handle;
  } catch (...) {
    return nullptr;
  }
}

extern "C" void rocaddr3_mapper_destroy(void* mapper) noexcept {
  try {
    if (mapper != nullptr) rocr::AddrDestroy(mapper);
  } catch (...) {
    // Destruction cannot unwind through the Rust ABI.
  }
}

extern "C" int rocaddr3_mapper_map_row(void* mapper,
                                        const RocAddr3Input* request,
                                        uint32_t swizzle_mode, uint32_t mip,
                                        uint32_t x, uint32_t y, uint32_t slice,
                                        uint32_t count, uint64_t* offsets) noexcept {
  if (mapper == nullptr || !ValidInput(request) || offsets == nullptr || count == 0 ||
      swizzle_mode >= ADDR3_MAX_TYPE || mip >= request->mip_levels) {
    return 0;
  }
  const uint32_t width = (request->width >> mip) == 0 ? 1 : request->width >> mip;
  const uint32_t height = (request->height >> mip) == 0 ? 1 : request->height >> mip;
  const uint32_t depth =
      request->resource_type == ADDR_RSRC_TEX_3D
          ? ((request->slices >> mip) == 0 ? 1 : request->slices >> mip)
          : request->slices;
  if (x >= width || count > width - x || y >= height || slice >= depth) {
    return 0;
  }
  try {
    rocr::ADDR3_COMPUTE_SURFACE_ADDRFROMCOORD_INPUT input = {};
    input.size = sizeof(input);
    input.y = y;
    input.slice = slice;
    input.mipId = mip;
    input.swizzleMode = static_cast<Addr3SwizzleMode>(swizzle_mode);
    input.flags.texture = 1;
    input.flags.view3dAs2dArray =
        request->resource_type == ADDR_RSRC_TEX_3D ? 1 : 0;
    input.resourceType = static_cast<AddrResourceType>(request->resource_type);
    input.bpp = request->bits_per_pixel;
    input.unAlignedDims.width = request->width;
    input.unAlignedDims.height = request->height;
    input.unAlignedDims.depth = request->slices;
    input.numMipLevels = request->mip_levels;
    input.numSamples = 1;
    for (uint32_t index = 0; index < count; ++index) {
      input.x = x + index;
      rocr::ADDR3_COMPUTE_SURFACE_ADDRFROMCOORD_OUTPUT output = {};
      output.size = sizeof(output);
      if (rocr::Addr3ComputeSurfaceAddrFromCoord(mapper, &input, &output) !=
              ADDR_OK || output.bitPosition != 0) {
        return 0;
      }
      offsets[index] = output.addr;
    }
    return 1;
  } catch (...) {
    return 0;
  }
}
