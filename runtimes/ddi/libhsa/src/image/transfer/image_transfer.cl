// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Byte-preserving image transfers use unsigned resource views, matching ROCr's
// ConvertImage path. All five image arguments name the same resource descriptor;
// geometry selects the correctly typed instruction. Compression remains enabled
// in that descriptor so loads and stores go through the hardware image path.

#define IMAGE_ARGS(ACCESS) \
  ACCESS image1d_t image1d, ACCESS image2d_t image2d, \
  ACCESS image3d_t image3d, ACCESS image1d_array_t image1da, \
  ACCESS image2d_array_t image2da

#define TRANSFER_COORDINATES \
  uint id = __builtin_amdgcn_workgroup_id_x() * 64 + __builtin_amdgcn_workitem_id_x(); \
  if (id >= extent.x * extent.y * extent.z) return; \
  uint4 position = origin + (uint4)(id % extent.x, \
      (id / extent.x) % extent.y, id / (extent.x * extent.y), 0); \
  uint offset = id * channels * component_bytes

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void image_read_bytes(__global uchar* buffer, IMAGE_ARGS(read_only),
    uint4 origin, uint4 extent, uint geometry, uint channels, uint component_bytes) {
  TRANSFER_COORDINATES;
  uint4 pixel = 0;
  switch (geometry) {
    case 0: pixel = read_imageui(image1d, position.x); break;
    case 1: case 6: pixel = read_imageui(image2d, convert_int2(position.xy)); break;
    case 2: pixel = read_imageui(image3d, convert_int4(position)); break;
    case 3: pixel = read_imageui(image1da, convert_int2(position.xy)); break;
    case 4: case 7: pixel = read_imageui(image2da, convert_int4(position)); break;
  }
  for (uint channel = 0; channel < channels; ++channel)
    for (uint byte = 0; byte < component_bytes; ++byte)
      buffer[offset + channel * component_bytes + byte] = pixel[channel] >> (8 * byte);
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void image_write_bytes(__global uchar* buffer, IMAGE_ARGS(write_only),
    uint4 origin, uint4 extent, uint geometry, uint channels, uint component_bytes) {
  TRANSFER_COORDINATES;
  uint4 pixel = 0;
  for (uint channel = 0; channel < channels; ++channel)
    for (uint byte = 0; byte < component_bytes; ++byte)
      pixel[channel] |= (uint)buffer[offset + channel * component_bytes + byte] << (8 * byte);
  switch (geometry) {
    case 0: write_imageui(image1d, position.x, pixel); break;
    case 1: case 6: write_imageui(image2d, convert_int2(position.xy), pixel); break;
    case 2: write_imageui(image3d, convert_int4(position), pixel); break;
    case 3: write_imageui(image1da, convert_int2(position.xy), pixel); break;
    case 4: case 7: write_imageui(image2da, convert_int4(position), pixel); break;
  }
}
