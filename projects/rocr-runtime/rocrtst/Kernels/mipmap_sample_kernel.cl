/*
* Copyright © Advanced Micro Devices, Inc., or its affiliates.
*
* SPDX-License-Identifier: MIT
*/


/**
 * @brief OpenCL kernel to read from a 2D image with explicit LOD
 *
 * Reads from a mipmapped image at specified coordinates and LOD level,
 * storing the result for host-side verification. Equivalent to tex2DLod in HIP.
 *
 * @param img_input Read-only image sampler
 * @param sampler_obj Sampler configuration
 * @param output Buffer to store sampled values
 * @param width Width of output buffer
 * @param height Height of output buffer
 * @param lod Mipmap level to sample from
 * @param offsetX X coordinate offset for sampling
 * @param offsetY Y coordinate offset for sampling
 */

__kernel void
sample_mipmap_2d(__read_only image2d_t img_input,
                 sampler_t sampler_obj,
                 __global float4 *output,
                 uint width,
                 uint height,
                 float lod,
                 float offsetX,
                 float offsetY) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  // Bounds checking
  if (x >= width || y >= height) return;

  // Compute normalized coordinates
  float px = 1.0f / (float)width;
  float py = 1.0f / (float)height;
  float2 coord = (float2)((x + offsetX) * px, (y + offsetY) * py);

  // Read from image at specified LOD
  // Note: OpenCL uses read_imagef with sampler for LOD-based sampling
  float4 value = read_imagef(img_input, sampler_obj, coord, lod);

  // Store result
  output[y * width + x] = value;
}

/**
 * @brief OpenCL kernel to write pattern data to a 2D image surface
 *
 * Writes a unique pattern to an image level for validation testing.
 * Each level gets a distinct pattern based on level ID.
 *
 * @param img_output Write-only image surface
 * @param width Width of image
 * @param height Height of image
 * @param level Mipmap level ID (used for pattern generation)
 */
__kernel void
write_mipmap_pattern(__write_only image2d_t img_output,
                     uint width,
                     uint height,
                     uint level) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  // Bounds checking
  if (x >= width || y >= height) return;

  // Generate unique pattern for this level
  // Pattern: (level_id, x_position, y_position, checksum)
  float4 pattern;
  pattern.x = (float)level / 255.0f;
  pattern.y = (float)x / (float)width;
  pattern.z = (float)y / (float)height;
  pattern.w = ((float)level + (float)x + (float)y) / 765.0f;

  int2 coord = (int2)(x, y);
  write_imagef(img_output, coord, pattern);
}

/**
 * @brief OpenCL kernel to read and validate pattern from 2D image
 *
 * Reads data from an image level and compares against expected pattern.
 * Sets validation flag to 0 if mismatch detected.
 *
 * @param img_input Read-only image
 * @param validation_result Output flag (1=pass, 0=fail)
 * @param width Width of image
 * @param height Height of image
 * @param expected_level Expected mipmap level ID
 */
__kernel void
validate_mipmap_pattern(__read_only image2d_t img_input,
                        __global uint *validation_result,
                        uint width,
                        uint height,
                        uint expected_level) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  // Bounds checking
  if (x >= width || y >= height) return;

  int2 coord = (int2)(x, y);
  sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                      CLK_ADDRESS_NONE |
                      CLK_FILTER_NEAREST;

  float4 actual = read_imagef(img_input, sampler, coord);

  // Generate expected pattern
  float4 expected;
  expected.x = (float)expected_level / 255.0f;
  expected.y = (float)x / (float)width;
  expected.z = (float)y / (float)height;
  expected.w = ((float)expected_level + (float)x + (float)y) / 765.0f;

  // Check tolerance with improved precision for 8-bit data
  float tolerance = 1.0f / 255.0f;
  if (fabs(actual.x - expected.x) > tolerance ||
      fabs(actual.y - expected.y) > tolerance ||
      fabs(actual.z - expected.z) > tolerance ||
      fabs(actual.w - expected.w) > tolerance) {
    // Validation failed - set result to 0
    atomic_xchg(validation_result, 0);
  }
}

/**
 * @brief OpenCL kernel for 1D mipmap sampling
 */
__kernel void
sample_mipmap_1d(__read_only image1d_t img_input,
                 sampler_t sampler_obj,
                 __global float4 *output,
                 uint width,
                 float lod,
                 float offsetX) {
  int x = get_global_id(0);

  // Bounds checking
  if (x >= width) return;

  float px = 1.0f / (float)width;
  float coord = (x + offsetX) * px;

  float4 value = read_imagef(img_input, sampler_obj, coord, lod);
  output[x] = value;
}

/**
 * @brief OpenCL kernel to write pattern data to a 1D image surface
 */
__kernel void
write_mipmap_pattern_1d(__write_only image1d_t img_output,
                        uint width,
                        uint level) {
  int x = get_global_id(0);

  if (x >= width) return;

  // Generate unique pattern for this level
  float4 pattern;
  pattern.x = (float)level / 255.0f;
  pattern.y = (float)x / (float)width;
  pattern.z = 0.0f;  // No Y dimension for 1D
  pattern.w = ((float)level + (float)x) / 510.0f;

  write_imagef(img_output, x, pattern);
}

/**
 * @brief OpenCL kernel to read and validate pattern from 1D image
 */
__kernel void
validate_mipmap_pattern_1d(__read_only image1d_t img_input,
                           __global uint *validation_result,
                           uint width,
                           uint expected_level) {
  int x = get_global_id(0);

  if (x >= width) return;

  sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                      CLK_ADDRESS_NONE |
                      CLK_FILTER_NEAREST;

  float4 actual = read_imagef(img_input, sampler, x);

  // Generate expected pattern
  float4 expected;
  expected.x = (float)expected_level / 255.0f;
  expected.y = (float)x / (float)width;
  expected.z = 0.0f;
  expected.w = ((float)expected_level + (float)x) / 510.0f;

  // Check tolerance with improved precision
  float tolerance = 1.0f / 255.0f;  // More appropriate for 8-bit data
  if (fabs(actual.x - expected.x) > tolerance ||
      fabs(actual.y - expected.y) > tolerance ||
      fabs(actual.z - expected.z) > tolerance ||
      fabs(actual.w - expected.w) > tolerance) {
    // Validation failed - set result to 0
    atomic_xchg(validation_result, 0);
  }
}

/**
 * @brief OpenCL kernel for 3D mipmap sampling
 */
__kernel void
sample_mipmap_3d(__read_only image3d_t img_input,
                 sampler_t sampler_obj,
                 __global float4 *output,
                 uint width,
                 uint height,
                 uint depth,
                 float lod,
                 float offsetX,
                 float offsetY,
                 float offsetZ) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int z = get_global_id(2);

  // Bounds checking
  if (x >= width || y >= height || z >= depth) return;

  float px = 1.0f / (float)width;
  float py = 1.0f / (float)height;
  float pz = 1.0f / (float)depth;
  float4 coord = (float4)((x + offsetX) * px, (y + offsetY) * py,
                          (z + offsetZ) * pz, 0.0f);

  float4 value = read_imagef(img_input, sampler_obj, coord, lod);
  output[z * width * height + y * width + x] = value;
}

/**
 * @brief OpenCL kernel to write pattern data to a 3D image surface
 */
__kernel void
write_mipmap_pattern_3d(__write_only image3d_t img_output,
                        uint width,
                        uint height,
                        uint depth,
                        uint level) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int z = get_global_id(2);

  if (x >= width || y >= height || z >= depth) return;

  // Generate unique pattern for this level
  float4 pattern;
  pattern.x = (float)level / 255.0f;
  pattern.y = (float)x / (float)width;
  pattern.z = (float)y / (float)height;
  pattern.w = ((float)level + (float)x + (float)y + (float)z) / 1020.0f;

  int4 coord = (int4)(x, y, z, 0);
  write_imagef(img_output, coord, pattern);
}

/**
 * @brief OpenCL kernel to read and validate pattern from 3D image
 */
__kernel void
validate_mipmap_pattern_3d(__read_only image3d_t img_input,
                           __global uint *validation_result,
                           uint width,
                           uint height,
                           uint depth,
                           uint expected_level) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int z = get_global_id(2);

  if (x >= width || y >= height || z >= depth) return;

  int4 coord = (int4)(x, y, z, 0);
  sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                      CLK_ADDRESS_NONE |
                      CLK_FILTER_NEAREST;

  float4 actual = read_imagef(img_input, sampler, coord);

  // Generate expected pattern
  float4 expected;
  expected.x = (float)expected_level / 255.0f;
  expected.y = (float)x / (float)width;
  expected.z = (float)y / (float)height;
  expected.w = ((float)expected_level + (float)x + (float)y + (float)z) / 1020.0f;

  // Check tolerance with improved precision
  float tolerance = 1.0f / 255.0f;  // More appropriate for 8-bit data
  if (fabs(actual.x - expected.x) > tolerance ||
      fabs(actual.y - expected.y) > tolerance ||
      fabs(actual.z - expected.z) > tolerance ||
      fabs(actual.w - expected.w) > tolerance) {
    // Validation failed - set result to 0
    atomic_xchg(validation_result, 0);
  }
}
