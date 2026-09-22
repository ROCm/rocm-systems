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

/// Kernel code for HSA image import/export/copy/clear in OpenCL C form.

uint4 read_image(__read_only image1d_t src1d,
                 __read_only image2d_t src2d,
                 __read_only image3d_t src3d,
                 __read_only image1d_array_t src1da,
                 __read_only image2d_array_t src2da,
                 uint format,
                 int4 coords) {
  switch (format) {
    case 0:  // 1D
      return read_imageui(src1d, coords.x);
      break;
    case 1:  // 2D
      return read_imageui(src2d, coords.xy);
      break;
    case 2:  // 3D
      return read_imageui(src3d, coords);
      break;
    case 3:  // 1DA
      return read_imageui(src1da, coords.xy);
      break;
    case 4:  // 2DA
      return read_imageui(src2da, coords);
      break;
    // case 5: //1DB
    //  return read_imageui(src1db, coords.x);
    //  break;
    default:  // Critical failure.
      return 0;
  }
}

void write_image(__write_only image1d_t src1d,
                 __write_only image2d_t src2d,
                 __write_only image3d_t src3d,
                 __write_only image1d_array_t src1da,
                 __write_only image2d_array_t src2da,
                 uint format,
                 int4 coords,
                 uint4 texel) {
  switch (format) {
    case 0:  // 1D
      write_imageui(src1d, coords.x, texel);
      break;
    case 1:  // 2D
      write_imageui(src2d, coords.xy, texel);
      break;
    case 2:  // 3D
      write_imageui(src3d, coords, texel);
      break;
    case 3:  // 1DA
      write_imageui(src1da, coords.xy, texel);
      break;
    case 4:  // 2DA
      write_imageui(src2da, coords, texel);
      break;
    // case 5: //1DB
    //  write_imageui(src1db, coords.x, texel);
    //  break;
    default:  // Critical failure.
      return;
  }
}

float4 read_image_float(__read_only image1d_t src1d,
                        __read_only image2d_t src2d,
                        __read_only image3d_t src3d,
                        __read_only image1d_array_t src1da,
                        __read_only image2d_array_t src2da,
                        uint format,
                        int4 coords) {
  switch (format) {
    case 0:  // 1D
      return read_imagef(src1d, coords.x);
      break;
    case 1:  // 2D
      return read_imagef(src2d, coords.xy);
      break;
    case 2:  // 3D
      return read_imagef(src3d, coords);
      break;
    case 3:  // 1DA
      return read_imagef(src1da, coords.xy);
      break;
    case 4:  // 2DA
      return read_imagef(src2da, coords);
      break;
    default:  // Critical failure.
      return 0;
  }
}

void write_image_float(__write_only image1d_t src1d,
                       __write_only image2d_t src2d,
                       __write_only image3d_t src3d,
                       __write_only image1d_array_t src1da,
                       __write_only image2d_array_t src2da,
                       uint format,
                       int4 coords,
                       float4 texel) {
  switch (format) {
    case 0:  // 1D
      write_imagef(src1d, coords.x, texel);
      break;
    case 1:  // 2D
      write_imagef(src2d, coords.xy, texel);
      break;
    case 2:  // 3D
      write_imagef(src3d, coords, texel);
      break;
    case 3:  // 1DA
      write_imagef(src1da, coords.xy, texel);
      break;
    case 4:  // 2DA
      write_imagef(src2da, coords, texel);
      break;
    default:  // Critical failure.
      return;
  }
}

void write_image_int(__write_only image1d_t src1d,
                     __write_only image2d_t src2d,
                     __write_only image3d_t src3d,
                     __write_only image1d_array_t src1da,
                     __write_only image2d_array_t src2da,
                     uint format,
                     int4 coords,
                     int4 texel) {
  switch (format) {
    case 0:  // 1D
      write_imagei(src1d, coords.x, texel);
      break;
    case 1:  // 2D
      write_imagei(src2d, coords.xy, texel);
      break;
    case 2:  // 3D
      write_imagei(src3d, coords, texel);
      break;
    case 3:  // 1DA
      write_imagei(src1da, coords.xy, texel);
      break;
    case 4:  // 2DA
      write_imagei(src2da, coords, texel);
      break;
    default:  // Critical failure.
      return;
  }
}

//image handle is repeated since OCL doesn't allow pointers to or casting of images.
//dst is start of output pixel in destination buffer
//format.x is element count
//format.y is element size
//format.z is max(dword per pixel, 1)
//format.w is texture type.
//srcOrigin is start pixel address.
//No export for 64, 96, 128 bit formats
__kernel void copy_image_to_buffer(
    __read_only image1d_t src1d,
    __read_only image2d_t src2d,
    __read_only image3d_t src3d,
    __read_only image1d_array_t src1da,
    __read_only image2d_array_t src2da,
    __global void* const dst,
    int4        srcOrigin,
    uint4       format,
    ulong       pitch,
    ulong       slice_pitch)
{
    ulong    idxDst;
    int4     coordsSrc;
    uint4    texel;

    __global uchar* const dstUChar = (__global uchar* const)dst;
    __global ushort* const dstUShort = (__global ushort* const)dst;
    __global uint* const dstUInt = (__global uint* const)dst;

    coordsSrc.x = get_global_id(0);
    coordsSrc.y = get_global_id(1);
    coordsSrc.z = get_global_id(2);
    coordsSrc.w = 0;

    idxDst = (coordsSrc.z * slice_pitch + coordsSrc.y * pitch +
        coordsSrc.x) * format.z;

    coordsSrc.x += srcOrigin.x;
    coordsSrc.y += srcOrigin.y;
    coordsSrc.z += srcOrigin.z;

    texel = read_image(src1d, src2d, src3d, src1da, src2da, format.w, coordsSrc);

    // Check components
    switch (format.x) {
    case 1:
        // Check size
        switch (format.y) {
        case 1:
            dstUChar[idxDst] = texel.x;
            break;
        case 2:
            dstUShort[idxDst] = texel.x;
            break;
        case 4:
            dstUInt[idxDst] = texel.x;
            break;
        }
    break;
    case 2:
        // Check size
        switch (format.y) {
        case 1:
            dstUShort[idxDst] = texel.x |
               (texel.y << 8);
            break;
        case 2:
            dstUInt[idxDst] = texel.x | (texel.y << 16);
            break;
        case 4:
            dstUInt[idxDst++] = texel.x;
            dstUInt[idxDst] = texel.y;
            break;
        }
    break;
    case 4:
        // Check size
        switch (format.y) {
        case 1:
            dstUInt[idxDst] = texel.x |
               (texel.y << 8) |
               (texel.z << 16) |
               (texel.w << 24);
            break;
        case 2:
            dstUInt[idxDst++] = texel.x | (texel.y << 16);
            dstUInt[idxDst] = texel.z | (texel.w << 16);
            break;
        case 4:
            dstUInt[idxDst++] = texel.x;
            dstUInt[idxDst++] = texel.y;
            dstUInt[idxDst++] = texel.z;
            dstUInt[idxDst] = texel.w;
            break;
        }
    break;
    }
}

__kernel void copy_buffer_to_image(__global uint* src,
                                   __write_only image1d_t dst1d,
                                   __write_only image2d_t dst2d,
                                   __write_only image3d_t dst3d,
                                   __write_only image1d_array_t dst1da,
                                   __write_only image2d_array_t dst2da,
                                   int4 dstOrigin,
                                   uint4 format,
                                   ulong pitch,
                                   ulong slice_pitch) {
  ulong idxSrc;
  int4 coordsDst;
  uint4 texel;

  __global uint* srcUInt = src;
  __global ushort* srcUShort = (__global ushort*)src;
  __global uchar* srcUChar = (__global uchar*)src;

  ushort tmpUShort;
  uint tmpUInt;

  coordsDst.x = get_global_id(0);
  coordsDst.y = get_global_id(1);
  coordsDst.z = get_global_id(2);
  coordsDst.w = 0;

  idxSrc = (coordsDst.z * slice_pitch + coordsDst.y * pitch + coordsDst.x) * format.z;

  coordsDst.x += dstOrigin.x;
  coordsDst.y += dstOrigin.y;
  coordsDst.z += dstOrigin.z;

  // Check components
  switch (format.x) {
    case 1:
        // Check size
        switch (format.y) {
          case 1:
            texel.x = (uint)srcUChar[idxSrc];
            break;
          case 2:
            texel.x = (uint)srcUShort[idxSrc];
            break;
          case 4:
            texel.x = srcUInt[idxSrc];
            break;
        }
    break;
    case 2:
        // Check size
        switch (format.y) {
          case 1:
            tmpUShort = srcUShort[idxSrc];
            texel.x = (uint)(tmpUShort & 0xff);
            texel.y = (uint)(tmpUShort >> 8);
            break;
          case 2:
            tmpUInt = srcUInt[idxSrc];
            texel.x = (tmpUInt & 0xffff);
            texel.y = (tmpUInt >> 16);
            break;
          case 4:
            texel.x = srcUInt[idxSrc++];
            texel.y = srcUInt[idxSrc];
            break;
        }
    break;
    case 4:
        // Check size
        switch (format.y) {
          case 1:
            tmpUInt = srcUInt[idxSrc];
            texel.x = tmpUInt & 0xff;
            texel.y = (tmpUInt >> 8) & 0xff;
            texel.z = (tmpUInt >> 16) & 0xff;
            texel.w = (tmpUInt >> 24) & 0xff;
            break;
          case 2:
            tmpUInt = srcUInt[idxSrc++];
            texel.x = tmpUInt & 0xffff;
            texel.y = (tmpUInt >> 16);
            tmpUInt = srcUInt[idxSrc];
            texel.z = tmpUInt & 0xffff;
            texel.w = (tmpUInt >> 16);
            break;
          case 4:
            texel.x = srcUInt[idxSrc++];
            texel.y = srcUInt[idxSrc++];
            texel.z = srcUInt[idxSrc++];
            texel.w = srcUInt[idxSrc];
            break;
        }
        break;
    }
    // Write the final pixel
    write_image(dst1d, dst2d, dst3d, dst1da, dst2da, format.w, coordsDst, texel);
}

__kernel void copy_image_default(__read_only image1d_t src1d,
                                 __read_only image2d_t src2d,
                                 __read_only image3d_t src3d,
                                 __read_only image1d_array_t src1da,
                                 __read_only image2d_array_t src2da,
                                 __write_only image1d_t dst1d,
                                 __write_only image2d_t dst2d,
                                 __write_only image3d_t dst3d,
                                 __write_only image1d_array_t dst1da,
                                 __write_only image2d_array_t dst2da,
                                 int4 srcOrigin,
                                 int4 dstOrigin,
                                 int srcFormat,
                                 int dstFormat) {
  int4 coordsDst;
  int4 coordsSrc;

  coordsDst.x = get_global_id(0);
  coordsDst.y = get_global_id(1);
  coordsDst.z = get_global_id(2);
  coordsDst.w = 0;

  coordsSrc = srcOrigin + coordsDst;
  coordsDst += dstOrigin;

  uint4 texel;
  texel = read_image(src1d, src2d, src3d, src1da, src2da, srcFormat, coordsSrc);
  write_image(dst1d, dst2d, dst3d, dst1da, dst2da, dstFormat, coordsDst, texel);
}

float linear_to_standard_rgba(float l_val) {
  float s_val = l_val;

  if (isnan(s_val)) s_val = 0.0f;

  if (s_val > 1.0f) {
    s_val = 1.0f;
  } else if (s_val < 0.0f) {
    s_val = 0.0f;
  } else if (s_val < 0.0031308f) {
    s_val = 12.92f * s_val;
  } else {
    s_val = (1.055f * pow(s_val, 5.0f / 12.0f)) - 0.055f;
  }

  return s_val;
}

__kernel void copy_image_linear_to_standard(
                                            __read_only image1d_t src1d,
                                            __read_only image2d_t src2d,
                                            __read_only image3d_t src3d,
                                            __read_only image1d_array_t src1da,
                                            __read_only image2d_array_t src2da,
                                            int srcFormat,
                                            __write_only image1d_t dst1d,
                                            __write_only image2d_t dst2d,
                                            __write_only image3d_t dst3d,
                                            __write_only image1d_array_t dst1da,
                                            __write_only image2d_array_t dst2da,
                                            int dstFormat,
                                            int4 srcOrigin,
                                            int4 dstOrigin) {
  int4 coordsDst;
  int4 coordsSrc;

  coordsDst.x = get_global_id(0);
  coordsDst.y = get_global_id(1);
  coordsDst.z = get_global_id(2);
  coordsDst.w = 0;

  coordsSrc = srcOrigin + coordsDst;
  coordsDst += dstOrigin;

  float4 texel;
  texel = read_image_float(src1d, src2d, src3d, src1da, src2da, srcFormat, coordsSrc);

  texel.x = linear_to_standard_rgba(texel.x);
  texel.y = linear_to_standard_rgba(texel.y);
  texel.z = linear_to_standard_rgba(texel.z);

  write_image_float(dst1d, dst2d, dst3d, dst1da, dst2da, dstFormat, coordsDst, texel);
}

__kernel void copy_image_standard_to_linear(
                                            __read_only image1d_t src1d,
                                            __read_only image2d_t src2d,
                                            __read_only image3d_t src3d,
                                            __read_only image1d_array_t src1da,
                                            __read_only image2d_array_t src2da,
                                            int srcFormat,
                                            __write_only image1d_t dst1d,
                                            __write_only image2d_t dst2d,
                                            __write_only image3d_t dst3d,
                                            __write_only image1d_array_t dst1da,
                                            __write_only image2d_array_t dst2da,
                                            int dstFormat,
                                            int4 srcOrigin,
                                            int4 dstOrigin) {
  int4 coordsDst;
  int4 coordsSrc;

  coordsDst.x = get_global_id(0);
  coordsDst.y = get_global_id(1);
  coordsDst.z = get_global_id(2);
  coordsDst.w = 0;

  coordsSrc = srcOrigin + coordsDst;
  coordsDst += dstOrigin;

  float4 texel;
  texel = read_image_float(src1d, src2d, src3d, src1da, src2da, srcFormat, coordsSrc);
  write_image_float(dst1d, dst2d, dst3d, dst1da, dst2da, dstFormat, coordsDst, texel);
}

__kernel void copy_image_1db(
                                            __read_only image1d_buffer_t src1d,
                                            __read_only image2d_t src2d,
                                            __read_only image3d_t src3d,
                                            __read_only image1d_array_t src1da,
                                            __read_only image2d_array_t src2da,
                                            int srcFormat,
                                            __write_only image1d_t dst1d,
                                            __write_only image2d_t dst2d,
                                            __write_only image3d_t dst3d,
                                            __write_only image1d_array_t dst1da,
                                            __write_only image2d_array_t dst2da,
                                            int dstFormat,
                                            int4 srcOrigin,
                                            int4 dstOrigin)
{
    int    coordDst;
    int    coordSrc;

    coordDst = get_global_id(0);

    coordSrc = srcOrigin.x + coordDst;
    coordDst += dstOrigin.x;

    uint4  texel;
    texel = read_imageui(src1d, coordSrc);
    write_imageui(dst1d, coordDst, texel);
}

__kernel void copy_image_1db_to_reg(
                                            __read_only image1d_buffer_t src1d,
                                            __read_only image2d_t src2d,
                                            __read_only image3d_t src3d,
                                            __read_only image1d_array_t src1da,
                                            __read_only image2d_array_t src2da,
                                            int srcFormat,
                                            __write_only image1d_t dst1d,
                                            __write_only image2d_t dst2d,
                                            __write_only image3d_t dst3d,
                                            __write_only image1d_array_t dst1da,
                                            __write_only image2d_array_t dst2da,
                                            int dstFormat,
                                            int4 srcOrigin,
                                            int4 dstOrigin)
{
    int4    coordsDst;
    int    coordSrc;

    coordsDst.x = get_global_id(0);
    coordsDst.y = get_global_id(1);
    coordsDst.z = get_global_id(2);
    coordsDst.w = 0;

    coordSrc = srcOrigin.x + coordsDst.x;
    coordsDst += dstOrigin;

    uint4  texel;
    texel = read_imageui(src1d, coordSrc);
    write_imageui(dst1d, coordsDst.x, texel);
}

__kernel void copy_image_reg_to_1db(
                                            __read_only image1d_t src1d,
                                            __read_only image2d_t src2d,
                                            __read_only image3d_t src3d,
                                            __read_only image1d_array_t src1da,
                                            __read_only image2d_array_t src2da,
                                            int srcFormat,
                                            __write_only image1d_buffer_t dst1d,
                                            __write_only image2d_t dst2d,
                                            __write_only image3d_t dst3d,
                                            __write_only image1d_array_t dst1da,
                                            __write_only image2d_array_t dst2da,
                                            int dstFormat,
                                            int4 srcOrigin,
                                            int4 dstOrigin)
{
    int    coordDst;
    int4    coordsSrc;

    coordsSrc.x = get_global_id(0);
    coordsSrc.y = get_global_id(1);
    coordsSrc.z = get_global_id(2);
    coordsSrc.w = 0;

    coordDst = dstOrigin.x + coordsSrc.x;
    coordsSrc += srcOrigin;

    uint4  texel;
    texel = read_imageui(src1d, coordsSrc.x);
    write_imageui(dst1d, coordDst, texel);
}

__kernel void clear_image(__write_only image1d_t dst1d,
                          __write_only image2d_t dst2d,
                          __write_only image3d_t dst3d,
                          __write_only image1d_array_t dst1da,
                          __write_only image2d_array_t dst2da,
                          int dstFormat,
                          uint type,
                          uint4 fill_data,
                          int4 origin) {
  int4 coords;

  coords.x = get_global_id(0);
  coords.y = get_global_id(1);
  coords.z = get_global_id(2);
  coords.w = 0;

  coords += origin;

  // Check components
  switch (type) {
    case 0:
      write_image_float(dst1d, dst2d, dst3d, dst1da, dst2da, dstFormat, coords, *(float4*)&fill_data);
      break;
    case 1:
      write_image_int(dst1d, dst2d, dst3d, dst1da, dst2da, dstFormat, coords, *(int4*)&fill_data);
      break;
    case 2:
      write_image(dst1d, dst2d, dst3d, dst1da, dst2da, dstFormat, coords, fill_data);
      break;
    }
}

__kernel void clear_image_1db(__write_only image1d_buffer_t dst1d,
                              __write_only image2d_t dst2d,
                              __write_only image3d_t dst3d,
                              __write_only image1d_array_t dst1da,
                              __write_only image2d_array_t dst2da,
                              int dstFormat,
                              uint4 fill_data,
                              int4 origin,
                              uint type) {
  int4 coords;

  coords.x = get_global_id(0);

  coords += origin;

  // Check components
  switch (type) {
    case 0:
      write_imagef(dst1d, coords.x, *(float4*)&fill_data);
      break;
    case 1:
      write_imagei(dst1d, coords.x, *(int4*)&fill_data);
      break;
    case 2:
      write_imageui(dst1d, coords.x, fill_data);
      break;
    }
}
