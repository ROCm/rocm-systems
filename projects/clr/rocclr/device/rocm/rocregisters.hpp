/* Copyright (c) 2016 - 2021 Advanced Micro Devices, Inc.

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

#if !defined(LITTLEENDIAN_CPU) && !defined(BIGENDIAN_CPU)
#error "Must define LITTLEENDIAN_CPU or BIGENDIAN_CPU"
#endif
#if defined(LITTLEENDIAN_CPU) && defined(BIGENDIAN_CPU)
#error "LITTLEENDIAN_CPU and BIGENDIAN_CPU are mutually exclusive"
#endif

namespace amd::roc {

enum SQ_RSRC_IMG_TYPES {
  SQ_RSRC_IMG_1D = 0x08,
  SQ_RSRC_IMG_2D = 0x09,
  SQ_RSRC_IMG_3D = 0x0A,
  SQ_RSRC_IMG_CUBE = 0x0B,
  SQ_RSRC_IMG_1D_ARRAY = 0x0C,
  SQ_RSRC_IMG_2D_ARRAY = 0x0D,
  SQ_RSRC_IMG_2D_MSAA = 0x0E,
  SQ_RSRC_IMG_2D_MSAA_ARRAY = 0x0F
};

union SQ_IMG_RSRC_WORD0 {
  struct {
#if defined(LITTLEENDIAN_CPU)
    unsigned int base_address : 32;
#elif defined(BIGENDIAN_CPU)
    unsigned int base_address : 32;
#endif
  } bitfields, bits;
  unsigned int u32_all;
  signed int i32_all;
  float f32_all;
};

union SQ_IMG_RSRC_WORD1 {
  struct {
#if defined(LITTLEENDIAN_CPU)
    unsigned int base_address_hi : 8;
    unsigned int min_lod : 12;
    unsigned int data_format : 6;
    unsigned int num_format : 4;
    unsigned int mtype : 2;
#elif defined(BIGENDIAN_CPU)
    unsigned int mtype : 2;
    unsigned int num_format : 4;
    unsigned int data_format : 6;
    unsigned int min_lod : 12;
    unsigned int base_address_hi : 8;
#endif
  } bitfields, bits;
  unsigned int u32_all;
  signed int i32_all;
  float f32_all;
};

union SQ_IMG_RSRC_WORD2 {
  struct {
#if defined(LITTLEENDIAN_CPU)
    unsigned int width : 14;
    unsigned int height : 14;
    unsigned int perf_mod : 3;
    unsigned int interlaced : 1;
#elif defined(BIGENDIAN_CPU)
    unsigned int interlaced : 1;
    unsigned int perf_mod : 3;
    unsigned int height : 14;
    unsigned int width : 14;
#endif
  } bitfields, bits;
  unsigned int u32_all;
  signed int i32_all;
  float f32_all;
};

union SQ_IMG_RSRC_WORD3 {
  struct {
#if defined(LITTLEENDIAN_CPU)
    unsigned int dst_sel_x : 3;
    unsigned int dst_sel_y : 3;
    unsigned int dst_sel_z : 3;
    unsigned int dst_sel_w : 3;
    unsigned int base_level : 4;
    unsigned int last_level : 4;
    unsigned int tiling_index : 5;
    unsigned int pow2_pad : 1;
    unsigned int mtype : 1;
    unsigned int atc : 1;
    unsigned int type : 4;
#elif defined(BIGENDIAN_CPU)
    unsigned int type : 4;
    unsigned int atc : 1;
    unsigned int mtype : 1;
    unsigned int pow2_pad : 1;
    unsigned int tiling_index : 5;
    unsigned int last_level : 4;
    unsigned int base_level : 4;
    unsigned int dst_sel_w : 3;
    unsigned int dst_sel_z : 3;
    unsigned int dst_sel_y : 3;
    unsigned int dst_sel_x : 3;
#endif
  } bitfields, bits;
  unsigned int u32_all;
  signed int i32_all;
  float f32_all;
};

union SQ_IMG_RSRC_WORD4 {
  struct {
#if defined(LITTLEENDIAN_CPU)
    unsigned int depth : 13;
    unsigned int pitch : 14;
    unsigned int : 5;
#elif defined(BIGENDIAN_CPU)
    unsigned int : 5;
    unsigned int pitch : 14;
    unsigned int depth : 13;
#endif
  } bitfields, bits;
  unsigned int u32_all;
  signed int i32_all;
  float f32_all;
};

union SQ_IMG_RSRC_WORD5 {
  struct {
#if defined(LITTLEENDIAN_CPU)
    unsigned int base_array : 13;
    unsigned int last_array : 13;
    unsigned int : 6;
#elif defined(BIGENDIAN_CPU)
    unsigned int : 6;
    unsigned int last_array : 13;
    unsigned int base_array : 13;
#endif
  } bitfields, bits;
  unsigned int u32_all;
  signed int i32_all;
  float f32_all;
};

union SQ_IMG_RSRC_WORD6 {
  struct {
#if defined(LITTLEENDIAN_CPU)
    unsigned int min_lod_warn : 12;
    unsigned int counter_bank_id : 8;
    unsigned int lod_hdw_cnt_en : 1;
    unsigned int compression_en : 1;
    unsigned int alpha_is_on_msb : 1;
    unsigned int color_transform : 1;
    unsigned int lost_alpha_bits : 4;
    unsigned int lost_color_bits : 4;
#elif defined(BIGENDIAN_CPU)
    unsigned int lost_color_bits : 4;
    unsigned int lost_alpha_bits : 4;
    unsigned int color_transform : 1;
    unsigned int alpha_is_on_msb : 1;
    unsigned int compression_en : 1;
    unsigned int lod_hdw_cnt_en : 1;
    unsigned int counter_bank_id : 8;
    unsigned int min_lod_warn : 12;
#endif
  } bitfields, bits;
  unsigned int u32All;
  signed int i32All;
  float f32All;
};

union SQ_IMG_RSRC_WORD7 {
  struct {
#if defined(LITTLEENDIAN_CPU)
    unsigned int meta_data_address : 32;
#elif defined(BIGENDIAN_CPU)
    unsigned int meta_data_address : 32;
#endif
  } bitfields, bits;
  unsigned int u32All;
  signed int i32All;
  float f32All;
};
}  // namespace amd::roc

