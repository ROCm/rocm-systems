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

#ifndef HSA_RUNTIME_EXT_IMAGE_RESOURCE_KV_H
#define HSA_RUNTIME_EXT_IMAGE_RESOURCE_KV_H

#if defined(LITTLEENDIAN_CPU)
#elif defined(BIGENDIAN_CPU)
#else
#error "BIGENDIAN_CPU or LITTLEENDIAN_CPU must be defined"
#endif

namespace rocr {
namespace image {

union SQ_BUF_RSRC_WORD0 {
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

union SQ_BUF_RSRC_WORD1 {
  struct {
#if defined(LITTLEENDIAN_CPU)
    unsigned int base_address_hi : 16;
    unsigned int stride : 14;
    unsigned int cache_swizzle : 1;
    unsigned int swizzle_enable : 1;
#elif defined(BIGENDIAN_CPU)
    unsigned int swizzle_enable : 1;
    unsigned int cache_swizzle : 1;
    unsigned int stride : 14;
    unsigned int base_address_hi : 16;
#endif
  } bitfields, bits;
  unsigned int u32_all;
  signed int i32_all;
  float f32_all;
};

union SQ_BUF_RSRC_WORD2 {
  struct {
#if defined(LITTLEENDIAN_CPU)
    unsigned int num_records : 32;
#elif defined(BIGENDIAN_CPU)
    unsigned int num_records : 32;
#endif
  } bitfields, bits;
  unsigned int u32_all;
  signed int i32_all;
  float f32_all;
};

union SQ_BUF_RSRC_WORD3 {
  struct {
#if defined(LITTLEENDIAN_CPU)
    unsigned int dst_sel_x : 3;
    unsigned int dst_sel_y : 3;
    unsigned int dst_sel_z : 3;
    unsigned int dst_sel_w : 3;
    unsigned int num_format : 3;
    unsigned int data_format : 4;
    unsigned int element_size : 2;
    unsigned int index_stride : 2;
    unsigned int add_tid_enable : 1;
    unsigned int atc : 1;
    unsigned int hash_enable : 1;
    unsigned int heap : 1;
    unsigned int mtype : 3;
    unsigned int type : 2;
#elif defined(BIGENDIAN_CPU)
    unsigned int type : 2;
    unsigned int mtype : 3;
    unsigned int heap : 1;
    unsigned int hash_enable : 1;
    unsigned int atc : 1;
    unsigned int add_tid_enable : 1;
    unsigned int index_stride : 2;
    unsigned int element_size : 2;
    unsigned int data_format : 4;
    unsigned int num_format : 3;
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
#if	defined(LITTLEENDIAN_CPU)
    unsigned int min_lod_warn : 12;
    unsigned int counter_bank_id : 8;
    unsigned int lod_hdw_cnt_en : 1;
    unsigned int compression_en : 1;
    unsigned int alpha_is_on_msb : 1;
    unsigned int color_transform : 1;
    unsigned int lost_alpha_bits : 4;
    unsigned int lost_color_bits : 4;
#elif	defined(BIGENDIAN_CPU)
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
  unsigned int u32_all;
  signed int i32_all;
  float f32_all;
};

union SQ_IMG_RSRC_WORD7 {
  struct {
#if		defined(LITTLEENDIAN_CPU)
    unsigned int meta_data_address : 32;
#elif		defined(BIGENDIAN_CPU)
    unsigned int meta_data_address : 32;
#endif
  } bitfields, bits;
  unsigned int u32_all;
  signed int i32_all;
  float f32_all;
};

union SQ_IMG_SAMP_WORD0 {
  struct {
#if defined(LITTLEENDIAN_CPU)
    unsigned int CLAMP_X : 3;
    unsigned int CLAMP_Y : 3;
    unsigned int CLAMP_Z : 3;
    unsigned int max_aniso_ratio : 3;
    unsigned int depth_compare_func : 3;
    unsigned int force_unormalized : 1;
    unsigned int aniso_threshold : 3;
    unsigned int mc_coord_trunc : 1;
    unsigned int force_degamma : 1;
    unsigned int aniso_bias : 6;
    unsigned int trunc_coord : 1;
    unsigned int disable_cube_wrap : 1;
    unsigned int filter_mode : 2;
    unsigned int compat_mode : 1;
#elif defined(BIGENDIAN_CPU)
    unsigned int compat_mode : 1;
    unsigned int filter_mode : 2;
    unsigned int disable_cube_wrap : 1;
    unsigned int trunc_coord : 1;
    unsigned int aniso_bias : 6;
    unsigned int force_degamma : 1;
    unsigned int mc_coord_trunc : 1;
    unsigned int aniso_threshold : 3;
    unsigned int force_unormalized : 1;
    unsigned int depth_compare_func : 3;
    unsigned int max_aniso_ratio : 3;
    unsigned int CLAMP_Z : 3;
    unsigned int CLAMP_Y : 3;
    unsigned int CLAMP_X : 3;
#endif
  } bitfields, bits;
  unsigned int u32_all;
  signed int i32_all;
  float f32_all;
};

union SQ_IMG_SAMP_WORD1 {
  struct {
#if defined(LITTLEENDIAN_CPU)
    unsigned int min_lod : 12;
    unsigned int max_lod : 12;
    unsigned int perf_mip : 4;
    unsigned int perf_z : 4;
#elif defined(BIGENDIAN_CPU)
    unsigned int perf_z : 4;
    unsigned int perf_mip : 4;
    unsigned int max_lod : 12;
    unsigned int min_lod : 12;
#endif
  } bitfields, bits;
  unsigned int u32_all;
  signed int i32_all;
  float f32_all;
};

union SQ_IMG_SAMP_WORD2 {
  struct {
#if defined(LITTLEENDIAN_CPU)
    unsigned int lod_bias : 14;
    unsigned int lod_bias_sec : 6;
    unsigned int xy_mag_filter : 2;
    unsigned int xy_min_filter : 2;
    unsigned int z_filter : 2;
    unsigned int mip_filter : 2;
    unsigned int mip_point_preclamp : 1;
    unsigned int disable_lsb_ceil : 1;
    unsigned int filter_prec_fix : 1;
    unsigned int aniso_override_vi : 1;
#elif defined(BIGENDIAN_CPU)
    unsigned int aniso_override_vi : 1;
    unsigned int filter_prec_fix : 1;
    unsigned int disable_lsb_ceil : 1;
    unsigned int mip_point_preclamp : 1;
    unsigned int mip_filter : 2;
    unsigned int z_filter : 2;
    unsigned int xy_min_filter : 2;
    unsigned int xy_mag_filter : 2;
    unsigned int lod_bias_sec : 6;
    unsigned int lod_bias : 14;
#endif
  } bitfields, bits;
  unsigned int u32_all;
  signed int i32_all;
  float f32_all;
};

union SQ_IMG_SAMP_WORD3 {
  struct {
#if defined(LITTLEENDIAN_CPU)
    unsigned int border_color_ptr : 12;
    unsigned int : 18;
    unsigned int border_color_type : 2;
#elif defined(BIGENDIAN_CPU)
    unsigned int border_color_type : 2;
    unsigned int : 18;
    unsigned int border_color_ptr : 12;
#endif
  } bitfields, bits;
  unsigned int u32_all;
  signed int i32_all;
  float f32_all;
};

typedef enum FMT {
  FMT_INVALID = 0x00000000,
  FMT_8 = 0x00000001,
  FMT_16 = 0x00000002,
  FMT_8_8 = 0x00000003,
  FMT_32 = 0x00000004,
  FMT_16_16 = 0x00000005,
  FMT_10_10_10_2 = 0x00000008,
  FMT_2_10_10_10 = 0x00000009,
  FMT_8_8_8_8 = 0x0000000a,
  FMT_32_32 = 0x0000000b,
  FMT_16_16_16_16 = 0x0000000c,
  FMT_32_32_32 = 0x0000000d,
  FMT_32_32_32_32 = 0x0000000e,
  FMT_5_6_5 = 0x00000010,
  FMT_1_5_5_5 = 0x00000011,
  FMT_5_5_5_1 = 0x00000012,
  FMT_8_24 = 0x00000014,
  FMT_24_8 = 0x00000015,
  FMT_X24_8_32 = 0x00000016,
  FMT_RESERVED_24__SI__CI = 0x00000018
} FMT;

typedef enum type {
  TYPE_UNORM = 0x00000000,
  TYPE_SNORM = 0x00000001,
  TYPE_UINT = 0x00000004,
  TYPE_SINT = 0x00000005,
  TYPE_FLOAT = 0x00000007,
  TYPE_SRGB = 0x00000009
} type;

typedef enum SEL {
  SEL_0 = 0x00000000,
  SEL_1 = 0x00000001,
  SEL_X = 0x00000004,
  SEL_Y = 0x00000005,
  SEL_Z = 0x00000006,
  SEL_W = 0x00000007,
} SEL;

typedef enum SQ_RSRC_IMG_TYPE {
  SQ_RSRC_IMG_1D = 0x00000008,
  SQ_RSRC_IMG_2D = 0x00000009,
  SQ_RSRC_IMG_3D = 0x0000000a,
  SQ_RSRC_IMG_1D_ARRAY = 0x0000000c,
  SQ_RSRC_IMG_2D_ARRAY = 0x0000000d,
} SQ_RSRC_IMG_TYPE;

typedef enum SQ_TEX_XY_FILTER {
  SQ_TEX_XY_FILTER_POINT = 0x00000000,
  SQ_TEX_XY_FILTER_BILINEAR = 0x00000001,
  SQ_TEX_XY_FILTER_ANISO_POINT = 0x00000002,
  SQ_TEX_XY_FILTER_ANISO_BILINEAR = 0x00000003,
} SQ_TEX_XY_FILTER;

typedef enum SQ_TEX_Z_FILTER {
  SQ_TEX_Z_FILTER_NONE = 0x00000000,
  SQ_TEX_Z_FILTER_POINT = 0x00000001,
  SQ_TEX_Z_FILTER_LINEAR = 0x00000002,
} SQ_TEX_Z_FILTER;

typedef enum SQ_TEX_MIP_FILTER {
  SQ_TEX_MIP_FILTER_NONE = 0x00000000,
  SQ_TEX_MIP_FILTER_POINT = 0x00000001,
  SQ_TEX_MIP_FILTER_LINEAR = 0x00000002,
  SQ_TEX_MIP_FILTER_POINT_ANISO_ADJ__VI = 0x00000003,
} SQ_TEX_MIP_FILTER;

typedef enum SQ_TEX_CLAMP {
  SQ_TEX_WRAP = 0x00000000,
  SQ_TEX_MIRROR = 0x00000001,
  SQ_TEX_CLAMP_LAST_TEXEL = 0x00000002,
  SQ_TEX_MIRROR_ONCE_LAST_TEXEL = 0x00000003,
  SQ_TEX_CLAMP_HALF_BORDER = 0x00000004,
  SQ_TEX_MIRROR_ONCE_HALF_BORDER = 0x00000005,
  SQ_TEX_CLAMP_BORDER = 0x00000006,
  SQ_TEX_MIRROR_ONCE_BORDER = 0x00000007,
} SQ_TEX_CLAMP;

typedef enum SQ_TEX_BORDER_COLOR {
  SQ_TEX_BORDER_COLOR_TRANS_BLACK = 0x00000000,
  SQ_TEX_BORDER_COLOR_OPAQUE_BLACK = 0x00000001,
  SQ_TEX_BORDER_COLOR_OPAQUE_WHITE = 0x00000002,
  SQ_TEX_BORDER_COLOR_REGISTER = 0x00000003,
} SQ_TEX_BORDER_COLOR;

typedef struct metadata_amd_ci_vi_s {
    uint32_t version; // Must be 1
    uint32_t vendorID; // AMD | CZ
    SQ_IMG_RSRC_WORD0 word0;
    SQ_IMG_RSRC_WORD1 word1;
    SQ_IMG_RSRC_WORD2 word2;
    SQ_IMG_RSRC_WORD3 word3;
    SQ_IMG_RSRC_WORD4 word4;
    SQ_IMG_RSRC_WORD5 word5;
    SQ_IMG_RSRC_WORD6 word6;
    SQ_IMG_RSRC_WORD7 word7;
    uint32_t mip_offsets[0]; //Mip level offset bits [39:8] for each level (if any)
} metadata_amd_ci_vi_t;

}  // namespace image
}  // namespace rocr
#endif  // HSA_RUNTIME_EXT_IMAGE_RESOURCE_KV_H
