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

#ifndef EXT_IMAGE_RESOURCE_NV_H_
#define EXT_IMAGE_RESOURCE_NV_H_

#if defined(LITTLEENDIAN_CPU)
#elif defined(BIGENDIAN_CPU)
#else
#error "BIGENDIAN_CPU or LITTLEENDIAN_CPU must be defined"
#endif

namespace rocr {
namespace image {

/**********************************************************/
/**********************************************************/
#define SQ_BUF_RSC_WRD0_REG_SZ 32
#define SQ_BUF_RSC_WRD0_BASE_ADDRESS_SZ 32

struct sq_buf_rsrc_word0_t {
#if defined(LITTLEENDIAN_CPU)
  unsigned int BASE_ADDRESS : SQ_BUF_RSC_WRD0_BASE_ADDRESS_SZ;
#elif defined(BIGENDIAN_CPU)
  unsigned int BASE_ADDRESS : SQ_BUF_RSC_WRD0_BASE_ADDRESS_SZ;
#endif
};

union SQ_BUF_RSRC_WORD0 {
  sq_buf_rsrc_word0_t bitfields, bits, f;
  uint32_t val : SQ_BUF_RSC_WRD0_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};

/***********/

#define SQ_BUF_RSC_WRD1_REG_SZ 32
#define SQ_BUF_RSC_WRD1_BASE_ADDRESS_HI_SZ  16
#define SQ_BUF_RSC_WRD1_STRIDE_SZ           14
#define SQ_BUF_RSC_WRD1_CACHE_SWIZZLE_SZ    1
#define SQ_BUF_RSC_WRD1_SWIZZLE_ENABLE_SZ   1
struct sq_buf_rsrc_word1_t {
#if defined(LITTLEENDIAN_CPU)
  unsigned int BASE_ADDRESS_HI : SQ_BUF_RSC_WRD1_BASE_ADDRESS_HI_SZ;
  unsigned int STRIDE          : SQ_BUF_RSC_WRD1_STRIDE_SZ;
  unsigned int CACHE_SWIZZLE   : SQ_BUF_RSC_WRD1_CACHE_SWIZZLE_SZ;
  unsigned int SWIZZLE_ENABLE  : SQ_BUF_RSC_WRD1_SWIZZLE_ENABLE_SZ;
#elif defined(BIGENDIAN_CPU)
  unsigned int SWIZZLE_ENABLE  : SQ_BUF_RSC_WRD1_SWIZZLE_ENABLE_SZ;
  unsigned int CACHE_SWIZZLE   : SQ_BUF_RSC_WRD1_CACHE_SWIZZLE_SZ;
  unsigned int STRIDE          : SQ_BUF_RSC_WRD1_STRIDE_SZ;
  unsigned int BASE_ADDRESS_HI : SQ_BUF_RSC_WRD1_BASE_ADDRESS_HI_SZ;
#endif
};

union SQ_BUF_RSRC_WORD1 {
  sq_buf_rsrc_word1_t bitfields, bits, f;
  uint32_t val : SQ_BUF_RSC_WRD1_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};
/***********/

#define SQ_BUF_RSC_WRD2_REG_SZ 32
#define SQ_BUF_RSC_WRD2_NUM_RECORDS_SZ 32
struct sq_buf_rsrc_word2_t {
#if defined(LITTLEENDIAN_CPU)
  unsigned int NUM_RECORDS : SQ_BUF_RSC_WRD2_NUM_RECORDS_SZ;
#elif defined(BIGENDIAN_CPU)
  unsigned int NUM_RECORDS : SQ_BUF_RSC_WRD2_NUM_RECORDS_SZ;
#endif
};
union SQ_BUF_RSRC_WORD2 {
  sq_buf_rsrc_word2_t bitfields, bits, f;
  uint32_t val : SQ_BUF_RSC_WRD2_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};
/***********/

#define SQ_BUF_RSC_WRD3_REG_SZ 32
#define SQ_BUF_RSC_WRD3_DST_SEL_X_SZ        3
#define SQ_BUF_RSC_WRD3_DST_SEL_Y_SZ        3
#define SQ_BUF_RSC_WRD3_DST_SEL_Z_SZ        3
#define SQ_BUF_RSC_WRD3_DST_SEL_W_SZ        3
#define SQ_BUF_RSC_WRD3_FORMAT_SZ           7
#define SQ_BUF_RSC_WRD3_INDEX_STRIDE_SZ     2
#define SQ_BUF_RSC_WRD3_ADD_TID_ENABLE_SZ   1
#define SQ_BUF_RSC_WRD3_RESOURCE_LEVEL      1
#define SQ_BUF_RSC_WRD3_RESERVED_1          2
#define SQ_BUF_RSC_WORD3_OOB_SELECT_SZ      2
#define SQ_BUF_RSC_WRD3_TYPE_SZ             2
struct sq_buf_rsrc_word3_t {
#if defined(LITTLEENDIAN_CPU)
  unsigned int DST_SEL_X      : SQ_BUF_RSC_WRD3_DST_SEL_X_SZ;
  unsigned int DST_SEL_Y      : SQ_BUF_RSC_WRD3_DST_SEL_Y_SZ;
  unsigned int DST_SEL_Z      : SQ_BUF_RSC_WRD3_DST_SEL_Z_SZ;
  unsigned int DST_SEL_W      : SQ_BUF_RSC_WRD3_DST_SEL_W_SZ;
  unsigned int FORMAT         : SQ_BUF_RSC_WRD3_FORMAT_SZ;
  unsigned int                : 2;
  unsigned int INDEX_STRIDE   : SQ_BUF_RSC_WRD3_INDEX_STRIDE_SZ;
  unsigned int ADD_TID_ENABLE : SQ_BUF_RSC_WRD3_ADD_TID_ENABLE_SZ;
  unsigned int RESOURCE_LEVEL : SQ_BUF_RSC_WRD3_RESOURCE_LEVEL;
  unsigned int                : 1;
  unsigned int RESERVED_1     : SQ_BUF_RSC_WRD3_RESERVED_1;
  unsigned int OOB_SELECT     : SQ_BUF_RSC_WORD3_OOB_SELECT_SZ;
  unsigned int TYPE           : SQ_BUF_RSC_WRD3_TYPE_SZ;

#elif defined(BIGENDIAN_CPU)
  unsigned int TYPE           : SQ_BUF_RSC_WRD3_TYPE_SZ;
  unsigned int OOB_SELECT     : SQ_BUF_RSC_WORD3_OOB_SELECT_SZ;
  unsigned int RESERVED_1     : SQ_BUF_RSC_WRD3_RESERVED_1;
  unsigned int                : 1;
  unsigned int RESOURCE_LEVEL : SQ_BUF_RSC_WRD3_RESOURCE_LEVEL;
  unsigned int ADD_TID_ENABLE : SQ_BUF_RSC_WRD3_ADD_TID_ENABLE_SZ;
  unsigned int INDEX_STRIDE   : SQ_BUF_RSC_WRD3_INDEX_STRIDE_SZ;
  unsigned int                : 2;
  unsigned int FORMAT         : SQ_BUF_RSC_WRD3_FORMAT_SZ;
  unsigned int DST_SEL_W      : SQ_BUF_RSC_WRD3_DST_SEL_W_SZ;
  unsigned int DST_SEL_Z      : SQ_BUF_RSC_WRD3_DST_SEL_Z_SZ;
  unsigned int DST_SEL_Y      : SQ_BUF_RSC_WRD3_DST_SEL_Y_SZ;
  unsigned int DST_SEL_X      : SQ_BUF_RSC_WRD3_DST_SEL_X_SZ;

#endif
};
union SQ_BUF_RSRC_WORD3 {
  sq_buf_rsrc_word3_t bitfields, bits, f;
  uint32_t val : SQ_BUF_RSC_WRD3_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};
/***********/

/**********************************************************/
/**********************************************************/
#define SQ_IMG_RSC_WRD0_REG_SZ 32
#define SQ_IMG_RSC_WRD0_BASE_ADDRESS_SZ 32
struct sq_img_rsrc_word0_t {
#if defined(LITTLEENDIAN_CPU)
  unsigned int BASE_ADDRESS : SQ_IMG_RSC_WRD0_BASE_ADDRESS_SZ;
#elif defined(BIGENDIAN_CPU)
  unsigned int BASE_ADDRESS : SQ_IMG_RSC_WRD0_BASE_ADDRESS_SZ;
#endif
};
union SQ_IMG_RSRC_WORD0 {
  sq_img_rsrc_word0_t bitfields, bits, f;
  uint32_t val : SQ_IMG_RSC_WRD0_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};
/***********/

#define SQ_IMG_RSC_WRD1_REG_SZ 32
#define SQ_IMG_RSC_WRD1_BASE_ADDRESS_HI_SZ  8
#define SQ_IMG_RSC_WRD1_MIN_LOD_SZ          12
#define SQ_IMG_RSC_WRD1_FORMAT_SZ           9
#define SQ_IMG_RSC_WRD1_WIDTH_LO            2

struct sq_img_rsrc_word1_t{
#if defined(LITTLEENDIAN_CPU)
  unsigned int BASE_ADDRESS_HI : SQ_IMG_RSC_WRD1_BASE_ADDRESS_HI_SZ;
  unsigned int MIN_LOD         : SQ_IMG_RSC_WRD1_MIN_LOD_SZ;
  unsigned int FORMAT          : SQ_IMG_RSC_WRD1_FORMAT_SZ;
  unsigned int                 : 1;
  unsigned int WIDTH           : SQ_IMG_RSC_WRD1_WIDTH_LO;
#elif defined(BIGENDIAN_CPU)
  unsigned int WIDTH           : SQ_IMG_RSC_WRD1_WIDTH_LO;
  unsigned int                 : 1;
  unsigned int FORMAT          : SQ_IMG_RSC_WRD1_FORMAT_SZ;
  unsigned int MIN_LOD         : SQ_IMG_RSC_WRD1_MIN_LOD_SZ;
  unsigned int BASE_ADDRESS_HI : SQ_IMG_RSC_WRD1_BASE_ADDRESS_HI_SZ;
#endif
};
union SQ_IMG_RSRC_WORD1 {
  sq_img_rsrc_word1_t bitfields, bits, f;
  uint32_t val : SQ_IMG_RSC_WRD1_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};
/***********/

#define SQ_IMG_RSC_WRD2_REG_SZ 32
#define SQ_IMG_RSC_WRD2_WIDTH_HI_SZ        12
#define SQ_IMG_RSC_WRD2_HEIGHT_SZ          14
#define SQ_IMG_RSC_WRD2_RESOURCE_LEVEL_SZ  1
struct sq_img_rsrc_word2_t {
#if defined(LITTLEENDIAN_CPU)
  unsigned int WIDTH_HI       : SQ_IMG_RSC_WRD2_WIDTH_HI_SZ;
  unsigned int                : 2;
  unsigned int HEIGHT         : SQ_IMG_RSC_WRD2_HEIGHT_SZ;
  unsigned int                : 2;
  unsigned int                : 1;
  unsigned int RESOURCE_LEVEL : SQ_IMG_RSC_WRD2_RESOURCE_LEVEL_SZ;
#elif defined(BIGENDIAN_CPU)
  unsigned int RESOURCE_LEVEL : SQ_IMG_RSC_WRD2_RESOURCE_LEVEL_SZ;
  unsigned int RESERVED       : 1;
  unsigned int RESERVED       : 2;
  unsigned int HEIGHT         : SQ_IMG_RSC_WRD2_HEIGHT_SZ;
  unsigned int                : 2;
  unsigned int WIDTH_HI       : SQ_IMG_RSC_WRD2_WIDTH_SZ;
#endif
};
union SQ_IMG_RSRC_WORD2 {
  sq_img_rsrc_word2_t bitfields, bits, f;
  uint32_t val : SQ_IMG_RSC_WRD2_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};
/***********/

#define SQ_IMG_RSC_WRD3_REG_SZ 32
#define SQ_IMG_RSC_WRD3_DST_SEL_X_SZ  3
#define SQ_IMG_RSC_WRD3_DST_SEL_Y_SZ  3
#define SQ_IMG_RSC_WRD3_DST_SEL_Z_SZ  3
#define SQ_IMG_RSC_WRD3_DST_SEL_W_SZ  3
#define SQ_IMG_RSC_WRD3_BASE_LEVEL_SZ 4
#define SQ_IMG_RSC_WRD3_LAST_LEVEL_SZ 4
#define SQ_IMG_RSC_WRD3_SW_MODE_SZ    5
#define SQ_IMG_RSC_WRD3_BC_SWIZZLE_SZ 3
#define SQ_IMG_RSC_WRD3_TYPE_SZ       4
struct sq_img_rsrc_word3_t {
#if defined(LITTLEENDIAN_CPU)
  unsigned int DST_SEL_X  : SQ_IMG_RSC_WRD3_DST_SEL_X_SZ;
  unsigned int DST_SEL_Y  : SQ_IMG_RSC_WRD3_DST_SEL_Y_SZ;
  unsigned int DST_SEL_Z  : SQ_IMG_RSC_WRD3_DST_SEL_Z_SZ;
  unsigned int DST_SEL_W  : SQ_IMG_RSC_WRD3_DST_SEL_W_SZ;
  unsigned int BASE_LEVEL : SQ_IMG_RSC_WRD3_BASE_LEVEL_SZ;
  unsigned int LAST_LEVEL : SQ_IMG_RSC_WRD3_LAST_LEVEL_SZ;
  unsigned int SW_MODE    : SQ_IMG_RSC_WRD3_SW_MODE_SZ;
  unsigned int BC_SWIZZLE : SQ_IMG_RSC_WRD3_BC_SWIZZLE_SZ;
  unsigned int TYPE       : SQ_IMG_RSC_WRD3_TYPE_SZ;
#elif defined(BIGENDIAN_CPU)
  unsigned int TYPE       : SQ_IMG_RSC_WRD3_TYPE_SZ;
  unsigned int BC_SWIZZLE : SQ_IMG_RSC_WRD3_BC_SWIZZLE_SZ;
  unsigned int W_MODE     : SQ_IMG_RSC_WRD3_SW_MODE_SZ;
  unsigned int LAST_LEVEL : SQ_IMG_RSC_WRD3_LAST_LEVEL_SZ;
  unsigned int BASE_LEVEL : SQ_IMG_RSC_WRD3_BASE_LEVEL_SZ;
  unsigned int DST_SEL_W  : SQ_IMG_RSC_WRD3_DST_SEL_W_SZ;
  unsigned int DST_SEL_Z  : SQ_IMG_RSC_WRD3_DST_SEL_Z_SZ;
  unsigned int DST_SEL_Y  : SQ_IMG_RSC_WRD3_DST_SEL_Y_SZ;
  unsigned int DST_SEL_X  : SQ_IMG_RSC_WRD3_DST_SEL_X_SZ;
#endif
};
union SQ_IMG_RSRC_WORD3 {
  sq_img_rsrc_word3_t bitfields, bits, f;
  uint32_t val : SQ_IMG_RSC_WRD3_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};
/***********/

#define SQ_IMG_RSC_WRD4_REG_SZ 32
#define SQ_IMG_RSC_WRD4_DEPTH_SZ    13
#define SQ_IMG_RSC_WRD4_BASE_ARR_SZ 13
#define SQ_IMG_RSC_WRD4_PITCH_SZ 14
union sq_img_rsrc_word4_t {
  struct {
#if defined(LITTLEENDIAN_CPU)
    // For arrays this is last slice in view, for 3D this is depth-1, For remaining this is pitch-1
    unsigned int DEPTH      : SQ_IMG_RSC_WRD4_DEPTH_SZ;
    unsigned int            : 1; //Pitch[13] in gfx1030
    unsigned int            : 2;
    unsigned int BASE_ARRAY : SQ_IMG_RSC_WRD4_BASE_ARR_SZ;
    unsigned int            : 3;
#elif defined(BIGENDIAN_CPU)
    unsigned int            : 3;
    unsigned int BASE_ARRAY : SQ_IMG_RSC_WRD4_BASE_ARR_SZ;
    unsigned int            : 2;
    unsigned int            : 1; //Pitch[13] in gfx1030
    unsigned int DEPTH      : SQ_IMG_RSC_WRD4_DEPTH_SZ; //Pitch[0:12] in gfx1030
#endif
  };
  struct {
#if defined(LITTLEENDIAN_CPU)
    // For 1d, 2d and 2d-msaa in gfx1030 this is pitch-1
    unsigned int PITCH      : SQ_IMG_RSC_WRD4_PITCH_SZ;
    unsigned int            : SQ_IMG_RSC_WRD4_REG_SZ-SQ_IMG_RSC_WRD4_PITCH_SZ;
#elif defined(BIGENDIAN_CPU)
    unsigned int            : SQ_IMG_RSC_WRD4_REG_SZ-SQ_IMG_RSC_WRD4_PITCH_SZ;
    unsigned int PITCH      : SQ_IMG_RSC_WRD4_PITCH_SZ;
#endif
  };
};
union SQ_IMG_RSRC_WORD4 {
  sq_img_rsrc_word4_t bitfields, bits, f;
  uint32_t val : SQ_IMG_RSC_WRD4_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};
/***********/

#define SQ_IMG_RSC_WRD5_REG_SZ 32
#define SQ_IMG_RSC_WRD5_ARRAY_PITCH_SZ               4
#define SQ_IMG_RSC_WRD5_MAX_MIP_SZ                   4
//#define SQ_IMG_RSC_WRD5_DSCAL_OR_MID_LOD_WRN_SZ      4
//#define SQ_IMG_RSC_WRD5_HSCAL_OR_MID_LOD_WRN_SZ      4
//#define SQ_IMG_RSC_WRD5_WSCAL_OR_MID_LOD_WRN_SZ      4
#define SQ_IMG_RSC_WRD5_MID_LOD_WRN_SZ               12
#define SQ_IMG_RSC_WRD5_PERF_MOD_SZ                  3
#define SQ_IMG_RSC_WRD5_CORNER_SAMPLES_SZ            1
#define SQ_IMG_RSC_WRD5_LINKED_RESOURCE_SZ           1
#define SQ_IMG_RSC_WRD5_LOD_HDW_CNT_EN_SZ            1
#define SQ_IMG_RSC_WRD5_PRT_DEFAULT_SZ               1
#define SQ_IMG_RSC_WRD5_BIG_PAGE_SZ                  1

struct sq_img_rsrc_word5_t {
#if defined(LITTLEENDIAN_CPU)
  unsigned int ARRAY_PITCH          : SQ_IMG_RSC_WRD5_ARRAY_PITCH_SZ;
  unsigned int MAX_MIP              : SQ_IMG_RSC_WRD5_MAX_MIP_SZ;
  unsigned int MID_LOD_WRN          : SQ_IMG_RSC_WRD5_MID_LOD_WRN_SZ;
//  unsigned int DSCAL_OR_MID_LOD_WRN : SQ_IMG_RSC_WRD5_DSCAL_OR_MID_LOD_WRN_SZ;
//  unsigned int HSCAL_OR_MID_LOD_WRN : SQ_IMG_RSC_WRD5_HSCAL_OR_MID_LOD_WRN_SZ;
//  unsigned int WSCAL_OR_MID_LOD_WRN : SQ_IMG_RSC_WRD5_WSCAL_OR_MID_LOD_WRN_SZ;
  unsigned int PERF_MOD             : SQ_IMG_RSC_WRD5_PERF_MOD_SZ;
  unsigned int CORNER_SAMPLES       : SQ_IMG_RSC_WRD5_CORNER_SAMPLES_SZ;
  unsigned int LINKED_RESOURCE      : SQ_IMG_RSC_WRD5_LINKED_RESOURCE_SZ;
  unsigned int LOD_HDW_CNT_EN       : SQ_IMG_RSC_WRD5_LOD_HDW_CNT_EN_SZ;
  unsigned int PRT_DEFAULT          : SQ_IMG_RSC_WRD5_PRT_DEFAULT_SZ;
  unsigned int                      : 4;
  unsigned int BIG_PAGE             : SQ_IMG_RSC_WRD5_BIG_PAGE_SZ;
#elif defined(BIGENDIAN_CPU)
  unsigned int BIG_PAGE             : SQ_IMG_RSC_WRD5_BIG_PAGE_SZ;
  unsigned int                      : 4;
  unsigned int PRT_DEFAULT          : SQ_IMG_RSC_WRD5_PRT_DEFAULT_SZ;
  unsigned int LOD_HDW_CNT_EN       : SQ_IMG_RSC_WRD5_LOD_HDW_CNT_EN_SZ;
  unsigned int LINKED_RESOURCE      : SQ_IMG_RSC_WRD5_LINKED_RESOURCE_SZ;
  unsigned int CORNER_SAMPLES       : SQ_IMG_RSC_WRD5_CORNER_SAMPLES_SZ;
  unsigned int PERF_MOD             : SQ_IMG_RSC_WRD5_PERF_MOD_SZ;
  unsigned int MID_LOD_WRN          : SQ_IMG_RSC_WRD5_MID_LOD_WRN_SZ;
//  unsigned int WSCAL_OR_MID_LOD_WRN : SQ_IMG_RSC_WRD5_WSCAL_OR_MID_LOD_WRN_SZ;
//  unsigned int HSCAL_OR_MID_LOD_WRN : SQ_IMG_RSC_WRD5_HSCAL_OR_MID_LOD_WRN_SZ;
//  unsigned int DSCAL_OR_MID_LOD_WRN : SQ_IMG_RSC_WRD5_DSCAL_OR_MID_LOD_WRN_SZ;
  unsigned int MAX_MIP              : SQ_IMG_RSC_WRD5_MAX_MIP_SZ;
  unsigned int ARRAY_PITCH          : SQ_IMG_RSC_WRD5_ARRAY_PITCH_SZ;
#endif
};

union SQ_IMG_RSRC_WORD5 {
  sq_img_rsrc_word5_t bitfields, bits, f;
  uint32_t val : SQ_IMG_RSC_WRD5_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};
/***********/

#define SQ_IMG_RSC_WRD6_REG_SZ 32
#define SQ_IMG_RSC_WRD6_COUNTER_BANK_ID_SZ        8
#define SQ_IMG_RSC_WRD6_RESERVED_2_SZ             2
#define SQ_IMG_RSC_WRD6_ITERATE_256_SZ            1
#define SQ_IMG_RSC_WRD6_MAX_UNCOMP_BLK_SZ_SZ      2
#define SQ_IMG_RSC_WRD6_MAX_COMP_BLK_SZ_SZ        2
#define SQ_IMG_RSC_WRD6_META_PIPE_ALIGNED_SZ      1
#define SQ_IMG_RSC_WRD6_WRITE_COMPRESS_EN_SZ      1
#define SQ_IMG_RSC_WRD6_COMPRESSION_ENABLE_SZ     1
#define SQ_IMG_RSC_WRD6_ALPHA_IS_ON_MSB_SZ        1
#define SQ_IMG_RSC_WRD6_COLOR_TRANSFORM_SZ        1
#define SQ_IMG_RSC_WRD6_META_DATA_ADDR_SZ         8
struct sq_img_rsrc_word6_t {
#if defined(LITTLEENDIAN_CPU)
  unsigned int COUNTER_BANK_ID       : SQ_IMG_RSC_WRD6_COUNTER_BANK_ID_SZ;
  unsigned int RESERVED_2            : SQ_IMG_RSC_WRD6_RESERVED_2_SZ;
  unsigned int ITERATE_256           : SQ_IMG_RSC_WRD6_ITERATE_256_SZ;
  unsigned int                       : 4;
  unsigned int MAX_UNCOMP_BLK_SZ     : SQ_IMG_RSC_WRD6_MAX_UNCOMP_BLK_SZ_SZ;
  unsigned int MAX_COMP_BLK_SZ       : SQ_IMG_RSC_WRD6_MAX_COMP_BLK_SZ_SZ;
  unsigned int META_PIPE_ALIGNED     : SQ_IMG_RSC_WRD6_META_PIPE_ALIGNED_SZ;
  unsigned int WRITE_COMPRESS_ENABLE : SQ_IMG_RSC_WRD6_WRITE_COMPRESS_EN_SZ;
  unsigned int COMPRESSION_ENABLE    : SQ_IMG_RSC_WRD6_COMPRESSION_ENABLE_SZ;
  unsigned int ALPHA_IS_ON_MSB       : SQ_IMG_RSC_WRD6_ALPHA_IS_ON_MSB_SZ;
  unsigned int COLOR_TRANSFORM       : SQ_IMG_RSC_WRD6_COLOR_TRANSFORM_SZ;
  unsigned int META_DATA_ADDRESS     : SQ_IMG_RSC_WRD6_META_DATA_ADDR_SZ;
#elif defined(BIGENDIAN_CPU)
  unsigned int META_DATA_ADDRESS     : SQ_IMG_RSC_WRD6_META_DATA_ADDR_SZ;
  unsigned int COLOR_TRANSFORM       : SQ_IMG_RSC_WRD6_COLOR_TRANSFORM_SZ;
  unsigned int ALPHA_IS_ON_MSB       : SQ_IMG_RSC_WRD6_ALPHA_IS_ON_MSB_SZ;
  unsigned int COMPRESSION_ENABLE    : SQ_IMG_RSC_WRD6_COMPRESSION_ENABLE_SZ;
  unsigned int WRITE_COMPRESS_ENABLE : SQ_IMG_RSC_WRD6_WRITE_COMPRESS_EN_SZ;
  unsigned int META_PIPE_ALIGNED     : SQ_IMG_RSC_WRD6_META_PIPE_ALIGNED_SZ;
  unsigned int MAX_COMP_BLK_SZ       : SQ_IMG_RSC_WRD6_MAX_COMP_BLK_SZ_SZ;
  unsigned int MAX_UNCOMP_BLK_SZ     : SQ_IMG_RSC_WRD6_MAX_UNCOMP_BLK_SZ_SZ;
  unsigned int                       : 4;
  unsigned int ITERATE_256           : SQ_IMG_RSC_WRD6_ITERATE_256_SZ;
  unsigned int RESERVED_2            : SQ_IMG_RSC_WRD6_RESERVED_2_SZ;
  unsigned int COUNTER_BANK_ID       : SQ_IMG_RSC_WRD6_COUNTER_BANK_ID_SZ;
#endif
};
union SQ_IMG_RSRC_WORD6 {
  sq_img_rsrc_word6_t bitfields, bits, f;
  uint32_t val : SQ_IMG_RSC_WRD6_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};
/***********/

#define SQ_IMG_RSC_WRD7_REG_SZ 32
#define SQ_IMG_RSC_WRD7_META_DATA_ADDRESS_HI_SZ 32
struct sq_img_rsrc_word7_t {
#if defined(LITTLEENDIAN_CPU)
  unsigned int META_DATA_ADDRESS_HI : SQ_IMG_RSC_WRD7_META_DATA_ADDRESS_HI_SZ;
#elif defined(BIGENDIAN_CPU)
  unsigned int META_DATA_ADDRESS_HI : SQ_IMG_RSC_WRD7_META_DATA_ADDRESS_HI_SZ;
#endif
};
union SQ_IMG_RSRC_WORD7 {
  sq_img_rsrc_word7_t bitfields, bits, f;
  uint32_t val : SQ_IMG_RSC_WRD7_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};
/***********/
/**********************************************************/
/**********************************************************/

#define SQ_IMG_SAMP_WORD0_REG_SZ 32
#define SQ_IMG_SAMP_WORD0_CLAMP_X_SZ            3
#define SQ_IMG_SAMP_WORD0_CLAMP_Y_SZ            3
#define SQ_IMG_SAMP_WORD0_CLAMP_Z_SZ            3
#define SQ_IMG_SAMP_WORD0_MAX_ANISO_RATIO_SZ    3
#define SQ_IMG_SAMP_WORD0_DEPTH_COMPARE_FUNC_SZ 3
#define SQ_IMG_SAMP_WORD0_FORCE_UNNORMALIZED_SZ 1
#define SQ_IMG_SAMP_WORD0_ANISO_THRESHOLD_SZ    3
#define SQ_IMG_SAMP_WORD0_MC_COORD_TRUNC_SZ     1
#define SQ_IMG_SAMP_WORD0_FORCE_DEGAMMA_SZ      1
#define SQ_IMG_SAMP_WORD0_ANISO_BIAS_SZ         6
#define SQ_IMG_SAMP_WORD0_TRUNC_COORD_SZ        1
#define SQ_IMG_SAMP_WORD0_DISABLE_CUBE_WRAP_SZ  1
#define SQ_IMG_SAMP_WORD0_FILTER_MODE_SZ        2
#define SQ_IMG_SAMP_WORD0_SKIP_DEGAMMA_SZ       1
struct sq_img_samp_word0_t {
#if defined(LITTLEENDIAN_CPU)
  unsigned int CLAMP_X            : SQ_IMG_SAMP_WORD0_CLAMP_X_SZ;
  unsigned int CLAMP_Y            : SQ_IMG_SAMP_WORD0_CLAMP_Y_SZ;
  unsigned int CLAMP_Z            : SQ_IMG_SAMP_WORD0_CLAMP_Z_SZ;
  unsigned int MAX_ANISO_RATIO    : SQ_IMG_SAMP_WORD0_MAX_ANISO_RATIO_SZ;
  unsigned int DEPTH_COMPARE_FUNC : SQ_IMG_SAMP_WORD0_DEPTH_COMPARE_FUNC_SZ;
  unsigned int FORCE_UNNORMALIZED : SQ_IMG_SAMP_WORD0_FORCE_UNNORMALIZED_SZ;
  unsigned int ANISO_THRESHOLD    : SQ_IMG_SAMP_WORD0_ANISO_THRESHOLD_SZ;
  unsigned int MC_COORD_TRUNC     : SQ_IMG_SAMP_WORD0_MC_COORD_TRUNC_SZ;
  unsigned int FORCE_DEGAMMA      : SQ_IMG_SAMP_WORD0_FORCE_DEGAMMA_SZ;
  unsigned int ANISO_BIAS         : SQ_IMG_SAMP_WORD0_ANISO_BIAS_SZ;
  unsigned int TRUNC_COORD        : SQ_IMG_SAMP_WORD0_TRUNC_COORD_SZ;
  unsigned int DISABLE_CUBE_WRAP  : SQ_IMG_SAMP_WORD0_DISABLE_CUBE_WRAP_SZ;
  unsigned int FILTER_MODE        : SQ_IMG_SAMP_WORD0_FILTER_MODE_SZ;
  unsigned int SKIP_DEGAMMA       : SQ_IMG_SAMP_WORD0_SKIP_DEGAMMA_SZ;
#elif defined(BIGENDIAN_CPU)
  unsigned int SKIP_DEGAMMA       : SQ_IMG_SAMP_WORD0_SKIP_DEGAMMA_SZ;
  unsigned int FILTER_MODE        : SQ_IMG_SAMP_WORD0_FILTER_MODE_SZ;
  unsigned int DISABLE_CUBE_WRAP  : SQ_IMG_SAMP_WORD0_DISABLE_CUBE_WRAP_SZ;
  unsigned int TRUNC_COORD        : SQ_IMG_SAMP_WORD0_TRUNC_COORD_SZ;
  unsigned int ANISO_BIAS         : SQ_IMG_SAMP_WORD0_ANISO_BIAS_SZ;
  unsigned int FORCE_DEGAMMA      : SQ_IMG_SAMP_WORD0_FORCE_DEGAMMA_SZ;
  unsigned int MC_COORD_TRUNC     : SQ_IMG_SAMP_WORD0_MC_COORD_TRUNC_SZ;
  unsigned int ANISO_THRESHOLD    : SQ_IMG_SAMP_WORD0_ANISO_THRESHOLD_SZ;
  unsigned int FORCE_UNNORMALIZED : SQ_IMG_SAMP_WORD0_FORCE_UNNORMALIZED_SZ;
  unsigned int DEPTH_COMPARE_FUNC : SQ_IMG_SAMP_WORD0_DEPTH_COMPARE_FUNC_SZ;
  unsigned int MAX_ANISO_RATIO    : SQ_IMG_SAMP_WORD0_MAX_ANISO_RATIO_SZ;
  unsigned int CLAMP_Z            : SQ_IMG_SAMP_WORD0_CLAMP_Z_SZ;
  unsigned int CLAMP_Y            : SQ_IMG_SAMP_WORD0_CLAMP_Y_SZ;
  unsigned int CLAMP_X            : SQ_IMG_SAMP_WORD0_CLAMP_X_SZ;
#endif
};

union SQ_IMG_SAMP_WORD0 {
  sq_img_samp_word0_t bitfields, bits, f;
  uint32_t val : SQ_IMG_SAMP_WORD0_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};
/***********/

#define SQ_IMG_SAMP_WORD1_REG_SZ 32
#define SQ_IMG_SAMP_WORD1_MIN_LOD_SZ  12
#define SQ_IMG_SAMP_WORD1_MAX_LOD_SZ  12
#define SQ_IMG_SAMP_WORD1_PERF_MIP_SZ 4
#define SQ_IMG_SAMP_WORD1_PERF_Z_SZ   4
struct sq_img_samp_word1_t {
#if defined(LITTLEENDIAN_CPU)
  unsigned int MIN_LOD  : SQ_IMG_SAMP_WORD1_MIN_LOD_SZ;
  unsigned int MAX_LOD  : SQ_IMG_SAMP_WORD1_MAX_LOD_SZ;
  unsigned int PERF_MIP : SQ_IMG_SAMP_WORD1_PERF_MIP_SZ;
  unsigned int PERF_Z   : SQ_IMG_SAMP_WORD1_PERF_Z_SZ;
#elif defined(BIGENDIAN_CPU)
  unsigned int PERF_Z   : SQ_IMG_SAMP_WORD1_PERF_Z_SZ;
  unsigned int PERF_MIP : SQ_IMG_SAMP_WORD1_PERF_MIP_SZ;
  unsigned int MAX_LOD  : SQ_IMG_SAMP_WORD1_MAX_LOD_SZ;
  unsigned int MIN_LOD  : SQ_IMG_SAMP_WORD1_MIN_LOD_SZ;
#endif
};

union SQ_IMG_SAMP_WORD1 {
  sq_img_samp_word1_t bitfields, bits, f;
  uint32_t val : SQ_IMG_SAMP_WORD1_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};
/***********/

#define SQ_IMG_SAMP_WORD2_REG_SZ 32
#define SQ_IMG_SAMP_WORD2_BC_LRS_LB_SZ            12
#define SQ_IMG_SAMP_WORD2_BC_OR_BCT_SZ            2
#define SQ_IMG_SAMP_WORD2_LOD_BIAS_SEC_SZ         6
#define SQ_IMG_SAMP_WORD2_XY_MAG_FILTER_SZ        2
#define SQ_IMG_SAMP_WORD2_XY_MIN_FILTER_SZ        2
#define SQ_IMG_SAMP_WORD2_Z_FILTER_SZ             2
#define SQ_IMG_SAMP_WORD2_MIP_FILTER_SZ           2
#define SQ_IMG_SAMP_WORD2_MIP_POINT_PRECLAMP_SZ   1
#define SQ_IMG_SAMP_WORD2_ANISO_OVERRIDE_SZ       1
#define SQ_IMG_SAMP_WORD2_BLEND_ZERO_PRT_SZ       1
#define SQ_IMG_SAMP_WORD2_DERIV_ADJUST_ENABLE_SZ  1
struct sq_img_samp_word2_t {
#if defined(LITTLEENDIAN_CPU)
  unsigned int BC_LRS_LB          : SQ_IMG_SAMP_WORD2_BC_LRS_LB_SZ;
  unsigned int BC_OR_BCT          : SQ_IMG_SAMP_WORD2_BC_OR_BCT_SZ;
  unsigned int LOD_BIAS_SEC       : SQ_IMG_SAMP_WORD2_LOD_BIAS_SEC_SZ;
  unsigned int XY_MAG_FILTER      : SQ_IMG_SAMP_WORD2_XY_MAG_FILTER_SZ;
  unsigned int XY_MIN_FILTER      : SQ_IMG_SAMP_WORD2_XY_MIN_FILTER_SZ;
  unsigned int Z_FILTER           : SQ_IMG_SAMP_WORD2_Z_FILTER_SZ;
  unsigned int MIP_FILTER         : SQ_IMG_SAMP_WORD2_MIP_FILTER_SZ;
  unsigned int MIP_POINT_PRECLAMP : SQ_IMG_SAMP_WORD2_MIP_POINT_PRECLAMP_SZ;
  unsigned int ANISO_OVERRIDE     : SQ_IMG_SAMP_WORD2_ANISO_OVERRIDE_SZ;
  unsigned int BLEND_ZERO_PRT     : SQ_IMG_SAMP_WORD2_BLEND_ZERO_PRT_SZ;
  unsigned int DERIV_ADJUST_EN    : SQ_IMG_SAMP_WORD2_DERIV_ADJUST_ENABLE_SZ;
#elif defined(BIGENDIAN_CPU)
  unsigned int DERIV_ADJUST_EN    : SQ_IMG_SAMP_WORD2_DERIV_ADJUST_ENABLE_SZ;
  unsigned int BLEND_ZERO_PRT     : SQ_IMG_SAMP_WORD2_BLEND_ZERO_PRT_SZ;
  unsigned int ANISO_OVERRIDE     : SQ_IMG_SAMP_WORD2_ANISO_OVERRIDE_SZ;
  unsigned int MIP_POINT_PRECLAMP : SQ_IMG_SAMP_WORD2_MIP_POINT_PRECLAMP_SZ;
  unsigned int MIP_FILTER         : SQ_IMG_SAMP_WORD2_MIP_FILTER_SZ;
  unsigned int Z_FILTER           : SQ_IMG_SAMP_WORD2_Z_FILTER_SZ;
  unsigned int XY_MIN_FILTER      : SQ_IMG_SAMP_WORD2_XY_MIN_FILTER_SZ;
  unsigned int XY_MAG_FILTER      : SQ_IMG_SAMP_WORD2_XY_MAG_FILTER_SZ;
  unsigned int LOD_BIAS_SEC       : SQ_IMG_SAMP_WORD2_LOD_BIAS_SEC_SZ;
  unsigned int BC_OR_BCT          : SQ_IMG_SAMP_WORD2_BC_OR_BCT_SZ;
  unsigned int LOD_BIAS           : SQ_IMG_SAMP_WORD2_BC_LRS_LB_SZ;
#endif
};

union SQ_IMG_SAMP_WORD2 {
  sq_img_samp_word2_t bitfields, bits, f;
  uint32_t val : SQ_IMG_SAMP_WORD2_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};
/***********/

#define SQ_IMG_SAMP_WORD3_REG_SZ 32
#define SQ_IMG_SAMP_WORD3_BCP_LRS_DAV_SZ      12
#define SQ_IMG_SAMP_WORD3_GRAD_ADJ_OR_DAV_SZ  16
#define SQ_IMG_SAMP_WORD3_RES_OR_DAV_SZ       2
#define SQ_IMG_SAMP_WORD3_BORD_COLOR_TYPE_SZ  2

struct sq_img_samp_word3_t {
#if defined(LITTLEENDIAN_CPU)
  unsigned int BCP_LRS_DAV       : SQ_IMG_SAMP_WORD3_BCP_LRS_DAV_SZ;
  unsigned int GRAD_ADJ_OR_DAV   : SQ_IMG_SAMP_WORD3_GRAD_ADJ_OR_DAV_SZ;
  unsigned int RES_OR_DAV        : SQ_IMG_SAMP_WORD3_RES_OR_DAV_SZ;
  unsigned int BORDER_COLOR_TYPE : SQ_IMG_SAMP_WORD3_BORD_COLOR_TYPE_SZ;
#elif defined(BIGENDIAN_CPU)
  unsigned int BORDER_COLOR_TYPE : SQ_IMG_SAMP_WORD3_BORD_COLOR_TYPE_SZ;
  unsigned int RES_OR_DAV        : SQ_IMG_SAMP_WORD3_RES_OR_DAV_SZ;
  unsigned int GRAD_ADJ_OR_DAV   : SQ_IMG_SAMP_WORD3_GRAD_ADJ_OR_DAV_SZ;
  unsigned int BCP_LRS_DAV       : SQ_IMG_SAMP_WORD3_BCP_LRS_DAV_SZ;
#endif
};

union SQ_IMG_SAMP_WORD3 {
  sq_img_samp_word3_t bitfields, bits, f;
  uint32_t val : SQ_IMG_SAMP_WORD3_REG_SZ;
  uint32_t u32All;
  int32_t  i32All;
  float    f32All;
};
/***********/

/**************************************************************/
/**************************************************************/
/**************************************************************/

typedef enum FMT {
FMT_INVALID                              = 0x00000000,
FMT_8                                    = 0x00000001,
FMT_16                                   = 0x00000002,
FMT_8_8                                  = 0x00000003,
FMT_32                                   = 0x00000004,
FMT_16_16                                = 0x00000005,
FMT_10_11_11                             = 0x00000006,
FMT_11_11_10                             = 0x00000007,
FMT_10_10_10_2                           = 0x00000008,
FMT_2_10_10_10                           = 0x00000009,
FMT_8_8_8_8                              = 0x0000000a,
FMT_32_32                                = 0x0000000b,
FMT_16_16_16_16                          = 0x0000000c,
FMT_32_32_32                             = 0x0000000d,
FMT_32_32_32_32                          = 0x0000000e,
FMT_RESERVED_78                          = 0x0000000f,
FMT_5_6_5                                = 0x00000010,
FMT_1_5_5_5                              = 0x00000011,
FMT_5_5_5_1                              = 0x00000012,
FMT_4_4_4_4                              = 0x00000013,
FMT_8_24                                 = 0x00000014,
FMT_24_8                                 = 0x00000015,
FMT_X24_8_32                             = 0x00000016,
FMT_RESERVED_155                         = 0x00000017,
FMT_1                                    = 0x00000018,
FMT_1_REVERSED                           = 0x00000019,
FMT_GB_GR                                = 0x0000001a,
FMT_BG_RG                                = 0x0000001b,
FMT_4_4                                  = 0x0000001c,
FMT_BC1                                  = 0x0000001d,
FMT_BC2                                  = 0x0000001e,
FMT_BC3                                  = 0x0000001f,
FMT_BC4                                  = 0x00000020,
FMT_BC5                                  = 0x00000021,
FMT_BC6                                  = 0x00000022,
FMT_BC7                                  = 0x00000023,
FMT_6E4                                  = 0x00000024,
FMT_5_9_9_9                              = 0x00000025,
FMT_FMASK8_S2                            = 0x00000026,
FMT_FMASK8_S4                            = 0x00000027,
FMT_FMASK8_S8                            = 0x00000028,
FMT_FMASK16_S16                          = 0x00000029,
FMT_FMASK16_S8                           = 0x0000002a,
FMT_FMASK32_S16                          = 0x0000002b,
FMT_FMASK32_S8                           = 0x0000002c,
FMT_FMASK64_S16                          = 0x0000002d,
FMT_ETC2_RGB                             = 0x0000002e,
FMT_ETC2_RGBA                            = 0x0000002f,
FMT_ETC2_R                               = 0x00000030,
FMT_ETC2_RG                              = 0x00000031,
FMT_ETC2_RGBA1                           = 0x00000032,
FMT_ASTC_2D_LDR                          = 0x00000033,
FMT_ASTC_2D_HDR                          = 0x00000034,
FMT_ASTC_2D_LDR_SRGB                     = 0x00000035,
FMT_ASTC_3D_LDR                          = 0x00000036,
FMT_ASTC_3D_HDR                          = 0x00000037,
FMT_ASTC_3D_LDR_SRGB                     = 0x00000038,
FMT_MM_8                                 = 0x00000039,
FMT_MM_8_8                               = 0x0000003a,
FMT_MM_8_8_8_8                           = 0x0000003b,
FMT_MM_VYUY8                             = 0x0000003c,
FMT_MM_10_11_11                          = 0x0000003d,
FMT_MM_2_10_10_10                        = 0x0000003e,
FMT_MM_16_16_16_16                       = 0x0000003f,
FMT_10_IN_16                             = 0x00000040,
FMT_10_IN_16_16                          = 0x00000041,
FMT_10_IN_16_16_16_16                    = 0x00000042,
FMT_7E3                                  = 0x00000043,
FMT_YCBCR                                = 0x00000044,
} FMT;

typedef enum type {
TYPE_UNORM                               = 0x00000000,
TYPE_SNORM                               = 0x00000001,
TYPE_USCALED                             = 0x00000002,
TYPE_SSCALED                             = 0x00000003,
TYPE_UINT                                = 0x00000004,
TYPE_SINT                                = 0x00000005,
TYPE_RESERVED_6                          = 0x00000006,
TYPE_FLOAT                               = 0x00000007,
TYPE_RESERVED_8                          = 0x00000008,
TYPE_SRGB                                = 0x00000009,
TYPE_UNORM_UINT                          = 0x0000000a,
TYPE_REVERSED_UNORM                      = 0x0000000b,
TYPE_FLOAT_CLAMP                         = 0x0000000c,
} type;

enum FORMAT {
CFMT_INVALID           = 0,
CFMT_8_UNORM           = 1,
CFMT_8_SNORM           = 2,
CFMT_8_UINT            = 5,
CFMT_8_SINT            = 6,
CFMT_16_UNORM          = 7,
CFMT_16_SNORM          = 8,
CFMT_16_UINT           = 11,
CFMT_16_SINT           = 12,
CFMT_16_FLOAT          = 13,
CFMT_8_8_UNORM         = 14,
CFMT_8_8_SNORM         = 15,
CFMT_8_8_UINT          = 18,
CFMT_8_8_SINT          = 19,
CFMT_32_UINT           = 20,
CFMT_32_SINT           = 21,
CFMT_32_FLOAT          = 22,
CFMT_16_16_UNORM       = 23,
CFMT_16_16_SNORM       = 24,
CFMT_16_16_UINT        = 27,
CFMT_16_16_SINT        = 28,
CFMT_16_16_FLOAT       = 29,
CFMT_10_10_10_2_UNORM  = 44,
CFMT_10_10_10_2_SNORM  = 45,
CFMT_10_10_10_2_UINT   = 48,
CFMT_10_10_10_2_SINT   = 49,
CFMT_2_10_10_10_UNORM  = 50,
CFMT_2_10_10_10_SNORM  = 51,
CFMT_2_10_10_10_UINT   = 54,
CFMT_2_10_10_10_SINT   = 55,
CFMT_8_8_8_8_UNORM     = 56,
CFMT_8_8_8_8_SNORM     = 57,
CFMT_8_8_8_8_UINT      = 60,
CFMT_8_8_8_8_SINT      = 61,
CFMT_32_32_UINT        = 62,
CFMT_32_32_SINT        = 63,
CFMT_32_32_FLOAT       = 64,
CFMT_16_16_16_16_UNORM = 65,
CFMT_16_16_16_16_SNORM = 66,
CFMT_16_16_16_16_UINT  = 69,
CFMT_16_16_16_16_SINT  = 70,
CFMT_16_16_16_16_FLOAT = 71,
CFMT_32_32_32_UINT     = 72,
CFMT_32_32_32_SINT     = 73,
CFMT_32_32_32_FLOAT    = 74,
CFMT_32_32_32_32_UINT  = 75,
CFMT_32_32_32_32_SINT  = 76,
CFMT_32_32_32_32_FLOAT = 77,
CFMT_8_SRGB            = 128,
CFMT_8_8_SRGB          = 129,
CFMT_8_8_8_8_SRGB      = 130,
CFMT_5_6_5_UNORM       = 133,
CFMT_1_5_5_5_UNORM     = 134,
CFMT_5_5_5_1_UNORM     = 135,
CFMT_8_24_UNORM        = 141,
CFMT_8_24_UINT         = 142,
CFMT_24_8_UNORM        = 143,
CFMT_24_8_UINT         = 144
};

typedef enum SEL {
  SEL_0 = 0x00000000,
  SEL_1 = 0x00000001,
  SEL_X = 0x00000004,
  SEL_Y = 0x00000005,
  SEL_Z = 0x00000006,
  SEL_W = 0x00000007,
} SEL;

typedef enum SQ_RSRC_IMG_TYPE {
  SQ_RSRC_IMG_1D            = 0x00000008,
  SQ_RSRC_IMG_2D            = 0x00000009,
  SQ_RSRC_IMG_3D            = 0x0000000a,
  SQ_RSRC_IMG_CUBE_ARRAY    = 0x0000000b,
  SQ_RSRC_IMG_1D_ARRAY      = 0x0000000c,
  SQ_RSRC_IMG_2D_ARRAY      = 0x0000000d,
  SQ_RSRC_IMG_2D_MSAA       = 0x0000000e,
  SQ_RSRC_IMG_2D_MSAA_ARRAY = 0x0000000f,
} SQ_RSRC_IMG_TYPE;

typedef enum SQ_TEX_XY_FILTER {
  SQ_TEX_XY_FILTER_POINT          = 0x00000000,
  SQ_TEX_XY_FILTER_BILINEAR       = 0x00000001,
  SQ_TEX_XY_FILTER_ANISO_POINT    = 0x00000002,
  SQ_TEX_XY_FILTER_ANISO_BILINEAR = 0x00000003,
} SQ_TEX_XY_FILTER;

typedef enum SQ_TEX_Z_FILTER {
  SQ_TEX_Z_FILTER_NONE   = 0x00000000,
  SQ_TEX_Z_FILTER_POINT  = 0x00000001,
  SQ_TEX_Z_FILTER_LINEAR = 0x00000002,
} SQ_TEX_Z_FILTER;

typedef enum SQ_TEX_MIP_FILTER {
  SQ_TEX_MIP_FILTER_NONE                = 0x00000000,
  SQ_TEX_MIP_FILTER_POINT               = 0x00000001,
  SQ_TEX_MIP_FILTER_LINEAR              = 0x00000002,
  SQ_TEX_MIP_FILTER_POINT_ANISO_ADJ__VI = 0x00000003,
} SQ_TEX_MIP_FILTER;

typedef enum SQ_TEX_CLAMP {
  SQ_TEX_WRAP                    = 0x00000000,
  SQ_TEX_MIRROR                  = 0x00000001,
  SQ_TEX_CLAMP_LAST_TEXEL        = 0x00000002,
  SQ_TEX_MIRROR_ONCE_LAST_TEXEL  = 0x00000003,
  SQ_TEX_CLAMP_HALF_BORDER       = 0x00000004,
  SQ_TEX_MIRROR_ONCE_HALF_BORDER = 0x00000005,
  SQ_TEX_CLAMP_BORDER            = 0x00000006,
  SQ_TEX_MIRROR_ONCE_BORDER      = 0x00000007,
} SQ_TEX_CLAMP;

typedef enum SQ_TEX_BORDER_COLOR {
  SQ_TEX_BORDER_COLOR_TRANS_BLACK  = 0x00000000,
  SQ_TEX_BORDER_COLOR_OPAQUE_BLACK = 0x00000001,
  SQ_TEX_BORDER_COLOR_OPAQUE_WHITE = 0x00000002,
  SQ_TEX_BORDER_COLOR_REGISTER     = 0x00000003,
} SQ_TEX_BORDER_COLOR;

typedef enum TEX_BC_SWIZZLE {
TEX_BC_Swizzle_XYZW = 0x00000000,
TEX_BC_Swizzle_XWYZ = 0x00000001,
TEX_BC_Swizzle_WZYX = 0x00000002,
TEX_BC_Swizzle_WXYZ = 0x00000003,
TEX_BC_Swizzle_ZYXW = 0x00000004,
TEX_BC_Swizzle_YXWZ = 0x00000005,
} TEX_BC_SWIZZLE;

typedef struct metadata_amd_nv_s {
    uint32_t version;  // Must be 1
    uint32_t vendorID;  // AMD
    SQ_IMG_RSRC_WORD0 word0;
    SQ_IMG_RSRC_WORD1 word1;
    SQ_IMG_RSRC_WORD2 word2;
    SQ_IMG_RSRC_WORD3 word3;
    SQ_IMG_RSRC_WORD4 word4;
    SQ_IMG_RSRC_WORD5 word5;
    SQ_IMG_RSRC_WORD6 word6;
    SQ_IMG_RSRC_WORD7 word7;
    uint32_t mip_offsets[0];
} metadata_amd_nv_t;

}  // namespace image
}  // namespace rocr
#endif  // EXT_IMAGE_RESOURCE_NV_H_
