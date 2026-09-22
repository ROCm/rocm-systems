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

#ifndef HSA_RUNTIME_EXT_IMAGE_RESOURCE_AI_H
#define HSA_RUNTIME_EXT_IMAGE_RESOURCE_AI_H

#if defined(LITTLEENDIAN_CPU)
#elif defined(BIGENDIAN_CPU)
#else
#error "BIGENDIAN_CPU or LITTLEENDIAN_CPU must be defined"
#endif

namespace rocr {
namespace image {

        union SQ_BUF_RSRC_WORD0 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int                    BASE_ADDRESS : 32;
#elif           defined(BIGENDIAN_CPU)
                unsigned int                    BASE_ADDRESS : 32;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };


        union SQ_BUF_RSRC_WORD1 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int                 BASE_ADDRESS_HI : 16;
                unsigned int                          STRIDE : 14;
                unsigned int                   CACHE_SWIZZLE : 1;
                unsigned int                  SWIZZLE_ENABLE : 1;
#elif           defined(BIGENDIAN_CPU)
                unsigned int                  SWIZZLE_ENABLE : 1;
                unsigned int                   CACHE_SWIZZLE : 1;
                unsigned int                          STRIDE : 14;
                unsigned int                 BASE_ADDRESS_HI : 16;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };


        union SQ_BUF_RSRC_WORD2 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int                     NUM_RECORDS : 32;
#elif           defined(BIGENDIAN_CPU)
                unsigned int                     NUM_RECORDS : 32;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };


        union SQ_BUF_RSRC_WORD3 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int                       DST_SEL_X : 3;
                unsigned int                       DST_SEL_Y : 3;
                unsigned int                       DST_SEL_Z : 3;
                unsigned int                       DST_SEL_W : 3;
                unsigned int                      NUM_FORMAT : 3;
                unsigned int                     DATA_FORMAT : 4;
                unsigned int                  USER_VM_ENABLE : 1;
                unsigned int                    USER_VM_MODE : 1;
                unsigned int                    INDEX_STRIDE : 2;
                unsigned int                  ADD_TID_ENABLE : 1;
                unsigned int                                 : 3;
                unsigned int                              NV : 1;
                unsigned int                                 : 2;
                unsigned int                            TYPE : 2;
#elif           defined(BIGENDIAN_CPU)
                unsigned int                            TYPE : 2;
                unsigned int                                 : 2;
                unsigned int                              NV : 1;
                unsigned int                                 : 3;
                unsigned int                  ADD_TID_ENABLE : 1;
                unsigned int                    INDEX_STRIDE : 2;
                unsigned int                    USER_VM_MODE : 1;
                unsigned int                  USER_VM_ENABLE : 1;
                unsigned int                     DATA_FORMAT : 4;
                unsigned int                      NUM_FORMAT : 3;
                unsigned int                       DST_SEL_W : 3;
                unsigned int                       DST_SEL_Z : 3;
                unsigned int                       DST_SEL_Y : 3;
                unsigned int                       DST_SEL_X : 3;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };


        union SQ_IMG_RSRC_WORD0 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int                    BASE_ADDRESS : 32;
#elif           defined(BIGENDIAN_CPU)
                unsigned int                    BASE_ADDRESS : 32;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };


        union SQ_IMG_RSRC_WORD1 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int                 BASE_ADDRESS_HI : 8;
                unsigned int                         MIN_LOD : 12;
                unsigned int                     DATA_FORMAT : 6;
                unsigned int                      NUM_FORMAT : 4;
                unsigned int                              NV : 1;
                unsigned int                     META_DIRECT : 1;
#elif           defined(BIGENDIAN_CPU)
                unsigned int                     META_DIRECT : 1;
                unsigned int                              NV : 1;
                unsigned int                      NUM_FORMAT : 4;
                unsigned int                     DATA_FORMAT : 6;
                unsigned int                         MIN_LOD : 12;
                unsigned int                 BASE_ADDRESS_HI : 8;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };


        union SQ_IMG_RSRC_WORD2 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int                           WIDTH : 14;
                unsigned int                          HEIGHT : 14;
                unsigned int                        PERF_MOD : 3;
                unsigned int                                 : 1;
#elif           defined(BIGENDIAN_CPU)
                unsigned int                                 : 1;
                unsigned int                        PERF_MOD : 3;
                unsigned int                          HEIGHT : 14;
                unsigned int                           WIDTH : 14;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };


        union SQ_IMG_RSRC_WORD3 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int                       DST_SEL_X : 3;
                unsigned int                       DST_SEL_Y : 3;
                unsigned int                       DST_SEL_Z : 3;
                unsigned int                       DST_SEL_W : 3;
                unsigned int                      BASE_LEVEL : 4;
                unsigned int                      LAST_LEVEL : 4;
                unsigned int                         SW_MODE : 5;
                unsigned int                                 : 3;
                unsigned int                            TYPE : 4;
#elif           defined(BIGENDIAN_CPU)
                unsigned int                            TYPE : 4;
                unsigned int                                 : 3;
                unsigned int                         SW_MODE : 5;
                unsigned int                      LAST_LEVEL : 4;
                unsigned int                      BASE_LEVEL : 4;
                unsigned int                       DST_SEL_W : 3;
                unsigned int                       DST_SEL_Z : 3;
                unsigned int                       DST_SEL_Y : 3;
                unsigned int                       DST_SEL_X : 3;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };


        union SQ_IMG_RSRC_WORD4 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int                           DEPTH : 13;
                unsigned int                           PITCH : 16;
                unsigned int                      BC_SWIZZLE : 3;
#elif           defined(BIGENDIAN_CPU)
                unsigned int                      BC_SWIZZLE : 3;
                unsigned int                           PITCH : 16;
                unsigned int                           DEPTH : 13;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };


        union SQ_IMG_RSRC_WORD5 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int                      BASE_ARRAY : 13;
                unsigned int                     ARRAY_PITCH : 4;
                unsigned int            META_DATA_ADDRESS_HI : 8;
                unsigned int                     META_LINEAR : 1;
                unsigned int               META_PIPE_ALIGNED : 1;
                unsigned int                 META_RB_ALIGNED : 1;
                unsigned int                         MAX_MIP : 4;
#elif           defined(BIGENDIAN_CPU)
                unsigned int                         MAX_MIP : 4;
                unsigned int                 META_RB_ALIGNED : 1;
                unsigned int               META_PIPE_ALIGNED : 1;
                unsigned int                     META_LINEAR : 1;
                unsigned int            META_DATA_ADDRESS_HI : 8;
                unsigned int                     ARRAY_PITCH : 4;
                unsigned int                      BASE_ARRAY : 13;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };


        union SQ_IMG_RSRC_WORD6 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int                    MIN_LOD_WARN : 12;
                unsigned int                 COUNTER_BANK_ID : 8;
                unsigned int                  LOD_HDW_CNT_EN : 1;
                unsigned int                  COMPRESSION_EN : 1;
                unsigned int                 ALPHA_IS_ON_MSB : 1;
                unsigned int                 COLOR_TRANSFORM : 1;
                unsigned int                 LOST_ALPHA_BITS : 4;
                unsigned int                 LOST_COLOR_BITS : 4;
#elif           defined(BIGENDIAN_CPU)
                unsigned int                 LOST_COLOR_BITS : 4;
                unsigned int                 LOST_ALPHA_BITS : 4;
                unsigned int                 COLOR_TRANSFORM : 1;
                unsigned int                 ALPHA_IS_ON_MSB : 1;
                unsigned int                  COMPRESSION_EN : 1;
                unsigned int                  LOD_HDW_CNT_EN : 1;
                unsigned int                 COUNTER_BANK_ID : 8;
                unsigned int                    MIN_LOD_WARN : 12;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };


        union SQ_IMG_RSRC_WORD7 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int               META_DATA_ADDRESS : 32;
#elif           defined(BIGENDIAN_CPU)
                unsigned int               META_DATA_ADDRESS : 32;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };


        union SQ_IMG_SAMP_WORD0 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int                         CLAMP_X : 3;
                unsigned int                         CLAMP_Y : 3;
                unsigned int                         CLAMP_Z : 3;
                unsigned int                 MAX_ANISO_RATIO : 3;
                unsigned int              DEPTH_COMPARE_FUNC : 3;
                unsigned int              FORCE_UNNORMALIZED : 1;
                unsigned int                 ANISO_THRESHOLD : 3;
                unsigned int                  MC_COORD_TRUNC : 1;
                unsigned int                   FORCE_DEGAMMA : 1;
                unsigned int                      ANISO_BIAS : 6;
                unsigned int                     TRUNC_COORD : 1;
                unsigned int               DISABLE_CUBE_WRAP : 1;
                unsigned int                     FILTER_MODE : 2;
                unsigned int                     COMPAT_MODE : 1;
#elif           defined(BIGENDIAN_CPU)
                unsigned int                     COMPAT_MODE : 1;
                unsigned int                     FILTER_MODE : 2;
                unsigned int               DISABLE_CUBE_WRAP : 1;
                unsigned int                     TRUNC_COORD : 1;
                unsigned int                      ANISO_BIAS : 6;
                unsigned int                   FORCE_DEGAMMA : 1;
                unsigned int                  MC_COORD_TRUNC : 1;
                unsigned int                 ANISO_THRESHOLD : 3;
                unsigned int              FORCE_UNNORMALIZED : 1;
                unsigned int              DEPTH_COMPARE_FUNC : 3;
                unsigned int                 MAX_ANISO_RATIO : 3;
                unsigned int                         CLAMP_Z : 3;
                unsigned int                         CLAMP_Y : 3;
                unsigned int                         CLAMP_X : 3;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };


        union SQ_IMG_SAMP_WORD1 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int                         MIN_LOD : 12;
                unsigned int                         MAX_LOD : 12;
                unsigned int                        PERF_MIP : 4;
                unsigned int                          PERF_Z : 4;
#elif           defined(BIGENDIAN_CPU)
                unsigned int                          PERF_Z : 4;
                unsigned int                        PERF_MIP : 4;
                unsigned int                         MAX_LOD : 12;
                unsigned int                         MIN_LOD : 12;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };


        union SQ_IMG_SAMP_WORD2 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int                        LOD_BIAS : 14;
                unsigned int                    LOD_BIAS_SEC : 6;
                unsigned int                   XY_MAG_FILTER : 2;
                unsigned int                   XY_MIN_FILTER : 2;
                unsigned int                        Z_FILTER : 2;
                unsigned int                      MIP_FILTER : 2;
                unsigned int              MIP_POINT_PRECLAMP : 1;
                unsigned int                  BLEND_ZERO_PRT : 1;
                unsigned int                 FILTER_PREC_FIX : 1;
                unsigned int                  ANISO_OVERRIDE : 1;
#elif           defined(BIGENDIAN_CPU)
                unsigned int                  ANISO_OVERRIDE : 1;
                unsigned int                 FILTER_PREC_FIX : 1;
                unsigned int                  BLEND_ZERO_PRT : 1;
                unsigned int              MIP_POINT_PRECLAMP : 1;
                unsigned int                      MIP_FILTER : 2;
                unsigned int                        Z_FILTER : 2;
                unsigned int                   XY_MIN_FILTER : 2;
                unsigned int                   XY_MAG_FILTER : 2;
                unsigned int                    LOD_BIAS_SEC : 6;
                unsigned int                        LOD_BIAS : 14;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };


        union SQ_IMG_SAMP_WORD3 {
        struct {
#if             defined(LITTLEENDIAN_CPU)
                unsigned int                BORDER_COLOR_PTR : 12;
                unsigned int                    SKIP_DEGAMMA : 1;
                unsigned int                                 : 17;
                unsigned int               BORDER_COLOR_TYPE : 2;
#elif           defined(BIGENDIAN_CPU)
                unsigned int               BORDER_COLOR_TYPE : 2;
                unsigned int                                 : 17;
                unsigned int                    SKIP_DEGAMMA : 1;
                unsigned int                BORDER_COLOR_PTR : 12;
#endif
        } bitfields, bits;
        unsigned int    u32All;
        signed int      i32All;
        float   f32All;
        };



#define SQ_BUF_RSRC_WORD0_REG_SIZE     32
#define SQ_BUF_RSRC_WORD0_BASE_ADDRESS_SIZE 32

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_buf_rsrc_word0_t {
          unsigned int base_address                   : SQ_BUF_RSRC_WORD0_BASE_ADDRESS_SIZE;
     } sq_buf_rsrc_word0_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_buf_rsrc_word0_t {
          unsigned int base_address                   : SQ_BUF_RSRC_WORD0_BASE_ADDRESS_SIZE;
     } sq_buf_rsrc_word0_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_buf_rsrc_word0_t f;
} sq_buf_rsrc_word0_u;

#define SQ_BUF_RSRC_WORD1_REG_SIZE     32
#define SQ_BUF_RSRC_WORD1_BASE_ADDRESS_HI_SIZE 16
#define SQ_BUF_RSRC_WORD1_STRIDE_SIZE  14
#define SQ_BUF_RSRC_WORD1_CACHE_SWIZZLE_SIZE 1
#define SQ_BUF_RSRC_WORD1_SWIZZLE_ENABLE_SIZE 1

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_buf_rsrc_word1_t {
          unsigned int base_address_hi                : SQ_BUF_RSRC_WORD1_BASE_ADDRESS_HI_SIZE;
          unsigned int stride                         : SQ_BUF_RSRC_WORD1_STRIDE_SIZE;
          unsigned int cache_swizzle                  : SQ_BUF_RSRC_WORD1_CACHE_SWIZZLE_SIZE;
          unsigned int swizzle_enable                 : SQ_BUF_RSRC_WORD1_SWIZZLE_ENABLE_SIZE;
     } sq_buf_rsrc_word1_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_buf_rsrc_word1_t {
          unsigned int swizzle_enable                 : SQ_BUF_RSRC_WORD1_SWIZZLE_ENABLE_SIZE;
          unsigned int cache_swizzle                  : SQ_BUF_RSRC_WORD1_CACHE_SWIZZLE_SIZE;
          unsigned int stride                         : SQ_BUF_RSRC_WORD1_STRIDE_SIZE;
          unsigned int base_address_hi                : SQ_BUF_RSRC_WORD1_BASE_ADDRESS_HI_SIZE;
     } sq_buf_rsrc_word1_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_buf_rsrc_word1_t f;
} sq_buf_rsrc_word1_u;

#define SQ_BUF_RSRC_WORD2_REG_SIZE     32
#define SQ_BUF_RSRC_WORD2_NUM_RECORDS_SIZE 32

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_buf_rsrc_word2_t {
          unsigned int num_records                    : SQ_BUF_RSRC_WORD2_NUM_RECORDS_SIZE;
     } sq_buf_rsrc_word2_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_buf_rsrc_word2_t {
          unsigned int num_records                    : SQ_BUF_RSRC_WORD2_NUM_RECORDS_SIZE;
     } sq_buf_rsrc_word2_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_buf_rsrc_word2_t f;
} sq_buf_rsrc_word2_u;

#define SQ_BUF_RSRC_WORD3_REG_SIZE     32
#define SQ_BUF_RSRC_WORD3_DST_SEL_X_SIZE 3
#define SQ_BUF_RSRC_WORD3_DST_SEL_Y_SIZE 3
#define SQ_BUF_RSRC_WORD3_DST_SEL_Z_SIZE 3
#define SQ_BUF_RSRC_WORD3_DST_SEL_W_SIZE 3
#define SQ_BUF_RSRC_WORD3_NUM_FORMAT_SIZE 3
#define SQ_BUF_RSRC_WORD3_DATA_FORMAT_SIZE 4
#define SQ_BUF_RSRC_WORD3_USER_VM_ENABLE_SIZE 1
#define SQ_BUF_RSRC_WORD3_USER_VM_MODE_SIZE 1
#define SQ_BUF_RSRC_WORD3_INDEX_STRIDE_SIZE 2
#define SQ_BUF_RSRC_WORD3_ADD_TID_ENABLE_SIZE 1
#define SQ_BUF_RSRC_WORD3_NV_SIZE      1
#define SQ_BUF_RSRC_WORD3_TYPE_SIZE    2

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_buf_rsrc_word3_t {
          unsigned int dst_sel_x                      : SQ_BUF_RSRC_WORD3_DST_SEL_X_SIZE;
          unsigned int dst_sel_y                      : SQ_BUF_RSRC_WORD3_DST_SEL_Y_SIZE;
          unsigned int dst_sel_z                      : SQ_BUF_RSRC_WORD3_DST_SEL_Z_SIZE;
          unsigned int dst_sel_w                      : SQ_BUF_RSRC_WORD3_DST_SEL_W_SIZE;
          unsigned int num_format                     : SQ_BUF_RSRC_WORD3_NUM_FORMAT_SIZE;
          unsigned int data_format                    : SQ_BUF_RSRC_WORD3_DATA_FORMAT_SIZE;
          unsigned int user_vm_enable                 : SQ_BUF_RSRC_WORD3_USER_VM_ENABLE_SIZE;
          unsigned int user_vm_mode                   : SQ_BUF_RSRC_WORD3_USER_VM_MODE_SIZE;
          unsigned int index_stride                   : SQ_BUF_RSRC_WORD3_INDEX_STRIDE_SIZE;
          unsigned int add_tid_enable                 : SQ_BUF_RSRC_WORD3_ADD_TID_ENABLE_SIZE;
          unsigned int                                : 3;
          unsigned int nv                             : SQ_BUF_RSRC_WORD3_NV_SIZE;
          unsigned int                                : 2;
          unsigned int type                           : SQ_BUF_RSRC_WORD3_TYPE_SIZE;
     } sq_buf_rsrc_word3_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_buf_rsrc_word3_t {
          unsigned int type                           : SQ_BUF_RSRC_WORD3_TYPE_SIZE;
          unsigned int                                : 2;
          unsigned int nv                             : SQ_BUF_RSRC_WORD3_NV_SIZE;
          unsigned int                                : 3;
          unsigned int add_tid_enable                 : SQ_BUF_RSRC_WORD3_ADD_TID_ENABLE_SIZE;
          unsigned int index_stride                   : SQ_BUF_RSRC_WORD3_INDEX_STRIDE_SIZE;
          unsigned int user_vm_mode                   : SQ_BUF_RSRC_WORD3_USER_VM_MODE_SIZE;
          unsigned int user_vm_enable                 : SQ_BUF_RSRC_WORD3_USER_VM_ENABLE_SIZE;
          unsigned int data_format                    : SQ_BUF_RSRC_WORD3_DATA_FORMAT_SIZE;
          unsigned int num_format                     : SQ_BUF_RSRC_WORD3_NUM_FORMAT_SIZE;
          unsigned int dst_sel_w                      : SQ_BUF_RSRC_WORD3_DST_SEL_W_SIZE;
          unsigned int dst_sel_z                      : SQ_BUF_RSRC_WORD3_DST_SEL_Z_SIZE;
          unsigned int dst_sel_y                      : SQ_BUF_RSRC_WORD3_DST_SEL_Y_SIZE;
          unsigned int dst_sel_x                      : SQ_BUF_RSRC_WORD3_DST_SEL_X_SIZE;
     } sq_buf_rsrc_word3_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_buf_rsrc_word3_t f;
} sq_buf_rsrc_word3_u;


#define SQ_IMG_RSRC_WORD0_REG_SIZE     32
#define SQ_IMG_RSRC_WORD0_BASE_ADDRESS_SIZE 32

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_img_rsrc_word0_t {
          unsigned int base_address                   : SQ_IMG_RSRC_WORD0_BASE_ADDRESS_SIZE;
     } sq_img_rsrc_word0_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_img_rsrc_word0_t {
          unsigned int base_address                   : SQ_IMG_RSRC_WORD0_BASE_ADDRESS_SIZE;
     } sq_img_rsrc_word0_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_img_rsrc_word0_t f;
} sq_img_rsrc_word0_u;

#define SQ_IMG_RSRC_WORD1_REG_SIZE     32
#define SQ_IMG_RSRC_WORD1_BASE_ADDRESS_HI_SIZE 8
#define SQ_IMG_RSRC_WORD1_MIN_LOD_SIZE 12
#define SQ_IMG_RSRC_WORD1_DATA_FORMAT_SIZE 6
#define SQ_IMG_RSRC_WORD1_NUM_FORMAT_SIZE 4
#define SQ_IMG_RSRC_WORD1_NV_SIZE      1
#define SQ_IMG_RSRC_WORD1_META_DIRECT_SIZE 1

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_img_rsrc_word1_t {
          unsigned int base_address_hi                : SQ_IMG_RSRC_WORD1_BASE_ADDRESS_HI_SIZE;
          unsigned int min_lod                        : SQ_IMG_RSRC_WORD1_MIN_LOD_SIZE;
          unsigned int data_format                    : SQ_IMG_RSRC_WORD1_DATA_FORMAT_SIZE;
          unsigned int num_format                     : SQ_IMG_RSRC_WORD1_NUM_FORMAT_SIZE;
          unsigned int nv                             : SQ_IMG_RSRC_WORD1_NV_SIZE;
          unsigned int meta_direct                    : SQ_IMG_RSRC_WORD1_META_DIRECT_SIZE;
     } sq_img_rsrc_word1_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_img_rsrc_word1_t {
          unsigned int meta_direct                    : SQ_IMG_RSRC_WORD1_META_DIRECT_SIZE;
          unsigned int nv                             : SQ_IMG_RSRC_WORD1_NV_SIZE;
          unsigned int num_format                     : SQ_IMG_RSRC_WORD1_NUM_FORMAT_SIZE;
          unsigned int data_format                    : SQ_IMG_RSRC_WORD1_DATA_FORMAT_SIZE;
          unsigned int min_lod                        : SQ_IMG_RSRC_WORD1_MIN_LOD_SIZE;
          unsigned int base_address_hi                : SQ_IMG_RSRC_WORD1_BASE_ADDRESS_HI_SIZE;
     } sq_img_rsrc_word1_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_img_rsrc_word1_t f;
} sq_img_rsrc_word1_u;

#define SQ_IMG_RSRC_WORD2_REG_SIZE     32
#define SQ_IMG_RSRC_WORD2_WIDTH_SIZE   14
#define SQ_IMG_RSRC_WORD2_HEIGHT_SIZE  14
#define SQ_IMG_RSRC_WORD2_PERF_MOD_SIZE 3

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_img_rsrc_word2_t {
          unsigned int width                          : SQ_IMG_RSRC_WORD2_WIDTH_SIZE;
          unsigned int height                         : SQ_IMG_RSRC_WORD2_HEIGHT_SIZE;
          unsigned int perf_mod                       : SQ_IMG_RSRC_WORD2_PERF_MOD_SIZE;
          unsigned int                                : 1;
     } sq_img_rsrc_word2_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_img_rsrc_word2_t {
          unsigned int                                : 1;
          unsigned int perf_mod                       : SQ_IMG_RSRC_WORD2_PERF_MOD_SIZE;
          unsigned int height                         : SQ_IMG_RSRC_WORD2_HEIGHT_SIZE;
          unsigned int width                          : SQ_IMG_RSRC_WORD2_WIDTH_SIZE;
     } sq_img_rsrc_word2_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_img_rsrc_word2_t f;
} sq_img_rsrc_word2_u;

#define SQ_IMG_RSRC_WORD3_REG_SIZE     32
#define SQ_IMG_RSRC_WORD3_DST_SEL_X_SIZE 3
#define SQ_IMG_RSRC_WORD3_DST_SEL_Y_SIZE 3
#define SQ_IMG_RSRC_WORD3_DST_SEL_Z_SIZE 3
#define SQ_IMG_RSRC_WORD3_DST_SEL_W_SIZE 3
#define SQ_IMG_RSRC_WORD3_BASE_LEVEL_SIZE 4
#define SQ_IMG_RSRC_WORD3_LAST_LEVEL_SIZE 4
#define SQ_IMG_RSRC_WORD3_SW_MODE_SIZE 5
#define SQ_IMG_RSRC_WORD3_TYPE_SIZE    4

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_img_rsrc_word3_t {
          unsigned int dst_sel_x                      : SQ_IMG_RSRC_WORD3_DST_SEL_X_SIZE;
          unsigned int dst_sel_y                      : SQ_IMG_RSRC_WORD3_DST_SEL_Y_SIZE;
          unsigned int dst_sel_z                      : SQ_IMG_RSRC_WORD3_DST_SEL_Z_SIZE;
          unsigned int dst_sel_w                      : SQ_IMG_RSRC_WORD3_DST_SEL_W_SIZE;
          unsigned int base_level                     : SQ_IMG_RSRC_WORD3_BASE_LEVEL_SIZE;
          unsigned int last_level                     : SQ_IMG_RSRC_WORD3_LAST_LEVEL_SIZE;
          unsigned int sw_mode                        : SQ_IMG_RSRC_WORD3_SW_MODE_SIZE;
          unsigned int                                : 3;
          unsigned int type                           : SQ_IMG_RSRC_WORD3_TYPE_SIZE;
     } sq_img_rsrc_word3_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_img_rsrc_word3_t {
          unsigned int type                           : SQ_IMG_RSRC_WORD3_TYPE_SIZE;
          unsigned int                                : 3;
          unsigned int sw_mode                        : SQ_IMG_RSRC_WORD3_SW_MODE_SIZE;
          unsigned int last_level                     : SQ_IMG_RSRC_WORD3_LAST_LEVEL_SIZE;
          unsigned int base_level                     : SQ_IMG_RSRC_WORD3_BASE_LEVEL_SIZE;
          unsigned int dst_sel_w                      : SQ_IMG_RSRC_WORD3_DST_SEL_W_SIZE;
          unsigned int dst_sel_z                      : SQ_IMG_RSRC_WORD3_DST_SEL_Z_SIZE;
          unsigned int dst_sel_y                      : SQ_IMG_RSRC_WORD3_DST_SEL_Y_SIZE;
          unsigned int dst_sel_x                      : SQ_IMG_RSRC_WORD3_DST_SEL_X_SIZE;
     } sq_img_rsrc_word3_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_img_rsrc_word3_t f;
} sq_img_rsrc_word3_u;

#define SQ_IMG_RSRC_WORD4_REG_SIZE     32
#define SQ_IMG_RSRC_WORD4_DEPTH_SIZE   13
#define SQ_IMG_RSRC_WORD4_PITCH_SIZE   16
#define SQ_IMG_RSRC_WORD4_BC_SWIZZLE_SIZE 3

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_img_rsrc_word4_t {
          unsigned int depth                          : SQ_IMG_RSRC_WORD4_DEPTH_SIZE;
          unsigned int pitch                          : SQ_IMG_RSRC_WORD4_PITCH_SIZE;
          unsigned int bc_swizzle                     : SQ_IMG_RSRC_WORD4_BC_SWIZZLE_SIZE;
     } sq_img_rsrc_word4_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_img_rsrc_word4_t {
          unsigned int bc_swizzle                     : SQ_IMG_RSRC_WORD4_BC_SWIZZLE_SIZE;
          unsigned int pitch                          : SQ_IMG_RSRC_WORD4_PITCH_SIZE;
          unsigned int depth                          : SQ_IMG_RSRC_WORD4_DEPTH_SIZE;
     } sq_img_rsrc_word4_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_img_rsrc_word4_t f;
} sq_img_rsrc_word4_u;

#define SQ_IMG_RSRC_WORD5_REG_SIZE     32
#define SQ_IMG_RSRC_WORD5_BASE_ARRAY_SIZE 13
#define SQ_IMG_RSRC_WORD5_ARRAY_PITCH_SIZE 4
#define SQ_IMG_RSRC_WORD5_META_DATA_ADDRESS_SIZE 8
#define SQ_IMG_RSRC_WORD5_META_LINEAR_SIZE 1
#define SQ_IMG_RSRC_WORD5_META_PIPE_ALIGNED_SIZE 1
#define SQ_IMG_RSRC_WORD5_META_RB_ALIGNED_SIZE 1
#define SQ_IMG_RSRC_WORD5_MAX_MIP_SIZE 4

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_img_rsrc_word5_t {
          unsigned int base_array                     : SQ_IMG_RSRC_WORD5_BASE_ARRAY_SIZE;
          unsigned int array_pitch                    : SQ_IMG_RSRC_WORD5_ARRAY_PITCH_SIZE;
          unsigned int meta_data_address              : SQ_IMG_RSRC_WORD5_META_DATA_ADDRESS_SIZE;
          unsigned int meta_linear                    : SQ_IMG_RSRC_WORD5_META_LINEAR_SIZE;
          unsigned int meta_pipe_aligned              : SQ_IMG_RSRC_WORD5_META_PIPE_ALIGNED_SIZE;
          unsigned int meta_rb_aligned                : SQ_IMG_RSRC_WORD5_META_RB_ALIGNED_SIZE;
          unsigned int max_mip                        : SQ_IMG_RSRC_WORD5_MAX_MIP_SIZE;
     } sq_img_rsrc_word5_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_img_rsrc_word5_t {
          unsigned int max_mip                        : SQ_IMG_RSRC_WORD5_MAX_MIP_SIZE;
          unsigned int meta_rb_aligned                : SQ_IMG_RSRC_WORD5_META_RB_ALIGNED_SIZE;
          unsigned int meta_pipe_aligned              : SQ_IMG_RSRC_WORD5_META_PIPE_ALIGNED_SIZE;
          unsigned int meta_linear                    : SQ_IMG_RSRC_WORD5_META_LINEAR_SIZE;
          unsigned int meta_data_address              : SQ_IMG_RSRC_WORD5_META_DATA_ADDRESS_SIZE;
          unsigned int array_pitch                    : SQ_IMG_RSRC_WORD5_ARRAY_PITCH_SIZE;
          unsigned int base_array                     : SQ_IMG_RSRC_WORD5_BASE_ARRAY_SIZE;
     } sq_img_rsrc_word5_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_img_rsrc_word5_t f;
} sq_img_rsrc_word5_u;

#define SQ_IMG_RSRC_WORD6_REG_SIZE     32
#define SQ_IMG_RSRC_WORD6_MIN_LOD_WARN_SIZE 12
#define SQ_IMG_RSRC_WORD6_COUNTER_BANK_ID_SIZE 8
#define SQ_IMG_RSRC_WORD6_LOD_HDW_CNT_EN_SIZE 1
#define SQ_IMG_RSRC_WORD6_COMPRESSION_EN_SIZE 1
#define SQ_IMG_RSRC_WORD6_ALPHA_IS_ON_MSB_SIZE 1
#define SQ_IMG_RSRC_WORD6_COLOR_TRANSFORM_SIZE 1
#define SQ_IMG_RSRC_WORD6_LOST_ALPHA_BITS_SIZE 4
#define SQ_IMG_RSRC_WORD6_LOST_COLOR_BITS_SIZE 4

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_img_rsrc_word6_t {
          unsigned int min_lod_warn                   : SQ_IMG_RSRC_WORD6_MIN_LOD_WARN_SIZE;
          unsigned int counter_bank_id                : SQ_IMG_RSRC_WORD6_COUNTER_BANK_ID_SIZE;
          unsigned int lod_hdw_cnt_en                 : SQ_IMG_RSRC_WORD6_LOD_HDW_CNT_EN_SIZE;
          unsigned int compression_en                 : SQ_IMG_RSRC_WORD6_COMPRESSION_EN_SIZE;
          unsigned int alpha_is_on_msb                : SQ_IMG_RSRC_WORD6_ALPHA_IS_ON_MSB_SIZE;
          unsigned int color_transform                : SQ_IMG_RSRC_WORD6_COLOR_TRANSFORM_SIZE;
          unsigned int lost_alpha_bits                : SQ_IMG_RSRC_WORD6_LOST_ALPHA_BITS_SIZE;
          unsigned int lost_color_bits                : SQ_IMG_RSRC_WORD6_LOST_COLOR_BITS_SIZE;
     } sq_img_rsrc_word6_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_img_rsrc_word6_t {
          unsigned int lost_color_bits                : SQ_IMG_RSRC_WORD6_LOST_COLOR_BITS_SIZE;
          unsigned int lost_alpha_bits                : SQ_IMG_RSRC_WORD6_LOST_ALPHA_BITS_SIZE;
          unsigned int color_transform                : SQ_IMG_RSRC_WORD6_COLOR_TRANSFORM_SIZE;
          unsigned int alpha_is_on_msb                : SQ_IMG_RSRC_WORD6_ALPHA_IS_ON_MSB_SIZE;
          unsigned int compression_en                 : SQ_IMG_RSRC_WORD6_COMPRESSION_EN_SIZE;
          unsigned int lod_hdw_cnt_en                 : SQ_IMG_RSRC_WORD6_LOD_HDW_CNT_EN_SIZE;
          unsigned int counter_bank_id                : SQ_IMG_RSRC_WORD6_COUNTER_BANK_ID_SIZE;
          unsigned int min_lod_warn                   : SQ_IMG_RSRC_WORD6_MIN_LOD_WARN_SIZE;
     } sq_img_rsrc_word6_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_img_rsrc_word6_t f;
} sq_img_rsrc_word6_u;

#define SQ_IMG_RSRC_WORD7_REG_SIZE     32
#define SQ_IMG_RSRC_WORD7_META_DATA_ADDRESS_SIZE 32

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_img_rsrc_word7_t {
          unsigned int meta_data_address              : SQ_IMG_RSRC_WORD7_META_DATA_ADDRESS_SIZE;
     } sq_img_rsrc_word7_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_img_rsrc_word7_t {
          unsigned int meta_data_address              : SQ_IMG_RSRC_WORD7_META_DATA_ADDRESS_SIZE;
     } sq_img_rsrc_word7_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_img_rsrc_word7_t f;
} sq_img_rsrc_word7_u;

#define SQ_IMG_SAMP_WORD0_REG_SIZE     32
#define SQ_IMG_SAMP_WORD0_CLAMP_X_SIZE 3
#define SQ_IMG_SAMP_WORD0_CLAMP_Y_SIZE 3
#define SQ_IMG_SAMP_WORD0_CLAMP_Z_SIZE 3
#define SQ_IMG_SAMP_WORD0_MAX_ANISO_RATIO_SIZE 3
#define SQ_IMG_SAMP_WORD0_DEPTH_COMPARE_FUNC_SIZE 3
#define SQ_IMG_SAMP_WORD0_FORCE_UNNORMALIZED_SIZE 1
#define SQ_IMG_SAMP_WORD0_ANISO_THRESHOLD_SIZE 3
#define SQ_IMG_SAMP_WORD0_MC_COORD_TRUNC_SIZE 1
#define SQ_IMG_SAMP_WORD0_FORCE_DEGAMMA_SIZE 1
#define SQ_IMG_SAMP_WORD0_ANISO_BIAS_SIZE 6
#define SQ_IMG_SAMP_WORD0_TRUNC_COORD_SIZE 1
#define SQ_IMG_SAMP_WORD0_DISABLE_CUBE_WRAP_SIZE 1
#define SQ_IMG_SAMP_WORD0_FILTER_MODE_SIZE 2
#define SQ_IMG_SAMP_WORD0_COMPAT_MODE_SIZE 1

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_img_samp_word0_t {
          unsigned int clamp_x                        : SQ_IMG_SAMP_WORD0_CLAMP_X_SIZE;
          unsigned int clamp_y                        : SQ_IMG_SAMP_WORD0_CLAMP_Y_SIZE;
          unsigned int clamp_z                        : SQ_IMG_SAMP_WORD0_CLAMP_Z_SIZE;
          unsigned int max_aniso_ratio                : SQ_IMG_SAMP_WORD0_MAX_ANISO_RATIO_SIZE;
          unsigned int depth_compare_func             : SQ_IMG_SAMP_WORD0_DEPTH_COMPARE_FUNC_SIZE;
          unsigned int force_unnormalized             : SQ_IMG_SAMP_WORD0_FORCE_UNNORMALIZED_SIZE;
          unsigned int aniso_threshold                : SQ_IMG_SAMP_WORD0_ANISO_THRESHOLD_SIZE;
          unsigned int mc_coord_trunc                 : SQ_IMG_SAMP_WORD0_MC_COORD_TRUNC_SIZE;
          unsigned int force_degamma                  : SQ_IMG_SAMP_WORD0_FORCE_DEGAMMA_SIZE;
          unsigned int aniso_bias                     : SQ_IMG_SAMP_WORD0_ANISO_BIAS_SIZE;
          unsigned int trunc_coord                    : SQ_IMG_SAMP_WORD0_TRUNC_COORD_SIZE;
          unsigned int disable_cube_wrap              : SQ_IMG_SAMP_WORD0_DISABLE_CUBE_WRAP_SIZE;
          unsigned int filter_mode                    : SQ_IMG_SAMP_WORD0_FILTER_MODE_SIZE;
          unsigned int compat_mode                    : SQ_IMG_SAMP_WORD0_COMPAT_MODE_SIZE;
     } sq_img_samp_word0_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_img_samp_word0_t {
          unsigned int compat_mode                    : SQ_IMG_SAMP_WORD0_COMPAT_MODE_SIZE;
          unsigned int filter_mode                    : SQ_IMG_SAMP_WORD0_FILTER_MODE_SIZE;
          unsigned int disable_cube_wrap              : SQ_IMG_SAMP_WORD0_DISABLE_CUBE_WRAP_SIZE;
          unsigned int trunc_coord                    : SQ_IMG_SAMP_WORD0_TRUNC_COORD_SIZE;
          unsigned int aniso_bias                     : SQ_IMG_SAMP_WORD0_ANISO_BIAS_SIZE;
          unsigned int force_degamma                  : SQ_IMG_SAMP_WORD0_FORCE_DEGAMMA_SIZE;
          unsigned int mc_coord_trunc                 : SQ_IMG_SAMP_WORD0_MC_COORD_TRUNC_SIZE;
          unsigned int aniso_threshold                : SQ_IMG_SAMP_WORD0_ANISO_THRESHOLD_SIZE;
          unsigned int force_unnormalized             : SQ_IMG_SAMP_WORD0_FORCE_UNNORMALIZED_SIZE;
          unsigned int depth_compare_func             : SQ_IMG_SAMP_WORD0_DEPTH_COMPARE_FUNC_SIZE;
          unsigned int max_aniso_ratio                : SQ_IMG_SAMP_WORD0_MAX_ANISO_RATIO_SIZE;
          unsigned int clamp_z                        : SQ_IMG_SAMP_WORD0_CLAMP_Z_SIZE;
          unsigned int clamp_y                        : SQ_IMG_SAMP_WORD0_CLAMP_Y_SIZE;
          unsigned int clamp_x                        : SQ_IMG_SAMP_WORD0_CLAMP_X_SIZE;
     } sq_img_samp_word0_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_img_samp_word0_t f;
} sq_img_samp_word0_u;

#define SQ_IMG_SAMP_WORD1_REG_SIZE     32
#define SQ_IMG_SAMP_WORD1_MIN_LOD_SIZE 12
#define SQ_IMG_SAMP_WORD1_MAX_LOD_SIZE 12
#define SQ_IMG_SAMP_WORD1_PERF_MIP_SIZE 4
#define SQ_IMG_SAMP_WORD1_PERF_Z_SIZE  4

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_img_samp_word1_t {
          unsigned int min_lod                        : SQ_IMG_SAMP_WORD1_MIN_LOD_SIZE;
          unsigned int max_lod                        : SQ_IMG_SAMP_WORD1_MAX_LOD_SIZE;
          unsigned int perf_mip                       : SQ_IMG_SAMP_WORD1_PERF_MIP_SIZE;
          unsigned int perf_z                         : SQ_IMG_SAMP_WORD1_PERF_Z_SIZE;
     } sq_img_samp_word1_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_img_samp_word1_t {
          unsigned int perf_z                         : SQ_IMG_SAMP_WORD1_PERF_Z_SIZE;
          unsigned int perf_mip                       : SQ_IMG_SAMP_WORD1_PERF_MIP_SIZE;
          unsigned int max_lod                        : SQ_IMG_SAMP_WORD1_MAX_LOD_SIZE;
          unsigned int min_lod                        : SQ_IMG_SAMP_WORD1_MIN_LOD_SIZE;
     } sq_img_samp_word1_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_img_samp_word1_t f;
} sq_img_samp_word1_u;

#define SQ_IMG_SAMP_WORD2_REG_SIZE     32
#define SQ_IMG_SAMP_WORD2_LOD_BIAS_SIZE 14
#define SQ_IMG_SAMP_WORD2_LOD_BIAS_SEC_SIZE 6
#define SQ_IMG_SAMP_WORD2_XY_MAG_FILTER_SIZE 2
#define SQ_IMG_SAMP_WORD2_XY_MIN_FILTER_SIZE 2
#define SQ_IMG_SAMP_WORD2_Z_FILTER_SIZE 2
#define SQ_IMG_SAMP_WORD2_MIP_FILTER_SIZE 2
#define SQ_IMG_SAMP_WORD2_MIP_POINT_PRECLAMP_SIZE 1
#define SQ_IMG_SAMP_WORD2_BLEND_ZERO_PRT_SIZE 1
#define SQ_IMG_SAMP_WORD2_FILTER_PREC_FIX_SIZE 1
#define SQ_IMG_SAMP_WORD2_ANISO_OVERRIDE_SIZE 1

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_img_samp_word2_t {
          unsigned int lod_bias                       : SQ_IMG_SAMP_WORD2_LOD_BIAS_SIZE;
          unsigned int lod_bias_sec                   : SQ_IMG_SAMP_WORD2_LOD_BIAS_SEC_SIZE;
          unsigned int xy_mag_filter                  : SQ_IMG_SAMP_WORD2_XY_MAG_FILTER_SIZE;
          unsigned int xy_min_filter                  : SQ_IMG_SAMP_WORD2_XY_MIN_FILTER_SIZE;
          unsigned int z_filter                       : SQ_IMG_SAMP_WORD2_Z_FILTER_SIZE;
          unsigned int mip_filter                     : SQ_IMG_SAMP_WORD2_MIP_FILTER_SIZE;
          unsigned int mip_point_preclamp             : SQ_IMG_SAMP_WORD2_MIP_POINT_PRECLAMP_SIZE;
          unsigned int blend_zero_prt                 : SQ_IMG_SAMP_WORD2_BLEND_ZERO_PRT_SIZE;
          unsigned int filter_prec_fix                : SQ_IMG_SAMP_WORD2_FILTER_PREC_FIX_SIZE;
          unsigned int aniso_override                 : SQ_IMG_SAMP_WORD2_ANISO_OVERRIDE_SIZE;
     } sq_img_samp_word2_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_img_samp_word2_t {
          unsigned int aniso_override                 : SQ_IMG_SAMP_WORD2_ANISO_OVERRIDE_SIZE;
          unsigned int filter_prec_fix                : SQ_IMG_SAMP_WORD2_FILTER_PREC_FIX_SIZE;
          unsigned int blend_zero_prt                 : SQ_IMG_SAMP_WORD2_BLEND_ZERO_PRT_SIZE;
          unsigned int mip_point_preclamp             : SQ_IMG_SAMP_WORD2_MIP_POINT_PRECLAMP_SIZE;
          unsigned int mip_filter                     : SQ_IMG_SAMP_WORD2_MIP_FILTER_SIZE;
          unsigned int z_filter                       : SQ_IMG_SAMP_WORD2_Z_FILTER_SIZE;
          unsigned int xy_min_filter                  : SQ_IMG_SAMP_WORD2_XY_MIN_FILTER_SIZE;
          unsigned int xy_mag_filter                  : SQ_IMG_SAMP_WORD2_XY_MAG_FILTER_SIZE;
          unsigned int lod_bias_sec                   : SQ_IMG_SAMP_WORD2_LOD_BIAS_SEC_SIZE;
          unsigned int lod_bias                       : SQ_IMG_SAMP_WORD2_LOD_BIAS_SIZE;
     } sq_img_samp_word2_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_img_samp_word2_t f;
} sq_img_samp_word2_u;

#define SQ_IMG_SAMP_WORD3_REG_SIZE     32
#define SQ_IMG_SAMP_WORD3_BORDER_COLOR_PTR_SIZE 12
#define SQ_IMG_SAMP_WORD3_SKIP_DEGAMMA_SIZE 1
#define SQ_IMG_SAMP_WORD3_BORDER_COLOR_TYPE_SIZE 2

#if             defined(LITTLEENDIAN_CPU)

     typedef struct _sq_img_samp_word3_t {
          unsigned int border_color_ptr               : SQ_IMG_SAMP_WORD3_BORDER_COLOR_PTR_SIZE;
          unsigned int skip_degamma                   : SQ_IMG_SAMP_WORD3_SKIP_DEGAMMA_SIZE;
          unsigned int                                : 17;
          unsigned int border_color_type              : SQ_IMG_SAMP_WORD3_BORDER_COLOR_TYPE_SIZE;
     } sq_img_samp_word3_t;

#elif           defined(BIGENDIAN_CPU)

     typedef struct _sq_img_samp_word3_t {
          unsigned int border_color_type              : SQ_IMG_SAMP_WORD3_BORDER_COLOR_TYPE_SIZE;
          unsigned int                                : 17;
          unsigned int skip_degamma                   : SQ_IMG_SAMP_WORD3_SKIP_DEGAMMA_SIZE;
          unsigned int border_color_ptr               : SQ_IMG_SAMP_WORD3_BORDER_COLOR_PTR_SIZE;
     } sq_img_samp_word3_t;

#endif

typedef union {
     unsigned int val : 32;
     sq_img_samp_word3_t f;
} sq_img_samp_word3_u;

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
FMT_RESERVED_4                           = 0x0000000f,
FMT_5_6_5                                = 0x00000010,
FMT_1_5_5_5                              = 0x00000011,
FMT_5_5_5_1                              = 0x00000012,
FMT_4_4_4_4                              = 0x00000013,
FMT_8_24                                 = 0x00000014,
FMT_24_8                                 = 0x00000015,
FMT_X24_8_32_FLOAT                       = 0x00000016,
FMT_RESERVED_33                          = 0x00000017,
FMT_11_11_10_FLOAT                       = 0x00000018,
FMT_16_FLOAT                             = 0x00000019,
FMT_32_FLOAT                             = 0x0000001a,
FMT_16_16_FLOAT                          = 0x0000001b,
FMT_8_24_FLOAT                           = 0x0000001c,
FMT_24_8_FLOAT                           = 0x0000001d,
FMT_32_32_FLOAT                          = 0x0000001e,
FMT_10_11_11_FLOAT                       = 0x0000001f,
FMT_16_16_16_16_FLOAT                    = 0x00000020,
FMT_3_3_2                                = 0x00000021,
FMT_6_5_5                                = 0x00000022,
FMT_32_32_32_32_FLOAT                    = 0x00000023,
FMT_RESERVED_36                          = 0x00000024,
FMT_1                                    = 0x00000025,
FMT_1_REVERSED                           = 0x00000026,
FMT_GB_GR                                = 0x00000027,
FMT_BG_RG                                = 0x00000028,
FMT_32_AS_8                              = 0x00000029,
FMT_32_AS_8_8                            = 0x0000002a,
FMT_5_9_9_9_SHAREDEXP                    = 0x0000002b,
FMT_8_8_8                                = 0x0000002c,
FMT_16_16_16                             = 0x0000002d,
FMT_16_16_16_FLOAT                       = 0x0000002e,
FMT_4_4                                  = 0x0000002f,
FMT_32_32_32_FLOAT                       = 0x00000030,
FMT_BC1                                  = 0x00000031,
FMT_BC2                                  = 0x00000032,
FMT_BC3                                  = 0x00000033,
FMT_BC4                                  = 0x00000034,
FMT_BC5                                  = 0x00000035,
FMT_BC6                                  = 0x00000036,
FMT_BC7                                  = 0x00000037,
FMT_32_AS_32_32_32_32                    = 0x00000038,
FMT_APC3                                 = 0x00000039,
FMT_APC4                                 = 0x0000003a,
FMT_APC5                                 = 0x0000003b,
FMT_APC6                                 = 0x0000003c,
FMT_APC7                                 = 0x0000003d,
FMT_CTX1                                 = 0x0000003e,
FMT_RESERVED_63                          = 0x0000003f,
} FMT;

typedef enum type {
TYPE_UNORM                     = 0x00000000,
TYPE_SNORM                     = 0x00000001,
TYPE_USCALED                   = 0x00000002,
TYPE_SSCALED                   = 0x00000003,
TYPE_UINT                      = 0x00000004,
TYPE_SINT                      = 0x00000005,
TYPE_RESERVED_6                = 0x00000006,
TYPE_FLOAT                     = 0x00000007,
TYPE_RESERVED_8                = 0x00000008,
TYPE_SRGB                      = 0x00000009,
TYPE_UNORM_UINT                = 0x0000000a,
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

typedef enum TEX_BC_SWIZZLE {
TEX_BC_Swizzle_XYZW                      = 0x00000000,
TEX_BC_Swizzle_XWYZ                      = 0x00000001,
TEX_BC_Swizzle_WZYX                      = 0x00000002,
TEX_BC_Swizzle_WXYZ                      = 0x00000003,
TEX_BC_Swizzle_ZYXW                      = 0x00000004,
TEX_BC_Swizzle_YXWZ                      = 0x00000005,
} TEX_BC_SWIZZLE;

typedef struct metadata_amd_ai_s {
    uint32_t version; // Must be 1
    uint32_t vendorID; // AMD
    SQ_IMG_RSRC_WORD0 word0;
    SQ_IMG_RSRC_WORD1 word1;
    SQ_IMG_RSRC_WORD2 word2;
    SQ_IMG_RSRC_WORD3 word3;
    SQ_IMG_RSRC_WORD4 word4;
    SQ_IMG_RSRC_WORD5 word5;
    SQ_IMG_RSRC_WORD6 word6;
    SQ_IMG_RSRC_WORD7 word7;
    uint32_t mip_offsets[0]; //Mip level offset bits [39:8] for each level (if any)
} metadata_amd_ai_t;

}  // namespace image
}  // namespace rocr
#endif  // HSA_RUNTIME_EXT_IMAGE_RESOURCE_AI_H
