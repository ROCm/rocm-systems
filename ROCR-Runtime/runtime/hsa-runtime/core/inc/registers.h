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

// This file is used only for open source cmake builds, if we hardcode the
// register values in amd_aql_queue.cpp then this file won't be required. For
// now we are using this file where register details are  spelled out in the
// structs/unions below.
#ifndef HSA_RUNTME_CORE_INC_REGISTERS_H_
#define HSA_RUNTME_CORE_INC_REGISTERS_H_

typedef enum SQ_RSRC_BUF_TYPE {
SQ_RSRC_BUF                              = 0x00000000,
SQ_RSRC_BUF_RSVD_1                       = 0x00000001,
SQ_RSRC_BUF_RSVD_2                       = 0x00000002,
SQ_RSRC_BUF_RSVD_3                       = 0x00000003,
} SQ_RSRC_BUF_TYPE;

typedef enum BUF_DATA_FORMAT {
BUF_DATA_FORMAT_INVALID                  = 0x00000000,
BUF_DATA_FORMAT_8                        = 0x00000001,
BUF_DATA_FORMAT_16                       = 0x00000002,
BUF_DATA_FORMAT_8_8                      = 0x00000003,
BUF_DATA_FORMAT_32                       = 0x00000004,
BUF_DATA_FORMAT_16_16                    = 0x00000005,
BUF_DATA_FORMAT_10_11_11                 = 0x00000006,
BUF_DATA_FORMAT_11_11_10                 = 0x00000007,
BUF_DATA_FORMAT_10_10_10_2               = 0x00000008,
BUF_DATA_FORMAT_2_10_10_10               = 0x00000009,
BUF_DATA_FORMAT_8_8_8_8                  = 0x0000000a,
BUF_DATA_FORMAT_32_32                    = 0x0000000b,
BUF_DATA_FORMAT_16_16_16_16              = 0x0000000c,
BUF_DATA_FORMAT_32_32_32                 = 0x0000000d,
BUF_DATA_FORMAT_32_32_32_32              = 0x0000000e,
BUF_DATA_FORMAT_RESERVED_15              = 0x0000000f,
} BUF_DATA_FORMAT;

typedef enum BUF_NUM_FORMAT {
BUF_NUM_FORMAT_UNORM                     = 0x00000000,
BUF_NUM_FORMAT_SNORM                     = 0x00000001,
BUF_NUM_FORMAT_USCALED                   = 0x00000002,
BUF_NUM_FORMAT_SSCALED                   = 0x00000003,
BUF_NUM_FORMAT_UINT                      = 0x00000004,
BUF_NUM_FORMAT_SINT                      = 0x00000005,
BUF_NUM_FORMAT_SNORM_OGL__SI__CI         = 0x00000006,
BUF_NUM_FORMAT_RESERVED_6__VI            = 0x00000006,
BUF_NUM_FORMAT_FLOAT                     = 0x00000007,
} BUF_NUM_FORMAT;

typedef enum BUF_FORMAT {
BUF_FORMAT_32_UINT                       = 0x00000014,
} BUF_FORMAT;

typedef enum SQ_SEL_XYZW01 {
SQ_SEL_0                                 = 0x00000000,
SQ_SEL_1                                 = 0x00000001,
SQ_SEL_RESERVED_0                        = 0x00000002,
SQ_SEL_RESERVED_1                        = 0x00000003,
SQ_SEL_X                                 = 0x00000004,
SQ_SEL_Y                                 = 0x00000005,
SQ_SEL_Z                                 = 0x00000006,
SQ_SEL_W                                 = 0x00000007,
} SQ_SEL_XYZW01;

	union COMPUTE_TMPRING_SIZE {
	struct {
#if		defined(LITTLEENDIAN_CPU)
		unsigned int                           WAVES : 12;
		unsigned int                        WAVESIZE : 13;
		unsigned int                                 : 7;
#elif		defined(BIGENDIAN_CPU)
		unsigned int                                 : 7;
		unsigned int                        WAVESIZE : 13;
		unsigned int                           WAVES : 12;
#endif
	} bitfields, bits;
	unsigned int	u32All;
	signed int	i32All;
	float	f32All;
	};

        union COMPUTE_TMPRING_SIZE_GFX11 {
          struct {
#if defined(LITTLEENDIAN_CPU)
            unsigned int WAVES : 12;
            unsigned int WAVESIZE : 15;
            unsigned int : 5;
#elif defined(BIGENDIAN_CPU)
            unsigned int : 5;
            unsigned int WAVESIZE : 15;
            unsigned int WAVES : 12;
#endif
          } bitfields, bits;
          unsigned int u32All;
          signed int i32All;
          float f32All;
        };

        union COMPUTE_TMPRING_SIZE_GFX12 {
          struct {
#if defined(LITTLEENDIAN_CPU)
            unsigned int WAVES : 12;
            unsigned int WAVESIZE : 18;
            unsigned int : 2;
#elif defined(BIGENDIAN_CPU)
            unsigned int : 2;
            unsigned int WAVESIZE : 18;
            unsigned int WAVES : 12;
#endif
          } bitfields, bits;
          unsigned int u32All;
          signed int i32All;
          float f32All;
        };


        union SQ_BUF_RSRC_WORD0 {
	struct {
#if		defined(LITTLEENDIAN_CPU)
		unsigned int                    BASE_ADDRESS : 32;
#elif		defined(BIGENDIAN_CPU)
		unsigned int                    BASE_ADDRESS : 32;
#endif
	} bitfields, bits;
	unsigned int	u32All;
	signed int	i32All;
	float	f32All;
	};


	union SQ_BUF_RSRC_WORD1 {
	struct {
#if		defined(LITTLEENDIAN_CPU)
		unsigned int                 BASE_ADDRESS_HI : 16;
		unsigned int                          STRIDE : 14;
		unsigned int                   CACHE_SWIZZLE : 1;
		unsigned int                  SWIZZLE_ENABLE : 1;
#elif		defined(BIGENDIAN_CPU)
		unsigned int                  SWIZZLE_ENABLE : 1;
		unsigned int                   CACHE_SWIZZLE : 1;
		unsigned int                          STRIDE : 14;
		unsigned int                 BASE_ADDRESS_HI : 16;
#endif
	} bitfields, bits;
	unsigned int	u32All;
	signed int	i32All;
	float	f32All;
	};

        union SQ_BUF_RSRC_WORD1_GFX11 {
          struct {
#if defined(LITTLEENDIAN_CPU)
            unsigned int BASE_ADDRESS_HI : 16;
            unsigned int STRIDE : 14;
            unsigned int SWIZZLE_ENABLE : 2;
#elif defined(BIGENDIAN_CPU)
            unsigned int SWIZZLE_ENABLE : 2;
            unsigned int STRIDE : 14;
            unsigned int BASE_ADDRESS_HI : 16;
#endif
          } bitfields, bits;
          unsigned int u32All;
          signed int i32All;
          float f32All;
        };


        union SQ_BUF_RSRC_WORD2 {
	struct {
#if		defined(LITTLEENDIAN_CPU)
		unsigned int                     NUM_RECORDS : 32;
#elif		defined(BIGENDIAN_CPU)
		unsigned int                     NUM_RECORDS : 32;
#endif
	} bitfields, bits;
	unsigned int	u32All;
	signed int	i32All;
	float	f32All;
	};


	union SQ_BUF_RSRC_WORD3 {
	struct {
#if		defined(LITTLEENDIAN_CPU)
                unsigned int                       DST_SEL_X : 3;
                unsigned int                       DST_SEL_Y : 3;
                unsigned int                       DST_SEL_Z : 3;
                unsigned int                       DST_SEL_W : 3;
                unsigned int                      NUM_FORMAT : 3;
                unsigned int                     DATA_FORMAT : 4;
                unsigned int                    ELEMENT_SIZE : 2;
                unsigned int                    INDEX_STRIDE : 2;
                unsigned int                  ADD_TID_ENABLE : 1;
                unsigned int                     ATC__CI__VI : 1;
                unsigned int                     HASH_ENABLE : 1;
                unsigned int                            HEAP : 1;
                unsigned int                   MTYPE__CI__VI : 3;
                unsigned int                            TYPE : 2;
#elif		defined(BIGENDIAN_CPU)
                unsigned int                            TYPE : 2;
                unsigned int                   MTYPE__CI__VI : 3;
                unsigned int                            HEAP : 1;
                unsigned int                     HASH_ENABLE : 1;
                unsigned int                     ATC__CI__VI : 1;
                unsigned int                  ADD_TID_ENABLE : 1;
                unsigned int                    INDEX_STRIDE : 2;
                unsigned int                    ELEMENT_SIZE : 2;
                unsigned int                     DATA_FORMAT : 4;
                unsigned int                      NUM_FORMAT : 3;
                unsigned int                       DST_SEL_W : 3;
                unsigned int                       DST_SEL_Z : 3;
                unsigned int                       DST_SEL_Y : 3;
                unsigned int                       DST_SEL_X : 3;
#endif
	} bitfields, bits;
	unsigned int	u32All;
	signed int	i32All;
	float	f32All;
	};

	union SQ_BUF_RSRC_WORD3_GFX10 {
	struct {
#if		defined(LITTLEENDIAN_CPU)
                unsigned int                       DST_SEL_X : 3;
                unsigned int                       DST_SEL_Y : 3;
                unsigned int                       DST_SEL_Z : 3;
                unsigned int                       DST_SEL_W : 3;
                unsigned int                          FORMAT : 7;
                unsigned int                       RESERVED1 : 2;
                unsigned int                    INDEX_STRIDE : 2;
                unsigned int                  ADD_TID_ENABLE : 1;
                unsigned int                  RESOURCE_LEVEL : 1;
                unsigned int                       RESERVED2 : 3;
                unsigned int                      OOB_SELECT : 2;
                unsigned int                            TYPE : 2;
#elif		defined(BIGENDIAN_CPU)
                unsigned int                            TYPE : 2;
                unsigned int                      OOB_SELECT : 2;
                unsigned int                       RESERVED2 : 3;
                unsigned int                  RESOURCE_LEVEL : 1;
                unsigned int                  ADD_TID_ENABLE : 1;
                unsigned int                    INDEX_STRIDE : 2;
                unsigned int                       RESERVED1 : 2;
                unsigned int                          FORMAT : 7;
                unsigned int                       DST_SEL_W : 3;
                unsigned int                       DST_SEL_Z : 3;
                unsigned int                       DST_SEL_Y : 3;
                unsigned int                       DST_SEL_X : 3;
#endif
        } bitfields, bits;
        unsigned int u32All;
        signed int i32All;
        float f32All;
        };

        // From V# Table
        union SQ_BUF_RSRC_WORD3_GFX11 {
          struct {
#if defined(LITTLEENDIAN_CPU)
            unsigned int DST_SEL_X : 3;
            unsigned int DST_SEL_Y : 3;
            unsigned int DST_SEL_Z : 3;
            unsigned int DST_SEL_W : 3;
            unsigned int FORMAT : 6;
            unsigned int RESERVED1 : 3;
            unsigned int INDEX_STRIDE : 2;
            unsigned int ADD_TID_ENABLE : 1;
            unsigned int RESERVED2 : 4;
            unsigned int OOB_SELECT : 2;
            unsigned int TYPE : 2;
#elif defined(BIGENDIAN_CPU)
            unsigned int TYPE : 2;
            unsigned int OOB_SELECT : 2;
            unsigned int RESERVED2 : 4;
            unsigned int ADD_TID_ENABLE : 1;
            unsigned int INDEX_STRIDE : 2;
            unsigned int RESERVED1 : 3;
            unsigned int FORMAT : 6;
            unsigned int DST_SEL_W : 3;
            unsigned int DST_SEL_Z : 3;
            unsigned int DST_SEL_Y : 3;
            unsigned int DST_SEL_X : 3;
#endif
          } bitfields, bits;
        unsigned int	u32All;
	signed int	i32All;
	float	f32All;
        };

        // From V# Table
        union SQ_BUF_RSRC_WORD3_GFX12 {
          struct {
#if defined(LITTLEENDIAN_CPU)
            unsigned int DST_SEL_X : 3;
            unsigned int DST_SEL_Y : 3;
            unsigned int DST_SEL_Z : 3;
            unsigned int DST_SEL_W : 3;
            unsigned int FORMAT : 6;
            unsigned int RESERVED1 : 3;
            unsigned int INDEX_STRIDE : 2;
            unsigned int ADD_TID_ENABLE : 1;
            unsigned int WRITE_COMPRESS_ENABLE : 1;
            unsigned int COMPRESSION_EN : 1;
            unsigned int COMPRESSION_ACCESS_MODE : 2;
            unsigned int OOB_SELECT : 2;
            unsigned int TYPE : 2;
#elif defined(BIGENDIAN_CPU)
            unsigned int TYPE : 2;
            unsigned int OOB_SELECT : 2;
            unsigned int COMPRESSION_ACCESS_MODE : 2;
            unsigned int COMPRESSION_EN : 1;
            unsigned int WRITE_COMPRESS_ENABLE : 1;
            unsigned int ADD_TID_ENABLE : 1;
            unsigned int INDEX_STRIDE : 2;
            unsigned int RESERVED1 : 3;
            unsigned int FORMAT : 6;
            unsigned int DST_SEL_W : 3;
            unsigned int DST_SEL_Z : 3;
            unsigned int DST_SEL_Y : 3;
            unsigned int DST_SEL_X : 3;
#endif
          } bitfields, bits;
        unsigned int	u32All;
	signed int	i32All;
	float	f32All;
        };
#endif  // header guard
