/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file rocencode.h
 *   rocEncode -- hardware video encode for AMD GPUs through the VCN engines.
 *
 *   The interface follows the shape of a conventional hardware encode API:
 *   a capability-query chain, an encoder session object, registered input and
 *   bitstream resources, and a submit/lock/unlock pipeline.  Entry points,
 *   structures, enumerations, bitfield widths and reserved array sizes have a
 *   fixed layout on x86-64; the `static_assert` block at the end of this file
 *   pins that down.
 *
 *   Naming conventions used uniformly:
 *     Types and constants are prefixed ROCENC_, functions rocEnc…, the
 *     calling-convention macro is ROCENCAPI, status codes are rocEncStatus.
 *     Device and stream handles are HIP: ROCENC_DEVICE_TYPE_HIP,
 *     ROCENC_HIPSTREAM_PTR, rocEncSetIOHipStreams.
 *
 *   Phase 1 implements the capability-query chain only.  Every other entry
 *   point is declared, wired into ROCENCODE_API_FUNCTION_LIST, and returns
 *   ROCENC_ERR_UNSUPPORTED_DEVICE (or ROCENC_ERR_INVALID_CALL where that is
 *   the better fit).  No entry in the function list is ever NULL.
 */

#ifndef _ROCENCODEAPI_H_
#define _ROCENCODEAPI_H_

#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup ENCODER_STRUCTURE rocEncode data structures
 * @{
 */

#ifdef _WIN32
#define ROCENCAPI     __stdcall
#else
#define ROCENCAPI
#endif

#ifdef _WIN32
typedef RECT ROCENC_RECT;
#else
/**
 * \struct _ROCENC_RECT
 * Defines a rectangle.
 */
typedef struct _ROCENC_RECT
{
    uint32_t left;                                        /**< [in]: X of the upper left corner.      */
    uint32_t top;                                         /**< [in]: Y of the upper left corner.      */
    uint32_t right;                                       /**< [in]: X of the bottom right corner.    */
    uint32_t bottom;                                      /**< [in]: Y of the bottom right corner.    */
} ROCENC_RECT;
#endif /* _WIN32 */

/** @} */ /* End of ROCENC_RECT structure grouping */

typedef void* ROCENC_INPUT_PTR;             /**< rocEncode API input buffer                             */
typedef void* ROCENC_OUTPUT_PTR;            /**< rocEncode API output buffer                            */
typedef void* ROCENC_REGISTERED_PTR;        /**< A resource registered with the rocEncode API           */
typedef void* ROCENC_HIPSTREAM_PTR;         /**< Pointer to a hipStream_t                               */

/* Interface version of this header.  It gates the per-structure _VER values
 * below and is unrelated to the rocEncode *library* version, which lives in
 * the package metadata, not here. */
#define ROCENCAPI_MAJOR_VERSION 12
#define ROCENCAPI_MINOR_VERSION 1

#define ROCENCAPI_VERSION (ROCENCAPI_MAJOR_VERSION | (ROCENCAPI_MINOR_VERSION << 24))

/**
 * Macro to generate the per-structure version for use with the API.
 */
#define ROCENCAPI_STRUCT_VERSION(ver) ((uint32_t)ROCENCAPI_VERSION | ((ver)<<16) | (0x7 << 28))

#define ROCENC_INFINITE_GOPLENGTH  0xffffffff

#define ROCENC_MAX_SEQ_HDR_LEN  (512)

#ifdef __GNUC__
#define ROCENC_DEPRECATED __attribute__ ((deprecated("WILL BE REMOVED IN A FUTURE rocEncode VERSION")))
#else
#define ROCENC_DEPRECATED
#endif

/**
 * \addtogroup ENCODER_STRUCTURE rocEncode data structures
 * @{
 */

/**
 * Encode codecs known to the rocEncode interface.
 * Applications must enumerate rather than hard-code support
 * (rocEncGetEncodeCodecs); a codec listed here may be absent on a given device.
 */
typedef enum _ROCENC_CODEC
{
    ROCENC_CODEC_UNDEFINED              = 0x0,   /**< Undefined. Invalid for encoding. */
    ROCENC_CODEC_H264                   = 0x1,   /**< H.264 / AVC.                     */
    ROCENC_CODEC_HEVC                   = 0x2,   /**< H.265 / HEVC.                    */
    ROCENC_CODEC_AV1                    = 0x3,   /**< AV1.                             */
} ROCENC_CODEC;

/**
 * Encode profiles known to the rocEncode interface.
 * Each value belongs to exactly one codec; enumerate with
 * rocEncGetEncodeProfiles for the device-specific subset.
 */
typedef enum _ROCENC_PROFILE
{
    ROCENC_PROFILE_AUTOSELECT               = 0x0,   /**< Let the library pick the profile from the configuration. Valid for every codec. */

    ROCENC_H264_PROFILE_BASELINE            = 0x100, /**< H.264 Baseline.              */
    ROCENC_H264_PROFILE_MAIN                = 0x101, /**< H.264 Main.                  */
    ROCENC_H264_PROFILE_HIGH                = 0x102, /**< H.264 High.                  */
    ROCENC_H264_PROFILE_HIGH_444            = 0x103, /**< H.264 High 4:4:4 Predictive. */
    ROCENC_H264_PROFILE_STEREO              = 0x104, /**< H.264 Stereo High (MVC).     */
    ROCENC_H264_PROFILE_PROGRESSIVE_HIGH    = 0x105, /**< H.264 Progressive High.      */
    ROCENC_H264_PROFILE_CONSTRAINED_HIGH    = 0x106, /**< H.264 Constrained High.      */

    ROCENC_HEVC_PROFILE_MAIN                = 0x200, /**< HEVC Main.                   */
    ROCENC_HEVC_PROFILE_MAIN10              = 0x201, /**< HEVC Main 10.                */
    ROCENC_HEVC_PROFILE_FREXT               = 0x202, /**< HEVC Main 4:4:4 8-bit and Main 4:4:4 10-bit only. */

    ROCENC_AV1_PROFILE_MAIN                 = 0x300, /**< AV1 Main (profile 0).        */
} ROCENC_PROFILE;

/**
 * Encode presets.  Performance degrades and quality improves moving from P1
 * to P7.
 */
typedef enum _ROCENC_PRESET
{
    ROCENC_PRESET_UNDEFINED             = 0x0,   /**< Undefined. Invalid for encoding. */
    ROCENC_PRESET_P1                    = 0x1,   /**< Fastest, lowest quality.         */
    ROCENC_PRESET_P2                    = 0x2,
    ROCENC_PRESET_P3                    = 0x3,
    ROCENC_PRESET_P4                    = 0x4,   /**< Balanced default.                */
    ROCENC_PRESET_P5                    = 0x5,
    ROCENC_PRESET_P6                    = 0x6,
    ROCENC_PRESET_P7                    = 0x7,   /**< Slowest, highest quality.        */
} ROCENC_PRESET;

/**
 * Input frame encode modes.
 */
typedef enum _ROCENC_PARAMS_FRAME_FIELD_MODE
{
    ROCENC_PARAMS_FRAME_FIELD_MODE_FRAME = 0x01,  /**< Frame mode */
    ROCENC_PARAMS_FRAME_FIELD_MODE_FIELD = 0x02,  /**< Field mode */
    ROCENC_PARAMS_FRAME_FIELD_MODE_MBAFF = 0x03   /**< MB adaptive frame/field */
} ROCENC_PARAMS_FRAME_FIELD_MODE;

/**
 * Rate control modes.
 */
typedef enum _ROCENC_PARAMS_RC_MODE
{
    ROCENC_PARAMS_RC_CONSTQP                = 0x0,       /**< Constant QP mode */
    ROCENC_PARAMS_RC_VBR                    = 0x1,       /**< Variable bitrate mode */
    ROCENC_PARAMS_RC_CBR                    = 0x2,       /**< Constant bitrate mode */
} ROCENC_PARAMS_RC_MODE;

/**
 * Multi pass encoding.
 */
typedef enum _ROCENC_MULTI_PASS
{
    ROCENC_MULTI_PASS_DISABLED              = 0x0,        /**< Single pass */
    ROCENC_TWO_PASS_QUARTER_RESOLUTION      = 0x1,        /**< Two pass, first pass quarter resolution */
    ROCENC_TWO_PASS_FULL_RESOLUTION         = 0x2,        /**< Two pass, first pass full resolution */
} ROCENC_MULTI_PASS;

/**
 * Restore encoder state.
 */
typedef enum _ROCENC_STATE_RESTORE_TYPE
{
    ROCENC_STATE_RESTORE_FULL               = 0x01,      /**< Restore full encoder state */
    ROCENC_STATE_RESTORE_RATE_CONTROL       = 0x02,      /**< Restore only rate control state */
    ROCENC_STATE_RESTORE_ENCODE             = 0x03,      /**< Restore full encoder state except rate control */
} ROCENC_STATE_RESTORE_TYPE;

/**
 * Granularity of the encoded frame output statistics.
 */
typedef enum _ROCENC_OUTPUT_STATS_LEVEL
{
    ROCENC_OUTPUT_STATS_NONE          = 0,             /**< No output stats */
    ROCENC_OUTPUT_STATS_BLOCK_LEVEL   = 1,             /**< Per block (CTB for HEVC, macroblock for H.264, super block for AV1) */
    ROCENC_OUTPUT_STATS_ROW_LEVEL     = 2,             /**< Per row of blocks */
} ROCENC_OUTPUT_STATS_LEVEL;

/**
 * Emphasis levels.
 */
typedef enum _ROCENC_EMPHASIS_MAP_LEVEL
{
    ROCENC_EMPHASIS_MAP_LEVEL_0               = 0x0,       /**< Zero delta QP */
    ROCENC_EMPHASIS_MAP_LEVEL_1               = 0x1,       /**< Very low delta QP */
    ROCENC_EMPHASIS_MAP_LEVEL_2               = 0x2,       /**< Low delta QP */
    ROCENC_EMPHASIS_MAP_LEVEL_3               = 0x3,       /**< Medium delta QP */
    ROCENC_EMPHASIS_MAP_LEVEL_4               = 0x4,       /**< High delta QP */
    ROCENC_EMPHASIS_MAP_LEVEL_5               = 0x5        /**< Very high delta QP */
} ROCENC_EMPHASIS_MAP_LEVEL;

/**
 * QP map mode.
 */
typedef enum _ROCENC_QP_MAP_MODE
{
    ROCENC_QP_MAP_DISABLED               = 0x0,             /**< ROCENC_PIC_PARAMS::qpDeltaMap is ignored. */
    ROCENC_QP_MAP_EMPHASIS               = 0x1,             /**< Treated as an emphasis level map (H.264 only). */
    ROCENC_QP_MAP_DELTA                  = 0x2,             /**< Treated as a QP delta map. */
    ROCENC_QP_MAP                        = 0x3,             /**< Not supported. Treated as an absolute QP map. */
} ROCENC_QP_MAP_MODE;

/**
 * Input picture structure.
 */
typedef enum _ROCENC_PIC_STRUCT
{
    ROCENC_PIC_STRUCT_FRAME             = 0x01,                 /**< Progressive frame */
    ROCENC_PIC_STRUCT_FIELD_TOP_BOTTOM  = 0x02,                 /**< Field encoding, top field first */
    ROCENC_PIC_STRUCT_FIELD_BOTTOM_TOP  = 0x03                  /**< Field encoding, bottom field first */
} ROCENC_PIC_STRUCT;

/**
 * Display picture structure.  Only used to decide the number of clock
 * timestamp sets in the picture timing / time code SEI.
 */
typedef enum _ROCENC_DISPLAY_PIC_STRUCT
{
    ROCENC_PIC_STRUCT_DISPLAY_FRAME             = 0x00,                 /**< Frame */
    ROCENC_PIC_STRUCT_DISPLAY_FIELD_TOP_BOTTOM  = 0x01,                 /**< Top field first */
    ROCENC_PIC_STRUCT_DISPLAY_FIELD_BOTTOM_TOP  = 0x02,                 /**< Bottom field first */
    ROCENC_PIC_STRUCT_DISPLAY_FRAME_DOUBLING    = 0x03,                 /**< Frame doubling */
    ROCENC_PIC_STRUCT_DISPLAY_FRAME_TRIPLING    = 0x04                  /**< Frame tripling */
} ROCENC_DISPLAY_PIC_STRUCT;

/**
 * Input picture type.
 */
typedef enum _ROCENC_PIC_TYPE
{
    ROCENC_PIC_TYPE_P               = 0x0,     /**< Forward predicted */
    ROCENC_PIC_TYPE_B               = 0x01,    /**< Bi-directionally predicted picture */
    ROCENC_PIC_TYPE_I               = 0x02,    /**< Intra predicted picture */
    ROCENC_PIC_TYPE_IDR             = 0x03,    /**< IDR picture */
    ROCENC_PIC_TYPE_BI              = 0x04,    /**< Bi-directionally predicted with only intra MBs */
    ROCENC_PIC_TYPE_SKIPPED         = 0x05,    /**< Picture is skipped */
    ROCENC_PIC_TYPE_INTRA_REFRESH   = 0x06,    /**< First picture in an intra refresh cycle */
    ROCENC_PIC_TYPE_NONREF_P        = 0x07,    /**< Non reference P picture */
    ROCENC_PIC_TYPE_UNKNOWN         = 0xFF     /**< Picture type unknown */
} ROCENC_PIC_TYPE;

/**
 * Motion vector precisions.
 */
typedef enum _ROCENC_MV_PRECISION
{
    ROCENC_MV_PRECISION_DEFAULT     = 0x0,     /**< Driver selects quarter-pel by default */
    ROCENC_MV_PRECISION_FULL_PEL    = 0x01,    /**< Full-pel motion vector precision */
    ROCENC_MV_PRECISION_HALF_PEL    = 0x02,    /**< Half-pel motion vector precision */
    ROCENC_MV_PRECISION_QUARTER_PEL = 0x03     /**< Quarter-pel motion vector precision */
} ROCENC_MV_PRECISION;

/**
 * Input buffer formats.
 */
typedef enum _ROCENC_BUFFER_FORMAT
{
    ROCENC_BUFFER_FORMAT_UNDEFINED                       = 0x00000000,  /**< Undefined buffer format */

    ROCENC_BUFFER_FORMAT_NV12                            = 0x00000001,  /**< Semi-planar YUV, Y plane then interleaved UV plane */
    ROCENC_BUFFER_FORMAT_YV12                            = 0x00000010,  /**< Planar YUV, Y plane then V and U planes */
    ROCENC_BUFFER_FORMAT_IYUV                            = 0x00000100,  /**< Planar YUV, Y plane then U and V planes */
    ROCENC_BUFFER_FORMAT_YUV444                          = 0x00001000,  /**< Planar YUV 4:4:4, Y plane then U and V planes */
    ROCENC_BUFFER_FORMAT_YUV420_10BIT                    = 0x00010000,  /**< 10 bit semi-planar YUV. 2 bytes per sample, data in the MS 10 bits. */
    ROCENC_BUFFER_FORMAT_YUV444_10BIT                    = 0x00100000,  /**< 10 bit planar YUV 4:4:4. 2 bytes per sample, data in the MS 10 bits. */
    ROCENC_BUFFER_FORMAT_ARGB                            = 0x01000000,  /**< 8 bit packed A8R8G8B8, word ordered: B in the low 8 bits, A in the high 8 bits. */
    ROCENC_BUFFER_FORMAT_ARGB10                          = 0x02000000,  /**< 10 bit packed A2R10G10B10, word ordered: B in the low 10 bits, A in the high 2 bits. */
    ROCENC_BUFFER_FORMAT_AYUV                            = 0x04000000,  /**< 8 bit packed A8Y8U8V8, word ordered: V in the low 8 bits, A in the high 8 bits. */
    ROCENC_BUFFER_FORMAT_ABGR                            = 0x10000000,  /**< 8 bit packed A8B8G8R8, word ordered: R in the low 8 bits, A in the high 8 bits. */
    ROCENC_BUFFER_FORMAT_ABGR10                          = 0x20000000,  /**< 10 bit packed A2B10G10R10, word ordered: R in the low 10 bits, A in the high 2 bits. */
    ROCENC_BUFFER_FORMAT_U8                              = 0x40000000,  /**< One-dimensional buffer.  Only valid when registering an output resource. */
} ROCENC_BUFFER_FORMAT;

#define ROCENC_BUFFER_FORMAT_NV12_PL ROCENC_BUFFER_FORMAT_NV12
#define ROCENC_BUFFER_FORMAT_YV12_PL ROCENC_BUFFER_FORMAT_YV12
#define ROCENC_BUFFER_FORMAT_IYUV_PL ROCENC_BUFFER_FORMAT_IYUV
#define ROCENC_BUFFER_FORMAT_YUV444_PL ROCENC_BUFFER_FORMAT_YUV444

/**
 * Encoding levels.
 */
typedef enum _ROCENC_LEVEL
{
    ROCENC_LEVEL_AUTOSELECT         = 0,

    ROCENC_LEVEL_H264_1             = 10,
    ROCENC_LEVEL_H264_1b            = 9,
    ROCENC_LEVEL_H264_11            = 11,
    ROCENC_LEVEL_H264_12            = 12,
    ROCENC_LEVEL_H264_13            = 13,
    ROCENC_LEVEL_H264_2             = 20,
    ROCENC_LEVEL_H264_21            = 21,
    ROCENC_LEVEL_H264_22            = 22,
    ROCENC_LEVEL_H264_3             = 30,
    ROCENC_LEVEL_H264_31            = 31,
    ROCENC_LEVEL_H264_32            = 32,
    ROCENC_LEVEL_H264_4             = 40,
    ROCENC_LEVEL_H264_41            = 41,
    ROCENC_LEVEL_H264_42            = 42,
    ROCENC_LEVEL_H264_5             = 50,
    ROCENC_LEVEL_H264_51            = 51,
    ROCENC_LEVEL_H264_52            = 52,
    ROCENC_LEVEL_H264_60            = 60,
    ROCENC_LEVEL_H264_61            = 61,
    ROCENC_LEVEL_H264_62            = 62,

    ROCENC_LEVEL_HEVC_1             = 30,
    ROCENC_LEVEL_HEVC_2             = 60,
    ROCENC_LEVEL_HEVC_21            = 63,
    ROCENC_LEVEL_HEVC_3             = 90,
    ROCENC_LEVEL_HEVC_31            = 93,
    ROCENC_LEVEL_HEVC_4             = 120,
    ROCENC_LEVEL_HEVC_41            = 123,
    ROCENC_LEVEL_HEVC_5             = 150,
    ROCENC_LEVEL_HEVC_51            = 153,
    ROCENC_LEVEL_HEVC_52            = 156,
    ROCENC_LEVEL_HEVC_6             = 180,
    ROCENC_LEVEL_HEVC_61            = 183,
    ROCENC_LEVEL_HEVC_62            = 186,

    ROCENC_TIER_HEVC_MAIN           = 0,
    ROCENC_TIER_HEVC_HIGH           = 1,

    ROCENC_LEVEL_AV1_2              = 0,
    ROCENC_LEVEL_AV1_21             = 1,
    ROCENC_LEVEL_AV1_22             = 2,
    ROCENC_LEVEL_AV1_23             = 3,
    ROCENC_LEVEL_AV1_3              = 4,
    ROCENC_LEVEL_AV1_31             = 5,
    ROCENC_LEVEL_AV1_32             = 6,
    ROCENC_LEVEL_AV1_33             = 7,
    ROCENC_LEVEL_AV1_4              = 8,
    ROCENC_LEVEL_AV1_41             = 9,
    ROCENC_LEVEL_AV1_42             = 10,
    ROCENC_LEVEL_AV1_43             = 11,
    ROCENC_LEVEL_AV1_5              = 12,
    ROCENC_LEVEL_AV1_51             = 13,
    ROCENC_LEVEL_AV1_52             = 14,
    ROCENC_LEVEL_AV1_53             = 15,
    ROCENC_LEVEL_AV1_6              = 16,
    ROCENC_LEVEL_AV1_61             = 17,
    ROCENC_LEVEL_AV1_62             = 18,
    ROCENC_LEVEL_AV1_63             = 19,
    ROCENC_LEVEL_AV1_7              = 20,
    ROCENC_LEVEL_AV1_71             = 21,
    ROCENC_LEVEL_AV1_72             = 22,
    ROCENC_LEVEL_AV1_73             = 23,
    ROCENC_LEVEL_AV1_AUTOSELECT         ,

    ROCENC_TIER_AV1_0               = 0,
    ROCENC_TIER_AV1_1               = 1
} ROCENC_LEVEL;

/**
 * Error codes.
 */
typedef enum _rocEncStatus
{
    /** The API call returned with no errors. */
    ROCENC_SUCCESS,

    /** No encode capable devices were detected. */
    ROCENC_ERR_NO_ENCODE_DEVICE,

    /** The device passed by the client is not supported. */
    ROCENC_ERR_UNSUPPORTED_DEVICE,

    /** The encoder device supplied by the client is not valid. */
    ROCENC_ERR_INVALID_ENCODERDEVICE,

    /** The device passed to the API call is invalid. */
    ROCENC_ERR_INVALID_DEVICE,

    /** The device is no longer available and needs to be reinitialized.  The
     *  client must destroy the current session, free its buffers and create a
     *  new encoding session. */
    ROCENC_ERR_DEVICE_NOT_EXIST,

    /** One or more of the pointers passed to the API call is invalid. */
    ROCENC_ERR_INVALID_PTR,

    /** The completion event passed in rocEncEncodePicture() is invalid. */
    ROCENC_ERR_INVALID_EVENT,

    /** One or more of the parameters passed to the API call is invalid. */
    ROCENC_ERR_INVALID_PARAM,

    /** An API call was made in the wrong sequence or order. */
    ROCENC_ERR_INVALID_CALL,

    /** The API call could not allocate enough memory. */
    ROCENC_ERR_OUT_OF_MEMORY,

    /** The encoder has not been initialized with rocEncInitializeEncoder(), or
     *  initialization failed. */
    ROCENC_ERR_ENCODER_NOT_INITIALIZED,

    /** An unsupported parameter was passed by the client. */
    ROCENC_ERR_UNSUPPORTED_PARAM,

    /** rocEncLockBitstream() failed to lock the output buffer.  Happens on a
     *  non-blocking lock (doNotWait).  Not fatal; retry after a few ms. */
    ROCENC_ERR_LOCK_BUSY,

    /** The user buffer passed by the client is too small. */
    ROCENC_ERR_NOT_ENOUGH_BUFFER,

    /** An invalid struct version was used by the client. */
    ROCENC_ERR_INVALID_VERSION,

    /** rocEncMapInputResource() failed to map the client provided resource. */
    ROCENC_ERR_MAP_FAILED,

    /** The encoder requires more input buffers to produce an output bitstream.
     *  Not fatal.  A client in synchronous mode must not call
     *  rocEncLockBitstream() on an output buffer whose rocEncEncodePicture()
     *  returned this code; keep submitting input until ROCENC_SUCCESS. */
    ROCENC_ERR_NEED_MORE_INPUT,

    /** The hardware encoder is busy.  Retry after a few milliseconds. */
    ROCENC_ERR_ENCODER_BUSY,

    /** The completion event has not been registered with
     *  rocEncRegisterAsyncEvent(). */
    ROCENC_ERR_EVENT_NOT_REGISTERD,

    /** An unknown internal error has occurred. */
    ROCENC_ERR_GENERIC,

    /** The client is attempting to use a feature not available for the license
     *  type of the current system. */
    ROCENC_ERR_INCOMPATIBLE_CLIENT_KEY,

    /** The client is attempting to use a feature that is not implemented for
     *  the current version. */
    ROCENC_ERR_UNIMPLEMENTED,

    /** rocEncRegisterResource() failed to register the resource. */
    ROCENC_ERR_RESOURCE_REGISTER_FAILED,

    /** Attempt to unregister a resource that was never registered. */
    ROCENC_ERR_RESOURCE_NOT_REGISTERED,

    /** Attempt to unmap a resource that was never mapped. */
    ROCENC_ERR_RESOURCE_NOT_MAPPED,

    /** The encoder requires more output buffers to write an output bitstream.
     *  Not fatal when returned from rocEncRestoreEncoderState(). */
    ROCENC_ERR_NEED_MORE_OUTPUT,

} rocEncStatus;

/**
 * Encode picture flags.
 */
typedef enum _ROCENC_PIC_FLAGS
{
    ROCENC_PIC_FLAG_FORCEINTRA         = 0x1,   /**< Encode the current picture as an intra picture */
    ROCENC_PIC_FLAG_FORCEIDR           = 0x2,   /**< Encode the current picture as an IDR picture.  Only valid when enablePTD == 1. */
    ROCENC_PIC_FLAG_OUTPUT_SPSPPS      = 0x4,   /**< Write the sequence and picture headers for this picture */
    ROCENC_PIC_FLAG_EOS                = 0x8,   /**< End of the input stream */
    ROCENC_PIC_FLAG_DISABLE_ENC_STATE_ADVANCE = 0x10,  /**< Do not advance encoder state during encode */
    ROCENC_PIC_FLAG_OUTPUT_RECON_FRAME        = 0x20,  /**< Write the reconstructed frame */
} ROCENC_PIC_FLAGS;

/**
 * Memory heap used to allocate input and output buffers.
 */
typedef enum _ROCENC_MEMORY_HEAP
{
    ROCENC_MEMORY_HEAP_AUTOSELECT      = 0, /**< Decided by the driver based on usage */
    ROCENC_MEMORY_HEAP_VID             = 1, /**< Local video memory */
    ROCENC_MEMORY_HEAP_SYSMEM_CACHED   = 2, /**< Cached system memory */
    ROCENC_MEMORY_HEAP_SYSMEM_UNCACHED = 3  /**< Uncached system memory */
} ROCENC_MEMORY_HEAP;

/**
 * B-frame used as reference modes.
 */
typedef enum _ROCENC_BFRAME_REF_MODE
{
    ROCENC_BFRAME_REF_MODE_DISABLED = 0x0,          /**< B frame is not used for reference */
    ROCENC_BFRAME_REF_MODE_EACH     = 0x1,          /**< Each B-frame is used for reference */
    ROCENC_BFRAME_REF_MODE_MIDDLE   = 0x2,          /**< Only the middle B-frame is used for reference */
} ROCENC_BFRAME_REF_MODE;

/**
 * H.264 entropy coding modes.
 */
typedef enum _ROCENC_H264_ENTROPY_CODING_MODE
{
    ROCENC_H264_ENTROPY_CODING_MODE_AUTOSELECT = 0x0,   /**< Selected by the driver */
    ROCENC_H264_ENTROPY_CODING_MODE_CABAC      = 0x1,   /**< CABAC */
    ROCENC_H264_ENTROPY_CODING_MODE_CAVLC      = 0x2    /**< CAVLC */
} ROCENC_H264_ENTROPY_CODING_MODE;

/**
 * H.264 specific BDirect modes.
 */
typedef enum _ROCENC_H264_BDIRECT_MODE
{
    ROCENC_H264_BDIRECT_MODE_AUTOSELECT = 0x0,          /**< Selected by the driver */
    ROCENC_H264_BDIRECT_MODE_DISABLE    = 0x1,          /**< Disable BDirect mode */
    ROCENC_H264_BDIRECT_MODE_TEMPORAL   = 0x2,          /**< Temporal BDirect mode */
    ROCENC_H264_BDIRECT_MODE_SPATIAL    = 0x3           /**< Spatial BDirect mode */
} ROCENC_H264_BDIRECT_MODE;

/**
 * H.264 specific FMO usage.
 */
typedef enum _ROCENC_H264_FMO_MODE
{
    ROCENC_H264_FMO_AUTOSELECT          = 0x0,          /**< Selected by the driver */
    ROCENC_H264_FMO_ENABLE              = 0x1,          /**< Enable FMO */
    ROCENC_H264_FMO_DISABLE             = 0x2,          /**< Disable FMO */
} ROCENC_H264_FMO_MODE;

/**
 * H.264 specific adaptive transform modes.
 */
typedef enum _ROCENC_H264_ADAPTIVE_TRANSFORM_MODE
{
    ROCENC_H264_ADAPTIVE_TRANSFORM_AUTOSELECT = 0x0,   /**< Selected by the driver */
    ROCENC_H264_ADAPTIVE_TRANSFORM_DISABLE    = 0x1,   /**< Adaptive transform 8x8 disabled */
    ROCENC_H264_ADAPTIVE_TRANSFORM_ENABLE     = 0x2,   /**< Adaptive transform 8x8 enabled */
} ROCENC_H264_ADAPTIVE_TRANSFORM_MODE;

/**
 * Stereo frame packing modes.
 */
typedef enum _ROCENC_STEREO_PACKING_MODE
{
    ROCENC_STEREO_PACKING_MODE_NONE             = 0x0,  /**< No stereo packing required */
    ROCENC_STEREO_PACKING_MODE_CHECKERBOARD     = 0x1,  /**< Checkerboard */
    ROCENC_STEREO_PACKING_MODE_COLINTERLEAVE    = 0x2,  /**< Column interleave */
    ROCENC_STEREO_PACKING_MODE_ROWINTERLEAVE    = 0x3,  /**< Row interleave */
    ROCENC_STEREO_PACKING_MODE_SIDEBYSIDE       = 0x4,  /**< Side-by-side */
    ROCENC_STEREO_PACKING_MODE_TOPBOTTOM        = 0x5,  /**< Top-bottom */
    ROCENC_STEREO_PACKING_MODE_FRAMESEQ         = 0x6   /**< Frame sequential */
} ROCENC_STEREO_PACKING_MODE;

/**
 * Input resource type.
 */
typedef enum _ROCENC_INPUT_RESOURCE_TYPE
{
    ROCENC_INPUT_RESOURCE_TYPE_DIRECTX          = 0x0,   /**< A DirectX surface.  Declared, not supported. */
    ROCENC_INPUT_RESOURCE_TYPE_HIPDEVICEPTR     = 0x1,   /**< A HIP device pointer surface */
    ROCENC_INPUT_RESOURCE_TYPE_HIPARRAY         = 0x2,   /**< A HIP array surface.  Declared, not supported. */
    ROCENC_INPUT_RESOURCE_TYPE_OPENGL_TEX       = 0x3    /**< An OpenGL texture.  Declared, not supported. */
} ROCENC_INPUT_RESOURCE_TYPE;

/**
 * Buffer usage.
 */
typedef enum _ROCENC_BUFFER_USAGE
{
    ROCENC_INPUT_IMAGE              = 0x0,          /**< Registered surface is used for the input image */
    ROCENC_OUTPUT_MOTION_VECTOR     = 0x1,          /**< Registered surface receives ME-only mode output */
    ROCENC_OUTPUT_BITSTREAM         = 0x2,          /**< Registered surface receives the output bitstream */
    ROCENC_OUTPUT_RECON             = 0x4,          /**< Registered surface receives the reconstructed frame */
} ROCENC_BUFFER_USAGE;

/**
 * Encoder device type.
 */
typedef enum _ROCENC_DEVICE_TYPE
{
    ROCENC_DEVICE_TYPE_DIRECTX          = 0x0,   /**< DirectX device.  Declared; rejected at session open. */
    ROCENC_DEVICE_TYPE_HIP              = 0x1,   /**< HIP device.  The only supported device type. */
    ROCENC_DEVICE_TYPE_OPENGL           = 0x2    /**< OpenGL device.  Declared; rejected at session open. */
} ROCENC_DEVICE_TYPE;

/**
 * Number of reference frames.
 */
typedef enum _ROCENC_NUM_REF_FRAMES
{
    ROCENC_NUM_REF_FRAMES_AUTOSELECT       = 0x0,          /**< Selected by the driver */
    ROCENC_NUM_REF_FRAMES_1                = 0x1,
    ROCENC_NUM_REF_FRAMES_2                = 0x2,
    ROCENC_NUM_REF_FRAMES_3                = 0x3,
    ROCENC_NUM_REF_FRAMES_4                = 0x4,
    ROCENC_NUM_REF_FRAMES_5                = 0x5,
    ROCENC_NUM_REF_FRAMES_6                = 0x6,
    ROCENC_NUM_REF_FRAMES_7                = 0x7
} ROCENC_NUM_REF_FRAMES;

/**
 * Encoder capabilities enumeration.  See docs/caps-mapping.md for how each
 * value is resolved (VA config attribute, AMDGPU_INFO_VIDEO_CAPS field, or a
 * hard 0).
 */
typedef enum _ROCENC_CAPS
{
    /** Maximum number of B-frames supported. */
    ROCENC_CAPS_NUM_MAX_BFRAMES,

    /** Rate control modes supported, as a bitmask of ROCENC_PARAMS_RC_MODE. */
    ROCENC_CAPS_SUPPORTED_RATECONTROL_MODES,

    /** Field mode encoding: 0 none, 1 field, 2 frame and field. */
    ROCENC_CAPS_SUPPORT_FIELD_ENCODING,

    /** Monochrome mode encoding. */
    ROCENC_CAPS_SUPPORT_MONOCHROME,

    /** Flexible macroblock ordering. */
    ROCENC_CAPS_SUPPORT_FMO,

    /** Quarter-pel motion estimation. */
    ROCENC_CAPS_SUPPORT_QPELMV,

    /** H.264 BDirect modes. */
    ROCENC_CAPS_SUPPORT_BDIRECT_MODE,

    /** H.264 CABAC entropy coding. */
    ROCENC_CAPS_SUPPORT_CABAC,

    /** Adaptive transform. */
    ROCENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM,

    /** Multi view coding (MVC). */
    ROCENC_CAPS_SUPPORT_STEREO_MVC,

    /** Maximum number of temporal layers. */
    ROCENC_CAPS_NUM_MAX_TEMPORAL_LAYERS,

    /** Hierarchical P frames. */
    ROCENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES,

    /** Hierarchical B frames. */
    ROCENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES,

    /** Maximum encoding level supported (see ROCENC_LEVEL). */
    ROCENC_CAPS_LEVEL_MAX,

    /** Minimum encoding level supported (see ROCENC_LEVEL). */
    ROCENC_CAPS_LEVEL_MIN,

    /** Separate colour plane encoding. */
    ROCENC_CAPS_SEPARATE_COLOUR_PLANE,

    /** Maximum output width supported. */
    ROCENC_CAPS_WIDTH_MAX,

    /** Maximum output height supported. */
    ROCENC_CAPS_HEIGHT_MAX,

    /** Temporal SVC encoding. */
    ROCENC_CAPS_SUPPORT_TEMPORAL_SVC,

    /** Dynamic encode resolution change. */
    ROCENC_CAPS_SUPPORT_DYN_RES_CHANGE,

    /** Dynamic encode bitrate change. */
    ROCENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE,

    /** Forcing constant QP on the fly. */
    ROCENC_CAPS_SUPPORT_DYN_FORCE_CONSTQP,

    /** Dynamic rate control mode change. */
    ROCENC_CAPS_SUPPORT_DYN_RCMODE_CHANGE,

    /** Subframe readback for slice-based encoding. */
    ROCENC_CAPS_SUPPORT_SUBFRAME_READBACK,

    /** Constrained encoding mode. */
    ROCENC_CAPS_SUPPORT_CONSTRAINED_ENCODING,

    /** Intra refresh mode. */
    ROCENC_CAPS_SUPPORT_INTRA_REFRESH,

    /** Custom VBV buffer size. */
    ROCENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE,

    /** Dynamic slice mode. */
    ROCENC_CAPS_SUPPORT_DYNAMIC_SLICE_MODE,

    /** Reference picture invalidation. */
    ROCENC_CAPS_SUPPORT_REF_PIC_INVALIDATION,

    /** Pre-processing support, as a bitmask. */
    ROCENC_CAPS_PREPROC_SUPPORT,

    /** Asynchronous encode mode. */
    ROCENC_CAPS_ASYNC_ENCODE_SUPPORT,

    /** Maximum macroblocks per frame. */
    ROCENC_CAPS_MB_NUM_MAX,

    /** Maximum aggregate throughput in macroblocks per second. */
    ROCENC_CAPS_MB_PER_SEC_MAX,

    /** YUV 4:4:4 mode encoding. */
    ROCENC_CAPS_SUPPORT_YUV444_ENCODE,

    /** Lossless encoding. */
    ROCENC_CAPS_SUPPORT_LOSSLESS_ENCODE,

    /** Sample adaptive offset. */
    ROCENC_CAPS_SUPPORT_SAO,

    /** Motion estimation only mode: 0 none, 1 I and P, 2 I, P and B. */
    ROCENC_CAPS_SUPPORT_MEONLY_MODE,

    /** Lookahead encoding. */
    ROCENC_CAPS_SUPPORT_LOOKAHEAD,

    /** Temporal AQ encoding. */
    ROCENC_CAPS_SUPPORT_TEMPORAL_AQ,

    /** 10 bit encoding. */
    ROCENC_CAPS_SUPPORT_10BIT_ENCODE,

    /** Maximum number of long term reference frames supported. */
    ROCENC_CAPS_NUM_MAX_LTR_FRAMES,

    /** Weighted prediction. */
    ROCENC_CAPS_SUPPORT_WEIGHTED_PREDICTION,

    /** Residual encoder capacity as a percentage of total capacity.  rocEncode
     *  has no virtualized-capacity query and always reports 100. */
    ROCENC_CAPS_DYNAMIC_QUERY_ENCODER_CAPACITY,

    /** B as reference: 0 no, 1 each B-frame, 2 only middle B-frame. */
    ROCENC_CAPS_SUPPORT_BFRAME_REF_MODE,

    /** Emphasis level map based delta QP computation. */
    ROCENC_CAPS_SUPPORT_EMPHASIS_LEVEL_MAP,

    /** Minimum input width supported. */
    ROCENC_CAPS_WIDTH_MIN,

    /** Minimum input height supported. */
    ROCENC_CAPS_HEIGHT_MIN,

    /** Multiple reference frames. */
    ROCENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES,

    /** HEVC with alpha encoding. */
    ROCENC_CAPS_SUPPORT_ALPHA_LAYER_ENCODING,

    /** Number of encoding engines present on the GPU. */
    ROCENC_CAPS_NUM_ENCODER_ENGINES,

    /** Single slice intra refresh. */
    ROCENC_CAPS_SINGLE_SLICE_INTRA_REFRESH,

    /** Encoding without advancing the encoder state. */
    ROCENC_CAPS_DISABLE_ENC_STATE_ADVANCE,

    /** Reconstructed output surface. */
    ROCENC_CAPS_OUTPUT_RECON_SURFACE,

    /** Encoded frame output stats per block. */
    ROCENC_CAPS_OUTPUT_BLOCK_STATS,

    /** Encoded frame output stats per row. */
    ROCENC_CAPS_OUTPUT_ROW_STATS,

    /** Reserved -- not to be used by clients. */
    ROCENC_CAPS_EXPOSED_COUNT

} ROCENC_CAPS;

/**
 * HEVC CU size.
 */
typedef enum _ROCENC_HEVC_CUSIZE
{
    ROCENC_HEVC_CUSIZE_AUTOSELECT = 0,
    ROCENC_HEVC_CUSIZE_8x8        = 1,
    ROCENC_HEVC_CUSIZE_16x16      = 2,
    ROCENC_HEVC_CUSIZE_32x32      = 3,
    ROCENC_HEVC_CUSIZE_64x64      = 4,
}ROCENC_HEVC_CUSIZE;

/**
 * AV1 partition size.
 */
typedef enum _ROCENC_AV1_PART_SIZE
{
    ROCENC_AV1_PART_SIZE_AUTOSELECT    = 0,
    ROCENC_AV1_PART_SIZE_4x4           = 1,
    ROCENC_AV1_PART_SIZE_8x8           = 2,
    ROCENC_AV1_PART_SIZE_16x16         = 3,
    ROCENC_AV1_PART_SIZE_32x32         = 4,
    ROCENC_AV1_PART_SIZE_64x64         = 5,
}ROCENC_AV1_PART_SIZE;

/**
 * VUI video format.
 */
typedef enum _ROCENC_VUI_VIDEO_FORMAT
{
    ROCENC_VUI_VIDEO_FORMAT_COMPONENT   = 0,
    ROCENC_VUI_VIDEO_FORMAT_PAL         = 1,
    ROCENC_VUI_VIDEO_FORMAT_NTSC        = 2,
    ROCENC_VUI_VIDEO_FORMAT_SECAM       = 3,
    ROCENC_VUI_VIDEO_FORMAT_MAC         = 4,
    ROCENC_VUI_VIDEO_FORMAT_UNSPECIFIED = 5,
}ROCENC_VUI_VIDEO_FORMAT;

/**
 * VUI colour primaries.
 */
typedef enum _ROCENC_VUI_COLOR_PRIMARIES
{
    ROCENC_VUI_COLOR_PRIMARIES_UNDEFINED   = 0,
    ROCENC_VUI_COLOR_PRIMARIES_BT709       = 1,
    ROCENC_VUI_COLOR_PRIMARIES_UNSPECIFIED = 2,
    ROCENC_VUI_COLOR_PRIMARIES_RESERVED    = 3,
    ROCENC_VUI_COLOR_PRIMARIES_BT470M      = 4,
    ROCENC_VUI_COLOR_PRIMARIES_BT470BG     = 5,
    ROCENC_VUI_COLOR_PRIMARIES_SMPTE170M   = 6,
    ROCENC_VUI_COLOR_PRIMARIES_SMPTE240M   = 7,
    ROCENC_VUI_COLOR_PRIMARIES_FILM        = 8,
    ROCENC_VUI_COLOR_PRIMARIES_BT2020      = 9,
    ROCENC_VUI_COLOR_PRIMARIES_SMPTE428    = 10,
    ROCENC_VUI_COLOR_PRIMARIES_SMPTE431    = 11,
    ROCENC_VUI_COLOR_PRIMARIES_SMPTE432    = 12,
    ROCENC_VUI_COLOR_PRIMARIES_JEDEC_P22   = 22,
}ROCENC_VUI_COLOR_PRIMARIES;

/**
 * VUI transfer characteristics.
 */
typedef enum _ROCENC_VUI_TRANSFER_CHARACTERISTIC
{
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_UNDEFINED     = 0,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_BT709         = 1,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_UNSPECIFIED   = 2,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_RESERVED      = 3,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_BT470M        = 4,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_BT470BG       = 5,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_SMPTE170M     = 6,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_SMPTE240M     = 7,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_LINEAR        = 8,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_LOG           = 9,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_LOG_SQRT      = 10,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_IEC61966_2_4  = 11,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_BT1361_ECG    = 12,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_SRGB          = 13,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_BT2020_10     = 14,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_BT2020_12     = 15,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_SMPTE2084     = 16,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_SMPTE428      = 17,
    ROCENC_VUI_TRANSFER_CHARACTERISTIC_ARIB_STD_B67  = 18,
}ROCENC_VUI_TRANSFER_CHARACTERISTIC;

/**
 * VUI matrix coefficients.
 */
typedef enum _ROCENC_VUI_MATRIX_COEFFS
{
    ROCENC_VUI_MATRIX_COEFFS_RGB         = 0,
    ROCENC_VUI_MATRIX_COEFFS_BT709       = 1,
    ROCENC_VUI_MATRIX_COEFFS_UNSPECIFIED = 2,
    ROCENC_VUI_MATRIX_COEFFS_RESERVED    = 3,
    ROCENC_VUI_MATRIX_COEFFS_FCC         = 4,
    ROCENC_VUI_MATRIX_COEFFS_BT470BG     = 5,
    ROCENC_VUI_MATRIX_COEFFS_SMPTE170M   = 6,
    ROCENC_VUI_MATRIX_COEFFS_SMPTE240M   = 7,
    ROCENC_VUI_MATRIX_COEFFS_YCGCO       = 8,
    ROCENC_VUI_MATRIX_COEFFS_BT2020_NCL  = 9,
    ROCENC_VUI_MATRIX_COEFFS_BT2020_CL   = 10,
    ROCENC_VUI_MATRIX_COEFFS_SMPTE2085   = 11,
}ROCENC_VUI_MATRIX_COEFFS;

/**
 * Input struct for querying encoding capabilities.
 */
typedef struct _ROCENC_CAPS_PARAM
{
    uint32_t version;                                  /**< [in]: Struct version. Must be ROCENC_CAPS_PARAM_VER */
    ROCENC_CAPS  capsToQuery;                          /**< [in]: Capability to query. A member of ROCENC_CAPS. */
    uint32_t reserved[62];                             /**< [in]: Reserved and must be set to 0 */
} ROCENC_CAPS_PARAM;

/** ROCENC_CAPS_PARAM struct version. */
#define ROCENC_CAPS_PARAM_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * Restore encoder state parameters.
 */
typedef struct _ROCENC_RESTORE_ENCODER_STATE_PARAMS
{
    uint32_t                  version;                 /**< [in]: Struct version. */
    uint32_t                  bufferIdx;               /**< [in]: State buffer index to restore from. */
    ROCENC_STATE_RESTORE_TYPE state;                   /**< [in]: State type to restore. */
    ROCENC_OUTPUT_PTR         outputBitstream;         /**< [in]: Output buffer pointer, AV1 only. */
    void*                     completionEvent;         /**< [in]: Completion event when async mode is enabled. AV1 only. */
    uint32_t                  reserved1[64];           /**< [in]: Reserved and must be set to 0 */
    void*                     reserved2[64];           /**< [in]: Reserved and must be set to NULL */
} ROCENC_RESTORE_ENCODER_STATE_PARAMS;

/** ROCENC_RESTORE_ENCODER_STATE_PARAMS struct version. */
#define ROCENC_RESTORE_ENCODER_STATE_PARAMS_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * Encoded frame information for every block.
 */
typedef struct _ROCENC_OUTPUT_STATS_BLOCK
{
   uint32_t                 version;                /**< [in]: Struct version */
   uint8_t                  QP;                     /**< [out]: QP of the block */
   uint8_t                  reserved[3];            /**< [in]: Reserved and must be set to 0 */
   uint32_t                 bitcount;               /**< [out]: Bitcount of the block */
   uint32_t                 reserved1[13];          /**< [in]: Reserved and must be set to 0 */
} ROCENC_OUTPUT_STATS_BLOCK;

/** ROCENC_OUTPUT_STATS_BLOCK struct version. */
#define ROCENC_OUTPUT_STATS_BLOCK_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * Encoded frame information for every row.
 */
typedef struct _ROCENC_OUTPUT_STATS_ROW
{
   uint32_t                 version;                /**< [in]: Struct version */
   uint8_t                  QP;                     /**< [out]: QP of the row */
   uint8_t                  reserved[3];            /**< [in]: Reserved and must be set to 0 */
   uint32_t                 bitcount;               /**< [out]: Bitcount of the row */
   uint32_t                 reserved1[13];          /**< [in]: Reserved and must be set to 0 */
} ROCENC_OUTPUT_STATS_ROW;

/** ROCENC_OUTPUT_STATS_ROW struct version. */
#define ROCENC_OUTPUT_STATS_ROW_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * Encoder output parameters.
 */
typedef struct _ROCENC_ENCODE_OUT_PARAMS
{
    uint32_t                  version;                 /**< [out]: Struct version. */
    uint32_t                  bitstreamSizeInBytes;    /**< [out]: Encoded bitstream size in bytes */
    uint32_t                  reserved[62];            /**< [out]: Reserved and must be set to 0 */
} ROCENC_ENCODE_OUT_PARAMS;

/** ROCENC_ENCODE_OUT_PARAMS struct version. */
#define ROCENC_ENCODE_OUT_PARAMS_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * Lookahead picture parameters.
 */
typedef struct _ROCENC_LOOKAHEAD_PIC_PARAMS
{
    uint32_t                  version;                 /**< [in]: Struct version. */
    ROCENC_INPUT_PTR          inputBuffer;             /**< [in]: Input buffer from rocEncCreateInputBuffer() or rocEncMapInputResource(). */
    ROCENC_PIC_TYPE           pictureType;             /**< [in]: Input picture type. Required when enablePTD is 0. */
    uint32_t                  reserved[64];            /**< [in]: Reserved and must be set to 0 */
    void*                     reserved1[64];           /**< [in]: Reserved and must be set to NULL */
} ROCENC_LOOKAHEAD_PIC_PARAMS;

/** ROCENC_LOOKAHEAD_PIC_PARAMS struct version. */
#define ROCENC_LOOKAHEAD_PIC_PARAMS_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * Creation parameters for an input buffer.
 */
typedef struct _ROCENC_CREATE_INPUT_BUFFER
{
    uint32_t                  version;                 /**< [in]: Struct version. Must be ROCENC_CREATE_INPUT_BUFFER_VER */
    uint32_t                  width;                   /**< [in]: Input frame width */
    uint32_t                  height;                  /**< [in]: Input frame height */
    ROCENC_MEMORY_HEAP        memoryHeap;              /**< [in]: Deprecated. Do not use. */
    ROCENC_BUFFER_FORMAT      bufferFmt;               /**< [in]: Input buffer format */
    uint32_t                  reserved;                /**< [in]: Reserved and must be set to 0 */
    ROCENC_INPUT_PTR          inputBuffer;             /**< [out]: Pointer to the input buffer */
    void*                     pSysMemBuffer;           /**< [in]: Pointer to an existing system memory buffer */
    uint32_t                  reserved1[57];           /**< [in]: Reserved and must be set to 0 */
    void*                     reserved2[63];           /**< [in]: Reserved and must be set to NULL */
} ROCENC_CREATE_INPUT_BUFFER;

/** ROCENC_CREATE_INPUT_BUFFER struct version. */
#define ROCENC_CREATE_INPUT_BUFFER_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * Creation parameters for an output bitstream buffer.
 */
typedef struct _ROCENC_CREATE_BITSTREAM_BUFFER
{
    uint32_t              version;                     /**< [in]: Struct version. Must be ROCENC_CREATE_BITSTREAM_BUFFER_VER */
    uint32_t              size;                        /**< [in]: Deprecated. Do not use. */
    ROCENC_MEMORY_HEAP    memoryHeap;                  /**< [in]: Deprecated. Do not use. */
    uint32_t              reserved;                    /**< [in]: Reserved and must be set to 0 */
    ROCENC_OUTPUT_PTR     bitstreamBuffer;             /**< [out]: Pointer to the output bitstream buffer */
    void*                 bitstreamBufferPtr;          /**< [out]: Reserved and should not be used */
    uint32_t              reserved1[58];               /**< [in]: Reserved and should be set to 0 */
    void*                 reserved2[64];               /**< [in]: Reserved and should be set to NULL */
} ROCENC_CREATE_BITSTREAM_BUFFER;

/** ROCENC_CREATE_BITSTREAM_BUFFER struct version. */
#define ROCENC_CREATE_BITSTREAM_BUFFER_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * Structs needed for ME only mode.
 */
typedef struct _ROCENC_MVECTOR
{
    int16_t             mvx;               /**< x component of the MV in quarter-pel units */
    int16_t             mvy;               /**< y component of the MV in quarter-pel units */
} ROCENC_MVECTOR;

/**
 * Motion vector structure per macroblock for H.264 motion estimation.
 */
typedef struct _ROCENC_H264_MV_DATA
{
    ROCENC_MVECTOR      mv[4];             /**< Up to 4 vectors for an 8x8 partition */
    uint8_t             mbType;            /**< 0 (I), 1 (P), 2 (IPCM), 3 (B) */
    uint8_t             partitionType;     /**< 0:16x16, 1:8x8, 2:16x8, 3:8x16 */
    uint16_t            reserved;          /**< Reserved padding for alignment */
    uint32_t            mbCost;
} ROCENC_H264_MV_DATA;

/**
 * Motion vector structure per CU for HEVC motion estimation.
 */
typedef struct _ROCENC_HEVC_MV_DATA
{
    ROCENC_MVECTOR    mv[4];               /**< Up to 4 vectors within a CU */
    uint8_t           cuType;              /**< 0 (I), 1 (P) */
    uint8_t           cuSize;              /**< 0: 8x8, 1: 16x16, 2: 32x32, 3: 64x64 */
    uint8_t           partitionMode;       /**< CU partition mode, 0 (2Nx2N) .. 7 (nRx2N) */
    uint8_t           lastCUInCTB;         /**< Marker separating CUs of the current CTB from the next */
} ROCENC_HEVC_MV_DATA;

/**
 * Creation parameters for the output motion vector buffer in ME only mode.
 */
typedef struct _ROCENC_CREATE_MV_BUFFER
{
    uint32_t            version;           /**< [in]: Struct version. Must be ROCENC_CREATE_MV_BUFFER_VER */
    ROCENC_OUTPUT_PTR   mvBuffer;          /**< [out]: Pointer to the output motion vector buffer */
    uint32_t            reserved1[255];    /**< [in]: Reserved and should be set to 0 */
    void*               reserved2[63];     /**< [in]: Reserved and should be set to NULL */
} ROCENC_CREATE_MV_BUFFER;

/** ROCENC_CREATE_MV_BUFFER struct version. */
#define ROCENC_CREATE_MV_BUFFER_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * QP values for frames.  The fields are uint32_t for layout reasons; treat
 * them as signed where negative QP values are meaningful.
 */
typedef struct _ROCENC_QP
{
    uint32_t        qpInterP;     /**< [in]: QP value for P-frames. */
    uint32_t        qpInterB;     /**< [in]: QP value for B-frames. */
    uint32_t        qpIntra;      /**< [in]: QP value for intra frames. */
} ROCENC_QP;

/**
 * Rate control configuration parameters.
 */
 typedef struct _ROCENC_RC_PARAMS
 {
    uint32_t                        version;
    ROCENC_PARAMS_RC_MODE           rateControlMode;                             /**< [in]: Rate control mode. Check ROCENC_CAPS_SUPPORTED_RATECONTROL_MODES. */
    ROCENC_QP                       constQP;                                     /**< [in]: Initial QP, used for all frames in constant QP mode. */
    uint32_t                        averageBitRate;                              /**< [in]: Average bitrate in bits/sec. */
    uint32_t                        maxBitRate;                                  /**< [in]: Maximum bitrate. Used for VBR, ignored for CBR. */
    uint32_t                        vbvBufferSize;                               /**< [in]: VBV (HRD) buffer size in bits. 0 for the default. */
    uint32_t                        vbvInitialDelay;                             /**< [in]: VBV (HRD) initial delay in bits. 0 for the default. */
    uint32_t                        enableMinQP          :1;                     /**< [in]: Use minQP for rate control. */
    uint32_t                        enableMaxQP          :1;                     /**< [in]: Use maxQP for rate control. */
    uint32_t                        enableInitialRCQP    :1;                     /**< [in]: Use initialRCQP for rate control. */
    uint32_t                        enableAQ             :1;                     /**< [in]: Enable spatial adaptive quantization. */
    uint32_t                        reservedBitField1    :1;                     /**< [in]: Reserved and must be set to 0. */
    uint32_t                        enableLookahead      :1;                     /**< [in]: Enable lookahead of depth lookaheadDepth. */
    uint32_t                        disableIadapt        :1;                     /**< [in]: Disable adaptive I-frame insertion at scene cuts. */
    uint32_t                        disableBadapt        :1;                     /**< [in]: Disable adaptive B-frame decision. */
    uint32_t                        enableTemporalAQ     :1;                     /**< [in]: Enable temporal AQ. */
    uint32_t                        zeroReorderDelay     :1;                     /**< [in]: Zero latency operation (num_reorder_frames = 0). */
    uint32_t                        enableNonRefP        :1;                     /**< [in]: Automatic insertion of non-reference P-frames. */
    uint32_t                        strictGOPTarget      :1;                     /**< [in]: Minimize GOP-to-GOP rate fluctuation. */
    uint32_t                        aqStrength           :4;                     /**< [in]: AQ strength, 1 (low) to 15 (aggressive). 0 selects automatically. */
    uint32_t                        enableExtLookahead   :1;                     /**< [in]: Enable externally driven lookahead (rocEncLookaheadPicture). */
    uint32_t                        reservedBitFields    :15;                    /**< [in]: Reserved bitfields and must be set to 0 */
    ROCENC_QP                       minQP;                                       /**< [in]: Minimum QP. Requires enableMinQP. */
    ROCENC_QP                       maxQP;                                       /**< [in]: Maximum QP. Requires enableMaxQP. */
    ROCENC_QP                       initialRCQP;                                 /**< [in]: Initial QP hint. Requires enableInitialRCQP. */
    uint32_t                        temporallayerIdxMask;                        /**< [in]: Bitmask of temporal layers whose QPs changed. Constant QP mode only. */
    uint8_t                         temporalLayerQP[8];                          /**< [in]: Per temporal layer QP. Constant QP mode only. */
    uint8_t                         targetQuality;                               /**< [in]: Target CQ level for VBR mode (0-51, 0 automatic). */
    uint8_t                         targetQualityLSB;                            /**< [in]: Fractional part of targetQuality (8.8 fixed point). */
    uint16_t                        lookaheadDepth;                              /**< [in]: Lookahead depth, 0 to (31 - number of B frames). */
    uint8_t                         lowDelayKeyFrameScale;                       /**< [in]: Ratio of I frame bits to P frame bits for single frame VBV and CBR. */
    int8_t                          yDcQPIndexOffset;                            /**< [in]: AV1 deltaQ_y_dc. */
    int8_t                          uDcQPIndexOffset;                            /**< [in]: AV1 deltaQ_u_dc. */
    int8_t                          vDcQPIndexOffset;                            /**< [in]: AV1 deltaQ_v_dc (future use). */
    ROCENC_QP_MAP_MODE              qpMapMode;                                   /**< [in]: How to interpret ROCENC_PIC_PARAMS::qpDeltaMap. */
    ROCENC_MULTI_PASS               multiPass;                                   /**< [in]: Enable multi-pass encoding for the given rate control mode. */
    uint32_t                        alphaLayerBitrateRatio;                      /**< [in]: Base:alpha layer bitrate split ratio. Default 15. */
    int8_t                          cbQPIndexOffset;                             /**< [in]: H.264 chroma_qp_index_offset / HEVC pps_cb_qp_offset / AV1 deltaQ_u_ac. */
    int8_t                          crQPIndexOffset;                             /**< [in]: H.264 second_chroma_qp_index_offset / HEVC pps_cr_qp_offset / AV1 deltaQ_v_ac. */
    uint16_t                        reserved2;
    uint32_t                        reserved[4];
 } ROCENC_RC_PARAMS;

/** Macro for constructing the version field of ::_ROCENC_RC_PARAMS */
#define ROCENC_RC_PARAMS_VER ROCENCAPI_STRUCT_VERSION(1)

#define ROCENC_MAX_NUM_CLOCK_TS    3

/**
 * Clock timestamp set parameters.  Populates the H.264 picture timing SEI and
 * the HEVC time code SEI.
 */
typedef struct _ROCENC_CLOCK_TIMESTAMP_SET
{
    uint32_t        countingType            : 1;    /**< [in] counting_type */
    uint32_t        discontinuityFlag       : 1;    /**< [in] discontinuity_flag */
    uint32_t        cntDroppedFrames        : 1;    /**< [in] cnt_dropped_flag */
    uint32_t        nFrames                 : 8;    /**< [in] n_frames */
    uint32_t        secondsValue            : 6;    /**< [in] seconds_value */
    uint32_t        minutesValue            : 6;    /**< [in] minutes_value */
    uint32_t        hoursValue              : 5;    /**< [in] hours_value */
    uint32_t        reserved2               : 4;    /**< [in] Reserved and must be set to 0 */
    uint32_t        timeOffset;                     /**< [in] time_offset_value */
} ROCENC_CLOCK_TIMESTAMP_SET;

typedef struct _ROCENC_TIME_CODE
{
    ROCENC_DISPLAY_PIC_STRUCT       displayPicStruct;                          /**< [in] Display picStruct */
    ROCENC_CLOCK_TIMESTAMP_SET      clockTimestamp[ROCENC_MAX_NUM_CLOCK_TS];   /**< [in] Clock timestamp set */
} ROCENC_TIME_CODE;

/**
 * \struct _ROCENC_CONFIG_H264_VUI_PARAMETERS
 * H.264 video usability information parameters.
 */
typedef struct _ROCENC_CONFIG_H264_VUI_PARAMETERS
{
    uint32_t                            overscanInfoPresentFlag;        /**< [in]: overscanInfo is present */
    uint32_t                            overscanInfo;                   /**< [in]: Overscan info (Annex E). */
    uint32_t                            videoSignalTypePresentFlag;     /**< [in]: videoFormat, videoFullRangeFlag and colourDescriptionPresentFlag are present. */
    ROCENC_VUI_VIDEO_FORMAT             videoFormat;                    /**< [in]: Source video format (Annex E). */
    uint32_t                            videoFullRangeFlag;             /**< [in]: Output range of the luma and chroma samples. */
    uint32_t                            colourDescriptionPresentFlag;   /**< [in]: colourPrimaries, transferCharacteristics and colourMatrix are present. */
    ROCENC_VUI_COLOR_PRIMARIES          colourPrimaries;                /**< [in]: Colour primaries. */
    ROCENC_VUI_TRANSFER_CHARACTERISTIC  transferCharacteristics;        /**< [in]: Opto-electronic transfer characteristics. */
    ROCENC_VUI_MATRIX_COEFFS            colourMatrix;                   /**< [in]: Matrix coefficients. */
    uint32_t                            chromaSampleLocationFlag;       /**< [in]: chromaSampleLocationTop and chromaSampleLocationBot are present. */
    uint32_t                            chromaSampleLocationTop;        /**< [in]: Chroma sample location for the top field. */
    uint32_t                            chromaSampleLocationBot;        /**< [in]: Chroma sample location for the bottom field. */
    uint32_t                            bitstreamRestrictionFlag;       /**< [in]: Bitstream restriction parameters are present. */
    uint32_t                            timingInfoPresentFlag;          /**< [in]: numUnitInTicks and timeScale are supplied by the application. */
    uint32_t                            numUnitInTicks;                 /**< [in]: Number of time units of the clock. */
    uint32_t                            timeScale;                      /**< [in]: Frequency of the clock. */
    uint32_t                            reserved[12];                   /**< [in]: Reserved and must be set to 0 */
}ROCENC_CONFIG_H264_VUI_PARAMETERS;

typedef ROCENC_CONFIG_H264_VUI_PARAMETERS ROCENC_CONFIG_HEVC_VUI_PARAMETERS;

/**
 * \struct _ROCENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE
 * External motion vector hint counts per block type.
 */
typedef struct _ROCENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE
{
    uint32_t   numCandsPerBlk16x16                   : 4;   /**< [in]: H.264, HEVC. Candidates per 16x16 block. */
    uint32_t   numCandsPerBlk16x8                    : 4;   /**< [in]: H.264 only. Candidates per 16x8 block. */
    uint32_t   numCandsPerBlk8x16                    : 4;   /**< [in]: H.264 only. Candidates per 8x16 block. */
    uint32_t   numCandsPerBlk8x8                     : 4;   /**< [in]: H.264, HEVC. Candidates per 8x8 block. */
    uint32_t   numCandsPerSb                         : 8;   /**< [in]: AV1 only. Candidates per SB. */
    uint32_t   reserved                              : 8;   /**< [in]: Reserved for padding. */
    uint32_t   reserved1[3];                                /**< [in]: Reserved for future use. */
} ROCENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE;

/**
 * \struct _ROCENC_EXTERNAL_ME_HINT
 * External motion vector hint for H.264 and HEVC.
 */
typedef struct _ROCENC_EXTERNAL_ME_HINT
{
    int32_t    mvx         : 12;                        /**< [in]: x component of the integer pixel MV, S12.0. */
    int32_t    mvy         : 10;                        /**< [in]: y component of the integer pixel MV, S10.0. */
    int32_t    refidx      : 5;                         /**< [in]: Reference index (31 = invalid). Must be 0. */
    int32_t    dir         : 1;                         /**< [in]: Direction. 0 = L0, 1 = L1. */
    int32_t    partType    : 2;                         /**< [in]: Block partition type. 0=16x16 1=16x8 2=8x16 3=8x8. */
    int32_t    lastofPart  : 1;                         /**< [in]: Set to 1 for the last MV of a (sub) partition. */
    int32_t    lastOfMB    : 1;                         /**< [in]: Set to 1 for the last MV of a macroblock. */
} ROCENC_EXTERNAL_ME_HINT;

/**
 * \struct _ROCENC_EXTERNAL_ME_SB_HINT
 * External motion vector SB hint for AV1.
 */
typedef struct _ROCENC_EXTERNAL_ME_SB_HINT
{
    int16_t    refidx         : 5;                      /**< [in]: Reference index (31 = invalid). */
    int16_t    direction      : 1;                      /**< [in]: Direction. 0 = L0, 1 = L1. */
    int16_t    bi             : 1;                      /**< [in]: Reference mode. 0 = single mv, 1 = compound mv. */
    int16_t    partition_type : 3;                      /**< [in]: Partition type. 0: 2Nx2N, 1: 2NxN, 2: Nx2N. */
    int16_t    x8             : 3;                      /**< [in]: Partition top left x position in 8 pixel units. */
    int16_t    last_of_cu     : 1;                      /**< [in]: Set to 1 for the last MV of the current CU. */
    int16_t    last_of_sb     : 1;                      /**< [in]: Set to 1 for the last MV of the current SB. */
    int16_t    reserved0      : 1;                      /**< [in]: Reserved and must be set to 0 */
    int16_t    mvx            : 14;                     /**< [in]: x component of the integer pixel MV, S12.2. */
    int16_t    cu_size        : 2;                      /**< [in]: CU size. 0: 8x8, 1: 16x16, 2: 32x32, 3: 64x64. */
    int16_t    mvy            : 12;                     /**< [in]: y component of the integer pixel MV, S10.2. */
    int16_t    y8             : 3;                      /**< [in]: Partition top left y position in 8 pixel units. */
    int16_t    reserved1      : 1;                      /**< [in]: Reserved and must be set to 0 */
} ROCENC_EXTERNAL_ME_SB_HINT;

/**
 * \struct _ROCENC_CONFIG_H264
 * H.264 encoder configuration parameters.
 */
typedef struct _ROCENC_CONFIG_H264
{
    uint32_t enableTemporalSVC         :1;                          /**< [in]: Enable temporal SVC. */
    uint32_t enableStereoMVC           :1;                          /**< [in]: Enable stereo MVC. */
    uint32_t hierarchicalPFrames       :1;                          /**< [in]: Enable hierarchical P frames. */
    uint32_t hierarchicalBFrames       :1;                          /**< [in]: Enable hierarchical B frames. */
    uint32_t outputBufferingPeriodSEI  :1;                          /**< [in]: Write the buffering period SEI. */
    uint32_t outputPictureTimingSEI    :1;                          /**< [in]: Write the picture timing SEI. */
    uint32_t outputAUD                 :1;                          /**< [in]: Write access unit delimiters. */
    uint32_t disableSPSPPS             :1;                          /**< [in]: Do not write SPS/PPS into the bitstream. */
    uint32_t outputFramePackingSEI     :1;                          /**< [in]: Write the frame packing arrangement SEI. */
    uint32_t outputRecoveryPointSEI    :1;                          /**< [in]: Write the recovery point SEI. */
    uint32_t enableIntraRefresh        :1;                          /**< [in]: Enable gradual decoder refresh. Ignored when B frames are used. */
    uint32_t enableConstrainedEncoding :1;                          /**< [in]: Each slice independent of the others. Check ROCENC_CAPS_SUPPORT_CONSTRAINED_ENCODING. */
    uint32_t repeatSPSPPS              :1;                          /**< [in]: Write SPS/PPS for every IDR frame. */
    uint32_t enableVFR                 :1;                          /**< [in]: Sets fixed_frame_rate_flag = 0 in the VUI only. */
    uint32_t enableLTR                 :1;                          /**< [in]: Enable long term reference frame support. Not supported with B frames. */
    uint32_t qpPrimeYZeroTransformBypassFlag :1;                    /**< [in]: Lossless encode. Requires QP 0, CONSTQP and a HIGH_444 profile. */
    uint32_t useConstrainedIntraPred   :1;                          /**< [in]: Enable constrained intra prediction. */
    uint32_t enableFillerDataInsertion :1;                          /**< [in]: Insert filler data. CBR with a non-zero frame rate only. */
    uint32_t disableSVCPrefixNalu      :1;                          /**< [in]: Do not write the SVC prefix NALU before each slice. */
    uint32_t enableScalabilityInfoSEI  :1;                          /**< [in]: Write the scalability information SEI before each IDR. */
    uint32_t singleSliceIntraRefresh   :1;                          /**< [in]: Maintain a single slice during intra refresh. */
    uint32_t enableTimeCode            :1;                          /**< [in]: Write clock timestamp sets in the picture timing SEI. */
    uint32_t reservedBitFields         :10;                         /**< [in]: Reserved bitfields and must be set to 0 */
    uint32_t level;                                                 /**< [in]: Encoding level. ROCENC_LEVEL_AUTOSELECT is recommended. */
    uint32_t idrPeriod;                                             /**< [in]: IDR interval. Defaults to gopLength. */
    uint32_t separateColourPlaneFlag;                               /**< [in]: Enable 4:4:4 separate colour planes. */
    uint32_t disableDeblockingFilterIDC;                            /**< [in]: disable_deblocking_filter_idc, range [0,2]. */
    uint32_t numTemporalLayers;                                     /**< [in]: Number of temporal layers, [1, ROCENC_CAPS_NUM_MAX_TEMPORAL_LAYERS]. */
    uint32_t spsId;                                                 /**< [in]: SPS id of the sequence header. */
    uint32_t ppsId;                                                 /**< [in]: PPS id of the picture header. */
    ROCENC_H264_ADAPTIVE_TRANSFORM_MODE adaptiveTransformMode;      /**< [in]: Adaptive transform mode. */
    ROCENC_H264_FMO_MODE                fmoMode;                    /**< [in]: FMO mode. */
    ROCENC_H264_BDIRECT_MODE            bdirectMode;                /**< [in]: BDirect mode. */
    ROCENC_H264_ENTROPY_CODING_MODE     entropyCodingMode;          /**< [in]: Entropy coding mode. */
    ROCENC_STEREO_PACKING_MODE          stereoMode;                 /**< [in]: Stereo frame packing mode signalled in the SEI. */
    uint32_t                            intraRefreshPeriod;         /**< [in]: Interval between successive intra refreshes. */
    uint32_t                            intraRefreshCnt;            /**< [in]: Length of an intra refresh in frames. */
    uint32_t                            maxNumRefFrames;            /**< [in]: DPB size used for encoding. 0 for the driver default. */
    uint32_t                            sliceMode;                  /**< [in]: 0 MB based, 1 byte based, 2 MB row based, 3 numSlices in picture. */
    uint32_t                            sliceModeData;              /**< [in]: Parameter for sliceMode. */
    ROCENC_CONFIG_H264_VUI_PARAMETERS   h264VUIParameters;          /**< [in]: H.264 video usability info parameters. */
    uint32_t                            ltrNumFrames;               /**< [in]: Number of LTR frames. Meaning depends on ltrTrustMode. */
    uint32_t                            ltrTrustMode;               /**< [in]: 1 for LTR Trust mode, 0 for LTR Per Picture mode. */
    uint32_t                            chromaFormatIDC;            /**< [in]: 1 for yuv420 input, 3 for yuv444 input. */
    uint32_t                            maxTemporalLayers;          /**< [in]: Max temporal layer for temporal SVC / hierarchical coding. */
    ROCENC_BFRAME_REF_MODE              useBFramesAsRef;            /**< [in]: B-frame as reference mode. */
    ROCENC_NUM_REF_FRAMES               numRefL0;                   /**< [in]: Max reference frames in L0. */
    ROCENC_NUM_REF_FRAMES               numRefL1;                   /**< [in]: Max reference frames in L1. */

    uint32_t                            reserved1[267];             /**< [in]: Reserved and must be set to 0 */
    void*                               reserved2[64];              /**< [in]: Reserved and must be set to NULL */
} ROCENC_CONFIG_H264;

/**
 * \struct _ROCENC_CONFIG_HEVC
 * HEVC encoder configuration parameters.
 */
typedef struct _ROCENC_CONFIG_HEVC
{
    uint32_t level;                                                 /**< [in]: Level of the encoded bitstream. */
    uint32_t tier;                                                  /**< [in]: Level tier of the encoded bitstream. */
    ROCENC_HEVC_CUSIZE minCUSize;                                   /**< [in]: Minimum size of a luma coding unit. */
    ROCENC_HEVC_CUSIZE maxCUSize;                                   /**< [in]: Maximum size of a luma coding unit. */
    uint32_t useConstrainedIntraPred               :1;              /**< [in]: Enable constrained intra prediction. */
    uint32_t disableDeblockAcrossSliceBoundary     :1;              /**< [in]: Disable in-loop filtering across slice boundaries. */
    uint32_t outputBufferingPeriodSEI              :1;              /**< [in]: Write the buffering period SEI. */
    uint32_t outputPictureTimingSEI                :1;              /**< [in]: Write the picture timing SEI. */
    uint32_t outputAUD                             :1;              /**< [in]: Write access unit delimiters. */
    uint32_t enableLTR                             :1;              /**< [in]: Enable long term reference frame support. */
    uint32_t disableSPSPPS                         :1;              /**< [in]: Do not signal VPS, SPS and PPS. */
    uint32_t repeatSPSPPS                          :1;              /**< [in]: Output VPS, SPS and PPS for every IDR frame. */
    uint32_t enableIntraRefresh                    :1;              /**< [in]: Enable gradual decoder refresh. */
    uint32_t chromaFormatIDC                       :2;              /**< [in]: 1 for yuv420 input, 3 for yuv444 input. */
    uint32_t pixelBitDepthMinus8                   :3;              /**< [in]: 0 for 8 bit input, 2 for 10 bit input. */
    uint32_t enableFillerDataInsertion             :1;              /**< [in]: Insert filler data. CBR with a non-zero frame rate only. */
    uint32_t enableConstrainedEncoding             :1;              /**< [in]: Each slice independent of the others. */
    uint32_t enableAlphaLayerEncoding              :1;              /**< [in]: Enable HEVC encode with an alpha layer. */
    uint32_t singleSliceIntraRefresh               :1;              /**< [in]: Maintain single slice frames during intra refresh. */
    uint32_t outputRecoveryPointSEI                :1;              /**< [in]: Write the recovery point SEI. */
    uint32_t outputTimeCodeSEI                     :1;              /**< [in]: Write the time code SEI. */
    uint32_t reserved                              :12;             /**< [in]: Reserved bitfields. */
    uint32_t idrPeriod;                                             /**< [in]: IDR interval. Defaults to gopLength. */
    uint32_t intraRefreshPeriod;                                    /**< [in]: Interval between successive intra refreshes. */
    uint32_t intraRefreshCnt;                                       /**< [in]: Length of an intra refresh in frames. */
    uint32_t maxNumRefFramesInDPB;                                  /**< [in]: Maximum number of reference frames in the DPB. */
    uint32_t ltrNumFrames;                                          /**< [in]: Number of LTR frames. Meaning depends on ltrTrustMode. */
    uint32_t vpsId;                                                 /**< [in]: VPS id of the video parameter set. */
    uint32_t spsId;                                                 /**< [in]: SPS id of the sequence header. */
    uint32_t ppsId;                                                 /**< [in]: PPS id of the picture header. */
    uint32_t sliceMode;                                             /**< [in]: 0 CTU based, 1 byte based, 2 CTU row based, 3 numSlices in picture. */
    uint32_t sliceModeData;                                         /**< [in]: Parameter for sliceMode. */
    uint32_t maxTemporalLayersMinus1;                               /**< [in]: Max temporal layer used for hierarchical coding. */
    ROCENC_CONFIG_HEVC_VUI_PARAMETERS   hevcVUIParameters;          /**< [in]: HEVC video usability info parameters. */
    uint32_t ltrTrustMode;                                          /**< [in]: 1 for LTR Trust mode, 0 for LTR Per Picture mode. */
    ROCENC_BFRAME_REF_MODE              useBFramesAsRef;            /**< [in]: B-frame as reference mode. */
    ROCENC_NUM_REF_FRAMES               numRefL0;                   /**< [in]: Max reference frames in L0. */
    ROCENC_NUM_REF_FRAMES               numRefL1;                   /**< [in]: Max reference frames in L1. */
    uint32_t                            reserved1[214];             /**< [in]: Reserved and must be set to 0. */
    void*                               reserved2[64];              /**< [in]: Reserved and must be set to NULL */
} ROCENC_CONFIG_HEVC;

#define ROCENC_MAX_TILE_COLS_AV1               64
#define ROCENC_MAX_TILE_ROWS_AV1               64

/**
 * \struct _ROCENC_FILM_GRAIN_PARAMS_AV1
 * AV1 film grain parameters.
 */
typedef struct _ROCENC_FILM_GRAIN_PARAMS_AV1
{
    uint32_t applyGrain                 :1;                         /**< [in]: Add film grain to the frame. */
    uint32_t chromaScalingFromLuma      :1;                         /**< [in]: Infer chroma scaling from luma scaling. */
    uint32_t overlapFlag                :1;                         /**< [in]: Apply overlap between film grain blocks. */
    uint32_t clipToRestrictedRange      :1;                         /**< [in]: Clip to studio range after adding film grain. */
    uint32_t grainScalingMinus8         :2;                         /**< [in]: Shift - 8 applied to the chroma component values. */
    uint32_t arCoeffLag                 :2;                         /**< [in]: Number of auto-regressive coefficients. */
    uint32_t numYPoints                 :4;                         /**< [in]: Points in the luma piecewise linear scaling function. */
    uint32_t numCbPoints                :4;                         /**< [in]: Points in the cb piecewise linear scaling function. */
    uint32_t numCrPoints                :4;                         /**< [in]: Points in the cr piecewise linear scaling function. */
    uint32_t arCoeffShiftMinus6         :2;                         /**< [in]: Range of the auto-regressive coefficients. */
    uint32_t grainScaleShift            :2;                         /**< [in]: Downscale applied to the Gaussian random numbers. */
    uint32_t reserved1                  :8;                         /**< [in]: Reserved bits field - should be set to 0 */
    uint8_t  pointYValue[14];                                       /**< [in]: x coordinates of the luma scaling function points. */
    uint8_t  pointYScaling[14];                                     /**< [in]: Output values of the luma scaling function points. */
    uint8_t  pointCbValue[10];                                      /**< [in]: x coordinates of the cb scaling function points. */
    uint8_t  pointCbScaling[10];                                    /**< [in]: Output values of the cb scaling function points. */
    uint8_t  pointCrValue[10];                                      /**< [in]: x coordinates of the cr scaling function points. */
    uint8_t  pointCrScaling[10];                                    /**< [in]: Output values of the cr scaling function points. */
    uint8_t  arCoeffsYPlus128[24];                                  /**< [in]: Auto-regressive coefficients for the Y plane. */
    uint8_t  arCoeffsCbPlus128[25];                                 /**< [in]: Auto-regressive coefficients for the U plane. */
    uint8_t  arCoeffsCrPlus128[25];                                 /**< [in]: Auto-regressive coefficients for the V plane. */
    uint8_t  reserved2[2];                                          /**< [in]: Reserved bytes - should be set to 0 */
    uint8_t  cbMult;                                                /**< [in]: Multiplier for the cb component. */
    uint8_t  cbLumaMult;                                            /**< [in]: Multiplier for the average luma component in the cb index. */
    uint16_t cbOffset;                                              /**< [in]: Offset used in the cb scaling function index. */
    uint8_t  crMult;                                                /**< [in]: Multiplier for the cr component. */
    uint8_t  crLumaMult;                                            /**< [in]: Multiplier for the average luma component in the cr index. */
    uint16_t crOffset;                                              /**< [in]: Offset used in the cr scaling function index. */
} ROCENC_FILM_GRAIN_PARAMS_AV1;

/**
 * \struct _ROCENC_CONFIG_AV1
 * AV1 encoder configuration parameters.
 */
typedef struct _ROCENC_CONFIG_AV1
{
    uint32_t level;                                                 /**< [in]: Level of the encoded bitstream. */
    uint32_t tier;                                                  /**< [in]: Level tier of the encoded bitstream. */
    ROCENC_AV1_PART_SIZE minPartSize;                               /**< [in]: Minimum size of a luma coding block partition. */
    ROCENC_AV1_PART_SIZE maxPartSize;                               /**< [in]: Maximum size of a luma coding block partition. */
    uint32_t outputAnnexBFormat             : 1;                    /**< [in]: Use Annex B format for the bitstream output. */
    uint32_t enableTimingInfo               : 1;                    /**< [in]: Write timing info into sequence/frame headers. */
    uint32_t enableDecoderModelInfo         : 1;                    /**< [in]: Write decoder model info into sequence/frame headers. */
    uint32_t enableFrameIdNumbers           : 1;                    /**< [in]: Write frame id numbers in the bitstream. */
    uint32_t disableSeqHdr                  : 1;                    /**< [in]: Do not signal the sequence header. */
    uint32_t repeatSeqHdr                   : 1;                    /**< [in]: Output the sequence header for every key frame. */
    uint32_t enableIntraRefresh             : 1;                    /**< [in]: Enable gradual decoder refresh. */
    uint32_t chromaFormatIDC                : 2;                    /**< [in]: 1 for yuv420 input (yuv444 not supported). */
    uint32_t enableBitstreamPadding         : 1;                    /**< [in]: Enable bitstream padding. */
    uint32_t enableCustomTileConfig         : 1;                    /**< [in]: Enable the custom tile configuration. */
    uint32_t enableFilmGrainParams          : 1;                    /**< [in]: Enable custom film grain parameters. */
    uint32_t inputPixelBitDepthMinus8       : 3;                    /**< [in]: Input pixel bit depth minus 8. */
    uint32_t pixelBitDepthMinus8            : 3;                    /**< [in]: Encoded pixel bit depth minus 8. */
    uint32_t reserved                       : 14;                   /**< [in]: Reserved bitfields. */
    uint32_t idrPeriod;                                             /**< [in]: IDR / key frame interval. Defaults to gopLength. */
    uint32_t intraRefreshPeriod;                                    /**< [in]: Interval between successive intra refreshes. */
    uint32_t intraRefreshCnt;                                       /**< [in]: Length of an intra refresh in frames. */
    uint32_t maxNumRefFramesInDPB;                                  /**< [in]: Maximum number of reference frames in the DPB. */
    uint32_t numTileColumns;                                        /**< [in]: Number of tile columns. See enableCustomTileConfig. */
    uint32_t numTileRows;                                           /**< [in]: Number of tile rows. See enableCustomTileConfig. */
    uint32_t *tileWidths;                                           /**< [in]: Width of each tile column in 64x64 CTU units. */
    uint32_t *tileHeights;                                          /**< [in]: Height of each tile row in 64x64 CTU units. */
    uint32_t maxTemporalLayersMinus1;                               /**< [in]: Max temporal layer used for hierarchical coding. */
    ROCENC_VUI_COLOR_PRIMARIES colorPrimaries;                      /**< [in]: As defined in ISO/IEC 23091-4 / ITU-T H.273. */
    ROCENC_VUI_TRANSFER_CHARACTERISTIC transferCharacteristics;     /**< [in]: As defined in ISO/IEC 23091-4 / ITU-T H.273. */
    ROCENC_VUI_MATRIX_COEFFS matrixCoefficients;                    /**< [in]: As defined in ISO/IEC 23091-4 / ITU-T H.273. */
    uint32_t colorRange;                                            /**< [in]: 0 studio swing, 1 full swing. */
    uint32_t chromaSamplePosition;                                  /**< [in]: 0 unknown, 1 horizontally collocated, 2 co-located. */
    ROCENC_BFRAME_REF_MODE useBFramesAsRef;                         /**< [in]: B-frame as reference mode. */
    ROCENC_FILM_GRAIN_PARAMS_AV1 *filmGrainParams;                  /**< [in]: Film grain parameters when enableFilmGrainParams is 1. */
    ROCENC_NUM_REF_FRAMES  numFwdRefs;                              /**< [in]: Max forward reference frames, range 1-4. Advisory. */
    ROCENC_NUM_REF_FRAMES  numBwdRefs;                              /**< [in]: Max L1 list reference frames, range 1-3. Advisory. */
    uint32_t reserved1[235];                                        /**< [in]: Reserved and must be set to 0. */
    void*    reserved2[62];                                         /**< [in]: Reserved and must be set to NULL */
} ROCENC_CONFIG_AV1;

/**
 * \struct _ROCENC_CONFIG_H264_MEONLY
 * H.264 encoder configuration parameters for ME only mode.
 */
typedef struct _ROCENC_CONFIG_H264_MEONLY
{
    uint32_t disablePartition16x16 :1;                          /**< [in]: Disable ME on 16x16 blocks. */
    uint32_t disablePartition8x16  :1;                          /**< [in]: Disable ME on 8x16 blocks. */
    uint32_t disablePartition16x8  :1;                          /**< [in]: Disable ME on 16x8 blocks. */
    uint32_t disablePartition8x8   :1;                          /**< [in]: Disable ME on 8x8 blocks. */
    uint32_t disableIntraSearch    :1;                          /**< [in]: Disable intra search during ME. */
    uint32_t bStereoEnable         :1;                          /**< [in]: Enable stereo mode for ME. */
    uint32_t reserved              :26;                         /**< [in]: Reserved and must be set to 0 */
    uint32_t reserved1 [255];                                   /**< [in]: Reserved and must be set to 0 */
    void*    reserved2[64];                                     /**< [in]: Reserved and must be set to NULL */
} ROCENC_CONFIG_H264_MEONLY;

/**
 * \struct _ROCENC_CONFIG_HEVC_MEONLY
 * HEVC encoder configuration parameters for ME only mode.
 */
typedef struct _ROCENC_CONFIG_HEVC_MEONLY
{
    uint32_t reserved [256];                                    /**< [in]: Reserved and must be set to 0 */
    void*    reserved1[64];                                     /**< [in]: Reserved and must be set to NULL */
} ROCENC_CONFIG_HEVC_MEONLY;

/**
 * \struct _ROCENC_CODEC_CONFIG
 * Codec-specific encoder configuration parameters.
 */
typedef union _ROCENC_CODEC_CONFIG
{
    ROCENC_CONFIG_H264        h264Config;                /**< [in]: H.264-specific encoder configuration. */
    ROCENC_CONFIG_HEVC        hevcConfig;                /**< [in]: HEVC-specific encoder configuration. */
    ROCENC_CONFIG_AV1         av1Config;                 /**< [in]: AV1-specific encoder configuration. */
    ROCENC_CONFIG_H264_MEONLY h264MeOnlyConfig;          /**< [in]: H.264-specific ME only encoder configuration. */
    ROCENC_CONFIG_HEVC_MEONLY hevcMeOnlyConfig;          /**< [in]: HEVC-specific ME only encoder configuration. */
    uint32_t                  reserved[320];             /**< [in]: Reserved and must be set to 0 */
} ROCENC_CODEC_CONFIG;

/**
 * \struct _ROCENC_CONFIG
 * Encoder configuration parameters set during initialization.
 */
typedef struct _ROCENC_CONFIG
{
    uint32_t                        version;                                     /**< [in]: Struct version. Must be ROCENC_CONFIG_VER. */
    ROCENC_PROFILE                  profile;                                     /**< [in]: Codec profile, or ROCENC_PROFILE_AUTOSELECT. */
    uint32_t                        gopLength;                                   /**< [in]: Pictures in one GOP, or ROCENC_INFINITE_GOPLENGTH. */
    int32_t                         frameIntervalP;                              /**< [in]: GOP pattern. 0: I, 1: IPP, 2: IBP, 3: IBBP. */
    uint32_t                        monoChromeEncoding;                          /**< [in]: Enable monochrome encoding for this session. */
    ROCENC_PARAMS_FRAME_FIELD_MODE  frameFieldMode;                              /**< [in]: Frame/field mode. Check ROCENC_CAPS_SUPPORT_FIELD_ENCODING. */
    ROCENC_MV_PRECISION             mvPrecision;                                 /**< [in]: Desired motion vector prediction precision. */
    ROCENC_RC_PARAMS                rcParams;                                    /**< [in]: Rate control parameters for this session. */
    ROCENC_CODEC_CONFIG             encodeCodecConfig;                           /**< [in]: Codec specific configuration. */
    uint32_t                        reserved [280];                              /**< [in]: Reserved and must be set to 0 */
    void*                           reserved2[64];                               /**< [in]: Reserved and must be set to NULL */
} ROCENC_CONFIG;

/** Macro for constructing the version field of ::_ROCENC_CONFIG */
#define ROCENC_CONFIG_VER (ROCENCAPI_STRUCT_VERSION(8) | ( 1<<31 ))

/**
 * Tuning information.  Not applicable to H.264 and HEVC ME only mode.
 */
typedef enum ROCENC_TUNING_INFO
{
    ROCENC_TUNING_INFO_UNDEFINED         = 0,                                     /**< Undefined. Invalid for encoding. */
    ROCENC_TUNING_INFO_HIGH_QUALITY      = 1,                                     /**< Latency tolerant encoding. */
    ROCENC_TUNING_INFO_LOW_LATENCY       = 2,                                     /**< Low latency streaming. */
    ROCENC_TUNING_INFO_ULTRA_LOW_LATENCY = 3,                                     /**< Ultra low latency streaming. */
    ROCENC_TUNING_INFO_LOSSLESS          = 4,                                     /**< Lossless encoding. */
    ROCENC_TUNING_INFO_COUNT                                                      /**< Count of tuning infos. Invalid value. */
}ROCENC_TUNING_INFO;

/**
 * Split encoding modes.  Not applicable to H.264.
 */
typedef enum _ROCENC_SPLIT_ENCODE_MODE
{
    ROCENC_SPLIT_AUTO_MODE               = 0,                                    /**< Forced mode disabled, auto mode enabled. */
    ROCENC_SPLIT_AUTO_FORCED_MODE        = 1,                                    /**< Forced mode with a driver-selected strip count. */
    ROCENC_SPLIT_TWO_FORCED_MODE         = 2,                                    /**< Forced 2-strip split frame encoding. */
    ROCENC_SPLIT_THREE_FORCED_MODE       = 3,                                    /**< Forced 3-strip split frame encoding. */
    ROCENC_SPLIT_DISABLE_MODE            = 15,                                   /**< Both auto and forced split modes disabled. */
} ROCENC_SPLIT_ENCODE_MODE;

/**
 * \struct _ROCENC_INITIALIZE_PARAMS
 * Encode session initialization parameters.
 */
typedef struct _ROCENC_INITIALIZE_PARAMS
{
    uint32_t                                   version;                         /**< [in]: Struct version. Must be ROCENC_INITIALIZE_PARAMS_VER. */
    ROCENC_CODEC                               codec;                           /**< [in]: Codec for which the encoder is created. */
    ROCENC_PRESET                              preset;                          /**< [in]: Preset. Applied before any other parameter. */
    uint32_t                                   encodeWidth;                     /**< [in]: Encode width. */
    uint32_t                                   encodeHeight;                    /**< [in]: Encode height. */
    uint32_t                                   darWidth;                        /**< [in]: Display aspect ratio width (H.264/HEVC) or render width (AV1). */
    uint32_t                                   darHeight;                       /**< [in]: Display aspect ratio height (H.264/HEVC) or render height (AV1). */
    uint32_t                                   frameRateNum;                    /**< [in]: Frame rate numerator. */
    uint32_t                                   frameRateDen;                    /**< [in]: Frame rate denominator. */
    uint32_t                                   enableEncodeAsync;               /**< [in]: Enable asynchronous mode. Must be 0 on rocEncode. */
    uint32_t                                   enablePTD;                       /**< [in]: Let the library take the picture type decision. */
    uint32_t                                   reportSliceOffsets        :1;    /**< [in]: Report slice offsets in ROCENC_LOCK_BITSTREAM. */
    uint32_t                                   enableSubFrameWrite       :1;    /**< [in]: Write out available bitstream at subframe intervals. */
    uint32_t                                   enableExternalMEHints     :1;    /**< [in]: Enable external ME hints for the current frame. */
    uint32_t                                   enableMEOnlyMode          :1;    /**< [in]: Enable ME only mode. */
    uint32_t                                   enableWeightedPrediction  :1;    /**< [in]: Enable weighted prediction. Not supported with B-frames. */
    uint32_t                                   splitEncodeMode           :4;    /**< [in]: Split encoding mode. Not applicable to H.264. */
    uint32_t                                   enableOutputInVidmem      :1;    /**< [in]: Output into an application-created video memory buffer. */
    uint32_t                                   enableReconFrameOutput    :1;    /**< [in]: Enable reconstructed frame output. */
    uint32_t                                   enableOutputStats         :1;    /**< [in]: Enable encoded frame output stats. */
    uint32_t                                   reservedBitFields         :20;   /**< [in]: Reserved bitfields and must be set to 0 */
    uint32_t                                   privDataSize;                    /**< [in]: Reserved private data buffer size. Must be 0. */
    void*                                      privData;                        /**< [in]: Reserved private data buffer. Must be NULL. */
    ROCENC_CONFIG*                             encodeConfig;                    /**< [in]: Advanced codec specific configuration. Overrides preset. */
    uint32_t                                   maxEncodeWidth;                  /**< [in]: Maximum encode width for dynamic resolution change. */
    uint32_t                                   maxEncodeHeight;                 /**< [in]: Maximum encode height for dynamic resolution change. */
    ROCENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE maxMEHintCountsPerBlock[2];    /**< [in]: Max hint candidates per block per direction. [0] L0, [1] L1. */
    ROCENC_TUNING_INFO                         tuningInfo;                      /**< [in]: Tuning info. Not applicable to ME only mode. */
    ROCENC_BUFFER_FORMAT                       bufferFormat;                    /**< [in]: Input buffer format. Used only by the DX12 interface. */
    uint32_t                                   numStateBuffers;                 /**< [in]: State buffers to allocate for encoding without advancing state. */
    ROCENC_OUTPUT_STATS_LEVEL                  outputStatsLevel;                /**< [in]: Granularity of the encoded frame output stats. */
    uint32_t                                   reserved [291];                  /**< [in]: Reserved and must be set to 0 */
    void*                                      reserved2[64];                   /**< [in]: Reserved and must be set to NULL */
} ROCENC_INITIALIZE_PARAMS;

/** Macro for constructing the version field of ::_ROCENC_INITIALIZE_PARAMS */
#define ROCENC_INITIALIZE_PARAMS_VER (ROCENCAPI_STRUCT_VERSION(6) | ( 1<<31 ))

/**
 * \struct _ROCENC_RECONFIGURE_PARAMS
 * Encode session reconfiguration parameters.
 */
typedef struct _ROCENC_RECONFIGURE_PARAMS
{
    uint32_t                                    version;                        /**< [in]: Struct version. Must be ROCENC_RECONFIGURE_PARAMS_VER. */
    ROCENC_INITIALIZE_PARAMS                    reInitEncodeParams;             /**< [in]: Encoder session re-initialization parameters. */
    uint32_t                                    resetEncoder            :1;     /**< [in]: Reset rate control and other internal encoder state. IDR only. */
    uint32_t                                    forceIDR                :1;     /**< [in]: Encode the current picture as IDR. Only valid when enablePTD is 1. */
    uint32_t                                    reserved                :30;

}ROCENC_RECONFIGURE_PARAMS;

/** Macro for constructing the version field of ::_ROCENC_RECONFIGURE_PARAMS */
#define ROCENC_RECONFIGURE_PARAMS_VER (ROCENCAPI_STRUCT_VERSION(1) | ( 1<<31 ))

/**
 * \struct _ROCENC_PRESET_CONFIG
 * Encoder preset configuration.
 */
typedef struct _ROCENC_PRESET_CONFIG
{
    uint32_t      version;                               /**< [in]:  Struct version. Must be ROCENC_PRESET_CONFIG_VER. */
    ROCENC_CONFIG presetCfg;                             /**< [out]: Preset configuration returned by the library. */
    uint32_t      reserved1[255];                        /**< [in]: Reserved and must be set to 0 */
    void*         reserved2[64];                         /**< [in]: Reserved and must be set to NULL */
}ROCENC_PRESET_CONFIG;

/** Macro for constructing the version field of ::_ROCENC_PRESET_CONFIG */
#define ROCENC_PRESET_CONFIG_VER (ROCENCAPI_STRUCT_VERSION(4) | ( 1<<31 ))

/**
 * \struct _ROCENC_PIC_PARAMS_MVC
 * MVC-specific per-frame parameters.
 */
typedef struct _ROCENC_PIC_PARAMS_MVC
{
    uint32_t version;                                    /**< [in]: Struct version. Must be ROCENC_PIC_PARAMS_MVC_VER. */
    uint32_t viewID;                                     /**< [in]: View ID of the current input view. */
    uint32_t temporalID;                                 /**< [in]: Temporal ID of the current input view. */
    uint32_t priorityID;                                 /**< [in]: Priority ID. Reserved and ignored. */
    uint32_t reserved1[12];                              /**< [in]: Reserved and must be set to 0. */
    void*    reserved2[8];                               /**< [in]: Reserved and must be set to NULL. */
}ROCENC_PIC_PARAMS_MVC;

/** Macro for constructing the version field of ::_ROCENC_PIC_PARAMS_MVC */
#define ROCENC_PIC_PARAMS_MVC_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * \union _ROCENC_PIC_PARAMS_H264_EXT
 * H.264 extension picture parameters.
 */
typedef union _ROCENC_PIC_PARAMS_H264_EXT
{
    ROCENC_PIC_PARAMS_MVC mvcPicParams;                  /**< [in]: MVC picture parameters. */
    uint32_t reserved1[32];                              /**< [in]: Reserved and must be set to 0. */
}ROCENC_PIC_PARAMS_H264_EXT;

/**
 * \struct _ROCENC_SEI_PAYLOAD
 * User SEI message.
 */
typedef struct _ROCENC_SEI_PAYLOAD
{
    uint32_t payloadSize;            /**< [in] SEI payload size in bytes. Must be byte aligned (Annex D). */
    uint32_t payloadType;            /**< [in] SEI payload type (Annex D of the H.264 specification). */
    uint8_t *payload;                /**< [in] Pointer to the user data. */
} ROCENC_SEI_PAYLOAD;

#define ROCENC_H264_SEI_PAYLOAD ROCENC_SEI_PAYLOAD

/**
 * \struct _ROCENC_PIC_PARAMS_H264
 * H.264 specific per-frame encode parameters.
 */
typedef struct _ROCENC_PIC_PARAMS_H264
{
    uint32_t displayPOCSyntax;                           /**< [in]: Display POC syntax. Required when the client takes the picture type decision. */
    uint32_t reserved3;                                  /**< [in]: Reserved and must be set to 0 */
    uint32_t refPicFlag;                                 /**< [in]: 1 for a reference picture. Ignored when enablePTD is 1. */
    uint32_t colourPlaneId;                              /**< [in]: Colour plane ID of the current input. */
    uint32_t forceIntraRefreshWithFrameCnt;              /**< [in]: Force an intra refresh of this duration. Not usable with B frames. */
    uint32_t constrainedFrame           :1;              /**< [in]: Encode this frame with fully independent slices. */
    uint32_t sliceModeDataUpdate        :1;              /**< [in]: Change sliceModeData to specify a new slice size. */
    uint32_t ltrMarkFrame               :1;              /**< [in]: Mark this frame as LTR. */
    uint32_t ltrUseFrames               :1;              /**< [in]: Allow encoding this frame using the LTR frames in ltrUseFrameBitmap. */
    uint32_t reservedBitFields          :28;             /**< [in]: Reserved bit fields and must be set to 0 */
    uint8_t* sliceTypeData;                              /**< [in]: Deprecated. */
    uint32_t sliceTypeArrayCnt;                          /**< [in]: Deprecated. */
    uint32_t seiPayloadArrayCnt;                         /**< [in]: Number of elements in seiPayloadArray. */
    ROCENC_SEI_PAYLOAD* seiPayloadArray;                 /**< [in]: SEI payloads inserted for this frame. */
    uint32_t sliceMode;                                  /**< [in]: 0 MB based, 1 byte based, 2 MB row based, 3 numSlices in picture. */
    uint32_t sliceModeData;                              /**< [in]: Parameter for sliceMode. */
    uint32_t ltrMarkFrameIdx;                            /**< [in]: LTR index used when marking this frame as LTR. */
    uint32_t ltrUseFrameBitmap;                          /**< [in]: Bitmap of LTR frame indices to use when encoding this frame. */
    uint32_t ltrUsageMode;                               /**< [in]: Not supported. Reserved and must be set to 0. */
    uint32_t forceIntraSliceCount;                       /**< [in]: Number of slices forced to intra. Requires sliceMode 3. */
    uint32_t *forceIntraSliceIdx;                        /**< [in]: Slice indices forced to intra. forceIntraSliceCount entries. */
    ROCENC_PIC_PARAMS_H264_EXT h264ExtPicParams;         /**< [in]: H.264 extension configuration parameters. */
    ROCENC_TIME_CODE timeCode;                           /**< [in]: Clock timestamp sets for the picture timing SEI. */
    uint32_t reserved [203];                             /**< [in]: Reserved and must be set to 0. */
    void*    reserved2[61];                              /**< [in]: Reserved and must be set to NULL. */
} ROCENC_PIC_PARAMS_H264;

/**
 * \struct _ROCENC_PIC_PARAMS_HEVC
 * HEVC specific per-frame encode parameters.
 */
typedef struct _ROCENC_PIC_PARAMS_HEVC
{
    uint32_t displayPOCSyntax;                           /**< [in]: Display POC syntax. Required when the client takes the picture type decision. */
    uint32_t refPicFlag;                                 /**< [in]: 1 for a reference picture. Ignored when enablePTD is 1. */
    uint32_t temporalId;                                 /**< [in]: Temporal id of the picture. */
    uint32_t forceIntraRefreshWithFrameCnt;              /**< [in]: Force an intra refresh of this duration. Not usable with B frames. */
    uint32_t constrainedFrame           :1;              /**< [in]: Encode this frame with fully independent slices. */
    uint32_t sliceModeDataUpdate        :1;              /**< [in]: Change sliceModeData to specify a new slice size. */
    uint32_t ltrMarkFrame               :1;              /**< [in]: Mark this frame as LTR. */
    uint32_t ltrUseFrames               :1;              /**< [in]: Allow encoding this frame using the LTR frames in ltrUseFrameBitmap. */
    uint32_t reservedBitFields          :28;             /**< [in]: Reserved bit fields and must be set to 0 */
    uint8_t* sliceTypeData;                              /**< [in]: Per-slice type array used to force a slice to intra. sliceMode 3 only. */
    uint32_t sliceTypeArrayCnt;                          /**< [in]: Number of elements in sliceTypeData, 0 when it is NULL. */
    uint32_t sliceMode;                                  /**< [in]: 0 CTU based, 1 byte based, 2 CTU row based, 3 numSlices in picture. */
    uint32_t sliceModeData;                              /**< [in]: Parameter for sliceMode. */
    uint32_t ltrMarkFrameIdx;                            /**< [in]: LTR index used when marking this frame as LTR. */
    uint32_t ltrUseFrameBitmap;                          /**< [in]: Bitmap of LTR frame indices to use when encoding this frame. */
    uint32_t ltrUsageMode;                               /**< [in]: Not supported. Reserved and must be set to 0. */
    uint32_t seiPayloadArrayCnt;                         /**< [in]: Number of elements in seiPayloadArray. */
    uint32_t reserved;                                   /**< [in]: Reserved and must be set to 0. */
    ROCENC_SEI_PAYLOAD* seiPayloadArray;                 /**< [in]: SEI payloads inserted for this frame. */
    ROCENC_TIME_CODE timeCode;                           /**< [in]: Clock timestamp sets for the time code SEI. */
    uint32_t reserved2 [237];                            /**< [in]: Reserved and must be set to 0. */
    void*    reserved3[61];                              /**< [in]: Reserved and must be set to NULL. */
} ROCENC_PIC_PARAMS_HEVC;

#define ROCENC_AV1_OBU_PAYLOAD ROCENC_SEI_PAYLOAD

/**
 * \struct _ROCENC_PIC_PARAMS_AV1
 * AV1 specific per-frame encode parameters.
 */
typedef struct _ROCENC_PIC_PARAMS_AV1
{
    uint32_t displayPOCSyntax;                           /**< [in]: Display POC syntax. Required when the client takes the picture type decision. */
    uint32_t refPicFlag;                                 /**< [in]: 1 for a reference picture. Ignored when enablePTD is 1. */
    uint32_t temporalId;                                 /**< [in]: Temporal id of the picture. */
    uint32_t forceIntraRefreshWithFrameCnt;              /**< [in]: Force an intra refresh of this duration. Not usable with B frames. */
    uint32_t goldenFrameFlag            : 1;             /**< [in]: Encode frame as a golden frame. Ignored when enablePTD is 1. */
    uint32_t arfFrameFlag               : 1;             /**< [in]: Encode frame as an alternate reference frame. */
    uint32_t arf2FrameFlag              : 1;             /**< [in]: Encode frame as an alternate reference 2 frame. */
    uint32_t bwdFrameFlag               : 1;             /**< [in]: Encode frame as a backward reference frame. */
    uint32_t overlayFrameFlag           : 1;             /**< [in]: Encode frame as an overlay frame. */
    uint32_t showExistingFrameFlag      : 1;             /**< [in]: Controls show_existing_frame for the overlay frame. Placeholder. */
    uint32_t errorResilientModeFlag     : 1;             /**< [in]: Encode frame independently of previously encoded frames. */
    uint32_t tileConfigUpdate           : 1;             /**< [in]: Overwrite the default tile configuration. */
    uint32_t enableCustomTileConfig     : 1;             /**< [in]: Enable the custom tile configuration. */
    uint32_t filmGrainParamsUpdate      : 1;             /**< [in]: Update the previous film grain parameters. */
    uint32_t reservedBitFields          : 22;            /**< [in]: Reserved bitfields and must be set to 0 */
    uint32_t numTileColumns;                             /**< [in]: Number of tile columns. See enableCustomTileConfig. */
    uint32_t numTileRows;                                /**< [in]: Number of tile rows. See enableCustomTileConfig. */
    uint32_t *tileWidths;                                /**< [in]: Width of each tile column in 64x64 CTU units. */
    uint32_t *tileHeights;                               /**< [in]: Height of each tile row in 64x64 CTU units. */
    uint32_t obuPayloadArrayCnt;                         /**< [in]: Number of elements in obuPayloadArray. */
    uint32_t reserved;                                   /**< [in]: Reserved and must be set to 0. */
    ROCENC_AV1_OBU_PAYLOAD* obuPayloadArray;             /**< [in]: OBU payloads inserted for this frame. */
    ROCENC_FILM_GRAIN_PARAMS_AV1 *filmGrainParams;       /**< [in]: Film grain parameters when filmGrainParamsUpdate is 1. */
    uint32_t reserved2[247];                             /**< [in]: Reserved and must be set to 0. */
    void*    reserved3[61];                              /**< [in]: Reserved and must be set to NULL. */
} ROCENC_PIC_PARAMS_AV1;

/**
 * Codec specific per-picture encoding parameters.
 */
typedef union _ROCENC_CODEC_PIC_PARAMS
{
    ROCENC_PIC_PARAMS_H264 h264PicParams;                /**< [in]: H.264 encode picture params. */
    ROCENC_PIC_PARAMS_HEVC hevcPicParams;                /**< [in]: HEVC encode picture params. */
    ROCENC_PIC_PARAMS_AV1  av1PicParams;                 /**< [in]: AV1 encode picture params. */
    uint32_t               reserved[256];                /**< [in]: Reserved and must be set to 0. */
} ROCENC_CODEC_PIC_PARAMS;

/**
 * \struct _ROCENC_PIC_PARAMS
 * Per-frame encoding parameters.
 */
typedef struct _ROCENC_PIC_PARAMS
{
    uint32_t                                    version;                        /**< [in]: Struct version. Must be ROCENC_PIC_PARAMS_VER. */
    uint32_t                                    inputWidth;                     /**< [in]: Input frame width. */
    uint32_t                                    inputHeight;                    /**< [in]: Input frame height. */
    uint32_t                                    inputPitch;                     /**< [in]: Input buffer pitch. Set to inputWidth if unknown. */
    uint32_t                                    encodePicFlags;                 /**< [in]: Bitwise OR of ROCENC_PIC_FLAGS. */
    uint32_t                                    frameIdx;                       /**< [in]: Frame index associated with the input frame [optional]. */
    uint64_t                                    inputTimeStamp;                 /**< [in]: Opaque data associated with the encoded frame, not encoded. */
    uint64_t                                    inputDuration;                  /**< [in]: Duration of the input picture. */
    ROCENC_INPUT_PTR                            inputBuffer;                    /**< [in]: Input buffer from rocEncCreateInputBuffer() or rocEncMapInputResource(). */
    ROCENC_OUTPUT_PTR                           outputBitstream;                /**< [in]: Output buffer from rocEncCreateBitstreamBuffer(). */
    void*                                       completionEvent;                /**< [in]: Event signalled on completion [asynchronous mode only]. */
    ROCENC_BUFFER_FORMAT                        bufferFmt;                      /**< [in]: Input buffer format. */
    ROCENC_PIC_STRUCT                           pictureStruct;                  /**< [in]: Structure of the input picture. */
    ROCENC_PIC_TYPE                             pictureType;                    /**< [in]: Input picture type. Required when enablePTD is 0. */
    ROCENC_CODEC_PIC_PARAMS                     codecPicParams;                 /**< [in]: Codec specific per-picture encoding parameters. */
    ROCENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE meHintCountsPerBlock[2];       /**< [in]: Hint candidates per block per direction. [0] L0, [1] L1. */
    ROCENC_EXTERNAL_ME_HINT                    *meExternalHints;                /**< [in]: ME external hints for the current frame (H.264, HEVC). */
    uint32_t                                    reserved1[6];                   /**< [in]: Reserved and must be set to 0 */
    void*                                       reserved2[2];                   /**< [in]: Reserved and must be set to NULL */
    int8_t                                     *qpDeltaMap;                     /**< [in]: Per-block signed byte array, interpreted per ROCENC_RC_PARAMS::qpMapMode. */
    uint32_t                                    qpDeltaMapSize;                 /**< [in]: Size in bytes of the qpDeltaMap surface. */
    uint32_t                                    reservedBitFields;              /**< [in]: Reserved bitfields and must be set to 0 */
    uint16_t                                    meHintRefPicDist[2];            /**< [in]: Temporal distance for the reference picture used in external ME. */
    ROCENC_INPUT_PTR                            alphaBuffer;                    /**< [in]: Input alpha buffer. HEVC with alpha layer only. */
    ROCENC_EXTERNAL_ME_SB_HINT                 *meExternalSbHints;              /**< [in]: AV1 ME external SB hints for the current frame. */
    uint32_t                                    meSbHintsCount;                 /**< [in]: AV1 total number of external ME SB hint candidates. */
    uint32_t                                    stateBufferIdx;                 /**< [in]: State buffer index, 0 to numStateBuffers - 1. */
    ROCENC_OUTPUT_PTR                           outputReconBuffer;              /**< [in]: Reconstructed frame output buffer. Requires enableReconFrameOutput. */
    uint32_t                                    reserved3[284];                 /**< [in]: Reserved and must be set to 0 */
    void*                                       reserved4[57];                  /**< [in]: Reserved and must be set to NULL */
} ROCENC_PIC_PARAMS;

/** Macro for constructing the version field of ::_ROCENC_PIC_PARAMS */
#define ROCENC_PIC_PARAMS_VER (ROCENCAPI_STRUCT_VERSION(6) | ( 1<<31 ))

/**
 * \struct _ROCENC_MEONLY_PARAMS
 * ME only parameters, sent per motion estimation call.
 */
typedef struct _ROCENC_MEONLY_PARAMS
{
    uint32_t                version;                            /**< [in]: Struct version. Must be ROCENC_MEONLY_PARAMS_VER. */
    uint32_t                inputWidth;                         /**< [in]: Input frame width. */
    uint32_t                inputHeight;                        /**< [in]: Input frame height. */
    ROCENC_INPUT_PTR        inputBuffer;                        /**< [in]: Input buffer from rocEncCreateInputBuffer() or rocEncMapInputResource(). */
    ROCENC_INPUT_PTR        referenceFrame;                     /**< [in]: Reference frame pointer. */
    ROCENC_OUTPUT_PTR       mvBuffer;                           /**< [in]: Motion vector data buffer from rocEncCreateMVBuffer(). */
    ROCENC_BUFFER_FORMAT    bufferFmt;                          /**< [in]: Input buffer format. */
    void*                   completionEvent;                    /**< [in]: Event signalled on completion [asynchronous mode only]. */
    uint32_t                viewID;                             /**< [in]: Left or right view id when bStereoEnable is set, 0 otherwise. */
    ROCENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE
                            meHintCountsPerBlock[2];            /**< [in]: Hint candidates per block for the current frame. [0] is L0. */
    ROCENC_EXTERNAL_ME_HINT *meExternalHints;                   /**< [in]: ME external hints for the current frame. */
    uint32_t                reserved1[243];                     /**< [in]: Reserved and must be set to 0 */
    void*                   reserved2[59];                      /**< [in]: Reserved and must be set to NULL */
} ROCENC_MEONLY_PARAMS;

/** ROCENC_MEONLY_PARAMS struct version. */
#define ROCENC_MEONLY_PARAMS_VER ROCENCAPI_STRUCT_VERSION(3)

/**
 * \struct _ROCENC_LOCK_BITSTREAM
 * Bitstream buffer lock parameters.
 */
typedef struct _ROCENC_LOCK_BITSTREAM
{
    uint32_t                version;                     /**< [in]: Struct version. Must be ROCENC_LOCK_BITSTREAM_VER. */
    uint32_t                doNotWait         :1;        /**< [in]: Return even if the operation has not completed. Otherwise the call blocks. */
    uint32_t                ltrFrame          :1;        /**< [out]: This frame is marked as an LTR frame. */
    uint32_t                getRCStats        :1;        /**< [in]: Also return intra/inter MB counts and average MVX, MVY. */
    uint32_t                reservedBitFields :29;       /**< [in]: Reserved bit fields and must be set to 0 */
    void*                   outputBitstream;             /**< [in]: Bitstream buffer being locked. */
    uint32_t*               sliceOffsets;                /**< [in, out]: Receives the slice (H.264/HEVC) or tile (AV1) offsets. */
    uint32_t                frameIdx;                    /**< [out]: Frame number whose bitstream is being retrieved. */
    uint32_t                hwEncodeStatus;              /**< [out]: Status for the locked picture. */
    uint32_t                numSlices;                   /**< [out]: Number of slices or tiles. Requires reportSliceOffsets. */
    uint32_t                bitstreamSizeInBytes;        /**< [out]: Bytes generated and copied to bitstreamBufferPtr. */
    uint64_t                outputTimeStamp;             /**< [out]: Presentation timestamp of the encoded output. */
    uint64_t                outputDuration;              /**< [out]: Presentation duration of the encoded output. */
    void*                   bitstreamBufferPtr;          /**< [out]: Pointer to the generated output bitstream. */
    ROCENC_PIC_TYPE         pictureType;                 /**< [out]: Picture type of the encoded picture. */
    ROCENC_PIC_STRUCT       pictureStruct;               /**< [out]: Structure of the generated output picture. */
    uint32_t                frameAvgQP;                  /**< [out]: Average QP of the frame. */
    uint32_t                frameSatd;                   /**< [out]: Total SATD cost for the whole frame. */
    uint32_t                ltrFrameIdx;                 /**< [out]: Frame index associated with this LTR frame. */
    uint32_t                ltrFrameBitmap;              /**< [out]: Bitmap of LTR frame indices used to encode this frame. */
    uint32_t                temporalId;                  /**< [out]: TemporalId of the frame when using temporal SVC encoding. */
    uint32_t                intraMBCount;                /**< [out]: Intra block count. Requires getRCStats. */
    uint32_t                interMBCount;                /**< [out]: Inter block count. Requires getRCStats. */
    int32_t                 averageMVX;                  /**< [out]: Average motion vector in X. Requires getRCStats. */
    int32_t                 averageMVY;                  /**< [out]: Average motion vector in Y. Requires getRCStats. */
    uint32_t                alphaLayerSizeInBytes;       /**< [out]: Bytes generated for the alpha layer. */
    uint32_t                outputStatsPtrSize;          /**< [in]: Size of the buffer pointed to by outputStatsPtr. */
    void*                   outputStatsPtr;              /**< [in, out]: Receives the encoded frame output stats. */
    uint32_t                frameIdxDisplay;             /**< [out]: Frame index in display order. */
    uint32_t                reserved1[220];              /**< [in]: Reserved and must be set to 0 */
    void*                   reserved2[63];               /**< [in]: Reserved and must be set to NULL */
    uint32_t                reservedInternal[8];         /**< [in]: Reserved and must be set to 0 */
} ROCENC_LOCK_BITSTREAM;

#define ROCENC_LOCK_BITSTREAM_VER (ROCENCAPI_STRUCT_VERSION(1) | ( 1<<31 ))

/**
 * \struct _ROCENC_LOCK_INPUT_BUFFER
 * Uncompressed input buffer lock parameters.
 */
typedef struct _ROCENC_LOCK_INPUT_BUFFER
{
    uint32_t                  version;                   /**< [in]:  Struct version. Must be ROCENC_LOCK_INPUT_BUFFER_VER. */
    uint32_t                  doNotWait         :1;      /**< [in]:  Make rocEncLockInputBuffer() non-blocking. */
    uint32_t                  reservedBitFields :31;     /**< [in]:  Reserved bitfields and must be set to 0 */
    ROCENC_INPUT_PTR          inputBuffer;               /**< [in]:  Input buffer to lock. */
    void*                     bufferDataPtr;             /**< [out]: Host pointer to the locked input buffer data. */
    uint32_t                  pitch;                     /**< [out]: Pitch of the locked input buffer. */
    uint32_t                  reserved1[251];            /**< [in]:  Reserved and must be set to 0  */
    void*                     reserved2[64];             /**< [in]:  Reserved and must be set to NULL  */
} ROCENC_LOCK_INPUT_BUFFER;

/** Macro for constructing the version field of ::_ROCENC_LOCK_INPUT_BUFFER */
#define ROCENC_LOCK_INPUT_BUFFER_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * \struct _ROCENC_MAP_INPUT_RESOURCE
 * Map an input resource to a rocEncode input buffer.
 */
typedef struct _ROCENC_MAP_INPUT_RESOURCE
{
    uint32_t                   version;                   /**< [in]:  Struct version. Must be ROCENC_MAP_INPUT_RESOURCE_VER. */
    uint32_t                   subResourceIndex;          /**< [in]:  Deprecated. Do not use. */
    void*                      inputResource;             /**< [in]:  Deprecated. Do not use. */
    ROCENC_REGISTERED_PTR      registeredResource;        /**< [in]:  Handle from rocEncRegisterResource(). */
    ROCENC_INPUT_PTR           mappedResource;            /**< [out]: Mapped pointer for ROCENC_PIC_PARAMS::inputBuffer. */
    ROCENC_BUFFER_FORMAT       mappedBufferFmt;           /**< [out]: Buffer format for ROCENC_PIC_PARAMS::bufferFmt. */
    uint32_t                   reserved1[251];            /**< [in]:  Reserved and must be set to 0. */
    void*                      reserved2[63];             /**< [in]:  Reserved and must be set to NULL */
} ROCENC_MAP_INPUT_RESOURCE;

/** Macro for constructing the version field of ::_ROCENC_MAP_INPUT_RESOURCE */
#define ROCENC_MAP_INPUT_RESOURCE_VER ROCENCAPI_STRUCT_VERSION(4)

/**
 * \struct _ROCENC_INPUT_RESOURCE_OPENGL_TEX
 * ROCENC_REGISTER_RESOURCE::resourceToRegister must point to a variable of
 * this type when resourceType is ROCENC_INPUT_RESOURCE_TYPE_OPENGL_TEX.
 */
typedef struct _ROCENC_INPUT_RESOURCE_OPENGL_TEX
{
    uint32_t texture;                                     /**< [in]: Name of the texture to be used. */
    uint32_t target;                                      /**< [in]: GL_TEXTURE_RECTANGLE or GL_TEXTURE_2D. */
} ROCENC_INPUT_RESOURCE_OPENGL_TEX;

/**
 * \struct _ROCENC_FENCE_POINT_D3D12
 * Fence and fence value for synchronization.  Declared for structural parity;
 * the D3D12 interface is not supported by rocEncode.
 */
typedef struct _ROCENC_FENCE_POINT_D3D12
{
    uint32_t                version;                   /**< [in]: Struct version. Must be ROCENC_FENCE_POINT_D3D12_VER. */
    uint32_t                reserved;                  /**< [in]: Reserved and must be set to 0. */
    void*                   pFence;                    /**< [in]: Pointer to an ID3D12Fence. */
    uint64_t                waitValue;                 /**< [in]: Fence value to reach before the GPU operation. */
    uint64_t                signalValue;               /**< [in]: Fence value to set after the GPU operation. */
    uint32_t                bWait:1;                   /**< [in]: Wait on waitValue before starting the GPU operation. */
    uint32_t                bSignal:1;                 /**< [in]: Signal signalValue after the GPU operation completes. */
    uint32_t                reservedBitField:30;       /**< [in]: Reserved and must be set to 0. */
    uint32_t                reserved1[7];              /**< [in]: Reserved and must be set to 0. */
} ROCENC_FENCE_POINT_D3D12;

#define ROCENC_FENCE_POINT_D3D12_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * \struct _ROCENC_INPUT_RESOURCE_D3D12
 * Declared for structural parity; the D3D12 interface is not supported.
 */
typedef struct _ROCENC_INPUT_RESOURCE_D3D12
{
    uint32_t                    version;                /**< [in]: Struct version. Must be ROCENC_INPUT_RESOURCE_D3D12_VER. */
    uint32_t                    reserved;               /**< [in]: Reserved and must be set to 0. */
    ROCENC_INPUT_PTR            pInputBuffer;           /**< [in]: Input surface from rocEncMapInputResource(). */
    ROCENC_FENCE_POINT_D3D12    inputFencePoint;        /**< [in]: Fence and fence values for the GPU wait and signal. */
    uint32_t                    reserved1[16];          /**< [in]: Reserved and must be set to 0. */
    void*                       reserved2[16];          /**< [in]: Reserved and must be set to NULL. */
} ROCENC_INPUT_RESOURCE_D3D12;

#define ROCENC_INPUT_RESOURCE_D3D12_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * \struct _ROCENC_OUTPUT_RESOURCE_D3D12
 * Declared for structural parity; the D3D12 interface is not supported.
 */
typedef struct _ROCENC_OUTPUT_RESOURCE_D3D12
{
    uint32_t                    version;                /**< [in]: Struct version. Must be ROCENC_OUTPUT_RESOURCE_D3D12_VER. */
    uint32_t                    reserved;               /**< [in]: Reserved and must be set to 0. */
    ROCENC_INPUT_PTR            pOutputBuffer;          /**< [in]: Output buffer from rocEncMapInputResource(). */
    ROCENC_FENCE_POINT_D3D12    outputFencePoint;       /**< [in]: Fence and fence values for the GPU wait and signal. */
    uint32_t                    reserved1[16];          /**< [in]: Reserved and must be set to 0. */
    void*                       reserved2[16];          /**< [in]: Reserved and must be set to NULL. */
} ROCENC_OUTPUT_RESOURCE_D3D12;

#define ROCENC_OUTPUT_RESOURCE_D3D12_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * \struct _ROCENC_REGISTER_RESOURCE
 * Register a resource for future use with the rocEncode interface.
 */
typedef struct _ROCENC_REGISTER_RESOURCE
{
    uint32_t                    version;                        /**< [in]: Struct version. Must be ROCENC_REGISTER_RESOURCE_VER. */
    ROCENC_INPUT_RESOURCE_TYPE  resourceType;                   /**< [in]: Type of resource to register.  rocEncode supports
                                                                           ROCENC_INPUT_RESOURCE_TYPE_HIPDEVICEPTR. */
    uint32_t                    width;                          /**< [in]: Input frame width. */
    uint32_t                    height;                         /**< [in]: Input frame height. */
    uint32_t                    pitch;                          /**< [in]: Input buffer pitch in bytes. Must be a multiple of 4. */
    uint32_t                    subResourceIndex;               /**< [in]: DirectX subresource index. 0 for other interfaces. */
    void*                       resourceToRegister;             /**< [in]: Handle to the resource being registered. */
    ROCENC_REGISTERED_PTR       registeredResource;             /**< [out]: Registered resource handle. */
    ROCENC_BUFFER_FORMAT        bufferFormat;                   /**< [in]: Buffer format of the resource. */
    ROCENC_BUFFER_USAGE         bufferUsage;                    /**< [in]: Usage of the resource. */
    ROCENC_FENCE_POINT_D3D12*   pInputFencePoint;               /**< [in]: D3D12 input fence. Not supported by rocEncode. */
    uint32_t                    chromaOffset[2];                /**< [out]: Chroma offset for a reconstructed output buffer. */
    uint32_t                    reserved1[245];                 /**< [in]: Reserved and must be set to 0. */
    void*                       reserved2[61];                  /**< [in]: Reserved and must be set to NULL. */
} ROCENC_REGISTER_RESOURCE;

/** Macro for constructing the version field of ::_ROCENC_REGISTER_RESOURCE */
#define ROCENC_REGISTER_RESOURCE_VER ROCENCAPI_STRUCT_VERSION(4)

/**
 * \struct _ROCENC_STAT
 * Encode statistics.
 */
typedef struct _ROCENC_STAT
{
    uint32_t            version;                         /**< [in]:  Struct version. Must be ROCENC_STAT_VER. */
    uint32_t            reserved;                        /**< [in]:  Reserved and must be set to 0 */
    ROCENC_OUTPUT_PTR   outputBitStream;                 /**< [out]: Pointer to the output bitstream. */
    uint32_t            bitStreamSize;                   /**< [out]: Size of the generated bitstream in bytes. */
    uint32_t            picType;                         /**< [out]: Picture type of the encoded picture. See ROCENC_PIC_TYPE. */
    uint32_t            lastValidByteOffset;             /**< [out]: Offset of the last valid byte of the completed bitstream. */
    uint32_t            sliceOffsets[16];                /**< [out]: Offsets of each slice. */
    uint32_t            picIdx;                          /**< [out]: Picture number. */
    uint32_t            frameAvgQP;                      /**< [out]: Average QP of the frame. */
    uint32_t            ltrFrame          :1;            /**< [out]: This frame is marked as an LTR frame. */
    uint32_t            reservedBitFields :31;           /**< [in]:  Reserved bit fields and must be set to 0 */
    uint32_t            ltrFrameIdx;                     /**< [out]: Frame index associated with this LTR frame. */
    uint32_t            intraMBCount;                    /**< [out]: Intra block count of the encoded frame. */
    uint32_t            interMBCount;                    /**< [out]: Inter block count of the encoded frame. */
    int32_t             averageMVX;                      /**< [out]: Average motion vector in X. */
    int32_t             averageMVY;                      /**< [out]: Average motion vector in Y. */
    uint32_t            reserved1[226];                  /**< [in]:  Reserved and must be set to 0 */
    void*               reserved2[64];                   /**< [in]:  Reserved and must be set to NULL */
} ROCENC_STAT;

/** Macro for constructing the version field of ::_ROCENC_STAT */
#define ROCENC_STAT_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * \struct _ROCENC_SEQUENCE_PARAM_PAYLOAD
 * Sequence and picture parameter payload.
 */
typedef struct _ROCENC_SEQUENCE_PARAM_PAYLOAD
{
    uint32_t            version;                         /**< [in]:  Struct version. Must be ROCENC_SEQUENCE_PARAM_PAYLOAD_VER. */
    uint32_t            inBufferSize;                    /**< [in]:  Size of spsppsBuffer provided by the client. */
    uint32_t            spsId;                           /**< [in]:  SPS id used in the sequence header. Default 0. */
    uint32_t            ppsId;                           /**< [in]:  PPS id used in the picture header. Default 0. */
    void*               spsppsBuffer;                    /**< [in]:  Client-managed header buffer of inBufferSize bytes. */
    uint32_t*           outSPSPPSPayloadSize;            /**< [out]: Size of the sequence and picture header in bytes. */
    uint32_t            reserved [250];                  /**< [in]:  Reserved and must be set to 0 */
    void*               reserved2[64];                   /**< [in]:  Reserved and must be set to NULL */
} ROCENC_SEQUENCE_PARAM_PAYLOAD;

/** Macro for constructing the version field of ::_ROCENC_SEQUENCE_PARAM_PAYLOAD */
#define ROCENC_SEQUENCE_PARAM_PAYLOAD_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * Event registration / unregistration parameters.
 */
typedef struct _ROCENC_EVENT_PARAMS
{
    uint32_t            version;                          /**< [in]: Struct version. Must be ROCENC_EVENT_PARAMS_VER. */
    uint32_t            reserved;                         /**< [in]: Reserved and must be set to 0 */
    void*               completionEvent;                  /**< [in]: Event to register or unregister. */
    uint32_t            reserved1[253];                   /**< [in]: Reserved and must be set to 0    */
    void*               reserved2[64];                    /**< [in]: Reserved and must be set to NULL */
} ROCENC_EVENT_PARAMS;

/** Macro for constructing the version field of ::_ROCENC_EVENT_PARAMS */
#define ROCENC_EVENT_PARAMS_VER ROCENCAPI_STRUCT_VERSION(1)

/**
 * Encoder session creation parameters.
 */
typedef struct _ROCENC_OPEN_ENCODE_SESSIONEX_PARAMS
{
    uint32_t            version;                          /**< [in]: Struct version. Must be ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER. */
    ROCENC_DEVICE_TYPE  deviceType;                       /**< [in]: Device type. Must be ROCENC_DEVICE_TYPE_HIP. */
    void*               device;                           /**< [in]: Pointer to the client device.  For ROCENC_DEVICE_TYPE_HIP this is
                                                                     the HIP device ordinal, passed by value through the pointer
                                                                     (i.e. (void*)(intptr_t)device_id).

                                                                     HAZARD: under the ordinal-by-value encoding a NULL device is
                                                                     indistinguishable from a legitimate request for device 0, and
                                                                     is accepted as such.  Code that zero-initialises this struct
                                                                     and never sets `device` will therefore bind device 0 silently
                                                                     rather than be rejected.  Set it explicitly, always.
                                                                     (Tracked as an open API decision; see the API plan §6.) */
    void*               reserved;                         /**< [in]: Reserved and must be set to 0. */
    uint32_t            apiVersion;                       /**< [in]: API version. Should be set to ROCENCAPI_VERSION. */
    uint32_t            reserved1[253];                   /**< [in]: Reserved and must be set to 0    */
    void*               reserved2[64];                    /**< [in]: Reserved and must be set to NULL */
} ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS;

/** Macro for constructing the version field of ::_ROCENC_OPEN_ENCODE_SESSIONEX_PARAMS */
#define ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER ROCENCAPI_STRUCT_VERSION(1)

/** @} */ /* END ENCODER_STRUCTURE */

/**
 * \addtogroup ENCODE_FUNC rocEncode functions
 * @{
 */

/**
 * \brief Opens an encoding session.  Deprecated; use rocEncOpenEncodeSessionEx().
 * \return ::ROCENC_ERR_INVALID_CALL
 */
rocEncStatus ROCENCAPI rocEncOpenEncodeSession                  (void* device, uint32_t deviceType, void** encoder);

/**
 * \brief Retrieves the number of encode-capable codecs on the device.
 */
rocEncStatus ROCENCAPI rocEncGetEncodeCodecCount                (void* encoder, uint32_t* codecCount);

/**
 * \brief Retrieves the encode-capable codecs on the device.
 */
rocEncStatus ROCENCAPI rocEncGetEncodeCodecs                    (void* encoder, ROCENC_CODEC* codecs, uint32_t codecArraySize, uint32_t* codecCount);

/**
 * \brief Retrieves the number of supported profiles for a codec.
 */
rocEncStatus ROCENCAPI rocEncGetEncodeProfileCount              (void* encoder, ROCENC_CODEC codec, uint32_t* profileCount);

/**
 * \brief Retrieves the supported profiles for a codec.
 */
rocEncStatus ROCENCAPI rocEncGetEncodeProfiles                  (void* encoder, ROCENC_CODEC codec, ROCENC_PROFILE* profiles, uint32_t profileArraySize, uint32_t* profileCount);

/**
 * \brief Retrieves the number of supported input formats for a codec.
 */
rocEncStatus ROCENCAPI rocEncGetInputFormatCount                (void* encoder, ROCENC_CODEC codec, uint32_t* inputFmtCount);

/**
 * \brief Retrieves the supported input formats for a codec.
 */
rocEncStatus ROCENCAPI rocEncGetInputFormats                    (void* encoder, ROCENC_CODEC codec, ROCENC_BUFFER_FORMAT* inputFmts, uint32_t inputFmtArraySize, uint32_t* inputFmtCount);

/**
 * \brief Retrieves the capability value for a given codec and capability.
 */
rocEncStatus ROCENCAPI rocEncGetEncodeCaps                      (void* encoder, ROCENC_CODEC codec, ROCENC_CAPS_PARAM* capsParam, int* capsVal);

/**
 * \brief Retrieves the number of supported presets for a codec.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncGetEncodePresetCount               (void* encoder, ROCENC_CODEC codec, uint32_t* presetCount);

/**
 * \brief Retrieves the supported presets for a codec.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncGetEncodePresets                   (void* encoder, ROCENC_CODEC codec, ROCENC_PRESET* presets, uint32_t presetArraySize, uint32_t* presetCount);

/**
 * \brief Retrieves the encoder configuration for a preset.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncGetEncodePresetConfig              (void* encoder, ROCENC_CODEC codec, ROCENC_PRESET preset, ROCENC_PRESET_CONFIG* presetConfig);

/**
 * \brief Retrieves the encoder configuration for a preset and tuning info.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncGetEncodePresetConfigEx            (void* encoder, ROCENC_CODEC codec, ROCENC_PRESET preset, ROCENC_TUNING_INFO tuningInfo, ROCENC_PRESET_CONFIG* presetConfig);

/**
 * \brief Initializes the encoder with the given configuration.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncInitializeEncoder                  (void* encoder, ROCENC_INITIALIZE_PARAMS* createEncodeParams);

/**
 * \brief Allocates an input buffer.  This is the host-write (slow) path; the
 *        zero-copy path is rocEncRegisterResource() / rocEncMapInputResource().
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncCreateInputBuffer                  (void* encoder, ROCENC_CREATE_INPUT_BUFFER* createInputBufferParams);

/**
 * \brief Destroys an input buffer.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncDestroyInputBuffer                 (void* encoder, ROCENC_INPUT_PTR inputBuffer);

/**
 * \brief Sets the HIP streams used for input and output processing.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncSetIOHipStreams                    (void* encoder, ROCENC_HIPSTREAM_PTR inputStream, ROCENC_HIPSTREAM_PTR outputStream);

/**
 * \brief Allocates an output bitstream buffer.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncCreateBitstreamBuffer              (void* encoder, ROCENC_CREATE_BITSTREAM_BUFFER* createBitstreamBufferParams);

/**
 * \brief Destroys an output bitstream buffer.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncDestroyBitstreamBuffer             (void* encoder, ROCENC_OUTPUT_PTR bitstreamBuffer);

/**
 * \brief Submits a picture for encoding.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncEncodePicture                      (void* encoder, ROCENC_PIC_PARAMS* encodePicParams);

/**
 * \brief Locks an output bitstream buffer and returns the encoded data.
 *        With doNotWait = 0 this call blocks until the encode completes.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncLockBitstream                      (void* encoder, ROCENC_LOCK_BITSTREAM* lockBitstreamBufferParams);

/**
 * \brief Unlocks an output bitstream buffer.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncUnlockBitstream                    (void* encoder, ROCENC_OUTPUT_PTR bitstreamBuffer);

/**
 * \brief Restores a previously saved encoder state.
 * \note No VCN equivalent.  Always returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncRestoreEncoderState                (void* encoder, ROCENC_RESTORE_ENCODER_STATE_PARAMS* restoreState);

/**
 * \brief Locks an input buffer and returns a host-writable pointer.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncLockInputBuffer                    (void* encoder, ROCENC_LOCK_INPUT_BUFFER* lockInputBufferParams);

/**
 * \brief Unlocks an input buffer.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncUnlockInputBuffer                  (void* encoder, ROCENC_INPUT_PTR inputBuffer);

/**
 * \brief Retrieves encode statistics.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncGetEncodeStats                     (void* encoder, ROCENC_STAT* encodeStats);

/**
 * \brief Retrieves the sequence and picture headers for the current session.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncGetSequenceParams                  (void* encoder, ROCENC_SEQUENCE_PARAM_PAYLOAD* sequenceParamPayload);

/**
 * \brief Retrieves the sequence and picture headers for a given configuration.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncGetSequenceParamEx                 (void* encoder, ROCENC_INITIALIZE_PARAMS* encInitParams, ROCENC_SEQUENCE_PARAM_PAYLOAD* sequenceParamPayload);

/**
 * \brief Registers an event for asynchronous completion notification.
 * \note No VCN equivalent.  Always returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncRegisterAsyncEvent                 (void* encoder, ROCENC_EVENT_PARAMS* eventParams);

/**
 * \brief Unregisters an asynchronous completion event.
 * \note No VCN equivalent.  Always returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncUnregisterAsyncEvent               (void* encoder, ROCENC_EVENT_PARAMS* eventParams);

/**
 * \brief Maps a registered resource for use as an encoder input buffer.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncMapInputResource                   (void* encoder, ROCENC_MAP_INPUT_RESOURCE* mapInputResParams);

/**
 * \brief Unmaps a previously mapped input resource.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncUnmapInputResource                 (void* encoder, ROCENC_INPUT_PTR mappedInputBuffer);

/**
 * \brief Destroys the encoder session and releases all its resources.
 */
rocEncStatus ROCENCAPI rocEncDestroyEncoder                     (void* encoder);

/**
 * \brief Invalidates reference frames that were lost at the client.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncInvalidateRefFrames                (void* encoder, uint64_t invalidRefFrameTimeStamp);

/**
 * \brief Opens an encoding session on the given device.
 */
rocEncStatus ROCENCAPI rocEncOpenEncodeSessionEx                (ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS* openSessionExParams, void** encoder);

/**
 * \brief Registers an application-owned resource with the encoder.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncRegisterResource                   (void* encoder, ROCENC_REGISTER_RESOURCE* registerResParams);

/**
 * \brief Unregisters a previously registered resource.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncUnregisterResource                 (void* encoder, ROCENC_REGISTERED_PTR registeredResource);

/**
 * \brief Reconfigures an existing encoder session.
 * \note Phase 1: returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncReconfigureEncoder                 (void* encoder, ROCENC_RECONFIGURE_PARAMS* reInitEncodeParams);

/**
 * \brief Allocates an output motion vector buffer for ME only mode.
 * \note No VCN equivalent.  Always returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncCreateMVBuffer                     (void* encoder, ROCENC_CREATE_MV_BUFFER* createMVBufferParams);

/**
 * \brief Destroys an output motion vector buffer.
 * \note No VCN equivalent.  Always returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncDestroyMVBuffer                    (void* encoder, ROCENC_OUTPUT_PTR mvBuffer);

/**
 * \brief Runs motion estimation without encoding.
 * \note No VCN equivalent.  Always returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncRunMotionEstimationOnly            (void* encoder, ROCENC_MEONLY_PARAMS* meOnlyParams);

/**
 * \brief Returns the highest API version supported by this library.
 */
rocEncStatus ROCENCAPI rocEncodeAPIGetMaxSupportedVersion       (uint32_t* version);

/**
 * \brief Returns the detail of the last error encountered on this session.
 */
const char * ROCENCAPI rocEncGetLastErrorString                 (void* encoder);

/**
 * \brief Submits an input picture for lookahead.
 * \note No VCN equivalent.  Always returns ::ROCENC_ERR_UNSUPPORTED_DEVICE.
 */
rocEncStatus ROCENCAPI rocEncLookaheadPicture                   (void* encoder, ROCENC_LOOKAHEAD_PIC_PARAMS* lookaheadParams);

/// \cond API PFN
typedef rocEncStatus (ROCENCAPI* PROCENCOPENENCODESESSION)         (void* device, uint32_t deviceType, void** encoder);
typedef rocEncStatus (ROCENCAPI* PROCENCGETENCODECODECCOUNT)       (void* encoder, uint32_t* codecCount);
typedef rocEncStatus (ROCENCAPI* PROCENCGETENCODECODECS)           (void* encoder, ROCENC_CODEC* codecs, uint32_t codecArraySize, uint32_t* codecCount);
typedef rocEncStatus (ROCENCAPI* PROCENCGETENCODEPROFILECOUNT)     (void* encoder, ROCENC_CODEC codec, uint32_t* profileCount);
typedef rocEncStatus (ROCENCAPI* PROCENCGETENCODEPROFILES)         (void* encoder, ROCENC_CODEC codec, ROCENC_PROFILE* profiles, uint32_t profileArraySize, uint32_t* profileCount);
typedef rocEncStatus (ROCENCAPI* PROCENCGETINPUTFORMATCOUNT)       (void* encoder, ROCENC_CODEC codec, uint32_t* inputFmtCount);
typedef rocEncStatus (ROCENCAPI* PROCENCGETINPUTFORMATS)           (void* encoder, ROCENC_CODEC codec, ROCENC_BUFFER_FORMAT* inputFmts, uint32_t inputFmtArraySize, uint32_t* inputFmtCount);
typedef rocEncStatus (ROCENCAPI* PROCENCGETENCODECAPS)             (void* encoder, ROCENC_CODEC codec, ROCENC_CAPS_PARAM* capsParam, int* capsVal);
typedef rocEncStatus (ROCENCAPI* PROCENCGETENCODEPRESETCOUNT)      (void* encoder, ROCENC_CODEC codec, uint32_t* presetCount);
typedef rocEncStatus (ROCENCAPI* PROCENCGETENCODEPRESETS)          (void* encoder, ROCENC_CODEC codec, ROCENC_PRESET* presets, uint32_t presetArraySize, uint32_t* presetCount);
typedef rocEncStatus (ROCENCAPI* PROCENCGETENCODEPRESETCONFIG)     (void* encoder, ROCENC_CODEC codec, ROCENC_PRESET preset, ROCENC_PRESET_CONFIG* presetConfig);
typedef rocEncStatus (ROCENCAPI* PROCENCGETENCODEPRESETCONFIGEX)   (void* encoder, ROCENC_CODEC codec, ROCENC_PRESET preset, ROCENC_TUNING_INFO tuningInfo, ROCENC_PRESET_CONFIG* presetConfig);
typedef rocEncStatus (ROCENCAPI* PROCENCINITIALIZEENCODER)         (void* encoder, ROCENC_INITIALIZE_PARAMS* createEncodeParams);
typedef rocEncStatus (ROCENCAPI* PROCENCCREATEINPUTBUFFER)         (void* encoder, ROCENC_CREATE_INPUT_BUFFER* createInputBufferParams);
typedef rocEncStatus (ROCENCAPI* PROCENCDESTROYINPUTBUFFER)        (void* encoder, ROCENC_INPUT_PTR inputBuffer);
typedef rocEncStatus (ROCENCAPI* PROCENCCREATEBITSTREAMBUFFER)     (void* encoder, ROCENC_CREATE_BITSTREAM_BUFFER* createBitstreamBufferParams);
typedef rocEncStatus (ROCENCAPI* PROCENCDESTROYBITSTREAMBUFFER)    (void* encoder, ROCENC_OUTPUT_PTR bitstreamBuffer);
typedef rocEncStatus (ROCENCAPI* PROCENCENCODEPICTURE)             (void* encoder, ROCENC_PIC_PARAMS* encodePicParams);
typedef rocEncStatus (ROCENCAPI* PROCENCLOCKBITSTREAM)             (void* encoder, ROCENC_LOCK_BITSTREAM* lockBitstreamBufferParams);
typedef rocEncStatus (ROCENCAPI* PROCENCUNLOCKBITSTREAM)           (void* encoder, ROCENC_OUTPUT_PTR bitstreamBuffer);
typedef rocEncStatus (ROCENCAPI* PROCENCLOCKINPUTBUFFER)           (void* encoder, ROCENC_LOCK_INPUT_BUFFER* lockInputBufferParams);
typedef rocEncStatus (ROCENCAPI* PROCENCUNLOCKINPUTBUFFER)         (void* encoder, ROCENC_INPUT_PTR inputBuffer);
typedef rocEncStatus (ROCENCAPI* PROCENCGETENCODESTATS)            (void* encoder, ROCENC_STAT* encodeStats);
typedef rocEncStatus (ROCENCAPI* PROCENCGETSEQUENCEPARAMS)         (void* encoder, ROCENC_SEQUENCE_PARAM_PAYLOAD* sequenceParamPayload);
typedef rocEncStatus (ROCENCAPI* PROCENCREGISTERASYNCEVENT)        (void* encoder, ROCENC_EVENT_PARAMS* eventParams);
typedef rocEncStatus (ROCENCAPI* PROCENCUNREGISTERASYNCEVENT)      (void* encoder, ROCENC_EVENT_PARAMS* eventParams);
typedef rocEncStatus (ROCENCAPI* PROCENCMAPINPUTRESOURCE)          (void* encoder, ROCENC_MAP_INPUT_RESOURCE* mapInputResParams);
typedef rocEncStatus (ROCENCAPI* PROCENCUNMAPINPUTRESOURCE)        (void* encoder, ROCENC_INPUT_PTR mappedInputBuffer);
typedef rocEncStatus (ROCENCAPI* PROCENCDESTROYENCODER)            (void* encoder);
typedef rocEncStatus (ROCENCAPI* PROCENCINVALIDATEREFFRAMES)       (void* encoder, uint64_t invalidRefFrameTimeStamp);
typedef rocEncStatus (ROCENCAPI* PROCENCOPENENCODESESSIONEX)       (ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS* openSessionExParams, void** encoder);
typedef rocEncStatus (ROCENCAPI* PROCENCREGISTERRESOURCE)          (void* encoder, ROCENC_REGISTER_RESOURCE* registerResParams);
typedef rocEncStatus (ROCENCAPI* PROCENCUNREGISTERRESOURCE)        (void* encoder, ROCENC_REGISTERED_PTR registeredRes);
typedef rocEncStatus (ROCENCAPI* PROCENCRECONFIGUREENCODER)        (void* encoder, ROCENC_RECONFIGURE_PARAMS* reInitEncodeParams);

typedef rocEncStatus (ROCENCAPI* PROCENCCREATEMVBUFFER)            (void* encoder, ROCENC_CREATE_MV_BUFFER* createMVBufferParams);
typedef rocEncStatus (ROCENCAPI* PROCENCDESTROYMVBUFFER)           (void* encoder, ROCENC_OUTPUT_PTR mvBuffer);
typedef rocEncStatus (ROCENCAPI* PROCENCRUNMOTIONESTIMATIONONLY)   (void* encoder, ROCENC_MEONLY_PARAMS* meOnlyParams);
typedef const char * (ROCENCAPI* PROCENCGETLASTERROR)              (void* encoder);
typedef rocEncStatus (ROCENCAPI* PROCENCSETIOHIPSTREAMS)           (void* encoder, ROCENC_HIPSTREAM_PTR inputStream, ROCENC_HIPSTREAM_PTR outputStream);
typedef rocEncStatus (ROCENCAPI* PROCENCGETSEQUENCEPARAMEX)        (void* encoder, ROCENC_INITIALIZE_PARAMS* encInitParams, ROCENC_SEQUENCE_PARAM_PAYLOAD* sequenceParamPayload);
typedef rocEncStatus (ROCENCAPI* PROCENCRESTOREENCODERSTATE)       (void* encoder, ROCENC_RESTORE_ENCODER_STATE_PARAMS* restoreState);
typedef rocEncStatus (ROCENCAPI* PROCENCLOOKAHEADPICTURE)          (void* encoder, ROCENC_LOOKAHEAD_PIC_PARAMS* lookaheadParams);
/// \endcond

/** @} */ /* END ENCODE_FUNC */

/**
 * \ingroup ENCODER_STRUCTURE
 * ROCENCODE_API_FUNCTION_LIST
 *
 * Field order is fixed: it is the ABI.  New entry points are appended, never
 * inserted.  No entry is ever NULL on a successfully created instance.
 */
typedef struct _ROCENCODE_API_FUNCTION_LIST
{
    uint32_t                        version;                           /**< [in]: Client should pass ROCENCODE_API_FUNCTION_LIST_VER.  */
    uint32_t                        reserved;                          /**< [in]: Reserved and should be set to 0.                     */
    PROCENCOPENENCODESESSION        rocEncOpenEncodeSession;           /**< [out]: ::rocEncOpenEncodeSession()        */
    PROCENCGETENCODECODECCOUNT      rocEncGetEncodeCodecCount;         /**< [out]: ::rocEncGetEncodeCodecCount()      */
    PROCENCGETENCODEPROFILECOUNT    rocEncGetEncodeProfileCount;       /**< [out]: ::rocEncGetEncodeProfileCount()    */
    PROCENCGETENCODEPROFILES        rocEncGetEncodeProfiles;           /**< [out]: ::rocEncGetEncodeProfiles()        */
    PROCENCGETENCODECODECS          rocEncGetEncodeCodecs;             /**< [out]: ::rocEncGetEncodeCodecs()          */
    PROCENCGETINPUTFORMATCOUNT      rocEncGetInputFormatCount;         /**< [out]: ::rocEncGetInputFormatCount()      */
    PROCENCGETINPUTFORMATS          rocEncGetInputFormats;             /**< [out]: ::rocEncGetInputFormats()          */
    PROCENCGETENCODECAPS            rocEncGetEncodeCaps;               /**< [out]: ::rocEncGetEncodeCaps()            */
    PROCENCGETENCODEPRESETCOUNT     rocEncGetEncodePresetCount;        /**< [out]: ::rocEncGetEncodePresetCount()     */
    PROCENCGETENCODEPRESETS         rocEncGetEncodePresets;            /**< [out]: ::rocEncGetEncodePresets()         */
    PROCENCGETENCODEPRESETCONFIG    rocEncGetEncodePresetConfig;       /**< [out]: ::rocEncGetEncodePresetConfig()    */
    PROCENCINITIALIZEENCODER        rocEncInitializeEncoder;           /**< [out]: ::rocEncInitializeEncoder()        */
    PROCENCCREATEINPUTBUFFER        rocEncCreateInputBuffer;           /**< [out]: ::rocEncCreateInputBuffer()        */
    PROCENCDESTROYINPUTBUFFER       rocEncDestroyInputBuffer;          /**< [out]: ::rocEncDestroyInputBuffer()       */
    PROCENCCREATEBITSTREAMBUFFER    rocEncCreateBitstreamBuffer;       /**< [out]: ::rocEncCreateBitstreamBuffer()    */
    PROCENCDESTROYBITSTREAMBUFFER   rocEncDestroyBitstreamBuffer;      /**< [out]: ::rocEncDestroyBitstreamBuffer()   */
    PROCENCENCODEPICTURE            rocEncEncodePicture;               /**< [out]: ::rocEncEncodePicture()            */
    PROCENCLOCKBITSTREAM            rocEncLockBitstream;               /**< [out]: ::rocEncLockBitstream()            */
    PROCENCUNLOCKBITSTREAM          rocEncUnlockBitstream;             /**< [out]: ::rocEncUnlockBitstream()          */
    PROCENCLOCKINPUTBUFFER          rocEncLockInputBuffer;             /**< [out]: ::rocEncLockInputBuffer()          */
    PROCENCUNLOCKINPUTBUFFER        rocEncUnlockInputBuffer;           /**< [out]: ::rocEncUnlockInputBuffer()        */
    PROCENCGETENCODESTATS           rocEncGetEncodeStats;              /**< [out]: ::rocEncGetEncodeStats()           */
    PROCENCGETSEQUENCEPARAMS        rocEncGetSequenceParams;           /**< [out]: ::rocEncGetSequenceParams()        */
    PROCENCREGISTERASYNCEVENT       rocEncRegisterAsyncEvent;          /**< [out]: ::rocEncRegisterAsyncEvent()       */
    PROCENCUNREGISTERASYNCEVENT     rocEncUnregisterAsyncEvent;        /**< [out]: ::rocEncUnregisterAsyncEvent()     */
    PROCENCMAPINPUTRESOURCE         rocEncMapInputResource;            /**< [out]: ::rocEncMapInputResource()         */
    PROCENCUNMAPINPUTRESOURCE       rocEncUnmapInputResource;          /**< [out]: ::rocEncUnmapInputResource()       */
    PROCENCDESTROYENCODER           rocEncDestroyEncoder;              /**< [out]: ::rocEncDestroyEncoder()           */
    PROCENCINVALIDATEREFFRAMES      rocEncInvalidateRefFrames;         /**< [out]: ::rocEncInvalidateRefFrames()      */
    PROCENCOPENENCODESESSIONEX      rocEncOpenEncodeSessionEx;         /**< [out]: ::rocEncOpenEncodeSessionEx()      */
    PROCENCREGISTERRESOURCE         rocEncRegisterResource;            /**< [out]: ::rocEncRegisterResource()         */
    PROCENCUNREGISTERRESOURCE       rocEncUnregisterResource;          /**< [out]: ::rocEncUnregisterResource()       */
    PROCENCRECONFIGUREENCODER       rocEncReconfigureEncoder;          /**< [out]: ::rocEncReconfigureEncoder()       */
    void*                           reserved1;
    PROCENCCREATEMVBUFFER           rocEncCreateMVBuffer;              /**< [out]: ::rocEncCreateMVBuffer()           */
    PROCENCDESTROYMVBUFFER          rocEncDestroyMVBuffer;             /**< [out]: ::rocEncDestroyMVBuffer()          */
    PROCENCRUNMOTIONESTIMATIONONLY  rocEncRunMotionEstimationOnly;     /**< [out]: ::rocEncRunMotionEstimationOnly()  */
    PROCENCGETLASTERROR             rocEncGetLastErrorString;          /**< [out]: ::rocEncGetLastErrorString()       */
    PROCENCSETIOHIPSTREAMS          rocEncSetIOHipStreams;             /**< [out]: ::rocEncSetIOHipStreams()          */
    PROCENCGETENCODEPRESETCONFIGEX  rocEncGetEncodePresetConfigEx;     /**< [out]: ::rocEncGetEncodePresetConfigEx()  */
    PROCENCGETSEQUENCEPARAMEX       rocEncGetSequenceParamEx;          /**< [out]: ::rocEncGetSequenceParamEx()       */
    PROCENCRESTOREENCODERSTATE      rocEncRestoreEncoderState;         /**< [out]: ::rocEncRestoreEncoderState()      */
    PROCENCLOOKAHEADPICTURE         rocEncLookaheadPicture;            /**< [out]: ::rocEncLookaheadPicture()         */
    void*                           reserved2[275];                    /**< [in]:  Reserved and must be set to NULL   */
} ROCENCODE_API_FUNCTION_LIST;

/** Macro for constructing the version field of ::_ROCENCODE_API_FUNCTION_LIST. */
#define ROCENCODE_API_FUNCTION_LIST_VER ROCENCAPI_STRUCT_VERSION(2)

/**
 * \ingroup ENCODE_FUNC
 * Entry point to the rocEncode interface.  Populates functionList with
 * pointers to every API routine.
 *
 * \param [out] functionList
 *
 * \return
 * ::ROCENC_SUCCESS \n
 * ::ROCENC_ERR_INVALID_PTR \n
 * ::ROCENC_ERR_INVALID_VERSION
 */
rocEncStatus ROCENCAPI rocEncodeAPICreateInstance(ROCENCODE_API_FUNCTION_LIST* functionList);

/* =========================================================================================
 * ABI size checks.
 *
 * Every value below was measured on x86-64 by compiling the corresponding
 * structure and printing sizeof(); they are not hand-computed.  These sizes
 * are the published ABI.  A mismatch here is a bug in the structure
 * definition, not a reason to change the constant.
 * ========================================================================================= */
#if defined(__cplusplus)
static_assert(sizeof(ROCENC_CODEC) == 4, "ABI: sizeof(ROCENC_CODEC) must be 4");
static_assert(sizeof(ROCENC_PROFILE) == 4, "ABI: sizeof(ROCENC_PROFILE) must be 4");
static_assert(sizeof(ROCENC_PRESET) == 4, "ABI: sizeof(ROCENC_PRESET) must be 4");
#if !defined(_WIN32)
static_assert(sizeof(ROCENC_RECT) == 16, "ABI: sizeof(ROCENC_RECT) must be 16");
#endif
static_assert(sizeof(ROCENC_CAPS_PARAM) == 256, "ABI: sizeof(ROCENC_CAPS_PARAM) must be 256");
static_assert(sizeof(ROCENC_RESTORE_ENCODER_STATE_PARAMS) == 800, "ABI: sizeof(ROCENC_RESTORE_ENCODER_STATE_PARAMS) must be 800");
static_assert(sizeof(ROCENC_OUTPUT_STATS_BLOCK) == 64, "ABI: sizeof(ROCENC_OUTPUT_STATS_BLOCK) must be 64");
static_assert(sizeof(ROCENC_OUTPUT_STATS_ROW) == 64, "ABI: sizeof(ROCENC_OUTPUT_STATS_ROW) must be 64");
static_assert(sizeof(ROCENC_ENCODE_OUT_PARAMS) == 256, "ABI: sizeof(ROCENC_ENCODE_OUT_PARAMS) must be 256");
static_assert(sizeof(ROCENC_LOOKAHEAD_PIC_PARAMS) == 792, "ABI: sizeof(ROCENC_LOOKAHEAD_PIC_PARAMS) must be 792");
static_assert(sizeof(ROCENC_CREATE_INPUT_BUFFER) == 776, "ABI: sizeof(ROCENC_CREATE_INPUT_BUFFER) must be 776");
static_assert(sizeof(ROCENC_CREATE_BITSTREAM_BUFFER) == 776, "ABI: sizeof(ROCENC_CREATE_BITSTREAM_BUFFER) must be 776");
static_assert(sizeof(ROCENC_MVECTOR) == 4, "ABI: sizeof(ROCENC_MVECTOR) must be 4");
static_assert(sizeof(ROCENC_H264_MV_DATA) == 24, "ABI: sizeof(ROCENC_H264_MV_DATA) must be 24");
static_assert(sizeof(ROCENC_HEVC_MV_DATA) == 20, "ABI: sizeof(ROCENC_HEVC_MV_DATA) must be 20");
static_assert(sizeof(ROCENC_CREATE_MV_BUFFER) == 1544, "ABI: sizeof(ROCENC_CREATE_MV_BUFFER) must be 1544");
static_assert(sizeof(ROCENC_QP) == 12, "ABI: sizeof(ROCENC_QP) must be 12");
static_assert(sizeof(ROCENC_RC_PARAMS) == 128, "ABI: sizeof(ROCENC_RC_PARAMS) must be 128");
static_assert(sizeof(ROCENC_CLOCK_TIMESTAMP_SET) == 8, "ABI: sizeof(ROCENC_CLOCK_TIMESTAMP_SET) must be 8");
static_assert(sizeof(ROCENC_TIME_CODE) == 28, "ABI: sizeof(ROCENC_TIME_CODE) must be 28");
static_assert(sizeof(ROCENC_CONFIG_H264_VUI_PARAMETERS) == 112, "ABI: sizeof(ROCENC_CONFIG_H264_VUI_PARAMETERS) must be 112");
static_assert(sizeof(ROCENC_CONFIG_HEVC_VUI_PARAMETERS) == 112, "ABI: sizeof(ROCENC_CONFIG_HEVC_VUI_PARAMETERS) must be 112");
static_assert(sizeof(ROCENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE) == 16, "ABI: sizeof(ROCENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE) must be 16");
static_assert(sizeof(ROCENC_EXTERNAL_ME_HINT) == 4, "ABI: sizeof(ROCENC_EXTERNAL_ME_HINT) must be 4");
static_assert(sizeof(ROCENC_EXTERNAL_ME_SB_HINT) == 6, "ABI: sizeof(ROCENC_EXTERNAL_ME_SB_HINT) must be 6");
static_assert(sizeof(ROCENC_CONFIG_H264) == 1792, "ABI: sizeof(ROCENC_CONFIG_H264) must be 1792");
static_assert(sizeof(ROCENC_CONFIG_HEVC) == 1560, "ABI: sizeof(ROCENC_CONFIG_HEVC) must be 1560");
static_assert(sizeof(ROCENC_FILM_GRAIN_PARAMS_AV1) == 156, "ABI: sizeof(ROCENC_FILM_GRAIN_PARAMS_AV1) must be 156");
static_assert(sizeof(ROCENC_CONFIG_AV1) == 1552, "ABI: sizeof(ROCENC_CONFIG_AV1) must be 1552");
static_assert(sizeof(ROCENC_CONFIG_H264_MEONLY) == 1536, "ABI: sizeof(ROCENC_CONFIG_H264_MEONLY) must be 1536");
static_assert(sizeof(ROCENC_CONFIG_HEVC_MEONLY) == 1536, "ABI: sizeof(ROCENC_CONFIG_HEVC_MEONLY) must be 1536");
static_assert(sizeof(ROCENC_CODEC_CONFIG) == 1792, "ABI: sizeof(ROCENC_CODEC_CONFIG) must be 1792");
static_assert(sizeof(ROCENC_CONFIG) == 3584, "ABI: sizeof(ROCENC_CONFIG) must be 3584");
static_assert(sizeof(ROCENC_INITIALIZE_PARAMS) == 1808, "ABI: sizeof(ROCENC_INITIALIZE_PARAMS) must be 1808");
static_assert(sizeof(ROCENC_RECONFIGURE_PARAMS) == 1824, "ABI: sizeof(ROCENC_RECONFIGURE_PARAMS) must be 1824");
static_assert(sizeof(ROCENC_PRESET_CONFIG) == 5128, "ABI: sizeof(ROCENC_PRESET_CONFIG) must be 5128");
static_assert(sizeof(ROCENC_PIC_PARAMS_MVC) == 128, "ABI: sizeof(ROCENC_PIC_PARAMS_MVC) must be 128");
static_assert(sizeof(ROCENC_PIC_PARAMS_H264_EXT) == 128, "ABI: sizeof(ROCENC_PIC_PARAMS_H264_EXT) must be 128");
static_assert(sizeof(ROCENC_SEI_PAYLOAD) == 16, "ABI: sizeof(ROCENC_SEI_PAYLOAD) must be 16");
static_assert(sizeof(ROCENC_PIC_PARAMS_H264) == 1536, "ABI: sizeof(ROCENC_PIC_PARAMS_H264) must be 1536");
static_assert(sizeof(ROCENC_PIC_PARAMS_HEVC) == 1536, "ABI: sizeof(ROCENC_PIC_PARAMS_HEVC) must be 1536");
static_assert(sizeof(ROCENC_PIC_PARAMS_AV1) == 1552, "ABI: sizeof(ROCENC_PIC_PARAMS_AV1) must be 1552");
static_assert(sizeof(ROCENC_CODEC_PIC_PARAMS) == 1552, "ABI: sizeof(ROCENC_CODEC_PIC_PARAMS) must be 1552");
static_assert(sizeof(ROCENC_PIC_PARAMS) == 3360, "ABI: sizeof(ROCENC_PIC_PARAMS) must be 3360");
static_assert(sizeof(ROCENC_MEONLY_PARAMS) == 1552, "ABI: sizeof(ROCENC_MEONLY_PARAMS) must be 1552");
static_assert(sizeof(ROCENC_LOCK_BITSTREAM) == 1552, "ABI: sizeof(ROCENC_LOCK_BITSTREAM) must be 1552");
static_assert(sizeof(ROCENC_LOCK_INPUT_BUFFER) == 1544, "ABI: sizeof(ROCENC_LOCK_INPUT_BUFFER) must be 1544");
static_assert(sizeof(ROCENC_MAP_INPUT_RESOURCE) == 1544, "ABI: sizeof(ROCENC_MAP_INPUT_RESOURCE) must be 1544");
static_assert(sizeof(ROCENC_INPUT_RESOURCE_OPENGL_TEX) == 8, "ABI: sizeof(ROCENC_INPUT_RESOURCE_OPENGL_TEX) must be 8");
static_assert(sizeof(ROCENC_FENCE_POINT_D3D12) == 64, "ABI: sizeof(ROCENC_FENCE_POINT_D3D12) must be 64");
static_assert(sizeof(ROCENC_INPUT_RESOURCE_D3D12) == 272, "ABI: sizeof(ROCENC_INPUT_RESOURCE_D3D12) must be 272");
static_assert(sizeof(ROCENC_OUTPUT_RESOURCE_D3D12) == 272, "ABI: sizeof(ROCENC_OUTPUT_RESOURCE_D3D12) must be 272");
static_assert(sizeof(ROCENC_REGISTER_RESOURCE) == 1536, "ABI: sizeof(ROCENC_REGISTER_RESOURCE) must be 1536");
static_assert(sizeof(ROCENC_STAT) == 1544, "ABI: sizeof(ROCENC_STAT) must be 1544");
static_assert(sizeof(ROCENC_SEQUENCE_PARAM_PAYLOAD) == 1544, "ABI: sizeof(ROCENC_SEQUENCE_PARAM_PAYLOAD) must be 1544");
static_assert(sizeof(ROCENC_EVENT_PARAMS) == 1544, "ABI: sizeof(ROCENC_EVENT_PARAMS) must be 1544");
static_assert(sizeof(ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS) == 1552, "ABI: sizeof(ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS) must be 1552");
static_assert(sizeof(ROCENCODE_API_FUNCTION_LIST) == 2552, "ABI: sizeof(ROCENCODE_API_FUNCTION_LIST) must be 2552");
#endif /* __cplusplus */

#ifdef __cplusplus
}
#endif

#endif /* _ROCENCODEAPI_H_ */
