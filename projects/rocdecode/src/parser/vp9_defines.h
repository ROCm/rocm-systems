/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

#include <stdint.h>

#define VP9_REFS_PER_FRAME          3 // Each inter frame can use up to 3 frames for reference
#define VP9_NUM_REF_FRAMES          8 // Number of frames that can be stored for future reference
#define VP9_MAX_REF_FRAMES          4 // Number of values that can be derived for ref_frame
#define VP9_MAX_SEGMENTS            8 // Number of segments allowed in segmentation map
#define VP9_SEG_LVL_ALT_Q           0 // Index for quantizer segment feature
#define VP9_SEG_LVL_ALT_L           1 // Index for loop filter segment feature
#define VP9_SEG_LVL_REF_FRAME       2 // Index for reference frame segment feature
#define VP9_SEG_LVL_SKIP            3 // Index for skip segment feature
#define VP9_SEG_LVL_MAX             4 // Number of segment features
#define MIN_TILE_WIDTH_B64          4 // Minimum width of a tile in units of superblocks (although tiles on the right hand edge can be narrower)
#define MAX_TILE_WIDTH_B64          64 // Maximum width of a tile in units of superblocks
#define MAX_MODE_LF_DELTAS          2  // Number of different mode types for loop filtering
#define VP9_MAX_LOOP_FILTER         63 // Maximum value used for loop filtering
#define VP9_BUFFER_POOL_MAX_SIZE    10 // Number of frames in buffer pool

typedef enum {
    kVp9KeyFrame        = 0,
    kVp9NonKeyFrame     = 1,
} Vp9FrameType;

typedef enum {
    CS_UNKNOWN =    0, // Unknown (in this case the color space must be signaled outside the VP9 bitstream).
    CS_BT_601 =     1, // Rec. ITU-R BT.601-7
    CS_BT_709 =     2, // Rec. ITU-R BT.709-6
    CS_SMPTE_170 =  3, // SMPTE-170
    CS_SMPTE_240 =  4, // SMPTE-240
    CS_BT_2020 =    5, // Rec. ITU-R BT.2020-2
    CS_RESERVED =   6, // Reserved
    CS_RGB =        7, // sRGB (IEC 61966-2-1)
} Vp9ColorSpace;

typedef enum {
    kStudioSwing    = 0, // Studio video range
    kFullSwing      = 1, // Full video range
} Vp9ColorRange;

typedef enum {
    kVp9IntraFrame     = 0,
    kVp9LastFrame      = 1,
    kVp9GoldenFrame    = 2,
    kVp9AltRefFrame    = 3,
} Vp9RefFrame;

typedef enum {
    kVp9EightTap        = 0,
    kVp9EightTapSmooth  = 1,
    kVp9EightTapSharp   = 2,
    kVp9Bilinear        = 3,
    kVp9Switchable      = 4,
} Vp9InterpolotionFilterType;

typedef struct {
    uint8_t frame_sync_byte_0;
    uint8_t frame_sync_byte_1;
    uint8_t frame_sync_byte_2;
} Vp9FrameSyncCode;

typedef struct {
    uint8_t ten_or_twelve_bit;
    uint8_t bit_depth;
    uint8_t color_space;
    uint8_t color_range;
    uint8_t subsampling_x;
    uint8_t subsampling_y;
    uint8_t reserved_zero;
} Vp9ColorConfig;

typedef struct {
    uint16_t frame_width_minus_1;
    uint16_t frame_height_minus_1;
    uint32_t frame_width;
    uint32_t frame_height;
    uint16_t mi_cols;
    uint16_t mi_rows;
    uint16_t sb64_cols;
    uint16_t sb64_rows;
} Vp9FrameSize;

typedef struct {
    uint8_t  render_and_frame_size_different;
    uint16_t render_width_minus_1;
    uint16_t render_height_minus_1;
    uint32_t render_width;
    uint32_t render_height;
} Vp9RenderSize;

typedef struct {
    uint8_t loop_filter_level;
    uint8_t loop_filter_sharpness;
    uint8_t loop_filter_delta_enabled;
    uint8_t loop_filter_delta_update;
    uint8_t update_ref_delta[4];
    int8_t  loop_filter_ref_deltas[4];
    uint8_t update_mode_delta[2];
    int8_t  loop_filter_mode_deltas[2];
} Vp9LoopFilterParams;

typedef struct {
    uint8_t base_q_idx;
    int8_t  delta_q_y_dc;
    int8_t  delta_q_uv_dc;
    int8_t  delta_q_uv_ac;
    uint8_t lossless;
} Vp9QuantizationParams;

typedef struct {
    uint8_t segmentation_enabled;
    uint8_t segmentation_update_map;
    uint8_t segmentation_tree_probs[7];
    uint8_t segmentation_temporal_update;
    uint8_t segmentation_pred_prob[3];
    uint8_t segmentation_update_data;
    uint8_t segmentation_abs_or_delta_update;
    uint8_t feature_enabled[VP9_MAX_SEGMENTS][VP9_SEG_LVL_MAX];
    int16_t feature_data[VP9_MAX_SEGMENTS][VP9_SEG_LVL_MAX];
} Vp9SegmentationParams;

typedef struct {
    uint16_t min_log2_tile_cols; // minLog2TileCols
    uint16_t max_log2_tile_cols; // maxLog2TileCols
    uint8_t  tile_cols_log2;
    uint8_t  tile_rows_log2;
} Vp9TileInfo;

typedef struct {
    uint8_t frame_marker;
    uint8_t profile_low_bit;
    uint8_t profile_high_bit;
    uint8_t profile;
    uint8_t reserved_zero;
    uint8_t show_existing_frame;
    uint8_t frame_to_show_map_idx;
    uint8_t frame_type;
    uint8_t show_frame;
    uint8_t error_resilient_mode;
    uint8_t intra_only;
    uint8_t reset_frame_context;
    Vp9FrameSyncCode frame_sync_code;
    Vp9ColorConfig color_config;
    Vp9FrameSize frame_size;
    Vp9RenderSize render_size;
    uint8_t refresh_frame_flags;
    uint8_t ref_frame_idx[VP9_REFS_PER_FRAME];
    uint8_t ref_frame_sign_bias[VP9_MAX_REF_FRAMES];
    uint8_t allow_high_precision_mv;
    uint8_t is_filter_switchable;
    uint8_t raw_interpolation_filter;
    uint8_t interpolation_filter;
    uint8_t refresh_frame_context;
    uint8_t frame_parallel_decoding_mode;
    uint8_t frame_context_idx;
    Vp9LoopFilterParams loop_filter_params;
    Vp9QuantizationParams quantization_params;
    Vp9SegmentationParams segmentation_params;
    Vp9TileInfo tile_info;
    uint16_t header_size_in_bytes;
} Vp9UncompressedHeader;