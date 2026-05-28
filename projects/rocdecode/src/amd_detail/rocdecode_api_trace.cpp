/*
Copyright (c) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include "../../api/amd_detail/rocdecode_api_trace.h"

#if defined(ROCDECODE_ROCPROFILER_REGISTER) && ROCDECODE_ROCPROFILER_REGISTER > 0
#include <rocprofiler-register/rocprofiler-register.h>

#define ROCDECODE_ROCP_REG_VERSION \
  ROCPROFILER_REGISTER_COMPUTE_VERSION_3(ROCDECODE_ROCP_REG_VERSION_MAJOR, ROCDECODE_ROCP_REG_VERSION_MINOR, \
                                         ROCDECODE_ROCP_REG_VERSION_PATCH)

ROCPROFILER_REGISTER_DEFINE_IMPORT(rocdecode, ROCDECODE_ROCP_REG_VERSION)
#elif !defined(ROCDECODE_ROCPROFILER_REGISTER)
#define ROCDECODE_ROCPROFILER_REGISTER 0
#endif

namespace rocdecode {
rocDecStatus ROCDECAPI rocDecCreateVideoParser(RocdecVideoParser *parser_handle, RocdecParserParams *params);
rocDecStatus ROCDECAPI rocDecParseVideoData(RocdecVideoParser parser_handle, RocdecSourceDataPacket *packet);
rocDecStatus ROCDECAPI rocDecDestroyVideoParser(RocdecVideoParser parser_handle);
rocDecStatus ROCDECAPI rocDecCreateDecoder(rocDecDecoderHandle *decoder_handle, RocDecoderCreateInfo *decoder_create_info);
rocDecStatus ROCDECAPI rocDecDestroyDecoder(rocDecDecoderHandle decoder_handle);
rocDecStatus ROCDECAPI rocDecGetDecoderCaps(RocdecDecodeCaps *decode_caps);
rocDecStatus ROCDECAPI rocDecDecodeFrame(rocDecDecoderHandle decoder_handle, RocdecPicParams *pic_params);
rocDecStatus ROCDECAPI rocDecGetDecodeStatus(rocDecDecoderHandle decoder_handle, int pic_idx, RocdecDecodeStatus *decode_status);
rocDecStatus ROCDECAPI rocDecReconfigureDecoder(rocDecDecoderHandle decoder_handle, RocdecReconfigureDecoderInfo *reconfig_params);
rocDecStatus ROCDECAPI rocDecGetVideoFrame(rocDecDecoderHandle decoder_handle, int pic_idx, void *dev_mem_ptr[3], uint32_t *horizontal_pitch, RocdecProcParams *vid_postproc_params);
const char *ROCDECAPI rocDecGetErrorName(rocDecStatus rocdec_status);
rocDecStatus ROCDECAPI rocDecCreateBitstreamReader(RocdecBitstreamReader *bs_reader_handle, const char *input_file_path);
rocDecStatus ROCDECAPI rocDecGetBitstreamCodecType(RocdecBitstreamReader bs_reader_handle, rocDecVideoCodec *codec_type);
rocDecStatus ROCDECAPI rocDecGetBitstreamBitDepth(RocdecBitstreamReader bs_reader_handle, int *bit_depth);
rocDecStatus ROCDECAPI rocDecGetBitstreamPicData(RocdecBitstreamReader bs_reader_handle, uint8_t **pic_data, int *pic_size, int64_t *pts);
rocDecStatus ROCDECAPI rocDecDestroyBitstreamReader(RocdecBitstreamReader bs_reader_handle);
}

namespace rocdecode {
namespace {
void UpdateDispatchTable(RocDecodeDispatchTable* ptr_dispatch_table) {
    ptr_dispatch_table->size = sizeof(RocDecodeDispatchTable);
    ptr_dispatch_table->pfn_rocdec_create_video_parser = rocdecode::rocDecCreateVideoParser;
    ptr_dispatch_table->pfn_rocdec_parse_video_data = rocdecode::rocDecParseVideoData;
    ptr_dispatch_table->pfn_rocdec_destroy_video_parser = rocdecode::rocDecDestroyVideoParser;
    ptr_dispatch_table->pfn_rocdec_create_decoder = rocdecode::rocDecCreateDecoder;
    ptr_dispatch_table->pfn_rocdec_destroy_decoder = rocdecode::rocDecDestroyDecoder;
    ptr_dispatch_table->pfn_rocdec_get_gecoder_caps = rocdecode::rocDecGetDecoderCaps;
    ptr_dispatch_table->pfn_rocdec_decode_frame = rocdecode::rocDecDecodeFrame;
    ptr_dispatch_table->pfn_rocdec_get_decode_status = rocdecode::rocDecGetDecodeStatus;
    ptr_dispatch_table->pfn_rocdec_reconfigure_decoder = rocdecode::rocDecReconfigureDecoder;
    ptr_dispatch_table->pfn_rocdec_get_video_frame = rocdecode::rocDecGetVideoFrame;
    ptr_dispatch_table->pfn_rocdec_get_error_name = rocdecode::rocDecGetErrorName;
    ptr_dispatch_table->pfn_rocdec_create_bitstream_reader = rocdecode::rocDecCreateBitstreamReader;
    ptr_dispatch_table->pfn_rocdec_get_bitstream_codec_type = rocdecode::rocDecGetBitstreamCodecType;
    ptr_dispatch_table->pfn_rocdec_get_bitstream_bit_depth = rocdecode::rocDecGetBitstreamBitDepth;
    ptr_dispatch_table->pfn_rocdec_get_bitstream_pic_data = rocdecode::rocDecGetBitstreamPicData;
    ptr_dispatch_table->pfn_rocdec_destroy_bitstream_reader = rocdecode::rocDecDestroyBitstreamReader;
}

#if ROCDECODE_ROCPROFILER_REGISTER > 0
template <typename Tp> struct dispatch_table_info;

#define ROCDECODE_DEFINE_DISPATCH_TABLE_INFO(TYPE, NAME) \
template <> struct dispatch_table_info<TYPE> { \
    static constexpr auto name = #NAME; \
    static constexpr auto version = ROCDECODE_ROCP_REG_VERSION; \
    static constexpr auto import_func = &ROCPROFILER_REGISTER_IMPORT_FUNC(NAME); \
};

ROCDECODE_DEFINE_DISPATCH_TABLE_INFO(RocDecodeDispatchTable, rocdecode)
#endif

template <typename Tp> void ToolInit(Tp* table) {
#if ROCDECODE_ROCPROFILER_REGISTER > 0
    auto table_array = std::array<void*, 1>{static_cast<void*>(table)};
    auto lib_id = rocprofiler_register_library_indentifier_t{};
    rocprofiler_register_library_api_table(
        dispatch_table_info<Tp>::name, dispatch_table_info<Tp>::import_func,
        dispatch_table_info<Tp>::version, table_array.data(), table_array.size(), &lib_id);
#else
    (void)table;
#endif
}

template <typename Tp> Tp& GetDispatchTableImpl() {
    static auto dispatch_table = Tp{};
    // Update all function pointers to reference the runtime implementation functions of rocDecode.
    UpdateDispatchTable(&dispatch_table);
    // The profiler registration process may encapsulate the function pointers.
    ToolInit(&dispatch_table);
    return dispatch_table;
}
} //namespace

const RocDecodeDispatchTable* GetRocDecodeDispatchTable() {
    static auto* rocdecode_dispatch_table = &GetDispatchTableImpl<RocDecodeDispatchTable>();
    return rocdecode_dispatch_table;
}
} //namespace rocdecode

#if !defined(_WIN32)
constexpr auto ComputeTableOffset(size_t num_funcs) {
    return (num_funcs * sizeof(void*)) + sizeof(size_t);
}

// The `ROCDECODE_ENFORCE_ABI_VERSIONING` macro will trigger a compiler error if the size of the rocDecode dispatch API table changes,
// which is most likely due to the addition of a new dispatch table entry. This serves as a reminder for developers to update the table
// versioning value before changing the value in `ROCDECODE_ENFORCE_ABI_VERSIONING`, ensuring that this static assertion passes.
//
// The `ROCDECODE_ENFORCE_ABI` macro will also trigger a compiler error if the order of the members in the rocDecode dispatch API table
// is altered. Therefore, it is essential to avoid reordering member variables.
//
// Please be aware that `rocprofiler` performs strict compile-time checks to ensure that these versioning values are correctly updated.
// Commenting out this check or merely updating the size field in `ROCDECODE_ENFORCE_ABI_VERSIONING` will cause the `rocprofiler` to fail
// during the build process.
#define ROCDECODE_ENFORCE_ABI_VERSIONING(TABLE, NUM) \
  static_assert( \
      sizeof(TABLE) == ComputeTableOffset(NUM), \
      "The size of the API table structure has been updated. Please modify the " \
      "STEP_VERSION number (or, in rare cases, the MAJOR_VERSION number) " \
      "in <rocDecode/api/amd_detail/rocdecode_api_trace.h> for the failing API " \
      "structure before changing the SIZE field passed to ROCDECODE_DEFINE_DISPATCH_TABLE_INFO.");

#define ROCDECODE_ENFORCE_ABI(TABLE, ENTRY, NUM) \
  static_assert(offsetof(TABLE, ENTRY) == ComputeTableOffset(NUM), \
                "ABI broke for " #TABLE "." #ENTRY \
                ", only add new function pointers at the end of the struct and do not rearrange them.");

// These ensure that function pointers are not re-ordered
// ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION == 0
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_create_video_parser, 0)
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_parse_video_data, 1)
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_destroy_video_parser, 2)
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_create_decoder, 3)
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_destroy_decoder, 4)
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_get_gecoder_caps, 5)
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_decode_frame, 6)
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_get_decode_status, 7)
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_reconfigure_decoder, 8)
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_get_video_frame, 9)
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_get_error_name, 10)
// ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION == 1
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_create_bitstream_reader, 11)
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_get_bitstream_codec_type, 12)
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_get_bitstream_bit_depth, 13)
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_get_bitstream_pic_data, 14)
ROCDECODE_ENFORCE_ABI(RocDecodeDispatchTable, pfn_rocdec_destroy_bitstream_reader, 15)
// ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION == 2

// If ROCDECODE_ENFORCE_ABI entries are added for each new function pointer in the table,
// the number below will be one greater than the number in the last ROCDECODE_ENFORCE_ABI line. For example:
//  ROCDECODE_ENFORCE_ABI(<table>, <functor>, 15)
//  ROCDECODE_ENFORCE_ABI_VERSIONING(<table>, 16) <- 15 + 1 = 16
ROCDECODE_ENFORCE_ABI_VERSIONING(RocDecodeDispatchTable, 16)

static_assert(ROCDECODE_RUNTIME_API_TABLE_MAJOR_VERSION == 0 && ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION == 1,
              "If you encounter this error, add the new ROCDECODE_ENFORCE_ABI(...) code for the updated function pointers, "
              "and then modify this check to ensure it evaluates to true.");
#endif