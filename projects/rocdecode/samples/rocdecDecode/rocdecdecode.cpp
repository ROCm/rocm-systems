/*
Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#if __cplusplus >= 201703L && __has_include(<filesystem>)
    #include <filesystem>
    namespace fs = std::filesystem;
#else
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#endif
#include <hip/hip_runtime.h>
#include <rocdecode/rocdecode.h>
#include <rocdecode/rocparser.h>
#if ENABLE_HOST_DECODE
    #include <rocdecode/rocdecode_host.h>
#endif


__attribute__((visibility("hidden"))) inline bool is_error(rocDecStatus status)
{
    return status != ROCDEC_SUCCESS;
}

__attribute__((visibility("hidden"))) inline const char* error_string(rocDecStatus status)
{
    return rocDecGetErrorName(status);
}

struct Rect {
    int left;
    int top;
    int right;
    int bottom;
};

template <typename Status, typename... Args>
__attribute__((visibility("hidden"))) inline void report_error(
    Status status, const char* function_name, const char* file_name, int line, Args&&... args)
{
    ((std::cerr << "ERROR: " << error_string(status) << "; " << function_name << "; "
                << file_name << ":" << line)
     << ... << std::forward<Args>(args))
        << std::endl;
    std::exit(EXIT_FAILURE);
}

//hardcoding for this sample
#define DEFAULT_WIDTH 2912
#define DEFAULT_HEIGHT 1888

// helper functions for saving output to file

static inline float GetChromaHeightFactor(rocDecVideoSurfaceFormat surface_format) {
    float factor = 0.5;
    switch (surface_format) {
    case rocDecVideoSurfaceFormat_NV12:
    case rocDecVideoSurfaceFormat_P016:
    case rocDecVideoSurfaceFormat_YUV420:
    case rocDecVideoSurfaceFormat_YUV420_16Bit:
        factor = 0.5;
        break;
    case rocDecVideoSurfaceFormat_YUV422:
    case rocDecVideoSurfaceFormat_YUV422_16Bit:
    case rocDecVideoSurfaceFormat_YUV444:
    case rocDecVideoSurfaceFormat_YUV444_16Bit:
        factor = 1.0;
        break;
    }

    return factor;
};

static inline rocDecVideoCodec CodecTypeToRocDecVideoCodec(int codec_type) {
    switch (codec_type) {
        case 0:     return rocDecVideoCodec_HEVC;
        case 1:     return rocDecVideoCodec_AVC;
        case 2:     return rocDecVideoCodec_AV1;
        case 3:     return rocDecVideoCodec_VP9;
        case 4:     return rocDecVideoCodec_VP8;
        case 5:     return rocDecVideoCodec_JPEG;
        default:    return rocDecVideoCodec_NumCodecs;
    }
}
static inline float GetChromaWidthFactor(rocDecVideoSurfaceFormat surface_format) {
    float factor = 0.5;
    switch (surface_format) {
    case rocDecVideoSurfaceFormat_NV12:
    case rocDecVideoSurfaceFormat_P016:
    case rocDecVideoSurfaceFormat_YUV444:
    case rocDecVideoSurfaceFormat_YUV444_16Bit:
        factor = 1.0;
        break;
    case rocDecVideoSurfaceFormat_YUV420:
    case rocDecVideoSurfaceFormat_YUV420_16Bit:
    case rocDecVideoSurfaceFormat_YUV422:
    case rocDecVideoSurfaceFormat_YUV422_16Bit:
        factor = 0.5;
        break;
    }
    return factor;
};

// only 2 types of memory mode is supported in this sample for simplicity.
typedef enum OutputSurfaceMemoryType_enum {
    OUT_SURFACE_MEM_DEV_INTERNAL = 0,      /**<  Internal interopped decoded surface memory(original mapped decoded surface) */
    OUT_SURFACE_MEM_HOST = 2,        /**<  decoded output will be in host memory (true for host based decoding) **/
} OutputSurfaceMemoryType;

// Enum for decoder backend
typedef enum DecoderBackend_enum {
    DECODER_BACKEND_DEVICE = 0,      /**<  Decoding using VCN hardware in the device specified by user */
    DECODER_BACKEND_HOST = 1,        /**<  decoded using host and ffmpeg avcodec **/
} DecoderBackend;


#define CHECK(callable, ...)                                                             \
    do                                                                                   \
    {                                                                                    \
        auto status__ = callable; /* invoke the callable and assign the return status */ \
        if (is_error(status__))                                                          \
        {                                                                                \
            report_error(status__, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);     \
        }                                                                                \
    } while (false)

/**
 * @brief Struct containing all the information for decoding and displaying output
 * 
 */
struct DecoderInfo {
    int dec_device_id;
    DecoderBackend backend;                //0: device, 1: host
    rocDecDecoderHandle decoder;
    RocdecVideoParser parser;
    std::uint32_t bit_depth;
    rocDecVideoCodec rocdec_codec_id;
    int dump_decoded_frames;
    std::string output_file_path;
    OutputSurfaceMemoryType mem_type;
    rocDecVideoSurfaceFormat surf_format;
    rocDecVideoSurfaceFormat video_chroma_format;
    uint32_t coded_width, coded_height;
    uint32_t bytes_per_pixel;
    bool is_decoder_reconfigured;
    Rect disp_rect;
    FILE *fp_out;
    DecoderInfo() : dec_device_id(0), backend(DECODER_BACKEND_DEVICE), decoder(nullptr), bit_depth(8), dump_decoded_frames(0), mem_type{OUT_SURFACE_MEM_DEV_INTERNAL},
                    surf_format{rocDecVideoSurfaceFormat_NV12}, video_chroma_format{rocDecVideoSurfaceFormat_NV12},
                    is_decoder_reconfigured{false}, fp_out{nullptr} {}
};

/**
 * @brief Funtion to save internal frame buffer to file for device buffer : chroma format is assumed to be NV12 for internal device memory
 * 
 * @param p_dec_info 
 * @param surf_mem  device mem pointers of luma and chroma planes
 * @param pitch  stride in bytes of luma and chroma planes
 */
void save_frame_to_file(DecoderInfo *p_dec_info, void *surf_mem[], uint32_t *pitch) {

    uint8_t *hst_ptr = nullptr;
    uint64_t output_image_size_luma = pitch[0] * p_dec_info->coded_height;
    uint64_t output_image_size_chroma = pitch[1] * ((p_dec_info->coded_height * GetChromaHeightFactor(p_dec_info->surf_format)));
    if (p_dec_info->mem_type == OUT_SURFACE_MEM_DEV_INTERNAL) {
        if (hst_ptr == nullptr) {
            hst_ptr = new uint8_t [output_image_size_luma + output_image_size_chroma];
        }
        hipError_t hip_status = hipSuccess;
        // copy luma
        hip_status = hipMemcpyDtoH((void *)hst_ptr, surf_mem[0], output_image_size_luma);
        if (hip_status != hipSuccess) {
            std::cerr << "ERROR: hipMemcpyDtoH failed for luma! (" << hipGetErrorName(hip_status) << ")" << std::endl;
            delete [] hst_ptr;
            return;
        }
        hip_status = hipMemcpyDtoH((void *)(hst_ptr + output_image_size_luma), surf_mem[1], output_image_size_chroma);
        if (hip_status != hipSuccess) {
            std::cerr << "ERROR: hipMemcpyDtoH failed for chroma! (" << hipGetErrorName(hip_status) << ")" << std::endl;
            delete [] hst_ptr;
            return;
        }
    } else
        hst_ptr = static_cast<uint8_t *> (surf_mem[0]);

    if (p_dec_info->is_decoder_reconfigured) {
        if (p_dec_info->fp_out) {
            fclose(p_dec_info->fp_out);
            p_dec_info->fp_out = nullptr;
        }
        p_dec_info->is_decoder_reconfigured = false;
    }

    if (p_dec_info->fp_out == nullptr && !p_dec_info->output_file_path.empty()) {
        p_dec_info->fp_out = fopen(p_dec_info->output_file_path.c_str(), "wb");
    }

    if (p_dec_info->fp_out) {
        uint8_t *tmp_hst_ptr = hst_ptr;
        if (p_dec_info->mem_type == OUT_SURFACE_MEM_DEV_INTERNAL) {
            tmp_hst_ptr += (p_dec_info->disp_rect.top * pitch[0]) + (p_dec_info->disp_rect.left * p_dec_info->bytes_per_pixel);
        }
        int img_width = p_dec_info->disp_rect.right - p_dec_info->disp_rect.left;
        int img_height = p_dec_info->disp_rect.bottom - p_dec_info->disp_rect.top;
        uint32_t output_stride =  pitch[0];
        if ((img_width * p_dec_info->bytes_per_pixel) == output_stride) {
            fwrite(tmp_hst_ptr, 1, output_image_size_luma, p_dec_info->fp_out);
            tmp_hst_ptr += output_image_size_luma;
            fwrite(tmp_hst_ptr, 1, output_image_size_chroma, p_dec_info->fp_out);
        } else {
            uint32_t width = img_width * p_dec_info->bytes_per_pixel;
            if (p_dec_info->bit_depth <= 16) {
                for (int i = 0; i < img_height; i++) {
                    fwrite(tmp_hst_ptr, 1, width, p_dec_info->fp_out);
                    tmp_hst_ptr += output_stride;
                }
                // dump chroma
                uint8_t *uv_hst_ptr = hst_ptr + output_image_size_luma;
                uint32_t chroma_height = static_cast<int>(GetChromaHeightFactor(p_dec_info->surf_format) * img_height);
                if (p_dec_info->mem_type == OUT_SURFACE_MEM_DEV_INTERNAL) {
                    uv_hst_ptr += ((p_dec_info->disp_rect.top >> 1) * output_stride) + (p_dec_info->disp_rect.left * p_dec_info->bytes_per_pixel);
                }
                for (uint32_t i = 0; i < chroma_height; i++) {
                    fwrite(uv_hst_ptr, 1, width, p_dec_info->fp_out);
                    uv_hst_ptr += pitch[1];
                }
            }
        }
    }

    if (hst_ptr != nullptr) {
        delete [] hst_ptr;
    }
}

/**
 * @brief Funtion to save internal frame buffer to file for host buffer
 * 
 * @param p_dec_info 
 * @param frame_mem 
 * @param pitch 
 */
void save_frame_to_file_host(DecoderInfo *p_dec_info, void *frame_mem[], uint32_t *pitch) {

    if (p_dec_info->is_decoder_reconfigured) {
        if (p_dec_info->fp_out) {
            fclose(p_dec_info->fp_out);
            p_dec_info->fp_out = nullptr;
        }
        p_dec_info->is_decoder_reconfigured = false;
    }
    
    if (p_dec_info->fp_out == nullptr && !p_dec_info->output_file_path.empty()) {
        p_dec_info->fp_out = fopen(p_dec_info->output_file_path.c_str(), "wb");
    }

    if (p_dec_info->fp_out) {
        uint8_t *p_src_ptr_y = static_cast<uint8_t *>(frame_mem[0]) + (p_dec_info->disp_rect.top * pitch[0] + p_dec_info->disp_rect.left  * p_dec_info->bytes_per_pixel);
        if (!p_src_ptr_y) {
            std::cerr << "save_frame_to_file_host: Invalid Memory address for src/dst" << std::endl;
            return;
        }
        int img_width = p_dec_info->disp_rect.right - p_dec_info->disp_rect.left;
        int img_height = p_dec_info->disp_rect.bottom - p_dec_info->disp_rect.top;
        int output_stride =  pitch[0];

        uint32_t width = img_width * p_dec_info->bytes_per_pixel;
        if (p_dec_info->bit_depth <= 16) {
            for (int i = 0; i < img_height; i++) {
                fwrite(p_src_ptr_y, 1, width, p_dec_info->fp_out);
                p_src_ptr_y += output_stride;
            }
            // dump chroma
            uint8_t *p_src_ptr_uv = static_cast<uint8_t *>(frame_mem[1]) + ((p_dec_info->disp_rect.top >> 1) * pitch[1] + (p_dec_info->disp_rect.left >> 1) * p_dec_info->bytes_per_pixel);
            int32_t chroma_height = static_cast<int>(GetChromaHeightFactor(p_dec_info->surf_format) * img_height);
            int32_t chroma_width = static_cast<int>(GetChromaWidthFactor(p_dec_info->surf_format) * img_width) * p_dec_info->bytes_per_pixel;
            for (int32_t i = 0; i < chroma_height; i++) {
                fwrite(p_src_ptr_uv, 1, chroma_width, p_dec_info->fp_out);
                p_src_ptr_uv += pitch[1];
            }
            if (frame_mem[2] != nullptr) {
                uint8_t *p_src_ptr_v = static_cast<uint8_t *>(frame_mem[2]) + p_dec_info->disp_rect.top * pitch[2] + (p_dec_info->disp_rect.left >> 1) * p_dec_info->bytes_per_pixel;
                for (int32_t i = 0; i < chroma_height; i++) {
                    fwrite(p_src_ptr_v, 1, chroma_width, p_dec_info->fp_out);
                    p_src_ptr_v += pitch[2];
                }
            }
        }
    }
}

std::vector<std::vector<uint8_t>> read_frames(std::vector<std::string>& names) {
    std::vector<std::vector<uint8_t>> frames;
    // sort the frames file so it is consecutive
    for (std::string name : names) {
        std::ifstream inputFile(name.c_str(), std::ios::binary);
        if (!inputFile) {
            std::cerr << "Error opening " << name << " for reading." << std::endl;
            std::abort();
        }
        std::cout << "Reading " << name << " for reading." << std::endl;
        // Determine the file size
        inputFile.seekg(0, std::ios::end);
        std::streamsize fileSize = inputFile.tellg();
        inputFile.seekg(0, std::ios::beg);

        // Read the file contents into a byte array
        std::vector<uint8_t> frame(fileSize);
        if (!inputFile.read(reinterpret_cast<char*>(frame.data()), fileSize)) {
            std::cerr << "Error reading " << name << "." << std::endl;
            std::abort();
        }
        // Close the file
        inputFile.close();
        frames.push_back(std::move(frame));
    }

    return frames;
}

void init() {}

void create_decoder(DecoderInfo& dec_info) {
    RocDecoderCreateInfo create_info = {};
    create_info.codec_type = dec_info.rocdec_codec_id;     // user specified codec_type for raw files
    create_info.max_width = DEFAULT_WIDTH;
    create_info.max_height = DEFAULT_HEIGHT;
    create_info.width = DEFAULT_WIDTH;
    create_info.height = DEFAULT_HEIGHT;
    create_info.num_decode_surfaces = 6;
    create_info.target_width = DEFAULT_WIDTH;
    create_info.target_height = DEFAULT_HEIGHT;
    create_info.display_rect.left = 0;
    create_info.display_rect.right = static_cast<short>(DEFAULT_WIDTH);
    create_info.display_rect.top = 0;
    create_info.display_rect.bottom = static_cast<short>(DEFAULT_HEIGHT);
    // for decode creation: assuming chroma_format is 4:2:0 and output_format is NV12.
    // video dimensions ( width, height, max_width, max_height), num_decode_surfaces, and bit_depth_minus_8 are hardcoded here
    // this will get changed in reconfigure when the sequence header is parsed from the stream to detect the actual video parameters
    create_info.chroma_format = rocDecVideoChromaFormat_420;
    create_info.output_format = rocDecVideoSurfaceFormat_NV12;
    create_info.bit_depth_minus_8 = 0;
    create_info.num_output_surfaces = 1;
    CHECK(rocDecCreateDecoder(&dec_info.decoder, &create_info));
}

#if ENABLE_HOST_DECODE
int ROCDECAPI handle_video_sequence_host(void* user_data, RocdecVideoFormatHost* format_host) {
    DecoderInfo *p_dec_info = static_cast<DecoderInfo *>(user_data);
    RocdecVideoFormat *format = &format_host->video_format;
    RocdecReconfigureDecoderInfo reconfig_params = {};
    reconfig_params.width = format->coded_width;
    reconfig_params.height = format->coded_height;
    reconfig_params.num_decode_surfaces = 6;
    reconfig_params.target_width = format->coded_width;
    reconfig_params.target_height = format->coded_height;
    reconfig_params.display_rect.left = 0;
    reconfig_params.display_rect.right = static_cast<short>(format->coded_width);
    reconfig_params.display_rect.top = 0;
    reconfig_params.display_rect.bottom = static_cast<short>(format->coded_height);
    p_dec_info->surf_format = format_host->video_surface_format;
    p_dec_info->disp_rect.top = format->display_area.top;
    p_dec_info->disp_rect.bottom = format->display_area.bottom;
    p_dec_info->disp_rect.left = format->display_area.left;
    p_dec_info->disp_rect.right = format->display_area.right;
    CHECK(rocDecReconfigureDecoderHost(p_dec_info->decoder, &reconfig_params));
    p_dec_info->is_decoder_reconfigured = true;
    int bitdepth_minus_8 = format->bit_depth_luma_minus8;
    p_dec_info->coded_width = format->coded_width;
    p_dec_info->coded_height = format->coded_height;
    p_dec_info->bytes_per_pixel = bitdepth_minus_8 > 0 ? 2 : 1;
    std::ostringstream input_video_info_str;
    input_video_info_str.str("");
    input_video_info_str.clear();
    input_video_info_str << "Input Video Information" << std::endl
        << "\tCodec        : " << format->codec << std::endl;
        if (format->frame_rate.numerator && format->frame_rate.denominator) {
            input_video_info_str << "\tFrame rate   : " << format->frame_rate.numerator << "/" << format->frame_rate.denominator << " = " << 1.0 * format->frame_rate.numerator / format->frame_rate.denominator << " fps" << std::endl;
        }
    input_video_info_str << "\tSequence     : " << (format->progressive_sequence ? "Progressive" : "Interlaced") << std::endl
        << "\tCoded size   : [" << format->coded_width << ", " << format->coded_height << "]" << std::endl
        << "\tDisplay area : [" << format->display_area.left << ", " << format->display_area.top << ", "
            << format->display_area.right << ", " << format->display_area.bottom << "]" << std::endl
        << "\tBit depth    : " << format->bit_depth_luma_minus8 + 8
    ;
    input_video_info_str << std::endl;
    std::cout << input_video_info_str.str();

    return 1;
}

int ROCDECAPI handle_picture_display_host(void* user_data, RocdecParserDispInfo* disp_info) {
    DecoderInfo *p_dec_info = static_cast<DecoderInfo *>(user_data);
    RocdecParserDispInfo *p_disp_info = static_cast<RocdecParserDispInfo *>(disp_info);
    RocdecProcParams params = {};
    params.progressive_frame = p_disp_info->progressive_frame;
    params.top_field_first = p_disp_info->top_field_first;
    void* frame_mem_ptr[3] = {nullptr};
    uint32_t pitch[3] = {0};
    CHECK(rocDecGetVideoFrameHost(p_dec_info->decoder, p_disp_info->picture_index, frame_mem_ptr, pitch, &params));
    p_dec_info->mem_type = OUT_SURFACE_MEM_HOST;
    if (p_dec_info->dump_decoded_frames) {
        save_frame_to_file_host(p_dec_info, frame_mem_ptr, pitch);
    }

    return 1;
}

void create_decoder_host(DecoderInfo& dec_info) {
    // many of the decoder parameters are hardcoded below for just creating the decoder.
    // In the handlevideosequence callback, the decoder will get reconfigured to the actual parameters in the sequence header
    RocDecoderHostCreateInfo create_info = {};
    create_info.codec_type = dec_info.rocdec_codec_id;
    create_info.num_decode_threads = 0;     // default
    create_info.max_width = DEFAULT_WIDTH;
    create_info.max_height = DEFAULT_HEIGHT;
    create_info.width = DEFAULT_WIDTH;
    create_info.height = DEFAULT_HEIGHT;
    create_info.target_width = DEFAULT_WIDTH;
    create_info.target_height = DEFAULT_HEIGHT;
    create_info.display_rect.left = 0;
    create_info.display_rect.right = static_cast<short>(DEFAULT_WIDTH);
    create_info.display_rect.top = 0;
    create_info.display_rect.bottom = static_cast<short>(DEFAULT_HEIGHT);
    create_info.chroma_format = rocDecVideoChromaFormat_420;
    create_info.output_format = rocDecVideoSurfaceFormat_P016;
    create_info.bit_depth_minus_8 = 2;
    create_info.num_output_surfaces = 1;
    create_info.user_data = &dec_info;
    create_info.pfn_sequence_callback = handle_video_sequence_host;
    create_info.pfn_display_picture = handle_picture_display_host;
    CHECK(rocDecCreateDecoderHost(&dec_info.decoder, &create_info));
    dec_info.backend = DECODER_BACKEND_HOST;
}
#endif

int ROCDECAPI handle_video_sequence(void* user_data, RocdecVideoFormat* format) {
    DecoderInfo *p_dec_info = static_cast<DecoderInfo *>(user_data);
    RocdecReconfigureDecoderInfo reconfig_params = {};
    int bitdepth_minus_8 = format->bit_depth_luma_minus8;
    uint32_t target_width = (format->display_area.right - format->display_area.left + 1) & ~1;
    uint32_t target_height = (format->display_area.bottom - format->display_area.top + 1) & ~1;
    reconfig_params.width = format->coded_width;
    reconfig_params.height = format->coded_height;
    reconfig_params.bit_depth_minus_8 = bitdepth_minus_8;
    reconfig_params.num_decode_surfaces = format->min_num_decode_surfaces;
    reconfig_params.target_width = target_width;
    reconfig_params.target_height = target_height;
    reconfig_params.display_rect.left = format->display_area.left;
    reconfig_params.display_rect.right = format->display_area.right;
    reconfig_params.display_rect.top = format->display_area.top;
    reconfig_params.display_rect.bottom = format->display_area.bottom;
    CHECK(rocDecReconfigureDecoder(p_dec_info->decoder, &reconfig_params));
    p_dec_info->is_decoder_reconfigured = true;
    p_dec_info->disp_rect.top = format->display_area.top;
    p_dec_info->disp_rect.bottom = format->display_area.bottom;
    p_dec_info->disp_rect.left = format->display_area.left;
    p_dec_info->disp_rect.right = format->display_area.right;
    rocDecVideoChromaFormat video_chroma_format = format->chroma_format;
    if (video_chroma_format == rocDecVideoChromaFormat_420 || rocDecVideoChromaFormat_Monochrome)
        p_dec_info->surf_format = bitdepth_minus_8 ? rocDecVideoSurfaceFormat_P016 : rocDecVideoSurfaceFormat_NV12;
    else if (video_chroma_format == rocDecVideoChromaFormat_444)
        p_dec_info->surf_format = bitdepth_minus_8 ? rocDecVideoSurfaceFormat_YUV444_16Bit : rocDecVideoSurfaceFormat_YUV444;
    else if (video_chroma_format == rocDecVideoChromaFormat_422)
        p_dec_info->surf_format = bitdepth_minus_8 ? rocDecVideoSurfaceFormat_YUV422_16Bit : rocDecVideoSurfaceFormat_YUV422;
    p_dec_info->coded_width = format->coded_width;
    p_dec_info->coded_height = format->coded_height;
    p_dec_info->bytes_per_pixel = bitdepth_minus_8 > 0 ? 2 : 1;
    std::ostringstream input_video_info_str;
    input_video_info_str.str("");
    input_video_info_str.clear();
    input_video_info_str << "Input Video Information" << std::endl
        << "\tCodec        : " << format->codec << std::endl;
        if (format->frame_rate.numerator && format->frame_rate.denominator) {
            input_video_info_str << "\tFrame rate   : " << format->frame_rate.numerator << "/" << format->frame_rate.denominator << " = " << 1.0 * format->frame_rate.numerator / format->frame_rate.denominator << " fps" << std::endl;
        }
    input_video_info_str << "\tSequence     : " << (format->progressive_sequence ? "Progressive" : "Interlaced") << std::endl
        << "\tCoded size   : [" << format->coded_width << ", " << format->coded_height << "]" << std::endl
        << "\tDisplay area : [" << format->display_area.left << ", " << format->display_area.top << ", "
            << format->display_area.right << ", " << format->display_area.bottom << "]" << std::endl
        << "\tBit depth    : " << format->bit_depth_luma_minus8 + 8
    ;
    input_video_info_str << std::endl;
    std::cout << input_video_info_str.str();
    return 1;
}

int ROCDECAPI handle_picture_decode(void* user_data, RocdecPicParams* params) {
    DecoderInfo *p_dec_info = static_cast<DecoderInfo *>(user_data);
    CHECK(rocDecDecodeFrame(p_dec_info->decoder, params));
    return 1;
}


int ROCDECAPI handle_picture_display(void* user_data, RocdecParserDispInfo* disp_info) {
    DecoderInfo *p_dec_info = static_cast<DecoderInfo *>(user_data);
    RocdecProcParams params = {};
    params.progressive_frame = disp_info->progressive_frame;
    params.top_field_first = disp_info->top_field_first;
    // get device memory pointer for decoded output surface
    void* dev_mem_ptr[3] = { 0 };
    uint32_t pitch[3] = { 0 };
    CHECK(rocDecGetVideoFrame(p_dec_info->decoder, disp_info->picture_index, dev_mem_ptr, pitch, &params));

    if (p_dec_info->dump_decoded_frames) {
        save_frame_to_file(p_dec_info, dev_mem_ptr, pitch);
    }
    return 1;
}


void create_parser(DecoderInfo& dec_info) {
    RocdecParserParams params = {};
    params.codec_type = dec_info.rocdec_codec_id;
    params.max_num_decode_surfaces = 6;
    params.max_display_delay = 1;       // min display delay of 1 is recommented to get optimal performance from hardware decoder
    params.user_data = &dec_info;
    params.pfn_sequence_callback = handle_video_sequence;
    params.pfn_decode_picture = handle_picture_decode;
    params.pfn_display_picture = handle_picture_display;
    CHECK(rocDecCreateVideoParser(&dec_info.parser, &params));
}

void decode_frames(DecoderInfo& dec_info, const std::vector<std::vector<uint8_t>>& frames) {
    // gpu backend using VCN
    if (dec_info.backend == DECODER_BACKEND_DEVICE) {
        for (int i=0; i < static_cast<int>(frames.size()); ++i) {
            RocdecSourceDataPacket packet = {};
            packet.payload_size = frames[i].size();
            packet.payload = frames[i].data();
            if (i == static_cast<int>(frames.size() - 1)) {
                packet.flags = ROCDEC_PKT_ENDOFPICTURE;     // mark end_of_picture flag for last frame
            }
            CHECK(rocDecParseVideoData(dec_info.parser, &packet));
        }
    }
#if ENABLE_HOST_DECODE
    else if (dec_info.backend == DECODER_BACKEND_HOST) {
        for (int i=0; i < static_cast<int>(frames.size()); ++i) {
            RocdecPicParamsHost pic_params = {};
            pic_params.bitstream_data_len = frames[i].size();
            pic_params.bitstream_data = frames[i].data();
            if (i == static_cast<int>(frames.size() - 1)) {
                pic_params.flags = ROCDEC_PKT_ENDOFPICTURE;     // mark end_of_picture flag for last frame
            }
            CHECK(rocDecDecodeFrameHost(dec_info.decoder, &pic_params));
        }   
    }
#endif
}

void destroy_decoder(DecoderInfo& dec_info) {
    if (dec_info.backend == DECODER_BACKEND_DEVICE) {
        CHECK(rocDecDestroyDecoder(dec_info.decoder));
    }
#if ENABLE_HOST_DECODE
    else if (dec_info.backend == DECODER_BACKEND_HOST) {
        CHECK(rocDecDestroyDecoderHost(dec_info.decoder));
    }
#endif
}

void destroy_parser(DecoderInfo& dec_info) {
    if (dec_info.backend == DECODER_BACKEND_DEVICE)
        CHECK(rocDecDestroyVideoParser(dec_info.parser));
}

void ShowHelpAndExit(const char *option = NULL) {
    std::cout << "Options:" << std::endl
    << "-i Input File Path - required" << std::endl
    << "-o Output File Path - dumps output if requested; optional" << std::endl
    << "-d GPU device ID (0 for the first device, 1 for the second, etc.); optional; default: 0" << std::endl
#if ENABLE_HOST_DECODE
    << "-b backend (0 for GPU, 1 CPU-FFMpeg); optional; default: 0" << std::endl
#else
    << "-b backend (0 for GPU); optional; default: 0" << std::endl
#endif
    << "-c codec (0 : HEVC, 1 : H264, 2: AV1, 4: VP9, 5: VP8 ); optional; default: 0" << std::endl
    << "-n Number of iteration - specify the number of iterations for performance evaluation; optional; default: 1" << std::endl
    << "-m output_surface_memory_type - decoded surface memory; optional; default - 0"
    << " [0 : OUT_SURFACE_MEM_DEV_INTERNAL/ 1 : OUT_SURFACE_MEM_DEV_COPIED/ 2 : OUT_SURFACE_MEM_HOST_COPIED/ 3 : OUT_SURFACE_MEM_NOT_MAPPED]" << std::endl;
    exit(0);
}

// helper function for sort
std::string getLastPart(const std::string& str, char delimiter) {
    size_t pos = str.find_last_of(delimiter);
    if (pos == std::string::npos) {
        return str; // Delimiter not found, return the whole string
    }
    return str.substr(pos + 1);
}

// helper function for sort
int extractNumber(const std::string& filename) {
    std::string numStr;
    for (char c : filename) {
        if (std::isdigit(c)) {
            numStr += c;
        } else if (!numStr.empty()) {
            break; // Stop at first non-digit after a digit sequence
        }
    }
    return numStr.empty() ? 0 : std::stoi(numStr);
}

// helper function for sort
// Sort entries based on the numerical part of their filenames
bool compareFilenames(const std::string& a, const std::string& b) {
    int num_a = extractNumber(a);
    int num_b = extractNumber(b);
    if (num_a != num_b) {
        return num_a < num_b;
    }
    return a < b; // Fallback to lexicographical comparison
};

int main(int argc, char** argv) {

    std::string input_file_path, output_file_path;
    int dump_output_frames = 0;
    int device_id = 0;
    DecoderBackend backend = DECODER_BACKEND_DEVICE;
    int num_iterations = 1; 
    std::vector<std::string> input_file_names;
    int codec_type = 0; // default for HEVC
    DecoderInfo dec_info;

    // Parse command-line arguments
    if(argc <= 1) {
        ShowHelpAndExit();
    }
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h")) {
            ShowHelpAndExit();
        }
        if (!strcmp(argv[i], "-i")) {
            if (++i == argc) {
                ShowHelpAndExit("-i");
            }
            input_file_path = argv[i];
            bool b_sort_filenames = false;
            if (fs::is_directory(input_file_path)) {
                for (const auto& entry : fs::directory_iterator(input_file_path)) {
                    if (entry.is_directory()) {
                        std::vector<std::string> file_names_sub_folder;
                        for (const auto& sub_entry : fs::directory_iterator(entry)) {
                            file_names_sub_folder.push_back(sub_entry.path());
                        }
                        std::sort(file_names_sub_folder.begin(), file_names_sub_folder.end(), compareFilenames);
                        input_file_names.insert(input_file_names.end(), file_names_sub_folder.begin(), file_names_sub_folder.end());
                        file_names_sub_folder.clear();
                    } else if(entry.is_regular_file()) {
                        b_sort_filenames = true;
                        input_file_names.push_back(entry.path());
                    }
                    else {
                        std::cout << "unknown file type in input folder: " << entry.path().string() << '\n';
                        continue;
                    }
                 }
                 if (b_sort_filenames) {
                    std::sort(input_file_names.begin(), input_file_names.end(), compareFilenames);
                 }
            } else {
                input_file_names.push_back(input_file_path);
            }
        
            std::cout << "Read " << input_file_names.size() << " frames from disk." << std::endl;
            continue;
        }
        if (!strcmp(argv[i], "-o")) {
            if (++i == argc) {
                ShowHelpAndExit("-o");
            }
            output_file_path = argv[i];
            dec_info.output_file_path = output_file_path;
            dump_output_frames = true;
            continue;
        }
        if (!strcmp(argv[i], "-b")) {
            if (++i == argc) {
                ShowHelpAndExit("-b");
            }
            backend = static_cast<DecoderBackend>(atoi(argv[i]));
            continue;
        }

        if (!strcmp(argv[i], "-d")) {
            if (++i == argc) {
                ShowHelpAndExit("-d");
            }
            device_id = atoi(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-c")) {
            if (++i == argc) {
                ShowHelpAndExit("-c");
            }
            codec_type = atoi(argv[i]);
            continue;
        }

        if (!strcmp(argv[i], "-n")) {
            if (++i == argc) {
                ShowHelpAndExit("-n");
            }
            num_iterations = atoi(argv[i]);
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
    dec_info.rocdec_codec_id = CodecTypeToRocDecVideoCodec(codec_type);
    dec_info.dec_device_id = device_id;
    dec_info.mem_type = (!backend) ? OUT_SURFACE_MEM_DEV_INTERNAL : OUT_SURFACE_MEM_HOST;
    init();
    if (backend == DECODER_BACKEND_DEVICE) {
        create_parser(dec_info);
        create_decoder(dec_info);
    }
#if ENABLE_HOST_DECODE
    else {
        create_decoder_host(dec_info);
    }
#endif
    dec_info.dump_decoded_frames = dump_output_frames;
    auto input_frames = read_frames(input_file_names);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        decode_frames(dec_info, input_frames);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Decoding time: " << elapsed << " microseconds" << std::endl;
    destroy_decoder(dec_info);
    destroy_parser(dec_info);
    std::cout << "Success." << std::endl << std::endl << std::endl;
    return 0;
}
