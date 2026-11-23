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

// rocencCaps -- prints every rocEncode capability for every codec on every
// visible device.
//
// A device with no encoder (Instinct MI300-class, Navi 24 / VCN 3.0.33) is
// expected output, not a failure: the session opens, zero codecs are
// enumerated, and the sample says so and exits 0.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <rocencode/rocencode.h>

namespace {

struct CapEntry {
    ROCENC_CAPS  id;
    const char*  name;
};

// All 55 queryable capabilities.  ROCENC_CAPS_EXPOSED_COUNT is reserved and is
// deliberately absent: querying it must fail, which the negative test checks.
const CapEntry kCaps[] = {
    {ROCENC_CAPS_NUM_MAX_BFRAMES,                "NUM_MAX_BFRAMES"},
    {ROCENC_CAPS_SUPPORTED_RATECONTROL_MODES,    "SUPPORTED_RATECONTROL_MODES"},
    {ROCENC_CAPS_SUPPORT_FIELD_ENCODING,         "SUPPORT_FIELD_ENCODING"},
    {ROCENC_CAPS_SUPPORT_MONOCHROME,             "SUPPORT_MONOCHROME"},
    {ROCENC_CAPS_SUPPORT_FMO,                    "SUPPORT_FMO"},
    {ROCENC_CAPS_SUPPORT_QPELMV,                 "SUPPORT_QPELMV"},
    {ROCENC_CAPS_SUPPORT_BDIRECT_MODE,           "SUPPORT_BDIRECT_MODE"},
    {ROCENC_CAPS_SUPPORT_CABAC,                  "SUPPORT_CABAC"},
    {ROCENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM,     "SUPPORT_ADAPTIVE_TRANSFORM"},
    {ROCENC_CAPS_SUPPORT_STEREO_MVC,             "SUPPORT_STEREO_MVC"},
    {ROCENC_CAPS_NUM_MAX_TEMPORAL_LAYERS,        "NUM_MAX_TEMPORAL_LAYERS"},
    {ROCENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES,   "SUPPORT_HIERARCHICAL_PFRAMES"},
    {ROCENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES,   "SUPPORT_HIERARCHICAL_BFRAMES"},
    {ROCENC_CAPS_LEVEL_MAX,                      "LEVEL_MAX"},
    {ROCENC_CAPS_LEVEL_MIN,                      "LEVEL_MIN"},
    {ROCENC_CAPS_SEPARATE_COLOUR_PLANE,          "SEPARATE_COLOUR_PLANE"},
    {ROCENC_CAPS_WIDTH_MAX,                      "WIDTH_MAX"},
    {ROCENC_CAPS_HEIGHT_MAX,                     "HEIGHT_MAX"},
    {ROCENC_CAPS_SUPPORT_TEMPORAL_SVC,           "SUPPORT_TEMPORAL_SVC"},
    {ROCENC_CAPS_SUPPORT_DYN_RES_CHANGE,         "SUPPORT_DYN_RES_CHANGE"},
    {ROCENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE,     "SUPPORT_DYN_BITRATE_CHANGE"},
    {ROCENC_CAPS_SUPPORT_DYN_FORCE_CONSTQP,      "SUPPORT_DYN_FORCE_CONSTQP"},
    {ROCENC_CAPS_SUPPORT_DYN_RCMODE_CHANGE,      "SUPPORT_DYN_RCMODE_CHANGE"},
    {ROCENC_CAPS_SUPPORT_SUBFRAME_READBACK,      "SUPPORT_SUBFRAME_READBACK"},
    {ROCENC_CAPS_SUPPORT_CONSTRAINED_ENCODING,   "SUPPORT_CONSTRAINED_ENCODING"},
    {ROCENC_CAPS_SUPPORT_INTRA_REFRESH,          "SUPPORT_INTRA_REFRESH"},
    {ROCENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE,    "SUPPORT_CUSTOM_VBV_BUF_SIZE"},
    {ROCENC_CAPS_SUPPORT_DYNAMIC_SLICE_MODE,     "SUPPORT_DYNAMIC_SLICE_MODE"},
    {ROCENC_CAPS_SUPPORT_REF_PIC_INVALIDATION,   "SUPPORT_REF_PIC_INVALIDATION"},
    {ROCENC_CAPS_PREPROC_SUPPORT,                "PREPROC_SUPPORT"},
    {ROCENC_CAPS_ASYNC_ENCODE_SUPPORT,           "ASYNC_ENCODE_SUPPORT"},
    {ROCENC_CAPS_MB_NUM_MAX,                     "MB_NUM_MAX"},
    {ROCENC_CAPS_MB_PER_SEC_MAX,                 "MB_PER_SEC_MAX"},
    {ROCENC_CAPS_SUPPORT_YUV444_ENCODE,          "SUPPORT_YUV444_ENCODE"},
    {ROCENC_CAPS_SUPPORT_LOSSLESS_ENCODE,        "SUPPORT_LOSSLESS_ENCODE"},
    {ROCENC_CAPS_SUPPORT_SAO,                    "SUPPORT_SAO"},
    {ROCENC_CAPS_SUPPORT_MEONLY_MODE,            "SUPPORT_MEONLY_MODE"},
    {ROCENC_CAPS_SUPPORT_LOOKAHEAD,              "SUPPORT_LOOKAHEAD"},
    {ROCENC_CAPS_SUPPORT_TEMPORAL_AQ,            "SUPPORT_TEMPORAL_AQ"},
    {ROCENC_CAPS_SUPPORT_10BIT_ENCODE,           "SUPPORT_10BIT_ENCODE"},
    {ROCENC_CAPS_NUM_MAX_LTR_FRAMES,             "NUM_MAX_LTR_FRAMES"},
    {ROCENC_CAPS_SUPPORT_WEIGHTED_PREDICTION,    "SUPPORT_WEIGHTED_PREDICTION"},
    {ROCENC_CAPS_DYNAMIC_QUERY_ENCODER_CAPACITY, "DYNAMIC_QUERY_ENCODER_CAPACITY"},
    {ROCENC_CAPS_SUPPORT_BFRAME_REF_MODE,        "SUPPORT_BFRAME_REF_MODE"},
    {ROCENC_CAPS_SUPPORT_EMPHASIS_LEVEL_MAP,     "SUPPORT_EMPHASIS_LEVEL_MAP"},
    {ROCENC_CAPS_WIDTH_MIN,                      "WIDTH_MIN"},
    {ROCENC_CAPS_HEIGHT_MIN,                     "HEIGHT_MIN"},
    {ROCENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES,    "SUPPORT_MULTIPLE_REF_FRAMES"},
    {ROCENC_CAPS_SUPPORT_ALPHA_LAYER_ENCODING,   "SUPPORT_ALPHA_LAYER_ENCODING"},
    {ROCENC_CAPS_NUM_ENCODER_ENGINES,            "NUM_ENCODER_ENGINES"},
    {ROCENC_CAPS_SINGLE_SLICE_INTRA_REFRESH,     "SINGLE_SLICE_INTRA_REFRESH"},
    {ROCENC_CAPS_DISABLE_ENC_STATE_ADVANCE,      "DISABLE_ENC_STATE_ADVANCE"},
    {ROCENC_CAPS_OUTPUT_RECON_SURFACE,           "OUTPUT_RECON_SURFACE"},
    {ROCENC_CAPS_OUTPUT_BLOCK_STATS,             "OUTPUT_BLOCK_STATS"},
    {ROCENC_CAPS_OUTPUT_ROW_STATS,               "OUTPUT_ROW_STATS"},
};
constexpr size_t kNumCaps = sizeof(kCaps) / sizeof(kCaps[0]);

const char* CodecName(ROCENC_CODEC c) {
    switch (c) {
        case ROCENC_CODEC_H264:      return "H.264";
        case ROCENC_CODEC_HEVC:      return "HEVC";
        case ROCENC_CODEC_AV1:       return "AV1";
        case ROCENC_CODEC_UNDEFINED: return "UNDEFINED";
    }
    return "unknown";
}

const char* ProfileName(ROCENC_PROFILE p) {
    switch (p) {
        case ROCENC_PROFILE_AUTOSELECT:            return "AUTOSELECT";
        case ROCENC_H264_PROFILE_BASELINE:         return "H264_BASELINE";
        case ROCENC_H264_PROFILE_MAIN:             return "H264_MAIN";
        case ROCENC_H264_PROFILE_HIGH:             return "H264_HIGH";
        case ROCENC_H264_PROFILE_HIGH_444:         return "H264_HIGH_444";
        case ROCENC_H264_PROFILE_STEREO:           return "H264_STEREO";
        case ROCENC_H264_PROFILE_PROGRESSIVE_HIGH: return "H264_PROGRESSIVE_HIGH";
        case ROCENC_H264_PROFILE_CONSTRAINED_HIGH: return "H264_CONSTRAINED_HIGH";
        case ROCENC_HEVC_PROFILE_MAIN:             return "HEVC_MAIN";
        case ROCENC_HEVC_PROFILE_MAIN10:           return "HEVC_MAIN10";
        case ROCENC_HEVC_PROFILE_FREXT:            return "HEVC_FREXT";
        case ROCENC_AV1_PROFILE_MAIN:              return "AV1_MAIN";
    }
    return "unknown";
}

const char* FormatName(ROCENC_BUFFER_FORMAT f) {
    switch (f) {
        case ROCENC_BUFFER_FORMAT_NV12:         return "NV12";
        case ROCENC_BUFFER_FORMAT_YV12:         return "YV12";
        case ROCENC_BUFFER_FORMAT_IYUV:         return "IYUV";
        case ROCENC_BUFFER_FORMAT_YUV444:       return "YUV444";
        case ROCENC_BUFFER_FORMAT_YUV420_10BIT: return "YUV420_10BIT";
        case ROCENC_BUFFER_FORMAT_YUV444_10BIT: return "YUV444_10BIT";
        case ROCENC_BUFFER_FORMAT_ARGB:         return "ARGB";
        case ROCENC_BUFFER_FORMAT_ARGB10:       return "ARGB10";
        case ROCENC_BUFFER_FORMAT_AYUV:         return "AYUV";
        case ROCENC_BUFFER_FORMAT_ABGR:         return "ABGR";
        case ROCENC_BUFFER_FORMAT_ABGR10:       return "ABGR10";
        case ROCENC_BUFFER_FORMAT_U8:           return "U8";
        case ROCENC_BUFFER_FORMAT_UNDEFINED:    return "UNDEFINED";
    }
    return "unknown";
}

// Reports one device.  Returns false when the ordinal does not exist, which is
// how the caller discovers the device count without linking against HIP.
bool ReportDevice(ROCENCODE_API_FUNCTION_LIST& api, int device_id, bool* saw_any_encoder) {
    ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS open_params;
    std::memset(&open_params, 0, sizeof(open_params));
    open_params.version = ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    open_params.deviceType = ROCENC_DEVICE_TYPE_HIP;
    open_params.device = reinterpret_cast<void*>(static_cast<intptr_t>(device_id));
    open_params.apiVersion = ROCENCAPI_VERSION;

    void* encoder = nullptr;
    rocEncStatus status = api.rocEncOpenEncodeSessionEx(&open_params, &encoder);
    if (status != ROCENC_SUCCESS) {
        return false;
    }

    std::printf("\n================ device %d ================\n", device_id);

    uint32_t codec_count = 0;
    status = api.rocEncGetEncodeCodecCount(encoder, &codec_count);
    if (status != ROCENC_SUCCESS) {
        std::printf("  rocEncGetEncodeCodecCount failed: %d (%s)\n", static_cast<int>(status),
                    api.rocEncGetLastErrorString(encoder));
        api.rocEncDestroyEncoder(encoder);
        return true;
    }

    if (codec_count == 0) {
        std::printf("  No encode-capable codecs on this device.\n");
        std::printf("  (Expected on Instinct MI300-class parts and on Navi 24 / VCN 3.0.33,\n");
        std::printf("   which are decode-only silicon.)\n");
        api.rocEncDestroyEncoder(encoder);
        return true;
    }
    *saw_any_encoder = true;

    std::vector<ROCENC_CODEC> codecs(codec_count);
    uint32_t returned = 0;
    status = api.rocEncGetEncodeCodecs(encoder, codecs.data(), codec_count, &returned);
    if (status != ROCENC_SUCCESS) {
        std::printf("  rocEncGetEncodeCodecs failed: %d\n", static_cast<int>(status));
        api.rocEncDestroyEncoder(encoder);
        return true;
    }

    for (uint32_t i = 0; i < returned; i++) {
        const ROCENC_CODEC codec = codecs[i];
        std::printf("\n  ---- codec %s ----\n", CodecName(codec));

        uint32_t profile_count = 0;
        if (api.rocEncGetEncodeProfileCount(encoder, codec, &profile_count) == ROCENC_SUCCESS &&
            profile_count > 0) {
            std::vector<ROCENC_PROFILE> profiles(profile_count);
            uint32_t profiles_returned = 0;
            if (api.rocEncGetEncodeProfiles(encoder, codec, profiles.data(), profile_count,
                                            &profiles_returned) == ROCENC_SUCCESS) {
                std::printf("  profiles (%u):\n", profiles_returned);
                for (uint32_t p = 0; p < profiles_returned; p++) {
                    std::printf("    %s\n", ProfileName(profiles[p]));
                }
            }
        }

        uint32_t fmt_count = 0;
        if (api.rocEncGetInputFormatCount(encoder, codec, &fmt_count) == ROCENC_SUCCESS &&
            fmt_count > 0) {
            std::vector<ROCENC_BUFFER_FORMAT> formats(fmt_count);
            uint32_t fmts_returned = 0;
            if (api.rocEncGetInputFormats(encoder, codec,formats.data(), fmt_count,
                                          &fmts_returned) == ROCENC_SUCCESS) {
                std::printf("  input formats (%u):", fmts_returned);
                for (uint32_t f = 0; f < fmts_returned; f++) {
                    std::printf(" %s", FormatName(formats[f]));
                }
                std::printf("\n");
            }
        }

        std::printf("  capabilities (%zu):\n", kNumCaps);
        for (size_t c = 0; c < kNumCaps; c++) {
            ROCENC_CAPS_PARAM caps_param;
            std::memset(&caps_param, 0, sizeof(caps_param));
            caps_param.version = ROCENC_CAPS_PARAM_VER;
            caps_param.capsToQuery = kCaps[c].id;
            int value = 0;
            rocEncStatus caps_status = api.rocEncGetEncodeCaps(encoder, codec,&caps_param, &value);
            if (caps_status == ROCENC_SUCCESS) {
                std::printf("    %-34s %d\n", kCaps[c].name, value);
            } else {
                std::printf("    %-34s <query failed: %d>\n", kCaps[c].name,
                            static_cast<int>(caps_status));
            }
        }
    }

    api.rocEncDestroyEncoder(encoder);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int only_device = -1;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            only_device = std::atoi(argv[++i]);
        } else {
            std::printf("usage: %s [-d <device_id>]\n", argv[0]);
            return 1;
        }
    }

    uint32_t max_version = 0;
    if (rocEncodeAPIGetMaxSupportedVersion(&max_version) != ROCENC_SUCCESS) {
        std::fprintf(stderr, "rocEncodeAPIGetMaxSupportedVersion failed\n");
        return 1;
    }
    std::printf("rocEncode API version %u.%u\n", max_version & 0xf, (max_version >> 24) & 0xf);

    ROCENCODE_API_FUNCTION_LIST api;
    std::memset(&api, 0, sizeof(api));
    api.version = ROCENCODE_API_FUNCTION_LIST_VER;
    rocEncStatus status = rocEncodeAPICreateInstance(&api);
    if (status != ROCENC_SUCCESS) {
        std::fprintf(stderr, "rocEncodeAPICreateInstance failed: %d\n", static_cast<int>(status));
        return 1;
    }

    bool saw_any_encoder = false;
    bool saw_any_device = false;
    if (only_device >= 0) {
        saw_any_device = ReportDevice(api, only_device, &saw_any_encoder);
        if (!saw_any_device) {
            std::fprintf(stderr, "device %d could not be opened\n", only_device);
            return 1;
        }
    } else {
        // Walk ordinals until one cannot be opened.  A device with no encoder
        // still opens, so this stops at the end of the device list rather than
        // at the first decode-only part.
        for (int device_id = 0; device_id < 64; device_id++) {
            if (!ReportDevice(api, device_id, &saw_any_encoder)) {
                break;
            }
            saw_any_device = true;
        }
    }

    if (!saw_any_device) {
        std::printf("\nNo devices could be opened.\n");
        return 1;
    }
    if (!saw_any_encoder) {
        std::printf("\nNo encode-capable device was found. This is a clean, expected result on\n"
                    "decode-only silicon; it is not an error.\n");
    }
    return 0;
}
