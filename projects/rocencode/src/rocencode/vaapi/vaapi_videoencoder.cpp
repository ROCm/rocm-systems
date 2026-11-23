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

#include "vaapi_videoencoder.h"

#include <algorithm>

namespace {

// Candidate profile order, matching VaEncodeCaps::profile_mask bit positions
// (see CandidateProfiles() in va_context.cpp):
//   H.264 : 0 ConstrainedBaseline, 1 Main, 2 High
//   HEVC  : 0 Main,                1 Main10
//   AV1   : 0 Profile0
constexpr uint32_t kH264BitConstrainedBaseline = 1u << 0;
constexpr uint32_t kH264BitMain                = 1u << 1;
constexpr uint32_t kH264BitHigh                = 1u << 2;
constexpr uint32_t kHevcBitMain                = 1u << 0;
constexpr uint32_t kHevcBitMain10              = 1u << 1;
constexpr uint32_t kAv1BitProfile0             = 1u << 0;

// ROCENC_CAPS_SUPPORTED_RATECONTROL_MODES convention: the mask is the OR of the
// ROCENC_PARAMS_RC_MODE *values themselves* (VBR -> 0x1, CBR -> 0x2), which is
// what NVENC means by "a bitmask of the values in NV_ENC_PARAMS_RC_MODE".
//
// Do not "fix" this to (1 << mode).  Ported code writes
//   if (caps & ROCENC_PARAMS_RC_CBR)
// and a shifted convention makes that test silently report VBR when the
// application asked about CBR.
//
// CONSTQP == 0 contributes no bit, exactly as on NVENC.  It is unconditionally
// available on every encode-capable VCN part, so nothing is lost: see
// docs/caps-mapping.md.
constexpr int kRcBitVbr = ROCENC_PARAMS_RC_VBR;
constexpr int kRcBitCbr = ROCENC_PARAMS_RC_CBR;

}  // namespace

bool VaapiVideoEncoder::Initialize(const HipDeviceIdentity& ident) {
    if (!VaContext::GetInstance().GetVaContext(ident, &va_ctx_id_)) {
        return false;
    }
    supported_codecs_.clear();
    for (RocEncCodec codec : {RocEncCodec::kH264, RocEncCodec::kHevc, RocEncCodec::kAv1}) {
        VaEncodeCaps caps;
        if (!VaContext::GetInstance().GetEncodeCaps(va_ctx_id_, codec, &caps)) {
            continue;
        }
        caps_[static_cast<size_t>(codec)] = caps;
        if (caps.supported) {
            supported_codecs_.push_back(codec);
        }
    }
    initialized_ = true;
    if (supported_codecs_.empty()) {
        // Not an error.  MI300-class parts and Navi 24 (VCN 3.0.33) land here
        // and must report no-encode cleanly.
        InfoLog(g_rocenc_logger,
                "No encode-capable VA profiles on device " + ROCENC_TOSTR(ident.device_id) +
                    " -- the session opens and reports zero codecs.");
    }
    return true;
}

const VaEncodeCaps& VaapiVideoEncoder::CapsFor(RocEncCodec codec) const {
    return caps_[static_cast<size_t>(codec)];
}

bool VaapiVideoEncoder::IsCodecSupported(RocEncCodec codec) const {
    return initialized_ && CapsFor(codec).supported;
}

std::string VaapiVideoEncoder::VendorString() const {
    if (!initialized_) {
        return std::string();
    }
    return VaContext::GetInstance().GetVendorString(va_ctx_id_);
}

std::vector<ROCENC_PROFILE> VaapiVideoEncoder::SupportedProfiles(RocEncCodec codec) const {
    std::vector<ROCENC_PROFILE> profiles;
    if (!IsCodecSupported(codec)) {
        return profiles;
    }
    const VaEncodeCaps& caps = CapsFor(codec);
    // AUTOSELECT is always offered when the codec itself is supported.
    profiles.push_back(ROCENC_PROFILE_AUTOSELECT);
    switch (codec) {
        case RocEncCodec::kH264:
            if (caps.profile_mask & kH264BitConstrainedBaseline) {
                profiles.push_back(ROCENC_H264_PROFILE_BASELINE);
            }
            if (caps.profile_mask & kH264BitMain) {
                profiles.push_back(ROCENC_H264_PROFILE_MAIN);
            }
            if (caps.profile_mask & kH264BitHigh) {
                profiles.push_back(ROCENC_H264_PROFILE_HIGH);
            }
            if (caps.supports_yuv444) {
                profiles.push_back(ROCENC_H264_PROFILE_HIGH_444);
            }
            break;
        case RocEncCodec::kHevc:
            if (caps.profile_mask & kHevcBitMain) {
                profiles.push_back(ROCENC_HEVC_PROFILE_MAIN);
            }
            if (caps.profile_mask & kHevcBitMain10) {
                profiles.push_back(ROCENC_HEVC_PROFILE_MAIN10);
            }
            if (caps.supports_yuv444) {
                profiles.push_back(ROCENC_HEVC_PROFILE_FREXT);
            }
            break;
        case RocEncCodec::kAv1:
            if (caps.profile_mask & kAv1BitProfile0) {
                profiles.push_back(ROCENC_AV1_PROFILE_MAIN);
            }
            break;
    }
    return profiles;
}

std::vector<ROCENC_BUFFER_FORMAT> VaapiVideoEncoder::SupportedInputFormats(RocEncCodec codec) const {
    std::vector<ROCENC_BUFFER_FORMAT> formats;
    if (!IsCodecSupported(codec)) {
        return formats;
    }
    const VaEncodeCaps& caps = CapsFor(codec);
    // NV12 is the native VCN encode input layout and is implied by an
    // EncSlice-capable 4:2:0 config.
    formats.push_back(ROCENC_BUFFER_FORMAT_NV12);
    if (caps.supports_10bit) {
        formats.push_back(ROCENC_BUFFER_FORMAT_YUV420_10BIT);
    }
    if (caps.supports_yuv444) {
        formats.push_back(ROCENC_BUFFER_FORMAT_YUV444);
    }
    return formats;
}

bool VaapiVideoEncoder::GetCapsValue(RocEncCodec codec, ROCENC_CAPS caps_to_query, int* caps_val) const {
    if (caps_val == nullptr) {
        return false;
    }
    if (caps_to_query < 0 || caps_to_query >= ROCENC_CAPS_EXPOSED_COUNT) {
        return false;
    }
    *caps_val = 0;

    const VaEncodeCaps& caps = CapsFor(codec);
    if (!initialized_ || !caps.supported) {
        // Every capability of a codec this device cannot encode is 0.
        return true;
    }

    auto max_width = [&]() -> int {
        if (caps.va_max_width != 0) return static_cast<int>(caps.va_max_width);
        return static_cast<int>(caps.kernel_max_width);
    };
    auto max_height = [&]() -> int {
        if (caps.va_max_height != 0) return static_cast<int>(caps.va_max_height);
        return static_cast<int>(caps.kernel_max_height);
    };

    switch (caps_to_query) {
        // ---- VA config attributes -------------------------------------
        case ROCENC_CAPS_NUM_MAX_BFRAMES:
            // VAConfigAttribEncMaxRefFrames, L1 count.  A device with no L1
            // reference slots cannot emit B-frames.
            *caps_val = static_cast<int>(caps.max_ref_l1);
            break;
        case ROCENC_CAPS_SUPPORTED_RATECONTROL_MODES: {
            // VA_RC_CQP is deliberately not folded in: ROCENC_PARAMS_RC_CONSTQP
            // is 0 and so has no bit to set, matching NVENC.
            int mask = 0;
            if (caps.rate_control_mask & VA_RC_VBR) mask |= kRcBitVbr;
            if (caps.rate_control_mask & VA_RC_CBR) mask |= kRcBitCbr;
            *caps_val = mask;
            break;
        }
        case ROCENC_CAPS_SUPPORT_FIELD_ENCODING:
            *caps_val = caps.interlaced ? 1 : 0;  // VAConfigAttribEncInterlaced
            break;
        case ROCENC_CAPS_NUM_MAX_TEMPORAL_LAYERS:
            *caps_val = static_cast<int>(caps.max_temporal_layers);  // VAConfigAttribEncRateControlExt
            break;
        case ROCENC_CAPS_SUPPORT_TEMPORAL_SVC:
            *caps_val = (caps.max_temporal_layers > 1) ? 1 : 0;      // VAConfigAttribEncRateControlExt
            break;
        case ROCENC_CAPS_SUPPORT_INTRA_REFRESH:
            *caps_val = (caps.intra_refresh != 0) ? 1 : 0;           // VAConfigAttribEncIntraRefresh
            break;
        case ROCENC_CAPS_SUPPORT_DYNAMIC_SLICE_MODE:
            *caps_val = (caps.slice_structure != 0) ? 1 : 0;         // VAConfigAttribEncSliceStructure
            break;
        case ROCENC_CAPS_SUPPORT_YUV444_ENCODE:
            *caps_val = caps.supports_yuv444 ? 1 : 0;                // VAConfigAttribRTFormat
            break;
        case ROCENC_CAPS_SUPPORT_10BIT_ENCODE:
            *caps_val = caps.supports_10bit ? 1 : 0;                 // VAConfigAttribRTFormat
            break;
        case ROCENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES:
            *caps_val = (caps.max_ref_l0 > 1) ? 1 : 0;               // VAConfigAttribEncMaxRefFrames
            break;
        case ROCENC_CAPS_WIDTH_MIN:
            *caps_val = static_cast<int>(caps.va_min_width);         // VASurfaceAttribMinWidth
            break;
        case ROCENC_CAPS_HEIGHT_MIN:
            *caps_val = static_cast<int>(caps.va_min_height);        // VASurfaceAttribMinHeight
            break;
        case ROCENC_CAPS_WIDTH_MAX:
            *caps_val = max_width();   // VASurfaceAttribMaxWidth, else kernel max_width
            break;
        case ROCENC_CAPS_HEIGHT_MAX:
            *caps_val = max_height();  // VASurfaceAttribMaxHeight, else kernel max_height
            break;

        // ---- AMDGPU_INFO_VIDEO_CAPS -----------------------------------
        case ROCENC_CAPS_LEVEL_MAX:
            *caps_val = static_cast<int>(caps.kernel_max_level);
            break;
        case ROCENC_CAPS_MB_NUM_MAX: {
            uint32_t pixels = caps.kernel_max_pixels_per_frame;
            if (pixels == 0) {
                pixels = static_cast<uint32_t>(max_width()) * static_cast<uint32_t>(max_height());
            }
            *caps_val = static_cast<int>(pixels / 256u);
            break;
        }

        // ---- Fixed values ---------------------------------------------
        case ROCENC_CAPS_NUM_ENCODER_ENGINES:
            // Reported, never used to decide capability.  See
            // VaContext::QueryEngineCount().
            *caps_val = static_cast<int>(caps.num_enc_engines);
            break;
        case ROCENC_CAPS_DYNAMIC_QUERY_ENCODER_CAPACITY:
            // There is no virtualized-capacity query on Linux; NVENC also
            // returns 100 on bare metal.
            *caps_val = 100;
            break;
        case ROCENC_CAPS_SUPPORT_QPELMV:
            // Quarter-pel motion compensation is mandatory in H.264 and HEVC,
            // so a device that encodes the codec at all does quarter-pel.  AV1
            // uses eighth-pel and is reported as 0.
            *caps_val = (codec == RocEncCodec::kAv1) ? 0 : 1;
            break;
        case ROCENC_CAPS_SUPPORT_CABAC:
            // H.264 only, and only where a CABAC-capable profile is actually
            // exposed.  CABAC is forbidden in Constrained Baseline, so a device
            // that offers VAEntrypointEncSlice on Constrained Baseline alone
            // must report 0.  VA exposes no attribute for entropy coding, so
            // the profile set is the only honest source.
            *caps_val = (codec == RocEncCodec::kH264 &&
                         (caps.profile_mask & (kH264BitMain | kH264BitHigh)) != 0)
                            ? 1
                            : 0;
            break;
        case ROCENC_CAPS_LEVEL_MIN:
            *caps_val = ROCENC_LEVEL_AUTOSELECT;  // 0
            break;

        // ---- Hard 0: no VA attribute and no kernel field --------------
        case ROCENC_CAPS_SUPPORT_MONOCHROME:
        case ROCENC_CAPS_SUPPORT_FMO:
        case ROCENC_CAPS_SUPPORT_BDIRECT_MODE:
        case ROCENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM:
        case ROCENC_CAPS_SUPPORT_STEREO_MVC:
        case ROCENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES:
        case ROCENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES:
        case ROCENC_CAPS_SEPARATE_COLOUR_PLANE:
        case ROCENC_CAPS_SUPPORT_DYN_RES_CHANGE:
        case ROCENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE:
        case ROCENC_CAPS_SUPPORT_DYN_FORCE_CONSTQP:
        case ROCENC_CAPS_SUPPORT_DYN_RCMODE_CHANGE:
        case ROCENC_CAPS_SUPPORT_SUBFRAME_READBACK:
        case ROCENC_CAPS_SUPPORT_CONSTRAINED_ENCODING:
        case ROCENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE:
        case ROCENC_CAPS_SUPPORT_REF_PIC_INVALIDATION:
        case ROCENC_CAPS_PREPROC_SUPPORT:
        case ROCENC_CAPS_ASYNC_ENCODE_SUPPORT:
        case ROCENC_CAPS_MB_PER_SEC_MAX:
        case ROCENC_CAPS_SUPPORT_LOSSLESS_ENCODE:
        case ROCENC_CAPS_SUPPORT_SAO:
        case ROCENC_CAPS_SUPPORT_MEONLY_MODE:
        case ROCENC_CAPS_SUPPORT_LOOKAHEAD:
        case ROCENC_CAPS_SUPPORT_TEMPORAL_AQ:
        case ROCENC_CAPS_NUM_MAX_LTR_FRAMES:
        case ROCENC_CAPS_SUPPORT_WEIGHTED_PREDICTION:
        case ROCENC_CAPS_SUPPORT_BFRAME_REF_MODE:
        case ROCENC_CAPS_SUPPORT_EMPHASIS_LEVEL_MAP:
        case ROCENC_CAPS_SUPPORT_ALPHA_LAYER_ENCODING:
        case ROCENC_CAPS_SINGLE_SLICE_INTRA_REFRESH:
        case ROCENC_CAPS_DISABLE_ENC_STATE_ADVANCE:
        case ROCENC_CAPS_OUTPUT_RECON_SURFACE:
        case ROCENC_CAPS_OUTPUT_BLOCK_STATS:
        case ROCENC_CAPS_OUTPUT_ROW_STATS:
            *caps_val = 0;
            break;

        default:
            return false;
    }
    return true;
}
