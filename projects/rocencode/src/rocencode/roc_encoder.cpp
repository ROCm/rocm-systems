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

#include "roc_encoder.h"

#include <algorithm>
#include <cctype>

#include <hip/hip_runtime.h>

HipDeviceIdentity RocEncoder::QueryHipDeviceIdentity(int device_id) {
    HipDeviceIdentity ident;
    ident.device_id = device_id;

    int num_devices = 0;
    hipError_t hip_status = hipGetDeviceCount(&num_devices);
    if (hip_status != hipSuccess) {
        THROW(ROCENC_STR("hipGetDeviceCount failed: ") + hipGetErrorName(hip_status));
    }
    if (num_devices < 1) {
        THROW(ROCENC_STR("No HIP devices found."));
    }
    if (device_id < 0 || device_id >= num_devices) {
        THROW(ROCENC_STR("Requested device_id ") + ROCENC_TOSTR(device_id) + " is out of range.");
    }

    hipDeviceProp_t prop;
    hip_status = hipGetDeviceProperties(&prop, device_id);
    if (hip_status != hipSuccess) {
        THROW(ROCENC_STR("hipGetDeviceProperties failed: ") + hipGetErrorName(hip_status));
    }
    ident.gpu_uuid.assign(prop.uuid.bytes, sizeof(prop.uuid.bytes));
    ident.name.assign(prop.name);

    char pci_bus_id_buf[64] = {0};
    if (hipDeviceGetPCIBusId(pci_bus_id_buf, sizeof(pci_bus_id_buf), device_id) == hipSuccess) {
        std::string bdf(pci_bus_id_buf);
        std::transform(bdf.begin(), bdf.end(), bdf.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        // Drop the PCI function suffix so partition children match their base BDF.
        size_t dot_pos = bdf.find_last_of('.');
        if (dot_pos != std::string::npos) {
            bdf = bdf.substr(0, dot_pos);
        }
        ident.gpu_pci_bdf = bdf;
    }
    return ident;
}

RocEncoder::RocEncoder(ROCENC_DEVICE_TYPE device_type, void* device) {
    if (device_type != ROCENC_DEVICE_TYPE_HIP) {
        THROW(ROCENC_STR("Unsupported device type ") + ROCENC_TOSTR(static_cast<int>(device_type)) +
              "; rocEncode supports ROCENC_DEVICE_TYPE_HIP only.");
    }
    // For ROCENC_DEVICE_TYPE_HIP the device field carries the HIP device
    // ordinal by value, not a pointer to one.
    int device_id = static_cast<int>(reinterpret_cast<intptr_t>(device));
    identity_ = QueryHipDeviceIdentity(device_id);

    vaapi_encoder_ = std::make_unique<VaapiVideoEncoder>();
    if (!vaapi_encoder_->Initialize(identity_)) {
        THROW(ROCENC_STR("Failed to initialize the VA-API encoder context for device ") +
              ROCENC_TOSTR(device_id));
    }
}

bool RocEncoder::CodecFromPublic(ROCENC_CODEC public_codec, RocEncCodec* codec) {
    switch (public_codec) {
        case ROCENC_CODEC_H264: *codec = RocEncCodec::kH264; return true;
        case ROCENC_CODEC_HEVC: *codec = RocEncCodec::kHevc; return true;
        case ROCENC_CODEC_AV1:  *codec = RocEncCodec::kAv1;  return true;
        case ROCENC_CODEC_UNDEFINED: break;
    }
    return false;
}

ROCENC_CODEC RocEncoder::PublicFromCodec(RocEncCodec codec) {
    switch (codec) {
        case RocEncCodec::kH264: return ROCENC_CODEC_H264;
        case RocEncCodec::kHevc: return ROCENC_CODEC_HEVC;
        case RocEncCodec::kAv1:  return ROCENC_CODEC_AV1;
    }
    return ROCENC_CODEC_UNDEFINED;
}

rocEncStatus RocEncoder::GetEncodeCodecCount(uint32_t* codec_count) {
    if (codec_count == nullptr) {
        SetLastError("rocEncGetEncodeCodecCount: codecCount is NULL");
        return ROCENC_ERR_INVALID_PTR;
    }
    *codec_count = static_cast<uint32_t>(vaapi_encoder_->SupportedCodecs().size());
    return ROCENC_SUCCESS;
}

rocEncStatus RocEncoder::GetEncodeCodecs(ROCENC_CODEC* codecs, uint32_t codec_array_size, uint32_t* codec_count) {
    if (codecs == nullptr || codec_count == nullptr) {
        SetLastError("rocEncGetEncodeCodecs: NULL output pointer");
        return ROCENC_ERR_INVALID_PTR;
    }
    const std::vector<RocEncCodec>& supported = vaapi_encoder_->SupportedCodecs();
    if (codec_array_size < supported.size()) {
        SetLastError("rocEncGetEncodeCodecs: codecArraySize is too small");
        return ROCENC_ERR_INVALID_PARAM;
    }
    for (size_t i = 0; i < supported.size(); i++) {
        codecs[i] = PublicFromCodec(supported[i]);
    }
    *codec_count = static_cast<uint32_t>(supported.size());
    return ROCENC_SUCCESS;
}

rocEncStatus RocEncoder::GetEncodeProfileCount(ROCENC_CODEC public_codec, uint32_t* profile_count) {
    if (profile_count == nullptr) {
        SetLastError("rocEncGetEncodeProfileCount: profileCount is NULL");
        return ROCENC_ERR_INVALID_PTR;
    }
    RocEncCodec codec;
    if (!CodecFromPublic(public_codec, &codec)) {
        SetLastError("rocEncGetEncodeProfileCount: unknown codec");
        return ROCENC_ERR_INVALID_PARAM;
    }
    *profile_count = static_cast<uint32_t>(vaapi_encoder_->SupportedProfiles(codec).size());
    return ROCENC_SUCCESS;
}

rocEncStatus RocEncoder::GetEncodeProfiles(ROCENC_CODEC public_codec, ROCENC_PROFILE* profiles,
                                           uint32_t profile_array_size, uint32_t* profile_count) {
    if (profiles == nullptr || profile_count == nullptr) {
        SetLastError("rocEncGetEncodeProfiles: NULL output pointer");
        return ROCENC_ERR_INVALID_PTR;
    }
    RocEncCodec codec;
    if (!CodecFromPublic(public_codec, &codec)) {
        SetLastError("rocEncGetEncodeProfiles: unknown codec");
        return ROCENC_ERR_INVALID_PARAM;
    }
    std::vector<ROCENC_PROFILE> supported = vaapi_encoder_->SupportedProfiles(codec);
    if (profile_array_size < supported.size()) {
        SetLastError("rocEncGetEncodeProfiles: profileArraySize is too small");
        return ROCENC_ERR_INVALID_PARAM;
    }
    for (size_t i = 0; i < supported.size(); i++) {
        profiles[i] = supported[i];
    }
    *profile_count = static_cast<uint32_t>(supported.size());
    return ROCENC_SUCCESS;
}

rocEncStatus RocEncoder::GetInputFormatCount(ROCENC_CODEC public_codec, uint32_t* input_fmt_count) {
    if (input_fmt_count == nullptr) {
        SetLastError("rocEncGetInputFormatCount: inputFmtCount is NULL");
        return ROCENC_ERR_INVALID_PTR;
    }
    RocEncCodec codec;
    if (!CodecFromPublic(public_codec, &codec)) {
        SetLastError("rocEncGetInputFormatCount: unknown codec");
        return ROCENC_ERR_INVALID_PARAM;
    }
    *input_fmt_count = static_cast<uint32_t>(vaapi_encoder_->SupportedInputFormats(codec).size());
    return ROCENC_SUCCESS;
}

rocEncStatus RocEncoder::GetInputFormats(ROCENC_CODEC public_codec, ROCENC_BUFFER_FORMAT* input_fmts,
                                         uint32_t input_fmt_array_size, uint32_t* input_fmt_count) {
    if (input_fmts == nullptr || input_fmt_count == nullptr) {
        SetLastError("rocEncGetInputFormats: NULL output pointer");
        return ROCENC_ERR_INVALID_PTR;
    }
    RocEncCodec codec;
    if (!CodecFromPublic(public_codec, &codec)) {
        SetLastError("rocEncGetInputFormats: unknown codec");
        return ROCENC_ERR_INVALID_PARAM;
    }
    std::vector<ROCENC_BUFFER_FORMAT> formats = vaapi_encoder_->SupportedInputFormats(codec);
    if (input_fmt_array_size < formats.size()) {
        SetLastError("rocEncGetInputFormats: inputFmtArraySize is too small");
        return ROCENC_ERR_INVALID_PARAM;
    }
    for (size_t i = 0; i < formats.size(); i++) {
        input_fmts[i] = formats[i];
    }
    *input_fmt_count = static_cast<uint32_t>(formats.size());
    return ROCENC_SUCCESS;
}

rocEncStatus RocEncoder::GetEncodeCaps(ROCENC_CODEC public_codec, ROCENC_CAPS_PARAM* caps_param, int* caps_val) {
    if (caps_param == nullptr || caps_val == nullptr) {
        SetLastError("rocEncGetEncodeCaps: NULL pointer");
        return ROCENC_ERR_INVALID_PTR;
    }
    if (caps_param->version != ROCENC_CAPS_PARAM_VER) {
        SetLastError("rocEncGetEncodeCaps: ROCENC_CAPS_PARAM::version mismatch");
        return ROCENC_ERR_INVALID_VERSION;
    }
    RocEncCodec codec;
    if (!CodecFromPublic(public_codec, &codec)) {
        SetLastError("rocEncGetEncodeCaps: unknown codec");
        return ROCENC_ERR_INVALID_PARAM;
    }
    if (!vaapi_encoder_->GetCapsValue(codec, caps_param->capsToQuery, caps_val)) {
        SetLastError("rocEncGetEncodeCaps: capsToQuery is out of range");
        return ROCENC_ERR_INVALID_PARAM;
    }
    return ROCENC_SUCCESS;
}
