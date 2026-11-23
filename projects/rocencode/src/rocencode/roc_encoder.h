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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vaapi/vaapi_videoencoder.h"
#include "../commons.h"
#include "../../api/rocencode/rocencode.h"

/**
 * @brief Session orchestration, and the only layer that touches HIP.
 *
 * Phase 1 uses HIP for one thing: turning the device ordinal the application
 * passed to rocEncOpenEncodeSessionEx() into a HipDeviceIdentity (UUID, PCI
 * BDF, marketing name) that VaContext can match against a DRM render node.
 * Everything below this class is VA-API only; nothing above it is.
 */
class RocEncoder {
public:
    RocEncoder(ROCENC_DEVICE_TYPE device_type, void* device);
    ~RocEncoder() = default;

    RocEncoder(const RocEncoder&) = delete;
    RocEncoder& operator=(const RocEncoder&) = delete;

    rocEncStatus GetEncodeCodecCount(uint32_t* codec_count);
    rocEncStatus GetEncodeCodecs(ROCENC_CODEC* codecs, uint32_t codec_array_size, uint32_t* codec_count);
    rocEncStatus GetEncodeProfileCount(ROCENC_CODEC codec, uint32_t* profile_count);
    rocEncStatus GetEncodeProfiles(ROCENC_CODEC codec, ROCENC_PROFILE* profiles,
                                   uint32_t profile_array_size, uint32_t* profile_count);
    rocEncStatus GetInputFormatCount(ROCENC_CODEC codec, uint32_t* input_fmt_count);
    rocEncStatus GetInputFormats(ROCENC_CODEC codec, ROCENC_BUFFER_FORMAT* input_fmts,
                                 uint32_t input_fmt_array_size, uint32_t* input_fmt_count);
    rocEncStatus GetEncodeCaps(ROCENC_CODEC codec, ROCENC_CAPS_PARAM* caps_param, int* caps_val);

    void SetLastError(const std::string& message) { last_error_ = message; }
    const char* GetLastError() const { return last_error_.c_str(); }

    int DeviceId() const { return identity_.device_id; }

private:
    HipDeviceIdentity identity_;
    std::unique_ptr<VaapiVideoEncoder> vaapi_encoder_;
    std::string last_error_;

    // The only HIP calls in the library.  Throws rocEncodeException on
    // failure.
    static HipDeviceIdentity QueryHipDeviceIdentity(int device_id);

    // Maps a public codec value onto the internal codec enum.  Returns false
    // for a value that is not a known codec.
    static bool CodecFromPublic(ROCENC_CODEC public_codec, RocEncCodec* codec);
    static ROCENC_CODEC PublicFromCodec(RocEncCodec codec);
};
