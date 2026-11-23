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

#include <string>
#include <vector>

#include "va_context.h"
#include "../../commons.h"
#include "../../../api/rocencode/rocencode.h"

/**
 * @brief The only layer that talks to VA-API.
 *
 * Phase 1 scope is the capability probe: which codecs the device can encode,
 * which profiles and input formats those codecs expose, and the value of every
 * ROCENC_CAPS_* query.  Surface allocation, context creation and encode
 * submission arrive in later phases and belong here too; nothing above this
 * class ever includes <va/va.h>.
 *
 * It never touches HIP.  The HIP device identity is handed to it by
 * RocEncoder.
 */
class VaapiVideoEncoder {
public:
    VaapiVideoEncoder() = default;
    ~VaapiVideoEncoder() = default;

    VaapiVideoEncoder(const VaapiVideoEncoder&) = delete;
    VaapiVideoEncoder& operator=(const VaapiVideoEncoder&) = delete;

    // Binds this encoder to the VA context of the given HIP device.  Returns
    // false only when the device or its render node cannot be reached; a
    // device with no encoder binds successfully and then reports no supported
    // codecs.
    bool Initialize(const HipDeviceIdentity& ident);

    // Codecs with at least one VAEntrypointEncSlice profile on this device.
    const std::vector<RocEncCodec>& SupportedCodecs() const { return supported_codecs_; }

    bool IsCodecSupported(RocEncCodec codec) const;

    // Profiles available for a codec, as public ROCENC_PROFILE values.
    std::vector<ROCENC_PROFILE> SupportedProfiles(RocEncCodec codec) const;

    // Input buffer formats accepted for a codec.
    std::vector<ROCENC_BUFFER_FORMAT> SupportedInputFormats(RocEncCodec codec) const;

    // Resolves one ROCENC_CAPS_* query.  Returns false when caps_to_query is
    // outside ROCENC_CAPS.  An unsupported capability is 0 with a true return.
    bool GetCapsValue(RocEncCodec codec, ROCENC_CAPS caps_to_query, int* caps_val) const;

    std::string VendorString() const;

private:
    uint32_t va_ctx_id_ = 0;
    bool initialized_ = false;
    std::vector<RocEncCodec> supported_codecs_;
    VaEncodeCaps caps_[3];

    const VaEncodeCaps& CapsFor(RocEncCodec codec) const;
};
