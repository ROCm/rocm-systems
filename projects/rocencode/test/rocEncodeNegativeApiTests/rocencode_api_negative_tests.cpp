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

// rocEncode negative API tests.
//
// Everything here must hold on any machine, encode-capable or not: the point
// is that bad arguments produce a status code rather than a crash, and that
// every declared-but-unimplemented entry point is reachable through the
// function list and refuses cleanly.

#include <cstdio>
#include <cstring>
#include <vector>

#include <rocencode/rocencode.h>

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool condition, const char* what) {
    g_checks++;
    if (!condition) {
        g_failures++;
        std::printf("  FAIL: %s\n", what);
    } else {
        std::printf("  ok  : %s\n", what);
    }
}

void CheckStatus(rocEncStatus actual, rocEncStatus expected, const char* what) {
    g_checks++;
    if (actual != expected) {
        g_failures++;
        std::printf("  FAIL: %s (expected %d, got %d)\n", what, static_cast<int>(expected),
                    static_cast<int>(actual));
    } else {
        std::printf("  ok  : %s\n", what);
    }
}

void TestCreateInstance() {
    std::printf("\n[rocEncodeAPICreateInstance]\n");

    CheckStatus(rocEncodeAPICreateInstance(nullptr), ROCENC_ERR_INVALID_PTR,
                "NULL function list is rejected");

    ROCENCODE_API_FUNCTION_LIST bad;
    std::memset(&bad, 0, sizeof(bad));
    bad.version = 0xdeadbeef;
    CheckStatus(rocEncodeAPICreateInstance(&bad), ROCENC_ERR_INVALID_VERSION,
                "wrong function list version is rejected");

    ROCENCODE_API_FUNCTION_LIST api;
    std::memset(&api, 0, sizeof(api));
    api.version = ROCENCODE_API_FUNCTION_LIST_VER;
    CheckStatus(rocEncodeAPICreateInstance(&api), ROCENC_SUCCESS, "valid function list is populated");

    // Every entry point must be wired; none may be NULL.
    const void* entries[] = {
        reinterpret_cast<const void*>(api.rocEncOpenEncodeSession),
        reinterpret_cast<const void*>(api.rocEncGetEncodeCodecCount),
        reinterpret_cast<const void*>(api.rocEncGetEncodeProfileCount),
        reinterpret_cast<const void*>(api.rocEncGetEncodeProfiles),
        reinterpret_cast<const void*>(api.rocEncGetEncodeCodecs),
        reinterpret_cast<const void*>(api.rocEncGetInputFormatCount),
        reinterpret_cast<const void*>(api.rocEncGetInputFormats),
        reinterpret_cast<const void*>(api.rocEncGetEncodeCaps),
        reinterpret_cast<const void*>(api.rocEncGetEncodePresetCount),
        reinterpret_cast<const void*>(api.rocEncGetEncodePresets),
        reinterpret_cast<const void*>(api.rocEncGetEncodePresetConfig),
        reinterpret_cast<const void*>(api.rocEncInitializeEncoder),
        reinterpret_cast<const void*>(api.rocEncCreateInputBuffer),
        reinterpret_cast<const void*>(api.rocEncDestroyInputBuffer),
        reinterpret_cast<const void*>(api.rocEncCreateBitstreamBuffer),
        reinterpret_cast<const void*>(api.rocEncDestroyBitstreamBuffer),
        reinterpret_cast<const void*>(api.rocEncEncodePicture),
        reinterpret_cast<const void*>(api.rocEncLockBitstream),
        reinterpret_cast<const void*>(api.rocEncUnlockBitstream),
        reinterpret_cast<const void*>(api.rocEncLockInputBuffer),
        reinterpret_cast<const void*>(api.rocEncUnlockInputBuffer),
        reinterpret_cast<const void*>(api.rocEncGetEncodeStats),
        reinterpret_cast<const void*>(api.rocEncGetSequenceParams),
        reinterpret_cast<const void*>(api.rocEncRegisterAsyncEvent),
        reinterpret_cast<const void*>(api.rocEncUnregisterAsyncEvent),
        reinterpret_cast<const void*>(api.rocEncMapInputResource),
        reinterpret_cast<const void*>(api.rocEncUnmapInputResource),
        reinterpret_cast<const void*>(api.rocEncDestroyEncoder),
        reinterpret_cast<const void*>(api.rocEncInvalidateRefFrames),
        reinterpret_cast<const void*>(api.rocEncOpenEncodeSessionEx),
        reinterpret_cast<const void*>(api.rocEncRegisterResource),
        reinterpret_cast<const void*>(api.rocEncUnregisterResource),
        reinterpret_cast<const void*>(api.rocEncReconfigureEncoder),
        reinterpret_cast<const void*>(api.rocEncCreateMVBuffer),
        reinterpret_cast<const void*>(api.rocEncDestroyMVBuffer),
        reinterpret_cast<const void*>(api.rocEncRunMotionEstimationOnly),
        reinterpret_cast<const void*>(api.rocEncGetLastErrorString),
        reinterpret_cast<const void*>(api.rocEncSetIOHipStreams),
        reinterpret_cast<const void*>(api.rocEncGetEncodePresetConfigEx),
        reinterpret_cast<const void*>(api.rocEncGetSequenceParamEx),
        reinterpret_cast<const void*>(api.rocEncRestoreEncoderState),
        reinterpret_cast<const void*>(api.rocEncLookaheadPicture),
    };
    bool all_set = true;
    for (size_t i = 0; i < sizeof(entries) / sizeof(entries[0]); i++) {
        if (entries[i] == nullptr) {
            std::printf("  (function list entry %zu is NULL)\n", i);
            all_set = false;
        }
    }
    Check(all_set, "no function list entry is NULL");
    Check(api.reserved1 == nullptr, "reserved1 is NULL");
}

void TestNoHandle(ROCENCODE_API_FUNCTION_LIST& api) {
    std::printf("\n[NULL / invalid handle]\n");

    uint32_t count = 0;
    CheckStatus(api.rocEncGetEncodeCodecCount(nullptr, &count), ROCENC_ERR_INVALID_ENCODERDEVICE,
                "rocEncGetEncodeCodecCount(NULL encoder)");
    CheckStatus(api.rocEncDestroyEncoder(nullptr), ROCENC_ERR_INVALID_ENCODERDEVICE,
                "rocEncDestroyEncoder(NULL encoder)");

    const char* err = api.rocEncGetLastErrorString(nullptr);
    Check(err != nullptr, "rocEncGetLastErrorString(NULL) returns a string, not NULL");

    CheckStatus(api.rocEncOpenEncodeSessionEx(nullptr, nullptr), ROCENC_ERR_INVALID_PTR,
                "rocEncOpenEncodeSessionEx(NULL params)");

    void* encoder = nullptr;
    ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS params;

    std::memset(&params, 0, sizeof(params));
    params.version = 0xdeadbeef;
    params.deviceType = ROCENC_DEVICE_TYPE_HIP;
    params.apiVersion = ROCENCAPI_VERSION;
    CheckStatus(api.rocEncOpenEncodeSessionEx(&params, &encoder), ROCENC_ERR_INVALID_VERSION,
                "rocEncOpenEncodeSessionEx with a bad struct version");

    std::memset(&params, 0, sizeof(params));
    params.version = ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    params.deviceType = ROCENC_DEVICE_TYPE_DIRECTX;
    params.apiVersion = ROCENCAPI_VERSION;
    CheckStatus(api.rocEncOpenEncodeSessionEx(&params, &encoder), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "ROCENC_DEVICE_TYPE_DIRECTX is rejected");

    params.deviceType = ROCENC_DEVICE_TYPE_OPENGL;
    CheckStatus(api.rocEncOpenEncodeSessionEx(&params, &encoder), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "ROCENC_DEVICE_TYPE_OPENGL is rejected");

    void* legacy = nullptr;
    CheckStatus(api.rocEncOpenEncodeSession(nullptr, ROCENC_DEVICE_TYPE_HIP, &legacy),
                ROCENC_ERR_INVALID_CALL, "deprecated rocEncOpenEncodeSession returns INVALID_CALL");
}

// Session-scoped checks.  Skipped (not failed) when no device can be opened,
// so the test is still meaningful on a machine with no GPU at all.
void TestWithSession(ROCENCODE_API_FUNCTION_LIST& api) {
    std::printf("\n[valid session]\n");

    ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS params;
    std::memset(&params, 0, sizeof(params));
    params.version = ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    params.deviceType = ROCENC_DEVICE_TYPE_HIP;
    params.device = reinterpret_cast<void*>(static_cast<intptr_t>(0));
    params.apiVersion = ROCENCAPI_VERSION;

    void* encoder = nullptr;
    rocEncStatus status = api.rocEncOpenEncodeSessionEx(&params, &encoder);
    if (status != ROCENC_SUCCESS) {
        std::printf("  skip: no device 0 (status %d); session-scoped checks not run\n",
                    static_cast<int>(status));
        return;
    }

    uint32_t count = 0xffffffffu;
    CheckStatus(api.rocEncGetEncodeCodecCount(encoder, &count), ROCENC_SUCCESS,
                "rocEncGetEncodeCodecCount succeeds");
    Check(count <= 3, "codec count is at most 3 (H.264, HEVC, AV1)");
    std::printf("  info: %u encode-capable codec(s) on device 0\n", count);

    CheckStatus(api.rocEncGetEncodeCodecCount(encoder, nullptr), ROCENC_ERR_INVALID_PTR,
                "rocEncGetEncodeCodecCount(NULL out)");

    // Codec values that are not one of ours.
    uint32_t profile_count = 0;
    CheckStatus(api.rocEncGetEncodeProfileCount(encoder, static_cast<ROCENC_CODEC>(0xABABABAB), &profile_count),
                ROCENC_ERR_INVALID_PARAM, "out-of-range codec is rejected");
    CheckStatus(api.rocEncGetEncodeProfileCount(encoder, ROCENC_CODEC_UNDEFINED, &profile_count),
                ROCENC_ERR_INVALID_PARAM, "ROCENC_CODEC_UNDEFINED is rejected");

    // Caps queries.
    ROCENC_CAPS_PARAM caps_param;
    int caps_val = 0;

    std::memset(&caps_param, 0, sizeof(caps_param));
    caps_param.version = 0xdeadbeef;
    caps_param.capsToQuery = ROCENC_CAPS_WIDTH_MAX;
    CheckStatus(api.rocEncGetEncodeCaps(encoder, ROCENC_CODEC_H264, &caps_param, &caps_val),
                ROCENC_ERR_INVALID_VERSION, "ROCENC_CAPS_PARAM version is checked");

    std::memset(&caps_param, 0, sizeof(caps_param));
    caps_param.version = ROCENC_CAPS_PARAM_VER;
    caps_param.capsToQuery = ROCENC_CAPS_EXPOSED_COUNT;
    CheckStatus(api.rocEncGetEncodeCaps(encoder, ROCENC_CODEC_H264, &caps_param, &caps_val),
                ROCENC_ERR_INVALID_PARAM, "ROCENC_CAPS_EXPOSED_COUNT is not queryable");

    std::memset(&caps_param, 0, sizeof(caps_param));
    caps_param.version = ROCENC_CAPS_PARAM_VER;
    caps_param.capsToQuery = static_cast<ROCENC_CAPS>(9999);
    CheckStatus(api.rocEncGetEncodeCaps(encoder, ROCENC_CODEC_H264, &caps_param, &caps_val),
                ROCENC_ERR_INVALID_PARAM, "out-of-range capability is rejected");

    std::memset(&caps_param, 0, sizeof(caps_param));
    caps_param.version = ROCENC_CAPS_PARAM_VER;
    caps_param.capsToQuery = ROCENC_CAPS_WIDTH_MAX;
    CheckStatus(api.rocEncGetEncodeCaps(encoder, ROCENC_CODEC_H264, &caps_param, nullptr),
                ROCENC_ERR_INVALID_PTR, "rocEncGetEncodeCaps(NULL capsVal)");

    // A codec the device cannot encode answers 0, not an error.
    caps_val = -1;
    std::memset(&caps_param, 0, sizeof(caps_param));
    caps_param.version = ROCENC_CAPS_PARAM_VER;
    caps_param.capsToQuery = ROCENC_CAPS_WIDTH_MAX;
    CheckStatus(api.rocEncGetEncodeCaps(encoder, ROCENC_CODEC_AV1, &caps_param, &caps_val),
                ROCENC_SUCCESS, "caps query on a possibly-unsupported codec still succeeds");
    Check(caps_val >= 0, "capability value is never negative");

    // Undersized output arrays.
    ROCENC_CODEC one_codec;
    uint32_t returned = 0;
    if (count > 1) {
        CheckStatus(api.rocEncGetEncodeCodecs(encoder, &one_codec, 1, &returned),
                    ROCENC_ERR_INVALID_PARAM, "undersized codec array is rejected");
    }

    // Every declared-but-unimplemented entry point refuses cleanly.
    std::printf("\n[unimplemented entry points]\n");
    CheckStatus(api.rocEncGetEncodePresetCount(encoder, ROCENC_CODEC_H264, &count),
                ROCENC_ERR_UNSUPPORTED_DEVICE, "rocEncGetEncodePresetCount");
    CheckStatus(api.rocEncGetEncodePresets(encoder, ROCENC_CODEC_H264, nullptr, 0, &count),
                ROCENC_ERR_UNSUPPORTED_DEVICE, "rocEncGetEncodePresets");
    CheckStatus(api.rocEncGetEncodePresetConfig(encoder, ROCENC_CODEC_H264, ROCENC_PRESET_P4,
                                                nullptr),
                ROCENC_ERR_UNSUPPORTED_DEVICE, "rocEncGetEncodePresetConfig");
    CheckStatus(api.rocEncGetEncodePresetConfigEx(encoder, ROCENC_CODEC_H264, ROCENC_PRESET_P4,
                                                  ROCENC_TUNING_INFO_HIGH_QUALITY, nullptr),
                ROCENC_ERR_UNSUPPORTED_DEVICE, "rocEncGetEncodePresetConfigEx");
    CheckStatus(api.rocEncInitializeEncoder(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncInitializeEncoder");
    CheckStatus(api.rocEncCreateInputBuffer(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncCreateInputBuffer");
    CheckStatus(api.rocEncDestroyInputBuffer(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncDestroyInputBuffer");
    CheckStatus(api.rocEncCreateBitstreamBuffer(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncCreateBitstreamBuffer");
    CheckStatus(api.rocEncDestroyBitstreamBuffer(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncDestroyBitstreamBuffer");
    CheckStatus(api.rocEncEncodePicture(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncEncodePicture");
    CheckStatus(api.rocEncLockBitstream(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncLockBitstream");
    CheckStatus(api.rocEncUnlockBitstream(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncUnlockBitstream");
    CheckStatus(api.rocEncLockInputBuffer(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncLockInputBuffer");
    CheckStatus(api.rocEncUnlockInputBuffer(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncUnlockInputBuffer");
    CheckStatus(api.rocEncGetEncodeStats(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncGetEncodeStats");
    CheckStatus(api.rocEncGetSequenceParams(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncGetSequenceParams");
    CheckStatus(api.rocEncGetSequenceParamEx(encoder, nullptr, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncGetSequenceParamEx");
    CheckStatus(api.rocEncRegisterAsyncEvent(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncRegisterAsyncEvent");
    CheckStatus(api.rocEncUnregisterAsyncEvent(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncUnregisterAsyncEvent");
    CheckStatus(api.rocEncMapInputResource(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncMapInputResource");
    CheckStatus(api.rocEncUnmapInputResource(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncUnmapInputResource");
    CheckStatus(api.rocEncInvalidateRefFrames(encoder, 0), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncInvalidateRefFrames");
    CheckStatus(api.rocEncRegisterResource(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncRegisterResource");
    CheckStatus(api.rocEncUnregisterResource(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncUnregisterResource");
    CheckStatus(api.rocEncReconfigureEncoder(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncReconfigureEncoder");
    CheckStatus(api.rocEncCreateMVBuffer(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncCreateMVBuffer");
    CheckStatus(api.rocEncDestroyMVBuffer(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncDestroyMVBuffer");
    CheckStatus(api.rocEncRunMotionEstimationOnly(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncRunMotionEstimationOnly");
    CheckStatus(api.rocEncRestoreEncoderState(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncRestoreEncoderState");
    CheckStatus(api.rocEncLookaheadPicture(encoder, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncLookaheadPicture");
    CheckStatus(api.rocEncSetIOHipStreams(encoder, nullptr, nullptr), ROCENC_ERR_UNSUPPORTED_DEVICE,
                "rocEncSetIOHipStreams");

    Check(api.rocEncGetLastErrorString(encoder) != nullptr,
          "rocEncGetLastErrorString returns a message after a failure");

    CheckStatus(api.rocEncDestroyEncoder(encoder), ROCENC_SUCCESS, "rocEncDestroyEncoder succeeds");
}

}  // namespace

int main() {
    std::printf("rocEncode negative API tests\n");

    TestCreateInstance();

    ROCENCODE_API_FUNCTION_LIST api;
    std::memset(&api, 0, sizeof(api));
    api.version = ROCENCODE_API_FUNCTION_LIST_VER;
    if (rocEncodeAPICreateInstance(&api) != ROCENC_SUCCESS) {
        std::printf("FATAL: rocEncodeAPICreateInstance failed\n");
        return 1;
    }

    TestNoHandle(api);
    TestWithSession(api);

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
