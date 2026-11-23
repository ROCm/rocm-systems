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

#include <cstring>
#include <new>
#include <string>

#include "enc_handle.h"
#include "roc_encoder.h"
#include "../commons.h"
#include "../../api/rocencode/rocencode.h"

namespace {

// Returned by rocEncGetLastErrorString when there is no usable handle to read
// a per-session message from.
const char kNoHandleError[] = "rocEncode: invalid or NULL encoder handle";
const char kNoErrorString[] = "";

EncHandle* AsHandle(void* encoder) {
    if (encoder == nullptr) {
        return nullptr;
    }
    EncHandle* handle = static_cast<EncHandle*>(encoder);
    if (!handle->IsValid()) {
        return nullptr;
    }
    return handle;
}

}  // namespace

extern "C" {

/*****************************************************************************************************/
//! Capability-query slice -- implemented in Phase 1.
/*****************************************************************************************************/

rocEncStatus ROCENCAPI rocEncOpenEncodeSessionEx(ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS* openSessionExParams,
                                                 void** encoder) {
    FunctionEntryLogWithArgs(g_rocenc_logger, RocEncFmtPtr(openSessionExParams) + ", " + RocEncFmtPtr(encoder));
    if (openSessionExParams == nullptr || encoder == nullptr) {
        CriticalLog(g_rocenc_logger, "Null pointer");
        return ROCENC_ERR_INVALID_PTR;
    }
    if (openSessionExParams->version != ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER) {
        CriticalLog(g_rocenc_logger, "ROCENC_OPEN_ENCODE_SESSION_EX_PARAMS::version mismatch");
        return ROCENC_ERR_INVALID_VERSION;
    }
    if (openSessionExParams->apiVersion > ROCENCAPI_VERSION) {
        CriticalLog(g_rocenc_logger, "apiVersion is newer than this library supports");
        return ROCENC_ERR_INVALID_VERSION;
    }
    if (openSessionExParams->deviceType != ROCENC_DEVICE_TYPE_HIP) {
        // ROCENC_DEVICE_TYPE_DIRECTX and _OPENGL are declared for structural
        // parity and rejected here, exactly as the plan specifies.
        CriticalLog(g_rocenc_logger, "Only ROCENC_DEVICE_TYPE_HIP is supported");
        return ROCENC_ERR_UNSUPPORTED_DEVICE;
    }

    EncHandle* handle = nullptr;
    try {
        handle = new EncHandle(openSessionExParams->deviceType, openSessionExParams->device);
    } catch (const std::exception& e) {
        CriticalLog(g_rocenc_logger, "Failed to open the encode session: " + ROCENC_STR(e.what()));
        delete handle;
        return ROCENC_ERR_NO_ENCODE_DEVICE;
    } catch (...) {
        CriticalLog(g_rocenc_logger, "Failed to open the encode session: unknown exception");
        return ROCENC_ERR_GENERIC;
    }
    *encoder = handle;
    return ROCENC_SUCCESS;
}

rocEncStatus ROCENCAPI rocEncOpenEncodeSession(void* device, uint32_t deviceType, void** encoder) {
    FunctionEntryLogWithArgs(g_rocenc_logger, RocEncFmtPtr(device) + ", " + ROCENC_TOSTR(deviceType));
    (void)device;
    (void)deviceType;
    (void)encoder;
    // Deprecated in NVENC and here: the Ex form is the only supported entry.
    return ROCENC_ERR_INVALID_CALL;
}

rocEncStatus ROCENCAPI rocEncGetEncodeCodecCount(void* encoder, uint32_t* codecCount) {
    FunctionEntryLogWithArgs(g_rocenc_logger, RocEncFmtPtr(encoder));
    EncHandle* handle = AsHandle(encoder);
    if (handle == nullptr) {
        return ROCENC_ERR_INVALID_ENCODERDEVICE;
    }
    try {
        return handle->roc_encoder_->GetEncodeCodecCount(codecCount);
    } catch (const std::exception& e) {
        handle->roc_encoder_->SetLastError(e.what());
        return ROCENC_ERR_GENERIC;
    }
}

rocEncStatus ROCENCAPI rocEncGetEncodeCodecs(void* encoder, ROCENC_CODEC* codecs, uint32_t codecArraySize,
                                             uint32_t* codecCount) {
    FunctionEntryLogWithArgs(g_rocenc_logger, RocEncFmtPtr(encoder));
    EncHandle* handle = AsHandle(encoder);
    if (handle == nullptr) {
        return ROCENC_ERR_INVALID_ENCODERDEVICE;
    }
    try {
        return handle->roc_encoder_->GetEncodeCodecs(codecs, codecArraySize, codecCount);
    } catch (const std::exception& e) {
        handle->roc_encoder_->SetLastError(e.what());
        return ROCENC_ERR_GENERIC;
    }
}

rocEncStatus ROCENCAPI rocEncGetEncodeProfileCount(void* encoder, ROCENC_CODEC codec, uint32_t* profileCount) {
    FunctionEntryLogWithArgs(g_rocenc_logger, RocEncFmtPtr(encoder));
    EncHandle* handle = AsHandle(encoder);
    if (handle == nullptr) {
        return ROCENC_ERR_INVALID_ENCODERDEVICE;
    }
    try {
        return handle->roc_encoder_->GetEncodeProfileCount(codec, profileCount);
    } catch (const std::exception& e) {
        handle->roc_encoder_->SetLastError(e.what());
        return ROCENC_ERR_GENERIC;
    }
}

rocEncStatus ROCENCAPI rocEncGetEncodeProfiles(void* encoder, ROCENC_CODEC codec, ROCENC_PROFILE* profiles,
                                               uint32_t profileArraySize, uint32_t* profileCount) {
    FunctionEntryLogWithArgs(g_rocenc_logger, RocEncFmtPtr(encoder));
    EncHandle* handle = AsHandle(encoder);
    if (handle == nullptr) {
        return ROCENC_ERR_INVALID_ENCODERDEVICE;
    }
    try {
        return handle->roc_encoder_->GetEncodeProfiles(codec, profiles, profileArraySize, profileCount);
    } catch (const std::exception& e) {
        handle->roc_encoder_->SetLastError(e.what());
        return ROCENC_ERR_GENERIC;
    }
}

rocEncStatus ROCENCAPI rocEncGetInputFormatCount(void* encoder, ROCENC_CODEC codec, uint32_t* inputFmtCount) {
    FunctionEntryLogWithArgs(g_rocenc_logger, RocEncFmtPtr(encoder));
    EncHandle* handle = AsHandle(encoder);
    if (handle == nullptr) {
        return ROCENC_ERR_INVALID_ENCODERDEVICE;
    }
    try {
        return handle->roc_encoder_->GetInputFormatCount(codec, inputFmtCount);
    } catch (const std::exception& e) {
        handle->roc_encoder_->SetLastError(e.what());
        return ROCENC_ERR_GENERIC;
    }
}

rocEncStatus ROCENCAPI rocEncGetInputFormats(void* encoder, ROCENC_CODEC codec,
                                             ROCENC_BUFFER_FORMAT* inputFmts, uint32_t inputFmtArraySize,
                                             uint32_t* inputFmtCount) {
    FunctionEntryLogWithArgs(g_rocenc_logger, RocEncFmtPtr(encoder));
    EncHandle* handle = AsHandle(encoder);
    if (handle == nullptr) {
        return ROCENC_ERR_INVALID_ENCODERDEVICE;
    }
    try {
        return handle->roc_encoder_->GetInputFormats(codec, inputFmts, inputFmtArraySize, inputFmtCount);
    } catch (const std::exception& e) {
        handle->roc_encoder_->SetLastError(e.what());
        return ROCENC_ERR_GENERIC;
    }
}

rocEncStatus ROCENCAPI rocEncGetEncodeCaps(void* encoder, ROCENC_CODEC codec,
                                           ROCENC_CAPS_PARAM* capsParam, int* capsVal) {
    FunctionEntryLogWithArgs(g_rocenc_logger, RocEncFmtPtr(encoder) + ", " + RocEncFmtPtr(capsParam));
    EncHandle* handle = AsHandle(encoder);
    if (handle == nullptr) {
        return ROCENC_ERR_INVALID_ENCODERDEVICE;
    }
    try {
        return handle->roc_encoder_->GetEncodeCaps(codec, capsParam, capsVal);
    } catch (const std::exception& e) {
        handle->roc_encoder_->SetLastError(e.what());
        return ROCENC_ERR_GENERIC;
    }
}

rocEncStatus ROCENCAPI rocEncDestroyEncoder(void* encoder) {
    FunctionEntryLogWithArgs(g_rocenc_logger, RocEncFmtPtr(encoder));
    EncHandle* handle = AsHandle(encoder);
    if (handle == nullptr) {
        return ROCENC_ERR_INVALID_ENCODERDEVICE;
    }
    delete handle;
    return ROCENC_SUCCESS;
}

const char* ROCENCAPI rocEncGetLastErrorString(void* encoder) {
    EncHandle* handle = AsHandle(encoder);
    if (handle == nullptr) {
        return kNoHandleError;
    }
    const char* msg = handle->roc_encoder_->GetLastError();
    return msg ? msg : kNoErrorString;
}

rocEncStatus ROCENCAPI rocEncodeAPIGetMaxSupportedVersion(uint32_t* version) {
    FunctionEntryLogWithArgs(g_rocenc_logger, RocEncFmtPtr(version));
    if (version == nullptr) {
        return ROCENC_ERR_INVALID_PTR;
    }
    *version = ROCENCAPI_VERSION;
    return ROCENC_SUCCESS;
}

/*****************************************************************************************************/
//! Declared, not implemented in Phase 1.
//!
//! Every one of these is wired into ROCENCODE_API_FUNCTION_LIST -- no entry is
//! ever NULL -- and returns a status rather than crashing.  Entry points with
//! no VCN equivalent at all return ROCENC_ERR_UNSUPPORTED_DEVICE permanently;
//! the rest return it until the phase that implements them.
/*****************************************************************************************************/

#define ROCENC_UNSUPPORTED_BODY(name)                                                   \
    do {                                                                                \
        EncHandle* handle = AsHandle(encoder);                                          \
        if (handle != nullptr) {                                                        \
            handle->roc_encoder_->SetLastError(                                         \
                ROCENC_STR(name) + " is not implemented in this release of rocEncode"); \
        }                                                                               \
        return ROCENC_ERR_UNSUPPORTED_DEVICE;                                           \
    } while (0)

rocEncStatus ROCENCAPI rocEncGetEncodePresetCount(void* encoder, ROCENC_CODEC codec, uint32_t* presetCount) {
    (void)codec; (void)presetCount;
    ROCENC_UNSUPPORTED_BODY("rocEncGetEncodePresetCount");
}

rocEncStatus ROCENCAPI rocEncGetEncodePresets(void* encoder, ROCENC_CODEC codec, ROCENC_PRESET* presets,
                                              uint32_t presetArraySize, uint32_t* presetCount) {
    (void)codec; (void)presets; (void)presetArraySize; (void)presetCount;
    ROCENC_UNSUPPORTED_BODY("rocEncGetEncodePresets");
}

rocEncStatus ROCENCAPI rocEncGetEncodePresetConfig(void* encoder, ROCENC_CODEC codec, ROCENC_PRESET preset,
                                                   ROCENC_PRESET_CONFIG* presetConfig) {
    (void)codec; (void)preset; (void)presetConfig;
    ROCENC_UNSUPPORTED_BODY("rocEncGetEncodePresetConfig");
}

rocEncStatus ROCENCAPI rocEncGetEncodePresetConfigEx(void* encoder, ROCENC_CODEC codec, ROCENC_PRESET preset,
                                                     ROCENC_TUNING_INFO tuningInfo,
                                                     ROCENC_PRESET_CONFIG* presetConfig) {
    (void)codec; (void)preset; (void)tuningInfo; (void)presetConfig;
    ROCENC_UNSUPPORTED_BODY("rocEncGetEncodePresetConfigEx");
}

rocEncStatus ROCENCAPI rocEncInitializeEncoder(void* encoder, ROCENC_INITIALIZE_PARAMS* createEncodeParams) {
    (void)createEncodeParams;
    ROCENC_UNSUPPORTED_BODY("rocEncInitializeEncoder");
}

rocEncStatus ROCENCAPI rocEncCreateInputBuffer(void* encoder,
                                               ROCENC_CREATE_INPUT_BUFFER* createInputBufferParams) {
    (void)createInputBufferParams;
    ROCENC_UNSUPPORTED_BODY("rocEncCreateInputBuffer");
}

rocEncStatus ROCENCAPI rocEncDestroyInputBuffer(void* encoder, ROCENC_INPUT_PTR inputBuffer) {
    (void)inputBuffer;
    ROCENC_UNSUPPORTED_BODY("rocEncDestroyInputBuffer");
}

rocEncStatus ROCENCAPI rocEncSetIOHipStreams(void* encoder, ROCENC_HIPSTREAM_PTR inputStream,
                                             ROCENC_HIPSTREAM_PTR outputStream) {
    (void)inputStream; (void)outputStream;
    ROCENC_UNSUPPORTED_BODY("rocEncSetIOHipStreams");
}

rocEncStatus ROCENCAPI rocEncCreateBitstreamBuffer(void* encoder,
                                                   ROCENC_CREATE_BITSTREAM_BUFFER* createBitstreamBufferParams) {
    (void)createBitstreamBufferParams;
    ROCENC_UNSUPPORTED_BODY("rocEncCreateBitstreamBuffer");
}

rocEncStatus ROCENCAPI rocEncDestroyBitstreamBuffer(void* encoder, ROCENC_OUTPUT_PTR bitstreamBuffer) {
    (void)bitstreamBuffer;
    ROCENC_UNSUPPORTED_BODY("rocEncDestroyBitstreamBuffer");
}

rocEncStatus ROCENCAPI rocEncEncodePicture(void* encoder, ROCENC_PIC_PARAMS* encodePicParams) {
    (void)encodePicParams;
    // The encoder was never initialized, so this is a sequencing error as much
    // as an unsupported one; ROCENC_ERR_ENCODER_NOT_INITIALIZED would mislead
    // an application into retrying after rocEncInitializeEncoder(), which also
    // fails.  Report the device limitation.
    ROCENC_UNSUPPORTED_BODY("rocEncEncodePicture");
}

rocEncStatus ROCENCAPI rocEncLockBitstream(void* encoder, ROCENC_LOCK_BITSTREAM* lockBitstreamBufferParams) {
    (void)lockBitstreamBufferParams;
    ROCENC_UNSUPPORTED_BODY("rocEncLockBitstream");
}

rocEncStatus ROCENCAPI rocEncUnlockBitstream(void* encoder, ROCENC_OUTPUT_PTR bitstreamBuffer) {
    (void)bitstreamBuffer;
    ROCENC_UNSUPPORTED_BODY("rocEncUnlockBitstream");
}

rocEncStatus ROCENCAPI rocEncLockInputBuffer(void* encoder,
                                             ROCENC_LOCK_INPUT_BUFFER* lockInputBufferParams) {
    (void)lockInputBufferParams;
    ROCENC_UNSUPPORTED_BODY("rocEncLockInputBuffer");
}

rocEncStatus ROCENCAPI rocEncUnlockInputBuffer(void* encoder, ROCENC_INPUT_PTR inputBuffer) {
    (void)inputBuffer;
    ROCENC_UNSUPPORTED_BODY("rocEncUnlockInputBuffer");
}

rocEncStatus ROCENCAPI rocEncGetEncodeStats(void* encoder, ROCENC_STAT* encodeStats) {
    (void)encodeStats;
    ROCENC_UNSUPPORTED_BODY("rocEncGetEncodeStats");
}

rocEncStatus ROCENCAPI rocEncGetSequenceParams(void* encoder,
                                               ROCENC_SEQUENCE_PARAM_PAYLOAD* sequenceParamPayload) {
    (void)sequenceParamPayload;
    ROCENC_UNSUPPORTED_BODY("rocEncGetSequenceParams");
}

rocEncStatus ROCENCAPI rocEncGetSequenceParamEx(void* encoder, ROCENC_INITIALIZE_PARAMS* encInitParams,
                                                ROCENC_SEQUENCE_PARAM_PAYLOAD* sequenceParamPayload) {
    (void)encInitParams; (void)sequenceParamPayload;
    ROCENC_UNSUPPORTED_BODY("rocEncGetSequenceParamEx");
}

rocEncStatus ROCENCAPI rocEncRegisterAsyncEvent(void* encoder, ROCENC_EVENT_PARAMS* eventParams) {
    (void)eventParams;
    // No VCN equivalent: rocEncode is synchronous-only, as NVENC is on Linux.
    ROCENC_UNSUPPORTED_BODY("rocEncRegisterAsyncEvent");
}

rocEncStatus ROCENCAPI rocEncUnregisterAsyncEvent(void* encoder, ROCENC_EVENT_PARAMS* eventParams) {
    (void)eventParams;
    ROCENC_UNSUPPORTED_BODY("rocEncUnregisterAsyncEvent");
}

rocEncStatus ROCENCAPI rocEncMapInputResource(void* encoder, ROCENC_MAP_INPUT_RESOURCE* mapInputResParams) {
    (void)mapInputResParams;
    ROCENC_UNSUPPORTED_BODY("rocEncMapInputResource");
}

rocEncStatus ROCENCAPI rocEncUnmapInputResource(void* encoder, ROCENC_INPUT_PTR mappedInputBuffer) {
    (void)mappedInputBuffer;
    ROCENC_UNSUPPORTED_BODY("rocEncUnmapInputResource");
}

rocEncStatus ROCENCAPI rocEncInvalidateRefFrames(void* encoder, uint64_t invalidRefFrameTimeStamp) {
    (void)invalidRefFrameTimeStamp;
    ROCENC_UNSUPPORTED_BODY("rocEncInvalidateRefFrames");
}

rocEncStatus ROCENCAPI rocEncRegisterResource(void* encoder, ROCENC_REGISTER_RESOURCE* registerResParams) {
    (void)registerResParams;
    ROCENC_UNSUPPORTED_BODY("rocEncRegisterResource");
}

rocEncStatus ROCENCAPI rocEncUnregisterResource(void* encoder, ROCENC_REGISTERED_PTR registeredResource) {
    (void)registeredResource;
    ROCENC_UNSUPPORTED_BODY("rocEncUnregisterResource");
}

rocEncStatus ROCENCAPI rocEncReconfigureEncoder(void* encoder, ROCENC_RECONFIGURE_PARAMS* reInitEncodeParams) {
    (void)reInitEncodeParams;
    ROCENC_UNSUPPORTED_BODY("rocEncReconfigureEncoder");
}

rocEncStatus ROCENCAPI rocEncCreateMVBuffer(void* encoder, ROCENC_CREATE_MV_BUFFER* createMVBufferParams) {
    (void)createMVBufferParams;
    // No VCN equivalent: there is no motion-estimation-only mode.
    ROCENC_UNSUPPORTED_BODY("rocEncCreateMVBuffer");
}

rocEncStatus ROCENCAPI rocEncDestroyMVBuffer(void* encoder, ROCENC_OUTPUT_PTR mvBuffer) {
    (void)mvBuffer;
    ROCENC_UNSUPPORTED_BODY("rocEncDestroyMVBuffer");
}

rocEncStatus ROCENCAPI rocEncRunMotionEstimationOnly(void* encoder, ROCENC_MEONLY_PARAMS* meOnlyParams) {
    (void)meOnlyParams;
    ROCENC_UNSUPPORTED_BODY("rocEncRunMotionEstimationOnly");
}

rocEncStatus ROCENCAPI rocEncRestoreEncoderState(void* encoder,
                                                 ROCENC_RESTORE_ENCODER_STATE_PARAMS* restoreState) {
    (void)restoreState;
    ROCENC_UNSUPPORTED_BODY("rocEncRestoreEncoderState");
}

rocEncStatus ROCENCAPI rocEncLookaheadPicture(void* encoder, ROCENC_LOOKAHEAD_PIC_PARAMS* lookaheadParams) {
    (void)lookaheadParams;
    ROCENC_UNSUPPORTED_BODY("rocEncLookaheadPicture");
}

#undef ROCENC_UNSUPPORTED_BODY

/*****************************************************************************************************/
//! The entry point.
/*****************************************************************************************************/

rocEncStatus ROCENCAPI rocEncodeAPICreateInstance(ROCENCODE_API_FUNCTION_LIST* functionList) {
    FunctionEntryLogWithArgs(g_rocenc_logger, RocEncFmtPtr(functionList));
    if (functionList == nullptr) {
        return ROCENC_ERR_INVALID_PTR;
    }
    if (functionList->version != ROCENCODE_API_FUNCTION_LIST_VER) {
        CriticalLog(g_rocenc_logger, "ROCENCODE_API_FUNCTION_LIST::version mismatch");
        return ROCENC_ERR_INVALID_VERSION;
    }

    uint32_t requested_version = functionList->version;
    std::memset(functionList, 0, sizeof(ROCENCODE_API_FUNCTION_LIST));
    functionList->version = requested_version;

    functionList->rocEncOpenEncodeSession          = rocEncOpenEncodeSession;
    functionList->rocEncGetEncodeCodecCount        = rocEncGetEncodeCodecCount;
    functionList->rocEncGetEncodeProfileCount      = rocEncGetEncodeProfileCount;
    functionList->rocEncGetEncodeProfiles          = rocEncGetEncodeProfiles;
    functionList->rocEncGetEncodeCodecs            = rocEncGetEncodeCodecs;
    functionList->rocEncGetInputFormatCount        = rocEncGetInputFormatCount;
    functionList->rocEncGetInputFormats            = rocEncGetInputFormats;
    functionList->rocEncGetEncodeCaps              = rocEncGetEncodeCaps;
    functionList->rocEncGetEncodePresetCount       = rocEncGetEncodePresetCount;
    functionList->rocEncGetEncodePresets           = rocEncGetEncodePresets;
    functionList->rocEncGetEncodePresetConfig      = rocEncGetEncodePresetConfig;
    functionList->rocEncInitializeEncoder          = rocEncInitializeEncoder;
    functionList->rocEncCreateInputBuffer          = rocEncCreateInputBuffer;
    functionList->rocEncDestroyInputBuffer         = rocEncDestroyInputBuffer;
    functionList->rocEncCreateBitstreamBuffer      = rocEncCreateBitstreamBuffer;
    functionList->rocEncDestroyBitstreamBuffer     = rocEncDestroyBitstreamBuffer;
    functionList->rocEncEncodePicture              = rocEncEncodePicture;
    functionList->rocEncLockBitstream              = rocEncLockBitstream;
    functionList->rocEncUnlockBitstream            = rocEncUnlockBitstream;
    functionList->rocEncLockInputBuffer            = rocEncLockInputBuffer;
    functionList->rocEncUnlockInputBuffer          = rocEncUnlockInputBuffer;
    functionList->rocEncGetEncodeStats             = rocEncGetEncodeStats;
    functionList->rocEncGetSequenceParams          = rocEncGetSequenceParams;
    functionList->rocEncRegisterAsyncEvent         = rocEncRegisterAsyncEvent;
    functionList->rocEncUnregisterAsyncEvent       = rocEncUnregisterAsyncEvent;
    functionList->rocEncMapInputResource           = rocEncMapInputResource;
    functionList->rocEncUnmapInputResource         = rocEncUnmapInputResource;
    functionList->rocEncDestroyEncoder             = rocEncDestroyEncoder;
    functionList->rocEncInvalidateRefFrames        = rocEncInvalidateRefFrames;
    functionList->rocEncOpenEncodeSessionEx        = rocEncOpenEncodeSessionEx;
    functionList->rocEncRegisterResource           = rocEncRegisterResource;
    functionList->rocEncUnregisterResource         = rocEncUnregisterResource;
    functionList->rocEncReconfigureEncoder         = rocEncReconfigureEncoder;
    functionList->rocEncCreateMVBuffer             = rocEncCreateMVBuffer;
    functionList->rocEncDestroyMVBuffer            = rocEncDestroyMVBuffer;
    functionList->rocEncRunMotionEstimationOnly    = rocEncRunMotionEstimationOnly;
    functionList->rocEncGetLastErrorString         = rocEncGetLastErrorString;
    functionList->rocEncSetIOHipStreams            = rocEncSetIOHipStreams;
    functionList->rocEncGetEncodePresetConfigEx    = rocEncGetEncodePresetConfigEx;
    functionList->rocEncGetSequenceParamEx         = rocEncGetSequenceParamEx;
    functionList->rocEncRestoreEncoderState        = rocEncRestoreEncoderState;
    functionList->rocEncLookaheadPicture           = rocEncLookaheadPicture;

    return ROCENC_SUCCESS;
}

}  // extern "C"
