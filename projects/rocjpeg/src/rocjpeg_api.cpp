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

#include "rocjpeg_api_stream_handle.h"
#include "rocjpeg_api_decoder_handle.h"
#include "rocjpeg_commons.h"

namespace rocjpeg {
/**
 * @brief Creates a RocJpegStreamHandle for JPEG stream processing.
 *
 * This function creates a RocJpegStreamHandle, which is used for processing JPEG streams.
 * The created handle is assigned to the provided jpeg_stream_handle pointer.
 *
 * @param jpeg_stream_handle A pointer to a RocJpegStreamHandle variable that will hold the created handle.
 * @return RocJpegStatus The status of the operation. Returns ROCJPEG_STATUS_SUCCESS if successful,
 *         ROCJPEG_STATUS_INVALID_PARAMETER if the jpeg_stream_handle pointer is nullptr,
 *         or ROCJPEG_STATUS_NOT_INITIALIZED if the rocJPEG stream handle failed to initialize.
 */
RocJpegStatus ROCJPEGAPI rocJpegStreamCreate(RocJpegStreamHandle *jpeg_stream_handle) {
    if (jpeg_stream_handle == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    RocJpegStreamHandle rocjpeg_stream_handle = nullptr;
    try {
        rocjpeg_stream_handle = new RocJpegStreamParserHandle();
    }
    catch(const std::exception& e) {
        ERR(STR("Failed to init the rocJPEG stream handle, ") + STR(e.what()));
        return ROCJPEG_STATUS_NOT_INITIALIZED;
    }
    *jpeg_stream_handle = rocjpeg_stream_handle;
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Parses a JPEG stream.
 *
 * This function parses a JPEG stream represented by the input data and length,
 * and updates the provided JPEG stream handle accordingly.
 *
 * @param data The pointer to the JPEG stream data.
 * @param length The length of the JPEG stream data.
 * @param jpeg_stream_handle The handle to the JPEG stream.
 * @return The status of the JPEG stream parsing operation.
 *         - ROCJPEG_STATUS_SUCCESS if the parsing is successful.
 *         - ROCJPEG_STATUS_INVALID_PARAMETER if the input parameters are invalid.
 *         - ROCJPEG_STATUS_BAD_JPEG if the JPEG stream is invalid.
 */
RocJpegStatus ROCJPEGAPI rocJpegStreamParse(const unsigned char *data, size_t length, RocJpegStreamHandle jpeg_stream_handle) {
    if (data == nullptr || jpeg_stream_handle == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    auto rocjpeg_stream_handle = static_cast<RocJpegStreamParserHandle*>(jpeg_stream_handle);
    if (!rocjpeg_stream_handle->rocjpeg_stream->ParseJpegStream(data, length)) {
        return ROCJPEG_STATUS_BAD_JPEG;
    }
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Destroys a RocJpegStreamHandle object and releases associated resources.
 *
 * @param jpeg_stream_handle The handle to the RocJpegStreamHandle object to be destroyed.
 * @return RocJpegStatus The status of the operation. Returns ROCJPEG_STATUS_SUCCESS if successful,
 *         or ROCJPEG_STATUS_INVALID_PARAMETER if the input handle is nullptr.
 */
RocJpegStatus ROCJPEGAPI rocJpegStreamDestroy(RocJpegStreamHandle jpeg_stream_handle) {
    if (jpeg_stream_handle == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    auto rocjpeg_stream_handle = static_cast<RocJpegStreamParserHandle*>(jpeg_stream_handle);
    delete rocjpeg_stream_handle;
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Creates a RocJpegHandle for JPEG decoding.
 *
 * This function creates a RocJpegHandle for JPEG decoding using the specified backend and device ID.
 *
 * @param backend The backend to be used for JPEG decoding.
 * @param device_id The ID of the device to be used for JPEG decoding.
 * @param handle Pointer to a RocJpegHandle variable to store the created handle.
 * @return The status of the operation. Returns ROCJPEG_STATUS_INVALID_PARAMETER if handle is nullptr,
 *         ROCJPEG_STATUS_NOT_INITIALIZED if the rocJPEG handle initialization fails, or the status
 *         returned by the InitializeDecoder function of the rocjpeg_decoder.
 */
RocJpegStatus ROCJPEGAPI rocJpegCreate(RocJpegBackend backend, int device_id, RocJpegHandle *handle) {
    if (handle == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    RocJpegHandle rocjpeg_handle = nullptr;
    try {
        rocjpeg_handle = new RocJpegDecoderHandle(backend, device_id);
    } catch(const std::exception& e) {
        ERR(STR("Failed to init the rocJPEG handle, ") + STR(e.what()));
        return ROCJPEG_STATUS_NOT_INITIALIZED;
    }
    *handle = rocjpeg_handle;
    return static_cast<RocJpegDecoderHandle *>(rocjpeg_handle)->rocjpeg_decoder->InitializeDecoder();
}

/**
 * @brief Destroys a RocJpegHandle object.
 *
 * This function destroys the RocJpegHandle object pointed to by the given handle.
 * It releases any resources associated with the handle and frees the memory.
 *
 * @param handle The handle to the RocJpegHandle object to be destroyed.
 * @return The status of the operation. Returns ROCJPEG_STATUS_SUCCESS if the handle was successfully destroyed,
 *         or ROCJPEG_STATUS_INVALID_PARAMETER if the handle is nullptr.
 */
RocJpegStatus ROCJPEGAPI rocJpegDestroy(RocJpegHandle handle) {
    if (handle == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    auto rocjpeg_handle = static_cast<RocJpegDecoderHandle*>(handle);
    delete rocjpeg_handle;
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Retrieves information about the JPEG image.
 *
 * This function retrieves the number of components, chroma subsampling, and dimensions (width and height) of the JPEG image
 * specified by the `jpeg_stream_handle`. The information is stored in the provided output parameters `num_components`,
 * `subsampling`, `widths`, and `heights`.
 *
 * @param handle The handle to the RocJpegDecoder instance.
 * @param jpeg_stream_handle The handle to the RocJpegStream instance representing the JPEG image.
 * @param num_components A pointer to an unsigned 8-bit integer that will store the number of components in the JPEG image.
 * @param subsampling A pointer to a RocJpegChromaSubsampling enum that will store the chroma subsampling information.
 * @param widths A pointer to an unsigned 32-bit integer array that will store the width of each component in the JPEG image.
 * @param heights A pointer to an unsigned 32-bit integer array that will store the height of each component in the JPEG image.
 *
 * @return The RocJpegStatus indicating the success or failure of the operation.
 *         - ROCJPEG_STATUS_SUCCESS: The operation was successful.
 *         - ROCJPEG_STATUS_INVALID_PARAMETER: One or more input parameters are invalid.
 *         - ROCJPEG_STATUS_RUNTIME_ERROR: An exception occurred during the operation.
 */
RocJpegStatus ROCJPEGAPI rocJpegGetImageInfo(RocJpegHandle handle, RocJpegStreamHandle jpeg_stream_handle, uint8_t *num_components,
    RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights) {
    if (handle == nullptr || num_components == nullptr ||
        subsampling == nullptr || widths == nullptr || heights == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    RocJpegStatus rocjpeg_status = ROCJPEG_STATUS_SUCCESS;
    auto rocjpeg_handle = static_cast<RocJpegDecoderHandle*>(handle);
    try {
        rocjpeg_status = rocjpeg_handle->rocjpeg_decoder->GetImageInfo(jpeg_stream_handle, num_components, subsampling, widths, heights);
    } catch (const std::exception& e) {
        rocjpeg_handle->CaptureError(e.what());
        ERR(e.what());
        return ROCJPEG_STATUS_RUNTIME_ERROR;
    }

    return rocjpeg_status;
}

/**
 * @brief Decodes a JPEG image using the rocJPEG library.
 *
 * This function decodes a JPEG image using the rocJPEG library. It takes a rocJpegHandle, a rocJpegStreamHandle,
 * a pointer to RocJpegDecodeParams, and a pointer to RocJpegImage as input parameters. The function returns a
 * RocJpegStatus indicating the success or failure of the decoding operation.
 *
 * @param handle The rocJpegHandle representing the rocJPEG decoder instance.
 * @param jpeg_stream_handle The rocJpegStreamHandle representing the input JPEG stream.
 * @param decode_params A pointer to RocJpegDecodeParams containing the decoding parameters.
 * @param destination A pointer to RocJpegImage where the decoded image will be stored.
 * @return A RocJpegStatus indicating the success or failure of the decoding operation.
 */
RocJpegStatus ROCJPEGAPI rocJpegDecode(RocJpegHandle handle, RocJpegStreamHandle jpeg_stream_handle, const RocJpegDecodeParams *decode_params,
    RocJpegImage *destination) {

    if (handle == nullptr || decode_params == nullptr || destination == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    RocJpegStatus rocjpeg_status = ROCJPEG_STATUS_SUCCESS;
    auto rocjpeg_handle = static_cast<RocJpegDecoderHandle*>(handle);
    try {
        rocjpeg_status = rocjpeg_handle->rocjpeg_decoder->Decode(jpeg_stream_handle, decode_params, destination);
    } catch (const std::exception& e) {
        rocjpeg_handle->CaptureError(e.what());
        ERR(e.what());
        return ROCJPEG_STATUS_RUNTIME_ERROR;
    }

    return rocjpeg_status;
}

/**
 * @brief Decodes a batch of JPEG images using the rocJPEG library.
 *
 * Decodes a batch of JPEG images using the specified handle, stream handles, decode parameters, and destination images.
 *
 * @param handle The handle to the RocJpeg decoder.
 * @param jpeg_stream_handles An array of stream handles for the JPEG images to be decoded.
 * @param decode_params The decode parameters for the decoding process.
 * @param destinations An array of RocJpegImage structures to store the decoded images.
 * @return The status of the decoding process. Returns ROCJPEG_STATUS_SUCCESS if successful, or an error code otherwise.
 */
RocJpegStatus ROCJPEGAPI rocJpegDecodeBatched(RocJpegHandle handle, RocJpegStreamHandle *jpeg_stream_handles, int batch_size, const RocJpegDecodeParams *decode_params, RocJpegImage *destinations) {
    if (handle == nullptr || jpeg_stream_handles == nullptr|| decode_params == nullptr || destinations == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    RocJpegStatus rocjpeg_status = ROCJPEG_STATUS_SUCCESS;
    auto rocjpeg_handle = static_cast<RocJpegDecoderHandle*>(handle);
    try {
        rocjpeg_status = rocjpeg_handle->rocjpeg_decoder->DecodeBatched(jpeg_stream_handles, batch_size, decode_params, destinations);
    } catch (const std::exception& e) {
        rocjpeg_handle->CaptureError(e.what());
        ERR(e.what());
        return ROCJPEG_STATUS_RUNTIME_ERROR;
    }

    return rocjpeg_status;
}
/**
 * @brief Returns the error name corresponding to the given RocJpegStatus.
 *
 * This function takes a RocJpegStatus enum value and returns the corresponding error name as a string.
 *
 * @param rocjpeg_status The RocJpegStatus enum value.
 * @return The error name as a string.
 */
extern const char* ROCJPEGAPI rocJpegGetErrorName(RocJpegStatus rocjpeg_status) {
    switch (rocjpeg_status) {
        case ROCJPEG_STATUS_SUCCESS:
            return "ROCJPEG_STATUS_SUCCESS";
        case ROCJPEG_STATUS_NOT_INITIALIZED:
            return "ROCJPEG_STATUS_NOT_INITIALIZED";
        case ROCJPEG_STATUS_INVALID_PARAMETER:
            return "ROCJPEG_STATUS_INVALID_PARAMETER";
        case ROCJPEG_STATUS_BAD_JPEG:
            return "ROCJPEG_STATUS_BAD_JPEG";
        case ROCJPEG_STATUS_JPEG_NOT_SUPPORTED:
            return "ROCJPEG_STATUS_JPEG_NOT_SUPPORTED";
        case ROCJPEG_STATUS_EXECUTION_FAILED:
            return "ROCJPEG_STATUS_EXECUTION_FAILED";
        case ROCJPEG_STATUS_ARCH_MISMATCH:
            return "ROCJPEG_STATUS_ARCH_MISMATCH";
        case ROCJPEG_STATUS_INTERNAL_ERROR:
            return "ROCJPEG_STATUS_INTERNAL_ERROR";
        case ROCJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
            return "ROCJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED";
        case ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED:
            return "ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED";
        case ROCJPEG_STATUS_RUNTIME_ERROR:
            return "ROCJPEG_STATUS_RUNTIME_ERROR";
        case ROCJPEG_STATUS_OUTOF_MEMORY:
            return "ROCJPEG_STATUS_OUTOF_MEMORY";
        case ROCJPEG_STATUS_NOT_IMPLEMENTED:
            return "ROCJPEG_STATUS_NOT_IMPLEMENTED";
        default:
            return "UNKNOWN_ERROR";
    }
}
} //namespace rocjpeg