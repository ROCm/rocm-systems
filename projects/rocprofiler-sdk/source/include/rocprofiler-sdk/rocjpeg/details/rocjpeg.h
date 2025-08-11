/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef ROC_JPEG_H
#define ROC_JPEG_H

#define ROCJPEGAPI

#pragma once
#include <rocprofiler-sdk/rocjpeg/details/rocjpeg_headers.h>
#include "hip/hip_runtime.h"

#if ROCPROFILER_SDK_USE_SYSTEM_ROCJPEG > 0
#    include <rocjpeg/rocjpeg_version.h>
#else
#    include <rocprofiler-sdk/rocjpeg/details/rocjpeg_version.h>
#endif

/**
 * @file rocjpeg.h
 * @brief The AMD rocJPEG Library.
 * @defgroup group_amd_rocjepg rocJPEG: AMD ROCm JPEG Decode API
 * @brief rocJPEG API is a toolkit to decode JPEG images using a hardware-accelerated JPEG decoder
 * on AMD’s GPUs.
 */

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus
/**
 * @def
 * @ingroup group_amd_rocjpeg
 * Maximum number of channels rocJPEG supports
 */
#define ROCJPEG_MAX_COMPONENT 4

/**
 * @enum RocJpegStatus
 * @ingroup group_amd_rocjpeg
 * @brief Enumeration representing the status codes for the rocJPEG library.
 */
typedef enum
{
    ROCJPEG_STATUS_SUCCESS            = 0,  /**< The operation completed successfully. */
    ROCJPEG_STATUS_NOT_INITIALIZED    = -1, /**< The rocJPEG library is not initialized. */
    ROCJPEG_STATUS_INVALID_PARAMETER  = -2, /**< An invalid parameter was passed to a function. */
    ROCJPEG_STATUS_BAD_JPEG           = -3, /**< The input JPEG data is corrupted or invalid. */
    ROCJPEG_STATUS_JPEG_NOT_SUPPORTED = -4, /**< The JPEG format is not supported. */
    ROCJPEG_STATUS_OUTOF_MEMORY       = -5, /**< Out of memory error. */
    ROCJPEG_STATUS_EXECUTION_FAILED   = -6, /**< The execution of a function failed. */
    ROCJPEG_STATUS_ARCH_MISMATCH      = -7, /**< The architecture is not supported. */
    ROCJPEG_STATUS_INTERNAL_ERROR     = -8, /**< Internal error occurred. */
    ROCJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED =
        -9, /**< The requested implementation is not supported. */
    ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED =
        -10,                              /**< Hardware JPEG decoder is not supported. */
    ROCJPEG_STATUS_RUNTIME_ERROR   = -11, /**< Runtime error occurred. */
    ROCJPEG_STATUS_NOT_IMPLEMENTED = -12, /**< The requested feature is not implemented. */
} RocJpegStatus;

/**
 * @enum RocJpegChromaSubsampling
 * @ingroup group_amd_rocjpeg
 * @brief Enum representing the chroma subsampling options for JPEG encoding/decoding.
 *
 * The `RocJpegChromaSubsampling` enum defines the available chroma subsampling options for JPEG
 * encoding/decoding. Chroma subsampling refers to the reduction of color information in an image to
 * reduce file size.
 *
 * The possible values are:
 * - `ROCJPEG_CSS_444`: Full chroma resolution (4:4:4).
 * - `ROCJPEG_CSS_440`: Chroma resolution reduced by half vertically (4:4:0).
 * - `ROCJPEG_CSS_422`: Chroma resolution reduced by half horizontally (4:2:2).
 * - `ROCJPEG_CSS_420`: Chroma resolution reduced by half both horizontally and vertically (4:2:0).
 * - `ROCJPEG_CSS_411`: Chroma resolution reduced by a quarter horizontally (4:1:1).
 * - `ROCJPEG_CSS_400`: No chroma information (4:0:0).
 * - `ROCJPEG_CSS_UNKNOWN`: Unknown chroma subsampling.
 */
typedef enum
{
    ROCJPEG_CSS_444     = 0,
    ROCJPEG_CSS_440     = 1,
    ROCJPEG_CSS_422     = 2,
    ROCJPEG_CSS_420     = 3,
    ROCJPEG_CSS_411     = 4,
    ROCJPEG_CSS_400     = 5,
    ROCJPEG_CSS_UNKNOWN = -1
} RocJpegChromaSubsampling;

/**
 * @struct RocJpegImage
 * @ingroup group_amd_rocjpeg
 * @brief Structure representing a JPEG image.
 *
 * This structure holds the information about a JPEG image, including the pointers to the image
 * channels and the pitch (stride) of each channel.
 */
typedef struct
{
    uint8_t* channel[ROCJPEG_MAX_COMPONENT]; /**< Pointers to the image channels. */
    uint32_t pitch[ROCJPEG_MAX_COMPONENT];   /**< Pitch (stride) of each channel. */
} RocJpegImage;

/**
 * @enum RocJpegOutputFormat
 * @ingroup group_amd_rocjpeg
 * @brief Enum representing the output format options for the RocJpegImage.
 *
 * The `RocJpegOutputFormat` enum specifies the different output formats that can be used when
 * decoding a JPEG image using the VCN JPEG decoder.
 *
 * The available output formats are:
 * - `ROCJPEG_OUTPUT_NATIVE`: Returns the native unchanged decoded YUV image from the VCN JPEG
 * decoder. The channel arrangement depends on the chroma subsampling format.
 * - `ROCJPEG_OUTPUT_YUV_PLANAR`: Extracts the Y, U, and V channels from the decoded YUV image and
 * writes them into separate channels of the RocJpegImage.
 * - `ROCJPEG_OUTPUT_Y`: Returns only the luma component (Y) and writes it to the first channel of
 * the RocJpegImage.
 * - `ROCJPEG_OUTPUT_RGB`: Converts the decoded image to interleaved RGB format using the VCN JPEG
 * decoder or HIP kernels and writes it to the first channel of the RocJpegImage.
 * - `ROCJPEG_OUTPUT_RGB_PLANAR`: Converts the decoded image to RGB PLANAR format using the VCN JPEG
 * decoder or HIP kernels and writes the RGB channels to separate channels of the RocJpegImage.
 * - `ROCJPEG_OUTPUT_FORMAT_MAX`: Maximum allowed value for the output format.
 */
typedef enum
{
    /**< return native unchanged decoded YUV image from the VCN JPEG decoder.
         For ROCJPEG_CSS_444 and ROCJPEG_CSS_440 write Y, U, and V to first, second, and third
       channels of RocJpegImage For ROCJPEG_CSS_422 write YUYV (packed) to first channel of
       RocJpegImage For ROCJPEG_CSS_420 write Y to first channel and UV (interleaved) to second
       channel of RocJpegImage For ROCJPEG_CSS_400 write Y to first channel of RocJpegImage */
    ROCJPEG_OUTPUT_NATIVE = 0,
    /**< extract Y, U, and V channels from the decoded YUV image from the VCN JPEG decoder and write
       into first, second, and third channel of RocJpegImage. For ROCJPEG_CSS_400 write Y to first
       channel of RocJpegImage */
    ROCJPEG_OUTPUT_YUV_PLANAR = 1,
    /**< return luma component (Y) and write to first channel of RocJpegImage */
    ROCJPEG_OUTPUT_Y = 2,
    /**< convert to interleaved RGB using VCN JPEG decoder (on MI300+) or using HIP kernels and
       write to first channel of RocJpegImage */
    ROCJPEG_OUTPUT_RGB = 3,
    /**< convert to RGB PLANAR using VCN JPEG decoder (on MI300+) or HIP kernels and write to first,
       second, and third channel of RocJpegImage. */
    ROCJPEG_OUTPUT_RGB_PLANAR = 4,
    ROCJPEG_OUTPUT_FORMAT_MAX = 5 /**< maximum allowed value */
} RocJpegOutputFormat;

/**
 * @struct RocJpegDecodeParams
 * @ingroup group_amd_rocjpeg
 * @brief Structure containing parameters for JPEG decoding.
 *
 * This structure defines the parameters for decoding a JPEG image using the rocJpeg library.
 * It specifies the output format, crop rectangle, and target dimensions for the decoded image.
 * Note that if both the crop rectangle and target dimensions are defined, cropping is done first,
 * followed by resizing the resulting ROI to the target dimension.
 */
typedef struct
{
    RocJpegOutputFormat
        output_format; /**< Output data format. See RocJpegOutputFormat for description. */
    struct
    {
        int16_t left;   /**< Left coordinate of the crop rectangle. */
        int16_t top;    /**< Top coordinate of the crop rectangle. */
        int16_t right;  /**< Right coordinate of the crop rectangle. */
        int16_t bottom; /**< Bottom coordinate of the crop rectangle. */
    } crop_rectangle;   /**< Defines the region of interest (ROI) to be copied into the RocJpegImage
                           output buffers. */
    struct
    {
        uint32_t width;  /**< Target width of the picture to be resized. */
        uint32_t height; /**< Target height of the picture to be resized. */
    } target_dimension;  /**< (future use) Defines the target width and height of the picture to be
                            resized. Both should be even.  If specified, allocate the RocJpegImage
                            buffers based on these dimensions. */
} RocJpegDecodeParams;

/**
 * @enum RocJpegBackend
 * @ingroup group_amd_rocjpeg
 * @brief The backend options for the rocJpeg library.
 *
 * This enum defines the available backend options for the rocJpeg library.
 * The backend can be either hardware or hybrid.
 */
typedef enum
{
    ROCJPEG_BACKEND_HARDWARE = 0, /**< Hardware backend option. */
    ROCJPEG_BACKEND_HYBRID   = 1  /**< Hybrid backend option. */
} RocJpegBackend;

/**
 * @brief A handle representing a RocJpegStream instance.
 *
 * The `RocJpegStreamHandle` is a pointer type used to represent a RocJpegStream instance.
 * It is used as a handle to parse and store various parameters from a JPEG stream.
 */
typedef void* RocJpegStreamHandle;

/**
 * @fn RocJpegStatus ROCJPEGAPI rocJpegStreamCreate(RocJpegStreamHandle *jpeg_stream_handle);
 * @ingroup group_amd_rocjpeg
 * @brief Creates a RocJpegStreamHandle for JPEG stream processing.
 *
 * This function creates a RocJpegStreamHandle, which is used for processing JPEG streams.
 * The created handle is stored in the `jpeg_stream_handle` parameter.
 *
 * @param jpeg_stream_handle Pointer to a RocJpegStreamHandle variable that will hold the created
 * handle.
 * @return RocJpegStatus Returns the status of the operation. Possible values are:
 *                      - ROCJPEG_STATUS_SUCCESS: The operation was successful.
 *                      - ROCJPEG_STATUS_INVALID_ARGUMENT: The `jpeg_stream_handle` parameter is
 * NULL.
 *                      - ROCJPEG_STATUS_OUT_OF_MEMORY: Failed to allocate memory for the handle.
 *                      - ROCJPEG_STATUS_UNKNOWN_ERROR: An unknown error occurred.
 */
RocJpegStatus ROCJPEGAPI
rocJpegStreamCreate(RocJpegStreamHandle* jpeg_stream_handle);

/**
 * @fn RocJpegStatus ROCJPEGAPI rocJpegStreamParse(const unsigned char *data, size_t length,
 * RocJpegStreamHandle jpeg_stream_handle);
 * @ingroup group_amd_rocjpeg
 * @brief Parses a JPEG stream.
 *
 * This function parses a JPEG stream represented by the `data` parameter of length `length`.
 * The parsed stream is associated with the `jpeg_stream_handle` provided.
 *
 * @param data The pointer to the JPEG stream data.
 * @param length The length of the JPEG stream data.
 * @param jpeg_stream_handle The handle to the JPEG stream.
 * @return The status of the JPEG stream parsing operation.
 */
RocJpegStatus ROCJPEGAPI
rocJpegStreamParse(const unsigned char* data,
                   size_t               length,
                   RocJpegStreamHandle  jpeg_stream_handle);

/**
 * @fn RocJpegStatus ROCJPEGAPI rocJpegStreamDestroy(RocJpegStreamHandle jpeg_stream_handle);
 * @ingroup group_amd_rocjpeg
 * @brief Destroys a RocJpegStreamHandle object and releases associated resources.
 *
 * This function destroys the RocJpegStreamHandle object specified by `jpeg_stream_handle` and
 * releases any resources associated with it. After calling this function, the `jpeg_stream_handle`
 * becomes invalid and should not be used anymore.
 *
 * @param jpeg_stream_handle The handle to the RocJpegStreamHandle object to be destroyed.
 * @return The status of the operation. Returns ROCJPEG_STATUS_SUCCESS if the operation is
 * successful, or an error code if an error occurs.
 */
RocJpegStatus ROCJPEGAPI
rocJpegStreamDestroy(RocJpegStreamHandle jpeg_stream_handle);

/**
 * @brief A handle representing a RocJpeg instance.
 *
 * The `RocJpegHandle` is a pointer type used to represent a RocJpeg instance.
 * It is used as a handle to perform various operations on the rocJpeg library.
 */
typedef void* RocJpegHandle;

/**
 * @fn RocJpegStatus ROCJPEGAPI rocJpegCreate(RocJpegBackend backend, int device_id, RocJpegHandle
 * *handle);
 * @ingroup group_amd_rocjpeg
 * @brief Creates a RocJpegHandle for JPEG decoding.
 *
 * This function creates a RocJpegHandle for JPEG decoding using the specified backend and device
 * ID.
 *
 * @param backend The backend to be used for JPEG decoding.
 * @param device_id The ID of the device to be used for JPEG decoding.
 * @param handle Pointer to a RocJpegHandle variable to store the created handle.
 * @return The status of the operation. Returns ROCJPEG_STATUS_INVALID_PARAMETER if handle is
 * nullptr, ROCJPEG_STATUS_NOT_INITIALIZED if the rocJPEG handle initialization fails, or the status
 *         returned by the InitializeDecoder function of the rocjpeg_decoder.
 */
RocJpegStatus ROCJPEGAPI
rocJpegCreate(RocJpegBackend backend, int device_id, RocJpegHandle* handle);

/**
 * @fn RocJpegStatus ROCJPEGAPI rocJpegDestroy(RocJpegHandle handle);
 * @ingroup group_amd_rocjpeg
 * @brief Destroys a RocJpegHandle object.
 *
 * This function destroys the RocJpegHandle object pointed to by the given handle.
 * It releases any resources associated with the handle and frees the memory.
 *
 * @param handle The handle to the RocJpegHandle object to be destroyed.
 * @return The status of the operation. Returns ROCJPEG_STATUS_SUCCESS if the handle was
 * successfully destroyed, or ROCJPEG_STATUS_INVALID_PARAMETER if the handle is nullptr.
 */

RocJpegStatus ROCJPEGAPI
rocJpegDestroy(RocJpegHandle handle);

/**
 * @fn RocJpegStatus ROCJPEGAPI rocJpegGetImageInfo(RocJpegHandle handle, RocJpegStreamHandle
 * jpeg_stream_handle, uint8_t *num_components, RocJpegChromaSubsampling *subsampling, uint32_t
 * *widths, uint32_t *heights);
 * @ingroup group_amd_rocjpeg
 * @brief Retrieves information about the JPEG image.
 *
 * This function retrieves the number of components, chroma subsampling, and dimensions (width and
 * height) of the JPEG image specified by the `jpeg_stream_handle`. The information is stored in the
 * provided output parameters `num_components`, `subsampling`, `widths`, and `heights`.
 *
 * @param handle The handle to the RocJpegDecoder instance.
 * @param jpeg_stream_handle The handle to the RocJpegStream instance representing the JPEG image.
 * @param num_components A pointer to an unsigned 8-bit integer that will store the number of
 * components in the JPEG image.
 * @param subsampling A pointer to a RocJpegChromaSubsampling enum that will store the chroma
 * subsampling information.
 * @param widths A pointer to an unsigned 32-bit integer array that will store the width of each
 * component in the JPEG image.
 * @param heights A pointer to an unsigned 32-bit integer array that will store the height of each
 * component in the JPEG image.
 *
 * @return The RocJpegStatus indicating the success or failure of the operation.
 *         - ROCJPEG_STATUS_SUCCESS: The operation was successful.
 *         - ROCJPEG_STATUS_INVALID_PARAMETER: One or more input parameters are invalid.
 *         - ROCJPEG_STATUS_RUNTIME_ERROR: An exception occurred during the operation.
 */
RocJpegStatus ROCJPEGAPI
rocJpegGetImageInfo(RocJpegHandle             handle,
                    RocJpegStreamHandle       jpeg_stream_handle,
                    uint8_t*                  num_components,
                    RocJpegChromaSubsampling* subsampling,
                    uint32_t*                 widths,
                    uint32_t*                 heights);

/**
 * @fn RocJpegStatus ROCJPEGAPI rocJpegDecode(RocJpegHandle handle, RocJpegStreamHandle
 * jpeg_stream_handle, const RocJpegDecodeParams *decode_params, RocJpegImage *destination);
 * @ingroup group_amd_rocjpeg
 * @brief Decodes a JPEG image using the rocJPEG library.
 *
 * This function decodes a JPEG image using the rocJPEG library. It takes a rocJpegHandle, a
 * rocJpegStreamHandle, a pointer to RocJpegDecodeParams, and a pointer to RocJpegImage as input
 * parameters. The function returns a RocJpegStatus indicating the success or failure of the
 * decoding operation.
 *
 * @param handle The rocJpegHandle representing the rocJPEG decoder instance.
 * @param jpeg_stream_handle The rocJpegStreamHandle representing the input JPEG stream.
 * @param decode_params A pointer to RocJpegDecodeParams containing the decoding parameters.
 * @param destination A pointer to RocJpegImage where the decoded image will be stored.
 * @return A RocJpegStatus indicating the success or failure of the decoding operation.
 */

RocJpegStatus ROCJPEGAPI
rocJpegDecode(RocJpegHandle              handle,
              RocJpegStreamHandle        jpeg_stream_handle,
              const RocJpegDecodeParams* decode_params,
              RocJpegImage*              destination);

/**
 * @fn RocJpegStatus ROCJPEGAPI rocJpegDecodeBatched(RocJpegHandle handle, RocJpegStreamHandle
 * *jpeg_stream_handles, int batch_size, const RocJpegDecodeParams *decode_params, RocJpegImage
 * *destinations);
 * @ingroup group_amd_rocjpeg
 * @brief Decodes a batch of JPEG images using the rocJPEG library.
 *
 * Decodes a batch of JPEG images using the rocJPEG library.
 *
 * @param handle The rocJPEG handle.
 * @param jpeg_stream_handles An array of rocJPEG stream handles representing the input JPEG
 * streams.
 * @param batch_size The number of JPEG streams in the batch.
 * @param decode_params The decode parameters for the JPEG decoding process.
 * @param destinations An array of rocJPEG images representing the output decoded images.
 * @return The status of the JPEG decoding operation.
 */
RocJpegStatus ROCJPEGAPI
rocJpegDecodeBatched(RocJpegHandle              handle,
                     RocJpegStreamHandle*       jpeg_stream_handles,
                     int                        batch_size,
                     const RocJpegDecodeParams* decode_params,
                     RocJpegImage*              destinations);

/**
 * @fn extern const char* ROCDECAPI rocJpegGetErrorName(RocJpegStatus rocjpeg_status);
 * @ingroup group_amd_rocjpeg
 * @brief Retrieves the name of the error associated with the given RocJpegStatus.
 *
 * This function returns a string representation of the error associated with the given
 * RocJpegStatus.
 *
 * @param rocjpeg_status The RocJpegStatus for which to retrieve the error name.
 * @return A pointer to a constant character string representing the error name.
 */
extern const char* ROCJPEGAPI
rocJpegGetErrorName(RocJpegStatus rocjpeg_status);

#if defined(__cplusplus)
}
#endif

#endif  // ROC_JPEG_H
