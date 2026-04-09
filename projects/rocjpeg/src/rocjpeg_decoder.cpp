/*
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "rocjpeg_decoder.h"

RocJpegDecoder::RocJpegDecoder(RocJpegBackend backend, int device_id) :
    num_devices_{0}, device_id_ {device_id}, hip_stream_ {0}, backend_{backend} {}

RocJpegDecoder::~RocJpegDecoder() {
    if (hip_stream_) {
        hipError_t hip_status = hipStreamDestroy(hip_stream_);
        if (hip_status != hipSuccess) {
            ERR("ERROR: Failed to destroy the HIP stream!");
        }
    }
}

/**
 * @brief Initializes the HIP environment for the RocJpegDecoder.
 *
 * This function initializes the HIP environment for the RocJpegDecoder by setting the device, 
 * creating a HIP stream, and retrieving device properties.
 *
 * @param device_id The ID of the device to be used for decoding.
 * @return The status of the initialization process.
 *         - ROCJPEG_STATUS_SUCCESS if the initialization is successful.
 *         - ROCJPEG_STATUS_NOT_INITIALIZED if no GPU device is found.
 *         - ROCJPEG_STATUS_INVALID_PARAMETER if the requested device_id is not found.
 */
RocJpegStatus RocJpegDecoder::InitHIP(int device_id) {
    CHECK_HIP(hipGetDeviceCount(&num_devices_));
    if (num_devices_ < 1) {
        ERR("ERROR: Failed to find any GPU!");
        return ROCJPEG_STATUS_NOT_INITIALIZED;
    }
    if (device_id >= num_devices_) {
        ERR("ERROR: the requested device_id is not found!");
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    CHECK_HIP(hipSetDevice(device_id));
    CHECK_HIP(hipGetDeviceProperties(&hip_dev_prop_, device_id));
    CHECK_HIP(hipStreamCreate(&hip_stream_));
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Initializes the RocJpegDecoder.
 *
 * This function initializes the RocJpegDecoder by performing the following steps:
 * 1. Initializes the HIP device.
 * 2. If the backend is ROCJPEG_BACKEND_HARDWARE, initializes the VA-API JPEG decoder.
 *
 * @return The status of the initialization process.
 *         - ROCJPEG_STATUS_SUCCESS if the initialization is successful.
 *         - An error code if the initialization fails.
 */
RocJpegStatus RocJpegDecoder::InitializeDecoder() {
    RocJpegStatus rocjpeg_status = ROCJPEG_STATUS_SUCCESS;
    rocjpeg_status = InitHIP(device_id_);
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
        ERR("ERROR: Failed to initilize the HIP!");
        return rocjpeg_status;
    }
    if (backend_ == ROCJPEG_BACKEND_HARDWARE) {
        std::string gpu_uuid(hip_dev_prop_.uuid.bytes, sizeof(hip_dev_prop_.uuid.bytes));
        rocjpeg_status = jpeg_vaapi_decoder_.InitializeDecoder(hip_dev_prop_.name, device_id_, gpu_uuid);
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
            ERR("ERROR: Failed to initialize the VA-API JPEG decoder!");
            return rocjpeg_status;
        }
    } else if (backend_ == ROCJPEG_BACKEND_HYBRID) {
        return ROCJPEG_STATUS_NOT_IMPLEMENTED;
    } else {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    return rocjpeg_status;
}

/**
 * @brief Decodes a JPEG image using the RocJpegDecoder.
 *
 * This function decodes a JPEG image from the provided JPEG stream handle and decode parameters,
 * and stores the decoded image in the destination buffer.
 *
 * @param jpeg_stream_handle The handle to the JPEG stream.
 * @param decode_params The decode parameters for the JPEG image.
 * @param destination The destination buffer to store the decoded image.
 * @return The status of the JPEG decoding operation.
 */
RocJpegStatus RocJpegDecoder::Decode(RocJpegStreamHandle jpeg_stream_handle, const RocJpegDecodeParams *decode_params, RocJpegImage *destination) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (jpeg_stream_handle == nullptr || decode_params == nullptr || destination == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    auto rocjpeg_stream_handle = static_cast<RocJpegStreamParserHandle*>(jpeg_stream_handle);
    const JpegStreamParameters *jpeg_stream_params = rocjpeg_stream_handle->rocjpeg_stream->GetJpegStreamParameters();

    VASurfaceID current_surface_id;
    CHECK_ROCJPEG(jpeg_vaapi_decoder_.SubmitDecode(jpeg_stream_params, current_surface_id, decode_params));

    HipInteropDeviceMem hip_interop_dev_mem = {};
    CHECK_ROCJPEG(jpeg_vaapi_decoder_.SyncSurface(current_surface_id));
    CHECK_ROCJPEG(jpeg_vaapi_decoder_.GetHipInteropMem(current_surface_id, hip_interop_dev_mem));

    uint16_t chroma_height = 0;
    uint16_t picture_width = 0;
    uint16_t picture_height = 0;
    bool is_roi_valid = false;
    uint32_t roi_width;
    uint32_t roi_height;
    roi_width = decode_params->crop_rectangle.right - decode_params->crop_rectangle.left;
    roi_height = decode_params->crop_rectangle.bottom - decode_params->crop_rectangle.top;

    if (roi_width > 0 && roi_height > 0 && roi_width <= jpeg_stream_params->picture_parameter_buffer.picture_width && roi_height <= jpeg_stream_params->picture_parameter_buffer.picture_height) {
        is_roi_valid = true;
    }

    picture_width = is_roi_valid ? roi_width : jpeg_stream_params->picture_parameter_buffer.picture_width;
    picture_height = is_roi_valid ? roi_height : jpeg_stream_params->picture_parameter_buffer.picture_height;

    VcnJpegSpec current_vcn_jpeg_spec = jpeg_vaapi_decoder_.GetCurrentVcnJpegSpec();
    if (is_roi_valid && current_vcn_jpeg_spec.can_roi_decode) {
        // Set is_roi_valid to false because in this case, the hardware handles the ROI decode and we don't
        // need to calculate the roi_offset later in the following functions (e.g., CopyChannel, GetPlanarYUVOutputFormat, etc) to copy the crop rectangle
        is_roi_valid = false;
    }

    switch (decode_params->output_format) {
        case ROCJPEG_OUTPUT_NATIVE:
            // Copy the native decoded output buffers from interop memory directly to the destination buffers
            CHECK_ROCJPEG(GetChromaHeight(hip_interop_dev_mem.surface_format, picture_height, chroma_height));

            // Copy Luma (first channel) for any surface format
            CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_width, picture_height, 0, destination, decode_params, is_roi_valid));

            if (hip_interop_dev_mem.surface_format == VA_FOURCC_NV12) {
                // Copy the second channel (UV interleaved) for NV12
                CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_width, chroma_height, 1, destination, decode_params, is_roi_valid));
            } else if (hip_interop_dev_mem.surface_format == VA_FOURCC_444P ||
                       hip_interop_dev_mem.surface_format == VA_FOURCC_422V) {
                // Copy the second and third channels for YUV444 and YUV440 (i.e., YUV422V)
                CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_width, chroma_height, 1, destination, decode_params, is_roi_valid));
                CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_width, chroma_height, 2, destination, decode_params, is_roi_valid));
            }
            break;
        case ROCJPEG_OUTPUT_YUV_PLANAR:
            CHECK_ROCJPEG(GetChromaHeight(hip_interop_dev_mem.surface_format, picture_height, chroma_height));
            CHECK_ROCJPEG(GetPlanarYUVOutputFormat(hip_interop_dev_mem, picture_width,
                                                   picture_height, chroma_height, destination, decode_params, is_roi_valid));
            break;
        case ROCJPEG_OUTPUT_Y:
            CHECK_ROCJPEG(GetYOutputFormat(hip_interop_dev_mem, picture_width,
                                           picture_height, destination, decode_params, is_roi_valid));
            break;
        case ROCJPEG_OUTPUT_RGB:
            CHECK_ROCJPEG(ColorConvertToRGB(hip_interop_dev_mem, picture_width,
                                                    picture_height, destination, decode_params, is_roi_valid));
            break;
        case ROCJPEG_OUTPUT_RGB_PLANAR:
            CHECK_ROCJPEG(ColorConvertToRGBPlanar(hip_interop_dev_mem, picture_width,
                                                    picture_height, destination, decode_params, is_roi_valid));
            break;
        default:
            break;
    }

    CHECK_ROCJPEG(jpeg_vaapi_decoder_.SetSurfaceAsIdle(current_surface_id));
    CHECK_HIP(hipStreamSynchronize(hip_stream_));
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * Decodes a batch of JPEG streams using the specified decode parameters and stores the decoded images in the provided destinations.
 *
 * @param jpeg_streams An array of RocJpegStreamHandle objects representing the JPEG streams to be decoded.
 * @param batch_size The number of JPEG streams in the batch.
 * @param decode_params A pointer to RocJpegDecodeParams object containing the decode parameters.
 * @param destinations An array of RocJpegImage objects where the decoded images will be stored.
 * @return A RocJpegStatus value indicating the success or failure of the decoding operation.
 */
RocJpegStatus RocJpegDecoder::DecodeBatched(RocJpegStreamHandle *jpeg_streams, int batch_size, const RocJpegDecodeParams *decode_params, RocJpegImage *destinations) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (jpeg_streams == nullptr || decode_params == nullptr || destinations == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }

    std::vector<VASurfaceID> current_surface_ids;
    std::vector<JpegStreamParameters> jpeg_streams_params;
    current_surface_ids.resize(batch_size);
    jpeg_streams_params.resize(batch_size);
    VcnJpegSpec current_vcn_jpeg_spec = jpeg_vaapi_decoder_.GetCurrentVcnJpegSpec();

    for (int i = 0; i < batch_size; i += current_vcn_jpeg_spec.num_jpeg_cores) {
        int batch_end = std::min(i + static_cast<int>(current_vcn_jpeg_spec.num_jpeg_cores), batch_size);
        int current_batch_size = batch_end - i;

        for (int j = i; j < batch_end; j++) {
            auto rocjpeg_stream_handle = static_cast<RocJpegStreamParserHandle*>(jpeg_streams[j]);
            const JpegStreamParameters *jpeg_stream_params = rocjpeg_stream_handle->rocjpeg_stream->GetJpegStreamParameters();
            jpeg_streams_params[j] = std::move(*jpeg_stream_params);
        }

        CHECK_ROCJPEG(jpeg_vaapi_decoder_.SubmitDecodeBatched(jpeg_streams_params.data() + i, current_batch_size, &decode_params[i], current_surface_ids.data() + i));

        for (int k = 0; k < current_batch_size; k++) {
            HipInteropDeviceMem hip_interop_dev_mem = {};
            VASurfaceID current_surface_id = *(current_surface_ids.data() + k + i);
            const JpegStreamParameters *jpeg_stream_params = jpeg_streams_params.data() + k + i;
            CHECK_ROCJPEG(jpeg_vaapi_decoder_.SyncSurface(current_surface_id));
            CHECK_ROCJPEG(jpeg_vaapi_decoder_.GetHipInteropMem(current_surface_id, hip_interop_dev_mem));

            uint16_t chroma_height = 0;
            uint16_t picture_width = 0;
            uint16_t picture_height = 0;
            bool is_roi_valid = false;
            uint32_t roi_width;
            uint32_t roi_height;
            roi_width = decode_params[k + i].crop_rectangle.right - decode_params[k + i].crop_rectangle.left;
            roi_height = decode_params[k + i].crop_rectangle.bottom - decode_params[k + i].crop_rectangle.top;
    
            if (roi_width > 0 && roi_height > 0 && roi_width <= jpeg_stream_params->picture_parameter_buffer.picture_width && roi_height <= jpeg_stream_params->picture_parameter_buffer.picture_height) {
                is_roi_valid = true;
            }

            picture_width = is_roi_valid ? roi_width : jpeg_stream_params->picture_parameter_buffer.picture_width;
            picture_height = is_roi_valid ? roi_height : jpeg_stream_params->picture_parameter_buffer.picture_height;

            if (is_roi_valid && current_vcn_jpeg_spec.can_roi_decode) {
                // Set is_roi_valid to false because in this case, the hardware handles the ROI decode and we don't need to calculate the roi_offset
                // later in the following functions (e.g., CopyChannel, GetPlanarYUVOutputFormat, etc) to copy the crop rectangle
                is_roi_valid = false;
            }

            switch (decode_params[k + i].output_format) {
                case ROCJPEG_OUTPUT_NATIVE:
                    // Copy the native decoded output buffers from interop memory directly to the destination buffers
                    CHECK_ROCJPEG(GetChromaHeight(hip_interop_dev_mem.surface_format, picture_height, chroma_height));
                    // Copy Luma (first channel) for any surface format
                    CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_width, picture_height, 0, &destinations[k + i], &decode_params[k + i], is_roi_valid));
                    if (hip_interop_dev_mem.surface_format == VA_FOURCC_NV12) {
                        // Copy the second channel (UV interleaved) for NV12
                        CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_width, chroma_height, 1, &destinations[k + i], &decode_params[k + i], is_roi_valid));
                    } else if (hip_interop_dev_mem.surface_format == VA_FOURCC_444P ||
                            hip_interop_dev_mem.surface_format == VA_FOURCC_422V) {
                        // Copy the second and third channels for YUV444 and YUV440 (i.e., YUV422V)
                        CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_width, chroma_height, 1, &destinations[k + i], &decode_params[k + i], is_roi_valid));
                        CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_width, chroma_height, 2, &destinations[k + i], &decode_params[k + i], is_roi_valid));
                    }
                    break;
                case ROCJPEG_OUTPUT_YUV_PLANAR:
                    CHECK_ROCJPEG(GetChromaHeight(hip_interop_dev_mem.surface_format, picture_height, chroma_height));
                    CHECK_ROCJPEG(GetPlanarYUVOutputFormat(hip_interop_dev_mem, picture_width,
                                                        picture_height, chroma_height, &destinations[k + i], &decode_params[k + i], is_roi_valid));
                    break;
                case ROCJPEG_OUTPUT_Y:
                    CHECK_ROCJPEG(GetYOutputFormat(hip_interop_dev_mem, picture_width,
                                                picture_height, &destinations[k + i], &decode_params[k + i], is_roi_valid));
                    break;
                case ROCJPEG_OUTPUT_RGB:
                    CHECK_ROCJPEG(ColorConvertToRGB(hip_interop_dev_mem, picture_width,
                                                            picture_height, &destinations[k + i], &decode_params[k + i], is_roi_valid));
                    break;
                case ROCJPEG_OUTPUT_RGB_PLANAR:
                    CHECK_ROCJPEG(ColorConvertToRGBPlanar(hip_interop_dev_mem, picture_width,
                                                            picture_height, &destinations[k + i], &decode_params[k + i], is_roi_valid));
                    break;
                default:
                    break;
            }
        }
        CHECK_HIP(hipStreamSynchronize(hip_stream_));
        for (int k = 0; k < current_batch_size; k++) {
            VASurfaceID current_surface_id = *(current_surface_ids.data() + k + i);
            CHECK_ROCJPEG(jpeg_vaapi_decoder_.SetSurfaceAsIdle(current_surface_id));
        }
    }

    return ROCJPEG_STATUS_SUCCESS;
}
/**
 * @brief Retrieves the image information from the JPEG stream.
 *
 * This function retrieves the number of components, chroma subsampling, widths, and heights
 * of the image from the given JPEG stream.
 *
 * @param jpeg_stream_handle The handle to the JPEG stream.
 * @param num_components Pointer to store the number of components in the image.
 * @param subsampling Pointer to store the chroma subsampling of the image.
 * @param widths Array to store the widths of the image components.
 * @param heights Array to store the heights of the image components.
 * @return The status of the operation. Returns ROCJPEG_STATUS_SUCCESS if successful,
 *         or ROCJPEG_STATUS_INVALID_PARAMETER if any of the input parameters are invalid.
 */
RocJpegStatus RocJpegDecoder::GetImageInfo(RocJpegStreamHandle jpeg_stream_handle, uint8_t *num_components, RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights){
    std::lock_guard<std::mutex> lock(mutex_);
    if (jpeg_stream_handle == nullptr || num_components == nullptr || subsampling == nullptr || widths == nullptr || heights == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    auto rocjpeg_stream_handle = static_cast<RocJpegStreamParserHandle*>(jpeg_stream_handle);
    const JpegStreamParameters *jpeg_stream_params = rocjpeg_stream_handle->rocjpeg_stream->GetJpegStreamParameters();

    *num_components = jpeg_stream_params->picture_parameter_buffer.num_components;
    widths[0] = jpeg_stream_params->picture_parameter_buffer.picture_width;
    heights[0] = jpeg_stream_params->picture_parameter_buffer.picture_height;
    widths[3] = 0;
    heights[3] = 0;

    switch (jpeg_stream_params->chroma_subsampling) {
        case CSS_444:
            *subsampling = ROCJPEG_CSS_444;
            widths[2] = widths[1] = widths[0];
            heights[2] = heights[1] = heights[0];
            break;
        case CSS_440:
            *subsampling = ROCJPEG_CSS_440;
            widths[2] = widths[1] = widths[0];
            heights[2] = heights[1] = heights[0] >> 1;
            break;
        case CSS_422:
            *subsampling = ROCJPEG_CSS_422;
            widths[2] = widths[1] = widths[0] >> 1;
            heights[2] = heights[1] = heights[0];
            break;
        case CSS_420:
            *subsampling = ROCJPEG_CSS_420;
            widths[2] = widths[1] = widths[0] >> 1;
            heights[2] = heights[1] = heights[0] >> 1;
            break;
        case CSS_400:
            *subsampling = ROCJPEG_CSS_400;
            widths[3] = widths[2] = widths[1] = 0;
            heights[3] = heights[2] = heights[1] = 0;
            break;
        case CSS_411:
            *subsampling = ROCJPEG_CSS_411;
            widths[2] = widths[1] = widths[0] >> 2;
            heights[2] = heights[1] = heights[0];
            break;
        default:
            *subsampling = ROCJPEG_CSS_UNKNOWN;
            break;
    }

    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Copies a channel from the `hip_interop_dev_mem` to the `destination` image.
 *
 * This function copies the channel specified by `channel_index` from the `hip_interop_dev_mem` to the `destination` image.
 * The `channel_height` parameter specifies the height of the channel.
 *
 * @param hip_interop_dev_mem The `HipInteropDeviceMem` object containing the source channel data.
 * @param channel_width The width of the channel to be copied.
 * @param channel_height The height of the channel to be copied.
 * @param channel_index The index of the channel to be copied.
 * @param destination The `RocJpegImage` object representing the destination image.
 * @return The status of the operation. Returns `ROCJPEG_STATUS_SUCCESS` if the channel was copied successfully.
 */
RocJpegStatus RocJpegDecoder::CopyChannel(HipInteropDeviceMem& hip_interop_dev_mem, uint16_t channel_width, uint16_t channel_height, uint8_t channel_index, RocJpegImage *destination, const RocJpegDecodeParams *decode_params, bool is_roi_valid) {
    if (hip_interop_dev_mem.pitch[channel_index] != 0 && destination->pitch[channel_index] != 0 && destination->channel[channel_index] != nullptr) {
        uint32_t roi_offset = 0;
        if (is_roi_valid) {
            int16_t top = decode_params->crop_rectangle.top;
            int16_t left = decode_params->crop_rectangle.left;
            // adjustments need to be made for these 3 pixel formats
            switch (hip_interop_dev_mem.surface_format) {
                case VA_FOURCC_NV12:
                case VA_FOURCC_422V:
                    top = (channel_index == 1 || channel_index == 2) ? top >> 1 : top;
                    break;
                case VA_FOURCC_YUY2:
                    left *= 2;
                    break;
            }
            roi_offset = top * hip_interop_dev_mem.pitch[channel_index] + left;
        }

        uint32_t channel_widths[ROCJPEG_MAX_COMPONENT] = {};
        uint32_t roi_width = decode_params->crop_rectangle.right - decode_params->crop_rectangle.left;
        bool is_roi_width_valid = roi_width > 0 && roi_width <= channel_width;
        switch (decode_params->output_format) {
            case ROCJPEG_OUTPUT_NATIVE:
                switch (hip_interop_dev_mem.surface_format) {
                    case VA_FOURCC_444P:
                        channel_widths[2] = channel_widths[1] = channel_widths[0] = is_roi_width_valid ? roi_width : channel_width;
                        break;
                    case VA_FOURCC_422V:
                        channel_widths[2] = channel_widths[1] = channel_widths[0] = is_roi_width_valid ? roi_width : channel_width;
                        break;
                    case VA_FOURCC_YUY2:
                        channel_widths[0] = (is_roi_width_valid ? roi_width : channel_width) * 2;
                        break;
                    case VA_FOURCC_NV12:
                        channel_widths[1] = channel_widths[0] = is_roi_width_valid ? roi_width : channel_width;
                        break;
                    case VA_FOURCC_Y800:
                        channel_widths[0] = is_roi_width_valid ? roi_width : channel_width;
                        break;
                    default:
                        ERR("Unknown output format!");
                        return ROCJPEG_STATUS_INVALID_PARAMETER;
                    }
                break;
            case ROCJPEG_OUTPUT_YUV_PLANAR:
                switch (hip_interop_dev_mem.surface_format) {
                    case VA_FOURCC_444P:
                        channel_widths[2] = channel_widths[1] = channel_widths[0] = is_roi_width_valid ? roi_width : channel_width;
                        break;
                    case VA_FOURCC_422V:
                        channel_widths[2] = channel_widths[1] = channel_widths[0] = is_roi_width_valid ? roi_width : channel_width;
                        break;
                    case VA_FOURCC_YUY2:
                        channel_widths[0] = is_roi_width_valid ? roi_width : channel_width;
                        channel_widths[2] = channel_widths[1] = channel_widths[0] >> 1;
                        break;
                    case VA_FOURCC_NV12:
                        channel_widths[0] = is_roi_width_valid ? roi_width : channel_width;
                        channel_widths[2] = channel_widths[1] = channel_widths[0] >> 1;
                        break;
                    case VA_FOURCC_Y800:
                        channel_widths[0] = is_roi_width_valid ? roi_width : channel_width;
                        break;
                    default:
                        ERR("Unknown output format!");
                        return ROCJPEG_STATUS_INVALID_PARAMETER;
                    }
                break;
            case ROCJPEG_OUTPUT_Y:
                channel_widths[0] = is_roi_width_valid ? roi_width : channel_width;
                break;
            case ROCJPEG_OUTPUT_RGB:
                channel_widths[0] = (is_roi_width_valid ? roi_width : channel_width) * 3;
                break;
            case ROCJPEG_OUTPUT_RGB_PLANAR:
                channel_widths[2] = channel_widths[1] = channel_widths[0] = is_roi_width_valid ? roi_width : channel_width;
                break;
            default:
                ERR("Unknown output format!");
                return ROCJPEG_STATUS_INVALID_PARAMETER;
        }

        if (destination->pitch[channel_index] == hip_interop_dev_mem.pitch[channel_index]) {
            uint32_t channel_size = destination->pitch[channel_index] * channel_height;
            CHECK_HIP(hipMemcpyDtoDAsync(destination->channel[channel_index], hip_interop_dev_mem.hip_mapped_device_mem + hip_interop_dev_mem.offset[channel_index] + roi_offset, channel_size, hip_stream_));
        } else {
            CHECK_HIP(hipMemcpy2DAsync(destination->channel[channel_index], destination->pitch[channel_index], hip_interop_dev_mem.hip_mapped_device_mem + hip_interop_dev_mem.offset[channel_index] + roi_offset, hip_interop_dev_mem.pitch[channel_index],
            channel_widths[channel_index], channel_height, hipMemcpyDeviceToDevice, hip_stream_));
        }
    }
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Calculates the chroma height based on the surface format and picture height.
 *
 * This function takes the surface format, picture height, and a reference to the chroma height.
 * It calculates the chroma height based on the surface format and assigns the result to the chroma_height parameter.
 *
 * @param surface_format The surface format of the image.
 * @param picture_height The height of the picture.
 * @param chroma_height  A reference to the variable where the calculated chroma height will be stored.
 *
 * @return The status of the operation. Returns ROCJPEG_STATUS_SUCCESS if successful, or ROCJPEG_STATUS_JPEG_NOT_SUPPORTED if the surface format is not supported.
 */
RocJpegStatus RocJpegDecoder::GetChromaHeight(uint32_t surface_format, uint16_t picture_height, uint16_t &chroma_height) {
    switch (surface_format) {
        case VA_FOURCC_NV12: /*NV12: two-plane 8-bit YUV 4:2:0*/
            chroma_height = picture_height >> 1;
            break;
        case VA_FOURCC_444P: /*444P: three-plane 8-bit YUV 4:4:4*/
            chroma_height = picture_height;
            break;
        case VA_FOURCC_Y800: /*Y800: one-plane 8-bit greyscale YUV 4:0:0*/
            chroma_height = 0;
            break;
        case VA_FOURCC_YUY2: /*YUYV: one-plane packed 8-bit YUV 4:2:2. Four bytes per pair of pixels: Y, U, Y, V*/
            chroma_height = picture_height;
            break;
        case VA_FOURCC_422V: /*422V: three-plane 8-bit YUV 4:4:0*/
            chroma_height = picture_height >> 1;
            break;
        default:
            return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
    }
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Converts the color format of the input image to RGB format.
 *
 * This function converts the color format of the input image to RGB format based on the surface format
 * specified in the `hip_interop_dev_mem` parameter. The converted image is stored in the `destination`
 * parameter.
 *
 * @param hip_interop_dev_mem The HipInteropDeviceMem object containing the input image data.
 * @param picture_width The width of the destination image.
 * @param picture_height The height of the destination image.
 * @param destination Pointer to the RocJpegImage object where the converted image will be stored.
 * @return The status of the color conversion operation. Returns ROCJPEG_STATUS_SUCCESS if the conversion
 *         is successful. Returns ROCJPEG_STATUS_JPEG_NOT_SUPPORTED if the surface format is not supported.
 */
RocJpegStatus RocJpegDecoder::ColorConvertToRGB(HipInteropDeviceMem& hip_interop_dev_mem, uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination, const RocJpegDecodeParams *decode_params, bool is_roi_valid) {
    uint32_t roi_offset = 0;
    uint32_t roi_uv_offset = 0;
    int16_t top = decode_params->crop_rectangle.top;
    int16_t left = decode_params->crop_rectangle.left;
    if (is_roi_valid) {
        if (hip_interop_dev_mem.surface_format == VA_FOURCC_422V || hip_interop_dev_mem.surface_format == VA_FOURCC_NV12){
            roi_uv_offset = (top >> 1) * hip_interop_dev_mem.pitch[1] + left;
        } else if (hip_interop_dev_mem.surface_format == VA_FOURCC_YUY2) {
            left *= 2;
        }
        roi_offset = top * hip_interop_dev_mem.pitch[0] + left;
    }
    switch (hip_interop_dev_mem.surface_format) {
        case VA_FOURCC_444P:
            ColorConvertYUV444ToRGB(hip_stream_, picture_width, picture_height, destination->channel[0], destination->pitch[0],
                                                  hip_interop_dev_mem.hip_mapped_device_mem + roi_offset, hip_interop_dev_mem.pitch[0], hip_interop_dev_mem.offset[1] + roi_offset, hip_interop_dev_mem.offset[2] + roi_offset);
            break;
        case VA_FOURCC_422V:
            ColorConvertYUV440ToRGB(hip_stream_, picture_width, picture_height, destination->channel[0], destination->pitch[0],
                                                  hip_interop_dev_mem.hip_mapped_device_mem + roi_offset, hip_interop_dev_mem.pitch[0], hip_interop_dev_mem.offset[1] /*+ roi_uv_offset*/, hip_interop_dev_mem.offset[2] /*+ roi_uv_offset*/);
            break;
        case VA_FOURCC_YUY2:
            ColorConvertYUYVToRGB(hip_stream_, picture_width, picture_height, destination->channel[0], destination->pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem + roi_offset, hip_interop_dev_mem.pitch[0]);
            break;
        case VA_FOURCC_NV12:
            ColorConvertNV12ToRGB(hip_stream_, picture_width, picture_height, destination->channel[0], destination->pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem + roi_offset, hip_interop_dev_mem.pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem + hip_interop_dev_mem.offset[1] + roi_uv_offset, hip_interop_dev_mem.pitch[1]);
            break;
        case VA_FOURCC_Y800:
            ColorConvertYUV400ToRGB(hip_stream_, picture_width, picture_height, destination->channel[0], destination->pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem + roi_offset, hip_interop_dev_mem.pitch[0]);
           break;
        case VA_FOURCC_RGBA:
            ColorConvertRGBAToRGB(hip_stream_, picture_width, picture_height, destination->channel[0], destination->pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem + roi_offset, hip_interop_dev_mem.pitch[0]);
           break;
        default:
            ERR("ERROR! surface format is not supported!");
            return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
    }
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Converts the color format of the input image to RGB planar format.
 *
 * This function converts the color format of the input image to RGB planar format.
 * The conversion is performed based on the surface format specified in the `hip_interop_dev_mem`.
 * The converted image is stored in the `destination` RocJpegImage object.
 *
 * @param hip_interop_dev_mem The HipInteropDeviceMem object containing the input image data.
 * @param picture_width The width of the destination image.
 * @param picture_height The height of the destination image.
 * @param destination Pointer to the RocJpegImage object where the converted image will be stored.
 * @return RocJpegStatus The status of the color conversion operation.
 *         Returns ROCJPEG_STATUS_SUCCESS if the conversion is successful.
 *         Returns ROCJPEG_STATUS_JPEG_NOT_SUPPORTED if the surface format is not supported.
 */
RocJpegStatus RocJpegDecoder::ColorConvertToRGBPlanar(HipInteropDeviceMem& hip_interop_dev_mem, uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination, const RocJpegDecodeParams *decode_params, bool is_roi_valid) {
    uint32_t roi_offset = 0;
    uint32_t roi_uv_offset = 0;
    int16_t top = decode_params->crop_rectangle.top;
    int16_t left = decode_params->crop_rectangle.left;
    if (is_roi_valid) {
        if (hip_interop_dev_mem.surface_format == VA_FOURCC_422V || hip_interop_dev_mem.surface_format == VA_FOURCC_NV12){
            roi_uv_offset = (top >> 1) * hip_interop_dev_mem.pitch[1] + left;
        } else if (hip_interop_dev_mem.surface_format == VA_FOURCC_YUY2) {
            left *= 2;
        }
        roi_offset = top * hip_interop_dev_mem.pitch[0] + left;
    }
    switch (hip_interop_dev_mem.surface_format) {
        case VA_FOURCC_444P:
            ColorConvertYUV444ToRGBPlanar(hip_stream_, picture_width, picture_height, destination->channel[0], destination->channel[1], destination->channel[2], destination->pitch[0],
                                                  hip_interop_dev_mem.hip_mapped_device_mem + roi_offset, hip_interop_dev_mem.pitch[0], hip_interop_dev_mem.offset[1] + roi_offset, hip_interop_dev_mem.offset[2] + roi_offset);
            break;
        case VA_FOURCC_422V:
            ColorConvertYUV440ToRGBPlanar(hip_stream_, picture_width, picture_height, destination->channel[0], destination->channel[1], destination->channel[2], destination->pitch[0],
                                                  hip_interop_dev_mem.hip_mapped_device_mem + roi_offset, hip_interop_dev_mem.pitch[0], hip_interop_dev_mem.offset[1] /*+ roi_uv_offset*/, hip_interop_dev_mem.offset[2] /*+ roi_uv_offset*/);
            break;
        case VA_FOURCC_YUY2:
            ColorConvertYUYVToRGBPlanar(hip_stream_, picture_width, picture_height, destination->channel[0], destination->channel[1], destination->channel[2], destination->pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem + roi_offset, hip_interop_dev_mem.pitch[0]);
            break;
        case VA_FOURCC_NV12:
            ColorConvertNV12ToRGBPlanar(hip_stream_, picture_width, picture_height, destination->channel[0], destination->channel[1], destination->channel[2], destination->pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem + roi_offset, hip_interop_dev_mem.pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem + hip_interop_dev_mem.offset[1] + roi_uv_offset, hip_interop_dev_mem.pitch[1]);
            break;
        case VA_FOURCC_Y800:
            ColorConvertYUV400ToRGBPlanar(hip_stream_, picture_width, picture_height, destination->channel[0], destination->channel[1], destination->channel[2], destination->pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem + roi_offset, hip_interop_dev_mem.pitch[0]);
           break;
        case VA_FOURCC_RGBP:
            // Copy red, green, and blue channels from the interop memory into the destination
            for (uint8_t channel_index = 0; channel_index < 3; channel_index++) {
                CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_width, picture_height, channel_index, destination, decode_params, is_roi_valid));
            }
           break;
        default:
            ERR("ERROR! surface format is not supported!");
            return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
    }
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Retrieves the planar YUV output format from the input image.
 *
 * This function converts the input image data to planar YUV format based on the surface format of the input data.
 * If the surface format is VA_FOURCC_YUY2, the function extracts the packed YUYV data and copies them into the
 * first, second, and third channels of the destination image. If the surface format is VA_FOURCC_NV12, the function
 * extracts the interleaved UV channels and copies them into the second and third channels of the destination image.
 * If the surface format is VA_FOURCC_444P, the function copies the luma channel and both chroma channels into the
 * destination image.
 *
 * @param hip_interop_dev_mem The HipInteropDeviceMem object containing the input image data.
 * @param picture_width The width of the input picture.
 * @param picture_height The height of the input picture.
 * @param chroma_height The height of the chroma channels.
 * @param destination Pointer to the RocJpegImage object where the converted image data will be stored.
 * @return The status of the operation. Returns ROCJPEG_STATUS_SUCCESS if successful.
 */
RocJpegStatus RocJpegDecoder::GetPlanarYUVOutputFormat(HipInteropDeviceMem& hip_interop_dev_mem, uint32_t picture_width, uint32_t picture_height, uint16_t chroma_height, RocJpegImage *destination, const RocJpegDecodeParams *decode_params, bool is_roi_valid) {
    uint32_t roi_offset = 0;
    if (is_roi_valid) {
         int16_t top = decode_params->crop_rectangle.top;
         int16_t left = decode_params->crop_rectangle.left;
         if (hip_interop_dev_mem.surface_format == VA_FOURCC_NV12){
            roi_offset = (top >> 1) * hip_interop_dev_mem.pitch[1] + left;
         } else if (hip_interop_dev_mem.surface_format == VA_FOURCC_YUY2) {
            roi_offset = top * hip_interop_dev_mem.pitch[0] + (left * 2);
         }
    }
    if (hip_interop_dev_mem.surface_format == VA_FOURCC_YUY2) {
        // Extract the packed YUYV and copy them into the first, second, and third channels of the destination.
        ConvertPackedYUYVToPlanarYUV(hip_stream_, picture_width, picture_height, destination->channel[0], destination->channel[1], destination->channel[2],
                                                  destination->pitch[0], destination->pitch[1], hip_interop_dev_mem.hip_mapped_device_mem + roi_offset, hip_interop_dev_mem.pitch[0]);
    } else {
        // Copy Luma
        CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_width, picture_height, 0, destination, decode_params, is_roi_valid));
        if (hip_interop_dev_mem.surface_format == VA_FOURCC_NV12) {
            // Extract the interleaved UV channels and copy them into the second and third channels of the destination.
            ConvertInterleavedUVToPlanarUV(hip_stream_, picture_width >> 1, picture_height >> 1, destination->channel[1], destination->channel[2],
                destination->pitch[1], hip_interop_dev_mem.hip_mapped_device_mem + hip_interop_dev_mem.offset[1] + roi_offset, hip_interop_dev_mem.pitch[1]);
        } else if (hip_interop_dev_mem.surface_format == VA_FOURCC_444P ||
                   hip_interop_dev_mem.surface_format == VA_FOURCC_422V) {
            CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_width, chroma_height, 1, destination, decode_params, is_roi_valid));
            CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_width, chroma_height, 2, destination, decode_params, is_roi_valid));
        }
    }
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Retrieves the Y output format from the input YUV image.
 *
 * This function extracts the Y output format from the RocJpegDecoder based on the provided parameters.
 * If the surface format is VA_FOURCC_YUY2, it calls the ExtractYFromPackedYUYV function to extract the Y component
 * from the packed YUYV format. Otherwise, it calls the CopyChannel function to copy the luma channel.
 *
 * @param hip_interop_dev_mem The HipInteropDeviceMem object containing the surface format and device memory.
 * @param picture_width The width of the picture.
 * @param picture_height The height of the picture.
 * @param destination Pointer to the RocJpegImage object where the extracted Y component will be stored.
 * @return The status of the operation. Returns ROCJPEG_STATUS_SUCCESS if successful.
 */
RocJpegStatus RocJpegDecoder::GetYOutputFormat(HipInteropDeviceMem& hip_interop_dev_mem, uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination, const RocJpegDecodeParams *decode_params, bool is_roi_valid) {
    uint32_t roi_offset = 0; 
    if (hip_interop_dev_mem.surface_format == VA_FOURCC_YUY2) {
        // calculate offset and add to hip_mapped_device_mem
        if (is_roi_valid) {
                int16_t top = decode_params->crop_rectangle.top;
                int16_t left = decode_params->crop_rectangle.left * 2;
                roi_offset = top * hip_interop_dev_mem.pitch[0] + left;
        }
        ExtractYFromPackedYUYV(hip_stream_, picture_width, picture_height, destination->channel[0], destination->pitch[0],
                              hip_interop_dev_mem.hip_mapped_device_mem + roi_offset, hip_interop_dev_mem.pitch[0]);
    } else {
        // Copy Luma
        CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_width, picture_height, 0, destination, decode_params, is_roi_valid));
    }
    return ROCJPEG_STATUS_SUCCESS;
}