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

#ifndef ROC_JPEG_DECODER_H_
#define ROC_JPEG_DECODER_H_

#include <unistd.h>
#include <vector>
#include <mutex>
#include <unordered_map>
#include "../api/rocjpeg/rocjpeg.h"
#include "rocjpeg_api_stream_handle.h"
#include "rocjpeg_parser.h"
#include "rocjpeg_commons.h"
#include "rocjpeg_vaapi_decoder.h"
#include "rocjpeg_hip_kernels.h"

/**
 * @brief Reusable scratch buffer for one batched kernel's per-image parameter array.
 *
 * Accumulates a host vector of per-image parameter structs for a single group,
 * tracks the maximum grid extents across the group, and lazily grows a device
 * buffer that the params are uploaded into before the batched launch.
 */
template <typename T>
class BatchedKernelParams {
public:
    BatchedKernelParams() = default;
    ~BatchedKernelParams() { if (dev_) { (void)hipFree(dev_); } }

    // Owns a raw device buffer; copying/moving would double-free it.
    BatchedKernelParams(const BatchedKernelParams &) = delete;
    BatchedKernelParams &operator=(const BatchedKernelParams &) = delete;
    BatchedKernelParams(BatchedKernelParams &&) = delete;
    BatchedKernelParams &operator=(BatchedKernelParams &&) = delete;

    void Reset() { host_.clear(); max_gx_ = 0; max_gy_ = 0; }

    // Append one image's params; gx/gy are that image's pre-block-division grid extents.
    void Add(const T &params, uint32_t gx, uint32_t gy) {
        host_.push_back(params);
        if (gx > max_gx_) max_gx_ = gx;
        if (gy > max_gy_) max_gy_ = gy;
    }

    bool Empty() const { return host_.empty(); }
    uint32_t Count() const { return static_cast<uint32_t>(host_.size()); }
    uint32_t MaxGx() const { return max_gx_; }
    uint32_t MaxGy() const { return max_gy_; }

    // Grow the device buffer on demand and upload the host params on the given stream.
    hipError_t Upload(hipStream_t stream, T **dev_out) {
        size_t bytes = host_.size() * sizeof(T);
        if (host_.size() > capacity_) {
            if (dev_) { (void)hipFree(dev_); dev_ = nullptr; }
            hipError_t status = hipMalloc(reinterpret_cast<void **>(&dev_), bytes);
            if (status != hipSuccess) return status;
            capacity_ = host_.size();
        }
        *dev_out = dev_;
        return hipMemcpyAsync(dev_, host_.data(), bytes, hipMemcpyHostToDevice, stream);
    }

private:
    std::vector<T> host_;
    T *dev_ = nullptr;
    size_t capacity_ = 0;
    uint32_t max_gx_ = 0;
    uint32_t max_gy_ = 0;
};

/**
 * @class RocJpegDecoder
 * @brief The RocJpegDecoder class represents a JPEG decoder.
 *
 * This class provides methods to initialize the decoder, retrieve image information,
 * and decode JPEG streams into RocJpegImage objects.
 */
/**
 * @brief The RocJpegDecoder class is responsible for decoding JPEG images using a hardware-accelerated jpeg decoder.
 */
class RocJpegDecoder {
public:
   /**
    * @brief Constructs a RocJpegDecoder object.
    * @param backend The ROCm backend to be used for decoding (default: ROCJPEG_BACKEND_HARDWARE).
    * @param device_id The ID of the device to be used for decoding (default: 0).
    */
   RocJpegDecoder(RocJpegBackend backend = ROCJPEG_BACKEND_HARDWARE, int device_id = 0);

   /**
    * @brief Destroys the RocJpegDecoder object.
    */
   ~RocJpegDecoder();

   /**
    * @brief Initializes the decoder.
    * @return The status of the initialization process.
    */
   RocJpegStatus InitializeDecoder();

   /**
    * @brief Retrieves information about the JPEG image.
    * @param jpeg_stream The handle to the JPEG stream.
    * @param num_components Pointer to store the number of color components in the image.
    * @param subsampling Pointer to store the chroma subsampling information.
    * @param widths Pointer to store the widths of the image components.
    * @param heights Pointer to store the heights of the image components.
    * @return The status of the operation.
    */
   RocJpegStatus GetImageInfo(RocJpegStreamHandle jpeg_stream, uint8_t *num_components, RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights);

   /**
    * @brief Decodes the JPEG image.
    * @param jpeg_stream The handle to the JPEG stream.
    * @param decode_params The decoding parameters.
    * @param destination Pointer to the destination image.
    * @return The status of the decoding process.
    */
   RocJpegStatus Decode(RocJpegStreamHandle jpeg_stream, const RocJpegDecodeParams *decode_params, RocJpegImage *destination);

   /**
    * Decodes a batch of JPEG streams.
    *
    * This function decodes a batch of JPEG streams specified by `jpeg_streams` into a batch of destination images specified by `destinations`.
    * The number of JPEG streams in the batch is specified by `batch_size`.
    * The decoding parameters are specified by `decode_params`.
    *
    * @param jpeg_streams The array of JPEG stream handles.
    * @param batch_size The number of JPEG streams in the batch.
    * @param decode_params The decoding parameters.
    * @param destinations The array of destination images.
    * @return The status of the decoding operation.
    */
   RocJpegStatus DecodeBatched(RocJpegStreamHandle *jpeg_streams, int batch_size, const RocJpegDecodeParams *decode_params, RocJpegImage *destinations);

   /**
    * @brief Submits a JPEG decode operation and stores pending state in this decoder handle.
    * @param jpeg_stream The handle to the JPEG stream.
    * @param decode_params The decoding parameters.
    * @param destination Pointer to the output image.
    * @return The status of the submit operation.
    */
   RocJpegStatus DecodeAsync(RocJpegStreamHandle jpeg_stream, const RocJpegDecodeParams *decode_params, RocJpegImage *destination);

   /**
    * @brief Synchronizes a pending asynchronous decode and copies/converts the output.
    * @param destination Pointer to the destination image.
    * @return The status of the sync operation.
    */
   RocJpegStatus DecodeSync(RocJpegImage *destination);

   /**
    * @brief Submits a batch of JPEG decode operations and stores pending state for all images.
    * @param jpeg_streams Array of handles to JPEG streams.
    * @param batch_size Number of images in the batch.
    * @param decode_params Array of decoding parameters, one per image.
    * @param destinations Array of output images, one per image.
    * @return The status of the submit operation.
    */
   RocJpegStatus DecodeBatchedAsync(RocJpegStreamHandle *jpeg_streams, int batch_size, const RocJpegDecodeParams *decode_params, RocJpegImage *destinations);

   /**
    * @brief Synchronizes all pending asynchronous decodes for a batch and copies/converts the output.
    * @param destinations Array of destination images identifying the pending batch.
    * @param batch_size Number of images in the batch.
    * @return The status of the sync operation.
    */
   RocJpegStatus DecodeBatchedSync(RocJpegImage *destinations, int batch_size);

private:
   struct AsyncDecodeState {
      VASurfaceID surface_id;
      JpegStreamParameters jpeg_stream_params;
      RocJpegDecodeParams decode_params;
   };

   /**
    * @brief Initializes the HIP framework.
    * @param device_id The ID of the device to be used for HIP operations.
    * @return The status of the initialization process.
    */
   RocJpegStatus InitHIP(int device_id);

   /**
    * @brief Waits for a submitted surface and copies/converts the decoded output.
    */
   RocJpegStatus FinalizeDecode(VASurfaceID surface_id, const JpegStreamParameters *jpeg_stream_params, const RocJpegDecodeParams *decode_params, RocJpegImage *destination);
   RocJpegStatus FinalizeDecodeBatched(const VASurfaceID *surface_ids, const JpegStreamParameters *jpeg_stream_params, const RocJpegDecodeParams *decode_params, RocJpegImage *destinations, int batch_size);

   /**
    * @brief Retrieves the height of the chroma channel.
    * @param surface_format The surface format of the image.
    * @param picture_height The height of the picture.
    * @param chroma_height Reference to store the height of the chroma channel.
    * @return The status of the operation.
    */
   RocJpegStatus GetChromaHeight(uint32_t surface_format, uint16_t picture_height, uint16_t &chroma_height);

   /**
    * @brief Copies a channel from the HIP interop device memory to the destination image.
    * @param hip_interop The HIP interop device memory.
    * @param channel_width The width of the channel.
    * @param channel_height The height of the channel.
    * @param channel_index The index of the channel.
    * @param destination Pointer to the destination image.
    * @return The status of the operation.
    */
   RocJpegStatus CopyChannel(HipInteropDeviceMem& hip_interop, uint16_t channel_width, uint16_t channel_height, uint8_t channel_index, RocJpegImage *destination, const RocJpegDecodeParams *decode_params, bool is_roi_valid);

   /**
    * @brief Converts the image to RGB color space.
    * @param hip_interop The HIP interop device memory.
    * @param picture_width The width of the picture.
    * @param picture_height The height of the picture.
    * @param destination Pointer to the destination image.
    * @return The status of the operation.
    */
   RocJpegStatus ColorConvertToRGB(HipInteropDeviceMem& hip_interop, uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination, const RocJpegDecodeParams *decode_params, bool is_roi_valid);

   /**
    * @brief Converts the image to RGB planar color space.
    * @param hip_interop The HIP interop device memory.
    * @param picture_width The width of the picture.
    * @param picture_height The height of the picture.
    * @param destination Pointer to the destination image.
    * @return The status of the operation.
    */
   RocJpegStatus ColorConvertToRGBPlanar(HipInteropDeviceMem& hip_interop, uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination, const RocJpegDecodeParams *decode_params, bool is_roi_valid);

   /**
    * @brief Retrieves the output format for planar YUV images.
    * @param hip_interop The HIP interop device memory.
    * @param picture_width The width of the picture.
    * @param picture_height The height of the picture.
    * @param chroma_height The height of the chroma channel.
    * @param destination Pointer to the destination image.
    * @return The status of the operation.
    */
   RocJpegStatus GetPlanarYUVOutputFormat(HipInteropDeviceMem& hip_interop, uint32_t picture_width, uint32_t picture_height, uint16_t chroma_height, RocJpegImage *destination, const RocJpegDecodeParams *decode_params, bool is_roi_valid);

   /**
    * @brief Retrieves the output format for Y images.
    * @param hip_interop The HIP interop device memory.
    * @param picture_width The width of the picture.
    * @param picture_height The height of the picture.
    * @param destination Pointer to the destination image.
    * @return The status of the operation.
    */
   RocJpegStatus GetYOutputFormat(HipInteropDeviceMem& hip_interop, uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination, const RocJpegDecodeParams *decode_params, bool is_roi_valid);

   // Batched-path accumulators: mirror the per-image color-convert/output helpers above,
   // but instead of launching a kernel per image they append params to the per-kernel
   // scratch buffers below. Memcpy-only work (CopyChannel) is still issued inline.
   RocJpegStatus AccumulateColorConvertToRGB(HipInteropDeviceMem& hip_interop, uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination, const RocJpegDecodeParams *decode_params, bool is_roi_valid);
   RocJpegStatus AccumulateColorConvertToRGBPlanar(HipInteropDeviceMem& hip_interop, uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination, const RocJpegDecodeParams *decode_params, bool is_roi_valid);
   RocJpegStatus AccumulatePlanarYUVOutputFormat(HipInteropDeviceMem& hip_interop, uint32_t picture_width, uint32_t picture_height, uint16_t chroma_height, RocJpegImage *destination, const RocJpegDecodeParams *decode_params, bool is_roi_valid);
   RocJpegStatus AccumulateYOutputFormat(HipInteropDeviceMem& hip_interop, uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination, const RocJpegDecodeParams *decode_params, bool is_roi_valid);
   void ResetBatchedParams();
   RocJpegStatus LaunchBatchedParams();

   // Uploads one accumulated buffer (if non-empty) and issues its batched launch.
   template <typename T>
   RocJpegStatus LaunchBatchedBuffer(BatchedKernelParams<T> &buf,
       void (*launch)(hipStream_t, uint32_t, uint32_t, const T *, uint32_t)) {
       if (buf.Empty()) return ROCJPEG_STATUS_SUCCESS;
       T *dev = nullptr;
       CHECK_HIP(buf.Upload(hip_stream_, &dev));
       launch(hip_stream_, buf.MaxGx(), buf.MaxGy(), dev, buf.Count());
       return ROCJPEG_STATUS_SUCCESS;
   }

   int num_devices_; // Number of available devices
   int device_id_; // ID of the device to be used
   hipDeviceProp_t hip_dev_prop_; // HIP device properties
   hipStream_t hip_stream_; // HIP stream
   std::mutex mutex_; // Mutex for thread safety
   RocJpegBackend backend_; // RocJpeg backend
   RocJpegVappiDecoder jpeg_vaapi_decoder_; // RocJpeg VAAPI decoder object
   std::unordered_map<RocJpegImage*, AsyncDecodeState> pending_decodes_; // Map of pending asynchronous decodes keyed by destination

   // Per-kernel scratch buffers for the batched output path, one per batched kernel.
   // Filled per group in FinalizeDecodeBatched; each device buffer grows on demand.
   BatchedKernelParams<RGBAToRGBBatchParams>              b_rgba_rgb_;
   BatchedKernelParams<PackedYUYVToPlanarYUVBatchParams>  b_yuyv_yuvp_;
   BatchedKernelParams<InterleavedUVToPlanarUVBatchParams> b_nv12_uvp_;
   BatchedKernelParams<YFromPackedYUYVBatchParams>        b_yuyv_y_;
   // Unified YUV->RGB / YUV->RGB_PLANAR buffers: a single kernel each handles all of
   // NV12/YUV444/YUV440/YUYV/YUV400, selected per image by a surface-layout tag.
   BatchedKernelParams<YUVToRGBBatchParams>            b_yuv_rgb_;
   BatchedKernelParams<YUVToRGBPlanarBatchParams>      b_yuv_rgbp_;
};

#endif //ROC_JPEG_DECODER_H_