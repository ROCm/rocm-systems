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

#ifndef ROC_JPEG_VAAPI_DECODER_H_
#define ROC_JPEG_VAAPI_DECODER_H_

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unordered_map>
#include <memory>
#include <functional>
#include <libdrm/amdgpu.h>
#include <libdrm/amdgpu_drm.h>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>
#include "rocjpeg_commons.h"
#include "rocjpeg_parser.h"
#include "../api/rocjpeg/rocjpeg.h"

/**
 * @brief Enumeration representing the compute partition for the MI300+ family of GPUs.
 */
typedef enum {
    kSpx = 0, /**< Single Partition Accelerator */
    kDpx = 1, /**< Dual Partition Accelerator */
    kTpx = 2, /**< Triple Partition Accelerator */
    kQpx = 3, /**< Quad Partition Accelerator */
    kCpx = 4, /**< Core Partition Accelerator */
} ComputePartition;

/**
 * @brief Structure representing the specifications of a VCN JPEG decoder.
 *
 * This structure contains information about the VCN JPEG decoder, including the number of JPEG cores,
 * whether it can convert to RGB, and whether it supports ROI (Region of Interest) decoding.
 */
typedef struct {
    uint32_t num_jpeg_cores; /**< Number of JPEG cores in the VCN JPEG decoder. */
    bool can_convert_to_rgb; /**< Flag indicating whether the VCN JPEG decoder can convert to RGB. */
    bool can_roi_decode; /**< Flag indicating whether the VCN JPEG decoder supports ROI decoding. */
} VcnJpegSpec;

/**
 * @brief Structure representing the HIP interop device memory.
 *
 * This structure holds information related to the HIP-VAAPI interop device memory.
 * It includes the HIP external memory interface, mapped device memory for the YUV plane,
 * pixel format fourcc of the whole surface, width and height of the surface in pixels,
 * offset and pitch of each plane, and the number of layers making up the surface.
 */
struct HipInteropDeviceMem {
    hipExternalMemory_t hip_ext_mem; /**< Interface to the vaapi-hip interop */
    uint8_t* hip_mapped_device_mem; /**< Mapped device memory for the YUV plane */
    uint32_t surface_format; /**< Pixel format fourcc of the whole surface */
    uint32_t width; /**< Width of the surface in pixels. */
    uint32_t height; /**< Height of the surface in pixels. */
    uint32_t size; /**< Size of the surface in pixels. */
    uint32_t offset[3]; /**< Offset of each plane */
    uint32_t pitch[3]; /**< Pitch of each plane */
    uint32_t num_layers; /**< Number of layers making up the surface */
};

/**
 * @brief Defines the enumeration MemPoolEntryStatus.
 */
typedef enum {
    kIdle = 0,
    kBusy = 1,
} MemPoolEntryStatus;

/**
 * @struct RocJpegVaapiMemPoolEntry
 * @brief Structure representing an entry in the RocJpegVaapiMemPool.
 *
 * This structure holds information about a memory pool entry used by the RocJpegVaapiDecoder.
 * It contains the image width and height, the entry status, an array of VA surface IDs,
 * and an array of HipInteropDeviceMem objects.
 */
struct RocJpegVaapiMemPoolEntry {
    uint32_t image_width;
    uint32_t image_height;
    MemPoolEntryStatus entry_status;
    std::vector<VASurfaceID> va_surface_ids;
    std::vector<HipInteropDeviceMem> hip_interops;
};

/**
 * @class RocJpegVaapiMemoryPool
 * @brief A class that represents a memory pool for VAAPI surfaces used by the RocJpegVappiDecoder.
 *
 * The RocJpegVaapiMemoryPool class provides methods to manage and allocate memory resources for VAAPI surfaces.
 * It allows setting the pool size, associating a VADisplay, finding surface IDs, getting pool entries, adding pool entries,
 * and retrieving HipInterop memory for a specific surface ID.
 */
class RocJpegVaapiMemoryPool {
    public:
        /**
         * @brief Default constructor for RocJpegVaapiMemoryPool.
         */
        RocJpegVaapiMemoryPool();

        /**
         * @brief Releases all the resources associated with the memory pool.
         */
        void ReleaseResources();

        /**
         * @brief Sets the maximum size of the memory pool.
         * @param max_pool_size The maximum size of the memory pool.
         */
        void SetPoolSize(uint32_t max_pool_size);

        /**
         * @brief Sets the VADisplay for the memory pool.
         * @param va_display The VADisplay to be set.
         */
        void SetVaapiDisplay(const VADisplay& va_display);

        /**
         * @brief Finds a surface ID in the memory pool.
         * @param surface_id The surface ID to find.
         * @return True if the surface ID is found, false otherwise.
         */
        bool FindSurfaceId(VASurfaceID surface_id);

        /**
         * @brief Gets a pool entry based on the surface format, image width, and image height.
         * @param surface_format The surface format of the pool entry.
         * @param image_width The image width of the pool entry.
         * @param image_height The image height of the pool entry.
         * @param num_surfaces The number of surfaces of the entry to retrieve.
         * @return The RocJpegVaapiMemPoolEntry object if found, otherwise a default-constructed object.
         */
        RocJpegVaapiMemPoolEntry GetEntry(uint32_t surface_format, uint32_t image_width, uint32_t image_height, uint32_t num_surfaces);

        /**
         * @brief Adds a pool entry to the memory pool.
         * @param surface_format The surface format of the pool entry.
         * @param pool_entry The RocJpegVaapiMemPoolEntry to be added.
         * @return The status of the operation.
         */
        RocJpegStatus AddPoolEntry(uint32_t surface_format, const RocJpegVaapiMemPoolEntry& pool_entry);

        /**
         * @brief Retrieves HipInterop memory for a specific surface ID.
         * @param surface_id The surface ID to retrieve HipInterop memory for.
         * @param hip_interop The HipInteropDeviceMem object to store the retrieved memory.
         * @return The status of the operation.
         */
        RocJpegStatus GetHipInteropMem(VASurfaceID surface_id, HipInteropDeviceMem& hip_interop);

        /**
         * @brief Sets a VASurfaceID as idle.
         *
         * This function sets the specified VASurfaceID as idle, indicating that it is available for reuse.
         *
         * @param surface_id The VASurfaceID to set as idle.
         * @return true if the VASurfaceID was successfully set as idle, false otherwise.
         */
        bool SetSurfaceAsIdle(VASurfaceID surface_id);

    private:
        VADisplay va_display_; // The VADisplay associated with the memory pool.
        uint32_t max_pool_size_; // The maximum pool size of the memory pool (mem_pool_) per entry.
        std::unordered_map<uint32_t, std::vector<RocJpegVaapiMemPoolEntry>> mem_pool_; // The memory pool.
        /**
         * @brief Retrieves the total size of the memory pool.
         *
         * @return The total size of the memory pool in bytes.
         */
        size_t GetTotalMemPoolSize() const;
        /**
         * @brief  Deletes an idle entry from the memory pool.
         *
         * This function is responsible for removing an idle entry from the memory pool.
         * It ensures that resources associated with the idle entry are properly released.
         *
         * @return true if the idle entry was successfully deleted, false otherwise.
         */
        bool DeleteIdleEntry();
};

/**
 * @brief Structure representing the key for a JPEG stream.
 *
 * This structure contains information about the surface format, pixel format, width, and height
 * of a JPEG stream. It is used for comparing two JpegStreamKey objects for equality.
 */
struct JpegStreamKey {
    uint32_t surface_format; /**< The surface format of the JPEG stream. */
    uint32_t pixel_format; /**< The pixel format of the JPEG stream. */
    uint32_t width; /**< The width of the JPEG stream. */
    uint32_t height; /**< The height of the JPEG stream. */

    /**
     * @brief Equality operator for comparing two JpegStreamKey objects.
     *
     * @param other The JpegStreamKey object to compare with.
     * @return true if the two objects are equal, false otherwise.
     */
    bool operator==(const JpegStreamKey& other) const {
        return surface_format == other.surface_format &&
               pixel_format == other.pixel_format &&
               width == other.width &&
               height == other.height;
    }
};


/**
 * @brief Specialization of the std::hash template for JpegStreamKey.
 *
 * This struct provides a hash function for the JpegStreamKey struct, which is used as a key in hash-based containers.
 * It calculates the hash value based on the surface_format, pixel_format, width, and height members of the JpegStreamKey struct.
 */
template <>
struct std::hash<JpegStreamKey> {
    /**
     * @brief Calculates the hash value for a given JpegStreamKey object.
     *
     * @param k The JpegStreamKey object to calculate the hash value for.
     * @return The calculated hash value.
     */
    std::size_t operator()(const JpegStreamKey& k) const {
        size_t result = std::hash<int>()(k.surface_format);
        result ^= std::hash<int>()(k.pixel_format) << 1;
        result ^= std::hash<uint32_t>()(k.width) << 1;
        result ^= std::hash<uint32_t>()(k.height) << 1;
        return result;
    }
};

/**
 * @brief The RocJpegVappiDecoder class represents a VAAPI-based JPEG decoder.
 */
class RocJpegVappiDecoder {
public:
    /**
     * @brief Constructs a RocJpegVappiDecoder object.
     * @param device_id The ID of the device to use for decoding (default is 0).
     */
    RocJpegVappiDecoder(int device_id = 0);

    /**
     * @brief Destroys the RocJpegVappiDecoder object.
     */
    ~RocJpegVappiDecoder();

    /**
     * @brief Initializes the decoder with the specified device, GCN architecture, and device ID.
     * @param device_name The name of the device.
     * @param device_id The ID of the device.
     * @param gpu_uuid The UUID of the GPU.
     * @return The status of the initialization.
     */
    RocJpegStatus InitializeDecoder(std::string device_name, int device_id, std::string& gpu_uuid);

    /**
     * @brief Submits a JPEG stream for decoding.
     * @param jpeg_stream_params The parameters of the JPEG stream.
     * @param surface_id The ID of the output surface.
     *  @param decode_params Additional parameters for the decode operation.
     * @return The status of the decoding operation.
     */
    RocJpegStatus SubmitDecode(const JpegStreamParameters *jpeg_stream_params, uint32_t &surface_id, const RocJpegDecodeParams *decode_params);

    /**
     * @brief Waits for the decoding operation to complete.
     * @param surface_id The ID of the output surface.
     * @return The status of the synchronization operation.
     */
    RocJpegStatus SyncSurface(VASurfaceID surface_id);

    /**
     * @brief Retrieves the HIP interop memory associated with the specified surface.
     * @param surface_id The ID of the surface.
     * @param hip_interop The HIP interop memory object to be filled.
     * @return The status of the retrieval operation.
     */
    RocJpegStatus GetHipInteropMem(VASurfaceID surface_id, HipInteropDeviceMem& hip_interop);

    /**
     * Submits a batch of JPEG streams for decoding using the VAAPI decoder.
     *
     * @param jpeg_streams_params An array of the JPEG streams parameters to be decoded.
     * @param batch_size The number of JPEG streams in the batch.
     * @param decode_params The decoding parameters for the VAAPI decoder.
     * @param surface_ids An array to store the surface IDs of the decoded frames.
     * @return The status of the decoding operation.
     */
    RocJpegStatus SubmitDecodeBatched(JpegStreamParameters *jpeg_streams_params, int batch_size, const RocJpegDecodeParams *decode_params, uint32_t *surface_ids);

    /**
     * @brief Returns the current VCN JPEG specification.
     * @return The current VCN JPEG specification.
     */
    const VcnJpegSpec& GetCurrentVcnJpegSpec() const {return current_vcn_jpeg_spec_;}

    /**
     * Sets the specified VASurfaceID as idle.
     *
     * @param surface_id The VASurfaceID to set as idle.
     * @return The status of the operation.
     */
    RocJpegStatus SetSurfaceAsIdle(VASurfaceID surface_id);
private:
    int device_id_; // The ID of the device
    int drm_fd_; // The file descriptor for the DRM device
    uint32_t min_picture_width_; // The minimum width of the picture
    uint32_t min_picture_height_; // The minimum height of the picture
    uint32_t max_picture_width_; // The maximum width of the picture
    uint32_t max_picture_height_; // The maximum height of the picture
    uint32_t default_surface_width_; // The default width of the surface
    uint32_t default_surface_height_; // The default height of the surface
    bool supports_modifiers_; // DRM format modifiers support
    VADisplay va_display_; // The VAAPI display
    VAContextID va_context_id_; // The VAAPI context ID
    std::vector<VAConfigAttrib> va_config_attrib_; // The VAAPI configuration attributes
    VAConfigID va_config_id_; // The VAAPI configuration ID
    VAProfile va_profile_; // The VAAPI profile
    std::unique_ptr<RocJpegVaapiMemoryPool> vaapi_mem_pool_; // The VAAPI memory pool
    VcnJpegSpec current_vcn_jpeg_spec_; // The current VCN JPEG specification
    VABufferID va_picture_parameter_buf_id_; // The VAAPI picture parameter buffer ID
    VABufferID va_quantization_matrix_buf_id_; // The VAAPI quantization matrix buffer ID
    VABufferID va_huffmantable_buf_id_; // The VAAPI Huffman table buffer ID
    VABufferID va_slice_param_buf_id_; // The VAAPI slice parameter buffer ID
    VABufferID va_slice_data_buf_id_; // The VAAPI slice data buffer ID
    /**
     * @brief A map that associates GPU UUIDs with their corresponding render node indices.
     * 
     * This unordered map uses GPU UUIDs as keys (std::string) and maps them to their 
     * respective render node indices (int). It provides a fast lookup mechanism to 
     * retrieve the render node index for a given GPU UUID.
     */
    std::unordered_map<std::string, int> gpu_uuids_to_render_nodes_map_;

    /**
     * @brief A map that associates GPU UUIDs with their corresponding compute partitions.
     *
     * This unordered map uses GPU UUIDs as keys (represented as strings) and maps them to
     * ComputePartition objects. It allows for efficient lookup and management of compute
     * partitions based on the unique identifiers of GPUs.
     */
    std::unordered_map<std::string, ComputePartition> gpu_uuids_to_compute_partition_map_;
    /**
     * @brief Initializes the VAAPI with the specified DRM node.
     * @param drm_node The DRM node to use for VAAPI initialization.
     * @return The status of the VAAPI initialization.
     */
    RocJpegStatus InitVAAPI(std::string drm_node);

    /**
     * @brief Creates the decoder configuration.
     * @return The status of the configuration creation.
     */
    RocJpegStatus CreateDecoderConfig();

    /**
     * @brief Creates the decoder context.
     *
     * This function initializes and sets up the necessary context for decoding
     * JPEG images using the VA-API.
     *
     * @return RocJpegStatus indicating the success or failure of the context creation.
     */
    RocJpegStatus CreateDecoderContext();

    /**
     * @brief Destroys the data buffers.
     * @return The status of the buffer destruction.
     */
    RocJpegStatus DestroyDataBuffers();

    /**
     * @brief Retrieves the visible devices.
     * @param visible_devices The vector to store the visible devices.
     */
    void GetVisibleDevices(std::vector<int>& visible_devices);

    /**
     * @brief Retrieves the DRM node offset.
     * @param device_name The name of the device.
     * @param device_id The ID of the device.
     * @param visible_devices The vector of visible devices.
     * @param current_compute_partition The current compute partition.
     * @param offset The offset of the DRM node.
     */
    void GetDrmNodeOffset(std::string device_name, uint8_t device_id, std::vector<int>& visible_devices,
                                    ComputePartition current_compute_partitions,
                                    int &offset);
    /**
     * @brief Retrieves GPU UUIDs and maps them to render node IDs.
    */
    void GetGpuUuids();

    /**
     * @brief Retrieves the number of JPEG cores available.
     *
     * This function is used to determine the number of JPEG decoding cores
     * that are available for use.
     */
    void GetNumJpegCores();
};

#endif // ROC_JPEG_VAAPI_DECODER_H_