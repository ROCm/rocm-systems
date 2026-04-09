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

#include "rocjpeg_vaapi_decoder.h"

/**
 * @brief Default constructor for RocJpegVaapiMemoryPool class.
 *
 * This constructor initializes the memory pool for different surface formats used in RocJpegVappiDecoder.
 * It creates an empty vector for each surface format and stores it in the mem_pool_ map.
 *
 * @param None
 * @return None
 */
RocJpegVaapiMemoryPool::RocJpegVaapiMemoryPool() {
    std::vector<uint32_t> surface_formats = {VA_FOURCC_RGBA, VA_FOURCC_RGBP, VA_FOURCC_444P, VA_FOURCC_422V, VA_FOURCC_YUY2, VA_FOURCC_NV12, VA_FOURCC_Y800};
    for (auto surface_format : surface_formats) {
        mem_pool_[surface_format] = std::vector<RocJpegVaapiMemPoolEntry>();
    }
    max_pool_size_ = 2;
}

/**
 * @brief Releases the resources used by the RocJpegVaapiMemoryPool.
 *
 * This function releases the VA-API contexts, surfaces, HIP device memory, and HIP external memory
 * associated with the memory pool. It iterates over each entry in the memory pool and checks if
 * the VA-API context ID, VA-API surface ID, HIP mapped device memory, or HIP external memory is
 * non-zero. If so, it destroys the corresponding resource using the appropriate API function.
 * Finally, it resets the HIP interop structure for each entry in the memory pool.
 */
void RocJpegVaapiMemoryPool::ReleaseResources() {
    VAStatus va_status;
    hipError_t hip_status;
    for (auto& pair : mem_pool_) {
        for (auto& entry : pair.second) {
            if (!entry.va_surface_ids.empty()) {
                va_status = vaDestroySurfaces(va_display_, entry.va_surface_ids.data(), entry.va_surface_ids.size());
                if (va_status != VA_STATUS_SUCCESS) {
                    ERR("ERROR: vaDestroySurfaces failed!");
                }
            }
            if (!entry.hip_interops.empty()) {
                for(auto& hip_interop_entry : entry.hip_interops) {
                    if (hip_interop_entry.hip_mapped_device_mem != nullptr) {
                        hip_status = hipFree(hip_interop_entry.hip_mapped_device_mem);
                        if (hip_status != hipSuccess) {
                            ERR("ERROR: hipFree failed!");
                        }
                    }
                    if (hip_interop_entry.hip_ext_mem != nullptr) {
                        hip_status = hipDestroyExternalMemory(hip_interop_entry.hip_ext_mem);
                        if (hip_status != hipSuccess) {
                            ERR("ERROR: hipDestroyExternalMemory failed!");
                        }
                    }
                }
            }
        }
    }
}

void RocJpegVaapiMemoryPool::SetPoolSize(uint32_t max_pool_size) {
    max_pool_size_ = max_pool_size;
}

void RocJpegVaapiMemoryPool::SetVaapiDisplay(const VADisplay& va_display) {
    va_display_ = va_display;
}

/**
 * @brief Retrieves the total size of the memory pool.
 *
 * This function iterates through the memory pool and sums up the sizes of all entries.
 *
 * @return The total size of the memory pool.
 */
size_t RocJpegVaapiMemoryPool::GetTotalMemPoolSize() const {
    size_t total_mem_pool_size = 0;
    for (const auto& pair : mem_pool_) {
        total_mem_pool_size += pair.second.size();
    }
    return total_mem_pool_size;
}

/**
 * @brief Deletes an idle entry from the memory pool.
 *
 * This function iterates through the memory pool and searches for an entry
 * with the status `kIdle`. If such an entry is found, it performs the following
 * cleanup operations:
 * - Destroys the VAAPI context if it exists.
 * - Destroys the VAAPI surfaces if they exist.
 * - Frees HIP mapped device memory and destroys HIP external memory if they exist.
 * - Resets the HIP interop entries.
 *
 * After performing the cleanup, the idle entry is removed from the memory pool.
 *
 * @return true if an idle entry was found and deleted, false otherwise.
 */
bool RocJpegVaapiMemoryPool::DeleteIdleEntry() {
    for (auto& pair : mem_pool_) {
        auto it = std::find_if(pair.second.begin(), pair.second.end(), [](const RocJpegVaapiMemPoolEntry& entry) {return entry.entry_status == kIdle;});
        if (it != pair.second.end()) {
            auto index = std::distance(pair.second.begin(), it);
            if (!pair.second[index].va_surface_ids.empty()) {
                CHECK_VAAPI(vaDestroySurfaces(va_display_, pair.second[index].va_surface_ids.data(), pair.second[index].va_surface_ids.size()));
                std::fill(pair.second[index].va_surface_ids.begin(), pair.second[index].va_surface_ids.end(), 0);
            }
            if (!pair.second[index].hip_interops.empty()) {
                for(auto& hip_interop_entry : pair.second[index].hip_interops) {
                    if (hip_interop_entry.hip_mapped_device_mem != nullptr)
                        CHECK_HIP(hipFree(hip_interop_entry.hip_mapped_device_mem));
                    if (hip_interop_entry.hip_ext_mem != nullptr)
                        CHECK_HIP(hipDestroyExternalMemory(hip_interop_entry.hip_ext_mem));
                    memset((void*)&hip_interop_entry, 0, sizeof(hip_interop_entry));
                }
            }
            pair.second.erase(it);
            return true;
        }
    }
    return false;
}

/**
 * @brief Adds a pool entry to the memory pool for a specific surface format.
 *
 * This function adds a pool entry to the memory pool for a specific surface format.
 * If the memory pool for the given surface format is not full, the new entry is added to the pool.
 * If the memory pool is full, the oldest entry is removed from the pool and replaced with the new entry.
 * If the removed entry has associated resources (VA context, VA surface, HIP memory), they are destroyed and freed.
 *
 * @param surface_format The surface format for which the pool entry is being added.
 * @param pool_entry The pool entry to be added.
 * @return The status of the operation. Returns ROCJPEG_STATUS_SUCCESS if the operation is successful.
 */
RocJpegStatus RocJpegVaapiMemoryPool::AddPoolEntry(uint32_t surface_format, const RocJpegVaapiMemPoolEntry& pool_entry) {
    size_t total_mem_pool_size = GetTotalMemPoolSize();
    auto& entries = mem_pool_[surface_format];
    if (total_mem_pool_size < max_pool_size_) {
        entries.push_back(pool_entry);
    } else {
        if (DeleteIdleEntry()) {
            entries.push_back(pool_entry);
        } else {
            ERR("cannot find an idle entry in the the memory pool!");
            return ROCJPEG_STATUS_INVALID_PARAMETER;
        }
    }
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Retrieves a `RocJpegVaapiMemPoolEntry` from the memory pool based on the specified surface format, image width, and image height.
 *
 * @param surface_format The surface pixel format of the entry to retrieve.
 * @param image_width The width of the image of the entry to retrieve.
 * @param image_height The height of the image of the entry to retrieve.
 * @param num_surfaces The number of surfaces of the entry to retrieve.
 * @return The matching `RocJpegVaapiMemPoolEntry` if found, or a default-initialized entry if not found.
 */
RocJpegVaapiMemPoolEntry RocJpegVaapiMemoryPool::GetEntry(uint32_t surface_format, uint32_t image_width, uint32_t image_height, uint32_t num_surfaces) {
    for (auto& entry : mem_pool_[surface_format]) {
        if (entry.image_width >= image_width && entry.image_height >= image_height && entry.va_surface_ids.size() == num_surfaces && entry.entry_status == kIdle) {
            entry.entry_status = kBusy;
            return entry;
        }
    }
    return {0, 0, kIdle, {}, {}};
}

bool RocJpegVaapiMemoryPool::FindSurfaceId(VASurfaceID surface_id) {
    for (auto& pair : mem_pool_) {
        for (auto& entry : pair.second) {
            if (std::find(entry.va_surface_ids.begin(), entry.va_surface_ids.end(), surface_id) != entry.va_surface_ids.end()) {
                return true;
            }
        }
    }
    return false;
}


/**
 * @brief Retrieves the HipInteropDeviceMem associated with a given VASurfaceID from the memory pool.
 *
 * This function searches the memory pool for the entry that matches the provided VASurfaceID.
 * If a matching entry is found and the associated HipInteropDeviceMem is not already initialized,
 * it initializes the HipInteropDeviceMem by exporting the VASurfaceID as a DRM prime surface handle,
 * importing it as an external memory object, and getting the mapped buffer for the external memory.
 * The function then updates the HipInteropDeviceMem with the surface format, width, height, offsets,
 * pitches, and number of layers from the exported surface descriptor.
 *
 * @param surface_id The VASurfaceID to retrieve the HipInteropDeviceMem for.
 * @param hip_interop [out] The retrieved HipInteropDeviceMem.
 * @return RocJpegStatus Returns ROCJPEG_STATUS_SUCCESS if the HipInteropDeviceMem is successfully retrieved,
 *         ROCJPEG_STATUS_INVALID_PARAMETER if the requested surface_id is not found in the memory pool.
 */
RocJpegStatus RocJpegVaapiMemoryPool::GetHipInteropMem(VASurfaceID surface_id, HipInteropDeviceMem& hip_interop) {
    for (auto& pair : mem_pool_) {
        auto& entries = pair.second;
        auto it = std::find_if(entries.begin(), entries.end(),
                              [surface_id](const RocJpegVaapiMemPoolEntry& entry){return std::find(entry.va_surface_ids.begin(), entry.va_surface_ids.end(), surface_id) != entry.va_surface_ids.end();});
        if (it != entries.end()) {
            auto idx = std::distance(it->va_surface_ids.begin(), std::find(it->va_surface_ids.begin(), it->va_surface_ids.end(), surface_id));
            #if defined(ENABLE_INTEROP_BUFFER_REUSE) && ENABLE_INTEROP_BUFFER_REUSE > 0
                // Perform the VA-API/HIP interop once for each surface in the memory pool and store it for reuse.
                if (it->hip_interops[idx].hip_mapped_device_mem == nullptr) {
                    VADRMPRIMESurfaceDescriptor va_drm_prime_surface_desc = {};
                    CHECK_VAAPI(vaExportSurfaceHandle(va_display_, surface_id, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                        VA_EXPORT_SURFACE_READ_ONLY | VA_EXPORT_SURFACE_SEPARATE_LAYERS,
                        &va_drm_prime_surface_desc));

                    hipExternalMemoryHandleDesc external_mem_handle_desc = {};
                    hipExternalMemoryBufferDesc external_mem_buffer_desc = {};
                    external_mem_handle_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
                    external_mem_handle_desc.handle.fd = va_drm_prime_surface_desc.objects[0].fd;
                    external_mem_handle_desc.size = va_drm_prime_surface_desc.objects[0].size;

                    CHECK_HIP(hipImportExternalMemory(&it->hip_interops[idx].hip_ext_mem, &external_mem_handle_desc));
                    external_mem_buffer_desc.size = va_drm_prime_surface_desc.objects[0].size;
                    CHECK_HIP(hipExternalMemoryGetMappedBuffer((void**)&it->hip_interops[idx].hip_mapped_device_mem, it->hip_interops[idx].hip_ext_mem, &external_mem_buffer_desc));

                    uint32_t surface_format = va_drm_prime_surface_desc.fourcc;
                    // Workaround Mesa <= 24.3 returning non-standard VA fourcc
                    if (surface_format == VA_FOURCC('Y', 'U', 'Y', 'V'))
                        surface_format = VA_FOURCC_YUY2;

                    it->hip_interops[idx].surface_format = surface_format;
                    it->hip_interops[idx].width = va_drm_prime_surface_desc.width;
                    it->hip_interops[idx].height = va_drm_prime_surface_desc.height;
                    it->hip_interops[idx].size = va_drm_prime_surface_desc.objects[0].size;
                    it->hip_interops[idx].offset[0] = va_drm_prime_surface_desc.layers[0].offset[0];
                    it->hip_interops[idx].offset[1] = va_drm_prime_surface_desc.layers[1].offset[0];
                    it->hip_interops[idx].offset[2] = va_drm_prime_surface_desc.layers[2].offset[0];
                    it->hip_interops[idx].pitch[0] = va_drm_prime_surface_desc.layers[0].pitch[0];
                    it->hip_interops[idx].pitch[1] = va_drm_prime_surface_desc.layers[1].pitch[0];
                    it->hip_interops[idx].pitch[2] = va_drm_prime_surface_desc.layers[2].pitch[0];
                    it->hip_interops[idx].num_layers = va_drm_prime_surface_desc.num_layers;

                    for (uint32_t i = 0; i < va_drm_prime_surface_desc.num_objects; ++i) {
                        close(va_drm_prime_surface_desc.objects[i].fd);
                    }
                }
            #else
                if (it->hip_interops[idx].hip_mapped_device_mem != nullptr) {
                    CHECK_HIP(hipFree(it->hip_interops[idx].hip_mapped_device_mem));
                    if (it->hip_interops[idx].hip_ext_mem != nullptr) {
                        CHECK_HIP(hipDestroyExternalMemory(it->hip_interops[idx].hip_ext_mem));
                    }
                }
                VADRMPRIMESurfaceDescriptor va_drm_prime_surface_desc = {};
                CHECK_VAAPI(vaExportSurfaceHandle(va_display_, surface_id, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                    VA_EXPORT_SURFACE_READ_ONLY | VA_EXPORT_SURFACE_SEPARATE_LAYERS,
                    &va_drm_prime_surface_desc));

                hipExternalMemoryHandleDesc external_mem_handle_desc = {};
                hipExternalMemoryBufferDesc external_mem_buffer_desc = {};
                external_mem_handle_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
                external_mem_handle_desc.handle.fd = va_drm_prime_surface_desc.objects[0].fd;
                external_mem_handle_desc.size = va_drm_prime_surface_desc.objects[0].size;

                CHECK_HIP(hipImportExternalMemory(&it->hip_interops[idx].hip_ext_mem, &external_mem_handle_desc));
                external_mem_buffer_desc.size = va_drm_prime_surface_desc.objects[0].size;
                CHECK_HIP(hipExternalMemoryGetMappedBuffer((void**)&it->hip_interops[idx].hip_mapped_device_mem, it->hip_interops[idx].hip_ext_mem, &external_mem_buffer_desc));

                uint32_t surface_format = va_drm_prime_surface_desc.fourcc;
                // Workaround Mesa <= 24.3 returning non-standard VA fourcc
                if (surface_format == VA_FOURCC('Y', 'U', 'Y', 'V'))
                    surface_format = VA_FOURCC_YUY2;

                it->hip_interops[idx].surface_format = surface_format;
                it->hip_interops[idx].width = va_drm_prime_surface_desc.width;
                it->hip_interops[idx].height = va_drm_prime_surface_desc.height;
                it->hip_interops[idx].size = va_drm_prime_surface_desc.objects[0].size;
                it->hip_interops[idx].offset[0] = va_drm_prime_surface_desc.layers[0].offset[0];
                it->hip_interops[idx].offset[1] = va_drm_prime_surface_desc.layers[1].offset[0];
                it->hip_interops[idx].offset[2] = va_drm_prime_surface_desc.layers[2].offset[0];
                it->hip_interops[idx].pitch[0] = va_drm_prime_surface_desc.layers[0].pitch[0];
                it->hip_interops[idx].pitch[1] = va_drm_prime_surface_desc.layers[1].pitch[0];
                it->hip_interops[idx].pitch[2] = va_drm_prime_surface_desc.layers[2].pitch[0];
                it->hip_interops[idx].num_layers = va_drm_prime_surface_desc.num_layers;

                for (uint32_t i = 0; i < va_drm_prime_surface_desc.num_objects; ++i) {
                    close(va_drm_prime_surface_desc.objects[i].fd);
                }
            #endif
            hip_interop = it->hip_interops[idx];
            return ROCJPEG_STATUS_SUCCESS;
        }
    }
    // it shouldn't reach here unless the requested surface_id is not in the memory pool.
    ERR("the surface_id: " + TOSTR(surface_id) + " was not found in the memory pool!");
    return ROCJPEG_STATUS_INVALID_PARAMETER;
}

bool RocJpegVaapiMemoryPool::SetSurfaceAsIdle(VASurfaceID surface_id) {
    for (auto& pair : mem_pool_) {
        for (auto& entry : pair.second) {
            if (std::find(entry.va_surface_ids.begin(), entry.va_surface_ids.end(), surface_id) != entry.va_surface_ids.end()) {
                entry.entry_status = kIdle;
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief Constructs a RocJpegVappiDecoder object.
 *
 * This constructor initializes a RocJpegVappiDecoder object with the specified device ID and default values for other member variables.
 *
 * @param device_id The ID of the device to be used for decoding.
 */
RocJpegVappiDecoder::RocJpegVappiDecoder(int device_id) : device_id_{device_id}, drm_fd_{-1}, min_picture_width_{64}, min_picture_height_{64},
    max_picture_width_{4096}, max_picture_height_{4096}, default_surface_width_{3840}, default_surface_height_{2160}, supports_modifiers_{false}, va_display_{0}, va_config_attrib_{{}}, va_config_id_{0}, va_profile_{VAProfileJPEGBaseline},
    vaapi_mem_pool_(std::make_unique<RocJpegVaapiMemoryPool>()), current_vcn_jpeg_spec_{}, va_picture_parameter_buf_id_{0}, va_quantization_matrix_buf_id_{0}, va_huffmantable_buf_id_{0},
    va_slice_param_buf_id_{0}, va_slice_data_buf_id_{0} {};

/**
 * @brief Destructor for the RocJpegVappiDecoder class.
 *
 * This destructor is responsible for cleaning up the resources used by the RocJpegVappiDecoder object.
 * It closes the DRM file descriptor, releases the VAAPI memory pool resources, destroys the VAAPI data buffers,
 * destroys the VAAPI configuration, and terminates the VAAPI display.
 *
 * @note If any of the cleanup operations fail, an error message will be printed.
 */
RocJpegVappiDecoder::~RocJpegVappiDecoder() {
    if (drm_fd_ != -1) {
        close(drm_fd_);
    }
    if (va_display_) {
        vaapi_mem_pool_->ReleaseResources();
        RocJpegStatus rocjpeg_status = DestroyDataBuffers();
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
            ERR("Error: Failed to destroy VAAPI buffer");
        }
        VAStatus va_status;
        if (va_context_id_ != 0) {
            va_status = vaDestroyContext(va_display_, va_context_id_);
            if (va_status != VA_STATUS_SUCCESS) {
                ERR("ERROR: vaDestroyContext failed!");
            }
        }
        if (va_config_id_) {
            va_status = vaDestroyConfig(va_display_, va_config_id_);
            if (va_status != VA_STATUS_SUCCESS) {
                ERR("ERROR: vaDestroyConfig failed!");
            }
        }
        va_status = vaTerminate(va_display_);
        if (va_status != VA_STATUS_SUCCESS) {
            ERR("ERROR: vaTerminate failed!");
        }

    }
}

/**
 * @brief Initializes the VAAPI decoder for RocJpeg.
 *
 * This function initializes the VAAPI decoder for RocJpeg by setting the device ID, GCN architecture name,
 * and other necessary parameters. It also sets up the VAAPI display and creates the decoder configuration.
 *
 * @param device_name The name of the device.
 * @param device_id The ID of the device.
 * @param gpu_uuid The UUID of the GPU.
 * @return The status of the initialization process.
 */
RocJpegStatus RocJpegVappiDecoder::InitializeDecoder(std::string device_name, int device_id, std::string& gpu_uuid) {
    device_id_ = device_id;
    std::vector<int> visible_devices;
    GetVisibleDevices(visible_devices);
    GetGpuUuids();

    int offset = 0;
    ComputePartition current_compute_partition = (gpu_uuids_to_compute_partition_map_.find(gpu_uuid) != gpu_uuids_to_compute_partition_map_.end()) ? gpu_uuids_to_compute_partition_map_[gpu_uuid] : kSpx;
    GetDrmNodeOffset(device_name, device_id_, visible_devices, current_compute_partition, offset);

    std::string drm_node = "/dev/dri/renderD";
    int render_node_id = (gpu_uuids_to_render_nodes_map_.find(gpu_uuid) != gpu_uuids_to_render_nodes_map_.end()) ? gpu_uuids_to_render_nodes_map_[gpu_uuid] : 128;
    drm_node += std::to_string(render_node_id + offset);

    CHECK_ROCJPEG(InitVAAPI(drm_node));
    CHECK_ROCJPEG(CreateDecoderConfig());
    CHECK_ROCJPEG(CreateDecoderContext());

    vaapi_mem_pool_->SetVaapiDisplay(va_display_);

    GetNumJpegCores();
    vaapi_mem_pool_->SetPoolSize(5 * current_vcn_jpeg_spec_.num_jpeg_cores + 1);

    return ROCJPEG_STATUS_SUCCESS;
}
/**
 * @brief Retrieves the number of JPEG cores available on the AMD GPU and updates the decoder's capabilities.
 *
 * This function initializes the AMD GPU device, queries the number of JPEG cores available, and updates
 * the current VCN JPEG specifications based on the number of cores. If the number of JPEG cores is 8 or more,
 * it sets the capabilities to support ROI decode and conversion to RGB.
 *
 * @note If the initialization of the AMD GPU device fails or querying the number of JPEG cores fails,
 *       appropriate error messages are logged.
 */
void RocJpegVappiDecoder::GetNumJpegCores() {
    amdgpu_device_handle dev_handle;
    uint32_t major_version = 0, minor_version = 0;
    uint32_t num_jpeg_cores = 0;
    int error_code = 0;
    const char *enable_vcn_hw_csc_str = std::getenv("ROCJPEG_ENABLE_VCN_HW_CSC");
    bool enable_vcn_hw_csc = (enable_vcn_hw_csc_str != nullptr && strcmp(enable_vcn_hw_csc_str, "1") == 0);
    if (amdgpu_device_initialize(drm_fd_, &major_version, &minor_version, &dev_handle)) {
        ERR("amdgpu_device_initialize failed!");
        return;
    }
    error_code = amdgpu_query_hw_ip_count(dev_handle, AMDGPU_HW_IP_VCN_JPEG, &num_jpeg_cores);
    if (!error_code) {
        current_vcn_jpeg_spec_.num_jpeg_cores = num_jpeg_cores;
        // Set the capabilities based on the number of JPEG cores
        current_vcn_jpeg_spec_.can_roi_decode = (num_jpeg_cores >= 8);
        current_vcn_jpeg_spec_.can_convert_to_rgb = (num_jpeg_cores >= 8) && enable_vcn_hw_csc;
    } else {
        ERR("Failed to get the number of jpeg cores.");
    }
    amdgpu_device_deinitialize(dev_handle);
}
/**
 * @brief Initializes the VAAPI decoder.
 *
 * This function initializes the VAAPI decoder by opening the DRM node, creating the va_display,
 * setting the info callback, and initializing the va_display.
 *
 * @param drm_node The path to the DRM node.
 * @return The status of the initialization process.
 *         - ROCJPEG_STATUS_SUCCESS if the initialization is successful.
 *         - ROCJPEG_STATUS_NOT_INITIALIZED if the initialization fails.
 */
RocJpegStatus RocJpegVappiDecoder::InitVAAPI(std::string drm_node) {
    drm_fd_ = open(drm_node.c_str(), O_RDWR);
    if (drm_fd_ < 0) {
        ERR("ERROR: failed to open drm node " + drm_node);
        return ROCJPEG_STATUS_NOT_INITIALIZED;
    }
    va_display_ = vaGetDisplayDRM(drm_fd_);
    if (!va_display_) {
        ERR("ERROR: failed to create va_display!");
        return ROCJPEG_STATUS_NOT_INITIALIZED;
    }
    vaSetInfoCallback(va_display_, NULL, NULL);
    int major_version = 0, minor_version = 0;
    CHECK_VAAPI(vaInitialize(va_display_, &major_version, &minor_version))
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Creates the decoder configuration for the RocJpegVappiDecoder.
 *
 * This function creates the decoder configuration by querying the VA API for supported entrypoints
 * and checking if the hardware JPEG decoder is supported. If the hardware JPEG decoder is supported,
 * it retrieves the maximum picture width and height attributes from the VA API and creates the configuration.
 *
 * @return The status of the decoder configuration creation.
 *         - ROCJPEG_STATUS_SUCCESS if the configuration is created successfully.
 *         - ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED if the hardware JPEG decoder is not supported.
 */
RocJpegStatus RocJpegVappiDecoder::CreateDecoderConfig() {
    int max_num_entrypoints = vaMaxNumEntrypoints(va_display_);
    std::vector<VAEntrypoint> jpeg_entrypoint_list;
    jpeg_entrypoint_list.resize(max_num_entrypoints);
    int num_entrypoints = 0;
    CHECK_VAAPI(vaQueryConfigEntrypoints(va_display_, va_profile_, jpeg_entrypoint_list.data(), &num_entrypoints));
    bool hw_jpeg_decoder_supported = false;
    if (num_entrypoints > 0) {
        for (auto entry_point : jpeg_entrypoint_list) {
            if (entry_point == VAEntrypointVLD) {
                hw_jpeg_decoder_supported = true;
                break;
            }
        }
    } else {
        return ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED;
    }

    if (hw_jpeg_decoder_supported) {
        va_config_attrib_.resize(3);
        va_config_attrib_[0].type = VAConfigAttribRTFormat;
        va_config_attrib_[1].type = VAConfigAttribMaxPictureWidth;
        va_config_attrib_[2].type = VAConfigAttribMaxPictureHeight;
        CHECK_VAAPI(vaGetConfigAttributes(va_display_, va_profile_, VAEntrypointVLD, va_config_attrib_.data(), va_config_attrib_.size()));
        CHECK_VAAPI(vaCreateConfig(va_display_, va_profile_, VAEntrypointVLD, &va_config_attrib_[0], 1, &va_config_id_));
        if (va_config_attrib_[1].value != VA_ATTRIB_NOT_SUPPORTED) {
            max_picture_width_ = va_config_attrib_[1].value;
        }
        if (va_config_attrib_[2].value != VA_ATTRIB_NOT_SUPPORTED) {
            max_picture_height_ = va_config_attrib_[2].value;
        }
        unsigned int num_attribs = 0;
        CHECK_VAAPI(vaQuerySurfaceAttributes(va_display_, va_config_id_, nullptr, &num_attribs));
        std::vector<VASurfaceAttrib> attribs(num_attribs);
        CHECK_VAAPI(vaQuerySurfaceAttributes(va_display_, va_config_id_, attribs.data(), &num_attribs));
        for (auto attrib : attribs) {
            if (attrib.type == VASurfaceAttribDRMFormatModifiers) {
                supports_modifiers_ = true;
                break;
            }
        }
        return ROCJPEG_STATUS_SUCCESS;
    } else {
        return ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED;
    }
}

/**
 * @brief Creates the decoder context for the VAAPI-based JPEG decoder.
 *
 * This function initializes the VAAPI decoder context.
 *
 * @return RocJpegStatus indicating the success or failure of the context creation.
 */
RocJpegStatus RocJpegVappiDecoder::CreateDecoderContext() {

    CHECK_VAAPI(vaCreateContext(va_display_, va_config_id_, min_picture_width_, min_picture_height_, VA_PROGRESSIVE, nullptr, 0, &va_context_id_));

    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Destroys the data buffers used by the RocJpegVappiDecoder.
 *
 * This function destroys the data buffers used by the RocJpegVappiDecoder, including the picture parameter buffer,
 * quantization matrix buffer, Huffman table buffer, slice parameter buffer, and slice data buffer.
 *
 * @return The status of the operation. Returns ROCJPEG_STATUS_SUCCESS if the data buffers were successfully destroyed.
 */
RocJpegStatus RocJpegVappiDecoder::DestroyDataBuffers() {
    if (va_picture_parameter_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, va_picture_parameter_buf_id_));
        va_picture_parameter_buf_id_ = 0;
    }
    if (va_quantization_matrix_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, va_quantization_matrix_buf_id_));
        va_quantization_matrix_buf_id_ = 0;
    }
    if (va_huffmantable_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, va_huffmantable_buf_id_));
        va_huffmantable_buf_id_ = 0;
    }
    if (va_slice_param_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, va_slice_param_buf_id_));
        va_slice_param_buf_id_ = 0;
    }
    if (va_slice_data_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, va_slice_data_buf_id_));
        va_slice_data_buf_id_ = 0;
    }
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Submits a JPEG decode operation to the VAAPI decoder.
 *
 * This function submits a JPEG decode operation to the VAAPI decoder using the provided JPEG stream parameters.
 * It checks for invalid parameters and unsupported image resolutions before proceeding with the decode operation.
 * The output format is determined based on the requested format and the capabilities of the hardware decoder.
 *
 * @param jpeg_stream_params The JPEG stream parameters for the decode operation.
 * @param surface_id [out] The ID of the output surface where the decoded image will be stored.
 * @param decode_params Additional parameters for the decode operation.
 * @return The status of the decode operation.
 *         - ROCJPEG_STATUS_SUCCESS if the decode operation was successful.
 *         - ROCJPEG_STATUS_INVALID_PARAMETER if the provided parameters are invalid.
 *         - ROCJPEG_STATUS_JPEG_NOT_SUPPORTED if the JPEG image resolution or chroma subsampling is not supported.
 */
RocJpegStatus RocJpegVappiDecoder::SubmitDecode(const JpegStreamParameters *jpeg_stream_params, uint32_t &surface_id, const RocJpegDecodeParams *decode_params) {
    if (jpeg_stream_params == nullptr || decode_params == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }

    if (sizeof(jpeg_stream_params->picture_parameter_buffer) != sizeof(VAPictureParameterBufferJPEGBaseline) ||
        sizeof(jpeg_stream_params->quantization_matrix_buffer) != sizeof(VAIQMatrixBufferJPEGBaseline) ||
        sizeof(jpeg_stream_params->huffman_table_buffer) != sizeof(VAHuffmanTableBufferJPEGBaseline) ||
        sizeof(jpeg_stream_params->slice_parameter_buffer) != sizeof(VASliceParameterBufferJPEGBaseline)) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }

    if (jpeg_stream_params->picture_parameter_buffer.picture_width < min_picture_width_ ||
        jpeg_stream_params->picture_parameter_buffer.picture_height < min_picture_height_ ||
        jpeg_stream_params->picture_parameter_buffer.picture_width > max_picture_width_ ||
        jpeg_stream_params->picture_parameter_buffer.picture_height > max_picture_height_) {
            ERR("The JPEG image resolution is not supported!");
            return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
        }

    uint32_t surface_format;
    std::vector<VASurfaceAttrib> surface_attribs;
    VASurfaceAttrib surface_attrib;
    surface_attrib.type = VASurfaceAttribPixelFormat;
    surface_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrib.value.type = VAGenericValueTypeInteger;

    // If RGB output format is requested, and the HW JPEG decoder has a built-in format conversion,
    // set the RGB surface format and attributes to obtain the RGB output directly from the JPEG HW decoder.
    // otherwise set the appropriate surface format and attributes based on the chroma subsampling of the image.
    if ((decode_params->output_format == ROCJPEG_OUTPUT_RGB || decode_params->output_format == ROCJPEG_OUTPUT_RGB_PLANAR) && current_vcn_jpeg_spec_.can_convert_to_rgb && jpeg_stream_params->chroma_subsampling != CSS_440) {
        if (decode_params->output_format == ROCJPEG_OUTPUT_RGB) {
            surface_format = VA_RT_FORMAT_RGB32;
            surface_attrib.value.value.i = VA_FOURCC_RGBA;
        } else if (decode_params->output_format == ROCJPEG_OUTPUT_RGB_PLANAR) {
            surface_format = VA_RT_FORMAT_RGBP;
            surface_attrib.value.value.i = VA_FOURCC_RGBP;
        }
    } else {
        switch (jpeg_stream_params->chroma_subsampling) {
            case CSS_444:
                surface_format = VA_RT_FORMAT_YUV444;
                surface_attrib.value.value.i = VA_FOURCC_444P;
                break;
            case CSS_440:
                surface_format = VA_RT_FORMAT_YUV422;
                surface_attrib.value.value.i = VA_FOURCC_422V;
                break;
            case CSS_422:
                surface_format = VA_RT_FORMAT_YUV422;
                surface_attrib.value.value.i = VA_FOURCC_YUY2;
                break;
            case CSS_420:
                surface_format = VA_RT_FORMAT_YUV420;
                surface_attrib.value.value.i = VA_FOURCC_NV12;
                break;
            case CSS_400:
                surface_format = VA_RT_FORMAT_YUV400;
                surface_attrib.value.value.i = VA_FOURCC_Y800;
                break;
            default:
                ERR("ERROR: The chroma subsampling is not supported by the VCN hardware!");
                return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
                break;
        }
    }
    uint32_t surface_pixel_format = static_cast<uint32_t>(surface_attrib.value.value.i);
    surface_attribs.push_back(surface_attrib);

    uint64_t mod_linear = 0;
    VADRMFormatModifierList modifier_list = {
        .num_modifiers = 1,
        .modifiers = &mod_linear,
    };
    if (supports_modifiers_) {
        surface_attrib.type = VASurfaceAttribDRMFormatModifiers;
        surface_attrib.value.type = VAGenericValueTypePointer;
        surface_attrib.value.value.p = &modifier_list;
        surface_attribs.push_back(surface_attrib);
    }

    // if the HW JPEG decoder has a built-in ROI-decode capability then fill the requested crop rectangle to the picture parameter buffer
    void *picture_parameter_buffer = (void*)&jpeg_stream_params->picture_parameter_buffer;
    if (current_vcn_jpeg_spec_.can_roi_decode) {
        uint32_t roi_width;
        uint32_t roi_height;
        roi_width = decode_params->crop_rectangle.right - decode_params->crop_rectangle.left;
        roi_height = decode_params->crop_rectangle.bottom - decode_params->crop_rectangle.top;
        if (roi_width > 0 && roi_height > 0 && roi_width <= jpeg_stream_params->picture_parameter_buffer.picture_width && roi_height <= jpeg_stream_params->picture_parameter_buffer.picture_height) {
#if VA_CHECK_VERSION(1, 21, 0)
            reinterpret_cast<VAPictureParameterBufferJPEGBaseline*>(picture_parameter_buffer)->crop_rectangle.x = decode_params->crop_rectangle.left;
            reinterpret_cast<VAPictureParameterBufferJPEGBaseline*>(picture_parameter_buffer)->crop_rectangle.y = decode_params->crop_rectangle.top;
            reinterpret_cast<VAPictureParameterBufferJPEGBaseline*>(picture_parameter_buffer)->crop_rectangle.width = roi_width;
            reinterpret_cast<VAPictureParameterBufferJPEGBaseline*>(picture_parameter_buffer)->crop_rectangle.height = roi_height;
#else
            reinterpret_cast<VAPictureParameterBufferJPEGBaseline*>(picture_parameter_buffer)->va_reserved[0] = decode_params->crop_rectangle.top << 16 | decode_params->crop_rectangle.left;
            reinterpret_cast<VAPictureParameterBufferJPEGBaseline*>(picture_parameter_buffer)->va_reserved[1] = roi_height << 16 | roi_width;
#endif
        }
    }

    RocJpegVaapiMemPoolEntry mem_pool_entry = vaapi_mem_pool_->GetEntry(surface_pixel_format, jpeg_stream_params->picture_parameter_buffer.picture_width, jpeg_stream_params->picture_parameter_buffer.picture_height, 1);
    if (mem_pool_entry.va_surface_ids.empty()) {
        mem_pool_entry.va_surface_ids.resize(1);
        uint32_t surface_width = (jpeg_stream_params->picture_parameter_buffer.picture_width > default_surface_width_) ? jpeg_stream_params->picture_parameter_buffer.picture_width : default_surface_width_;
        uint32_t surface_height = (jpeg_stream_params->picture_parameter_buffer.picture_height > default_surface_height_) ? jpeg_stream_params->picture_parameter_buffer.picture_height : default_surface_height_;
        CHECK_VAAPI(vaCreateSurfaces(va_display_, surface_format, surface_width, surface_height, mem_pool_entry.va_surface_ids.data(), 1, surface_attribs.data(), surface_attribs.size()));
        mem_pool_entry.image_width = surface_width;
        mem_pool_entry.image_height = surface_height;
        mem_pool_entry.hip_interops.resize(1, HipInteropDeviceMem());
        surface_id = mem_pool_entry.va_surface_ids[0];
        mem_pool_entry.entry_status = kBusy;
        CHECK_ROCJPEG(vaapi_mem_pool_->AddPoolEntry(surface_pixel_format, mem_pool_entry));
    } else {
        surface_id = mem_pool_entry.va_surface_ids[0];
    }

    CHECK_ROCJPEG(DestroyDataBuffers());

    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VAPictureParameterBufferType, sizeof(VAPictureParameterBufferJPEGBaseline), 1, picture_parameter_buffer, &va_picture_parameter_buf_id_));
    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VAIQMatrixBufferType, sizeof(VAIQMatrixBufferJPEGBaseline), 1, (void *)&jpeg_stream_params->quantization_matrix_buffer, &va_quantization_matrix_buf_id_));
    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VAHuffmanTableBufferType, sizeof(VAHuffmanTableBufferJPEGBaseline), 1, (void *)&jpeg_stream_params->huffman_table_buffer, &va_huffmantable_buf_id_));
    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VASliceParameterBufferType, sizeof(VASliceParameterBufferJPEGBaseline), 1, (void *)&jpeg_stream_params->slice_parameter_buffer, &va_slice_param_buf_id_));
    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VASliceDataBufferType, jpeg_stream_params->slice_parameter_buffer.slice_data_size, 1, (void *)jpeg_stream_params->slice_data_buffer, &va_slice_data_buf_id_));

    CHECK_VAAPI(vaBeginPicture(va_display_, va_context_id_,  surface_id));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &va_picture_parameter_buf_id_, 1));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &va_quantization_matrix_buf_id_, 1));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &va_huffmantable_buf_id_, 1));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &va_slice_param_buf_id_, 1));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &va_slice_data_buf_id_, 1));
    CHECK_VAAPI(vaEndPicture(va_display_, va_context_id_));

    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus RocJpegVappiDecoder::SubmitDecodeBatched(JpegStreamParameters *jpeg_streams_params, int batch_size, const RocJpegDecodeParams *decode_params, uint32_t *surface_ids) {
    if (jpeg_streams_params == nullptr || decode_params == nullptr || surface_ids == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }

    // Group the JPEG streams in the jpeg_streams_params array based on their chroma subsampling, width, and height.
    // Store the groups in an unordered map, where the key is a JpegStreamKey struct and the value is a vector of integers
    // representing the indices of the JPEG streams in the batch.
    std::unordered_map<JpegStreamKey, std::vector<int>> jpeg_stream_groups;
    for (int i = 0; i < batch_size; i++) {
        if (sizeof(jpeg_streams_params[i].picture_parameter_buffer) != sizeof(VAPictureParameterBufferJPEGBaseline) ||
            sizeof(jpeg_streams_params[i].quantization_matrix_buffer) != sizeof(VAIQMatrixBufferJPEGBaseline) ||
            sizeof(jpeg_streams_params[i].huffman_table_buffer) != sizeof(VAHuffmanTableBufferJPEGBaseline) ||
            sizeof(jpeg_streams_params[i].slice_parameter_buffer) != sizeof(VASliceParameterBufferJPEGBaseline)) {
            return ROCJPEG_STATUS_INVALID_PARAMETER;
        }
        JpegStreamKey jpeg_stream_key = {};
        jpeg_stream_key.width = jpeg_streams_params[i].picture_parameter_buffer.picture_width;
        jpeg_stream_key.height = jpeg_streams_params[i].picture_parameter_buffer.picture_height;
        if (jpeg_stream_key.width < min_picture_width_ ||
            jpeg_stream_key.height < min_picture_height_ ||
            jpeg_stream_key.width > max_picture_width_ ||
            jpeg_stream_key.height > max_picture_height_) {
                ERR("The JPEG image resolution is not supported!");
                return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
            }

        if ((decode_params[i].output_format == ROCJPEG_OUTPUT_RGB || decode_params[i].output_format == ROCJPEG_OUTPUT_RGB_PLANAR) && current_vcn_jpeg_spec_.can_convert_to_rgb && jpeg_streams_params[i].chroma_subsampling != CSS_440) {
            if (decode_params[i].output_format == ROCJPEG_OUTPUT_RGB) {
                jpeg_stream_key.surface_format = VA_RT_FORMAT_RGB32;
                jpeg_stream_key.pixel_format = VA_FOURCC_RGBA;
            } else if (decode_params[i].output_format == ROCJPEG_OUTPUT_RGB_PLANAR) {
                jpeg_stream_key.surface_format = VA_RT_FORMAT_RGBP;
                jpeg_stream_key.pixel_format = VA_FOURCC_RGBP;
            }
        } else {
            switch (jpeg_streams_params[i].chroma_subsampling) {
                case CSS_444:
                    jpeg_stream_key.surface_format = VA_RT_FORMAT_YUV444;
                    jpeg_stream_key.pixel_format = VA_FOURCC_444P;
                    break;
                case CSS_440:
                    jpeg_stream_key.surface_format = VA_RT_FORMAT_YUV422;
                    jpeg_stream_key.pixel_format = VA_FOURCC_422V;
                    break;
                case CSS_422:
                    jpeg_stream_key.surface_format = VA_RT_FORMAT_YUV422;
                    jpeg_stream_key.pixel_format = VA_FOURCC_YUY2;
                    break;
                case CSS_420:
                    jpeg_stream_key.surface_format = VA_RT_FORMAT_YUV420;
                    jpeg_stream_key.pixel_format = VA_FOURCC_NV12;
                    break;
                case CSS_400:
                    jpeg_stream_key.surface_format = VA_RT_FORMAT_YUV400;
                    jpeg_stream_key.pixel_format = VA_FOURCC_Y800;
                    break;
                default:
                    ERR("ERROR: The chroma subsampling is not supported by the VCN hardware!");
                    return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
                    break;
            }
        }
        jpeg_stream_groups[jpeg_stream_key].push_back(i);
    }

    uint32_t surface_format;
    std::vector<VASurfaceAttrib> surface_attribs(2);
    surface_attribs[0].type = VASurfaceAttribPixelFormat;
    surface_attribs[0].flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attribs[0].value.type = VAGenericValueTypeInteger;

    uint64_t mod_linear = 0;
    VADRMFormatModifierList modifier_list = {
        .num_modifiers = 1,
        .modifiers = &mod_linear,
    };
    if (supports_modifiers_) {
        surface_attribs[1].type = VASurfaceAttribDRMFormatModifiers;
        surface_attribs[1].flags = VA_SURFACE_ATTRIB_SETTABLE;
        surface_attribs[1].value.type = VAGenericValueTypePointer;
        surface_attribs[1].value.value.p = &modifier_list;
    }

    uint32_t roi_width;
    uint32_t roi_height;

    // Iterate through all entries of jpeg_stream_groups.
    // Check if there is a matching entry in the memory pool.
    // If not, allocate surfaces and create a context for each group.
    // Submit the JPEG streams to the hardware for decoding.
    for (const auto& group : jpeg_stream_groups) {
        const JpegStreamKey& key = group.first;
        const std::vector<int>& indices = group.second;

        surface_format = key.surface_format;
        surface_attribs[0].value.value.i = key.pixel_format;

        RocJpegVaapiMemPoolEntry mem_pool_entry = vaapi_mem_pool_->GetEntry(key.pixel_format, key.width, key.height, indices.size());
        if (mem_pool_entry.va_surface_ids.empty()) {
            mem_pool_entry.va_surface_ids.resize(indices.size());
            uint32_t surface_width = (key.width > default_surface_width_) ? key.width : default_surface_width_;
            uint32_t surface_height = (key.height > default_surface_height_) ? key.height : default_surface_height_;
            CHECK_VAAPI(vaCreateSurfaces(va_display_, surface_format, surface_width, surface_height, mem_pool_entry.va_surface_ids.data(), mem_pool_entry.va_surface_ids.size(), surface_attribs.data(), supports_modifiers_ ? 2 : 1));
            mem_pool_entry.image_width = surface_width;
            mem_pool_entry.image_height = surface_height;
            for (size_t i = 0; i < mem_pool_entry.va_surface_ids.size(); i++) {
                surface_ids[indices[i]] = mem_pool_entry.va_surface_ids[i];
            }
            mem_pool_entry.hip_interops.resize(indices.size(), HipInteropDeviceMem());
            mem_pool_entry.entry_status = kBusy;
            CHECK_ROCJPEG(vaapi_mem_pool_->AddPoolEntry(key.pixel_format, mem_pool_entry));
        } else {
            for (size_t i = 0; i < mem_pool_entry.va_surface_ids.size(); i++) {
                surface_ids[indices[i]] = mem_pool_entry.va_surface_ids[i];
            }
        }

        for (int idx : indices) {
            // if the HW JPEG decoder has a built-in ROI-decode capability then fill the requested crop rectangle to the picture parameter buffer
            void* picture_parameter_buffer = &jpeg_streams_params[idx].picture_parameter_buffer;
            roi_width = decode_params[idx].crop_rectangle.right - decode_params[idx].crop_rectangle.left;
            roi_height = decode_params[idx].crop_rectangle.bottom - decode_params[idx].crop_rectangle.top;
            if (current_vcn_jpeg_spec_.can_roi_decode && roi_width > 0 && roi_height > 0 &&
                roi_width <= jpeg_streams_params[idx].picture_parameter_buffer.picture_width &&
                roi_height <= jpeg_streams_params[idx].picture_parameter_buffer.picture_height) {
#if VA_CHECK_VERSION(1, 21, 0)
            reinterpret_cast<VAPictureParameterBufferJPEGBaseline*>(picture_parameter_buffer)->crop_rectangle.x = decode_params[idx].crop_rectangle.left;
            reinterpret_cast<VAPictureParameterBufferJPEGBaseline*>(picture_parameter_buffer)->crop_rectangle.y = decode_params[idx].crop_rectangle.top;
            reinterpret_cast<VAPictureParameterBufferJPEGBaseline*>(picture_parameter_buffer)->crop_rectangle.width = roi_width;
            reinterpret_cast<VAPictureParameterBufferJPEGBaseline*>(picture_parameter_buffer)->crop_rectangle.height = roi_height;
#else
            reinterpret_cast<VAPictureParameterBufferJPEGBaseline*>(picture_parameter_buffer)->va_reserved[0] = decode_params[idx].crop_rectangle.top << 16 | decode_params[idx].crop_rectangle.left;
            reinterpret_cast<VAPictureParameterBufferJPEGBaseline*>(picture_parameter_buffer)->va_reserved[1] = roi_height << 16 | roi_width;
#endif
            }
            CHECK_ROCJPEG(DestroyDataBuffers());
            CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VAPictureParameterBufferType, sizeof(VAPictureParameterBufferJPEGBaseline), 1, picture_parameter_buffer, &va_picture_parameter_buf_id_));
            CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VAIQMatrixBufferType, sizeof(VAIQMatrixBufferJPEGBaseline), 1, (void *)&jpeg_streams_params[idx].quantization_matrix_buffer, &va_quantization_matrix_buf_id_));
            CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VAHuffmanTableBufferType, sizeof(VAHuffmanTableBufferJPEGBaseline), 1, (void *)&jpeg_streams_params[idx].huffman_table_buffer, &va_huffmantable_buf_id_));
            CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VASliceParameterBufferType, sizeof(VASliceParameterBufferJPEGBaseline), 1, (void *)&jpeg_streams_params[idx].slice_parameter_buffer, &va_slice_param_buf_id_));
            CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VASliceDataBufferType, jpeg_streams_params[idx].slice_parameter_buffer.slice_data_size, 1, (void *)jpeg_streams_params[idx].slice_data_buffer, &va_slice_data_buf_id_));

            CHECK_VAAPI(vaBeginPicture(va_display_, va_context_id_, surface_ids[idx]));
            CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &va_picture_parameter_buf_id_, 1));
            CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &va_quantization_matrix_buf_id_, 1));
            CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &va_huffmantable_buf_id_, 1));
            CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &va_slice_param_buf_id_, 1));
            CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &va_slice_data_buf_id_, 1));
            CHECK_VAAPI(vaEndPicture(va_display_, va_context_id_));
        }
    }


    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Synchronizes the specified VASurfaceID.
 *
 * This function synchronizes the specified VASurfaceID by querying its status and waiting until it becomes ready.
 * If the surface ID is not found in the VAAPI memory pool, it returns ROCJPEG_STATUS_INVALID_PARAMETER.
 * If any error occurs during synchronization, it returns ROCJPEG_STATUS_RUNTIME_ERROR.
 *
 * @param surface_id The VASurfaceID to synchronize.
 * @return The status of the synchronization operation.
 */
RocJpegStatus RocJpegVappiDecoder::SyncSurface(VASurfaceID surface_id) {
    if (!vaapi_mem_pool_->FindSurfaceId(surface_id)) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    CHECK_VAAPI(vaSyncSurface(va_display_, surface_id));
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Retrieves the HipInteropDeviceMem associated with the specified VASurfaceID.
 *
 * This function retrieves the HipInteropDeviceMem associated with the specified VASurfaceID
 * from the vaapi_mem_pool_ and stores it in the provided `hip_interop` parameter.
 *
 * @param surface_id The VASurfaceID of the surface to retrieve the HipInteropDeviceMem for.
 * @param hip_interop The reference to a HipInteropDeviceMem object where the retrieved memory will be stored.
 * @return The RocJpegStatus indicating the success or failure of the operation.
 */
RocJpegStatus RocJpegVappiDecoder::GetHipInteropMem(VASurfaceID surface_id, HipInteropDeviceMem& hip_interop) {
    return vaapi_mem_pool_->GetHipInteropMem(surface_id, hip_interop);
}

/**
 * @brief Retrieves the visible devices for the RocJpegVappiDecoder.
 *
 * This function retrieves the visible devices for the RocJpegVappiDecoder by reading the value of the environment variable "HIP_VISIBLE_DEVICES".
 * The visible devices are stored in the provided vector `visible_devices_vector`.
 *
 * @param visible_devices_vector The vector to store the visible devices.
 */
void RocJpegVappiDecoder::GetVisibleDevices(std::vector<int>& visible_devices_vetor) {
    // First, check if the ROCR_VISIBLE_DEVICES environment variable is present
    char *visible_devices = std::getenv("ROCR_VISIBLE_DEVICES");
    // If ROCR_VISIBLE_DEVICES is not present, check if HIP_VISIBLE_DEVICES is present
    if (visible_devices == nullptr) {
        visible_devices = std::getenv("HIP_VISIBLE_DEVICES");
    }
    if (visible_devices != nullptr) {
        char *token = std::strtok(visible_devices,",");
        while (token != nullptr) {
            visible_devices_vetor.push_back(std::atoi(token));
            token = std::strtok(nullptr,",");
        }
    std::sort(visible_devices_vetor.begin(), visible_devices_vetor.end());
    }
}

/**
 * @brief Calculates the offset for the DRM node based on the device name, device ID, visible devices,
 *        current compute partitions, and the selected compute partition.
 *
 * @param device_name The name of the device.
 * @param device_id The ID of the device.
 * @param visible_devices A vector containing the IDs of the visible devices.
 * @param current_compute_partition the current compute partition.
 * @param offset The calculated offset for the DRM node.
 */
void RocJpegVappiDecoder::GetDrmNodeOffset(std::string device_name, uint8_t device_id, std::vector<int>& visible_devices,
                                                   ComputePartition current_compute_partition,
                                                   int &offset) {
    switch (current_compute_partition) {
        case kSpx:
            offset = 0;
            break;
        case kDpx:
            if (device_id < visible_devices.size()) {
                offset = (visible_devices[device_id] % 2);
            } else {
                offset = (device_id % 2);
            }
            break;
        case kTpx:
            if (device_id < visible_devices.size()) {
                offset = (visible_devices[device_id] % 3);
            } else {
                offset = (device_id % 3);
            }
            break;
        case kQpx:
            if (device_id < visible_devices.size()) {
                offset = (visible_devices[device_id] % 4);
            } else {
                offset = (device_id % 4);
            }
            break;
        case kCpx:
            // Note: The MI300 series share the same gfx_arch_name (gfx942).
            // Therefore, we cannot use gfx942 to distinguish between MI300X, MI300A etc.
            // Instead, use the device name to identify MI300A, etc.
            std::string mi300a = "MI300A";
            size_t found_mi300a = device_name.find(mi300a);
            if (found_mi300a != std::string::npos) {
                if (device_id < visible_devices.size()) {
                    offset = (visible_devices[device_id] % 6);
                } else {
                    offset = (device_id % 6);
                }
            } else {
                if (device_id < visible_devices.size()) {
                    offset = (visible_devices[device_id] % 8);
                } else {
                    offset = (device_id % 8);
                }
            }
            break;
    }
}

/**
 * @brief Sets a VASurfaceID as idle.
 *
 * This function sets the specified VASurfaceID as idle in the RocJpegVappiDecoder's vaapi_mem_pool.
 * If the surface cannot be set as idle, it returns ROCJPEG_STATUS_INVALID_PARAMETER.
 *
 * @param surface_id The VASurfaceID to set as idle.
 * @return RocJpegStatus The status of the operation. Returns ROCJPEG_STATUS_SUCCESS if successful,
 *         or ROCJPEG_STATUS_INVALID_PARAMETER if the surface cannot be set as idle.
 */
RocJpegStatus RocJpegVappiDecoder::SetSurfaceAsIdle(VASurfaceID surface_id) {
    if (!vaapi_mem_pool_->SetSurfaceAsIdle(surface_id)) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Retrieves GPU UUIDs and maps them to render node IDs and compute partitions.
 *
 * This function iterates through all render nodes in the /dev/dri directory,
 * extracts the render node ID from the filename, and then reads the unique GPU
 * UUID from the corresponding sysfs path. It maps each unique GPU UUID to its
 * corresponding render node ID and stores this mapping in the gpu_uuids_to_render_nodes_map_.
 * Additionally, it maps the unique GPU UUID to the current compute partition if available.
 */
void RocJpegVappiDecoder::GetGpuUuids() {
    std::string dri_path = "/dev/dri";
    DIR* dir = opendir(dri_path.c_str());
    if (dir) {
        struct dirent* entry;
        // Iterate through all render nodes
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            // Check if the file name starts with "renderD"
            if (filename.find("renderD") == 0) {
                // Extract the integer part from the render node name (e.g., 128 from renderD128)
                int render_id = std::stoi(filename.substr(7));
                std::string sys_device_path = "/sys/class/drm/" + filename + "/device";
                struct stat info;
                if (stat(sys_device_path.c_str(), &info) == 0) {
                    std::string unique_id_path = sys_device_path + "/unique_id";
                    std::ifstream unique_id_file(unique_id_path);
                    std::string unique_id;
                    if (unique_id_file.is_open() && std::getline(unique_id_file, unique_id)) {
                        if (!unique_id.empty()) {
                            // Map the unique GPU UUID to the render node ID
                            gpu_uuids_to_render_nodes_map_[unique_id] = render_id;
                        }
                    }
                    unique_id_file.close();
                    if (!unique_id.empty()) {
                        unique_id_path = sys_device_path + "/current_compute_partition";
                        std::ifstream partition_file(unique_id_path);
                        std::string partition;
                        ComputePartition current_compute_partition = kSpx;
                        if (partition_file.is_open() && std::getline(partition_file, partition)) {
                            if (!partition.empty()) {
                                if (partition.compare("SPX") == 0 || partition.compare("spx") == 0) {
                                    current_compute_partition = kSpx;
                                } else if (partition.compare("DPX") == 0 || partition.compare("dpx") == 0) {
                                    current_compute_partition = kDpx;
                                } else if (partition.compare("TPX") == 0 || partition.compare("tpx") == 0) {
                                    current_compute_partition = kTpx;
                                } else if (partition.compare("QPX") == 0 || partition.compare("qpx") == 0) {
                                    current_compute_partition = kQpx;
                                } else if (partition.compare("CPX") == 0 || partition.compare("cpx") == 0) {
                                    current_compute_partition = kCpx;
                                }
                                // Map the unique GPU UUID to the compute partition
                                gpu_uuids_to_compute_partition_map_[unique_id] = current_compute_partition;
                            }
                        }
                        partition_file.close();
                    }
                }
            }
        }
        closedir(dir);
    }
}