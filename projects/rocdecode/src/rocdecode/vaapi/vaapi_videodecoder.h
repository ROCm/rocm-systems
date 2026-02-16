/*
Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <cstring>
#include <mutex>
#include <algorithm>
#include <unordered_map>
#include <libdrm/amdgpu_drm.h>
#include <libdrm/amdgpu.h>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>
#include "../../commons.h"
#include "../../../api/rocdecode/rocdecode.h"

#define CHECK_HIP(call) {\
    hipError_t hip_status = call;\
    if (hip_status != hipSuccess) {\
        logger_.CriticalLog(MakeMsg("HIP failure: " + #call + " failed with 'status: " + STR(hipGetErrorName(hip_status)) + "' at " + __FILE__ + ":" + TOSTR(__LINE__)));\
        return ROCDEC_RUNTIME_ERROR;\
    }\
}

#define CHECK_VAAPI(call) {\
    VAStatus va_status = call;\
    if (va_status != VA_STATUS_SUCCESS) {\
        logger_.CriticalLog(MakeMsg("VAAPI failure: " + #call + " failed with 'status: " + TOSTR(va_status) + ": " + vaErrorStr(va_status) + "' at " + __FILE__ + ":" + TOSTR(__LINE__)));\
        return ROCDEC_RUNTIME_ERROR;\
    }\
}

#define INIT_SLICE_PARAM_LIST_NUM 16 // initial slice parameter buffer list size

typedef enum {
    kSpx = 0, // Single Partition Accelerator
    kDpx = 1, // Dual Partition Accelerator
    kTpx = 2, // Triple Partition Accelerator
    kQpx = 3, // Quad Partition Accelerator
    kCpx = 4, // Core Partition Accelerator
} ComputePartition;

typedef struct {
    int device_id;
    std::string gpu_uuid;
    int drm_fd;
    VADisplay va_display;
    hipDeviceProp_t hip_dev_prop;
    uint32_t num_dec_engines;
    int num_va_profiles;
    std::vector<VAProfile> va_profile_list; // supported profiles by the current GPU
    VAProfile va_profile; // current profile used
    VAConfigID va_config_id;
    bool config_attributes_probed;
    uint32_t rt_format_attrib;
    uint32_t output_format_mask;
    uint32_t max_width;
    uint32_t max_height;
    uint32_t min_width;
    uint32_t min_height;
} VaContextInfo;

class VaapiVideoDecoder {
public:
    VaapiVideoDecoder(RocDecoderCreateInfo &decoder_create_info);
    ~VaapiVideoDecoder();
    rocDecStatus InitializeDecoder();
    rocDecStatus SubmitDecode(RocdecPicParams *pPicParams);
    rocDecStatus GetDecodeStatus(int pic_idx, RocdecDecodeStatus* decode_status);
    rocDecStatus ExportSurface(int pic_idx, VADRMPRIMESurfaceDescriptor &va_drm_prime_surface_desc);
    rocDecStatus SyncSurface(int pic_idx);
    rocDecStatus ReconfigureDecoder(RocdecReconfigureDecoderInfo *reconfig_params);

private:
    RocDecoderCreateInfo decoder_create_info_;
    VADisplay va_display_;
    VAConfigAttrib va_config_attrib_;
    VAConfigID va_config_id_;
    VAProfile va_profile_;
    VAContextID va_context_id_;
    std::vector<VASurfaceID> va_surface_ids_;
    bool supports_modifiers_;

    VABufferID pic_params_buf_id_;
    VABufferID iq_matrix_buf_id_;
    std::vector<VABufferID> slice_params_buf_id_ = std::vector<VABufferID>(INIT_SLICE_PARAM_LIST_NUM, 0);
    uint32_t num_slices_;
    VABufferID slice_data_buf_id_;

    RocDecLogger logger_;

    bool IsCodecConfigSupported(int device_id, rocDecVideoCodec codec_type, rocDecVideoChromaFormat chroma_format, uint32_t bit_depth_minus8, rocDecVideoSurfaceFormat output_format);
    rocDecStatus CreateDecoderConfig();
    rocDecStatus CreateSurfaces();
    rocDecStatus CreateContext();
    rocDecStatus DestroyDataBuffers();
};

// The VaContext singleton class providing access to the the GPU VA services
class VaContext {
public:
    int num_devices_;
    std::vector<VaContextInfo> va_contexts_;

    static VaContext& GetInstance() {
        static VaContext instance;
        return instance;
    }
    rocDecStatus GetVaContext(int device_id, uint32_t *va_ctx_id);
    rocDecStatus GetVaDisplay(uint32_t va_ctx_id, VADisplay *va_display);
    rocDecStatus CheckDecCapForCodecType(RocdecDecodeCaps *dec_cap);

private:
    std::mutex mutex;
    /**
     * @brief A map that associates GPU UUIDs with their corresponding render node indices.
     * 
     * This unordered map uses GPU UUIDs as keys (std::string) and maps them to their 
     * respective render node indices (int). It provides a fast lookup mechanism to 
     * retrieve the render node index for a given GPU UUID.
     */
    std::unordered_map<std::string, int> gpu_uuids_to_render_nodes_map_;
    std::unordered_map<std::string, ComputePartition> gpu_uuids_to_compute_partition_map_;
    VaContext();
    VaContext(const VaContext&) = delete;
    VaContext& operator = (const VaContext) = delete;
    ~VaContext();

    RocDecLogger logger_;

    rocDecStatus InitHIP(int device_id, hipDeviceProp_t& hip_dev_prop);
    rocDecStatus InitVAAPI(int va_ctx_idx, std::string drm_node);
    void GetVisibleDevices(std::vector<int>& visible_devices_vetor);
    void GetDrmNodeOffset(std::string device_name, uint8_t device_id, std::vector<int>& visible_devices, ComputePartition current_compute_partition, int &offset);
    void GetGpuUuids();
};