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

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>

#include "vaapi_loader.h"
#include "../../commons.h"
#include "../../../api/rocencode/rocencode.h"

// Compute partition modes read from sysfs (current_compute_partition).
typedef enum {
    kSpx = 0,  // Single Partition Accelerator
    kDpx = 1,  // Dual Partition Accelerator
    kTpx = 2,  // Triple Partition Accelerator
    kQpx = 3,  // Quad Partition Accelerator
    kCpx = 4,  // Core Partition Accelerator
} ComputePartition;

// Codecs rocEncode can be asked about.  Kept separate from the public
// ROCENC_CODEC enum so that the VA layer never has to know about it.
enum class RocEncCodec {
    kH264 = 0,
    kHevc = 1,
    kAv1  = 2,
};

const char* RocEncCodecName(RocEncCodec codec);

/**
 * @brief Everything the capability surface can learn about one codec on one
 *        device.
 *
 * Filled exclusively from VAEntrypointEncSlice config attributes and, where VA
 * has nothing to say, from AMDGPU_INFO_VIDEO_CAPS.  It is never derived from a
 * GPU name, a gfx target, or the VCN encode-ring count: MI300 reports
 * num_enc_rings = 1 and has no encoder at all.
 *
 * `supported == false` is the normal, non-exceptional answer for a decode-only
 * part (MI300-class) and for Navi 24 (VCN 3.0.33).
 */
struct VaEncodeCaps {
    bool     supported = false;          // VAEntrypointEncSlice exists for at least one profile of this codec
    bool     kernel_caps_valid = false;  // AMDGPU_INFO_VIDEO_CAPS reported a valid encode entry

    // Union over every supported profile of this codec.
    uint32_t rt_format_mask = 0;         // VAConfigAttribRTFormat
    uint32_t rate_control_mask = 0;      // VAConfigAttribRateControl
    uint32_t packed_headers = 0;         // VAConfigAttribEncPackedHeaders
    uint32_t max_ref_l0 = 0;             // VAConfigAttribEncMaxRefFrames, low 16 bits
    uint32_t max_ref_l1 = 0;             // VAConfigAttribEncMaxRefFrames, high 16 bits
    uint32_t max_slices = 0;             // VAConfigAttribEncMaxSlices
    uint32_t quality_range = 0;          // VAConfigAttribEncQualityRange
    uint32_t intra_refresh = 0;          // VAConfigAttribEncIntraRefresh
    uint32_t roi_attrib = 0;             // VAConfigAttribEncROI (raw)
    uint32_t slice_structure = 0;        // VAConfigAttribEncSliceStructure
    uint32_t rate_control_ext = 0;       // VAConfigAttribEncRateControlExt (raw)
    uint32_t max_temporal_layers = 0;    // derived from rate_control_ext
    uint32_t profile_mask = 0;           // bit i set when candidate profile i has VAEntrypointEncSlice
    bool     interlaced = false;         // VAConfigAttribEncInterlaced != 0/NOT_SUPPORTED
    bool     supports_10bit = false;     // RTFormat YUV420_10
    bool     supports_yuv444 = false;    // RTFormat YUV444

    // Surface geometry limits from vaQuerySurfaceAttributes on an EncSlice config.
    uint32_t va_min_width = 0;
    uint32_t va_min_height = 0;
    uint32_t va_max_width = 0;
    uint32_t va_max_height = 0;

    // AMDGPU_INFO_VIDEO_CAPS (encode sub-query).  Zero when unavailable.
    uint32_t kernel_max_width = 0;
    uint32_t kernel_max_height = 0;
    uint32_t kernel_max_pixels_per_frame = 0;
    uint32_t kernel_max_level = 0;

    // Number of VCN instances.  Informational only; never used to decide
    // whether the device can encode.
    uint32_t num_enc_engines = 0;
};

/**
 * @brief Identity of a HIP device, supplied by the HIP layer.
 *
 * VaContext matches HIP ordinals to DRM render nodes with this and never calls
 * HIP itself -- that keeps the "exactly one layer touches HIP" invariant
 * (RocEncoder fills this in).
 */
struct HipDeviceIdentity {
    int         device_id = 0;
    std::string gpu_uuid;     // raw bytes of hipDeviceProp_t::uuid
    std::string gpu_pci_bdf;  // lowercased, function suffix stripped
    std::string name;         // hipDeviceProp_t::name
};

struct VaContextInfo {
    int device_id = -1;
    std::string gpu_uuid;
    std::string gpu_pci_bdf;
    std::string device_name;
    int drm_fd = -1;
    VADisplay va_display = nullptr;
    int num_va_profiles = 0;
    std::vector<VAProfile> va_profile_list;  // profiles reported by the driver
    uint32_t num_enc_engines = 0;
    bool caps_probed[3] = {false, false, false};
    VaEncodeCaps caps[3];
};

/**
 * @brief Process-wide singleton owning the libva handle, the render nodes and
 *        the per-device capability cache.
 *
 * Ported from rocDecode's VaContext with one substitution and one deliberate
 * non-substitution:
 *   - VAEntrypointEncSlice replaces VAEntrypointVLD in the config queries, and
 *     that query is what establishes whether a device can encode at all.
 *   - amdgpu_query_hw_ip_count(AMDGPU_HW_IP_VCN_DEC) is NOT turned into the
 *     _ENC variant and used as a capability test.  The engine count is read
 *     for reporting only (ROCENC_CAPS_NUM_ENCODER_ENGINES) and is never
 *     consulted when deciding `VaEncodeCaps::supported`.
 */
class VaContext {
public:
    static VaContext& GetInstance() {
        static VaContext instance;
        return instance;
    }

    // Finds or creates the VA context for a HIP device.  Returns false and
    // leaves *va_ctx_id untouched on failure.
    bool GetVaContext(const HipDeviceIdentity& ident, uint32_t* va_ctx_id);

    // Returns the cached encode capabilities for one codec on one context.
    // A device with no encoder yields caps.supported == false and a true
    // return: "no encode" is an answer, not an error.
    bool GetEncodeCaps(uint32_t va_ctx_id, RocEncCodec codec, VaEncodeCaps* out);

    // Vendor string of the VA driver, for diagnostics.  Never gated on.
    std::string GetVendorString(uint32_t va_ctx_id);

private:
    VaContext();
    ~VaContext();
    VaContext(const VaContext&) = delete;
    VaContext& operator=(const VaContext&) = delete;

    std::mutex mutex_;
    std::vector<VaContextInfo> va_contexts_;

    // GPU UUID / PCI BDF -> render node index and compute partition.
    std::unordered_map<std::string, int> gpu_uuids_to_render_nodes_map_;
    std::unordered_map<std::string, ComputePartition> gpu_uuids_to_compute_partition_map_;
    std::unordered_map<std::string, int> gpu_pci_bdf_to_render_nodes_map_;
    std::unordered_map<std::string, ComputePartition> gpu_pci_bdf_to_compute_partition_map_;

    // Exclusively owns the dlopen handle and the VA function pointer table.
    // Outlives every VADisplay handle.
    std::unique_ptr<VaapiLoader> va_loader_;

    bool InitVaapi(uint32_t va_ctx_idx, const std::string& drm_node);
    void ProbeEncodeCaps(uint32_t va_ctx_idx, RocEncCodec codec);
    bool QueryKernelVideoCaps(int drm_fd, RocEncCodec codec, VaEncodeCaps* caps);
    void QueryEngineCount(uint32_t va_ctx_idx, const std::string& drm_node);

    static void GetVisibleDevices(std::vector<int>& visible_devices);
    static void GetDrmNodeOffset(const std::string& device_name, uint8_t device_id,
                                 const std::vector<int>& visible_devices,
                                 ComputePartition current_compute_partition, int& offset);
    void GetGpuUuids();
    static std::string GetRenderNodeBusId(const std::string& render_node_name);
    static std::string GetFirstAvailableDrmNode();
};
