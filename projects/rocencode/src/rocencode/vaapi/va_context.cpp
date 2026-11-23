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

#include "va_context.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

// amdgpu_drm.h supplies the AMDGPU_INFO_VIDEO_CAPS request/payload layout;
// xf86drm.h supplies drmCommandWrite().  libdrm's higher-level <libdrm/amdgpu.h>
// is deliberately NOT included: no amdgpu_* entry point is called from here.
#include <libdrm/amdgpu_drm.h>
#include <xf86drm.h>

// ---------------------------------------------------------------------------
// VA-API call redirection through the dlopen vtable.
//
// Every va*() call in this translation unit is macro-redirected through
// g_va_loader->fn.*, which resolves to the isolated librocm_sysdeps_va.so.2
// loaded with dlopen(RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND).  This stops a
// system libva.so.2 (e.g. pulled in by libavcodec in the same process) from
// winning the symbol table and intercepting rocEncode's VA calls.
//
// The macros are defined after the headers so that the type declarations from
// <va/va.h> stay visible.
// ---------------------------------------------------------------------------
static VaapiLoader* g_va_loader = nullptr;

// clang-format off
#define vaGetDisplayDRM(...)          (g_va_loader->fn.vaGetDisplayDRM(__VA_ARGS__))
#define vaInitialize(...)             (g_va_loader->fn.vaInitialize(__VA_ARGS__))
#define vaTerminate(...)              (g_va_loader->fn.vaTerminate(__VA_ARGS__))
#define vaSetInfoCallback(...)        (g_va_loader->fn.vaSetInfoCallback(__VA_ARGS__))
#define vaQueryVendorString(...)      (g_va_loader->fn.vaQueryVendorString(__VA_ARGS__))
#define vaErrorStr(...)               (g_va_loader->fn.vaErrorStr(__VA_ARGS__))
#define vaMaxNumProfiles(...)         (g_va_loader->fn.vaMaxNumProfiles(__VA_ARGS__))
#define vaMaxNumEntrypoints(...)      (g_va_loader->fn.vaMaxNumEntrypoints(__VA_ARGS__))
#define vaQueryConfigProfiles(...)    (g_va_loader->fn.vaQueryConfigProfiles(__VA_ARGS__))
#define vaQueryConfigEntrypoints(...) (g_va_loader->fn.vaQueryConfigEntrypoints(__VA_ARGS__))
#define vaGetConfigAttributes(...)    (g_va_loader->fn.vaGetConfigAttributes(__VA_ARGS__))
#define vaCreateConfig(...)           (g_va_loader->fn.vaCreateConfig(__VA_ARGS__))
#define vaDestroyConfig(...)          (g_va_loader->fn.vaDestroyConfig(__VA_ARGS__))
#define vaQuerySurfaceAttributes(...) (g_va_loader->fn.vaQuerySurfaceAttributes(__VA_ARGS__))
// clang-format on

const char* RocEncCodecName(RocEncCodec codec) {
    switch (codec) {
        case RocEncCodec::kH264: return "H.264";
        case RocEncCodec::kHevc: return "HEVC";
        case RocEncCodec::kAv1:  return "AV1";
    }
    return "unknown";
}

namespace {

// Candidate VA profiles per codec.  The capability record is the union over
// whichever of these the driver reports with VAEntrypointEncSlice.
const std::vector<VAProfile>& CandidateProfiles(RocEncCodec codec) {
    static const std::vector<VAProfile> kH264 = {
        VAProfileH264ConstrainedBaseline, VAProfileH264Main, VAProfileH264High};
    static const std::vector<VAProfile> kHevc = {VAProfileHEVCMain, VAProfileHEVCMain10};
    static const std::vector<VAProfile> kAv1 = {VAProfileAV1Profile0};
    switch (codec) {
        case RocEncCodec::kH264: return kH264;
        case RocEncCodec::kHevc: return kHevc;
        case RocEncCodec::kAv1:  return kAv1;
    }
    return kH264;
}

inline size_t CodecIndex(RocEncCodec codec) { return static_cast<size_t>(codec); }

}  // namespace

VaContext::VaContext() {
    // Create the loader before any VA call so the redirect macros are valid.
    // Throws std::runtime_error on failure, which propagates to the first
    // caller of VaContext::GetInstance().
    va_loader_ = std::make_unique<VaapiLoader>();
    g_va_loader = va_loader_.get();
    GetGpuUuids();
}

VaContext::~VaContext() {
    for (auto& ctx : va_contexts_) {
        if (ctx.va_display) {
            if (vaTerminate(ctx.va_display) != VA_STATUS_SUCCESS) {
                CriticalLog(g_rocenc_logger, "Failed to terminate VA");
            }
            ctx.va_display = nullptr;
        }
        if (ctx.drm_fd != -1) {
            close(ctx.drm_fd);
            ctx.drm_fd = -1;
        }
    }
    // Null the global pointer before destroying the loader so that an
    // accidental post-destruction macro invocation fails visibly rather than
    // calling through a dangling pointer.
    g_va_loader = nullptr;
    va_loader_.reset();
}

bool VaContext::GetVaContext(const HipDeviceIdentity& ident, uint32_t* va_ctx_id) {
    FunctionEntryLogWithArgs(g_rocenc_logger, ROCENC_TOSTR(ident.device_id));
    std::lock_guard<std::mutex> lock(mutex_);

    for (uint32_t i = 0; i < va_contexts_.size(); i++) {
        if ((!ident.gpu_pci_bdf.empty() && ident.gpu_pci_bdf == va_contexts_[i].gpu_pci_bdf) ||
            (!ident.gpu_uuid.empty() && ident.gpu_uuid == va_contexts_[i].gpu_uuid)) {
            *va_ctx_id = i;
            return true;
        }
    }

    va_contexts_.resize(va_contexts_.size() + 1);
    uint32_t idx = static_cast<uint32_t>(va_contexts_.size() - 1);
    va_contexts_[idx].device_id = ident.device_id;
    va_contexts_[idx].gpu_uuid = ident.gpu_uuid;
    va_contexts_[idx].gpu_pci_bdf = ident.gpu_pci_bdf;
    va_contexts_[idx].device_name = ident.name;

    std::vector<int> visible_devices;
    GetVisibleDevices(visible_devices);

    ComputePartition partition = kSpx;
    {
        auto bdf_it = gpu_pci_bdf_to_compute_partition_map_.find(ident.gpu_pci_bdf);
        if (bdf_it != gpu_pci_bdf_to_compute_partition_map_.end()) {
            partition = bdf_it->second;
        } else {
            auto uuid_it = gpu_uuids_to_compute_partition_map_.find(ident.gpu_uuid);
            if (uuid_it != gpu_uuids_to_compute_partition_map_.end()) {
                partition = uuid_it->second;
            }
        }
    }

    int offset = 0;
    GetDrmNodeOffset(ident.name, static_cast<uint8_t>(ident.device_id), visible_devices, partition, offset);

    int render_node_id = -1;
    {
        auto bdf_it = gpu_pci_bdf_to_render_nodes_map_.find(ident.gpu_pci_bdf);
        if (bdf_it != gpu_pci_bdf_to_render_nodes_map_.end()) {
            render_node_id = bdf_it->second;
        } else {
            auto uuid_it = gpu_uuids_to_render_nodes_map_.find(ident.gpu_uuid);
            if (uuid_it != gpu_uuids_to_render_nodes_map_.end()) {
                render_node_id = uuid_it->second;
            }
        }
    }

    std::string drm_node;
    if (render_node_id >= 0) {
        drm_node = "/dev/dri/renderD" + std::to_string(render_node_id + offset);
    } else {
        drm_node = GetFirstAvailableDrmNode();
        if (drm_node.empty()) {
            drm_node = "/dev/dri/renderD128";
        }
    }
    InfoLog(g_rocenc_logger, "Selected DRM node: " + drm_node);

    if (!InitVaapi(idx, drm_node)) {
        CriticalLog(g_rocenc_logger, "Failed to initialize VA-API on " + drm_node);
        va_contexts_.pop_back();
        return false;
    }

    // Informational only.  Never used to decide encode capability.
    QueryEngineCount(idx, drm_node);

    va_contexts_[idx].num_va_profiles = vaMaxNumProfiles(va_contexts_[idx].va_display);
    if (va_contexts_[idx].num_va_profiles > 0) {
        va_contexts_[idx].va_profile_list.resize(va_contexts_[idx].num_va_profiles);
        VAStatus va_status = vaQueryConfigProfiles(va_contexts_[idx].va_display,
                                                   va_contexts_[idx].va_profile_list.data(),
                                                   &va_contexts_[idx].num_va_profiles);
        if (va_status != VA_STATUS_SUCCESS) {
            CriticalLog(g_rocenc_logger,
                        ROCENC_STR("vaQueryConfigProfiles failed: ") + vaErrorStr(va_status));
            va_contexts_[idx].num_va_profiles = 0;
            va_contexts_[idx].va_profile_list.clear();
        } else {
            va_contexts_[idx].va_profile_list.resize(va_contexts_[idx].num_va_profiles);
        }
    }

    *va_ctx_id = idx;
    return true;
}

bool VaContext::GetEncodeCaps(uint32_t va_ctx_id, RocEncCodec codec, VaEncodeCaps* out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (va_ctx_id >= va_contexts_.size() || out == nullptr) {
        return false;
    }
    size_t ci = CodecIndex(codec);
    if (!va_contexts_[va_ctx_id].caps_probed[ci]) {
        ProbeEncodeCaps(va_ctx_id, codec);
        va_contexts_[va_ctx_id].caps_probed[ci] = true;
    }
    *out = va_contexts_[va_ctx_id].caps[ci];
    return true;
}

std::string VaContext::GetVendorString(uint32_t va_ctx_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (va_ctx_id >= va_contexts_.size() || va_contexts_[va_ctx_id].va_display == nullptr) {
        return std::string();
    }
    const char* vendor = vaQueryVendorString(va_contexts_[va_ctx_id].va_display);
    return vendor ? std::string(vendor) : std::string();
}

bool VaContext::InitVaapi(uint32_t va_ctx_idx, const std::string& drm_node) {
    va_contexts_[va_ctx_idx].drm_fd = open(drm_node.c_str(), O_RDWR | O_CLOEXEC);
    if (va_contexts_[va_ctx_idx].drm_fd < 0) {
        CriticalLog(g_rocenc_logger, "Failed to open DRM node: " + drm_node);
        return false;
    }
    va_contexts_[va_ctx_idx].va_display = vaGetDisplayDRM(va_contexts_[va_ctx_idx].drm_fd);
    if (!va_contexts_[va_ctx_idx].va_display) {
        CriticalLog(g_rocenc_logger, "Failed to create VA display.");
        close(va_contexts_[va_ctx_idx].drm_fd);
        va_contexts_[va_ctx_idx].drm_fd = -1;
        return false;
    }

    std::string va_driver_path;
    vaSetInfoCallback(va_contexts_[va_ctx_idx].va_display, [](void* user_context, const char* message) {
        std::string msg(message);
        if (msg.find("Trying to open") != std::string::npos) {
            *static_cast<std::string*>(user_context) = msg;
        }
    }, &va_driver_path);
    int major_version = 0, minor_version = 0;
    VAStatus va_status = vaInitialize(va_contexts_[va_ctx_idx].va_display, &major_version, &minor_version);
    vaSetInfoCallback(va_contexts_[va_ctx_idx].va_display, nullptr, nullptr);
    if (va_status != VA_STATUS_SUCCESS) {
        CriticalLog(g_rocenc_logger, ROCENC_STR("vaInitialize failed: ") + vaErrorStr(va_status));
        va_contexts_[va_ctx_idx].va_display = nullptr;
        close(va_contexts_[va_ctx_idx].drm_fd);
        va_contexts_[va_ctx_idx].drm_fd = -1;
        return false;
    }
    InfoLog(g_rocenc_logger, "VA-API version " + std::to_string(major_version) + "." + std::to_string(minor_version));
    const char* vendor_str = vaQueryVendorString(va_contexts_[va_ctx_idx].va_display);
    // Logged for diagnostics only.  Capability is never gated on the Mesa
    // version string -- it is gated on vaGetConfigAttributes.
    InfoLog(g_rocenc_logger, "VA-API vendor: " + std::string(vendor_str ? vendor_str : "<unknown>"));
    if (!va_driver_path.empty()) {
        InfoLog(g_rocenc_logger, va_driver_path);
    }
    return true;
}

void VaContext::ProbeEncodeCaps(uint32_t va_ctx_idx, RocEncCodec codec) {
    VaContextInfo& ctx = va_contexts_[va_ctx_idx];
    VaEncodeCaps& caps = ctx.caps[CodecIndex(codec)];
    caps = VaEncodeCaps{};
    caps.num_enc_engines = ctx.num_enc_engines;

    if (ctx.va_display == nullptr) {
        return;
    }

    const std::vector<VAProfile>& candidates = CandidateProfiles(codec);
    int max_entrypoints = vaMaxNumEntrypoints(ctx.va_display);
    if (max_entrypoints <= 0) {
        return;
    }
    std::vector<VAEntrypoint> entrypoints(static_cast<size_t>(max_entrypoints));

    for (size_t pi = 0; pi < candidates.size(); pi++) {
        VAProfile profile = candidates[pi];
        if (std::find(ctx.va_profile_list.begin(), ctx.va_profile_list.end(), profile) ==
            ctx.va_profile_list.end()) {
            continue;
        }
        int num_entrypoints = 0;
        if (vaQueryConfigEntrypoints(ctx.va_display, profile, entrypoints.data(), &num_entrypoints) !=
            VA_STATUS_SUCCESS) {
            continue;
        }
        // THE capability test.  VAEntrypointEncSlice, and nothing else.
        bool has_enc_slice = false;
        for (int i = 0; i < num_entrypoints; i++) {
            if (entrypoints[static_cast<size_t>(i)] == VAEntrypointEncSlice) {
                has_enc_slice = true;
                break;
            }
        }
        if (!has_enc_slice) {
            continue;
        }

        // Mesa silently ignores unsupported misc parameter buffers and returns
        // success, so a VA return code never proves a feature works.  Only the
        // attribute values below are trusted.
        VAConfigAttrib attribs[] = {
            {VAConfigAttribRTFormat, 0},
            {VAConfigAttribRateControl, 0},
            {VAConfigAttribEncPackedHeaders, 0},
            {VAConfigAttribEncMaxRefFrames, 0},
            {VAConfigAttribEncMaxSlices, 0},
            {VAConfigAttribEncQualityRange, 0},
            {VAConfigAttribEncIntraRefresh, 0},
            {VAConfigAttribEncROI, 0},
            {VAConfigAttribEncSliceStructure, 0},
            {VAConfigAttribEncInterlaced, 0},
            {VAConfigAttribEncRateControlExt, 0},
        };
        const int num_attribs = static_cast<int>(sizeof(attribs) / sizeof(attribs[0]));
        for (int i = 0; i < num_attribs; i++) {
            attribs[i].value = VA_ATTRIB_NOT_SUPPORTED;
        }
        VAStatus va_status =
            vaGetConfigAttributes(ctx.va_display, profile, VAEntrypointEncSlice, attribs, num_attribs);
        if (va_status != VA_STATUS_SUCCESS) {
            DebugLog(g_rocenc_logger, ROCENC_STR("vaGetConfigAttributes failed for ") +
                                          RocEncCodecName(codec) + ": " + vaErrorStr(va_status));
            continue;
        }

        caps.supported = true;
        caps.profile_mask |= (1u << pi);

        auto value_of = [&](VAConfigAttribType type) -> uint32_t {
            for (int i = 0; i < num_attribs; i++) {
                if (attribs[i].type == type && attribs[i].value != VA_ATTRIB_NOT_SUPPORTED) {
                    return attribs[i].value;
                }
            }
            return 0;
        };

        uint32_t rt_format = value_of(VAConfigAttribRTFormat);
        caps.rt_format_mask |= rt_format;
        caps.rate_control_mask |= value_of(VAConfigAttribRateControl);
        caps.packed_headers |= value_of(VAConfigAttribEncPackedHeaders);
        caps.quality_range = std::max(caps.quality_range, value_of(VAConfigAttribEncQualityRange));
        caps.intra_refresh |= value_of(VAConfigAttribEncIntraRefresh);
        caps.roi_attrib |= value_of(VAConfigAttribEncROI);
        caps.slice_structure |= value_of(VAConfigAttribEncSliceStructure);
        caps.max_slices = std::max(caps.max_slices, value_of(VAConfigAttribEncMaxSlices));
        uint32_t rc_ext = value_of(VAConfigAttribEncRateControlExt);
        caps.rate_control_ext |= rc_ext;
        // VAConfigAttribValEncRateControlExt: bits 0-7 carry
        // max_num_temporal_layers_minus1, so the count is always +1 and is
        // computed unconditionally.  A driver that reports the attribute with
        // value 0 means "one temporal layer", not "none"; an absent attribute
        // likewise means the base layer only.  Guarding on rc_ext != 0 would
        // report 0 layers for both, which would read as "cannot encode".
        caps.max_temporal_layers =
            std::max(caps.max_temporal_layers, (rc_ext & 0xffu) + 1u);
        if (value_of(VAConfigAttribEncInterlaced) != 0) {
            caps.interlaced = true;
        }

        uint32_t max_refs = value_of(VAConfigAttribEncMaxRefFrames);
        if (max_refs != 0) {
            caps.max_ref_l0 = std::max(caps.max_ref_l0, max_refs & 0xffffu);
            caps.max_ref_l1 = std::max(caps.max_ref_l1, (max_refs >> 16) & 0xffffu);
        }

        if (rt_format & VA_RT_FORMAT_YUV420_10) {
            caps.supports_10bit = true;
        }
        if (rt_format & VA_RT_FORMAT_YUV444) {
            caps.supports_yuv444 = true;
        }

        // Surface geometry limits.  vaQuerySurfaceAttributes needs a real
        // config, so create a minimal one and destroy it immediately.
        VAConfigAttrib create_attrib = {VAConfigAttribRTFormat,
                                        rt_format ? rt_format : static_cast<uint32_t>(VA_RT_FORMAT_YUV420)};
        VAConfigID config_id = VA_INVALID_ID;
        if (vaCreateConfig(ctx.va_display, profile, VAEntrypointEncSlice, &create_attrib, 1, &config_id) ==
            VA_STATUS_SUCCESS) {
            unsigned int num_surf_attribs = 0;
            if (vaQuerySurfaceAttributes(ctx.va_display, config_id, nullptr, &num_surf_attribs) ==
                    VA_STATUS_SUCCESS &&
                num_surf_attribs > 0) {
                std::vector<VASurfaceAttrib> surf_attribs(num_surf_attribs);
                if (vaQuerySurfaceAttributes(ctx.va_display, config_id, surf_attribs.data(),
                                             &num_surf_attribs) == VA_STATUS_SUCCESS) {
                    for (unsigned int i = 0; i < num_surf_attribs; i++) {
                        if (!(surf_attribs[i].flags & VA_SURFACE_ATTRIB_GETTABLE)) {
                            continue;
                        }
                        uint32_t v = static_cast<uint32_t>(surf_attribs[i].value.value.i);
                        switch (surf_attribs[i].type) {
                            case VASurfaceAttribMinWidth:
                                caps.va_min_width = caps.va_min_width ? std::min(caps.va_min_width, v) : v;
                                break;
                            case VASurfaceAttribMinHeight:
                                caps.va_min_height = caps.va_min_height ? std::min(caps.va_min_height, v) : v;
                                break;
                            case VASurfaceAttribMaxWidth:
                                caps.va_max_width = std::max(caps.va_max_width, v);
                                break;
                            case VASurfaceAttribMaxHeight:
                                caps.va_max_height = std::max(caps.va_max_height, v);
                                break;
                            default:
                                break;
                        }
                    }
                }
            }
            vaDestroyConfig(ctx.va_display, config_id);
        }
    }

    if (!caps.supported) {
        InfoLog(g_rocenc_logger, ROCENC_STR("No VAEntrypointEncSlice for ") + RocEncCodecName(codec) +
                                     " on device " + ROCENC_TOSTR(ctx.device_id) +
                                     " -- reporting no encode support.");
        return;
    }

    // Kernel limits fill in what VA does not report (max level, pixel budget).
    // A failure here is not fatal: the VA answer already stands on its own.
    if (QueryKernelVideoCaps(ctx.drm_fd, codec, &caps)) {
        caps.kernel_caps_valid = true;
    }
}

bool VaContext::QueryKernelVideoCaps(int drm_fd, RocEncCodec codec, VaEncodeCaps* caps) {
#if defined(AMDGPU_INFO_VIDEO_CAPS) && defined(AMDGPU_INFO_VIDEO_CAPS_ENCODE)
    if (drm_fd < 0) {
        return false;
    }
    unsigned int codec_idx = 0;
    switch (codec) {
        case RocEncCodec::kH264: codec_idx = AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_MPEG4_AVC; break;
        case RocEncCodec::kHevc: codec_idx = AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_HEVC; break;
        case RocEncCodec::kAv1:  codec_idx = AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_AV1; break;
    }

    // libdrm's amdgpu_query_info() zeroes the request union, which pins the
    // sub-query to DECODE.  The encode sub-query therefore needs the ioctl
    // issued directly.
    struct drm_amdgpu_info_video_caps video_caps;
    std::memset(&video_caps, 0, sizeof(video_caps));
    struct drm_amdgpu_info request;
    std::memset(&request, 0, sizeof(request));
    request.return_pointer = reinterpret_cast<uintptr_t>(&video_caps);
    request.return_size = sizeof(video_caps);
    request.query = AMDGPU_INFO_VIDEO_CAPS;
    // Note: the member of `struct drm_amdgpu_info` is `video_cap`, singular,
    // even though the payload struct it selects is `..._video_caps`.
    request.video_cap.type = AMDGPU_INFO_VIDEO_CAPS_ENCODE;

    if (drmCommandWrite(drm_fd, DRM_AMDGPU_INFO, &request, sizeof(request)) != 0) {
        DebugLog(g_rocenc_logger, "AMDGPU_INFO_VIDEO_CAPS (encode) query failed; kernel limits unavailable.");
        return false;
    }
    if (codec_idx >= AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_COUNT) {
        return false;
    }
    const struct drm_amdgpu_info_video_codec_info& info = video_caps.codec_info[codec_idx];
    if (!info.valid) {
        // The kernel says this device cannot encode this codec.  VA already
        // had the last word on `supported`; this only means no extra limits.
        return false;
    }
    caps->kernel_max_width = info.max_width;
    caps->kernel_max_height = info.max_height;
    caps->kernel_max_pixels_per_frame = info.max_pixels_per_frame;
    caps->kernel_max_level = info.max_level;
    return true;
#else
    (void)drm_fd;
    (void)codec;
    (void)caps;
    DebugLog(g_rocenc_logger,
             "libdrm headers do not expose AMDGPU_INFO_VIDEO_CAPS; kernel encode limits unavailable.");
    return false;
#endif
}

void VaContext::QueryEngineCount(uint32_t va_ctx_idx, const std::string& drm_node) {
    // Reported through ROCENC_CAPS_NUM_ENCODER_ENGINES only.  Deliberately not
    // AMDGPU_HW_IP_VCN_ENC: MI300 reports num_enc_rings = 1 with no encoder,
    // so the ring count answers "how many engines", never "can this encode".
    // The value is taken from the number of VCN instances the VA driver
    // exposes, which is 1 whenever any encode entrypoint exists.  A real
    // multi-engine count needs a source that does not conflate the two; until
    // then this reports 1 for an encode-capable device and 0 otherwise, and
    // ProbeEncodeCaps() overwrites it per codec.
    (void)drm_node;
    va_contexts_[va_ctx_idx].num_enc_engines = 1;
}

void VaContext::GetVisibleDevices(std::vector<int>& visible_devices) {
    const char* env = std::getenv("ROCR_VISIBLE_DEVICES");
    if (env == nullptr) {
        env = std::getenv("HIP_VISIBLE_DEVICES");
    }
    visible_devices = ParseVisibleDevicesCsv(env);
    std::sort(visible_devices.begin(), visible_devices.end());
}

void VaContext::GetDrmNodeOffset(const std::string& device_name, uint8_t device_id,
                                 const std::vector<int>& visible_devices,
                                 ComputePartition current_compute_partition, int& offset) {
    auto mod = [&](int divisor) {
        return (static_cast<size_t>(device_id) < visible_devices.size())
                   ? (visible_devices[device_id] % divisor)
                   : (static_cast<int>(device_id) % divisor);
    };
    switch (current_compute_partition) {
        case kSpx:
            offset = 0;
            break;
        case kDpx:
            offset = mod(2);
            break;
        case kTpx:
            offset = mod(3);
            break;
        case kQpx:
            offset = mod(4);
            break;
        case kCpx:
            // The MI300 series share gfx942, so the device name is the only way
            // to tell MI300A/MI308 apart.
            if (device_name.find("MI300A") != std::string::npos) {
                offset = mod(6);
            } else if (device_name.find("MI308") != std::string::npos) {
                offset = mod(4);
            } else {
                offset = mod(8);
            }
            break;
    }
}

void VaContext::GetGpuUuids() {
    std::string dri_path = "/dev/dri";
    DIR* dir = opendir(dri_path.c_str());
    if (!dir) {
        return;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.rfind("renderD", 0) != 0 || filename.size() <= 7) {
            continue;
        }
        int render_id = 0;
        try {
            render_id = std::stoi(filename.substr(7));
        } catch (...) {
            continue;
        }
        std::string sys_device_path = "/sys/class/drm/" + filename + "/device";
        struct stat info;
        if (stat(sys_device_path.c_str(), &info) != 0) {
            continue;
        }
        std::string bus_id = GetRenderNodeBusId(filename);
        if (!bus_id.empty()) {
            gpu_pci_bdf_to_render_nodes_map_[bus_id] = render_id;
        }
        std::string unique_id;
        {
            std::ifstream unique_id_file(sys_device_path + "/unique_id");
            if (unique_id_file.is_open() && std::getline(unique_id_file, unique_id) && !unique_id.empty()) {
                gpu_uuids_to_render_nodes_map_[unique_id] = render_id;
            }
        }
        if (unique_id.empty()) {
            continue;
        }
        std::ifstream partition_file(sys_device_path + "/current_compute_partition");
        std::string partition;
        if (partition_file.is_open() && std::getline(partition_file, partition) && !partition.empty()) {
            std::transform(partition.begin(), partition.end(), partition.begin(),
                           [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            ComputePartition current = kSpx;
            if (partition == "SPX") {
                current = kSpx;
            } else if (partition == "DPX") {
                current = kDpx;
            } else if (partition == "TPX") {
                current = kTpx;
            } else if (partition == "QPX") {
                current = kQpx;
            } else if (partition == "CPX") {
                current = kCpx;
            }
            gpu_uuids_to_compute_partition_map_[unique_id] = current;
            if (!bus_id.empty()) {
                gpu_pci_bdf_to_compute_partition_map_[bus_id] = current;
            }
        }
    }
    closedir(dir);
}

std::string VaContext::GetRenderNodeBusId(const std::string& render_node_name) {
    std::string device_link = "/sys/class/drm/" + render_node_name + "/device";
    char* resolved = realpath(device_link.c_str(), nullptr);
    if (resolved == nullptr) {
        return "";
    }
    std::string path(resolved);
    free(resolved);
    size_t pos = path.find_last_of('/');
    std::string bus_id = (pos == std::string::npos) ? path : path.substr(pos + 1);
    if (bus_id.find(':') == std::string::npos || bus_id.find('.') == std::string::npos) {
        return "";
    }
    std::transform(bus_id.begin(), bus_id.end(), bus_id.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    size_t dot_pos = bus_id.find_last_of('.');
    if (dot_pos != std::string::npos) {
        bus_id = bus_id.substr(0, dot_pos);
    }
    return bus_id;
}

std::string VaContext::GetFirstAvailableDrmNode() {
    DIR* dir = opendir("/dev/dri");
    if (!dir) {
        return "";
    }
    int min_render_id = -1;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.rfind("renderD", 0) == 0 && filename.size() > 7) {
            try {
                int render_id = std::stoi(filename.substr(7));
                if (min_render_id < 0 || render_id < min_render_id) {
                    min_render_id = render_id;
                }
            } catch (...) {
            }
        }
    }
    closedir(dir);
    if (min_render_id < 0) {
        return "";
    }
    return "/dev/dri/renderD" + std::to_string(min_render_id);
}
