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

#ifdef ROCDECODE_BUILD_WINDOWS

// Disable MSVC warnings for this file
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4365)  // signed/unsigned mismatch
#pragma warning(disable: 4244)  // conversion from type1 to type2, possible loss of data
#pragma warning(disable: 4267)  // conversion from size_t to smaller type
#endif

#include "pal_videodecoder.h"
#include "../../commons.h"
#include <cstring>
#include <algorithm>

namespace rocdec {

// Static members initialization
std::recursive_mutex PalVideoDecoder::platform_mutex_;
Pal::IPlatform* PalVideoDecoder::platform_ = nullptr;
void* PalVideoDecoder::platform_mem_ = nullptr;
int PalVideoDecoder::platform_ref_count_ = 0;
Pal::IDevice* PalVideoDecoder::device_instance_ = nullptr;
Pal::EngineType PalVideoDecoder::video_decode_engine_ = Pal::EngineTypeCount;

/* ================================================================ */
/* PalDpb implementation                                            */
/* ================================================================ */

int32_t PalDpb::AllocSlot() {
    for (uint32_t i = 0; i < slot_count_; ++i) {
        if (!slots_[i].occupied) {
            slots_[i].occupied = true;
            return static_cast<int32_t>(i);
        }
    }
    if (slot_count_ < kMaxSlots) {
        int32_t idx = static_cast<int32_t>(slot_count_++);
        slots_[idx].occupied = true;
        return idx;
    }
    return -1;
}

void PalDpb::FreeSlot(int32_t idx) {
    if (idx >= 0 && static_cast<uint32_t>(idx) < slot_count_) {
        slots_[idx].occupied = false;
    }
}

int32_t PalDpb::FindByPoc(int32_t poc) const {
    for (uint32_t i = 0; i < slot_count_; ++i) {
        if (slots_[i].occupied && slots_[i].poc == poc) {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

void PalDpb::Clear() {
    for (uint32_t i = 0; i < slot_count_; ++i) {
        slots_[i].image.Reset();
        slots_[i].image_memory.Reset();
        slots_[i].target_image.Reset();
        slots_[i].target_image_memory.Reset();
        slots_[i].fence.Reset();
        slots_[i].occupied = false;
        slots_[i].poc = -1;
        slots_[i].frame_idx = -1;
    }
    slot_count_ = 0;
}

/* ================================================================ */
/* PAL Platform singleton                                           */
/* ================================================================ */

namespace {
    // PAL allocator callbacks (wrapper around malloc/free)
    void* PalAlloc(void*, size_t size, size_t, Util::SystemAllocType) {
        return malloc(size);
    }

    void PalFree(void*, void* ptr) {
        free(ptr);
    }
}

Pal::IPlatform* PalVideoDecoder::GetPalPlatform() {
    InfoLog(g_rocdec_logger, "PAL: GetPalPlatform - acquiring mutex...");
    std::lock_guard<std::recursive_mutex> lock(platform_mutex_);
    InfoLog(g_rocdec_logger, "PAL: GetPalPlatform - mutex acquired");

    if (platform_) {
        InfoLog(g_rocdec_logger, "PAL: GetPalPlatform - returning cached platform");
        platform_ref_count_++;
        return platform_;
    }

    InfoLog(g_rocdec_logger, "PAL: GetPalPlatform - creating new platform");

    // Create PAL platform
    Pal::PlatformCreateInfo ci = {};
    ci.pSettingsPath = "rocDecode";
    ci.flags.supportRgpTraces = true;
    ci.flags.requestShadowDescriptorVaRange = true;

    Util::AllocCallbacks alloc_cb = {};
    alloc_cb.pfnAlloc = PalAlloc;
    alloc_cb.pfnFree = PalFree;
    alloc_cb.pClientData = nullptr;
    ci.pAllocCb = &alloc_cb;

    InfoLog(g_rocdec_logger, "PAL: GetPalPlatform - calling GetPlatformSize...");
    size_t plat_size = Pal::GetPlatformSize();
    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: GetPalPlatform - platform size = ") + ROCDEC_TOSTR(plat_size));

    InfoLog(g_rocdec_logger, "PAL: GetPalPlatform - allocating platform memory...");
    platform_mem_ = malloc(plat_size);
    if (!platform_mem_) {
        CriticalLog(g_rocdec_logger, "PAL: Failed to allocate platform memory");
        return nullptr;
    }
    InfoLog(g_rocdec_logger, "PAL: GetPalPlatform - platform memory allocated");

    InfoLog(g_rocdec_logger, "PAL: GetPalPlatform - calling CreatePlatform...");
    Pal::Result res = Pal::CreatePlatform(ci, platform_mem_, &platform_);
    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: GetPalPlatform - CreatePlatform returned result = ") + ROCDEC_TOSTR((int)res));

    if (Util::IsErrorResult(res)) {
        CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: CreatePlatform failed with result ") + ROCDEC_TOSTR((int)res));
        free(platform_mem_);
        platform_mem_ = nullptr;
        return nullptr;
    }

    platform_ref_count_ = 1;
    InfoLog(g_rocdec_logger, "PAL: Platform created successfully");
    return platform_;
}

void PalVideoDecoder::ReleasePalPlatform() {
    std::lock_guard<std::recursive_mutex> lock(platform_mutex_);

    if (--platform_ref_count_ == 0) {
        if (platform_) {
            platform_->Destroy();
            platform_ = nullptr;
        }
        if (platform_mem_) {
            free(platform_mem_);
            platform_mem_ = nullptr;
        }
        device_instance_ = nullptr;
        InfoLog(g_rocdec_logger, "PAL: Platform destroyed");
    }
}

Pal::IDevice* PalVideoDecoder::GetPalDevice() {
    std::lock_guard<std::recursive_mutex> lock(platform_mutex_);

    if (device_instance_) {
        InfoLog(g_rocdec_logger, "PAL: Returning cached device instance");
        return device_instance_;
    }

    InfoLog(g_rocdec_logger, "PAL: Getting PAL platform...");
    Pal::IPlatform* platform = GetPalPlatform();
    if (!platform) {
        CriticalLog(g_rocdec_logger, "PAL: Failed to get platform");
        return nullptr;
    }

    // Enumerate devices
    uint32_t device_count = 0;
    Pal::IDevice* raw_devices[Pal::MaxDevices] = {};

    InfoLog(g_rocdec_logger, "PAL: Enumerating devices...");
    Pal::Result res = platform->EnumerateDevices(&device_count, raw_devices);
    if (Util::IsErrorResult(res) || device_count == 0) {
        CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: EnumerateDevices failed - result=") +
                    ROCDEC_TOSTR((int)res) + " device_count=" + ROCDEC_TOSTR(device_count));
        return nullptr;
    }

    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Found ") + ROCDEC_TOSTR(device_count) + " device(s)");

    // Find first device with VCN decode support
    for (uint32_t i = 0; i < device_count; ++i) {
        Pal::IDevice* dev = raw_devices[i];
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Checking device ") + ROCDEC_TOSTR(i));

        // Commit settings
        res = dev->CommitSettingsAndInit();
        if (Util::IsErrorResult(res)) {
            ErrorLog(g_rocdec_logger, ROCDEC_STR("PAL: Device ") + ROCDEC_TOSTR(i) +
                     " CommitSettingsAndInit failed with result " + ROCDEC_TOSTR((int)res));
            continue;
        }

        // Check for VCN hardware
        Pal::DeviceProperties dev_props = {};
        InfoLog(g_rocdec_logger, "PAL: Getting properties for device " + ROCDEC_TOSTR(i));
        dev->GetProperties(&dev_props);

        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Device ") + ROCDEC_TOSTR(i) +
                " VCN level=" + ROCDEC_TOSTR((int)dev_props.vcnLevel) +
                " engineCount[VcnDecode=" + ROCDEC_TOSTR((int)Pal::EngineTypeVcnDecode) + "]=" +
                    ROCDEC_TOSTR(dev_props.engineProperties[Pal::EngineTypeVcnDecode].engineCount) +
                " engineCount[VcnUnified=" + ROCDEC_TOSTR((int)Pal::EngineTypeVcnUnified) + "]=" +
                    ROCDEC_TOSTR(dev_props.engineProperties[Pal::EngineTypeVcnUnified].engineCount) +
                " EngineTypeCount=" + ROCDEC_TOSTR((int)Pal::EngineTypeCount));

        if (dev_props.vcnLevel == Pal::VcnIpLevel::None ||
            dev_props.vcnLevel == Pal::VcnIpLevel::_None) {
            InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Device ") + ROCDEC_TOSTR(i) + " has no VCN support");
            continue;
        }

        // Check for video decode engines
        Pal::EngineType video_engine = Pal::EngineTypeCount;
        if (dev_props.engineProperties[Pal::EngineTypeVcnDecode].engineCount > 0) {
            video_engine = Pal::EngineTypeVcnDecode;
        } else if (dev_props.engineProperties[Pal::EngineTypeVcnUnified].engineCount > 0) {
            video_engine = Pal::EngineTypeVcnUnified;
        }

        if (video_engine == Pal::EngineTypeCount) {
            continue;
        }

        // Finalize device
        Pal::DeviceFinalizeInfo fi = {};
        fi.requestedEngineCounts[video_engine].engines = 1;
        fi.requestedEngineCounts[Pal::EngineTypeUniversal].engines =
            (dev_props.engineProperties[Pal::EngineTypeUniversal].engineCount > 0) ? 1 : 0;

        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Finalizing device ") + ROCDEC_TOSTR(i) +
                " with engineType=" + ROCDEC_TOSTR((int)video_engine) +
                " universalEngines=" + ROCDEC_TOSTR(fi.requestedEngineCounts[Pal::EngineTypeUniversal].engines));
        res = dev->Finalize(fi);
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Finalize result=") + ROCDEC_TOSTR((int)res));
        if (Util::IsErrorResult(res)) {
            continue;
        }

        device_instance_ = dev;
        video_decode_engine_ = video_engine;

        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Device ") + ROCDEC_TOSTR(i) +
                " initialized with VCN support, engineType=" + ROCDEC_TOSTR((int)video_engine) +
                " (EngineTypeVcnDecode=" + ROCDEC_TOSTR((int)Pal::EngineTypeVcnDecode) +
                " EngineTypeVcnUnified=" + ROCDEC_TOSTR((int)Pal::EngineTypeVcnUnified) + ")");
        return device_instance_;
    }

    CriticalLog(g_rocdec_logger, "PAL: No device with VCN decode support found");
    return nullptr;
}

/* ================================================================ */
/* PalVideoDecoder implementation                                   */
/* ================================================================ */

PalVideoDecoder::PalVideoDecoder()
    : device_(nullptr)
    , codec_(rocDecVideoCodec_NumCodecs)
    , pal_format_(Pal::ChNumFormat::Undefined)
    , max_coded_width_(0)
    , max_coded_height_(0)
    , bit_depth_(0)
    , max_dpb_slots_(0)
    , initialized_(false)
    , session_begun_(false)
    , last_submitted_slot_idx_(-1) {
}

PalVideoDecoder::~PalVideoDecoder() {
    Destroy();
}

rocDecStatus PalVideoDecoder::Initialize(rocDecVideoCodec codec_type,
                                         uint32_t width,
                                         uint32_t height,
                                         uint32_t bit_depth,
                                         uint32_t max_dpb_slots) {
    InfoLog(g_rocdec_logger, "PAL: Initialize called - codec=" + ROCDEC_TOSTR((int)codec_type) +
            " width=" + ROCDEC_TOSTR(width) + " height=" + ROCDEC_TOSTR(height) +
            " bit_depth=" + ROCDEC_TOSTR(bit_depth) + " max_dpb_slots=" + ROCDEC_TOSTR(max_dpb_slots));

    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        InfoLog(g_rocdec_logger, "PAL: Already initialized, returning success");
        return ROCDEC_SUCCESS;
    }

    codec_ = codec_type;
    max_coded_width_ = width;
    max_coded_height_ = height;
    bit_depth_ = bit_depth;
    max_dpb_slots_ = std::min(max_dpb_slots, static_cast<uint32_t>(PalDpb::kMaxSlots));

    // Determine format based on bit depth
    pal_format_ = (bit_depth == 10) ? Pal::ChNumFormat::P010 : Pal::ChNumFormat::NV12;
    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Using format ") +
            (pal_format_ == Pal::ChNumFormat::P010 ? "P010" : "NV12"));

    // Get PAL device
    InfoLog(g_rocdec_logger, "PAL: Getting PAL device...");
    device_ = GetPalDevice();
    if (!device_) {
        CriticalLog(g_rocdec_logger, "PAL: Failed to get device");
        return ROCDEC_NOT_INITIALIZED;
    }
    InfoLog(g_rocdec_logger, "PAL: Device obtained successfully");

    Pal::EngineType eng = video_decode_engine_;
    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Using engine type ") + ROCDEC_TOSTR((int)eng));
    Pal::Result res;

    // Create video queue
    InfoLog(g_rocdec_logger, "PAL: Creating video queue...");
    {
        Pal::QueueCreateInfo qci = {};
        qci.engineType = eng;
        qci.engineIndex = 0;
        qci.queueType = Pal::QueueTypeVideoDecode;

        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Queue create - engineType=") + ROCDEC_TOSTR((int)qci.engineType) +
                " queueType=" + ROCDEC_TOSTR((int)qci.queueType));

        size_t sz = device_->GetQueueSize(qci, &res);
        if (Util::IsErrorResult(res)) {
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: GetQueueSize failed with result ") + ROCDEC_TOSTR((int)res));
            return ROCDEC_RUNTIME_ERROR;
        }
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Queue size = ") + ROCDEC_TOSTR(sz));

        void* mem = malloc(sz);
        Pal::IQueue* q = nullptr;
        res = device_->CreateQueue(qci, mem, &q);
        if (Util::IsErrorResult(res) || !q) {
            free(mem);
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: CreateQueue failed with result ") + ROCDEC_TOSTR((int)res));
            return ROCDEC_RUNTIME_ERROR;
        }

        *video_queue_.PtrAddr() = q;
        *video_queue_.MemAddr() = mem;
        InfoLog(g_rocdec_logger, "PAL: Video queue created successfully");
    }

    // Create command allocator
    // PAL recommends Local heap for UVD/VCN engine for best performance
    InfoLog(g_rocdec_logger, "PAL: Creating command allocator...");
    {
        // Use device-preferred heaps for the VCN engine (matches DXCP pattern)
        Pal::DeviceProperties dev_props_for_alloc = {};
        device_->GetProperties(&dev_props_for_alloc);
        const auto& eng_props = dev_props_for_alloc.engineProperties[eng];

        Pal::CmdAllocatorCreateInfo aci = {};
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: CmdAllocator create - engineType=") + ROCDEC_TOSTR((int)eng) +
                " queueType=" + ROCDEC_TOSTR((int)Pal::QueueTypeVideoDecode) +
                " preferred heaps: " +
                ROCDEC_TOSTR(eng_props.preferredCmdAllocHeaps[0]) + ", " +
                ROCDEC_TOSTR(eng_props.preferredCmdAllocHeaps[1]) + ", " +
                ROCDEC_TOSTR(eng_props.preferredCmdAllocHeaps[2]) + ", " +
                ROCDEC_TOSTR(eng_props.preferredCmdAllocHeaps[3]));
        aci.flags.threadSafe = 1;

        // Initialize all allocator types from the device's preferred heaps for this engine
        for (uint32_t i = 0; i < Pal::CmdAllocatorTypeCount; ++i) {
            aci.allocInfo[i].allocHeap = eng_props.preferredCmdAllocHeaps[i];
            aci.allocInfo[i].suballocSize = 64 * 1024;
            aci.allocInfo[i].allocSize    = 256 * 1024;
        }
        // VCN command data always goes on local heap
        aci.allocInfo[Pal::CommandDataAlloc].allocHeap = Pal::GpuHeapLocal;
        aci.allocInfo[Pal::EmbeddedDataAlloc].allocHeap = Pal::GpuHeapLocal;

        size_t sz = device_->GetCmdAllocatorSize(aci, &res);
        if (Util::IsErrorResult(res)) {
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: GetCmdAllocatorSize failed with result ") + ROCDEC_TOSTR((int)res));
            return ROCDEC_RUNTIME_ERROR;
        }

        void* mem = malloc(sz);
        Pal::ICmdAllocator* alloc = nullptr;
        res = device_->CreateCmdAllocator(aci, mem, &alloc);
        if (Util::IsErrorResult(res) || !alloc) {
            free(mem);
            CriticalLog(g_rocdec_logger, "PAL: CreateCmdAllocator failed");
            return ROCDEC_RUNTIME_ERROR;
        }

        *cmd_allocator_.PtrAddr() = alloc;
        *cmd_allocator_.MemAddr() = mem;
    }

    // Create command buffer (DXCP videoCmdList pattern: pCmdAllocator=nullptr at creation,
    // allocator bound at Reset() time before each frame recording).
    {
        Pal::CmdBufferCreateInfo cbci = {};
        cbci.engineType = eng;
        cbci.queueType = Pal::QueueTypeVideoDecode;
        //cbci.pCmdAllocator = nullptr;
        cbci.pCmdAllocator = static_cast<Pal::ICmdAllocator*>(*cmd_allocator_.PtrAddr());

        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: CmdBufferCreateInfo - engineType=") + ROCDEC_TOSTR((int)cbci.engineType) +
                " queueType=" + ROCDEC_TOSTR((int)cbci.queueType) +
                " (EngineTypeVcnDecode=" + ROCDEC_TOSTR((int)Pal::EngineTypeVcnDecode) +
                " EngineTypeVcnUnified=" + ROCDEC_TOSTR((int)Pal::EngineTypeVcnUnified) +
                " QueueTypeVideoDecode=" + ROCDEC_TOSTR((int)Pal::QueueTypeVideoDecode) + ")");

        size_t sz = device_->GetCmdBufferSize(cbci, &res);
        if (Util::IsErrorResult(res) || sz == 0) {
            // size=0 with Success means PAL was built without PAL_BUILD_VIDEO_DECODER — the
            // video decode command buffer code is not compiled into the library.
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: GetCmdBufferSize failed - result=") + ROCDEC_TOSTR((int)res) +
                        " size=" + ROCDEC_TOSTR(sz) +
                        " (size=0 indicates PAL library was built without PAL_BUILD_VIDEO_DECODER=1)");
            return ROCDEC_RUNTIME_ERROR;
        }
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: CmdBuffer size = ") + ROCDEC_TOSTR(sz));

        void* mem = malloc(sz);
        Pal::ICmdBuffer* cb = nullptr;
        res = device_->CreateCmdBuffer(cbci, mem, &cb);
        if (Util::IsErrorResult(res) || !cb) {
            free(mem);
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: CreateCmdBuffer failed with result ") + ROCDEC_TOSTR((int)res));
            return ROCDEC_RUNTIME_ERROR;
        }

        *cmd_buffer_.PtrAddr() = cb;
        *cmd_buffer_.MemAddr() = mem;
        InfoLog(g_rocdec_logger, "PAL: Command buffer created successfully");
    }

    // Allocate DPB slots
    for (uint32_t i = 0; i < max_dpb_slots_; ++i) {
        int32_t slot_idx = dpb_.AllocSlot();
        if (slot_idx < 0) {
            CriticalLog(g_rocdec_logger, "PAL: Failed to allocate DPB slot");
            return ROCDEC_OUTOF_MEMORY;
        }

        PalDpbSlot* slot = dpb_.GetSlot(slot_idx);
        rocDecStatus status = AllocateDecodedFrame(width, height, *slot);
        if (status != ROCDEC_SUCCESS) {
            return status;
        }
    }

    // Map rocDecode codec + bit depth to a PAL decode type once; reused below for both
    // the video decoder object and the decoder-heap size query.
    Pal::VideoDecodeType decode_type;
    switch (codec_) {
        case rocDecVideoCodec_HEVC:
            decode_type = (bit_depth_ == 10) ? Pal::VideoDecodeType::Hevc10Bit : Pal::VideoDecodeType::Hevc;
            break;
        case rocDecVideoCodec_AVC:
            decode_type = Pal::VideoDecodeType::H264;
            break;
        case rocDecVideoCodec_VP9:
            decode_type = (bit_depth_ == 10) ? Pal::VideoDecodeType::Vp910Bit : Pal::VideoDecodeType::Vp9;
            break;
        case rocDecVideoCodec_AV1:
            decode_type = Pal::VideoDecodeType::Av1;
            break;
        default:
            CriticalLog(g_rocdec_logger, "PAL: Unsupported codec type");
            return ROCDEC_NOT_SUPPORTED;
    }

    // Create video decoder object
    InfoLog(g_rocdec_logger, "PAL: Creating video decoder...");
    {
        Pal::VideoDecoderCreateInfo vd_ci = {};
        vd_ci.engineType = eng;
        vd_ci.decodeType = decode_type;

        size_t sz = device_->GetVideoDecoderSize(vd_ci, &res);
        if (Util::IsErrorResult(res)) {
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: GetVideoDecoderSize failed with result ") + ROCDEC_TOSTR((int)res));
            return ROCDEC_RUNTIME_ERROR;
        }
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Video decoder size = ") + ROCDEC_TOSTR(sz));

        void* mem = malloc(sz);
        Pal::IVideoDecoder* vd = nullptr;
        res = device_->CreateVideoDecoder(vd_ci, mem, &vd);
        if (Util::IsErrorResult(res) || !vd) {
            free(mem);
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: CreateVideoDecoder failed with result ") + ROCDEC_TOSTR((int)res));
            return ROCDEC_RUNTIME_ERROR;
        }

        *video_decoder_.PtrAddr() = vd;
        *video_decoder_.MemAddr() = mem;
        InfoLog(g_rocdec_logger, "PAL: Video decoder created successfully");
    }

    // Allocate decoder heap (VCN internal memory — size queried from PAL based on codec/resolution/DPB count)
    InfoLog(g_rocdec_logger, "PAL: Allocating decoder heap...");
    {
        Pal::VideoDecoderGpuMemInfo heap_mem_info = {};
        heap_mem_info.engineType = eng;
        heap_mem_info.decodeType = decode_type;
        heap_mem_info.decodeExtent.width  = max_coded_width_;
        heap_mem_info.decodeExtent.height = max_coded_height_;
        heap_mem_info.maxDpbFrames = max_dpb_slots_;
        heap_mem_info.frameRate = 30.0f;
        heap_mem_info.bitRate   = 0;

        Pal::GpuMemoryRequirements heap_mem_reqs = {};
        res = device_->GetVideoDecoderGpuMemorySize(heap_mem_info, &heap_mem_reqs);
        if (Util::IsErrorResult(res)) {
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: GetVideoDecoderGpuMemorySize failed with result ") + ROCDEC_TOSTR((int)res));
            return ROCDEC_RUNTIME_ERROR;
        }

        Pal::GpuMemoryCreateInfo heap_ci = {};
        // Use PAL-reported size; fall back to a generous 32MB if the query returns 0.
        heap_ci.size = (heap_mem_reqs.size > 0) ? heap_mem_reqs.size : (32ULL * 1024 * 1024);
        heap_ci.alignment = (heap_mem_reqs.alignment > 0) ? heap_mem_reqs.alignment : 4096;
        heap_ci.vaRange = Pal::VaRange::Default;
        heap_ci.priority = Pal::GpuMemPriority::Normal;
        if (heap_mem_reqs.heapCount > 0) {
            heap_ci.heapCount = heap_mem_reqs.heapCount;
            for (uint32_t h = 0; h < heap_mem_reqs.heapCount && h < Pal::GpuHeapCount; h++) {
                heap_ci.heaps[h] = heap_mem_reqs.heaps[h];
            }
        } else {
            heap_ci.heapCount = 1;
            heap_ci.heaps[0] = Pal::GpuHeapLocal;  // fallback: local heap for VCN
        }

        size_t heap_obj_size = device_->GetGpuMemorySize(heap_ci, &res);
        if (Util::IsErrorResult(res)) {
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: GetGpuMemorySize (decoder heap) failed with result ") + ROCDEC_TOSTR((int)res));
            return ROCDEC_RUNTIME_ERROR;
        }

        void* heap_obj_mem = malloc(heap_obj_size);
        Pal::IGpuMemory* heap_mem = nullptr;
        res = device_->CreateGpuMemory(heap_ci, heap_obj_mem, &heap_mem);
        if (Util::IsErrorResult(res) || !heap_mem) {
            free(heap_obj_mem);
            CriticalLog(g_rocdec_logger, "PAL: CreateGpuMemory (decoder heap) failed");
            return ROCDEC_RUNTIME_ERROR;
        }

        *decoder_heap_.PtrAddr() = heap_mem;
        *decoder_heap_.MemAddr() = heap_obj_mem;

        Pal::GpuMemoryRef mem_ref = {};
        mem_ref.pGpuMemory = heap_mem;
        Pal::Result add_res = device_->AddGpuMemoryReferences(1, &mem_ref, nullptr, 0);
        if (Util::IsErrorResult(add_res)) {
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: AddGpuMemoryReferences (decoder heap) failed with result ") +
                        ROCDEC_TOSTR((int)add_res));
            return ROCDEC_RUNTIME_ERROR;
        }
        InfoLog(g_rocdec_logger, "PAL: Decoder heap allocated successfully");
    }

    // Allocate bitstream buffer (16 MB initial)
    rocDecStatus status = AllocateBitstreamBuffer(16 * 1024 * 1024);
    if (status != ROCDEC_SUCCESS) {
        return status;
    }

    initialized_ = true;
    InfoLog(g_rocdec_logger, "PAL: Decoder initialized successfully");
    return ROCDEC_SUCCESS;
}

rocDecStatus PalVideoDecoder::AllocateDecodedFrame(uint32_t width, uint32_t height,
                                                    PalDpbSlot& slot) {
    Pal::Result res;

    // --- Tiled DPB surface (VCN reconstruction + reference reads) ---
    // VCN hardcodes the DPB swizzle to SW_256B_S on VCN4 (vcn3DecodeCmdBuffer.cpp).
    // Creating this surface Optimal + videoDecoder usage lets addrlib pick the matching
    // video-decode tiled mode. It is never touched by CPU/HIP, so no sharing flags.
    Pal::ImageCreateInfo ici = {};
    ici.imageType = Pal::ImageType::Tex2d;
    ici.swizzledFormat.format = pal_format_;
    ici.swizzledFormat.swizzle.r = Pal::ChannelSwizzle::X;
    ici.swizzledFormat.swizzle.g = Pal::ChannelSwizzle::Y;
    ici.swizzledFormat.swizzle.b = Pal::ChannelSwizzle::Z;
    ici.swizzledFormat.swizzle.a = Pal::ChannelSwizzle::W;
    ici.extent.width = width;
    // Round luma height up to a multiple of 16. The interop consumer (RocVideoDecoder's
    // GetSurfaceStrideInternal) assumes vstride = align(height, 16) for both NV12 and P016,
    // which fixes the chroma-plane offset and the total surface size so a flat DtoH copy of
    // the decoded surface stays within the exported allocation.
    ici.extent.height = (height + 15) & ~15u;
    ici.extent.depth = 1;
    ici.mipLevels = 1;
    ici.arraySize = 1;
    ici.samples = 1;
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 961
    ici.fragments = 1;
#endif
    ici.tiling = Pal::ImageTiling::Optimal;
    ici.usageFlags.videoDecoder = 1;

    size_t img_size = device_->GetImageSize(ici, &res);
    if (Util::IsErrorResult(res)) {
        CriticalLog(g_rocdec_logger, "PAL: GetImageSize failed");
        return ROCDEC_RUNTIME_ERROR;
    }

    void* img_mem = malloc(img_size);
    Pal::IImage* img = nullptr;
    res = device_->CreateImage(ici, img_mem, &img);
    if (Util::IsErrorResult(res) || !img) {
        free(img_mem);
        CriticalLog(g_rocdec_logger, "PAL: CreateImage failed");
        return ROCDEC_RUNTIME_ERROR;
    }

    *slot.image.PtrAddr() = img;
    *slot.image.MemAddr() = img_mem;

    // Allocate and bind GPU memory
    Pal::GpuMemoryRequirements mem_reqs = {};
    img->GetGpuMemoryRequirements(&mem_reqs);

    Pal::GpuMemoryCreateInfo mem_ci = {};
    mem_ci.size = mem_reqs.size;
    mem_ci.alignment = mem_reqs.alignment;
    // Tiled DPB: only VCN touches it (reconstruction + reference reads), never CPU/HIP,
    // so no sharing flags. GpuHeapInvisible fails AddGpuMemoryReferences(-9) on this WDDM
    // setup (no device-wide residency), so use a GART heap — proven to work with residency.
    // Tiling is an image-layout property and is independent of which heap backs the bytes.
    mem_ci.flags.interprocess = 0;
    mem_ci.flags.shareable = 0;
    mem_ci.heapCount = 1;
    mem_ci.heaps[0] = Pal::GpuHeapGartCacheable;

    size_t gpu_mem_size = device_->GetGpuMemorySize(mem_ci, &res);
    if (Util::IsErrorResult(res)) {
        CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: GetGpuMemorySize (DPB surface) failed result=") + ROCDEC_TOSTR((int)res));
        return ROCDEC_RUNTIME_ERROR;
    }

    void* gpu_mem_obj = malloc(gpu_mem_size);
    Pal::IGpuMemory* gpu_mem = nullptr;
    res = device_->CreateGpuMemory(mem_ci, gpu_mem_obj, &gpu_mem);
    if (Util::IsErrorResult(res) || !gpu_mem) {
        free(gpu_mem_obj);
        CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: CreateGpuMemory failed result=") + ROCDEC_TOSTR((int)res));
        return ROCDEC_RUNTIME_ERROR;
    }

    *slot.image_memory.PtrAddr() = gpu_mem;
    *slot.image_memory.MemAddr() = gpu_mem_obj;

    // Bind image to memory
    res = img->BindGpuMemory(gpu_mem, 0);
    if (Util::IsErrorResult(res)) {
        CriticalLog(g_rocdec_logger, "PAL: BindGpuMemory failed");
        return ROCDEC_RUNTIME_ERROR;
    }

    // Register with WDDM VidMm so the KMD keeps it resident for GPU access.
    // On WDDM, per-submit gpuMemRefs are not supported; residency is managed device-wide.
    {
        Pal::GpuMemoryRef mem_ref = {};
        mem_ref.pGpuMemory = gpu_mem;
        Pal::Result add_res = device_->AddGpuMemoryReferences(1, &mem_ref, nullptr, 0);
        if (Util::IsErrorResult(add_res)) {
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: AddGpuMemoryReferences (DPB slot) failed with result ") +
                        ROCDEC_TOSTR((int)add_res));
            return ROCDEC_RUNTIME_ERROR;
        }
    }

    // --- Linear decode-target surface (clean display copy, exported to HIP) ---
    // VCN writes a linear copy of the reconstructed frame here. It must be CPU-aperture
    // backed (GartCacheable) and interprocess=1 so a WDDM KMT handle can be exported.
    {
        Pal::ImageCreateInfo tici = ici;          // same format/extent/usage as the DPB
        tici.tiling = Pal::ImageTiling::Linear;

        size_t timg_size = device_->GetImageSize(tici, &res);
        if (Util::IsErrorResult(res)) {
            CriticalLog(g_rocdec_logger, "PAL: GetImageSize (target) failed");
            return ROCDEC_RUNTIME_ERROR;
        }

        void* timg_mem = malloc(timg_size);
        Pal::IImage* timg = nullptr;
        res = device_->CreateImage(tici, timg_mem, &timg);
        if (Util::IsErrorResult(res) || !timg) {
            free(timg_mem);
            CriticalLog(g_rocdec_logger, "PAL: CreateImage (target) failed");
            return ROCDEC_RUNTIME_ERROR;
        }

        *slot.target_image.PtrAddr() = timg;
        *slot.target_image.MemAddr() = timg_mem;

        Pal::GpuMemoryRequirements tmem_reqs = {};
        timg->GetGpuMemoryRequirements(&tmem_reqs);

        Pal::GpuMemoryCreateInfo tmem_ci = {};
        tmem_ci.size = tmem_reqs.size;
        tmem_ci.alignment = tmem_reqs.alignment;
        // GartCacheable is the only WDDM heap that supports interprocess=1 KMT export.
        tmem_ci.flags.interprocess = 1;
        tmem_ci.flags.shareable = 0;
        tmem_ci.heapCount = 1;
        tmem_ci.heaps[0] = Pal::GpuHeapGartCacheable;

        size_t tgpu_mem_size = device_->GetGpuMemorySize(tmem_ci, &res);
        if (Util::IsErrorResult(res)) {
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: GetGpuMemorySize (target) failed result=") + ROCDEC_TOSTR((int)res));
            return ROCDEC_RUNTIME_ERROR;
        }

        void* tgpu_mem_obj = malloc(tgpu_mem_size);
        Pal::IGpuMemory* tgpu_mem = nullptr;
        res = device_->CreateGpuMemory(tmem_ci, tgpu_mem_obj, &tgpu_mem);
        if (Util::IsErrorResult(res) || !tgpu_mem) {
            free(tgpu_mem_obj);
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: CreateGpuMemory (target) failed result=") + ROCDEC_TOSTR((int)res));
            return ROCDEC_RUNTIME_ERROR;
        }

        *slot.target_image_memory.PtrAddr() = tgpu_mem;
        *slot.target_image_memory.MemAddr() = tgpu_mem_obj;

        res = timg->BindGpuMemory(tgpu_mem, 0);
        if (Util::IsErrorResult(res)) {
            CriticalLog(g_rocdec_logger, "PAL: BindGpuMemory (target) failed");
            return ROCDEC_RUNTIME_ERROR;
        }

        Pal::GpuMemoryRef tmem_ref = {};
        tmem_ref.pGpuMemory = tgpu_mem;
        Pal::Result tadd_res = device_->AddGpuMemoryReferences(1, &tmem_ref, nullptr, 0);
        if (Util::IsErrorResult(tadd_res)) {
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: AddGpuMemoryReferences (target) failed with result ") +
                        ROCDEC_TOSTR((int)tadd_res));
            return ROCDEC_RUNTIME_ERROR;
        }
    }

    slot.image_index = 0;

    // Create a fence for this slot so GetDecodeStatus/SyncSurface can track
    // when a decode targeting this slot has completed, independently of other slots.
    {
        Pal::FenceCreateInfo fci = {};
        fci.flags.signaled = 1; // Start signaled — no pending work yet

        size_t fence_sz = device_->GetFenceSize(&res);
        if (Util::IsErrorResult(res)) {
            CriticalLog(g_rocdec_logger, "PAL: GetFenceSize failed for DPB slot fence");
            return ROCDEC_RUNTIME_ERROR;
        }

        void* fence_mem = malloc(fence_sz);
        Pal::IFence* f = nullptr;
        res = device_->CreateFence(fci, fence_mem, &f);
        if (Util::IsErrorResult(res) || !f) {
            free(fence_mem);
            CriticalLog(g_rocdec_logger, "PAL: CreateFence failed for DPB slot");
            return ROCDEC_RUNTIME_ERROR;
        }

        *slot.fence.PtrAddr() = f;
        *slot.fence.MemAddr() = fence_mem;
    }

    return ROCDEC_SUCCESS;
}

rocDecStatus PalVideoDecoder::AllocateBitstreamBuffer(size_t capacity) {
    Pal::Result res;

    // Unmap and release old buffer if exists
    if (bitstream_.gpu_memory.Get()) {
        if (bitstream_.mapped_ptr) {
            bitstream_.gpu_memory->Unmap();
            bitstream_.mapped_ptr = nullptr;
        }
        bitstream_.gpu_memory.Reset();
    }

    Pal::GpuMemoryCreateInfo mem_ci = {};
    mem_ci.size = capacity;
    mem_ci.alignment = 256;
    mem_ci.vaRange = Pal::VaRange::Default;
    mem_ci.priority = Pal::GpuMemPriority::Normal;
    mem_ci.heapCount = 1;
    mem_ci.heaps[0] = Pal::GpuHeapGartUswc;
    // TODO: PAL version mismatch - cpuVisible not found
    // mem_ci.flags.cpuVisible = 1;

    size_t obj_size = device_->GetGpuMemorySize(mem_ci, &res);
    if (Util::IsErrorResult(res)) {
        CriticalLog(g_rocdec_logger, "PAL: GetGpuMemorySize (bitstream) failed");
        return ROCDEC_RUNTIME_ERROR;
    }

    void* obj_mem = malloc(obj_size);
    Pal::IGpuMemory* gpu_mem = nullptr;
    res = device_->CreateGpuMemory(mem_ci, obj_mem, &gpu_mem);
    if (Util::IsErrorResult(res) || !gpu_mem) {
        free(obj_mem);
        CriticalLog(g_rocdec_logger, "PAL: CreateGpuMemory (bitstream) failed");
        return ROCDEC_RUNTIME_ERROR;
    }

    *bitstream_.gpu_memory.PtrAddr() = gpu_mem;
    *bitstream_.gpu_memory.MemAddr() = obj_mem;
    bitstream_.capacity = capacity;
    bitstream_.used = 0;

    // Persistent CPU mapping
    res = gpu_mem->Map(&bitstream_.mapped_ptr);
    if (Util::IsErrorResult(res)) {
        CriticalLog(g_rocdec_logger, "PAL: Map (bitstream) failed");
        return ROCDEC_RUNTIME_ERROR;
    }

    // Register with WDDM VidMm for GPU residency
    if (device_) {
        Pal::GpuMemoryRef mem_ref = {};
        mem_ref.pGpuMemory = gpu_mem;
        mem_ref.flags.readOnly = 1;  // GPU only reads the bitstream
        Pal::Result add_res = device_->AddGpuMemoryReferences(1, &mem_ref, nullptr, 0);
        if (Util::IsErrorResult(add_res)) {
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: AddGpuMemoryReferences (bitstream) failed with result ") +
                        ROCDEC_TOSTR((int)add_res));
            return ROCDEC_RUNTIME_ERROR;
        }
    }

    return ROCDEC_SUCCESS;
}

rocDecStatus PalVideoDecoder::UploadBitstream(const uint8_t* data, size_t size) {
    if (!bitstream_.gpu_memory.Get() || !bitstream_.mapped_ptr) {
        CriticalLog(g_rocdec_logger, "PAL: Bitstream buffer not allocated");
        return ROCDEC_NOT_INITIALIZED;
    }

    if (size > bitstream_.capacity) {
        CriticalLog(g_rocdec_logger, "PAL: Bitstream too large for buffer");
        return ROCDEC_OUTOF_MEMORY;
    }

    // Copy bitstream data to mapped GPU memory
    memcpy(bitstream_.mapped_ptr, data, size);
    bitstream_.used = size;

    return ROCDEC_SUCCESS;
}

rocDecStatus PalVideoDecoder::DecodeFrame(RocdecPicParams* pic_params) {
    if (!initialized_ || !pic_params) {
        return ROCDEC_INVALID_PARAMETER;
    }

    // Dispatch to codec-specific decode
    switch (codec_) {
        case rocDecVideoCodec_AVC:
            return DecodeH264(static_cast<RocdecPicParams*>(pic_params));
        case rocDecVideoCodec_HEVC:
            return DecodeHEVC(static_cast<RocdecPicParams*>(pic_params));
        case rocDecVideoCodec_AV1:
            return DecodeAV1(static_cast<RocdecPicParams*>(pic_params));
        case rocDecVideoCodec_VP9:
            return DecodeVP9(static_cast<RocdecPicParams*>(pic_params));
        default:
            return ROCDEC_NOT_SUPPORTED;
    }
}

rocDecStatus PalVideoDecoder::DecodeH264(RocdecPicParams* params) {
    (void)params; // Suppress unused parameter warning
    // TODO: Implement H.264 decode
    CriticalLog(g_rocdec_logger, "PAL: H.264 decode not yet implemented");
    return ROCDEC_NOT_IMPLEMENTED;
}

rocDecStatus PalVideoDecoder::DecodeHEVC(RocdecPicParams* params) {
    if (!params || params->num_slices == 0) {
        CriticalLog(g_rocdec_logger, "PAL: Invalid HEVC parameters");
        return ROCDEC_INVALID_PARAMETER;
    }

    const RocdecHevcPicParams* hevc = &params->pic_params.hevc;

    // Map rocDecode HEVC parameters to PAL/UVD format
    hevc_t pal_hevc = {};

    // SPS flags
    pal_hevc.sps_info_flags = 0;
    if (hevc->pic_fields.bits.scaling_list_enabled_flag)
        pal_hevc.sps_info_flags |= (1 << 0);  // scalingListEnabled
    if (hevc->pic_fields.bits.amp_enabled_flag)
        pal_hevc.sps_info_flags |= (1 << 1);  // ampEnabled
    if (hevc->slice_parsing_fields.bits.sample_adaptive_offset_enabled_flag)
        pal_hevc.sps_info_flags |= (1 << 2);  // sampleAdaptiveOffsetEnabled
    if (hevc->pic_fields.bits.pcm_enabled_flag)
        pal_hevc.sps_info_flags |= (1 << 3);  // pcmEnabled
    if (hevc->pic_fields.bits.pcm_loop_filter_disabled_flag)
        pal_hevc.sps_info_flags |= (1 << 4);  // pcmLoopFilterDisabled
    if (hevc->slice_parsing_fields.bits.long_term_ref_pics_present_flag)
        pal_hevc.sps_info_flags |= (1 << 5);  // longTermRefPicsPresent
    if (hevc->slice_parsing_fields.bits.sps_temporal_mvp_enabled_flag)
        pal_hevc.sps_info_flags |= (1 << 6);  // spsTemporalMvpEnabled
    if (hevc->pic_fields.bits.strong_intra_smoothing_enabled_flag)
        pal_hevc.sps_info_flags |= (1 << 7);  // strongIntraSmoothingEnabled
    if (hevc->pic_fields.bits.separate_colour_plane_flag)
        pal_hevc.sps_info_flags |= (1 << 8);  // separateColourPlane

    // PPS flags
    pal_hevc.pps_info_flags = 0;
    if (hevc->slice_parsing_fields.bits.dependent_slice_segments_enabled_flag)
        pal_hevc.pps_info_flags |= (1 << 0);  // dependentSliceSegmentsEnabled
    if (hevc->slice_parsing_fields.bits.output_flag_present_flag)
        pal_hevc.pps_info_flags |= (1 << 1);  // outputFlagPresent
    if (hevc->pic_fields.bits.sign_data_hiding_enabled_flag)
        pal_hevc.pps_info_flags |= (1 << 2);  // signDataHidingEnable
    if (hevc->slice_parsing_fields.bits.cabac_init_present_flag)
        pal_hevc.pps_info_flags |= (1 << 3);  // cabacInitPresent
    if (hevc->pic_fields.bits.constrained_intra_pred_flag)
        pal_hevc.pps_info_flags |= (1 << 4);  // constrainedIntraPred
    if (hevc->pic_fields.bits.transform_skip_enabled_flag)
        pal_hevc.pps_info_flags |= (1 << 5);  // transformSkipEnabled
    if (hevc->pic_fields.bits.cu_qp_delta_enabled_flag)
        pal_hevc.pps_info_flags |= (1 << 6);  // cuQpDeltaEnabled
    if (hevc->slice_parsing_fields.bits.pps_slice_chroma_qp_offsets_present_flag)
        pal_hevc.pps_info_flags |= (1 << 7);  // ppsSliceChromaQpOffsetsPresent
    if (hevc->pic_fields.bits.weighted_pred_flag)
        pal_hevc.pps_info_flags |= (1 << 8);  // weightedPred
    if (hevc->pic_fields.bits.weighted_bipred_flag)
        pal_hevc.pps_info_flags |= (1 << 9);  // weightedBiPred
    if (hevc->pic_fields.bits.transquant_bypass_enabled_flag)
        pal_hevc.pps_info_flags |= (1 << 10); // transQuantBypassEnabled
    if (hevc->pic_fields.bits.tiles_enabled_flag)
        pal_hevc.pps_info_flags |= (1 << 11); // tilesEnabled
    if (hevc->pic_fields.bits.entropy_coding_sync_enabled_flag)
        pal_hevc.pps_info_flags |= (1 << 12); // entropyCodingSyncEnabled
    // uniformSpacing flag not in rocDecode params, assume false
    if (hevc->pic_fields.bits.loop_filter_across_tiles_enabled_flag)
        pal_hevc.pps_info_flags |= (1 << 14); // loopFilterAcrossTilesEnabled
    if (hevc->pic_fields.bits.pps_loop_filter_across_slices_enabled_flag)
        pal_hevc.pps_info_flags |= (1 << 15); // ppsLoopFilterAcrossSlicesEnabled
    if (hevc->slice_parsing_fields.bits.deblocking_filter_override_enabled_flag)
        pal_hevc.pps_info_flags |= (1 << 16); // deblockingFilterOverrideEnabled
    if (hevc->slice_parsing_fields.bits.pps_disable_deblocking_filter_flag)
        pal_hevc.pps_info_flags |= (1 << 17); // ppsDeblockingFilterDisabled
    if (hevc->slice_parsing_fields.bits.lists_modification_present_flag)
        pal_hevc.pps_info_flags |= (1 << 18); // listsModificationPresent
    if (hevc->slice_parsing_fields.bits.slice_segment_header_extension_present_flag)
        pal_hevc.pps_info_flags |= (1 << 19); // sliceSegmentHeaderExtensionPresent

    // SPS/PPS parameters
    pal_hevc.chroma_format = hevc->pic_fields.bits.chroma_format_idc;
    pal_hevc.bit_depth_luma_minus8 = hevc->bit_depth_luma_minus8;
    pal_hevc.bit_depth_chroma_minus8 = hevc->bit_depth_chroma_minus8;
    pal_hevc.log2_max_pic_order_cnt_lsb_minus4 = hevc->log2_max_pic_order_cnt_lsb_minus4;
    pal_hevc.sps_max_dec_pic_buffering_minus1 = hevc->sps_max_dec_pic_buffering_minus1;
    pal_hevc.log2_min_luma_coding_block_size_minus3 = hevc->log2_min_luma_coding_block_size_minus3;
    pal_hevc.log2_diff_max_min_luma_coding_block_size = hevc->log2_diff_max_min_luma_coding_block_size;
    pal_hevc.log2_min_transform_block_size_minus2 = hevc->log2_min_luma_transform_block_size_minus2;
    pal_hevc.log2_diff_max_min_transform_block_size = hevc->log2_diff_max_min_luma_transform_block_size;
    pal_hevc.max_transform_hierarchy_depth_inter = hevc->max_transform_hierarchy_depth_inter;
    pal_hevc.max_transform_hierarchy_depth_intra = hevc->max_transform_hierarchy_depth_intra;
    pal_hevc.pcm_sample_bit_depth_luma_minus1 = hevc->pcm_sample_bit_depth_luma_minus1;
    pal_hevc.pcm_sample_bit_depth_chroma_minus1 = hevc->pcm_sample_bit_depth_chroma_minus1;
    pal_hevc.log2_min_pcm_luma_coding_block_size_minus3 = hevc->log2_min_pcm_luma_coding_block_size_minus3;
    pal_hevc.log2_diff_max_min_pcm_luma_coding_block_size = hevc->log2_diff_max_min_pcm_luma_coding_block_size;
    pal_hevc.num_extra_slice_header_bits = hevc->num_extra_slice_header_bits;
    pal_hevc.num_short_term_ref_pic_sets = hevc->num_short_term_ref_pic_sets;
    pal_hevc.num_long_term_ref_pic_sps = hevc->num_long_term_ref_pic_sps;
    pal_hevc.num_ref_idx_l0_default_active_minus1 = hevc->num_ref_idx_l0_default_active_minus1;
    pal_hevc.num_ref_idx_l1_default_active_minus1 = hevc->num_ref_idx_l1_default_active_minus1;
    pal_hevc.pps_cb_qp_offset = hevc->pps_cb_qp_offset;
    pal_hevc.pps_cr_qp_offset = hevc->pps_cr_qp_offset;
    pal_hevc.pps_beta_offset_div2 = hevc->pps_beta_offset_div2;
    pal_hevc.pps_tc_offset_div2 = hevc->pps_tc_offset_div2;
    pal_hevc.diff_cu_qp_delta_depth = hevc->diff_cu_qp_delta_depth;
    pal_hevc.num_tile_columns_minus1 = hevc->num_tile_columns_minus1;
    pal_hevc.num_tile_rows_minus1 = hevc->num_tile_rows_minus1;
    pal_hevc.log2_parallel_merge_level_minus2 = hevc->log2_parallel_merge_level_minus2;
    pal_hevc.init_qp_minus26 = hevc->init_qp_minus26;
    pal_hevc.st_rps_bits = hevc->st_rps_bits;

    // Tile dimensions
    for (size_t i = 0; i < 19; i++) {
        pal_hevc.column_width_minus1[i] = hevc->column_width_minus1[i];
    }
    for (size_t i = 0; i < 21; i++) {
        pal_hevc.row_height_minus1[i] = hevc->row_height_minus1[i];
    }

    // Current picture POC
    pal_hevc.curr_poc = hevc->curr_pic.poc;
    pal_hevc.curr_idx = static_cast<unsigned char>(params->curr_pic_idx);

    // Reference picture list — hevc_t.ref_pic_list[16], rocDecode provides 15 ref slots.
    // Fill all 16 entries; the 16th (index 15) has no rocDecode counterpart so mark invalid.
    for (size_t i = 0; i < 15; i++) {
        if (hevc->ref_frames[i].pic_idx != 0xFF && hevc->ref_frames[i].pic_idx >= 0) {
            pal_hevc.ref_pic_list[i] = static_cast<unsigned char>(hevc->ref_frames[i].pic_idx);
            pal_hevc.poc_list[i] = hevc->ref_frames[i].poc;
        } else {
            pal_hevc.ref_pic_list[i] = 0xFF;  // Invalid
            pal_hevc.poc_list[i] = 0;
        }
    }
    pal_hevc.ref_pic_list[15] = 0xFF;  // No rocDecode counterpart — must be explicitly invalid

    // 10-bit mode
    if (hevc->bit_depth_luma_minus8 == 2) {
        pal_hevc.p010_mode = 1;
        pal_hevc.luma_10to8 = 0;
        pal_hevc.chroma_10to8 = 0;
    } else {
        pal_hevc.p010_mode = 0;
    }

    // TODO: Fill in remaining fields (direct_reflist, scaling lists, etc.)

    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: HEVC decode - POC=") + ROCDEC_TOSTR(pal_hevc.curr_poc) +
            " size=" + ROCDEC_TOSTR(hevc->picture_width_in_luma_samples) + "x" +
            ROCDEC_TOSTR(hevc->picture_height_in_luma_samples));

    // Get current DPB slot for output
    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Getting DPB slot for curr_pic_idx=") + ROCDEC_TOSTR(params->curr_pic_idx));
    PalDpbSlot* output_slot = dpb_.GetSlot(params->curr_pic_idx);
    if (!output_slot || !output_slot->image.Get() || !output_slot->target_image.Get()) {
        CriticalLog(g_rocdec_logger, "PAL: Invalid output slot");
        return ROCDEC_INVALID_PARAMETER;
    }
    InfoLog(g_rocdec_logger, "PAL: DPB slot acquired");

    // Build PAL decode frame info
    Pal::VideoCodecInfo codec_info = {};
    memcpy(&codec_info.hevcCodecData, &pal_hevc, sizeof(hevc_t));

    Pal::VideoDecodeFrameInfo decode_info = {};

    // Decode type
    decode_info.decodeType = (bit_depth_ == 10) ? Pal::VideoDecodeType::Hevc10Bit : Pal::VideoDecodeType::Hevc;

    // Source extent
    decode_info.srcExtent.width = hevc->picture_width_in_luma_samples;
    decode_info.srcExtent.height = hevc->picture_height_in_luma_samples;

    // Bitstream buffer
    decode_info.pBitstreamBuffer = bitstream_.gpu_memory.Get();
    decode_info.bitstreamBufferSize = params->bitstream_data_len;
    decode_info.bitstreamBufferOffset = 0;

    // Decode target = linear display copy (exported to HIP). DPB curr/refs below use the
    // tiled output_slot->image, which VCN reconstructs into and reads references from.
    decode_info.pDecodeTargetBuffer = output_slot->target_image.Get();
    decode_info.decodeTargetArraySlice = output_slot->image_index;

    // Codec info
    decode_info.pCodecInfoBuffer = &codec_info;

    // Decoder heap (VCN internal memory)
    decode_info.pDecoderHeapBuffer = decoder_heap_.Get();
    decode_info.decoderHeapOffset = 0;

    // DPB configuration
    decode_info.dpbArraySize = max_dpb_slots_;

    // Query device properties to determine DPB tier
    InfoLog(g_rocdec_logger, "PAL: Calling device_->GetProperties ...");
    Pal::DeviceProperties dev_props = {};
    device_->GetProperties(&dev_props);
    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: GetProperties done - supportUnifiedDecodeTarget=") +
            ROCDEC_TOSTR(dev_props.vcnipProperties.flags.supportUnifiedDecodeTarget) +
            " supportArrayOfTextures=" + ROCDEC_TOSTR(dev_props.vcnipProperties.flags.supportArrayOfTextures));

    if (dev_props.vcnipProperties.flags.supportUnifiedDecodeTarget) {
        // Tier 2: Array of textures (preferred)
        decode_info.dpbConfig.dynamicDpbTier1 = 0;
        decode_info.dpbConfig.dynamicDpbTier2 = 1;
        decode_info.dpbConfig.dynamicDpbTier3 = 0;
    } else if (dev_props.vcnipProperties.flags.supportArrayOfTextures) {
        // Tier 2: Array of textures
        decode_info.dpbConfig.dynamicDpbTier1 = 0;
        decode_info.dpbConfig.dynamicDpbTier2 = 1;
        decode_info.dpbConfig.dynamicDpbTier3 = 0;
    } else {
        // Tier 1: Single texture array
        decode_info.dpbConfig.dynamicDpbTier1 = 1;
        decode_info.dpbConfig.dynamicDpbTier2 = 0;
        decode_info.dpbConfig.dynamicDpbTier3 = 0;
    }

    if (decode_info.dpbConfig.dynamicDpbTier2) {
        // Tier 2: Separate current and reference buffers
        decode_info.pDpbCurrBuffer = output_slot->image.Get();
        decode_info.dpbCurArraySlice = output_slot->image_index;

        // Initialize reference frames array
        for (uint32_t i = 0; i < Pal::MaxDpbSliceCount; i++) {
            decode_info.dpbRefArraySlices[i] = 0xFF;  // Mark as invalid
            decode_info.pDpbRefBuffers[i] = nullptr;
        }

        // Map reference frames using the reference list from codec parameters
        for (uint32_t i = 0; i < Pal::MaxDpbSliceCount; i++) {
            uint8_t pic_entry = pal_hevc.ref_pic_list[i] & 0x7F;

            if (pic_entry != 0x7F && pic_entry < max_dpb_slots_) {
                PalDpbSlot* ref_slot = dpb_.GetSlot(pic_entry);
                if (ref_slot && ref_slot->image.Get()) {
                    decode_info.pDpbRefBuffers[i] = ref_slot->image.Get();
                    decode_info.dpbRefArraySlices[i] = ref_slot->image_index;
                }
            }
        }
    } else {
        // Tier 1: Single DPB array
        decode_info.pDpbCurrBuffer = output_slot->image.Get();
        decode_info.dpbCurArraySlice = output_slot->image_index;

        for (uint32_t i = 0; i < Pal::MaxDpbSliceCount; i++) {
            decode_info.pDpbRefBuffers[i] = nullptr;
            if (i < max_dpb_slots_) {
                PalDpbSlot* slot = dpb_.GetSlot(i);
                if (slot) {
                    decode_info.dpbRefArraySlices[i] = slot->image_index;
                } else {
                    decode_info.dpbRefArraySlices[i] = 0xFF;
                }
            } else {
                decode_info.dpbRefArraySlices[i] = 0xFF;
            }
        }
    }

    // Wait for the previous frame's slot fence before recycling the shared cmd_allocator.
    // The GPU queue serializes execution, but cmd_allocator_->Reset frees CPU-side memory
    // that the GPU may still be reading for the previous submission.
    if (last_submitted_slot_idx_ >= 0) {
        PalDpbSlot* prev_slot = dpb_.GetSlot(last_submitted_slot_idx_);
        if (prev_slot && prev_slot->fence.Get()) {
            Pal::Result fence_status = prev_slot->fence->GetStatus();
            if (fence_status == Pal::Result::NotReady) {
                InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Waiting for slot ") +
                        ROCDEC_TOSTR(last_submitted_slot_idx_) + " fence before cmd_buffer Reset ...");
                const Pal::IFence* wait_fences[] = { prev_slot->fence.Get() };
                Pal::Result wait_res = device_->WaitForFences(1, wait_fences, true,
                                                              std::chrono::nanoseconds(5000000000LL));
                if (Util::IsErrorResult(wait_res)) {
                    CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: WaitForFences failed with result ") +
                                ROCDEC_TOSTR((int)wait_res));
                    return ROCDEC_RUNTIME_ERROR;
                }
            }
        }
    }

    // Upload bitstream after the fence wait so Frame N-1's GPU read of the shared
    // bitstream buffer has completed before Frame N overwrites it.
    {
        size_t bitstream_size = params->bitstream_data_len;
        if (bitstream_size > bitstream_.capacity) {
            rocDecStatus alloc_status = AllocateBitstreamBuffer(bitstream_size * 2);
            if (alloc_status != ROCDEC_SUCCESS) {
                return alloc_status;
            }
        }
        rocDecStatus upload_status = UploadBitstream(
            static_cast<const uint8_t*>(params->bitstream_data),
            bitstream_size
        );
        if (upload_status != ROCDEC_SUCCESS) {
            return upload_status;
        }
    }

    // Reset and begin command buffer
    InfoLog(g_rocdec_logger, "PAL: Calling cmd_buffer_->Reset ...");
    Pal::Result res = cmd_buffer_->Reset(cmd_allocator_.Get(), true);
    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: cmd_buffer_->Reset returned ") + ROCDEC_TOSTR((int)res));
    if (Util::IsErrorResult(res)) {
        CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: CmdBuffer Reset failed with result ") + ROCDEC_TOSTR((int)res));
        return ROCDEC_RUNTIME_ERROR;
    }

    Pal::CmdBufferBuildInfo build_info = {};
    InfoLog(g_rocdec_logger, "PAL: Calling cmd_buffer_->Begin ...");
    res = cmd_buffer_->Begin(build_info);
    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: cmd_buffer_->Begin returned ") + ROCDEC_TOSTR((int)res));
    if (Util::IsErrorResult(res)) {
        CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: CmdBuffer Begin failed with result ") + ROCDEC_TOSTR((int)res));
        return ROCDEC_RUNTIME_ERROR;
    }

    InfoLog(g_rocdec_logger, "PAL: Calling CmdBindVideoDecoder ...");
    cmd_buffer_->CmdBindVideoDecoder(*video_decoder_.Get());

    InfoLog(g_rocdec_logger, "PAL: Calling CmdBeginVideoDecode ...");
    Pal::VideoDecodeBeginInfo begin_info = {};
    begin_info.srcExtent.width = hevc->picture_width_in_luma_samples;
    begin_info.srcExtent.height = hevc->picture_height_in_luma_samples;
    cmd_buffer_->CmdBeginVideoDecode(begin_info);
    InfoLog(g_rocdec_logger, "PAL: CmdBeginVideoDecode done");

    // Submit decode frame command
    // --- Frame parameter dump for submission debugging ---
    {
        static int s_frame_idx = 0;
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: === FRAME ") + ROCDEC_TOSTR(s_frame_idx) + " DECODE_INFO DUMP ===");
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL:   curr_pic_idx=") + ROCDEC_TOSTR(params->curr_pic_idx) +
                " last_submitted_slot_idx=" + ROCDEC_TOSTR(last_submitted_slot_idx_));
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL:   decodeType=") + ROCDEC_TOSTR((int)decode_info.decodeType) +
                " srcExtent=" + ROCDEC_TOSTR(decode_info.srcExtent.width) + "x" + ROCDEC_TOSTR(decode_info.srcExtent.height));
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL:   bitstreamBufferSize=") + ROCDEC_TOSTR(decode_info.bitstreamBufferSize) +
                " bitstreamBufferOffset=" + ROCDEC_TOSTR(decode_info.bitstreamBufferOffset) +
                " pBitstreamBuffer=" + ROCDEC_TOSTR((uintptr_t)decode_info.pBitstreamBuffer));
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL:   pDecodeTargetBuffer=") + ROCDEC_TOSTR((uintptr_t)decode_info.pDecodeTargetBuffer) +
                " decodeTargetArraySlice=" + ROCDEC_TOSTR(decode_info.decodeTargetArraySlice));
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL:   pDpbCurrBuffer=") + ROCDEC_TOSTR((uintptr_t)decode_info.pDpbCurrBuffer) +
                " dpbCurArraySlice=" + ROCDEC_TOSTR(decode_info.dpbCurArraySlice) +
                " dpbArraySize=" + ROCDEC_TOSTR(decode_info.dpbArraySize));
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL:   dpbTier1=") + ROCDEC_TOSTR(decode_info.dpbConfig.dynamicDpbTier1) +
                " tier2=" + ROCDEC_TOSTR(decode_info.dpbConfig.dynamicDpbTier2) +
                " tier3=" + ROCDEC_TOSTR(decode_info.dpbConfig.dynamicDpbTier3));
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL:   pDecoderHeapBuffer=") + ROCDEC_TOSTR((uintptr_t)decode_info.pDecoderHeapBuffer) +
                " decoderHeapOffset=" + ROCDEC_TOSTR(decode_info.decoderHeapOffset));
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL:   pCodecInfoBuffer=") + ROCDEC_TOSTR((uintptr_t)decode_info.pCodecInfoBuffer));
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL:   output_slot->fence=") + ROCDEC_TOSTR((uintptr_t)output_slot->fence.Get()) +
                " fence_status=" + ROCDEC_TOSTR((int)output_slot->fence->GetStatus()));

        // Dump reference buffer pointers to catch null/invalid refs
        for (uint32_t i = 0; i < Pal::MaxDpbSliceCount; i++) {
            if (decode_info.pDpbRefBuffers[i] != nullptr || decode_info.dpbRefArraySlices[i] != 0xFF) {
                InfoLog(g_rocdec_logger, ROCDEC_STR("PAL:   ref[") + ROCDEC_TOSTR(i) + "]=" +
                        ROCDEC_TOSTR((uintptr_t)decode_info.pDpbRefBuffers[i]) +
                        " slice=" + ROCDEC_TOSTR((int)decode_info.dpbRefArraySlices[i]));
            }
        }

        // Dump HEVC codec params that differ most between frames
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL:   hevc.curr_poc=") + ROCDEC_TOSTR(pal_hevc.curr_poc) +
                " curr_idx=" + ROCDEC_TOSTR((int)pal_hevc.curr_idx) +
                " sps_info_flags=" + ROCDEC_TOSTR(pal_hevc.sps_info_flags) +
                " pps_info_flags=" + ROCDEC_TOSTR(pal_hevc.pps_info_flags));
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL:   ref_pic_list[0..3]=") +
                ROCDEC_TOSTR((int)pal_hevc.ref_pic_list[0]) + "," +
                ROCDEC_TOSTR((int)pal_hevc.ref_pic_list[1]) + "," +
                ROCDEC_TOSTR((int)pal_hevc.ref_pic_list[2]) + "," +
                ROCDEC_TOSTR((int)pal_hevc.ref_pic_list[3]));
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL:   poc_list[0..3]=") +
                ROCDEC_TOSTR(pal_hevc.poc_list[0]) + "," +
                ROCDEC_TOSTR(pal_hevc.poc_list[1]) + "," +
                ROCDEC_TOSTR(pal_hevc.poc_list[2]) + "," +
                ROCDEC_TOSTR(pal_hevc.poc_list[3]));
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: === END FRAME ") + ROCDEC_TOSTR(s_frame_idx) + " DUMP ===");
        s_frame_idx++;
    }
    cmd_buffer_->CmdDecodeVideoFrame(decode_info);

    // End command buffer
    res = cmd_buffer_->End();
    if (Util::IsErrorResult(res)) {
        CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: CmdBuffer End failed with result ") + ROCDEC_TOSTR((int)res));
        return ROCDEC_RUNTIME_ERROR;
    }

    // Reset this slot's fence so it can be signaled when this decode completes.
    // The queue serializes GPU work, so resetting after the previous submit for a
    // different slot is safe — each slot has its own independent fence.
    {
        Pal::IFence* fences_to_reset[] = { output_slot->fence.Get() };
        res = device_->ResetFences(1, fences_to_reset);
        if (Util::IsErrorResult(res)) {
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: ResetFences failed with result ") + ROCDEC_TOSTR((int)res));
            return ROCDEC_RUNTIME_ERROR;
        }
    }

    // Submit to queue, attaching this slot's fence
    Pal::ICmdBuffer* cmd_bufs[] = { cmd_buffer_.Get() };
    Pal::PerSubQueueSubmitInfo per_queue_info = {};
    per_queue_info.cmdBufferCount = 1;
    per_queue_info.ppCmdBuffers = cmd_bufs;

    Pal::IFence* submit_fence = output_slot->fence.Get();
    Pal::MultiSubmitInfo submit_info = {};
    submit_info.perSubQueueInfoCount = 1;
    submit_info.pPerSubQueueInfo = &per_queue_info;
    submit_info.ppFences = &submit_fence;
    submit_info.fenceCount = 1;

    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Calling video_queue_->Submit - cmdBufCount=") +
            ROCDEC_TOSTR(per_queue_info.cmdBufferCount) +
            " fenceCount=" + ROCDEC_TOSTR(submit_info.fenceCount) +
            " fence=" + ROCDEC_TOSTR((uintptr_t)submit_fence));
    res = video_queue_->Submit(submit_info);
    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: video_queue_->Submit returned ") + ROCDEC_TOSTR((int)res));
    if (Util::IsErrorResult(res)) {
        CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: Queue Submit failed with result ") + ROCDEC_TOSTR((int)res) +
                    " curr_pic_idx=" + ROCDEC_TOSTR(params->curr_pic_idx) +
                    " last_submitted_slot_idx=" + ROCDEC_TOSTR(last_submitted_slot_idx_));
        return ROCDEC_RUNTIME_ERROR;
    }

    last_submitted_slot_idx_ = params->curr_pic_idx;
    InfoLog(g_rocdec_logger, "PAL: HEVC frame decode submitted successfully");
    return ROCDEC_SUCCESS;
}

rocDecStatus PalVideoDecoder::DecodeAV1(RocdecPicParams* params) {
    (void)params; // Suppress unused parameter warning
    // TODO: Implement AV1 decode
    CriticalLog(g_rocdec_logger, "PAL: AV1 decode not yet implemented");
    return ROCDEC_NOT_IMPLEMENTED;
}

rocDecStatus PalVideoDecoder::DecodeVP9(RocdecPicParams* params) {
    (void)params; // Suppress unused parameter warning
    // TODO: Implement VP9 decode
    CriticalLog(g_rocdec_logger, "PAL: VP9 decode not yet implemented");
    return ROCDEC_NOT_IMPLEMENTED;
}

rocDecStatus PalVideoDecoder::GetDecodeStatus(int pic_idx, RocdecDecodeStatus* dec_pic) {
    if (!dec_pic || pic_idx < 0) {
        return ROCDEC_INVALID_PARAMETER;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        dec_pic->decode_status = static_cast<rocDecDecodeStatus>(0);
        dec_pic->reserved[0] = 0;
        return ROCDEC_NOT_INITIALIZED;
    }

    // Check if picture index is valid
    PalDpbSlot* slot = dpb_.GetSlot(pic_idx);
    if (!slot || !slot->occupied || !slot->fence.Get()) {
        dec_pic->decode_status = static_cast<rocDecDecodeStatus>(0);  // Invalid
        dec_pic->reserved[0] = 0;
        return ROCDEC_INVALID_PARAMETER;
    }

    // Non-blocking poll on this slot's fence — mirrors vaQuerySurfaceStatus
    Pal::Result res = slot->fence->GetStatus();

    if (res == Pal::Result::Success) {
        // Fence is signaled - decode complete
        dec_pic->decode_status = static_cast<rocDecDecodeStatus>(1);  // Success
        dec_pic->reserved[0] = 0;
    } else if (res == Pal::Result::NotReady) {
        // Decode in progress
        dec_pic->decode_status = static_cast<rocDecDecodeStatus>(2);  // In progress
        dec_pic->reserved[0] = 0;
    } else {
        // Error
        dec_pic->decode_status = static_cast<rocDecDecodeStatus>(0);  // Invalid/error
        dec_pic->reserved[0] = 0;
        ErrorLog(g_rocdec_logger, ROCDEC_STR("PAL: GetStatus failed with result ") + ROCDEC_TOSTR((int)res));
    }

    return ROCDEC_SUCCESS;
}

rocDecStatus PalVideoDecoder::Reconfigure(uint32_t width, uint32_t height) {
    std::lock_guard<std::mutex> lock(mutex_);

    // TODO: Implement reconfiguration
    max_coded_width_ = width;
    max_coded_height_ = height;

    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Reconfigure to ") +
            ROCDEC_TOSTR(width) + "x" + ROCDEC_TOSTR(height));

    return ROCDEC_SUCCESS;
}

void PalVideoDecoder::Destroy() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return;
    }

    // Wait for any in-flight decode to finish before tearing down resources.
    // Only the last submitted slot can have a pending fence; waiting on it is sufficient.
    if (device_ && last_submitted_slot_idx_ >= 0) {
        PalDpbSlot* slot = dpb_.GetSlot(last_submitted_slot_idx_);
        if (slot && slot->fence.Get()) {
            const Pal::IFence* fences[] = { slot->fence.Get() };
            device_->WaitForFences(1, fences, true, std::chrono::nanoseconds::max());
        }
    }

    // Unregister GPU memory from WDDM VidMm residency tracking before releasing
    {
        // Collect all registered allocations (two surfaces per DPB slot: tiled + linear)
        Pal::IGpuMemory* to_remove[2 + 2 * PalDpb::kMaxSlots] = {};
        uint32_t remove_count = 0;

        if (bitstream_.gpu_memory.Get())
            to_remove[remove_count++] = bitstream_.gpu_memory.Get();
        if (decoder_heap_.Get())
            to_remove[remove_count++] = decoder_heap_.Get();
        for (uint32_t i = 0; i < dpb_.GetSlotCount(); i++) {
            PalDpbSlot* slot = dpb_.GetSlot(static_cast<int32_t>(i));
            if (slot && slot->image_memory.Get())
                to_remove[remove_count++] = slot->image_memory.Get();
            if (slot && slot->target_image_memory.Get())
                to_remove[remove_count++] = slot->target_image_memory.Get();
        }

        if (remove_count > 0)
            device_->RemoveGpuMemoryReferences(remove_count, to_remove, nullptr);
    }

    // Unmap bitstream
    if (bitstream_.gpu_memory.Get() && bitstream_.mapped_ptr) {
        bitstream_.gpu_memory->Unmap();
        bitstream_.mapped_ptr = nullptr;
    }

    // Release resources (PalObject destructors handle this)
    bitstream_.gpu_memory.Reset();
    dpb_.Clear();  // Resets image, image_memory, and fence PalObjects for all slots
    decoder_heap_.Reset();
    cmd_buffer_.Reset();
    cmd_allocator_.Reset();
    video_queue_.Reset();
    video_decoder_.Reset();

    initialized_ = false;
    session_begun_ = false;
    last_submitted_slot_idx_ = -1;
    device_ = nullptr;

    ReleasePalPlatform();
}

rocDecStatus PalVideoDecoder::ExportSurface(int pic_idx,
                                             uint32_t& width,
                                             uint32_t& height,
                                             uint32_t pitch[3],
                                             uint32_t offset[3],
                                             uint32_t& num_planes,
                                             void*& kmt_handle,
                                             size_t& mem_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || pic_idx < 0) {
        return ROCDEC_INVALID_PARAMETER;
    }

    // Get DPB slot
    PalDpbSlot* slot = dpb_.GetSlot(pic_idx);
    if (!slot || !slot->target_image.Get() || !slot->target_image_memory.Get()) {
        CriticalLog(g_rocdec_logger, "PAL: Invalid DPB slot for export");
        return ROCDEC_INVALID_PARAMETER;
    }

    // Wait for the decode targeting this slot to complete before the CPU reads it.
    // PAL decode is submitted via a PAL queue; HIP/hipMemcpyDtoH has no visibility into
    // pending PAL work and will not wait automatically.
    if (slot->fence.Get()) {
        Pal::Result fence_status = slot->fence->GetStatus();
        if (fence_status == Pal::Result::NotReady) {
            InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: ExportSurface waiting for slot ") +
                    ROCDEC_TOSTR(pic_idx) + " fence ...");
            const Pal::IFence* wait_fences[] = { slot->fence.Get() };
            Pal::Result wait_res = device_->WaitForFences(1, wait_fences, true,
                                                          std::chrono::nanoseconds(5000000000LL));
            if (Util::IsErrorResult(wait_res)) {
                CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: ExportSurface WaitForFences failed result=") +
                            ROCDEC_TOSTR((int)wait_res));
                return ROCDEC_RUNTIME_ERROR;
            }
        }
    }

    // Export the linear target surface (not the tiled DPB) — this is the clean display
    // copy VCN wrote, and the only surface allocated interprocess=1 for KMT export.
    Pal::IImage* image = slot->target_image.Get();
    Pal::IGpuMemory* gpu_mem = slot->target_image_memory.Get();

    const Pal::ImageCreateInfo& create_info = image->GetImageCreateInfo();
    width  = create_info.extent.width;
    height = create_info.extent.height;

    // NV12/P010 has 2 planes (Y plane and UV interleaved plane)
    num_planes = 2;

    // Query subresource layout for each plane
    for (uint32_t plane = 0; plane < num_planes; plane++) {
        Pal::SubresId subres = {};
        subres.plane = static_cast<uint8_t>(plane);
        subres.mipLevel = 0;
        subres.arraySlice = slot->image_index;

        Pal::SubresLayout layout = {};
        Pal::Result res = image->GetSubresourceLayout(subres, &layout);
        if (Util::IsErrorResult(res)) {
            CriticalLog(g_rocdec_logger, ROCDEC_STR("PAL: GetSubresourceLayout failed for plane ") +
                        ROCDEC_TOSTR(plane) + " with result " + ROCDEC_TOSTR((int)res));
            return ROCDEC_RUNTIME_ERROR;
        }

        offset[plane] = static_cast<uint32_t>(layout.offset);
        pitch[plane] = static_cast<uint32_t>(layout.rowPitch);

        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Plane ") + ROCDEC_TOSTR(plane) +
                " offset=" + ROCDEC_TOSTR(offset[plane]) +
                " pitch=" + ROCDEC_TOSTR(pitch[plane]));
    }

    // Third plane unused for NV12/P010
    offset[2] = 0;
    pitch[2] = 0;

    // Get total memory size
    const Pal::GpuMemoryDesc& mem_desc = gpu_mem->Desc();
    mem_size = mem_desc.size;

    // Export GPU memory as KMT opaque handle (sharedViaNtHandle=0 path, uses m_hGlobalShare).
    // The caller must import with hipExternalMemoryHandleTypeOpaqueWin32Kmt (not the NT variant).
#if defined(PAL_KMT_BUILD)
    Pal::GpuMemoryExportInfo export_info = {};
    export_info.exportType        = Pal::ExportHandleType::Default;
    export_info.pSecurityAttributes = nullptr;
    export_info.pNtObjectName     = nullptr;
    export_info.accessFlags       = 0; // Unused for KMT (non-NT) path

    Pal::OsExternalHandle external_handle = gpu_mem->ExportExternalHandle(export_info);
    if (external_handle == nullptr) {
        CriticalLog(g_rocdec_logger, "PAL: ExportExternalHandle failed (is memory allocated with interprocess=1?)");
        return ROCDEC_RUNTIME_ERROR;
    }

    kmt_handle = external_handle;
    InfoLog(g_rocdec_logger, "PAL: Exported KMT opaque handle — import with hipExternalMemoryHandleTypeOpaqueWin32Kmt");
#else
    // PAL_KMT_BUILD not defined - export not available
    CriticalLog(g_rocdec_logger, "PAL: ExportExternalHandle not available (PAL_KMT_BUILD not defined)");
    return ROCDEC_NOT_SUPPORTED;
#endif

    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Exported surface pic_idx=") + ROCDEC_TOSTR(pic_idx) +
            " size=" + ROCDEC_TOSTR(width) + "x" + ROCDEC_TOSTR(height) +
            " mem_size=" + ROCDEC_TOSTR(mem_size));

    return ROCDEC_SUCCESS;
}

rocDecStatus PalVideoDecoder::QueryDecoderCaps(RocdecDecodeCaps* caps) {
    if (!caps) {
        return ROCDEC_INVALID_PARAMETER;
    }

    // Initialize output fields to zero
    caps->is_supported = 0;
    caps->num_decoders = 0;
    caps->output_format_mask = 0;
    caps->max_width = 0;
    caps->max_height = 0;
    caps->min_width = 0;
    caps->min_height = 0;

    // Get PAL device to query capabilities
    Pal::IDevice* device = GetPalDevice();
    if (!device) {
        CriticalLog(g_rocdec_logger, "PAL: Failed to get device for capability query");
        return ROCDEC_NOT_INITIALIZED;
    }

    Pal::DeviceProperties dev_props = {};
    device->GetProperties(&dev_props);

    // Check if VCN is available
    if (dev_props.vcnLevel == Pal::VcnIpLevel::None ||
        dev_props.vcnLevel == Pal::VcnIpLevel::_None) {
        InfoLog(g_rocdec_logger, "PAL: No VCN support, codec not supported");
        return ROCDEC_SUCCESS;
    }

    // Check engine availability
    bool has_vcn_decode = (dev_props.engineProperties[Pal::EngineTypeVcnDecode].engineCount > 0);
    bool has_vcn_unified = (dev_props.engineProperties[Pal::EngineTypeVcnUnified].engineCount > 0);

    if (!has_vcn_decode && !has_vcn_unified) {
        InfoLog(g_rocdec_logger, "PAL: No VCN decode engines available");
        return ROCDEC_SUCCESS;
    }

    // Map codec type to support (all modern VCN supports H.264, HEVC, VP9, AV1)
    bool codec_supported = false;
    if (caps->codec_type == rocDecVideoCodec_AVC ||   // H.264
        caps->codec_type == rocDecVideoCodec_HEVC ||  // H.265
        caps->codec_type == rocDecVideoCodec_VP9 ||
        caps->codec_type == rocDecVideoCodec_AV1) {
        codec_supported = true;
    } else {
        codec_supported = false;  // MPEG1/2/4, VP8, JPEG not supported on modern VCN
    }

    if (!codec_supported) {
        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Codec type ") +
                ROCDEC_TOSTR((int)caps->codec_type) + " not supported");
        return ROCDEC_SUCCESS;
    }

    // Check bit depth support (8-bit and 10-bit supported)
    if (caps->bit_depth_minus_8 > 2) {
        InfoLog(g_rocdec_logger, "PAL: Bit depth > 10 not supported");
        return ROCDEC_SUCCESS;
    }

    // Check chroma format (4:2:0 is primary, 4:4:4 may be supported)
    if (caps->chroma_format != rocDecVideoChromaFormat_420 &&
        caps->chroma_format != rocDecVideoChromaFormat_Monochrome) {
        InfoLog(g_rocdec_logger, "PAL: Only 4:2:0 chroma format fully supported");
        // Still mark as supported for 4:2:0
    }

    // Fill output caps
    caps->is_supported = 1;
    caps->num_decoders = has_vcn_decode ?
        dev_props.engineProperties[Pal::EngineTypeVcnDecode].engineCount :
        dev_props.engineProperties[Pal::EngineTypeVcnUnified].engineCount;

    // Output format mask (NV12 for 8-bit, P010 for 10-bit)
    if (caps->bit_depth_minus_8 == 0) {
        caps->output_format_mask = (1 << rocDecVideoSurfaceFormat_NV12);
    } else if (caps->bit_depth_minus_8 == 2) {
        caps->output_format_mask = (1 << rocDecVideoSurfaceFormat_P016);
    }

    // VCN decode resolution limits (conservative estimates based on VCN IP level)
    // These are typical limits - actual limits may vary by VCN generation
    caps->min_width = 64;
    caps->min_height = 64;

    // Max resolution (conservative 8K limit for all modern VCN)
    // Actual limits depend on VCN IP level, but 8K is safe for VCN 2.0+
    caps->max_width = 7680;
    caps->max_height = 4320;

    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Codec ") + ROCDEC_TOSTR((int)caps->codec_type) +
            " supported, max resolution " + ROCDEC_TOSTR(caps->max_width) + "x" +
            ROCDEC_TOSTR(caps->max_height));

    return ROCDEC_SUCCESS;
}

} // namespace rocdec

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif // ROCDECODE_BUILD_WINDOWS
