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
        dev->GetProperties(&dev_props);

        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Device ") + ROCDEC_TOSTR(i) +
                " VCN level=" + ROCDEC_TOSTR((int)dev_props.vcnLevel));

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

        res = dev->Finalize(fi);
        if (Util::IsErrorResult(res)) {
            continue;
        }

        device_instance_ = dev;
        video_decode_engine_ = video_engine;

        InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Device ") + ROCDEC_TOSTR(i) + " initialized with VCN support");
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
    , initialized_(false) {
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
        qci.queueType = Pal::QueueTypeUniversal /* TODO: QueueTypeVideoDecode */;

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
    InfoLog(g_rocdec_logger, "PAL: Creating command allocator...");
    {
        Pal::CmdAllocatorCreateInfo aci = {};
        aci.flags.threadSafe = 1;
        aci.allocInfo[Pal::CommandDataAlloc].allocHeap = Pal::GpuHeapGartUswc;
        aci.allocInfo[Pal::CommandDataAlloc].suballocSize = 64 * 1024;
        aci.allocInfo[Pal::CommandDataAlloc].allocSize = 256 * 1024;
        aci.allocInfo[Pal::EmbeddedDataAlloc].allocHeap = Pal::GpuHeapGartUswc;
        aci.allocInfo[Pal::EmbeddedDataAlloc].suballocSize = 64 * 1024;
        aci.allocInfo[Pal::EmbeddedDataAlloc].allocSize = 256 * 1024;
        aci.allocInfo[Pal::GpuScratchMemAlloc].allocHeap = Pal::GpuHeapInvisible;
        aci.allocInfo[Pal::GpuScratchMemAlloc].suballocSize = 64 * 1024;
        aci.allocInfo[Pal::GpuScratchMemAlloc].allocSize = 256 * 1024;

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

    // Create command buffer
    {
        Pal::CmdBufferCreateInfo cbci = {};
        cbci.engineType = eng;
        cbci.queueType = Pal::QueueTypeUniversal /* TODO: QueueTypeVideoDecode */;
        cbci.pCmdAllocator = cmd_allocator_.Get();

        size_t sz = device_->GetCmdBufferSize(cbci, &res);
        if (Util::IsErrorResult(res)) {
            CriticalLog(g_rocdec_logger, "PAL: GetCmdBufferSize failed");
            return ROCDEC_RUNTIME_ERROR;
        }

        void* mem = malloc(sz);
        Pal::ICmdBuffer* cb = nullptr;
        res = device_->CreateCmdBuffer(cbci, mem, &cb);
        if (Util::IsErrorResult(res) || !cb) {
            free(mem);
            CriticalLog(g_rocdec_logger, "PAL: CreateCmdBuffer failed");
            return ROCDEC_RUNTIME_ERROR;
        }

        *cmd_buffer_.PtrAddr() = cb;
        *cmd_buffer_.MemAddr() = mem;
    }

    // Create fence
    {
        Pal::FenceCreateInfo fci = {};
        fci.flags.signaled = 0;

        size_t sz = device_->GetFenceSize(&res);
        if (Util::IsErrorResult(res)) {
            CriticalLog(g_rocdec_logger, "PAL: GetFenceSize failed");
            return ROCDEC_RUNTIME_ERROR;
        }

        void* mem = malloc(sz);
        Pal::IFence* f = nullptr;
        res = device_->CreateFence(fci, mem, &f);
        if (Util::IsErrorResult(res) || !f) {
            free(mem);
            CriticalLog(g_rocdec_logger, "PAL: CreateFence failed");
            return ROCDEC_RUNTIME_ERROR;
        }

        *fence_.PtrAddr() = f;
        *fence_.MemAddr() = mem;
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

    // Create image
    Pal::ImageCreateInfo ici = {};
    ici.imageType = Pal::ImageType::Tex2d;
    ici.swizzledFormat.format = pal_format_;
    ici.swizzledFormat.swizzle.r = Pal::ChannelSwizzle::X;
    ici.swizzledFormat.swizzle.g = Pal::ChannelSwizzle::Y;
    ici.swizzledFormat.swizzle.b = Pal::ChannelSwizzle::Z;
    ici.swizzledFormat.swizzle.a = Pal::ChannelSwizzle::W;
    ici.extent.width = width;
    ici.extent.height = height;
    ici.extent.depth = 1;
    ici.mipLevels = 1;
    ici.arraySize = 1;
    ici.samples = 1;
    ici.fragments = 1;
    ici.tiling = Pal::ImageTiling::Optimal;
    // TODO: PAL version mismatch - videoDecodeTarget not found
    // ici.usageFlags.videoDecodeTarget = 1;

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
    mem_ci.vaRange = Pal::VaRange::Default;
    mem_ci.priority = Pal::GpuMemPriority::Normal;
    mem_ci.heapCount = mem_reqs.heapCount;
    for (uint32_t i = 0; i < mem_reqs.heapCount; ++i) {
        mem_ci.heaps[i] = mem_reqs.heaps[i];
    }

    size_t gpu_mem_size = device_->GetGpuMemorySize(mem_ci, &res);
    if (Util::IsErrorResult(res)) {
        CriticalLog(g_rocdec_logger, "PAL: GetGpuMemorySize failed");
        return ROCDEC_RUNTIME_ERROR;
    }

    void* gpu_mem_obj = malloc(gpu_mem_size);
    Pal::IGpuMemory* gpu_mem = nullptr;
    res = device_->CreateGpuMemory(mem_ci, gpu_mem_obj, &gpu_mem);
    if (Util::IsErrorResult(res) || !gpu_mem) {
        free(gpu_mem_obj);
        CriticalLog(g_rocdec_logger, "PAL: CreateGpuMemory failed");
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

    slot.image_index = 0;
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

    // Upload bitstream to GPU
    size_t bitstream_size = params->bitstream_data_len;
    if (bitstream_size > bitstream_.capacity) {
        rocDecStatus status = AllocateBitstreamBuffer(bitstream_size * 2);
        if (status != ROCDEC_SUCCESS) {
            return status;
        }
    }

    rocDecStatus status = UploadBitstream(
        static_cast<const uint8_t*>(params->bitstream_data),
        bitstream_size
    );
    if (status != ROCDEC_SUCCESS) {
        return status;
    }

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

    // Reference picture list
    for (size_t i = 0; i < 15; i++) {
        // Check if reference is valid (not using is_invalid flag directly, check pic_idx)
        if (hevc->ref_frames[i].pic_idx != 0xFF && hevc->ref_frames[i].pic_idx >= 0) {
            pal_hevc.ref_pic_list[i] = static_cast<unsigned char>(hevc->ref_frames[i].pic_idx);
            pal_hevc.poc_list[i] = hevc->ref_frames[i].poc;
        } else {
            pal_hevc.ref_pic_list[i] = 0xFF;  // Invalid
            pal_hevc.poc_list[i] = 0;
        }
    }

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

    // TODO: Submit decode command to PAL
    // For now, return success to test parameter mapping
    CriticalLog(g_rocdec_logger, "PAL: HEVC parameter mapping complete, decode submission not yet implemented");

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

    if (!initialized_ || !fence_.Get()) {
        dec_pic->decode_status = static_cast<rocDecDecodeStatus>(0);  // Invalid/not ready
        dec_pic->reserved[0] = 0;
        return ROCDEC_NOT_INITIALIZED;
    }

    // Check if picture index is valid
    PalDpbSlot* slot = dpb_.GetSlot(pic_idx);
    if (!slot || !slot->occupied) {
        dec_pic->decode_status = static_cast<rocDecDecodeStatus>(0);  // Invalid
        dec_pic->reserved[0] = 0;
        return ROCDEC_INVALID_PARAMETER;
    }

    // Query fence status (non-blocking)
    Pal::Result res = fence_->GetStatus();

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

    // Wait for pending operations
    if (device_ && fence_.Get()) {
        const Pal::IFence* fences[] = { fence_.Get() };
        device_->WaitForFences(1, fences, true, std::chrono::nanoseconds::max());
    }

    // Unmap bitstream
    if (bitstream_.gpu_memory.Get() && bitstream_.mapped_ptr) {
        bitstream_.gpu_memory->Unmap();
        bitstream_.mapped_ptr = nullptr;
    }

    // Release resources (PalObject destructors handle this)
    bitstream_.gpu_memory.Reset();
    dpb_.Clear();
    fence_.Reset();
    cmd_buffer_.Reset();
    cmd_allocator_.Reset();
    video_queue_.Reset();

    initialized_ = false;
    device_ = nullptr;

    ReleasePalPlatform();
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
