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

#include "pal_videodecoder.h"
#include "../../commons.h"
#include <cstring>
#include <algorithm>

namespace rocdec {

// Static members initialization
std::mutex PalVideoDecoder::platform_mutex_;
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
    std::lock_guard<std::mutex> lock(platform_mutex_);

    if (platform_) {
        platform_ref_count_++;
        return platform_;
    }

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

    size_t plat_size = Pal::GetPlatformSize();
    platform_mem_ = malloc(plat_size);
    if (!platform_mem_) {
        CriticalLog(g_rocdec_logger, "PAL: Failed to allocate platform memory");
        return nullptr;
    }

    Pal::Result res = Pal::CreatePlatform(ci, platform_mem_, &platform_);
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
    std::lock_guard<std::mutex> lock(platform_mutex_);

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
    std::lock_guard<std::mutex> lock(platform_mutex_);

    if (device_instance_) {
        return device_instance_;
    }

    Pal::IPlatform* platform = GetPalPlatform();
    if (!platform) {
        return nullptr;
    }

    // Enumerate devices
    uint32_t device_count = 0;
    Pal::IDevice* raw_devices[Pal::MaxDevices] = {};

    Pal::Result res = platform->EnumerateDevices(&device_count, raw_devices);
    if (Util::IsErrorResult(res) || device_count == 0) {
        CriticalLog(g_rocdec_logger, "PAL: No devices found");
        return nullptr;
    }

    InfoLog(g_rocdec_logger, ROCDEC_STR("PAL: Found ") + ROCDEC_TOSTR(device_count) + " device(s)");

    // Find first device with VCN decode support
    for (uint32_t i = 0; i < device_count; ++i) {
        Pal::IDevice* dev = raw_devices[i];

        // Commit settings
        res = dev->CommitSettingsAndInit();
        if (Util::IsErrorResult(res)) {
            continue;
        }

        // Check for VCN hardware
        Pal::DeviceProperties dev_props = {};
        dev->GetProperties(&dev_props);

        if (dev_props.vcnLevel == Pal::VcnIpLevel::None ||
            dev_props.vcnLevel == Pal::VcnIpLevel::_None) {
            continue;
        }

        // TODO: Check for video decode engines - stubbed for compilation
        // Pal::EngineType video_engine = Pal::EngineTypeCount;
        // if (dev_props.engineProperties[Pal::EngineTypeVcnDecode].engineCount > 0) {
        //     video_engine = Pal::EngineTypeVcnDecode;
        // } else if (dev_props.engineProperties[Pal::EngineTypeVcnUnified].engineCount > 0) {
        //     video_engine = Pal::EngineTypeVcnUnified;
        // }

        // if (video_engine == Pal::EngineTypeCount) {
        //     continue;
        // }

        // Finalize device
        Pal::DeviceFinalizeInfo fi = {};
        // TODO: Set engine counts - stubbed
        // fi.requestedEngineCounts[video_engine].engines = 1;
        // fi.requestedEngineCounts[Pal::EngineTypeUniversal].engines =
        //     (dev_props.engineProperties[Pal::EngineTypeUniversal].engineCount > 0) ? 1 : 0;

        res = dev->Finalize(fi);
        if (Util::IsErrorResult(res)) {
            continue;
        }

        device_instance_ = dev;
        // video_decode_engine_ = video_engine;

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
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        return ROCDEC_SUCCESS;
    }

    codec_ = codec_type;
    max_coded_width_ = width;
    max_coded_height_ = height;
    bit_depth_ = bit_depth;
    max_dpb_slots_ = std::min(max_dpb_slots, static_cast<uint32_t>(PalDpb::kMaxSlots));

    // Determine format based on bit depth
    pal_format_ = (bit_depth == 10) ? Pal::ChNumFormat::P010 : Pal::ChNumFormat::NV12;

    // Get PAL device
    device_ = GetPalDevice();
    if (!device_) {
        CriticalLog(g_rocdec_logger, "PAL: Failed to get device");
        return ROCDEC_NOT_INITIALIZED;
    }

    Pal::EngineType eng = video_decode_engine_;
    Pal::Result res;

    // Create video queue
    {
        Pal::QueueCreateInfo qci = {};
        qci.engineType = eng;
        qci.engineIndex = 0;
        qci.queueType = Pal::QueueTypeUniversal /* TODO: QueueTypeVideoDecode */;

        size_t sz = device_->GetQueueSize(qci, &res);
        if (Util::IsErrorResult(res)) {
            CriticalLog(g_rocdec_logger, "PAL: GetQueueSize failed");
            return ROCDEC_RUNTIME_ERROR;
        }

        void* mem = malloc(sz);
        Pal::IQueue* q = nullptr;
        res = device_->CreateQueue(qci, mem, &q);
        if (Util::IsErrorResult(res) || !q) {
            free(mem);
            CriticalLog(g_rocdec_logger, "PAL: CreateQueue failed");
            return ROCDEC_RUNTIME_ERROR;
        }

        *video_queue_.PtrAddr() = q;
        *video_queue_.MemAddr() = mem;
    }

    // Create command allocator
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
            CriticalLog(g_rocdec_logger, "PAL: GetCmdAllocatorSize failed");
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
    // TODO: Implement H.264 decode
    CriticalLog(g_rocdec_logger, "PAL: H.264 decode not yet implemented");
    return ROCDEC_NOT_IMPLEMENTED;
}

rocDecStatus PalVideoDecoder::DecodeHEVC(RocdecPicParams* params) {
    // TODO: Implement HEVC decode
    CriticalLog(g_rocdec_logger, "PAL: HEVC decode not yet implemented");
    return ROCDEC_NOT_IMPLEMENTED;
}

rocDecStatus PalVideoDecoder::DecodeAV1(RocdecPicParams* params) {
    // TODO: Implement AV1 decode
    CriticalLog(g_rocdec_logger, "PAL: AV1 decode not yet implemented");
    return ROCDEC_NOT_IMPLEMENTED;
}

rocDecStatus PalVideoDecoder::DecodeVP9(RocdecPicParams* params) {
    // TODO: Implement VP9 decode
    CriticalLog(g_rocdec_logger, "PAL: VP9 decode not yet implemented");
    return ROCDEC_NOT_IMPLEMENTED;
}

rocDecStatus PalVideoDecoder::GetDecodeStatus(int pic_idx, RocdecDecodeStatus* dec_pic) {
    if (!dec_pic) {
        return ROCDEC_INVALID_PARAMETER;
    }

    // TODO: Implement status query
    dec_pic->decode_status = static_cast<rocDecDecodeStatus>(0);
    dec_pic->reserved[0] = 0;
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

} // namespace rocdec

#endif // ROCDECODE_BUILD_WINDOWS
