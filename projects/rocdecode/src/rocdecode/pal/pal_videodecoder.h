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

#ifdef ROCDECODE_BUILD_WINDOWS

// Prevent Windows.h from defining min/max macros (PAL requires this)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// Define PAL_BUILD_VIDEO for video decoder support
#ifndef PAL_BUILD_VIDEO
#define PAL_BUILD_VIDEO 1
#endif

#include <windows.h>
#include <vector>
#include <memory>
#include <mutex>

// PAL headers
#include "pal.h"
#include "palPlatform.h"
#include "palDevice.h"
#include "palQueue.h"
#include "palCmdBuffer.h"
#include "palCmdAllocator.h"
#include "palGpuMemory.h"
#include "palImage.h"
#include "palFence.h"
#include "palVideoDecoder.h"

// Include PAL UVD interface for codec structures
#if PAL_CLOSED_SOURCE
#include "drv_uvd_if.h"
#else
#include "drv_uvd_if_open.h"
#endif

// rocDecode headers
#include "rocdecode/rocdecode.h"

namespace rocdec {

/**
 * @brief RAII wrapper for PAL placement-new objects
 *
 * PAL objects are created via GetXxxSize() → malloc() → CreateXxx() pattern.
 * This wrapper owns both the interface pointer and backing memory.
 */
template<typename T>
class PalObject {
public:
    PalObject() = default;

    PalObject(PalObject&& o) noexcept : ptr_(o.ptr_), mem_(o.mem_) {
        o.ptr_ = nullptr;
        o.mem_ = nullptr;
    }

    PalObject& operator=(PalObject&& o) noexcept {
        Reset();
        ptr_ = o.ptr_;
        mem_ = o.mem_;
        o.ptr_ = nullptr;
        o.mem_ = nullptr;
        return *this;
    }

    // Non-copyable
    PalObject(const PalObject&) = delete;
    PalObject& operator=(const PalObject&) = delete;

    ~PalObject() { Reset(); }

    void Reset() {
        if (ptr_) { ptr_->Destroy(); ptr_ = nullptr; }
        if (mem_) { free(mem_); mem_ = nullptr; }
    }

    T* Get() const { return ptr_; }
    T* operator->() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }

    T** PtrAddr() { return &ptr_; }
    void** MemAddr() { return &mem_; }

private:
    T* ptr_ = nullptr;
    void* mem_ = nullptr;
};

/**
 * @brief DPB (Decoded Picture Buffer) slot for reference frames
 */
struct PalDpbSlot {
    // Tiled DPB surface — VCN reconstructs here (SW_256B_S). Used as DPB curr/ref.
    PalObject<Pal::IImage> image;            // Tiled DPB surface (NV12/P010)
    PalObject<Pal::IGpuMemory> image_memory; // Backing GPU memory (GPU-local)
    // Linear display surface — VCN writes a clean linear copy here; exported to HIP.
    PalObject<Pal::IImage> target_image;            // Linear decode-target surface
    PalObject<Pal::IGpuMemory> target_image_memory; // Backing GPU memory (GART, interprocess=1)
    PalObject<Pal::IFence> fence;            // Signaled when this slot's decode completes
    uint32_t image_index;                    // Array layer (usually 0)

    bool occupied;
    int32_t poc;                             // Picture order count
    int32_t frame_idx;                       // Frame number / POC LSB

    PalDpbSlot() : image_index(0), occupied(false), poc(-1), frame_idx(-1) {}

    // Non-copyable (contains PalObject)
    PalDpbSlot(const PalDpbSlot&) = delete;
    PalDpbSlot& operator=(const PalDpbSlot&) = delete;
    PalDpbSlot(PalDpbSlot&&) = default;
    PalDpbSlot& operator=(PalDpbSlot&&) = default;
};

/**
 * @brief Decoded picture buffer manager
 */
class PalDpb {
public:
    static constexpr uint32_t kMaxSlots = 17; // H.264 max DPB + current frame

    PalDpb() : slot_count_(0) {}

    int32_t AllocSlot();
    void FreeSlot(int32_t idx);
    int32_t FindByPoc(int32_t poc) const;
    void Clear();

    PalDpbSlot* GetSlot(int32_t idx) {
        return (idx >= 0 && (uint32_t)idx < slot_count_) ? &slots_[idx] : nullptr;
    }

    uint32_t GetSlotCount() const { return slot_count_; }

private:
    PalDpbSlot slots_[kMaxSlots];
    uint32_t slot_count_;
};

/**
 * @brief Bitstream upload buffer (CPU-writable, GPU-readable)
 */
struct PalBitstreamBuffer {
    PalObject<Pal::IGpuMemory> gpu_memory;
    void* mapped_ptr;
    size_t capacity;
    size_t used;

    PalBitstreamBuffer() : mapped_ptr(nullptr), capacity(0), used(0) {}
};

/**
 * @brief PAL-based video decoder context
 *
 * Implements hardware video decoding using AMD PAL (Platform Abstraction Library)
 * which provides direct access to the VCN (Video Core Next) decode engine.
 */
class PalVideoDecoder {
public:
    PalVideoDecoder();
    ~PalVideoDecoder();

    // Non-copyable
    PalVideoDecoder(const PalVideoDecoder&) = delete;
    PalVideoDecoder& operator=(const PalVideoDecoder&) = delete;

    /**
     * @brief Initialize PAL decoder session
     *
     * @param codec_type Codec to decode (H264, HEVC, VP9, AV1)
     * @param width Maximum coded width
     * @param height Maximum coded height
     * @param bit_depth Bit depth (8 or 10)
     * @param max_dpb_slots Maximum DPB slots needed
     * @return rocDecStatus
     */
    rocDecStatus Initialize(rocDecVideoCodec codec_type,
                           uint32_t width,
                           uint32_t height,
                           uint32_t bit_depth,
                           uint32_t max_dpb_slots);

    /**
     * @brief Decode a video frame
     *
     * @param pic_params Codec-specific picture parameters
     * @return rocDecStatus
     */
    rocDecStatus DecodeFrame(RocdecPicParams* pic_params);

    /**
     * @brief Get decoded frame
     *
     * @param pic_idx Picture index to retrieve
     * @param dec_pic Output decoded picture info
     * @return rocDecStatus
     */
    rocDecStatus GetDecodeStatus(int pic_idx, RocdecDecodeStatus* dec_pic);

    /**
     * @brief Reconfigure decoder for new resolution
     *
     * @param width New width
     * @param height New height
     * @return rocDecStatus
     */
    rocDecStatus Reconfigure(uint32_t width, uint32_t height);

    /**
     * @brief Clean up resources
     */
    void Destroy();

    /**
     * @brief Query decoder capabilities
     *
     * @param caps Capability query structure (IN/OUT)
     * @return rocDecStatus
     */
    static rocDecStatus QueryDecoderCaps(RocdecDecodeCaps* caps);

    /**
     * @brief Export decoded surface for HIP interop
     *
     * @param pic_idx Picture index
     * @param width Output width
     * @param height Output height
     * @param pitch Output pitch for each plane
     * @param offset Output offset for each plane
     * @param num_planes Output number of planes
     * @param kmt_handle Output Windows KMT handle for HIP import
     * @param mem_size Output total memory size
     * @return rocDecStatus
     */
    rocDecStatus ExportSurface(int pic_idx,
                               uint32_t& width,
                               uint32_t& height,
                               uint32_t pitch[3],
                               uint32_t offset[3],
                               uint32_t& num_planes,
                               void*& kmt_handle,
                               size_t& mem_size);

private:
    // PAL platform and device (shared across all decoders)
    static Pal::IPlatform* GetPalPlatform();
    static void ReleasePalPlatform();
    static Pal::IDevice* GetPalDevice();

    // Decode operations
    rocDecStatus DecodeH264(RocdecPicParams* params);
    rocDecStatus DecodeHEVC(RocdecPicParams* params);
    rocDecStatus DecodeAV1(RocdecPicParams* params);
    rocDecStatus DecodeVP9(RocdecPicParams* params);

    // Helper functions
    rocDecStatus AllocateDecodedFrame(uint32_t width, uint32_t height,
                                     PalDpbSlot& slot);
    rocDecStatus AllocateBitstreamBuffer(size_t capacity);
    rocDecStatus UploadBitstream(const uint8_t* data, size_t size);
    rocDecStatus SyncQueue();

    // State
    Pal::IDevice* device_;
    PalObject<Pal::IQueue> video_queue_;
    PalObject<Pal::ICmdAllocator> cmd_allocator_;
    PalObject<Pal::ICmdBuffer> cmd_buffer_;
    PalObject<Pal::IVideoDecoder> video_decoder_;

    rocDecVideoCodec codec_;
    Pal::ChNumFormat pal_format_;
    uint32_t max_coded_width_;
    uint32_t max_coded_height_;
    uint32_t bit_depth_;
    uint32_t max_dpb_slots_;

    PalDpb dpb_;
    PalBitstreamBuffer bitstream_;
    PalObject<Pal::IGpuMemory> decoder_heap_;  // VCN decoder internal heap

    bool initialized_;
    bool session_begun_;
    int32_t last_submitted_slot_idx_; // DPB slot index of the most recently submitted frame
    std::mutex mutex_;

    // Platform singleton state (recursive mutex because GetPalDevice calls GetPalPlatform)
    static std::recursive_mutex platform_mutex_;
    static Pal::IPlatform* platform_;
    static void* platform_mem_;
    static int platform_ref_count_;
    static Pal::IDevice* device_instance_;
    static Pal::EngineType video_decode_engine_;
};

} // namespace rocdec

#endif // ROCDECODE_BUILD_WINDOWS
