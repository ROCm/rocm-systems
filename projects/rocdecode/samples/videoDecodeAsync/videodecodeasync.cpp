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

// videoDecodeAsync: a producer-consumer decode sample used to test display-callback designs.
//
// The producer (main thread) demuxes and parses the stream and runs the parser callbacks.
// The display callback hands each frame to a consumer thread through a bounded queue,
// following a display policy:
//   sync-copy  rocDecGetVideoFrame, copy the frame into a sample-owned buffer and wait for
//              the copy, then queue the buffer. Safe: the decode surface may be reused as
//              soon as the callback returns.
//   no-copy    rocDecGetVideoFrame and queue the returned pointers without copying. Unsafe
//              on purpose: the pointer is an alias of the decode surface (a second reference
//              to the same memory, not a copy), and the consumer reads it later, after the
//              parser may have reused that surface for another picture. This is the control
//              the Map/Unmap tests compare against; its MD5 is expected to differ.
// The consumer copies (no-copy) and hashes the frames in display order.
//
// The sample uses the rocDecode core API directly rather than the RocVideoDecoder utility,
// because the designs under test differ exactly in the display callback that the utility
// hides. It uses only header-level helpers from utils: VideoDemuxer, MD5Generator and the
// OutputSurfaceInfo struct and surface-format helpers from roc_video_dec.h.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <hip/hip_runtime.h>
#include "video_demuxer.h"
#include "roc_video_dec.h"
#include "md5.h"

using Clock = std::chrono::steady_clock;

enum class Policy { SyncCopy, NoCopy };
enum class Workload { Md5, Lag, Hold };

// Geometry of the frames the sample outputs: the display area of the decoded picture,
// rounded up to even width and height, packed with one pitch for every plane.
struct FrameGeometry {
    Rect disp_rect = {};
    uint32_t width = 0, height = 0;          // target (output) size
    uint32_t chroma_width = 0;               // samples per chroma row: interleaved UV counts both
    uint32_t chroma_height = 0;
    uint32_t num_chroma_planes = 0;
    uint32_t bytes_per_pixel = 1;
    uint32_t bit_depth = 8;
    rocDecVideoSurfaceFormat surface_format = rocDecVideoSurfaceFormat_NV12;
    uint32_t Pitch() const { return width * bytes_per_pixel; }
    uint32_t ChromaPitch() const { return chroma_width * bytes_per_pixel; }
    size_t FrameSize() const {
        return static_cast<size_t>(Pitch()) * height + static_cast<size_t>(ChromaPitch()) * chroma_height * num_chroma_planes;
    }
};

struct FrameItem {
    uint64_t index = 0;                       // display order
    int64_t pts = 0;
    FrameGeometry geometry;
    uint8_t *buffer = nullptr;                // sync-copy: packed frame in a pool buffer
    void *src[3] = {};                        // no-copy: decode surface planes
    uint32_t src_pitch[3] = {};
};

struct Options {
    std::string input_path;
    int device_id = 0;
    Policy policy = Policy::SyncCopy;
    Workload workload = Workload::Md5;
    uint32_t lag = 0;
    uint32_t hold_min_ms = 0, hold_max_ms = 0;
    uint32_t display_delay = 1;
    uint32_t decode_surfaces = 1;             // 1: let the parser choose the pool size
    uint32_t queue_depth = 4;
    bool md5 = false;
    std::string md5_check_path;
    std::string md5_frames_path;
    bool expect_mismatch = false;
};

struct AsyncDecoder {
    Options opt;
    rocDecVideoCodec codec = rocDecVideoCodec_HEVC;
    RocdecVideoParser parser = nullptr;
    rocDecDecoderHandle decoder = nullptr;
    hipStream_t producer_stream = nullptr;

    // Current stream format.
    rocDecVideoChromaFormat chroma_format = rocDecVideoChromaFormat_420;
    uint32_t bit_depth_minus_8 = 0;
    uint32_t coded_width = 0, coded_height = 0;
    uint32_t num_decode_surfaces = 0;
    int max_width = 0, max_height = 0;
    FrameGeometry geometry;
    bool unsupported = false;
    bool failed = false;

    // Queue between producer and consumer.
    std::mutex mtx;
    std::condition_variable cv;               // any change to the queue, pool or counters
    std::deque<FrameItem> queue;
    uint64_t displayed = 0;                   // frames pushed by the display callback
    uint64_t consumed = 0;                    // frames finished by the consumer
    bool draining = false;                    // producer waits for the queue to empty
    bool eos = false;
    bool consumer_stopped = false;            // the consumer exited early; producer waits must not block

    // sync-copy buffer pool.
    std::vector<uint8_t *> all_buffers;
    std::vector<uint8_t *> free_buffers;
    size_t buffer_size = 0;

    // Statistics.
    uint64_t decoded_pictures = 0;
    double display_callback_ms = 0;           // producer time inside the display callback
};

// Each queue slot holds a full-frame pool buffer in sync-copy, so larger depths would only fail
// to allocate; the cap also keeps queue_depth + 2 and lag + 1 from wrapping.
constexpr uint32_t kMaxQueueDepth = 256;

static void ShowHelpAndExit(const char *option = nullptr) {
    std::cout << "Options:" << std::endl
    << "-i Input file path - required" << std::endl
    << "-d GPU device ID (0 for the first device, 1 for the second, etc.); optional; default: 0" << std::endl
    << "-p Display policy: sync-copy | no-copy; optional; default: sync-copy" << std::endl
    << "      sync-copy: copy each frame in the display callback (safe baseline)" << std::endl
    << "      no-copy:   queue the decode surface pointer without copying (unsafe control)" << std::endl
    << "-w Consumer workload: md5 | lag:<L> | hold:<min>-<max>; optional; default: md5" << std::endl
    << "      md5:       hash each frame as soon as it is dequeued" << std::endl
    << "      lag:<L>:   hash frame n only after frame n+L has been displayed (deterministic hold), L 1-" << kMaxQueueDepth - 1 << std::endl
    << "      hold:<min>-<max>: wait a random time in milliseconds before hashing each frame" << std::endl
    << "-dd Display delay passed to the parser; optional; default: 1" << std::endl
    << "-ds Max number of decode surfaces passed to the parser; optional; default: 1 (parser decides)" << std::endl
    << "-q Queue depth between producer and consumer, 1-" << kMaxQueueDepth << "; optional; default: 4 (raised to L+1 for lag:<L>)" << std::endl
    << "-md5 Generate the MD5 digest of the output frames; optional" << std::endl
    << "-md5_check MD5 file path - generate the MD5 digest and compare it to the reference in the file; optional" << std::endl
    << "-md5_frames File path - write one MD5 digest per output frame; optional" << std::endl
    << "-expect_mismatch Invert the -md5_check result: succeed only if the digest does NOT match (control runs); optional" << std::endl;
    exit(option == nullptr ? 0 : 1);
}

// Parse a decimal number in [min_value, max_value]. std::stoul alone would accept a sign (wrapping
// -1 to a huge value), drop trailing text and throw on input that is not a number.
static bool ParseUint(const std::string &s, uint32_t min_value, uint32_t max_value, uint32_t &value) {
    if (s.empty() || s.size() > 10 || !std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); })) return false;
    uint64_t v = std::stoull(s);
    if (v < min_value || v > max_value) return false;
    value = static_cast<uint32_t>(v);
    return true;
}

static bool ParseWorkload(const std::string &arg, Options &opt) {
    if (arg == "md5") {
        opt.workload = Workload::Md5;
        return true;
    }
    if (arg.rfind("lag:", 0) == 0) {
        opt.workload = Workload::Lag;
        return ParseUint(arg.substr(4), 1, kMaxQueueDepth - 1, opt.lag);
    }
    if (arg.rfind("hold:", 0) == 0) {
        size_t dash = arg.find('-', 5);
        if (dash == std::string::npos) return false;
        opt.workload = Workload::Hold;
        return ParseUint(arg.substr(5, dash - 5), 0, UINT32_MAX, opt.hold_min_ms) &&
               ParseUint(arg.substr(dash + 1), 0, UINT32_MAX, opt.hold_max_ms) && opt.hold_min_ms <= opt.hold_max_ms;
    }
    return false;
}

// Log a failed cleanup call; cleanup continues either way.
static void CheckCleanup(hipError_t status, const char *call) {
    if (status != hipSuccess) std::cerr << "Error: " << call << " failed: " << hipGetErrorName(status) << std::endl;
}

static std::string DigestToString(const uint8_t *digest) {
    std::ostringstream s;
    for (int i = 0; i < 16; i++) s << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[i]);
    return s.str();
}

// Describe a packed frame (see FrameGeometry), copied to host memory, for MD5Generator. The frame
// is already cropped, so the display rectangle is the whole frame.
static OutputSurfaceInfo DescribePackedFrame(const FrameGeometry &g) {
    OutputSurfaceInfo info = {};
    info.output_width = g.width;
    info.output_height = g.height;
    info.output_pitch = g.Pitch();
    info.output_vstride = g.height;
    info.chroma_height = g.chroma_height;
    info.disp_rect = {0, 0, static_cast<int>(g.width), static_cast<int>(g.height)};
    info.bytes_per_pixel = g.bytes_per_pixel;
    info.bit_depth = g.bit_depth;
    info.num_chroma_planes = g.num_chroma_planes;
    info.output_surface_size_in_bytes = g.FrameSize();
    info.surface_format = g.surface_format;
    info.mem_type = OUT_SURFACE_MEM_HOST_COPIED;
    return info;
}

// Copy the display area of a decoded picture into a packed device buffer. The layout matches
// what -m 0 (device internal) hashes in videoDecode: the even-rounded target size from the
// display origin, with subsampled chroma offset by half the top row or half the left column.
static hipError_t CopyFrame(void *const src[3], const uint32_t src_pitch[3], const FrameGeometry &g,
                            uint8_t *dst, hipStream_t stream) {
    const uint32_t row_bytes = g.Pitch();
    const uint8_t *luma = static_cast<const uint8_t *>(src[0]) + g.disp_rect.top * src_pitch[0] + g.disp_rect.left * g.bytes_per_pixel;
    hipError_t status = hipMemcpy2DAsync(dst, row_bytes, luma, src_pitch[0], row_bytes, g.height, hipMemcpyDeviceToDevice, stream);
    uint8_t *dst_plane = dst + static_cast<size_t>(row_bytes) * g.height;
    const uint32_t chroma_row_bytes = g.ChromaPitch();
    // Half-height chroma (4:2:0) starts at half the display top; half-width planar chroma (4:2:2)
    // at half the display left. Interleaved UV keeps the luma offset, as each pair is two samples.
    const uint32_t top = (g.chroma_height < g.height) ? (g.disp_rect.top >> 1) : g.disp_rect.top;
    const uint32_t left = (g.chroma_width < g.width) ? (g.disp_rect.left >> 1) : g.disp_rect.left;
    for (uint32_t p = 0; p < g.num_chroma_planes && status == hipSuccess; p++) {
        const uint8_t *chroma = static_cast<const uint8_t *>(src[1 + p]) + top * src_pitch[1 + p] + left * g.bytes_per_pixel;
        status = hipMemcpy2DAsync(dst_plane, chroma_row_bytes, chroma, src_pitch[1 + p], chroma_row_bytes, g.chroma_height, hipMemcpyDeviceToDevice, stream);
        dst_plane += static_cast<size_t>(chroma_row_bytes) * g.chroma_height;
    }
    return status;
}

// ------------------------------------------------------------------------------------------
// sync-copy buffer pool. Sized to the queue depth plus one buffer held by the consumer and one
// being filled, so the producer only waits when the consumer has fallen behind.
// ------------------------------------------------------------------------------------------
static void FreeBufferPool(AsyncDecoder &dec) {
    for (uint8_t *b : dec.all_buffers) {
        CheckCleanup(hipFree(b), "hipFree");
    }
    dec.all_buffers.clear();
    dec.free_buffers.clear();
    dec.buffer_size = 0;
}

static bool AllocateBufferPool(AsyncDecoder &dec) {
    FreeBufferPool(dec);
    dec.buffer_size = dec.geometry.FrameSize();
    uint32_t count = dec.opt.queue_depth + 2;
    for (uint32_t i = 0; i < count; i++) {
        uint8_t *b = nullptr;
        hipError_t status = hipMalloc(reinterpret_cast<void **>(&b), dec.buffer_size);
        if (status != hipSuccess) {
            std::cerr << "Error: hipMalloc of " << dec.buffer_size << " bytes failed: " << hipGetErrorName(status) << std::endl;
            return false;
        }
        dec.all_buffers.push_back(b);
        dec.free_buffers.push_back(b);
    }
    return true;
}

// Wait until the consumer has finished every queued frame. Called before a reconfigure,
// which frees the decode surfaces' HIP mappings that no-copy frames point into, and before the
// buffer pool is resized. Lag and hold waits end early while draining.
// Returns false if the consumer stopped early, so the queue can never drain.
static bool DrainQueue(AsyncDecoder &dec) {
    std::unique_lock<std::mutex> lock(dec.mtx);
    dec.draining = true;
    dec.cv.notify_all();
    dec.cv.wait(lock, [&] { return dec.consumed == dec.displayed || dec.consumer_stopped; });
    dec.draining = false;
    return !dec.consumer_stopped;
}

static void UpdateGeometry(AsyncDecoder &dec, const RocdecVideoFormat *format) {
    FrameGeometry &g = dec.geometry;
    g.disp_rect.left = format->display_area.left;
    g.disp_rect.top = format->display_area.top;
    g.disp_rect.right = format->display_area.right;
    g.disp_rect.bottom = format->display_area.bottom;
    g.width = (format->display_area.right - format->display_area.left + 1) & ~1;
    g.height = (format->display_area.bottom - format->display_area.top + 1) & ~1;
    g.bit_depth = format->bit_depth_luma_minus8 + 8;
    g.bytes_per_pixel = format->bit_depth_luma_minus8 > 0 ? 2 : 1;
    g.surface_format = SelectSurfaceFormat(format->chroma_format, format->bit_depth_luma_minus8);
    // Planar 4:2:0 and 4:2:2 chroma rows are half the luma width; interleaved UV (NV12/P016) and
    // planar 4:4:4 rows hold as many samples as luma.
    bool half_width = g.surface_format == rocDecVideoSurfaceFormat_YUV420 || g.surface_format == rocDecVideoSurfaceFormat_YUV420_16Bit ||
                      g.surface_format == rocDecVideoSurfaceFormat_YUV422 || g.surface_format == rocDecVideoSurfaceFormat_YUV422_16Bit;
    g.chroma_width = half_width ? g.width / 2 : g.width;
    g.chroma_height = static_cast<uint32_t>(std::ceil(g.height * GetChromaHeightFactor(g.surface_format)));
    g.num_chroma_planes = (format->chroma_format == rocDecVideoChromaFormat_Monochrome) ? 0 : GetChromaPlaneCount(g.surface_format);
}

// ------------------------------------------------------------------------------------------
// Parser callbacks (producer thread). They return 0 on failure, which stops the parser.
// Ported from RocVideoDecoder::HandleVideoSequence/ReconfigureDecoder, except that every
// output size is recomputed on each reconfigure.
// ------------------------------------------------------------------------------------------
static int ROCDECAPI HandleVideoSequence(void *user_data, RocdecVideoFormat *format) {
    AsyncDecoder &dec = *static_cast<AsyncDecoder *>(user_data);
    // MD5Generator handles only 4:2:0 with interleaved chroma: for monochrome it would read and
    // write past the luma-only frame, and planar 4:4:4 or 4:2:2 would be hashed incorrectly.
    if ((dec.opt.md5 || !dec.opt.md5_frames_path.empty()) && format->chroma_format != rocDecVideoChromaFormat_420) {
        std::cerr << "Error: MD5 hashing (-md5, -md5_check, -md5_frames) supports only 4:2:0 streams" << std::endl;
        dec.failed = true;
        return 0;
    }
    RocdecDecodeCaps caps = {};
    caps.codec_type = format->codec;
    caps.chroma_format = format->chroma_format;
    caps.bit_depth_minus_8 = format->bit_depth_luma_minus8;
    caps.device_id = dec.opt.device_id;
    rocDecStatus caps_status = rocDecGetDecoderCaps(&caps);
    if (caps_status != ROCDEC_SUCCESS) {
        std::cerr << "Error: rocDecGetDecoderCaps failed: " << rocDecGetErrorName(caps_status) << std::endl;
        dec.failed = true;
        return 0;
    }
    if (!caps.is_supported || format->coded_width > caps.max_width || format->coded_height > caps.max_height) {
        // Skip (exit 77) only before anything is decoded; a later sequence the GPU cannot decode
        // truncates the output, which is a failure.
        if (dec.decoder != nullptr) {
            std::cerr << "Error: reconfigured stream format not supported on this GPU" << std::endl;
            dec.failed = true;
            return 0;
        }
        std::cout << "info: Stream format not supported on this GPU" << std::endl;
        dec.unsupported = true;
        return 0;
    }

    if (dec.decoder == nullptr) {
        dec.codec = format->codec;
        dec.chroma_format = format->chroma_format;
        dec.bit_depth_minus_8 = format->bit_depth_luma_minus8;
        dec.coded_width = format->coded_width;
        dec.coded_height = format->coded_height;
        dec.num_decode_surfaces = format->min_num_decode_surfaces;
        // AV1 carries the maximum frame size of the sequence in its sequence header.
        if (dec.codec == rocDecVideoCodec_AV1 && format->seqhdr_data_length > 0) {
            const RocdecVideoFormatEx *ex = reinterpret_cast<const RocdecVideoFormatEx *>(format);
            dec.max_width = ex->max_width;
            dec.max_height = ex->max_height;
        }
        dec.max_width = std::max(dec.max_width, static_cast<int>(format->coded_width));
        dec.max_height = std::max(dec.max_height, static_cast<int>(format->coded_height));
        UpdateGeometry(dec, format);

        RocDecoderCreateInfo create_info = {};
        create_info.device_id = static_cast<uint8_t>(dec.opt.device_id);
        create_info.codec_type = dec.codec;
        create_info.chroma_format = dec.chroma_format;
        create_info.output_format = rocDecVideoSurfaceFormat_Native;
        create_info.bit_depth_minus_8 = dec.bit_depth_minus_8;
        create_info.num_decode_surfaces = dec.num_decode_surfaces;
        create_info.width = dec.coded_width;
        create_info.height = dec.coded_height;
        create_info.max_width = dec.max_width;
        create_info.max_height = dec.max_height;
        create_info.display_rect.left = static_cast<int16_t>(dec.geometry.disp_rect.left);
        create_info.display_rect.top = static_cast<int16_t>(dec.geometry.disp_rect.top);
        create_info.display_rect.right = static_cast<int16_t>(dec.geometry.disp_rect.right);
        create_info.display_rect.bottom = static_cast<int16_t>(dec.geometry.disp_rect.bottom);
        create_info.target_width = dec.geometry.width;
        create_info.target_height = dec.geometry.height;
        rocDecStatus status = rocDecCreateDecoder(&dec.decoder, &create_info);
        if (status != ROCDEC_SUCCESS) {
            std::cerr << "Error: rocDecCreateDecoder failed: " << rocDecGetErrorName(status) << std::endl;
            dec.failed = true;
            return 0;
        }
        if (dec.opt.policy == Policy::SyncCopy && !AllocateBufferPool(dec)) {
            dec.failed = true;
            return 0;
        }
        std::cout << "info: Coded size " << dec.coded_width << "x" << dec.coded_height << ", output "
                  << dec.geometry.width << "x" << dec.geometry.height << ", bit depth " << dec.geometry.bit_depth
                  << ", decode surfaces " << dec.num_decode_surfaces << std::endl;
        return static_cast<int>(dec.num_decode_surfaces);
    }

    // A later sequence header. Frames already queued were displayed with the old format and,
    // for no-copy, point into decode surfaces a reconfigure would unmap, so finish them first.
    if (!DrainQueue(dec)) {
        dec.failed = true;
        return 0;
    }

    if (format->codec != dec.codec || format->chroma_format != dec.chroma_format) {
        std::cerr << "Error: Reconfigure is not supported for a codec or chroma format change" << std::endl;
        dec.failed = true;
        return 0;
    }
    bool res_changed = format->coded_width != dec.coded_width || format->coded_height != dec.coded_height;
    bool rect_changed = format->display_area.left != dec.geometry.disp_rect.left || format->display_area.top != dec.geometry.disp_rect.top ||
                        format->display_area.right != dec.geometry.disp_rect.right || format->display_area.bottom != dec.geometry.disp_rect.bottom;
    bool depth_changed = format->bit_depth_luma_minus8 != dec.bit_depth_minus_8;
    bool surfaces_changed = format->min_num_decode_surfaces != dec.num_decode_surfaces;
    if (!res_changed && !rect_changed && !depth_changed && !surfaces_changed) {
        return 1;
    }

    dec.coded_width = format->coded_width;
    dec.coded_height = format->coded_height;
    dec.bit_depth_minus_8 = format->bit_depth_luma_minus8;
    dec.num_decode_surfaces = format->min_num_decode_surfaces;
    UpdateGeometry(dec, format);
    if (dec.opt.policy == Policy::SyncCopy && !AllocateBufferPool(dec)) {
        dec.failed = true;
        return 0;
    }

    // Only a reconfigure that needs new surfaces goes to the decoder; a display area change
    // alone just changes what the sample copies.
    if (format->reconfig_options == ROCDEC_RECONFIG_NEW_SURFACES) {
        RocdecReconfigureDecoderInfo reconfig = {};
        reconfig.width = dec.coded_width;
        reconfig.height = dec.coded_height;
        reconfig.target_width = dec.geometry.width;
        reconfig.target_height = dec.geometry.height;
        reconfig.bit_depth_minus_8 = dec.bit_depth_minus_8;
        reconfig.num_decode_surfaces = dec.num_decode_surfaces;
        reconfig.display_rect.left = static_cast<int16_t>(dec.geometry.disp_rect.left);
        reconfig.display_rect.top = static_cast<int16_t>(dec.geometry.disp_rect.top);
        reconfig.display_rect.right = static_cast<int16_t>(dec.geometry.disp_rect.right);
        reconfig.display_rect.bottom = static_cast<int16_t>(dec.geometry.disp_rect.bottom);
        rocDecStatus status = rocDecReconfigureDecoder(dec.decoder, &reconfig);
        if (status != ROCDEC_SUCCESS) {
            std::cerr << "Error: rocDecReconfigureDecoder failed: " << rocDecGetErrorName(status) << std::endl;
            dec.failed = true;
            return 0;
        }
    }
    std::cout << "info: Reconfigured: coded size " << dec.coded_width << "x" << dec.coded_height << ", output "
              << dec.geometry.width << "x" << dec.geometry.height << ", bit depth " << dec.geometry.bit_depth << std::endl;
    return 1;
}

static int ROCDECAPI HandlePictureDecode(void *user_data, RocdecPicParams *params) {
    AsyncDecoder &dec = *static_cast<AsyncDecoder *>(user_data);
    rocDecStatus status = rocDecDecodeFrame(dec.decoder, params);
    if (status != ROCDEC_SUCCESS) {
        std::cerr << "Error: rocDecDecodeFrame failed: " << rocDecGetErrorName(status) << std::endl;
        dec.failed = true;
        return 0;
    }
    dec.decoded_pictures++;
    return 1;
}

static int ROCDECAPI HandlePictureDisplay(void *user_data, RocdecParserDispInfo *disp_info) {
    AsyncDecoder &dec = *static_cast<AsyncDecoder *>(user_data);
    // The parser ignores the return value and keeps displaying the rest of a flush, so stop here
    // once the run has failed; otherwise failed copies could empty the pool and block forever.
    if (dec.failed) return 0;
    auto start = Clock::now();
    RocdecProcParams proc_params = {};
    proc_params.progressive_frame = disp_info->progressive_frame;
    proc_params.top_field_first = disp_info->top_field_first;
    void *src[3] = {};
    uint32_t src_pitch[3] = {};
    rocDecStatus status = rocDecGetVideoFrame(dec.decoder, disp_info->picture_index, src, src_pitch, &proc_params);
    if (status != ROCDEC_SUCCESS) {
        std::cerr << "Error: rocDecGetVideoFrame failed: " << rocDecGetErrorName(status) << std::endl;
        dec.failed = true;
        return 0;
    }

    FrameItem item;
    item.pts = disp_info->pts;
    item.geometry = dec.geometry;
    if (dec.opt.policy == Policy::SyncCopy) {
        {
            // Back-pressure: wait for a free pool buffer if the consumer has fallen behind.
            std::unique_lock<std::mutex> lock(dec.mtx);
            dec.cv.wait(lock, [&] { return !dec.free_buffers.empty() || dec.consumer_stopped; });
            if (dec.consumer_stopped) {
                dec.failed = true;
                return 0;
            }
            item.buffer = dec.free_buffers.back();
            dec.free_buffers.pop_back();
        }
        hipError_t hip_status = CopyFrame(src, src_pitch, item.geometry, item.buffer, dec.producer_stream);
        if (hip_status == hipSuccess) hip_status = hipStreamSynchronize(dec.producer_stream);
        if (hip_status != hipSuccess) {
            std::cerr << "Error: frame copy failed: " << hipGetErrorName(hip_status) << std::endl;
            std::lock_guard<std::mutex> lock(dec.mtx);
            dec.free_buffers.push_back(item.buffer);
            dec.failed = true;
            return 0;
        }
    } else {
        std::copy(src, src + 3, item.src);
        std::copy(src_pitch, src_pitch + 3, item.src_pitch);
    }

    {
        std::unique_lock<std::mutex> lock(dec.mtx);
        dec.cv.wait(lock, [&] { return dec.queue.size() < dec.opt.queue_depth || dec.consumer_stopped; });
        if (dec.consumer_stopped) {
            dec.failed = true;
            return 0;
        }
        item.index = dec.displayed++;
        dec.queue.push_back(item);
    }
    dec.cv.notify_all();
    dec.display_callback_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return 1;
}

// ------------------------------------------------------------------------------------------
// Consumer thread: waits as the workload asks, copies no-copy frames, hashes in display order.
// ------------------------------------------------------------------------------------------
struct ConsumerResult {
    bool ok = true;
    uint64_t frames = 0;
};

// Called when the consumer exits before the end of the stream. Without it the producer would
// wait forever for a free buffer, queue space or a drain that only the consumer can provide.
static void StopConsumer(AsyncDecoder &dec, ConsumerResult *result, const char *reason) {
    std::cerr << "Error: consumer stopped: " << reason << std::endl;
    result->ok = false;
    {
        std::lock_guard<std::mutex> lock(dec.mtx);
        dec.consumer_stopped = true;
    }
    dec.cv.notify_all();
}

static void ConsumerLoop(AsyncDecoder &dec, MD5Generator *md5, std::ofstream *md5_frames, ConsumerResult *result) {
    // HIP's current device is per thread, so select the decoder's device here as well.
    if (hipSetDevice(dec.opt.device_id) != hipSuccess) {
        StopConsumer(dec, result, "hipSetDevice failed");
        return;
    }
    hipStream_t stream = nullptr;
    uint8_t *late_copy_buffer = nullptr;
    size_t late_copy_size = 0;
    std::vector<uint8_t> host_frame;
    std::mt19937 rng(12345);
    if (hipStreamCreate(&stream) != hipSuccess) {
        StopConsumer(dec, result, "hipStreamCreate failed");
        return;
    }

    while (true) {
        FrameItem item;
        {
            std::unique_lock<std::mutex> lock(dec.mtx);
            dec.cv.wait(lock, [&] { return !dec.queue.empty() || dec.eos; });
            if (dec.queue.empty()) break;  // end of stream and nothing left
            item = dec.queue.front();
            dec.queue.pop_front();
        }
        dec.cv.notify_all();  // a queue slot is free

        if (dec.opt.workload == Workload::Lag) {
            // Deterministic hold: frame n is handled only after frame n+L has been displayed. The
            // display callback waits for each decode, so those decodes have finished by then.
            std::unique_lock<std::mutex> lock(dec.mtx);
            dec.cv.wait(lock, [&] { return dec.displayed > item.index + dec.opt.lag || dec.draining || dec.eos; });
        } else if (dec.opt.workload == Workload::Hold) {
            std::uniform_int_distribution<uint32_t> dist(dec.opt.hold_min_ms, dec.opt.hold_max_ms);
            auto until = Clock::now() + std::chrono::milliseconds(dist(rng));
            std::unique_lock<std::mutex> lock(dec.mtx);
            dec.cv.wait_until(lock, until, [&] { return dec.draining; });
        }

        uint8_t *frame = item.buffer;
        if (dec.opt.policy == Policy::NoCopy) {
            // Read the decode surface now, possibly after it has been reused.
            size_t size = item.geometry.FrameSize();
            if (size > late_copy_size) {
                if (late_copy_buffer) CheckCleanup(hipFree(late_copy_buffer), "hipFree");
                late_copy_buffer = nullptr;
                if (hipMalloc(reinterpret_cast<void **>(&late_copy_buffer), size) != hipSuccess) {
                    result->ok = false;
                    late_copy_size = 0;
                } else {
                    late_copy_size = size;
                }
            }
            frame = late_copy_buffer;
            if (late_copy_buffer && (CopyFrame(item.src, item.src_pitch, item.geometry, late_copy_buffer, stream) != hipSuccess ||
                                 hipStreamSynchronize(stream) != hipSuccess)) {
                result->ok = false;
                frame = nullptr;
            }
        }

        // Copy the frame to the host here rather than in MD5Generator, which logs a failed copy
        // but cannot report it, so the digest would silently leave the frame out.
        if (frame && (md5 || md5_frames)) {
            host_frame.resize(item.geometry.FrameSize());
            // Copy on the consumer's stream: hipMemcpyDtoH uses the legacy null stream, which waits for
            // the producer's stream and would count the consumer's copies as producer time.
            hipError_t status = hipMemcpyAsync(host_frame.data(), frame, host_frame.size(), hipMemcpyDeviceToHost, stream);
            if (status == hipSuccess) status = hipStreamSynchronize(stream);
            if (status != hipSuccess) {
                std::cerr << "Error: copying frame " << item.index << " to the host failed: " << hipGetErrorName(status) << std::endl;
                result->ok = false;
            } else {
                OutputSurfaceInfo info = DescribePackedFrame(item.geometry);
                if (md5) md5->UpdateMd5ForFrame(host_frame.data(), &info);
                if (md5_frames) {
                    MD5Generator frame_md5;
                    frame_md5.InitMd5();
                    frame_md5.UpdateMd5ForFrame(host_frame.data(), &info);
                    uint8_t *digest = nullptr;
                    frame_md5.FinalizeMd5(&digest);
                    *md5_frames << item.index << " " << DigestToString(digest) << std::endl;
                    if (!*md5_frames) {
                        std::cerr << "Error: cannot write the MD5 of frame " << item.index << " to the -md5_frames file" << std::endl;
                        result->ok = false;
                        md5_frames = nullptr;  // stop writing; the run fails
                    }
                }
            }
        }
        result->frames++;

        {
            std::lock_guard<std::mutex> lock(dec.mtx);
            if (item.buffer) dec.free_buffers.push_back(item.buffer);
            dec.consumed++;
        }
        dec.cv.notify_all();
    }
    if (late_copy_buffer) CheckCleanup(hipFree(late_copy_buffer), "hipFree");
    CheckCleanup(hipStreamDestroy(stream), "hipStreamDestroy");
}

int main(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next = [&](const char *name) -> std::string {
            if (++i >= argc) ShowHelpAndExit(name);
            return argv[i];
        };
        if (arg == "-h") ShowHelpAndExit();
        else if (arg == "-i") opt.input_path = next("-i");
        else if (arg == "-d") {
            uint32_t device_id = 0;
            if (!ParseUint(next("-d"), 0, UINT8_MAX, device_id)) ShowHelpAndExit("-d");
            opt.device_id = static_cast<int>(device_id);
        }
        else if (arg == "-p") {
            std::string p = next("-p");
            if (p == "sync-copy") opt.policy = Policy::SyncCopy;
            else if (p == "no-copy") opt.policy = Policy::NoCopy;
            else ShowHelpAndExit("-p");
        }
        else if (arg == "-w") { if (!ParseWorkload(next("-w"), opt)) ShowHelpAndExit("-w"); }
        else if (arg == "-dd") { if (!ParseUint(next("-dd"), 0, UINT32_MAX, opt.display_delay)) ShowHelpAndExit("-dd"); }
        else if (arg == "-ds") { if (!ParseUint(next("-ds"), 0, UINT32_MAX, opt.decode_surfaces)) ShowHelpAndExit("-ds"); }
        else if (arg == "-q") { if (!ParseUint(next("-q"), 1, kMaxQueueDepth, opt.queue_depth)) ShowHelpAndExit("-q"); }
        else if (arg == "-md5") opt.md5 = true;
        else if (arg == "-md5_check") { opt.md5 = true; opt.md5_check_path = next("-md5_check"); }
        else if (arg == "-md5_frames") opt.md5_frames_path = next("-md5_frames");
        else if (arg == "-expect_mismatch") opt.expect_mismatch = true;
        else ShowHelpAndExit(argv[i]);
    }
    if (opt.input_path.empty()) ShowHelpAndExit("-i");
    if (opt.expect_mismatch && opt.md5_check_path.empty()) {
        std::cerr << "Error: -expect_mismatch needs -md5_check" << std::endl;
        return 1;
    }
    // A consumer holding frame n waits for frame n+L to be displayed, which needs L more queue slots.
    if (opt.workload == Workload::Lag && opt.queue_depth < opt.lag + 1) {
        std::cout << "info: Raising queue depth from " << opt.queue_depth << " to " << opt.lag + 1 << " for lag:" << opt.lag << std::endl;
        opt.queue_depth = opt.lag + 1;
    }

    AsyncDecoder dec;
    dec.opt = opt;
    int exit_code = 0;
    try {
        if (hipSetDevice(opt.device_id) != hipSuccess) {
            std::cerr << "Error: hipSetDevice(" << opt.device_id << ") failed" << std::endl;
            return 1;
        }
        VideoDemuxer demuxer(opt.input_path.c_str());
        // The demuxer logs a failed open or a missing video stream instead of throwing.
        if (demuxer.GetCodecID() == AV_CODEC_ID_NONE) {
            std::cerr << "Error: cannot open a video stream in " << opt.input_path << std::endl;
            return 1;
        }
        dec.codec = AVCodec2RocDecVideoCodec(demuxer.GetCodecID());

        // Skip (exit 77) when the GPU does not support the codec, as the other samples do.
        RocdecDecodeCaps caps = {};
        caps.device_id = opt.device_id;
        caps.codec_type = dec.codec;
        caps.chroma_format = rocDecVideoChromaFormat_420;
        caps.bit_depth_minus_8 = demuxer.GetBitDepth() - 8;
        rocDecStatus caps_status = rocDecGetDecoderCaps(&caps);
        if (caps_status != ROCDEC_SUCCESS) {
            std::cerr << "Error: rocDecGetDecoderCaps failed: " << rocDecGetErrorName(caps_status) << std::endl;
            return 1;
        }
        if (!caps.is_supported) {
            std::cout << "info: Codec not supported on GPU, skipping!" << std::endl;
            return 77;
        }

        if (hipStreamCreate(&dec.producer_stream) != hipSuccess) {
            std::cerr << "Error: hipStreamCreate failed" << std::endl;
            return 1;
        }
        RocdecParserParams parser_params = {};
        parser_params.codec_type = dec.codec;
        parser_params.max_num_decode_surfaces = opt.decode_surfaces;
        parser_params.max_display_delay = opt.display_delay;
        parser_params.user_data = &dec;
        parser_params.pfn_sequence_callback = HandleVideoSequence;
        parser_params.pfn_decode_picture = HandlePictureDecode;
        parser_params.pfn_display_picture = HandlePictureDisplay;
        rocDecStatus status = rocDecCreateVideoParser(&dec.parser, &parser_params);
        if (status != ROCDEC_SUCCESS) {
            std::cerr << "Error: rocDecCreateVideoParser failed: " << rocDecGetErrorName(status) << std::endl;
            return 1;
        }

        MD5Generator md5;
        if (opt.md5) md5.InitMd5();
        std::ofstream md5_frames;
        if (!opt.md5_frames_path.empty()) {
            md5_frames.open(opt.md5_frames_path);
            if (!md5_frames) {
                std::cerr << "Error: cannot open " << opt.md5_frames_path << std::endl;
                return 1;
            }
        }
        ConsumerResult consumer_result;
        std::thread consumer(ConsumerLoop, std::ref(dec), opt.md5 ? &md5 : nullptr,
                             md5_frames.is_open() ? &md5_frames : nullptr, &consumer_result);

        std::cout << "info: Policy " << (opt.policy == Policy::SyncCopy ? "sync-copy" : "no-copy")
                  << ", queue depth " << opt.queue_depth << ", display delay " << opt.display_delay << std::endl;
        auto start = Clock::now();
        uint8_t *packet = nullptr;
        int packet_size = 0;
        int64_t pts = 0;
        bool parse_failed = false;
        do {
            demuxer.Demux(&packet, &packet_size, &pts);
            // An empty packet is also returned on a demux error; don't treat that as the end of the
            // stream, or a truncated decode could pass (and -expect_mismatch would report a mismatch).
            if (demuxer.DemuxFailed()) {
                std::cerr << "Error: demuxing failed" << std::endl;
                parse_failed = true;
                break;
            }
            RocdecSourceDataPacket pkt = {};
            pkt.payload = packet;
            pkt.payload_size = static_cast<uint32_t>(packet_size);
            pkt.flags = ROCDEC_PKT_TIMESTAMP | (packet_size == 0 ? ROCDEC_PKT_ENDOFSTREAM : 0);
            pkt.pts = pts;
            if (rocDecParseVideoData(dec.parser, &pkt) != ROCDEC_SUCCESS || dec.failed || dec.unsupported) {
                parse_failed = true;
                break;
            }
        } while (packet_size);

        {
            std::lock_guard<std::mutex> lock(dec.mtx);
            dec.eos = true;
        }
        dec.cv.notify_all();
        consumer.join();
        double elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

        rocDecDestroyVideoParser(dec.parser);
        if (dec.decoder) rocDecDestroyDecoder(dec.decoder);
        FreeBufferPool(dec);
        CheckCleanup(hipStreamDestroy(dec.producer_stream), "hipStreamDestroy");

        if (dec.unsupported) {
            std::cout << "info: Codec not supported on GPU, skipping!" << std::endl;
            return 77;
        }
        if (parse_failed || !consumer_result.ok) {
            std::cerr << "Error: decoding failed" << std::endl;
            return 1;
        }
        std::cout << "info: Total pictures decoded: " << dec.decoded_pictures << std::endl;
        std::cout << "info: Total frames output/displayed: " << consumer_result.frames << std::endl;
        if (consumer_result.frames == 0) {
            std::cerr << "Error: No frames were decoded!" << std::endl;
            return 1;
        }
        std::cout << "info: avg decode FPS: " << dec.decoded_pictures * 1000.0 / elapsed_ms << std::endl;
        std::cout << "info: avg end-to-end FPS: " << consumer_result.frames * 1000.0 / elapsed_ms << std::endl;
        std::cout << "info: producer time in the display callback: " << dec.display_callback_ms << " ms ("
                  << 100.0 * dec.display_callback_ms / elapsed_ms << "% of " << elapsed_ms << " ms)" << std::endl;

        if (opt.md5) {
            uint8_t *digest = nullptr;
            md5.FinalizeMd5(&digest);
            std::string digest_string = DigestToString(digest);
            std::cout << "MD5 message digest: " << digest_string << std::endl;
            if (!opt.md5_check_path.empty()) {
                std::ifstream ref_file(opt.md5_check_path);
                std::string ref_string;
                if (!(ref_file >> ref_string)) {
                    std::cerr << "Error: cannot read the reference MD5 from " << opt.md5_check_path << std::endl;
                    return 1;
                }
                // Reject a malformed reference, which would otherwise count as a mismatch and let
                // -expect_mismatch pass whether or not the output was corrupted.
                if (ref_string.size() != 32 || !std::all_of(ref_string.begin(), ref_string.end(), [](unsigned char c) { return std::isxdigit(c); })) {
                    std::cerr << "Error: the reference in " << opt.md5_check_path << " is not a 32-digit hexadecimal MD5 digest" << std::endl;
                    return 1;
                }
                std::transform(ref_string.begin(), ref_string.end(), ref_string.begin(), ::tolower);
                bool match = ref_string == digest_string;
                if (match) std::cout << "MD5 digest matches the reference MD5 digest: " << ref_string << std::endl;
                else std::cout << "MD5 digest does not match the reference MD5 digest: " << ref_string << std::endl;
                if (opt.expect_mismatch) {
                    std::cout << "info: -expect_mismatch: " << (match ? "FAILED, the output was not corrupted" : "PASSED, the mismatch was detected") << std::endl;
                    exit_code = match ? 1 : 0;
                } else {
                    exit_code = match ? 0 : 1;
                }
            }
        }
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return exit_code;
}
