
/*
Copyright (c) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <assert.h>
#include <chrono>
#include <cstring>
#include <exception>
#include <hip/hip_runtime.h>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <string.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
extern "C"
{
#include <libavutil/md5.h>
#include <libavutil/mem.h>
}
#include <rocdecode/rocdecode.h>
#include <rocdecode/rocparser.h>

/*!
 * \file
 * \brief The AMD Video Decode Library.
 *
 * \defgroup group_amd_roc_video_dec rocDecode Video Decode: AMD Video Decode API
 * \brief AMD The rocDecode video decoder for AMD’s GPUs.
 */

#define MAX_FRAME_NUM 16

typedef int(ROCDECAPI* PFNRECONFIGUEFLUSHCALLBACK)(void*, uint32_t, void*);

typedef enum SeiAvcHevcPayloadType_enum
{
    SEI_TYPE_TIME_CODE              = 136,
    SEI_TYPE_USER_DATA_UNREGISTERED = 5
} SeiAvcHevcPayloadType;

typedef enum OutputSurfaceMemoryType_enum
{
    OUT_SURFACE_MEM_DEV_INTERNAL = 0, /**<  Internal interopped decoded surface
                                         memory(original mapped decoded surface) */
    OUT_SURFACE_MEM_DEV_COPIED =
        1, /**<  decoded output will be copied to a separate device memory (the user
              doesn't need to call release) **/
    OUT_SURFACE_MEM_HOST_COPIED =
        2, /**<  decoded output will be copied to a separate host memory (the user doesn't
              need to call release) **/
    OUT_SURFACE_MEM_NOT_MAPPED = 3 /**< <  decoded output is not available (interop won't
                                      be used): useful for decode only performance app*/
} OutputSurfaceMemoryType;

#define TOSTR(X) std::to_string(static_cast<int>(X))
#define STR(X)   std::string(X)

#if DBGINFO
#    define INFO(X)                                                                      \
        std::clog << "[INF] " << " {" << __func__ << "} " << " " << X << std::endl;
#else
#    define INFO(X) ;
#endif
#define ERR(X) std::cerr << "[ERR] " << " {" << __func__ << "} " << " " << X << std::endl;

class RocVideoDecodeException : public std::exception
{
public:
    explicit RocVideoDecodeException(const std::string& message, const int err_code)
    : _message(message)
    , _err_code(err_code)
    {}
    explicit RocVideoDecodeException(const std::string& message)
    : _message(message)
    , _err_code(-1)
    {}
    virtual const char* what() const throw() override { return _message.c_str(); }
    int                 Geterror_code() const { return _err_code; }

private:
    std::string _message;
    int         _err_code;
};

#define ROCDEC_THROW(X, CODE)                                                            \
    throw RocVideoDecodeException(" { " + std::string(__func__) + " } " + X, CODE);
#define THROW(X) throw RocVideoDecodeException(" { " + std::string(__func__) + " } " + X);

#define ROCDEC_API_CALL(rocDecAPI)                                                       \
    do                                                                                   \
    {                                                                                    \
        rocDecStatus error_code = rocDecAPI;                                             \
        if(error_code != ROCDEC_SUCCESS)                                                 \
        {                                                                                \
            std::ostringstream error_log;                                                \
            error_log << #rocDecAPI << " returned " << rocDecGetErrorName(error_code)    \
                      << " at " << __FILE__ << ":" << __LINE__;                          \
            ROCDEC_THROW(error_log.str(), error_code);                                   \
        }                                                                                \
    } while(0)

#define HIP_API_CALL(call)                                                               \
    do                                                                                   \
    {                                                                                    \
        hipError_t hip_status = call;                                                    \
        if(hip_status != hipSuccess)                                                     \
        {                                                                                \
            const char* sz_err_name = NULL;                                              \
            sz_err_name             = hipGetErrorName(hip_status);                       \
            std::ostringstream error_log;                                                \
            error_log << "hip API error " << sz_err_name;                                \
            ROCDEC_THROW(error_log.str(), hip_status);                                   \
        }                                                                                \
    } while(0)

struct Rect
{
    int left;
    int top;
    int right;
    int bottom;
};

struct Dim
{
    int w, h;
};

static inline int
align(int value, int alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

typedef struct DecFrameBuffer_
{
    uint8_t* frame_ptr;     /**< device memory pointer for the decoded frame */
    int64_t  pts;           /**<  timestamp for the decoded frame */
    int      picture_index; /**<  surface index for the decoded frame */
} DecFrameBuffer;

typedef struct OutputSurfaceInfoType
{
    uint32_t output_width;   /**< Output width of decoded surface*/
    uint32_t output_height;  /**< Output height of decoded surface*/
    uint32_t output_pitch;   /**< Output pitch in bytes of luma plane, chroma pitch can be
                                inferred based on chromaFormat*/
    uint32_t output_vstride; /**< Output vertical stride in case of using internal mem
                                pointer **/
    uint32_t bytes_per_pixel;              /**< Output BytesPerPixel of decoded image*/
    uint32_t bit_depth;                    /**< Output BitDepth of the image*/
    uint32_t num_chroma_planes;            /**< Output Chroma number of planes*/
    uint64_t output_surface_size_in_bytes; /**< Output Image Size in Bytes; including both
                                              luma and chroma planes*/
    rocDecVideoSurfaceFormat surface_format; /**< Chroma format of the decoded image*/
    OutputSurfaceMemoryType  mem_type;       /**< Output mem_type of the surface*/
} OutputSurfaceInfo;

typedef struct ReconfigParams_t
{
    PFNRECONFIGUEFLUSHCALLBACK p_fn_reconfigure_flush;
    void*                      p_reconfig_user_struct;
    uint32_t                   reconfig_flush_mode;
} ReconfigParams;

class RocVideoDecoder
{
public:
    /**
     * @brief Construct a new Roc Video Decoder object
     *
     * @param hip_ctx
     * @param b_use_device_mem
     * @param codec
     * @param device_id
     * @param b_low_latency
     * @param device_frame_pitched
     * @param p_crop_rect
     * @param extract_user_SEI_Message
     * @param max_width
     * @param max_height
     * @param clk_rate
     * @param force_zero_latency
     */
    RocVideoDecoder(int device_id, OutputSurfaceMemoryType out_mem_type,
                    rocDecVideoCodec codec, bool force_zero_latency = false,
                    const Rect* p_crop_rect       = nullptr,
                    bool extract_user_SEI_Message = false, uint32_t disp_delay = 0,
                    int max_width = 0, int max_height = 0, uint32_t clk_rate = 1000);
    ~RocVideoDecoder();

    rocDecVideoCodec GetCodecId() { return codec_id_; }

    hipStream_t GetStream() { return hip_stream_; }

    /**
     * @brief Get the output frame width
     */
    uint32_t GetWidth()
    {
        assert(disp_width_);
        return disp_width_;
    }

    /**
     *  @brief  This function is used to get the actual decode width
     */
    int GetDecodeWidth()
    {
        assert(coded_width_);
        return coded_width_;
    }

    /**
     * @brief Get the output frame height
     */
    uint32_t GetHeight()
    {
        assert(disp_height_);
        return disp_height_;
    }

    /**
     *  @brief  This function is used to get the current chroma height.
     */
    int GetChromaHeight()
    {
        assert(chroma_height_);
        return chroma_height_;
    }

    /**
     *  @brief  This function is used to get the number of chroma planes.
     */
    int GetNumChromaPlanes()
    {
        assert(num_chroma_planes_);
        return num_chroma_planes_;
    }

    /**
     *   @brief  This function is used to get the current frame size based on pixel
     * format.
     */
    int GetFrameSize()
    {
        assert(disp_width_);
        return disp_width_ * (disp_height_ + (chroma_height_ * num_chroma_planes_)) *
               byte_per_pixel_;
    }

    /**
     *   @brief  This function is used to get the current frame size based on pitch
     */
    int GetFrameSizePitched()
    {
        assert(surface_stride_);
        return surface_stride_ * (disp_height_ + (chroma_height_ * num_chroma_planes_));
    }

    /**
     * @brief Get the Bit Depth and BytesPerPixel associated with the pixel format
     *
     * @return uint32_t
     */
    uint32_t GetBitDepth()
    {
        assert(bitdepth_minus_8_);
        return (bitdepth_minus_8_ + 8);
    }
    uint32_t GetBytePerPixel()
    {
        assert(byte_per_pixel_);
        return byte_per_pixel_;
    }
    /**
     * @brief Functions to get the output surface attributes
     */
    size_t GetSurfaceSize()
    {
        assert(surface_size_);
        return surface_size_;
    }
    uint32_t GetSurfaceStride()
    {
        assert(surface_stride_);
        return surface_stride_;
    }
    // RocDecImageFormat GetSubsampling() { return subsampling_; }
    /**
     * @brief Get the name of the output format
     *
     * @param codec_id
     * @return std::string
     */
    const char* GetCodecFmtName(rocDecVideoCodec codec_id);

    /**
     * @brief function to return the name from surface_format_id
     *
     * @param surface_format_id - enum for surface format
     * @return const char*
     */
    const char* GetSurfaceFmtName(rocDecVideoSurfaceFormat surface_format_id);

    /**
     * @brief Get the pointer to the Output Image Info
     *
     * @param surface_info ptr to output surface info
     * @return true
     * @return false
     */
    bool GetOutputSurfaceInfo(OutputSurfaceInfo** surface_info);

    /**
     * @brief Function to set the Reconfig Params object
     *
     * @param p_reconfig_params: pointer to reconfig params struct
     * @return true : success
     * @return false : fail
     */
    bool SetReconfigParams(ReconfigParams* p_reconfig_params,
                           bool            b_force_reconfig_flush = false);

    /**
     * @brief Function to force Reconfigure Flush: needed for random seeking to key frames
     *
     * @return int 1: Success 0: Fail
     */
    int FlushAndReconfigure();
    /**
     * @brief this function decodes a frame and returns the number of frames avalable for
     * display
     *
     * @param data - pointer to the data buffer that is to be decode
     * @param size - size of the data buffer in bytes
     * @param pts - presentation timestamp
     * @param flags - video packet flags
     * @param num_decoded_pics - nummber of pictures decoded in this call
     * @return int - num of frames to display
     */
    int DecodeFrame(const uint8_t* data, size_t size, int pkt_flags, int64_t pts = 0,
                    int* num_decoded_pics = nullptr);
    /**
     * @brief This function returns a decoded frame and timestamp. This should be called
     * in a loop fetching all the available frames
     *
     */
    uint8_t* GetFrame(int64_t* pts);

    /**
     * @brief function to release frame after use by the application: Only used with
     * "OUT_SURFACE_MEM_DEV_INTERNAL"
     *
     * @param pTimestamp - timestamp of the frame to be released (unmapped)
     * @param b_flushing - true when flushing
     * @return true      - success
     * @return false     - falied
     */
    bool ReleaseFrame(int64_t pTimestamp, bool b_flushing = false);

    /**
     * @brief utility function to save image to a file
     *
     * @param output_file_name - file to write
     * @param dev_mem - dev_memory pointer of the frame
     * @param image_info - output image info
     * @param is_output_RGB - to write in RGB
     */
    // void SaveImage(std::string output_file_name, void* dev_mem, OutputImageInfo*
    // image_info, bool is_output_RGB = 0);

    /**
     * @brief Get the Device info for the current device
     *
     * @param device_name
     * @param gcn_arch_name
     * @param pci_bus_id
     * @param pci_domain_id
     * @param pci_device_id
     */
    void GetDeviceinfo(std::string& device_name, std::string& gcn_arch_name,
                       int& pci_bus_id, int& pci_domain_id, int& pci_device_id);

    /**
     * @brief Helper function to dump decoded output surface to file
     *
     * @param output_file_name  - Output file name
     * @param dev_mem           - pointer to surface memory
     * @param surf_info         - surface info
     * @param rgb_image_size    - image size for rgb (optional). A non_zero value
     * indicates the surf_mem holds an rgb interleaved image and the entire size will be
     * dumped to file
     */
    void SaveFrameToFile(std::string output_file_name, void* surf_mem,
                         OutputSurfaceInfo* surf_info, size_t rgb_image_size = 0);

    /**
     * @brief Helper funtion to close a existing file and dump to new file in case of
     * multiple files using same decoder
     */
    void ResetSaveFrameToFile();

    /**
     * @brief Helper function to start MD5 calculation
     */
    void InitMd5();

    void UpdateMd5ForDataBuffer(void* pDevMem, int rgb_image_size);

    /**
     * @brief Helper function to dump decoded output surface to file
     *
     * @param dev_mem           - pointer to surface memory
     * @param surf_info         - surface info
     */
    void UpdateMd5ForFrame(void* surf_mem, OutputSurfaceInfo* surf_info);

    /**
     * @brief Helper function to complete MD5 calculation
     *
     * @param [out] digest Pointer to the 16 byte message digest
     */
    void FinalizeMd5(uint8_t** digest);
    /**
     * @brief Get the Num Of Flushed Frames from video decoder object
     *
     * @return int32_t
     */
    int32_t GetNumOfFlushedFrames() { return num_frames_flushed_during_reconfig_; }

    /*! \brief Function to wait for the decode completion of the last submitted picture
     */
    void WaitForDecodeCompletion();

    // Session overhead refers to decoder initialization and deinitialization time
    void AddDecoderSessionOverHead(std::thread::id session_id, double duration)
    {
        session_overhead_[session_id] += duration;
    }
    double GetDecoderSessionOverHead(std::thread::id session_id)
    {
        if(session_overhead_.find(session_id) != session_overhead_.end())
        {
            return session_overhead_[session_id];
        }
        else
        {
            return 0;
        }
    }

    /**
     * @brief Check if the given Video Codec is supported on the given GPU
     *
     * @return rocDecStatus
     */
    bool CodecSupported(int device_id, rocDecVideoCodec codec_id, uint32_t bit_depth);

private:
    /**
     *   @brief  Callback function to be registered for getting a callback when decoding
     * of sequence starts
     */
    static int ROCDECAPI HandleVideoSequenceProc(void*              p_user_data,
                                                 RocdecVideoFormat* p_video_format)
    {
        return ((RocVideoDecoder*) p_user_data)->HandleVideoSequence(p_video_format);
    }

    /**
     *   @brief  Callback function to be registered for getting a callback when a decoded
     * frame is ready to be decoded
     */
    static int ROCDECAPI HandlePictureDecodeProc(void*            p_user_data,
                                                 RocdecPicParams* p_pic_params)
    {
        return ((RocVideoDecoder*) p_user_data)->HandlePictureDecode(p_pic_params);
    }

    /**
     *   @brief  Callback function to be registered for getting a callback when a decoded
     * frame is available for display
     */
    static int ROCDECAPI HandlePictureDisplayProc(void*                 p_user_data,
                                                  RocdecParserDispInfo* p_disp_info)
    {
        return ((RocVideoDecoder*) p_user_data)->HandlePictureDisplay(p_disp_info);
    }

    /**
     *   @brief  Callback function to be registered for getting a callback when all the
     * unregistered user SEI Messages are parsed for a frame.
     */
    static int ROCDECAPI HandleSEIMessagesProc(void*                 p_user_data,
                                               RocdecSeiMessageInfo* p_sei_message_info)
    {
        return ((RocVideoDecoder*) p_user_data)->GetSEIMessage(p_sei_message_info);
    }

    /**
     *   @brief  This function gets called when a sequence is ready to be decoded. The
     function also gets called when there is format change
    */
    int HandleVideoSequence(RocdecVideoFormat* p_video_format);

    /**
     *   @brief  This function gets called when a picture is ready to be decoded.
     * cuvidDecodePicture is called from this function to decode the picture
     */
    int HandlePictureDecode(RocdecPicParams* p_pic_params);

    /**
     *   @brief  This function gets called after a picture is decoded and available for
     display. Frames are fetched and stored in internal buffer
    */
    int HandlePictureDisplay(RocdecParserDispInfo* p_disp_info);
    /**
     *   @brief  This function gets called when all unregistered user SEI messages are
     * parsed for a frame
     */
    int GetSEIMessage(RocdecSeiMessageInfo* p_sei_message_info);

    /**
     *   @brief  This function reconfigure decoder if there is a change in sequence
     * params.
     */
    int ReconfigureDecoder(RocdecVideoFormat* p_video_format);

    /**
     * @brief function to release all internal frames and clear the vp_frames_q_ (used
     * with reconfigure): Only used with "OUT_SURFACE_MEM_DEV_INTERNAL"
     *
     * @return true      - success
     * @return false     - falied
     */
    bool ReleaseInternalFrames();

    /**
     * @brief Function to Initialize GPU-HIP
     *
     */
    bool InitHIP(int device_id);

    /**
     * @brief Function to get start time
     *
     */
    std::chrono::_V2::system_clock::time_point StartTimer();

    /**
     * @brief Function to get elapsed time
     *
     */
    double StopTimer(const std::chrono::_V2::system_clock::time_point& start_time);

    int                      num_devices_;
    int                      device_id_;
    RocdecVideoParser        rocdec_parser_         = nullptr;
    rocDecDecoderHandle      roc_decoder_           = nullptr;
    OutputSurfaceMemoryType  out_mem_type_          = OUT_SURFACE_MEM_DEV_INTERNAL;
    bool                     b_extract_sei_message_ = false;
    bool                     b_force_zero_latency_  = false;
    uint32_t                 disp_delay_;
    ReconfigParams*          p_reconfig_params_                  = nullptr;
    bool                     b_force_recofig_flush_              = false;
    int32_t                  num_frames_flushed_during_reconfig_ = 0;
    hipDeviceProp_t          hip_dev_prop_;
    hipStream_t              hip_stream_;
    rocDecVideoCodec         codec_id_             = rocDecVideoCodec_NumCodecs;
    rocDecVideoChromaFormat  video_chroma_format_  = rocDecVideoChromaFormat_420;
    rocDecVideoSurfaceFormat video_surface_format_ = rocDecVideoSurfaceFormat_NV12;
    RocdecSeiMessageInfo*    curr_sei_message_ptr_ = nullptr;
    RocdecSeiMessageInfo     sei_message_display_q_[MAX_FRAME_NUM];
    RocdecVideoFormat*       curr_video_format_ptr_ = nullptr;
    int                      output_frame_cnt_ = 0, output_frame_cnt_ret_ = 0;
    int                      decoded_pic_cnt_ = 0;
    int                      decode_poc_      = 0, pic_num_in_dec_order_[MAX_FRAME_NUM];
    int                      num_alloced_frames_   = 0;
    int                      last_decode_surf_idx_ = 0;
    std::ostringstream       input_video_info_str_;
    int                      bitdepth_minus_8_ = 0;
    uint32_t                 byte_per_pixel_   = 1;
    uint32_t                 coded_width_      = 0;
    uint32_t                 disp_width_       = 0;
    uint32_t                 coded_height_     = 0;
    uint32_t                 disp_height_      = 0;
    uint32_t                 target_width_     = 0;
    uint32_t                 target_height_    = 0;
    int                      max_width_ = 0, max_height_ = 0;
    uint32_t                 chroma_height_     = 0;
    uint32_t                 num_chroma_planes_ = 0;
    uint32_t                 num_components_    = 0;
    uint32_t                 surface_stride_    = 0;
    uint32_t
        surface_vstride_ = 0,
        chroma_vstride_ =
            0;  // vertical stride between planes: used when using internal dev memory
    size_t                      surface_size_        = 0;
    OutputSurfaceInfo           output_surface_info_ = {};
    std::mutex                  mtx_vp_frame_;
    std::vector<DecFrameBuffer> vp_frames_;  // vector of decoded frames
    std::queue<DecFrameBuffer>  vp_frames_q_;
    Rect disp_rect_ = {};  // displayable area specified in the bitstream
    Rect crop_rect_ = {};  // user specified region of interest within diplayable area
                           // disp_rect_
    FILE*           fp_sei_ = NULL;
    FILE*           fp_out_ = NULL;
    struct AVMD5*   md5_ctx_;
    uint8_t         md5_digest_[16];
    bool            is_decoder_reconfigured_ = false;
    std::string     current_output_filename  = "";
    uint32_t        extra_output_file_count_ = 0;
    std::thread::id decoder_session_id_;  // Decoder session identifier. Used to gather
                                          // session level stats.
    std::unordered_map<std::thread::id, double>
        session_overhead_;  // Records session overhead of initialization+deinitialization
                            // time. Format is (thread id, duration)
};
