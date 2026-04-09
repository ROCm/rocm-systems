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

#include "vp9_defines.h"
#include "roc_video_parser.h"

#define INVALID_INDEX -1  // Invalid buffer index.

class Vp9VideoParser : public RocVideoParser {
public:
    /*! \brief Vp9VideoParser constructor
     */
    Vp9VideoParser();

    /*! \brief Vp9VideoParser destructor
     */
    virtual ~Vp9VideoParser();

    /*! \brief Function to Initialize the parser
     * \param [in] p_params Input of <tt>RocdecParserParams</tt> with codec type to initialize parser.
     * \return <tt>rocDecStatus</tt> Returns success on completion, else error code for failure
     */
    virtual rocDecStatus Initialize(RocdecParserParams *p_params);

    /*! \brief Function to Parse video data: Typically called from application when a demuxed picture is ready to be parsed
     * \param [in] p_data Pointer to picture data of type <tt>RocdecSourceDataPacket</tt>
     * @return <tt>rocDecStatus</tt> Returns success on completion, else error_code for failure
     */
    virtual rocDecStatus ParseVideoData(RocdecSourceDataPacket *p_data);

    /*! \brief function to uninitialize AV1 parser
     * @return rocDecStatus 
     */
    virtual rocDecStatus UnInitialize();     // derived method

protected:
    typedef struct {
        int      pic_idx;
        int      dec_buf_idx;  // frame index in decode/display buffer pool
        uint32_t use_status;    // refer to FrameBufUseStatus
    } Vp9Picture;

    /*! \brief Decoded picture buffer
     */
    typedef struct {
        Vp9Picture frame_store[VP9_BUFFER_POOL_MAX_SIZE]; // BufferPool
        int dec_ref_count[VP9_BUFFER_POOL_MAX_SIZE]; // frame ref count
        // A list of all frame buffers that may be used for reference of the current picture or any
        // subsequent pictures. The value is the index of a frame in DPB buffer pool. If an entry is
        // not used as reference, the value should be -1. Borrowed from AV1.
        int virtual_buffer_index[VP9_NUM_REF_FRAMES];
        uint32_t ref_frame_width[VP9_NUM_REF_FRAMES]; // RefFrameWidth
        uint32_t ref_frame_height[VP9_NUM_REF_FRAMES]; // RefFrameHeight
        uint32_t ref_subsampling_x[VP9_NUM_REF_FRAMES]; // RefSubsamplingX
        uint32_t ref_subsampling_y[VP9_NUM_REF_FRAMES]; // RefSubsamplingY
        uint32_t ref_bit_depth[VP9_NUM_REF_FRAMES]; // RefBitDepth
    } DecodedPictureBuffer;

    Vp9UncompressedHeader uncompressed_header_;
    uint32_t uncomp_header_size_;
    uint8_t last_frame_type_; // LastFrameType
    uint8_t frame_is_intra_;
    RocdecVp9SliceParams tile_params_;
    int16_t y_dequant_[VP9_MAX_SEGMENTS][2];
    int16_t uv_dequant_[VP9_MAX_SEGMENTS][2];
    uint8_t lvl_lookup_[VP9_MAX_SEGMENTS][VP9_MAX_REF_FRAMES][MAX_MODE_LF_DELTAS];
    
    int num_frames_in_chunck_; // can be more than 1 for superframes
    std::vector<uint32_t> frame_sizes_; // frame size of a single coded frame or sizes of multiple coded frames in a superframe
    DecodedPictureBuffer dpb_buffer_;
    Vp9Picture curr_pic_;

    // Current surface size, used to support size change (down) on inter frames, where we keep the
    // previously allocated surfaces and use them to store the smaller images.
    uint32_t curr_surface_width_;
    uint32_t curr_surface_height_;
    uint32_t reconfig_option_;

    /*! \brief Function to parse one picture bit stream received from the demuxer.
     * \param [in] p_stream A pointer of <tt>uint8_t</tt> for the input stream to be parsed
     * \param [in] pic_data_size Size of the input stream
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParsePictureData(const uint8_t *p_stream, uint32_t pic_data_size);

    /*! \brief Function to detect a superframe and parse the frame sizes. Annex B.
     *  \param [in] p_stream Pointer to the frame data chunk
     *  \param [in] chunk_data_size Size of the frame data chunk
     *  \return None
     */
    void CheckSuperframe(const uint8_t *p_stream, uint32_t chunk_data_size);

    /*! \brief Function to notify decoder about new sequence format through callback
     * \return <tt>ParserResult</tt>
     */
    ParserResult NotifyNewSequence(Vp9UncompressedHeader *p_uncomp_header);

    /*! \brief Function to fill the decode parameters and call back decoder to decode a picture
     * \return <tt>ParserResult</tt>
     */
    ParserResult SendPicForDecode();

    /*! \brief Function to do reference frame update process. 8.10.
     *  \return None
     */
    void UpdateRefFrames();

    /*! Function to initialize the local DPB (BufferPool)
     *  \return None
     */
    void InitDpb();

    /*! \brief Function to send out the remaining pictures that need for output in decode frame buffer.
     * \return <tt>ParserResult</tt>
     */
    ParserResult FlushDpb();

    /*! \brief Function to find a free buffer in the decode buffer pool
     *  \return <tt>ParserResult</tt>
     */
    ParserResult FindFreeInDecBufPool();

    /*! \brief Function to find a free buffer in DPB for the current picture and mark it.
     * \return <tt>ParserResult</tt>
     */
    ParserResult FindFreeInDpbAndMark();

    /*! \brief Function to check the frame stores that are done decoding and update status in DPB and decode/disp pool.
     *  \return None.
     */
    void CheckAndUpdateDecStatus();

    /*! \brief Function to parse an uncompressed header (uncompressed_header(), 6.2)
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] size Byte size of the stream
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParseUncompressedHeader(uint8_t *p_stream, size_t size);

    /*! \brief Function to parse frame sync syntax (frame_sync_code(), 6.2.1)
     * \param [in] p_stream Pointer to the bit stream
     * \param [inout] offset Bit offset
     * \param [out] p_uncomp_header Pointer to uncompressed header struct
     * \return <tt>ParserResult</tt>
     */
    ParserResult FrameSyncCode(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header);

    /*! \brief Function to parse color config syntax (color_config(), 6.2.2)
     * \param [in] p_stream Pointer to the bit stream
     * \param [inout] offset Bit offset
     * \param [out] p_uncomp_header Pointer to uncompressed header struct
     * \return <tt>ParserResult</tt>
     */
    ParserResult ColorConfig(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header);

    /*! \brief Function to parse frame size syntax (frame_size(), 6.2.3)
     * \param [in] p_stream Pointer to the bit stream
     * \param [inout] offset Bit offset
     * \param [out] p_uncomp_header Pointer to uncompressed header struct
     *  \return None
     */
    void FrameSize(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header);

    /*! \brief Function to parse render size syntax (frame_size(), 6.2.4)
     * \param [in] p_stream Pointer to the bit stream
     * \param [inout] offset Bit offset
     * \param [out] p_uncomp_header Pointer to uncompressed header struct
     *  \return None
     */
    void RenderSize(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header);

    /*! \brief Function to parse frame size with refs syntax (frame_size_with_refs(), 6.2.5)
     * \param [in] p_stream Pointer to the bit stream
     * \param [inout] offset Bit offset
     * \param [out] p_uncomp_header Pointer to uncompressed header struct
     *  \return None
     */
    void FrameSizeWithRefs(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header);

    /*! \brief Function to calculate 8x8 and 64x64 block columns and rows of the frame (compute_image_size(), 6.2.6)
     * \param [inout] p_uncomp_header Pointer to uncompressed header struct
     * \return None
     */
    void ComputeImageSize(Vp9UncompressedHeader *p_uncomp_header);

    /*! \brief Function to indicate that this frame can be decoded without dependence on previous coded frames. setup_past_independence() in spec.
     * \param [out] p_frame_header Pointer to frame header struct
     */
    void SetupPastIndependence(Vp9UncompressedHeader *p_uncomp_header);

    /*! \brief Function to parse loop filter params syntax (loop_filter_params(), 6.2.8)
     * \param [in] p_stream Pointer to the bit stream
     * \param [inout] offset Bit offset
     * \param [out] p_uncomp_header Pointer to uncompressed header struct
     *  \return None
     */
    void LoopFilterParams(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header);

    /*! \brief Function to parse quantization params syntax (quantization_params(), 6.2.9)
     * \param [in] p_stream Pointer to the bit stream
     * \param [inout] offset Bit offset
     * \param [out] p_uncomp_header Pointer to uncompressed header struct
     *  \return None
     */
    void QuantizationParams(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header);

    /*! \brief Function to parse delta quantizer syntax (read_delta_q(), 6.2.10)
     * \param [in] p_stream Pointer to the bit stream
     * \param [inout] offset Bit offset
     * \return <tt>int8_t</tt>
     */
    int8_t ReadDeltaQ(const uint8_t *p_stream, size_t &offset);

    /*! \brief Function to parse segmentation params syntax (segmentation_params(), 6.2.11)
     * \param [in] p_stream Pointer to the bit stream
     * \param [inout] offset Bit offset
     * \param [out] p_uncomp_header Pointer to uncompressed header struct
     * \return <tt>ParserResult</tt>
     */
    ParserResult SegmentationParams(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header);

    /*! \brief Function to parse probability syntax (read_delta_q(), 6.2.12)
     * \param [in] p_stream Pointer to the bit stream
     * \param [inout] offset Bit offset
     * \return <tt>uint8_t</tt>
     */
    uint8_t ReadProb(const uint8_t *p_stream, size_t &offset);

    /*! \brief Function to parse tile info syntax (tile_info(), 6.2.13)
     * \param [in] p_stream Pointer to the bit stream
     * \param [inout] offset Bit offset
     * \param [out] p_uncomp_header Pointer to uncompressed header struct
     * \return <tt>ParserResult</tt>
     */
    ParserResult TileInfo(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header);

    /*! \brief Function to set up Y and UV dequantization values based on segmentation parameters. 8.6.1.
     *  \param [in] p_uncomp_header Pointer to uncompressed header struct
     *  \return None
     */
    void SetupSegDequant(Vp9UncompressedHeader *p_uncomp_header);

    /*! \brief Function to return the quantizer index for the current block.
     *  \param [in] p_uncomp_header Pointer to uncompressed header struct
     *  \param [in] seg_id Segment id
     *  \return quantizer index
     */
    int GetQIndex(Vp9UncompressedHeader *p_uncomp_header, int seg_id);

    /*! \brief Function to get DC quantization parameter
     *  \param [in] bit_depth Bit depth
     *  \param [in] seg_id index Quantization parameter index
     *  \return DC quantization parameter
     */
    int DcQ(int bit_depth, int index);

    /*! \brief Function to get AC quantization parameter
     *  \param [in] bit_depth Bit depth
     *  \param [in] seg_id index Quantization parameter index
     *  \return AC quantization parameter
     */
    int AcQ(int bit_depth, int index);

    /*! \brief Function to prepare a filter strength lookup table once per frame. 8.8.1.
     *  \param [in] p_uncomp_header Pointer to uncompressed header struct
     *  \return None
     */
    void LoopFilterFrameInit(Vp9UncompressedHeader *p_uncomp_header);

    /*! \brief Function to read igned integer using n bits for the value and 1 bit for a sign flag 4.10.6. su(n).
     * \param [in] p_stream Bit stream pointer
     * \param [in] bit_offset Starting bit offset
     * \param [out] bit_offset Updated bit offset
     * \param [in] num_bits Number of bits to read
     * \return The signed value
     */
    inline int32_t ReadSigned(const uint8_t *p_stream, size_t &bit_offset, int num_bits) {
        uint32_t u_value = Parser::ReadBits(p_stream, bit_offset, num_bits);
        uint8_t sign = Parser::GetBit(p_stream, bit_offset);
        return sign ? -u_value : u_value;
    }

#if DBGINFO
    /*! \brief Function to log VAAPI parameters
     */
    void PrintVaapiParams();
    /*! \brief Function to log DPB and decode/display buffer pool info
     */
    void PrintDpb();
#endif // DBGINFO
};