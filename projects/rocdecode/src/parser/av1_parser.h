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

#include "av1_defines.h"
#include "roc_video_parser.h"

#define OBU_HEADER_SIZE 1
#define OBU_EXTENSION_SIZE 1
#define INVALID_INDEX -1  // Invalid buffer index.

class Av1VideoParser : public RocVideoParser {
public:
    /*! \brief Av1VideoParser constructor
     */
    Av1VideoParser();

    /*! \brief Av1VideoParser destructor
     */
    virtual ~Av1VideoParser();

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

    typedef struct {
        uint32_t tile_offset;
        uint32_t tile_size;
        uint32_t tile_col;
        uint32_t tile_row;
    } Av1TileDataInfo;

    typedef struct {
        uint8_t *buffer_ptr;  // base pointer to all tile data.
        uint32_t buffer_size;  // total size of all tile data, can come from multiple tile group OBUs.
        uint32_t num_tiles; // total number of tiles in the picture
        uint32_t tile_group_num; // current tile group number, counting from 0.
        uint32_t tg_start; // tg_start of the current tile group
        uint32_t tg_end; // tg_end of the current tile group
        uint32_t num_tiles_parsed; // number of parsed tiles for the current frame
        Av1TileDataInfo tile_data_info[MAX_TILE_ROWS * MAX_TILE_COLS];
    } Av1TileGroupDataInfo;

    typedef struct {
        int      pic_idx;
        int      dec_buf_idx;  // frame index in decode/display buffer pool
        int      fg_buf_idx;  // frame index for film grain synthesis output in decode/display buffer pool
        uint32_t current_frame_id;
        uint32_t order_hint;
        uint32_t frame_type;
        uint32_t use_status;    // refer to FrameBufUseStatus
        uint32_t show_frame;
    } Av1Picture;

    /*! \brief Decoded picture buffer
     */
    typedef struct {
        Av1Picture frame_store[BUFFER_POOL_MAX_SIZE]; // BufferPool
        int dec_ref_count[BUFFER_POOL_MAX_SIZE]; // DecoderRefCount
        // A list of all frame buffers that may be used for reference of the current picture or any
        // subsequent pictures. The value is the index of a frame in DPB buffer pool. If an entry is
        // not used as reference, the value should be -1.
        int virtual_buffer_index[NUM_REF_FRAMES]; // VBI
        int ref_valid[NUM_REF_FRAMES]; // RefValid
        int ref_frame_type[NUM_REF_FRAMES]; // RefFrameType
        int ref_frame_id[NUM_REF_FRAMES]; // RefFrameId
        uint32_t ref_upscaled_width[NUM_REF_FRAMES]; // RefUpscaledWidth
        uint32_t ref_frame_width[NUM_REF_FRAMES]; // RefFrameWidth
        uint32_t ref_frame_height[NUM_REF_FRAMES]; // RefFrameHeight
        uint32_t ref_render_width[NUM_REF_FRAMES]; // RefRenderWidth
        uint32_t ref_render_height[NUM_REF_FRAMES]; // RefRenderHeight
        int ref_order_hint[NUM_REF_FRAMES]; // RefOrderHint
        uint32_t saved_order_hints[NUM_REF_FRAMES][NUM_REF_FRAMES]; // SavedOrderHints
        int32_t saved_gm_params[NUM_REF_FRAMES][NUM_REF_FRAMES][6]; // SavedGmParams
        int32_t saved_loop_filter_ref_deltas[NUM_REF_FRAMES][TOTAL_REFS_PER_FRAME];
        int32_t saved_loop_filter_mode_deltas[NUM_REF_FRAMES][2];
        uint8_t saved_feature_enabled[NUM_REF_FRAMES][MAX_SEGMENTS][SEG_LVL_MAX];
        int16_t saved_feature_data[NUM_REF_FRAMES][MAX_SEGMENTS][SEG_LVL_MAX];
        Av1FilmGrainParams saved_film_grain_params[NUM_REF_FRAMES];
    } DecodedPictureBuffer;

protected:
    Av1ObuHeader obu_header_;
    uint64_t obu_size_; // current OBU size in byte, not including header and size bytes
    uint32_t obu_byte_offset_; // current OBU byte offset, not including header and obu_size syntax elements

    uint32_t seen_frame_header_; // SeenFrameHeader
    Av1SequenceHeader seq_header_;
    Av1FrameHeader frame_header_;
    Av1TileGroupDataInfo tile_group_data_;
    std::vector<RocdecAv1SliceParams> tile_param_list_;

    int temporal_id_; //  temporal level of the data contained in the OBU
    int spatial_id_;  // spatial level of the data contained in the OBU

    DecodedPictureBuffer dpb_buffer_;
    Av1Picture curr_pic_;

    int32_t prev_gm_params_[NUM_REF_FRAMES][6];

    /*! \brief Function to parse one picture bit stream received from the demuxer.
     * \param [in] p_stream A pointer of <tt>uint8_t</tt> for the input stream to be parsed
     * \param [in] pic_data_size Size of the input stream
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParsePictureData(const uint8_t *p_stream, uint32_t pic_data_size);

    /*! \brief Function to notify decoder about new sequence format through callback
     * \param [in] p_seq_header Pointer to the current sequence header
     * \param [in] p_frame_header Ponter to the current frame header
     * \return <tt>ParserResult</tt>
     */
    ParserResult NotifyNewSequence(Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header);

    /*! \brief Function to fill the decode parameters and call back decoder to decode a picture
     * \return <tt>ParserResult</tt>
     */
    ParserResult SendPicForDecode();

    /*! Function to initialize the local DPB (BufferPool)
     *  \return None
     */
     void InitDpb();

    /*! \brief Function to send out the remaining pictures that need for output in decode frame buffer.
     * \return <tt>ParserResult</tt>
     */
    ParserResult FlushDpb();

    /*! \brief Function to do reference frame update process. 7.20.
     *  \return None
     */
    void UpdateRefFrames();

    /*! \brief Function to load saved values for a previous reference frame back into the current frame variables. 7.21.
     *  \return None
     */
    void LoadRefFrame();

    /*! \brief Function to wrap up frame decode process. 7.4.
     * \return <tt>ParserResult</tt>
     */
    ParserResult DecodeFrameWrapup();

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

    /*! \brief Function to parse an OBU header
     * \param [in] p_stream Pointer to the bit stream
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParseObuHeader(const uint8_t *p_stream);

    /*! \brief Function to parse an OBU header and size
     * \return <tt>ParserResult</tt>
     */
    ParserResult ReadObuHeaderAndSize();

    /*! \brief Function to parse a sequence header OBU. 5.5.
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] size Byte size of the stream
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParseSequenceHeaderObu(uint8_t *p_stream, size_t size);

    /*! \brief Function to parse a frame header OBU. 5.9.
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] size Byte size of the stream
     * \param [out] p_bytes_parsed Number of bytes that have been parsed
     * \return None
     */
    ParserResult ParseFrameHeaderObu(uint8_t *p_stream, size_t size, int *p_bytes_parsed);

    /*! \brief Function to parse a frame header OBU
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] size Byte size of the stream
     * \param [out] p_bytes_parsed Number of bytes that have been parsed
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParseUncompressedHeader(uint8_t *p_stream, size_t size, int *p_bytes_parsed);

    /*! \brief Function to parse a tile group OBU
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] size Byte size of the stream
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParseTileGroupObu(uint8_t *p_stream, size_t size);

    /*! \brief Function to parse color config in sequence header
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [out] p_seq_header Pointer to sequence header struct
     * \return None
     */
    void ParseColorConfig(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header);

    /*! \brief Function to mark reference frames
     * \param [in] p_seq_header Pointer to sequence header
     * \param [in] p_frame_header Ponter to frame header
     * \param [in] id_len Current frame id length
     */
    void MarkRefFrames(Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header, uint32_t id_len);

    /*! \brief Function to parse frame size
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [in] p_seq_header Pointer to sequence header struct
     * \param [out] p_frame_header Pointer to frame header struct
     * \return <tt>ParserResult</tt>
     */
    ParserResult FrameSize(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header);

    /*! \brief Function to parse super res parameters
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [in] p_seq_header Pointer to sequence header struct
     * \param [out] p_frame_header Pointer to frame header struct
     * \return None
     */
    void SuperResParams(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header);

    /*! \brief Function to calculate 4x4 block columns and rows of the frame
     * \param [in] p_frame_header Pointer to frame header struct
     * \param [out] p_frame_header Pointer to frame header struct
     * \return None
     */
    void ComputeImageSize(Av1FrameHeader *p_frame_header);

    /*! \brief Function to parse render size info
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [out] p_frame_header Pointer to frame header struct
     * \return None
     */
    void RenderSize(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header);

    /*! \brief Function to compute the distance between two order hints by sign extending the result of subtracting the values.
     * \param [in] p_seq_header Pointer to sequence header struct
     * \param [in] a One order hint
     * \param [in] b The other order hint
     * \return The order hint distance
     */
    int GetRelativeDist(Av1SequenceHeader *p_seq_header, int a, int b);

    /*! \brief Function to compute the elements in the ref_frame_idx array. 7.8. Set frame refs process.
     * \param [in] p_seq_header Pinter to sequence header struct
     * \param [out] p_frame_header Pointer to frame header struct
     * \return None
     */
    void SetFrameRefs(Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header);

    /*! \brief Function to find the latest backward reference
     * \param [in] shifted_order_hints An array containing the expected output order shifted such that the current
     *             frame has hint equal to curr_frame_hint
     * \param [in] used_frame An array marking which reference frames have been used
     * \param [in] curr_frame_hint A variable set equal to 1 << (OrderHintBits - 1)
     * \param [in] latest_order_hint Latest order hint
     */
    int FindLatestBackward(int *shifted_order_hints, int *used_frame, int curr_frame_hint, int &latest_order_hint);

    /*! \brief Function to find the earliest backward reference
     * \param [in] shifted_order_hints An array containing the expected output order shifted such that the current
     *             frame has hint equal to curr_frame_hint
     * \param [in] used_frame An array marking which reference frames have been used
     * \param [in] curr_frame_hint A variable set equal to 1 << (OrderHintBits - 1)
     * \param [in] earliest_order_hint Eearliest order hint
     */
    int FindEarliestBackward(int *shifted_order_hints, int *used_frame, int curr_frame_hint, int &earliest_order_hint);

    /*! \brief Function to find the latest forward reference
     * \param [in] shifted_order_hints An array containing the expected output order shifted such that the current
     *             frame has hint equal to curr_frame_hint
     * \param [in] used_frame An array marking which reference frames have been used
     * \param [in] curr_frame_hint A variable set equal to 1 << (OrderHintBits - 1)
     * \param [in] latest_order_hint Latest order hint
     */
    int FindLatestForward(int *shifted_order_hints, int *used_frame, int curr_frame_hint, int &latest_order_hint);

    /*! \brief Function to parse frame size with refs info
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [in] p_seq_header Pointer to sequence header struct
     * \param [out] p_frame_header Pointer to frame header struct
     * \return None
     */
    void FrameSizeWithRefs(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header);

    /*! \brief Function to indicate that this frame can be decoded without dependence on previous coded frames. setup_past_independence() in spec.
     * \param [out] p_frame_header Pointer to frame header struct
     */
    void SetupPastIndependence(Av1FrameHeader *p_frame_header);

    /*! \brief Function to indicate that information from a previous frame may be loaded for use in decoding the current frame. load_previous() in spec.
     * \param [out] p_frame_header Pointer to frame header struct
     */
    void LoadPrevious(Av1FrameHeader *p_frame_header);

    /*! \brief Function to parse tile info
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [in] p_seq_header Pointer to sequence header struct
     * \param [out] p_frame_header Pointer to frame header struct
     * \return <tt>ParserResult</tt>
     */
    ParserResult TileInfo(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header);

    /*! \brief Function to calculate the smallest value for k such that blk_size << k is greater than or equal to target.
     * \param [in] blk_size Block size
     * \param [in] target target value
     * \return 32-bit unsigned
     */
    uint32_t TileLog2(uint32_t blk_size, uint32_t target);

    /*! \brief Function to parse quantization parameters
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [in] p_seq_header Pointer to sequence header struct
     * \param [out] p_frame_header Pointer to frame header struct
     * \return None
     */
    void QuantizationParams(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header);

    /*! \brief Function to read delta quantizer
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [out] p_frame_header Pointer to frame header struct
     * \return Delta quantizer value
     */
    int32_t ReadDeltaQ(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header);

    /*! \brief Function to segmentation parameters
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [out] p_frame_header Pointer to frame header struct
     * \return None
     */
    void SegmentationParams(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header);

    /*! \brief Function to parse quantizer index delta parameters
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [out] p_frame_header Pointer to frame header struct
     * \return None
     */
    void DeltaQParams(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header);

    /*! \brief Function to parse loop filter delta parameters
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [out] p_frame_header Pointer to frame header struct
     * \return None
     */
    void DeltaLFParams(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header);

    /*! \brief Function to return the quantizer index for the current block
     *  \param [in] p_frame_header Pointer to frame header struct
     *  \param [in] ignore_delta_q Indicator to ignore the Q index delta
     *  \param [in] segment_id Segment id
     *  \return Quantizer index
     */
    int GetQIndex(Av1FrameHeader *p_frame_header, int ignore_delta_q, int segment_id);

    /*! \brief Function to parse loop filter parameters
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [in] p_seq_header Pointer to sequence header struct
     * \param [out] p_frame_header Pointer to frame header struct
     * \return None
     */
    void LoopFilterParams(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header);

    /*! \brief Function to parse CDEF parameters
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [in] p_seq_header Pointer to sequence header struct
     * \param [out] p_frame_header Pointer to frame header struct
     * \return None
     */
    void CdefParams(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header);

    /*! \brief Function to loop restoration parameters
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [in] p_seq_header Pointer to sequence header struct
     * \param [out] p_frame_header Pointer to frame header struct
     * \return None
     */
    void LrParams(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header);

    /*! \brief Function to parse TX mode
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [out] p_frame_header Pointer to frame header struct
     * \return None
     */
    void ReadTxMode(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header);

    /*! \brief Function to skip mode parameters
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [in] p_seq_header Pointer to sequence header struct
     * \param [out] p_frame_header Pointer to frame header struct
     * \return None
     */
    void SkipModeParams(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header);

    /*! \brief Function to parse global motion parameters
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [out] p_frame_header Pointer to frame header struct
     * \return None
     */
    void GlobalMotionParams(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header);

    /*! \brief Function to calculate global motion parameters
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [out] p_frame_header Pointer to frame header struct
     * \param [in] type Motion type
     * \param [in] ref Reference frame
     * \param [in] idx Parameter index
     * \return None
     */
    void ReadGlobalParam(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header, int type, int ref, int idx);

    /*! \brief Function to decode signed subexp with ref. 5.9.26. decode_signed_subexp_with_ref()
     */
    int DecodeSignedSubexpWithRef(const uint8_t *p_stream, size_t &offset, int low, int high, int r);

    /*! \brief Function to decode unsigned subexp with ref. 5.9.27. decode_unsigned_subexp_with_ref()
     */
    int DecodeUnsignedSubexpWithRef(const uint8_t *p_stream, size_t &offset, int mx, int r);

    /*! \brief Function to decode subexp. 5.9.28. decode_subexp()
     */
    int DecodeSubexp(const uint8_t *p_stream, size_t &offset, int num_syms);

    /*! \brief Function to inverse recenter. 5.9.29. inverse_recenter()
     */
    int InverseRecenter(int r, int v);

    /*! \brief Function to check the validness of an array of warp parameters. 7.11.3.6.
     *  \param [in] p_warp_params Pointer to the warp parameters
     *  \return 1: valid; 0: invalid
     */
    int ShearParamsValidation(int32_t *p_warp_params);

    /*! \brief Function to calculate variables div_shift and div_shift that can be used to perform an approximate division by d
     *         via multiplying by div_factor and shifting right by div_shift.
     *  \param [in] d Input variable d
     *  \param [out] div_shift The output shift value
     *  \param [out] div_factor The ouput factor value
     *  \return None
     */
    void ResolveDivisor(int d, int *div_shift, int *div_factor);

    /*! \brief Function to parse film grain parameters
     * \param [in] p_stream Pointer to the bit stream
     * \param [in] offset Starting bit offset
     * \param [out] offset Updated bit offset
     * \param [in] p_seq_header Pointer to sequence header struct
     * \param [out] p_frame_header Pointer to frame header struct
     * \return <tt>ParserResult</tt>
     */
    ParserResult FilmGrainParams(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header);

    /*! \brief Function to round a number to 2^n
     *  \param [in] x The number to be rounded
     *  \param [in] n The exponent
     *  \return The rounded value
     */
    inline uint64_t Round2(uint64_t x, int n) {
        if (n == 0) {
            return x;
        }
        return ((x + ((uint64_t)1 << (n - 1))) >> n);
    }

    /*! \brief Function to round a signed number to 2^n
     *  \param [in] x The number to be rounded
     *  \param [in] n The exponent
     *  \return The rounded value
     */
    inline int64_t Round2Signed(int64_t x, int n) {
        return ((x < 0) ? -((int64_t)Round2(-x, n)) : (int64_t)Round2(x, n));
    }

    /*! \brief Function to calculate the floor of the base 2 logarithm of the input x
     * \param [in] x A 32-bit unsigned integer
     * \return the location of the most significant bit in x
     */
    inline uint32_t FloorLog2(uint32_t x) {
        uint32_t saved_x = x;
        uint32_t s = 0;
        while (x != 0) {
            x = x >> 1;
            s++;
        }
        return saved_x ? s - 1 : 0;
    }

    /*! \brief Function to read variable length unsigned n-bit number appearing directly in the bitstream. 4.10.3. uvlc().
     * \param [in] p_stream Bit stream pointer
     * \param [in] bit_offset Starting bit offset
     * \param [out] bit_offset Updated bit offset
     * \return The unsigned value
     */
    inline uint32_t ReadUVLC(const uint8_t *p_stream, size_t &bit_offset) {
        int leading_zeros = 0;
        while (!Parser::GetBit(p_stream, bit_offset)) {
            ++leading_zeros;
        }
        // Maximum 32 bits.
        if (leading_zeros >= 32) {
            return 0xFFFFFFFF;
        }
        uint32_t base = (1u << leading_zeros) - 1;
        uint32_t value = Parser::ReadBits(p_stream, bit_offset, leading_zeros);
        return base + value;
    }

    /*! \brief Function to read unsigned little-endian num_bytes-byte number appearing directly in the bitstream. 4.10.4. le(n).
     * \param [in] p_stream Bit stream pointer
     * \param [in] num_bytes Number of bytes to read
     * \return The unsigned value
     */
    inline uint32_t ReadLeBytes(const uint8_t *p_stream, int num_bytes) {
        uint32_t t = 0;
        for (int i = 0; i < num_bytes; i++) {
            t += (p_stream[i] << ( i * 8 ) );
        }
        return t;
    }

    /*! \brief Function to read unsigned integer represented by a variable number of little-endian bytes, which
     *         is less than or equal to (1 << 32) - 1. 4.10.5. leb128().
     * \param [in] p_stream Bit stream pointer
     * \param [out] p_num_bytes_read Number of bytes read
     * \return The unsigned value
     */
    inline uint64_t ReadLeb128(const uint8_t *p_stream, uint32_t *p_num_bytes_read) {
        uint32_t value = 0;
        *p_num_bytes_read = 0;
        uint32_t len;
        for (len = 0; len < 8; ++len) {
            value |= (p_stream[len] & 0x7F) << (len * 7);
            if ((p_stream[len] & 0x80) == 0) {
                ++len;
                *p_num_bytes_read = len;
                break;
            }
        }
        return value;
    }

    /*! \brief Function to read signed integer converted from an n bits unsigned integer in the bitstream. 4.10.6. su(n).
     * \param [in] p_stream Bit stream pointer
     * \param [in] bit_offset Starting bit offset
     * \param [out] bit_offset Updated bit offset
     * \param [in] num_bits Number of bits to read
     * \return The signed value
     */
    inline int32_t ReadSigned(const uint8_t *p_stream, size_t &bit_offset, int num_bits) {
        int32_t value;
        uint32_t u_value = Parser::ReadBits(p_stream, bit_offset, num_bits);
        uint32_t sign_mask = 1 << (num_bits - 1);
        if ( u_value & sign_mask ) {
            value = u_value - 2 * sign_mask;
        } else {
            value = u_value;
        }
        return value;
    }

    /*! \brief Function to read unsigned encoded (non-symmetric) integer with maximum number of values num_bits 
     *         (i.e. output in range 0..num_bits-1). This encoding is non-symmetric because the values are not all 
     *         coded with the same number of bits. 4.10.7. ns(n).
     * \param [in] p_stream Bit stream pointer
     * \param [in] bit_offset Starting bit offset
     * \param [out] bit_offset Updated bit offset
     * \param [in] num_bits Number of bits to read
     * \return The unsigned value
     */
    inline uint32_t ReadUnsignedNonSymmetic(const uint8_t *p_stream, size_t &bit_offset, int num_bits) {
        uint32_t w = FloorLog2(num_bits) + 1;
        uint32_t m = (1 << w) - num_bits;
        uint32_t v = Parser::ReadBits(p_stream, bit_offset, w - 1);
        if (v < m) {
            return v;
        }
        uint32_t extra_bit = Parser::GetBit(p_stream, bit_offset);
        return (v << 1) - m + extra_bit;
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