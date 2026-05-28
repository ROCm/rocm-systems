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

#include "../commons.h"
#include "roc_video_parser.h"
#include "hevc_defines.h"

#include <map>
#include <algorithm>

class HevcVideoParser : public RocVideoParser {

public:
    /*! \brief Construct a new HEVCParser object
     */
    HevcVideoParser();

    /**
     * @brief HEVCParser object destructor
     */
    virtual ~HevcVideoParser();

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

    /**
     * @brief function to uninitialize hevc parser
     * 
     * @return rocDecStatus 
     */
    virtual rocDecStatus UnInitialize();     // derived method :: nothing to do for this

protected:
    /*! \brief Inline function to Parse the NAL Unit Header
     * 
     * \param [in] nal_unit A pointer of <tt>uint8_t</tt> containing the Demuxed output stream 
     * \return Returns an object of NALUnitHeader
     */
    static inline HevcNalUnitHeader ParseNalUnitHeader(uint8_t *nal_unit) {
        HevcNalUnitHeader nalu_header;
        nalu_header.num_emu_byte_removed = 0;
        //read nalu header
        nalu_header.forbidden_zero_bit = (uint32_t) ((nal_unit[0] >> 7) & 1);
        nalu_header.nal_unit_type = (uint32_t) ((nal_unit[0] >> 1) & 63);
        nalu_header.nuh_layer_id = (uint32_t) (((nal_unit[0] & 1) << 6) | ((nal_unit[1] & 248) >> 3));
        nalu_header.nuh_temporal_id_plus1 = (uint32_t) (nal_unit[1] & 7);

        return nalu_header;
    }

    /*! \brief Slice info of a picture
     */
    typedef struct {
        HevcSliceSegHeader slice_header;
        uint32_t slice_data_offset; // offset in the slice data buffer of this slice
        uint32_t slice_data_size; // slice data size in bytes
        uint8_t ref_pic_list_0_[HEVC_MAX_NUM_REF_PICS];  // RefPicList0
        uint8_t ref_pic_list_1_[HEVC_MAX_NUM_REF_PICS];  // RefPicList1
    } HevcSliceInfo;

    /*! \brief Picture info for decoding process
     */
    typedef struct {
        int     pic_idx;  // picture index or id
        int     dec_buf_idx;  // frame index in decode buffer pool
        // POC info
        int32_t pic_order_cnt;  // PicOrderCnt
        int32_t prev_poc_lsb;  // prevPicOrderCntLsb
        int32_t prev_poc_msb;  // prevPicOrderCntMsb
        uint32_t slice_pic_order_cnt_lsb; // for long term ref pic identification
        uint32_t decode_order_count;  // to record relative time in DPB

        uint32_t pic_output_flag;  // PicOutputFlag
        uint32_t is_reference;
        uint32_t use_status;    // refer to FrameBufUseStatus
    } HevcPicInfo;

    /*! \brief Decoded picture buffer
     */
    typedef struct
    {
        uint32_t dpb_size;  // DPB buffer size in number of frames
        uint32_t num_pics_needed_for_output;  // number of pictures in DPB that need to be output
        uint32_t dpb_fullness;  // number of pictures in DPB
        HevcPicInfo frame_buffer_list[HEVC_MAX_DPB_FRAMES];
    } DecodedPictureBuffer;

    // Data members of HEVC class
    HevcNalUnitHeader   nal_unit_header_;
    int32_t             active_vps_id_;
    int32_t             active_sps_id_;
    int32_t             active_pps_id_;
    HevcVideoParamSet   vps_list_[MAX_VPS_COUNT];
    HevcSeqParamSet     sps_list_[MAX_SPS_COUNT];
    HevcPicParamSet     pps_list_[MAX_PPS_COUNT];
    HevcSliceSegHeader  slice_header_copy_;
    std::vector<HevcSliceInfo> slice_info_list_;
    std::vector<RocdecHevcSliceParams> slice_param_list_;

    HevcNalUnitHeader   slice_nal_unit_header_;
    HevcPicInfo         curr_pic_info_;

    int first_pic_after_eos_nal_unit_; // to flag the first picture after EOS
    int no_rasl_output_flag_; // NoRaslOutputFlag

    uint32_t max_tb_log2_size_y_; // MaxTbLog2SizeY
    int ctb_log2_size_y_; // CtbLog2SizeY
    int pic_width_in_ctbs_y_;  // PicWidthInCtbsY
    int pic_height_in_ctbs_y_;  // PicHeightInCtbsY
    int pic_size_in_ctbs_y_;  // PicSizeInCtbsY
    int sub_width_c_; // SubWidthC
    int sub_height_c_; // SubHeightC

    // DPB
    DecodedPictureBuffer dpb_buffer_;
    int no_output_of_prior_pics_flag;  // NoOutputOfPriorPicsFlag

    uint32_t num_pic_total_curr_;  // NumPicTotalCurr

    // Reference picture set
    uint32_t num_poc_st_curr_before_;  // NumPocStCurrBefore;
    uint32_t num_poc_st_curr_after_;  // NumPocStCurrAfter;
    uint32_t num_poc_st_foll_;  // NumPocStFoll;
    uint32_t num_poc_lt_curr_;  // NumPocLtCurr;
    uint32_t num_poc_lt_foll_;  // NumPocLtFoll;

    int32_t poc_st_curr_before_[HEVC_MAX_NUM_REF_PICS];  // PocStCurrBefore
    int32_t poc_st_curr_after_[HEVC_MAX_NUM_REF_PICS];  // PocStCurrAfter
    int32_t poc_st_foll_[HEVC_MAX_NUM_REF_PICS];  // PocStFoll
    int32_t poc_lt_curr_[HEVC_MAX_NUM_REF_PICS];  // PocLtCurr
    int32_t poc_lt_foll_[HEVC_MAX_NUM_REF_PICS];  // PocLtFoll
    uint8_t ref_pic_set_st_curr_before_[HEVC_MAX_NUM_REF_PICS];  // RefPicSetStCurrBefore
    uint8_t ref_pic_set_st_curr_after_[HEVC_MAX_NUM_REF_PICS];  // RefPicSetStCurrAfter
    uint8_t ref_pic_set_st_foll_[HEVC_MAX_NUM_REF_PICS];  // RefPicSetStFoll
    uint8_t ref_pic_set_lt_curr_[HEVC_MAX_NUM_REF_PICS];  // RefPicSetLtCurr
    uint8_t ref_pic_set_lt_foll_[HEVC_MAX_NUM_REF_PICS];  // RefPicSetLtFoll

    /*! \brief Function to parse Video Parameter Set 
     * \param [in] nalu A pointer of <tt>uint8_t</tt> for the input stream to be parsed
     * \param [in] size Size of the input stream
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParseVps(uint8_t *nalu, size_t size);

    /*! \brief Function to parse Sequence Parameter Set 
     * \param [in] nalu A pointer of <tt>uint8_t</tt> for the input stream to be parsed
     * \param [in] size Size of the input stream
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParseSps(uint8_t *nalu, size_t size);

    /*! \brief Function to parse Picture Parameter Set 
     * \param [in] nalu A pointer of <tt>uint8_t</tt> for the input stream to be parsed
     * \param [in] size Size of the input stream
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParsePps(uint8_t *nalu, size_t size);

    /*! \brief Function to parse Profiles, Tiers and Levels
     * \param [out] ptl A pointer of <tt>HevcProfileTierLevel</tt> for the output from teh parsed stream
     * \param [in] profile_present_flag Input of <tt>bool</tt> - 1 specifies profile information is present, else 0
     * \param [in] max_num_sub_layers_minus1 Input of <tt>uint32_t</tt> - plus 1 specifies the maximum number of temporal sub-layers that may be present
     * \param [in] nalu A pointer of <tt>uint8_t</tt> for the input stream to be parsed
     * \param [in] size Size of the input stream
     * \param [in] offset Reference to the offset in the input buffer
     * \return No return value
     */
    void ParsePtl(HevcProfileTierLevel *ptl, bool profile_present_flag, uint32_t max_num_sub_layers_minus1, uint8_t *nalu, size_t size, size_t &offset);
    
    /*! \brief Function to parse Sub Layer Hypothetical Reference Decoder Parameters
     * \param [out] sub_hrd A pointer of <tt>HevcSubLayerHrdParameters</tt> for the output from teh parsed stream
     * \param [in] cpb_cnt Input of <tt>uint32_t</tt> - specifies the coded picture buffer count in a HRD buffer
     * \param [in] sub_pic_hrd_params_present_flag Input of <tt>bool</tt> - 1 specifies sub layer HRD information is present, else 0
     * \param [in] nalu A pointer of <tt>uint8_t</tt> for the input stream to be parsed
     * \param [in] size Size of the input stream
     * \param [in] offset Reference to the offset in the input buffer
     * \return No return value
     */
    void ParseSubLayerHrdParameters(HevcSubLayerHrdParameters *sub_hrd, uint32_t cpb_cnt, bool sub_pic_hrd_params_present_flag, uint8_t *nalu, size_t size, size_t &offset);
    
    /*! \brief Function to parse Hypothetical Reference Decoder Parameters
     * \param [out] hrd A pointer of <tt>HevcHrdParameters</tt> for the output from the parsed stream
     * \param [in] common_inf_present_flag Input of <tt>bool</tt> - 1 specifies HRD information is present, else 0
     * \param [in] max_num_sub_layers_minus1 Input of <tt>uint32_t</tt> - plus 1 specifies the maximum number of temporal sub-layers that may be present
     * \param [in] nalu A pointer of <tt>uint8_t</tt> for the input stream to be parsed
     * \param [in] size Size of the input stream
     * \param [in] offset Reference to the offset in the input buffer
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParseHrdParameters(HevcHrdParameters *hrd, bool common_inf_present_flag, uint32_t max_num_sub_layers_minus1, uint8_t *nalu, size_t size, size_t &offset);
    
    /*! \brief Function to set the default values to the scaling list
     * \param [out] sl_ptr A pointer to the scaling list <tt>HevcScalingListData</tt>
     * \return No return value
     */
    void SetDefaultScalingList(HevcScalingListData *sl_ptr);

    /*! \brief Function to parse Scaling List
     * \param [out] sl_ptr A pointer of <tt>HevcScalingListData</tt> for the output from the parsed stream
     * \param [in] data A pointer of <tt>uint8_t</tt> for the input stream to be parsed
     * \param [in] size Size of the input stream
     * \param [in] offset Reference to the offset in the input buffer
     * \param [in] sps_ptr Pointer to the current SPS
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParseScalingList(HevcScalingListData * sl_ptr, uint8_t *data, size_t size, size_t &offset, HevcSeqParamSet *sps_ptr);
    
    /*! \brief Function to parse Video Usability Information
     * \param [in/out] sps_ptr The pointer to the associated SPS
     * \param [in] data A pointer of <tt>uint8_t</tt> for the input stream to be parsed
     * \param [in] size Size of the input stream
     * \param [in] offset Reference to the offset in the input buffer
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParseVui(HevcSeqParamSet *sps_ptr, uint8_t *data, size_t size, size_t &offset);
    
    /*! \brief Function to parse Short Term Reference Picture Set
     * \param [in] sps_ptr The pointer to the associated SPS
     * \param [out] rps A pointer of <tt>HevcShortTermRps</tt> for the output from the parsed stream
     * \param [in] st_rps_idx specifies the index in the RPS buffer
     * \param [in] num_short_term_ref_pic_sets Specifies the count of Short Term RPS in <tt>uint32_t</tt>
     * \param [in] rps_ref A reference of <tt>HevcShortTermRps</tt> to the RPS buffer
     * \param [in] data A pointer of <tt>uint8_t</tt> for the input stream to be parsed
     * \param [in] size Size of the input stream
     * \param [in] offset Reference to the offset in the input buffer
     * \return No return value
     */
    ParserResult ParseShortTermRefPicSet(HevcSeqParamSet *sps_ptr, HevcShortTermRps *rps, uint32_t st_rps_idx, uint32_t num_short_term_ref_pic_sets, HevcShortTermRps rps_ref[], uint8_t *data, size_t size, size_t &offset);
    
    /*! \brief Function to parse weighted prediction table
     * \param [in/out] Slice_header_ptr Pointer to the slice segment header
     * \param [in] chroma_array_type ChromaArrayType
     * \param [in] stream_ptr Bit stream pointer
     * \param [in/out] offset Bit offset of the current parsing action
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParsePredWeightTable(HevcSliceSegHeader *slice_header_ptr, int chroma_array_type, uint8_t *stream_ptr, size_t &offset);

    /*! \brief Function to parse Slice Header
     * \param [in] nalu A pointer of <tt>uint8_t</tt> for the input stream to be parsed
     * \param [in] size Size of the input stream
     * \param [out] p_slice_header Pointer to the slice header struct
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParseSliceHeader(uint8_t *nalu, size_t size, HevcSliceSegHeader *p_slice_header);

    /*! \brief Function to calculate the picture order count of the current picture. Once per picutre. (8.3.1)
     */
    void CalculateCurrPoc();

    /*! \brief Function to perform decoding process for reference picture set. Once per picture. (8.3.2)
     */
    void DecodeRps();

    /*! \brief Function to perform decoding process for reference picture lists construction per slice. (8.3.4)
     * \param [in] p_slice_info Pointer to the slice info struct
     */
    void ConstructRefPicLists(HevcSliceInfo *p_slice_info);

    /*! \brief Function to initialize DPB buffer.
     */
    void InitDpb();

    /*! \brief Function to clear DPB buffer.
     */
    void EmptyDpb();

    /*! \brief Function to send out the remaining pictures that need for output in DPB buffer.
     * \return Code in ParserResult form.
     */
    int FlushDpb();

    /*! \brief Function to output and remove pictures from DPB. C.5.2.2.
     * \return Code in ParserResult form.
     */
    int MarkOutputPictures();

    /*! \brief Function to find a free buffer in the decode buffer pool
     *  \return <tt>ParserResult</tt>
     */
    ParserResult FindFreeInDecBufPool();

    /*! \brief Function to find a free buffer in DPB for the current picture and mark it. Additional picture
     *         bumping is done if needed. C.5.2.3.
     * \return <tt>ParserResult</tt>
     */
    ParserResult FindFreeInDpbAndMark();

    /*! \brief Function to bump one picture out of DPB. C.5.2.4.
     * \return Code in ParserResult form.
     */
    int BumpPicFromDpb();

    /*! \brief Function to parse one picture bit stream received from the demuxer.
     * \param [in] p_stream A pointer of <tt>uint8_t</tt> for the input stream to be parsed
     * \param [in] pic_data_size Size of the input stream
     * \return <tt>ParserResult</tt>
     */
    ParserResult ParsePictureData(const uint8_t* p_stream, uint32_t pic_data_size);

#if DBGINFO
    void PrintVps(HevcVideoParamSet *vps_ptr);
    void PrintSps(HevcSeqParamSet *sps_ptr);
    void PrintPps(HevcPicParamSet *pps_ptr);
    void PrintSliceSegHeader(HevcSliceSegHeader *slice_header_ptr);
    void PrintStRps(HevcShortTermRps *rps_ptr);
    void PrintLtRefInfo(HevcLongTermRps *lt_info_ptr);
    void PrintDpb();
    void PrintVappiBufInfo();
#endif // DBGINFO

private:
    /*! \brief Callback function to notify decoder about new SPS.
     */
    int FillSeqCallbackFn(HevcSeqParamSet* sps_data);

    /*! \brief Callback function to send parsed SEI playload to decoder.
     */
    void SendSeiMsgPayload();

    /*! \brief Callback function to fill the decode parameters and call decoder to decode a picture
     * \return Return code in ParserResult form
     */
    int SendPicForDecode();

    bool IsIdrPic(HevcNalUnitHeader *nal_header_ptr);
    bool IsCraPic(HevcNalUnitHeader *nal_header_ptr);
    bool IsBlaPic(HevcNalUnitHeader *nal_header_ptr);
    bool IsIrapPic(HevcNalUnitHeader *nal_header_ptr);
    bool IsRaslPic(HevcNalUnitHeader *nal_header_ptr);
    bool IsRadlPic(HevcNalUnitHeader *nal_header_ptr);
    bool IsRefPic(HevcNalUnitHeader *nal_header_ptr);
};