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

#include <iostream>
#include <fstream>
#include <vector>
#include "rocdecode/rocdecode.h"
#include "roc_video_parser.h"

#define BS_RING_SIZE (16 * 1024 * 1024)
#define INIT_PIC_DATA_SIZE (2 * 1024 * 1024)

enum {
    kStreamTypeUnsupported = -1,
    kStreamTypeAvcElementary = 0,
    kStreamTypeHevcElementary,
    kStreamTypeAv1Elementary,
    kStreamTypeAv1Ivf,
    kStreamTypeVp9Ivf,
    kStreamTypeNumSupported
} StreamFileType;

#define STREAM_PROBE_SIZE 2 * 1024
#define STREAM_TYPE_SCORE_THRESHOLD 50
#define STREAM_TYPE_HIGH_CONFIDENCE_SCORE_THRESHOLD 90

class RocVideoESParser {
    public:
        RocVideoESParser(const char *input_file_path);
        RocVideoESParser();
        ~RocVideoESParser();

        /*! \brief Function to probe the bitstream file and try to get the codec id
         * \retrun Codec id
         */
        rocDecVideoCodec GetCodecId();

        /*! \brief Function to retrieve the bitstream of a picture
         * \param [out] p_pic_data Pointer to the picture data
         * \param [out] pic_size Size of the picture in bytes
         * \param [out] pts Presentation time stamp
         */
        int GetPicData(uint8_t **p_pic_data, int *pic_size, int64_t *pts);

        /*! \brief Function to return the bit depth of the stream
         */
        int GetBitDepth() {return bit_depth_;};

    private:
        std::ifstream p_stream_file_;
        int stream_type_;
        int bit_depth_;

        // Bitstream ring buffer
        uint8_t bs_ring_[BS_RING_SIZE];
        uint32_t read_ptr_; /// start position of unprocessed stream in the ring
        uint32_t write_ptr_;  /// end position of unprocessed stream in the ring
        bool end_of_file_;
        bool end_of_stream_;
        int curr_byte_offset_;
        // AVC/HEVC
        int num_start_code_;
        int curr_start_code_offset_;
        int next_start_code_offset_;
        //int nal_unit_size_;
        // AV1
        int obu_byte_offset_; // header offset
        int obu_size_; // including header
        int num_td_obus_; // number of temporal delimiter OBUs

        // Picture data (linear buffer)
        std::vector<uint8_t> pic_data_;
        int pic_data_size_;
        // AVC/HEVC
        int curr_pic_end_;
        int next_pic_start_;
        int num_pictures_;
        // AV1
        int num_temp_units_; // number of temporal units

        bool ivf_file_header_read_; // indicator if IVF file header has been checked

        RocDecLogger logger_;

        /*! \brief Function to retrieve the bitstream of a picture for AVC/HEVC
         * \param [out] p_pic_data Pointer to the picture data
         * \param [out] pic_size Size of the picture in bytes
         */
        int GetPicDataAvcHevc(uint8_t **p_pic_data, int *pic_size);

        /*! \brief Function to retrieve the bitstream of a temporal unit for AV1
         * \param [out] p_pic_data Pointer to the picture data
         * \param [out] pic_size Size of the picture in bytes
         */
        int GetPicDataAv1(uint8_t **p_pic_data, int *pic_size);

        /*! \brief Function to retrieve the bitstream of a frame from IVF container
         * \param [out] p_pic_data Pointer to the picture data
         * \param [out] pic_size Size of the picture in bytes
         */
        int GetPicDataIvf(uint8_t **p_pic_data, int *pic_size);

        /*! \brief Function to read bitstream from file and fill into the ring buffer.
        * \return Number of bytes read from file.
        */
        int FetchBitStream();

        /*! \brief Function to check the remaining data size in the ring buffer
         * \return Number of bytes still available in the ring
         */
        int GetDataSizeInRB();

        /*! \brief Function to read one byte from the ring buffer without advancing the read pointer
         * \param [in] offset The byte offset to read
         * \param [out] data The byte read
         * \return True: success; False: no more byte available.
         */
        bool GetByte(int offset, uint8_t *data);

        /*! \brief Function to read the specified bytes from the ring buffer without advancing the read pointer
         * \param [in] offset The starting byte offset to read
         * \param [in] size The numbers of bytes to read
         * \param [out] data The bytes read
         * \return True: success; False: can not read the set bytes
         */
        bool ReadBytes(int offset, int size, uint8_t *data);

        /*! \brief Function to update the read pointer by the set bytes
         * \param [in] value The new read pointer value
         */
        void SetReadPointer(int value);

        /*! \brief Function to find the start codes from the ring buffer to locate the NAL units
        * \return Returns: true: a new start code is found or end of stream reached; false: no start code found. 
        */
        bool FindStartCode();

        /*! \brief Function to check if an HEVC NAL is the (first) slice of a picture
         * \param [in] start_code_offset Start code location of the NAL unit
         * \param [out] slice_flag Slice NAL unit indicator
         * \param [out] first_slice_flag First slice indicator
         */
        void CheckHevcNalForSlice(int start_code_offset, int *slice_flag, int *first_slice_flag);

        /*! \brief Function to check if an AVC NAL is the (first) slice of a picture
         * \param [in] start_code_offset Start code location of the NAL unit
         * \param [out] slice_flag Slice NAL unit indicator
         * \param [out] first_slice_flag First slice indicator
         */
        void CheckAvcNalForSlice(int start_code_offset, int *slice_flag, int *first_slice_flag);

        /*! \brief Function to copy a NAL unit from the bitstream ring buffer to the linear picture data buffer
         */
        void CopyNalUnitFromRing();

        /*! \brief Function to parse an OBU header and size
        * \param [out] obu_type Pointer to the returned OBU type
        * \return true if success
        */
        bool ReadObuHeaderAndSize(int *obu_type);
    
        /*! \brief Function to copy an OBU from the bitstream ring buffer to the linear picture data buffer
         * \return true if success
         */
        bool CopyObuFromRing();

        /*! \brief Function to probe the bitstream file and try to find if it is one of types supported.
         * \return Elementary stream file type
         */
        int ProbeStreamType();

        /*! \brief Function to check the likelihood of a stream to be an AVC elementary stream.
         * \param [in] p_stream Pointer to the stream
         * \param [in] stream_size Size of the stream in bytes
         * \return The likelihood score
         */
        int CheckAvcEStream(uint8_t *p_stream, int stream_size);

        /*! \brief Function to check the likelihood of a stream to be an HEVC elementary stream.
         * \param [in] p_stream Pointer to the stream
         * \param [in] stream_size Size of the stream in bytes
         * \return The likelihood score
         */
        int CheckHevcEStream(uint8_t *p_stream, int stream_size);

        /*! \brief Function to convert from Encapsulated Byte Sequence Packets to Raw Byte Sequence Payload
        * \param [inout] stream_buffer A pointer of <tt>uint8_t</tt> for the converted RBSP buffer.
        * \param [in] begin_bytepos Start position in the EBSP buffer to convert
        * \param [in] end_bytepos End position in the EBSP buffer to convert, generally it's size.
        * \return Returns the size of the converted buffer
        */
        int EbspToRbsp(uint8_t *stream_buffer, int begin_bytepos, int end_bytepos);

        /*! \brief Function to check the likelihood of a stream to be an AV1 elementary stream.
         * \param [in] p_stream Pointer to the stream
         * \param [in] stream_size Size of the stream in bytes
         * \return The likelihood score
         */
        int CheckAv1EStream(uint8_t *p_stream, int stream_size);

        /*! \brief Function to check the likelihood of a stream to be an IVF container of AV1 elementary stream.
         * \param [in] p_stream Pointer to the stream
         * \param [in] stream_size Size of the stream in bytes
         * \return The likelihood score
         */
        int CheckIvfAv1Stream(uint8_t *p_stream, int stream_size);

        /*! \brief Function to check the likelihood of a stream to be a VP9 elementary stream.
         * \param [in] p_stream Pointer to the stream
         * \param [in] stream_size Size of the stream in bytes
         * \return The likelihood score
         */
        int CheckVp9EStream(uint8_t *p_stream, int stream_size);

        /*! \brief Function to check the likelihood of a stream to be an IVF container of VP9 elementary stream.
         * \param [in] p_stream Pointer to the stream
         * \param [in] stream_size Size of the stream in bytes
         * \return The likelihood score
         */
        int CheckIvfVp9Stream(uint8_t *p_stream, int stream_size);

        /*! \brief Function to read variable length unsigned n-bit number appearing directly in the bitstream. 4.10.3. uvlc().
        * \param [in] p_stream Bit stream pointer
        * \param [in] bit_offset Starting bit offset
        * \param [out] bit_offset Updated bit offset
        * \return The unsigned value
        */
        uint32_t ReadUVLC(const uint8_t *p_stream, size_t &bit_offset);
};