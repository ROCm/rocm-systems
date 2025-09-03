/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <rocdecode/roc_bitstream_reader.h>
#include <rocdecode/rocdecode.h>
#include <rocdecode/rocparser.h>
#include <cstring>
#include <iostream>

// Call rocDecode API with nullptrs to test rocDecode trace
// Eventually should replace with functional calls, but this requires
// finding and linking several FFmpeg and AV.. libraries (AVUTIL, AVCODEC, AVFORMAT)
// Additionally, some rocdecode/share files like ffmpeg_video_dec.cpp would need
// to be updated to resolve compiler warnings to compile with rocprofiler-sdk
int
test_rocdecode_decoder()
{
    rocDecStatus rocdecode_status = rocDecCreateDecoder(nullptr, nullptr);
    if(rocdecode_status != ROCDEC_INVALID_PARAMETER)
    {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER\n";
        return 1;
    }
    if(rocDecGetErrorName(rocdecode_status) == nullptr)
    {
        std::cerr << "Expected error name to not be null\n";
        return 1;
    }
    if(rocDecCreateVideoParser(nullptr, nullptr) != ROCDEC_INVALID_PARAMETER)
    {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER\n";
        return 1;
    }
    if(rocDecParseVideoData(nullptr, nullptr) != ROCDEC_INVALID_PARAMETER)
    {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER\n";
        return 1;
    }
    if(rocDecDestroyVideoParser(nullptr) != ROCDEC_INVALID_PARAMETER)
    {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER\n";
        return 1;
    }
    if(rocDecDestroyDecoder(nullptr) != ROCDEC_INVALID_PARAMETER)
    {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER\n";
        return 1;
    }
    if(rocDecGetDecoderCaps(nullptr) != ROCDEC_INVALID_PARAMETER)
    {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER\n";
        return 1;
    }
    if(rocDecGetDecodeStatus(nullptr, 0, nullptr) != ROCDEC_INVALID_PARAMETER)
    {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER\n";
        return 1;
    }
    if(rocDecReconfigureDecoder(nullptr, nullptr) != ROCDEC_INVALID_PARAMETER)
    {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER\n";
        return 1;
    }
    if(rocDecGetVideoFrame(nullptr, 0, nullptr, nullptr, nullptr) != ROCDEC_INVALID_PARAMETER)
    {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER\n";
        return 1;
    }
    return 0;
}

int
test_rocdecode_bitstream_reader(const std::string& input_file)
{
    // Set up bitstreamreader
    RocdecBitstreamReader bs_reader = nullptr;
    rocDecVideoCodec      rocdec_codec_id{};
    int                   bit_depth{};
    if(rocDecCreateBitstreamReader(&bs_reader, input_file.c_str()) != ROCDEC_SUCCESS)
    {
        std::cerr << "Failed to create the bitstream reader." << std::endl;
        return 1;
    }
    if(rocDecGetBitstreamCodecType(bs_reader, &rocdec_codec_id) != ROCDEC_SUCCESS)
    {
        std::cerr << "Failed to get stream codec type." << std::endl;
        return 1;
    }
    if(rocdec_codec_id >= rocDecVideoCodec_NumCodecs)
    {
        std::cerr << "Unsupported stream file type or codec type by the bitstream reader. Exiting."
                  << std::endl;
        return 1;
    }
    if(rocDecGetBitstreamBitDepth(bs_reader, &bit_depth) != ROCDEC_SUCCESS)
    {
        std::cerr << "Failed to get stream bit depth." << std::endl;
        return 1;
    }

    uint8_t* pvideo        = nullptr;
    int      n_video_bytes = 0;
    int64_t  pts           = 0;
    if(rocDecGetBitstreamPicData(bs_reader, &pvideo, &n_video_bytes, &pts) != ROCDEC_SUCCESS)
    {
        std::cerr << "Failed to get picture data." << std::endl;
        return 1;
    }

    // The following commands were originally used via the RocVideoDecoder from the
    // roc_video_dec.cpp file from the rocDecode repository. However, this file kept causing
    // compiler issues, so the file was removed. The following commands will return errors, but they
    // should should be tracked for testing purposes

    // rocDecCreateVideoParser
    RocdecVideoParser  rocdec_parser_     = nullptr;
    RocdecParserParams parser_params      = {};
    parser_params.codec_type              = rocdec_codec_id;
    parser_params.max_num_decode_surfaces = 1;
    parser_params.clock_rate              = 0;
    parser_params.max_display_delay       = 1;
    parser_params.pfn_sequence_callback   = nullptr;
    parser_params.pfn_decode_picture      = nullptr;
    rocDecCreateVideoParser(&rocdec_parser_, &parser_params);

    // rocDecGetDecoderCaps
    RocdecDecodeCaps decode_caps{};
    decode_caps.codec_type        = rocdec_codec_id;
    decode_caps.chroma_format     = rocDecVideoChromaFormat_420;
    decode_caps.bit_depth_minus_8 = 0;
    rocDecGetDecoderCaps(&decode_caps);

    // rocDecCreateDecoder
    rocDecDecoderHandle  roc_decoder_          = nullptr;
    RocDecoderCreateInfo videoDecodeCreateInfo = {};
    videoDecodeCreateInfo.device_id            = 0;
    videoDecodeCreateInfo.codec_type           = rocdec_codec_id;
    videoDecodeCreateInfo.chroma_format        = rocDecVideoChromaFormat_420;
    videoDecodeCreateInfo.bit_depth_minus_8    = 0;
    rocDecCreateDecoder(&roc_decoder_, &videoDecodeCreateInfo);

    // rocDecDecodeFrame
    RocdecPicParams* pPicParams = nullptr;
    rocDecDecodeFrame(roc_decoder_, pPicParams);

    // rocDecParseVideoData
    RocdecSourceDataPacket packet = {};
    rocDecParseVideoData(rocdec_parser_, &packet);

    // rocDecGetVideoFrame
    void*            src_dev_ptr[3]    = {0};
    uint32_t         src_pitch[3]      = {0};
    RocdecProcParams video_proc_params = {};
    rocDecGetVideoFrame(roc_decoder_, 0, src_dev_ptr, src_pitch, &video_proc_params);

    // rocDecGetDecodeStatus
    RocdecDecodeStatus dec_status{};
    rocDecGetDecodeStatus(roc_decoder_, 0, &dec_status);
    if(bs_reader)
    {
        rocDecDestroyBitstreamReader(bs_reader);
    }
    return 0;
}

int
main(int argc, char** argv)
{
    // Get input file
    std::string input_file_path{};
    for(int i = 1; i < argc; i++)
    {
        if(!strcmp(argv[i], "-i"))
        {
            if(++i == argc)
            {
                std::cerr << "Provide path to input file" << std::endl;
            }
            input_file_path = argv[i];
            continue;
        }
    }
    if(test_rocdecode_bitstream_reader(input_file_path) != 0)
    {
        std::cerr << "rocDecode bitsream reader test failed\n";
        return 1;
    }
    if(test_rocdecode_decoder() != 0)
    {
        std::cerr << "rocDecode decoder test failed\n";
        return 1;
    }
    return 0;
}
