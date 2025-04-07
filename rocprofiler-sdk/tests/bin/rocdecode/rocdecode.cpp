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
#include <iostream>
#include "roc_video_dec.h"

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
    // Set up bitstreamreader
    RocdecBitstreamReader bs_reader = nullptr;
    rocDecVideoCodec      rocdec_codec_id{};
    int                   bit_depth{};
    if(rocDecCreateBitstreamReader(&bs_reader, input_file_path.c_str()) != ROCDEC_SUCCESS)
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

    // Set up video decoder
    int                     device_id              = 0;
    OutputSurfaceMemoryType mem_type               = OUT_SURFACE_MEM_DEV_INTERNAL;
    bool                    b_force_zero_latency   = false;
    Rect*                   p_crop_rect            = nullptr;
    int                     disp_delay             = 1;
    bool                    b_extract_sei_messages = false;
    RocVideoDecoder*        viddec                 = new RocVideoDecoder(device_id,
                                                  mem_type,
                                                  rocdec_codec_id,
                                                  b_force_zero_latency,
                                                  p_crop_rect,
                                                  b_extract_sei_messages,
                                                  disp_delay);

    uint8_t* pvideo        = nullptr;
    int      n_video_bytes = 0;
    int64_t  pts           = 0;
    int      pkg_flags     = 0;
    int      decoded_pics  = 0;
    if(rocDecGetBitstreamPicData(bs_reader, &pvideo, &n_video_bytes, &pts) != ROCDEC_SUCCESS)
    {
        std::cerr << "Failed to get picture data." << std::endl;
        return 1;
    }
    // Treat 0 bitstream size as end of stream indicator
    if(n_video_bytes == 0)
    {
        pkg_flags |= ROCDEC_PKT_ENDOFSTREAM;
    }
    viddec->DecodeFrame(pvideo, n_video_bytes, pkg_flags, pts, &decoded_pics);
    viddec->DecodeFrame(pvideo, n_video_bytes, pkg_flags, pts, &decoded_pics);
    viddec->DecodeFrame(pvideo, n_video_bytes, pkg_flags, pts, &decoded_pics);
    if(bs_reader)
    {
        rocDecDestroyBitstreamReader(bs_reader);
    }
}
