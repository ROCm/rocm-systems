// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// rocDecode workload module for the anytime-tool-config tests.
//
// PURPOSE: exercise the rocprofiler-sdk ROCDECODE_API *tracing*, not video decoding itself.
// rocprofiler intercepts at the dispatch table (call entry), so every rocDecode API call is
// traced regardless of whether GPU decode works. We are testing the SDK, not rocDecode.
//
// This module drives the rocDecode *bitstream reader* API (rocDecCreateBitstreamReader,
// rocDecGetBitstreamCodecType, rocDecGetBitstreamBitDepth, rocDecGetBitstreamPicData,
// rocDecDestroyBitstreamReader) -- all of which are in the traced rocDecode dispatch table
// and are CPU-side, so they work and are traced even when the AMD VCN VA-API video decoder
// is unavailable. It deliberately avoids the VA-API decoder path (rocDecCreateDecoder /
// RocVideoDecoder), which would crash without a working backend.
//
// C ABI (loaded via ctypes):
//   rocdecode_decode_file(path) -> open the bitstream, query codec/bit-depth, then read all
//                                  picture packets to EOS, then destroy the reader. Returns
//                                  the number of packets read (>= 0), or -1 on error.

#include <rocdecode/roc_bitstream_reader.h>
#include <rocdecode/rocdecode.h>
#include <rocdecode/rocparser.h>

#include <cstdio>

#define ROCDECODE_WORKLOAD_PUBLIC_API __attribute__((visibility("default")))

extern "C" {

ROCDECODE_WORKLOAD_PUBLIC_API int
rocdecode_decode_file(const char* path)
{
    RocdecBitstreamReader bs_reader = nullptr;
    if(rocDecCreateBitstreamReader(&bs_reader, path) != ROCDEC_SUCCESS)
    {
        fprintf(stderr, "[python-rocdecode] could not open bitstream %s\n", path);
        return -1;
    }

    rocDecVideoCodec codec_id  = rocDecVideoCodec_NumCodecs;
    int              bit_depth = 8;
    // Traced rocDecode API calls (CPU-side; independent of the VA-API video decoder).
    rocDecGetBitstreamCodecType(bs_reader, &codec_id);
    rocDecGetBitstreamBitDepth(bs_reader, &bit_depth);

    int      packets       = 0;
    uint8_t* pvideo        = nullptr;
    int      n_video_bytes = 0;
    int64_t  pts           = 0;
    do
    {
        if(rocDecGetBitstreamPicData(bs_reader, &pvideo, &n_video_bytes, &pts) != ROCDEC_SUCCESS)
            break;
        if(n_video_bytes > 0) ++packets;
    } while(n_video_bytes > 0);

    rocDecDestroyBitstreamReader(bs_reader);
    fprintf(stderr, "[python-rocdecode] read %d packets from %s\n", packets, path);
    return packets;
}

}  // extern "C"
