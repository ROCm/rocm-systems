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

// rocJPEG workload module for the anytime-tool-config tests.
//
// PURPOSE: exercise the rocprofiler-sdk ROCJPEG_API *tracing*, not JPEG decoding itself.
// rocprofiler intercepts at the dispatch table (call entry), so every rocJPEG API call is
// traced even when the call fails -- e.g. when the AMD VCN VA-API driver is absent and the
// hardware JPEG decoder cannot initialize. We are testing the SDK, not rocJPEG.
//
// Therefore this module issues a deterministic sequence of rocJPEG API calls and IGNORES
// their return codes. It deliberately uses only calls that are safe to invoke when the
// backend is unavailable: rocJpegCreate, and the stream create/parse/destroy trio (stream
// parsing is CPU-side and does not touch the VA decoder). It does NOT call rocJpegDecode /
// rocJpegDestroy on a failed decoder handle, which would dereference uninitialized VA state
// and crash inside librocjpeg.
//
// C ABI (loaded via ctypes):
//   rocjpeg_init()                 -> rocJpegCreate (traced; may fail) + a stream handle.
//   rocjpeg_decode_file(path, n)   -> n reps of streamCreate/streamParse/streamDestroy on
//                                     the file's bytes. Returns 0 (calls issued).
//   rocjpeg_fini()                 -> destroy the stream handle.
//
// Per rocjpeg_decode_file rep: 3 traced stream calls (deterministic, backend-independent).

#include <rocjpeg/rocjpeg.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#define ROCJPEG_WORKLOAD_PUBLIC_API __attribute__((visibility("default")))

namespace
{
RocJpegHandle       handle        = nullptr;
RocJpegStreamHandle stream_handle = nullptr;
}  // namespace

extern "C" {

ROCJPEG_WORKLOAD_PUBLIC_API int
rocjpeg_init()
{
    // rocJpegCreate is traced whether or not the VA-API backend initializes. We keep the
    // (possibly-failed) handle but never pass it to decode/destroy.
    rocJpegCreate(ROCJPEG_BACKEND_HARDWARE, 0, &handle);
    rocJpegStreamCreate(&stream_handle);
    fprintf(stderr, "[python-rocjpeg] init calls issued\n");
    return 0;
}

ROCJPEG_WORKLOAD_PUBLIC_API int
rocjpeg_decode_file(const char* path, int reps)
{
    std::vector<uint8_t> data;
    std::ifstream        input(path, std::ios::binary | std::ios::ate);
    if(input.is_open())
    {
        std::streamsize size = input.tellg();
        data.resize(static_cast<size_t>(size));
        input.seekg(0, std::ios::beg);
        input.read(reinterpret_cast<char*>(data.data()), size);
    }

    for(int i = 0; i < reps; ++i)
    {
        RocJpegStreamHandle sh = nullptr;
        rocJpegStreamCreate(&sh);
        rocJpegStreamParse(data.data(), data.size(), sh);
        rocJpegStreamDestroy(sh);
    }
    return 0;
}

ROCJPEG_WORKLOAD_PUBLIC_API int
rocjpeg_fini()
{
    rocJpegStreamDestroy(stream_handle);
    stream_handle = nullptr;
    handle        = nullptr;
    fprintf(stderr, "[python-rocjpeg] finalized\n");
    return 0;
}

}  // extern "C"
