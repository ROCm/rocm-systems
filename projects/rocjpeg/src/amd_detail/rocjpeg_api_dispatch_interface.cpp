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
#include "../../api/amd_detail/rocjpeg_api_trace.h"

namespace rocjpeg {
const RocJpegDispatchTable* GetRocJpegDispatchTable();
} //namespace rocjpeg


RocJpegStatus ROCJPEGAPI rocJpegStreamCreate(RocJpegStreamHandle *jpeg_stream_handle) {
    return rocjpeg::GetRocJpegDispatchTable()->pfn_rocjpeg_stream_create(jpeg_stream_handle);
}
RocJpegStatus ROCJPEGAPI rocJpegStreamParse(const unsigned char *data, size_t length, RocJpegStreamHandle jpeg_stream_handle) {
    return rocjpeg::GetRocJpegDispatchTable()->pfn_rocjpeg_stream_parse(data, length, jpeg_stream_handle);
}
RocJpegStatus ROCJPEGAPI rocJpegStreamDestroy(RocJpegStreamHandle jpeg_stream_handle) {
    return rocjpeg::GetRocJpegDispatchTable()->pfn_rocjpeg_stream_destroy(jpeg_stream_handle);
}
RocJpegStatus ROCJPEGAPI rocJpegCreate(RocJpegBackend backend, int device_id, RocJpegHandle *handle) {
    return rocjpeg::GetRocJpegDispatchTable()->pfn_rocjpeg_create(backend, device_id, handle);
}
RocJpegStatus ROCJPEGAPI rocJpegDestroy(RocJpegHandle handle) {
    return rocjpeg::GetRocJpegDispatchTable()->pfn_rocjpeg_destroy(handle);
}
RocJpegStatus ROCJPEGAPI rocJpegGetImageInfo(RocJpegHandle handle, RocJpegStreamHandle jpeg_stream_handle, uint8_t *num_components, RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights) {
    return rocjpeg::GetRocJpegDispatchTable()->pfn_rocjpeg_get_image_info(handle, jpeg_stream_handle, num_components, subsampling, widths, heights);
}
RocJpegStatus ROCJPEGAPI rocJpegDecode(RocJpegHandle handle, RocJpegStreamHandle jpeg_stream_handle, const RocJpegDecodeParams *decode_params, RocJpegImage *destination) {
    return rocjpeg::GetRocJpegDispatchTable()->pfn_rocjpeg_decode(handle, jpeg_stream_handle, decode_params, destination);
}
RocJpegStatus ROCJPEGAPI rocJpegDecodeBatched(RocJpegHandle handle, RocJpegStreamHandle *jpeg_stream_handles, int batch_size, const RocJpegDecodeParams *decode_params, RocJpegImage *destinations) {
    return rocjpeg::GetRocJpegDispatchTable()->pfn_rocjpeg_decode_batched(handle, jpeg_stream_handles, batch_size, decode_params, destinations);
}
const char* ROCJPEGAPI rocJpegGetErrorName(RocJpegStatus rocjpeg_status) {
    return rocjpeg::GetRocJpegDispatchTable()->pfn_rocjpeg_get_error_name(rocjpeg_status);
}