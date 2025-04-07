#pragma once

#define ROCDECODE_RUNTIME_API_TABLE_MAJOR_VERSION 0
#define ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION  1

#include <cstddef>
#include <cstdint>

extern "C" {
// fake rccl function
enum RocJpegStatus
{
};

enum RocJpegStreamHandle
{
};

RocJpegStatus
rocJpegStreamCreate(RocJpegStreamHandle* jpeg_stream_handle)
    __attribute__((visibility("default")));
}

namespace rocjpeg
{
struct rocjpegApiFuncTable
{
    uint64_t                         size                   = 0;
    decltype(::rocJpegStreamCreate)* rocJpegStreamCreate_fn = nullptr;
};

RocJpegStatus
rocJpegStreamCreate(RocJpegStreamHandle* jpeg_stream_handle);

// populates rocjpeg api table with function pointers
inline void
initialize_rocjpeg_api_table(rocjpegApiFuncTable* dst)
{
    dst->size                   = sizeof(rocjpegApiFuncTable);
    dst->rocJpegStreamCreate_fn = &::rocjpeg::rocJpegStreamCreate;
}

// copies the api table from src to dst
inline void
copy_rocjpeg_api_table(rocjpegApiFuncTable* dst, const rocjpegApiFuncTable* src)
{
    *dst = *src;
}
}  // namespace rocjpeg
