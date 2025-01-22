#pragma once

#define ROCDECODE_RUNTIME_API_TABLE_MAJOR_VERSION 0
#define ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION  1

#include <cstddef>
#include <cstdint>

extern "C" {
// fake rccl function
enum rocDecStatus
{
};

enum rocDecDecoderHandle
{
};

enum RocDecoderCreateInfo
{
};

rocDecStatus
rocDecCreateDecoder(rocDecDecoderHandle*  decoder_handle,
                    RocDecoderCreateInfo* decoder_create_info)
    __attribute__((visibility("default")));
}

namespace rocdecode
{
struct rocdecodeApiFuncTable
{
    uint64_t                         size                   = 0;
    decltype(::rocDecCreateDecoder)* rocDecCreateDecoder_fn = nullptr;
};

rocDecStatus
rocDecCreateDecoder(rocDecDecoderHandle*  decoder_handle,
                    RocDecoderCreateInfo* decoder_create_info);

// populates rocdecode api table with function pointers
inline void
initialize_rocdecode_api_table(rocdecodeApiFuncTable* dst)
{
    dst->size                   = sizeof(rocdecodeApiFuncTable);
    dst->rocDecCreateDecoder_fn = &::rocdecode::rocDecCreateDecoder;
}

// copies the api table from src to dst
inline void
copy_rocdecode_api_table(rocdecodeApiFuncTable* dst, const rocdecodeApiFuncTable* src)
{
    *dst = *src;
}
}  // namespace rocdecode
