
#pragma once

#include <dlfcn.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(ROCP_REG_TEST_WEAK) && ROCP_REG_TEST_WEAK > 0
#    pragma weak hip_init
#    pragma weak hsa_init
#    pragma weak roctxRangePush
#    pragma weak roctxRangePop
#    pragma weak ncclGetVersion
#    pragma weak rocDecCreateDecoder
#    pragma weak rocJpegStreamCreate
#endif

extern void
hip_init(void);

extern void
hsa_init(void);

extern void
roctxRangePush(const char*);

extern void
roctxRangePop(const char*);

enum ncclResult_t
{
};

extern ncclResult_t
ncclGetVersion(int* version);

enum rocDecStatus
{
};

enum rocDecDecoderHandle
{
};

enum RocDecoderCreateInfo
{
};

extern rocDecStatus
rocDecCreateDecoder(rocDecDecoderHandle*  decoder_handle,
                    RocDecoderCreateInfo* decoder_create_info);

enum RocJpegStatus
{
};

enum RocJpegStreamHandle
{
};

extern RocJpegStatus
rocJpegStreamCreate(RocJpegStreamHandle* jpeg_stream_handle);

#ifdef __cplusplus
}
#endif
