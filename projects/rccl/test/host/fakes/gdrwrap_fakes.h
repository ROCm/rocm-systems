/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for the file-static GDRCopy helpers in gdrwrap.h.
// Defaults fail loudly; fixtures install only the operations their path uses.

#ifndef RCCL_TEST_HOST_FAKES_GDRWRAP_FAKES_H_
#define RCCL_TEST_HOST_FAKES_GDRWRAP_FAKES_H_

#include <functional>

#include "gdrwrap.h"

extern std::function<ncclResult_t(void*, size_t, gdr_mh_t*, void**, size_t*, gdr_info_t*, uint32_t)>
    g_ncclGdrCudaMapPointer;
extern std::function<ncclResult_t(gdr_mh_t, void*, size_t)> g_ncclGdrCudaUnmapPointer;
extern std::function<ncclResult_t(gdr_mh_t, void*, const void*, size_t)> g_wrapGdrCopyToMapping;
extern std::function<ncclResult_t(gdr_mh_t, void*, const void*, size_t)> g_wrapGdrCopyFromMapping;

// Trampolines named by gdrwrap_seam.h. Keep these declarations guarded so a
// translation unit may bracket more than one included production unit.
ncclResult_t micro_ncclGdrCudaMapPointer(void* data, size_t size, gdr_mh_t* mh, void** map, size_t* mapSize,
                                         gdr_info_t* info, uint32_t pinFlags = 0);
ncclResult_t micro_ncclGdrCudaUnmapPointer(gdr_mh_t mh, void* map, size_t mapSize);
ncclResult_t micro_wrapGdrCopyToMapping(gdr_mh_t mh, void* map, const void* host, size_t size);
ncclResult_t micro_wrapGdrCopyFromMapping(gdr_mh_t mh, void* host, const void* map, size_t size);

void ResetGdrwrapFakes();

#endif  // RCCL_TEST_HOST_FAKES_GDRWRAP_FAKES_H_
