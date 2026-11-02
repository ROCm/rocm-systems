/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Installs the libdl seam over a unit under test. Include it immediately BEFORE
// the #include <UNIT>_CC_PATH and fakes/dl_seam_undef.h straight after, as with
// fakes/libc_seam.h. No include guard: the define/undef pair is positional.

#include <dlfcn.h>

extern "C" {
void* micro_dlopen(const char*, int);
void* micro_dlsym(void*, const char*);
void* micro_dlvsym(void*, const char*, const char*);
int micro_dlclose(void*);
char* micro_dlerror(void);
}  // extern "C"

#define dlopen micro_dlopen
#define dlsym micro_dlsym
#define dlvsym micro_dlvsym
#define dlclose micro_dlclose
#define dlerror micro_dlerror
