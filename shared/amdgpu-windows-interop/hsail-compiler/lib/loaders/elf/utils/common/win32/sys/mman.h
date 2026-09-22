/*
************************************************************************************************************************
*
*  Copyright (C) 2020 Advanced Micro Devices, Inc.  All rights reserved.
*
***********************************************************************************************************************/

#ifndef _MMAN_H_
#define _MMAN_H_
#if defined(WIN32)

void *mmap(void*, size_t, int, int, int, unsigned);
int munmap(void*, size_t);
#endif
#endif // _MMAN_H_
