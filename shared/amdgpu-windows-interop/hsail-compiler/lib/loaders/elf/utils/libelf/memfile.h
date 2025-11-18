/*
************************************************************************************************************************
*
*  Copyright (C) 2020 Advanced Micro Devices, Inc.  All rights reserved.
*
***********************************************************************************************************************/

//
// Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef _MEMFILE_H
#define _MEMFILE_H

#include <sys/types.h>
#if !defined(_MSC_VER)
#include <sys/stat.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif

// Acts the same as open(), but path can be NULL, which is a request for in memory file
extern int     mem_open(const char *path, int oflag, int pmode);
extern off_t   mem_read(int fd, void *buffer, size_t count);
extern off_t   mem_write(int fd, const void *buffer, size_t count);
extern int     mem_close(int fd);
extern off_t   mem_lseek(int fd, off_t offset, int origin);
extern int     mem_fstat(int fd, struct stat *buf);
extern int     mem_ftruncate(int fd, size_t len);
extern off_t   mem_sendfile(int out_fd, int in_fd, off_t *offset, size_t count);
extern void*   mem_mmap(void* start, size_t length, int prot, int flags, int fd, unsigned offset);
extern int     mem_munmap(void* start, size_t length);

#if defined(__cplusplus)
}
#endif

#endif // !_MEMFILE_H
