/*
************************************************************************************************************************
*
*  Copyright (C) 2020 Advanced Micro Devices, Inc.  All rights reserved.
*
***********************************************************************************************************************/

//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

#include "elf_utils.hpp"
#include "memfile.h"

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>

/*
   See elf_utils.hpp for descriptions about each functions
 */

namespace amd {

#define ELF_OPEN             mem_open
#define ELF_READ(f, b, l)    mem_read((f), (b), (unsigned int)(l))
#define ELF_WRITE            mem_write
#define ELF_CLOSE            mem_close
#define ELF_LSEEK            mem_lseek

/*
   Save the error string in _lastErrMsg.  If it is built without NDEBUG, the program
   will terminate immediately with exit(1).
 */
void OclElfErr::xfail(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(&_lastErrMsg[0], (size_t)MAX_ERROR_MESSAGE_LENGTH, fmt, ap);
    va_end(ap);

#ifndef NDEBUG
    printf("%s\n", _lastErrMsg);
    exit(1);
#endif
}

namespace oclelfutils {

/*
   Wrap malloc() with xfail(), so this returns newly-allocated memory or 0.
   The memory is guaranteed to be initialized to zero.
 */
void* xmalloc(OclElfErr& err, const size_t len)
{
    void *retval = calloc(1, len);
    if (retval == NULL) {
        err.xfail("xmalloc failed: out of memory");
        return NULL;
    }
    return retval;
}

/*
   Return file descriptor on success; return -1 on error and invoke xfail()
   to record the error.
 */
int xopen(OclElfErr& err, const char *fname, const int in_flags, const int perms)
{
    const int retval = ELF_OPEN(fname, in_flags, perms);
    if (retval == -1) {
        err.xfail("Failed to open '%s': %s", fname, strerror(errno));
        return -1;
    }
    return retval;
}

/*
   Return 0 on success; return -1 on error.
 */
int xclose(OclElfErr& err, const char *fname, const int fd)
{
    int rc;
    while ( ((rc = :: ELF_CLOSE(fd)) == -1) && (errno == EINTR) ) { ;/* spin. */ }
    if (rc == -1) {
        err.xfail("Failed to close '%s': %s", fname, strerror(errno));
        return -1;
    }
    return rc;
}

/*
   Return the file offset location on success; return -1 on error.
 */
off_t xlseek(
   OclElfErr&     err,
   const char* fname,
   const int   fd,
   const off_t offset,
   const int   whence)
{
    // For really big file  _lseeki64/lseek64 are needed. For now,
    // lseek/_lseek is enough.
    off_t res = ELF_LSEEK(fd, offset, whence);
    if (res == -1) {
        err.xfail("Failed to seek in '%s': %s", fname, strerror(errno));
        return -1;
    }
    return res;
}

/*
   Return the number of bytes that are read on success; return -1 on error.
 */
ssize_t xread(
    OclElfErr&   err,
    const char*  fname,
    const int    fd,
    void*        buf,
    const size_t buf_len
    )
{
    ssize_t rc;
    while (((rc = ELF_READ(fd, buf, buf_len)) == -1) && (errno == EINTR)) { ;/* spin */ }
    if (rc < 0) {
        err.xfail("Failed to read '%s': %s", fname, strerror(errno));
        return -1;
    }
    return rc;
}

#if 0

/*
   Return the number of bytes that have been written on success; return -1 on error.
 */
ssize_t xwrite(OclElfErr&      err,
               const char*  fname,
               const int    fd,
               const void*  buf,
               const size_t len)
{
    ssize_t rc;
    while (((rc = ELF_WRITE(fd, buf, len)) == -1) && (errno == EINTR)) { ;/* spin */ }
    if ( (rc == -1) || (rc != (ssize_t)len) ) {
        err.xfail("Failed to write '%s': %s", fname, strerror(errno));
        return -1;
    }
    return rc;
}

/*
   Allocate a copy of (str), invoke xfail() on failure.
   Returns NULL on error, or address of the allocated copy
 */
char* xstrdup(OclElfErr& err, const char *str)
{
    char* retval = (char*)xmalloc(err, strlen(str) + 1);
    if (retval == NULL) {
        err.xfail("xstrdup failed: cannot allocate new char string");
        return NULL;
    }
    strcpy(retval, str);
    return retval;
}

/*
   get the length of an open file in bytes. return -1 on error.
 */
uint64_t xget_file_size(OclElfErr& err, const char *fname, const int fd)
{
    struct stat statbuf;
    if (fstat(fd, &statbuf) == -1) {
        err.xfail("Failed to fstat '%s': %s", fname, strerror(errno));
        return -1;
    }
    return (uint64_t) statbuf.st_size;
}

/*
   Copy file 'infd' to file 'outfd'.
   Return the total number of bytes copied on success; return -1 on error.
 */
int64_t xcopyfile(
    OclElfErr&     err,
    const char* in,
    const int   infd,
    const char* out,
    const int   outfd
    )
{
    uint64_t retval = 0;
    ssize_t rc = 0;
    off_t res = xlseek(err, in, infd, 0, SEEK_SET);
    if (res == -1) {
        err.xfail("xcopyfile failed in xlseek : in %s, out %s", in, out);
        return -1;
    }

    uint8_t* copybuf = err._copyBuffer;
    if (copybuf == NULL) {
        copybuf = (uint8_t*)xmalloc(err, IO_BUF_SIZE);
        err._copyBuffer = copybuf;
    }

    while ( (rc = xread(err, in, infd, copybuf, IO_BUF_SIZE)) > 0 ) {
        retval += (uint64_t) rc;
        int ret = xwrite(err, out, outfd, copybuf, rc);
        if (ret == -1) {
            err.xfail("xcopyfile failed in xwrite: in %s, out %s", in, out);
            return -1;
        }
    }
    if (rc == -1) {
        err.xfail("xcopyfile failed in xread: in %s, out %s", in, out);
        return -1;
    }
    return retval;
}

/*
   Copy file from 'infd' to current offset in 'outfd', for 'size' bytes.
   Return 'size' on success; return -1 on error.
 */
int64_t
xcopyfile_range(
    OclElfErr& err, const char *in, const int infd,
    const char *out, const int outfd,
    const uint64_t offset, const uint64_t size
    )
{
    uint8_t* copybuf = err._copyBuffer;
    if (copybuf == NULL) {
        copybuf = (uint8_t*)xmalloc(err, IO_BUF_SIZE);
        err._copyBuffer = copybuf;
    }

    ssize_t rc = xlseek(err, in, infd, (off_t) offset, SEEK_SET);
    if (rc == -1) {
        err.xfail("xcopyfile_range: xlseek() failed: %s", in);
        return -1;
    }

    uint64_t remaining = size;
    while (remaining >= IO_BUF_SIZE) {
        rc = xread(err, in, infd, copybuf, IO_BUF_SIZE);
        if ((rc == -1) || (rc != IO_BUF_SIZE)) {
            err.xfail("xcopyfile_range: xread() failed %s", in);
            return -1;
        }
        rc = xwrite(err, out, outfd, copybuf, IO_BUF_SIZE);
        if (rc == -1) {
            err.xfail("xcopyfile_range: xwrite() failed: %s", out);
        }

        remaining -= (uint64_t) IO_BUF_SIZE;
    }

    if (remaining > 0) {
        rc = xread(err, in, infd, copybuf, IO_BUF_SIZE);
        if ((rc == -1) || (rc != (ssize_t)remaining)) {
            err.xfail("xcopyfile_range: xread() failed %s", in);
            return -1;
        }
        rc = xwrite(err, out, outfd, copybuf, rc);
        if (rc == -1) {
            err.xfail("xcopyfile_range: xwrite() failed: %s", out);
        }
    }
    return size;
}

uint64_t
align_to_page(const uint64_t offset)
{
    // TODO_jugu don't use hardcoded pagesize.
    return (offset + ((1LL << 12) -1)) & ((uint64_t)(-(1LL << 12)));
}

#endif

} // namespace elfutils

} // namespace amd
