/*
************************************************************************************************************************
*
*  Copyright (C) 2020 Advanced Micro Devices, Inc.  All rights reserved.
*
***********************************************************************************************************************/

//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _ELF_UTILS_HPP
#define _ELF_UTILS_HPP

#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "top.hpp"

namespace amd {

#define MAX_ERROR_MESSAGE_LENGTH 1024
#define IO_BUF_SIZE            16 * 1024

class OclElfErr
{
public:
    // Temperary buffer for copying from file to file
    uint8_t* _copyBuffer;   // Initialized first time it is used

private:
    //  Keep the last error message.
    char _lastErrMsg[MAX_ERROR_MESSAGE_LENGTH];

public:

    OclElfErr() : _copyBuffer(NULL) { _lastErrMsg[0] = 0; }
    ~OclElfErr()  {
        if (_copyBuffer) {
            free(_copyBuffer);
        }
    }

    void Init() { _lastErrMsg[0] = 0; }

    void Fini() {
        _lastErrMsg[0] = 0;
        if (_copyBuffer) {
            free(_copyBuffer);
        }
        _copyBuffer = NULL;
    }

    // Return the last error message.
    const char* getOclElfError() const { return _lastErrMsg; }

    //
    // Save the error string in ErrorMessage.  If it is built without NDEBUG, the program
    // will terminate immediately with exit(1).
    //
    void xfail(const char *fmt, ...);

};

namespace oclelfutils {

/*
   Wrap malloc() with xfail(), so this returns newly-allocated memory or 0.
   The memory is guaranteed to be initialized to zero.
 */
void* xmalloc(OclElfErr& err, const size_t len);

/*
   Return file descriptor on success; return -1 on error and invoke xfail()
   to record the error.
 */
int xopen(OclElfErr& err, const char *fname, const int flags, const int perms);

/*
   Return 0 on success; return -1 on error.
 */
int xclose(OclElfErr& err, const char *fname, const int fd);

/*
   Return the file offset location on success; return -1 on error.
 */
off_t xlseek(OclElfErr& err, const char *fname, const int fd,
             const off_t o, const int whence);

/*
   Return the number of bytes that are read on success; return -1 on error.
 */
ssize_t xread(
    OclElfErr&   err,
    const char*  fname,   // File name for file descriptor 'fd'
    const int    fd,      // File descriptor
    void*        buf,     // buffer for reading
    const size_t buf_len  // capacity of buffer in bytes
    );

#if 0

/*
   Return the number of bytes that have been written on success; return -1 on error.
 */
ssize_t xwrite(
    OclElfErr&   err,
    const char*  fname,  // File name for file descriptor 'fd'
    const int    fd,     // File descriptor
    const void*  buf,    // data buffer to be written out
    const size_t buf_len // the size of data in bytes
    );

/*
   Allocate a copy of (str), invoke xfail() on failure.
   Returns 0 on error, or address of the allocated copy
 */
char* xstrdup(OclElfErr& err, const char *str);

/*
   get the length of an open file in bytes. return -1 if error.
 */
uint64_t xget_file_size(OclElfErr& err, const char *fname, const int fd);

/*
   Copy file 'infd' to file 'outfd'.
   Return the total number of bytes copied on success; return -1 on error.
 */
int64_t xcopyfile(OclElfErr& err, const char *in, const int infd,
                  const char *out, const int outfd);

/*
   Copy file from 'infd' to current offset in 'outfd', for 'size' bytes.
   Return 'size' on success; return -1 on error.
 */
int64_t xcopyfile_range(OclElfErr& err, const char *in, const int infd,
                     const char *out, const int outfd,
                     const uint64_t offset, const uint64_t size);

// Align a value to the page size.
uint64_t align_to_page(const uint64_t offset);

#endif

} // namespace elfutils

} // namespace amd

#endif
