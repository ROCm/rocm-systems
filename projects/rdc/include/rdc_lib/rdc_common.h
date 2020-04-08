/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef RDC_LIB_RDC_COMMON_H_
#define RDC_LIB_RDC_COMMON_H_
#include <iostream>

#define RDC_ERROR  0
#define RDC_INFO   1
#define RDC_DEBUG  2

#define RDC_LOG(debug_level, msg) do { \
    auto& logger = amd::rdc::RdcLogger::getLogger(); \
    if (logger.should_log((debug_level))) { \
        logger.get_ostream() << \
        logger.get_log_header((debug_level), __FILE__, __LINE__) << \
        msg << std::endl; \
    } \
} while (0)

/**
 *  @brief The strncpy but with null terminated
 *
 *  @details It will copy at most n-1 bytes from src to dst, and
 *  always adds a null terminator following the bytes copied to dst.
 *
 *  @param[out] dest The destination string to copy
 *
 *  @param[in] src The source string to be copied
 *
 *  @param[in] n At most n-1 bytes will be copied
 *
 *  @retval Return a pointer to the destination string.
 */
char *strncpy_with_null(char *dest, const char *src, size_t n);


#endif  // RDC_LIB_RDC_COMMON_H_
