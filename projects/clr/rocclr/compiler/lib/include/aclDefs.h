/* Copyright (c) 2011 - 2021 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef _ACL_DEFS_0_8_H_
#define _ACL_DEFS_0_8_H_

#ifndef ACL_API_ENTRY
#if defined(_WIN32) || defined(__CYGWIN__)
#define ACL_API_ENTRY __stdcall
#else
#define ACL_API_ENTRY
#endif
#endif

#ifndef ACL_API_0_8
#define ACL_API_0_8
#endif

#ifndef BIF_API_2_0
#define BIF_API_2_0
#endif

#ifndef BIF_API_2_1
#define BIF_API_2_1
#endif

#ifndef BIF_API_3_0
#define BIF_API_3_0
#endif

#ifndef MAX_HIDDEN_KERNARGS_NUM
#define MAX_HIDDEN_KERNARGS_NUM 6
#else
#error "MAX_HIDDEN_KERNARGS_NUM is already defined"
#endif

#endif  // _ACL_DEFS_0_8_H_
