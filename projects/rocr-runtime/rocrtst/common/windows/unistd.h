/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

// Windows unistd.h stub header
// This provides minimal unistd definitions for compilation compatibility

#ifndef _UNISTD_H_STUB
#define _UNISTD_H_STUB

#ifndef _WIN32
#include_next <unistd.h>
#else
#include <windows.h>
#include <io.h>
#include <process.h>
#include <direct.h>

// POSIX function mappings for Windows
#define close _close
#define read _read
#define write _write
#define open _open
#define lseek _lseek
#define getcwd _getcwd
#define chdir _chdir
#define rmdir _rmdir
#define unlink _unlink
#define getpid _getpid

// POSIX constants
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#define STDIN_FILENO 0

// Access modes
#define R_OK 4
#define W_OK 2
#define X_OK 1
#define F_OK 0

// File type constants
#define S_IRUSR 0400
#define S_IWUSR 0200
#define S_IXUSR 0100

// Define pid_t for Windows - use intptr_t (pointer-sized signed integer)
typedef intptr_t pid_t;

#ifdef __cplusplus
extern "C" {
#endif
pid_t fork(void);
#ifdef __cplusplus
}
#endif

#endif // _WIN32

#endif // _UNISTD_H_STUB
