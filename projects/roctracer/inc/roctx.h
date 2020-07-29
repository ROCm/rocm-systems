/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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

////////////////////////////////////////////////////////////////////////////////
//
// ROC-TX API
//
// ROC-TX library, Code Annotation API.
// The goal of the implementation is to provide functionality for annotating
// events, code ranges, and resources in applications.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef INC_ROCTX_H_
#define INC_ROCTX_H_

#include <stdint.h>

#define ROCTX_VERSION_MAJOR 1
#define ROCTX_VERSION_MINOR 0

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

////////////////////////////////////////////////////////////////////////////////
// Returning library version
uint32_t roctx_version_major();
uint32_t roctx_version_minor();

////////////////////////////////////////////////////////////////////////////////
// Returning the last error
const char* roctracer_error_string();

////////////////////////////////////////////////////////////////////////////////
// Markers annotating API

// A marker created by given ASCII massage
void roctxMarkA(const char* message);
#define roctxMark(message) roctxMarkA(message)

////////////////////////////////////////////////////////////////////////////////
// Ranges annotating API

// Returns the 0 based level of a nested range being started by given message associated to this range.
// A negative value is returned on the error.
int roctxRangePushA(const char* message);
#define roctxRangePush(message) roctxRangePushA(message)

// Marks the end of a nested range.
// A negative value is returned on the error.
int roctxRangePop();

// ROCTX range id type
typedef uint64_t roctx_range_id_t;

// Starts a process range
roctx_range_id_t roctxRangeStartA(const char* message);
#define roctxRangeStart(message) roctxRangeStartA(message)

// Stop a process range
void roctxRangeStop(roctx_range_id_t id);

#ifdef __cplusplus
}  // extern "C" block
#endif  // __cplusplus

#endif  // INC_ROCTX_H_
