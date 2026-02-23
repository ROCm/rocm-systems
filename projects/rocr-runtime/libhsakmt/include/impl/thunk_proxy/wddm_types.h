////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2015, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <stdint.h>
#include <no_sal2.h>

typedef uint32_t UINT, *UINT_PTR;
typedef int32_t  INT32;
typedef int32_t  LONG;
typedef uint32_t ULONG, *ULONG_PTR;
typedef int64_t  LONGLONG;
typedef int64_t  LONG64;
typedef uint64_t ULONGLONG;
typedef uint64_t ULONG64, *ULONG64_PTR;
typedef uint8_t  BYTE;
typedef uint16_t WORD;
typedef uint32_t DWORD;
typedef int32_t  BOOL;
typedef int32_t  NTSTATUS;
typedef uint16_t USHORT;
typedef uint16_t UINT16;
typedef uint32_t UINT32;
typedef uint64_t UINT64;
typedef int32_t  INT;
typedef uint64_t SIZE_T;
typedef void VOID;
typedef float FLOAT;
typedef char CHAR;
typedef unsigned char UCHAR;
typedef UCHAR BOOLEAN;
typedef int16_t WCHAR;
typedef void *HANDLE;
typedef void *PVOID;
typedef void *LPVOID;
typedef const int16_t *PCWSTR;

#define ULONG ULONG
#define ULONG_PTR ULONG_PTR
#define USHORT USHORT

#define DECLARE_HANDLE(name) struct name##__{int unused;}; typedef struct name##__ *name
#define C_ASSERT(e) typedef char __C_ASSERT__[(e)?1:-1]

DECLARE_HANDLE(HWND);
DECLARE_HANDLE(HDC);
DECLARE_HANDLE(PALETTEENTRY);

typedef struct tagPOINT {
    LONG x;
    LONG y;
} POINT;

typedef struct tagRECT {
    LONG left;
    LONG top;
    LONG right;
    LONG bottom;
} RECT;

typedef struct tagRECTL {
    LONG left;
    LONG top;
    LONG right;
    LONG bottom;
} RECTL;

typedef union _LARGE_INTEGER {
	struct {
		DWORD LowPart;
		DWORD HighPart;
	} u;
	LONGLONG QuadPart;
} LARGE_INTEGER;

typedef LARGE_INTEGER *PLARGE_INTEGER;

typedef union _ULARGE_INTEGER {
    struct {
        ULONG LowPart;
        ULONG HighPart;
    } DUMMYSTRUCTNAME;
    struct {
        ULONG LowPart;
        ULONG HighPart;
    } u;
    ULONGLONG QuadPart;
} ULARGE_INTEGER;

typedef ULARGE_INTEGER *PULARGE_INTEGER;

typedef struct _LUID {
    ULONG LowPart;
    LONG HighPart;
} LUID, *PLUID;

typedef enum _DEVICE_POWER_STATE {
    PowerDeviceUnspecified = 0,
    PowerDeviceD0,
    PowerDeviceD1,
    PowerDeviceD2,
    PowerDeviceD3,
    PowerDeviceMaximum
} DEVICE_POWER_STATE, *PDEVICE_POWER_STATE;

#define _Check_return_
#define APIENTRY
#define CONST const
#define IN
#define OUT
#define FAR
#define MAX_PATH 260
#define __stdcall

#ifndef GUID_DEFINED
#define GUID_DEFINED
typedef struct _GUID {
    uint32_t Data1;
    uint16_t Data2;
    uint16_t Data3;
    uint8_t  Data4[ 8 ];
} GUID;
#endif

#include <guiddef.h>

