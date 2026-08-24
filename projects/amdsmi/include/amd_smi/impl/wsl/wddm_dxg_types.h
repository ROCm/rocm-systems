/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 */

#pragma once

#if defined(__linux__)
#include <ntstatus.h>

#include <cstdint>

#include "amd_smi/impl/wsl/wddm_types.h"
// windows wchar is 16bit, but linux is 32bit
// seems libdxcore (not dxgkrnl.ko) convert thunk windows wchar to linux one
// so only accept 32bit wchar args. note driver private data structure still
// use 16bit wchar
#define WCHAR wchar_t
#define PCWSTR const wchar_t*
#else
#define WIN32_NO_STATUS
#include <Windows.h>
#undef WIN32_NO_STATUS
#endif
#include <d3dkmthk.h>

#include "ntstatus.h"
#if defined(__linux__)
#undef WCHAR
#undef PCWSTR
#endif

using gpusize = uint64_t;  // Used to specify GPU addresses and sizes of GPU allocations
using WinAllocationHandle = D3DKMT_HANDLE;
using WinResourceHandle = D3DKMT_HANDLE;
using WinContextHandle = D3DKMT_HANDLE;
using WinDeviceHandle = D3DKMT_HANDLE;
using WinAdapterHandle = D3DKMT_HANDLE;

// reference dk/winnt.h
#define STANDARD_RIGHTS_REQUIRED (0x000F0000L)

// reference dk/ntdef.h
#define OBJ_INHERIT (0x00000002L)
typedef WCHAR *PWCHAR, *LPWCH, *PWCH;
typedef struct _UNICODE_STRING {
  USHORT Length;
  USHORT MaximumLength;
#ifdef MIDL_PASS
  [ size_is(MaximumLength / 2), length_is((Length) / 2) ] USHORT* Buffer;
#else   // MIDL_PASS
  _Field_size_bytes_part_opt_(MaximumLength, Length) PWCH Buffer;
#endif  // MIDL_PASS
} UNICODE_STRING;
typedef UNICODE_STRING* PUNICODE_STRING;
typedef const UNICODE_STRING* PCUNICODE_STRING;

typedef struct _OBJECT_ATTRIBUTES {
  ULONG Length;
  HANDLE RootDirectory;
  PUNICODE_STRING ObjectName;
  ULONG Attributes;
  PVOID SecurityDescriptor;
  PVOID SecurityQualityOfService;
} OBJECT_ATTRIBUTES;
#define InitializeObjectAttributes(p, n, a, r, s) \
  {                                               \
    (p)->Length = sizeof(OBJECT_ATTRIBUTES);      \
    (p)->RootDirectory = r;                       \
    (p)->Attributes = a;                          \
    (p)->ObjectName = n;                          \
    (p)->SecurityDescriptor = s;                  \
    (p)->SecurityQualityOfService = NULL;         \
  }
