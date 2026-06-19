/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef ARCHINFO_H_
#define ARCHINFO_H_

#include <stdint.h>
#include <string.h>

/*
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
*/

void GcnArchNameFormat(char *gcnArchName, char* out);
void convertGcnArchToGcnArchName(const char* gcnArch, const char** gcnArchName);
int GetGcnArchName(int deviceId, char* out);
double GetDeviceWallClockRateInKhz(int deviceId);
bool IsArchMatch(char const* arch, char const* target);

/* Host Code: Must match NCCL_LL128_LINESIZE / NCCL_LL128_LINEELEMS in device 
 * code for the same arch. */
inline int rcclLL128LineElemsFromArch(char const* arch) {
  return IsArchMatch(arch, "gfx1250") ? 128 / (int)sizeof(uint64_t) : 64 / (int)sizeof(uint64_t);
}
inline int rcclLL128DataElemsFromArch(char const* arch) {
  return rcclLL128LineElemsFromArch(arch) - 1;
}

#endif // ARCHINFO_H
