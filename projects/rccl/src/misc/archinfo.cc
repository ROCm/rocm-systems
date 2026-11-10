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

#include "archinfo.h"
#include "checks.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

void GcnArchNameFormat(char* gcnArchName, char* out) {
  // this function parses the char array from the device properties into something easier to handle.
  // as the gcnArchName attribute looks something like: "gfx900:xnack+:blah-:etc-"
  char* gcnArchNameToken = strtok(gcnArchName, ":");
  strcpy(out, gcnArchNameToken);
}

void convertGcnArchToGcnArchName(const char* gcnArch, const char** gcnArchName) {
  // gcnArch is deprecated and we should instead use gcnArchName; however, some data files still have
  // the older gcnArch value.  There's only a handful of architectures that were coded prior to deprecation,
  // so we handle those cases here.
  if (strcmp(gcnArch, "906") == 0) *gcnArchName = "gfx906";
  else if (strcmp(gcnArch, "908") == 0) *gcnArchName = "gfx908";
  else if (strcmp(gcnArch, "910") == 0) *gcnArchName = "gfx90a";
  else if (strcmp(gcnArch, "942") == 0) *gcnArchName = "gfx942";
  else if (strcmp(gcnArch, "950") == 0) *gcnArchName = "gfx950";
  else if (strcmp(gcnArch, "1250") == 0) *gcnArchName = "gfx1250";
  else *gcnArchName = gcnArch;
}

int GetGcnArchName(int deviceId, char* out) {
  // this is a generic call in to get a consistent gcnArchName
  hipDeviceProp_t devProp;
  CUDACHECK(hipGetDeviceProperties(&devProp, deviceId));
  GcnArchNameFormat(devProp.gcnArchName, out);
  return 0;
}

double GetDeviceWallClockRateInKhz(int deviceId) {
  char gcn[256];
  GetGcnArchName(deviceId, gcn);
  if (strncmp("gfx942", gcn, 6) == 0) return 1.0E5;
  else if (strncmp("gfx950", gcn, 6) == 0) return 1.0E5;
  else if (strncmp("gfx1250", gcn, 7) == 0) return 1.0E5;
  else return 2.5E4;
}
/**
 * prefix comparison
 */
bool IsArchMatch(char const* arch, char const* target) {
  // helper function to reduce clutter in code elsewhere.  Returns true on match.
  if (arch == nullptr || target == nullptr) return false;
  return (strncmp(arch, target, strlen(target)) == 0);
}

bool rcclArchIsBuilt(char const* archName) {
  char const* targets = RCCL_BUILT_GPU_TARGETS;
  // Nothing to check against, or one code object covers a whole family: assume supported.
  if (targets[0] == '\0' || strstr(targets, "-generic") != nullptr) return true;
  if (archName == nullptr) return true;

  const size_t n = strcspn(archName, ":");  // base name, feature suffix dropped
  if (n == 0) return true;

  for (char const* p = targets; *p != '\0';) {
    // GPU_TARGETS verbatim, so an entry ends at ';' (next target) or ':' (its own
    // suffix, e.g. gfx942:xnack+). Comparing exactly n bytes AND requiring a
    // delimiter is what keeps a shorter target ("gfx110") from accepting a longer
    // device name ("gfx1103"), and "gfx1250" from accepting "gfx1250-strict".
    if (strncmp(p, archName, n) == 0 && (p[n] == '\0' || p[n] == ';' || p[n] == ':')) return true;
    char const* sep = strchr(p, ';');
    if (sep == nullptr) break;
    p = sep + 1;
  }
  return false;
}
