/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

#include "pfm.h"

#ifdef _WIN32
#include <io.h>
#endif

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

unsigned int SavePFM(const char* filename, const float* buffer, unsigned int width,
                     unsigned int height, unsigned int components) {
  unsigned int error = 0;

  //
  // open the image file for writing
  //
  FILE* fh;
  if ((fh = fopen(filename, "wb")) == NULL) {
    return 1;
  }

  //
  // write the PFM header
  //
#define PFMEOL "\x0a"
  fprintf(fh, "PF" PFMEOL "%d %d" PFMEOL "-1" PFMEOL, width, height);
  fflush(fh);

  //
  // write each scanline
  //
  const unsigned int lineSize = width * 3;
  float line[3 * 4096];
  for (unsigned int y = height; y > 0; y--) {
    const float* v = buffer + components * width * (y - 1);
    for (unsigned int x = 0; x < width; x++) {
      line[x * 3 + 0] = v[x * components + 0];
      line[x * 3 + 1] = (components > 1) ? v[x * components + 1] : v[x * components + 0];
      line[x * 3 + 2] = (components > 2) ? v[x * components + 2] : v[x * components + 0];
    }
    unsigned int written = (unsigned int)fwrite(line, (unsigned int)sizeof(float), lineSize, fh);
    if (written != lineSize) {
      error = 1;
      break;
    }
    fflush(fh);
  }
  fflush(fh);
  fclose(fh);

  return error;
}
