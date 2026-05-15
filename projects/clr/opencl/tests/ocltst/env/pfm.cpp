/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
