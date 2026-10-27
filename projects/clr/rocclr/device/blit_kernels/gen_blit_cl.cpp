/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Writes the OpenCL C source of the blit program from the strings the runtime compiles
// (blitcl.cpp, rocblitcl.cpp), so the kernels built offline and at runtime come from one source.
// All kernels go into one program: the linear blits, the HIP stream ops and heap kernels, the
// OpenCL device enqueue scheduler, and on request the GWS init kernel and the image blits (the
// runtime leaves them out where GWS or images aren't supported).
//
// Usage: gen_blit_cl <out.cl> [gws] [images]

#include <cstdio>
#include <cstring>

namespace amd::device {
extern const char* BlitLinearSourceCode;
extern const char* BlitImageSourceCode;
extern const char* HipExtraSourceCode;
extern const char* HipExtraSourceCodeNoGWS;
}  // namespace amd::device

namespace amd::roc {
extern const char* SchedulerSourceCode;
}  // namespace amd::roc

int main(int argc, char** argv) {
  bool gws = false;
  bool images = false;
  bool ok = (argc >= 2);
  for (int i = 2; ok && i < argc; ++i) {
    if (std::strcmp(argv[i], "gws") == 0) {
      gws = true;
    } else if (std::strcmp(argv[i], "images") == 0) {
      images = true;
    } else {
      ok = false;
    }
  }
  if (!ok) {
    std::fprintf(stderr, "Usage: %s <out.cl> [gws] [images]\n", argv[0]);
    return 1;
  }
  std::FILE* out = std::fopen(argv[1], "w");
  if (out == nullptr) {
    std::perror(argv[1]);
    return 1;
  }
  const char* parts[] = {amd::device::BlitLinearSourceCode,
                         images ? amd::device::BlitImageSourceCode : "",
                         gws ? amd::device::HipExtraSourceCode
                             : amd::device::HipExtraSourceCodeNoGWS,
                         amd::roc::SchedulerSourceCode};
  for (const char* part : parts) {
    std::fputs(part, out);
    std::fputs("\n", out);
  }
  return std::fclose(out) == 0 ? 0 : 1;
}
