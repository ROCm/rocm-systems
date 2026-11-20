/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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


// Linux regression test: when the VA driver loads and responds but exposes no decode profile, rocDecGetDecoderCaps
// must return ROCDEC_NOT_INITIALIZED rather than ROCDEC_SUCCESS with is_supported = 0, so a broken VA stack cannot be
// mistaken for a GPU without a decoder.
//
// The test points libva at a stub driver (stub_drv_video.cpp) that advertises only VAProfileNone -- the profile drivers
// list for the video post-processing entrypoint -- and checks every codec, plus a repeated call, against it.

#include <dlfcn.h>

#include <cstdlib>
#include <iostream>
#include <string>

#include "rocdecode/rocdecode.h"

namespace {

int failures = 0;

void Check(bool condition, const std::string &message) {
    std::cout << (condition ? "PASS: " : "FAIL: ") << message << std::endl;
    if (!condition) {
        failures++;
    }
}

// Reports which librocdecode the process loaded, so a failure caused by LD_LIBRARY_PATH picking up a different build
// is recognizable from the test output alone.
std::string LoadedLibraryPath() {
    Dl_info info = {};
    if (dladdr(reinterpret_cast<void *>(&rocDecGetDecoderCaps), &info) == 0 || info.dli_fname == nullptr) {
        return "<unknown>";
    }
    return info.dli_fname;
}

}  // namespace

int main() {
    // libva reads these when the VA display is first initialized, which happens on the first rocDecode call below.
    if (setenv("LIBVA_DRIVERS_PATH", STUB_DRIVER_DIR, 1) != 0 || setenv("LIBVA_DRIVER_NAME", STUB_DRIVER_NAME, 1) != 0) {
        std::cerr << "Error: failed to point libva at the stub driver" << std::endl;
        return EXIT_FAILURE;
    }

    const struct {
        rocDecVideoCodec codec;
        const char *name;
    } codecs[] = {
        {rocDecVideoCodec_HEVC, "HEVC"},
        {rocDecVideoCodec_AVC, "AVC"},
        {rocDecVideoCodec_AV1, "AV1"},
        {rocDecVideoCodec_VP9, "VP9"},
    };
    // Query twice: the second pass is answered from the cached VA context rather than a fresh probe.
    for (int pass = 1; pass <= 2; pass++) {
        for (const auto &c : codecs) {
            RocdecDecodeCaps decode_caps = {};
            decode_caps.device_id = 0;
            decode_caps.codec_type = c.codec;
            decode_caps.chroma_format = rocDecVideoChromaFormat_420;
            decode_caps.bit_depth_minus_8 = 0;
            decode_caps.is_supported = 1;
            rocDecStatus status = rocDecGetDecoderCaps(&decode_caps);
            std::string tag = std::string(" for ") + c.name + " (pass " + std::to_string(pass) + ")";
            Check(status == ROCDEC_NOT_INITIALIZED,
                  "rocDecGetDecoderCaps returns ROCDEC_NOT_INITIALIZED" + tag + ", got " + rocDecGetErrorName(status) +
                  " (" + std::to_string(status) + ")");
            Check(decode_caps.is_supported == 0, "is_supported is cleared" + tag);
        }
    }

    if (failures) {
        std::cerr << failures << " check(s) failed; librocdecode loaded from " << LoadedLibraryPath() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "All checks passed" << std::endl;
    return EXIT_SUCCESS;
}
