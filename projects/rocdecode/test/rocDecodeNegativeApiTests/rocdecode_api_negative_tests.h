/*
Copyright (c) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef ROCDECODE_API_NEGATIVE_TESTS_H
#define ROCDECODE_API_NEGATIVE_TESTS_H

#include <iostream>
#include <rocdecode/rocdecode.h>
#include <rocdecode/rocparser.h>
#include <rocdecode/roc_bitstream_reader.h>

/**
 * @class RocDecodeApiNegativeTests
 * @brief A class to perform negative API tests for the rocDecode library.
 *
 * This class contains a set of test cases designed to validate the behavior
 * of the RocDecode library when invalid or unexpected inputs are provided.
 * It ensures the robustness and error handling capabilities of the library.
 */
class RocDecodeApiNegativeTests {
    public:
        RocDecodeApiNegativeTests();
        ~RocDecodeApiNegativeTests();
        int RunTests();
    private:
        int TestInvalidCreateDecoder();
        int TestInvalidDestroyDecoder();
        int TestInvalidGetDecoderCaps();
        int TestInvalidDecodeFrame();
        int TestInvalidGetDecodeStatus();
        int TestInvalidReconfigureDecoder();
        int TestinvalidGetVideoFrame();
        int TestinvalidGetErrorName();
        int TestinvalidCreateVideoParser();
        int TestinvalidParseVideoData();
        int TestinvalidDestroyVideoParser();
        int TestinvalidCreateBitstreamReader();
        int TestinvalidGetBitstreamCodecType();
        int TestinvalidGetBitstreamBitDepth();
        int TestinvalidGetBitstreamPicData();
        int TestinvalidDestroyBitstreamReader();
        rocDecDecoderHandle decoder_handle_;
        RocDecoderCreateInfo decoder_create_info_;
        RocdecVideoParser parser_handle_;
};

#endif // ROCDECODE_API_NEGATIVE_TESTS_H