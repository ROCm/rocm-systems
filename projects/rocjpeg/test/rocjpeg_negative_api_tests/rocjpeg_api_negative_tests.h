/*
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef ROCJPEG_API_NEGATIVE_TESTS_H
#define ROCJPEG_API_NEGATIVE_TESTS_H

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <rocjpeg/rocjpeg.h>

/**
 * @class RocJpegApiNegativeTests
 * @brief A class to perform negative API tests for the rocJPEG library.
 *
 * This class contains a set of test cases designed to validate the behavior
 * of the rocJPEG library when invalid or unexpected inputs are provided.
 * It ensures the robustness and error handling capabilities of the library.
 *
 * In addition to the hand-written negative scenarios, the class hosts a fuzz
 * target for rocJpegStreamParse (TestStreamParseFuzz). The fuzz target derives
 * malformed streams from embedded valid JPEGs and checks two properties on
 * every one of them: the parser must not read out of bounds (observable under
 * a sanitizer build), and a stream the parser accepts must actually describe a
 * decodable image. The second property is what catches over-acceptance, which
 * a corpus of valid images alone can never detect.
 *
 * The class also hosts a two-GPU regression test (TestCrossDeviceDecode). It
 * checks that a decode is finished on the handle's device whatever the calling
 * thread's current device is, and that the thread's current device is left
 * unchanged. It is skipped unless at least two GPUs are visible. A handle on
 * the last device is used from threads left on device 0, and a handle on
 * device 0 from threads on the last device. Both directions run because rocJPEG
 * cannot decode on every GPU that HIP lists: with a discrete GPU and an
 * integrated GPU that rocJPEG has no kernels for, only the direction with the
 * handle on the discrete GPU tests anything, and that GPU may be the first or
 * the last device. A direction whose handle device cannot decode is skipped.
 */
class RocJpegApiNegativeTests {
    public:
        RocJpegApiNegativeTests();
        ~RocJpegApiNegativeTests();
        int RunTests();
    private:
        int TestInvalidStreamCreate();
        int TestInvalidStreamParse();
        int TestInvalidStreamDestroy();
        int TestInvalidCreate();
        int TestInvalidDestroy();
        int TestInvalidGetImageInfo();
        int TestInvalidDecode();
        int TestInvalidDecodeBatched();
        int TestInvalidGetErrorName();
        int TestInvalidGetDecoderCaps();
        int TestStreamParseFuzz();
        int TestCrossDeviceDecode();

        /**
         * @brief Parses a stream out of an exactly sized heap allocation.
         *
         * Copying the case into a buffer whose allocation is exactly as large as
         * the stream puts a sanitizer redzone immediately after the final byte,
         * so a one-byte overread is trapped instead of landing in slack space.
         *
         * @param data The bytes to hand to rocJpegStreamParse.
         * @return The status reported by rocJpegStreamParse.
         */
        RocJpegStatus ParseExactBuffer(const std::vector<uint8_t> &data);

        /**
         * @brief Parses a fuzz case and verifies the parser's postconditions.
         *
         * Any status is acceptable as long as it is a documented one. When the
         * stream is accepted, the resulting image description is required to be
         * usable: a supported component count, non-zero dimensions and a known
         * chroma subsampling. The image description is only inspected when
         * check_image_info_ is set, since reading it needs a decoder handle.
         *
         * @param data The bytes to parse.
         * @param case_name A label printed alongside the offending bytes on failure.
         * @return EXIT_SUCCESS when every postcondition holds, EXIT_FAILURE otherwise.
         */
        int CheckParseInvariants(const std::vector<uint8_t> &data, const std::string &case_name);

        /**
         * @brief Decodes with handles on one GPU from threads on another.
         *
         * A 64x64 JPEG is first decoded to RGB with handle_device current, as the
         * reference. It is then decoded with rocJpegDecode and with
         * rocJpegDecodeBatched, each from a new thread whose current device is
         * caller_device, and with rocJpegDecodeAsync and rocJpegDecodeBatchedAsync
         * submitted from this thread and synchronized by rocJpegDecodeSync and
         * rocJpegDecodeBatchedSync on such a thread. Every decode uses a handle of
         * its own, created on handle_device, and every output is compared with the
         * reference. If rocJPEG cannot decode on handle_device at all (creating
         * the handles, allocating the outputs or the reference decode fails, or
         * the reference is blank), the comparisons would test nothing, so this
         * direction is skipped instead.
         *
         * @param handle_device The device the decoder handles are created on.
         * @param caller_device The current device of the threads that decode.
         * @return EXIT_SUCCESS when every decode succeeds, matches the reference and
         *         leaves its thread on caller_device, or when the direction is
         *         skipped; EXIT_FAILURE otherwise.
         */
        int CheckCrossDeviceDecode(int handle_device, int caller_device);

        /**
         * @brief Destroys the decoder handle and clears the member.
         *
         * TestInvalidCreate destroys and recreates the handle several times over.
         * Clearing it is what keeps those rounds independent: rocJpegCreate only
         * writes through its out parameter once construction has succeeded, so a
         * failure that returns early - the throw path in rocJpegCreate, for one -
         * leaves whatever was there before, and the destructor would then hand a
         * freed pointer back to rocJpegDestroy. A cleared member turns that into
         * the documented null case instead of a double free.
         *
         * @return EXIT_SUCCESS when the destroy reported success.
         */
        int DestroyHandle();

        RocJpegHandle rocjpeg_handle_ = nullptr;
        RocJpegStreamHandle rocjpeg_stream_handle_ = nullptr;
        /**
         * @brief Whether the fuzz target can inspect parsed image descriptions.
         *
         * Reading them goes through rocJpegGetImageInfo, which needs a working
         * decoder handle. In the normal test flow TestInvalidCreate has already
         * established one; when the fuzz target runs without a usable device it
         * falls back to checking only the reported status.
         */
        bool check_image_info_ = false;
};

#endif // ROCJPEG_API_NEGATIVE_TESTS_H