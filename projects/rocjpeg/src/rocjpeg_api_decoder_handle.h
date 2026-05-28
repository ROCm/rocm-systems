/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef ROC_JPEG_DECODER_HANDLE_H
#define ROC_JPEG_DECODER_HANDLE_H

#pragma once

#include "rocjpeg_decoder.h"

/**
 * @brief The RocJpegDecoderHandle class represents a handle to the RocJpegDecoder object.
 *
 * This class provides a convenient way to manage the lifetime of a RocJpegDecoder object.
 * It encapsulates the RocJpegDecoder object and provides error handling functionality.
 */
class RocJpegDecoderHandle {
public:
    /**
     * @brief Constructs a RocJpegDecoderHandle object with the specified backend and device ID.
     *
     * @param backend The backend to use for decoding.
     * @param device_id The ID of the device to use for decoding.
     */
    explicit RocJpegDecoderHandle(RocJpegBackend backend, int device_id) : rocjpeg_decoder(std::make_shared<RocJpegDecoder>(backend, device_id)) {};

    /**
     * @brief Destructor for the RocJpegDecoderHandle class.
     *
     * Clears any errors associated with the handle.
     */
    ~RocJpegDecoderHandle() { ClearErrors(); }

    /**
     * @brief The RocJpegDecoder object associated with the handle.
     */
    std::shared_ptr<RocJpegDecoder> rocjpeg_decoder;

    /**
     * @brief Checks if there are no errors associated with the handle.
     *
     * @return true if there are no errors, false otherwise.
     */
    bool NoError() { return error_.empty(); }

    /**
     * @brief Gets the error message associated with the handle.
     *
     * @return The error message as a C-style string.
     */
    const char* ErrorMsg() { return error_.c_str(); }

    /**
     * @brief Captures an error message for the handle.
     *
     * @param err_msg The error message to capture.
     */
    void CaptureError(const std::string& err_msg) { error_ = err_msg; }

private:
    /**
     * @brief Clears any errors associated with the handle.
     */
    void ClearErrors() { error_ = ""; }

    std::string error_;
};

#endif //ROC_JPEG_DECODER_HANDLE_H