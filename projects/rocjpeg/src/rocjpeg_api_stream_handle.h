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

#ifndef ROC_JPEG_STREAM_HANDLE_H
#define ROC_JPEG_STREAM_HANDLE_H

#pragma once

#include <memory>
#include "rocjpeg_parser.h"

/**
 * @brief The RocJpegStreamParserHandle class represents a handle to the RocJpegStreamParser object.
 *
 * This class provides a convenient way to manage the RocJpegStreamParser object by encapsulating it
 * within a shared pointer. It also provides error handling functionality.
 */
class RocJpegStreamParserHandle {
    public:
        /**
         * @brief Constructs a RocJpegStreamParserHandle object.
         *
         * This constructor initializes the rocjpeg_stream member with a new instance of RocJpegStreamParser
         * using std::make_shared.
         */
        explicit RocJpegStreamParserHandle() : rocjpeg_stream(std::make_shared<RocJpegStreamParser>()) {};

        /**
         * @brief Destroys the RocJpegStreamParserHandle object.
         *
         * This destructor clears any errors associated with the handle.
         */
        ~RocJpegStreamParserHandle() { ClearErrors(); }

        std::shared_ptr<RocJpegStreamParser> rocjpeg_stream; /**< The RocJpegStreamParser object. */

        /**
         * @brief Checks if there are no errors associated with the handle.
         * @return true if there are no errors, false otherwise.
         */
        bool NoError() { return error_.empty(); }

        /**
         * @brief Gets the error message associated with the handle.
         * @return The error message as a C-style string.
         */
        const char* ErrorMsg() { return error_.c_str(); }

        /**
         * @brief Captures an error message.
         * @param err_msg The error message to capture.
         */
        void CaptureError(const std::string& err_msg) { error_ = err_msg; }

    private:
        /**
         * @brief Clears any errors associated with the handle.
         */
        void ClearErrors() { error_ = "";}

        std::string error_; /**< The error message. */
};

#endif //ROC_JPEG_STREAM_HANDLE_H