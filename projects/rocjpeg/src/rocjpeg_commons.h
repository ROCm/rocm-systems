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

#ifndef ROC_JPEG_COMMON_H_
#define ROC_JPEG_COMMON_H_

#pragma once
#include <stdexcept>
#include <exception>
#include <string>
#include <iostream>
#include <cstring>

#define TOSTR(X) std::to_string(static_cast<int>(X))
#define STR(X) std::string(X)

#if DBGINFO
#define INFO(X) std::clog << "[INF] " << " {" << __func__ <<"} " << " " << X << std::endl;
#else
#define INFO(X) ;
#endif
#define ERR(X) std::cerr << "[ERR] "  << " {" << __func__ <<"} " << " " << X << std::endl;

#define CHECK_VAAPI(call) {                                               \
    VAStatus va_status = (call);                                          \
    if (va_status != VA_STATUS_SUCCESS) {                                 \
        std::cerr << "VAAPI failure: " << #call << " failed with status: " << std::hex << "0x" << va_status << std::dec << " = '" << vaErrorStr(va_status) << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
        return ROCJPEG_STATUS_EXECUTION_FAILED;                           \
    }                                                                     \
}

#define CHECK_HIP(call) {                                             \
    hipError_t hip_status = (call);                                   \
    if (hip_status != hipSuccess) {                                   \
        std::cerr << "HIP failure: 'status: " << hipGetErrorName(hip_status) << "' at " << __FILE__ << ":" << __LINE__ << std::endl;\
        return ROCJPEG_STATUS_EXECUTION_FAILED;                       \
    }                                                                 \
}

#define CHECK_ROCJPEG(call) {                                             \
    RocJpegStatus rocjpeg_status = (call);                                \
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {                       \
        std::cerr << #call << " returned " << rocJpegGetErrorName(rocjpeg_status) << " at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
        return rocjpeg_status;                                                          \
    }                                                                     \
}

static inline int align(int value, int alignment) {
   return (value + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Custom exception class for RocJpeg.
 *
 * This exception class is used to handle errors and exceptions that occur during RocJpeg operations.
 * It inherits from the std::exception class and provides an implementation for the what() function.
 */
class RocJpegException : public std::exception {
    public:
        /**
         * @brief Constructs a RocJpegException object with the specified error message.
         *
         * @param message The error message associated with the exception.
         */
        explicit RocJpegException(const std::string& message):message_(message){}

        /**
         * @brief Returns a C-style string describing the exception.
         *
         * This function overrides the what() function from the std::exception class and returns
         * the error message associated with the exception.
         *
         * @return A C-style string describing the exception.
         */
        virtual const char* what() const throw() override {
            return message_.c_str();
        }

    private:
        std::string message_;
};

#define THROW(X) throw RocJpegException(" { "+std::string(__func__)+" } " + X);

#endif //ROC_JPEG_COMMON_H_