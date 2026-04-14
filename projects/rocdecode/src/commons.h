/*
Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once
#include <stdexcept>
#include <exception>
#include <string>

#include "rocdecode/logger.h"

#if DBGINFO
#define MSG(X) std::clog << X << std::endl;
#define MSG_NO_NEWLINE(X) std::clog << X;
#else
#define MSG(X) ;
#define MSG_NO_NEWLINE(X) ;
#endif

class rocDecodeException : public std::exception {
public:

    explicit rocDecodeException(const std::string& OutputMsg):_message(OutputMsg){}
    virtual const char* what() const throw() override {
        return _message.c_str();
    }
private:
    std::string _message;
};

#define THROW(X) throw rocDecodeException(" { "+std::string(__func__)+" } " + X);
