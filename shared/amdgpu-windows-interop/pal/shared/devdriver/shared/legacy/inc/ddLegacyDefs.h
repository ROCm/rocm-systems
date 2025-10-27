/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 **********************************************************************************************************************/

#pragma once

#ifndef DD_PLATFORM_WINDOWS_UM
    #if _WIN32 && !_KERNEL_MODE
        #define DD_PLATFORM_WINDOWS_UM 1
        #define DD_PLATFORM_IS_UM      1
    #endif
#endif

#ifndef DD_PLATFORM_WINDOWS_KM
    #if _WIN32 && _KERNEL_MODE
        #define DD_PLATFORM_WINDOWS_KM 1
        #define DD_PLATFORM_IS_KM      1
    #endif
#endif

#ifndef DD_PLATFORM_LINUX_UM
    #ifdef __linux__
        #define DD_PLATFORM_LINUX_UM 1
        #define DD_PLATFORM_IS_UM    1
        #define DD_PLATFORM_IS_GNU   1
    #endif
#endif
