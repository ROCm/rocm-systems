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

/*
ROCPROFILER_SDK_USE_SYSTEM_ROCDECODE is set to 0 due to issues with the
rocDecode cmake setup and header files. Once they are resolved, the first
if-condition should set the ROCPROFILER_SDK_USE_SYSTEM_ROCDECODE variable
to 1
*/
#if !defined(ROCPROFILER_SDK_USE_SYSTEM_ROCDECODE)
#    if defined __has_include
#        if __has_include(<rocdecode/rocdecode.h>)
#            define ROCPROFILER_SDK_USE_SYSTEM_ROCDECODE 0
#        else
#            define ROCPROFILER_SDK_USE_SYSTEM_ROCDECODE 0
#        endif
#    else
#        define ROCPROFILER_SDK_USE_SYSTEM_ROCDECODE 0
#    endif
#endif
