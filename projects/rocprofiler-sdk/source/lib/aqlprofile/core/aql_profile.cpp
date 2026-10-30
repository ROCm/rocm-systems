// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/aqlprofile/core/pm4_factory.h"

#ifdef _WIN32
#    define DESTRUCTOR_API
#else
#    define DESTRUCTOR_API __attribute__((destructor))
#endif

namespace aql_profile
{
bool                     Pm4Factory::concurrent_create_mode_ = false;
Pm4Factory::mutex_t      Pm4Factory::mutex_;
Pm4Factory::instances_t* Pm4Factory::instances_ = nullptr;

DESTRUCTOR_API void
destructor()
{
    Pm4Factory::Destroy();
}

}  // namespace aql_profile

#ifdef _WIN32
#    include <windows.h>
extern "C" BOOL WINAPI
DllMain(HINSTANCE /*hinstDLL*/, DWORD fdwReason, LPVOID /*lpvReserved*/)
{
    if(fdwReason == DLL_PROCESS_DETACH)
    {
        aql_profile::destructor();
    }
    return TRUE;
}
#endif
