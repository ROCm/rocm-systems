// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ratio>
#include <string>
#include <thread>

#define NOINLINE __attribute__((noinline))

NOINLINE size_t
inner();

NOINLINE size_t
outer_a();

NOINLINE size_t
outer_b();

NOINLINE size_t
outer_c();

NOINLINE size_t
outer_d();

NOINLINE size_t
outer_e();

int
main(int argc, char** argv)
{
    int nrepeat = 1;
    if(argc > 1) nrepeat = atol(argv[1]);

    std::string _name = argv[0];
    auto        _pos  = _name.find_last_of('/');
    if(_pos != std::string::npos) _name = _name.substr(_pos + 1);

    size_t nitr = 0;
    for(int i = 0; i < nrepeat; ++i)
    {
        nitr += outer_a();
        nitr += outer_b();
        nitr += outer_c();
        nitr += outer_d();
        nitr += outer_e();
        printf("[%s][%i] number of calls made = %zu\n", _name.c_str(), i, nitr);
    }
}

size_t
inner(size_t _duration)
{
    static int64_t _n = 0;

    if(_n++ % 5 == 2)
    {
        using clock_type = std::chrono::high_resolution_clock;
        auto   _end      = clock_type::now() + std::chrono::milliseconds{ _duration };
        size_t nitr      = 0;
        while(clock_type::now() < _end)
        {
            ++nitr;
        }
        return nitr;
    }
    else
    {
        std::this_thread::sleep_for(std::chrono::milliseconds{ _duration });
        return 1;
    }
}

#define OUTER_FUNCTION(TAG)                                                              \
    size_t outer_##TAG() { return inner(500); }

OUTER_FUNCTION(a)
OUTER_FUNCTION(b)
OUTER_FUNCTION(c)
OUTER_FUNCTION(d)
OUTER_FUNCTION(e)
