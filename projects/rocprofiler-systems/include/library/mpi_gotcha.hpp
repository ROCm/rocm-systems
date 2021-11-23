// Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include "library/common.hpp"
#include "library/timemory.hpp"

// this is used to wrap MPI_Init and MPI_Init_thread
struct mpi_gotcha : comp::base<mpi_gotcha, void>
{
    using gotcha_data_t = comp::gotcha_data;

    TIMEMORY_DEFAULT_OBJECT(mpi_gotcha)

    // this will get called right before MPI_Init with that functions arguments
    void audit(const gotcha_data_t& _data, audit::incoming, int*, char***);

    // this will get called right before MPI_Init_thread with that functions arguments
    void audit(const gotcha_data_t& _data, audit::incoming, int*, char***, int, int*);

    // this will get called right after MPI_Init and MPI_Init_thread with the return value
    void audit(const gotcha_data_t& _data, audit::outgoing, int _retval);

    // this will get called right before MPI_Finalize
    void audit(const gotcha_data_t& _data, audit::incoming);
};

using mpi_gotcha_t = comp::gotcha<4, tim::component_tuple<mpi_gotcha>, hosttrace>;
