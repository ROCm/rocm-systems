/*
Copyright (c) 2015-2020 Advanced Micro Devices, Inc. All rights reserved.

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

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <type_traits>
#include <vector>

static std::mutex print_lock{};
using auto_lock_t = std::unique_lock<std::mutex>;

#include <mpi.h>

std::string _name = {};

template <typename Tp, size_t N>
void
all2all(int _rank)
{
    static_assert(N > 0, "Error! N must be greater than zero!");

    auto _mt   = std::mt19937_64{ size_t(_rank + 100) };
    auto _dist = []() {
        if constexpr(std::is_integral<Tp>::value)
        {
            return std::uniform_int_distribution<Tp>(1, N * N);
        }
        else
        {
            return std::uniform_real_distribution<Tp>(1.0, N * N);
        }
    }();

    auto _get_values_str = [](const auto& _data) {
        std::stringstream _ss{};
        for(auto&& itr : _data)
            _ss << ", " << std::setw(6) << std::setprecision(2) << std::fixed << itr;
        return _ss.str().substr(1);
    };

    std::array<Tp, N> values_sent = {};
    std::array<Tp, N> values_recv = {};
    for(size_t i = 0; i < N; ++i)
        values_sent[i] = _dist(_mt);

    if(_rank == 0)
        printf("[%s][%i] values sent (# = %zu) :: %s.\n", _name.c_str(), _rank,
               values_sent.size(), _get_values_str(values_sent).c_str());

    auto _dtype = MPI_INT;
    if(std::is_same<Tp, long>::value)
        _dtype = MPI_LONG;
    else if(std::is_same<Tp, float>::value)
        _dtype = MPI_FLOAT;
    else if(std::is_same<Tp, double>::value)
        _dtype = MPI_DOUBLE;

    MPI_Alltoall(&values_sent[_rank], 1, _dtype, &values_recv[_rank], 1, _dtype,
                 MPI_COMM_WORLD);

    if(_rank == 0)
        printf("[%s][%i] values recv (# = %zu) :: %s.\n", _name.c_str(), _rank,
               values_sent.size(), _get_values_str(values_recv).c_str());
}

int
main(int argc, char** argv)
{
    int rank = 0;
    int size = 1;
    int nitr = 1;
    if(argc > 1) nitr = atoi(argv[2]);

    _name     = argv[0];
    auto _pos = _name.find_last_of('/');
    if(_pos < _name.length()) _name = _name.substr(_pos + 1);

    printf("[%s] Number of iterations: %i\n", _name.c_str(), nitr);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    for(int i = 0; i < nitr; ++i)
    {
        all2all<int, 3>(rank);
        all2all<long, 4>(rank);
        MPI_Barrier(MPI_COMM_WORLD);
        all2all<float, 5>(rank);
        all2all<double, 6>(rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
