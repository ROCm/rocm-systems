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

#include <mpi.h>

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
#include <sstream>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <vector>

std::string _name = {};

template <typename Tp, size_t N>
void
all2all(int _rank, MPI_Comm _comm)
{
    if(_comm == MPI_COMM_NULL) return;
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

    auto _dtype = MPI_INT;  // NOLINT
    if(std::is_same<Tp, long>::value)
        _dtype = MPI_LONG;
    else if(std::is_same<Tp, float>::value)
        _dtype = MPI_FLOAT;
    else if(std::is_same<Tp, double>::value)
        _dtype = MPI_DOUBLE;

    MPI_Alltoall(&values_sent[_rank], 1, _dtype, &values_recv[_rank], 1, _dtype, _comm);

    if(_rank == 0)
        printf("[%s][%i] values recv (# = %zu) :: %s.\n", _name.c_str(), _rank,
               values_sent.size(), _get_values_str(values_recv).c_str());
}

void
run(MPI_Comm _comm, int nitr)
{
    if(_comm == MPI_COMM_NULL) return;
    int _rank = 0;
    int _size = 0;
    MPI_Comm_rank(_comm, &_rank);
    MPI_Comm_size(_comm, &_size);

    printf("[%s][%i] running %i iterations on %i ranks...\n", _name.c_str(), _rank, nitr,
           _size);

    MPI_Barrier(_comm);
    for(int i = 0; i < nitr; ++i)
    {
        all2all<int, 3>(_rank, _comm);
        all2all<long, 4>(_rank, _comm);
        MPI_Barrier(_comm);
        all2all<float, 5>(_rank, _comm);
        all2all<double, 6>(_rank, _comm);
    }
    MPI_Barrier(_comm);
}

void
print_info(MPI_Comm _comm, bool _verbose, std::string _msg = {})
{
    if(_comm == MPI_COMM_NULL) return;
    int _rank = 0;
    int _size = 1;
    MPI_Comm_rank(_comm, &_rank);
    MPI_Comm_size(_comm, &_size);

    if(!_msg.empty()) _msg = "[" + _msg + "] ";

    if(_verbose)
    {
        auto              _ppid = getppid();
        std::ifstream     _ifs{ "/proc/" + std::to_string(_ppid) + "/task/" +
                            std::to_string(_ppid) + "/children" };
        std::stringstream _ss{};
        while(_ifs)
        {
            std::string _s{};
            _ifs >> _s;
            _ss << _s << " ";
        }
        if(_rank == 0)
            printf("[%s]%s RANK = %i (out of %i), PID = %i, PPID = %i :: %s\n",
                   _name.c_str(), _msg.c_str(), _rank, _size, getpid(), getppid(),
                   _ss.str().c_str());
    }
    else
    {
        if(_rank == 0)
            printf("[%s]%s RANK = %i (out of %i), PID = %i, PPID = %i\n", _name.c_str(),
                   _msg.c_str(), _rank, _size, getpid(), getppid());
    }
}

int
main(int argc, char** argv)
{
    int _mpi_thread_provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &_mpi_thread_provided);

    int rank = 0;
    int size = 1;
    int nitr = 1;

    if(argc > 1) nitr = atoi(argv[2]);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    _name     = argv[0];
    auto _pos = _name.find_last_of('/');
    if(_pos < _name.length()) _name = _name.substr(_pos + 1);

    printf("[%s] Number of iterations: %i\n", _name.c_str(), nitr);

    printf("[%s][%i] running with MPI_COMM_WORLD...\n", _name.c_str(), getpid());
    run(MPI_COMM_WORLD, nitr);

    print_info(MPI_COMM_WORLD, true, "MPI_COMM_WORLD");

    printf("[%s]\n", _name.c_str());

    if(size > 1)
    {
        MPI_Comm dup;
        printf("[%s][%i] Duplicating MPI_COMM_WORLD...\n", _name.c_str(), getpid());
        MPI_Comm_dup(MPI_COMM_WORLD, &dup);

        printf("[%s][%i] running with duplicated comm of MPI_COMM_WORLD...\n",
               _name.c_str(), getpid());
        run(dup, nitr);

        MPI_Comm_rank(dup, &rank);
        if(rank == 0) printf("[%s]\n", _name.c_str());
        printf("[%s][%i] RANK = %i on duplicated MPI_COMM_WORLD...\n", _name.c_str(),
               getpid(), rank);

        if(size > 3)
        {
            std::vector<MPI_Comm> comms(3);
            for(int i = 0; i < size; ++i)
            {
                auto _idx = i % 3;
                printf("[%s][%i] Splitting duplicated MPI_COMM_WORLD %i (rank = %i)...\n",
                       _name.c_str(), getpid(), _idx, rank);
                MPI_Comm* comm = &comms.at(_idx);
                MPI_Comm_split(dup, _idx, rank, comm);
            }

            for(auto itr : comms)  // NOLINT
                MPI_Barrier(itr);

            for(int i = 0; i < size; ++i)
            {
                auto _idx  = i % 3;
                int  _rank = 0;
                MPI_Comm_rank(comms.at(_idx), &_rank);
                printf("[%s][%i] Running on split communicator %i (rank = %i)...\n",
                       _name.c_str(), getpid(), _idx, _rank);
                run(comms.at(_idx), nitr);
            }

            // Get the group of processes in MPI_COMM_WORLD
            MPI_Group world_group;
            MPI_Comm_group(MPI_COMM_WORLD, &world_group);

            int       n        = 0;
            const int ranks[7] = { 1, 2, 3, 5, 7, 11, 13 };
            for(int rank : ranks)
                if(rank < size) ++n;

            // Construct a group containing all of the prime ranks in world_group
            MPI_Group prime_group;
            MPI_Group_incl(world_group, n, ranks, &prime_group);

            // Create a new communicator based on the group
            MPI_Comm prime_comm;
            MPI_Comm_create_group(MPI_COMM_WORLD, prime_group, 0, &prime_comm);

            MPI_Group nonprime_group;
            MPI_Group_difference(world_group, prime_group, &nonprime_group);

            MPI_Comm nonprime_comm;
            MPI_Comm_create_group(MPI_COMM_WORLD, nonprime_group, 1, &nonprime_comm);

            print_info(prime_comm, false, "Prime comm");
            print_info(nonprime_comm, false, "Non-prime comm");

            run(prime_comm, nitr);
            run(nonprime_comm, nitr);

            MPI_Group_free(&world_group);
            MPI_Group_free(&prime_group);
            MPI_Group_free(&nonprime_group);
        }

        print_info(dup, false);
    }

    MPI_Finalize();
    return 0;
}
