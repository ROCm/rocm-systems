// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <mpi.h>

#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace
{
auto _name = std::string{};
}  // namespace

template <typename Tp>
auto
get_values_str(const Tp& _data)
{
    std::stringstream _ss{};
    for(auto&& itr : _data)
        _ss << ", " << std::setw(6) << std::setprecision(2) << std::fixed << itr;
    return _ss.str().substr(1);
}

template <typename Tp, size_t N>
auto
get_dist(std::mt19937_64& _mt)
{
    static auto _dist = []() {
        if constexpr(std::is_integral<Tp>::value)
        {
            return std::uniform_int_distribution<Tp>(1, N * N);
        }
        else
        {
            return std::uniform_real_distribution<Tp>(1.0, N * N);
        }
    }();
    return _dist(_mt);
}

template <typename Tp>
auto
get_dtype()
{
    auto _dtype = MPI_INT;  // NOLINT
    if(std::is_same<Tp, long>::value)
        _dtype = MPI_LONG;
    else if(std::is_same<Tp, float>::value)
        _dtype = MPI_FLOAT;
    else if(std::is_same<Tp, double>::value)
        _dtype = MPI_DOUBLE;
    return _dtype;
}

template <typename Tp, size_t N>
void
all2all(int _rank, MPI_Comm _comm)
{
    if(_comm == MPI_COMM_NULL) return;
    static_assert(N > 0, "Error! N must be greater than zero!");

    auto _dtype      = get_dtype<Tp>();
    auto _mt         = std::mt19937_64{ size_t(_rank + 100) };
    auto values_sent = std::array<Tp, N>{};
    auto values_recv = std::array<Tp, N>{};
    for(size_t i = 0; i < N; ++i)
        values_sent[i] = get_dist<Tp, N>(_mt);

    if(_rank == 0)
        printf("[%s][%s][%2i] values sent (# = %zu) :: %s.\n", _name.c_str(),
               __FUNCTION__, _rank, values_sent.size(),
               get_values_str(values_sent).c_str());

    MPI_Alltoall(&values_sent[_rank], 1, _dtype, &values_recv[_rank], 1, _dtype, _comm);

    if(_rank == 0)
        printf("[%s][%s][%2i] values recv (# = %zu) :: %s.\n", _name.c_str(),
               __FUNCTION__, _rank, values_sent.size(),
               get_values_str(values_recv).c_str());
}

template <typename Tp, size_t N>
void
send_recv(int _rank, MPI_Comm _comm)
{
    if(_comm == MPI_COMM_NULL) return;
    static_assert(N > 0, "Error! N must be greater than zero!");
    int _size = 0;
    MPI_Comm_size(_comm, &_size);

    auto _dtype      = get_dtype<Tp>();
    auto _mt         = std::mt19937_64{ size_t(_rank + 100) };
    auto values_sent = std::array<Tp, N>{};
    auto values_recv = std::array<Tp, N>{};
    for(size_t i = 0; i < N; ++i)
        values_sent[i] = get_dist<Tp, N>(_mt);

    if(_rank == 0 || _rank == _size - 1)
        printf("[%s][%s][%2i] values sent (# = %zu) :: %s.\n", _name.c_str(),
               __FUNCTION__, _rank, values_sent.size(),
               get_values_str(values_sent).c_str());

    for(int i = 0; i < _size; ++i)
    {
        if(i != _rank) MPI_Send(&values_sent[_rank], 1, _dtype, i, N, _comm);
    }

    for(int i = 0; i < _size; ++i)
    {
        if(i != _rank)
        {
            MPI_Status _status;
            MPI_Recv(&values_recv[i], 1, _dtype, i, N, _comm, &_status);
        }
    }

    if(_rank == 0 || _rank == _size - 1)
        printf("[%s][%s][%2i] values recv (# = %zu) :: %s.\n", _name.c_str(),
               __FUNCTION__, _rank, values_sent.size(),
               get_values_str(values_recv).c_str());
}

void
run(MPI_Comm _comm, int nitr)
{
    if(_comm == MPI_COMM_NULL) return;
    int _rank = 0;
    int _size = 0;
    MPI_Comm_rank(_comm, &_rank);
    MPI_Comm_size(_comm, &_size);

    printf("[%s][%s][%2i] running %i iterations on %i ranks... \n", _name.c_str(),
           __FUNCTION__, _rank, nitr, _size);

    MPI_Barrier(_comm);
    for(int i = 0; i < nitr; ++i)
    {
        send_recv<int, 3>(_rank, _comm);
        send_recv<long, 4>(_rank, _comm);
        send_recv<float, 5>(_rank, _comm);
        send_recv<double, 6>(_rank, _comm);
        MPI_Barrier(_comm);
        all2all<int, 3>(_rank, _comm);
        all2all<long, 4>(_rank, _comm);
        all2all<float, 5>(_rank, _comm);
        all2all<double, 6>(_rank, _comm);
    }
    MPI_Barrier(_comm);

    printf("[%s][%s][%2i] running %i iterations on %i ranks... Done\n", _name.c_str(),
           __FUNCTION__, _rank, nitr, _size);
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

void
run_main(int argc, char** argv)
{
    int rank = 0;
    int size = 1;
    int nitr = 1;

    if(argc > 1) nitr = atoi(argv[1]);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    _name     = argv[0];
    auto _pos = _name.find_last_of('/');
    if(_pos < _name.length()) _name = _name.substr(_pos + 1);

    printf("[%s] Number of iterations: %i\n", _name.c_str(), nitr);

    printf("[%s][%2i] running with MPI_COMM_WORLD...\n", _name.c_str(), getpid());
    run(MPI_COMM_WORLD, nitr);

    print_info(MPI_COMM_WORLD, true, "MPI_COMM_WORLD");

    if(size > 1)
    {
        MPI_Comm dup;
        printf("[%s][%2i] Duplicating MPI_COMM_WORLD...\n", _name.c_str(), getpid());
        MPI_Comm_dup(MPI_COMM_WORLD, &dup);

        printf("[%s][%2i] running with duplicated comm of MPI_COMM_WORLD...\n",
               _name.c_str(), getpid());
        run(dup, nitr);

        MPI_Comm_rank(dup, &rank);
        if(rank == 0) printf("[%s]\n", _name.c_str());
        printf("[%s][%2i] RANK = %i on duplicated MPI_COMM_WORLD...\n", _name.c_str(),
               getpid(), rank);

        if(size > 3)
        {
            std::vector<MPI_Comm> comms(3);
            for(int i = 0; i < size; ++i)
            {
                auto _idx = i % 3;
                printf(
                    "[%s][%2i] Splitting duplicated MPI_COMM_WORLD %i (rank = %i)...\n",
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
                printf("[%s][%2i] Running on split communicator %i (rank = %i)...\n",
                       _name.c_str(), getpid(), _idx, _rank);
                run(comms.at(_idx), nitr);
            }

            // Get the group of processes in MPI_COMM_WORLD
            MPI_Group world_group;
            MPI_Comm_group(MPI_COMM_WORLD, &world_group);

            int       n        = 0;
            const int ranks[7] = { 1, 2, 3, 5, 7, 11, 13 };
            for(int r : ranks)
                if(r < size) ++n;

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

    printf("[%s][%i of %i] %s... Done\n", _name.c_str(), rank, size, __FUNCTION__);
}

int
main(int argc, char** argv)
{
    std::this_thread::sleep_for(std::chrono::seconds{ 2 });
    int _mpi_thread_requested = MPI_THREAD_SERIALIZED;
    int _mpi_thread_provided  = 0;
    MPI_Init_thread(&argc, &argv, _mpi_thread_requested, &_mpi_thread_provided);

    if(_mpi_thread_provided != _mpi_thread_requested)
        throw std::runtime_error("Error! requested thread mode != provided thread mode");

    auto _prom = std::promise<void>{};
    auto _fut  = _prom.get_future();

    std::thread{ [&]() {
        _prom.set_value_at_thread_exit();
        run_main(argc, argv);
    } }.join();

    _fut.wait();

    MPI_Finalize();
    return EXIT_SUCCESS;
}
