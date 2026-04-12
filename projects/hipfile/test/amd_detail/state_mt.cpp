/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// Multithreading stress test for hipFile's global state
//
// Just run the program. It takes no special arguments.

#include "context.h"
#include "file.h"
#include "hipfile.h"
#include "state.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <sys/resource.h>
#include <sys/stat.h>
#include <vector>

using namespace hipFile;

using namespace std;

// TODO: Use getopt() to take N_THREADS and RUNTIME_S from the
//       command line
constexpr size_t N_THREADS = 10;
constexpr int    RUNTIME_S = 10;

constexpr size_t ALLOC_SIZE = 1024;

static mutex cout_mutex;

// Tracks how many files the process can open before it will hit resource limits
static atomic<unsigned long> n_free_files{0};

static atomic<bool> run_flag{true};

static int
make_temp_file()
{
    // Create a unique temporary file
    char pathname[]{"state_mt_tmp.XXXXXX"};
    int  fd{mkstemp(pathname)};
    if (-1 == fd) {
        cerr << "mkstemp() failed! " << strerror(errno) << endl;
        exit(EXIT_FAILURE);
    }

    // Ensure the file is deleted when fd is closed
    if (-1 == unlink(pathname)) {
        cerr << "unlink() failed! " << strerror(errno) << endl;
        exit(EXIT_FAILURE);
    }

    return fd;
}

void thread_function(int);

void
thread_function(int id)
{
    constexpr int N_CYCLES  = 100; // # of cycles before checking the run flag
    constexpr int N_PRELOAD = 10;  // # of files/buffers to load before cycling

    auto *ds = Context<DriverState>::get();

    vector<pair<int, hipFileHandle_t>> files;
    vector<void *>                     buffers;
    vector<hipStream_t>                hip_streams;

    // Preload with data
    for (int i{0}; i < N_PRELOAD; i++) {
        unsigned long nff = n_free_files.load();
        while (0 < nff) {
            if (n_free_files.compare_exchange_weak(nff, nff - 1)) {
                int fd = make_temp_file();
                files.push_back({fd, ds->registerFile(fd)});
                break;
            }
        }

        void      *buf = nullptr;
        hipError_t err = hipMalloc(&buf, ALLOC_SIZE);
        if (err != hipSuccess) {
            cerr << "hipMalloc() failed!" << endl;
            exit(EXIT_FAILURE);
        }

        buffers.push_back(buf);
        ds->registerBuffer(buf, ALLOC_SIZE, 0);

        hipStream_t hip_stream;
        err = hipStreamCreateWithFlags(&hip_stream, hipStreamNonBlocking);
        if (err != hipSuccess) {
            cerr << "hipStreamCreateWithFlags() failed!" << endl;
            exit(EXIT_FAILURE);
        }

        hip_streams.push_back(hip_stream);
        ds->registerStream(hip_stream, 0);
    }

    // Random number setup
    random_device rd;
    mt19937       gen{rd()};

    // Distribution for the operation selection
    uniform_int_distribution<int> op_dist{1, 12};

    while (run_flag) {
        for (int i{0}; i < N_CYCLES; i++) {
            // Get a random number in range of ops
            int choice = op_dist(gen);

            switch (choice) {
                // Case 1: Register a new file
                case 1: {
                    unsigned long nff = n_free_files.load();
                    while (0 < nff) {
                        if (n_free_files.compare_exchange_weak(nff, nff - 1)) {
                            int fd = make_temp_file();
                            files.push_back({fd, ds->registerFile(fd)});
                            break;
                        }
                    }
                } break;

                // Case 2: Deregister a random file
                case 2: {
                    if (files.size() == 0) {
                        break;
                    }
                    uniform_int_distribution<size_t> vec_dist{0, files.size() - 1};
                    size_t                           idx = vec_dist(gen);

                    auto [fd, handle] = files[idx];
                    ds->deregisterFile(handle);
                    if (-1 == close(fd)) {
                        cerr << "close() failed! " << strerror(errno) << endl;
                        exit(EXIT_FAILURE);
                    }
                    n_free_files++;

                    swap(files[idx], files.back());
                    files.pop_back();
                } break;

                // Case 3: Register a new buffer
                case 3: {
                    void      *buf = nullptr;
                    hipError_t err = hipMalloc(&buf, ALLOC_SIZE);
                    if (err != hipSuccess) {
                        cerr << "hipMalloc() failed!" << endl;
                        exit(EXIT_FAILURE);
                    }

                    buffers.push_back(buf);
                    ds->registerBuffer(buf, ALLOC_SIZE, 0);
                } break;

                // Case 4: Deregister a random buffer
                case 4: {
                    if (buffers.size() == 0) {
                        break;
                    }
                    uniform_int_distribution<size_t> vec_dist{0, buffers.size() - 1};
                    size_t                           idx = vec_dist(gen);

                    ds->deregisterBuffer(buffers[idx]);
                    hipError_t err = hipFree(buffers[idx]);
                    if (err != hipSuccess) {
                        cerr << "hipFree() failed!" << endl;
                        exit(EXIT_FAILURE);
                    }

                    swap(buffers[idx], buffers.back());
                    buffers.pop_back();
                } break;

                // Case 5: Register a new stream
                case 5: {
                    hipStream_t hip_stream;
                    hipError_t  err = hipStreamCreateWithFlags(&hip_stream, hipStreamNonBlocking);
                    if (err != hipSuccess) {
                        cerr << "hipStreamCreateWithFlags() failed!" << endl;
                        exit(EXIT_FAILURE);
                    }

                    hip_streams.push_back(hip_stream);
                    ds->registerStream(hip_stream, 0);
                } break;

                // Case 6: Deregister a random stream
                case 6: {
                    if (hip_streams.empty()) {
                        break;
                    }
                    uniform_int_distribution<size_t> vec_dist{0, hip_streams.size() - 1};
                    size_t                           idx = vec_dist(gen);

                    ds->deregisterStream(hip_streams[idx]);
                    hipError_t err = hipStreamDestroy(hip_streams[idx]);
                    if (err != hipSuccess) {
                        cerr << "hipStreamDestroy() failed!" << endl;
                        exit(EXIT_FAILURE);
                    }

                    swap(hip_streams[idx], hip_streams.back());
                    hip_streams.pop_back();
                } break;

                // Case 7: Get a random file
                case 7: {
                    if (files.size() == 0) {
                        break;
                    }
                    uniform_int_distribution<size_t> vec_dist{0, files.size() - 1};
                    size_t                           idx = vec_dist(gen);

                    auto data = ds->getFile(files[idx].second);

                    // TODO: Maintain a data collection to ensure there are no races
                    //       on the data
                } break;

                // Case 8: Get a random buffer (1)
                case 8: {
                    if (buffers.size() == 0) {
                        break;
                    }
                    uniform_int_distribution<size_t> vec_dist{0, buffers.size() - 1};
                    size_t                           idx = vec_dist(gen);

                    auto data = ds->getRegisteredBuffer(buffers[idx]);

                    // TODO: Maintain a data collection to ensure there are no races
                    //       on the data
                } break;

                // Case 9: Get a random buffer (2)
                case 9: {
                    if (buffers.size() == 0) {
                        break;
                    }
                    uniform_int_distribution<size_t> vec_dist{0, buffers.size() - 1};
                    size_t                           idx = vec_dist(gen);

                    auto data = ds->getBuffer(buffers[idx]);

                    // TODO: Maintain a data collection to ensure there are no races
                    //       on the data
                } break;

                // Case 10: Get a random stream
                case 10: {
                    if (hip_streams.empty()) {
                        break;
                    }
                    uniform_int_distribution<size_t> vec_dist{0, hip_streams.size() - 1};
                    size_t                           idx = vec_dist(gen);

                    auto stream = ds->getStream(hip_streams[idx]);

                } break;

                // Case 11: Get a file and a buffer
                case 11: {
                    if (buffers.size() == 0 || files.size() == 0) {
                        break;
                    }

                    size_t idx;

                    uniform_int_distribution<size_t> file_dist{0, files.size() - 1};
                    idx     = file_dist(gen);
                    auto fh = files[idx].second;

                    uniform_int_distribution<size_t> buf_dist{0, buffers.size() - 1};
                    idx      = buf_dist(gen);
                    auto buf = buffers[idx];

                    auto [file, buffer] = ds->getFileAndBuffer(fh, buf);

                    // TODO: Maintain a data collection to ensure there are no races
                    //       on the data
                } break;

                // Case 12: Get a file, buffer, and a stream
                //
                case 12: {
                    if (buffers.empty() || files.empty() || hip_streams.empty()) {
                        break;
                    }

                    size_t idx;

                    uniform_int_distribution<size_t> file_dist{0, files.size() - 1};
                    idx     = file_dist(gen);
                    auto fh = files[idx].second;

                    uniform_int_distribution<size_t> buf_dist{0, buffers.size() - 1};
                    idx      = buf_dist(gen);
                    auto buf = buffers[idx];

                    uniform_int_distribution<size_t> stream_dist{0, hip_streams.size() - 1};
                    idx             = stream_dist(gen);
                    auto hip_stream = hip_streams[idx];

                    auto [file, buffer, stream] = ds->getFileBufferAndStream(fh, buf, hip_stream);
                } break;

                default:
                    cout << "ERROR: Tried to pick a non-existent case" << endl;
                    exit(EXIT_FAILURE);
            }
        }
    }

    auto n_files       = files.size();
    auto n_buffers     = buffers.size();
    auto n_hip_streams = hip_streams.size();

    for (auto [fd, fh] : files) {
        ds->deregisterFile(fh);
        if (-1 == close(fd)) {
            cerr << "close() failed! " << strerror(errno) << endl;
        }
    }

    for (auto b : buffers) {
        ds->deregisterBuffer(b);
        hipError_t err = hipFree(b);
        if (err != hipSuccess) {
            cerr << "hipFree() failed!" << endl;
            exit(EXIT_FAILURE);
        }
    }

    for (auto s : hip_streams) {
        ds->deregisterStream(s);
        hipError_t err = hipStreamDestroy(s);
        if (err != hipSuccess) {
            cerr << "hipStreamDestroy() failed!" << endl;
            exit(EXIT_FAILURE);
        }
    }

    cout_mutex.lock();
    cout << "Thread " << id << ": DONE (final sizes: ";
    cout << to_string(n_files) << " files, ";
    cout << to_string(n_buffers) << " buffers, ";
    cout << to_string(n_hip_streams) << " hip streams";
    cout << ")" << endl;
    cout_mutex.unlock();
}

int
main()
{
    // Security analyzers will be sad if you don't set umask 0077
    // before creating temporary files
    //
    // Doing this here to avoid inadvertently synchronizing files if we
    // added a mutex in the temp file function
    //
    // Ignoring return type since we never switch it back
    (void)umask(S_IRWXG | S_IRWXO);

    struct rlimit nofile;
    if (-1 == getrlimit(RLIMIT_NOFILE, &nofile)) {
        cerr << "getrlimit() failed! " << strerror(errno) << endl;
        exit(EXIT_FAILURE);
    }

    // The process will already have a few open files. Reduce the maximum
    // number of open files by this amount.
    const unsigned n_file_reserved = 8;

    // Each thread should be able to create about 10 files
    const unsigned n_files_per_thread = 10;

    // hipfile makes an additional file descriptor when a file is registered.
    // Ensure that we do not exceed the allowed number of open file descriptors
    unsigned n_fds_per_registered_file = 2;
    if (nofile.rlim_cur <= n_file_reserved + (N_THREADS * n_files_per_thread * n_fds_per_registered_file)) {
        cerr << "RLIMIT_NOFILE (ulimit -n) is to low" << endl;
        exit(EXIT_FAILURE);
    }
    n_free_files = (nofile.rlim_cur - n_file_reserved) / n_fds_per_registered_file;

    array<thread, N_THREADS> threads;

    for (size_t i{0}; i < N_THREADS; i++) {
        threads[i] = thread{thread_function, i};
    }

    sleep(RUNTIME_S);
    run_flag = false;

    for (size_t i{0}; i < N_THREADS; i++) {
        threads[i].join();
    }

    cout << "DONE!" << endl;
    return 0;
}
