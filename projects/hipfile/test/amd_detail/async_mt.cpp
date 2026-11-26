/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// Multithreading stress test for hipFile's asynchronous offload path.
//
// Just run the program. It takes no special arguments. Many threads hammer the
// public async API on a small pool of shared streams, buffers, and files:
// concurrent read/write submissions, register/deregister churn that drains
// in-flight work, and submissions on unregistered streams that run transiently.
// It checks for crashes, deadlocks, and lock errors; it does not verify data.

#include "hipfile.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

using namespace std;

// TODO: Use getopt() to take N_THREADS and RUNTIME_S from the
//       command line
constexpr size_t N_THREADS = 10;
constexpr int    RUNTIME_S = 10;

constexpr size_t ALLOC_SIZE = 1 << 20;  // GPU buffer size (1 MiB)
constexpr size_t FILE_SIZE  = 1 << 20;  // Temp file size (1 MiB)
constexpr size_t IO_SIZE    = 64 << 10; // Bytes moved per op (64 KiB)

constexpr size_t N_FILES         = 8;
constexpr size_t N_BUFFERS       = 8;
constexpr size_t N_REG_STREAMS   = 6;   // Pre-registered; threads churn registration on these
constexpr size_t N_IMPL_STREAMS  = 6;   // Never registered; threads submit on them transiently
constexpr size_t MAX_OUTSTANDING = 256; // Reap a thread's op records once this many are in flight

// Guards all writes to cout/cerr from worker threads. Worker threads share
// these streams, so unsynchronized formatted output is a data race on the
// stream object's internal state.
static mutex io_mutex;

static atomic<bool> run_flag{true};

// Work counters, summed once per thread at exit so the run shows it did something.
static atomic<uint64_t> g_submits{0};     // Async ops the threads issued
static atomic<uint64_t> g_submit_ok{0};   // ... that the API accepted (queued or ran)
static atomic<uint64_t> g_completed{0};   // ... that transferred data, confirmed after a sync
static atomic<uint64_t> g_bytes{0};       // Total bytes those completed ops moved
static atomic<uint64_t> g_registers{0};   // hipFileStreamRegister calls
static atomic<uint64_t> g_deregisters{0}; // hipFileStreamDeregister calls
static atomic<uint64_t> g_syncs{0};       // hipStreamSynchronize calls

// Per-thread tallies, folded into the atomics above at thread exit.
struct Stats {
    uint64_t submits{0};
    uint64_t submit_ok{0};
    uint64_t completed{0};
    uint64_t bytes{0};
    uint64_t registers{0};
    uint64_t deregisters{0};
    uint64_t syncs{0};
};

// Shared pools, built by main() before any thread starts and only read by the
// threads after that, so the containers need no lock; the contention under test
// is on the hipFile state the handles name.
struct SharedPools {
    vector<pair<int, hipFileHandle_t>> files;
    vector<void *>                     buffers;
    vector<hipStream_t>                reg_streams;
    vector<hipStream_t>                impl_streams;
};

static bool
ok(hipFileError_t err)
{
    return err.err == hipFileSuccess;
}

// Reports a failure in a worker thread and aborts the process.
//
// Don't call exit() from a worker thread: it destroys static objects, such
// as the DriverState singleton, while the other threads are still using
// them. abort() stops the process where it is, and when the test runs under
// gdb, gdb prints every thread's backtrace.
[[noreturn]] static void
fatal(const string &msg)
{
    {
        lock_guard<mutex> guard{io_mutex};
        cerr << msg << endl;
    }
    abort();
}

static void
hip_check(hipError_t err, const char *what)
{
    if (err != hipSuccess) {
        fatal(string{what} + " failed: " + hipGetErrorString(err));
    }
}

static int
make_temp_file()
{
    char pathname[]{"async_mt_tmp.XXXXXX"};
    int  fd{mkstemp(pathname)};
    if (-1 == fd) {
        fatal(string{"mkstemp() failed: "} + strerror(errno));
    }
    // Ensure the file is deleted when fd is closed.
    if (-1 == unlink(pathname)) {
        fatal(string{"unlink() failed: "} + strerror(errno));
    }
    // Give it a body so reads stay inside the file.
    if (-1 == ftruncate(fd, static_cast<off_t>(FILE_SIZE))) {
        fatal(string{"ftruncate() failed: "} + strerror(errno));
    }
    return fd;
}

// Stable storage for one in-flight op's parameters. The async API may read these
// through pointers after the submitting call returns, so each record must outlive
// its op. A thread frees its records only after a device synchronize, which
// guarantees every op it launched has completed.
struct OpRecord {
    size_t  size;
    hoff_t  file_off;
    hoff_t  buf_off;
    ssize_t bytes;
};

void thread_function(int, const SharedPools &);

void
thread_function(int id, const SharedPools &pools)
{
    constexpr int N_CYCLES = 100; // # of cycles before checking the run flag

    Stats           stats;
    deque<OpRecord> inflight; // deque for pointer stability across growth

    random_device rd;
    mt19937       gen{rd()};

    uniform_int_distribution<int>      op_dist{1, 6};
    uniform_int_distribution<size_t>   file_dist{0, pools.files.size() - 1};
    uniform_int_distribution<size_t>   buf_dist{0, pools.buffers.size() - 1};
    uniform_int_distribution<size_t>   reg_dist{0, pools.reg_streams.size() - 1};
    uniform_int_distribution<size_t>   impl_dist{0, pools.impl_streams.size() - 1};
    uniform_int_distribution<size_t>   file_off_dist{0, FILE_SIZE - IO_SIZE};
    uniform_int_distribution<size_t>   buf_off_dist{0, ALLOC_SIZE - IO_SIZE};
    uniform_int_distribution<unsigned> flag_dist{0, 7}; // the three fixed-* flags
    uniform_int_distribution<size_t>   all_stream_dist{0, N_REG_STREAMS + N_IMPL_STREAMS - 1};

    // Sync the device, then free the completed op records. Safe because the sync
    // waits for every op this thread launched (gated ops hold the stream until the
    // pool bumps the signal after the IO reads these records).
    auto reap = [&]() {
        hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize");
        // After the sync, every op's result is final. Tally what actually transferred.
        for (const OpRecord &r : inflight) {
            if (r.bytes > 0) {
                stats.completed++;
                stats.bytes += static_cast<uint64_t>(r.bytes);
            }
        }
        inflight.clear();
    };

    // Submit one async op. Concurrency makes several hipFileError_t outcomes valid
    // (e.g. the stream is mid-deregistration), so the return is accepted as-is.
    auto submit = [&](hipStream_t stream, bool is_write) {
        if (inflight.size() >= MAX_OUTSTANDING) {
            reap();
        }
        inflight.push_back(OpRecord{IO_SIZE, static_cast<hoff_t>(file_off_dist(gen)),
                                    static_cast<hoff_t>(buf_off_dist(gen)), 0});
        OpRecord &r   = inflight.back();
        auto      fh  = pools.files[file_dist(gen)].second;
        auto      buf = pools.buffers[buf_dist(gen)];
        auto      fn  = is_write ? hipFileWriteAsync : hipFileReadAsync;
        stats.submits++;
        if (ok(fn(fh, buf, &r.size, &r.file_off, &r.buf_off, &r.bytes, stream))) {
            stats.submit_ok++;
        }
    };

    auto pick_any_stream = [&]() {
        size_t idx = all_stream_dist(gen);
        return idx < N_REG_STREAMS ? pools.reg_streams[idx] : pools.impl_streams[idx - N_REG_STREAMS];
    };

    while (run_flag) {
        for (int i{0}; i < N_CYCLES; i++) {
            switch (op_dist(gen)) {
                // Case 1: Async read on a registered shared stream
                case 1:
                    submit(pools.reg_streams[reg_dist(gen)], false);
                    break;

                // Case 2: Async write on a registered shared stream
                case 2:
                    submit(pools.reg_streams[reg_dist(gen)], true);
                    break;

                // Case 3: Async read/write on an unregistered stream, which runs the transfer transiently
                case 3:
                    submit(pools.impl_streams[impl_dist(gen)], (gen() & 1u) != 0);
                    break;

                // Case 4: (Re)register a shared stream, with random fixed-* flags
                case 4:
                    (void)hipFileStreamRegister(pools.reg_streams[reg_dist(gen)], flag_dist(gen));
                    stats.registers++;
                    break;

                // Case 5: Deregister a shared stream; drains in-flight work on it
                case 5:
                    if ((gen() & 1u) != 0) {
                        (void)hipFileStreamDeregister(pools.reg_streams[reg_dist(gen)]);
                    }
                    else {
                        (void)hipFileStreamDeregister(pools.impl_streams[impl_dist(gen)]);
                    }
                    stats.deregisters++;
                    break;

                // Case 6: Synchronize a random shared stream, contending with submits and drains
                case 6:
                    hip_check(hipStreamSynchronize(pick_any_stream()), "hipStreamSynchronize");
                    stats.syncs++;
                    break;

                default:
                    fatal("ERROR: Tried to pick a non-existent case");
            }
        }
    }

    // Drain everything this thread launched before its records go out of scope.
    reap();

    g_submits.fetch_add(stats.submits, memory_order_relaxed);
    g_submit_ok.fetch_add(stats.submit_ok, memory_order_relaxed);
    g_completed.fetch_add(stats.completed, memory_order_relaxed);
    g_bytes.fetch_add(stats.bytes, memory_order_relaxed);
    g_registers.fetch_add(stats.registers, memory_order_relaxed);
    g_deregisters.fetch_add(stats.deregisters, memory_order_relaxed);
    g_syncs.fetch_add(stats.syncs, memory_order_relaxed);

    {
        lock_guard<mutex> guard{io_mutex};
        cout << "Thread " << id << ": DONE (" << stats.submits << " submits, " << stats.submit_ok << " ok, "
             << stats.completed << " completed, " << stats.bytes << " bytes; " << stats.registers << " reg, "
             << stats.deregisters << " dereg, " << stats.syncs << " sync)" << endl;
    }
}

int
main()
{
    // Security analyzers will be sad if you don't set umask 0077 before
    // creating temporary files.
    (void)umask(S_IRWXG | S_IRWXO);

    if (!ok(hipFileDriverOpen())) {
        fatal("hipFileDriverOpen() failed");
    }

    SharedPools pools;

    for (size_t i{0}; i < N_FILES; i++) {
        int             fd = make_temp_file();
        hipFileHandle_t fh;
        hipFileDescr_t  descr{hipFileHandleTypeOpaqueFD, {fd}, nullptr};
        if (!ok(hipFileHandleRegister(&fh, &descr))) {
            fatal("hipFileHandleRegister() failed");
        }
        pools.files.push_back({fd, fh});
    }

    for (size_t i{0}; i < N_BUFFERS; i++) {
        void *buf = nullptr;
        hip_check(hipMalloc(&buf, ALLOC_SIZE), "hipMalloc");
        if (!ok(hipFileBufRegister(buf, ALLOC_SIZE, 0))) {
            fatal("hipFileBufRegister() failed");
        }
        pools.buffers.push_back(buf);
    }

    for (size_t i{0}; i < N_REG_STREAMS; i++) {
        hipStream_t stream;
        hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        pools.reg_streams.push_back(stream);
    }
    for (size_t i{0}; i < N_IMPL_STREAMS; i++) {
        hipStream_t stream;
        hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        pools.impl_streams.push_back(stream);
    }

    // Start the registered subset registered; threads churn it from there. The
    // implicit subset stays unregistered so submissions on it run transiently.
    for (auto stream : pools.reg_streams) {
        if (!ok(hipFileStreamRegister(stream, 0))) {
            fatal("hipFileStreamRegister() failed");
        }
    }

    array<thread, N_THREADS> threads;
    for (size_t i{0}; i < N_THREADS; i++) {
        threads[i] = thread{thread_function, static_cast<int>(i), cref(pools)};
    }

    sleep(RUNTIME_S);
    run_flag = false;

    for (size_t i{0}; i < N_THREADS; i++) {
        threads[i].join();
    }

    // Teardown. Everything is drained first, then released in dependency order.
    hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize");

    for (auto stream : pools.reg_streams) {
        (void)hipFileStreamDeregister(stream);
        hip_check(hipStreamDestroy(stream), "hipStreamDestroy");
    }
    for (auto stream : pools.impl_streams) {
        (void)hipFileStreamDeregister(stream);
        hip_check(hipStreamDestroy(stream), "hipStreamDestroy");
    }
    for (auto buf : pools.buffers) {
        (void)hipFileBufDeregister(buf);
        hip_check(hipFree(buf), "hipFree");
    }
    for (auto [fd, fh] : pools.files) {
        hipFileHandleDeregister(fh);
        if (-1 == close(fd)) {
            lock_guard<mutex> guard{io_mutex};
            cerr << "close() failed: " << strerror(errno) << endl;
        }
    }
    (void)hipFileDriverClose();

    cout << "TOTAL: " << g_submits.load() << " submits, " << g_submit_ok.load() << " ok, "
         << g_completed.load() << " completed, " << g_bytes.load() << " bytes; " << g_registers.load()
         << " reg, " << g_deregisters.load() << " dereg, " << g_syncs.load() << " sync" << endl;

    // Guard against a silent no-op pass: if not a single async op ever completed, the offload path
    // did no work and the run proves nothing.
    if (g_completed.load() == 0) {
        cerr << "FAILED: no asynchronous op completed; the offload path did no work" << endl;
        return EXIT_FAILURE;
    }

    cout << "DONE!" << endl;
    return 0;
}
