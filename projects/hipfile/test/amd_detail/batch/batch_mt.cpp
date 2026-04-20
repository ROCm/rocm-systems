/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// Multithreading stress test for hipFile's batch context map
//
// Just run the program. It takes no special arguments.
//

#include "batch/batch.h"
#include "hipfile.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace hipFile;

using namespace std;

// TODO: Use getopt() to take N_THREADS and RUNTIME_S from the
//       command line
constexpr size_t N_THREADS = 10;
constexpr int    RUNTIME_S = 10;

static mutex cout_mutex;

static atomic<bool> run_flag{true};

void thread_function(int);

void
thread_function(int id)
{
    constexpr int N_CYCLES  = 100; // # of cycles before checking the run flag
    constexpr int N_PRELOAD = 10;  // # of handles to load before cycling
    constexpr int CAPACITY  = 64;  // Arbitrary

    auto bcm = BatchContextMap{};

    vector<hipFileBatchHandle_t> handles;

    // Preload with data
    for (int i{0}; i < N_PRELOAD; i++) {
        handles.push_back(bcm.createContext(CAPACITY));
    }

    // Random number setup
    random_device rd;
    mt19937       gen{rd()};

    // Distribution for the operation selection
    uniform_int_distribution<int> op_dist{1, 3};

    while (run_flag) {
        for (int i{0}; i < N_CYCLES; i++) {
            // Get a random number from 1-3
            int choice = op_dist(gen);

            switch (choice) {
                // Case 1: Add a new handle
                case 1: {
                    handles.push_back(bcm.createContext(CAPACITY));
                } break;

                // Case 2: Remove a random handle
                case 2: {
                    if (handles.size() == 0) {
                        break;
                    }
                    uniform_int_distribution<size_t> vec_dist{0, handles.size() - 1};
                    size_t                           idx = vec_dist(gen);

                    hipFileBatchHandle_t bh = handles[idx];
                    bcm.destroyContext(bh);

                    swap(handles[idx], handles.back());
                    handles.pop_back();
                } break;

                // Case 3: Get the data for a random handle in the vector
                case 3: {
                    if (handles.size() == 0) {
                        break;
                    }
                    uniform_int_distribution<size_t> vec_dist{0, handles.size() - 1};
                    size_t                           idx = vec_dist(gen);

                    auto data = bcm.get(handles[idx]);

                    // TODO: Maintain a data collection to ensure there are no races
                    //       on the data
                } break;
                default:
                    cout << "ERROR: Tried to pick a non-existent case" << endl;
                    exit(EXIT_FAILURE);
            }
        }
    }

    auto final_size = handles.size();
    for (auto h : handles) {
        bcm.destroyContext(h);
    }

    cout_mutex.lock();
    cout << "Thread " << id << ": DONE (final size: " << to_string(final_size) << ")" << endl;
    cout_mutex.unlock();
}

int
main()
{
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
