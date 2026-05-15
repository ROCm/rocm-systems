/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include "RcclMockFuncs.hpp"
#include "comm.h"

#include <thread>

namespace RcclUnitTesting
{
  /**
   * \brief Verify correctness of Recorder record() correctness in binary mode
   * ******************************************************************************************/
  TEST(Recorder, ParseBinary)
  {
    // to add after binary export of logging is supported
  }

  /**
   * \brief Verify correctness of Recorder record() correctness in json mode
   * ******************************************************************************************/
  TEST(Recorder, ParseJson)
  {
    setenv("RCCL_REPLAY_FILE", "/tmp/test.json", 1);

    int pid = getpid();
    hipStream_t stream;
    hipStreamCreate(&stream);

    int array[] = {2, 3, 5};
    ncclComm comm{.nRanks = 1, .localRank = 1, .localRankToRank = array, .opCount = 8, .planner = {.nTasksColl = 13, .nTasksP2p = 21}};
    rccl::rcclApiCall call(rccl::rrAllToAllv, {.sendbuff = (void*)0x7f22f9600000, .recvbuff = (void*)0x7f22f9601000, .count = 0, .datatype = ncclFloat32, .comm = &comm, .stream = stream});
    rccl::Recorder::instance().record(call);

    std::vector<rccl::rcclApiCall> calls;
    char entry[4096];
    gethostname(entry, 256);
    //parse the outfile
    std::string filename = "/tmp/test." + std::to_string(pid) + "." + std::string(entry) + ".json";
    std::ifstream fp(filename);
    fp.getline(entry, 4096);
    fp.getline(entry, 4096);
    fp.getline(entry, 4096);
    parseJsonEntry(entry, calls);
    // compare only the fields after the pid field
    int result = memcmp(&(calls[0].pid)+1, &(call.pid)+1, sizeof(rccl::rcclApiCall)-sizeof(call.pid));
    fp.close(); // care that recorder is not designed to anticipate fp closing before destructor
    //remove(filename.c_str()); TODO: after implement fp closing mechanism via envvar, remove file after test case
    unsetenv("RCCL_REPLAY_FILE");
    assert(!result);
  }

  /**
   * \brief Verify RCCL Recorder's integrity in multithread context by comparing Recorder
   * instance across different threads.
   * ******************************************************************************************/
  static void recorderCmp(void** recorder)
  {
    *recorder = &(rccl::Recorder::instance());
  }
  TEST(Recorder, VerifyMultithread)
  {
    void *p1, *p2;
    std::thread t1(recorderCmp, &p1);
    std::thread t2(recorderCmp, &p2);
    t1.join();
    t2.join();
    assert(p1 == p2);
  }
}
