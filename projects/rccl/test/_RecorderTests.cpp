/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include "RcclMockFuncs.hpp"
#include "comm.h"
#include "common/ProcessIsolatedTestRunner.hpp"

#include <cstdio>
#include <fstream>
#include <memory>
#include <thread>

#define HIPCALL(cmd)                                                                          \
    do {                                                                                      \
        hipError_t error = (cmd);                                                             \
        GTEST_ASSERT_EQ(error, hipSuccess);                                                   \
    } while (0)

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
    RUN_ISOLATED_TEST_WITH_ENV(
      "ParseJson",
      []()
      {
        int pid = getpid();
        hipStream_t stream;
        HIPCALL(hipStreamCreate(&stream));

        int array[] = {2, 3, 5};
        ncclComm comm{.nRanks = 1, .localRank = 1, .localRankToRank = array, .opCount = 8, .planner = {.nTasksColl = 13, .nTasksP2p = 21}};
        rccl::rcclApiCall call(rccl::rrAllToAllv, {.sendbuff = (void*)0x7f22f9600000, .recvbuff = (void*)0x7f22f9601000, .count = 0, .datatype = ncclFloat32, .comm = &comm, .stream = stream});
        rccl::Recorder::instance().record(call);

        std::vector<rccl::rcclApiCall> calls;
        char entry[4096];
        gethostname(entry, 256);
        // Parse the output file written by the Recorder
        std::string filename = "/tmp/test." + std::to_string(pid) + "." + std::string(entry) + ".json";
        std::ifstream fp(filename);
        ASSERT_TRUE(fp.is_open()) << "Recorder did not create expected file: " << filename;
        fp.getline(entry, 4096); // line 1: "{"
        fp.getline(entry, 4096); // line 2: "  version : 1,"
        fp.getline(entry, 4096); // line 3: the serialised API call
        parseJsonEntry(entry, calls);
        ASSERT_FALSE(calls.empty()) << "parseJsonEntry produced no results; raw line: " << entry;
        // Compare all fields after pid (pid is the child's pid, not the outer test's)
        int result = memcmp(&(calls[0].pid)+1, &(call.pid)+1, sizeof(rccl::rcclApiCall)-sizeof(call.pid));
        fp.close();
        EXPECT_EQ(result, 0) << "Round-tripped rcclApiCall fields do not match";
      },
      {{"RCCL_REPLAY_FILE", "/tmp/test.json"}}
    );
  }

  /**
   * \brief Verify the RedOp record() overload writes PreMulSum scalar/datatype/residence and Destroy op/comm to json
   * ******************************************************************************************/
  TEST(Recorder, RedOpRecordJson)
  {
    RUN_ISOLATED_TEST_WITH_ENV(
      "RedOpRecordJson",
      []()
      {
        int array[] = {2, 3, 5};
        ncclComm comm{.nRanks = 1, .localRank = 1, .localRankToRank = array, .opCount = 8,
                      .planner = {.nTasksColl = 13, .nTasksP2p = 21}};
        float scalar = 2.0f;
        // Non-zero datatype and residence so a dropped PreMulSum branch is not hidden by the zero-initialised defaults.
        ASSERT_EQ(ncclSuccess, rccl::Recorder::instance().record(rccl::rrRedOpCreatePreMulSum, ncclMax, &comm,
                                                                 ncclFloat32, ncclScalarHostImmediate, &scalar));
        ASSERT_EQ(ncclSuccess, rccl::Recorder::instance().record(rccl::rrRedOpDestroy, ncclMax, &comm));

        char line[4096];
        gethostname(line, sizeof(line));
        std::string filename = "/tmp/test_redop." + std::to_string(getpid()) + "." + std::string(line) + ".json";
        std::ifstream fp(filename);
        ASSERT_TRUE(fp.is_open()) << "Recorder did not create expected file: " << filename;
        fp.getline(line, sizeof(line)); // line 1: "{"
        fp.getline(line, sizeof(line)); // line 2: "  version : 1,"
        fp.getline(line, sizeof(line)); // line 3: RedOpCreatePreMulSum
        std::string create(line);
        fp.getline(line, sizeof(line)); // line 4: RedOpDestroy
        std::string destroy(line);
        fp.close();
        std::remove(filename.c_str());

        // parseJsonEntry is not used: its RedOp sscanf calls pass fewer pointers than conversions (UB) inside assert().
        size_t pos = create.find("RedOpCreatePreMulSum : [");
        ASSERT_NE(pos, std::string::npos) << "raw line: " << create;
        void* gotScalar = nullptr;
        void* gotComm = nullptr;
        int gotDatatype = -1;
        int gotOp = -1;
        int gotResidence = -1;
        constexpr const char* kCreateFmt =
          "RedOpCreatePreMulSum : [scalar : %p, datatype : %d, op : %d, residence : %d, comm : %p,";
        ASSERT_EQ(5, std::sscanf(create.c_str() + pos, kCreateFmt, &gotScalar, &gotDatatype, &gotOp, &gotResidence,
                                 &gotComm))
          << "raw line: " << create;
        EXPECT_EQ(gotScalar, static_cast<void*>(&scalar));
        EXPECT_EQ(gotDatatype, ncclFloat32);
        EXPECT_EQ(gotOp, ncclMax);
        EXPECT_EQ(gotResidence, ncclScalarHostImmediate);
        EXPECT_EQ(gotComm, static_cast<void*>(&comm));

        pos = destroy.find("RedOpDestroy : [");
        ASSERT_NE(pos, std::string::npos) << "raw line: " << destroy;
        gotOp = -1;
        gotComm = nullptr;
        ASSERT_EQ(2, std::sscanf(destroy.c_str() + pos, "RedOpDestroy : [op : %d, comm : %p,", &gotOp, &gotComm))
          << "raw line: " << destroy;
        EXPECT_EQ(gotOp, ncclMax);
        EXPECT_EQ(gotComm, static_cast<void*>(&comm));
      },
      {{"RCCL_REPLAY_FILE", "/tmp/test_redop.json"}}
    );
  }

  /**
   * \brief Verify binary mode writes PreMulSum scalar/datatype/residence as sendbuff/datatype/root; Destroy drops them
   * ******************************************************************************************/
  TEST(Recorder, RedOpRecordBinary)
  {
    RUN_ISOLATED_TEST_WITH_ENV(
      "RedOpRecordBinary",
      []()
      {
        int ranks[] = {2, 3, 5};
        auto comm = std::make_unique<ncclComm>();
        comm->nRanks = 4;
        comm->localRank = 1;
        comm->localRankToRank = ranks;
        float scalar = 2.0f;
        double ignored = 4.0;
        // Non-default datatype and residence, so a dropped or inverted PreMulSum guard cannot match the defaults.
        ASSERT_EQ(ncclSuccess, rccl::Recorder::instance().record(rccl::rrRedOpCreatePreMulSum, ncclMax, comm.get(),
                                                                 ncclFloat32, ncclScalarHostImmediate, &scalar));
        // Destroy gets non-default scalar arguments too, so a guard that always copies them is caught.
        ASSERT_EQ(ncclSuccess, rccl::Recorder::instance().record(rccl::rrRedOpDestroy, ncclMax, comm.get(),
                                                                 ncclFloat64, ncclScalarHostImmediate, &ignored));

        char host[256];
        gethostname(host, sizeof(host));
        std::string filename = "/tmp/test_redop." + std::to_string(getpid()) + "." + std::string(host) + ".bin";
        std::ifstream fp(filename, std::ios::binary);
        ASSERT_TRUE(fp.is_open()) << "Recorder did not create expected file: " << filename;
        // One spare slot so a stray extra record shows up in bytesRead.
        rccl::rcclApiCall calls[3];
        fp.read(reinterpret_cast<char*>(calls), sizeof(calls));
        std::streamsize bytesRead = fp.gcount();
        fp.close();
        std::remove(filename.c_str());
        // Binary mode writes exactly one raw struct per call and no header.
        ASSERT_EQ(bytesRead, static_cast<std::streamsize>(2 * sizeof(rccl::rcclApiCall)));

        const rccl::rcclApiCall& create = calls[0];
        EXPECT_EQ(create.type, rccl::rrRedOpCreatePreMulSum);
        EXPECT_EQ(create.op, ncclMax);
        EXPECT_EQ(create.comm, comm.get());
        // nRanks and globalRank are taken from the comm and only reach disk in binary mode.
        EXPECT_EQ(create.nRanks, 4);
        EXPECT_EQ(create.globalRank, 3);
        EXPECT_EQ(create.sendbuff, static_cast<const void*>(&scalar)) << "sendbuff must come from scalar";
        EXPECT_EQ(create.datatype, ncclFloat32) << "datatype must come from the datatype argument";
        EXPECT_EQ(create.root, static_cast<int>(ncclScalarHostImmediate)) << "root must come from residence";

        const rccl::rcclApiCall& destroy = calls[1];
        EXPECT_EQ(destroy.type, rccl::rrRedOpDestroy);
        EXPECT_EQ(destroy.op, ncclMax);
        EXPECT_EQ(destroy.comm, comm.get());
        // record() builds from the aggregate ncclInfo{.op, .comm}, so fields Destroy must not set are zero.
        EXPECT_EQ(destroy.sendbuff, nullptr) << "Destroy must not record the scalar";
        EXPECT_EQ(destroy.datatype, ncclInt8) << "Destroy must not record the datatype";
        EXPECT_EQ(destroy.root, 0) << "Destroy must not record the residence";
      },
      {{"RCCL_REPLAY_FILE", "/tmp/test_redop.bin"}}
    );
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
