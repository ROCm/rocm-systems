/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "CallCollectiveForked.hpp"
#include "CollectiveArgs.hpp"
#include <rccl/rccl.h>
#include <gtest/gtest.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#define HIPCALL(cmd)                                                                          \
    do {                                                                                      \
        hipError_t error = (cmd);                                                             \
        if (error != hipSuccess)                                                              \
        {                                                                                     \
            printf("Encountered HIP error (%s) at line %d in file %s\n",                      \
                                  hipGetErrorString(error), __LINE__, __FILE__);              \
            exit(-1);                                                                         \
        }                                                                                     \
    } while (0)

#define NCCLCHECK(cmd) do {                                     \
    ncclResult_t res = cmd;                                     \
    if (res != ncclSuccess) {                                   \
         printf("NCCL failure %s:%d '%s'\n",                    \
            __FILE__,__LINE__,ncclGetErrorString(res));         \
         exit(-1);                                              \
    }                                                           \
} while(0)

namespace RcclUnitTesting
{
static constexpr int kChildSkipped = 77;

void callCollective(ncclUniqueId id, int collID, int rank, int nranks, const std::vector<int>& send, std::vector<int>& recv, bool use_managed_mem){
    switch(collID){
        case ncclCollAllReduce:
        case ncclCollAllGather:
        break;

        default:
            TEST_ERROR("This collective is not implemented for callCollective routine");
    }
    
    HIPCALL(hipSetDevice(rank));
    hipStream_t stream;
    HIPCALL(hipStreamCreate(&stream));
    ncclComm_t comm;
    
    

    NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
    int *sendbuff;
    int *recvbuff;
    void *sendRegHandle;
    void *recvRegHandle;
    

    
    size_t sendSize = 0;
    size_t recvSize = 0;

     switch(collID){
      case ncclCollAllReduce:
        sendSize = send.size();
        recvSize = recv.size();
        break;
      case ncclCollAllGather:
        sendSize = send.size();
        recvSize = nranks*send.size();
        break;
      default: exit(0);
    }

    if(!use_managed_mem){
      HIPCALL(hipMalloc((void **)&sendbuff, sendSize * sizeof(int)));
      HIPCALL(hipMalloc((void **)&recvbuff, recvSize * sizeof(int)));
    }
    else{
      HIPCALL(hipMallocManaged((void **)&sendbuff, sendSize * sizeof(int)));
      HIPCALL(hipMallocManaged((void **)&recvbuff, recvSize * sizeof(int)));
    }    
   
    NCCLCHECK(ncclCommRegister(comm, sendbuff, sendSize * sizeof(int), &sendRegHandle));
    NCCLCHECK(ncclCommRegister(comm, recvbuff, recvSize * sizeof(int), &recvRegHandle));

    HIPCALL(hipMemcpy(sendbuff, send.data(), sizeof(int) * sendSize, hipMemcpyHostToDevice));
    HIPCALL(hipMemcpy(recvbuff, recv.data(), sizeof(int) *recvSize, hipMemcpyHostToDevice));

    switch(collID){
      case ncclCollAllReduce:
        NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, sendSize, ncclInt, ncclSum, comm, stream));
        break;
      case ncclCollAllGather:
        NCCLCHECK(ncclAllGather(sendbuff, recvbuff, sendSize, ncclInt, comm, stream));
        break;
      default: exit(0);
    }

    HIPCALL(hipStreamSynchronize(stream));
    HIPCALL(hipMemcpy(recv.data(), recvbuff, sizeof(int) * recvSize, hipMemcpyDeviceToHost));
    
    NCCLCHECK(ncclCommDeregister(comm, sendRegHandle));
    NCCLCHECK(ncclCommDeregister(comm, recvRegHandle));

    HIPCALL(hipFree(sendbuff));
    HIPCALL(hipFree(recvbuff));
    ncclCommDestroy(comm);
  }

void callCollectiveForked(int nranks,  int collID, const std::vector<int>& sendBuff, std::vector<int>& recvBuff, const std::vector<int>& expected, bool use_managed_mem){
    std::vector<pid_t> children(nranks, 0);
    std::vector<std::vector<int>> childPipes(nranks, std::vector<int>(2,0));
    ncclUniqueId id;

    for(int r = 0; r < nranks; ++r){
      if(pipe(childPipes[r].data()) == -1)
        TEST_ERROR("child %i pipe Failed", r);
    } 

    auto createNCCLid = [&](int rank){
        ncclGetUniqueId(&id);
        close(childPipes[rank][0]);
        write(childPipes[rank][1], &id, sizeof(ncclUniqueId));
        close(childPipes[rank][1]);
    };

    auto getNCCLidFromParent = [&](int rank){
      close(childPipes[rank][1]); //close write to child0
      read(childPipes[rank][0], &id, sizeof(ncclUniqueId));
      close(childPipes[rank][0]);
    };

    auto getAndDistributeNCCLid = [&](int nranks){
      close(childPipes[0][1]); //close write to child0
      read(childPipes[0][0], &id, sizeof(ncclUniqueId)); //read from child0
      for(int r = 1; r < nranks; ++r){
        write(childPipes[r][1], &id, sizeof(ncclUniqueId));
        close(childPipes[r][1]);
      }
    };

    // A child must never return into gtest: it would go on to run the rest of the suite.
    for(int r = 0; r < nranks; ++r){
      children[r] = fork();
      if(children[r] == 0){
        int ngpus = 0;
        HIPCALL(hipGetDeviceCount(&ngpus));
        if(ngpus != nranks){
          _exit(kChildSkipped);
        }
        //child processes
        if(r == 0)
          createNCCLid(r);
        else
          getNCCLidFromParent(r);

        callCollective(id, collID, r, nranks, sendBuff, recvBuff, use_managed_mem);
        int status = 0;
        for(size_t i = 0; i < recvBuff.size(); ++i){
          if(recvBuff[i] != expected[i]){
            printf("Rank %d: recvBuff[%zu] = %d, expected %d\n", r, i, recvBuff[i], expected[i]);
            status = 1;
            break;
          }
        }
        fflush(stdout);
        _exit(status);
      }
    }

    getAndDistributeNCCLid(nranks);

    int numSkipped = 0;
    for(int r = 0; r < nranks; ++r){
      if(children[r] < 0){
        ADD_FAILURE() << "fork failed for rank " << r;
        continue;
      }
      int status = 0;
      if(waitpid(children[r], &status, 0) != children[r]){
        ADD_FAILURE() << "waitpid failed for rank " << r;
        continue;
      }
      if(WIFEXITED(status) && WEXITSTATUS(status) == kChildSkipped){
        ++numSkipped;
        continue;
      }
      EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0)
          << "rank " << r << " child "
          << (WIFSIGNALED(status) ? "killed by signal " + std::to_string(WTERMSIG(status))
                                  : "exited with status " + std::to_string(WEXITSTATUS(status)));
    }
    if(numSkipped == nranks)
      GTEST_SKIP() << "Requires exactly " << nranks << " GPUs";
    EXPECT_EQ(numSkipped, 0) << "only some ranks saw " << nranks << " GPUs";
}

}
