// Compiles MemManagerTests.cpp with real RCCL types via hipcc.
#include "comm.h"
#include "mem_manager.h"
#include "alloc.h"
#include "utils.h"

#include "mem_manager/MemManagerTests.cpp"

#include <memory>

// Seams in wrapper_link_stubs.cpp and init_stubs.cpp; ncclCommSuspend_impl runs from mem_manager_src_wrapper.cc.
extern int64_t g_wrapperEnqueueRearchEnable;
extern ncclResult_t (*g_wrapperMgmtTaskEnqueue)(struct ncclAsyncJob*, ncclResult_t (*)(struct ncclAsyncJob*),
                                                void (*)(void*), struct ncclComm*);

namespace {

int g_suspendJobsDestroyed = 0;
struct ncclAsyncJob* g_suspendJob = nullptr;
void (*g_suspendJobFree)(void*) = nullptr;
struct ncclComm* g_suspendJobComm = nullptr;

class CommSuspendRearchTest : public ::testing::Test {
protected:
    std::unique_ptr<ncclComm> comm = std::make_unique<ncclComm>();

    void SetUp() override {
        g_suspendJobsDestroyed = 0;
        g_suspendJob = nullptr;
        g_suspendJobFree = nullptr;
        g_suspendJobComm = nullptr;
        g_wrapperEnqueueRearchEnable = 1;
        comm->config.blocking = 1;
    }
    void TearDown() override {
        g_wrapperEnqueueRearchEnable = 0;
        g_wrapperMgmtTaskEnqueue = nullptr;
    }
};

TEST_F(CommSuspendRearchTest, HandsTheSuspendJobToTheManagementQueue) {
    g_wrapperMgmtTaskEnqueue = [](struct ncclAsyncJob* job, ncclResult_t (*)(struct ncclAsyncJob*),
                                  void (*destructor)(void*), struct ncclComm* c) {
        g_suspendJob = job;
        g_suspendJobFree = destructor;
        g_suspendJobComm = c;
        return ncclSuccess;
    };

    ASSERT_EQ(ncclCommSuspend_impl(comm.get(), NCCL_SUSPEND_MEM), ncclSuccess);
    ASSERT_NE(g_suspendJob, nullptr);
    ASSERT_NE(g_suspendJobFree, nullptr);
    EXPECT_EQ(g_suspendJobComm, comm.get());
    EXPECT_TRUE(ncclIntruQueueEmpty(&comm->suspendTaskQueue));
    g_suspendJobFree(g_suspendJob);
}

// ncclMgmtTaskEnqueue owns the job once called: when it rejects (a blocking and a nonblocking comm in one
// group) it runs the destructor itself, so fail: must not free the job again.
TEST_F(CommSuspendRearchTest, EnqueueRejects_JobIsDestroyedOnlyByTheEnqueue) {
    g_wrapperMgmtTaskEnqueue = [](struct ncclAsyncJob* job, ncclResult_t (*)(struct ncclAsyncJob*),
                                  void (*destructor)(void*), struct ncclComm*) {
        destructor(job);
        ++g_suspendJobsDestroyed;
        return ncclInvalidArgument;
    };

    EXPECT_EQ(ncclCommSuspend_impl(comm.get(), NCCL_SUSPEND_MEM), ncclInvalidArgument);
    EXPECT_EQ(g_suspendJobsDestroyed, 1);
    EXPECT_TRUE(ncclIntruQueueEmpty(&comm->suspendTaskQueue));
}

} // namespace
