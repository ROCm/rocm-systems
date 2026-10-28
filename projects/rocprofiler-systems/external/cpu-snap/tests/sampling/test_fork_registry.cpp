#include <gtest/gtest.h>

#include <cstdlib>

#include <sys/wait.h>
#include <unistd.h>

#include "cpu_snap/sampling/fork_registry.hpp"

namespace
{

namespace fork_registry = cpu_snap::sampling::internal::fork_registry;

struct probe
{
    int calls = 0;
};

void
count_call(void* object) noexcept
{
    ++static_cast<probe*>(object)->calls;
}

class ForkRegistryTest : public ::testing::Test
{
protected:
    void TearDown() override
    {
        fork_registry::remove(&m_a);
        fork_registry::remove(&m_b);
    }

    // Forks; the child reports its callback counts through the exit code
    // (a * 10 + b) and the parent returns the child's exit code.
    int run_child()
    {
        const pid_t pid = ::fork();
        if(pid == 0)
        {
            ::_exit(m_a.calls * 10 + m_b.calls);
        }
        int status = 0;
        EXPECT_EQ(::waitpid(pid, &status, 0), pid);
        EXPECT_TRUE(WIFEXITED(status));
        return WEXITSTATUS(status);
    }

    probe m_a;
    probe m_b;
};

TEST_F(ForkRegistryTest, ChildRunsEachRegisteredCallbackOnceParentNone)
{
    fork_registry::add(&m_a, &count_call);
    fork_registry::add(&m_b, &count_call);

    EXPECT_EQ(run_child(), 11);
    EXPECT_EQ(m_a.calls, 0);
    EXPECT_EQ(m_b.calls, 0);
}

TEST_F(ForkRegistryTest, RemovedObjectIsNotCalled)
{
    fork_registry::add(&m_a, &count_call);
    fork_registry::add(&m_b, &count_call);
    fork_registry::remove(&m_a);

    EXPECT_EQ(run_child(), 1);
}

TEST_F(ForkRegistryTest, RemoveOfUnknownObjectIsNoOp)
{
    fork_registry::remove(&m_a);
    EXPECT_EQ(run_child(), 0);
}

TEST_F(ForkRegistryTest, AddAfterForkWorks)
{
    fork_registry::add(&m_a, &count_call);
    EXPECT_EQ(run_child(), 10);

    fork_registry::add(&m_b, &count_call);
    EXPECT_EQ(run_child(), 11);
}

TEST_F(ForkRegistryTest, ChildClearsRegistryAfterRunningCallbacks)
{
    fork_registry::add(&m_a, &count_call);
    const pid_t pid = ::fork();
    if(pid == 0)
    {
        const pid_t grandchild = ::fork();
        if(grandchild == 0) ::_exit(m_a.calls);
        int status = 0;
        ::waitpid(grandchild, &status, 0);
        ::_exit(WEXITSTATUS(status) * 10 + m_a.calls);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 11);
}

}  // namespace
