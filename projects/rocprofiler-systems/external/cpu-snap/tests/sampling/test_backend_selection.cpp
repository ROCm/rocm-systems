#include <gtest/gtest.h>

#include <cerrno>
#include <cstddef>
#include <system_error>

#include "cpu_snap/error.hpp"
#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/sampling/backend_selection.hpp"
#include "cpu_snap/sampling/sampler.hpp"

namespace
{

using cpu_snap::sampling::backend_kind;
using cpu_snap::sampling::perf_probe_result;
using cpu_snap::sampling::resolve_backend;
using cpu_snap::sampling::sampler;

int  g_probe_calls          = 0;
bool g_probe_exclude_kernel = false;

perf_probe_result
probe_ok(bool exclude_kernel)
{
    ++g_probe_calls;
    g_probe_exclude_kernel = exclude_kernel;
    return { true, 0 };
}

perf_probe_result
probe_eacces(bool exclude_kernel)
{
    ++g_probe_calls;
    g_probe_exclude_kernel = exclude_kernel;
    return { false, EACCES };
}

perf_probe_result
probe_enosys(bool)
{
    ++g_probe_calls;
    return { false, ENOSYS };
}

class ResolveBackendTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_probe_calls          = 0;
        g_probe_exclude_kernel = false;
    }
};

TEST_F(ResolveBackendTest, ExplicitSignalNeverProbes)
{
    const auto choice = resolve_backend(backend_kind::signal, true, &probe_ok);
    EXPECT_EQ(choice.kind, backend_kind::signal);
    EXPECT_TRUE(choice.reason.empty());
    EXPECT_EQ(g_probe_calls, 0);
}

TEST_F(ResolveBackendTest, ExplicitPerfWithAvailablePerfIsPerf)
{
    const auto choice = resolve_backend(backend_kind::perf, true, &probe_ok);
    EXPECT_EQ(choice.kind, backend_kind::perf);
    EXPECT_TRUE(choice.reason.empty());
}

TEST_F(ResolveBackendTest, ExplicitPerfDeniedThrowsActionableError)
{
    try
    {
        (void) resolve_backend(backend_kind::perf, true, &probe_eacces);
        FAIL() << "expected cpu_snap::environment_error";
    }
    catch(const cpu_snap::environment_error& e)
    {
        EXPECT_EQ(e.code().value(), EACCES);
        EXPECT_NE(std::string(e.what()).find("perf_event_paranoid"), std::string::npos);
        EXPECT_NE(std::string(e.what()).find("signal"), std::string::npos);
    }
}

TEST_F(ResolveBackendTest, AutomaticWithAvailablePerfIsPerfWithoutReason)
{
    const auto choice = resolve_backend(backend_kind::automatic, true, &probe_ok);
    EXPECT_EQ(choice.kind, backend_kind::perf);
    EXPECT_TRUE(choice.reason.empty());
}

TEST_F(ResolveBackendTest, AutomaticDeniedFallsBackToSignalWithReason)
{
    const auto choice = resolve_backend(backend_kind::automatic, true, &probe_eacces);
    EXPECT_EQ(choice.kind, backend_kind::signal);
    EXPECT_FALSE(choice.reason.empty());
}

TEST_F(ResolveBackendTest, AutomaticFallsBackOnAnyErrno)
{
    const auto choice = resolve_backend(backend_kind::automatic, true, &probe_enosys);
    EXPECT_EQ(choice.kind, backend_kind::signal);
    EXPECT_FALSE(choice.reason.empty());
}

TEST_F(ResolveBackendTest, ExcludeKernelIsForwardedToProbe)
{
    (void) resolve_backend(backend_kind::automatic, true, &probe_ok);
    EXPECT_TRUE(g_probe_exclude_kernel);
    (void) resolve_backend(backend_kind::automatic, false, &probe_ok);
    EXPECT_FALSE(g_probe_exclude_kernel);
}

sampler::config
make_config(cpu_snap::core::memory_sink& sink)
{
    sampler::config cfg;
    cfg.streaming.sink = std::ref(sink);
    return cfg;
}

TEST(SamplerBackendSelectionTest, DefaultConfigResolvesToConcreteBackend)
{
    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = 1U << 20 });
    auto cfg = make_config(sink);
    ASSERT_EQ(cfg.backend, backend_kind::automatic);
    const bool perf_ok = cpu_snap::sampling::probe_perf_open(cfg.perf.exclude_kernel).available;

    sampler smp{ std::move(cfg) };

    const auto& choice = smp.selected_backend();
    EXPECT_NE(choice.kind, backend_kind::automatic);
    if(perf_ok)
    {
        EXPECT_EQ(choice.kind, backend_kind::perf);
        EXPECT_TRUE(choice.reason.empty());
    }
    else
    {
        EXPECT_EQ(choice.kind, backend_kind::signal);
        EXPECT_FALSE(choice.reason.empty());
    }
}

TEST(SamplerBackendSelectionTest, ExplicitPerfThrowsAtConstructionWhenPerfDenied)
{
    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = 1U << 20 });
    auto cfg    = make_config(sink);
    cfg.backend = backend_kind::perf;
    if(cpu_snap::sampling::probe_perf_open(cfg.perf.exclude_kernel).available)
    {
        GTEST_SKIP() << "perf_event_open is permitted here";
    }
    EXPECT_THROW(sampler{ std::move(cfg) }, cpu_snap::environment_error);
}

}  // namespace
