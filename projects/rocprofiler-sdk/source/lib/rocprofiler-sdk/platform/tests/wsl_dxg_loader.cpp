// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// The librocdxg lookup against the real dynamic loader. Each case copies
// fixture librocdxg and libhsa-runtime64 objects into a scratch directory tree
// and runs wsl-dxg-loader-probe on it in a child process, so LD_LIBRARY_PATH is
// read at startup as the loader requires and no loader state carries over from
// one case to the next. No GPU, real thunk or HSA runtime is involved.
//
// The child starts through the dynamic loader with --inhibit-cache, because a
// librocdxg listed in the host's ld.so.cache would otherwise satisfy every bare
// name. LD_LIBRARY_PATH, DT_RPATH/DT_RUNPATH and the default directories are
// still searched, so the cases that need a bare name to fail first check that
// the default directories do not hold one. LD_LIBRARY_PATH holds only what a
// case sets; probe_startup() explains how the probe's own libraries are found.

#include "lib/rocprofiler-sdk/platform/wsl/dxg_thunk.hpp"

#include <gtest/gtest.h>

#include <link.h>
#include <spawn.h>
#include <sys/auxv.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern char** environ;

namespace
{
namespace fs = std::filesystem;

using rocprofiler::platform::wsl::kLibRocdxgUnversionedName;
using rocprofiler::platform::wsl::kLibRocdxgVersionedName;

// Built and installed beside this test.
constexpr const char* kProbe         = "wsl-dxg-loader-probe";
constexpr const char* kThunkFixture  = "libwsl-dxg-fixture-thunk.so";
constexpr const char* kRunpathCaller = "libwsl-dxg-fixture-runpath-caller.so";
constexpr const char* kRpathCaller   = "libwsl-dxg-fixture-rpath-caller.so";
constexpr const char* kPlainCaller   = "libwsl-dxg-fixture-plain-caller.so";

constexpr const char* kCaller     = "libcaller.so";
constexpr const char* kHsaRuntime = "libhsa-runtime64.so.1";

const std::string kVersioned   = kLibRocdxgVersionedName;
const std::string kUnversioned = kLibRocdxgUnversionedName;

fs::path
fixture_dir()
{
    return fs::canonical("/proc/self/exe").parent_path();
}

// The dynamic loader this process runs under, which starts the child too.
std::string
dynamic_loader()
{
    struct Search
    {
        ElfW(Addr) base  = 0;
        std::string path = {};
    } search{static_cast<ElfW(Addr)>(::getauxval(AT_BASE)), {}};

    ::dl_iterate_phdr(
        [](dl_phdr_info* info, size_t, void* data) -> int {
            auto* out = static_cast<Search*>(data);
            if(info->dlpi_addr != out->base || info->dlpi_name == nullptr) return 0;
            out->path = info->dlpi_name;
            return 1;
        },
        &search);
    return search.path;
}

// A scratch directory tree, removed with everything in it.
class Layout
{
public:
    Layout()
    {
        auto pattern = (fs::temp_directory_path() / "wsl-dxg-loader-XXXXXX").string();
        if(::mkdtemp(pattern.data()) == nullptr)
            throw std::runtime_error{std::string{"mkdtemp: "} + std::strerror(errno)};
        root_ = fs::canonical(pattern);
    }

    ~Layout()
    {
        auto ec = std::error_code{};
        fs::remove_all(root_, ec);
    }

    Layout(const Layout&) = delete;
    Layout& operator=(const Layout&) = delete;

    // Copy `fixture` to `dir`/`name`. A copy rather than a link, so the loader
    // sees a distinct file - and maps a distinct object - at every path.
    fs::path add(const std::string& dir, const char* fixture, const std::string& name) const
    {
        fs::create_directories(root_ / dir);
        auto path = root_ / dir / name;
        fs::copy_file(fixture_dir() / fixture, path);
        return path;
    }

    const fs::path& root() const { return root_; }

private:
    fs::path root_ = {};
};

// What the probe printed.
struct ProbeRun
{
    int         status = -1;
    std::string output = {};

    // The rest of every line that starts with `key `.
    std::vector<std::string> values(std::string_view key) const
    {
        auto out    = std::vector<std::string>{};
        auto stream = std::istringstream{output};
        for(auto line = std::string{}; std::getline(stream, line);)
            if(line.size() > key.size() && line.compare(0, key.size(), key) == 0 &&
               line[key.size()] == ' ')
                out.emplace_back(line.substr(key.size() + 1));
        return out;
    }

    std::string value(std::string_view key) const
    {
        const auto all = values(key);
        return all.size() == 1 ? all.front() : std::to_string(all.size()) + " values";
    }
};

std::vector<std::string>
paths(std::initializer_list<fs::path> list)
{
    auto out = std::vector<std::string>{};
    for(const auto& path : list)
        out.emplace_back(path.string());
    return out;
}

// Run `argv` with exactly the environment `envp` and wait for it, collecting
// what it writes to stdout and stderr.
ProbeRun
spawn_and_wait(std::vector<std::string> argv, std::vector<std::string> envp)
{
    auto c_strings = [](std::vector<std::string>& strings) {
        auto out = std::vector<char*>{};
        for(auto& itr : strings)
            out.emplace_back(itr.data());
        out.emplace_back(nullptr);
        return out;
    };
    auto c_argv = c_strings(argv);
    auto c_envp = c_strings(envp);

    auto run = ProbeRun{};
    int  out[2];
    if(::pipe(out) != 0)
    {
        run.output = std::string{"pipe: "} + std::strerror(errno);
        return run;
    }

    posix_spawn_file_actions_t actions;
    ::posix_spawn_file_actions_init(&actions);
    ::posix_spawn_file_actions_adddup2(&actions, out[1], STDOUT_FILENO);
    ::posix_spawn_file_actions_adddup2(&actions, out[1], STDERR_FILENO);
    ::posix_spawn_file_actions_addclose(&actions, out[0]);
    ::posix_spawn_file_actions_addclose(&actions, out[1]);

    auto      pid = pid_t{};
    const int spawned =
        ::posix_spawn(&pid, c_argv[0], &actions, nullptr, c_argv.data(), c_envp.data());
    ::posix_spawn_file_actions_destroy(&actions);
    ::close(out[1]);

    char buffer[4096];
    while(true)
    {
        const auto n = ::read(out[0], buffer, sizeof(buffer));
        if(n > 0)
            run.output.append(buffer, static_cast<size_t>(n));
        else if(n == 0 || errno != EINTR)
            break;
    }
    ::close(out[0]);

    if(spawned != 0)
    {
        run.output = std::string{"posix_spawn: "} + std::strerror(spawned);
        return run;
    }

    int wait_status = 0;
    while(::waitpid(pid, &wait_status, 0) < 0 && errno == EINTR)
    {}
    run.status = WIFEXITED(wait_status) ? WEXITSTATUS(wait_status) : 128 + WTERMSIG(wait_status);
    return run;
}

// This environment minus anything that would steer the dynamic loader or the
// lookup.
std::vector<std::string>
clean_environment()
{
    auto envp = std::vector<std::string>{};
    for(char** entry = environ; *entry != nullptr; ++entry)
    {
        const auto var = std::string_view{*entry};
        if(var.rfind("LD_LIBRARY_PATH=", 0) == 0 || var.rfind("LD_PRELOAD=", 0) == 0 ||
           var.rfind("ROCPROFILER_REGISTER_TOOL_ATTACHED=", 0) == 0)
            continue;
        envp.emplace_back(var);
    }
    return envp;
}

// The real path of the object this process has mapped under file name `name`,
// or an empty string.
std::string
mapped_object(const std::string& name)
{
    struct Search
    {
        std::string_view name = {};
        std::string      path = {};
    } search{name, {}};

    ::dl_iterate_phdr(
        [](dl_phdr_info* info, size_t, void* data) -> int {
            auto*      out   = static_cast<Search*>(data);
            const auto path  = std::string_view{info->dlpi_name != nullptr ? info->dlpi_name : ""};
            const auto slash = path.find_last_of('/');
            if(slash == std::string_view::npos || path.substr(slash + 1) != out->name) return 0;
            out->path = std::string{path};
            return 1;
        },
        &search);

    auto ec = std::error_code{};
    return search.path.empty() ? search.path : fs::canonical(search.path, ec).string();
}

// Every library the dynamic loader could not find, by the name it was looked
// for under: from the "NAME => not found" lines of a traced load, or the
// "error while loading shared libraries: NAME: ..." of an aborted one.
std::vector<std::string>
unresolved_dependencies(const std::string& loader_output)
{
    constexpr auto traced  = std::string_view{" => not found"};
    constexpr auto aborted = std::string_view{"error while loading shared libraries: "};

    auto out    = std::vector<std::string>{};
    auto stream = std::istringstream{loader_output};
    for(auto line = std::string{}; std::getline(stream, line);)
    {
        if(const auto at = line.find(traced); at != std::string::npos)
        {
            const auto begin = line.find_first_not_of(" \t");
            if(begin < at) out.emplace_back(line.substr(begin, at - begin));
        }
        else if(const auto from = line.find(aborted); from != std::string::npos)
        {
            const auto begin = from + aborted.size();
            const auto end   = line.find(':', begin);
            if(end != std::string::npos && end > begin)
                out.emplace_back(line.substr(begin, end - begin));
        }
    }
    return out;
}

// How the probe's own dependencies are supplied: the absolute paths to preload,
// joined with ':', or why they could not be found.
struct ProbeStartup
{
    std::string preload = {};
    std::string error   = {};
};

// The probe links what this test links, but its run path does not always reach
// those libraries. TheRock rewrites the run path of every installed binary as
// if it sat in a bin/ beside lib/, which the unit-test bin/ under share/ does
// not; this test itself starts on an LD_LIBRARY_PATH the probe cannot be given.
//
// So each library the dynamic loader cannot find for the probe is preloaded by
// absolute path, taken from the copy this process is running with. A preloaded
// object only satisfies the DT_NEEDED entries naming its soname. It adds no
// directory to LD_LIBRARY_PATH, to any DT_RPATH or DT_RUNPATH, or to the
// default directories, so every bare-name lookup under test still searches
// exactly the paths its case lays out. The directories are deliberately not
// passed on: in an installed tree the one holding libamd_comgr also holds the
// production librocdxg, which would answer the bare names.
const ProbeStartup&
probe_startup()
{
    static const auto startup = []() {
        const auto probe   = (fixture_dir() / kProbe).string();
        auto       preload = std::vector<std::string>{};
        auto       joined  = std::string{};
        for(auto round = 0; round < 8; ++round)
        {
            // A traced load, as ldd runs one: every dependency is resolved and
            // listed, unresolved ones included, and the probe itself never runs.
            auto envp = clean_environment();
            envp.emplace_back("LD_TRACE_LOADED_OBJECTS=1");
            if(!joined.empty()) envp.emplace_back("LD_PRELOAD=" + joined);
            const auto list = spawn_and_wait({dynamic_loader(), "--inhibit-cache", probe}, envp);

            const auto missing = unresolved_dependencies(list.output);
            if(missing.empty() && list.status == 0) return ProbeStartup{joined, {}};
            if(missing.empty())
                return ProbeStartup{{},
                                    std::string{"tracing the dependencies of "} + kProbe +
                                        " failed (exit " + std::to_string(list.status) + "):\n" +
                                        list.output};

            for(const auto& name : missing)
            {
                const auto path = mapped_object(name);
                if(path.empty() || std::find(preload.begin(), preload.end(), path) != preload.end())
                    return ProbeStartup{
                        {},
                        kProbe + std::string{" needs "} + name +
                            ", which neither its own run path nor this test process provides:\n" +
                            list.output};
                preload.emplace_back(path);
                joined += (joined.empty() ? "" : ":") + path;
            }
        }
        return ProbeStartup{{}, kProbe + std::string{" dependencies did not resolve"}};
    }();
    return startup;
}

// Run the probe with `args`, in the clean environment plus LD_LIBRARY_PATH set
// to `library_path` if there is one, and its own dependencies preloaded.
ProbeRun
run_probe(const std::vector<std::string>& args,
          const std::optional<fs::path>&  library_path = std::nullopt)
{
    const auto& startup = probe_startup();
    if(!startup.error.empty()) return ProbeRun{-1, startup.error};

    auto argv = std::vector<std::string>{
        dynamic_loader(), "--inhibit-cache", (fixture_dir() / kProbe).string()};
    argv.insert(argv.end(), args.begin(), args.end());

    auto envp = clean_environment();
    if(library_path) envp.emplace_back("LD_LIBRARY_PATH=" + library_path->string());
    if(!startup.preload.empty()) envp.emplace_back("LD_PRELOAD=" + startup.preload);
    return spawn_and_wait(std::move(argv), std::move(envp));
}

// Whether the probe reached main(). Exiting before then - 127 with "error while
// loading shared libraries", typically - is the dynamic loader failing to start
// it, and says nothing about the lookup under test.
::testing::AssertionResult
started(const ProbeRun& run)
{
    if(run.value("probe") == "started") return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << kProbe << " did not start (exit " << run.status
           << "): the dynamic loader failed before the lookup under test ran\n"
           << run.output;
}

// The lookup with nothing of the test's own on the search path: LD_LIBRARY_PATH
// unset and a caller directory holding only the caller. Anything it finds came
// from the default directories.
ProbeRun
lookup_with_an_empty_search_path()
{
    const auto layout = Layout{};
    return run_probe({layout.add("caller", kRunpathCaller, kCaller).string()});
}

class wsl_dxg_loader : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(kVersioned.empty()) GTEST_SKIP() << "built against headers without rocdxg_version.h";
        ASSERT_TRUE(fs::exists(fixture_dir() / kProbe))
            << kProbe << " is built and installed beside this test";
        ASSERT_FALSE(dynamic_loader().empty());
        ASSERT_EQ(probe_startup().error, "");
    }
};

// Skips the calling test when the default directories hold a librocdxg, since
// a failing bare-name search cannot be arranged then.
#define SKIP_IF_A_BARE_NAME_RESOLVES_FROM_THE_HOST()                                               \
    do                                                                                             \
    {                                                                                              \
        const auto host = lookup_with_an_empty_search_path();                                      \
        ASSERT_TRUE(started(host));                                                                \
        ASSERT_EQ(host.status, 0) << host.output;                                                  \
        if(host.value("after-read") != "none")                                                     \
            GTEST_SKIP() << "a bare librocdxg name resolves from the default directories: "        \
                         << host.value("after-read");                                              \
    } while(false)
}  // namespace

// librocprofiler-sdk.so is linked with an $ORIGIN run path. Emitted as
// DT_RUNPATH, it is searched after LD_LIBRARY_PATH, so a copy in
// LD_LIBRARY_PATH is taken over one beside the caller.
TEST_F(wsl_dxg_loader, ld_library_path_is_searched_before_the_runpath_of_a_runpath_caller)
{
    const auto layout   = Layout{};
    const auto from_env = layout.add("env", kThunkFixture, kVersioned);
    const auto caller   = layout.add("caller", kRunpathCaller, kCaller);
    layout.add("caller", kThunkFixture, kVersioned);

    const auto run = run_probe({caller.string()}, from_env.parent_path());

    ASSERT_TRUE(started(run));
    ASSERT_EQ(run.status, 0) << run.output;
    ASSERT_EQ(run.value("caller-tag"), "RUNPATH") << run.output;
    EXPECT_EQ(run.value("nodes"), "1") << run.output;
    EXPECT_EQ(run.values("after-read"), paths({from_env})) << run.output;
}

// The same layout with DT_RPATH instead: the caller's RPATH is searched before
// LD_LIBRARY_PATH. LD_LIBRARY_PATH does not take precedence for every caller,
// so which tag librocprofiler-sdk.so is linked with decides it.
TEST_F(wsl_dxg_loader, the_rpath_of_an_rpath_caller_is_searched_before_ld_library_path)
{
    const auto layout      = Layout{};
    const auto from_env    = layout.add("env", kThunkFixture, kVersioned);
    const auto caller      = layout.add("caller", kRpathCaller, kCaller);
    const auto from_caller = layout.add("caller", kThunkFixture, kVersioned);

    const auto run = run_probe({caller.string()}, from_env.parent_path());

    ASSERT_TRUE(started(run));
    ASSERT_EQ(run.status, 0) << run.output;
    ASSERT_EQ(run.value("caller-tag"), "RPATH") << run.output;
    EXPECT_EQ(run.value("nodes"), "1") << run.output;
    EXPECT_EQ(run.values("after-read"), paths({from_caller})) << run.output;
}

// The packaged layout: librocdxg.so.<major> only beside libhsa-runtime64, where
// no bare-name search reaches. It is found there, and once the lookup has
// closed its handle the HSA runtime's own dlopen of the versioned name gets
// that same copy by DT_SONAME rather than a second one. That last part holds
// because the fixture, like librocdxg, is linked with -z nodelete.
TEST_F(wsl_dxg_loader, the_copy_beside_the_only_libhsa_runtime64_is_used_when_no_bare_name_resolves)
{
    SKIP_IF_A_BARE_NAME_RESOLVES_FROM_THE_HOST();

    const auto layout = Layout{};
    const auto caller = layout.add("caller", kRunpathCaller, kCaller);
    const auto hsa    = layout.add("hsa", kPlainCaller, kHsaRuntime);
    const auto beside = layout.add("hsa", kThunkFixture, kVersioned);

    const auto run = run_probe(
        {caller.string(), "--preload", hsa.string(), "--then-open", hsa.string(), kVersioned});

    ASSERT_TRUE(started(run));
    ASSERT_EQ(run.status, 0) << run.output;
    ASSERT_EQ(run.values("hsa-runtime"), paths({hsa})) << run.output;
    EXPECT_EQ(run.value("nodes"), "1") << run.output;
    EXPECT_EQ(run.values("after-read"), paths({beside})) << run.output;
    EXPECT_EQ(run.value("then-open"), beside.string()) << run.output;
    EXPECT_EQ(run.values("after-then-open"), paths({beside})) << run.output;
}

// The directory beside libhsa-runtime64 is only a fallback: a copy the bare
// name reaches through LD_LIBRARY_PATH is taken over the one beside it.
TEST_F(wsl_dxg_loader, a_bare_name_that_resolves_is_taken_over_the_copy_beside_libhsa_runtime64)
{
    const auto layout   = Layout{};
    const auto from_env = layout.add("env", kThunkFixture, kVersioned);
    const auto caller   = layout.add("caller", kRunpathCaller, kCaller);
    const auto hsa      = layout.add("hsa", kPlainCaller, kHsaRuntime);
    layout.add("hsa", kThunkFixture, kVersioned);

    const auto run =
        run_probe({caller.string(), "--preload", hsa.string()}, from_env.parent_path());

    ASSERT_TRUE(started(run));
    ASSERT_EQ(run.status, 0) << run.output;
    ASSERT_EQ(run.values("hsa-runtime"), paths({hsa})) << run.output;
    EXPECT_EQ(run.value("nodes"), "1") << run.output;
    EXPECT_EQ(run.values("after-read"), paths({from_env})) << run.output;
}

// The HSA runtime already loaded the thunk by explicit path, under either file
// name, as its fallbacks beside itself do. Its DT_SONAME is versioned, so the
// resident probe finds it, and the copy in LD_LIBRARY_PATH is never mapped.
TEST_F(wsl_dxg_loader, a_resident_copy_is_found_by_soname_and_no_second_copy_is_mapped)
{
    for(const auto& file_name : {kVersioned, kUnversioned})
    {
        const auto layout   = Layout{};
        const auto from_env = layout.add("env", kThunkFixture, kVersioned);
        const auto caller   = layout.add("caller", kRunpathCaller, kCaller);
        const auto hsa      = layout.add("hsa", kPlainCaller, kHsaRuntime);
        const auto resident = layout.add("hsa", kThunkFixture, file_name);

        const auto run =
            run_probe({caller.string(), "--preload", hsa.string(), "--preload", resident.string()},
                      from_env.parent_path());

        ASSERT_TRUE(started(run)) << file_name;
        ASSERT_EQ(run.status, 0) << file_name << "\n" << run.output;
        EXPECT_EQ(run.value("nodes"), "1") << file_name << "\n" << run.output;
        EXPECT_EQ(run.values("after-read"), paths({resident})) << file_name << "\n" << run.output;
    }
}

// libhsa-runtime64 mapped from two directories, each holding a thunk: neither
// is searched, rather than guessing which runtime will load the thunk.
TEST_F(wsl_dxg_loader, two_libhsa_runtime64_directories_are_not_searched_beside)
{
    SKIP_IF_A_BARE_NAME_RESOLVES_FROM_THE_HOST();

    const auto layout = Layout{};
    const auto caller = layout.add("caller", kRunpathCaller, kCaller);
    const auto first  = layout.add("first", kPlainCaller, kHsaRuntime);
    const auto second = layout.add("second", kPlainCaller, kHsaRuntime);
    layout.add("first", kThunkFixture, kVersioned);
    layout.add("second", kThunkFixture, kVersioned);

    const auto run =
        run_probe({caller.string(), "--preload", first.string(), "--preload", second.string()});

    ASSERT_TRUE(started(run));
    ASSERT_EQ(run.status, 0) << run.output;
    ASSERT_EQ(run.values("hsa-runtime"), paths({first, second})) << run.output;
    EXPECT_EQ(run.value("nodes"), "0") << run.output;
    EXPECT_EQ(run.values("after-read"), std::vector<std::string>{"none"}) << run.output;
    EXPECT_NE(run.output.find("libhsa-runtime64 is mapped from 2 directories"), std::string::npos)
        << run.output;
}

// Two runtimes mapped from one directory, one of them through a symlink to it:
// that is still one directory, and it is searched.
TEST_F(wsl_dxg_loader, one_libhsa_runtime64_directory_reached_through_a_symlink_is_one_directory)
{
    SKIP_IF_A_BARE_NAME_RESOLVES_FROM_THE_HOST();

    const auto layout = Layout{};
    const auto caller = layout.add("caller", kRunpathCaller, kCaller);
    const auto hsa    = layout.add("hsa", kPlainCaller, kHsaRuntime);
    layout.add("hsa", kPlainCaller, std::string{kHsaRuntime} + ".0");
    const auto beside = layout.add("hsa", kThunkFixture, kVersioned);
    fs::create_directory_symlink(hsa.parent_path(), layout.root() / "link");
    const auto linked = layout.root() / "link" / (std::string{kHsaRuntime} + ".0");

    const auto run =
        run_probe({caller.string(), "--preload", hsa.string(), "--preload", linked.string()});

    ASSERT_TRUE(started(run));
    ASSERT_EQ(run.status, 0) << run.output;
    ASSERT_EQ(run.values("hsa-runtime"), paths({hsa, linked})) << run.output;
    EXPECT_EQ(run.value("nodes"), "1") << run.output;
    EXPECT_EQ(run.values("after-read"), paths({beside})) << run.output;
}

// --- the harness -----------------------------------------------------------

// What is preloaded for the probe's own startup can be neither a thunk nor an
// HSA runtime: either would become the resident copy or the mapped runtime the
// cases above arrange for themselves.
TEST_F(wsl_dxg_loader, nothing_preloaded_for_the_probe_is_a_thunk_or_an_hsa_runtime)
{
    auto stream = std::istringstream{probe_startup().preload};
    for(auto path = std::string{}; std::getline(stream, path, ':');)
    {
        const auto name = fs::path{path}.filename().string();
        EXPECT_NE(name.rfind("librocdxg", 0), 0) << path;
        EXPECT_NE(name.rfind("libhsa-runtime64", 0), 0) << path;
    }

    const auto run = lookup_with_an_empty_search_path();
    ASSERT_TRUE(started(run));
    EXPECT_EQ(run.values("hsa-runtime"), std::vector<std::string>{"none"}) << run.output;
}

// A probe that never reached main() is a startup failure, not a lookup result.
// The output is what CI printed when the probe's dependencies were unreachable.
TEST(wsl_dxg_loader_harness, a_probe_that_never_reached_main_is_a_startup_failure)
{
    const auto unstarted =
        ProbeRun{127,
                 std::string{kProbe} +
                     ": error while loading shared libraries: librocm_sysdeps_dw.so.1: cannot "
                     "open shared object file: No such file or directory\n"};
    const auto result = started(unstarted);
    EXPECT_FALSE(result);
    EXPECT_NE(std::string{result.message()}.find("did not start (exit 127)"), std::string::npos)
        << result.message();

    EXPECT_FALSE(started(ProbeRun{0, "nodes 0\n"})) << "a clean exit without the marker";
    EXPECT_TRUE(started(ProbeRun{0, "probe started\nnodes 0\n"}));
}

// Unresolved dependencies are read from a traced load and from an aborted one.
TEST(wsl_dxg_loader_harness, unresolved_dependencies_are_read_from_the_dynamic_loader)
{
    const auto traced =
        std::string{"\tlinux-vdso.so.1 (0x00007ffd5a1f2000)\n"
                    "\tlibrocm_sysdeps_dw.so.1 => not found\n"
                    "\tlibamd_comgr.so.3 => /opt/rocm/lib/libamd_comgr.so.3 (0x00007f0e10000000)\n"
                    "\tlibrocm_sysdeps_drm.so.2 => not found\n"};
    EXPECT_EQ(unresolved_dependencies(traced),
              (std::vector<std::string>{"librocm_sysdeps_dw.so.1", "librocm_sysdeps_drm.so.2"}));

    const auto aborted = std::string{kProbe} +
                         ": error while loading shared libraries: librocm_sysdeps_dw.so.1: "
                         "cannot open shared object file: No such file or directory\n";
    EXPECT_EQ(unresolved_dependencies(aborted),
              std::vector<std::string>{"librocm_sysdeps_dw.so.1"});

    EXPECT_TRUE(
        unresolved_dependencies("\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x1)\n").empty());
}
