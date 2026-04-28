// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "analysis.hpp"

#include "core/demangler.hpp"
#include "fwd.hpp"
#include "module_function.hpp"

#include <BPatch_addressSpace.h>

#include <timemory/components/timing/wall_clock.hpp>

#include <algorithm>
#include <cctype>
#include <csetjmp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <set>
#include <sys/wait.h>
#include <type_traits>
#include <unistd.h>
#include <utility>

namespace analysis
{
namespace
{
using index_vec_t                     = std::vector<std::size_t>;
using token_set_t                     = std::set<std::string>;
using tokens_by_index_t               = std::map<std::size_t, token_set_t>;
constexpr int exec_failure_exit_code  = 127;
constexpr int signal_exit_code_offset = 128;  // shell convention for signal exits
// Minimum length of a family token to be considered for grouping
constexpr std::size_t min_family_token_length = 4;

struct family_group_candidate
{
    std::string token{};
    index_vec_t indices{};
};

sigjmp_buf            guarded_signal_jmp_buf{};
volatile sig_atomic_t guarded_signal_caught = 0;
const char*           signal_message        = nullptr;
std::size_t           signal_message_length = 0;
struct sigaction signal_message_old_segv
{};
struct sigaction signal_message_old_bus
{};

// timemory signal handling is termination-oriented, we need to recover
extern "C" void
guarded_signal_handler(int sig, siginfo_t* /*info*/, void* /*ucontext*/)
{
    guarded_signal_caught = sig;
    siglongjmp(guarded_signal_jmp_buf, 1);
}

extern "C" void
signal_message_handler(int sig, siginfo_t* info, void* ucontext)
{
    if(signal_message && signal_message_length > 0)
    {
        auto written = write(STDERR_FILENO, signal_message, signal_message_length);
        (void) written;
    }

    auto* previous = (sig == SIGBUS) ? &signal_message_old_bus : &signal_message_old_segv;
    if((previous->sa_flags & SA_SIGINFO) != 0 && previous->sa_sigaction)
    {
        previous->sa_sigaction(sig, info, ucontext);
        // If the previous handler returns, preserve signal-style process status.
        _exit(signal_exit_code_offset + sig);
    }

    if(previous->sa_handler != SIG_DFL && previous->sa_handler != SIG_IGN &&
       previous->sa_handler)
    {
        previous->sa_handler(sig);
        // If the previous handler returns, preserve signal-style process status.
        _exit(signal_exit_code_offset + sig);
    }

    sigaction(sig, previous, nullptr);
    kill(getpid(), sig);
    _exit(signal_exit_code_offset + sig);
}

// Runs an operation with a temporary message-only signal handler. If the operation
// crashes, the message is written first and then the previous handler, typically
// timemory's backtrace handler, receives the original signal context. The message
// must outlive the operation and any signal handling; use static storage.
template <typename FuncT>
guarded_result
run_with_signal_message(const char* msg, FuncT&& func)
{
    signal_message        = msg;
    signal_message_length = (msg) ? std::strlen(msg) : 0;

    struct sigaction new_sa
    {};
    std::memset(&new_sa, 0, sizeof(new_sa));
    new_sa.sa_sigaction = &signal_message_handler;
    new_sa.sa_flags     = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&new_sa.sa_mask);

    sigaction(SIGSEGV, &new_sa, &signal_message_old_segv);
    sigaction(SIGBUS, &new_sa, &signal_message_old_bus);

    struct restore_signal_handlers
    {
        ~restore_signal_handlers()
        {
            sigaction(SIGSEGV, &signal_message_old_segv, nullptr);
            sigaction(SIGBUS, &signal_message_old_bus, nullptr);
            signal_message        = nullptr;
            signal_message_length = 0;
        }
    } restore;

    return std::forward<FuncT>(func)();
}

// Runs an operation with temporary SIGSEGV/SIGBUS handlers so analysis can report
// the crash and choose a safer fallback instead of aborting outright.
template <typename FuncT, typename SignalFuncT>
guarded_result
run_with_signal_guard(const char* context, FuncT&& func, SignalFuncT&& on_signal)
{
    // siglongjmp() skips C++ unwinding, so keep guarded callbacks trivially destructible
    static_assert(std::is_trivially_destructible_v<std::decay_t<FuncT>>,
                  "signal-guarded callback must be trivially destructible");
    static_assert(std::is_trivially_destructible_v<std::decay_t<SignalFuncT>>,
                  "signal-guarded signal callback must be trivially destructible");

    guarded_signal_caught = 0;

    struct sigaction new_sa
    {};
    std::memset(&new_sa, 0, sizeof(new_sa));
    new_sa.sa_sigaction = &guarded_signal_handler;
    new_sa.sa_flags     = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&new_sa.sa_mask);

    struct sigaction old_segv
    {};
    struct sigaction old_bus
    {};
    sigaction(SIGSEGV, &new_sa, &old_segv);
    sigaction(SIGBUS, &new_sa, &old_bus);

    guarded_result result = guarded_result::fail;

    if(sigsetjmp(guarded_signal_jmp_buf, 1) == 0)
        result = std::forward<FuncT>(func)();
    else
        result = std::forward<SignalFuncT>(on_signal)();

    sigaction(SIGSEGV, &old_segv, nullptr);
    sigaction(SIGBUS, &old_bus, nullptr);

    if(result == guarded_result::signaled)
    {
        verbprintf(0, "[analysis] caught signal %d (%s) inside %s\n",
                   static_cast<int>(guarded_signal_caught),
                   strsignal(guarded_signal_caught), context);
    }

    return result;
}

std::vector<std::string>
read_proc_cmdline()
{
    std::vector<std::string> args;
    std::ifstream            f{ "/proc/self/cmdline", std::ios::binary };
    if(!f) return args;

    std::string s;
    char        c;
    while(f.get(c))
    {
        if(c == '\0')
        {
            args.emplace_back(std::move(s));
            s.clear();
        }
        else
        {
            s.push_back(c);
        }
    }
    if(!s.empty()) args.emplace_back(std::move(s));
    return args;
}

std::string
regex_escape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 4);
    for(char c : s)
    {
        switch(c)
        {
            case '.':
            case '\\':
            case '+':
            case '*':
            case '?':
            case '(':
            case ')':
            case '[':
            case ']':
            case '{':
            case '}':
            case '^':
            case '$':
            case '|': out.push_back('\\'); break;
            default: break;
        }
        out.push_back(c);
    }
    return out;
}

std::string
build_restrict_regex(const std::vector<procedure_id>& universe,
                     const index_vec_t&               indices)
{
    std::string r = "^(";
    for(std::size_t i = 0; i < indices.size(); ++i)
    {
        if(i > 0) r.push_back('|');
        r += regex_escape(universe.at(indices.at(i)).function_name);
    }
    r += ")$";
    return r;
}

// ------------------------------------------------------------
// Functions used in the name grouping fallback

std::string
index_subset_key(const index_vec_t& indices)
{
    std::string key{};
    for(auto idx : indices)
    {
        key += std::to_string(idx);
        key.push_back(',');
    }
    return key;
}

// For a given function, demangle the name and then tokenize the result
token_set_t
extract_family_tokens(const std::string& name)
{
    auto text = rocprofsys::utility::demangle(name);
    if(text.empty()) text = name;

    auto add_token = [](token_set_t& out, std::string& token) {
        auto has_alphabetic = std::any_of(
            token.begin(), token.end(), [](unsigned char c) { return std::isalpha(c); });
        if(has_alphabetic && token.length() >= min_family_token_length)
        {
            std::transform(
                token.begin(), token.end(), token.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            out.emplace(token);
        }
        token.clear();
    };

    token_set_t out{};
    std::string token{};
    for(char c : text)
    {
        auto uc = static_cast<unsigned char>(c);
        if(std::isalnum(uc) || c == '_')
            token.push_back(c);
        else
            add_token(out, token);
    }
    add_token(out, token);
    return out;
}

bool
is_unhelpful_family_token(const std::string& token)
{
    // Discourage grouping by common tokens
    static const std::set<std::string> generic = { "bool",
                                                   "char",
                                                   "clone",
                                                   "common",
                                                   "connection",
                                                   "const",
                                                   "constprop",
                                                   "direction",
                                                   "emit",
                                                   "error",
                                                   "erroneousiostatementstate",
                                                   "external",
                                                   "externalformattediostatementstate",
                                                   "externallistiostatementstate",
                                                   "externalmisciostatementstate",
                                                   "externalunformattediostatementstate",
                                                   "fortran",
                                                   "formatted",
                                                   "get",
                                                   "handle",
                                                   "input",
                                                   "internal",
                                                   "internalformattediostatementstate",
                                                   "internallistiostatementstate",
                                                   "inquire",
                                                   "inquireiolengthstate",
                                                   "inquirenounitstate",
                                                   "inquireunconnectedfilestate",
                                                   "inquireunitstate",
                                                   "int",
                                                   "io",
                                                   "ioerrorhandler",
                                                   "iostatementstate",
                                                   "isra",
                                                   "list",
                                                   "long",
                                                   "misc",
                                                   "noop",
                                                   "openstatementstate",
                                                   "close",
                                                   "closestatementstate",
                                                   "position",
                                                   "receive",
                                                   "record",
                                                   "referencewrapper",
                                                   "relative",
                                                   "runtime",
                                                   "state",
                                                   "statement",
                                                   "std",
                                                   "unformatted",
                                                   "unit",
                                                   "variant",
                                                   "void" };

    if(token.length() < min_family_token_length) return true;
    if(generic.count(token) > 0) return true;

    // State-machine implementation names are broad and drown out useful tokens
    if(token.size() >= 5 && ((token.rfind("state") == token.size() - 5) ||
                             (token.rfind("helper") == std::string::npos &&
                              token.find("statementstate") != std::string::npos)))
        return true;
    return false;
}

bool
prefer_family_token(const std::string& tokenA, const std::string& tokenB)
{
    auto score = [](const std::string& token) {
        auto penalty = (is_unhelpful_family_token(token)) ? 1000 : 0;
        return penalty + static_cast<int>(token.length());
    };

    auto ls = score(tokenA);
    auto rs = score(tokenB);
    if(ls != rs) return ls < rs;
    return tokenA < tokenB;
}

// Group a set of functions by common family tokens as a last resort
// Find common tokens from left and right subsets to construct new ones
std::vector<family_group_candidate>
build_common_name_groups(const std::vector<procedure_id>& universe,
                         const index_vec_t& parent, const index_vec_t& left,
                         const index_vec_t& right)
{
    // Extract tokens from all functions in the parent
    tokens_by_index_t tokens_by_idx{};
    for(auto idx : parent)
        tokens_by_idx[idx] = extract_family_tokens(universe.at(idx).function_name);

    // Collect every token seen in each half of the failed parent subset
    token_set_t left_tokens{};
    token_set_t right_tokens{};
    for(auto idx : left)
        left_tokens.insert(tokens_by_idx.at(idx).begin(), tokens_by_idx.at(idx).end());
    for(auto idx : right)
        right_tokens.insert(tokens_by_idx.at(idx).begin(), tokens_by_idx.at(idx).end());

    // Shared tokens identify candidate families that cross the passing halves
    std::vector<std::string> common_tokens{};
    std::set_intersection(left_tokens.begin(), left_tokens.end(), right_tokens.begin(),
                          right_tokens.end(), std::back_inserter(common_tokens));

    std::map<std::string, family_group_candidate> dedup{};
    for(const auto& token : common_tokens)
    {
        index_vec_t grouped{};
        bool        has_left  = false;
        bool        has_right = false;

        // Build the new candidate group
        for(auto idx : parent)
        {
            if(tokens_by_idx.at(idx).count(token) == 0) continue;

            grouped.emplace_back(idx);
            has_left  = has_left || std::binary_search(left.begin(), left.end(), idx);
            has_right = has_right || std::binary_search(right.begin(), right.end(), idx);
        }

        // The candidate group should be larger than 1, and contain functions from both
        // halves
        if(grouped.size() <= 1 || grouped.size() >= parent.size()) continue;
        if(!(has_left && has_right)) continue;

        auto key = index_subset_key(grouped);
        auto itr = dedup.find(key);
        if(itr == dedup.end() || prefer_family_token(token, itr->second.token))
            dedup[key] = { token, grouped };
    }

    auto groups = std::vector<family_group_candidate>{};
    groups.reserve(dedup.size());
    for(auto& [_, candidate] : dedup)
        groups.emplace_back(std::move(candidate));

    // Try smaller, better-labeled groups first.
    std::sort(groups.begin(), groups.end(), [](const auto& lhs, const auto& rhs) {
        if(lhs.indices.size() != rhs.indices.size())
            return lhs.indices.size() < rhs.indices.size();
        return prefer_family_token(lhs.token, rhs.token);
    });
    return groups;
}

//
// ------------------------------------------------------------

// Tracks bisection trials over queued procedures
struct function_bisect_state
{
    const std::vector<procedure_id>& queued;
    const std::vector<std::string>&  parent_argv;

    std::size_t                            trials = 0;
    std::vector<std::vector<procedure_id>> failing_subsets{};
    // Needed incase both disjoint subsets of a set fail
    std::vector<std::vector<procedure_id>> fallback_failing_subsets{};
    bool                                   terminated = false;

    std::vector<procedure_id> to_procedures(const index_vec_t& subset) const
    {
        std::vector<procedure_id> procedures{};
        procedures.reserve(subset.size());
        for(auto idx : subset)
            procedures.emplace_back(queued[idx]);
        return procedures;
    }

    void record_failing_subset(const index_vec_t& subset)
    {
        if(subset.empty()) return;
        failing_subsets.emplace_back(to_procedures(subset));
    }

    void record_fallback_failing_subset(const index_vec_t& subset)
    {
        if(subset.empty()) return;
        fallback_failing_subsets.emplace_back(to_procedures(subset));
    }

    trial_result run_subset(const index_vec_t& indices, bool is_bisect = true)
    {
        ++trials;
        auto regex = build_restrict_regex(queued, indices);
        auto first = indices.empty() ? 0 : indices.front();
        auto last  = indices.empty() ? 0 : (indices.back() + 1);
        auto _wc   = tim::component::wall_clock{};
        _wc.start();
        if(is_bisect)
        {
            verbprintf(0, "[analysis] trial %zu: subset [%zu, %zu) (size=%zu)\n", trials,
                       first, last, indices.size());
        }
        else
        {
            // Indices make no sense in name based grouping
            verbprintf(0, "[analysis] trial %zu: regrouped subset (size=%zu)\n", trials,
                       indices.size());
        }

        auto result = analysis::run_trial(parent_argv, regex);
        _wc.stop();
        print_trial_result(trials, result, _wc.get());
        return result;
    }

    bool bisect(const index_vec_t& subset)
    {
        if(terminated) return false;

        // The caller has already verified that this subset fails
        if(subset.size() == 1)
        {
            record_failing_subset(subset);
            return true;
        }

        auto        mid = subset.size() / 2;
        index_vec_t left{ subset.begin(), subset.begin() + mid };
        index_vec_t right{ subset.begin() + mid, subset.end() };

        auto left_result  = run_subset(left);
        auto right_result = run_subset(right);
        if(left_result == trial_result::unexpected ||
           right_result == trial_result::unexpected)
        {
            terminated = true;
            verbprintf(0, "[analysis] terminating analysis due to unexpected trial "
                          "failure\n");
            return false;
        }
        auto left_bad  = (left_result == trial_result::fail);
        auto right_bad = (right_result == trial_result::fail);

        if(left_bad || right_bad)
        {
            bool found_precise_subset = false;
            if(left_bad)
            {
                found_precise_subset = bisect(left) || found_precise_subset;
                if(terminated) return false;
            }
            if(right_bad)
            {
                found_precise_subset = bisect(right) || found_precise_subset;
                if(terminated) return false;
            }
            return found_precise_subset;
        }

        // It may be the case that a set of N functions fail, but both
        // of their subsets pass. In this case, we attempt to group the functions
        // by common family tokens as a last resort
        auto groups = build_common_name_groups(queued, subset, left, right);
        for(const auto& group : groups)
        {
            const auto& token   = group.token;
            const auto& grouped = group.indices;
            verbprintf(0, "[analysis]   regroup candidate token='%s' size=%zu\n",
                       token.c_str(), grouped.size());

            auto grouped_result = run_subset(grouped, false);
            if(grouped_result == trial_result::unexpected)
            {
                terminated = true;
                verbprintf(0, "[analysis] terminating analysis due to unexpected "
                              "regrouped trial failure\n");
                return false;
            }

            if(grouped_result == trial_result::fail)
            {
                verbprintf(0,
                           "[analysis]   -> regrouped family '%s' still crashes; "
                           "recording grouped subset\n",
                           token.c_str());
                record_failing_subset(grouped);
                return true;
            }
        }

        record_fallback_failing_subset(subset);
        return false;
    }
};

// Converts a module_function into a lightweight procedure_id object
procedure_id
to_procedure_id(const module_function& mf)
{
    procedure_id out{};
    out.module_name   = mf.module_name;
    out.function_name = mf.function_name;
    return out;
}

}  // namespace

guarded_result
finalize_insertion_set(address_space_t* addr_space, bool* modified_out,
                       bool debug_analysis)
{
    if(!addr_space) return guarded_result::fail;

    auto finalize = [addr_space, modified_out]() {
        bool modified = true;
        bool ok       = addr_space->finalizeInsertionSet(true, &modified);
        if(modified_out) *modified_out = modified;
        return ok ? guarded_result::pass : guarded_result::fail;
    };

    if(!debug_analysis)
    {
        static constexpr const char* finalize_crash_msg =
            "[rocprof-sys][exe] Error! finalizeInsertionSet crashed. This may be due "
            "to dyninst failing to instrument one or more functions.\n"
            "[rocprof-sys][exe] Re-run with '--debug-analysis' to bisect the queued "
            "functions and report "
            "the offending procedure subset.\n";
        return run_with_signal_message(finalize_crash_msg, finalize);
    }

    return run_with_signal_guard("finalizeInsertionSet", finalize, [modified_out]() {
        if(modified_out) *modified_out = false;
        return guarded_result::signaled;
    });
}

bool
is_analysis_child()
{
    const char* s = std::getenv(child_analysis_env);
    return s && *s && std::strcmp(s, "0") != 0;
}

// Forks a child process to run the target with a given restricted function regex
// Generally, with runtime-instrument, the main time cost is attaching to the process
// via processAttach() and the number of procedures being analyzed
trial_result
run_trial(const std::vector<std::string>& parent_argv, const std::string& restrict_regex)
{
    std::vector<std::string> child_args;
    child_args.reserve(parent_argv.size() + 2);

    bool injected = false;
    for(std::size_t i = 0; i < parent_argv.size(); ++i)
    {
        const auto& a = parent_argv.at(i);
        if(a == "--")
        {
            child_args.emplace_back("-R");
            child_args.emplace_back(restrict_regex);
            injected = true;
            child_args.insert(child_args.end(), parent_argv.begin() + i,
                              parent_argv.end());
            break;
        }

        // Replace rocprof-sys function restrictions before "--" with this trial's
        // subset regex
        if(a == "-R" || a == "--function-restrict")
        {
            if(i + 1 < parent_argv.size() && parent_argv.at(i + 1) != "--") ++i;
            continue;
        }
        if(a.rfind("--function-restrict=", 0) == 0) continue;
        child_args.emplace_back(a);
    }
    if(!injected)
    {
        if(child_args.size() >= 1)
            child_args.insert(child_args.begin() + 1, { "-R", restrict_regex });
        else
            child_args.insert(child_args.end(), { "-R", restrict_regex });
    }

    std::vector<char*> argv_ptrs;
    argv_ptrs.reserve(child_args.size() + 1);
    for(auto& s : child_args)
        argv_ptrs.push_back(const_cast<char*>(s.c_str()));
    argv_ptrs.push_back(nullptr);

    pid_t pid = fork();
    if(pid < 0)
    {
        verbprintf(0, "[analysis] fork failed: %s\n", std::strerror(errno));
        return trial_result::unexpected;
    }

    if(pid == 0)
    {
        const char* output_enabled  = std::getenv(analysis_output_env);
        const bool  suppress_output = (!output_enabled || !*output_enabled ||
                                      std::strcmp(output_enabled, "0") == 0);
        if(suppress_output)
        {
            int fd = open("/dev/null", O_WRONLY);
            if(fd >= 0)
            {
                dup2(fd, STDOUT_FILENO);
                dup2(fd, STDERR_FILENO);
                close(fd);
            }
        }

        setenv(child_analysis_env, "1", 1);
        execvp(argv_ptrs[0], argv_ptrs.data());
        std::fprintf(stderr, "[analysis-child] execvp(%s) failed: %s\n", argv_ptrs[0],
                     std::strerror(errno));
        _exit(exec_failure_exit_code);
    }

    int status = 0;
    while(true)
    {
        pid_t w = waitpid(pid, &status, 0);
        if(w < 0)
        {
            if(errno == EINTR) continue;
            verbprintf(0, "[analysis] waitpid failed: %s\n", std::strerror(errno));
            return trial_result::unexpected;
        }
        break;
    }

    if(WIFSIGNALED(status))
    {
        int sig = WTERMSIG(status);
        verbprintf(1, "[analysis] child pid=%d killed by signal %d (%s)\n", (int) pid,
                   sig, strsignal(sig));
        if(sig == SIGSEGV || sig == SIGBUS || sig == SIGABRT) return trial_result::fail;
        return trial_result::unexpected;
    }

    if(WIFEXITED(status))
    {
        int code = WEXITSTATUS(status);
        if(code == 0) return trial_result::pass;
        if(code == CHILD_ANALYSIS_EXIT) return trial_result::fail;
        verbprintf(0, "[analysis] child pid=%d exited with unexpected code %d\n",
                   (int) pid, code);
        return trial_result::unexpected;
    }

    return trial_result::unexpected;
}

void
print_trial_result(size_t trial_idx, trial_result result, double elapsed_seconds)
{
    switch(result)
    {
        case trial_result::pass:
            verbprintf(0, "[analysis]   trial %zu -> pass (%.3f sec)\n", trial_idx,
                       elapsed_seconds);
            break;
        case trial_result::fail:
            verbprintf(0, "[analysis]   trial %zu -> fail (%.3f sec)\n", trial_idx,
                       elapsed_seconds);
            break;
        case trial_result::unexpected:
            verbprintf(
                0,
                "[analysis]   An unexpected error occurred while running trial %zu "
                "(%.3f sec)\n",
                trial_idx, elapsed_seconds);
            break;
    }
}

void
print_analysis_result(const std::vector<std::vector<procedure_id>>& failing_subsets)
{
    if(failing_subsets.empty()) return;

    verbprintf(0, "[analysis] failing procedure subset(s): %zu\n",
               failing_subsets.size());

    for(std::size_t i = 0; i < failing_subsets.size(); ++i)
    {
        const auto& subset = failing_subsets.at(i);
        if(subset.size() == 1)
        {
            const auto& p = subset.front();
            verbprintf(0, "[analysis]   subset %zu singleton: module=%s function=%s\n",
                       i + 1, p.module_name.c_str(), p.function_name.c_str());
            continue;
        }

        verbprintf(0, "[analysis]   subset %zu cooperative procedures: %zu\n", i + 1,
                   subset.size());
        for(const auto& p : subset)
        {
            verbprintf(0, "[analysis]     cooperative: function=%s\n",
                       p.function_name.c_str());
        }
    }
}

void
run_analysis(analysis_type type, fmodset_t& instrumented_module_functions)
{
    switch(type)
    {
        case analysis_type::insertion_set:
            run_insertion_analysis(instrumented_module_functions);
            return;
    }
}

void
run_insertion_analysis(fmodset_t& instrumented_module_functions)
{
    auto _wc = tim::component::wall_clock{};
    _wc.start();

    if(instrumented_module_functions.empty())
    {
        verbprintf(0, "[analysis] empty queue, nothing to analyze\n");
        return;
    }

    auto queued = std::vector<procedure_id>{};
    queued.reserve(instrumented_module_functions.size());
    for(const auto& mf : instrumented_module_functions)
        queued.emplace_back(to_procedure_id(mf));

    auto parent_argv = read_proc_cmdline();
    if(parent_argv.empty())
    {
        verbprintf(0, "[analysis] could not read /proc/self/cmdline; aborting\n");
        return;
    }

    function_bisect_state state{ queued, parent_argv };

    index_vec_t root{};
    root.reserve(queued.size());
    for(std::size_t i = 0; i < queued.size(); ++i)
        root.emplace_back(i);

    state.bisect(root);  // Core
    if(state.terminated)
    {
        verbprintf(0, "[analysis] analysis terminated before finding a failing subset\n");
        return;
    }

    auto root_recorded_as_fallback = std::any_of(
        state.fallback_failing_subsets.begin(), state.fallback_failing_subsets.end(),
        [&queued](const auto& subset) { return subset.size() == queued.size(); });
    if(root_recorded_as_fallback)
    {
        verbprintf(0, "[analysis] root set failed, but both halves and grouped candidate "
                      "trials passed. The failure may not be due to individual functions "
                      "alone.\n");
    }

    // Singleton + name-regrouped subsets that fail
    auto reported_subsets = state.failing_subsets;
    // Append the case where parent set fails, but both subsets
    // and name grouping attempts pass
    reported_subsets.insert(reported_subsets.end(),
                            state.fallback_failing_subsets.begin(),
                            state.fallback_failing_subsets.end());
    print_analysis_result(reported_subsets);

    verbprintf(0, "[analysis] Excluding the functions from instrumentation via "
                  "--function-exclude is recommended\n");
    _wc.stop();
    verbprintf(0, "[analysis] total wall-clock time: %.3f sec\n", _wc.get());
}

}  // namespace analysis
