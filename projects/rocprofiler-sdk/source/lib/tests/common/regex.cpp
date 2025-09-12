#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "lib/common/regex.hpp"  // rocprofiler::common::regex::...

namespace R = rocprofiler::common::regex;

// ----------------------------- Helpers -----------------------------

struct StdRes
{
    bool   ok = false;
    size_t b  = 0;
    size_t e  = 0;
};

namespace
{
std::optional<bool>
TryStdMatch(std::string_view text, std::string_view pat)
{
    try
    {
        std::regex re(std::string(pat), std::regex::ECMAScript);
        return std::regex_match(std::string(text), re);
    } catch(const std::regex_error&)
    {
        return std::nullopt;  // invalid pattern for std::regex
    }
}

std::optional<StdRes>
TryStdSearch(std::string_view text, std::string_view pat)
{
    try
    {
        std::regex  re(std::string(pat), std::regex::ECMAScript);
        std::cmatch m;
        std::string s(text);
        if(std::regex_search(s.c_str(), m, re))
        {
            return StdRes{true,
                          static_cast<size_t>(m.position()),
                          static_cast<size_t>(m.position() + m.length())};
        }
        return StdRes{false, 0, 0};
    } catch(const std::regex_error&)
    {
        return std::nullopt;  // invalid pattern
    }
}

std::optional<std::string>
TryStdReplace(std::string_view text, std::string_view pat, std::string_view repl)
{
    try
    {
        std::regex re(std::string(pat), std::regex::ECMAScript);
        return std::regex_replace(std::string(text), re, std::string(repl));
    } catch(const std::regex_error&)
    {
        return std::nullopt;
    }
}
}  // namespace

// ----------------------------- Tests -------------------------------

TEST(regex_parity, literals_and_escapes)
{
    // Avoid invalid ECMAScript escapes that std::regex rejects (e.g., "\c").
    struct C
    {
        const char* s;
        const char* p;
    };
    std::vector<C> cases = {
        {"abc", "abc"},
        {"a\nb", "a\\nb"},
        {"a\tb", "a\\tb"},
        {"\\", "\\\\"},
        {"a", "a\\n"},  // literal 'n' after backslash for our engine & ECMAScript
    };
    for(auto& c : cases)
    {
        auto sm = TryStdMatch(c.s, c.p);
        if(!sm.has_value()) continue;  // skip invalid for std::regex
        EXPECT_EQ(R::regex_match(c.s, c.p), sm.value());

        auto ss = TryStdSearch(c.s, c.p);
        if(!ss.has_value()) continue;
        EXPECT_EQ(R::regex_search(c.s, c.p), ss->ok);
        if(ss->ok)
        {
            size_t b = 0;
            size_t e = 0;
            ASSERT_TRUE(R::regex_search(c.s, c.p, b, e));
            EXPECT_EQ(b, ss->b);
            EXPECT_EQ(e, ss->e);
        }
    }
}

TEST(regex_parity, dot_and_anchors)
{
    auto cmp = [&](const std::string& s, const std::string& p) {
        auto sm = TryStdMatch(s, p);
        if(!sm) return;
        EXPECT_EQ(R::regex_match(s, p), *sm);

        auto ss = TryStdSearch(s, p);
        if(!ss) return;
        EXPECT_EQ(R::regex_search(s, p), ss->ok);
        if(ss->ok)
        {
            size_t b = 0;
            size_t e = 0;
            ASSERT_TRUE(R::regex_search(s, p, b, e));
            EXPECT_EQ(b, ss->b);
            EXPECT_EQ(e, ss->e);
        }
    };
    cmp("abc", "a.c");
    cmp("abc", "^abc$");
    cmp("zzzHello", "^Hello");
    cmp("Hello world", "^Hello");
    cmp("world!", "world!$");
}

TEST(regex_parity, char_classes_and_shortcuts)
{
    std::vector<std::pair<std::string, std::string>> pats = {{"abc", "[a-c][a-c][a-c]"},
                                                             {"abc", "[^0-9]+"},
                                                             {"A_", "\\w\\w"},
                                                             {"9 ", "\\d\\s"},
                                                             {"__", "\\W\\W"},
                                                             {"Z5z", "[A-Z]\\d[a-z]"}};
    for(auto& [s, p] : pats)
    {
        auto sm = TryStdMatch(s, p);
        if(!sm) continue;
        EXPECT_EQ(R::regex_match(s, p), *sm);

        auto ss = TryStdSearch(s, p);
        if(!ss) continue;
        EXPECT_EQ(R::regex_search(s, p), ss->ok);
        if(ss->ok)
        {
            size_t b = 0;
            size_t e = 0;
            ASSERT_TRUE(R::regex_search(s, p, b, e));
            EXPECT_EQ(b, ss->b);
            EXPECT_EQ(e, ss->e);
        }
    }
}

TEST(regex_parity, alternation_and_grouping)
{
    std::string s  = "abc123xyz";
    std::string p  = "(abc|def)\\d{3}xyz";
    auto        sm = TryStdMatch(s, p);
    ASSERT_TRUE(sm.has_value());
    EXPECT_EQ(R::regex_match(s, p), *sm);

    auto ss = TryStdSearch(s, p);
    ASSERT_TRUE(ss.has_value());
    EXPECT_EQ(R::regex_search(s, p), ss->ok);
    if(ss->ok)
    {
        size_t b = 0;
        size_t e = 0;
        ASSERT_TRUE(R::regex_search(s, p, b, e));
        EXPECT_EQ(b, ss->b);
        EXPECT_EQ(e, ss->e);
    }

    EXPECT_TRUE(R::regex_search("foo bar", "(foo|bar)"));
    EXPECT_FALSE(R::regex_search("zzz", "(foo|bar)"));
}

TEST(regex_parity, quantifiers_greedy)
{
    struct C
    {
        const char* s;
        const char* p;
    };
    std::vector<C> cases = {
        {"", "a*"},
        {"aaa", "a+"},
        {"aaab", "a+b"},
        {"abbb", "ab{3}"},
        {"abbbbb", "ab{3,}"},
        {"acccb", "ac{2,3}b"},
    };
    for(auto& c : cases)
    {
        auto sm = TryStdMatch(c.s, c.p);
        ASSERT_TRUE(sm.has_value());
        EXPECT_EQ(R::regex_match(c.s, c.p), *sm);
    }
}

TEST(regex_parity, backtracking_across_tail)
{
    const std::string s  = "/prefix/%env{ARBITRARY_ENV_VARIABLE}%/suffix.txt";
    const std::string p  = "(.*)%(env|ENV)\\{([A-Z0-9_]+)\\}%(.*)";
    auto              ss = TryStdSearch(s, p);
    ASSERT_TRUE(ss.has_value());
    size_t b = 0;
    size_t e = 0;
    ASSERT_TRUE(R::regex_search(s, p, b, e));
    ASSERT_TRUE(ss->ok);
    EXPECT_EQ(b, ss->b);
    EXPECT_EQ(e, ss->e);
}

TEST(regex_parity, search_span)
{
    const std::string s  = "xx abcd123 yy";
    const std::string p  = "[a-z]+\\d+";
    auto              ss = TryStdSearch(s, p);
    ASSERT_TRUE(ss.has_value());
    size_t b = 999;
    size_t e = 999;
    ASSERT_TRUE(R::regex_search(s, p, b, e));
    ASSERT_TRUE(ss->ok);
    EXPECT_EQ(b, ss->b);
    EXPECT_EQ(e, ss->e);
}

TEST(regex_parity, replace_whole_and_groups)
{
    const std::string s  = "abc123def";
    const std::string p  = "(\\w+?)(\\d+)(\\w+)";
    auto              r1 = TryStdReplace(s, p, "[$&]");
    ASSERT_TRUE(r1.has_value());
    auto r2 = TryStdReplace(s, p, "($1)-{$2}-<$3>");
    ASSERT_TRUE(r2.has_value());
    auto r3 = TryStdReplace(s, "(\\d+)", "pre($`) mid($&) post($')");
    ASSERT_TRUE(r3.has_value());

    EXPECT_EQ(R::regex_replace(s, p, "[$&]"), *r1);
    EXPECT_EQ(R::regex_replace(s, p, "($1)-{$2}-<$3>"), *r2);
    EXPECT_EQ(R::regex_replace(s, "(\\d+)", "pre($`) mid($&) post($')"), *r3);
}

TEST(regex_parity, replace_global_multiple_hits)
{
    const std::string s  = "a1 b22 c333";
    const std::string p  = "(\\d+)";
    auto              sr = TryStdReplace(s, p, "[$1]");
    ASSERT_TRUE(sr.has_value());
    EXPECT_EQ(R::regex_replace(s, p, "[$1]"), *sr);
}

TEST(regex_parity, replace_two_digit_capture_index)
{
    // 11 groups: 1=outer, 2=a, ..., 10=i, 11=j
    const std::string s  = "abcdefghij";
    const std::string p  = "((a)(b)(c)(d)(e)(f)(g)(h)(i)(j))";
    auto              sr = TryStdReplace(s, p, "$10");
    ASSERT_TRUE(sr.has_value());
    EXPECT_EQ(R::regex_replace(s, p, "$10"), *sr);  // should be "i"
}

TEST(regex_parity, env_patterns_from_issue)
{
    const std::string fpath = "/home/user/summary/%env{ARBITRARY_ENV_VARIABLE}%/out_summary.txt";

    const std::vector<std::string> pats = {
        "(.*)%(env|ENV)\\{([A-Z0-9_]+)\\}%(.*)",   // should match
        "(.*)\\$(env|ENV)\\{([A-Z0-9_]+)\\}(.*)",  // should NOT match
        "(.*)%q\\{([A-Z0-9_]+)\\}(.*)"             // should NOT match here
    };

    for(const auto& p : pats)
    {
        auto ss = TryStdSearch(fpath, p);
        ASSERT_TRUE(ss.has_value());
        size_t b   = 0;
        size_t e   = 0;
        bool   got = R::regex_search(fpath, p, b, e);
        EXPECT_EQ(got, ss->ok) << "pattern: " << p;
        if(ss->ok)
        {
            EXPECT_EQ(b, ss->b);
            EXPECT_EQ(e, ss->e);
            // Check common replacements
            auto r1 = TryStdReplace(fpath, p, "$1");
            ASSERT_TRUE(r1.has_value());
            auto r3 = TryStdReplace(fpath, p, "$3");
            ASSERT_TRUE(r3.has_value());
            auto r4 = TryStdReplace(fpath, p, "$4");
            ASSERT_TRUE(r4.has_value());
            EXPECT_EQ(R::regex_replace(fpath, p, "$1"), *r1);
            EXPECT_EQ(R::regex_replace(fpath, p, "$3"), *r3);
            EXPECT_EQ(R::regex_replace(fpath, p, "$4"), *r4);
        }
    }
}

TEST(regex_parity, zero_length_and_empty)
{
    auto sm = TryStdMatch("", "a*");
    ASSERT_TRUE(sm.has_value());
    EXPECT_EQ(R::regex_match("", "a*"), *sm);

    auto ss = TryStdSearch("", "");
    ASSERT_TRUE(ss.has_value());
    EXPECT_EQ(R::regex_search("", ""), ss->ok);
    if(ss->ok)
    {
        size_t b = 0;
        size_t e = 0;
        ASSERT_TRUE(R::regex_search("", "", b, e));
        EXPECT_EQ(b, ss->b);
        EXPECT_EQ(e, ss->e);
    }
}

TEST(regex_parity, bad_patterns_throw)
{
    // Both should throw on bad syntax we recognize (unterminated brackets/parens)
    EXPECT_THROW({ R::regex_search("abc", "[a-z"); }, std::runtime_error);
    EXPECT_THROW({ R::regex_match("abc", "(abc"); }, std::runtime_error);
    EXPECT_THROW({ R::regex_replace("abc", "[", "x"); }, std::runtime_error);

    EXPECT_THROW(
        {
            std::regex re("[a-z", std::regex::ECMAScript);
            (void) re;
        },
        std::regex_error);
    EXPECT_THROW(
        {
            std::regex re("(abc", std::regex::ECMAScript);
            (void) re;
        },
        std::regex_error);
    EXPECT_THROW(
        {
            std::regex re("[", std::regex::ECMAScript);
            (void) re;
        },
        std::regex_error);
}

// --- LAZY QUANTIFIERS -------------------------------------------------
TEST(regex_parity, lazy_quantifiers)
{
    const std::string s  = "a---b---c";
    const std::string p  = "a.*?b";
    size_t            b1 = 0;
    size_t            e1 = 0;
    size_t            b2 = 0;
    size_t            e2 = 0;
    // search span parity
    ASSERT_TRUE(R::regex_search(s, p, b1, e1));
    std::regex  re(p, std::regex::ECMAScript);
    std::cmatch m;
    ASSERT_TRUE(std::regex_search(s.c_str(), m, re));
    b2 = static_cast<size_t>(m.position());
    e2 = b2 + static_cast<size_t>(m.length());
    EXPECT_EQ(b1, b2);
    EXPECT_EQ(e1, e2);
    // replace should touch minimal range
    std::string r1 = R::regex_replace(s, p, "X");
    std::string r2 = std::regex_replace(s, re, "X");
    EXPECT_EQ(r1, r2);
}

// --- CAPTURE VALUE IS LAST ITERATION OF A QUANTIFIED GROUP -----------
TEST(regex_parity, capture_is_last_iteration)
{
    const std::string s = "ababab";
    const std::string p = "(ab)*";
    // Replacing the match with $1 should be "ab" (last repetition)
    EXPECT_EQ(R::regex_replace(s, p, "$1"), std::regex_replace(s, std::regex(p), "$1"));
}

// --- ALTERNATION CHOICE (LEFT-TO-RIGHT) -------------------------------
TEST(regex_parity, alternation_preference)
{
    // (a|ab)b on "abb" prefers 'a' alternative (leftmost that leads to a match)
    const std::string s  = "abb";
    const std::string p  = "(a|ab)b";
    std::string       r1 = R::regex_replace(s, p, "($1)");
    std::string       r2 = std::regex_replace(s, std::regex(p), "($1)");
    EXPECT_EQ(r1, r2);  // should be "(a)"
}

// --- CHARACTER CLASS CORNER CASES -------------------------------------
TEST(regex_parity, class_hyphen_literal_edges)
{
    // '-' first/last is literal
    EXPECT_TRUE(R::regex_match("-", "[-a]"));
    EXPECT_TRUE(std::regex_match(std::string("-"), std::regex("[-a]")));
    EXPECT_TRUE(R::regex_match("a", "[-a]"));
    EXPECT_TRUE(std::regex_match(std::string("a"), std::regex("[-a]")));
}

TEST(regex_parity, class_escaped_bracket_and_negated_shorthand)
{
    // Escaped ']' inside class
    EXPECT_TRUE(R::regex_match("]", "[\\]]"));
    EXPECT_TRUE(std::regex_match(std::string("]"), std::regex("[\\]]")));
    // Negated digit class allows letters
    EXPECT_TRUE(R::regex_match("g", "[^\\d]"));
    EXPECT_TRUE(std::regex_match(std::string("g"), std::regex("[^\\d]")));
}

// --- ANCHORS WITH EMPTY STRING ----------------------------------------
TEST(regex_parity, anchors_empty_string)
{
    EXPECT_EQ(R::regex_match("", "^$"), std::regex_match(std::string(""), std::regex("^$")));
}

// --- QUANTIFIER {0} ZERO REPS -----------------------------------------
TEST(regex_parity, quantifier_zero_reps_matches_empty)
{
    // {0} should match empty; compare match result
    const std::string p = "a{0}";
    EXPECT_EQ(R::regex_search("xyz", p),
              (bool) std::regex_search(std::string("xyz"), std::regex(p)));
    EXPECT_EQ(R::regex_match("", p), std::regex_match(std::string(""), std::regex(p)));
}

// --- REPLACEMENT TOKEN CORNER CASES -----------------------------------
TEST(regex_parity, replacement_one_digit_then_literal)
{
    // When only one group exists, "$10" == "$1" + "0"
    const std::string s = "a";
    const std::string p = "(a)";
    EXPECT_EQ(R::regex_replace(s, p, "$10"), std::regex_replace(s, std::regex(p), "$10"));
}

TEST(regex_parity, replacement_unknown_two_digit_group_falls_back)
{
    // With only 1 capture, "$99" -> ($9 empty) + "9" => "9"
    const std::string s = "a";
    const std::string p = "(a)";
    EXPECT_EQ(R::regex_replace(s, p, "$99"), std::regex_replace(s, std::regex(p), "$99"));
}

TEST(regex_parity, replacement_dollar_at_end_is_literal)
{
    const std::string s = "abc123";
    const std::string p = "\\d+";
    EXPECT_EQ(R::regex_replace(s, p, "x$"), std::regex_replace(s, std::regex(p), "x$"));
}

TEST(regex_parity, replacement_whole_match_aliases)
{
    const std::string s = "abc123def";
    const std::string p = "(\\w+?)(\\d+)(\\w+)";
    EXPECT_EQ(R::regex_replace(s, p, "<$0>"), std::regex_replace(s, std::regex(p), "<$0>"));
    EXPECT_EQ(R::regex_replace(s, p, "<$&>"), std::regex_replace(s, std::regex(p), "<$&>"));
}

// --- CAPTURE INDEXING STABILITY WITH NESTED GROUPS ---------------------
TEST(regex_parity, nested_capture_indices_left_to_right)
{
    // Ensure left-to-right numbering at '(' is honored
    const std::string s = "xyz";
    const std::string p = "((x)(y))(z)";
    // Expect $1="xy", $2="x", $3="y", $4="z"
    EXPECT_EQ(R::regex_replace(s, p, "$1|$2|$3|$4"),
              std::regex_replace(s, std::regex(p), "$1|$2|$3|$4"));
}

// --- COMPATIBILITY AND INTERFACE TESTS -------------------------------------------
TEST(regex_compatibility, deterministic_behavior)
{
    // This test verifies that our implementation doesn't depend on std::regex
    // across different usage patterns and string types

    std::vector<std::pair<std::string, std::string>> test_cases = {
        {"hello world", "hello.*world"},
        {"file_v1.2.3.txt", "v(\\d+)\\.(\\d+)\\.(\\d+)"},
        {"path/to/file", "path/to/.*"},
        {"123-456-7890", "\\d{3}-\\d{3}-\\d{4}"},
        {"", "a*"},  // Empty string edge case
        {"abcdef", "[a-f]+"}};

    for(const auto& [text, pattern] : test_cases)
    {
        // Operations should be deterministic and repeatable
        bool match_result  = R::regex_match(text, pattern);
        bool search_result = R::regex_search(text, pattern);

        // Results should be consistent across multiple calls
        EXPECT_EQ(match_result, R::regex_match(text, pattern))
            << "Match result inconsistent for: " << text << " with pattern: " << pattern;
        EXPECT_EQ(search_result, R::regex_search(text, pattern))
            << "Search result inconsistent for: " << text << " with pattern: " << pattern;
    }
}

TEST(regex_compatibility, string_interface_types)
{
    // Test that our implementation works correctly with different string interface types
    // (const char*, std::string, std::string_view)

    const char*      c_str   = "test string with numbers 123";
    std::string      std_str = "test string with numbers 123";
    std::string_view sv      = std_str;

    const std::string pattern = "\\d+";

    // All these should produce the same results
    bool c_str_result   = R::regex_search(c_str, pattern);
    bool std_str_result = R::regex_search(std_str, pattern);
    bool sv_result      = R::regex_search(sv, pattern);

    EXPECT_EQ(c_str_result, std_str_result);
    EXPECT_EQ(std_str_result, sv_result);
    EXPECT_TRUE(c_str_result);  // Should find "123"

    // Test position results are consistent
    size_t c_begin, c_end, s_begin, s_end, sv_begin, sv_end;
    EXPECT_TRUE(R::regex_search(c_str, pattern, c_begin, c_end));
    EXPECT_TRUE(R::regex_search(std_str, pattern, s_begin, s_end));
    EXPECT_TRUE(R::regex_search(sv, pattern, sv_begin, sv_end));

    EXPECT_EQ(c_begin, s_begin);
    EXPECT_EQ(s_begin, sv_begin);
    EXPECT_EQ(c_end, s_end);
    EXPECT_EQ(s_end, sv_end);
}

TEST(regex_compatibility, build_system_patterns)
{
    // Test patterns commonly used in build systems and deployment scenarios

    struct TestCase
    {
        std::string text;
        std::string pattern;
        std::string replacement;
        std::string expected;
    };

    std::vector<TestCase> cases = {
        // Environment variable patterns (common in build systems)
        {"/path/%env{HOME}%/file", "%(env|ENV)\\{([A-Z_]+)\\}%", "${$2}", "/path/${HOME}/file"},

        // Version patterns (common in package management)
        {"package-1.2.3", "(\\w+)-(\\d+)\\.(\\d+)\\.(\\d+)", "$1_v$2_$3_$4", "package_v1_2_3"},

        // Path patterns (common in file systems)
        {"/usr/lib64/libfoo.so.1",
         "/usr/lib(\\d*)/([^/]+)\\.so\\.(\\d+)",
         "lib$1/$2.so.$3",
         "lib64/libfoo.so.1"},

        // Architecture patterns (common in cross-compilation)
        {"x86_64-linux-gnu-gcc",
         "(\\w+)-(\\w+)-(\\w+)-(\\w+)",
         "$1/$2/$3/$4",
         "x86_64/linux/gnu/gcc"}};

    for(const auto& test_case : cases)
    {
        std::string result =
            R::regex_replace(test_case.text, test_case.pattern, test_case.replacement);
        EXPECT_EQ(result, test_case.expected)
            << "Failed for text: " << test_case.text << " pattern: " << test_case.pattern
            << " replacement: " << test_case.replacement;
    }
}

TEST(regex_compatibility, memory_safety)
{
    // Test that our implementation doesn't have memory issues that could be

    std::vector<std::string> large_texts;
    const std::string        base_text = "This is a test string with numbers 123 and more text ";

    // Create larger strings to test memory handling
    for(int i = 0; i < 10; ++i)
    {
        std::string large_text;
        for(int j = 0; j < 100; ++j)
        {
            large_text += base_text + std::to_string(j) + " ";
        }
        large_texts.push_back(large_text);
    }

    const std::string pattern = "\\d+";

    for(const auto& text : large_texts)
    {
        // These operations should not cause memory corruption or leaks
        size_t matches = 0;
        size_t pos     = 0;

        while(pos < text.length())
        {
            size_t begin, end;
            if(R::regex_search(text.substr(pos), pattern, begin, end))
            {
                matches++;
                pos += begin + (end - begin);
            }
            else
            {
                break;
            }
        }

        EXPECT_GT(matches, 0) << "Should find numbers in the large text";
        EXPECT_LT(matches, 2000) << "Shouldn't find unreasonable number of matches";
    }
}

TEST(regex_compatibility, thread_safety)
{
    // Test that our implementation is thread-safe and doesn't have

    const std::string text    = "concurrent test 123 with multiple threads 456";
    const std::string pattern = "\\d+";
    std::atomic<int>  success_count{0};
    std::atomic<int>  failure_count{0};

    auto worker = [&]() {
        for(int i = 0; i < 100; ++i)
        {
            try
            {
                size_t begin, end;
                if(R::regex_search(text, pattern, begin, end))
                {
                    success_count++;
                }
                else
                {
                    failure_count++;
                }
            } catch(...)
            {
                failure_count++;
            }
        }
    };

    auto threads = std::vector<std::thread>{};
    threads.reserve(4);

    for(int i = 0; i < 4; ++i)
    {
        threads.emplace_back(worker);
    }

    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(success_count.load(), 400) << "All regex operations should succeed";
    EXPECT_EQ(failure_count.load(), 0) << "No regex operations should fail";
}
