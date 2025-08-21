#include <gtest/gtest.h>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

#include "lib/common/regex.hpp"  // rocprofiler::common::regex::...

namespace R = rocprofiler::common::regex;

// ----------------------------- Helpers -----------------------------

struct StdRes
{
    bool   ok = false;
    size_t b = 0, e = 0;
};

static std::optional<bool>
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

static std::optional<StdRes>
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

static std::optional<std::string>
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

// ----------------------------- Tests -------------------------------

TEST(RegexParity, LiteralsAndEscapes)
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
            size_t b = 0, e = 0;
            ASSERT_TRUE(R::regex_search(c.s, c.p, b, e));
            EXPECT_EQ(b, ss->b);
            EXPECT_EQ(e, ss->e);
        }
    }
}

TEST(RegexParity, DotAndAnchors)
{
    auto cmp = [&](std::string s, std::string p) {
        auto sm = TryStdMatch(s, p);
        if(!sm) return;
        EXPECT_EQ(R::regex_match(s, p), *sm);

        auto ss = TryStdSearch(s, p);
        if(!ss) return;
        EXPECT_EQ(R::regex_search(s, p), ss->ok);
        if(ss->ok)
        {
            size_t b = 0, e = 0;
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

TEST(RegexParity, CharClassesAndShortcuts)
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
            size_t b = 0, e = 0;
            ASSERT_TRUE(R::regex_search(s, p, b, e));
            EXPECT_EQ(b, ss->b);
            EXPECT_EQ(e, ss->e);
        }
    }
}

TEST(RegexParity, AlternationAndGrouping)
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
        size_t b = 0, e = 0;
        ASSERT_TRUE(R::regex_search(s, p, b, e));
        EXPECT_EQ(b, ss->b);
        EXPECT_EQ(e, ss->e);
    }

    EXPECT_TRUE(R::regex_search("foo bar", "(foo|bar)"));
    EXPECT_FALSE(R::regex_search("zzz", "(foo|bar)"));
}

TEST(RegexParity, QuantifiersGreedy)
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

TEST(RegexParity, BacktrackingAcrossTail)
{
    const std::string s  = "/prefix/%env{ARBITRARY_ENV_VARIABLE}%/suffix.txt";
    const std::string p  = "(.*)%(env|ENV)\\{([A-Z0-9_]+)\\}%(.*)";
    auto              ss = TryStdSearch(s, p);
    ASSERT_TRUE(ss.has_value());
    size_t b = 0, e = 0;
    ASSERT_TRUE(R::regex_search(s, p, b, e));
    ASSERT_TRUE(ss->ok);
    EXPECT_EQ(b, ss->b);
    EXPECT_EQ(e, ss->e);
}

TEST(RegexParity, SearchSpan)
{
    const std::string s  = "xx abcd123 yy";
    const std::string p  = "[a-z]+\\d+";
    auto              ss = TryStdSearch(s, p);
    ASSERT_TRUE(ss.has_value());
    size_t b = 999, e = 999;
    ASSERT_TRUE(R::regex_search(s, p, b, e));
    ASSERT_TRUE(ss->ok);
    EXPECT_EQ(b, ss->b);
    EXPECT_EQ(e, ss->e);
}

TEST(RegexParity, ReplaceWholeAndGroups)
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

TEST(RegexParity, ReplaceGlobalMultipleHits)
{
    const std::string s  = "a1 b22 c333";
    const std::string p  = "(\\d+)";
    auto              sr = TryStdReplace(s, p, "[$1]");
    ASSERT_TRUE(sr.has_value());
    EXPECT_EQ(R::regex_replace(s, p, "[$1]"), *sr);
}

TEST(RegexParity, ReplaceTwoDigitCaptureIndex)
{
    // 11 groups: 1=outer, 2=a, ..., 10=i, 11=j
    const std::string s  = "abcdefghij";
    const std::string p  = "((a)(b)(c)(d)(e)(f)(g)(h)(i)(j))";
    auto              sr = TryStdReplace(s, p, "$10");
    ASSERT_TRUE(sr.has_value());
    EXPECT_EQ(R::regex_replace(s, p, "$10"), *sr);  // should be "i"
}

TEST(RegexParity, EnvPatternsFromIssue)
{
    const std::string fpath = "/home/user/summary/%env{ARBITRARY_ENV_VARIABLE}%/out_summary.txt";

    const std::vector<std::string> pats = {
        "(.*)%(env|ENV)\\{([A-Z0-9_]+)\\}%(.*)",   // should match
        "(.*)\\$(env|ENV)\\{([A-Z0-9_]+)\\}(.*)",  // should NOT match
        "(.*)%q\\{([A-Z0-9_]+)\\}(.*)"             // should NOT match here
    };

    for(auto& p : pats)
    {
        auto ss = TryStdSearch(fpath, p);
        ASSERT_TRUE(ss.has_value());
        size_t b = 0, e = 0;
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

TEST(RegexParity, ZeroLengthAndEmpty)
{
    auto sm = TryStdMatch("", "a*");
    ASSERT_TRUE(sm.has_value());
    EXPECT_EQ(R::regex_match("", "a*"), *sm);

    auto ss = TryStdSearch("", "");
    ASSERT_TRUE(ss.has_value());
    EXPECT_EQ(R::regex_search("", ""), ss->ok);
    if(ss->ok)
    {
        size_t b = 0, e = 0;
        ASSERT_TRUE(R::regex_search("", "", b, e));
        EXPECT_EQ(b, ss->b);
        EXPECT_EQ(e, ss->e);
    }
}

TEST(RegexErrors, BadPatternsThrow)
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
TEST(RegexParity, LazyQuantifiers)
{
    const std::string s  = "a---b---c";
    const std::string p  = "a.*?b";
    size_t            b1 = 0, e1 = 0, b2 = 0, e2 = 0;
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
TEST(RegexParity, CaptureIsLastIteration)
{
    const std::string s = "ababab";
    const std::string p = "(ab)*";
    // Replacing the match with $1 should be "ab" (last repetition)
    EXPECT_EQ(R::regex_replace(s, p, "$1"), std::regex_replace(s, std::regex(p), "$1"));
}

// --- ALTERNATION CHOICE (LEFT-TO-RIGHT) -------------------------------
TEST(RegexParity, AlternationPreference)
{
    // (a|ab)b on "abb" prefers 'a' alternative (leftmost that leads to a match)
    const std::string s  = "abb";
    const std::string p  = "(a|ab)b";
    std::string       r1 = R::regex_replace(s, p, "($1)");
    std::string       r2 = std::regex_replace(s, std::regex(p), "($1)");
    EXPECT_EQ(r1, r2);  // should be "(a)"
}

// --- CHARACTER CLASS CORNER CASES -------------------------------------
TEST(RegexParity, ClassHyphenLiteralEdges)
{
    // '-' first/last is literal
    EXPECT_TRUE(R::regex_match("-", "[-a]"));
    EXPECT_TRUE(std::regex_match(std::string("-"), std::regex("[-a]")));
    EXPECT_TRUE(R::regex_match("a", "[-a]"));
    EXPECT_TRUE(std::regex_match(std::string("a"), std::regex("[-a]")));
}

TEST(RegexParity, ClassEscapedBracketAndNegatedShorthand)
{
    // Escaped ']' inside class
    EXPECT_TRUE(R::regex_match("]", "[\\]]"));
    EXPECT_TRUE(std::regex_match(std::string("]"), std::regex("[\\]]")));
    // Negated digit class allows letters
    EXPECT_TRUE(R::regex_match("g", "[^\\d]"));
    EXPECT_TRUE(std::regex_match(std::string("g"), std::regex("[^\\d]")));
}

// --- ANCHORS WITH EMPTY STRING ----------------------------------------
TEST(RegexParity, AnchorsEmptyString)
{
    EXPECT_EQ(R::regex_match("", "^$"), std::regex_match(std::string(""), std::regex("^$")));
}

// --- QUANTIFIER {0} ZERO REPS -----------------------------------------
TEST(RegexParity, QuantifierZeroRepsMatchesEmpty)
{
    // {0} should match empty; compare match result
    const std::string p = "a{0}";
    EXPECT_EQ(R::regex_search("xyz", p),
              (bool) std::regex_search(std::string("xyz"), std::regex(p)));
    EXPECT_EQ(R::regex_match("", p), std::regex_match(std::string(""), std::regex(p)));
}

// --- REPLACEMENT TOKEN CORNER CASES -----------------------------------
TEST(RegexParity, ReplacementOneDigitThenLiteral)
{
    // When only one group exists, "$10" == "$1" + "0"
    const std::string s = "a";
    const std::string p = "(a)";
    EXPECT_EQ(R::regex_replace(s, p, "$10"), std::regex_replace(s, std::regex(p), "$10"));
}

TEST(RegexParity, ReplacementUnknownTwoDigitGroupFallsBack)
{
    // With only 1 capture, "$99" -> ($9 empty) + "9" => "9"
    const std::string s = "a";
    const std::string p = "(a)";
    EXPECT_EQ(R::regex_replace(s, p, "$99"), std::regex_replace(s, std::regex(p), "$99"));
}

TEST(RegexParity, ReplacementDollarAtEndIsLiteral)
{
    const std::string s = "abc123";
    const std::string p = "\\d+";
    EXPECT_EQ(R::regex_replace(s, p, "x$"), std::regex_replace(s, std::regex(p), "x$"));
}

TEST(RegexParity, ReplacementWholeMatchAliases)
{
    const std::string s = "abc123def";
    const std::string p = "(\\w+?)(\\d+)(\\w+)";
    EXPECT_EQ(R::regex_replace(s, p, "<$0>"), std::regex_replace(s, std::regex(p), "<$0>"));
    EXPECT_EQ(R::regex_replace(s, p, "<$&>"), std::regex_replace(s, std::regex(p), "<$&>"));
}

// --- CAPTURE INDEXING STABILITY WITH NESTED GROUPS ---------------------
TEST(RegexParity, NestedCaptureIndicesLeftToRight)
{
    // Ensure left-to-right numbering at '(' is honored
    const std::string s = "xyz";
    const std::string p = "((x)(y))(z)";
    // Expect $1="xy", $2="x", $3="y", $4="z"
    EXPECT_EQ(R::regex_replace(s, p, "$1|$2|$3|$4"),
              std::regex_replace(s, std::regex(p), "$1|$2|$3|$4"));
}
