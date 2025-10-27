// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/common/regex.hpp"

#include <algorithm>
#include <cctype>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace common
{
namespace regex
{
// =============================== AST ===============================

struct Node
{
    enum Kind
    {
        LITERAL,
        DOT,
        CLASS,
        ANCHOR_BOL,
        ANCHOR_EOL,
        SEQ,
        ALT,
        QUANT,
        CAP
    } kind;
    char ch = 0;

    struct Class
    {
        std::function<bool(unsigned char)> pred;
    };
    std::optional<Class> cls;  // for CLASS

    std::vector<Node> children;  // for SEQ/ALT

    struct Quant
    {
        std::unique_ptr<Node> sub;
        size_t                min    = 0;
        size_t                max    = std::numeric_limits<size_t>::max();
        bool                  greedy = true;
    };
    std::unique_ptr<Quant> quant;  // for QUANT

    int                   cap_index = -1;  // for CAP (1..N)
    std::unique_ptr<Node> cap_sub;         // for CAP

    // Ctors / simple factories
    explicit Node(Kind k)
    : kind(k)
    {}
    explicit Node(char c)
    : kind(LITERAL)
    , ch(c)
    {}

    static Node dot() { return Node(DOT); }
    static Node bol() { return Node(ANCHOR_BOL); }
    static Node eol() { return Node(ANCHOR_EOL); }

    static Node seq(std::vector<Node> v)
    {
        Node n(SEQ);
        n.children = std::move(v);
        return n;
    }
    static Node alt(std::vector<Node> v)
    {
        Node n(ALT);
        n.children = std::move(v);
        return n;
    }

    static Node make_class(std::function<bool(unsigned char)> p)
    {
        Node n(CLASS);
        n.cls = Class{std::move(p)};
        return n;
    }
    static Node make_quant(Node sub, size_t mi, size_t ma, bool greedy)
    {
        Node n(QUANT);
        n.quant         = std::make_unique<Quant>();
        n.quant->sub    = std::make_unique<Node>(std::move(sub));
        n.quant->min    = mi;
        n.quant->max    = ma;
        n.quant->greedy = greedy;
        return n;
    }
    static Node make_cap(int idx, Node sub)
    {
        Node n(CAP);
        n.cap_index = idx;
        n.cap_sub   = std::make_unique<Node>(std::move(sub));
        return n;
    }
};

// ============================= Parser ==============================

struct Parser
{
    std::string_view pat;
    size_t           i              = 0;
    int              next_cap_index = 1;

    explicit Parser(std::string_view p)
    : pat(p)
    {}

    bool end() const { return i >= pat.size(); }
    char peek() const { return end() ? '\0' : pat[i]; }
    char get() { return end() ? '\0' : pat[i++]; }
    bool eat(char c)
    {
        if(!end() && pat[i] == c)
        {
            ++i;
            return true;
        }
        return false;
    }
    static bool is_digit(char c) { return c >= '0' && c <= '9'; }

    std::vector<std::function<bool(unsigned char)>> special_preds;

    Node parse_escape_in_atom()
    {
        get();
        char e = get();
        if(e == '\0') return Node('\\');
        auto make_cls = [&](auto p) { return Node::make_class(std::move(p)); };
        switch(e)
        {
            case 'd': return make_cls([](unsigned char x) { return std::isdigit(x) != 0; });
            case 'D': return make_cls([](unsigned char x) { return std::isdigit(x) == 0; });
            case 'w':
                return make_cls([](unsigned char x) { return (std::isalnum(x) != 0) || x == '_'; });
            case 'W':
                return make_cls(
                    [](unsigned char x) { return !((std::isalnum(x) != 0) || x == '_'); });
            case 's': return make_cls([](unsigned char x) { return std::isspace(x) != 0; });
            case 'S': return make_cls([](unsigned char x) { return std::isspace(x) == 0; });
            case 'n': return Node('\n');
            case 't': return Node('\t');
            case 'r': return Node('\r');
            case 'f': return Node('\f');
            case 'v': return Node('\v');
            default: return Node(e);
        }
    }

    Node parse_class()
    {
        bool negate = false;
        if(eat('^')) negate = true;
        struct Range
        {
            unsigned char a, b;
        };
        std::vector<Range>         ranges;
        std::vector<unsigned char> singles;
        auto                       add_char = [&](unsigned char c) { singles.push_back(c); };
        bool                       first    = true;
        unsigned char              prev     = 0;
        bool                       has_prev = false;
        while(!end() && peek() != ']')
        {
            unsigned char c;
            if(eat('\\'))
            {
                char e = get();
                if(e == 'd' || e == 'D' || e == 'w' || e == 'W' || e == 's' || e == 'S')
                {
                    switch(e)
                    {
                        case 'd':
                            special_preds.emplace_back(
                                [](unsigned char x) { return std::isdigit(x) != 0; });
                            break;
                        case 'D':
                            special_preds.emplace_back(
                                [](unsigned char x) { return std::isdigit(x) == 0; });
                            break;
                        case 'w':
                            special_preds.emplace_back(
                                [](unsigned char x) { return (std::isalnum(x) != 0) || x == '_'; });
                            break;
                        case 'W':
                            special_preds.emplace_back([](unsigned char x) {
                                return !((std::isalnum(x) != 0) || x == '_');
                            });
                            break;
                        case 's':
                            special_preds.emplace_back(
                                [](unsigned char x) { return std::isspace(x) != 0; });
                            break;
                    }
                    continue;
                }
                else
                    c = static_cast<unsigned char>(e);
            }
            else
                c = static_cast<unsigned char>(get());

            if(!first && c == '-' && peek() != ']' && has_prev)
            {
                unsigned char nxt;
                if(eat('\\'))
                    nxt = static_cast<unsigned char>(get());
                else
                    nxt = static_cast<unsigned char>(get());
                if(prev <= nxt)
                    ranges.push_back({prev, nxt});
                else
                    ranges.push_back({nxt, prev});
                has_prev = false;
                first    = false;
                continue;
            }
            else
            {
                if(has_prev) add_char(prev);
                prev     = c;
                has_prev = true;
            }
            first = false;
        }
        if(has_prev) add_char(prev);
        if(!eat(']')) throw std::runtime_error("Unterminated character class");

        auto rs       = std::move(ranges);
        auto ss       = std::move(singles);
        auto specials = std::move(special_preds);
        auto pred     = [rs, ss, specials, negate](unsigned char x) {
            bool in = false;
            for(const auto& r : rs)
            {
                if(r.a <= x && x <= r.b)
                {
                    in = true;
                    break;
                }
            }
            if(!in)
            {
                for(auto c : ss)
                {
                    if(c == x)
                    {
                        in = true;
                        break;
                    }
                }
            }
            if(!in)
            {
                for(const auto& sp : specials)
                {
                    if(sp(x))
                    {
                        in = true;
                        break;
                    }
                }
            }
            return negate ? !in : in;
        };
        return Node::make_class(pred);
    }

    struct RangeQ
    {
        size_t min, max;
        bool   ok;
    };

    RangeQ parse_brace_quant()
    {
        size_t save = i;
        if(!eat('{')) return {0, 0, false};
        auto read_num = [&]() -> std::optional<size_t> {
            if(end() || !is_digit(peek())) return std::nullopt;
            size_t v = 0;
            while(!end() && is_digit(peek()))
                v = v * 10 + (get() - '0');
            return v;
        };
        auto m = read_num();
        if(!m)
        {
            i = save;
            return {0, 0, false};
        }
        size_t mn = *m;
        size_t mx = mn;
        if(eat('}')) return {mn, mx, true};
        if(!eat(','))
        {
            i = save;
            return {0, 0, false};
        }
        if(peek() == '}')
        {
            get();
            return {mn, std::numeric_limits<size_t>::max(), true};
        }
        auto n = read_num();
        if(!n || !eat('}'))
        {
            i = save;
            return {0, 0, false};
        }
        if(*n < mn)
            std::swap(mn, *n);
        else
            mx = *n;
        return {mn, mx, true};
    }

    Node parse_atom()
    {
        if(end()) throw std::runtime_error("Unexpected end in atom");
        char c = peek();
        if(c == '.')
        {
            get();
            return Node::dot();
        }
        if(c == '^')
        {
            get();
            return Node::bol();
        }
        if(c == '$')
        {
            get();
            return Node::eol();
        }
        if(c == '[')
        {
            get();
            return parse_class();
        }
        if(c == '(')
        {
            get();
            int  idx   = next_cap_index++;  // assign at '(' (left-to-right)
            Node inner = parse_alt();
            if(!eat(')')) throw std::runtime_error("Unmatched '('");
            return Node::make_cap(idx, std::move(inner));
        }
        if(c == '\\') return parse_escape_in_atom();
        get();
        return Node(c);
    }

    Node parse_atom_with_quant()
    {
        Node atom       = parse_atom();
        auto apply_lazy = [&](Node& q) {
            if(eat('?'))
                if(q.kind == Node::QUANT && q.quant) q.quant->greedy = false;
        };
        if(!end())
        {
            if(eat('*'))
            {
                Node q =
                    Node::make_quant(std::move(atom), 0, std::numeric_limits<size_t>::max(), true);
                apply_lazy(q);
                return q;
            }
            if(eat('+'))
            {
                Node q =
                    Node::make_quant(std::move(atom), 1, std::numeric_limits<size_t>::max(), true);
                apply_lazy(q);
                return q;
            }
            if(eat('?'))
            {
                Node q = Node::make_quant(std::move(atom), 0, 1, true);
                apply_lazy(q);
                return q;
            }
            auto br = parse_brace_quant();
            if(br.ok)
            {
                Node q = Node::make_quant(std::move(atom), br.min, br.max, true);
                apply_lazy(q);
                return q;
            }
        }
        return atom;
    }

    Node parse_seq()
    {
        std::vector<Node> v;
        while(!end())
        {
            char c = peek();
            if(c == ')' || c == '|') break;
            v.push_back(parse_atom_with_quant());
        }
        if(v.empty()) return Node::seq(std::move(v));
        if(v.size() == 1) return std::move(v[0]);
        return Node::seq(std::move(v));
    }

    Node parse_alt()
    {
        std::vector<Node> branches;
        branches.push_back(parse_seq());
        while(eat('|'))
            branches.push_back(parse_seq());
        if(branches.size() == 1) return std::move(branches[0]);
        return Node::alt(std::move(branches));
    }

    Node parse_all()
    {
        Node n = parse_alt();
        if(!end()) throw std::runtime_error("Trailing pattern content");
        return n;
    }
};

// ============================= Matchers ============================

struct FastMatcher
{
    const Node&      root;
    std::string_view s;

    struct Key
    {
        const Node* node;
        size_t      idx;
        bool        operator==(const Key& o) const { return node == o.node && idx == o.idx; }
    };
    struct KeyHash
    {
        size_t operator()(const Key& k) const noexcept
        {
            return std::hash<const void*>()(k.node) ^ (std::hash<size_t>()(k.idx) << 1);
        }
    };
    std::unordered_map<Key, std::optional<size_t>, KeyHash> memo;

    FastMatcher(const Node& r, std::string_view sv)
    : root(r)
    , s(sv)
    {}

    std::optional<size_t> match(const Node* n, size_t i)
    {
        Key k{n, i};
        if(auto it = memo.find(k); it != memo.end()) return it->second;
        auto r = match_impl(n, i);
        memo.emplace(k, r);
        return r;
    }

    std::optional<size_t> match_seq_from(const std::vector<Node>& children, size_t k, size_t pos)
    {
        if(k == children.size()) return pos;

        const Node& ch = children[k];

        if(ch.kind != Node::QUANT)
        {
            if(ch.kind == Node::CAP && ch.cap_sub && ch.cap_sub->kind == Node::QUANT)
            {
                const auto& q = *ch.cap_sub->quant;

                std::vector<size_t> ends;
                ends.push_back(pos);  // 0 reps -> pos
                size_t cur   = pos;
                size_t count = 0;
                while(count < q.max)
                {
                    auto r = match(q.sub.get(), cur);
                    if(!r) break;
                    if(*r == cur) break;  // zero-length guard
                    cur = *r;
                    ++count;
                    ends.push_back(cur);
                    if(cur > s.size()) break;
                }

                if(q.greedy)
                {
                    for(size_t used = ends.size(); used-- > 0;)
                    {
                        if(used < q.min) continue;
                        auto tail = match_seq_from(children, k + 1, ends[used]);
                        if(tail) return tail;
                    }
                }
                else
                {
                    for(size_t used = 0; used < ends.size(); ++used)
                    {
                        if(used < q.min) continue;
                        auto tail = match_seq_from(children, k + 1, ends[used]);
                        if(tail) return tail;
                    }
                }
                return std::nullopt;
            }

            auto r = match(&ch, pos);
            if(!r) return std::nullopt;
            return match_seq_from(children, k + 1, *r);
        }

        const auto& q = *ch.quant;

        std::vector<size_t> ends;
        ends.push_back(pos);  // 0 reps -> pos
        size_t cur   = pos;
        size_t count = 0;
        while(count < q.max)
        {
            auto r = match(q.sub.get(), cur);
            if(!r) break;
            if(*r == cur) break;  // zero-length guard
            cur = *r;
            ++count;
            ends.push_back(cur);
            if(cur > s.size()) break;
        }

        if(q.greedy)
        {
            for(size_t used = ends.size(); used-- > 0;)
            {
                if(used < q.min) continue;
                auto tail = match_seq_from(children, k + 1, ends[used]);
                if(tail) return tail;
            }
        }
        else
        {
            for(size_t used = 0; used < ends.size(); ++used)
            {
                if(used < q.min) continue;
                auto tail = match_seq_from(children, k + 1, ends[used]);
                if(tail) return tail;
            }
        }
        return std::nullopt;
    }

    std::optional<size_t> match_impl(const Node* n, size_t i)
    {
        switch(n->kind)
        {
            case Node::LITERAL:
            {
                if(i < s.size() && (unsigned char) s[i] == (unsigned char) n->ch) return i + 1;
                return std::nullopt;
            }
            case Node::DOT:
            {
                if(i < s.size()) return i + 1;
                return std::nullopt;
            }
            case Node::CLASS:
            {
                if(i < s.size() && n->cls && n->cls->pred((unsigned char) s[i])) return i + 1;
                return std::nullopt;
            }
            case Node::ANCHOR_BOL:
            {
                if(i == 0) return i;
                return std::nullopt;
            }
            case Node::ANCHOR_EOL:
            {
                if(i == s.size()) return i;
                return std::nullopt;
            }
            case Node::SEQ:
            {
                return match_seq_from(n->children, 0, i);
            }
            case Node::ALT:
            {
                for(const auto& br : n->children)
                {
                    auto r = match(&br, i);
                    if(r) return r;
                }
                return std::nullopt;
            }
            case Node::QUANT:
            {
                const auto&         q = *n->quant;
                std::vector<size_t> ends;
                ends.push_back(i);  // 0 reps
                size_t pos   = i;
                size_t count = 0;
                while(count < q.max)
                {
                    auto r = match(q.sub.get(), pos);
                    if(!r) break;
                    if(*r == pos) break;  // zero-length guard
                    pos = *r;
                    ++count;
                    ends.push_back(pos);
                    if(pos > s.size()) break;
                }

                if(ends.size() - 1 < q.min) return std::nullopt;

                if(q.greedy)
                    return ends.back();
                else
                    return ends[q.min];
            }
            case Node::CAP:
            {
                return match(n->cap_sub.get(), i);  // fast path ignores recording
            }
        }
        return std::nullopt;
    }

    bool full_match()
    {
        auto r = match(&root, 0);
        return r && *r == s.size();
    }

    std::optional<std::pair<size_t, size_t>> find_first()
    {
        for(size_t pos = 0; pos <= s.size(); ++pos)
        {
            auto end = match(&root, pos);
            if(end) return std::make_pair(pos, *end);
        }
        return std::nullopt;
    }
};

struct CaptureMatcher
{
    const Node&                            root;
    std::string_view                       s;
    std::vector<std::pair<size_t, size_t>> groups;  // [0]=whole

    CaptureMatcher(const Node& r, std::string_view sv, int num_caps)
    : root(r)
    , s(sv)
    , groups(static_cast<size_t>(num_caps) + 1, {std::string::npos, std::string::npos})
    {}

    bool run_from(size_t start)
    {
        auto end = match_node(&root, start);
        if(!end) return false;
        groups[0] = {start, *end};
        return true;
    }

    std::optional<size_t> match_seq_from(const std::vector<Node>& children, size_t k, size_t pos)
    {
        if(k == children.size()) return pos;

        const Node& ch = children[k];

        if(ch.kind != Node::QUANT)
        {
            if(ch.kind == Node::CAP && ch.cap_sub && ch.cap_sub->kind == Node::QUANT)
            {
                const auto& q = *ch.cap_sub->quant;

                std::vector<size_t> ends;
                ends.push_back(pos);
                std::vector<std::vector<std::pair<size_t, size_t>>> snaps;
                snaps.push_back(groups);

                size_t cur   = pos;
                size_t count = 0;
                while(count < q.max)
                {
                    auto saved = groups;
                    auto r     = match_node(q.sub.get(), cur);
                    if(!r)
                    {
                        groups = std::move(saved);
                        break;
                    }
                    if(*r == cur)
                    {
                        groups = std::move(saved);
                        break;
                    }  // zero-length guard
                    cur = *r;
                    ++count;
                    ends.push_back(cur);
                    snaps.push_back(groups);
                    if(cur > s.size()) break;
                }

                if(q.greedy)
                {
                    for(size_t used = ends.size(); used-- > 0;)
                    {
                        if(used < q.min) continue;
                        groups               = snaps[used];
                        groups[ch.cap_index] = {pos, ends[used]};
                        auto tail            = match_seq_from(children, k + 1, ends[used]);
                        if(tail) return tail;
                    }
                }
                else
                {
                    for(size_t used = 0; used < ends.size(); ++used)
                    {
                        if(used < q.min) continue;
                        groups               = snaps[used];
                        groups[ch.cap_index] = {pos, ends[used]};
                        auto tail            = match_seq_from(children, k + 1, ends[used]);
                        if(tail) return tail;
                    }
                }
                return std::nullopt;
            }

            auto r = match_node(&ch, pos);
            if(!r) return std::nullopt;
            return match_seq_from(children, k + 1, *r);
        }

        const auto& q = *ch.quant;

        std::vector<size_t> ends;
        ends.push_back(pos);
        std::vector<std::vector<std::pair<size_t, size_t>>> snaps;
        snaps.push_back(groups);

        size_t cur   = pos;
        size_t count = 0;
        while(count < q.max)
        {
            auto saved = groups;
            auto r     = match_node(q.sub.get(), cur);
            if(!r)
            {
                groups = std::move(saved);
                break;
            }
            if(*r == cur)
            {
                groups = std::move(saved);
                break;
            }  // zero-length guard
            cur = *r;
            ++count;
            ends.push_back(cur);
            snaps.push_back(groups);
            if(cur > s.size()) break;
        }

        if(q.greedy)
        {
            for(size_t used = ends.size(); used-- > 0;)
            {
                if(used < q.min) continue;
                groups    = snaps[used];
                auto tail = match_seq_from(children, k + 1, ends[used]);
                if(tail) return tail;
            }
        }
        else
        {
            for(size_t used = 0; used < ends.size(); ++used)
            {
                if(used < q.min) continue;
                groups    = snaps[used];
                auto tail = match_seq_from(children, k + 1, ends[used]);
                if(tail) return tail;
            }
        }
        return std::nullopt;
    }

    std::optional<size_t> match_node(const Node* n, size_t i)
    {
        switch(n->kind)
        {
            case Node::LITERAL:
            {
                if(i < s.size() && (unsigned char) s[i] == (unsigned char) n->ch) return i + 1;
                return std::nullopt;
            }
            case Node::DOT:
            {
                if(i < s.size()) return i + 1;
                return std::nullopt;
            }
            case Node::CLASS:
            {
                if(i < s.size() && n->cls && n->cls->pred((unsigned char) s[i])) return i + 1;
                return std::nullopt;
            }
            case Node::ANCHOR_BOL:
            {
                if(i == 0) return i;
                return std::nullopt;
            }
            case Node::ANCHOR_EOL:
            {
                if(i == s.size()) return i;
                return std::nullopt;
            }
            case Node::SEQ:
            {
                return match_seq_from(n->children, 0, i);
            }
            case Node::ALT:
            {
                for(const auto& br : n->children)
                {
                    auto saved = groups;
                    auto r     = match_node(&br, i);
                    if(r) return r;
                    groups = std::move(saved);
                }
                return std::nullopt;
            }
            case Node::QUANT:
            {
                const auto&         q = *n->quant;
                std::vector<size_t> ends;
                ends.push_back(i);
                std::vector<std::vector<std::pair<size_t, size_t>>> snaps;
                snaps.push_back(groups);

                size_t pos   = i;
                size_t count = 0;
                while(count < q.max)
                {
                    auto saved = groups;
                    auto r     = match_node(q.sub.get(), pos);
                    if(!r)
                    {
                        groups = std::move(saved);
                        break;
                    }
                    if(*r == pos)
                    {
                        groups = std::move(saved);
                        break;
                    }
                    pos = *r;
                    ++count;
                    ends.push_back(pos);
                    snaps.push_back(groups);
                    if(pos > s.size()) break;
                }

                if(q.greedy)
                {
                    for(size_t k = ends.size(); k-- > 0;)
                    {
                        if(k < q.min) continue;
                        groups = snaps[k];
                        return ends[k];
                    }
                }
                else
                {
                    for(size_t used = 0; used < ends.size(); ++used)
                    {
                        if(used < q.min) continue;
                        groups = snaps[used];
                        return ends[used];
                    }
                }
                return std::nullopt;
            }
            case Node::CAP:
            {
                size_t start_i = i;
                auto   saved   = groups;
                auto   r       = match_node(n->cap_sub.get(), i);
                if(!r)
                {
                    groups = std::move(saved);
                    return std::nullopt;
                }
                groups[n->cap_index] = {start_i, *r};
                return r;
            }
        }
        return std::nullopt;
    }
};

namespace
{
int
count_captures(const Node& n)
{
    switch(n.kind)
    {
        case Node::CAP: return std::max(n.cap_index, count_captures(*n.cap_sub));
        case Node::SEQ:
        case Node::ALT:
        {
            int m = 0;
            for(const auto& c : n.children)
                m = std::max(m, count_captures(c));
            return m;
        }
        case Node::QUANT: return count_captures(*n.quant->sub);
        default: return 0;
    }
}

// Expand replacement with captures for a single match span [b,e)
std::string
expand_replacement(std::string_view                              text,
                   const std::vector<std::pair<size_t, size_t>>& groups,
                   size_t                                        b,
                   size_t                                        e,
                   std::string_view                              repl)
{
    std::string out;
    const int   max_group = static_cast<int>(groups.size()) - 1;  // groups[0] = whole match

    for(size_t i = 0; i < repl.size(); ++i)
    {
        char c = repl[i];

        if(c != '$' || i + 1 >= repl.size())
        {
            out.push_back(c);
            continue;
        }

        char n1 = repl[i + 1];

        // $` and $'
        if(n1 == '`')
        {
            out.append(text.substr(0, b));
            ++i;
            continue;
        }
        if(n1 == '\'')
        {
            out.append(text.substr(e));
            ++i;
            continue;
        }

        // $& or $0 => whole match
        if(n1 == '&' || n1 == '0')
        {
            out.append(text.substr(b, e - b));
            ++i;
            continue;
        }

        // $1..$99 (ECMAScript semantics: if two digits are present, always consume both)
        if(std::isdigit(static_cast<unsigned char>(n1)) != 0)
        {
            int    idx = n1 - '0';
            size_t j   = i + 2;

            if(j < repl.size() && (std::isdigit(static_cast<unsigned char>(repl[j])) != 0))
            {
                int d2 = repl[j] - '0';
                idx    = idx * 10 + d2;  // ALWAYS consume the second digit if present
                ++j;
            }

            if(idx >= 0 && idx <= max_group)
            {
                auto [gb, ge] = groups[static_cast<size_t>(idx)];
                if(gb != std::string::npos && ge != std::string::npos && ge >= gb)
                    out.append(text.substr(gb, ge - gb));
            }

            i = j - 1;  // advance past digits
            continue;
        }

        // Otherwise: treat as literal
        out.push_back('$');
        out.push_back(n1);
        ++i;
    }

    return out;
}
}  // namespace

// ============================ Public API ===========================

bool
regex_match(std::string_view text, std::string_view pattern)
{
    Parser P(pattern);
    Node   ast = P.parse_all();

    // Build ^ (ast) $
    std::vector<Node> seq_nodes;
    seq_nodes.emplace_back(Node::bol());
    seq_nodes.emplace_back(std::move(ast));
    seq_nodes.emplace_back(Node::eol());
    Node wrapped = Node::seq(std::move(seq_nodes));

    FastMatcher M(wrapped, text);
    return M.full_match();
}

bool
regex_search(std::string_view text, std::string_view pattern)
{
    Parser      P(pattern);
    Node        ast = P.parse_all();
    FastMatcher M(ast, text);
    return M.find_first().has_value();
}

bool
regex_search(std::string_view text,
             std::string_view pattern,
             size_t&          match_begin,
             size_t&          match_end)
{
    Parser      P(pattern);
    Node        ast = P.parse_all();
    FastMatcher M(ast, text);
    if(auto r = M.find_first())
    {
        match_begin = r->first;
        match_end   = r->second;
        return true;
    }
    return false;
}

inline std::string
regex_replace(std::string_view text, std::string_view pattern, std::string_view replacement)
{
    Parser    P(pattern);
    Node      ast      = P.parse_all();
    const int num_caps = count_captures(ast);

    std::string  result;
    size_t       cur = 0;
    const size_t n   = text.size();

    while(cur <= n)
    {
        // Find first match at or after 'cur' using CaptureMatcher only
        bool                                   found = false;
        size_t                                 mb    = std::string::npos;
        size_t                                 me    = std::string::npos;
        std::vector<std::pair<size_t, size_t>> groups;

        for(size_t pos = cur; pos <= n; ++pos)
        {
            CaptureMatcher cap(ast, text, num_caps);
            if(cap.run_from(pos))
            {
                auto [b0, e0] = cap.groups[0];
                if(b0 != std::string::npos && e0 != std::string::npos && e0 >= b0)
                {
                    found  = true;
                    mb     = b0;
                    me     = e0;
                    groups = std::move(cap.groups);
                    break;
                }
            }
        }

        if(!found)
        {
            // No more matches; append the remainder and finish
            result.append(text.substr(cur));
            break;
        }

        // Append text before the match
        result.append(text.substr(cur, mb - cur));

        // Expand replacement using these exact groups
        result += expand_replacement(text, groups, mb, me, replacement);

        // Zero-length guard like standard regex_replace
        if(me == mb)
        {
            if(me < n)
            {
                // copy one char and advance, to ensure progress
                result.push_back(text[me]);
                cur = me + 1;
            }
            else
            {
                // at end: done
                break;
            }
        }
        else
        {
            cur = me;
        }
    }

    return result;
}

}  // namespace regex
}  // namespace common
}  // namespace rocprofiler

// Global forwards for convenience
bool
regex_match(std::string_view s, std::string_view p)
{
    return rocprofiler::common::regex::regex_match(s, p);
}
bool
regex_search(std::string_view s, std::string_view p)
{
    return rocprofiler::common::regex::regex_search(s, p);
}
bool
regex_search(std::string_view s, std::string_view p, size_t& b, size_t& e)
{
    return rocprofiler::common::regex::regex_search(s, p, b, e);
}
std::string
regex_replace(std::string_view s, std::string_view p, std::string_view r)
{
    return rocprofiler::common::regex::regex_replace(s, p, r);
}
