#pragma once

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <glog/logging.h>

namespace rocprofiler
{
namespace counters
{
enum NodeType
{
    NONE = 0,
    ADDITION_NODE,
    DIVIDE_NODE,
    MULTIPLY_NODE,
    NUMBER_NODE,
    RANGE_NODE,
    REDUCE_NODE,
    REFERENCE_NODE,
    REFERENCE_SET,
    SELECT_NODE,
    SUBTRACTION_NODE,
};

struct RawAST
{
    // Node type
    NodeType type{NONE};  // Operation to perform on the counter set
    NodeType operation{NONE};

    // Stores either the name or digit dependening on whether this
    // is a name or number
    std::variant<std::monostate, std::string, int64_t> value{std::monostate{}};

    // Counter set of ASTs needed to compute this counter.
    // Operation is applied to all counters in this set.
    std::vector<RawAST*> counter_set;

    // Reference set to remove dimensions (such as shader)
    // from the result. This is a future looking change and
    // will be unsupported in 6.0.
    std::vector<RawAST*> reference_set;

    // Range restriction on this node
    RawAST* range{nullptr};

    ~RawAST()
    {
        auto deleteVec = [](auto& vec) {
            for(auto val : vec)
            {
                delete val;
            }
        };

        deleteVec(reference_set);
        deleteVec(counter_set);
        delete range;
    }

    // Constructors for raw value types
    RawAST(NodeType t, const char* v)
    : type(t)
    , value(std::string{CHECK_NOTNULL(v)})
    {}

    RawAST(NodeType t, int64_t v)
    : type(t)
    , value(v)
    {}

    // Reduce/Select operation constructor. Counter is the counter AST
    // to use for the reduce/select op, op is how to reduce (i.e. SUM,AVG,etc),
    // refs is the reference set AST. This reference set is copied to flatten
    // the AST.
    RawAST(NodeType t, RawAST* counter, const char* op, RawAST* refs = nullptr)
    : type(t)
    , value(std::string{CHECK_NOTNULL(op)})
    , counter_set({counter})
    {
        copy_reference_set(refs);
    }

    RawAST(NodeType t, std::vector<RawAST*> c)
    : type(t)
    , counter_set(std::move(c))
    {}

    // Following two calls are for future reference set settings
    // for select/reduce ops.

    // Referene set constructor, refs is a pointer to an existing
    // reference set when multiple references are given (i.e.
    // shader=X,anotherRef=Y,....).
    RawAST(NodeType t, const char* v, RawAST* r, RawAST* refs = nullptr)
    : type(t)
    , value(std::string{CHECK_NOTNULL(v)})
    , range(r)
    {
        LOG(ERROR) << "BUilding bad ast";
        copy_reference_set(refs);
    }

    // Flattens reference set tree into this node.
    void copy_reference_set(RawAST* ast)
    {
        if(!ast) return;
        reference_set.push_back(ast);
        reference_set.insert(
            reference_set.end(), ast->reference_set.begin(), ast->reference_set.end());
        ast->reference_set.clear();
    }
};
}  // namespace counters
}  // namespace rocprofiler

namespace fmt
{
// fmt::format support for RawAST
template <>
struct formatter<rocprofiler::counters::RawAST>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(rocprofiler::counters::RawAST const& ast, Ctx& ctx) const
    {
        static const std::map<rocprofiler::counters::NodeType, std::string> NodeTypeToString = {
            {rocprofiler::counters::NONE, "NONE"},
            {rocprofiler::counters::ADDITION_NODE, "ADDITION_NODE"},
            {rocprofiler::counters::DIVIDE_NODE, "DIVIDE_NODE"},
            {rocprofiler::counters::MULTIPLY_NODE, "MULTIPLY_NODE"},
            {rocprofiler::counters::NUMBER_NODE, "NUMBER_NODE"},
            {rocprofiler::counters::RANGE_NODE, "RANGE_NODE"},
            {rocprofiler::counters::REDUCE_NODE, "REDUCE_NODE"},
            {rocprofiler::counters::REFERENCE_NODE, "REFERENCE_NODE"},
            {rocprofiler::counters::REFERENCE_SET, "REFERENCE_SET"},
            {rocprofiler::counters::SELECT_NODE, "SELECT_NODE"},
            {rocprofiler::counters::SUBTRACTION_NODE, "SUBTRACTION_NODE"},
        };

        auto out = fmt::format_to(ctx.out(),
                                  "{{\"Type\":\"{}\", \"Operation\":\"{}\",",
                                  NodeTypeToString.at(ast.type),
                                  NodeTypeToString.at(ast.operation));

        if(const auto* string_val = std::get_if<std::string>(&ast.value))
        {
            out = fmt::format_to(out, " \"Value\":\"{}\",", *string_val);
        }
        else if(const auto* int_val = std::get_if<int64_t>(&ast.value))
        {
            out = fmt::format_to(out, " \"Value\":{},", *int_val);
        }

        if(ast.range)
        {
            out = fmt::format_to(out, " \"Range\":{},", *ast.range);
        }

        out = fmt::format_to(out, "\"ReferenceSet\":[");
        for(const auto& ref : ast.reference_set)
        {
            out = fmt::format_to(
                out, "{}{}", *CHECK_NOTNULL(ref), ref == ast.reference_set.back() ? "" : ",");
        }

        out = fmt::format_to(out, "], \"CounterSet\":[");
        for(const auto& ref : ast.counter_set)
        {
            out = fmt::format_to(
                out, "{}{}", *CHECK_NOTNULL(ref), ref == ast.counter_set.back() ? "" : ",");
        }
        return fmt::format_to(out, "]}}");
    }
};
}  // namespace fmt
