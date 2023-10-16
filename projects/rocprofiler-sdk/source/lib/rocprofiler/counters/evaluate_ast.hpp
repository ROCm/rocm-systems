#pragma once

#include <iostream>
#include <set>
#include <unordered_map>

#include "lib/common/utility.hpp"
#include "lib/rocprofiler/counters/metrics.hpp"
#include "lib/rocprofiler/counters/parser/raw_ast.hpp"

namespace rocprofiler
{
namespace counters
{
class EvaluateAST
{
public:
    EvaluateAST(const std::unordered_map<std::string, Metric>& metrics, const RawAST& ast)
    : _type(ast.type)
    , _op(ast.operation)
    {
        if(_type == NodeType::REFERENCE_NODE)
        {
            _metric = metrics.at(std::get<std::string>(ast.value));
            // LOG(ERROR) << fmt::format("CHILD METRIC {}", _metric);
        }

        if(_type == NodeType::NUMBER_NODE)
        {
            _raw_value = std::get<int64_t>(ast.value);
        }

        for(const auto& nextAst : ast.counter_set)
        {
            _children.emplace_back(metrics, *nextAst);
        }
    }

    void get_required_counters(const std::unordered_map<std::string, EvaluateAST>& asts,
                               std::set<Metric>&                                   counters) const
    {
        if(!_metric.empty() && children().empty() && _type != NodeType::NUMBER_NODE)
        {
            // Base counter
            if(_metric.expression().empty())
            {
                counters.insert(_metric);
                return;
            }

            // Derrived Counter
            const auto* expr_ptr = rocprofiler::common::get_val(asts, _metric.name());
            if(!expr_ptr) throw std::runtime_error("could not find derived counter");
            expr_ptr->get_required_counters(asts, counters);
            return;
        }

        for(const auto& child : children())
        {
            child.get_required_counters(asts, counters);
        }
    }

    NodeType                        type() const { return _type; }
    NodeType                        op() const { return _op; }
    const std::vector<EvaluateAST>& children() const { return _children; }
    const Metric&                   metric() const { return _metric; }

private:
    NodeType                 _type{NONE};
    NodeType                 _op{NONE};
    Metric                   _metric;
    double                   _raw_value{0};
    std::vector<EvaluateAST> _children;
};

using EvaluateASTMap = std::unordered_map<std::string, EvaluateAST>;

/**
 * Construct the ASTs for all counters appearing in basic/derrived counter
 * definition files.
 */
const std::unordered_map<std::string, EvaluateASTMap>&
get_ast_map();

/**
 * Get the required basic/hardware counters needed to evaluate a
 * specific metric (may be multiple HW counters if a derrived metric).
 */
std::optional<std::set<Metric>>
get_required_hardware_counters(const std::string& agent, const Metric& metric);
}  // namespace counters
}  // namespace rocprofiler
