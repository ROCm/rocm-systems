#include "lib/rocprofiler/counters/evaluate_ast.hpp"

#include <optional>

#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler/counters/parser/reader.hpp"

namespace rocprofiler
{
namespace counters
{
const std::unordered_map<std::string, EvaluateASTMap>&
get_ast_map()
{
    static std::unordered_map<std::string, EvaluateASTMap> ast_map = []() {
        std::unordered_map<std::string, EvaluateASTMap> data;
        const auto&                                     metric_map = counters::getMetricMap();
        for(const auto& [gfx, metrics] : metric_map)
        {
            // TODO: Remove global XML from derrived counters...
            if(gfx == "global") continue;

            std::unordered_map<std::string, Metric> by_name;
            for(const auto& metric : metrics)
            {
                by_name.emplace(metric.name(), metric);
            }

            auto& eval_map = data.emplace(gfx, EvaluateASTMap{}).first->second;
            for(auto& [_, metric] : by_name)
            {
                RawAST* ast = nullptr;
                auto*   buf =
                    yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                               : metric.expression().c_str());
                yyparse(&ast);
                if(!ast)
                {
                    LOG(ERROR) << fmt::format("Unable to parse metric {}", metric);
                    throw std::runtime_error(fmt::format("Unable to parse metric {}", metric));
                }
                try
                {
                    eval_map.emplace(metric.name(), EvaluateAST(by_name, *ast));
                } catch(std::out_of_range& e)
                {
                    throw std::runtime_error(
                        fmt::format("AST was not generated for {}:{}, Counter will be unavailable. "
                                    "Likely cause is a base counter not being defined used in a "
                                    "derrived counter.",
                                    gfx,
                                    metric.name()));
                }
                yy_delete_buffer(buf);
                delete ast;
            }
        }
        return data;
    }();
    return ast_map;
}

std::optional<std::set<Metric>>
get_required_hardware_counters(const std::string& agent, const Metric& metric)
{
    const auto& asts      = get_ast_map();
    const auto* agent_map = rocprofiler::common::get_val(asts, agent);
    if(!agent_map) return std::nullopt;
    const auto* counter_ast = rocprofiler::common::get_val(*agent_map, metric.name());
    if(!counter_ast) return std::nullopt;

    std::set<Metric> required_counters;
    counter_ast->get_required_counters(*agent_map, required_counters);
    return required_counters;
}

}  // namespace counters
}  // namespace rocprofiler
