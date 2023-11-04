#pragma once

#include <exception>
#include <iostream>
#include <set>
#include <unordered_map>

#include "lib/common/utility.hpp"
#include "lib/rocprofiler/aql/packet_construct.hpp"
#include "lib/rocprofiler/counters/dimensions.hpp"
#include "lib/rocprofiler/counters/metrics.hpp"
#include "lib/rocprofiler/counters/parser/raw_ast.hpp"

namespace rocprofiler
{
namespace counters
{
struct metric_result
{
    uint64_t         metric_id;
    std::vector<int> sample_values;
};

enum DimensionTypes
{
    DIMENSION_NONE          = 0,
    DIMENSION_XCC           = 1 << 0,
    DIMENSION_SHADER_ENGINE = 1 << 1,
    DIMENSION_AGENT         = 1 << 2,
    DIMENSION_PMC_CHANNEL   = 1 << 3,
    DIMENSION_CU            = 1 << 4,
    DIMENSION_LAST          = 1 << 5,
};

enum ReduceOperation
{
    REDUCE_NONE,
    REDUCE_MIN,
    REDUCE_MAX,
    REDUCE_SUM,
    REDUCE_AVG,
};

class EvaluateAST
{
public:
    EvaluateAST(const std::unordered_map<std::string, Metric>& metrics,
                const RawAST&                                  ast,
                std::string                                    agent);

    std::vector<rocprofiler_record_counter_t>* evaluate(
        std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>>& results_map);

    DimensionTypes set_dimensions();

    bool validate_raw_ast(const std::unordered_map<std::string, Metric>& metrics);

    void get_required_counters(const std::unordered_map<std::string, EvaluateAST>& asts,
                               std::set<Metric>&                                   counters) const;

    static std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>> read_pkt(
        const aql::AQLPacketConstruct* pkt_gen,
        hsa::AQLPacket&                pkt);

    NodeType                        type() const { return _type; }
    ReduceOperation                 reduce_op() const { return _reduce_op; }
    const std::vector<EvaluateAST>& children() const { return _children; }
    const Metric&                   metric() const { return _metric; }
    DimensionTypes                  dimension_types() const { return _dimension_types; }

private:
    NodeType                                                       _type{NONE};
    ReduceOperation                                                _reduce_op{REDUCE_NONE};
    Metric                                                         _metric;
    double                                                         _raw_value{0};
    std::vector<EvaluateAST>                                       _children;
    std::string                                                    _agent;
    DimensionTypes                                                 _dimension_types{DIMENSION_LAST};
    std::vector<rocprofiler_record_counter_t>                      _static_value;
    std::unordered_set<rocprofiler_profile_counter_instance_types> _reduce_dimension_set;
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

namespace fmt
{
template <>
struct formatter<rocprofiler_record_counter_t>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(rocprofiler_record_counter_t const& data, Ctx& ctx) const
    {
        return fmt::format_to(ctx.out(),
                              "(CounterId: {}, Dimension: {:x}, Value [D]: {}, Value [I]: {})",
                              rocprofiler::counters::rec_to_counter_id(data.id).handle,
                              rocprofiler::counters::rec_to_dim_pos(
                                  data.id, rocprofiler::counters::ROCPROFILER_DIMENSION_NONE),
                              data.derived_counter,
                              data.hw_counter);
    }
};
}  // namespace fmt