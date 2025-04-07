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

#pragma once

#include <set>
#include <unordered_map>

#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/counters/dimensions.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/counters/parser/raw_ast.hpp"

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
    DIMENSION_AID           = 1 << 1,
    DIMENSION_SHADER_ENGINE = 1 << 2,
    DIMENSION_AGENT         = 1 << 3,
    DIMENSION_PMC_CHANNEL   = 1 << 4,
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
    EvaluateAST(rocprofiler_counter_id_t                       out_id,
                const std::unordered_map<std::string, Metric>& metrics,
                const RawAST&                                  ast,
                std::string                                    agent);

    /**
     * @brief Evaluates the AST, returning a pointer to the location where the output
     *        results are stored. The output results will reuse a vector contained in
     *        result map (and the map should be treated as tainted after this call).
     *        For simple base counters, evaluate performs no copy operations. For
     *        derived counters, the number of data copies is contingent on the complexity
     *        of the counter.
     *
     * @param [in] results_map Results decoded from the AQL packet
     * @param [in] cache       Used to store results generated from derived counter
     *                         computations. This is needed to avoid destroying data
     *                         in the result map that may be used by other evaluate calls.
     *
     * @return std::vector<rocprofiler_counter_record_t>* A pointer to the output records.
     *          This pointer SHOULD NOT BE FREE'D/DELETED BY THE CALLER.
     */
    std::vector<rocprofiler_counter_record_t>* evaluate(
        std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_t>>& results_map,
        std::vector<std::unique_ptr<std::vector<rocprofiler_counter_record_t>>>& cache);

    /**
     * @brief Expand derived counter ASTs contained within this AST to full hardware counter
     *        representations.
     *
     * @param [in] asts all ASTs read for this agent.
     */
    void expand_derived(std::unordered_map<std::string, EvaluateAST>& asts);

    /**
     * @brief Set the dimensions for this AST and its sub-nodes. Returns the dimension
     *        of this AST. Can throw if the AST is invalid (i.e. dimension mismatch in
     *        child nodes of this AST). This is done in a recursive fashion.
     *
     * @return std::vector<MetricDimension> dimension of the output of this AST.
     */
    std::vector<MetricDimension> set_dimensions();

    bool validate_raw_ast(const std::unordered_map<std::string, Metric>& metrics);

    /**
     * @brief Get the base hardware counters required to evaluate the expressions in the
     *        AST structure. This is primarily useful if the AST contains derived metrics
     *        which will be converted into the base metrics needed to evaluate the derived.
     *
     * @param [in] asts         All constructed ASTs returned by get_ast_map()
     * @param [out] counters    Base metrics that need to be collected to evaluate this AST
     */
    void get_required_counters(const std::unordered_map<std::string, EvaluateAST>& asts,
                               std::set<Metric>&                                   counters) const;

    /**
     * @brief Read the AQL packet and construct rocprofiler_counter_record_t. This call
     *        does not perform any evaluation, only dumping the packet contents into
     *        rocprofiler_counter_record_t.
     *
     * @param [in] pkt_gen packet generator used to generate the AQL packet. This packet
     *                     generator contains information, such as the ordering of instances
     *                     contained in the return packet, that is required to decode what
     *                     data goes with what base counters.
     * @param [in] pkt     AQL packet structure to decode
     * @return std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_t>> map of
     *         {metric->id(), vector<records>}
     *
     */
    static std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_t>> read_pkt(
        const aql::CounterPacketConstruct* pkt_gen,
        hsa::AQLPacket&                    pkt);

    /**
     * @brief Insert special counter values, such as constants of the agent (i.e. max waves)
     *        and kernel duration into the output map.
     *
     * @param [in] agent                        Agent of the output
     * @param [in] required_special_counters     Special counters that are required for eval
     * @param [out] out_map                     Where the special counter values will be written
     */
    static void read_special_counters(
        const rocprofiler_agent_t&        agent,
        const std::set<counters::Metric>& required_special_counters,
        std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_t>>& out_map);

    NodeType                            type() const { return _type; }
    ReduceOperation                     reduce_op() const { return _reduce_op; }
    const std::vector<EvaluateAST>&     children() const { return _children; }
    const Metric&                       metric() const { return _metric; }
    const std::vector<MetricDimension>& dimension_types() const { return _dimension_types; }

    /**
     * @brief When an evaluation is complete, set the output id of the results. This is called
     *        externally to reduce the number of times the id is set to only the end result.
     *
     * @param [in] results computed results that will have their id modified to be counter _out_id
     */
    void set_out_id(std::vector<rocprofiler_counter_record_t>& results) const;

    const rocprofiler_counter_id_t& out_id() const { return _out_id; }

private:
    NodeType                                                          _type{NONE};
    ReduceOperation                                                   _reduce_op{REDUCE_NONE};
    Metric                                                            _metric;
    double                                                            _raw_value{0};
    std::vector<EvaluateAST>                                          _children;
    std::string                                                       _agent;
    std::vector<MetricDimension>                                      _dimension_types{};
    std::vector<rocprofiler_counter_record_t>                         _static_value;
    std::unordered_set<rocprofiler_profile_counter_instance_types>    _reduce_dimension_set;
    std::map<rocprofiler_profile_counter_instance_types, std::string> _select_dimension_map;
    bool                                                              _expanded{false};
    rocprofiler_counter_id_t                                          _out_id{.handle = 0};
};

using EvaluateASTMap = std::unordered_map<std::string, EvaluateAST>;

struct ASTs
{
    const std::unordered_map<std::string, EvaluateASTMap> arch_to_counter_asts;
};

rocprofiler_status_t
check_ast_generation(std::string_view arch, Metric metric);

/**
 * Construct the ASTs for all counters appearing in basic/derived counter
 * definition files.
 */
std::shared_ptr<const ASTs>
get_ast_map(bool reload = false);

/**
 * Get the required basic/hardware counters needed to evaluate a
 * specific metric (may be multiple HW counters if a derived metric).
 */
std::optional<std::set<Metric>>
get_required_hardware_counters(const std::unordered_map<std::string, EvaluateASTMap>& asts,
                               const std::string&                                     agent,
                               const Metric&                                          metric);
int64_t
get_agent_property(std::string_view property, const rocprofiler_agent_t& agent);
}  // namespace counters
}  // namespace rocprofiler

namespace fmt
{
template <>
struct formatter<rocprofiler_counter_record_t>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(rocprofiler_counter_record_t const& data, Ctx& ctx) const
    {
        return fmt::format_to(ctx.out(),
                              "(CounterId: {}, Dimension: {:x}, Value [D]: {})",
                              rocprofiler::counters::rec_to_counter_id(data.id).handle,
                              rocprofiler::counters::rec_to_dim_pos(
                                  data.id, rocprofiler::counters::ROCPROFILER_DIMENSION_NONE),
                              data.counter_value);
    }
};
}  // namespace fmt
