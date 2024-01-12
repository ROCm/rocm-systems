// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/counters/evaluate_ast.hpp"
#include <fmt/core.h>

#include <exception>
#include <optional>

#include <numeric>
#include <stdexcept>
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/counters/dimensions.hpp"
#include "lib/rocprofiler-sdk/counters/parser/reader.hpp"

namespace rocprofiler
{
namespace counters
{
namespace
{
ReduceOperation
get_reduce_op_type_from_string(const std::string& op)
{
    static const std::unordered_map<std::string, ReduceOperation> reduce_op_string_to_type = {
        {"min", REDUCE_MIN}, {"max", REDUCE_MAX}, {"sum", REDUCE_SUM}, {"avr", REDUCE_AVG}};

    ReduceOperation type = REDUCE_NONE;
    if(op.empty()) return REDUCE_NONE;
    const auto* reduce_op_type = rocprofiler::common::get_val(reduce_op_string_to_type, op);
    if(reduce_op_type) type = *reduce_op_type;
    return type;
}

std::vector<rocprofiler_record_counter_t>*
perform_reduction(ReduceOperation reduce_op, std::vector<rocprofiler_record_counter_t>* input_array)
{
    rocprofiler_record_counter_t result{.id = 0, .counter_value = 0};
    if(input_array->empty()) return input_array;
    switch(reduce_op)
    {
        case REDUCE_NONE: break;
        case REDUCE_MIN:
        {
            result =
                *std::min_element(input_array->begin(), input_array->end(), [](auto& a, auto& b) {
                    return a.counter_value < b.counter_value;
                });
            break;
        }
        case REDUCE_MAX:
        {
            result =
                *std::max_element(input_array->begin(), input_array->end(), [](auto& a, auto& b) {
                    return a.counter_value > b.counter_value;
                });
            break;
        }
        case REDUCE_SUM:
        {
            result = std::accumulate(input_array->begin(),
                                     input_array->end(),
                                     rocprofiler_record_counter_t{.id = 0, .counter_value = 0},
                                     [](auto& a, auto& b) {
                                         return rocprofiler_record_counter_t{
                                             .id            = a.id,
                                             .counter_value = a.counter_value + b.counter_value};
                                     });
            break;
        }
        case REDUCE_AVG:
        {
            result = std::accumulate(input_array->begin(),
                                     input_array->end(),
                                     rocprofiler_record_counter_t{.id = 0, .counter_value = 0},
                                     [](auto& a, auto& b) {
                                         return rocprofiler_record_counter_t{
                                             .id            = a.id,
                                             .counter_value = a.counter_value + b.counter_value};
                                     });
            result.counter_value /= input_array->size();
            break;
        }
    }
    input_array->clear();
    input_array->push_back(result);
    set_dim_in_rec(input_array->begin()->id, ROCPROFILER_DIMENSION_NONE, 0);
    return input_array;
}

}  // namespace

const std::unordered_map<std::string, EvaluateASTMap>&
get_ast_map()
{
    static std::unordered_map<std::string, EvaluateASTMap> ast_map = []() {
        std::unordered_map<std::string, EvaluateASTMap> data;
        const auto& metric_map = *CHECK_NOTNULL(counters::getMetricMap());
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
                    auto& evaluate_ast_node =
                        eval_map
                            .emplace(metric.name(),
                                     EvaluateAST({.handle = metric.id()}, by_name, *ast, gfx))
                            .first->second;
                    evaluate_ast_node.validate_raw_ast(
                        by_name);  // TODO: refactor and consolidate internal post-construction
                                   // logic as a Finish() method
                } catch(std::exception& e)
                {
                    LOG(ERROR) << e.what();
                    throw std::runtime_error(
                        fmt::format("AST was not generated for {}:{}", gfx, metric.name()));
                }
                yy_delete_buffer(buf);
                delete ast;
            }

            for(auto& [name, ast] : eval_map)
            {
                ast.expand_derived(eval_map);
            }
        }

        return data;
    }();
    return ast_map;
}

std::optional<std::set<Metric>>
get_required_hardware_counters(const std::unordered_map<std::string, EvaluateASTMap>& asts,
                               const std::string&                                     agent,
                               const Metric&                                          metric)
{
    const auto* agent_map = rocprofiler::common::get_val(asts, agent);
    if(!agent_map) return std::nullopt;
    const auto* counter_ast = rocprofiler::common::get_val(*agent_map, metric.name());
    if(!counter_ast) return std::nullopt;

    std::set<Metric> required_counters;
    counter_ast->get_required_counters(*agent_map, required_counters);
    return required_counters;
}

EvaluateAST::EvaluateAST(rocprofiler_counter_id_t                       out_id,
                         const std::unordered_map<std::string, Metric>& metrics,
                         const RawAST&                                  ast,
                         std::string                                    agent)
: _type(ast.type)
, _reduce_op(get_reduce_op_type_from_string(ast.reduce_op))
, _agent(std::move(agent))
, _reduce_dimension_set(ast.reduce_dimension_set)
, _out_id(out_id)
{
    if(_type == NodeType::REFERENCE_NODE)
    {
        try
        {
            _metric = metrics.at(std::get<std::string>(ast.value));
        } catch(std::exception& e)
        {
            throw std::runtime_error(
                fmt::format("Unable to lookup metric {}", std::get<std::string>(ast.value)));
        }
    }

    if(_type == NodeType::NUMBER_NODE)
    {
        _raw_value = std::get<int64_t>(ast.value);
        _static_value.push_back(
            {.id = 0, .counter_value = static_cast<double>(std::get<int64_t>(ast.value))});
    }

    for(const auto& nextAst : ast.counter_set)
    {
        _children.emplace_back(_out_id, metrics, *nextAst, _agent);
    }
}

DimensionTypes
EvaluateAST::set_dimensions()
{
    if(_dimension_types != DIMENSION_LAST)
    {
        return _dimension_types;
    }

    auto get_dim_types = [&](auto& metric) {
        int dim_types = 0;
        for(const auto& dim : getBlockDimensions(_agent, metric))
        {
            dim_types |= (dim.type() != ROCPROFILER_DIMENSION_NONE) ? (0x1 << dim.type()) : 0;
        }
        return static_cast<DimensionTypes>(dim_types);
    };

    switch(_type)
    {
        case NONE:
        case RANGE_NODE:
        case CONSTANT_NODE:
        case NUMBER_NODE: break;
        case ADDITION_NODE:
        case SUBTRACTION_NODE:
        case MULTIPLY_NODE:
        case DIVIDE_NODE:
        {
            if(_children[0].set_dimensions() != _children[1].set_dimensions() &&
               _children[0].type() != NUMBER_NODE && _children[1].type() != NUMBER_NODE)
                throw std::runtime_error(fmt::format("Dimension mis-mismatch: {} and {}",
                                                     _children[0].metric(),
                                                     _children[1].metric()));
            _dimension_types = (_children[0].type() != NUMBER_NODE) ? _children[0].set_dimensions()
                                                                    : _children[1].set_dimensions();
        }
        break;
        case REFERENCE_NODE:
        {
            _dimension_types = get_dim_types(_metric);
        }
        break;
        case REDUCE_NODE:
        {
            // There is only one child node in case of REDUCE_NODE and that
            // child node denotes the expression on which the reduce is applied.
            // The resulting dimension of REDUCE_NODE will be the child's dimension
            // minus the dimensions specified in the reduce_dimension_set.
            int original_dim  = static_cast<int>(_children[0].set_dimensions());
            int turn_off_dims = 0;
            for(auto dim : _reduce_dimension_set)
            {
                turn_off_dims |= (dim != ROCPROFILER_DIMENSION_NONE) ? (0x1 << dim) : 1;
            }
            int final_dims   = _reduce_dimension_set.empty() ? ROCPROFILER_DIMENSION_NONE
                                                             : (original_dim & ~turn_off_dims);
            _dimension_types = static_cast<DimensionTypes>(final_dims);
        }
        break;
        case SELECT_NODE:
        {
            // TODO: future scope
        }
        break;
    }
    return _dimension_types;
}

void
EvaluateAST::get_required_counters(const std::unordered_map<std::string, EvaluateAST>& asts,
                                   std::set<Metric>& counters) const
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
        // TODO: Add guards against infinite recursion
        return;
    }

    for(const auto& child : children())
    {
        child.get_required_counters(asts, counters);
    }
}

bool
EvaluateAST::validate_raw_ast(const std::unordered_map<std::string, Metric>& metrics)
{
    bool ret = true;

    try
    {
        switch(_type)
        {
            case NONE:
            case RANGE_NODE:
            case CONSTANT_NODE:
            case NUMBER_NODE: break;
            case ADDITION_NODE:
            case SUBTRACTION_NODE:
            case MULTIPLY_NODE:
            case DIVIDE_NODE:
            {
                // For arithmetic operations '+' '-' '*' '/' check if
                // dimensions of both operands are matching. (handled in set_dimensions())
                for(auto& child : _children)
                {
                    child.validate_raw_ast(metrics);
                }
            }
            break;
            case REFERENCE_NODE:
            {
                // handled in constructor
            }
            break;
            case REDUCE_NODE:
            {
                // Future TODO
                // Check #1 : Should be applied on a base metric. Derived metric support will be
                // added later. Check #2 : Operation should be a supported operation. Check #3 :
                // Dimensions specified should be valid for this metric and GPU

                // validate the members of RawAST, not the members of this class
            }
            break;
            case SELECT_NODE:
            {
                // Future TODO
                // Check #1 : Should be applied on a base metric. Derived metric support will be
                // added later. Check #2 : Operation should be a supported operation. Check #3 :
                // Dimensions specified should be valid for this metric and GPU. Check #4 :
                // Dimensionindex values should be within limits for this metric and GPU.
            }
            break;
        }
    } catch(std::exception& e)
    {
        throw;
    }

    // Future TODO:
    // check if there are cycles in the graph

    return ret;
}

namespace
{
using property_function_t = int64_t (*)(const rocprofiler_agent_t&);
#define GEN_MAP_ENTRY(name, value)                                                                 \
    {                                                                                              \
        name, property_function_t([](const rocprofiler_agent_t& agent_info) {                      \
            return static_cast<int64_t>(value);                                                    \
        })                                                                                         \
    }

int64_t
get_agent_property(const std::string& property, const rocprofiler_agent_t& agent)
{
    static std::unordered_map<std::string, property_function_t> props = {
        GEN_MAP_ENTRY("cpu_cores_count", agent_info.cpu_cores_count),
        GEN_MAP_ENTRY("simd_count", agent_info.simd_count),
        GEN_MAP_ENTRY("mem_banks_count", agent_info.mem_banks_count),
        GEN_MAP_ENTRY("caches_count", agent_info.caches_count),
        GEN_MAP_ENTRY("io_links_count", agent_info.io_links_count),
        GEN_MAP_ENTRY("cpu_core_id_base", agent_info.cpu_core_id_base),
        GEN_MAP_ENTRY("simd_id_base", agent_info.simd_id_base),
        GEN_MAP_ENTRY("max_waves_per_simd", agent_info.max_waves_per_simd),
        GEN_MAP_ENTRY("lds_size_in_kb", agent_info.lds_size_in_kb),
        GEN_MAP_ENTRY("gds_size_in_kb", agent_info.gds_size_in_kb),
        GEN_MAP_ENTRY("num_gws", agent_info.num_gws),
        GEN_MAP_ENTRY("wave_front_size", agent_info.wave_front_size),
        GEN_MAP_ENTRY("array_count", agent_info.array_count),
        GEN_MAP_ENTRY("simd_arrays_per_engine", agent_info.simd_arrays_per_engine),
        GEN_MAP_ENTRY("cu_per_simd_array", agent_info.cu_per_simd_array),
        GEN_MAP_ENTRY("simd_per_cu", agent_info.simd_per_cu),
        GEN_MAP_ENTRY("max_slots_scratch_cu", agent_info.max_slots_scratch_cu),
        GEN_MAP_ENTRY("gfx_target_version", agent_info.gfx_target_version),
        GEN_MAP_ENTRY("vendor_id", agent_info.vendor_id),
        GEN_MAP_ENTRY("device_id", agent_info.device_id),
        GEN_MAP_ENTRY("location_id", agent_info.location_id),
        GEN_MAP_ENTRY("domain", agent_info.domain),
        GEN_MAP_ENTRY("drm_render_minor", agent_info.drm_render_minor),
        GEN_MAP_ENTRY("hive_id", agent_info.hive_id),
        GEN_MAP_ENTRY("num_sdma_engines", agent_info.num_sdma_engines),
        GEN_MAP_ENTRY("num_sdma_xgmi_engines", agent_info.num_sdma_xgmi_engines),
        GEN_MAP_ENTRY("num_sdma_queues_per_engine", agent_info.num_sdma_queues_per_engine),
        GEN_MAP_ENTRY("num_cp_queues", agent_info.num_cp_queues),
        GEN_MAP_ENTRY("max_engine_clk_ccompute", agent_info.max_engine_clk_ccompute),
    };
    if(const auto* func = rocprofiler::common::get_val(props, property))
    {
        return (*func)(agent);
    }

    LOG(ERROR) << fmt::format("Unsupported special property {}", property);
    return 0.0;
}
}  // namespace

void
EvaluateAST::read_special_counters(
    const rocprofiler_agent_t&        agent,
    const std::set<counters::Metric>& required_special_counters,
    std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>>& out_map)
{
    for(const auto& metric : required_special_counters)
    {
        if(!out_map[metric.id()].empty()) out_map[metric.id()].clear();
        auto& record = out_map[metric.id()].emplace_back();
        set_counter_in_rec(record.id, {.handle = metric.id()});
        set_dim_in_rec(record.id, ROCPROFILER_DIMENSION_NONE, 0);

        record.counter_value = get_agent_property(metric.name(), agent);
    }
}

std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>>
EvaluateAST::read_pkt(const aql::AQLPacketConstruct* pkt_gen, hsa::AQLPacket& pkt)
{
    struct it_data
    {
        std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>>* data;
        const aql::AQLPacketConstruct*                                           pkt_gen;
    };

    std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>> ret;
    if(pkt.empty) return ret;
    it_data aql_data{.data = &ret, .pkt_gen = pkt_gen};
    ;
    hsa_status_t status = hsa_ven_amd_aqlprofile_iterate_data(
        &pkt.profile,
        [](hsa_ven_amd_aqlprofile_info_type_t  info_type,
           hsa_ven_amd_aqlprofile_info_data_t* info_data,
           void*                               data) {
            CHECK(data);
            auto& it = *static_cast<it_data*>(data);
            if(info_type != HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA) return HSA_STATUS_SUCCESS;
            const auto* metric = it.pkt_gen->event_to_metric(info_data->pmc_data.event);
            if(!metric) return HSA_STATUS_SUCCESS;
            auto& vec = it.data->emplace(metric->id(), std::vector<rocprofiler_record_counter_t>{})
                            .first->second;
            auto& next_rec = vec.emplace_back();
            set_counter_in_rec(next_rec.id, {.handle = metric->id()});
            // Actual dimension info needs to be used here in the future
            set_dim_in_rec(next_rec.id, ROCPROFILER_DIMENSION_NONE, vec.size() - 1);
            // Note: in the near future we need to use hw_counter here instead
            next_rec.counter_value = info_data->pmc_data.result;
            return HSA_STATUS_SUCCESS;
        },
        &aql_data);
    CHECK(status == HSA_STATUS_SUCCESS);
    return ret;
}

void
EvaluateAST::set_out_id(std::vector<rocprofiler_record_counter_t>& results) const
{
    for(auto& record : results)
    {
        set_counter_in_rec(record.id, _out_id);
    }
}

void
EvaluateAST::expand_derived(std::unordered_map<std::string, EvaluateAST>& asts)
{
    if(_expanded) return;
    _expanded = true;
    for(auto& child : _children)
    {
        if(auto* ptr = rocprofiler::common::get_val(asts, child.metric().name()))
        {
            ptr->expand_derived(asts);
            child = *ptr;
        }
        else
        {
            child.expand_derived(asts);
        }
    }

    /**
     * This covers cases where a derived metric is not a child at all. I.e.
     * <metric name="MemWrites32B" expr=WRITE_REQ_32B>. This will expand
     * WRITE_REQ_32B to its proper expression.
     */
    if(!_metric.expression().empty())
    {
        if(auto* ptr = rocprofiler::common::get_val(asts, _metric.name()))
        {
            ptr->expand_derived(asts);
            _children  = ptr->children();
            _type      = ptr->type();
            _reduce_op = ptr->reduce_op();
        }
    }
}

// convert to buffer at some point
std::vector<rocprofiler_record_counter_t>*
EvaluateAST::evaluate(
    std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>>& results_map,
    std::vector<std::unique_ptr<std::vector<rocprofiler_record_counter_t>>>& cache)
{
    auto perform_op = [&](auto&& op) {
        auto* r1 = _children.at(0).evaluate(results_map, cache);
        auto* r2 = _children.at(1).evaluate(results_map, cache);

        if(r1->size() < r2->size()) swap(r1, r2);

        cache.emplace_back(std::make_unique<std::vector<rocprofiler_record_counter_t>>());
        *cache.back() = *r1;
        r1            = cache.back().get();

        CHECK(!r1->empty() && !r2->empty());

        if(r2->size() == 1)
        {
            // Special operation on either a number node
            // or special node. This is typically a multiple/divide
            // or some other type of constant op.
            for(auto& val : *r1)
            {
                val = op(val, *r2->begin());
            }
        }
        else if(r2->size() == r1->size())
        {
            // Normal combination
            std::transform(r1->begin(), r1->end(), r2->begin(), r1->begin(), op);
        }
        else
        {
            throw std::runtime_error(
                fmt::format("Mismatched Sizes {}, {}", r1->size(), r2->size()));
        }
        return r1;
    };

    switch(_type)
    {
        case NONE:
        case CONSTANT_NODE:
        case RANGE_NODE: break;
        case NUMBER_NODE: return &_static_value;
        case ADDITION_NODE:
            return perform_op([](auto& a, auto& b) {
                return rocprofiler_record_counter_t{
                    .id = a.id, .counter_value = a.counter_value + b.counter_value};
            });
        case SUBTRACTION_NODE:
            return perform_op([](auto& a, auto& b) {
                return rocprofiler_record_counter_t{
                    .id = a.id, .counter_value = a.counter_value - b.counter_value};
            });
        case MULTIPLY_NODE:
            return perform_op([](auto& a, auto& b) {
                return rocprofiler_record_counter_t{
                    .id = a.id, .counter_value = a.counter_value * b.counter_value};
            });
        case DIVIDE_NODE:
            return perform_op([](auto& a, auto& b) {
                return rocprofiler_record_counter_t{
                    .id = a.id,
                    .counter_value =
                        (b.counter_value == 0 ? 0 : a.counter_value / b.counter_value)};
            });
        case REFERENCE_NODE:
        {
            auto* result = rocprofiler::common::get_val(results_map, _metric.id());
            if(!result)
                throw std::runtime_error(
                    fmt::format("Unable to lookup results for metric {}", _metric.name()));

            return result;
        }
        break;
        case REDUCE_NODE:
        {
            auto* result = rocprofiler::common::get_val(results_map, _children[0]._metric.id());
            if(!result)
                throw std::runtime_error(fmt::format("Unable to lookup results for metric {}",
                                                     _children[0]._metric.name()));

            if(_reduce_op == REDUCE_NONE)
                throw std::runtime_error(fmt::format("Invalid Second argument to reduce(): {}",
                                                     static_cast<int>(_reduce_op)));
            return perform_reduction(_reduce_op, result);
        }
        // Currently unsupported
        case SELECT_NODE: break;
    }

    return nullptr;
}

}  // namespace counters
}  // namespace rocprofiler
