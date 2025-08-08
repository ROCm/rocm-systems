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

#include "lib/rocprofiler-sdk/counters/evaluate_ast.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/counters/dimensions.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/counters/parser/raw_ast.hpp"
#include "lib/rocprofiler-sdk/counters/parser/reader.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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

void
perform_reduction_to_single_instance(ReduceOperation                            reduce_op,
                                     std::vector<rocprofiler_counter_record_t>* input_array,
                                     rocprofiler_counter_record_t*              result)
{
    switch(reduce_op)
    {
        case REDUCE_NONE: break;
        case REDUCE_MIN:
        {
            *result =
                *std::min_element(input_array->begin(), input_array->end(), [](auto& a, auto& b) {
                    return a.counter_value < b.counter_value;
                });
            break;
        }
        case REDUCE_MAX:
        {
            *result =
                *std::max_element(input_array->begin(), input_array->end(), [](auto& a, auto& b) {
                    return a.counter_value < b.counter_value;
                });
            break;
        }
        case REDUCE_SUM: [[fallthrough]];
        case REDUCE_AVG:
        {
            *result = std::accumulate(
                input_array->begin(),
                input_array->end(),
                rocprofiler_counter_record_t{.id            = input_array->begin()->id,
                                             .counter_value = 0,
                                             .dispatch_id   = input_array->begin()->dispatch_id,
                                             .user_data     = input_array->begin()->user_data,
                                             .agent_id      = input_array->begin()->agent_id},
                [](auto& a, auto& b) {
                    return rocprofiler_counter_record_t{
                        .id            = a.id,
                        .counter_value = a.counter_value + b.counter_value,
                        .dispatch_id   = a.dispatch_id,
                        .user_data     = a.user_data,
                        .agent_id      = a.agent_id};
                });
            if(reduce_op == REDUCE_AVG)
            {
                (*result).counter_value /= input_array->size();
            }
            break;
        }
    }
}

std::vector<rocprofiler_counter_record_t>*
perform_reduction(
    ReduceOperation                                                       reduce_op,
    std::vector<rocprofiler_counter_record_t>*                            input_array,
    const std::unordered_set<rocprofiler_profile_counter_instance_types>& _reduce_dimension_set)
{
    if(input_array->empty()) return input_array;
    if(_reduce_dimension_set.empty() ||
       _reduce_dimension_set.size() == ROCPROFILER_DIMENSION_LAST - 1)
    {
        rocprofiler_counter_record_t result{.id            = 0,
                                            .counter_value = 0,
                                            .dispatch_id   = 0,
                                            .user_data     = {.value = 0},
                                            .agent_id      = input_array->begin()->agent_id};
        perform_reduction_to_single_instance(reduce_op, input_array, &result);
        input_array->clear();
        input_array->push_back(result);
        set_dim_in_rec(input_array->begin()->id, ROCPROFILER_DIMENSION_NONE, 0);
        return input_array;
    }

    std::unordered_map<int64_t, std::vector<rocprofiler_counter_record_t>> rec_groups;
    size_t bit_length = DIM_BIT_LENGTH / ROCPROFILER_DIMENSION_LAST;

    for(auto& rec : *input_array)
    {
        for(auto dim : _reduce_dimension_set)
        {
            int64_t mask_dim = (MAX_64 >> (64 - bit_length)) << ((dim - 1) * bit_length);

            rec.id = rec.id | mask_dim;
            rec.id = rec.id ^ mask_dim;
        }
        rec_groups[rec.id].push_back(rec);
    }

    input_array->clear();
    for(auto& rec_pair : rec_groups)
    {
        rocprofiler_counter_record_t result{.id            = 0,
                                            .counter_value = 0,
                                            .dispatch_id   = 0,
                                            .user_data     = {.value = 0},
                                            .agent_id      = {.handle = 0}};

        perform_reduction_to_single_instance(reduce_op, &rec_pair.second, &result);
        input_array->push_back(result);
    }
    if(input_array->size() == 1)
    {
        set_dim_in_rec(input_array->begin()->id, ROCPROFILER_DIMENSION_NONE, 0);
    }
    return input_array;
}

int64_t
get_int_encoded_dimensions_from_string(const std::string& rangeStr)
{
    int64_t            result = 0;
    std::istringstream iss(rangeStr);
    std::string        token;
    size_t             bit_length = DIM_BIT_LENGTH / ROCPROFILER_DIMENSION_LAST;

    while(std::getline(iss, token, ','))
    {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());

        size_t dash_pos = token.find(':');
        if(dash_pos != std::string::npos)
        {
            throw std::runtime_error(
                fmt::format("Range based selection not supported by Dimension API. only select "
                            "single value for each dimension."));
            int start = std::stoi(token.substr(0, dash_pos));
            int end   = std::stoi(token.substr(dash_pos + 1));
            result |= (1LL << std::min(64, end + 1)) - (1LL << std::max(start, 0));
        }
        else
        {
            int num = std::stoi(token);
            if(num < (1 << bit_length))
            {
                result |= (1LL << num);
            }
            else
            {
                throw std::runtime_error(fmt::format("Dimension value exceeds max allowed."));
            }
        }
    }
    return result;
}

std::vector<rocprofiler_counter_record_t>*
perform_selection(std::map<rocprofiler_profile_counter_instance_types, std::string>& dimension_map,
                  std::vector<rocprofiler_counter_record_t>*                         input_array)
{
    if(input_array->empty()) return input_array;
    for(auto& dim_pair : dimension_map)
    {
        int64_t encoded_dim_values = get_int_encoded_dimensions_from_string(dim_pair.second);
        size_t  bit_length         = DIM_BIT_LENGTH / ROCPROFILER_DIMENSION_LAST;
        int64_t mask = (MAX_64 >> (64 - bit_length)) << ((dim_pair.first - 1) * bit_length);

        input_array->erase(std::remove_if(input_array->begin(),
                                          input_array->end(),
                                          [&](rocprofiler_counter_record_t& rec) {
                                              bool should_remove =
                                                  (encoded_dim_values &
                                                   (1 << rocprofiler::counters::rec_to_dim_pos(
                                                        rec.id, dim_pair.first))) == 0;
                                              if(!should_remove)
                                              {
                                                  rec.id = rec.id | mask;
                                                  rec.id = rec.id ^ mask;
                                              }
                                              return should_remove;
                                          }),
                           input_array->end());
    }

    return input_array;
}

const ASTs
load_asts()
{
    std::unordered_map<std::string, EvaluateASTMap> data;

    auto        mets       = counters::loadMetrics(true);
    const auto& metric_map = mets->arch_to_metric;
    for(const auto& [gfx, metrics] : metric_map)
    {
        // TODO: Remove global XML from derived counters...
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
            auto*   buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                     : metric.expression().c_str());
            yyparse(&ast);
            if(!ast)
            {
                ROCP_ERROR << fmt::format("Unable to parse metric {}", metric);
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
                ROCP_ERROR << e.what();
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

    return {.arch_to_counter_asts = data};
}

}  // namespace

rocprofiler_status_t
check_ast_generation(std::string_view arch, Metric metric)
{
    auto        metrics = counters::loadMetrics();
    const auto* metric_list =
        rocprofiler::common::get_val(metrics->arch_to_metric, std::string(arch));
    if(!metric_list) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    RawAST* ast = nullptr;
    auto*   buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                             : metric.expression().c_str());

    auto delete_ast = [&]() {
        yy_delete_buffer(buf);
        delete ast;
    };

    yyparse(&ast);
    if(!ast)
    {
        if(buf) yy_delete_buffer(buf);
        ROCP_ERROR << fmt::format("Unable to parse metric {}", metric);
        return ROCPROFILER_STATUS_ERROR_AST_GENERATION_FAILED;
    }

    std::unordered_map<std::string, Metric> by_name;
    for(const auto& existing_metric : *metric_list)
    {
        by_name.emplace(existing_metric.name(), existing_metric);
    }

    if(!by_name.emplace(metric.name(), metric).second)
    {
        delete_ast();
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        auto evaluate_ast_node =
            EvaluateAST({.handle = metric.id()}, by_name, *ast, std::string(arch));
        evaluate_ast_node.validate_raw_ast(by_name);
    } catch(std::exception& e)
    {
        ROCP_ERROR << fmt::format("Unable to generate AST for {} error: {}", metric, e.what());
        delete_ast();
        return ROCPROFILER_STATUS_ERROR_AST_GENERATION_FAILED;
    }

    delete_ast();
    return ROCPROFILER_STATUS_SUCCESS;
}

std::shared_ptr<const ASTs>
get_ast_map(bool reload)
{
    using ASTSync             = common::Synchronized<std::shared_ptr<const ASTs>>;
    static ASTSync*& ast_data = common::static_object<ASTSync>::construct(
        [&]() { return std::make_shared<const ASTs>(load_asts()); }());

    if(!reload)
    {
        return ast_data->rlock([](const auto& data) {
            CHECK(data);
            return data;
        });
    }

    return ast_data->wlock([&](auto& data) {
        data = std::make_shared<const ASTs>(load_asts());
        CHECK(data);
        return data;
    });
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
, _select_dimension_map(ast.select_dimension_map)
, _out_id(out_id)
{
    if(_type == NodeType::REFERENCE_NODE || _type == NodeType::ACCUMULATE_NODE)
    {
        try
        {
            _metric = metrics.at(std::get<std::string>(ast.value));
            if(_type == NodeType::ACCUMULATE_NODE)
            {
                ROCP_FATAL_IF(ast.accumulate_op != ACCUMULATE_OP_TYPE::NONE &&
                              _metric.block() != "SQ")
                    << fmt::format("Accumulate High_RES/Low_RES only works for counters from SQ "
                                   "block: invalid operation on {} counter.",
                                   _metric.name());
                _metric.setflags(static_cast<int>(ast.accumulate_op));
            }
        } catch(std::exception& e)
        {
            throw std::runtime_error(
                fmt::format("Unable to lookup metric {}", std::get<std::string>(ast.value)));
        }
    }

    if(_type == NodeType::NUMBER_NODE)
    {
        _raw_value = std::get<int64_t>(ast.value);
        _static_value.push_back({.id            = 0,
                                 .counter_value = static_cast<double>(std::get<int64_t>(ast.value)),
                                 .dispatch_id   = 0,
                                 .user_data     = {.value = 0},
                                 .agent_id      = {.handle = 0}});
    }

    for(const auto& nextAst : ast.counter_set)
    {
        _children.emplace_back(_out_id, metrics, *nextAst, _agent);
    }
}

std::vector<MetricDimension>
EvaluateAST::set_dimensions()
{
    if(!_dimension_types.empty())
    {
        return _dimension_types;
    }

    auto get_dim_types = [&](auto& metric) { return getBlockDimensions(_agent, metric); };

    switch(_type)
    {
        case NONE:
        case RANGE_NODE:
        case CONSTANT_NODE:
        case NUMBER_NODE:
        {
            _dimension_types =
                std::vector<MetricDimension>{{dimension_map().at(ROCPROFILER_DIMENSION_INSTANCE),
                                              1,
                                              ROCPROFILER_DIMENSION_INSTANCE}};
        }
        break;
        case ADDITION_NODE:
        case SUBTRACTION_NODE:
        case MULTIPLY_NODE:
        case DIVIDE_NODE:
        {
            auto first  = _children[0].set_dimensions();
            auto second = _children[1].set_dimensions();
            // - first.size() > 1 && second.size() > 1
            // This is an explicit compatibility change to allow existing integer * COUNTER
            // derived counters to function
            if(first != second && first.size() > 1 && second.size() > 1)
                throw std::runtime_error(
                    fmt::format("Dimension mis-mismatch: {} (dims: {}) and {} (dims: {})",
                                _children[0].metric(),
                                fmt::join(_children[0].set_dimensions(), ","),
                                _children[1].metric(),
                                fmt::join(_children[1].set_dimensions(), ",")));
            _dimension_types = first.size() > second.size() ? first : second;
        }
        break;
        case ACCUMULATE_NODE:
        case REFERENCE_NODE:
        {
            _dimension_types = get_dim_types(_metric);
        }
        break;
        case REDUCE_NODE:
        {
            if(_reduce_dimension_set.empty())
            {
                _dimension_types = std::vector<MetricDimension>{
                    {dimension_map().at(ROCPROFILER_DIMENSION_INSTANCE),
                     1,
                     ROCPROFILER_DIMENSION_INSTANCE}};
            }

            else
            {
                _dimension_types = std::vector<MetricDimension>{
                    {dimension_map().at(ROCPROFILER_DIMENSION_INSTANCE),
                     1,
                     ROCPROFILER_DIMENSION_INSTANCE}};
                auto first = _children[0].set_dimensions();
                first.erase(std::remove_if(first.begin(),
                                           first.end(),
                                           [&](const MetricDimension& dim) {
                                               return _reduce_dimension_set.find(dim.type()) !=
                                                      _reduce_dimension_set.end();
                                           }),
                            first.end());
                if(!first.empty()) _dimension_types = first;
            }
        }
        break;
        case SELECT_NODE:
        {
            auto first = _children[0].set_dimensions();
            first.erase(std::remove_if(first.begin(),
                                       first.end(),
                                       [&](const MetricDimension& dim) {
                                           return _select_dimension_map.find(dim.type()) !=
                                                  _select_dimension_map.end();
                                       }),
                        first.end());
            if(first.empty())
            {
                _dimension_types = std::vector<MetricDimension>{
                    {dimension_map().at(ROCPROFILER_DIMENSION_INSTANCE),
                     1,
                     ROCPROFILER_DIMENSION_INSTANCE}};
            }
            else
            {
                _dimension_types = first;
            }
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
            case ACCUMULATE_NODE:
            {
                // Future todo only to be applied on sq metric
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
}  // namespace

int64_t
get_agent_property(std::string_view property, const rocprofiler_agent_t& agent)
{
    using map_t = std::unordered_map<std::string_view, property_function_t>;

    static auto*& _props = common::static_object<common::Synchronized<map_t>>::construct(map_t{
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
    });

    return CHECK_NOTNULL(_props)->wlock([&property, &agent](map_t& props) -> int64_t {
        if(const auto* func = rocprofiler::common::get_val(props, property))
        {
            return (*func)(agent);
        }
        return 0;
    });
}

void
EvaluateAST::read_special_counters(
    const rocprofiler_agent_t&        agent,
    const std::set<counters::Metric>& required_special_counters,
    std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_t>>& out_map)
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

std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_t>>
EvaluateAST::read_pkt(const aql::CounterPacketConstruct* pkt_gen, hsa::AQLPacket& pkt)
{
    struct it_data
    {
        std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_t>>* data;
        const aql::CounterPacketConstruct*                                       pkt_gen;
        aqlprofile_agent_handle_t                                                agent;
    };

    auto aql_agent = *CHECK_NOTNULL(rocprofiler::agent::get_aql_agent(pkt_gen->agent()));

    std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_t>> ret;
    if(pkt.empty) return ret;
    it_data aql_data{.data = &ret, .pkt_gen = pkt_gen, .agent = aql_agent};

    hsa_status_t status = aqlprofile_pmc_iterate_data(
        pkt.handle,
        [](aqlprofile_pmc_event_t event, uint64_t counter_id, uint64_t counter_value, void* data) {
            CHECK(data);
            auto&       it     = *static_cast<it_data*>(data);
            const auto* metric = it.pkt_gen->event_to_metric(event);

            if(!metric) return HSA_STATUS_SUCCESS;

            auto& vec = it.data->emplace(metric->id(), std::vector<rocprofiler_counter_record_t>{})
                            .first->second;
            auto& next_rec = vec.emplace_back();
            set_counter_in_rec(next_rec.id, {.handle = metric->id()});
            // Actual dimension info needs to be used here in the future
            auto aql_status = aql::set_dim_id_from_sample(next_rec.id, it.agent, event, counter_id);
            CHECK_EQ(aql_status, ROCPROFILER_STATUS_SUCCESS)
                << rocprofiler_get_status_string(aql_status);

            // set_dim_in_rec(next_rec.id, ROCPROFILER_DIMENSION_NONE, vec.size() - 1);
            // Note: in the near future we need to use hw_counter here instead
            next_rec.counter_value = counter_value;
            return HSA_STATUS_SUCCESS;
        },
        &aql_data);

    if(status != HSA_STATUS_SUCCESS)
    {
        ROCP_ERROR << "AqlProfile could not decode packet " << status;
    }
    return ret;
}

void
EvaluateAST::set_out_id(std::vector<rocprofiler_counter_record_t>& results) const
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
        if(child._type == NodeType::ACCUMULATE_NODE) continue;
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
std::vector<rocprofiler_counter_record_t>*
EvaluateAST::evaluate(
    std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_t>>& results_map,
    std::vector<std::unique_ptr<std::vector<rocprofiler_counter_record_t>>>& cache)
{
    auto perform_op = [&](auto&& op) {
        auto* r1 = _children.at(0).evaluate(results_map, cache);
        auto* r2 = _children.at(1).evaluate(results_map, cache);

        if(r1->size() < r2->size()) swap(r1, r2);

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
        case NUMBER_NODE:
        {
            cache.emplace_back(std::make_unique<std::vector<rocprofiler_counter_record_t>>());
            *cache.back() = _static_value;
            return cache.back().get();
        }
        case ADDITION_NODE:
            return perform_op([](auto& a, auto& b) {
                return rocprofiler_counter_record_t{
                    .id            = a.id,
                    .counter_value = a.counter_value + b.counter_value,
                    .dispatch_id   = a.dispatch_id,
                    .user_data     = {.value = 0},
                    .agent_id      = {.handle = 0}};
            });
        case SUBTRACTION_NODE:
            return perform_op([](auto& a, auto& b) {
                return rocprofiler_counter_record_t{
                    .id            = a.id,
                    .counter_value = a.counter_value - b.counter_value,
                    .dispatch_id   = a.dispatch_id,
                    .user_data     = {.value = 0},
                    .agent_id      = {.handle = 0}};
            });
        case MULTIPLY_NODE:
            return perform_op([](auto& a, auto& b) {
                return rocprofiler_counter_record_t{
                    .id            = a.id,
                    .counter_value = a.counter_value * b.counter_value,
                    .dispatch_id   = a.dispatch_id,
                    .user_data     = {.value = 0},
                    .agent_id      = {.handle = 0}};
            });
        case DIVIDE_NODE:
            return perform_op([](auto& a, auto& b) {
                return rocprofiler_counter_record_t{
                    .id            = a.id,
                    .counter_value = (b.counter_value == 0 ? 0 : a.counter_value / b.counter_value),
                    .dispatch_id   = a.dispatch_id,
                    .user_data     = {.value = 0},
                    .agent_id      = {.handle = 0}};
            });
        case ACCUMULATE_NODE:
        // todo update how to read the hybrid metric
        case REFERENCE_NODE:
        {
            auto* result = rocprofiler::common::get_val(results_map, _metric.id());
            if(!result)
                throw std::runtime_error(
                    fmt::format("Unable to lookup results for metric {}", _metric.name()));

            cache.emplace_back(std::make_unique<std::vector<rocprofiler_counter_record_t>>());
            *cache.back() = *result;
            result        = cache.back().get();
            return result;
        }
        break;
        case REDUCE_NODE:
        {
            auto* result = _children.at(0).evaluate(results_map, cache);
            if(_reduce_op == REDUCE_NONE)
                throw std::runtime_error(fmt::format("Invalid Second argument to reduce(): {}",
                                                     static_cast<int>(_reduce_op)));
            return perform_reduction(_reduce_op, result, _reduce_dimension_set);
        }
        case SELECT_NODE:
        {
            auto* result = _children.at(0).evaluate(results_map, cache);
            return perform_selection(_select_dimension_map, result);
        }
    }

    return nullptr;
}

}  // namespace counters
}  // namespace rocprofiler
