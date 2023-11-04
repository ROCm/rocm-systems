#include "lib/rocprofiler/counters/evaluate_ast.hpp"
#include <fmt/core.h>

#include <exception>
#include <optional>

#include <numeric>
#include <stdexcept>
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler/counters/dimensions.hpp"
#include "lib/rocprofiler/counters/parser/reader.hpp"

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

    ReduceOperation type           = REDUCE_NONE;
    const auto*     reduce_op_type = rocprofiler::common::get_val(reduce_op_string_to_type, op);
    if(reduce_op_type) type = *reduce_op_type;
    return type;
}

std::vector<rocprofiler_record_counter_t>*
perform_reduction(ReduceOperation reduce_op, std::vector<rocprofiler_record_counter_t>* input_array)
{
    rocprofiler_record_counter_t result{.id = 0, .derived_counter = 0};
    if(input_array->empty()) return input_array;
    switch(reduce_op)
    {
        case REDUCE_NONE: break;
        case REDUCE_MIN:
        {
            result =
                *std::min_element(input_array->begin(), input_array->end(), [](auto& a, auto& b) {
                    return a.derived_counter < b.derived_counter;
                });
            break;
        }
        case REDUCE_MAX:
        {
            result =
                *std::max_element(input_array->begin(), input_array->end(), [](auto& a, auto& b) {
                    return a.derived_counter > b.derived_counter;
                });
            break;
        }
        case REDUCE_SUM:
        {
            result = std::accumulate(
                input_array->begin(),
                input_array->end(),
                rocprofiler_record_counter_t{.id = 0, .derived_counter = 0},
                [](auto& a, auto& b) {
                    return rocprofiler_record_counter_t{
                        .id = a.id, .derived_counter = a.derived_counter + b.derived_counter};
                });
            break;
        }
        case REDUCE_AVG:
        {
            result = std::accumulate(
                input_array->begin(),
                input_array->end(),
                rocprofiler_record_counter_t{.id = 0, .derived_counter = 0},
                [](auto& a, auto& b) {
                    return rocprofiler_record_counter_t{
                        .id = a.id, .derived_counter = a.derived_counter + b.derived_counter};
                });
            result.derived_counter /= input_array->size();
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
                    auto& evaluate_ast_node =
                        eval_map.emplace(metric.name(), EvaluateAST(by_name, *ast, gfx))
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
            // Set dimensions after all ASTs loaded for arch.
            for(auto& [name, ast] : eval_map)
            {
                try
                {
                    ast.set_dimensions();
                } catch(std::exception& e)
                {
                    LOG(ERROR) << "Could not set dimensions for " << name << " failed with "
                               << e.what();
                }
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

EvaluateAST::EvaluateAST(const std::unordered_map<std::string, Metric>& metrics,
                         const RawAST&                                  ast,
                         std::string                                    agent)
: _type(ast.type)
, _reduce_op(get_reduce_op_type_from_string(ast.reduce_op))
, _agent(std::move(agent))
, _reduce_dimension_set(ast.reduce_dimension_set)
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
        _static_value.push_back({.id = 0, .hw_counter = std::get<int64_t>(ast.value)});
    }

    for(const auto& nextAst : ast.counter_set)
    {
        _children.emplace_back(metrics, *nextAst, _agent);
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

std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>>
EvaluateAST::read_pkt(const aql::AQLPacketConstruct* pkt_gen, hsa::AQLPacket& pkt)
{
    struct it_data
    {
        std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>>* data;
        const aql::AQLPacketConstruct*                                           pkt_gen;
    };

    std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>> ret;
    it_data aql_data{.data = &ret, .pkt_gen = pkt_gen};

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
            next_rec.derived_counter = info_data->pmc_data.result;
            return HSA_STATUS_SUCCESS;
        },
        &aql_data);
    CHECK(status == HSA_STATUS_SUCCESS);
    return ret;
}

// convert to buffer at some point
std::vector<rocprofiler_record_counter_t>*
EvaluateAST::evaluate(
    std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>>& results_map)
{
    auto perform_op = [&](auto&& op) {
        auto* r1 = _children[0].evaluate(results_map);
        auto* r2 = _children[1].evaluate(results_map);

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
        case RANGE_NODE: break;
        case NUMBER_NODE: return &_static_value;
        case ADDITION_NODE:
            return perform_op([](auto& a, auto& b) {
                return rocprofiler_record_counter_t{
                    .id = a.id, .derived_counter = a.derived_counter + b.derived_counter};
            });
        case SUBTRACTION_NODE:
            return perform_op([](auto& a, auto& b) {
                return rocprofiler_record_counter_t{
                    .id = a.id, .derived_counter = a.derived_counter - b.derived_counter};
            });
        case MULTIPLY_NODE:
            return perform_op([](auto& a, auto& b) {
                return rocprofiler_record_counter_t{
                    .id = a.id, .derived_counter = a.derived_counter * b.derived_counter};
            });
        case DIVIDE_NODE:
            return perform_op([](auto& a, auto& b) {
                return rocprofiler_record_counter_t{
                    .id = a.id,
                    .derived_counter =
                        (b.derived_counter == 0 ? 0 : a.derived_counter / b.derived_counter)};
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
