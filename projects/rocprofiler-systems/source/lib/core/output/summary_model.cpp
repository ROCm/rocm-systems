// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/summary_model.hpp"

#include "output/helper_collapser.hpp"
#include "output/role_classifier.hpp"
#include "output/run_metadata.hpp"
#include "output_file_registry.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <unordered_set>
#include <utility>

namespace rocprofsys::output
{

namespace
{
std::size_t
count_distinct_pids(const std::vector<output_file>& rows)
{
    std::unordered_set<pid_t> pid_set;
    for(const auto& row : rows)
        pid_set.insert(row.pid);
    return pid_set.size();
}

std::string
derive_output_dir(const run_metadata& meta, const std::vector<output_file>& rows)
{
    if(!meta.output_dir_abs.empty()) return meta.output_dir_abs;
    if(rows.empty()) return "?";
    auto parent = std::filesystem::path{ rows.front().path }.parent_path().string();
    return parent.empty() ? std::string{ "?" } : parent;
}

std::uintmax_t
total_known_output_bytes(const std::vector<output_file>& rows)
{
    std::uintmax_t total = 0;
    for(const auto& row : rows)
        if(row.size_bytes) total += *row.size_bytes;
    return total;
}

std::vector<int>
union_of_utilized_gpus(const std::vector<process_node>& roots)
{
    std::vector<int> utilized;
    for(const auto& root : roots)
        utilized.insert(utilized.end(), root.effective_gpu_ids.begin(),
                        root.effective_gpu_ids.end());
    std::sort(utilized.begin(), utilized.end());
    utilized.erase(std::unique(utilized.begin(), utilized.end()), utilized.end());
    return utilized;
}
}  // namespace

summary_model
build_summary_model(const output_file_registry& registry, const run_metadata& meta)
{
    summary_model model{};

    auto rows = registry.rows();
    if(rows.empty()) return model;

    auto processes    = registry.processes();
    model.built       = build_tree(rows, processes);
    model.built.roots = collapse_helpers(std::move(model.built.roots));
    classify(model.built.roots, getpid());
    compute_subtree_sizes(model.built.roots);

    model.process_count      = count_distinct_pids(rows);
    model.output_dir         = derive_output_dir(meta, rows);
    model.total_output_bytes = total_known_output_bytes(rows);
    model.utilized_gpu_ids   = union_of_utilized_gpus(model.built.roots);
    model.node_gpu_count     = registry.node_gpu_count();
    return model;
}

}  // namespace rocprofsys::output
