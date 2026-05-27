// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/summary_model.hpp"

#include "output/helper_collapser.hpp"
#include "output/role_classifier.hpp"
#include "output/run_metadata.hpp"
#include "output_file_registry.hpp"

#include <unistd.h>

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
    for(const auto& r : rows)
        pid_set.insert(r.pid);
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

    model.process_count = count_distinct_pids(rows);
    model.output_dir    = derive_output_dir(meta, rows);
    return model;
}

}  // namespace rocprofsys::output
