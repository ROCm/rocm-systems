#pragma once

/**
 * @file frame_filters.hpp
 * @brief Reusable frame_predicate builders for sample_postprocessor
 *        and slice_emitter consumers.
 *
 * Two builders are provided:
 *   - make_main_exe_only_filter: drops every frame whose IP belongs
 *     to a shared library (libc, libstdc++, ld-linux, vDSO, ...).
 *     Keeps frames inside the main executable mapping and frames
 *     whose IP falls outside any known mapping (JIT / anonymous).
 *   - make_library_exclude_filter: drops frames whose owning module
 *     path matches any of a caller-supplied substring list.
 *
 * Both builders snapshot the main-executable path at construction
 * and capture a reference to the caller-owned proc_maps instance.
 * The proc_maps instance MUST outlive the returned predicate.
 *
 * Header-only. C++20.
 */

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cpu_snap/core/module_map.hpp"
#include "cpu_snap/core/self_exe.hpp"
#include "cpu_snap/processing/symbols/proc_maps.hpp"

namespace cpu_snap::processing::symbols
{

/**
 * @brief Build a frame_predicate that drops every frame NOT in the
 *        main executable.
 *
 * Uses @p maps (already populated) to resolve the owning module per
 * IP. Frames whose mapping is the main executable, or whose IP does
 * not fall inside any known mapping (anonymous / JIT pages), are
 * kept. Frames inside shared libraries are dropped.
 *
 * @param maps Live proc_maps reference. Must outlive the returned
 *             predicate.
 * @return     Predicate `bool(std::uint64_t ip, std::string_view)`
 *             returning true to keep, false to drop.
 *
 * @note Cost per call: one linear mapping scan + a path equality
 *       check. The main-exe path is resolved once at construction
 *       via cpu_snap::core::self_exe_path().
 */
[[nodiscard]] inline std::function<bool(std::uint64_t, std::string_view)>
make_main_exe_only_filter(cpu_snap::processing::symbols::proc_maps& maps)
{
    std::string main_exe = cpu_snap::core::self_exe_path();
    if(main_exe.empty())
    {
        main_exe = maps.main_exe();
    }
    return [&maps, main_exe = std::move(main_exe)](
               std::uint64_t ip, std::string_view /*name*/) -> bool {
        const cpu_snap::processing::symbols::vm_mapping* mapping = maps.find(ip);
        if(mapping == nullptr)
        {
            return true;
        }
        if(mapping->is_main_exe)
        {
            return true;
        }
        return mapping->path == main_exe;
    };
}

/**
 * @brief Build a frame_predicate that drops frames whose owning
 *        module path contains any of @p excluded_libraries as a
 *        substring.
 *
 * Substring match; e.g. passing `{"libc.so", "libstdc++.so"}`
 * drops any frame whose mapping path contains either substring.
 * Frames whose IP falls outside any known mapping are kept.
 *
 * @param maps               Live proc_maps reference. Must outlive
 *                           the returned predicate.
 * @param excluded_libraries Substring patterns to exclude.
 * @return                   Predicate filtering matching frames.
 */
[[nodiscard]] inline std::function<bool(std::uint64_t, std::string_view)>
make_library_exclude_filter(cpu_snap::processing::symbols::proc_maps& maps,
                            std::vector<std::string>   excluded_libraries)
{
    return [&maps, patterns = std::move(excluded_libraries)](
               std::uint64_t ip, std::string_view /*name*/) -> bool {
        const cpu_snap::processing::symbols::vm_mapping* mapping = maps.find(ip);
        if(mapping == nullptr)
        {
            return true;
        }
        return std::none_of(
            patterns.begin(), patterns.end(), [&](const std::string& pat) {
                return mapping->path.find(pat) != std::string::npos;
            });
    };
}

/**
 * @brief Module_map-based variant of make_main_exe_only_filter(), for
 *        offline / cross-process use where no live /proc/self/maps is
 *        available.
 *
 * @p timestamp_ns is fixed for the returned predicate's lifetime: the
 * module set is resolved once per call, against whatever was loaded at
 * that single timestamp. Unlike the live-process filter (which always
 * reflects the current mapping list), this does not track a capture's
 * module layout changing mid-stream -- fine for one representative
 * timestamp (e.g. session start), but callers needing true per-sample
 * time-varying filtering should resolve per sample instead of caching
 * one predicate.
 *
 * @param modules      Time-aware module map. Must outlive the returned
 *                     predicate.
 * @param timestamp_ns Timestamp to resolve every ip against.
 * @param main_exe_path Absolute path of the module considered "main
 *                      executable"; module_map carries no such flag
 *                      itself, so the caller must supply it.
 * @return             Predicate `bool(std::uint64_t ip, std::string_view)`.
 */
[[nodiscard]] inline std::function<bool(std::uint64_t, std::string_view)>
make_main_exe_only_filter(const cpu_snap::core::module_map& modules, std::uint64_t timestamp_ns,
                          std::string main_exe_path)
{
    return [&modules, timestamp_ns, main_exe_path = std::move(main_exe_path)](
               std::uint64_t ip, std::string_view /*name*/) -> bool {
        const auto mod = modules.resolve(ip, timestamp_ns);
        if(!mod)
        {
            return true;
        }
        return mod->path == main_exe_path;
    };
}

/**
 * @brief Module_map-based variant of make_library_exclude_filter(); see
 *        make_main_exe_only_filter(module_map, ...) for the fixed-timestamp
 *        caveat.
 *
 * @param modules            Time-aware module map. Must outlive the
 *                           returned predicate.
 * @param timestamp_ns       Timestamp to resolve every ip against.
 * @param excluded_libraries Substring patterns to exclude.
 * @return                   Predicate filtering matching frames.
 */
[[nodiscard]] inline std::function<bool(std::uint64_t, std::string_view)>
make_library_exclude_filter(const cpu_snap::core::module_map& modules, std::uint64_t timestamp_ns,
                            std::vector<std::string> excluded_libraries)
{
    return [&modules, timestamp_ns, patterns = std::move(excluded_libraries)](
               std::uint64_t ip, std::string_view /*name*/) -> bool {
        const auto mod = modules.resolve(ip, timestamp_ns);
        if(!mod)
        {
            return true;
        }
        return std::none_of(patterns.begin(), patterns.end(),
                            [&](const std::string& pat) {
                                return mod->path.find(pat) != std::string::npos;
                            });
    };
}

/**
 * @brief True for a symbolizer label ("name [module]") that belongs to the C++
 *        standard library.
 *
 * A label is stdlib when it names a frame of libstdc++ (module suffix) or a
 * function whose own qualified name is in `std::` or `__gnu_cxx::`. Only the
 * function name counts: `std::` in a parameter list, a template argument or the
 * return type of another function does not match, and operator names that
 * contain spaces (`operator bool`, `operator unsigned int`) are kept in one
 * piece. The check works on the demangled text, so it is a heuristic.
 *
 * @param label Resolved label as produced by the symbolizer.
 */
[[nodiscard]] inline bool
is_stdlib_label(std::string_view label)
{
    if(label.find("[libstdc++") != std::string_view::npos)
    {
        return true;
    }
    const std::string_view name = label.substr(0, label.rfind(" ["));

    int         depth       = 0;
    bool        in_operator = false;
    std::size_t token_start = 0;
    std::size_t end         = name.size();
    for(std::size_t i = 0; i < name.size(); ++i)
    {
        const char c = name[i];
        if(c == '<')
        {
            ++depth;
        }
        else if(c == '>')
        {
            --depth;
        }
        else if(depth == 0 && c == '(')
        {
            end = i;
            break;
        }
        else if(depth == 0 && c == ' ' && !in_operator)
        {
            // "operator bool" / "operator unsigned int": the type is part of the name.
            in_operator = name.substr(token_start, i - token_start).ends_with("operator");
            if(!in_operator)
            {
                token_start = i + 1;
            }
        }
    }
    const std::string_view function = name.substr(token_start, end - token_start);
    return function.starts_with("std::") || function.starts_with("__gnu_cxx::");
}

/**
 * @brief Build a frame_predicate that drops C++ standard library frames.
 *
 * @param resolve Callable `std::uint64_t -> std::string` or `std::string_view`
 *                returning the symbolizer label for an IP; it must stay valid
 *                for the predicate's lifetime. Frames it cannot name (hex
 *                labels) are kept.
 * @return        Predicate `bool(ip, name)` returning true to keep the frame.
 * @see is_stdlib_label
 */
template <typename Resolve>
[[nodiscard]] std::function<bool(std::uint64_t, std::string_view)>
make_stdlib_exclude_filter(Resolve resolve)
{
    return [resolve = std::move(resolve)](std::uint64_t ip, std::string_view /*name*/) -> bool {
        const auto label = resolve(ip);
        return !is_stdlib_label(label);
    };
}

}  // namespace cpu_snap::processing::symbols
