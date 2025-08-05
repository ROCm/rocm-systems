// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "binary/dwarf_entry.hpp"
#include "binary/symbol.hpp"
#include "core/containers/c_array.hpp"
#include "core/defines.hpp"
#include "core/utility.hpp"
#include "library/causal/components/backtrace.hpp"
#include "library/causal/components/progress_point.hpp"
#include "library/causal/data.hpp"
#include "library/causal/selected_entry.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/tpls/cereal/cereal.hpp>
#include <timemory/tpls/cereal/cereal/cereal.hpp>
#include <timemory/unwind/types.hpp>
#include <timemory/utility/unwind.hpp>

#include <atomic>
#include <cstdint>
#include <unordered_map>

namespace rocprofsys
{
namespace causal
{
using hash_value_t = ::tim::hash_value_t;

struct experiment
{
    using progress_points_t =
        std::unordered_map<tim::hash_value_t, component::progress_point>;
    using experiments_t     = std::vector<experiment>;
    using filename_config_t = settings::compose_filename_config;
    using period_stats_t    = tim::statistics<int64_t>;

    struct sample : unwind::processed_entry
    {
        using base_type = unwind::processed_entry;

        sample() = default;
        sample(const base_type&, uint64_t);

        mutable uint64_t                    count   = 0;
        std::vector<binary::inlined_symbol> inlines = {};

        bool        operator==(const sample&) const;
        bool        operator<(const sample&) const;
        const auto& operator+=(const sample&) const;

        template <typename ArchiveT>
        void serialize(ArchiveT& ar, const unsigned);

        std::string get_identifier() const;
    };

    struct record
    {
        int64_t                 startup     = 0;
        uint64_t                runtime     = 0;
        std::vector<experiment> experiments = {};
        std::vector<sample>     samples     = {};

        template <typename ArchiveT>
        void serialize(ArchiveT& ar, const unsigned);
    };

    static std::string                     label();
    static std::string                     description();
    static const std::atomic<experiment*>& get_current_experiment();

    ROCPROFSYS_DEFAULT_OBJECT(experiment)

    bool        start();
    bool        wait() const;  // returns false if interrupted
    bool        stop();
    std::string as_string() const;

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned version);

    // in nanoseconds
    static uint64_t      get_delay();
    static double        get_delay_scaling();
    static uint32_t      get_index();
    static bool          is_active();
    static bool          is_selected(uint64_t);
    static bool          is_selected(unwind_addr_t);
    static bool          is_selected(container::c_array<uint64_t>);
    static void          add_selected();
    static experiments_t get_experiments();

    template <size_t N>
    static bool is_selected(std::array<uint64_t, N> _v)
    {
        return is_selected(container::c_array<uint64_t>{ _v.data(), _v.size() });
    }

    static void                save_experiments();
    static void                save_experiments(std::string, const filename_config_t&);
    static std::vector<record> load_experiments(bool _throw_on_err = true);
    static std::vector<record> load_experiments(std::string, const filename_config_t&,
                                                bool = true);

    bool              running         = false;
    uint16_t          virtual_speedup = 0;    /// 0-100 in multiples of 5
    uint32_t          index           = 0;    /// experiment number
    uint64_t          sampling_period = 0;    /// period b/t samples [nsec]
    uint64_t          start_time      = 0;    /// start of experiment [nsec]
    uint64_t          end_time        = 0;    /// end of experiment [nsec]
    uint64_t          experiment_time = 0;    /// how long the experiment ran [nsec]
    uint64_t          duration        = 0;    /// runtime - delays [nsec]
    uint64_t          batch_size      = 10;   /// batch factor for experiment/cooloff
    uint64_t          scaling_factor  = 100;  /// scaling factor for experiment time
    uint64_t          sample_delay    = 0;    /// how long to delay [nsec]
    uint64_t          total_delay     = 0;    /// total delays [nsec]
    uint64_t          selected        = 0;    /// num times selected line sampled
    uint64_t          global_delay    = 0;
    double            delay_scaling   = 0.0;  /// virtual_speedup / 100.
    selected_entry    selection       = {};   /// which line was selected
    progress_points_t init_progress   = {};   /// progress points at start
    progress_points_t fini_progress   = {};   /// progress points at end
};
}  // namespace causal
}  // namespace rocprofsys
