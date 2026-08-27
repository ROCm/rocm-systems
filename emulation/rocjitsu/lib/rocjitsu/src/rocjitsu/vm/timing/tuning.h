// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file tuning.h
/// @brief Every number the timing plane uses, read from the one config file.
///
/// @details The plane contains no constants. Each field below names a key under
/// the config's `timing.machine` object. There is no second file and no search
/// path: the architecture config rocjitsu is given is the whole description of
/// the part, and whoever produced that config -- mirage, or a person -- is
/// responsible for what is in it. That is the point of baking the numbers in
/// rather than referencing them: a run's timing is reproducible from the one
/// artefact it was given, and a config can be handed to someone without them
/// needing anything else to reproduce it.
///
/// Every value a config does not name resolves to the *slowest reasonable*
/// value for that parameter, not the typical one, and the run reports which
/// ones it fell back to. A forgotten parameter then makes a run read slow and
/// say so, rather than read fast and look like an answer.

#pragma once

#include "rocjitsu/vm/timing/inst_class.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu::timing {

/// @brief Geometry and rates of one cache level.
struct CacheTuning {
  std::uint64_t line_bytes = 4;
  std::uint64_t sets = 1;
  std::uint64_t ways = 1;
  /// @brief Cycles from request to data when the line is resident.
  std::uint64_t hit_cycles = 10000;
  /// @brief Lines this level returns per cycle, per instance.
  double lines_per_cycle = 1.0;
};

/// @brief Which matrix-core rate an opcode's inputs select.
enum class MatrixType : std::uint8_t { Default, F64, F32, F16, BF16, Narrow, Integer, Count };
inline constexpr std::size_t kNumMatrixTypes = static_cast<std::size_t>(MatrixType::Count);

/// @brief The part, as far as the timing plane is concerned.
struct Tuning {
  bool enabled = false;
  double clock_mhz = 1000.0;

  std::uint64_t compute_units = 1;
  std::uint64_t xcds = 1;
  std::uint64_t simd_lanes = 1;
  std::uint64_t wave_slots_per_cu = 1;
  std::uint64_t vector_registers_per_cu = 1;
  std::uint64_t scalar_registers_per_cu = 1;
  std::uint64_t lds_bytes_per_cu = 1;

  std::array<std::uint64_t, kNumInstClasses> issue_cycles{};
  /// @brief How many of a unit's operations retire per cycle.
  ///
  /// @details Fractional, and not because any unit has a fractional number of
  /// pipes. A class's issue occupancy is an integer count of cycles because
  /// that is what a pipeline stage is, but the *average* rate a unit sustains
  /// over a mixed instruction stream is not an integer, and this is the only
  /// place that average can be expressed. Writing it here rather than rounding
  /// a per-instruction cost keeps the per-instruction costs honest.
  std::array<double, kNumFunctionalUnits> ports{};

  /// @brief Cycles from an instruction's issue until a dependent instruction
  ///        can read the registers it wrote.
  ///
  /// @details Separate from issue_cycles because the two answer different
  /// questions: issue occupancy is how long the port is busy, this is how deep
  /// the pipeline behind it is. A model carrying only the first runs every
  /// compute unit at peak issue rate, which no pipeline does, and it does so
  /// worst on exactly the instruction-dense kernels a timing model is asked
  /// about. A memory access takes none of these: its destination is ready when
  /// the access completes, which the memory path computes.
  std::uint64_t vector_alu_result_cycles = 10000;
  std::uint64_t scalar_alu_result_cycles = 10000;
  std::uint64_t transcendental_result_cycles = 10000;
  std::uint64_t matrix_multiply_result_cycles = 10000;

  std::array<std::uint64_t, kNumMatrixTypes> matrix_macs_per_cycle{};
  double lane_addresses_per_cycle = 1.0;
  /// @brief Cycles before a wavefront may issue again, when its next
  ///        instruction goes to a different functional unit.
  ///
  /// @details A unit's issue occupancy is a throughput cost: how long that unit
  /// is unavailable to everyone. It is not how long the issuing wavefront must
  /// wait before its own next instruction, unless that instruction wants the
  /// same unit. Charging the full occupancy on the wavefront's critical path
  /// serialises a mixed instruction stream that the hardware pipelines, and it
  /// falls hardest on exactly the kernels with no parallelism to absorb it: a
  /// warp-shuffle softmax, which alternates transcendentals, shuffles and
  /// arithmetic, read 1.52 times its measured duration. Real dependencies are
  /// still enforced -- the register scoreboard holds an instruction until its
  /// operands are ready -- so this only removes a stall the hardware does not
  /// have. Defaults to the occupancy itself, which is the old behaviour.
  std::uint64_t wavefront_issue_cycles = 0;
  /// @brief Cycles one compute unit's front end spends on every instruction,
  ///        whatever functional unit it then goes to.
  ///
  /// @details A compute unit can start only so many instructions per cycle
  /// across all its units together, so a kernel whose work is spread thinly
  /// over many of them is limited by the front end rather than by any one of
  /// them. This is the term that binds instruction-dense kernels: at one cycle
  /// per instruction, which is the peak the part can do and not a rate it
  /// sustains, matrix and convolution kernels read 0.62 to 0.72 of measured
  /// while streaming kernels were unaffected, because only the dense ones reach
  /// the front end at all.
  ///
  /// Read from its own key rather than borrowed from the nop class, which is
  /// what it used to be: the value is load bearing and deserves a name that
  /// says what it is.
  std::uint64_t front_end_cycles = 0;
  /// @brief Front-end occupancy of one instruction of each class, in
  ///        sixteenths of a cycle.
  ///
  /// @details The front end is the one resource every instruction takes before
  /// it reaches any execution unit, and charging all of them the same is what
  /// made the model's accuracy a straight median-versus-maximum trade: at one
  /// cycle per instruction the instruction-dense matrix kernels read 0.62 to
  /// 0.72 of measured, and at two the memory-bound elementwise kernels read
  /// 2.37. Neither is a property of the kernels. A matrix instruction, a
  /// scalar load and a lane-parallel add do not occupy the same issue
  /// resources for the same time, and separating them removes the trade: the
  /// worst case fell from 46.9 to 34.7 per cent and the median from 21.8 to
  /// 12.9 with nothing else changed.
  ///
  /// Sixteenths because the useful values are fractions of a cycle and the
  /// accumulator they feed is an integer count of cycles; kFrontEndScale is
  /// divided back out where the total is used.
  std::array<std::uint64_t, kNumInstClasses> front_end_sixteenths{};

  /// @brief Cycles a dispatch's slowest wavefront runs beyond its average one,
  ///        per doubling of the wavefront count.
  ///
  /// @details A dispatch ends when its last wavefront ends. Every compute unit
  /// here runs the same instruction stream and finishes at the same modelled
  /// cycle, so the modelled maximum is the modelled mean and the straggler is
  /// missing entirely. The expected maximum of many independent noisy
  /// durations grows with the logarithm of how many there are, and the
  /// measurements agree: a launch-dominated kernel costs about the same extra
  /// per doubling of its wavefront count, from four wavefronts to a thousand.
  std::uint64_t straggler_cycles = 0;

  /// @brief Calibration factors, not measurements.
  ///
  /// @details Each scales a term the model composes, and each is here because
  /// the term is right in shape and wrong in size for a reason this model
  /// cannot presently name. They are separated from the machine parameters
  /// deliberately: a reader has to be able to tell what is a property of the
  /// part from what is a number fitted to a corpus.
  /// @brief Cycles a workgroup barrier costs a wavefront, beyond the spread
  ///        between the wavefronts the model can see.
  ///
  /// @details A barrier costs the difference between the first wavefront to
  /// arrive and the last, and this model gives every wavefront in a group the
  /// same instruction stream, so the modelled difference is near zero and a
  /// barrier comes out nearly free. On hardware the wavefronts arrive skewed
  /// by whatever their memory accesses did, and a kernel that barriers
  /// thousands of times per wavefront -- a blocked matrix multiply staging
  /// tiles through the local data share does exactly that -- pays for all of
  /// them. Fitted, so it lives with the calibration; zero by default, which is
  /// the old behaviour.
  std::uint64_t barrier_cycles = 0;

  /// @brief How the issue term scales with wavefronts resident per compute
  ///        unit, as an exponent on that count.
  ///
  /// @details Zero is the linear model: a unit's queue is the work on it and
  /// nothing else. Measurement disagrees. The same kernel at four wavefronts
  /// per compute unit reaches a higher instruction rate than at sixteen --
  /// 1.35 against 0.86 instructions per cycle -- because fewer wavefronts
  /// contend for the same issue slot, and a linear term over-charges the
  /// sparse case and under-charges the dense one by exactly that ratio. This
  /// is the smallest correction of the right shape, and it is fitted rather
  /// than derived, which is why it lives with the calibration.
  double issue_occupancy_exponent = 0.0;
  double stall_exposed_fraction = 1.0;
  double latency_exposure_scale = 1.0;
  double fill_exposure_scale = 1.0;
  double fill_ramp_scale = 1.0;

  CacheTuning l1_vector;
  CacheTuning l1_scalar;
  CacheTuning l1_instruction;
  CacheTuning l2;
  CacheTuning mall;

  std::uint64_t dram_latency_cycles = 10000;
  double dram_bytes_per_cycle = 1.0;
  double mall_bytes_per_cycle = 1.0;
  /// @brief Bytes per cycle across the link between the dies and memory,
  ///        charged on every second-level miss whether or not the memory-side
  ///        cache then serves it.
  ///
  /// @details The memory-side cache sits behind this link, so a hit in it saves
  /// a DRAM access and saves nothing on the way there. Leaving this out lets a
  /// working set that happens to fit in the memory-side cache read several
  /// times faster than the part has ever been measured to go: on an eight
  /// megabyte copy it produced 4.96 microseconds against a measured 10.88, and
  /// adding it produced 11.72.
  double fabric_bytes_per_cycle = 1.0;
  std::uint64_t memory_channels = 1;
  std::uint64_t fabric_request_bytes = 64;
  /// @brief Bytes in one DRAM row, per channel.
  std::uint64_t dram_row_bytes = 1024;
  /// @brief Extra cycles a channel spends closing one row and activating
  ///        another. Zero disables the row model.
  std::uint64_t dram_row_miss_cycles = 0;
  std::uint64_t miss_status_registers_per_cu = 1;
  /// @brief How many resident wavefronts' memory stalls genuinely overlap.
  ///
  /// @details Hiding a stall needs another wavefront with work that is ready,
  /// and in a kernel whose wavefronts all do the same thing they are not: they
  /// issue their loads together and wait together, so there is nothing ready to
  /// run. Crediting every resident wavefront's whole issue stream as available
  /// to overlap assumes perfect staggering, which is the best case and not the
  /// common one -- a strided copy with eight wavefronts per compute unit was
  /// credited with 4760 cycles of hiding against a 1320-cycle critical path and
  /// came out at 0.46 of its measured duration. One means the stalls do not
  /// overlap at all; the resident count means they overlap perfectly.
  std::uint64_t stall_overlap_wavefronts = 1;

  std::uint64_t lds_banks = 1;
  std::uint64_t lds_bank_bytes = 4;
  std::uint64_t lds_latency_cycles = 10000;
  std::uint64_t lds_lanes_per_phase = 1;

  std::uint64_t dispatch_start_cycles = 10000;
  std::uint64_t dispatch_end_cycles = 10000;
  std::uint64_t launch_invalidate_cycles = 10000;
  double workgroups_per_cycle = 0.001;

  /// @brief Keys the config named, and keys it did not, for the run's report.
  std::vector<std::string> resolved;
  std::vector<std::string> fell_back;

  /// @brief Read the `timing` block out of the raw config JSON.
  ///
  /// @details A second, schema-free pass over the same document, for the same
  /// reason the plugin block needs one: the typed load runs with unexpected
  /// fields skipped, so a `timing` block on the typed path would be dropped in
  /// silence and the plane would run entirely on fallbacks with nothing to say
  /// it had.
  static Tuning parse(const std::string &config_json);

  /// @brief Append what was in effect and what was not.
  void write_report(std::string &out) const;
};

} // namespace rocjitsu::timing
