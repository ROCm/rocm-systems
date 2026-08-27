//! The built-in timing table each builtin agent carries, and where every
//! number in it came from.
//!
//! Mirage bakes these into the one architecture config it hands the
//! emulator, so a run's timing is reproducible from that single file and
//! nothing else. That makes the provenance of each number part of the
//! deliverable: a table nobody can account for reads exactly like one
//! that was correlated against hardware, and the whole point of writing
//! it down here is that the two can be told apart.
//!
//! # Classification
//!
//! Everything below is **public-derived or assumed. None of it is
//! measured.** Three kinds of value appear, and each is marked at its
//! definition:
//!
//! * *Derived* — computed from something the agent already carries (its
//!   clock, memory width, preset per-CU limits, component tree) or from
//!   a published headline figure, by an arithmetic written out in the
//!   code so it can be checked.
//! * *Published* — a number AMD has stated for the part.
//! * *Assumed* — a plausible value for a parameter AMD has not published
//!   in this detail. These are the ones a measured table replaces first,
//!   and each says so.
//!
//! # Where a private table goes
//!
//! A table with measured numbers in it does not belong in this file and
//! must not be committed. Keep it as a JSON tuning file outside the
//! repository and pass it with `mirage run --timing-tuning PATH`; mirage
//! reads it, merges it over what is here, and writes the result only
//! into the session's scratch directory under `$XDG_RUNTIME_DIR`. See
//! `docs/timing.md`.

use std::collections::BTreeMap;

use mirage_core::agent::{ComponentDef, KfdDeviceInfo};
use mirage_core::timing::{CLOCK_KEY, TimingTable};
use serde_json::Number;

use crate::presets::Preset;

/// The part-specific inputs a timing table is built from.
///
/// Only what cannot be read off the agent itself is here. Everything
/// derivable — the clock, the memory width, the per-CU limits, the CU
/// and XCD counts — is taken from the [`KfdDeviceInfo`], the
/// [`Preset`] and the component tree, so a table cannot drift from the
/// device it describes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Part {
    /// Lanes one SIMD issues per pass.
    ///
    /// *Published geometry.* A CDNA SIMD is 16 lanes wide, so a wave64
    /// instruction is four passes through it. `gfx1250` is a wave32 part
    /// and its SIMD is 32 lanes, so a wavefront is one pass.
    pub(crate) simd_lanes: u64,
    /// Peak HBM bandwidth, bytes per second.
    ///
    /// *Published.* Stated directly rather than recomputed from the
    /// agent's `mem_width` and `mem_clk_max`, because what those two mean
    /// together depends on the HBM generation's pump rate: MI300X's
    /// published 5.3 TB/s over a 8192-bit bus needs a 5.2 GT/s pin rate,
    /// which is four times the 1300 MHz the agent reports, not two.
    pub(crate) dram_bytes_per_second: f64,
    /// Memory-side cache capacity, bytes. *Published.*
    pub(crate) mall_bytes: u64,
    /// Memory-side cache peak bandwidth, bytes per second.
    pub(crate) mall_bytes_per_second: f64,
    /// Matrix multiply-accumulates one compute unit retires per cycle.
    pub(crate) matrix: Matrix,
}

/// Matrix-core rate per input element type, in multiply-accumulates one
/// compute unit retires per cycle.
///
/// One rate per type rather than one rate: the spread inside a single
/// part is a factor of thirty-two between double precision and the
/// narrow formats, and costing every shape at one rate is wrong by that
/// spread in the fast direction for exactly the kernels a corpus is most
/// likely to contain.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Matrix {
    pub(crate) f64: u64,
    pub(crate) f32: u64,
    pub(crate) f16: u64,
    pub(crate) bf16: u64,
    /// FP8 and the formats below it.
    pub(crate) narrow: u64,
    pub(crate) integer: u64,
}

/// MI300X: *derived* from AMD's published dense matrix peaks for the
/// part, at its 304 compute units and 2100 MHz boost clock, as
/// `TFLOPS / 2 / (CUs * clock)`.
///
/// FP64 and FP32 matrix 163.4 TFLOPS give 128; FP16 and BF16 1307.4
/// TFLOPS give 1024; FP8 2614.9 TFLOPS and INT8 2614.9 TOPS give 2048.
/// The rate is per compute unit, so it does not change with the 256-CU
/// grid mirage emulates.
const MI300X_MATRIX: Matrix = Matrix {
    f64: 128,
    f32: 128,
    f16: 1024,
    bf16: 1024,
    narrow: 2048,
    integer: 2048,
};

/// MI350X: *derived* the same way from the MI355X published dense peaks,
/// at 256 compute units and 2400 MHz.
///
/// FP64 matrix 78.6 TFLOPS gives 64; FP32 matrix 157.3 TFLOPS gives 128;
/// FP16 and BF16 2.5 PFLOPS give 2048; FP8 5.0 PFLOPS and INT8 5.0 POPS
/// give 4096. FP4 and FP6 run at 10.1 PFLOPS, twice the FP8 rate, but
/// the timing plane carries one rate for every format below FP16 and
/// this takes the slower of the two.
const MI350X_MATRIX: Matrix = Matrix {
    f64: 64,
    f32: 128,
    f16: 2048,
    bf16: 2048,
    narrow: 4096,
    integer: 4096,
};

/// MI300X. Bandwidth figures are AMD's published peaks for the part:
/// 5.3 TB/s HBM3 and 17.2 TB/s Infinity Cache, over 256 MB of it.
pub(crate) const MI300X: Part = Part {
    simd_lanes: 16,
    dram_bytes_per_second: 5.3e12,
    mall_bytes: 256 * 1024 * 1024,
    mall_bytes_per_second: 17.2e12,
    matrix: MI300X_MATRIX,
};

/// MI350X. HBM3E peak is AMD's published 8.0 TB/s for MI355X. No
/// Infinity Cache bandwidth is published for CDNA4, so MI300X's
/// published 17.2 TB/s is carried forward unchanged, which is
/// conservative for a part whose DRAM got faster.
pub(crate) const MI350X: Part = Part {
    simd_lanes: 16,
    dram_bytes_per_second: 8.0e12,
    mall_bytes: 256 * 1024 * 1024,
    mall_bytes_per_second: 17.2e12,
    matrix: MI350X_MATRIX,
};

/// MI450X. The builtin MI450X agent is itself a placeholder — its KFD
/// identity is `gfx1250` with invented ids — and nothing about the
/// MI400 series is published in the detail a timing table needs. Every
/// rate here is MI350X's, which makes MI450X timing a statement about
/// CDNA4 running a CDNA5 ISA and not a prediction of the part.
///
/// The one thing that is this part's own is `simd_lanes`: `gfx1250` is
/// a wave32 target, so its SIMD is 32 lanes and a wavefront issues in
/// one pass rather than four.
pub(crate) const MI450X: Part = Part {
    simd_lanes: 32,
    ..MI350X
};

// -- Assumed latencies, in nanoseconds ---------------------------------------
//
// Nanoseconds rather than cycles so the same assumption produces the
// right cycle count on a 2100 MHz part and a 2700 MHz one. Every one of
// them is a plausible value for something AMD has not published, drawn
// from the shape of the memory hierarchy rather than from a
// measurement, and every one is a candidate for replacement by a
// private table.

/// Unloaded load-to-use latency from a second-level miss to HBM.
const DRAM_LATENCY_NS: f64 = 380.0;
/// Hit latency of the memory-side cache.
const MALL_HIT_NS: f64 = 140.0;
/// Hit latency of the per-XCD second-level cache.
const L2_HIT_NS: f64 = 130.0;
/// Hit latency of the per-CU vector and scalar first-level caches.
const L1_HIT_NS: f64 = 50.0;
/// Hit latency of the instruction cache. Short because the fetch unit
/// runs ahead of issue, and the timing plane exposes this once per
/// wavefront rather than once per miss.
const INSTRUCTION_FETCH_NS: f64 = 10.0;
/// Local data share load-to-use latency.
const LDS_LATENCY_NS: f64 = 25.0;
/// The launch acquire, invalidating every first-level cache.
const LAUNCH_INVALIDATE_NS: f64 = 50.0;
/// From the packet arriving to the first wavefront issuing: one fabric
/// round trip in shape, and assumed.
const DISPATCH_START_NS: f64 = 430.0;

// -- Assumed rates and counts ------------------------------------------------

/// Lane addresses the vector-memory address path retires per cycle.
///
/// *Assumed*: one quad of lanes per cycle, so a wave64 access walks
/// sixteen cycles of addresses however well it coalesces. AMD has not
/// published this pipeline in this detail.
const LANE_ADDRESSES_PER_CYCLE: f64 = 4.0;

/// Outstanding first-level misses one compute unit may have.
///
/// *Assumed.* This is the cap that turns a memory-divergent kernel from
/// bandwidth limited into latency limited, and its effect grows with the
/// compute-unit count, so it matters more here than its confidence
/// deserves.
const MISS_STATUS_REGISTERS_PER_CU: u64 = 32;

/// Workgroups the command processor places per cycle, device wide.
/// *Assumed.*
const WORKGROUPS_PER_CYCLE: f64 = 1.0;

/// How much wider than the DRAM interface the die-to-memory fabric is.
///
/// *Assumed, and the least-supported number in this file.* The link must
/// saturate DRAM and leave headroom for memory-side cache hits; AMD has
/// not published its width. It is charged on every second-level miss
/// whether or not the memory-side cache then serves it, which makes it
/// the number a streaming kernel is most sensitive to — leaving the
/// charge out entirely made an eight-megabyte copy read 4.96 us against
/// a measured 10.88. A measured table should replace this first.
const FABRIC_WIDTH_OVER_DRAM: f64 = 2.0;

/// Bytes one request occupies on the fabric: one second-level line.
const L2_LINE_BYTES: u64 = 128;
/// Ways in the second-level and memory-side caches. *Assumed
/// organisation* for a published capacity.
const L2_WAYS: u64 = 16;
/// First-level line size and associativity. *Assumed organisation.*
const L1_LINE_BYTES: u64 = 64;
const L1_WAYS: u64 = 4;
/// Published first-level capacities per compute unit: a 32 KiB vector
/// cache, a 16 KiB scalar cache and a 64 KiB instruction cache. The
/// scalar and instruction caches are shared between pairs of compute
/// units on the real part; modelling them per-CU overstates capacity and
/// understates contention, and is flagged here rather than corrected,
/// because the timing plane has no notion of a shared first level.
const L1_VECTOR_BYTES: u64 = 32 * 1024;
const L1_SCALAR_BYTES: u64 = 16 * 1024;
const L1_INSTRUCTION_BYTES: u64 = 64 * 1024;

/// Lines the per-XCD second-level cache returns per cycle.
///
/// *Assumed organisation*: sixteen channels, each returning one 128-byte
/// line per cycle, so an XCD delivers 2 KiB per clock. Across eight XCDs
/// at 2100 MHz that is 34 TB/s, which is the order the published
/// aggregate L2 read bandwidth of MI300X sits at.
const L2_LINES_PER_CYCLE: f64 = 16.0;

/// Local data share banks and their width. *Published geometry*: 32
/// banks of one dword.
const LDS_BANKS: u64 = 32;
const LDS_BANK_BYTES: u64 = 4;
/// Lanes the local data share resolves in one conflict-free phase.
///
/// *Published behaviour*: a dword access is resolved 32 lanes at a time,
/// so a wave64 takes two phases and conflicts are counted inside a
/// phase, never across the wavefront.
const LDS_LANES_PER_PHASE: u64 = 32;

/// Bits one HBM pseudo-channel is wide, used to turn the agent's
/// `mem_width` into a channel count. *Published*: HBM3 and HBM3E
/// pseudo-channels are 64 bits.
const HBM_PSEUDO_CHANNEL_BITS: u64 = 64;

/// Issue occupancy of one pass, by instruction class.
///
/// *Assumed* from the published CDNA pipeline description: one cycle per
/// pass on the simple pipes, four on the ones that are quarter rate. It
/// is an assumption about a microarchitecture AMD has not published at
/// this granularity, not a measurement. `unknown` is not listed: it is
/// filled in with the largest cost here, so that an opcode nobody
/// classified makes a run read slow and look suspicious rather than
/// costing nothing.
const ISSUE_CYCLES: [(&str, u64); 19] = [
    ("vector_alu", 1),
    ("scalar_alu", 1),
    ("transcendental", 4),
    ("matrix_multiply", 4),
    ("lds_read", 1),
    ("lds_write", 1),
    ("vector_memory_read", 1),
    ("vector_memory_write", 1),
    ("vector_memory_atomic", 4),
    ("scalar_memory", 1),
    ("tensor_memory", 4),
    ("export", 1),
    ("branch", 1),
    ("wait_counter", 1),
    ("delay_alu", 1),
    ("barrier", 1),
    ("message", 1),
    ("nop", 1),
    ("terminate", 1),
];

/// Issue ports per compute unit, by functional unit.
///
/// *Derived* from the four SIMDs a CDNA compute unit has, and one of
/// everything else. `none` is the unit a wait, a barrier or a no-op
/// occupies; it is given a port so that a rate is never divided by zero.
///
/// Fractional, and not because any unit has a fractional number of
/// pipes. A class's issue occupancy is a whole number of cycles because
/// that is what a pipeline stage is; the *average* rate a unit sustains
/// over a mixed instruction stream is not, and this is where that shows
/// up. Calibration overlays it; what is written here is the geometry.
const PORTS: [(&str, f64); 10] = [
    ("none", 1.0),
    ("vector_alu", 4.0),
    ("scalar_alu", 1.0),
    ("transcendental", 4.0),
    ("matrix_multiply", 4.0),
    ("local_data_share", 1.0),
    ("vector_memory", 1.0),
    ("scalar_memory", 1.0),
    ("branch", 1.0),
    ("export", 1.0),
];

/// Front-end occupancy of one instruction of each class, in cycles.
///
/// The front end is the one resource every instruction takes before it
/// reaches any execution unit. One cycle for all of them is the peak the
/// part can reach rather than a rate it sustains, and it is what these
/// default to: the *differences* between classes are calibration and
/// arrive with the tuning overlay, not from here. What this table
/// establishes is that the keys exist and that a run without an overlay
/// behaves exactly as one front-end figure for everything used to.
const FRONT_END_CYCLES: f64 = 1.0;

/// Terms the composition applies to what the machine parameters produce.
///
/// Every one is a calibration factor rather than a property of a part,
/// so every one is neutral here and moves only under an overlay. They
/// are named rather than folded into the parameters they scale so that a
/// reader can tell the two apart.
const COMPOSITION: [(&str, f64); 4] = [
    ("stall_exposed_fraction", 1.0),
    ("latency_exposure_scale", 1.0),
    ("fill_exposure_scale", 1.0),
    ("fill_ramp_scale", 1.0),
];

/// Build one device's timing table.
///
/// Everything derivable is derived here rather than written into
/// [`Part`], so that a change to the agent's clock, memory width,
/// preset limits or component tree carries the timing table with it.
pub(crate) fn table(
    device: &KfdDeviceInfo,
    preset: &Preset,
    part: &Part,
    root: &ComponentDef,
) -> TimingTable {
    let clock_mhz = f64::from(device.max_engine_clk_fcompute);
    let cycles = |ns: f64| whole((ns * clock_mhz / 1000.0).round());
    let per_cycle = |bytes_per_second: f64| bytes_per_second / (clock_mhz * 1.0e6);

    let compute_units = u64::from(count_of_type(root, "compute_unit"));
    let xcds = u64::from(count_of_type(root, "xcd")).max(1);
    let memory_channels = (u64::from(device.mem_width) / HBM_PSEUDO_CHANNEL_BITS).max(1);
    let dram_bytes_per_cycle = per_cycle(part.dram_bytes_per_second);
    // The second-level cache is per XCD, and its capacity is what the
    // device already advertises to the guest.
    let l2_bytes_per_xcd = u64::from(device.l2_size_kb) * 1024;

    let keys = &mut BTreeMap::new();

    // The clock itself, so that it merges, validates and reports through
    // the same path as everything else; it is lifted back out to
    // `timing.clock_mhz` when the block is emitted.
    count(keys, CLOCK_KEY, u64::from(device.max_engine_clk_fcompute));

    // -- Shape. Every one of these is read off the agent. ------------------
    count(keys, "compute_units", compute_units);
    count(keys, "xcds", xcds);
    count(keys, "simd_lanes", part.simd_lanes);
    count(
        keys,
        "wave_slots_per_cu",
        preset_number(preset.num_wf_slots),
    );
    // The vector register file, in the same units a wavefront's
    // allocation is counted in: the per-wavefront maximum times the four
    // SIMDs it is spread over. A wave taking the whole allocation then
    // gets one slot per SIMD, and a wave taking a sixteenth of it fills
    // the wave slots, which is how the real part behaves.
    count(
        keys,
        "vector_registers_per_cu",
        preset_number(preset.vgprs_per_wf) * u64::from(device.simd_per_cu).max(1),
    );
    // The scalar file has not limited occupancy since GCN. Sized at the
    // per-wavefront allocation times the wave-slot count so that it never
    // binds, rather than invented as a number that might.
    count(
        keys,
        "scalar_registers_per_cu",
        preset_number(preset.sgprs_per_wf) * preset_number(preset.num_wf_slots),
    );
    count(
        keys,
        "lds_bytes_per_cu",
        preset_number(preset.lds_size_kb) * 1024,
    );

    // -- Front end ----------------------------------------------------------
    let unknown_issue_cycles = ISSUE_CYCLES.iter().map(|(_, c)| *c).max().unwrap_or(1);
    count(keys, "unknown.issue_cycles", unknown_issue_cycles);
    for (class, issue_cycles) in ISSUE_CYCLES {
        count(keys, &format!("{class}.issue_cycles"), issue_cycles);
    }
    for (unit, ports) in PORTS {
        ratio(keys, &format!("{unit}.ports"), ports);
    }
    // The front end, per class. Uniform here; an overlay is what makes
    // them differ, because the differences are measured and not derived.
    ratio(keys, "front_end.issue_cycles", FRONT_END_CYCLES);
    ratio(keys, "front_end.unknown.issue_cycles", FRONT_END_CYCLES);
    for (class, _) in ISSUE_CYCLES {
        ratio(
            keys,
            &format!("front_end.{class}.issue_cycles"),
            FRONT_END_CYCLES,
        );
    }
    for (key, value) in COMPOSITION {
        ratio(keys, key, value);
    }
    // A dispatch ends when its slowest wavefront ends, not its average
    // one. Off unless an overlay says otherwise: the size of the effect
    // is measured, and nothing here can derive it.
    count(keys, "straggler_cycles", 0);
    count(keys, "stall_overlap_wavefronts", 1);
    // The untyped rate is what an opcode whose input type went
    // unrecognised is charged, so it is the slowest the part has rather
    // than an average of the ones below it.
    let matrix = part.matrix;
    count(
        keys,
        "matrix_multiply.macs_per_cycle",
        matrix.f64.min(matrix.f32).min(matrix.integer),
    );
    count(keys, "matrix_multiply.macs_per_cycle.f64", matrix.f64);
    count(keys, "matrix_multiply.macs_per_cycle.f32", matrix.f32);
    count(keys, "matrix_multiply.macs_per_cycle.f16", matrix.f16);
    count(keys, "matrix_multiply.macs_per_cycle.bf16", matrix.bf16);
    count(keys, "matrix_multiply.macs_per_cycle.narrow", matrix.narrow);
    count(
        keys,
        "matrix_multiply.macs_per_cycle.integer",
        matrix.integer,
    );
    ratio(
        keys,
        "vector_memory.lane_addresses_per_cycle",
        LANE_ADDRESSES_PER_CYCLE,
    );

    // -- Caches -------------------------------------------------------------
    cache(
        keys,
        "l1_vector",
        L1_VECTOR_BYTES,
        L1_LINE_BYTES,
        L1_WAYS,
        cycles(L1_HIT_NS),
        1.0,
    );
    cache(
        keys,
        "l1_scalar",
        L1_SCALAR_BYTES,
        L1_LINE_BYTES,
        L1_WAYS,
        cycles(L1_HIT_NS),
        1.0,
    );
    cache(
        keys,
        "l1_instruction",
        L1_INSTRUCTION_BYTES,
        L1_LINE_BYTES,
        L1_WAYS,
        cycles(INSTRUCTION_FETCH_NS),
        1.0,
    );
    cache(
        keys,
        "l2",
        l2_bytes_per_xcd,
        L2_LINE_BYTES,
        L2_WAYS,
        cycles(L2_HIT_NS),
        L2_LINES_PER_CYCLE,
    );
    // The memory-side cache is described per channel, because that is the
    // instance the timing plane replicates; its aggregate rate is the
    // separate `mall.bytes_per_cycle` below.
    cache(
        keys,
        "mall",
        part.mall_bytes / memory_channels,
        L2_LINE_BYTES,
        L2_WAYS,
        cycles(MALL_HIT_NS),
        1.0,
    );

    // -- Memory -------------------------------------------------------------
    count(keys, "dram.latency_cycles", cycles(DRAM_LATENCY_NS));
    ratio(keys, "dram.bytes_per_cycle", dram_bytes_per_cycle);
    ratio(
        keys,
        "mall.bytes_per_cycle",
        per_cycle(part.mall_bytes_per_second),
    );
    ratio(
        keys,
        "fabric.bytes_per_cycle",
        dram_bytes_per_cycle * FABRIC_WIDTH_OVER_DRAM,
    );
    count(keys, "memory_channels", memory_channels);
    count(keys, "fabric_request_bytes", L2_LINE_BYTES);
    count(
        keys,
        "miss_status_registers_per_cu",
        MISS_STATUS_REGISTERS_PER_CU,
    );

    // -- Local data share ---------------------------------------------------
    count(keys, "lds.banks", LDS_BANKS);
    count(keys, "lds.bank_bytes", LDS_BANK_BYTES);
    count(keys, "lds.latency_cycles", cycles(LDS_LATENCY_NS));
    count(keys, "lds.lanes_per_phase", LDS_LANES_PER_PHASE);

    // -- Dispatch -----------------------------------------------------------
    count(keys, "dispatch.start_cycles", cycles(DISPATCH_START_NS));
    // The completion signal is one memory round trip, so it is exactly the
    // DRAM latency rather than a second assumption.
    count(keys, "dispatch.end_cycles", cycles(DRAM_LATENCY_NS));
    count(
        keys,
        "dispatch.invalidate_cycles",
        cycles(LAUNCH_INVALIDATE_NS),
    );
    ratio(keys, "dispatch.workgroups_per_cycle", WORKGROUPS_PER_CYCLE);

    TimingTable::from_pairs(std::mem::take(keys))
}

/// One whole-numbered key: a count, a capacity or a cycle latency.
fn count(keys: &mut BTreeMap<String, Number>, key: &str, value: u64) {
    keys.insert(key.to_string(), Number::from(value));
}

/// One rate.
fn ratio(keys: &mut BTreeMap<String, Number>, key: &str, value: f64) {
    keys.insert(key.to_string(), rate(value));
}

/// One cache level's geometry and rates, with the set count derived from
/// the capacity so that the capacity is what is actually stated.
fn cache(
    keys: &mut BTreeMap<String, Number>,
    prefix: &str,
    capacity_bytes: u64,
    line_bytes: u64,
    ways: u64,
    hit_cycles: u64,
    lines_per_cycle: f64,
) {
    let sets = (capacity_bytes / (line_bytes * ways)).max(1);
    keys.insert(format!("{prefix}.line_bytes"), Number::from(line_bytes));
    keys.insert(format!("{prefix}.sets"), Number::from(sets));
    keys.insert(format!("{prefix}.ways"), Number::from(ways));
    keys.insert(format!("{prefix}.hit_cycles"), Number::from(hit_cycles));
    keys.insert(format!("{prefix}.lines_per_cycle"), rate(lines_per_cycle));
}

/// A rate as a JSON number, rounded to a thousandth.
///
/// Rounded because the config is read by people as well as by the
/// emulator, and `2962.9629629629626` says nothing `2962.963` does not.
/// The fallback cannot be reached: `Number::from_f64` refuses only NaN
/// and infinity, and every rate here is a finite quotient of two
/// positive constants.
fn rate(value: f64) -> Number {
    let rounded = (value * 1000.0).round() / 1000.0;
    Number::from_f64(rounded).unwrap_or_else(|| Number::from(whole(rounded)))
}

/// A non-negative `f64` as a `u64`, saturating rather than wrapping.
///
/// Every caller passes a rounded product of positive constants, so the
/// clamp is a guard and not a conversion policy.
fn whole(value: f64) -> u64 {
    if value.is_finite() && value > 0.0 {
        value as u64
    } else {
        0
    }
}

/// One preset value, which is carried as a string because that is what a
/// component's `config` holds.
///
/// A preset that has lost its number would have failed the build script
/// long before this, so the fallback is unreachable; it is a `1` rather
/// than a panic because a timing table is not worth aborting a run over,
/// and a `1` makes the affected parameter obviously wrong instead of
/// plausibly wrong.
fn preset_number(value: &str) -> u64 {
    value.parse::<u64>().unwrap_or(1)
}

/// How many components of `kind` a tree holds, multiplying out the
/// `[0:N]` ranges on the way down.
pub(crate) fn count_of_type(node: &ComponentDef, kind: &str) -> u32 {
    let span = node
        .name
        .rsplit_once("[0:")
        .and_then(|(_, rest)| rest.strip_suffix(']'))
        .and_then(|n| n.parse::<u32>().ok())
        .unwrap_or(1);
    if node.r#type == kind {
        return span;
    }
    span * node
        .children
        .iter()
        .map(|child| count_of_type(child, kind))
        .sum::<u32>()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::agents::{mi300x, mi350x, mi450x};

    /// Every key the timing plane reads must be in the table, because a
    /// key the config does not name resolves to the slowest reasonable
    /// value for it — which is the right behaviour for a forgotten
    /// parameter and the wrong one for a shipped table.
    ///
    /// The list is `rocjitsu/lib/rocjitsu/src/rocjitsu/vm/timing/tuning.h`
    /// transcribed. It is a transcription and not a generated list, so it
    /// is the thing to update when that header grows a field — and this
    /// test is what says so out loud.
    #[test]
    fn every_agent_names_every_key_the_plane_reads() {
        let mut want: Vec<String> = vec![
            CLOCK_KEY.into(),
            "compute_units".into(),
            "xcds".into(),
            "simd_lanes".into(),
            "wave_slots_per_cu".into(),
            "vector_registers_per_cu".into(),
            "scalar_registers_per_cu".into(),
            "lds_bytes_per_cu".into(),
            "unknown.issue_cycles".into(),
            "matrix_multiply.macs_per_cycle".into(),
            "matrix_multiply.macs_per_cycle.f64".into(),
            "matrix_multiply.macs_per_cycle.f32".into(),
            "matrix_multiply.macs_per_cycle.f16".into(),
            "matrix_multiply.macs_per_cycle.bf16".into(),
            "matrix_multiply.macs_per_cycle.narrow".into(),
            "matrix_multiply.macs_per_cycle.integer".into(),
            "vector_memory.lane_addresses_per_cycle".into(),
            "dram.latency_cycles".into(),
            "dram.bytes_per_cycle".into(),
            "mall.bytes_per_cycle".into(),
            "fabric.bytes_per_cycle".into(),
            "memory_channels".into(),
            "fabric_request_bytes".into(),
            "miss_status_registers_per_cu".into(),
            "lds.banks".into(),
            "lds.bank_bytes".into(),
            "lds.latency_cycles".into(),
            "lds.lanes_per_phase".into(),
            "dispatch.start_cycles".into(),
            "dispatch.end_cycles".into(),
            "dispatch.invalidate_cycles".into(),
            "dispatch.workgroups_per_cycle".into(),
        ];
        for (class, _) in ISSUE_CYCLES {
            want.push(format!("{class}.issue_cycles"));
        }
        for (unit, _) in PORTS {
            want.push(format!("{unit}.ports"));
        }
        want.push("front_end.issue_cycles".into());
        want.push("front_end.unknown.issue_cycles".into());
        for (class, _) in ISSUE_CYCLES {
            want.push(format!("front_end.{class}.issue_cycles"));
        }
        for (key, _) in COMPOSITION {
            want.push(key.into());
        }
        want.push("straggler_cycles".into());
        want.push("stall_overlap_wavefronts".into());
        for level in ["l1_vector", "l1_scalar", "l1_instruction", "l2", "mall"] {
            for field in [
                "line_bytes",
                "sets",
                "ways",
                "hit_cycles",
                "lines_per_cycle",
            ] {
                want.push(format!("{level}.{field}"));
            }
        }
        want.sort();

        for (name, agent) in crate::agents::agents() {
            let mut have: Vec<String> = agent.timing.keys().map(str::to_string).collect();
            have.sort();
            assert_eq!(have, want, "`{name}` timing table");
        }
    }

    /// An opcode nobody classified has to cost the most, not the least.
    #[test]
    fn an_unclassified_opcode_costs_the_most_the_table_names() {
        for (name, agent) in crate::agents::agents() {
            let unknown = agent
                .timing
                .get("unknown.issue_cycles")
                .and_then(Number::as_u64)
                .unwrap();
            for (class, _) in ISSUE_CYCLES {
                let cost = agent
                    .timing
                    .get(&format!("{class}.issue_cycles"))
                    .and_then(Number::as_u64)
                    .unwrap();
                assert!(cost <= unknown, "`{name}` {class} costs more than unknown");
            }
        }
    }

    /// The shape half of the table is read off the agent, not written
    /// beside it. Tautological against the current code and deliberately
    /// so: it fails the day somebody writes a literal back in, which is
    /// exactly how the per-CU limits drifted from their preset before.
    #[test]
    fn the_shape_comes_from_the_agent() {
        for (agent, wave_slots, vgprs, lds_kb) in [
            (mi300x(), 32, 512, 64),
            (mi350x(), 32, 512, 160),
            (mi450x(), 64, 1024, 320),
        ] {
            let get = |key: &str| agent.timing.get(key).and_then(Number::as_u64).unwrap();
            assert_eq!(get("compute_units"), 256);
            assert_eq!(get("xcds"), 8);
            assert_eq!(get("wave_slots_per_cu"), wave_slots);
            assert_eq!(get("vector_registers_per_cu"), vgprs * 4);
            assert_eq!(get("lds_bytes_per_cu"), lds_kb * 1024);
            assert_eq!(
                get(CLOCK_KEY),
                u64::from(agent.vm.gpu.device.max_engine_clk_fcompute)
            );
            assert_eq!(
                get("memory_channels"),
                u64::from(agent.vm.gpu.device.mem_width) / 64
            );
        }
    }

    /// The bandwidths really are the published peaks divided by the
    /// agent's own clock. Worked here at the two clocks the builtins use,
    /// because an arithmetic slip in `per_cycle` is invisible in a table
    /// of plausible-looking numbers.
    #[test]
    fn bandwidth_is_the_published_peak_over_the_clock() {
        let rate_of = |agent: &mirage_core::agent::AgentDef, key: &str| {
            agent.timing.get(key).and_then(Number::as_f64).unwrap()
        };
        // MI300X: 5.3 TB/s at 2100 MHz.
        assert!((rate_of(&mi300x(), "dram.bytes_per_cycle") - 2523.81).abs() < 0.01);
        // MI350X: 8.0 TB/s at 2700 MHz, and a fabric twice as wide.
        assert!((rate_of(&mi350x(), "dram.bytes_per_cycle") - 2962.963).abs() < 0.01);
        assert!((rate_of(&mi350x(), "fabric.bytes_per_cycle") - 5925.926).abs() < 0.01);
        // The memory-side cache is the one place the two parts share a
        // published figure, so the ratio of their rates is the ratio of
        // their clocks.
        let a = rate_of(&mi300x(), "mall.bytes_per_cycle");
        let b = rate_of(&mi350x(), "mall.bytes_per_cycle");
        assert!((a / b - 2700.0 / 2100.0).abs() < 0.001);
    }

    /// A cache's stated capacity has to be what its geometry multiplies
    /// out to, or the timing plane models a cache nobody described.
    #[test]
    fn cache_geometry_multiplies_out_to_the_stated_capacity() {
        let agent = mi350x();
        let get = |key: &str| agent.timing.get(key).and_then(Number::as_u64).unwrap();
        for (level, capacity) in [
            ("l1_vector", L1_VECTOR_BYTES),
            ("l1_scalar", L1_SCALAR_BYTES),
            ("l1_instruction", L1_INSTRUCTION_BYTES),
            ("l2", u64::from(agent.vm.gpu.device.l2_size_kb) * 1024),
            ("mall", 256 * 1024 * 1024 / 128),
        ] {
            let product = get(&format!("{level}.line_bytes"))
                * get(&format!("{level}.sets"))
                * get(&format!("{level}.ways"));
            assert_eq!(product, capacity, "{level}");
        }
    }

    /// Latencies are one assumption in nanoseconds, so the faster part
    /// must come out with more cycles for the same wall time.
    #[test]
    fn a_faster_clock_buys_more_cycles_of_the_same_latency() {
        let cycles = |agent: &mirage_core::agent::AgentDef, key: &str| {
            agent.timing.get(key).and_then(Number::as_u64).unwrap()
        };
        assert_eq!(cycles(&mi300x(), "dram.latency_cycles"), 798);
        assert_eq!(cycles(&mi350x(), "dram.latency_cycles"), 1026);
        // And the completion signal is that same round trip, not a
        // second guess at it.
        assert_eq!(
            cycles(&mi350x(), "dispatch.end_cycles"),
            cycles(&mi350x(), "dram.latency_cycles")
        );
    }

    /// MI450X carries CDNA4's rates because nothing about the MI400
    /// series is published in the detail a timing table needs. Asserted
    /// so that a future MI400-series table is a deliberate change to this
    /// test as well as to the numbers.
    #[test]
    fn mi450x_carries_cdna4_rates() {
        assert_eq!(MI450X.matrix.f16, MI350X.matrix.f16);
        assert_eq!(MI450X.dram_bytes_per_second, MI350X.dram_bytes_per_second);
        // Except its SIMD, which is its own.
        assert_eq!(MI450X.simd_lanes, 32);
        assert_eq!(
            mi450x().timing.get("simd_lanes").and_then(Number::as_u64),
            Some(32)
        );
    }
}
