//! Rewrite the application's kernel for one dispatch and lay out its launch.

use std::path::Path;

use anyhow::{bail, Context, Result};
use remapper_rewrite::{
    manifest::{LaunchLayout, Manifest},
    rewrite::{self, LaunchModel, Prefetch, RewriteOptions},
    sparse_table,
};
use serde_json::{json, Value};

use crate::{config::LaunchChoice, Config, Dispatch, KernelInfo};

/// Resident waves per WGP assumed when sizing the admission slack (4 SIMDs × 32 wave slots); parked arrivals per WGP
/// cannot exceed the workgroups it can hold, so this bounds the slack a launch can need.
pub const ADMISSION_WAVES_PER_WGP: u32 = 128;

pub fn cluster_dims(d: &Dispatch) -> [u32; 3] {
    if d.cluster_size == [0; 3] {
        [1, 1, 1]
    } else {
        d.cluster_size
    }
}

pub fn workgroups(d: &Dispatch) -> Result<[u64; 3]> {
    if d.workgroup_size.contains(&0) {
        bail!("zero workgroup dimension");
    }
    Ok(std::array::from_fn(|i| {
        (d.grid_size[i] as u64).div_ceil(d.workgroup_size[i] as u64)
    }))
}

/// Rewrite `code`'s `symbol` into `dir` and fill the launch layout; the result is saved as `dir/manifest.json`.
pub fn prepare(
    cfg: &Config,
    info: &KernelInfo<'_>,
    dir: &Path,
    report: &mut Value,
) -> Result<Manifest> {
    let (app, code) = (&info.app, info.code_object);
    let symbol = info.symbol.strip_suffix(".kd").unwrap_or(info.symbol);
    let workgroups = workgroups(app)?;
    let waves = (app
        .workgroup_size
        .iter()
        .map(|&v| v as u64)
        .product::<u64>())
    .div_ceil(32) as u32;
    let cluster_dims = cluster_dims(app);
    if cluster_dims.contains(&0) {
        bail!("invalid zero cluster dimension");
    }
    let cluster_size: u32 = cluster_dims.iter().product();
    let se = info.shader_engines;
    let num_workgroups: u64 = workgroups.iter().product();
    let padded = se as u64 * cfg.wgps_per_se as u64;
    let sparse_reason = if cluster_size > 1 {
        Some("cluster dispatch")
    } else if num_workgroups >= padded {
        Some("the launch has at least one workgroup per WGP")
    } else {
        None
    };
    let launch_model = match (cfg.launch_model, sparse_reason) {
        (LaunchChoice::Table, _) => LaunchModel::Table,
        (LaunchChoice::Admission, _) => LaunchModel::Admission,
        (LaunchChoice::Sparse, Some(r)) => bail!("launch model sparse does not apply: {r}"),
        (LaunchChoice::Sparse, None) => LaunchModel::Sparse,
        (LaunchChoice::Auto, None) => LaunchModel::Sparse,
        (LaunchChoice::Auto, Some(_)) => LaunchModel::Table,
    };
    let sparse = launch_model == LaunchModel::Sparse;
    let admission = launch_model == LaunchModel::Admission;
    report["app_dispatch"] = serde_json::to_value(app)?;
    report["collection"] = json!({
        "target_cu": cfg.target_cu,
        "shader_engines": se,
        "repetitions": cfg.repetitions,
        "launch_model": launch_model.name(),
        "natural_baseline": cfg.natural_baseline && !sparse,
        "original_baseline": cfg.original_baseline,
    });
    if let Some(m) = cfg.loop_markers {
        report["collection"]["loop_markers"] = json!(m.name());
    }
    // Admission slack in workgroups; cluster launches add whole clusters, as whole rows of clusters along Y when the
    // grid is two-dimensional, so flat cluster ids (y * clusters per row + x) stay valid.
    let mut slack = if admission {
        cfg.admission_slack
            .unwrap_or(padded * (ADMISSION_WAVES_PER_WGP as u64).div_ceil(waves as u64))
    } else {
        0
    };
    let mut padded_wg_counts = workgroups;
    if admission && cluster_size > 1 {
        let [cx, cy, _] = cluster_dims.map(u64::from);
        if workgroups[1] > 1 {
            padded_wg_counts[1] += slack.div_ceil(workgroups[0] * cy).max(1) * cy;
        } else {
            padded_wg_counts[0] += slack.div_ceil(cx).max(1) * cx;
        }
        slack = padded_wg_counts.iter().product::<u64>() - num_workgroups;
    }
    // Every real workgroup of a sparse launch is the first on its WGP, where a cold body hurts most; cluster
    // prologues need prefetch to avoid displacing the body's instruction working set.
    let prefetch = cfg.prefetch.unwrap_or(if cluster_size > 1 || sparse {
        Prefetch::PerWave
    } else {
        Prefetch::None
    });
    let opts = RewriteOptions {
        dispatch_index: info.dispatch_id,
        workgroups,
        waves_per_workgroup: waves,
        cluster_size,
        cluster_dims,
        shader_engines: se,
        traced_wgp: cfg.target_cu,
        prefetch,
        launch_model,
        padded_workgroups: num_workgroups + slack,
        loop_markers: cfg.loop_markers,
    };
    let mut m = rewrite::rewrite(code, symbol, &opts, dir)?;
    let wg = app.workgroup_size;
    if admission {
        let total = num_workgroups + slack;
        let grid = |i: usize| {
            u32::try_from(padded_wg_counts[i] * wg[i] as u64).context("padded grid overflow")
        };
        m.launch = Some(LaunchLayout {
            model: launch_model.name().into(),
            // Non-cluster launches flatten into X (the schedule supplies every id); clusters keep their layout.
            padded_grid_size: if cluster_size > 1 {
                [grid(0)?, grid(1)?, grid(2)?]
            } else {
                [
                    u32::try_from(total * wg[0] as u64).context("padded grid overflow")?,
                    wg[1],
                    wg[2],
                ]
            },
            padded_workgroups: total,
            assignment: vec![],
            traced_locations: (0..se)
                .map(|e| sparse_table::location(e, 0, cfg.target_cu))
                .collect(),
            slack,
        });
    }
    if sparse {
        m.launch = Some(LaunchLayout {
            model: launch_model.name().into(),
            padded_grid_size: [
                u32::try_from(padded * wg[0] as u64).context("padded grid overflow")?,
                wg[1],
                wg[2],
            ],
            padded_workgroups: padded,
            assignment: vec![],
            traced_locations: vec![],
            slack: 0,
        });
        m.num_log_records = padded;
    }
    if let Some(l) = &m.launch {
        check_padded_grid(app, l.padded_grid_size)?;
    }
    m.save(dir)?;
    report["symbol"] = json!(m.symbol);
    report["rewrite"] = serde_json::to_value(&m.rewrite)?;
    Ok(m)
}

/// A padded grid must be whole workgroups and, for clusters, whole clusters whose counts fit the extended packet.
pub fn check_padded_grid(app: &Dispatch, grid: [u32; 3]) -> Result<()> {
    let wg = app.workgroup_size;
    if wg.contains(&0) || (0..3).any(|i| grid[i] % wg[i] != 0) {
        bail!("padded grid {grid:?} is not a whole number of {wg:?} workgroups");
    }
    if app.cluster_size == [0; 3] {
        return Ok(());
    }
    let counts: [u32; 3] = std::array::from_fn(|i| grid[i] / wg[i]);
    let dims = app.cluster_size;
    if dims.contains(&0) || (0..3).any(|i| counts[i] % dims[i] != 0) {
        bail!("padded grid {counts:?} is not a whole number of {dims:?} clusters");
    }
    u16::try_from(counts[1] / dims[1]).context("padded cluster count in Y exceeds 65535")?;
    u16::try_from(counts[2] / dims[2]).context("padded cluster count in Z exceeds 65535")?;
    Ok(())
}
