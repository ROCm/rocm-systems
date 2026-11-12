//! Sparse launch model: probe validation, assignment of logical workgroups to locations, the one-entry-per-location
//! table, and placement checks. Fail closed: a launch that does not put exactly one workgroup on every WGP, or a
//! replay whose workgroups did not run where they were assigned, invalidates the run.
use super::manifest::{AssignedWorkgroup, Manifest, FAKE, NO_ENTRY};
use super::table::Log;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// Location key layout shared with the prologue: `(se * 2 + sa) * 16 + wgp`.
pub fn location(se: u32, sa: u32, wgp: u32) -> u32 {
    (se * 2 + sa) * 16 + wgp
}

/// Shader engines in assignment order: one per XCC first (0, 2, 4, ...), then the other one of each XCC.
pub fn engine_order(shader_engines: u32) -> Vec<u32> {
    if shader_engines % 2 != 0 {
        return (0..shader_engines).collect();
    }
    let half = shader_engines / 2;
    (0..shader_engines)
        .map(|i| 2 * (i % half) + (i / half) % 2)
        .collect()
}

/// Each probe log must hold one record per padded workgroup, every one the first arrival at a distinct location,
/// with the same set of locations in every probe replay and every traced location present. Returns the locations.
pub fn validate_probe(
    m: &Manifest,
    logs: &[Log],
    shader_engines: u32,
    traced_wgp: u32,
) -> Result<BTreeSet<u32>> {
    let mut set: Option<BTreeSet<u32>> = None;
    for (i, log) in logs.iter().enumerate() {
        if log.len() as u64 != m.num_log_records {
            bail!(
                "probe {i}: {} records, expected {}",
                log.len(),
                m.num_log_records
            );
        }
        let mut here = BTreeSet::new();
        for r in log {
            if r[1] != 0 || !here.insert(r[0]) {
                bail!("padded launch did not place one workgroup per WGP (location {} got a second workgroup in probe {i}); check --wgps-per-se", r[0]);
            }
            if r[0] as u64 >= m.num_keys || r[2] != FAKE {
                bail!("probe {i}: malformed record {r:?}");
            }
        }
        match &set {
            None => set = Some(here),
            Some(s) if *s != here => {
                bail!("the padded launch used a different set of WGPs in probe {i}")
            }
            _ => {}
        }
    }
    let set = set.context("no probe logs")?;
    for se in 0..shader_engines {
        if !set.contains(&location(se, 0, traced_wgp)) {
            bail!("traced WGP {traced_wgp} of shader engine {se} received no workgroup in the padded launch");
        }
    }
    Ok(set)
}

/// Assign the N logical workgroups (linear order, x fastest): the first one per shader engine to its traced WGP,
/// in `engine_order`, and any further ones round-robin over shader engines to their other WGPs. Returns the
/// assignment and the traced locations that hold a real workgroup.
pub fn assign(
    locations: &BTreeSet<u32>,
    workgroups: [u64; 3],
    shader_engines: u32,
    traced_wgp: u32,
) -> Result<(Vec<AssignedWorkgroup>, Vec<u32>)> {
    let [gx, gy, gz] = workgroups;
    let n = gx * gy * gz;
    if n == 0 || n > locations.len() as u64 {
        bail!(
            "{n} workgroups do not fit one per WGP on {} WGPs",
            locations.len()
        );
    }
    if gy > 0xffff || gz > 0xffff {
        bail!("grid Y/Z exceed ttmp7's 16-bit fields");
    }
    let order = engine_order(shader_engines);
    let traced: Vec<u32> = order
        .iter()
        .map(|&se| location(se, 0, traced_wgp))
        .collect();
    // Remaining locations per shader engine in cursor order (shader array 0, then 1), excluding the traced WGP.
    let mut spare: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for &loc in locations {
        if !traced.contains(&loc) {
            spare.entry(loc / 32).or_default().push(loc);
        }
    }
    let mut out = Vec::with_capacity(n as usize);
    let mut next_engine = 0usize;
    for j in 0..n {
        let (x, y, z) = (j % gx, (j / gx) % gy, j / (gx * gy));
        let loc = if (j as usize) < traced.len() {
            traced[j as usize]
        } else {
            let mut found = None;
            for _ in 0..order.len() {
                let se = order[next_engine % order.len()];
                next_engine += 1;
                if let Some(list) = spare.get_mut(&se).filter(|l| !l.is_empty()) {
                    found = Some(list.remove(0));
                    break;
                }
            }
            found.context("ran out of WGPs while assigning workgroups")?
        };
        out.push(AssignedWorkgroup {
            location: loc,
            ttmp9: x as u32,
            ttmp7: (y | z << 16) as u32,
        });
    }
    let traced_used = traced.into_iter().take(n as usize).collect();
    Ok((out, traced_used))
}

/// `num_keys` entries of (ttmp9, ttmp7): assigned locations hold their logical workgroup, other probed locations
/// FAKE, locations never probed NO_ENTRY.
pub fn build_table(
    num_keys: u64,
    locations: &BTreeSet<u32>,
    assignment: &[AssignedWorkgroup],
) -> Result<Vec<[u32; 2]>> {
    let mut entries = vec![[NO_ENTRY, NO_ENTRY]; num_keys as usize];
    for &loc in locations {
        *entries
            .get_mut(loc as usize)
            .context("location outside the key space")? = [FAKE, 0];
    }
    for a in assignment {
        let e = entries
            .get_mut(a.location as usize)
            .context("assigned location outside the key space")?;
        if e[0] != FAKE {
            bail!("location {} assigned twice or never probed", a.location);
        }
        *e = [a.ttmp9, a.ttmp7];
    }
    Ok(entries)
}

pub fn table_bytes(entries: &[[u32; 2]]) -> Vec<u8> {
    entries
        .iter()
        .flat_map(|e| e.iter().flat_map(|x| x.to_le_bytes()))
        .collect()
}

#[derive(Serialize, Debug)]
pub struct SparsePlacement {
    pub placed: usize,
    pub total: usize,
    pub fakes: usize,
    pub every_tile_once: bool,
}

/// A table-mode log: one first arrival per probed location, each assigned location running its logical workgroup
/// and every other location FAKE.
pub fn check_placement(
    m: &Manifest,
    log: &Log,
    locations: &BTreeSet<u32>,
    assignment: &[AssignedWorkgroup],
) -> Result<SparsePlacement> {
    if log.len() as u64 != m.num_log_records {
        bail!("{} log records, expected {}", log.len(), m.num_log_records);
    }
    let expected: BTreeMap<u32, [u32; 2]> = assignment
        .iter()
        .map(|a| (a.location, [a.ttmp9, a.ttmp7]))
        .collect();
    let mut seen = BTreeSet::new();
    let (mut placed, mut fakes) = (0, 0);
    for r in log {
        if r[1] != 0 || !seen.insert(r[0]) || !locations.contains(&r[0]) {
            bail!(
                "location {} received an unexpected workgroup (turn {})",
                r[0],
                r[1]
            );
        }
        match expected.get(&r[0]) {
            Some(&want) if [r[2], r[3]] == want => placed += 1,
            None if r[2] == FAKE => fakes += 1,
            _ => bail!(
                "location {} ran {:?}, expected {:?}",
                r[0],
                [r[2], r[3]],
                expected.get(&r[0])
            ),
        }
    }
    Ok(SparsePlacement {
        placed,
        total: assignment.len(),
        fakes,
        every_tile_once: placed == assignment.len(),
    })
}
