//! Placement tables and fail-closed record/placement validation.
use super::manifest::Manifest;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::{collections::BTreeSet, path::Path};
pub type Log = Vec<[u32; 8]>;
pub struct Table {
    pub max_k: u32,
    pub entries: Vec<[u32; 2]>,
    pub counts: Vec<u32>,
}
#[derive(Serialize, Debug)]
pub struct PlacementReport {
    pub placed: usize,
    pub total: usize,
    pub every_tile_once: bool,
    pub natural_match: f64,
    pub traced_sets_match: Option<bool>,
    pub renamed: Option<usize>,
    pub slot_ok: Option<bool>,
}
impl PlacementReport {
    pub fn valid(&self) -> bool {
        self.placed == self.total
            && self.every_tile_once
            && self.traced_sets_match != Some(false)
            && self.slot_ok != Some(false)
    }
}
pub fn read_log(path: &Path, m: &Manifest) -> Result<Log> {
    let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if m.log_words != 8 || b.len() as u64 != m.num_log_records * 32 {
        bail!("{}: incorrect log length", path.display());
    }
    let log = b
        .chunks_exact(32)
        .map(|b| {
            std::array::from_fn(|i| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()))
        })
        .collect::<Log>();
    if log.is_empty() || log.iter().all(|r| r[4] == 0) {
        bail!("log is empty (no HW_ID1 values)");
    }
    Ok(log)
}
fn tiles(m: &Manifest, log: &Log, hardware: bool) -> Result<bool> {
    let mut ids = BTreeSet::new();
    for r in log {
        let i = if hardware { 6 } else { 2 };
        let id = if let Some(c) = &m.cluster {
            if r[i] as u64 >= c.num_clusters {
                return Ok(false);
            }
            r[i] as u64
        } else {
            let [gx, gy, gz] = m.workgroups;
            // A kernel that does not enable workgroup ids Y and Z finds ttmp7 unwritten on gfx1250, so
            // in a one-dimensional grid it carries no id.
            let (y, z) = if gy == 1 && gz == 1 {
                (0, 0)
            } else {
                ((r[i + 1] & 0xffff) as u64, (r[i + 1] >> 16) as u64)
            };
            let x = r[i] as u64;
            if x >= gx || y >= gy || z >= gz {
                return Ok(false);
            }
            (z * gy + y) * gx + x
        };
        if !ids.insert(id) {
            return Ok(false);
        }
    }
    Ok(ids.len() as u64 == m.num_log_records)
}
pub fn counts(m: &Manifest, log: &Log) -> Result<Vec<u32>> {
    if log.len() as u64 != m.num_log_records {
        bail!("incorrect record count");
    }
    let mut counts = vec![0u32; m.num_keys as usize];
    let mut pairs = BTreeSet::new();
    for r in log {
        if r[0] as u64 >= m.num_keys {
            bail!("placement key {} outside 0..{}", r[0], m.num_keys);
        }
        if !pairs.insert((r[0], r[1])) {
            bail!("duplicate (key, turn)");
        }
        counts[r[0] as usize] = counts[r[0] as usize]
            .checked_add(1)
            .context("turn count overflow")?;
        if let Some(c) = &m.cluster {
            if !c.valid_rank(r[3]) || !c.valid_rank(r[7]) {
                bail!("cluster rank outside its range");
            }
        }
    }
    for r in log {
        if r[1] >= counts[r[0] as usize] {
            bail!("turn sequence contains a gap or overflow");
        }
    }
    Ok(counts)
}
impl Table {
    pub fn build(m: &Manifest, log: &Log) -> Result<Self> {
        let counts = counts(m, log)?;
        if !tiles(m, log, true)? || !tiles(m, log, false)? {
            bail!("record run did not run every tile exactly once");
        }
        if log
            .iter()
            .any(|r| r[2] != r[6] || (m.cluster.is_none() && r[3] != r[7]))
        {
            bail!("record run changed hardware ids");
        }
        let max_k = *counts.iter().max().context("no placement keys")?;
        if max_k == 0 {
            bail!("empty placement table");
        }
        let mut entries = vec![[u32::MAX; 2]; m.num_keys as usize * max_k as usize];
        for r in log {
            entries[r[0] as usize * max_k as usize + r[1] as usize] = if m.cluster.is_some() {
                [r[2], r[3]]
            } else {
                [r[6], r[7]]
            };
        }
        Ok(Self {
            max_k,
            entries,
            counts,
        })
    }
    pub fn check_record(&self, m: &Manifest, log: &Log) -> Result<()> {
        if counts(m, log)? != self.counts {
            bail!("per-key turn counts changed between record runs; placement is unstable and the table would overflow");
        }
        if !tiles(m, log, true)? || !tiles(m, log, false)? {
            bail!("record run did not run every tile once");
        }
        Ok(())
    }
    pub fn write(&self, path: &Path) -> Result<()> {
        std::fs::write(
            path,
            self.entries
                .iter()
                .flat_map(|e| e.iter().flat_map(|x| x.to_le_bytes()))
                .collect::<Vec<_>>(),
        )?;
        Ok(())
    }
    pub fn check(&self, m: &Manifest, log: &Log, slot: Option<&[u32]>) -> Result<PlacementReport> {
        let got_counts = counts(m, log)?;
        if got_counts != self.counts {
            bail!("placement turn counts differ from the table");
        }
        let mut placed = 0;
        let mut natural = 0;
        let mut renamed = 0;
        let mut slot_ok = slot.map(|_| true);
        let mut traced_ok = true;
        for r in log {
            let entry = self
                .entries
                .get(r[0] as usize * self.max_k as usize + r[1] as usize)
                .context("table index out of range")?;
            let traced = m.cluster.as_ref().is_some_and(|c| r[0] < c.shader_engines);
            if r[1] < self.max_k
                && entry[0] != u32::MAX
                && r[2] == entry[0]
                && (if m.cluster.is_some() {
                    !traced || r[3] ^ r[7] == entry[1]
                } else {
                    r[3] == entry[1]
                })
            {
                placed += 1;
            }
            if r[2] == r[6] && (m.cluster.is_some() || r[3] == r[7]) {
                natural += 1;
            }
            if traced {
                if r[7] != 0 {
                    renamed += 1;
                }
                if let Some(slot) = slot {
                    let v = *slot
                        .get(r[6] as usize)
                        .context("slot hardware cluster id out of bounds")?;
                    let c = m.cluster.as_ref().unwrap();
                    if v >> 31 != 1
                        || v & 0x3ff != r[0] * 32 + c.traced_wgp
                        || (v >> 16) & 0xff != r[3]
                    {
                        slot_ok = Some(false);
                    }
                }
            }
        }
        if let Some(c) = &m.cluster {
            for key in 0..c.shader_engines {
                let expected: BTreeSet<_> = self.entries
                    [key as usize * self.max_k as usize..(key + 1) as usize * self.max_k as usize]
                    .iter()
                    .filter(|e| e[0] != u32::MAX)
                    .map(|e| (e[0], e[1]))
                    .collect();
                let got: BTreeSet<_> = log
                    .iter()
                    .filter(|r| r[0] == key)
                    .map(|r| (r[2], r[3] ^ r[7]))
                    .collect();
                traced_ok &= expected == got;
            }
        }
        Ok(PlacementReport {
            placed,
            total: log.len(),
            every_tile_once: tiles(m, log, false)?,
            natural_match: natural as f64 / log.len() as f64,
            traced_sets_match: m.cluster.as_ref().map(|_| traced_ok),
            renamed: m.cluster.as_ref().map(|_| renamed),
            slot_ok,
        })
    }
}
