//! Opt-in loop markers. Selected barrier instructions in the body become an `s_branch` to an out-of-line stub placed
//! after the prologue, which writes an `s_ttracedata` word, runs the original instruction and branches back. ATT
//! records the words of every wave on a shader engine, not only the traced WGP's, so each replay shows when every
//! workgroup reached each marked barrier. Every other body byte and address is unchanged.
//!
//! Marker word: `tag << 24 | site << 16 | ttmp6[15:0]` (tag A3 cluster signal, A4 cluster wait, A5 workgroup wait;
//! site = index into the manifest's marker sites; ttmp6[15:0] carries the hardware cluster rank). The stub writes
//! only M0, with `s_mov_b32`/`s_pack_ll_b32_b16`, which leave SCC alone. When the body may read M0 (names it, or uses
//! an instruction that reads it implicitly) the stub saves and restores it in an SGPR the body never references.

use super::analyze::{Analysis, NUM_SGPRS};
use super::isa::Inst;
use super::manifest::{LoopMarkerInfo, MarkerSite};
use super::prologue::assemble;
use anyhow::{anyhow, bail, Result};
use serde::Serialize;

pub const TAG_CLUSTER_SIGNAL: u8 = 0xA3;
pub const TAG_CLUSTER_WAIT: u8 = 0xA4;
pub const TAG_WORKGROUP_WAIT: u8 = 0xA5;
const MAX_SITES: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LoopMarkers {
    /// `s_barrier_signal -3` (A3) and `s_barrier_wait 0xfffd` (A4).
    Cluster,
    /// The cluster sites plus every workgroup barrier wait `s_barrier_wait 0xffff` (A5).
    ClusterAndWorkgroup,
}

impl LoopMarkers {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cluster => "cluster",
            Self::ClusterAndWorkgroup => "cluster-and-workgroup",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        [Self::Cluster, Self::ClusterAndWorkgroup]
            .into_iter()
            .find(|m| m.name() == s)
    }
}

/// Stub size: [save M0], marker word (pack with a literal), s_ttracedata, [restore M0], original, branch back.
pub fn stub_bytes(save: Option<u32>) -> usize {
    if save.is_some() {
        28
    } else {
        20
    }
}

fn tag_of(mnemonic: &str, operands: &str, which: LoopMarkers) -> Option<u8> {
    match (mnemonic, operands) {
        ("s_barrier_signal", "-3") => Some(TAG_CLUSTER_SIGNAL),
        ("s_barrier_wait", "0xfffd" | "-3") => Some(TAG_CLUSTER_WAIT),
        ("s_barrier_wait", "0xffff" | "-1") if which == LoopMarkers::ClusterAndWorkgroup => {
            Some(TAG_WORKGROUP_WAIT)
        }
        _ => None,
    }
}

/// Body instructions to mark: (instruction index, tag).
pub fn select_sites(a: &Analysis, which: LoopMarkers) -> Result<Vec<(usize, u8)>> {
    let sites: Vec<_> = a
        .insts
        .iter()
        .enumerate()
        .filter_map(|(i, x)| tag_of(&x.mnemonic, &x.operands, which).map(|t| (i, t)))
        .collect();
    if sites.is_empty() {
        bail!(
            "--loop-markers {}: the kernel has no matching barrier instructions",
            which.name()
        );
    }
    if sites.len() > MAX_SITES {
        bail!(
            "--loop-markers: {} sites exceed the {MAX_SITES} the marker word can index",
            sites.len()
        );
    }
    for &(i, _) in &sites {
        let x = &a.insts[i];
        if x.size() != 4 || a.sites.iter().any(|s| s.index == i) {
            bail!(
                "loop marker site at {:#x} is not a plain 4-byte instruction",
                x.addr
            );
        }
    }
    Ok(sites)
}

/// Why the body may read M0, if it may: an operand naming it, or an instruction that reads it implicitly (block
/// loads/stores and cluster loads take a mask from it, s_sendmsg a payload, movrel an index, LDS parameter loads an
/// address). Messages that carry no payload are exempt.
pub fn m0_reader(insts: &[Inst]) -> Option<String> {
    const IMPLICIT: &[&str] = &[
        "s_movrel",
        "v_movrel",
        "cluster_load",
        "lds_param_load",
        "lds_direct_load",
        "v_interp",
        "ds_gws",
        "ds_ordered_count",
        "s_ttracedata",
        "global_load_lds",
        "buffer_load_lds",
    ];
    insts.iter().find_map(|i| {
        let m = i.mnemonic.as_str();
        let names_m0 = i
            .operands
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .any(|t| t == "m0");
        let implicit = (IMPLICIT.iter().any(|p| m.starts_with(p)) && m != "s_ttracedata_imm")
            || m.ends_with("_block")
            || (m.starts_with("buffer_") && i.operands.split(' ').any(|t| t == "lds"))
            || (m.starts_with("s_sendmsg")
                && !(i.operands.contains("MSG_RTN_GET_")
                    || i.operands.contains("MSG_DEALLOC_VGPRS")));
        (names_m0 || implicit).then(|| format!("{:#x}: {}", i.addr, i.text()))
    })
}

/// Highest SGPR the body never references, outside the live-in user SGPRs and the reserved workgroup-id registers.
pub fn save_sgpr(a: &Analysis) -> Result<u32> {
    (a.live_in_sgprs..NUM_SGPRS)
        .rev()
        .find(|r| !a.sgprs_used.contains(r) && !a.reserved.values().any(|v| v == r))
        .ok_or_else(|| anyhow!("--loop-markers: M0 may be live and no SGPR is free to save it in"))
}

/// SGPR to save M0 in, or `None` when the body never reads M0.
pub fn m0_save(a: &Analysis) -> Result<Option<u32>> {
    match m0_reader(&a.insts) {
        Some(why) => save_sgpr(a)
            .map(Some)
            .map_err(|e| anyhow!("{e} (the body may read M0 at {why})")),
        None => Ok(None),
    }
}

/// Stubs for `sites` placed at `base`: bytes, assembly source and manifest entries. Each stub branches back to the
/// instruction after its site.
pub fn build_stubs(
    a: &Analysis,
    sites: &[(usize, u8)],
    save: Option<u32>,
    base: u64,
) -> Result<(Vec<u8>, String, Vec<MarkerSite>)> {
    let len = stub_bytes(save);
    let mut source = String::from("\t.text\n");
    for (n, &(i, tag)) in sites.iter().enumerate() {
        let x = &a.insts[i];
        source.push_str(&format!("\t// site {n} at {:#x}\n", x.addr));
        if let Some(s) = save {
            source.push_str(&format!("\ts_mov_b32 s{s}, m0\n"));
        }
        source.push_str(&format!(
            "\ts_pack_ll_b32_b16 m0, ttmp6, {:#x}\n\ts_ttracedata\n",
            (tag as u32) << 8 | n as u32
        ));
        if let Some(s) = save {
            source.push_str(&format!("\ts_mov_b32 m0, s{s}\n"));
        }
        source.push_str(&format!("\t{}\n\ts_branch 0\n", x.text()));
    }
    let mut code = assemble(&source)?;
    if code.len() != sites.len() * len {
        bail!(
            "loop marker stubs assembled to {} bytes, expected {}",
            code.len(),
            sites.len() * len
        );
    }
    let mut out = vec![];
    for (n, &(i, tag)) in sites.iter().enumerate() {
        let x = &a.insts[i];
        let stub = base + (n * len) as u64;
        let o = n * len;
        if code[o + len - 8..o + len - 4] != x.bytes[..] {
            bail!(
                "loop marker stub {n} does not reproduce the original instruction at {:#x}",
                x.addr
            );
        }
        code[o + len - 4..o + len]
            .copy_from_slice(&branch_word(stub + len as u64 - 4, x.addr + 4)?);
        out.push(MarkerSite {
            addr: x.addr,
            original: x.text(),
            tag,
            index: n as u32,
            stub_vaddr: stub,
        });
    }
    Ok((code, source, out))
}

/// `s_branch` at `from` to `to`.
pub fn branch_word(from: u64, to: u64) -> Result<[u8; 4]> {
    let diff = to as i64 - (from + 4) as i64;
    let simm = i16::try_from(diff / 4)
        .ok()
        .filter(|_| diff % 4 == 0)
        .ok_or_else(|| {
            anyhow!("loop marker branch {from:#x} -> {to:#x} is out of s_branch range")
        })?;
    Ok((0xBFA00000u32 | simm as u16 as u32).to_le_bytes())
}

pub fn info(
    which: LoopMarkers,
    save: Option<u32>,
    base: u64,
    sites: Vec<MarkerSite>,
) -> LoopMarkerInfo {
    LoopMarkerInfo {
        sites_kind: which.name().into(),
        save_sgpr: save,
        stubs_vaddr: base,
        stubs_bytes: (sites.len() * stub_bytes(save)) as u64,
        sites,
    }
}
