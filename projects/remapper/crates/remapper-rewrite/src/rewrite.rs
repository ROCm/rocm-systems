//! Static code-object rewriting; the original body retains its addresses and all non-operand bytes.
use super::{
    analyze::analyze_code_object,
    elf::{self, Elf},
    manifest::*,
    markers, prologue,
};
use anyhow::{bail, Context, Result};
use object::{Object, ObjectSymbol};
use serde::Serialize;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Prefetch {
    PerWave,
    Wave0,
    None,
}
impl Prefetch {
    pub const ALL: [Prefetch; 3] = [Prefetch::PerWave, Prefetch::Wave0, Prefetch::None];

    pub fn name(self) -> &'static str {
        match self {
            Self::PerWave => "per-wave",
            Self::Wave0 => "wave0",
            Self::None => "none",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == s)
    }
}
/// How the replayer launches the rewritten kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LaunchModel {
    /// The captured grid; the k-th workgroup at a location takes `table[location][k]`.
    Table,
    /// One workgroup per WGP; assigned locations run the body, every other workgroup ends in the prologue.
    Sparse,
    /// The captured grid plus slack; each location runs its recorded schedule and arrivals past its quota park.
    Admission,
}
impl LaunchModel {
    pub fn name(self) -> &'static str {
        match self {
            Self::Table => "table",
            Self::Sparse => "sparse",
            Self::Admission => "admission",
        }
    }
}
/// Admission buffer, dwords per shader engine: tiles still to claim, parked arrivals, and parked arrivals that found
/// tiles still unclaimed (held a WGP).
pub const ADMISSION_ENGINES: usize = 32;
pub const ADMISSION_WORDS: usize = 3 * ADMISSION_ENGINES;
/// Workgroups with fewer waves than a WGP has SIMDs occupy only some of its SIMDs, chosen by the hardware per
/// workgroup, so the placement is keyed by (location, SIMD of wave 0) to give each traced SIMD the same workgroups in
/// every pass. Workgroups of 4 or more waves put a wave on every SIMD and keep the per-WGP key.
pub fn keys_by_simd(waves_per_workgroup: u32) -> bool {
    waves_per_workgroup < 4
}
pub struct RewriteOptions {
    pub dispatch_index: u64,
    pub workgroups: [u64; 3],
    pub waves_per_workgroup: u32,
    pub cluster_size: u32,
    /// Workgroups per cluster in X, Y, Z.
    pub cluster_dims: [u32; 3],
    pub shader_engines: u32,
    pub traced_wgp: u32,
    pub prefetch: Prefetch,
    pub launch_model: LaunchModel,
    /// Admission model: workgroups launched in table modes (captured workgroups plus slack).
    pub padded_workgroups: u64,
    /// Opt-in `s_ttracedata` stubs at body barriers; `None` leaves the rewrite byte-identical to before.
    pub loop_markers: Option<markers::LoopMarkers>,
}

pub fn rewrite(
    code: &[u8],
    symbol: &str,
    opts: &RewriteOptions,
    out_dir: &Path,
) -> Result<Manifest> {
    let mut a = analyze_code_object(code, Some(symbol))?;
    a.zero_constant_ids(opts.workgroups);
    a.refusal()?;
    let cluster = opts.cluster_size > 1;
    let num_workgroups = opts
        .workgroups
        .iter()
        .try_fold(1u64, |n, &v| n.checked_mul(v))
        .context("workgroup count overflow")?;
    if num_workgroups == 0 || opts.waves_per_workgroup == 0 {
        bail!("empty dispatch");
    }
    if num_workgroups
        .checked_mul(opts.waves_per_workgroup as u64)
        .context("marker count overflow")?
        >= 1 << 24
    {
        bail!("marker uid range exceeds 24 bits");
    }
    if !cluster && a.reads_cluster_rank() {
        bail!("reads ttmp6 but the dispatch is not clustered");
    }
    let [cx, cy, cz] = opts.cluster_dims;
    let [gx, gy, gz] = opts.workgroups;
    if cluster
        && (cx * cy * cz != opts.cluster_size
            || !cx.is_power_of_two()
            || !cy.is_power_of_two()
            || opts.cluster_size > 16
            || cz != 1
            || gz != 1
            || gx % cx as u64 != 0
            || gy % cy as u64 != 0
            || (gx / cx as u64) * (gy / cy as u64) > 0xffff)
    {
        bail!(
            "cluster remap requires a one- or two-dimensional grid of power-of-two X/Y clusters of at most 16 \
             workgroups that tile the grid (grid {gx}x{gy}x{gz}, cluster {cx}x{cy}x{cz})"
        );
    }
    if opts.shader_engines == 0 || opts.shader_engines > 32 || opts.traced_wgp >= 32 {
        bail!("invalid gfx1250 topology");
    }
    if opts.launch_model != LaunchModel::Table {
        let model = opts.launch_model.name();
        if cluster && opts.launch_model == LaunchModel::Sparse {
            bail!("the {model} launch model does not support cluster dispatches");
        }
        let problems = a.dispatch_packet_grid_problems();
        if !problems.is_empty() {
            bail!(
                "{} refused for a {model} launch: {}",
                a.symbol,
                problems.join("; ")
            );
        }
    }
    let admission = opts.launch_model == LaunchModel::Admission;
    if admission
        && (opts.padded_workgroups <= num_workgroups || opts.padded_workgroups > u32::MAX as u64)
    {
        bail!(
            "admission launch of {} workgroups for {num_workgroups} captured ones",
            opts.padded_workgroups
        );
    }
    // Admission clusters: the park code is cluster id 0xffff, so every padded cluster id must stay below it.
    let padded_clusters = opts.padded_workgroups / opts.cluster_size.max(1) as u64;
    if admission
        && cluster
        && (opts.padded_workgroups % opts.cluster_size as u64 != 0 || padded_clusters >= 0xffff)
    {
        bail!(
            "admission launch of {} workgroups is not a whole number of clusters below 65535",
            opts.padded_workgroups
        );
    }
    let original = Elf::parse(code.to_vec())?;
    let text = original.section(".text")?;
    let mut addr = (text.addr + text.size + 255) & !255;
    let ctl = (a.kd.kernarg_size as u64 + 7) & !7;
    let (mut prologue, mut source, mut registers) = prologue::build_prologue(&a, opts, addr, ctl)?;
    // Loop-marker stubs follow the prologue; their length does not depend on where they are placed.
    let marker_plan = match opts.loop_markers {
        Some(which) => Some((
            which,
            markers::select_sites(&a, which)?,
            markers::m0_save(&a)?,
        )),
        None => None,
    };
    let stub_len = marker_plan
        .as_ref()
        .map_or(0, |(_, s, save)| s.len() * markers::stub_bytes(*save));
    let stubs_at = |base: u64| -> Result<(Vec<u8>, String, Vec<MarkerSite>)> {
        match &marker_plan {
            Some((_, sites, save)) => markers::build_stubs(&a, sites, *save, base),
            None => Ok((vec![], String::new(), vec![])),
        }
    };
    let in_text = addr + (prologue.len() + stub_len) as u64 + 256 <= elf::text_limit(&original)?;
    let stubs;
    let mut grown = if in_text {
        stubs = stubs_at(addr + prologue.len() as u64)?;
        let blob = [prologue.as_slice(), &stubs.0].concat();
        let grown = elf::grow_text(&original, addr, &blob, 256, 0xBF9F0000)?;
        log::debug!(
            "remap ELF growth {} bytes; file insertion {} bytes",
            grown.growth,
            grown.file_delta
        );
        grown.elf
    } else {
        // No slack after .text: place the prologue just below it. Its PC-relative operands depend on its address,
        // its length does not.
        let len = prologue.len();
        addr = text.addr - ((len + stub_len) as u64).div_ceil(256) * 256;
        (prologue, source, registers) = prologue::build_prologue(&a, opts, addr, ctl)?;
        if prologue.len() != len {
            bail!("prologue length depends on its address");
        }
        log::info!("remap prologue does not fit after .text; placed below it at {addr:#x}");
        stubs = stubs_at(addr + prologue.len() as u64)?;
        let blob = [prologue.as_slice(), &stubs.0].concat();
        let (elf, placed) = elf::grow_text_down(&original, &blob, 0xBF9F0000)?;
        if placed != addr {
            bail!("prologue placed at {placed:#x}, assembled for {addr:#x}");
        }
        elf
    };
    let added = (prologue.len() + stub_len) as u64;
    // The kernel symbol covers the prologue and stubs (the ATT decoder disassembles whole symbols).
    for table in [".symtab", ".dynsym"] {
        for sym in grown.symbols(table)?.iter().filter(|s| s.name == a.symbol) {
            let (start, end) = if in_text {
                (sym.value, addr + added)
            } else {
                (addr, sym.value + sym.size)
            };
            elf::write_u64(&mut grown.data, sym.entry_off + 8, start)?;
            elf::write_u64(&mut grown.data, sym.entry_off + 16, end - start)?;
        }
    }
    let kd = grown.vaddr_to_offset(a.kd.vaddr)?;
    elf::write_u64(&mut grown.data, kd + 16, addr.wrapping_sub(a.kd.vaddr))?;
    let descriptor_lines = prologue.len().div_ceil(128) as u32;
    elf::write_u32(
        &mut grown.data,
        kd + 44,
        (a.kd.rsrc3 & !0xFF0) | (descriptor_lines << 4),
    )?;
    elf::write_u32(&mut grown.data, kd + 8, (ctl + 8).try_into()?)?;
    if a.kd.preload_length() > 0 {
        elf::write_u16(&mut grown.data, kd + 58, 0)?;
    }
    let sites = prologue::substitute_sites(&a)?;
    for (site, bytes) in &sites {
        let off = grown.vaddr_to_offset(site.addr)?;
        grown.data[off..off + bytes.len()].copy_from_slice(bytes);
    }
    let branch = addr + prologue.len() as u64 - 4;
    let diff = a.kd.entry_vaddr as i64 - (branch + 4) as i64;
    if diff % 4 != 0 || i16::try_from(diff / 4).is_err() {
        bail!("original entry out of s_branch range");
    }
    let off = grown.vaddr_to_offset(branch)?;
    elf::write_u32(
        &mut grown.data,
        off,
        0xBFA00000 | ((diff / 4) as u16 as u32),
    )?;
    let (_, stub_source, marker_sites) = stubs;
    for m in &marker_sites {
        let off = grown.vaddr_to_offset(m.addr)?;
        grown.data[off..off + 4].copy_from_slice(&markers::branch_word(m.addr, m.stub_vaddr)?);
    }
    let listing = super::disassemble_insts(&grown.data)?;
    let new = listing
        .iter()
        .filter(|i| i.addr >= a.kernel_vaddr && i.addr < a.kernel_vaddr + a.kernel_size)
        .cloned()
        .collect::<Vec<_>>();
    if new.len() != a.insts.len() {
        bail!("rewritten body instruction count changed");
    }
    for (old, new) in a.insts.iter().zip(&new) {
        if old.addr != new.addr || old.size() != new.size() {
            bail!("body instruction addresses or sizes changed");
        }
        if let Some(m) = marker_sites.iter().find(|m| m.addr == old.addr) {
            if new.mnemonic != "s_branch" || new.branch_target() != Some(m.stub_vaddr) {
                bail!("loop marker site verification failed at {:#x}", old.addr);
            }
        } else if let Some((site, bytes)) = sites.iter().find(|(s, _)| s.addr == old.addr) {
            if new.bytes != *bytes
                || new.text().split_whitespace().collect::<Vec<_>>()
                    != site.patched.split_whitespace().collect::<Vec<_>>()
            {
                bail!("patched body verification failed at {:#x}", old.addr);
            }
        } else if old.bytes != new.bytes {
            bail!("body changed at {:#x}", old.addr);
        }
    }
    // Each stub: [save M0], marker word, s_ttracedata, [restore M0], the original instruction, branch to the next one.
    if let Some((_, _, save)) = &marker_plan {
        let mut want: Vec<String> = vec![];
        if let Some(s) = save {
            want.push(format!("s_mov_b32 s{s}, m0"));
        }
        want.extend(["s_pack_ll_b32_b16".into(), "s_ttracedata".into()]);
        if let Some(s) = save {
            want.push(format!("s_mov_b32 m0, s{s}"));
        }
        let len = markers::stub_bytes(*save) as u64;
        for m in &marker_sites {
            let stub: Vec<_> = listing
                .iter()
                .filter(|i| i.addr >= m.stub_vaddr && i.addr < m.stub_vaddr + len)
                .collect();
            let original = &a
                .insts
                .iter()
                .find(|i| i.addr == m.addr)
                .context("marker site vanished")?
                .bytes;
            let n = want.len();
            let ok = stub.len() == n + 2
                && want.iter().zip(&stub).all(|(w, i)| {
                    if w == "s_pack_ll_b32_b16" {
                        i.text()
                            == format!(
                                "s_pack_ll_b32_b16 m0, ttmp6, {:#x}",
                                (m.tag as u32) << 8 | m.index
                            )
                    } else {
                        i.text() == *w
                    }
                })
                && stub[n].bytes == *original
                && stub[n + 1].mnemonic == "s_branch"
                && stub[n + 1].branch_target() == Some(m.addr + 4);
            if !ok {
                bail!(
                    "loop marker stub verification failed at {:#x}",
                    m.stub_vaddr
                );
            }
        }
    }
    let obj = object::File::parse(grown.data.as_slice())?;
    let func = obj
        .symbols()
        .find(|s| s.name().ok() == Some(a.symbol.as_str()))
        .context("rewritten function missing")?;
    if func.address() > addr
        || func.address() + func.size() < addr + added
        || !obj
            .symbols()
            .any(|s| s.name().ok() == Some(&format!("{}.kd", a.symbol)))
    {
        bail!("rewritten symbols do not cover prologue");
    }
    let dir = out_dir.join("rewrite");
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("source.hsaco"), code)?;
    std::fs::write(dir.join("kernel.hsaco"), &grown.data)?;
    std::fs::write(dir.join("prologue.s"), source)?;
    if marker_plan.is_some() {
        std::fs::write(dir.join("loop_markers.s"), &stub_source)?;
    }
    std::fs::write(dir.join("analysis.json"), serde_json::to_vec_pretty(&a)?)?;
    let nc = num_workgroups / opts.cluster_size.max(1) as u64;
    Ok(Manifest {
        version: VERSION,
        symbol: a.symbol.clone(),
        code_object: "rewrite/kernel.hsaco".into(),
        source_code_object: "rewrite/source.hsaco".into(),
        dispatch_index: opts.dispatch_index,
        workgroups: opts.workgroups,
        num_workgroups,
        waves_per_workgroup: opts.waves_per_workgroup,
        control_kernarg_offset: ctl as usize,
        control_bytes: CONTROL_BYTES,
        num_keys: if cluster {
            2 * opts.shader_engines as u64
        } else if opts.launch_model != LaunchModel::Sparse && keys_by_simd(opts.waves_per_workgroup)
        {
            4096
        } else {
            1024
        },
        log_words: LOG_WORDS,
        num_log_records: if cluster { nc } else { num_workgroups },
        min_group_segment_bytes: if !cluster && opts.waves_per_workgroup > 1 {
            8
        } else {
            0
        },
        extra_buffers: if cluster {
            // Slot and mailbox are indexed by hardware cluster id, which covers the slack in admission launches.
            let hw_clusters = if admission { padded_clusters } else { nc };
            let mut v = vec![
                ExtraBuffer {
                    name: "slot".into(),
                    control_offset: 40,
                    bytes: hw_clusters * 4,
                    fill: 0,
                    init: None,
                },
                ExtraBuffer {
                    name: "mailbox".into(),
                    control_offset: 48,
                    bytes: hw_clusters * 4,
                    fill: 0,
                    init: None,
                },
            ];
            if admission {
                v.push(ExtraBuffer {
                    name: "admission".into(),
                    control_offset: 56,
                    bytes: ADMISSION_WORDS as u64 * 4,
                    fill: 0,
                    init: None,
                });
            }
            v
        } else if admission {
            // Per-engine tile counts are written once the record run has fixed the schedule.
            vec![ExtraBuffer {
                name: "admission".into(),
                control_offset: 40,
                bytes: ADMISSION_WORDS as u64 * 4,
                fill: 0,
                init: None,
            }]
        } else {
            vec![]
        },
        cluster: cluster.then_some(ClusterLayout {
            size: opts.cluster_size,
            dims: opts.cluster_dims,
            num_clusters: nc,
            shader_engines: opts.shader_engines,
            traced_wgp: opts.traced_wgp,
        }),
        table: None,
        rewrite: RewriteInfo {
            prologue_vaddr: addr,
            prologue_bytes: prologue.len() as u64,
            original_entry_vaddr: a.kd.entry_vaddr,
            kernel_vaddr: a.kernel_vaddr,
            kernel_size: a.kernel_size,
            original_prefetch_lines: a.kd.inst_pref_lines(),
            descriptor_prefetch_lines: descriptor_lines,
            body_prefetch: opts.prefetch.name().into(),
            reserved_sgprs: a
                .reserved
                .iter()
                .map(|(n, r)| (format!("ttmp{n}"), *r))
                .collect(),
            prologue_sgprs: registers,
            workgroup_id_sites: sites.into_iter().map(|s| s.0).collect(),
            cluster_rank_sites: a
                .rank_reads
                .iter()
                .map(|r| RankSite {
                    addr: r.addr,
                    text: r.text.clone(),
                    kind: match r.kind {
                        super::analyze::RankKind::Data => "data",
                        _ => "multicast",
                    }
                    .into(),
                })
                .collect(),
            preload_converted: a.kd.preload_length() > 0,
            loop_markers: marker_plan.map(|(which, _, save)| {
                markers::info(which, save, addr + prologue.len() as u64, marker_sites)
            }),
        },
        launch: None,
    })
}
