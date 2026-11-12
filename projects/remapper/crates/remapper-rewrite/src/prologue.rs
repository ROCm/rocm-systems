//! Assembly templates and checked, byte-preserving operand substitution.
use super::{
    analyze::Analysis,
    elf::Elf,
    isa::{Inst, TTMP_OPERAND_BASE},
    manifest::PatchedSite,
    rewrite::{LaunchModel, Prefetch, RewriteOptions},
};
use anyhow::{bail, Context, Result};
use std::{collections::BTreeMap, process::Command};

pub fn assemble(src: &str) -> Result<Vec<u8>> {
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("prologue.s");
    let output = dir.path().join("prologue.o");
    std::fs::write(&input, src)?;
    let out = Command::new(super::llvm_tool("llvm-mc")?)
        .args([
            "-triple=amdgcn-amd-amdhsa",
            "-mcpu=gfx1250",
            "-filetype=obj",
        ])
        .arg(input)
        .arg("-o")
        .arg(&output)
        .output()?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() || stderr.contains("error") || stderr.contains("warning") {
        bail!("llvm-mc: {stderr}");
    }
    let elf = Elf::parse(std::fs::read(output)?)?;
    if elf.sections.iter().any(|s| s.name.starts_with(".rel")) {
        bail!("assembly contains relocations");
    }
    let sec = elf.section(".text")?;
    Ok(elf.data[sec.offset as usize..(sec.offset + sec.size) as usize].to_vec())
}

fn render(template: &str, values: &BTreeMap<String, String>) -> Result<String> {
    let mut out = String::new();
    let mut rest = template;
    while let Some((before, after)) = rest.split_once("${") {
        out.push_str(before);
        let (key, tail) = after.split_once('}').context("unterminated template key")?;
        out.push_str(
            values
                .get(key)
                .with_context(|| format!("unknown template key {key}"))?,
        );
        rest = tail;
    }
    out.push_str(rest);
    if out.contains("${") {
        bail!("unexpanded template key");
    }
    Ok(out)
}
fn block(lines: Vec<String>) -> String {
    lines
        .iter()
        .map(|s| format!("\t{s}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Polls (one per ~8k shader clocks) a parked arrival waits without a claim in its shader engine before ending, so a
/// launch whose slack ran out ends (and fails golden verification) instead of hanging.
pub const ADMISSION_STALL_POLLS: u32 = 256;

pub fn build_prologue(
    a: &Analysis,
    opts: &RewriteOptions,
    addr: u64,
    ctl: u64,
) -> Result<(Vec<u8>, String, String)> {
    let cluster = opts.cluster_size > 1;
    let admission = opts.launch_model == LaunchModel::Admission;
    let mut base = (a.live_in_sgprs + 3) & !3;
    let count = if cluster {
        32
    } else if admission {
        28
    } else {
        24
    };
    if cluster {
        while a.reserved.values().any(|&r| base <= r && r < base + count) {
            base += 4;
        }
    }
    if base + count > if cluster { 104 } else { 106 }
        || a.reserved.values().any(|&r| base <= r && r < base + count)
    {
        bail!("prologue SGPR block overlaps reserved registers or exceeds allocation");
    }
    let s = |i| format!("s{}", base + i);
    let r = |i, n| format!("s[{}:{}]", base + i, base + i + n - 1);
    let mut v = BTreeMap::new();
    for (key, i, n) in [
        ("CTL", 0, 2),
        ("MODE_MAXK", 2, 2),
        ("PTRS", 4, 8),
        ("ARRIVAL", 4, 2),
        ("TABLE", 6, 2),
        ("COUNTERS", 8, 2),
        ("LOG", 10, 2),
    ] {
        v.insert(key.into(), r(i, n));
    }
    v.insert("MODE".into(), s(2));
    v.insert("MAXK".into(), s(3));
    let fields: &[(&str, u32)] = if cluster {
        &[
            ("WAVE", 16),
            ("HWID1", 17),
            ("SEAID", 18),
            ("LOC", 19),
            ("K", 20),
            ("CID", 21),
            ("C", 22),
            ("SAVE_EXEC", 23),
            ("TMP", 24),
            ("TMP2", 25),
            ("HWCID", 26),
            ("HWRANK", 27),
            ("KEY", 28),
            ("AR", 29),
            ("ENTRY_LO", 30),
            ("ENTRY_HI", 31),
        ]
    } else {
        &[
            ("WAVE", 12),
            ("HWID1", 13),
            ("SEAID", 14),
            ("LOC", 15),
            ("K", 16),
            ("T9", 17),
            ("T7", 18),
            ("SAVE_EXEC", 19),
            ("TMP", 20),
            ("TMP2", 21),
            ("ENTRY_LO", 22),
            ("ENTRY_HI", 23),
        ]
    };
    for &(k, i) in fields {
        v.insert(k.into(), s(i));
    }
    v.insert("ENTRY".into(), r(if cluster { 30 } else { 22 }, 2));
    // location = location * 4 + SIMD of wave 0 (HW_ID1[9:8]).
    let simd_key = !cluster
        && opts.launch_model != LaunchModel::Sparse
        && super::rewrite::keys_by_simd(opts.waves_per_workgroup);
    if !cluster {
        let lines = if simd_key {
            vec![
                format!("s_lshl_b32 {0}, {0}, 2", v["LOC"]),
                format!("s_bfe_u32 {}, {}, 0x20008", v["TMP"], v["HWID1"]),
                format!("s_add_co_i32 {0}, {0}, {1}", v["LOC"], v["TMP"]),
            ]
        } else {
            vec![]
        };
        // Appended to the preceding line in prologue.s.in, so per-WGP keys render byte-identical source.
        v.insert(
            "SIMD_KEY_LINES".into(),
            lines.iter().map(|l| format!("\n\t{l}")).collect(),
        );
        v.insert("SIMD_KEY".into(), block(lines));
    }
    if admission && cluster {
        // The parked path no longer needs the key/rank pair or the table entry.
        v.insert("ADMP".into(), r(28, 2));
        v.insert("POLL".into(), s(30));
    } else if admission {
        v.insert("ADM".into(), r(24, 2));
        v.insert("SE".into(), s(26));
        v.insert("POLL".into(), s(27));
    }
    if admission {
        v.insert("STALL_POLLS".into(), ADMISSION_STALL_POLLS.to_string());
        v.insert("SE_SHIFT".into(), if simd_key { "7" } else { "5" }.into());
        v.insert(
            "PARK_SYNC".into(),
            block(if opts.waves_per_workgroup > 1 {
                vec!["s_barrier_signal -1".into(), "s_barrier_wait -1".into()]
            } else {
                vec![]
            }),
        );
    }
    for (n, fallback) in if cluster {
        vec![(9, "CID"), (6, "TMP2")]
    } else {
        vec![(9, "T9"), (7, "T7")]
    } {
        v.insert(
            format!("R{n}"),
            a.reserved
                .get(&n)
                .map(|r| format!("s{r}"))
                .unwrap_or_else(|| v[fallback].clone()),
        );
    }
    if cluster {
        for (k, i, n) in [("PTRS2", 12, 4), ("SLOT", 12, 2), ("MAILBOX", 14, 2)] {
            v.insert(k.into(), r(i, n));
        }
        v.insert(
            "CLUSTER_SHIFT".into(),
            opts.cluster_size.trailing_zeros().to_string(),
        );
        v.insert("TRACED_WGP".into(), opts.traced_wgp.to_string());
        v.insert("NUM_SE".into(), opts.shader_engines.to_string());
        let [cx, cy, _] = opts.cluster_dims;
        let ncx =
            u32::try_from(opts.workgroups[0] / cx as u64).context("cluster count overflow")?;
        let two_d = opts.workgroups[1] > 1;
        v.insert("CLUSTER_COUNT_X".into(), ncx.to_string());
        let cid_src = if two_d {
            v["HWCID"].clone()
        } else {
            "ttmp9".into()
        };
        v.insert("CID_SRC".into(), cid_src);
        // One-dimensional clusters keep the 4-bit rank (wg_y is 0) and their validated instruction sequence.
        v.insert(
            "RANK_MASK".into(),
            if cy > 1 { "0xff" } else { "15" }.into(),
        );
        v.insert(
            "RANK_BFE".into(),
            if cy > 1 { "0x80010" } else { "0x40010" }.into(),
        );
        v.insert(
            "FLAT_RANK".into(),
            block(if cy > 1 {
                vec![
                    format!("s_lshr_b32 {}, {}, 4", v["TMP2"], v["TMP"]),
                    format!("s_mul_i32 {0}, {0}, {cx}", v["TMP2"]),
                    format!("s_and_b32 {0}, {0}, 15", v["TMP"]),
                    format!("s_add_co_i32 {0}, {0}, {1}", v["TMP"], v["TMP2"]),
                ]
            } else {
                vec![]
            }),
        );
        // ttmp7 is initialized only for multi-dimensional grids.
        v.insert(
            "HWCID_INIT".into(),
            block(if two_d {
                vec![
                    format!("s_and_b32 {}, ttmp7, 0xffff", v["HWCID"]),
                    format!("s_mul_i32 {0}, {0}, {ncx}", v["HWCID"]),
                    format!("s_add_co_i32 {0}, {0}, ttmp9", v["HWCID"]),
                ]
            } else {
                vec![format!("s_mov_b32 {}, ttmp9", v["HWCID"])]
            }),
        );
        let (cid, tmp, tmp2, r9) = (&v["CID"], &v["TMP"], &v["TMP2"], &v["R9"]);
        let mut ids = if !two_d {
            vec![format!("s_mov_b32 {r9}, {cid}")]
        } else if ncx.is_power_of_two() {
            vec![
                format!("s_lshr_b32 {tmp}, {cid}, {}", ncx.trailing_zeros()),
                format!("s_and_b32 {r9}, {cid}, {}", ncx - 1),
            ]
        } else {
            // floor(cid / ncx) = mul_hi(cid, ceil(2^32 / ncx)) for cid, ncx < 2^16.
            let magic = (1u64 << 32).div_ceil(ncx as u64) as u32;
            vec![
                format!("s_mul_hi_u32 {tmp}, {cid}, {magic:#x}"),
                format!("s_mul_i32 {tmp2}, {tmp}, {ncx}"),
                format!("s_sub_co_u32 {r9}, {cid}, {tmp2}"),
            ]
        };
        if two_d {
            if let Some(r7) = a.reserved.get(&7) {
                ids.push(format!("s_mov_b32 s{r7}, {tmp}"));
            }
        } else if cy != 1 {
            bail!("a two-dimensional cluster needs a two-dimensional grid");
        }
        v.insert("CLUSTER_IDS".into(), block(ids));
    } else {
        let (mut publish, mut sync) = (vec![], vec![]);
        if opts.waves_per_workgroup > 1 {
            publish = vec![
                "v_mov_b32_e32 v1, 0".into(),
                format!("v_mov_b32_e32 v4, {}", v["T9"]),
                format!("v_mov_b32_e32 v5, {}", v["T7"]),
                "s_wait_alu 0x0".into(),
                "ds_store_b64 v1, v[4:5]".into(),
                "s_wait_dscnt 0x0".into(),
            ];
            sync = vec![
                "s_barrier_signal -1".into(),
                "s_barrier_wait -1".into(),
                "v_mov_b32_e32 v1, 0".into(),
                "s_wait_alu 0x0".into(),
                "ds_load_b64 v[4:5], v1".into(),
                "s_wait_dscnt 0x0".into(),
                format!("v_readfirstlane_b32 {}, v4", v["T9"]),
                format!("v_readfirstlane_b32 {}, v5", v["T7"]),
                "s_wait_alu 0x0".into(),
                "s_barrier_signal -1".into(),
                "s_barrier_wait -1".into(),
            ];
        }
        v.insert("PUBLISH".into(), block(publish));
        v.insert("SYNC".into(), block(sync));
        let [gx, gy, gz] = opts.workgroups;
        let uid = if gy == 1 && gz == 1 {
            vec![format!("s_mov_b32 {}, {}", v["TMP"], v["T9"])]
        } else {
            if !a.reads_ttmp(7) {
                bail!("multi-dimensional grid but the body never reads ttmp7");
            }
            vec![
                format!("s_lshr_b32 {}, {}, 16", v["TMP"], v["T7"]),
                format!("s_mul_i32 {0}, {0}, {gy}", v["TMP"]),
                format!("s_and_b32 {}, {}, 0xffff", v["TMP2"], v["T7"]),
                format!("s_add_co_i32 {0}, {0}, {1}", v["TMP"], v["TMP2"]),
                format!("s_mul_i32 {0}, {0}, {gx}", v["TMP"]),
                format!("s_add_co_i32 {0}, {0}, {1}", v["TMP"], v["T9"]),
            ]
        };
        v.insert("UID".into(), block(uid));
    }
    let mut preload = vec![];
    let mut i = 0;
    while i < a.kd.preload_length() {
        let n = if a.kd.preload_length() - i >= 2 && (a.preload_sgpr_base + i) % 2 == 0 {
            2
        } else {
            1
        };
        let dst = if n == 2 {
            format!(
                "s[{}:{}]",
                a.preload_sgpr_base + i,
                a.preload_sgpr_base + i + 1
            )
        } else {
            format!("s{}", a.preload_sgpr_base + i)
        };
        preload.push(format!(
            "s_load_b{} {dst}, s[0:1], {:#x}",
            32 * n,
            (a.kd.preload_offset() + i) * 4
        ));
        i += n;
    }
    if !preload.is_empty() {
        preload.push("s_wait_kmcnt 0x0".into());
    }
    v.insert("PRELOAD".into(), block(preload));
    let lines = if opts.prefetch == Prefetch::None {
        0
    } else {
        a.kd.inst_pref_lines()
    };
    let wave0 = lines > 0 && opts.prefetch == Prefetch::Wave0;
    let tmp = v["TMP"].clone();
    // Body prefetch starting at `start`; s_prefetch_inst_pc_rel offsets are relative to the next instruction.
    let prefetch_block = |start: u64| -> String {
        let mut prefetch = vec![];
        let mut pc = start;
        if wave0 {
            prefetch.extend([
                format!("s_bfe_u32 {tmp}, ttmp8, 0x50019"),
                format!("s_cmp_lg_u32 {tmp}, 0"),
                format!("s_cbranch_scc1 {}", 2 * lines.div_ceil(31)),
            ]);
            pc += 16;
        }
        let mut done = 0;
        while done < lines {
            let n = (lines - done).min(31);
            pc += 8;
            let off = a.kd.entry_vaddr as i64 + done as i64 * 128 - pc as i64;
            let hex = if off < 0 {
                format!("-0x{:x}", -off)
            } else {
                format!("0x{off:x}")
            };
            prefetch.push(format!("s_prefetch_inst_pc_rel {hex}, null, {n}"));
            done += n;
        }
        block(prefetch)
    };
    let prefetch = prefetch_block(addr);
    v.insert("PREFETCH".into(), prefetch);
    v.insert("NUM_WAVES".into(), opts.waves_per_workgroup.to_string());
    v.insert("CTL_KERNARG_OFFSET".into(), format!("{ctl:#x}"));
    let ka = a.kd.kernarg_sgpr().context("no kernarg segment pointer")?;
    v.insert("KERNARG".into(), format!("s[{ka}:{}]", ka + 1));
    let template = match (cluster, opts.launch_model) {
        (true, LaunchModel::Table) => include_str!("prologue_cluster.s.in"),
        (false, LaunchModel::Table) => include_str!("prologue.s.in"),
        (false, LaunchModel::Sparse) => include_str!("prologue_sparse.s.in"),
        (false, LaunchModel::Admission) => include_str!("prologue_admission.s.in"),
        (true, LaunchModel::Admission) => include_str!("prologue_admission_cluster.s.in"),
        (true, m) => bail!(
            "the {} launch model does not support cluster dispatches",
            m.name()
        ),
    };
    let mut source = render(template, &v)?;
    let mut code = assemble(&source)?;
    if opts.launch_model == LaunchModel::Sparse && lines > 0 {
        // The sparse template prefetches just before its final branch, so only workgroups that run the body
        // prefetch it; place the block there now that the prologue length is known.
        let bytes = 8 * lines.div_ceil(31) as u64 + if wave0 { 16 } else { 0 };
        let start = addr + code.len() as u64 - 4 - bytes;
        let prefetch = prefetch_block(start);
        v.insert("PREFETCH".into(), prefetch);
        source = render(template, &v)?;
        let placed = assemble(&source)?;
        if placed.len() != code.len() {
            bail!("sparse prologue length changed when placing the body prefetch");
        }
        code = placed;
    }
    if code.len() < 4 || code[code.len() - 4..] != 0xBFA00000u32.to_le_bytes() {
        bail!("prologue must end with s_branch 0");
    }
    if code.len().div_ceil(128) > 255 {
        bail!("prologue exceeds descriptor prefetch range");
    }
    Ok((code, source, format!("s{base}..s{}", base + count - 1)))
}

/// Every changed bit of the re-assembled instruction must lie in an 8-bit window that holds a ttmp operand code in
/// the original and its reserved SGPR number in the patch. Windows may start at any bit: VOP3 source fields are 9
/// bits wide and src1/src2 straddle bytes; their high bit is 0 for both codes and must stay unchanged.
/// Operand code of the inline constant 0.
const INLINE_ZERO: u8 = 128;

pub fn operand_patch_ok(old: &[u8], new: &[u8], subs: &[(u8, u8)]) -> bool {
    if old.len() != new.len() || old.is_empty() || old.len() > 16 {
        return false;
    }
    let word = |b: &[u8]| b.iter().rev().fold(0u128, |acc, &x| acc << 8 | x as u128);
    let (o, w) = (word(old), word(new));
    let mut covered = 0u128;
    for &(from, to) in subs {
        for bit in 0..=(old.len() * 8 - 8) {
            if (o >> bit) & 0xff == from as u128 && (w >> bit) & 0xff == to as u128 {
                covered |= 0xff << bit;
            }
        }
    }
    o != w && (o ^ w) & !covered == 0
}

pub fn substitute_sites(a: &Analysis) -> Result<Vec<(PatchedSite, Vec<u8>)>> {
    let mut texts = vec![];
    for site in &a.sites {
        // Tokenize identifiers so ttmp6 cannot accidentally replace ttmp60.
        let text = site
            .text
            .split_inclusive(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '.')
            .map(|token| {
                let n = token
                    .trim_end_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '.');
                for &r in &site.ttmps {
                    if n == format!("ttmp{r}") {
                        return format!("s{}{}", a.reserved[&r], &token[n.len()..]);
                    }
                }
                if site.zeroed.iter().any(|&r| n == format!("ttmp{r}")) {
                    return format!("0{}", &token[n.len()..]);
                }
                token.to_string()
            })
            .collect::<String>();
        texts.push(text);
    }
    let bytes = assemble(&format!("\t.text\n\t{}\n", texts.join("\n\t")))?;
    let expected: usize = a.sites.iter().map(|s| a.insts[s.index].bytes.len()).sum();
    if bytes.len() != expected {
        bail!("operand substitution changed instruction sizes");
    }
    let mut off = 0;
    let mut out = vec![];
    for (site, patched) in a.sites.iter().zip(texts) {
        let old: &Inst = &a.insts[site.index];
        let new = bytes[off..off + old.bytes.len()].to_vec();
        off += old.bytes.len();
        let changed: Vec<_> = (0..new.len()).filter(|&i| new[i] != old.bytes[i]).collect();
        let subs: Vec<(u8, u8)> = site
            .ttmps
            .iter()
            .map(|&n| (TTMP_OPERAND_BASE + n as u8, a.reserved[&n] as u8))
            .chain(
                site.zeroed
                    .iter()
                    .map(|&n| (TTMP_OPERAND_BASE + n as u8, INLINE_ZERO)),
            )
            .collect();
        if !operand_patch_ok(&old.bytes, &new, &subs) {
            bail!(
                "unexpected operand bytes at {:#x}: {:02x?} -> {:02x?}",
                site.addr,
                old.bytes,
                new
            );
        }
        out.push((
            PatchedSite {
                addr: site.addr,
                original: site.text.clone(),
                patched,
                bytes_changed: changed,
            },
            new,
        ));
    }
    Ok(out)
}
