//! Static analysis of a gfx1250 kernel for the binary workgroup remap.
//!
//! Finds every instruction that reads the architected workgroup-id registers (ttmp9 = X, ttmp7 = Y | Z << 16) or
//! the cluster register (ttmp6, rank in [3:0]), classifies each cluster-rank read, picks SGPRs the kernel never
//! references to carry the remapped values into the body, and refuses (fails closed) every kernel the rewrite
//! cannot handle safely.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{bail, Context, Result};
use serde::Serialize;

use super::elf::Elf;
use super::isa::{Inst, RegClass, RegRange};

pub const NUM_SGPRS: u32 = 106;
pub const KD_SIZE: usize = 64;
/// Workgroup X, Y|Z<<16 and the cluster register.
pub const TTMP_WG_X: u32 = 9;
pub const TTMP_WG_YZ: u32 = 7;
pub const TTMP_CLUSTER: u32 = 6;

#[derive(Clone, Debug, Serialize)]
pub struct KernelDescriptor {
    pub vaddr: u64,
    pub file_offset: usize,
    pub group_segment_fixed_size: u32,
    pub private_segment_fixed_size: u32,
    pub kernarg_size: u32,
    pub entry_vaddr: u64,
    pub rsrc1: u32,
    pub rsrc2: u32,
    pub rsrc3: u32,
    pub code_properties: u16,
    pub kernarg_preload: u16,
}

impl KernelDescriptor {
    pub fn parse(elf: &Elf, vaddr: u64) -> Result<Self> {
        let off = elf.vaddr_to_offset(vaddr)?;
        let d = elf
            .data
            .get(off..off + KD_SIZE)
            .context("kernel descriptor out of bounds")?;
        let u32_at = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
        let entry = i64::from_le_bytes(d[16..24].try_into().unwrap());
        Ok(KernelDescriptor {
            vaddr,
            file_offset: off,
            group_segment_fixed_size: u32_at(0),
            private_segment_fixed_size: u32_at(4),
            kernarg_size: u32_at(8),
            entry_vaddr: (vaddr as i64 + entry) as u64,
            rsrc3: u32_at(44),
            rsrc1: u32_at(48),
            rsrc2: u32_at(52),
            code_properties: u16::from_le_bytes([d[56], d[57]]),
            kernarg_preload: u16::from_le_bytes([d[58], d[59]]),
        })
    }

    /// COMPUTE_PGM_RSRC2.USER_SGPR_COUNT is 6 bits ([6:1]) on gfx125x, so kernels can preload up to 32 SGPRs.
    pub fn user_sgpr_count(&self) -> u32 {
        (self.rsrc2 >> 1) & 0x3f
    }
    pub fn tg_size_enabled(&self) -> bool {
        self.rsrc2 >> 10 & 1 != 0
    }
    pub fn wave32(&self) -> bool {
        self.code_properties >> 10 & 1 != 0
    }
    pub fn preload_length(&self) -> u32 {
        (self.kernarg_preload & 0x7f) as u32
    }
    pub fn preload_offset(&self) -> u32 {
        (self.kernarg_preload >> 7) as u32
    }
    pub fn inst_pref_lines(&self) -> u32 {
        (self.rsrc3 >> 4) & 0xff
    }
    /// VGPRs allocated per wave32 lane: granules of 8.
    pub fn vgprs(&self) -> u32 {
        ((self.rsrc1 & 0x3f) + 1) * 8
    }
    /// Enabled user-SGPR inputs, in hardware order.
    pub fn user_sgpr_inputs(&self) -> Vec<&'static str> {
        const NAMES: [&str; 7] = [
            "private_segment_buffer",
            "dispatch_ptr",
            "queue_ptr",
            "kernarg_segment_ptr",
            "dispatch_id",
            "flat_scratch_init",
            "private_segment_size",
        ];
        (0..7)
            .filter(|b| self.code_properties >> b & 1 != 0)
            .map(|b| NAMES[b])
            .collect()
    }
    /// First SGPR of the dispatch packet pointer, after the private segment buffer if that is enabled.
    pub fn dispatch_ptr_sgpr(&self) -> Option<u32> {
        (self.code_properties >> 1 & 1 != 0).then(|| 4 * (self.code_properties & 1) as u32)
    }
    /// First SGPR of the kernarg segment pointer, after the user-SGPR inputs that precede it in hardware order.
    pub fn kernarg_sgpr(&self) -> Option<u32> {
        const WIDTHS: [u32; 3] = [4, 2, 2];
        (self.code_properties >> 3 & 1 != 0).then(|| {
            (0..3)
                .filter(|b| self.code_properties >> b & 1 != 0)
                .map(|b| WIDTHS[b])
                .sum()
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RankKind {
    /// Feeds data offsets: the body reads `ttmp6 ^ c` (logical rank) here.
    Data,
    /// Only the shift amount of `s_lshl_b32 dst, imm, x`: a multicast mask over hardware workgroups, which keeps the
    /// hardware rank. XOR by c maps every multicast group (ranks agreeing on the masked bits) onto a group.
    Multicast,
}

#[derive(Clone, Debug, Serialize)]
pub struct RankRead {
    pub index: usize,
    pub addr: u64,
    pub text: String,
    pub kind: RankKind,
}

/// An instruction whose ttmp source operands the rewrite redirects to reserved SGPRs.
#[derive(Clone, Debug, Serialize)]
pub struct Site {
    pub index: usize,
    pub addr: u64,
    pub text: String,
    pub ttmps: Vec<u32>,
    /// ttmps whose reads become the constant 0: workgroup ids the grid does not vary.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub zeroed: Vec<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Analysis {
    pub symbol: String,
    pub kd: KernelDescriptor,
    pub kernel_vaddr: u64,
    pub kernel_size: u64,
    #[serde(skip)]
    pub insts: Vec<Inst>,
    pub instructions: usize,
    pub sites: Vec<Site>,
    pub rank_reads: Vec<RankRead>,
    pub sgprs_used: BTreeSet<u32>,
    pub live_in_sgprs: u32,
    /// ttmp number -> reserved SGPR carrying its remapped value into the body.
    pub reserved: BTreeMap<u32, u32>,
    /// Set when the body references too many SGPRs to reserve one per remapped ttmp.
    pub sgpr_shortage: Option<String>,
    pub preload_sgpr_base: u32,
    pub problems: Vec<String>,
}

impl Analysis {
    pub fn reads_cluster_rank(&self) -> bool {
        !self.rank_reads.is_empty()
    }

    pub fn reads_ttmp(&self, ttmp: u32) -> bool {
        self.sites.iter().any(|s| s.ttmps.contains(&ttmp))
    }

    pub fn refusal(&self) -> Result<()> {
        let problems: Vec<&str> = self
            .problems
            .iter()
            .map(String::as_str)
            .chain(self.sgpr_shortage.as_deref())
            .collect();
        if !problems.is_empty() {
            bail!("{} refused: {}", self.symbol, problems.join("; "));
        }
        Ok(())
    }

    /// Reserve, for each ttmp the sites read, an SGPR outside the live-in user SGPRs that the body never references.
    fn reserve(&mut self) {
        let free: Vec<_> = (self.live_in_sgprs..NUM_SGPRS)
            .rev()
            .filter(|r| !self.sgprs_used.contains(r))
            .collect();
        let need: Vec<_> = self
            .sites
            .iter()
            .flat_map(|s| s.ttmps.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .rev()
            .collect();
        self.sgpr_shortage = (need.len() > free.len()).then(|| {
            format!(
                "need {} SGPRs never referenced by the kernel, found {}",
                need.len(),
                free.len()
            )
        });
        self.reserved = need.into_iter().zip(free).collect();
    }

    /// A workgroup id the logical grid does not vary (X when it is one workgroup wide, Y and Z when it is
    /// one-dimensional) is 0 in every tile, so reads of its ttmp become the constant 0 instead of a reserved SGPR.
    /// The hardware id may differ: sparse and admission launches flatten the padded grid into X, and gfx1250 leaves
    /// ttmp7 unwritten when a kernel does not enable workgroup ids Y and Z.
    pub fn zero_constant_ids(&mut self, workgroups: [u64; 3]) {
        let [gx, gy, gz] = workgroups;
        let fixed: Vec<u32> = [(TTMP_WG_X, gx == 1), (TTMP_WG_YZ, gy == 1 && gz == 1)]
            .into_iter()
            .filter_map(|(n, f)| f.then_some(n))
            .collect();
        let remapped_before = self.reads_ttmp(TTMP_WG_X) || self.reads_ttmp(TTMP_WG_YZ);
        for s in &mut self.sites {
            let (zeroed, kept): (Vec<u32>, Vec<u32>) =
                s.ttmps.iter().partition(|n| fixed.contains(n));
            s.ttmps = kept;
            s.zeroed.extend(zeroed);
        }
        if remapped_before && !self.reads_ttmp(TTMP_WG_X) && !self.reads_ttmp(TTMP_WG_YZ) {
            self.problems
                .push("kernel reads no workgroup id the grid varies (nothing to remap)".into());
        }
        self.reserve();
    }

    /// A sparse launch dispatches more workgroups than were captured. Kernarg copies of the grid size keep their
    /// captured values, but the dispatch packet holds the padded grid, so the body may read the packet only through
    /// `s_load` with an immediate offset outside grid_size_x/y/z (bytes 12..24) while the pointer is unmodified.
    pub fn dispatch_packet_grid_problems(&self) -> Vec<String> {
        let Some(dp) = self.kd.dispatch_ptr_sgpr() else {
            return vec![];
        };
        let pointer = RegRange {
            class: RegClass::Sgpr,
            lo: dp,
            hi: dp + 1,
        };
        let succ = successors(&self.insts);
        // live[i]: bit 0/1 set if s[dp]/s[dp+1] may still hold the entry value when instruction i starts.
        let mut live = vec![0u8; self.insts.len()];
        let mut work = vec![(0usize, 3u8)];
        let mut problems = BTreeSet::new();
        while let Some((i, bits)) = work.pop() {
            if live[i] | bits == live[i] {
                continue;
            }
            live[i] |= bits;
            let state = live[i];
            let inst = &self.insts[i];
            let def = inst.scalar_def();
            let reads_pointer = inst.scalar_uses(def).iter().any(|r| {
                (state & 1 != 0 && r.contains(RegClass::Sgpr, dp))
                    || (state & 2 != 0 && r.contains(RegClass::Sgpr, dp + 1))
            });
            if reads_pointer {
                if let Err(e) = packet_load_outside_grid(inst, pointer, state) {
                    problems.insert(format!("{e} at {:#x}: {}", inst.addr, inst.text()));
                }
            }
            let mut out = state;
            if let Some(d) = def.filter(|d| d.class == RegClass::Sgpr) {
                for (bit, reg) in [(1u8, dp), (2, dp + 1)] {
                    if d.lo <= reg && reg <= d.hi {
                        out &= !bit;
                    }
                }
            }
            for &s in &succ[i] {
                if out & !live[s] != 0 {
                    work.push((s, out));
                }
            }
        }
        problems.into_iter().collect()
    }
}

/// Accept `s_load_<w> dst, s[dp:dp+1], <imm>` whose bytes miss the packet's grid_size fields.
fn packet_load_outside_grid(inst: &Inst, pointer: RegRange, state: u8) -> Result<(), String> {
    let ops = inst.operand_list();
    let width = inst.mnemonic.strip_prefix("s_load_").and_then(|w| {
        Some(match w.trim_start_matches(['b', 'u', 'i']) {
            "8" => 1,
            "16" => 2,
            "32" => 4,
            "64" => 8,
            "96" => 12,
            "128" => 16,
            "256" => 32,
            "512" => 64,
            _ => return None,
        })
    });
    let (Some(width), 3) = (width, ops.len()) else {
        return Err("dispatch packet pointer used outside an immediate-offset s_load".into());
    };
    let offset = ops[2].split_whitespace().next().unwrap_or("");
    let offset = match offset.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16).ok(),
        None => offset.parse().ok(),
    };
    match offset {
        Some(off) if state == 3 && ops[1].trim() == pointer.render() => {
            if off < 24 && off + width > 12 {
                Err("reads the dispatch packet's grid size".into())
            } else {
                Ok(())
            }
        }
        _ => Err("dispatch packet pointer used outside an immediate-offset s_load".into()),
    }
}

pub fn pick_kernel(elf: &Elf, symbol: Option<&str>) -> Result<String> {
    if let Some(s) = symbol {
        return Ok(s.strip_suffix(".kd").unwrap_or(s).to_string());
    }
    match elf.kernels()?.as_slice() {
        [one] => Ok(one.clone()),
        many => bail!(
            "code object has {} kernels; select one: {many:?}",
            many.len()
        ),
    }
}

/// Analyze `symbol` (or the only kernel) of `code`, given its objdump listing.
pub fn analyze(code: &[u8], symbol: Option<&str>, listing: &str) -> Result<Analysis> {
    analyze_insts(code, symbol, super::isa::parse_objdump(listing))
}

/// Analyze `symbol` (or the only kernel) of `code`, given its instructions.
pub fn analyze_insts(code: &[u8], symbol: Option<&str>, all: Vec<Inst>) -> Result<Analysis> {
    // Let the object reader validate the container before using descriptor/layout offsets.
    object::File::parse(code).context("invalid code-object ELF")?;
    let elf = Elf::parse(code.to_vec())?;
    let symbol = pick_kernel(&elf, symbol)?;
    let kd_sym = elf.symbol(&format!("{symbol}.kd"))?;
    let func = elf.symbol(&symbol)?;
    let kd = KernelDescriptor::parse(&elf, kd_sym.value)?;
    let lo = func.value;
    let hi = if func.size > 0 {
        lo + func.size
    } else {
        // Run-time generated code objects (gfxjit) leave the function size 0: the kernel extends to the next symbol
        // in its section, or the section end.
        let sec = elf
            .sections
            .iter()
            .find(|s| s.flags & 0x4 != 0 && s.addr <= lo && lo < s.addr + s.size)
            .context("kernel symbol outside every executable section")?;
        elf.symbols(".symtab")?
            .iter()
            .map(|s| s.value)
            .filter(|&v| lo < v && v < sec.addr + sec.size)
            .fold(sec.addr + sec.size, u64::min)
    };
    let insts: Vec<Inst> = all
        .into_iter()
        .filter(|i| lo <= i.addr && i.addr < hi)
        .collect();
    if insts.is_empty() {
        bail!("no instructions of {symbol} in the listing");
    }
    let mut problems = Vec::new();
    let mut next = lo;
    for i in &insts {
        if i.addr != next || i.mnemonic.contains("unknown") || i.mnemonic.contains("invalid") {
            problems.push(format!(
                "disassembly does not tile the kernel at {next:#x} (undecodable bytes?)"
            ));
        }
        next = i.end();
    }
    if next != hi {
        let tail_ok = next < hi
            && (hi - next) % 4 == 0
            && elf
                .vaddr_to_offset(next)
                .ok()
                .and_then(|o| elf.data.get(o..o + (hi - next) as usize))
                .is_some_and(|b| b.chunks_exact(4).all(|w| w == 0xBF9F0000u32.to_le_bytes()));
        if !tail_ok {
            problems.push(format!(
                "disassembly does not tile the kernel at {next:#x} (undecodable bytes?)"
            ));
        }
    }
    // Words LLVM rejects were decoded from the ISA table where it identifies them (`disassemble_insts`). Their
    // operands are modeled from the ISA, but they are never patched, so one naming a ttmp is refused.
    for i in &insts {
        if i.mnemonic == ".long" {
            problems.push(format!("undecodable instruction word at {:#x}", i.addr));
        } else if i.spec && i.scalar_regs().iter().any(|r| r.class == RegClass::Ttmp) {
            problems.push(format!(
                "{} at {:#x} (decoded from the ISA table) names a ttmp",
                i.mnemonic, i.addr
            ));
        }
    }
    if !kd.wave32() {
        problems.push("not a wave32 kernel".into());
    }
    if kd.entry_vaddr != lo {
        problems.push("kernel entry is not the start of the kernel symbol".into());
    }
    // Scratch needs no handling: gfx1250 has architected flat scratch, so the hardware initializes FLAT_SCRATCH and
    // no wave-offset SGPR follows the user SGPRs.
    let inputs = kd.user_sgpr_inputs();
    if kd.kernarg_sgpr().is_none() {
        problems.push("no kernarg segment pointer".into());
    }
    if kd.vgprs() < 6 {
        problems.push("prologue needs v1..v5".into());
    }
    let mut sgprs_used = BTreeSet::new();
    let addresses: HashMap<_, _> = insts.iter().enumerate().map(|(i, x)| (x.addr, i)).collect();
    for i in &insts {
        if i.is_call() {
            problems.push(format!("call at {:#x}: callees are not analyzed", i.addr));
        }
        if i.mnemonic.starts_with("s_movrel") || i.mnemonic.starts_with("s_set_gpr_idx") {
            problems.push(format!("indirect SGPR access at {:#x}", i.addr));
        }
        for r in i.scalar_regs() {
            if r.class == RegClass::Sgpr {
                sgprs_used.extend(r.lo..=r.hi);
            } else if r.lo != r.hi && [6, 7, 9].iter().any(|&n| r.contains(RegClass::Ttmp, n)) {
                problems.push(format!(
                    "ttmp range overlapping workgroup registers at {:#x}",
                    i.addr
                ));
            }
        }
        for n in [TTMP_CLUSTER, TTMP_WG_YZ, TTMP_WG_X] {
            if i.writes_ttmp(n) {
                problems.push(format!("writes ttmp{n} at {:#x}", i.addr));
            }
        }
        if let Some(t) = i.branch_target() {
            if !addresses.contains_key(&t) {
                problems.push(format!(
                    "branch at {:#x} to {t:#x} leaves the kernel",
                    i.addr
                ));
            }
        }
    }
    let succ = successors(&insts);
    let mut sites = Vec::new();
    let mut rank_reads = Vec::new();
    for (index, i) in insts.iter().enumerate() {
        let mut ttmps: Vec<_> = [9, 7].into_iter().filter(|&n| i.reads_ttmp(n)).collect();
        if i.reads_ttmp(6) {
            match classify_rank(&insts, &succ, index) {
                Ok(kind) => {
                    if kind == RankKind::Data {
                        ttmps.push(6);
                    }
                    rank_reads.push(RankRead {
                        index,
                        addr: i.addr,
                        text: i.text(),
                        kind,
                    });
                }
                Err(e) => problems.push(format!("{e}: {}", i.text())),
            }
        }
        if !ttmps.is_empty() {
            sites.push(Site {
                index,
                addr: i.addr,
                text: i.text(),
                ttmps,
                zeroed: vec![],
            });
        }
    }
    let live_in_sgprs = kd.user_sgpr_count() + kd.tg_size_enabled() as u32;
    if !sites
        .iter()
        .any(|s| s.ttmps.contains(&9) || s.ttmps.contains(&7))
    {
        problems.push("kernel never reads its workgroup id (nothing to remap)".into());
    }
    if kd.preload_length() > 0 && inputs != ["kernarg_segment_ptr"] {
        problems.push(
            "preload conversion assumes only the kernarg pointer precedes preloaded SGPRs".into(),
        );
    }
    let preload_sgpr_base = kd
        .user_sgpr_count()
        .checked_sub(kd.preload_length())
        .unwrap_or_else(|| {
            problems.push("preload length exceeds user SGPR count".into());
            0
        });
    let mut a = Analysis {
        symbol,
        kd,
        kernel_vaddr: lo,
        kernel_size: hi - lo,
        instructions: insts.len(),
        insts,
        sites,
        rank_reads,
        sgprs_used,
        live_in_sgprs,
        reserved: BTreeMap::new(),
        sgpr_shortage: None,
        preload_sgpr_base,
        problems,
    };
    a.reserve();
    Ok(a)
}

pub fn analyze_code_object(code: &[u8], symbol: Option<&str>) -> Result<Analysis> {
    analyze_insts(code, symbol, super::disassemble_insts(code)?)
}

fn successors(insts: &[Inst]) -> Vec<Vec<usize>> {
    let addresses: HashMap<_, _> = insts.iter().enumerate().map(|(i, x)| (x.addr, i)).collect();
    insts
        .iter()
        .enumerate()
        .map(|(j, i)| {
            if i.ends_flow() {
                return vec![];
            }
            let mut out = Vec::new();
            if (!i.is_branch() || i.is_conditional_branch()) && j + 1 < insts.len() {
                out.push(j + 1);
            }
            if let Some(t) = i.branch_target().and_then(|t| addresses.get(&t)) {
                out.push(*t);
            }
            out
        })
        .collect()
}

fn classify_rank(insts: &[Inst], succ: &[Vec<usize>], index: usize) -> Result<RankKind> {
    let inst = &insts[index];
    if inst.is_mask_shift_of(RegRange {
        class: RegClass::Ttmp,
        lo: 6,
        hi: 6,
    }) {
        return Ok(RankKind::Multicast);
    }
    let Some(d) = inst
        .scalar_def()
        .filter(|d| d.class == RegClass::Sgpr && d.lo == d.hi)
    else {
        return Ok(RankKind::Data);
    };
    // Follow the value, and scalar ALU results computed from it, to their uses: shift amounts of s_lshl_b32 are
    // multicast masks; every other use is data.
    let mut work: Vec<(usize, u32)> = succ[index].iter().map(|&j| (j, d.lo)).collect();
    let mut seen = BTreeSet::new();
    let (mut mask, mut data) = (false, false);
    while let Some((j, r)) = work.pop() {
        if !seen.insert((j, r)) {
            continue;
        }
        let i = &insts[j];
        let reg = RegRange {
            class: RegClass::Sgpr,
            lo: r,
            hi: r,
        };
        let def = i.scalar_def();
        if i.scalar_uses(def)
            .iter()
            .any(|u| u.contains(RegClass::Sgpr, r))
        {
            match def.filter(|x| x.class == RegClass::Sgpr && x.lo == x.hi) {
                _ if i.is_shift_amount_of(reg) => mask = true,
                Some(x) if i.is_scalar_derivation() => {
                    work.extend(succ[j].iter().map(|&k| (k, x.lo)));
                }
                _ => data = true,
            }
        }
        if !def.is_some_and(|x| x.contains(RegClass::Sgpr, r)) {
            work.extend(succ[j].iter().map(|&k| (k, r)));
        }
    }
    if mask && data {
        bail!(
            "ttmp6 value at {:#x} feeds both a multicast mask and data",
            inst.addr
        );
    }
    Ok(if mask {
        RankKind::Multicast
    } else {
        RankKind::Data
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kernarg_sgpr_follows_enabled_inputs() {
        let kd = |code_properties| KernelDescriptor {
            vaddr: 0,
            file_offset: 0,
            group_segment_fixed_size: 0,
            private_segment_fixed_size: 0,
            kernarg_size: 0,
            entry_vaddr: 0,
            rsrc1: 0,
            rsrc2: 0,
            rsrc3: 0,
            code_properties,
            kernarg_preload: 0,
        };
        // kernarg only; dispatch + queue + kernarg + dispatch_id; all three preceding inputs; none.
        for (bits, expected) in [
            (0b1000, Some(0)),
            (0b1_1110, Some(4)),
            (0b1111, Some(8)),
            (0b0110, None),
        ] {
            assert_eq!(kd(bits).kernarg_sgpr(), expected);
        }
        let mut wide = kd(0b1000);
        wide.rsrc2 = 32 << 1 | 1 << 7;
        assert_eq!(wide.user_sgpr_count(), 32);
    }
    #[test]
    fn reaching_rank_uses_respect_kills() {
        for (tail, expected) in [
            (
                "s_lshl_b32 s2, 1, s0\ns_mov_b32 s0, 0\ns_add_u32 s3, s0, 1",
                true,
            ),
            ("s_lshl_b32 s2, 1, s0\ns_cmp_eq_u32 s0, 1", false),
        ] {
            let text = format!("s_and_b32 s0, ttmp6, 3\n{tail}");
            let listing = text
                .lines()
                .enumerate()
                .map(|(j, l)| format!("\t{l} // {:012X}: 00000000\n", j * 4))
                .collect::<String>();
            let insts = super::super::isa::parse_objdump(&listing);
            let result = classify_rank(&insts, &successors(&insts), 0);
            if expected {
                assert_eq!(result.unwrap(), RankKind::Multicast);
            } else {
                assert!(result.is_err());
            }
        }
    }
    #[test]
    fn rank_classification_follows_scalar_arithmetic() {
        // gfxjit GEMM: a 2-D cluster's row mask 15 << (wg_y * 4), and tile coordinates from wg_x.
        for (text, expected) in [
            (
                "s_mov_b32 s7, ttmp6\ns_bfe_u32 s13, s7, 0x40004\ns_mul_i32 s13, s13, 4\ns_mov_b32 s7, 15\n\
                 s_lshl_b32 s7, s7, s13",
                RankKind::Multicast,
            ),
            (
                "s_and_b32 s8, ttmp6, 15\ns_add_co_u32 s8, s8, s10\ns_lshl_b32 s9, s8, 2\ns_cmp_eq_u32 s9, 0",
                RankKind::Data,
            ),
        ] {
            let listing = text
                .lines()
                .enumerate()
                .map(|(j, l)| format!("\t{l} // {:012X}: 00000000\n", j * 4))
                .collect::<String>();
            let insts = super::super::isa::parse_objdump(&listing);
            assert_eq!(classify_rank(&insts, &successors(&insts), 0).unwrap(), expected);
        }
    }
}
