//! gfx1250 instructions as printed by `llvm-objdump -d`, with the scalar-register reads and writes the remap
//! analysis needs. Only scalar registers are modeled; everything the model is unsure about is treated
//! conservatively by the caller (as a read that keeps a value live, never as a kill).

/// Scalar operand code of ttmp0 in SALU/VALU source fields.
pub const TTMP_OPERAND_BASE: u8 = 108;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Inst {
    pub addr: u64,
    pub bytes: Vec<u8>,
    pub mnemonic: String,
    /// Operand text after the mnemonic, with runs of whitespace collapsed.
    pub operands: String,
    /// Decoded from the ISA table (`spec.rs`) because LLVM's disassembler rejects the encoding.
    pub spec: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegClass {
    Sgpr,
    Ttmp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegRange {
    pub class: RegClass,
    pub lo: u32,
    pub hi: u32,
}

impl RegRange {
    pub fn contains(&self, class: RegClass, reg: u32) -> bool {
        self.class == class && self.lo <= reg && reg <= self.hi
    }
}

/// Parse every instruction line (`\t<text> // <ADDR>: <WORD>... [<label>]`) of an objdump listing.
pub fn parse_objdump(listing: &str) -> Vec<Inst> {
    let mut out = Vec::new();
    for line in listing.lines() {
        if !line.starts_with(['\t', ' ']) {
            continue;
        }
        let Some((text, tail)) = line.rsplit_once("//") else {
            continue;
        };
        let mut fields = tail.split_whitespace();
        let Some(addr) = fields.next().and_then(|a| a.strip_suffix(':')) else {
            continue;
        };
        let Ok(addr) = u64::from_str_radix(addr, 16) else {
            continue;
        };
        let mut bytes = Vec::new();
        for word in fields.take_while(|w| !w.starts_with('<')) {
            match u32::from_str_radix(word, 16) {
                Ok(w) if word.len() == 8 => bytes.extend_from_slice(&w.to_le_bytes()),
                _ => break,
            }
        }
        let mut cleaned = String::new();
        let mut rest = text;
        while let Some((before, comment)) = rest.split_once("/*") {
            cleaned.push_str(before);
            let Some((_, after)) = comment.split_once("*/") else {
                break;
            };
            rest = after;
        }
        cleaned.push_str(rest);
        let text = cleaned.trim();
        if bytes.is_empty() || text.is_empty() {
            continue;
        }
        let (mnemonic, operands) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
        out.push(Inst {
            addr,
            bytes,
            mnemonic: mnemonic.to_string(),
            operands: operands.split_whitespace().collect::<Vec<_>>().join(" "),
            spec: false,
        });
    }
    out
}

impl Inst {
    pub fn text(&self) -> String {
        if self.operands.is_empty() {
            self.mnemonic.clone()
        } else {
            format!("{} {}", self.mnemonic, self.operands)
        }
    }

    pub fn size(&self) -> u64 {
        self.bytes.len() as u64
    }

    pub fn end(&self) -> u64 {
        self.addr + self.size()
    }

    /// Top-level comma-separated operands (commas inside `()` or `[]` do not split).
    pub fn operand_list(&self) -> Vec<&str> {
        split_operands(&self.operands)
    }

    /// Every scalar register or range named in the operands.
    pub fn scalar_regs(&self) -> Vec<RegRange> {
        scalar_regs(&self.operands)
    }

    pub fn reads_ttmp(&self, reg: u32) -> bool {
        let def = self.scalar_def();
        self.scalar_uses(def)
            .iter()
            .any(|r| r.contains(RegClass::Ttmp, reg))
    }

    pub fn writes_ttmp(&self, reg: u32) -> bool {
        // Any first operand naming the ttmp counts, not only certain definitions: a write we can't model must refuse.
        let first = self
            .operand_list()
            .first()
            .map(|o| scalar_regs(o))
            .unwrap_or_default();
        !self.is_non_writing() && first.iter().any(|r| r.contains(RegClass::Ttmp, reg))
    }

    /// The scalar destination this instruction certainly writes, if it is a form whose first operand is a full
    /// (unconditional) destination. `None` for stores, compares, branches, conditional moves and anything unknown.
    pub fn scalar_def(&self) -> Option<RegRange> {
        let m = self.mnemonic.as_str();
        // Unknown opcodes must keep all operands live, never invent a kill.
        const DEFINES: &[&str] = &[
            "s_mov_b",
            "s_movk_",
            "s_add_",
            "s_addc_",
            "s_sub_",
            "s_subb_",
            "s_mul_",
            "s_min_",
            "s_max_",
            "s_and_",
            "s_andn",
            "s_or_",
            "s_orn",
            "s_xor_",
            "s_xnor_",
            "s_nand_",
            "s_nor_",
            "s_not_",
            "s_lshl",
            "s_lshr",
            "s_ashr",
            "s_bfe_",
            "s_bfm_",
            "s_brev_",
            "s_bcnt",
            "s_ff0_",
            "s_ff1_",
            "s_flbit_",
            "s_sext_",
            "s_abs_",
            "s_cselect_",
            "s_pack_",
            "s_load_",
            "s_buffer_load_",
            "s_getreg_",
            "s_getpc_",
            "s_sendmsg_rtn_",
            "s_bitset",
            "s_addk_",
            "s_mulk_",
        ];
        let writes_first = DEFINES.iter().any(|p| m.starts_with(p))
            || m.starts_with("v_readlane_b32")
            || m.starts_with("v_readfirstlane_b32");
        if !writes_first {
            return None;
        }
        let first = *self.operand_list().first()?;
        match scalar_regs(first).as_slice() {
            [r] if first.trim() == r.render() => Some(*r),
            _ => None,
        }
    }

    /// Scalar registers read: every named register except a certain destination, plus the destination itself for
    /// read-modify-write forms.
    pub fn scalar_uses(&self, def: Option<RegRange>) -> Vec<RegRange> {
        let skip_first = def.is_some() && !self.reads_own_destination();
        self.operand_list()
            .iter()
            .enumerate()
            .filter(|(i, _)| !(*i == 0 && skip_first))
            .flat_map(|(_, op)| scalar_regs(op))
            .collect()
    }

    pub fn is_branch(&self) -> bool {
        self.mnemonic == "s_branch" || self.mnemonic.starts_with("s_cbranch_")
    }

    pub fn is_conditional_branch(&self) -> bool {
        self.mnemonic.starts_with("s_cbranch_")
    }

    pub fn ends_flow(&self) -> bool {
        matches!(
            self.mnemonic.as_str(),
            "s_endpgm" | "s_endpgm_saved" | "s_code_end" | "s_setpc_b64" | "s_rfe_b64"
        )
    }

    pub fn is_call(&self) -> bool {
        let m = self.mnemonic.as_str();
        m.starts_with("s_swappc")
            || m.starts_with("s_setpc")
            || m.starts_with("s_call")
            || m.starts_with("s_rfe")
    }

    /// SOPP branch target: PC + 4 + simm16 * 4.
    pub fn branch_target(&self) -> Option<u64> {
        if !self.is_branch() || self.bytes.len() < 4 {
            return None;
        }
        let simm = i16::from_le_bytes([self.bytes[0], self.bytes[1]]) as i64;
        Some((self.addr as i64 + 4 + simm * 4) as u64)
    }

    /// `s_lshl_b32 dst, <immediate>, reg`: the shift amount of a constant, i.e. the base of a multicast mask.
    pub fn is_mask_shift_of(&self, reg: RegRange) -> bool {
        if self.mnemonic != "s_lshl_b32" {
            return false;
        }
        let ops = self.operand_list();
        ops.len() == 3 && is_immediate(ops[1]) && ops[2].trim() == reg.render()
    }

    /// `s_lshl_b32 dst, x, reg` with `reg` only as the shift amount.
    pub fn is_shift_amount_of(&self, reg: RegRange) -> bool {
        if self.mnemonic != "s_lshl_b32" {
            return false;
        }
        let ops = self.operand_list();
        ops.len() == 3 && ops[2].trim() == reg.render() && ops[1].trim() != reg.render()
    }

    /// Scalar ALU arithmetic whose single SGPR result is a function of its sources.
    pub fn is_scalar_derivation(&self) -> bool {
        const OPS: &[&str] = &[
            "s_mov_b32",
            "s_and_b32",
            "s_or_b32",
            "s_xor_b32",
            "s_bfe_u32",
            "s_bfe_i32",
            "s_lshl_b32",
            "s_lshr_b32",
            "s_ashr_i32",
            "s_mul_i32",
            "s_add_co_u32",
            "s_add_co_i32",
            "s_add_u32",
            "s_add_i32",
            "s_sub_co_u32",
            "s_sub_co_i32",
            "s_sub_u32",
            "s_sub_i32",
        ];
        OPS.contains(&self.mnemonic.as_str())
    }

    fn reads_own_destination(&self) -> bool {
        let m = self.mnemonic.as_str();
        m.starts_with("s_bitset")
            || m.starts_with("s_addk")
            || m.starts_with("s_mulk")
            || m.starts_with("s_cmov")
    }

    /// Scalar instructions whose first operand is not a written scalar destination.
    fn is_non_writing(&self) -> bool {
        const PREFIXES: &[&str] = &[
            "s_cmp",
            "s_bitcmp",
            "s_cbranch",
            "s_branch",
            "s_wait",
            "s_barrier",
            "s_nop",
            "s_sleep",
            "s_setreg",
            "s_store",
            "s_buffer_store",
            "s_scratch_store",
            "s_atomic",
            "s_buffer_atomic",
            "s_cmov",
            "s_ttracedata",
            "s_prefetch",
            "s_set_vgpr_msb",
            "s_delay_alu",
            "s_setprio",
            "s_endpgm",
            "s_trap",
            "s_dcache",
            "s_icache",
            "s_setpc",
            "s_rfe",
            "s_setkill",
            "s_set_inst_prefetch",
            "s_clause",
            "s_code_end",
            "s_round_mode",
            "s_denorm_mode",
            "s_singleuse",
            "s_version",
            "s_wakeup",
            "s_monitor_sleep",
            "s_signal",
            "s_cluster",
            "s_dealloc",
            "s_alloc",
            "s_inst_prefetch",
            "s_ttrace",
            "s_mov_regrd",
            "s_movreld",
            "s_icache_inv",
        ];
        let m = self.mnemonic.as_str();
        m == "s_sendmsg" || m == "s_sendmsghalt" || PREFIXES.iter().any(|p| m.starts_with(p))
    }
}

impl RegRange {
    pub fn render(&self) -> String {
        let name = match self.class {
            RegClass::Sgpr => "s",
            RegClass::Ttmp => "ttmp",
        };
        if self.lo == self.hi {
            format!("{name}{}", self.lo)
        } else {
            format!("{name}[{}:{}]", self.lo, self.hi)
        }
    }
}

pub fn split_operands(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0i32, 0usize);
    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    if !s[start..].trim().is_empty() {
        out.push(s[start..].trim());
    }
    out
}

fn is_immediate(op: &str) -> bool {
    let op = op.trim();
    let digits = op.strip_prefix('-').unwrap_or(op);
    !digits.is_empty() && digits.as_bytes()[0].is_ascii_digit()
}

/// Every `sN`, `s[a:b]`, `ttmpN` and `ttmp[a:b]` in `s`, at identifier boundaries.
pub fn scalar_regs(s: &str) -> Vec<RegRange> {
    let b = s.as_bytes();
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c == b'.';
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if i > 0 && ident(b[i - 1]) {
            i += 1;
            continue;
        }
        let (class, prefix) = if b[i..].starts_with(b"ttmp") {
            (RegClass::Ttmp, 4)
        } else if b[i] == b's' {
            (RegClass::Sgpr, 1)
        } else {
            i += 1;
            continue;
        };
        let j = i + prefix;
        if let Some((lo, hi, next)) = parse_number_or_range(b, j) {
            if next >= b.len() || !ident(b[next]) {
                out.push(RegRange { class, lo, hi });
                i = next;
                continue;
            }
        }
        i += 1;
    }
    out
}

fn parse_number_or_range(b: &[u8], j: usize) -> Option<(u32, u32, usize)> {
    let number = |mut k: usize| -> Option<(u32, usize)> {
        let start = k;
        while k < b.len() && b[k].is_ascii_digit() {
            k += 1;
        }
        (k > start)
            .then(|| {
                std::str::from_utf8(&b[start..k])
                    .ok()?
                    .parse()
                    .ok()
                    .map(|v| (v, k))
            })
            .flatten()
    };
    if j < b.len() && b[j] == b'[' {
        let (lo, k) = number(j + 1)?;
        if b.get(k) != Some(&b':') {
            return None;
        }
        let (hi, k) = number(k + 1)?;
        if b.get(k) != Some(&b']') {
            return None;
        }
        return Some((lo, hi, k + 1));
    }
    let (v, k) = number(j)?;
    Some((v, v, k))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(text: &str) -> Inst {
        let (m, o) = text.split_once(' ').unwrap_or((text, ""));
        Inst {
            addr: 0x1000,
            bytes: vec![0; 4],
            mnemonic: m.into(),
            operands: o.into(),
            spec: false,
        }
    }

    #[test]
    fn parses_objdump_lines_including_branch_labels() {
        let listing = "\ts_and_b32 s15, ttmp6, 15                                   // 0000000019EC: 8B0F8F72\n\
                       \ts_cbranch_scc1 3                // 000000001F44: BFA20003 <k+0x750>\n\
                       \ts_bfe_u32 s33, ttmp8, 0x50019   // 000000001830: 9321FF74 00050019\n";
        let insts = parse_objdump(listing);
        assert_eq!(insts.len(), 3);
        assert_eq!(insts[0].bytes, vec![0x72, 0x8f, 0x0f, 0x8b]);
        assert_eq!(insts[1].branch_target(), Some(0x1F44 + 4 + 12));
        assert_eq!(insts[2].size(), 8);
    }

    #[test]
    fn scans_scalar_registers_at_identifier_boundaries() {
        let regs = scalar_regs("s[4:11], s[0:1], 0x28 nv ttmp6 scope:SCOPE_SE s_x vcc s105");
        let r: Vec<String> = regs.iter().map(|r| r.render()).collect();
        assert_eq!(r, ["s[4:11]", "s[0:1]", "ttmp6", "s105"]);
        assert!(scalar_regs("hwreg(HW_REG_HW_ID1)").is_empty());
    }

    #[test]
    fn models_definitions_and_uses() {
        let i = inst("s_and_b32 s0, ttmp6, 3");
        assert_eq!(i.scalar_def().map(|r| r.render()), Some("s0".into()));
        assert!(i.reads_ttmp(6) && !i.writes_ttmp(6));
        let shift = inst("s_lshl_b32 s0, 0x1111, s0");
        let s0 = RegRange {
            class: RegClass::Sgpr,
            lo: 0,
            hi: 0,
        };
        assert!(shift.is_mask_shift_of(s0));
        assert!(shift
            .scalar_uses(shift.scalar_def())
            .iter()
            .any(|r| r.contains(RegClass::Sgpr, 0)));
        assert!(!inst("s_lshl_b32 s21, ttmp6, 6").is_mask_shift_of(s0));
        assert_eq!(inst("s_cmp_eq_u32 s0, 0").scalar_def(), None);
        assert_eq!(
            inst("s_cselect_b32 s3, -1, 0")
                .scalar_def()
                .map(|r| r.render()),
            Some("s3".into())
        );
        assert!(inst("s_mov_b32 ttmp9, s0").writes_ttmp(9));
        assert!(inst("s_swappc_b64 s[30:31], s[16:17]").is_call());
        assert_eq!(inst("s_unknown_opcode s0, s1").scalar_def(), None);
        let commented =
            parse_objdump("\tv_wmma v0 /*v256*/, v1 /*v257*/// 00001000: CC620078 1DE20108");
        assert_eq!(commented.len(), 1);
        assert_eq!(commented[0].size(), 8);
        assert!(!commented[0].operands.contains("/*"));
    }
}
