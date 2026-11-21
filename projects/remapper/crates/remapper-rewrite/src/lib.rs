//! Binary workgroup remap for multi-SIMD ATT on gfx1250: the code-object rewrite and the placement tables.
//!
//! A kernel's code object is rewritten statically: a prologue appended after the kernel picks each workgroup's
//! tile from `table[key][turn]`, keyed by its hardware location, and the body's workgroup-id reads are redirected
//! to reserved SGPRs by changing one operand byte, so body addresses and every other byte stay the same. Replays
//! then put the same work on the traced WGP in every run, which is what lets ATT samples from different replays
//! (one per SIMD) be merged into one view.
//!
//! - [`analyze`]: kernel descriptor, disassembly, workgroup-id and cluster-rank sites, refusals.
//! - [`spec`]: instructions LLVM's disassembler rejects, decoded from AMD's machine-readable ISA.
//! - [`prologue`]: prologue rendering and assembly, operand substitution.
//! - [`elf`]: code-object growth and patching.
//! - [`markers`]: opt-in `s_ttracedata` stubs at body barriers.
//! - [`table`]: record logs to placement table, and placement checks.
//! - [`sparse_table`]: the sparse launch model (one workgroup per WGP) for launches smaller than the GPU.
//! - [`mode`]: what the prologue does in a replay, and what a replay left behind.

pub mod analyze;
pub mod elf;
pub mod isa;
pub mod manifest;
pub mod markers;
pub mod mode;
pub mod prologue;
pub mod rewrite;
pub mod sparse_table;
pub mod spec;
pub mod table;

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};

pub const MCPU: &str = "gfx1250";

/// A ROCm LLVM tool: `$REMAPPER_LLVM_BIN`, `$ROCM_PATH/llvm/bin`, `/opt/rocm/llvm/bin`, the newest `/opt/rocm-*`,
/// then `PATH`.
pub fn llvm_tool(name: &str) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dir) = std::env::var_os("REMAPPER_LLVM_BIN") {
        candidates.push(PathBuf::from(dir).join(name));
    }
    if let Some(rocm) = std::env::var_os("ROCM_PATH") {
        candidates.push(PathBuf::from(rocm).join("llvm/bin").join(name));
    }
    candidates.push(PathBuf::from("/opt/rocm/llvm/bin").join(name));
    if let Ok(entries) = std::fs::read_dir("/opt") {
        let mut versioned: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("rocm-"))
            .map(|e| e.path().join("llvm/bin").join(name))
            .collect();
        versioned.sort();
        versioned.reverse();
        candidates.extend(versioned);
    }
    if let Some(path) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&path).map(|d| d.join(name)));
    }
    candidates
        .into_iter()
        .find(|p| p.is_file())
        .with_context(|| format!("could not find {name}; set REMAPPER_LLVM_BIN or ROCM_PATH"))
}

/// `llvm-objdump -d --mcpu=gfx1250` of a code object.
pub fn disassemble(code_object: &Path) -> Result<String> {
    let tool = llvm_tool("llvm-objdump")?;
    let out = Command::new(&tool)
        .args(["-d", &format!("--mcpu={MCPU}")])
        .arg(code_object)
        .output()
        .with_context(|| format!("running {}", tool.display()))?;
    if !out.status.success() {
        bail!(
            "llvm-objdump failed on {}: {}",
            code_object.display(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// `s_nop 0`, which stands in for table-decoded instructions so LLVM resynchronizes after them.
const S_NOP: [u8; 4] = 0xBF80_0000u32.to_le_bytes();

/// The instructions of a code object: LLVM's disassembly, with every word LLVM rejects decoded from the ISA table
/// where the table identifies it (see `spec.rs`). Words neither decodes stay `.long` entries.
pub fn disassemble_insts(code: &[u8]) -> Result<Vec<isa::Inst>> {
    let elf = elf::Elf::parse(code.to_vec())?;
    let mut masked = code.to_vec();
    let mut decoded: std::collections::BTreeMap<u64, isa::Inst> = Default::default();
    // Each pass masks the instructions it decodes with `s_nop`, so the next pass sees LLVM's decoding of what
    // follows them instead of words read out of step.
    for _ in 0..8 {
        let listing = isa::parse_objdump(&disassemble_bytes(&masked)?);
        let mut covered = 0u64;
        let mut progress = false;
        for i in &listing {
            if i.mnemonic != ".long"
                || i.addr < covered
                || decoded
                    .range(..=i.addr)
                    .next_back()
                    .is_some_and(|(_, d)| i.addr < d.end())
            {
                continue;
            }
            let Ok(off) = elf.vaddr_to_offset(i.addr) else {
                continue;
            };
            let Some(d) = spec::decode(&code[off..code.len().min(off + 16)]) else {
                continue;
            };
            for w in masked[off..off + d.len].chunks_exact_mut(4) {
                w.copy_from_slice(&S_NOP);
            }
            decoded.insert(
                i.addr,
                isa::Inst {
                    addr: i.addr,
                    bytes: code[off..off + d.len].to_vec(),
                    mnemonic: d.name,
                    operands: d.operands.join(", "),
                    spec: true,
                },
            );
            covered = i.addr + d.len as u64;
            progress = true;
        }
        if !progress {
            let mut out = Vec::with_capacity(listing.len());
            let mut skip_until = 0u64;
            for i in listing {
                if i.addr < skip_until {
                    continue;
                }
                match decoded.get(&i.addr) {
                    Some(d) => {
                        skip_until = d.end();
                        out.push(d.clone());
                    }
                    None => out.push(i),
                }
            }
            return Ok(out);
        }
    }
    bail!("instruction decoding did not settle after 8 passes")
}

pub fn disassemble_bytes(code: &[u8]) -> Result<String> {
    let file = tempfile::NamedTempFile::new()?;
    std::fs::write(file.path(), code)?;
    disassemble(file.path())
}

#[cfg(test)]
mod tests;
