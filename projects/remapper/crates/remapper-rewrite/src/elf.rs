//! Minimal ELF64 little-endian access for AMDGPU code objects, and the two ways of adding the remap prologue
//! without moving any existing virtual address: growing `.text` upward into the slack before the next allocated
//! section, or downward into the unmapped gap below it.

use anyhow::{bail, Context, Result};

pub const SHF_ALLOC: u64 = 2;
pub const SHT_NULL: u32 = 0;
pub const SHT_NOBITS: u32 = 8;
pub const PT_NULL: u32 = 0;
pub const PT_LOAD: u32 = 1;
pub const PHDR_SIZE: usize = 56;
pub const PAGE: u64 = 0x1000;
pub const SYM_ENTRY: usize = 24;

#[derive(Clone, Debug)]
pub struct Section {
    pub index: usize,
    pub hdr_off: usize,
    pub name: String,
    pub kind: u32,
    pub flags: u64,
    pub addr: u64,
    pub offset: u64,
    pub size: u64,
    pub link: u32,
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub hdr_off: usize,
    pub kind: u32,
    pub offset: u64,
    pub vaddr: u64,
    pub filesz: u64,
    pub memsz: u64,
}

#[derive(Clone, Debug)]
pub struct Symbol {
    pub name: String,
    pub entry_off: usize,
    pub value: u64,
    pub size: u64,
}

#[derive(Clone)]
pub struct Elf {
    pub data: Vec<u8>,
    pub e_phoff: u64,
    pub e_shoff: u64,
    pub sections: Vec<Section>,
    pub segments: Vec<Segment>,
}

pub fn read_u16(d: &[u8], off: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        d.get(off..off + 2)
            .context("ELF read out of bounds")?
            .try_into()?,
    ))
}

pub fn read_u32(d: &[u8], off: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        d.get(off..off + 4)
            .context("ELF read out of bounds")?
            .try_into()?,
    ))
}

pub fn read_u64(d: &[u8], off: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        d.get(off..off + 8)
            .context("ELF read out of bounds")?
            .try_into()?,
    ))
}

pub fn write_u16(d: &mut [u8], off: usize, v: u16) -> Result<()> {
    d.get_mut(off..off + 2)
        .context("ELF write out of bounds")?
        .copy_from_slice(&v.to_le_bytes());
    Ok(())
}

pub fn write_u32(d: &mut [u8], off: usize, v: u32) -> Result<()> {
    d.get_mut(off..off + 4)
        .context("ELF write out of bounds")?
        .copy_from_slice(&v.to_le_bytes());
    Ok(())
}

pub fn write_u64(d: &mut [u8], off: usize, v: u64) -> Result<()> {
    d.get_mut(off..off + 8)
        .context("ELF write out of bounds")?
        .copy_from_slice(&v.to_le_bytes());
    Ok(())
}

fn cstr(d: &[u8], off: usize) -> Result<String> {
    let tail = d.get(off..).context("string table offset out of bounds")?;
    let end = tail
        .iter()
        .position(|&c| c == 0)
        .context("unterminated ELF string")?;
    Ok(String::from_utf8_lossy(&tail[..end]).into_owned())
}

impl Elf {
    pub fn parse(data: Vec<u8>) -> Result<Self> {
        if data.len() < 64 || &data[..4] != b"\x7fELF" || data[4] != 2 || data[5] != 1 {
            bail!("not a little-endian ELF64 object");
        }
        let e_phoff = read_u64(&data, 0x20)?;
        let e_shoff = read_u64(&data, 0x28)?;
        let e_phentsize = read_u16(&data, 0x36)? as usize;
        let e_phnum = read_u16(&data, 0x38)? as usize;
        let e_shentsize = read_u16(&data, 0x3a)? as usize;
        let e_shnum = read_u16(&data, 0x3c)? as usize;
        let e_shstrndx = read_u16(&data, 0x3e)? as usize;
        let mut segments = Vec::with_capacity(e_phnum);
        for i in 0..e_phnum {
            let h = e_phoff as usize + i * e_phentsize;
            segments.push(Segment {
                hdr_off: h,
                kind: read_u32(&data, h)?,
                offset: read_u64(&data, h + 0x08)?,
                vaddr: read_u64(&data, h + 0x10)?,
                filesz: read_u64(&data, h + 0x20)?,
                memsz: read_u64(&data, h + 0x28)?,
            });
        }
        let mut raw = Vec::with_capacity(e_shnum);
        for i in 0..e_shnum {
            let h = e_shoff as usize + i * e_shentsize;
            raw.push((
                h,
                read_u32(&data, h)?,
                Section {
                    index: i,
                    hdr_off: h,
                    name: String::new(),
                    kind: read_u32(&data, h + 0x04)?,
                    flags: read_u64(&data, h + 0x08)?,
                    addr: read_u64(&data, h + 0x10)?,
                    offset: read_u64(&data, h + 0x18)?,
                    size: read_u64(&data, h + 0x20)?,
                    link: read_u32(&data, h + 0x28)?,
                },
            ));
        }
        let shstr_off = raw.get(e_shstrndx).context("bad e_shstrndx")?.2.offset as usize;
        let mut sections = Vec::with_capacity(e_shnum);
        for (_, name_off, mut s) in raw {
            s.name = cstr(&data, shstr_off + name_off as usize)?;
            sections.push(s);
        }
        Ok(Elf {
            data,
            e_phoff,
            e_shoff,
            sections,
            segments,
        })
    }

    pub fn section(&self, name: &str) -> Result<&Section> {
        let mut found = self.sections.iter().filter(|s| s.name == name);
        let s = found.next().with_context(|| format!("no {name} section"))?;
        if found.next().is_some() {
            bail!("more than one {name} section");
        }
        Ok(s)
    }

    /// Entries of `.symtab` or `.dynsym` (empty if the table is absent).
    pub fn symbols(&self, table: &str) -> Result<Vec<Symbol>> {
        let Some(sec) = self.sections.iter().find(|s| s.name == table) else {
            return Ok(Vec::new());
        };
        let strtab = self
            .sections
            .get(sec.link as usize)
            .context("symbol table has no string table")?;
        let mut out = Vec::new();
        for i in 0..(sec.size as usize / SYM_ENTRY) {
            let off = sec.offset as usize + i * SYM_ENTRY;
            out.push(Symbol {
                name: cstr(
                    &self.data,
                    strtab.offset as usize + read_u32(&self.data, off)? as usize,
                )?,
                entry_off: off,
                value: read_u64(&self.data, off + 8)?,
                size: read_u64(&self.data, off + 16)?,
            });
        }
        Ok(out)
    }

    pub fn symbol(&self, name: &str) -> Result<Symbol> {
        let mut found = self
            .symbols(".symtab")?
            .into_iter()
            .filter(|s| s.name == name);
        let s = found
            .next()
            .with_context(|| format!("symbol {name} not found in .symtab"))?;
        if found.next().is_some() {
            bail!("symbol {name} is defined more than once");
        }
        Ok(s)
    }

    /// Kernels (`<name>.kd` symbols) in `.symtab`.
    pub fn kernels(&self) -> Result<Vec<String>> {
        let mut out: Vec<String> = self
            .symbols(".symtab")?
            .into_iter()
            .filter_map(|s| s.name.strip_suffix(".kd").map(str::to_string))
            .collect();
        out.sort();
        out.dedup();
        Ok(out)
    }

    pub fn vaddr_to_offset(&self, vaddr: u64) -> Result<usize> {
        self.sections
            .iter()
            .find(|s| {
                s.kind != SHT_NOBITS && s.addr != 0 && s.addr <= vaddr && vaddr < s.addr + s.size
            })
            .map(|s| (s.offset + vaddr - s.addr) as usize)
            .with_context(|| format!("vaddr {vaddr:#x} is not in any file-backed section"))
    }
}

/// First virtual address past `.text` that is taken by another allocated section or PT_LOAD segment.
pub fn text_limit(elf: &Elf) -> Result<u64> {
    let text = elf.section(".text")?;
    let sections = elf
        .sections
        .iter()
        .filter(|s| s.flags & SHF_ALLOC != 0 && s.addr > text.addr)
        .map(|s| s.addr);
    let segments = elf
        .segments
        .iter()
        .filter(|p| p.kind == PT_LOAD && p.vaddr > text.addr)
        .map(|p| p.vaddr);
    Ok(sections.chain(segments).min().unwrap_or(u64::MAX))
}

/// `.text` grown downward: `code` placed `span` bytes below `.text` (`span` = `code.len()` rounded up to 256), with
/// `fill` words up to the original start, so the prologue directly precedes the body. File offsets from `.text` on
/// move by whole pages (the inserted page holds the prologue); no existing virtual address changes. Used when no slack
/// follows `.text`; the kernel symbol can then cover prologue and body as one contiguous range, which the ATT decoder
/// needs (it disassembles whole `.dynsym` FUNC symbols).
pub fn grow_text_down(elf: &Elf, code: &[u8], fill: u32) -> Result<(Elf, u64)> {
    let text = elf.section(".text")?.clone();
    let span = (code.len() as u64).div_ceil(256) * 256;
    if text.addr % 256 != 0 || code.len() % 4 != 0 || span > text.addr {
        bail!("cannot place the prologue below .text at {:#x}", text.addr);
    }
    let place = text.addr - span;
    let below = elf
        .sections
        .iter()
        .filter(|s| s.flags & SHF_ALLOC != 0 && s.addr < text.addr && s.index != text.index)
        .map(|s| s.addr + s.size)
        .chain(
            elf.segments
                .iter()
                .filter(|p| p.kind == PT_LOAD && p.vaddr < text.addr)
                .map(|p| p.vaddr + p.memsz),
        )
        .max()
        .unwrap_or(0);
    if place < below.div_ceil(PAGE) * PAGE {
        bail!(
            "no room for the prologue after .text (next allocation) or before it: {span:#x} bytes below {:#x} reach the page of {below:#x}",
            text.addr
        );
    }
    let cut = text.offset;
    let segs: Vec<&Segment> = elf
        .segments
        .iter()
        .filter(|p| p.kind == PT_LOAD && p.offset <= cut && cut < p.offset + p.filesz)
        .collect();
    let [seg] = segs.as_slice() else {
        bail!(".text must lie in exactly one PT_LOAD segment");
    };
    if seg.offset != cut || seg.vaddr != text.addr {
        bail!(".text does not start its PT_LOAD segment");
    }
    for s in &elf.sections {
        if s.kind != SHT_NOBITS && s.offset < cut && cut < s.offset + s.size {
            bail!("section {} spans the start of .text", s.name);
        }
    }
    for p in &elf.segments {
        if p.kind != PT_NULL
            && !std::ptr::eq(p, *seg)
            && p.offset < cut
            && cut < p.offset + p.filesz
        {
            bail!("a segment of type {} spans the start of .text", p.kind);
        }
    }
    let phdr_end = elf.e_phoff + (elf.segments.len() * PHDR_SIZE) as u64;
    if elf.e_phoff < cut && cut < phdr_end {
        bail!("program header table spans the start of .text");
    }

    let delta = span.div_ceil(PAGE) * PAGE;
    let mut data = Vec::with_capacity(elf.data.len() + delta as usize);
    data.extend_from_slice(&elf.data[..cut as usize]);
    data.resize((cut + delta - span) as usize, 0);
    data.extend_from_slice(code);
    data.extend(
        fill.to_le_bytes()
            .iter()
            .copied()
            .cycle()
            .take(span as usize - code.len()),
    );
    data.extend_from_slice(&elf.data[cut as usize..]);

    let moved = |off: u64| if off >= cut { off + delta } else { off };
    let (e_shoff, e_phoff) = (moved(elf.e_shoff), moved(elf.e_phoff));
    write_u64(&mut data, 0x28, e_shoff)?;
    write_u64(&mut data, 0x20, e_phoff)?;
    for s in &elf.sections {
        let h = (e_shoff + (s.hdr_off as u64 - elf.e_shoff)) as usize;
        if s.index == text.index {
            write_u64(&mut data, h + 0x10, place)?;
            write_u64(&mut data, h + 0x18, cut + delta - span)?;
            write_u64(&mut data, h + 0x20, text.size + span)?;
        } else if s.kind != SHT_NULL && s.offset >= cut {
            write_u64(&mut data, h + 0x18, s.offset + delta)?;
        }
    }
    for p in &elf.segments {
        let h = (e_phoff + (p.hdr_off as u64 - elf.e_phoff)) as usize;
        if std::ptr::eq(p, *seg) {
            write_u64(&mut data, h + 0x08, cut + delta - span)?;
            write_u64(&mut data, h + 0x10, place)?;
            write_u64(&mut data, h + 0x18, place)?;
            write_u64(&mut data, h + 0x20, p.filesz + span)?;
            write_u64(&mut data, h + 0x28, p.memsz + span)?;
        } else if p.kind != PT_NULL && p.offset >= cut {
            write_u64(&mut data, h + 0x08, p.offset + delta)?;
        }
    }
    Ok((Elf::parse(data)?, place))
}

/// `.text` grown in place: `code` placed at `place_vaddr` (past the end of `.text`), framed by `fill` words, with
/// `tail` fill bytes after it. File offsets after `.text` move by whole pages so every later segment keeps its
/// offset/address congruence; no virtual address changes.
pub struct Grown {
    pub elf: Elf,
    pub growth: u64,
    pub file_delta: u64,
}

pub fn grow_text(elf: &Elf, place_vaddr: u64, code: &[u8], tail: u64, fill: u32) -> Result<Grown> {
    let text = elf.section(".text")?.clone();
    let text_end = text.addr + text.size;
    let text_end_file = text.offset + text.size;
    if place_vaddr < text_end || (place_vaddr - text_end) % 4 != 0 || tail % 4 != 0 {
        bail!("prologue placement {place_vaddr:#x} is not a dword-aligned address past .text end {text_end:#x}");
    }
    let padded_end = place_vaddr + code.len() as u64 + tail;
    let limit = elf
        .sections
        .iter()
        .filter(|s| s.flags & SHF_ALLOC != 0 && s.addr > text.addr)
        .map(|s| s.addr)
        .min()
        .unwrap_or(u64::MAX);
    if padded_end > limit {
        bail!(
            "prologue would end at {padded_end:#x}, past the next allocated section at {limit:#x}"
        );
    }
    let text_segments: Vec<&Segment> = elf
        .segments
        .iter()
        .filter(|p| {
            p.kind == PT_LOAD && p.offset <= text.offset && text_end_file <= p.offset + p.filesz
        })
        .collect();
    let [seg] = text_segments.as_slice() else {
        bail!(
            ".text must lie in exactly one PT_LOAD segment, found {}",
            text_segments.len()
        );
    };
    if text.addr.checked_sub(seg.vaddr) != text.offset.checked_sub(seg.offset) {
        bail!(".text virtual address does not match its PT_LOAD mapping");
    }
    if elf
        .segments
        .iter()
        .any(|p| p.kind == PT_LOAD && p.vaddr > seg.vaddr && padded_end > p.vaddr)
    {
        bail!("prologue overlaps the next PT_LOAD virtual address");
    }
    if seg.offset + seg.filesz != text_end_file || seg.memsz != seg.filesz {
        bail!(".text is not the last file-backed content of its PT_LOAD segment");
    }
    for p in &elf.segments {
        let straddles = p.offset < text_end_file && text_end_file < p.offset + p.filesz;
        if p.kind != PT_NULL && straddles && !std::ptr::eq(p, *seg) {
            bail!("a segment of type {} spans the end of .text", p.kind);
        }
    }
    if elf.e_phoff < text_end_file && elf.e_phoff + (elf.segments.len() as u64) * 56 > text_end_file
    {
        bail!("program header table spans the end of .text");
    }

    let growth = padded_end - text_end;
    let delta = growth.div_ceil(PAGE) * PAGE;
    let fill_bytes = |n: u64| {
        fill.to_le_bytes()
            .iter()
            .copied()
            .cycle()
            .take(n as usize)
            .collect::<Vec<u8>>()
    };
    let mut inserted = fill_bytes(place_vaddr - text_end);
    inserted.extend_from_slice(code);
    inserted.extend(fill_bytes(tail));
    let cut = text_end_file as usize;
    let mut data = Vec::with_capacity(elf.data.len() + delta as usize);
    data.extend_from_slice(&elf.data[..cut]);
    data.extend_from_slice(&inserted);
    data.resize(cut + delta as usize, 0);
    data.extend_from_slice(&elf.data[cut..]);

    let moved = |off: u64| {
        if off >= text_end_file {
            off + delta
        } else {
            off
        }
    };
    let e_shoff = moved(elf.e_shoff);
    let e_phoff = moved(elf.e_phoff);
    write_u64(&mut data, 0x28, e_shoff)?;
    write_u64(&mut data, 0x20, e_phoff)?;
    for s in &elf.sections {
        let h = (e_shoff + (s.hdr_off as u64 - elf.e_shoff)) as usize;
        if s.index == text.index {
            write_u64(&mut data, h + 0x20, text.size + growth)?;
        } else if s.kind != SHT_NULL && s.offset >= text_end_file {
            write_u64(&mut data, h + 0x18, s.offset + delta)?;
        }
    }
    for p in &elf.segments {
        let h = (e_phoff + (p.hdr_off as u64 - elf.e_phoff)) as usize;
        if std::ptr::eq(p, *seg) {
            write_u64(&mut data, h + 0x20, p.filesz + growth)?;
            write_u64(&mut data, h + 0x28, p.memsz + growth)?;
        } else if p.kind != PT_NULL && p.offset >= text_end_file {
            write_u64(&mut data, h + 0x08, p.offset + delta)?;
        }
    }
    Ok(Grown {
        elf: Elf::parse(data)?,
        growth,
        file_delta: delta,
    })
}
