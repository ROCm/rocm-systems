#!/usr/bin/env python3
# Post-build safety check for the .hip_fatbin tail-placement linker script.
#
# What it proves
# --------------
# librccl.so embeds a large (eventually >2 GiB) device fat binary in
# .hip_fatbin.  Host code is compiled in the small code model, so every
# host->host reference uses a *signed 32-bit* relocation (PC32 / PLT32 /
# GOTPCRELX / 32S) whose displacement must fit in [-2^31, 2^31 - 1] (+/- 2 GiB).
#
# By design nothing takes a range-sensitive 32-bit reference *to* .hip_fatbin:
# the HIP runtime reaches the blob only through the wrapper in
# .hipFatBinSegment, which holds a full 64-bit pointer (R_X86_64_64 /
# R_X86_64_RELATIVE).  That makes .hip_fatbin an inert spacer -- it is never
# the site or the target of a 32-bit relocation.
#
# Given that, the only way a 32-bit host relocation can overflow is if two
# *host* sections (everything except .hip_fatbin) end up more than 2 GiB
# apart.  So the definitive, linker-independent safety condition is:
#
#     span(all SHF_ALLOC sections except .hip_fatbin) <= INT32_MAX
#
# where span = max(sh_addr + sh_size) - min(sh_addr).  If that holds, *every*
# possible 32-bit host relocation is in range, no matter how large the fatbin
# grows.  Placing .hip_fatbin at the tail is simply the mechanism that keeps
# this host window small; this check verifies the property directly instead of
# inspecting section order, and it catches the case where the linker script
# silently failed to move the blob (the window then straddles the fatbin and
# blows past 2 GiB).
#
# The section table is parsed straight out of the ELF (no readelf text
# scraping, no third-party modules).
#
# Exit status:
#   0  verified safe (host window <= 2 GiB), or the check does not apply
#      (non-x86-64 host, or a 32-bit object).
#   1  host window exceeds 2 GiB (the relocation-range guarantee is lost), or
#      the library could not be read/parsed as ELF.
#   2  bad usage.
#
# NOTE: correctness rests on the "nothing 32-bit-references .hip_fatbin"
# property above.  If a future toolchain/runtime takes a PC-relative or other
# 32-bit reference to __hip_fatbin, tail placement would itself create the
# overflow and excluding .hip_fatbin from the span would be invalid.
#
# Usage: check_hip_fatbin_tail.py <path-to-librccl.so>

import os
import struct
import sys

PREFIX = "[rccl] hip_fatbin tail check:"

SHF_ALLOC = 0x2
SHF_TLS = 0x400
EM_X86_64 = 62
ELFCLASS64 = 2
ELFDATA2LSB = 1
SHN_XINDEX = 0xFFFF
INT32_MAX = (1 << 31) - 1
MIB = 1024.0 * 1024.0
FATBIN = ".hip_fatbin"


class ElfError(Exception):
    pass


def parse_elf(f):
    """Parse an ELF64 section table from an open binary file.

    Returns (sections, machine) where sections is a list of dicts with keys
    name/type/flags/addr/size/offset, or (None, machine) for a non-ELF64
    object the check does not apply to.  Raises ElfError on a malformed ELF.

    Only the ELF header, the section header table, and the section-name string
    table are read (a few KiB), so cost is independent of the file size -- the
    multi-hundred-MB .hip_fatbin payload is never touched.
    """
    def read_at(off, n, what):
        f.seek(off)
        b = f.read(n)
        if len(b) != n:
            raise ElfError(f"short read for {what}")
        return b

    ehdr = read_at(0, 64, "ELF header")
    if ehdr[:4] != b"\x7fELF":
        raise ElfError("not an ELF file")
    if ehdr[4] != ELFCLASS64:
        return None, None  # 32-bit object: the >2 GiB concern does not apply
    endian = "<" if ehdr[5] == ELFDATA2LSB else ">"

    # Elf64_Ehdr, skipping the 16-byte e_ident.
    (e_type, e_machine, e_version, e_entry, e_phoff, e_shoff, e_flags,
     e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum,
     e_shstrndx) = struct.unpack_from(endian + "HHIQQQIHHHHHH", ehdr, 16)

    if e_shoff == 0 or e_shentsize < 64:
        raise ElfError("no usable section header table")

    def parse_shdr(raw):
        (sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size,
         sh_link, sh_info, sh_addralign, sh_entsize) = struct.unpack_from(
            endian + "IIQQQQIIQQ", raw, 0)
        return dict(name_off=sh_name, type=sh_type, flags=sh_flags,
                    addr=sh_addr, offset=sh_offset, size=sh_size,
                    link=sh_link)

    # Section 0 carries the real section count / shstrtab index when they do
    # not fit the 16-bit header fields.
    sec0 = parse_shdr(read_at(e_shoff, e_shentsize, "section header 0"))
    shnum = e_shnum if e_shnum != 0 else sec0["size"]
    shstrndx = sec0["link"] if e_shstrndx == SHN_XINDEX else e_shstrndx
    if shnum == 0:
        raise ElfError("zero sections")

    # Read the whole section header table in one go (shnum * e_shentsize bytes).
    sht = read_at(e_shoff, shnum * e_shentsize, "section header table")
    headers = [parse_shdr(sht[i * e_shentsize:(i + 1) * e_shentsize])
               for i in range(shnum)]

    if shstrndx >= shnum:
        raise ElfError("bad shstrndx")
    strtab = headers[shstrndx]
    strblob = read_at(strtab["offset"], strtab["size"], "section name table")

    def name_at(name_off):
        end = strblob.find(b"\x00", name_off)
        if end < 0:
            end = len(strblob)
        return strblob[name_off:end].decode("utf-8", errors="replace")

    for s in headers:
        s["name"] = name_at(s["name_off"])
    return headers, e_machine


def main(argv):
    if len(argv) != 2:
        print("usage: check_hip_fatbin_tail.py <librccl.so>", file=sys.stderr)
        return 2

    so = argv[1]
    if not os.path.exists(so):
        print(f"{PREFIX} FAIL ({so} not found)", file=sys.stderr)
        return 1

    try:
        with open(so, "rb") as f:
            sections, machine = parse_elf(f)
    except (OSError, ElfError, struct.error) as e:
        print(f"{PREFIX} FAIL (could not parse {so}: {e})", file=sys.stderr)
        return 1

    if sections is None:
        print(f"{PREFIX} SKIP (not a 64-bit ELF; check is x86-64 specific)")
        return 0
    if machine != EM_X86_64:
        print(f"{PREFIX} SKIP (e_machine={machine}; 32-bit-reloc concern is "
              f"x86-64 specific)")
        return 0

    # Host sections that can be the site or target of a 32-bit relocation:
    # everything allocated, minus TLS (different reloc class) and minus the
    # inert .hip_fatbin spacer.
    host = [s for s in sections
            if (s["flags"] & SHF_ALLOC)
            and not (s["flags"] & SHF_TLS)
            and s["name"] != FATBIN]
    if not host:
        print(f"{PREFIX} FAIL (no allocated host sections found)",
              file=sys.stderr)
        return 1

    lo = min(host, key=lambda s: s["addr"])
    hi = max(host, key=lambda s: s["addr"] + s["size"])
    span = (hi["addr"] + hi["size"]) - lo["addr"]

    fatbins = [s for s in sections if s["name"] == FATBIN]
    if fatbins:
        fb = max(fatbins, key=lambda s: s["size"])
        alloc = [s for s in sections
                 if (s["flags"] & SHF_ALLOC) and not (s["flags"] & SHF_TLS)]
        highest = max(alloc, key=lambda s: s["addr"] + s["size"])
        at_tail = highest["name"] == FATBIN
        fb_desc = (f".hip_fatbin {fb['size'] / MIB:.1f} MiB at file offset "
                   f"0x{fb['offset']:x}, "
                   f"{'at image tail' if at_tail else 'NOT at image tail'}")
    else:
        fb_desc = ".hip_fatbin not present (nothing to displace host sections)"

    if span <= INT32_MAX:
        margin = INT32_MAX - span
        print(f"{PREFIX} OK (host window {span / MIB:.1f} MiB <= 2 GiB, "
              f"margin {margin / MIB:.1f} MiB; {fb_desc})")
        return 0

    over = span - INT32_MAX
    print(f"{PREFIX} FAIL: host window is {span / MIB:.1f} MiB "
          f"(> 2 GiB by {over / MIB:.1f} MiB).", file=sys.stderr)
    print(f"  Lowest host section : {lo['name']} @ 0x{lo['addr']:x}",
          file=sys.stderr)
    print(f"  Highest host section: {hi['name']} @ "
          f"0x{hi['addr'] + hi['size']:x}", file=sys.stderr)
    print(f"  {fb_desc}", file=sys.stderr)
    print("  Small-code-model host relocations (PC32/PLT32/GOTPCRELX/32S) can "
          "overflow.", file=sys.stderr)
    print("  Check that the linker honoured "
          "cmake/linker_scripts/hip_fatbin_tail.ld and moved .hip_fatbin to "
          "the tail.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
