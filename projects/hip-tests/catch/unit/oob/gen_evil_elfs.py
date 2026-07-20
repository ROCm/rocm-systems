#!/usr/bin/env python3
"""gen_evil_elfs.py <bundle_or_elf.co> <outdir>

Accepts a clang offload bundle OR a raw ELF. Extracts the first AMDGPU ELF
entry (so arch is always whatever was just compiled), then writes malformed
copies that exercise the getElfSize rejection paths.
"""


"""
This script corrupts the Elf64_Ehdr of a module.
It generates 5 outputs
elf_valid.co - exactly the same as input
elf_huge_shnum.co - corrupt the e_shnum, we read past the buffer
elf_bad_shoff.co - corrupt the e_shoff, section table start is past end of file
elf_table_spill.co - corrupts e_shnum and e_shoff, table starts inbounds but end past file
elf_sh_overflow.co - corrupts sh_offset + sh_size, total size is too big
"""
import struct, sys, os

def extract_elf(data):
    if data[:4] == b'\x7fELF':
        return data
    assert data[:24] == b'__CLANG_OFFLOAD_BUNDLE__', "unrecognised format"
    off = 24
    num, = struct.unpack_from('<Q', data, off); off += 8
    for _ in range(num):
        o, s, idl = struct.unpack_from('<QQQ', data, off); off += 24
        bid = data[off:off+idl].decode(); off += idl
        if s > 0 and data[o:o+4] == b'\x7fELF':
            return data[o:o+s]
    raise ValueError("no ELF entry in bundle")

def u16(b, o): return struct.unpack_from('<H', b, o)[0]
def u64(b, o): return struct.unpack_from('<Q', b, o)[0]
def w16(b, o, v): struct.pack_into('<H', b, o, v)
def w64(b, o, v): struct.pack_into('<Q', b, o, v)

elf   = extract_elf(open(sys.argv[1], 'rb').read())
out   = sys.argv[2]
os.makedirs(out, exist_ok=True)

assert elf[4] == 2, "ELFCLASS64 only"
shoff     = u64(elf, 40)
shentsize = u16(elf, 58)
shnum     = u16(elf, 60)
size      = len(elf)

def emit(name, b):
    p = os.path.join(out, name)
    open(p, 'wb').write(bytes(b))
    print(f"  {p}")

emit('elf_valid.co',       elf)

b = bytearray(elf); w16(b, 60, 0xFFFF)
emit('elf_huge_shnum.co',  b)

b = bytearray(elf); w64(b, 40, 1 << 40)
emit('elf_bad_shoff.co',   b)

b = bytearray(elf); w64(b, 40, size - 10); w16(b, 60, 1)
emit('elf_table_spill.co', b)

b = bytearray(elf)
s5 = shoff + 5 * shentsize
struct.pack_into('<I', b, s5 + 4, 1)          # sh_type = SHT_PROGBITS
w64(b, s5 + 24, 0xFFFFFFFFFFFFFFFF)           # sh_offset -> overflow on +sh_size
w64(b, s5 + 32, 0x10)
emit('elf_sh_overflow.co', b)
