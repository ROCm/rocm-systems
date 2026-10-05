#!/usr/bin/env python3
"""Generate checked GFX1201 descriptors and text for device-producer.c."""

import argparse
import hashlib
from pathlib import Path
import struct
import subprocess
import tempfile


def elf_parts(path: Path, symbol: str) -> tuple[bytes, bytes, int]:
    data = path.read_bytes()
    if data[:9] != b"\x7fELF\x02\x01\x01\x40\x02":
        raise ValueError("expected an ELF64 little-endian AMDGPU HSA v4 object")
    if struct.unpack_from("<H", data, 18)[0] != 0xE0:
        raise ValueError("expected EM_AMDGPU")
    if struct.unpack_from("<I", data, 48)[0] != 0x4E:
        raise ValueError("expected the gfx1201 target flag")
    section_offset = struct.unpack_from("<Q", data, 40)[0]
    section_size, section_count, name_index = struct.unpack_from("<HHH", data, 58)
    if section_size != 64 or section_offset + section_count * section_size > len(data):
        raise ValueError("invalid ELF section table")
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, section_offset + index * section_size)
        for index in range(section_count)
    ]
    if name_index >= section_count:
        raise ValueError("invalid section-name table")

    def contents(section: tuple[int, ...]) -> bytes:
        start, size = section[4:6]
        if start + size > len(data):
            raise ValueError("section exceeds ELF file")
        return data[start : start + size]

    names = contents(sections[name_index])

    def name(offset: int) -> str:
        end = names.find(b"\0", offset)
        if end < 0:
            raise ValueError("unterminated ELF section name")
        return names[offset:end].decode("ascii")

    by_name = {name(section[0]): (index, section) for index, section in enumerate(sections)}
    for required in (".text", ".rodata", ".rela.rodata", ".symtab"):
        if required not in by_name:
            raise ValueError(f"missing {required} section")
    if [name(section[0]) for section in sections if section[1] == 4] != [".rela.rodata"]:
        raise ValueError("unexpected code-object relocations")
    text_index, text_section = by_name[".text"]
    rodata_index, rodata_section = by_name[".rodata"]
    descriptor = contents(rodata_section)
    text = contents(text_section)
    if len(descriptor) != 64 or text_section[8] < 256:
        raise ValueError("unexpected kernel descriptor or text alignment")
    private_bytes = struct.unpack_from("<I", descriptor, 4)[0]
    if private_bytes != 0 or descriptor[16:24] != bytes(8):
        raise ValueError("kernel unexpectedly requires scratch or has an entry offset")

    _, symbol_section = by_name[".symtab"]
    if symbol_section[6] >= section_count or symbol_section[9] != 24:
        raise ValueError("invalid symbol table")
    strings = contents(sections[symbol_section[6]])
    symbols = {}
    raw_symbols = contents(symbol_section)
    for index in range(len(raw_symbols) // 24):
        offset, info, _other, section, value, size = struct.unpack_from(
            "<IBBHQQ", raw_symbols, index * 24
        )
        end = strings.find(b"\0", offset)
        if end < 0:
            raise ValueError("invalid symbol name")
        symbols[strings[offset:end].decode("ascii")] = (index, info, section, value, size)
    kernel = symbols.get(symbol)
    descriptor_symbol = symbols.get(f"{symbol}.kd")
    if kernel is None or descriptor_symbol is None:
        raise ValueError("missing kernel or descriptor symbol")
    if kernel[1] & 15 != 2 or kernel[2] != text_index or kernel[3] != 0 or kernel[4] == 0:
        raise ValueError("kernel must begin at the text section start")
    if descriptor_symbol[1] & 15 != 1 or descriptor_symbol[2:] != (rodata_index, 0, 64):
        raise ValueError("descriptor symbol does not cover .rodata")
    if kernel[4] > len(text) or kernel[4] > 4096 - 256:
        raise ValueError("kernel text exceeds the consumer's code allocation")

    _, reloc_section = by_name[".rela.rodata"]
    relocs = contents(reloc_section)
    if reloc_section[6] != by_name[".symtab"][0] or reloc_section[7] != rodata_index or len(relocs) != 24:
        raise ValueError("expected one descriptor-entry relocation")
    offset, info, addend = struct.unpack_from("<QQq", relocs)
    if (offset, info >> 32, info & 0xFFFFFFFF, addend) != (16, kernel[0], 5, 16):
        raise ValueError("unexpected descriptor relocation")
    return descriptor, text[: kernel[4]], private_bytes


def render_array(name: str, data: bytes) -> str:
    lines = [f"static const unsigned char {name}[{len(data)}] = {{"]
    for offset in range(0, len(data), 12):
        lines.append("  " + ", ".join(f"0x{byte:02x}" for byte in data[offset : offset + 12]) + ",")
    return "\n".join(lines + ["};"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", choices=("publisher", "target"))
    parser.add_argument("output", type=Path, help="path to generated C include")
    parser.add_argument("--llvm-bin", type=Path, default=Path("/opt/rocm/llvm/bin"))
    args = parser.parse_args()
    source = Path(__file__).with_name(f"device-producer-{args.kernel}.cl")
    symbol = "amdf_publish_queue" if args.kernel == "publisher" else "amdf_device_target"
    prefix = "producer_kernel" if args.kernel == "publisher" else "target_kernel"
    compiler = args.llvm_bin / "clang"
    with tempfile.TemporaryDirectory(prefix="amdf-device-producer-") as temporary:
        obj = Path(temporary) / f"{args.kernel}.o"
        subprocess.run(
            [str(compiler), "-x", "cl", "--target=amdgcn-amd-amdhsa", "-mcpu=gfx1201",
             "-mcode-object-version=4", "-O2", "-c", str(source), "-o", str(obj)],
            check=True,
        )
        descriptor, text, private_bytes = elf_parts(obj, symbol)
        digest = hashlib.sha256(obj.read_bytes()).hexdigest()
    output = (
        f"/* GFX1201 code-object v4; source object SHA-256: {digest}.\n"
        " * The consumer replaces descriptor bytes 16..23 with its code offset. */\n"
        + render_array(f"{prefix}_descriptor", descriptor) + "\n\n"
        + render_array(f"{prefix}_text", text) + "\n"
    )
    args.output.write_text(output)
    print(f"{args.output}: {len(text)} text bytes, {private_bytes} private bytes")


if __name__ == "__main__":
    main()
