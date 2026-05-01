#!/usr/bin/env bash
# Dump host (x86-64) disassembly from librccl.so with line numbers and source
# interleaving where DWARF exists, and extract / disassemble embedded AMDGPU
# device code (ELF64 amdgcn images embedded in the host .so).
#
# Device handling:
#   Scans the host library for ELF64 EM_AMDGPU (0xe0) headers and slices out
#   each complete embedded object using program-header file ranges. This works
#   for typical RCCL HIP-linked binaries on ROCm 6.x. On ROCm 7+, you may also
#   use llvm-objcopy --dump-offload-bundle='<URI from llvm-readobj --offloading>'
#   manually for compressed bundles; this script does not invoke that yet when
#   no bare AMDGPU ELF is present in the file.
#
# Usage:
#   ./dump_librccl_asm.sh [PATH_TO_LIBRCCL] [OUTPUT_DIR]
#
# Writes:
#   OUTPUT_DIR/host_asm.s
#   OUTPUT_DIR/offload_list.txt          (llvm-readobj --offloading)
#   OUTPUT_DIR/device/amdgcn_N.elf       (extracted code objects)
#   OUTPUT_DIR/device/amdgcn_N.mcpu.txt  (gfx used for llvm-objdump --mcpu)
#   OUTPUT_DIR/device/amdgcn_N.s         (ISA disassembly)
#
# Environment:
#   ROCM_PATH     default: /opt/rocm
#   LLVM_OBJDUMP  override llvm-objdump path
#   LLVM_READOBJ  override llvm-readobj path

set -euo pipefail

usage() {
  echo "Usage: $0 [PATH_TO_LIBRCCL] [OUTPUT_DIR]" >&2
  echo "  PATH_TO_LIBRCCL  default: ../../build/release/librccl.so.1.0 (from tools/scripts)" >&2
  echo "  OUTPUT_DIR       default: ./librccl_asm_dump" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LIB="${SCRIPT_DIR}/../../build/release/librccl.so.1.0"
LIB="${1:-$DEFAULT_LIB}"
OUTDIR="${2:-${PWD}/librccl_asm_dump}"

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
if [[ -n "${LLVM_OBJDUMP:-}" ]]; then
  OBJDUMP="$LLVM_OBJDUMP"
else
  OBJDUMP="${ROCM_PATH}/llvm/bin/llvm-objdump"
fi
if [[ -n "${LLVM_READOBJ:-}" ]]; then
  READOBJ="$LLVM_READOBJ"
else
  READOBJ="${ROCM_PATH}/llvm/bin/llvm-readobj"
fi

for t in "$OBJDUMP" "$READOBJ"; do
  if [[ ! -x "$t" ]]; then
    echo "error: required tool not executable: $t" >&2
    echo "Set ROCM_PATH or LLVM_* overrides." >&2
    exit 1
  fi
done

if [[ ! -f "$LIB" ]]; then
  echo "error: library not found: $LIB" >&2
  exit 1
fi

LIB_REAL="$(readlink -f "$LIB")"
mkdir -p "$OUTDIR/device"

echo "Using llvm-objdump:  $OBJDUMP"
echo "Using llvm-readobj:  $READOBJ"
echo "Input library:       $LIB_REAL"
echo "Output directory:    $OUTDIR"

HOST_OUT="${OUTDIR}/host_asm.s"
OFFLOAD_TXT="${OUTDIR}/offload_list.txt"

"$READOBJ" --offloading "$LIB_REAL" >"$OFFLOAD_TXT" 2>&1 || true
echo "Wrote offload listing: $OFFLOAD_TXT"

"$OBJDUMP" \
  -d -l --source --demangle --no-show-raw-insn \
  "$LIB_REAL" >"$HOST_OUT"

verify_host() {
  local f="$1"
  local err=0

  if [[ ! -s "$f" ]]; then
    echo "HOST VERIFY FAIL: output missing or empty: $f" >&2
    return 1
  fi

  if ! grep -q "Disassembly of section" "$f"; then
    echo "HOST VERIFY FAIL: no disassembly section headers in $f" >&2
    err=1
  fi

  local insns
  insns="$(grep -cE '^[[:space:]]*[0-9a-f]+:' "$f" || true)"
  if [[ "${insns:-0}" -lt 500 ]]; then
    echo "HOST VERIFY FAIL: too few instruction lines ($insns)" >&2
    err=1
  fi

  if [[ "$err" -ne 0 ]]; then
    return 1
  fi

  echo "HOST VERIFY OK: $(wc -l <"$f") lines, $insns instruction lines"
  if grep -qE '\.(c|cc|cpp|cxx|h|hpp):[0-9]+' "$f"; then
    echo "HOST VERIFY OK: source line markers present (debug build)."
  else
    echo "HOST VERIFY NOTE: no .cpp/.cc source lines (non-debug or limited DWARF)."
  fi
  return 0
}

verify_host "$HOST_OUT"

extract_device_embedded_elfs() {
  python3 - "$LIB_REAL" "$OUTDIR" "$OBJDUMP" "$READOBJ" <<'PY'
import re
import struct
import subprocess
import sys
from pathlib import Path

lib_path = Path(sys.argv[1])
outdir = Path(sys.argv[2])
objdump = Path(sys.argv[3])
readobj = Path(sys.argv[4])
devdir = outdir / "device"

data = lib_path.read_bytes()
EM_AMDGPU = 0xE0


def amdgcn_elf_spans(blob: bytes):
    """Byte ranges [start, end) of complete ELF64 AMDGPU objects embedded in blob."""
    spans = []
    n = len(blob)
    i = 0
    while i + 64 <= n:
        if blob[i : i + 4] != b"\x7fELF":
            i += 4
            continue
        if blob[i + 4] != 2:
            i += 4
            continue
        if struct.unpack_from("<H", blob, i + 18)[0] != EM_AMDGPU:
            i += 4
            continue
        e_shoff = struct.unpack_from("<Q", blob, i + 40)[0]
        e_shentsize = struct.unpack_from("<H", blob, i + 58)[0]
        e_shnum = struct.unpack_from("<H", blob, i + 60)[0]
        e_phoff = struct.unpack_from("<Q", blob, i + 32)[0]
        e_phentsize = struct.unpack_from("<H", blob, i + 54)[0]
        e_phnum = struct.unpack_from("<H", blob, i + 56)[0]
        end_rel = 0
        for p in range(e_phnum):
            o = i + e_phoff + p * e_phentsize
            if o + 56 > n:
                break
            _pt, _fl, p_offset, _v, _p, p_filesz, _m, _al = struct.unpack_from("<IIQQQQQQ", blob, o)
            if p_filesz:
                end_rel = max(end_rel, p_offset + p_filesz)
        if e_shoff and e_shentsize and e_shnum:
            end_rel = max(end_rel, e_shoff + e_shentsize * e_shnum)
        if end_rel > 0 and i + end_rel <= n:
            spans.append((i, i + end_rel))
        i += 4
    return spans


def read_mcpu(elf_path: Path) -> str:
    r = subprocess.run(
        [str(readobj), "-n", str(elf_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode == 0:
        m = re.search(r"amdhsa\.target:\s*amdgcn-amd-amdhsa--([A-Za-z0-9_:+-]+)", r.stdout)
        if m:
            return m.group(1).split(":")[0]
    return "native"


def main():
    spans = amdgcn_elf_spans(data)
    if not spans:
        print(
            "DEVICE: no embedded ELF64 AMDGPU images found in the host .so.\n"
            "       (Some ROCm layouts only expose device code via compressed offload\n"
            "        bundles — use ROCm 7+ llvm-objcopy --dump-offload-bundle= with the\n"
            "        file URI from offload_list.txt, or capture ISA at build time.)"
        )
        sys.exit(0)
    print(f"DEVICE: found {len(spans)} embedded amdgcn ELF image(s).")
    for idx, (a, b) in enumerate(spans):
        elf_path = devdir / f"amdgcn_{idx}.elf"
        asm_path = devdir / f"amdgcn_{idx}.s"
        mcpu_path = devdir / f"amdgcn_{idx}.mcpu.txt"
        chunk = data[a:b]
        elf_path.write_bytes(chunk)
        mcpu = read_mcpu(elf_path)
        mcpu_path.write_text(mcpu + "\n")
        cmd = [str(objdump), "-d", "--no-show-raw-insn", f"--mcpu={mcpu}", str(elf_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        text = r.stdout + r.stderr
        if r.returncode != 0 or "error:" in text.lower():
            cmd = [str(objdump), "-d", "--no-show-raw-insn", str(elf_path)]
            r2 = subprocess.run(cmd, capture_output=True, text=True)
            text = r2.stdout + r2.stderr
        asm_path.write_text(text)
        # AMDGPU listings use tab-indented mnemonics (s_, v_, global_, buffer_, …), not x86 "addr:".
        _op = re.compile(
            r"^\s+(?:s_|v_|buffer_|global_|flat_|ds_|scratch_|s_memtime|s_sleep|s_barrier)\w*"
        )
        insns = sum(1 for line in text.splitlines() if _op.search(line))
        print(f"DEVICE: wrote {elf_path.name} ({len(chunk)} bytes), {asm_path.name} ({insns} AMDGPU op lines), mcpu={mcpu}")


main()
PY
}

verify_device() {
  local any_elf=0
  local insns_total=0
  local f
  shopt -s nullglob
  for f in "$OUTDIR"/device/*.s; do
    [[ -f "$f" ]] || continue
    local c=0
    if grep -q "file format elf64-amdgpu" "$f" 2>/dev/null; then
      c="$(grep -cE '^[[:space:]]+(s_|v_|buffer_|global_|flat_|ds_|scratch_|s_memtime|s_sleep|s_barrier)' "$f" || true)"
    else
      c="$(grep -cE '^[[:space:]]*[0-9a-f]+:' "$f" || true)"
    fi
    if [[ "${c:-0}" -gt 50 ]]; then
      any_elf=1
      insns_total=$((insns_total + c))
    fi
  done
  shopt -u nullglob
  if [[ "$any_elf" -eq 0 ]]; then
    echo "DEVICE VERIFY NOTE: no device .s files with AMDGPU op lines (see DEVICE: messages above)."
    return 0
  fi
  echo "DEVICE VERIFY OK: $OUTDIR/device/ ($insns_total AMDGPU op lines across *.s)"
  return 0
}

echo "Device: scanning for embedded ELF64 AMDGPU objects..."
extract_device_embedded_elfs
verify_device
