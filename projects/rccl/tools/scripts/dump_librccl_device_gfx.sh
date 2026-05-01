#!/usr/bin/env bash
# EM_AMDGPU ELF64 blobs inside a HIP host .so -> stem.<gfx>.s (GNU grep/dd/od).
set -euo pipefail
F="${1:?$0 lib.so [outdir]}"; D="${2:-.}"
P="${ROCM_PATH:-/opt/rocm}/llvm/bin"
X="${LLVM_OBJDUMP:-$P/llvm-objdump}"; Y="${LLVM_READOBJ:-$P/llvm-readobj}"
[[ -f $F && -x $X && -x $Y ]] || exit 1
L=$(readlink -f "$F"); mkdir -p "$D"
S=$(basename "$L"); S="${S%%.so*}"
u16(){ od -An -tu2 -j"$2" -N2 "$1"|awk 'NF{print$1;exit}'; }
u64(){ od -An -tu8 -j"$2" -N8 "$1"|awk 'NF{print$1;exit}'; }
span(){ # $1=file $2=elf offset -> echo "start end"
  local z=$(wc -c <"$1") pos=$2 ph i n fs e er=0 x
  ((pos+64<=z))||return 1
  [[ $(od -An -tu1 -j$((pos+4)) -N1 "$1"|awk 'NF{print$1;exit}') -eq 2 ]]||return 1
  [[ $(u16 "$1" $((pos+18))) -eq 224 ]]||return 1
  ph=$(u64 "$1" $((pos+32))); i=$(u16 "$1" $((pos+54))); n=$(u16 "$1" $((pos+56)))
  for((j=0;j<n;j++)); do fs=$(u64 "$1" $((pos+ph+j*i+32)));((fs>er))&&er=$fs;done
  e=$(u64 "$1" $((pos+40))); i=$(u16 "$1" $((pos+58))); n=$(u16 "$1" $((pos+60)))
  ((e&&i&&n))&&{ x=$((e+i*n));((x>er))&&er=$x; }
  ((er>0&&pos+er<=z))||return 1; echo "$pos $((pos+er))"
}
gfx(){ $Y -n "$1" 2>/dev/null|grep -Fm1 amdhsa.target:|sed -n 's/.*amdgcn-amd-amdhsa--\([^:[:space:]]*\).*/\1/p'|head -1; }
nextout(){
  local b o k=0
  b="$D/$S.$1"; o="${b}.s"
  while [[ -f $o ]]; do k=$((k+1)); o="${b}.${k}.s"; done
  echo "$o"
}
n=0
while IFS= read -r ln; do
  p="${ln%%:*}"; [[ $p =~ ^[0-9]+$ ]]||continue
  s=$(span "$L" "$p")||continue
  read -r a z <<<"$s"
  t=$(mktemp)
  dd if="$L" of="$t" ibs=1 skip="$a" count=$((z-a)) iflag=skip_bytes,count_bytes status=none
  g=$(gfx "$t"); [[ -n $g ]]||g=unknown
  o=$(nextout "$g")
  { $X -d --no-show-raw-insn --mcpu="$g" "$t" 2>&1||$X -d --no-show-raw-insn "$t" 2>&1; }|sed '/file format elf64-amdgpu/d;/./,$!d' >"$o"
  echo "$o"; n=$((n+1)); rm -f "$t"
done < <(LC_ALL=C grep -aboa $'\x7fELF' "$L")
((n))||exit 1
