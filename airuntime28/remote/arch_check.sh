#!/bin/bash
# DEBUG_CLR_BLIT_NONTEMPORAL is not gated by architecture, and blitcl.cpp is
# compiled at runtime for whatever device is present. So the same source runs on
# MI200, MI300 and Navi too - where __builtin_nontemporal_store lowers to a
# completely different cache-control encoding than gfx12's th: field.
#
# Everything measured in this investigation was on gfx1250. This shows what the
# other targets would actually execute, which is the honest scope of the risk.
set -uo pipefail
CLANG=/opt/rocm/llvm/bin/clang

SRC=$(mktemp /tmp/ntarch.XXXX.cl)
cat > "$SRC" <<'EOF'
__kernel void nt_store(__global ulong2* src, __global ulong2* dst) {
  __builtin_nontemporal_store(src[get_global_id(0)], &dst[get_global_id(0)]);
}
__kernel void plain_store(__global ulong2* src, __global ulong2* dst) {
  dst[get_global_id(0)] = src[get_global_id(0)];
}
EOF

printf '%-10s %-14s %-38s %s\n' "target" "part" "non-temporal store emits" "plain store emits"
printf -- '%.0s-' {1..112}; echo

for pair in "gfx90a:MI200" "gfx942:MI300" "gfx950:MI350" "gfx1100:Navi3" "gfx1250:MI450"; do
  arch=${pair%%:*}; part=${pair##*:}
  asm=$($CLANG --target=amdgcn-amd-amdhsa -mcpu=$arch -nogpulib -O2 \
        -x cl -cl-std=CL2.0 -Xclang -finclude-default-header -S "$SRC" -o - 2>/dev/null)
  [ -z "$asm" ] && { printf '%-10s %-14s %s\n' "$arch" "$part" "(target not supported by this compiler)"; continue; }

  nt=$(echo "$asm" | awk '/^nt_store:/{f=1} f&&/^plain_store:/{f=0} f&&/(global|flat|buffer)_store/{gsub(/^[ \t]+/,"");print;exit}')
  pl=$(echo "$asm" | awk '/^plain_store:/{f=1} f&&/\.size/{f=0} f&&/(global|flat|buffer)_store/{gsub(/^[ \t]+/,"");print;exit}')
  # Strip registers, keep opcode plus any cache-control modifiers.
  clean() { echo "$1" | sed -E 's/v\[[0-9:]+\]/V/g; s/v[0-9]+/V/g; s/s\[[0-9:]+\]/S/g; s/,//g; s/  +/ /g'; }
  printf '%-10s %-14s %-38s %s\n' "$arch" "$part" "$(clean "$nt")" "$(clean "$pl")"
done
rm -f "$SRC"

echo
echo "If the modifier differs between rows, the flag means something different on"
echo "that part, and none of the gfx1250 measurements transfer to it."
