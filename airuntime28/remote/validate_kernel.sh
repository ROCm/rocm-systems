#!/bin/bash
# Validate the *shipped* blit kernel source, not a hand-copied approximation.
#
# blitcl.cpp is standalone C++ whose only content is two stringified OpenCL
# blobs, so it can be compiled against a tiny driver that prints the blob back
# out. That emitted .cl is then compiled for gfx1250 and inspected. If the
# kernel in CLR ever drifts from what we measured, this catches it.
set -euo pipefail
cd ~/airuntime28/clr-validate

CLANG=/opt/rocm/llvm/bin/clang
ARCH=gfx1250

cat > dump_blit.cpp <<'EOF'
#include <cstdio>
namespace amd::device {
extern const char* BlitLinearSourceCode;
extern const char* BlitImageSourceCode;
}
int main() { std::printf("%s\n", amd::device::BlitLinearSourceCode); return 0; }
EOF

echo "=== extracting BlitLinearSourceCode from the real blitcl.cpp ==="
g++ -std=c++17 -O0 dump_blit.cpp blitcl.cpp -o dump_blit
./dump_blit > blit_linear.cl
wc -l blit_linear.cl
grep -c '__kernel' blit_linear.cl | sed 's/^/kernels found: /'

echo
echo "=== both copy kernels present? ==="
grep -o '__amd_rocclr_copyBuffer[A-Za-z]*' blit_linear.cl | sort -u

echo
echo "=== compiling extracted OpenCL for ${ARCH} ==="
# -nogpulib because the extern __amd_* helpers live in device-libs and are only
# needed at link time; -S stops before that.
$CLANG --target=amdgcn-amd-amdhsa -mcpu=$ARCH -nogpulib -O3 \
       -x cl -cl-std=CL2.0 -Xclang -finclude-default-header \
       -S blit_linear.cl -o blit_linear.s 2>&1 | grep -vE 'warning: (unused|argument)' || true
wc -l blit_linear.s

echo
echo "=== memory ops in __amd_rocclr_copyBuffer (baseline, must have NO th:) ==="
awk '/^__amd_rocclr_copyBuffer:/{f=1} f&&/^\s*\.size|^__amd_rocclr_copyBufferNT:/{f=0} f&&/global_(load|store)/{gsub(/^[ \t]+/,"");print "  "$0}' blit_linear.s | sort -u

echo
echo "=== memory ops in __amd_rocclr_copyBufferNT (must be b128 + TH_STORE_NT) ==="
awk '/^__amd_rocclr_copyBufferNT:/{f=1} f&&/^\s*\.size/{f=0} f&&/global_(load|store)/{gsub(/^[ \t]+/,"");print "  "$0}' blit_linear.s | sort -u

echo
echo "=== assertions ==="
NT_B128=$(awk '/^__amd_rocclr_copyBufferNT:/{f=1} f&&/^\s*\.size/{f=0} f' blit_linear.s | grep -c 'global_store_b128 .*th:TH_STORE_NT' || true)
NT_LOAD_PLAIN=$(awk '/^__amd_rocclr_copyBufferNT:/{f=1} f&&/^\s*\.size/{f=0} f' blit_linear.s | grep -c 'global_load_b128 .*th:' || true)
BASE_NT=$(awk '/^__amd_rocclr_copyBuffer:/{f=1} f&&(/^\s*\.size/||/^__amd_rocclr_copyBufferNT:/){f=0} f' blit_linear.s | grep -c 'th:' || true)

[ "$NT_B128" -ge 1 ] && echo "  [OK ] NT kernel emits global_store_b128 th:TH_STORE_NT (full 128-bit width)" \
                     || echo "  [BAD] NT kernel did NOT emit a 128-bit non-temporal store"
[ "$NT_LOAD_PLAIN" -eq 0 ] && echo "  [OK ] NT kernel load carries no temporal hint (source stays cacheable)" \
                           || echo "  [BAD] NT kernel load unexpectedly carries a hint"
[ "$BASE_NT" -eq 0 ] && echo "  [OK ] baseline __amd_rocclr_copyBuffer unchanged (no th: hints)" \
                     || echo "  [BAD] baseline kernel changed"

echo
echo "VALIDATE_DONE"
