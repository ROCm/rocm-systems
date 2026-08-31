#!/bin/bash
# Run the end-to-end test against the patched CLR, flag off and flag on, plus a
# null control.
#
# The flag is read once at runtime init and cannot be changed inside a process,
# so this is a between-process comparison and much noisier than the paired
# within-run comparisons everywhere else. The null control - two runs at the same
# setting - is what makes the off/on difference interpretable: any off/on gap
# smaller than the off/off gap means nothing.
#
#   ./e2e_run.sh            30 iterations
#   ./e2e_run.sh 60
set -uo pipefail
cd "$(dirname "$0")"

INSTALL=~/airuntime28-clr-install
LIB="$INSTALL/lib"
ITERS=${1:-30}
NOISE='rj warn|Resource leak detected|LoadLib\(|Secondary CUID|AMDCUID|^ROW'

if [ ! -d "$LIB" ]; then
  echo "No patched CLR at $INSTALL. Run ./clr_setup.sh then ./clr_build.sh first."
  exit 1
fi

echo "=== which libamdhip64 will be used ==="
ls -la "$LIB"/libamdhip64.so*

./build.sh e2e_memcpy || exit 1

# The built libamdhip64 must come first, but the rest of the ROCm stack
# (rocprofiler-register, hsa-runtime, comgr) still resolves from /opt/rocm.
export LD_LIBRARY_PATH="$LIB:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
echo "resolved libamdhip64: $(ldd build/e2e_memcpy | grep amdhip64)"

run() {  # run <flag> <label>
  echo
  echo "############################################################"
  echo "### DEBUG_CLR_BLIT_NONTEMPORAL=$1   ($2)"
  echo "############################################################"
  DEBUG_CLR_BLIT_NONTEMPORAL=$1 ./build/e2e_memcpy --iters "$ITERS" 2>&1 | grep -vE "$NOISE"
}

run 0 "baseline"
run 0 "null control: identical setting, so any difference here is noise"
run 1 "non-temporal stores enabled"

echo
echo "=== does the flag actually switch which kernel is dispatched ==="
for NT in 0 1; do
  echo "--- DEBUG_CLR_BLIT_NONTEMPORAL=$NT ---"
  DEBUG_CLR_BLIT_NONTEMPORAL=$NT AMD_LOG_LEVEL=4 ./build/e2e_memcpy --iters 1 --warmup 0 2>&1 \
    | grep -oE '__amd_rocclr_copyBuffer[A-Za-z]*' | sort | uniq -c
done
echo "(expect copyBufferNT to appear only at =1; if it appears at both or neither,"
echo " the flag is not reaching the kernel selection and the comparison is void)"

echo
echo "=== compare the three RESULT lines above ==="
echo "The off-vs-off gap is the noise floor for this comparison. Only an off-vs-on"
echo "gap larger than it means anything."
echo "E2E_DONE"
