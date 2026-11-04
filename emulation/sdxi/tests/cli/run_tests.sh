#!/usr/bin/env bash
# SDXI CLI integration tests — run inside a QEMU VM with sdxi_char.ko loaded.
#
# Prerequisites:
#   - sdxi_char.ko loaded (insmod drivers/sdxi_char/sdxi_char.ko)
#   - sdxi-cli in PATH or set SDXI_CLI=/path/to/sdxi-cli
#   - /dev/sdxi present (created by sdxi_char.ko)
#
# Usage: bash run_tests.sh [-d /dev/sdxi_other] [-v]

set -euo pipefail

SDXI_CLI="${SDXI_CLI:-sdxi-cli}"
# Auto-detect /dev/sdxi* if the caller didn't specify a device
if [[ -z "${DEVICE:-}" ]]; then
    DEVICE="$(ls /dev/sdxi* 2>/dev/null | head -n1)"
    [[ -z "$DEVICE" ]] && { echo "ERROR: /dev/sdxi not found — is sdxi_char.ko loaded?"; exit 1; }
fi
VERBOSE=0
PASS=0
FAIL=0
TMPDIR_TEST="$(mktemp -d /tmp/sdxi_test.XXXXXX)"
trap 'rm -rf "$TMPDIR_TEST"' EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d) DEVICE="$2"; shift 2 ;;
        -v) VERBOSE=1; shift ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

OPTS="-d $DEVICE"
[[ $VERBOSE -eq 1 ]] && OPTS="$OPTS -v"

check() {
    local desc="$1" rc="$2"
    if [[ $rc -eq 0 ]]; then
        echo "PASS: $desc"
        ((PASS++)) || true
    else
        echo "FAIL: $desc"
        ((FAIL++)) || true
    fi
}

require_device() {
    if [[ ! -c "$DEVICE" ]]; then
        echo "ERROR: $DEVICE not found — is sdxi_char.ko loaded?"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# test_copy_small: 4 KiB block, verify byte-for-byte match
# ---------------------------------------------------------------------------
test_copy_small() {
    local src="$TMPDIR_TEST/copy_src.bin"
    local dst="$TMPDIR_TEST/copy_dst.bin"
    dd if=/dev/urandom bs=4096 count=1 of="$src" 2>/dev/null
    $SDXI_CLI $OPTS copy "$src" "$dst"
    cmp -s "$src" "$dst"
    check "copy 4 KiB: data matches" $?
}

# ---------------------------------------------------------------------------
# test_copy_1m: 1 MiB block
# ---------------------------------------------------------------------------
test_copy_1m() {
    local src="$TMPDIR_TEST/copy1m_src.bin"
    local dst="$TMPDIR_TEST/copy1m_dst.bin"
    dd if=/dev/urandom bs=1M count=1 of="$src" 2>/dev/null
    $SDXI_CLI $OPTS copy "$src" "$dst"
    cmp -s "$src" "$dst"
    check "copy 1 MiB: data matches" $?
}

# ---------------------------------------------------------------------------
# test_compress_text: compress a text file, verify output is smaller
# ---------------------------------------------------------------------------
test_compress_text() {
    local src="$TMPDIR_TEST/text.txt"
    local cmp_out="$TMPDIR_TEST/text.zlib"
    # Generate highly compressible text
    python3 -c "print('SDXI test pattern ' * 1000)" > "$src"
    $SDXI_CLI $OPTS compress "$src" "$cmp_out"
    local orig compressed
    orig=$(wc -c < "$src")
    compressed=$(wc -c < "$cmp_out")
    [[ $compressed -lt $orig ]]
    check "compress text: output smaller than input ($compressed < $orig)" $?
}

# ---------------------------------------------------------------------------
# test_roundtrip_text: compress + decompress a text file
# ---------------------------------------------------------------------------
test_roundtrip_text() {
    local src="$TMPDIR_TEST/rt_text.txt"
    printf 'SDXI compress/decompress roundtrip test\n%.0s' {1..500} > "$src"
    $SDXI_CLI $OPTS roundtrip "$src"
    check "roundtrip text: data matches" $?
}

# ---------------------------------------------------------------------------
# test_roundtrip_binary: compress + decompress random binary data
# ---------------------------------------------------------------------------
test_roundtrip_binary() {
    local src="$TMPDIR_TEST/rt_bin.bin"
    dd if=/dev/urandom bs=65536 count=1 of="$src" 2>/dev/null
    # Random data compresses poorly but must round-trip correctly
    $SDXI_CLI $OPTS roundtrip "$src"
    check "roundtrip binary 64 KiB: data matches" $?
}

# ---------------------------------------------------------------------------
# test_decompress_bad: feeding garbage should return an error
# ---------------------------------------------------------------------------
test_decompress_bad() {
    local garbage="$TMPDIR_TEST/garbage.bin"
    local out="$TMPDIR_TEST/garbage_dec.bin"
    dd if=/dev/urandom bs=256 count=1 of="$garbage" 2>/dev/null
    if $SDXI_CLI $OPTS decompress "$garbage" "$out" 2>/dev/null; then
        check "decompress bad data: returns error" 1
    else
        check "decompress bad data: returns error" 0
    fi
}

# ---------------------------------------------------------------------------
# test_bench_copy: quick benchmark (5 iterations)
# ---------------------------------------------------------------------------
test_bench_copy() {
    $SDXI_CLI $OPTS bench copy $((1024*1024)) 5 > /dev/null
    check "bench copy 1 MiB × 5: completes without error" $?
}

# ---------------------------------------------------------------------------
# test_bench_compress: quick compress benchmark
# ---------------------------------------------------------------------------
test_bench_compress() {
    $SDXI_CLI $OPTS bench compress $((64*1024)) 5 > /dev/null
    check "bench compress 64 KiB × 5: completes without error" $?
}

# ---------------------------------------------------------------------------
# test_bench_decompress: quick decompress benchmark
# ---------------------------------------------------------------------------
test_bench_decompress() {
    $SDXI_CLI $OPTS bench decompress $((64*1024)) 5 > /dev/null
    check "bench decompress 64 KiB × 5: completes without error" $?
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
require_device

echo "=== SDXI CLI tests (device: $DEVICE) ==="

test_copy_small
test_copy_1m
test_compress_text
test_roundtrip_text
test_roundtrip_binary
test_decompress_bad
test_bench_copy
test_bench_compress
test_bench_decompress

echo ""
echo "Results: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]]
