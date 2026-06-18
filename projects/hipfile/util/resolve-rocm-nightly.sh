#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Resolves the latest ROCm nightly build and the clean ROCm version it pins.
# Used in CI to target a reproducible nightly when building the hipFile CI image.
#
# Two ways to use it:
#   Executed (CI):  ./resolve-rocm-nightly.sh
#                   -> emits (GITHUB_OUTPUT form) on stdout:
#                        nightly_build=<YYYYMMDD-gha_run_id>
#                        nightly_rocm_version=<MAJOR.MINOR.PATCH>
#   Tests:          bash resolve-rocm-nightly.sh --test
#
# The parsing/selection helpers are pure (stdin -> stdout), so --test validates
# them against fixtures offline -- without the network round-trips main() makes.
# Because the same helpers are both shipped to CI and exercised by --test, the
# two can never drift.

NIGHTLY_DEB_BASE="https://rocm.nightlies.amd.com/packages-multi-arch/deb"

# --- pure helpers ---------------------------------------------------------

# Reads a /deb/ directory index on stdin and prints the latest build id.
# Build ids are YYYYMMDD-<gha_run_id>; the date dominates, the numeric run id
# breaks ties so same-day builds resolve to a single deterministic match.
select_latest_build() {
    # awk 'NR==1' (rather than head) consumes the whole stream so sort does not
    # receive SIGPIPE under pipefail. The `|| true` keeps a no-match grep (exit 1)
    # from aborting the caller under `set -e`/pipefail: we emit nothing and exit 0
    # so the caller's own empty-result check can report a clean error.
    { grep -oE '[0-9]{8}-[0-9]+' || true; } \
        | sort -u -t- -k1,1nr -k2,2nr \
        | awk 'NR==1 { print }'
}

# Reads a binary-amd64/Packages file on stdin and prints the clean
# MAJOR.MINOR.PATCH version of the amdrocm-runtime-dev meta-package (the nightly
# Debian version is e.g. 7.14.0~20260602-26796279962; everything from '~' on is
# stripped).
extract_clean_version() {
    # Read the whole file (no early exit) so the upstream feeder does not get
    # SIGPIPE under pipefail; keep the first matching version.
    awk '
        /^Package:/ { in_pkg = ($0 == "Package: amdrocm-runtime-dev") }
        in_pkg && /^Version:/ && ver == "" { ver = $2; sub(/~.*/, "", ver) }
        END { if (ver != "") print ver }
    '
}

# --- main -----------------------------------------------------------------

main() {
    set -euo pipefail
    local index build packages version

    index="$(curl -fsSL "${NIGHTLY_DEB_BASE}/")"
    build="$(printf '%s' "${index}" | select_latest_build)"
    if [ -z "${build}" ]; then
        echo "resolve-rocm-nightly: no nightly build found at ${NIGHTLY_DEB_BASE}/" >&2
        exit 1
    fi

    packages="$(curl -fsSL "${NIGHTLY_DEB_BASE}/${build}/dists/stable/main/binary-amd64/Packages")"
    version="$(printf '%s' "${packages}" | extract_clean_version)"
    if [ -z "${version}" ]; then
        echo "resolve-rocm-nightly: could not determine ROCm version for build ${build}" >&2
        exit 1
    fi

    printf 'nightly_build=%s\n' "${build}"
    printf 'nightly_rocm_version=%s\n' "${version}"
}

# --- test suite -----------------------------------------------------------

_assert_eq() {
    local label="$1" actual="$2" expected="$3"
    if [ "${actual}" = "${expected}" ]; then
        printf '  PASS  %s\n        -> %s\n' "${label}" "${actual}"
    else
        printf '  FAIL  %s\n        expected: %s\n        got:      %s\n' \
            "${label}" "${expected}" "${actual}"
        return 1
    fi
}

_run_tests() {
    # not -e: the `read -r -d ''` fixtures below return non-zero at EOF, and we
    # want to keep going and tally failures rather than abort on the first one.
    set -uo pipefail
    local failures=0

    # Sample /deb/ index, mixing dates and same-date run ids, with href + text
    # duplicates as the real S3 listing produces.
    local INDEX_FIXTURE
    read -r -d '' INDEX_FIXTURE <<'EOF'
<a href="20260530-26673017729/">20260530-26673017729/</a>
<a href="20260602-26796000000/">20260602-26796000000/</a>
<a href="20260601-26733368587/">20260601-26733368587/</a>
<a href="20260602-26796279962/">20260602-26796279962/</a>
EOF

    # Sample Packages: the versioned -dev7.14 stanza must NOT be matched; only the
    # bare amdrocm-runtime-dev meta-package supplies the version.
    local PACKAGES_FIXTURE
    read -r -d '' PACKAGES_FIXTURE <<'EOF'
Package: amdrocm-runtime-dev7.14
Version: 7.14.0~20260602-26796279962
Architecture: amd64
Filename: pool/main/amdrocm-runtime-dev7.14_7.14.0~20260602-26796279962_amd64.deb

Package: amdrocm-runtime-dev
Version: 7.14.0~20260602-26796279962
Architecture: amd64
Filename: pool/main/amdrocm-runtime-dev_7.14.0~20260602-26796279962_amd64.deb
EOF

    # A different pinned version, meta stanza appearing first.
    local PACKAGES_FIXTURE_ALT
    read -r -d '' PACKAGES_FIXTURE_ALT <<'EOF'
Package: amdrocm-runtime-dev
Version: 7.15.1~20260701-30000000001
Architecture: amd64

Package: amdrocm-runtime-dev7.15
Version: 7.15.1~20260701-30000000001
Architecture: amd64
EOF

    echo "Test 1: latest build picks the highest date"
    _assert_eq "  select_latest_build" \
        "$(printf '%s' "${INDEX_FIXTURE}" | select_latest_build)" \
        "20260602-26796279962" || failures=$((failures+1))

    echo
    echo "Test 2: same-date tie-break picks the larger numeric run id"
    local INDEX_TIE
    read -r -d '' INDEX_TIE <<'EOF'
<a href="20260602-26796000000/">20260602-26796000000/</a>
<a href="20260602-9999999999/">20260602-9999999999/</a>
<a href="20260602-26796279962/">20260602-26796279962/</a>
EOF
    _assert_eq "  select_latest_build" \
        "$(printf '%s' "${INDEX_TIE}" | select_latest_build)" \
        "20260602-26796279962" || failures=$((failures+1))

    echo
    echo "Test 3: clean version from meta-package (strips ~build, ignores -dev7.14)"
    _assert_eq "  extract_clean_version" \
        "$(printf '%s' "${PACKAGES_FIXTURE}" | extract_clean_version)" \
        "7.14.0" || failures=$((failures+1))

    echo
    echo "Test 4: clean version when meta stanza is first"
    _assert_eq "  extract_clean_version" \
        "$(printf '%s' "${PACKAGES_FIXTURE_ALT}" | extract_clean_version)" \
        "7.15.1" || failures=$((failures+1))

    echo
    echo "Test 5: index with no build ids -> empty output and exit 0 (no-match grep"
    echo "        must not abort the caller under set -e/pipefail)"
    local nomatch_out nomatch_rc
    nomatch_out="$(printf '%s' '<a href="stable/">stable/</a>' | select_latest_build)"
    nomatch_rc=$?
    _assert_eq "  select_latest_build (output)" "${nomatch_out}" "" \
        || failures=$((failures+1))
    _assert_eq "  select_latest_build (exit)" "${nomatch_rc}" "0" \
        || failures=$((failures+1))

    echo
    if [ "${failures}" -eq 0 ]; then
        echo "All tests passed."
    else
        echo "${failures} test(s) FAILED."
        return 1
    fi
}

# --- dispatch -------------------------------------------------------------
# When sourced, ${BASH_SOURCE[0]} (this file) differs from ${0} (the caller's
# shell); when executed they are equal. So "!=" means "we are being sourced".
# Do NOT set shell options at top level: sourcing must not mutate the caller's
# shell. main and _run_tests each set the options they want.
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    : # sourced: helper functions are now available; do nothing else
else
    case "${1:-}" in
        --test) _run_tests ;;
        "")     main ;;
        *)      echo "usage: ./resolve-rocm-nightly.sh [--test]" >&2; exit 2 ;;
    esac
fi
