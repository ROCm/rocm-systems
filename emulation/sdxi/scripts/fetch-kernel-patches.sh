#!/usr/bin/env bash
# Fetch the SDXI Linux kernel driver patch series from public mailing list
# archives and store them under patches/kernel/.
#
# Usage: ./scripts/fetch-kernel-patches.sh [--version v1|v2|v3|rfc] [--all]
#
# Patches are saved as patches/kernel/<version>/<NNN>-<slug>.patch so they
# can be applied with:
#   git am patches/kernel/v3/*.patch

set -euo pipefail

PATCHES_DIR="$(dirname "$0")/../patches/kernel"

# Mailing list thread roots (message-id index pages on ratatoskr.run)
# v4 URL is not yet known; fill it in when the series is posted.
declare -A THREAD_URLS=(
    [rfc]="https://www.spinics.net/lists/kernel/msg5839841.html"
    [v1]="https://ratatoskr.run/lkml/2026/04/3508169/t"
    [v2]="https://ratatoskr.run/linux-pci/2026/05/8996752/t"
    [v3]="https://ratatoskr.run/linux-pci/2026/06/17096410/t"
    [v4]=""   # TODO: fill in when v4 is posted to linux-pci
)

# lore.kernel.org query URLs — the canonical source for downloadable mboxes
# These return a mbox bundle of the entire series.
declare -A LORE_URLS=(
    [rfc]="https://lore.kernel.org/dmaengine/?q=SDXI+basic+support+RFC"
    [v1]="https://lore.kernel.org/linux-pci/?q=dmaengine+SDXI+v1"
    [v2]="https://lore.kernel.org/linux-pci/?q=dmaengine+SDXI+v2"
    [v3]="https://lore.kernel.org/linux-pci/?q=dmaengine+SDXI+v3"
    [v4]=""   # TODO: fill in when v4 is posted
)

usage() {
    echo "Usage: $0 [--version <rfc|v1|v2|v3|v4>] [--all]"
    echo "       $0 --list   # print known thread URLs"
    exit 1
}

list_urls() {
    echo "Known SDXI patch series:"
    for ver in rfc v1 v2 v3 v4; do
        url="${THREAD_URLS[$ver]}"
        echo "  $ver: ${url:-<not yet posted>}"
    done
}

fetch_version() {
    local ver="$1"
    local url="${THREAD_URLS[$ver]:-__missing__}"
    if [[ "$url" == "__missing__" ]]; then
        echo "Unknown version: $ver (not in THREAD_URLS)" >&2
        exit 1
    fi
    if [[ -z "$url" ]]; then
        echo "Version $ver is not yet posted; update THREAD_URLS[$ver] when the series appears." >&2
        exit 1
    fi

    local outdir="$PATCHES_DIR/$ver"
    mkdir -p "$outdir"

    echo "Fetching $ver thread index: $url"
    # Download the thread index to find individual patch URLs
    local index_file="$outdir/00-thread-index.html"
    curl -fsSL --max-time 30 "$url" -o "$index_file"
    echo "  Saved thread index to $index_file"

    # Extract .patch or /raw URLs from the index and download each one.
    # ratatoskr.run uses hrefs like /linux-pci/2026/06/<msgid>
    # lore.kernel.org patches are at https://lore.kernel.org/<list>/<msgid>/raw
    local patch_urls
    patch_urls=$(grep -oE 'https?://[^"]+\.patch' "$index_file" 2>/dev/null || true)
    if [[ -z "$patch_urls" ]]; then
        # Try extracting message hrefs and constructing /raw URLs
        patch_urls=$(grep -oE '/[a-z-]+/[0-9]{4}/[0-9]{2}/[0-9]+' "$index_file" \
            | sort -u \
            | sed 's|^|https://lore.kernel.org|; s|$|/raw|' || true)
    fi

    if [[ -z "$patch_urls" ]]; then
        echo "  Could not extract individual patch URLs from index."
        echo "  Browse manually: $url"
        echo "  Thread index saved at: $index_file"
        return
    fi

    local n=1
    while IFS= read -r purl; do
        local fname
        fname=$(printf "%03d.patch" "$n")
        echo "  Fetching patch $n: $purl"
        curl -fsSL --max-time 30 "$purl" -o "$outdir/$fname" || {
            echo "  Warning: failed to fetch $purl" >&2
        }
        n=$((n + 1))
    done <<< "$patch_urls"

    echo "  Done — $((n-1)) patches in $outdir/"
}

# --- main ---
versions=()
fetch_all=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version|-v) shift; versions+=("$1") ;;
        --all)        fetch_all=true ;;
        --list)       list_urls; exit 0 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown argument: $1" >&2; usage ;;
    esac
    shift
done

if $fetch_all; then
    versions=(rfc v1 v2 v3)   # v4 excluded until URL is known
fi

if [[ ${#versions[@]} -eq 0 ]]; then
    echo "Fetching latest (v3) by default. Use --all or --version <ver> for others."
    versions=(v3)
fi

mkdir -p "$PATCHES_DIR"

for ver in "${versions[@]}"; do
    fetch_version "$ver"
done

echo ""
echo "Patch index stored under $PATCHES_DIR/"
echo "Thread URLs:"
for ver in "${versions[@]}"; do
    echo "  $ver: ${THREAD_URLS[$ver]}"
done
