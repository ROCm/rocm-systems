#!/bin/bash
# detect_asic_filter.sh — Detect the GPU ASIC and build the appropriate
# gtest negative filter from amdsmitst.exclude.
#
# Usage:
#   source amdsmitst.exclude        # loads FILTER[] and BLACKLIST_ALL_ASICS
#   source detect_asic_filter.sh    # sets GTEST_EXCLUDE
#   ./amdsmitst --gtest_filter="-${GTEST_EXCLUDE}"
#
# Requires: FILTER (associative array) and BLACKLIST_ALL_ASICS from
#           amdsmitst.exclude to already be sourced.

# ---------- gfx string to ASIC name mapping ----------
# Maps TARGET_GRAPHICS_VERSION strings from amd-smi to the ASIC name or
# gfx_target_version used as FILTER keys in amdsmitst.exclude.
declare -A GFX_TO_ASIC=(
    [gfx900]=vega10
    [gfx906]=vega20
    [gfx908]=arcturus
    [gfx90a]=aldebaran
    [gfx940]=90400
    [gfx941]=90401
    [gfx942]=90402
    [gfx1030]=sienna_cichlid
    [gfx1100]=strix_point
    [gfx1150]=gfx1150
    [gfx1151]=gfx1151
    [gfx1152]=gfx1152
    [gfx1153]=gfx1153
)

# ---------- ASIC detection ----------
# Method 1: KFD topology nodes expose the ASIC name.  Newer GPUs
# ("ip discovery") require gfx_target_version from the node properties.
ASIC_NAME=""
for node_dir in /sys/class/kfd/kfd/topology/nodes/*/; do
    if [ -f "${node_dir}name" ]; then
        node_name=$(cat "${node_dir}name" 2>/dev/null | tr -d '[:space:]')
        if [ -n "$node_name" ] && [ "$node_name" != "ip_discovery" ] && \
           [ "$node_name" != "ipdiscovery" ]; then
            ASIC_NAME="$node_name"
            break
        fi
        # For ip discovery nodes, fall back to gfx_target_version
        if [ "$node_name" = "ip_discovery" ] || [ "$node_name" = "ipdiscovery" ]; then
            if [ -f "${node_dir}properties" ]; then
                gfx_ver=$(grep 'gfx_target_version' "${node_dir}properties" 2>/dev/null | awk '{print $2}')
                if [ -n "$gfx_ver" ] && [ "$gfx_ver" != "0" ]; then
                    ASIC_NAME="$gfx_ver"
                    break
                fi
            fi
        fi
    fi
done

# Method 2: If KFD topology didn't yield an ASIC, try amd-smi.
# This is more reliable inside Docker containers where sysfs may be
# partially mounted.
if [ -z "$ASIC_NAME" ] && command -v amd-smi >/dev/null 2>&1; then
    gfx_str=$(amd-smi static --asic 2>/dev/null \
              | grep 'TARGET_GRAPHICS_VERSION' \
              | head -1 \
              | awk '{print $NF}')
    if [ -n "$gfx_str" ]; then
        # Map gfx string to FILTER key via lookup table
        if [ -n "${GFX_TO_ASIC[$gfx_str]+x}" ]; then
            ASIC_NAME="${GFX_TO_ASIC[$gfx_str]}"
            echo "KFD detection failed — amd-smi detected $gfx_str → $ASIC_NAME"
        else
            echo "KFD detection failed — amd-smi detected $gfx_str (no FILTER mapping)"
        fi
    fi
fi

# ---------- Virtualization detection ----------
VIRT_MODE=""
virt_file=$(find /sys/class/drm/card*/device/ -name 'current_virtualization_mode' 2>/dev/null | head -1)
if [ -n "$virt_file" ]; then
    virt_val=$(cat "$virt_file" 2>/dev/null | tr -d '[:space:]')
    if [ "$virt_val" = "SRIOV" ] || [ "$virt_val" = "VF" ]; then
        VIRT_MODE="virtualization"
    fi
fi

# ---------- Build combined gtest exclude filter ----------
# Start with the ASIC-specific blacklist, or fall back to the global one.
if [ -n "$ASIC_NAME" ] && [ -n "${FILTER[$ASIC_NAME]+x}" ]; then
    GTEST_EXCLUDE="${FILTER[$ASIC_NAME]}"
    echo "Detected ASIC: $ASIC_NAME — using device-specific blacklist"
else
    GTEST_EXCLUDE="${BLACKLIST_ALL_ASICS}"
    echo "ASIC '${ASIC_NAME:-unknown}' has no specific filter — using global blacklist"
fi

# Layer on virtualization exclusions if running in a VM/VF.
if [ -n "$VIRT_MODE" ] && [ -n "${FILTER[$VIRT_MODE]+x}" ]; then
    GTEST_EXCLUDE="${GTEST_EXCLUDE}${FILTER[$VIRT_MODE]}"
    echo "Virtualization detected — appending virtualization blacklist"
fi

echo "Final gtest negative filter: -${GTEST_EXCLUDE}"
