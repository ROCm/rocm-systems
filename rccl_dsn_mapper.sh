#!/bin/bash
#
# rccl_dsn_mapper.sh — Build inter-switch-link mapping for RCCL topology inference
#
# Discovers Broadcom PEX multi-host switch partitions by reading the PCIe Device
# Serial Number (DSN) from extended config space. All upstream ports sharing the
# same DSN belong to the same physical switch — their cross-partition paths use
# internal fabric, not the CPU.
#
# Populates the sysfs-compatible directory structure that RCCL reads via
# ncclOsGetBcmLinks() (src/os/linux.cc:702-730):
#   /sys/kernel/pci_switch_link/virtual_switch_links/<busid>/<peer_busid>
#
# Usage:
#   sudo ./rccl_dsn_mapper.sh [--discover] [--populate] [--clean] [--sysfs-base PATH] [--log-file PATH]
#
# Modes:
#   --discover     Print discovered switch groups and peer links (default)
#   --populate     Create the sysfs directory structure for RCCL consumption
#   --clean        Remove all files/directories previously created by --populate
#
# Options:
#   --sysfs-base   Override the base path (default: /sys/kernel/pci_switch_link/virtual_switch_links)
#   --log-file     Override the change log path (default: /var/run/rccl_dsn_mapper.log)
#
# Modeled after /apps/shared/disable_acs.sh — intended to run as a privileged
# step during NIC bring-up, before RCCL initializes.
#

set -euo pipefail

# --- Configuration ---
SYSFS_BASE="/var/run/rccl_bcm_links"
CHANGE_LOG="/var/run/rccl_dsn_mapper.log"
MODE="discover"        # discover | populate | clean
BCM_VENDOR="1000"      # Broadcom vendor ID
BCM_SWITCH_DEV="c030"  # PEX890xx device ID

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --discover)   MODE="discover"; shift ;;
        --populate)   MODE="populate"; shift ;;
        --clean)      MODE="clean"; shift ;;
        --sysfs-base) SYSFS_BASE="$2"; shift 2 ;;
        --log-file)   CHANGE_LOG="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--discover] [--populate] [--clean] [--sysfs-base PATH] [--log-file PATH]"
            echo ""
            echo "Modes:"
            echo "  --discover     Print discovered switch groups and peer links (default)"
            echo "  --populate     Create sysfs directory structure for RCCL"
            echo "  --clean        Remove all files/directories created by a previous --populate"
            echo ""
            echo "Options:"
            echo "  --sysfs-base   Override base path (default: /sys/kernel/pci_switch_link/virtual_switch_links)"
            echo "  --log-file     Override change log path (default: /var/run/rccl_dsn_mapper.log)"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Usage: $0 [--discover] [--populate] [--clean] [--sysfs-base PATH] [--log-file PATH]"
            exit 1
            ;;
    esac
done

# --- Root check (same pattern as disable_acs.sh) ---
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: $0 must be run as root"
    exit 1
fi

# --- Clean mode: remove previously created entries and exit ---
if [ "${MODE}" = "clean" ]; then
    echo "============================================================"
    echo "  RCCL DSN Mapper — Clean Mode"
    echo "============================================================"
    echo ""

    if [ ! -f "${CHANGE_LOG}" ]; then
        echo "No change log found at: ${CHANGE_LOG}"
        echo "Nothing to clean."
        exit 0
    fi

    echo "Reading change log: ${CHANGE_LOG}"
    echo ""

    # Read the log and remove entries in reverse order (files first, then dirs)
    # Log format: TIMESTAMP ACTION PATH
    REMOVED=0
    FAILED=0

    # Pass 1: remove files (link entries)
    while IFS=' ' read -r TIMESTAMP ACTION ENTRY_PATH; do
        if [ "${ACTION}" != "CREATE_FILE" ]; then
            continue
        fi
        if [ -e "${ENTRY_PATH}" ]; then
            rm -f "${ENTRY_PATH}"
            echo "  Removed file: ${ENTRY_PATH}"
            logger "rccl_dsn_mapper: Cleaned file ${ENTRY_PATH}"
            REMOVED=$((REMOVED + 1))
        else
            echo "  Already absent: ${ENTRY_PATH}"
        fi
    done < "${CHANGE_LOG}"

    # Pass 2: remove directories (in reverse order so children are removed before parents)
    tac "${CHANGE_LOG}" | while IFS=' ' read -r TIMESTAMP ACTION ENTRY_PATH; do
        if [ "${ACTION}" != "CREATE_DIR" ]; then
            continue
        fi
        if [ -d "${ENTRY_PATH}" ]; then
            # Only remove if empty (safety check)
            if [ -z "$(ls -A "${ENTRY_PATH}" 2>/dev/null)" ]; then
                rmdir "${ENTRY_PATH}"
                echo "  Removed dir:  ${ENTRY_PATH}"
                logger "rccl_dsn_mapper: Cleaned directory ${ENTRY_PATH}"
                REMOVED=$((REMOVED + 1))
            else
                echo "  Skipped non-empty dir: ${ENTRY_PATH}"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "  Already absent: ${ENTRY_PATH}"
        fi
    done

    # Remove the change log itself
    rm -f "${CHANGE_LOG}"
    echo ""
    echo "  Entries removed: ${REMOVED}"
    echo "  Skipped:         ${FAILED}"
    echo "  Change log removed: ${CHANGE_LOG}"
    echo ""
    echo "Done."
    exit 0
fi

# --- Platform identification ---
PLATFORM=$(dmidecode --string system-product-name 2>/dev/null || echo "unknown")
logger "rccl_dsn_mapper: PLATFORM=${PLATFORM}"

# --- Phase 1: Discover Broadcom upstream ports and read DSN ---

declare -A DSN_TO_PORTS     # DSN string -> space-separated upstream BDFs
declare -A BDF_TO_DSN       # BDF -> DSN string
UPSTREAM_COUNT=0
SKIPPED_COUNT=0

for BDF in $(lspci -D -d ${BCM_VENDOR}:${BCM_SWITCH_DEV} | awk '{print $1}'); do
    # Check if this is an Upstream Port (skip downstream ports, mgmt endpoints)
    PORT_TYPE=$(lspci -vvs ${BDF} 2>/dev/null | grep -oP 'Express \(v2\) \K(Upstream Port|Downstream Port)' | head -1)
    if [ "${PORT_TYPE}" != "Upstream Port" ]; then
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi

    # Read DSN from extended config space (offset 0x104 = lower 32b, 0x108 = upper 32b)
    DSN_LO=$(setpci -s ${BDF} 0x104.L 2>/dev/null) || continue
    DSN_HI=$(setpci -s ${BDF} 0x108.L 2>/dev/null) || continue

    # Skip if DSN read failed (returns all-ones without root)
    if [ "${DSN_LO}" = "ffffffff" ] && [ "${DSN_HI}" = "ffffffff" ]; then
        logger "rccl_dsn_mapper: WARNING: Cannot read DSN for ${BDF} (permission denied?)"
        continue
    fi

    DSN="${DSN_HI}:${DSN_LO}"
    DSN_TO_PORTS["${DSN}"]="${DSN_TO_PORTS[${DSN}]:-} ${BDF}"
    BDF_TO_DSN["${BDF}"]="${DSN}"
    UPSTREAM_COUNT=$((UPSTREAM_COUNT + 1))
done

# --- Phase 2: Build peer link pairs from DSN groups ---

declare -a LINK_PAIRS=()    # Array of "BDF_A BDF_B" pairs
MULTI_PARTITION_GROUPS=0
SOLO_GROUPS=0

for DSN in "${!DSN_TO_PORTS[@]}"; do
    # Trim leading space and split into array
    PORTS=(${DSN_TO_PORTS[$DSN]})

    if [ ${#PORTS[@]} -lt 2 ]; then
        SOLO_GROUPS=$((SOLO_GROUPS + 1))
        continue
    fi

    MULTI_PARTITION_GROUPS=$((MULTI_PARTITION_GROUPS + 1))

    # Create bidirectional peer links for all pairs within this group
    for ((i=0; i<${#PORTS[@]}; i++)); do
        for ((j=i+1; j<${#PORTS[@]}; j++)); do
            LINK_PAIRS+=("${PORTS[$i]} ${PORTS[$j]}")
        done
    done
done

# --- Phase 3: Output / Populate ---

echo "============================================================"
echo "  RCCL DSN-Based Inter-Switch-Link Mapper"
echo "  Platform: ${PLATFORM}"
echo "  Mode: ${MODE}"
echo "============================================================"
echo ""
echo "Discovery Summary:"
echo "  Broadcom PEX upstream ports found: ${UPSTREAM_COUNT}"
echo "  Non-upstream ports skipped:        ${SKIPPED_COUNT}"
echo "  Multi-partition switch groups:     ${MULTI_PARTITION_GROUPS}"
echo "  Solo-partition switch groups:      ${SOLO_GROUPS}"
echo "  Peer link pairs to create:         ${#LINK_PAIRS[@]}"
echo ""

# Print switch groups
echo "Switch Groups (by DSN):"
echo "------------------------------------------------------------"
for DSN in $(echo "${!DSN_TO_PORTS[@]}" | tr ' ' '\n' | sort); do
    PORTS=(${DSN_TO_PORTS[$DSN]})
    if [ ${#PORTS[@]} -ge 2 ]; then
        LABEL="MULTI-PARTITION"
    else
        LABEL="SOLO"
    fi
    echo "  DSN ${DSN} [${LABEL}]"
    for PORT in "${PORTS[@]}"; do
        DEV_DESC=$(lspci -s ${PORT} 2>/dev/null | sed "s/^${PORT} //")
        echo "    ${PORT}  ${DEV_DESC}"
    done
done
echo ""

# Print peer links
if [ ${#LINK_PAIRS[@]} -gt 0 ]; then
    echo "Peer Links:"
    echo "------------------------------------------------------------"
    for PAIR in "${LINK_PAIRS[@]}"; do
        read -r BDF_A BDF_B <<< "${PAIR}"
        echo "  ${BDF_A}  <-->  ${BDF_B}  (DSN: ${BDF_TO_DSN[$BDF_A]})"
    done
    echo ""
fi

# Populate sysfs if requested
if [ "${MODE}" = "populate" ]; then
    if [ ${#LINK_PAIRS[@]} -eq 0 ]; then
        echo "No peer links to create. Nothing to populate."
        exit 0
    fi

    echo "Populating sysfs at: ${SYSFS_BASE}"
    echo "Change log:          ${CHANGE_LOG}"
    echo "------------------------------------------------------------"

    # Initialize change log (append mode — preserves entries from previous runs)
    # Header with timestamp for this run
    TIMESTAMP=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    echo "# --- populate run: ${TIMESTAMP} sysfs_base=${SYSFS_BASE} ---" >> "${CHANGE_LOG}"

    # Create base directory if it doesn't exist
    if [ ! -d "${SYSFS_BASE}" ]; then
        mkdir -p "${SYSFS_BASE}"
        echo "${TIMESTAMP} CREATE_DIR ${SYSFS_BASE}" >> "${CHANGE_LOG}"
        logger "rccl_dsn_mapper: Created base directory ${SYSFS_BASE}"
        echo "  Created base directory: ${SYSFS_BASE}"
    fi

    CREATED=0
    EXISTED=0
    for PAIR in "${LINK_PAIRS[@]}"; do
        read -r BDF_A BDF_B <<< "${PAIR}"

        # Create A -> B link
        LINK_DIR="${SYSFS_BASE}/${BDF_A}"
        LINK_ENTRY="${LINK_DIR}/${BDF_B}"
        if [ ! -e "${LINK_ENTRY}" ]; then
            if [ ! -d "${LINK_DIR}" ]; then
                mkdir -p "${LINK_DIR}"
                echo "${TIMESTAMP} CREATE_DIR ${LINK_DIR}" >> "${CHANGE_LOG}"
            fi
            touch "${LINK_ENTRY}"
            echo "${TIMESTAMP} CREATE_FILE ${LINK_ENTRY}" >> "${CHANGE_LOG}"
            CREATED=$((CREATED + 1))
            logger "rccl_dsn_mapper: Created link ${BDF_A} -> ${BDF_B}"
        else
            EXISTED=$((EXISTED + 1))
        fi

        # Create B -> A link (bidirectional)
        LINK_DIR="${SYSFS_BASE}/${BDF_B}"
        LINK_ENTRY="${LINK_DIR}/${BDF_A}"
        if [ ! -e "${LINK_ENTRY}" ]; then
            if [ ! -d "${LINK_DIR}" ]; then
                mkdir -p "${LINK_DIR}"
                echo "${TIMESTAMP} CREATE_DIR ${LINK_DIR}" >> "${CHANGE_LOG}"
            fi
            touch "${LINK_ENTRY}"
            echo "${TIMESTAMP} CREATE_FILE ${LINK_ENTRY}" >> "${CHANGE_LOG}"
            CREATED=$((CREATED + 1))
            logger "rccl_dsn_mapper: Created link ${BDF_B} -> ${BDF_A}"
        else
            EXISTED=$((EXISTED + 1))
        fi
    done

    echo "  Links created:  ${CREATED}"
    echo "  Already existed: ${EXISTED}"
    echo ""
    echo "Verification:"
    for DIR in "${SYSFS_BASE}"/*/; do
        if [ -d "${DIR}" ]; then
            BDF=$(basename "${DIR}")
            PEERS=$(ls "${DIR}" 2>/dev/null | tr '\n' ' ')
            echo "  ${BDF} -> [${PEERS}]"
        fi
    done
    echo ""
    echo "Change log written to: ${CHANGE_LOG}"
    echo "  To undo: sudo $0 --clean --log-file ${CHANGE_LOG}"
fi

echo ""
echo "Done."
exit 0
