#!/bin/bash
#
# rccl_bcm_isl_discover.sh — Build inter-switch-link mapping for RCCL topology inference
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
#   sudo ./rccl_bcm_isl_discover.sh [--discover] [--populate] [--clean] [--sysfs-base PATH] [--log-file PATH] [--rccl-conf PATH] [--no-conf]
#
# Modes:
#   --discover     Print discovered switch groups and peer links (default)
#   --populate     Create the sysfs directory structure for RCCL consumption
#                  and register the path in /var/run/rccl.conf
#   --clean        Remove all files/directories previously created by --populate
#                  and remove the RCCL_BCM_LINKS_PATH entry from rccl.conf
#
# Options:
#   --sysfs-base   Override the base path (default: /var/run/rccl_bcm_links)
#   --log-file     Override the change log path (default: /var/run/rccl_bcm_isl_discover.log)
#   --rccl-conf    Override RCCL config file path (default: /var/run/rccl.conf)
#   --no-conf      Skip updating rccl.conf
#
# Modeled after /apps/shared/disable_acs.sh — intended to run as a privileged
# step during NIC bring-up, before RCCL initializes.
#

set -euo pipefail

# --- Configuration ---
SYSFS_BASE="/var/run/rccl_bcm_links"
CHANGE_LOG="/var/run/rccl_bcm_isl_discover.log"
RCCL_CONF="/var/run/rccl.conf"
UPDATE_CONF=1          # 1 = update rccl.conf, 0 = skip
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
        --rccl-conf)  RCCL_CONF="$2"; shift 2 ;;
        --no-conf)    UPDATE_CONF=0; shift ;;
        -h|--help)
            echo "Usage: $0 [--discover] [--populate] [--clean] [--sysfs-base PATH] [--log-file PATH] [--rccl-conf PATH] [--no-conf]"
            echo ""
            echo "Modes:"
            echo "  --discover     Print discovered switch groups and peer links (default)"
            echo "  --populate     Create sysfs directory structure for RCCL and register path in"
            echo "                 /var/run/rccl.conf (default: ${RCCL_CONF})"
            echo "  --clean        Remove all files/directories created by a previous --populate and"
            echo "                 remove the RCCL_BCM_LINKS_PATH entry from rccl.conf"
            echo ""
            echo "Options:"
            echo "  --sysfs-base   Override base path (default: /var/run/rccl_bcm_links)"
            echo "  --log-file     Override change log path (default: /var/run/rccl_bcm_isl_discover.log)"
            echo "  --rccl-conf    Override RCCL config file path (default: /var/run/rccl.conf)"
            echo "  --no-conf      Skip updating rccl.conf (path must be passed via env var instead)"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Usage: $0 [--discover] [--populate] [--clean] [--sysfs-base PATH] [--log-file PATH] [--rccl-conf PATH] [--no-conf]"
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
    echo "  RCCL BCM ISL Discover — Clean Mode"
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
            logger "rccl_bcm_isl_discover: Cleaned file ${ENTRY_PATH}"
            REMOVED=$((REMOVED + 1))
        else
            echo "  Already absent: ${ENTRY_PATH}"
        fi
    done < "${CHANGE_LOG}"

    # Pass 2: remove directories (in reverse order so children are removed before parents)
    while IFS=' ' read -r TIMESTAMP ACTION ENTRY_PATH; do
        if [ "${ACTION}" != "CREATE_DIR" ]; then
            continue
        fi
        if [ -d "${ENTRY_PATH}" ]; then
            # Only remove if empty (safety check)
            if [ -z "$(ls -A "${ENTRY_PATH}" 2>/dev/null)" ]; then
                rmdir "${ENTRY_PATH}"
                echo "  Removed dir:  ${ENTRY_PATH}"
                logger "rccl_bcm_isl_discover: Cleaned directory ${ENTRY_PATH}"
                REMOVED=$((REMOVED + 1))
            else
                echo "  Skipped non-empty dir: ${ENTRY_PATH}"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "  Already absent: ${ENTRY_PATH}"
        fi
    done < <(tac "${CHANGE_LOG}")

    # Remove RCCL_BCM_LINKS_PATH from rccl.conf
    if [ "${UPDATE_CONF}" -eq 1 ] && [ -f "${RCCL_CONF}" ]; then
        if grep -q '^RCCL_BCM_LINKS_PATH=' "${RCCL_CONF}" 2>/dev/null; then
            sed -i '/^RCCL_BCM_LINKS_PATH=/d' "${RCCL_CONF}"
            # Remove the marker comment if present
            sed -i '/^# Added by rccl_bcm_isl_discover/d' "${RCCL_CONF}"
            echo "  Removed RCCL_BCM_LINKS_PATH from ${RCCL_CONF}"
            logger "rccl_bcm_isl_discover: Removed RCCL_BCM_LINKS_PATH from ${RCCL_CONF}"
        else
            echo "  No RCCL_BCM_LINKS_PATH entry in ${RCCL_CONF}"
        fi
    fi

    echo ""
    echo "  Entries removed: ${REMOVED}"
    echo "  Skipped:         ${FAILED}"
    # Keep the change log when entries were skipped so a subsequent
    # --clean can retry; only remove it when everything was cleaned.
    if [ "${FAILED}" -eq 0 ]; then
        rm -f "${CHANGE_LOG}"
        echo "  Change log removed: ${CHANGE_LOG}"
    else
        echo "  Change log kept:    ${CHANGE_LOG} (${FAILED} entries remain)"
    fi
    echo ""
    echo "Done."
    exit 0
fi

# --- Platform identification ---
PLATFORM=$(dmidecode --string system-product-name 2>/dev/null || echo "unknown")
logger "rccl_bcm_isl_discover: PLATFORM=${PLATFORM}"

# --- Phase 1: Discover Broadcom upstream ports and read DSN ---

declare -A DSN_TO_PORTS     # DSN string -> space-separated upstream BDFs
declare -A BDF_TO_DSN       # BDF -> DSN string
UPSTREAM_COUNT=0
SKIPPED_COUNT=0

for BDF in $(lspci -D -d ${BCM_VENDOR}:${BCM_SWITCH_DEV} | awk '{print $1}'); do
    # Check if this is an Upstream Port (skip downstream ports, mgmt endpoints)
    PORT_TYPE=$(lspci -vvs ${BDF} 2>/dev/null | grep -oP 'Express \(v2\) \K(Upstream Port|Downstream Port)' | head -1 || true)
    if [ "${PORT_TYPE}" != "Upstream Port" ]; then
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi

    # Walk the PCIe extended capability linked list to find the DSN
    # capability (ID 0x0003).  Extended caps start at offset 0x100;
    # each header is a 32-bit word: bits[15:0] = cap ID, bits[31:20] =
    # next-cap offset.  If AER (0x0001) or another cap sits first,
    # reading 0x104/0x108 blindly would return its registers — identical
    # across all ports — collapsing every port into one false DSN group.
    DSN_CAP_OFFSET=""
    CAP_OFFSET="100"
    while [ -n "${CAP_OFFSET}" ] && [ "${CAP_OFFSET}" != "000" ]; do
        CAP_HDR=$(setpci -s ${BDF} 0x${CAP_OFFSET}.L 2>/dev/null) || break
        CAP_ID=$((16#${CAP_HDR} & 0xFFFF))
        if [ "${CAP_ID}" -eq 3 ]; then
            DSN_CAP_OFFSET="${CAP_OFFSET}"
            break
        fi
        # Next-cap pointer is bits[31:20], which gives a 12-bit offset
        NEXT=$((16#${CAP_HDR} >> 20 & 0xFFF))
        if [ "${NEXT}" -eq 0 ]; then
            break
        fi
        CAP_OFFSET=$(printf '%03x' ${NEXT})
    done

    if [ -z "${DSN_CAP_OFFSET}" ]; then
        logger "rccl_bcm_isl_discover: WARNING: ${BDF} has no DSN capability in ext-cap list — skipping"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi

    # Read DSN payload: lower 32b at cap+4, upper 32b at cap+8
    DSN_DATA_LO=$(printf '%03x' $(( 16#${DSN_CAP_OFFSET} + 4 )))
    DSN_DATA_HI=$(printf '%03x' $(( 16#${DSN_CAP_OFFSET} + 8 )))
    DSN_LO=$(setpci -s ${BDF} 0x${DSN_DATA_LO}.L 2>/dev/null) || continue
    DSN_HI=$(setpci -s ${BDF} 0x${DSN_DATA_HI}.L 2>/dev/null) || continue

    # Skip if DSN read failed (returns all-ones without root)
    if [ "${DSN_LO}" = "ffffffff" ] && [ "${DSN_HI}" = "ffffffff" ]; then
        logger "rccl_bcm_isl_discover: WARNING: Cannot read DSN for ${BDF} (permission denied?)"
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
        logger "rccl_bcm_isl_discover: Created base directory ${SYSFS_BASE}"
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
            logger "rccl_bcm_isl_discover: Created link ${BDF_A} -> ${BDF_B}"
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
            logger "rccl_bcm_isl_discover: Created link ${BDF_B} -> ${BDF_A}"
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

    # Register RCCL_BCM_LINKS_PATH in rccl.conf so RCCL discovers the
    # links automatically without requiring an env var in mpirun/launcher.
    # Uses setenv(..., 0) semantics — env var override still takes precedence.
    if [ "${UPDATE_CONF}" -eq 1 ]; then
        CONF_ENTRY="RCCL_BCM_LINKS_PATH=${SYSFS_BASE}"
        if [ -f "${RCCL_CONF}" ] && grep -q "^RCCL_BCM_LINKS_PATH=" "${RCCL_CONF}" 2>/dev/null; then
            # Update existing entry if path changed
            EXISTING=$(grep "^RCCL_BCM_LINKS_PATH=" "${RCCL_CONF}" | head -1)
            if [ "${EXISTING}" = "${CONF_ENTRY}" ]; then
                echo ""
                echo "RCCL config: ${RCCL_CONF} already contains ${CONF_ENTRY}"
            else
                sed -i "s|^RCCL_BCM_LINKS_PATH=.*|${CONF_ENTRY}|" "${RCCL_CONF}"
                echo ""
                echo "RCCL config: Updated ${RCCL_CONF} with ${CONF_ENTRY}"
                logger "rccl_bcm_isl_discover: Updated ${RCCL_CONF} with ${CONF_ENTRY}"
            fi
        else
            # Append new entry
            echo "# Added by rccl_bcm_isl_discover ($(date -u '+%Y-%m-%dT%H:%M:%SZ'))" >> "${RCCL_CONF}"
            echo "${CONF_ENTRY}" >> "${RCCL_CONF}"
            echo ""
            echo "RCCL config: Added ${CONF_ENTRY} to ${RCCL_CONF}"
            logger "rccl_bcm_isl_discover: Added ${CONF_ENTRY} to ${RCCL_CONF}"
        fi
    fi
fi

echo ""
echo "Done."
exit 0
