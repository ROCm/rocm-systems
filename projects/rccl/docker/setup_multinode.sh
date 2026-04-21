#!/bin/bash
#
# setup_multinode.sh - Build and launch multi-node ROCm containers
#
# Single-command workflow:
#   1. Creates host shared directories (SSH keys only with --ssh-key or --ssh-keygen)
#   2. Auto-detects SLURM allocation and generates hostfile (if missing)
#   3. Builds the Dockerfile.Multinode.Ubuntu image on top of any ROCm base
#   4. Optionally launches a container with all multi-node plumbing
#   5. Optionally verifies SSH connectivity across nodes
#
# Usage:
#   ./setup_multinode.sh [OPTIONS] [ROCM_IMAGE]
#
# Options:
#   --run                 Launch a container after building
#   --name NAME           Container name              (default: rccl-mn)
#   --gpus N              Number of GPUs               (default: auto-detect)
#   --ssh-port PORT       SSH port inside container    (default: 2224)
#   --shm-size SIZE       Shared memory size           (default: 64g)
#   --hostfile PATH       MPI hostfile on the host     (default: ~/.mpi_hostfile; auto from SLURM)
#   --volume SRC:DST      Extra host volume mount      (repeatable)
#   --shared-dir PATH     Shared workspace on host     (default: ~/.docker-shared)
#   --builds-dir PATH     Shared builds dir on host    (default: ~/.docker-builds)
#   --ssh-key-dir PATH    SSH key directory on host    (default: ~/.docker-ssh-keys)
#   --ssh-key PATH        Use existing SSH key pair for inter-container SSH
#   --ssh-authorized-keys PATH  Custom authorized_keys (for mesh SSH setups)
#   --ssh-keygen          Auto-generate a shared SSH key pair
#   --rebuild             Force image rebuild and replace existing containers
#   --run-only            Skip build, just launch the container
#   --launch-all          Build + launch on ALL nodes in the hostfile via SSH
#   --stop-all            Stop + remove containers on ALL nodes in the hostfile
#   --setup-deps          Build shared deps (UCX, OpenMPI) into shared dir (once)
#   --post-setup PATH     Post-setup dir with setup.sh/env.sh   (optional)
#   --verify              Verify SSH connectivity to all hosts in hostfile
#   --host-ssh-port PORT  SSH port for host-to-host access  (default: 22)
#   --verbose             Enable detailed debug logging
#   --help                Show this help message
#
# Environment Variables (alternative to flags):
#   SHARED_DIR, BUILDS_DIR, SSH_KEY_DIR, SSH_KEY, SSH_AUTHORIZED_KEYS, HOSTFILE, SSH_PORT, GPUS,
#   POST_SETUP_DIR, VERBOSE
#
# Path expansion:
#   All path options (--hostfile, --shared-dir, --builds-dir, --ssh-key-dir,
#   --ssh-key, --ssh-authorized-keys, --volume) support ~ and $VAR / ${VAR} expansion. Examples:
#     --hostfile '~/my_hostfile'
#     --shared-dir '$HOME/shared'
#     HOSTFILE='~/.mpi_hostfile' ./setup_multinode.sh --launch-all
#
# Examples:
#   ./setup_multinode.sh                                          # build default
#   ./setup_multinode.sh rocm/dev-ubuntu-24.04:7.1.1-complete     # specific ROCm
#   ./setup_multinode.sh --run --gpus 16                          # build + launch
#   ./setup_multinode.sh --run --volume /data:/data --name node0  # extra mount
#   ./setup_multinode.sh --run-only rocm-multinode:7.1.1-complete # launch existing
#   ./setup_multinode.sh --verify                                 # check SSH
#   ./setup_multinode.sh --run --verbose                          # full debug output
#   ./setup_multinode.sh --launch-all --ssh-keygen                 # build+launch, auto SSH keys
#   ./setup_multinode.sh --launch-all --ssh-key ~/.ssh/id_rsa     # build+launch, use your keys
#   ./setup_multinode.sh --launch-all --ssh-key ~/.ssh/id_rsa \
#                        --ssh-authorized-keys ~/.ssh/authorized_keys  # mesh SSH
#   ./setup_multinode.sh --setup-deps                             # build shared UCX/MPI (once)
#   ./setup_multinode.sh --stop-all                               # stop containers everywhere
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Defaults
# ============================================================================
ROCM_IMAGE="${ROCM_IMAGE:-rocm/dev-ubuntu-24.04:7.1.1-complete}"
CONTAINER_NAME="${CONTAINER_NAME:-rccl-mn}"
SSH_PORT="${SSH_PORT:-2224}"
SHM_SIZE="${SHM_SIZE:-64g}"
DO_RUN=false
DO_BUILD=true
DO_VERIFY=false
DO_LAUNCH_ALL=false
DO_STOP_ALL=false
DO_SETUP_DEPS=false
FORCE_REBUILD=false
VERBOSE="${VERBOSE:-}"
HOST_SSH_PORT="${HOST_SSH_PORT:-22}"

SHARED_DIR="${SHARED_DIR:-${HOME}/.docker-shared}"
BUILDS_DIR="${BUILDS_DIR:-${HOME}/.docker-builds}"
SSH_KEY_DIR="${SSH_KEY_DIR:-${HOME}/.docker-ssh-keys}"
HOSTFILE="${HOSTFILE:-${HOME}/.mpi_hostfile}"
POST_SETUP_DIR="${POST_SETUP_DIR:-}"
SSH_KEY="${SSH_KEY:-}"
SSH_AUTHORIZED_KEYS="${SSH_AUTHORIZED_KEYS:-}"
DO_SSH_KEYGEN=false

EXTRA_VOLUMES=()

log_verbose() {
    [[ -n "${VERBOSE}" ]] && echo "  [verbose] $*" || true
}

# Timer helpers for verbose mode
_timer_start=0
timer_start() {
    _timer_start=$(date +%s)
}
timer_end() {
    local elapsed=$(( $(date +%s) - _timer_start ))
    local m=$((elapsed / 60)) s=$((elapsed % 60))
    log_verbose "⏱ $1 completed in ${m}m ${s}s"
}

# Expand ~ and environment variables in a path string.
# Handles ~/..., $VAR, and ${VAR} patterns.
# Uses envsubst instead of eval to avoid command injection.
expand_path() {
    local p="$1"
    # Tilde expansion (envsubst doesn't handle ~)
    if [[ "$p" == "~/"* ]]; then
        p="${HOME}/${p#\~/}"
    elif [[ "$p" == "~" ]]; then
        p="${HOME}"
    fi
    # Safe environment variable expansion via envsubst
    echo "$p" | envsubst
}

auto_detect_gpus() {
    local count=0
    if [[ -d /sys/class/kfd/kfd/topology/nodes ]]; then
        for node in /sys/class/kfd/kfd/topology/nodes/*/gpu_id; do
            [[ -f "$node" ]] && [[ "$(cat "$node")" != "0" ]] && ((count++))
        done
    fi
    if [[ "${count}" -eq 0 ]] && [[ -d /dev/dri ]]; then
        count=$(ls -1 /dev/dri/renderD* 2>/dev/null | wc -l)
    fi
    if [[ "${count}" -eq 0 ]]; then
        echo "WARNING: Could not detect GPU count; set --gpus manually" >&2
        echo 0
    else
        echo "${count}"
    fi
}

GPUS="${GPUS:-$(auto_detect_gpus)}"

# ============================================================================
# Argument parsing
# ============================================================================
show_help() {
    sed -n '/^# Usage:/,/^[^#]/{ /^#/{ s/^# \?//; p } }' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)          DO_RUN=true;             shift ;;
        --run-only)     DO_RUN=true; DO_BUILD=false; shift ;;
        --verify)       DO_VERIFY=true; DO_BUILD=false; shift ;;
        --name)         CONTAINER_NAME="$2";     shift 2 ;;
        --gpus)         GPUS="$2"; GPUS_EXPLICIT=true; shift 2 ;;
        --ssh-port)     SSH_PORT="$2";           shift 2 ;;
        --shm-size)     SHM_SIZE="$2";           shift 2 ;;
        --hostfile)     HOSTFILE="$2";           shift 2 ;;
        --volume|-v)    EXTRA_VOLUMES+=("$2");   shift 2 ;;
        --shared-dir)   SHARED_DIR="$2";         shift 2 ;;
        --builds-dir)   BUILDS_DIR="$2";         shift 2 ;;
        --ssh-key-dir)  SSH_KEY_DIR="$2";        shift 2 ;;
        --rebuild)      FORCE_REBUILD=true;      shift ;;
        --launch-all)   DO_LAUNCH_ALL=true;      shift ;;
        --stop-all)     DO_STOP_ALL=true; DO_BUILD=false; shift ;;
        --setup-deps)   DO_SETUP_DEPS=true;      shift ;;
        --post-setup)   POST_SETUP_DIR="$2";     shift 2 ;;
        --ssh-key)      SSH_KEY="$2";            shift 2 ;;
        --ssh-authorized-keys) SSH_AUTHORIZED_KEYS="$2"; shift 2 ;;
        --ssh-keygen)   DO_SSH_KEYGEN=true;      shift ;;
        --host-ssh-port) HOST_SSH_PORT="$2";     shift 2 ;;
        --verbose)      VERBOSE=1;               shift ;;
        --help|-h)      show_help ;;
        -*)             echo "Unknown option: $1" >&2; exit 1 ;;
        *)              ROCM_IMAGE="$1";         shift ;;
    esac
done

# ============================================================================
# Expand ~ and $VAR in all path-valued options
# ============================================================================
SHARED_DIR="$(expand_path "${SHARED_DIR}")"
BUILDS_DIR="$(expand_path "${BUILDS_DIR}")"
SSH_KEY_DIR="$(expand_path "${SSH_KEY_DIR}")"
HOSTFILE="$(expand_path "${HOSTFILE}")"
[[ -n "${POST_SETUP_DIR}" ]] && POST_SETUP_DIR="$(expand_path "${POST_SETUP_DIR}")"
[[ -n "${SSH_KEY}" ]] && SSH_KEY="$(expand_path "${SSH_KEY}")"
[[ -n "${SSH_AUTHORIZED_KEYS}" ]] && SSH_AUTHORIZED_KEYS="$(expand_path "${SSH_AUTHORIZED_KEYS}")"
expanded_vols=()
for vol in "${EXTRA_VOLUMES[@]+"${EXTRA_VOLUMES[@]}"}"; do
    expanded_vols+=("$(expand_path "$vol")")
done
EXTRA_VOLUMES=("${expanded_vols[@]+"${expanded_vols[@]}"}")

# ============================================================================
# Auto-detect SLURM allocation: generate hostfile + configure SSH keys
# ============================================================================
detect_slurm_nodes() {
    if [[ -f "${HOSTFILE}" ]]; then
        return
    fi

    local nodelist="${SLURM_NODELIST:-${SLURM_JOB_NODELIST:-}}"
    if [[ -z "${nodelist}" ]]; then
        return
    fi

    if ! command -v scontrol &>/dev/null; then
        echo "WARNING: SLURM allocation detected (SLURM_NODELIST=${nodelist})" >&2
        echo "  but 'scontrol' not found in PATH; cannot expand node list" >&2
        echo "  Install slurm-client or create a hostfile manually" >&2
        return
    fi

    # --- Determine slots per node ---
    local slots="${GPUS}"
    if [[ "${slots}" -eq 0 ]]; then
        local slurm_gpus="${SLURM_GPUS_PER_NODE:-}"
        if [[ -n "${slurm_gpus}" ]]; then
            slurm_gpus="${slurm_gpus##*:}"
            slurm_gpus="${slurm_gpus%%\(*}"
            [[ "${slurm_gpus}" =~ ^[0-9]+$ ]] && slots="${slurm_gpus}"
        fi
    fi
    if [[ "${slots}" -eq 0 && -n "${SLURM_NTASKS_PER_NODE:-}" ]]; then
        slots="${SLURM_NTASKS_PER_NODE}"
    fi
    if [[ "${slots}" -eq 0 ]]; then
        slots=1
        echo "WARNING: Could not determine GPU/slot count from SLURM; defaulting to slots=1" >&2
    fi

    echo "=== SLURM allocation detected ==="
    echo "  SLURM_NODELIST : ${nodelist}"
    echo "  SLURM_NNODES   : ${SLURM_NNODES:-unknown}"
    echo "  Slots per node : ${slots}"
    log_verbose "SLURM_JOB_ID=${SLURM_JOB_ID:-} SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-} SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-}"

    # --- Generate hostfile ---
    local hosts
    hosts=$(scontrol show hostnames "${nodelist}")

    mkdir -p "$(dirname "${HOSTFILE}")"
    : > "${HOSTFILE}"
    while IFS= read -r host; do
        [[ -n "${host}" ]] && echo "${host} slots=${slots}" >> "${HOSTFILE}"
    done <<< "${hosts}"

    echo "  Generated hostfile: ${HOSTFILE}"
    if [[ -n "${VERBOSE}" ]]; then
        log_verbose "Hostfile contents:"
        while read -r line; do
            log_verbose "  ${line}"
        done < "${HOSTFILE}"
    fi

    # --- Auto-configure SSH keys from the user's ~/.ssh ---
    if [[ -n "${SSH_KEY}" ]] || [[ "${DO_SSH_KEYGEN}" == true ]]; then
        log_verbose "SSH key explicitly provided; skipping SLURM SSH auto-detect"
    elif [[ -f "${HOME}/.ssh/id_rsa" ]]; then
        SSH_KEY="${HOME}/.ssh/id_rsa"
        echo "  SSH key        : ${SSH_KEY} (auto-detected)"
        if [[ -f "${HOME}/.ssh/authorized_keys" ]]; then
            SSH_AUTHORIZED_KEYS="${HOME}/.ssh/authorized_keys"
            echo "  authorized_keys: ${SSH_AUTHORIZED_KEYS} (auto-detected)"
        fi
    elif [[ -f "${HOME}/.ssh/id_ed25519" ]]; then
        SSH_KEY="${HOME}/.ssh/id_ed25519"
        echo "  SSH key        : ${SSH_KEY} (auto-detected)"
        if [[ -f "${HOME}/.ssh/authorized_keys" ]]; then
            SSH_AUTHORIZED_KEYS="${HOME}/.ssh/authorized_keys"
            echo "  authorized_keys: ${SSH_AUTHORIZED_KEYS} (auto-detected)"
        fi
    else
        DO_SSH_KEYGEN=true
        echo "  SSH key        : (none found at ~/.ssh/id_rsa; will auto-generate)"
    fi

    echo ""
}

detect_slurm_nodes

# ============================================================================
# Validate option combinations
# ============================================================================
validate_options() {
    local errors=()
    local warnings=()

    # Count mutually exclusive action modes
    local action_count=0
    local actions=()

    if [[ "${DO_VERIFY}" == true ]]; then
        action_count=$((action_count + 1)); actions+=("--verify")
    fi
    if [[ "${DO_STOP_ALL}" == true ]]; then
        action_count=$((action_count + 1)); actions+=("--stop-all")
    fi
    if [[ "${DO_LAUNCH_ALL}" == true ]]; then
        action_count=$((action_count + 1)); actions+=("--launch-all")
    fi
    if [[ "${DO_SETUP_DEPS}" == true ]]; then
        action_count=$((action_count + 1)); actions+=("--setup-deps")
    fi
    # --run and --run-only share DO_RUN; only count once
    if [[ "${DO_RUN}" == true && "${DO_LAUNCH_ALL}" == false ]]; then
        action_count=$((action_count + 1))
        if [[ "${DO_BUILD}" == false ]]; then
            actions+=("--run-only")
        else
            actions+=("--run")
        fi
    fi

    # --- Mutually exclusive actions ---
    if [[ "${DO_VERIFY}" == true && "${action_count}" -gt 1 ]]; then
        local others=()
        for a in "${actions[@]}"; do [[ "$a" != "--verify" ]] && others+=("$a"); done
        errors+=("--verify is a standalone check; cannot be combined with ${others[*]}")
    fi
    if [[ "${DO_STOP_ALL}" == true && ("${DO_RUN}" == true || "${DO_LAUNCH_ALL}" == true || "${DO_SETUP_DEPS}" == true) ]]; then
        errors+=("--stop-all cannot be combined with --run, --launch-all, or --setup-deps")
    fi
    if [[ "${DO_LAUNCH_ALL}" == true && "${DO_RUN}" == true ]]; then
        errors+=("--launch-all already launches containers on all nodes; do not combine with --run or --run-only")
    fi
    if [[ "${DO_LAUNCH_ALL}" == true && "${DO_SETUP_DEPS}" == true ]]; then
        warnings+=("--setup-deps is redundant with --launch-all (shared deps are built automatically)")
    fi

    # --- Hostfile required ---
    if [[ "${DO_LAUNCH_ALL}" == true || "${DO_STOP_ALL}" == true || "${DO_VERIFY}" == true ]]; then
        if [[ ! -f "${HOSTFILE}" ]]; then
            errors+=("Hostfile not found: ${HOSTFILE}
  Required by: ${actions[*]}
  Create it:   echo 'hostname slots=8' > ${HOSTFILE}
  Or specify:  --hostfile /path/to/hostfile
  Or run inside a SLURM allocation (auto-detected from SLURM_NODELIST)")
        fi
    fi

    # --- --rebuild without a build/run/launch action ---
    if [[ "${FORCE_REBUILD}" == true ]]; then
        if [[ "${DO_RUN}" == false && "${DO_LAUNCH_ALL}" == false && "${DO_SETUP_DEPS}" == false && "${DO_BUILD}" == true ]]; then
            # build-only mode: rebuild is fine (forces image rebuild)
            :
        elif [[ "${DO_VERIFY}" == true || "${DO_STOP_ALL}" == true ]]; then
            warnings+=("--rebuild has no effect with ${actions[*]}; it only applies to build/run/launch actions")
        fi
    fi

    # --- --run-only without a pre-built image ---
    if [[ "${DO_RUN}" == true && "${DO_BUILD}" == false ]]; then
        if ! docker image inspect "rocm-multinode:${ROCM_IMAGE##*:}" &>/dev/null 2>&1; then
            errors+=("--run-only requires a pre-built image, but 'rocm-multinode:${ROCM_IMAGE##*:}' was not found
  Build first: $0 ${ROCM_IMAGE}
  Or use:      $0 --run ${ROCM_IMAGE}")
        fi
    fi

    # --- SSH key validation ---
    if [[ -n "${SSH_KEY}" ]] && [[ "${DO_SSH_KEYGEN}" == true ]]; then
        errors+=("--ssh-key and --ssh-keygen are mutually exclusive")
    fi
    if [[ -n "${SSH_KEY}" ]]; then
        local priv_key pub_key
        if [[ "${SSH_KEY}" == *.pub ]]; then
            pub_key="${SSH_KEY}"; priv_key="${SSH_KEY%.pub}"
        else
            priv_key="${SSH_KEY}"; pub_key="${SSH_KEY}.pub"
        fi
        [[ ! -f "${priv_key}" ]] && errors+=("SSH private key not found: ${priv_key}")
        [[ ! -f "${pub_key}" ]]  && errors+=("SSH public key not found: ${pub_key}")
    fi
    if [[ -n "${SSH_AUTHORIZED_KEYS}" ]]; then
        [[ ! -f "${SSH_AUTHORIZED_KEYS}" ]] && errors+=("SSH authorized_keys file not found: ${SSH_AUTHORIZED_KEYS}")
        if [[ -z "${SSH_KEY}" ]] && [[ "${DO_SSH_KEYGEN}" == false ]]; then
            errors+=("--ssh-authorized-keys requires --ssh-key or --ssh-keygen (need a private key for outbound SSH)")
        fi
    fi

    # --- Post-setup validation ---
    if [[ -n "${POST_SETUP_DIR}" ]]; then
        if [[ ! -d "${POST_SETUP_DIR}" ]]; then
            errors+=("Post-setup directory not found: ${POST_SETUP_DIR}")
        elif [[ ! -f "${POST_SETUP_DIR}/setup.sh" ]] && [[ ! -f "${POST_SETUP_DIR}/env.sh" ]]; then
            errors+=("Post-setup dir must contain setup.sh and/or env.sh: ${POST_SETUP_DIR}")
        fi
    fi

    # --- Options that have no effect without their action ---
    if [[ "${GPUS_EXPLICIT:-}" == true && "${DO_RUN}" == false && "${DO_LAUNCH_ALL}" == false ]]; then
        warnings+=("--gpus has no effect without --run, --run-only, or --launch-all")
    fi

    if [[ "${DO_LAUNCH_ALL}" == false && "${DO_STOP_ALL}" == false ]]; then
        if [[ "${HOST_SSH_PORT}" != "22" ]]; then
            warnings+=("--host-ssh-port has no effect without --launch-all or --stop-all")
        fi
    fi

    # --- Print warnings ---
    for w in "${warnings[@]+"${warnings[@]}"}"; do
        echo "WARNING: ${w}" >&2
    done

    # --- Print errors and exit ---
    if [[ "${#errors[@]}" -gt 0 ]]; then
        echo "" >&2
        for e in "${errors[@]}"; do
            echo "ERROR: ${e}" >&2
        done
        echo "" >&2
        echo "Run '$0 --help' for usage information." >&2
        exit 1
    fi
}

IMAGE_TAG="rocm-multinode:${ROCM_IMAGE##*:}"

validate_options

if [[ -n "${VERBOSE}" ]]; then
    echo "=== Verbose mode enabled ==="
    log_verbose "SCRIPT_DIR=${SCRIPT_DIR}"
    log_verbose "ROCM_IMAGE=${ROCM_IMAGE}"
    log_verbose "IMAGE_TAG=${IMAGE_TAG}"
    log_verbose "CONTAINER_NAME=${CONTAINER_NAME}"
    log_verbose "GPUS=${GPUS}"
    log_verbose "SSH_PORT=${SSH_PORT}"
    log_verbose "SHM_SIZE=${SHM_SIZE}"
    log_verbose "SHARED_DIR=${SHARED_DIR}"
    log_verbose "BUILDS_DIR=${BUILDS_DIR}"
    log_verbose "SSH_KEY_DIR=${SSH_KEY_DIR}"
    log_verbose "HOSTFILE=${HOSTFILE}"
    log_verbose "POST_SETUP_DIR=${POST_SETUP_DIR:-}"
    log_verbose "SSH_KEY=${SSH_KEY:-}"
    log_verbose "SSH_AUTHORIZED_KEYS=${SSH_AUTHORIZED_KEYS:-}"
    log_verbose "DO_SSH_KEYGEN=${DO_SSH_KEYGEN}"
    log_verbose "DO_BUILD=${DO_BUILD}  DO_RUN=${DO_RUN}  DO_VERIFY=${DO_VERIFY}  DO_LAUNCH_ALL=${DO_LAUNCH_ALL}  DO_STOP_ALL=${DO_STOP_ALL}  DO_SETUP_DEPS=${DO_SETUP_DEPS}"
    log_verbose "HOST_SSH_PORT=${HOST_SSH_PORT}"
    log_verbose "FORCE_REBUILD=${FORCE_REBUILD}"
    log_verbose "EXTRA_VOLUMES=(${EXTRA_VOLUMES[*]+"${EXTRA_VOLUMES[*]}"})"
    log_verbose "Host kernel: $(uname -r)"
    log_verbose "Docker version: $(docker --version 2>/dev/null || echo 'not found')"
    echo ""
fi

# ============================================================================
# Write SSH config + set permissions for the shared key directory
# ============================================================================
write_ssh_config() {
    cat > "${SSH_KEY_DIR}/config" << EOF
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    Port ${SSH_PORT}
    IdentityFile ~/.ssh/id_rsa
EOF
    chmod 600 "${SSH_KEY_DIR}/id_rsa"
    chmod 644 "${SSH_KEY_DIR}/id_rsa.pub" "${SSH_KEY_DIR}/authorized_keys" "${SSH_KEY_DIR}/config"
}

# ============================================================================
# Host setup (shared dirs + SSH keys) - idempotent
# ============================================================================
setup_host() {
    timer_start
    echo "=== Host setup ==="
    for dir in "${SHARED_DIR}" "${BUILDS_DIR}"; do
        if [[ ! -d "${dir}" ]]; then
            mkdir -p "${dir}"
            chmod 777 "${dir}" 2>/dev/null || chmod 755 "${dir}" 2>/dev/null || true
            echo "  Created ${dir}"
        else
            echo "  Exists  ${dir}"
        fi
        log_verbose "$(ls -ld "${dir}" 2>/dev/null)"
    done

    if [[ ! -d "${SSH_KEY_DIR}" ]]; then
        mkdir -p "${SSH_KEY_DIR}"
        chmod 700 "${SSH_KEY_DIR}"
        echo "  Created ${SSH_KEY_DIR} (mode 700)"
    else
        echo "  Exists  ${SSH_KEY_DIR}"
    fi
    log_verbose "$(ls -ld "${SSH_KEY_DIR}" 2>/dev/null)"

    if [[ -f "${SSH_KEY_DIR}/id_rsa" ]]; then
        echo "  SSH keys exist at ${SSH_KEY_DIR}"
    elif [[ -n "${SSH_KEY}" ]]; then
        local priv_key pub_key
        if [[ "${SSH_KEY}" == *.pub ]]; then
            pub_key="${SSH_KEY}"; priv_key="${SSH_KEY%.pub}"
        else
            priv_key="${SSH_KEY}"; pub_key="${SSH_KEY}.pub"
        fi
        echo "  Configuring SSH keys from ${SSH_KEY}..."
        cp "${priv_key}" "${SSH_KEY_DIR}/id_rsa"
        cp "${pub_key}" "${SSH_KEY_DIR}/id_rsa.pub"
        if [[ -n "${SSH_AUTHORIZED_KEYS}" ]]; then
            cp "${SSH_AUTHORIZED_KEYS}" "${SSH_KEY_DIR}/authorized_keys"
            cat "${pub_key}" >> "${SSH_KEY_DIR}/authorized_keys"
            log_verbose "authorized_keys: merged from ${SSH_AUTHORIZED_KEYS} + ${pub_key}"
        else
            cp "${pub_key}" "${SSH_KEY_DIR}/authorized_keys"
        fi
        write_ssh_config
        echo "  SSH keys configured at ${SSH_KEY_DIR}"
    elif [[ "${DO_SSH_KEYGEN}" == true ]]; then
        echo "  Generating shared SSH keys..."
        ssh-keygen -t rsa -b 4096 -N "" -f "${SSH_KEY_DIR}/id_rsa" -C "docker-shared-key" -q
        if [[ -n "${SSH_AUTHORIZED_KEYS}" ]]; then
            cp "${SSH_AUTHORIZED_KEYS}" "${SSH_KEY_DIR}/authorized_keys"
            cat "${SSH_KEY_DIR}/id_rsa.pub" >> "${SSH_KEY_DIR}/authorized_keys"
            log_verbose "authorized_keys: merged from ${SSH_AUTHORIZED_KEYS} + generated key"
        else
            cp "${SSH_KEY_DIR}/id_rsa.pub" "${SSH_KEY_DIR}/authorized_keys"
        fi
        write_ssh_config
        echo "  Keys generated at ${SSH_KEY_DIR}"
    else
        echo "  No SSH keys configured (containers will generate local-only keys)"
        echo "  For multi-node SSH, use one of:"
        echo "    --ssh-key ~/.ssh/id_rsa                                              # shared key pair"
        echo "    --ssh-key ~/.ssh/id_rsa --ssh-authorized-keys ~/.ssh/authorized_keys # mesh SSH (per-node keys)"
        echo "    --ssh-keygen                                                         # generate a new pair"
    fi

    if [[ -n "${VERBOSE}" ]]; then
        log_verbose "SSH key dir contents:"
        ls -la "${SSH_KEY_DIR}/" 2>/dev/null | while read -r line; do
            log_verbose "  ${line}"
        done
        log_verbose "Hostfile: ${HOSTFILE} $([ -f "${HOSTFILE}" ] && echo "(exists, $(wc -l < "${HOSTFILE}") lines)" || echo "(not found)")"
    fi
    timer_end "Host setup"
    echo ""
}

# ============================================================================
# Build shared dependencies (UCX + OpenMPI) into SHARED_DIR — runs once
# ============================================================================
setup_shared_deps() {
    local ucx_dir="${SHARED_DIR}/ucx"
    local ompi_dir="${SHARED_DIR}/ompi"

    if [[ -x "${ompi_dir}/bin/mpirun" ]] && [[ "${FORCE_REBUILD}" == false ]]; then
        echo "=== Shared deps already installed (use --rebuild to force) ==="
        log_verbose "UCX: ${ucx_dir} $(ls -d "${ucx_dir}/lib/libucp.so" 2>/dev/null && echo '(ok)' || echo '(missing)')"
        log_verbose "OpenMPI: $(${ompi_dir}/bin/mpirun --version 2>/dev/null | head -1 || echo 'unknown')"
        echo ""
        return
    fi

    local log_dir="${SHARED_DIR}/logs"
    mkdir -p "${log_dir}"
    local ucx_log="${log_dir}/ucx-build.log"
    local ompi_log="${log_dir}/ompi-build.log"

    echo "=== Building shared dependencies ==="
    echo "  UCX     → ${ucx_dir}"
    echo "  OpenMPI → ${ompi_dir}"
    echo "  Logs    → ${log_dir}/"
    echo ""

    if ! docker image inspect "${IMAGE_TAG}" &>/dev/null; then
        echo "  Image ${IMAGE_TAG} not found; building first..."
        build_image
    fi

    timer_start
    echo "  Building UCX 1.16.0 (log: ${ucx_log})..."
    if ! docker run --rm \
        -v "${SHARED_DIR}:/opt/shared" \
        "${IMAGE_TAG}" \
        bash -c "set -e \
            && cd /tmp \
            && wget -q https://github.com/openucx/ucx/releases/download/v1.16.0/ucx-1.16.0.tar.gz \
            && mkdir -p ucx && tar -zxf ucx-1.16.0.tar.gz -C ucx --strip-components=1 \
            && cd ucx && mkdir build && cd build \
            && ../configure --prefix=/opt/shared/ucx --with-rocm=/opt/rocm \
               --with-verbs --with-rdmacm --enable-mt \
               --disable-examples --silent \
            && make -j\$(nproc) install \
            && echo '>>> UCX installed successfully'" \
        > "${ucx_log}" 2>&1; then
        echo "  [FAIL] UCX build failed. See ${ucx_log}"
        tail -20 "${ucx_log}"
        exit 1
    fi
    timer_end "UCX build"
    echo "  [OK] UCX installed"

    timer_start
    echo "  Building OpenMPI 4.1.6 (log: ${ompi_log})..."
    if ! docker run --rm \
        -v "${SHARED_DIR}:/opt/shared" \
        "${IMAGE_TAG}" \
        bash -c "set -e \
            && cd /tmp \
            && wget -q https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz \
            && mkdir -p ompi4 && tar -zxf openmpi-4.1.6.tar.gz -C ompi4 --strip-components=1 \
            && cd ompi4 && mkdir build && cd build \
            && ../configure --prefix=/opt/shared/ompi --with-ucx=/opt/shared/ucx \
               --disable-oshmem --disable-mpi-fortran --disable-mpi-cxx \
               --enable-orterun-prefix-by-default --silent \
            && make -j\$(nproc) install \
            && echo '>>> OpenMPI installed successfully'" \
        > "${ompi_log}" 2>&1; then
        echo "  [FAIL] OpenMPI build failed. See ${ompi_log}"
        tail -20 "${ompi_log}"
        exit 1
    fi
    timer_end "OpenMPI build"
    echo "  [OK] OpenMPI installed"

    echo ""
    echo "  Shared deps installed:"
    echo "    UCX:     ${ucx_dir}"
    echo "    OpenMPI: ${ompi_dir}"
    echo "    Logs:    ${log_dir}/"
    log_verbose "mpirun version: $(${ompi_dir}/bin/mpirun --version 2>/dev/null | head -1 || echo 'check container')"
    echo ""
}

# ============================================================================
# Build Docker image
# ============================================================================
build_image() {
    if [[ "${FORCE_REBUILD}" == false ]] && docker image inspect "${IMAGE_TAG}" &>/dev/null; then
        echo "=== Image ${IMAGE_TAG} already exists (use --rebuild to force) ==="
        log_verbose "Image details: $(docker image inspect "${IMAGE_TAG}" --format '{{.Id}} created={{.Created}} size={{.Size}}' 2>/dev/null)"
        echo ""
        return
    fi

    timer_start
    echo "=== Building image ==="
    echo "  Base  : ${ROCM_IMAGE}"
    echo "  Tag   : ${IMAGE_TAG}"
    echo ""

    local build_args=(
        --build-arg "ROCM_IMAGE=${ROCM_IMAGE}"
        --build-arg "SSH_PORT=${SSH_PORT}"
        -t "${IMAGE_TAG}"
        -f "${SCRIPT_DIR}/Dockerfile.Multinode.Ubuntu"
    )

    if [[ -n "${VERBOSE}" ]]; then
        build_args+=(--progress=plain)
        log_verbose "docker build ${build_args[*]} ${SCRIPT_DIR}"
    fi

    docker build "${build_args[@]}" "${SCRIPT_DIR}"

    timer_end "Image build"
    echo ""
    echo "  Built: ${IMAGE_TAG}"
    log_verbose "Image ID: $(docker image inspect "${IMAGE_TAG}" --format '{{.Id}}' 2>/dev/null)"
    echo ""
}

# ============================================================================
# Launch container (idempotent — skips if already running, --rebuild replaces)
# ============================================================================
launch_container() {
    if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        local state
        state=$(docker inspect "${CONTAINER_NAME}" --format '{{.State.Status}}' 2>/dev/null || echo "unknown")

        if [[ "${FORCE_REBUILD}" == true ]]; then
            echo "=== Replacing container '${CONTAINER_NAME}' (--rebuild) ==="
            docker rm -f "${CONTAINER_NAME}" &>/dev/null || true
        elif [[ "${state}" == "running" ]]; then
            echo "=== Container '${CONTAINER_NAME}' is already running (use --rebuild to replace) ==="
            log_verbose "Container ID: $(docker inspect "${CONTAINER_NAME}" --format '{{.Id}}' 2>/dev/null | head -c 12)"
            return 0
        else
            echo "=== Container '${CONTAINER_NAME}' exists but is ${state}, removing and re-launching ==="
            docker rm -f "${CONTAINER_NAME}" &>/dev/null || true
        fi
    fi

    timer_start
    echo "=== Launching container ==="
    echo "  Image     : ${IMAGE_TAG}"
    echo "  Container : ${CONTAINER_NAME}"
    echo "  GPUs      : ${GPUS}"
    echo "  SSH port  : ${SSH_PORT}"
    echo "  SHM size  : ${SHM_SIZE}"
    echo ""

    local run_args=(
        -d
        --name "${CONTAINER_NAME}"
        --privileged
        --security-opt apparmor=unconfined
        --security-opt seccomp=unconfined
        --restart unless-stopped
        --group-add video
        --group-add "$(getent group render 2>/dev/null | cut -d: -f3 || echo 109)"
        --cap-add SYS_PTRACE
        --network host
        --ipc host
        --shm-size "${SHM_SIZE}"
        --ulimit memlock=-1

        -e GPUS="${GPUS}"
        -e HOST_UID="$(id -u)"
        -e HOST_GID="$(id -g)"
        -e RENDER_GID="$(getent group render 2>/dev/null | cut -d: -f3 || echo 109)"
    )

    # Pass verbose flag into the container for entrypoint logging
    [[ -n "${VERBOSE}" ]] && run_args+=(-e VERBOSE=1)

    [[ -e /dev/kfd ]] && run_args+=(--device /dev/kfd)
    [[ -d /dev/dri ]] && run_args+=(--device /dev/dri)

    if [[ -d /dev/infiniband ]]; then
        run_args+=(--device /dev/infiniband:/dev/infiniband)
        for dev in /dev/infiniband/*; do
            [[ -e "$dev" ]] && run_args+=(--device "$dev:$dev")
        done
        log_verbose "InfiniBand devices: $(ls /dev/infiniband/ 2>/dev/null | tr '\n' ' ')"

        # Bind-mount host MLNX_OFED / rdma-core libraries so the container
        # uses the exact same ibverbs stack as the host kernel drivers.
        # Mount resolved (versioned) .so files plus symlinks and provider dir.
        local ib_lib_dir="/usr/lib/x86_64-linux-gnu"
        for lib in libibverbs libmlx5 libmlx4 libefa; do
            for f in "${ib_lib_dir}/${lib}".so*; do
                [[ -e "$f" ]] && run_args+=(-v "$f:$f:ro")
            done
        done
        local ib_provider="${ib_lib_dir}/libibverbs"
        [[ -d "${ib_provider}" ]] && run_args+=(-v "${ib_provider}:${ib_provider}:ro")
        [[ -d /etc/libibverbs.d ]] && run_args+=(-v "/etc/libibverbs.d:/etc/libibverbs.d:ro")
        log_verbose "RDMA libs bind-mounted from host"
    else
        log_verbose "No InfiniBand devices found at /dev/infiniband"
    fi

    [[ -f "${HOSTFILE}" ]] && run_args+=(-v "${HOSTFILE}:${HOSTFILE}:ro")

    run_args+=(
        -v "${SHARED_DIR}:/opt/shared"
        -v "${BUILDS_DIR}:/opt/builds"
        -v "${SSH_KEY_DIR}:/opt/ssh-keys:ro"
    )

    if [[ -n "${POST_SETUP_DIR}" ]]; then
        run_args+=(-v "${POST_SETUP_DIR}:/opt/post-setup:ro")
    fi

    for vol in "${EXTRA_VOLUMES[@]+"${EXTRA_VOLUMES[@]}"}"; do
        run_args+=(-v "$vol")
    done

    if [[ -n "${VERBOSE}" ]]; then
        log_verbose "docker run arguments:"
        for arg in "${run_args[@]}"; do
            log_verbose "  ${arg}"
        done
    fi

    docker run "${run_args[@]}" "${IMAGE_TAG}"
    timer_end "Container launch"

    echo ""
    echo "=== Container '${CONTAINER_NAME}' is running ==="
    echo ""
    echo "  Shell (root)  : docker exec -it ${CONTAINER_NAME} bash"
    echo "  Shell (ubuntu): docker exec -it -u ubuntu ${CONTAINER_NAME} bash"
    echo "  SSH port      : ${SSH_PORT}"
    echo ""
    echo "  Shared directories (same on all nodes):"
    echo "    /opt/shared    <- ${SHARED_DIR}"
    echo "    /opt/builds    <- ${BUILDS_DIR}"
    echo "    /opt/ssh-keys  <- ${SSH_KEY_DIR} (read-only)"
    echo ""
    echo "  Verify SSH across nodes:"
    echo "    $0 --verify"

    if [[ -n "${VERBOSE}" ]]; then
        echo ""
        log_verbose "Container status: $(docker inspect "${CONTAINER_NAME}" --format '{{.State.Status}}' 2>/dev/null)"
        log_verbose "Container ID: $(docker inspect "${CONTAINER_NAME}" --format '{{.Id}}' 2>/dev/null | head -c 12)"
        log_verbose "Waiting 3s for entrypoint to finish..."
        sleep 3
        log_verbose "Container logs (last 20 lines):"
        docker logs "${CONTAINER_NAME}" 2>&1 | tail -20 | while read -r line; do
            log_verbose "  ${line}"
        done
    fi
}

# ============================================================================
# Verify SSH connectivity to all hosts in the hostfile
# ============================================================================
verify_ssh() {
    timer_start
    echo "=== Verifying SSH connectivity (port ${SSH_PORT}) ==="

    if [[ ! -f "${HOSTFILE}" ]]; then
        echo "  Hostfile not found: ${HOSTFILE}"
        echo "  Create it first:  echo 'hostname slots=8' > ${HOSTFILE}"
        exit 1
    fi

    log_verbose "Hostfile: ${HOSTFILE}"
    if [[ -n "${VERBOSE}" ]]; then
        log_verbose "Hostfile contents:"
        while read -r line; do
            log_verbose "  ${line}"
        done < "${HOSTFILE}"
    fi

    local hosts failed=0
    hosts=$(grep -v '^#' "${HOSTFILE}" | grep -v '^$' | awk '{print $1}' | sort -u)

    local ssh_key="${SSH_KEY_DIR}/id_rsa"
    if [[ ! -f "${ssh_key}" ]]; then
        echo "  Shared SSH key not found: ${ssh_key}"
        echo ""
        echo "  Set up SSH keys first:"
        echo "    $0 --launch-all --ssh-key ~/.ssh/id_rsa   # use your key pair"
        echo "    $0 --launch-all --ssh-keygen              # generate a new pair"
        echo "  For mesh SSH (per-node keys), also pass --ssh-authorized-keys"
        exit 1
    fi
    log_verbose "Using SSH key: ${ssh_key}"

    local ssh_opts=(
        -p "${SSH_PORT}"
        -i "${ssh_key}"
        -o StrictHostKeyChecking=no
        -o UserKnownHostsFile=/dev/null
        -o ConnectTimeout=5
        -o BatchMode=yes
        -o LogLevel=ERROR
    )

    local test_users=("root" "ubuntu")

    for host in ${hosts}; do
        local host_ok=1
        for user in "${test_users[@]}"; do
            log_verbose "Testing SSH to ${user}@${host}:${SSH_PORT}..."
            if ssh "${ssh_opts[@]}" "${user}@${host}" hostname 2>/dev/null; then
                echo "  [OK]   ${user}@${host}"
            else
                echo "  [FAIL] ${user}@${host}"
                host_ok=0
                failed=1
                if [[ -n "${VERBOSE}" ]]; then
                    log_verbose "SSH debug for ${user}@${host}:"
                    ssh -v "${ssh_opts[@]}" "${user}@${host}" hostname 2>&1 | tail -20 | while read -r line; do
                        log_verbose "  ${line}"
                    done
                fi
            fi
        done
    done

    if [[ "${failed}" -eq 1 ]]; then
        echo ""
        echo "Fix failed hosts:"
        echo "  1. Ensure the container is running: docker ps"
        echo "  2. Check sshd: docker exec <container> ss -tlnp | grep ${SSH_PORT}"
        echo "  3. Restart sshd: docker exec <container> /usr/sbin/sshd -p${SSH_PORT}"
        echo ""
        echo "  If SSH keys are not set up, re-launch with:"
        echo "    $0 --launch-all --ssh-key ~/.ssh/id_rsa   # use your key pair"
        echo "    $0 --launch-all --ssh-keygen              # generate a new pair"
        echo "  For mesh SSH (per-node keys), also pass --ssh-authorized-keys"
        exit 1
    fi

    timer_end "SSH verification"
    echo ""
    echo "All hosts reachable (as ${test_users[*]}). Ready for MPI workloads."
}

# ============================================================================
# Parse hostfile into a list of unique hostnames
# ============================================================================
parse_hostfile() {
    if [[ ! -f "${HOSTFILE}" ]]; then
        echo "  Hostfile not found: ${HOSTFILE}" >&2
        echo "  Create it first:  echo 'hostname slots=8' > ${HOSTFILE}" >&2
        exit 1
    fi
    grep -v '^#' "${HOSTFILE}" | grep -v '^$' | awk '{print $1}' | sort -u
}

# ============================================================================
# SSH wrapper for host-to-host access (port 22 by default, not container SSH)
# ============================================================================
host_ssh() {
    local host="$1"; shift
    ssh -p "${HOST_SSH_PORT}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=10 \
        -o BatchMode=yes \
        -o LogLevel=ERROR \
        "${host}" "$@"
}

# ============================================================================
# Launch containers on all nodes in the hostfile
# ============================================================================
launch_all() {
    timer_start
    echo "=== Launching containers on all nodes ==="
    echo ""

    local hosts
    hosts=$(parse_hostfile)
    local host_count
    host_count=$(echo "${hosts}" | wc -w)
    local local_hostname
    local_hostname=$(hostname -f 2>/dev/null || hostname)
    local local_short
    local_short=$(hostname -s 2>/dev/null || hostname)

    echo "  Hostfile  : ${HOSTFILE} (${host_count} nodes)"
    echo "  Image     : ${IMAGE_TAG}"
    echo "  Container : ${CONTAINER_NAME}"
    echo "  Script    : ${SCRIPT_DIR}/setup_multinode.sh"
    echo ""

    local forward_args=()
    forward_args+=(--run)
    forward_args+=(--name "${CONTAINER_NAME}")
    forward_args+=(--ssh-port "${SSH_PORT}")
    forward_args+=(--shm-size "${SHM_SIZE}")
    forward_args+=(--shared-dir "${SHARED_DIR}")
    forward_args+=(--builds-dir "${BUILDS_DIR}")
    forward_args+=(--ssh-key-dir "${SSH_KEY_DIR}")
    forward_args+=(--hostfile "${HOSTFILE}")
    [[ "${FORCE_REBUILD}" == true ]] && forward_args+=(--rebuild)
    [[ -n "${VERBOSE}" ]] && forward_args+=(--verbose)
    [[ -n "${POST_SETUP_DIR}" ]] && forward_args+=(--post-setup "${POST_SETUP_DIR}")
    [[ -n "${SSH_KEY}" ]] && forward_args+=(--ssh-key "${SSH_KEY}")
    [[ -n "${SSH_AUTHORIZED_KEYS}" ]] && forward_args+=(--ssh-authorized-keys "${SSH_AUTHORIZED_KEYS}")
    [[ "${DO_SSH_KEYGEN}" == true ]] && forward_args+=(--ssh-keygen)
    for vol in "${EXTRA_VOLUMES[@]+"${EXTRA_VOLUMES[@]}"}"; do
        forward_args+=(--volume "$vol")
    done
    forward_args+=("${ROCM_IMAGE}")

    local failed=0

    for host in ${hosts}; do
        echo "--- ${host} ---"

        if [[ "${host}" == "${local_hostname}" || "${host}" == "${local_short}" || "${host}" == "localhost" ]]; then
            log_verbose "Local node detected, running directly"
            if "${SCRIPT_DIR}/setup_multinode.sh" "${forward_args[@]}"; then
                echo "  [OK]   ${host}"
            else
                echo "  [FAIL] ${host} (exit $?)"
                failed=1
            fi
        else
            log_verbose "Remote node, SSHing via port ${HOST_SSH_PORT}"
            log_verbose "Remote command: ${SCRIPT_DIR}/setup_multinode.sh ${forward_args[*]}"
            if host_ssh "${host}" "${SCRIPT_DIR}/setup_multinode.sh" "${forward_args[@]}"; then
                echo "  [OK]   ${host}"
            else
                echo "  [FAIL] ${host} (exit $?)"
                failed=1
            fi
        fi
        echo ""
    done

    if [[ "${failed}" -eq 1 ]]; then
        echo "Some nodes failed. Check output above and fix, then re-run:"
        echo "  $0 --launch-all"
        exit 1
    fi

    timer_end "Launch all nodes"
    echo "=== All ${host_count} containers launched ==="
    echo ""
    echo "  Verify container SSH:"
    echo "    $0 --verify"
}

# ============================================================================
# Stop and remove containers on all nodes in the hostfile
# ============================================================================
stop_all() {
    echo "=== Stopping containers on all nodes ==="
    echo ""

    local hosts
    hosts=$(parse_hostfile)
    local local_hostname
    local_hostname=$(hostname -f 2>/dev/null || hostname)
    local local_short
    local_short=$(hostname -s 2>/dev/null || hostname)

    local stop_cmd="docker rm -f ${CONTAINER_NAME} 2>/dev/null && echo 'removed' || echo 'not running'"

    for host in ${hosts}; do
        printf "  %-20s " "${host}"

        if [[ "${host}" == "${local_hostname}" || "${host}" == "${local_short}" || "${host}" == "localhost" ]]; then
            eval "${stop_cmd}"
        else
            host_ssh "${host}" "${stop_cmd}"
        fi
    done

    echo ""
    echo "=== Done ==="
}

# ============================================================================
# Main
# ============================================================================
if [[ "${DO_VERIFY}" == true ]]; then
    verify_ssh
    exit 0
fi

if [[ "${DO_STOP_ALL}" == true ]]; then
    stop_all
    exit 0
fi

if [[ "${DO_SETUP_DEPS}" == true ]]; then
    setup_host
    build_image
    setup_shared_deps
    exit 0
fi

if [[ "${DO_LAUNCH_ALL}" == true ]]; then
    setup_host
    build_image
    setup_shared_deps
    launch_all
    exit 0
fi

setup_host

if [[ "${DO_BUILD}" == true ]]; then
    build_image
fi

if [[ "${DO_RUN}" == true ]]; then
    launch_container
else
    echo "Image ready: ${IMAGE_TAG}"
    echo ""
    echo "To launch a container:"
    echo "  $0 --run [--name NAME] [--gpus N]"
    echo ""
    echo "Or manually:"
    echo "  docker run -d --name rccl-mn --privileged --network host \\"
    echo "    --device /dev/kfd --device /dev/dri \\"
    echo "    -v ${SHARED_DIR}:/opt/shared \\"
    echo "    -v ${BUILDS_DIR}:/opt/builds \\"
    echo "    -v ${SSH_KEY_DIR}:/opt/ssh-keys:ro \\"
    echo "    ${IMAGE_TAG}"
fi
