#!/bin/bash
#
# Container entrypoint - runtime-only setup that cannot be done at build time.
#
# 1. Remaps the non-root user's UID/GID to match the host (NFS access)
# 2. Copies shared SSH keys into each user's ~/.ssh
# 3. Starts sshd
# 4. Executes the given command, LAUNCH_SCRIPT, or idles
#
# Environment (all optional):
#   HOST_UID / HOST_GID  - target UID/GID for the non-root user  (default: 1000)
#   SSH_PORT             - sshd listen port                      (default: 2224)
#   SSH_KEY_SOURCE       - mounted dir with id_rsa/pub           (default: /opt/ssh-keys)
#   CONTAINER_USER       - non-root user name                    (default: ubuntu)
#   LAUNCH_SCRIPT        - script to exec after setup            (default: "")
#   LAUNCH_SCRIPT_ARGS   - args for the launch script            (default: "")
#   VERBOSE              - set to 1 for detailed debug logging   (default: "")
#

set -e

SSH_PORT="${SSH_PORT:-2224}"
SSH_KEY_SOURCE="${SSH_KEY_SOURCE:-/opt/ssh-keys}"
CONTAINER_USER="${CONTAINER_USER:-ubuntu}"
VERBOSE="${VERBOSE:-}"

log_verbose() {
    [[ -n "${VERBOSE}" ]] && echo "  [verbose] $*" || true
}

# ============================================================================
# Remap non-root user UID/GID to match host
# ============================================================================
setup_container_user() {
    local target_uid="${HOST_UID:-1000}"
    local target_gid="${HOST_GID:-1000}"
    local user_home="/home/${CONTAINER_USER}"

    log_verbose "setup_container_user: target_uid=${target_uid} target_gid=${target_gid} user=${CONTAINER_USER}"

    if id "${CONTAINER_USER}" &>/dev/null; then
        local current_uid
        current_uid=$(id -u "${CONTAINER_USER}")
        log_verbose "User ${CONTAINER_USER} exists with UID ${current_uid}"
        if [ "$current_uid" != "$target_uid" ]; then
            echo "  Remapping ${CONTAINER_USER} UID ${current_uid} -> ${target_uid}"
            usermod -u "$target_uid" "${CONTAINER_USER}" 2>/dev/null || true
            groupmod -g "$target_gid" "${CONTAINER_USER}" 2>/dev/null || true
        fi
    else
        log_verbose "User ${CONTAINER_USER} does not exist, creating"
        groupadd -g "$target_gid" "${CONTAINER_USER}" 2>/dev/null || true
        useradd -u "$target_uid" -g "$target_gid" -m -s /bin/bash "${CONTAINER_USER}" 2>/dev/null || true
    fi

    mkdir -p "${user_home}"
    chown "${target_uid}:${target_gid}" "${user_home}" 2>/dev/null || true

    # Ensure render group exists with the host's GID
    local render_gid="${RENDER_GID:-109}"
    if ! getent group render &>/dev/null; then
        if ! groupadd -g "${render_gid}" render 2>/dev/null; then
            # GID may be taken by another group; try without specifying GID
            groupadd render 2>/dev/null || true
        fi
        log_verbose "Created render group ($(getent group render 2>/dev/null || echo 'failed'))"
    fi
    usermod -aG video,render "${CONTAINER_USER}" 2>/dev/null || true
    usermod -aG video,render root 2>/dev/null || true

    # Open GPU devices to all users
    chmod 666 /dev/kfd 2>/dev/null || true
    chmod 666 /dev/dri/render* 2>/dev/null || true
    chmod 666 /dev/dri/card* 2>/dev/null || true

    if [[ -n "${VERBOSE}" ]]; then
        log_verbose "Final user info: $(id "${CONTAINER_USER}" 2>/dev/null || echo 'n/a')"
        log_verbose "Home dir: $(ls -ld "${user_home}" 2>/dev/null || echo 'missing')"
        log_verbose "GPU devices:"
        ls -l /dev/kfd /dev/dri/render* /dev/dri/card* 2>/dev/null | while read -r line; do
            log_verbose "  ${line}"
        done
    fi
}

# ============================================================================
# Distribute shared SSH keys to a user's ~/.ssh
# ============================================================================
setup_user_ssh() {
    local user_home="$1" user_name="$2"

    log_verbose "setup_user_ssh: user=${user_name} home=${user_home} source=${SSH_KEY_SOURCE}"

    mkdir -p "${user_home}/.ssh"
    chmod 700 "${user_home}/.ssh"

    if [ -d "${SSH_KEY_SOURCE}" ] && [ -f "${SSH_KEY_SOURCE}/id_rsa" ]; then
        cp "${SSH_KEY_SOURCE}/id_rsa"          "${user_home}/.ssh/id_rsa"
        cp "${SSH_KEY_SOURCE}/id_rsa.pub"      "${user_home}/.ssh/id_rsa.pub"
        cp "${SSH_KEY_SOURCE}/authorized_keys" "${user_home}/.ssh/authorized_keys"
        if [ -f "${SSH_KEY_SOURCE}/config" ]; then
            cp "${SSH_KEY_SOURCE}/config" "${user_home}/.ssh/config"
        else
            printf 'Host *\n    StrictHostKeyChecking no\n    UserKnownHostsFile /dev/null\n    LogLevel ERROR\n    Port %s\n' \
                "${SSH_PORT}" > "${user_home}/.ssh/config"
        fi
        log_verbose "Copied shared keys from ${SSH_KEY_SOURCE}"
    else
        echo "  WARN: no shared keys at ${SSH_KEY_SOURCE}; generating local keys"
        [ -f "${user_home}/.ssh/id_rsa" ] || {
            ssh-keygen -t rsa -b 4096 -N "" -f "${user_home}/.ssh/id_rsa" -C "local-key" -q
            cat "${user_home}/.ssh/id_rsa.pub" >> "${user_home}/.ssh/authorized_keys"
        }
        printf 'Host *\n    StrictHostKeyChecking no\n    UserKnownHostsFile /dev/null\n    LogLevel ERROR\n    Port %s\n' \
            "${SSH_PORT}" > "${user_home}/.ssh/config"
    fi

    chmod 600 "${user_home}/.ssh/id_rsa" "${user_home}/.ssh/authorized_keys" "${user_home}/.ssh/config"
    chmod 644 "${user_home}/.ssh/id_rsa.pub"
    id "${user_name}" &>/dev/null && chown -R "${user_name}:${user_name}" "${user_home}/.ssh" 2>/dev/null || true

    if [[ -n "${VERBOSE}" ]]; then
        log_verbose "SSH dir contents for ${user_name}:"
        ls -la "${user_home}/.ssh/" 2>/dev/null | while read -r line; do
            log_verbose "  ${line}"
        done
    fi
}

# ============================================================================
# Main
# ============================================================================
echo "=== Container entrypoint ==="

if [[ -n "${VERBOSE}" ]]; then
    log_verbose "Environment:"
    log_verbose "  SSH_PORT=${SSH_PORT}"
    log_verbose "  SSH_KEY_SOURCE=${SSH_KEY_SOURCE}"
    log_verbose "  CONTAINER_USER=${CONTAINER_USER}"
    log_verbose "  HOST_UID=${HOST_UID:-1000}  HOST_GID=${HOST_GID:-1000}"
    log_verbose "  LAUNCH_SCRIPT=${LAUNCH_SCRIPT:-}"
    log_verbose "  GPUS=${GPUS:-}"
    log_verbose "Mounted volumes:"
    mount | grep -E '/opt/(shared|builds|ssh-keys)' | while read -r line; do
        log_verbose "  ${line}"
    done
    log_verbose "ROCm version: $(cat /opt/rocm/.info/version 2>/dev/null || echo 'unknown')"
fi

setup_container_user
setup_user_ssh "/root" "root"
user_home="/home/${CONTAINER_USER}"
if id "${CONTAINER_USER}" &>/dev/null; then
    setup_user_ssh "${user_home}" "${CONTAINER_USER}"
fi

# Shared mount-point permissions (may fail on NFS, that's fine)
chmod 777 /opt/shared /opt/builds 2>/dev/null || true
log_verbose "Shared dirs: $(ls -ld /opt/shared /opt/builds 2>/dev/null | tr '\n' ' ')"

echo "  Starting sshd on port ${SSH_PORT}"
if [[ -n "${VERBOSE}" ]]; then
    /usr/sbin/sshd -p"${SSH_PORT}" -e 2>&1 | while read -r line; do
        log_verbose "sshd: ${line}"
    done
    log_verbose "sshd config: Port=$(grep -E '^Port ' /etc/ssh/sshd_config 2>/dev/null || echo 'default')"
    log_verbose "sshd config: PermitRootLogin=$(grep -E '^PermitRootLogin ' /etc/ssh/sshd_config 2>/dev/null | awk '{print $2}')"
    log_verbose "sshd process: $(ps aux | grep '[s]shd' | head -1)"
else
    /usr/sbin/sshd -p"${SSH_PORT}"
fi

echo "  User: ${CONTAINER_USER} ($(id ${CONTAINER_USER} 2>/dev/null || echo 'n/a'))"
echo "=== Ready ==="

# Exec command, launch script, or idle
if [ $# -gt 0 ]; then
    log_verbose "Executing command: $*"
    exec "$@"
elif [ -n "${LAUNCH_SCRIPT:-}" ] && [ -f "${LAUNCH_SCRIPT}" ]; then
    log_verbose "Executing launch script: ${LAUNCH_SCRIPT} ${LAUNCH_SCRIPT_ARGS:-}"
    exec "${LAUNCH_SCRIPT}" ${LAUNCH_SCRIPT_ARGS:-}
else
    log_verbose "No command or launch script; idling with tail -f /dev/null"
    exec tail -f /dev/null
fi
