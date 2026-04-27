#!/bin/bash
#
# standalone/entrypoint.sh -- runtime setup for the standalone RCCL image.
#
# Performs only what cannot be done at build time:
#   1. Remap the non-root CONTAINER_USER's UID/GID to match the host
#      (so that NFS / bind-mounted home directories stay accessible).
#   2. Install the shared SSH keys mounted at /opt/ssh-keys into both
#      /root/.ssh and /home/$CONTAINER_USER/.ssh.
#   3. Open GPU devices to all users (chmod 666 /dev/kfd, render*, card*).
#   4. Start sshd on $SSH_PORT.
#   5. Exec the user-supplied command, or idle.
#
# Environment (all optional; defaults match the Dockerfile):
#   HOST_UID / HOST_GID     target UID/GID for CONTAINER_USER  (default 1000)
#   RENDER_GID              GID for the host's `render` group  (default 109)
#   SSH_PORT                sshd listen port                   (default 2224)
#   SSH_KEY_SOURCE          dir holding id_rsa{,.pub}/auth-keys (default /opt/ssh-keys)
#   CONTAINER_USER          non-root user name                 (default ubuntu)
#   VERBOSE                 set to 1 for extra debug lines
#
set -e

: "${SSH_PORT:=2224}"
: "${SSH_KEY_SOURCE:=/opt/ssh-keys}"
: "${CONTAINER_USER:=ubuntu}"
: "${HOST_UID:=1000}"
: "${HOST_GID:=1000}"
: "${RENDER_GID:=109}"
: "${VERBOSE:=}"

log()         { echo "  $*"; }
log_verbose() { [[ -n "${VERBOSE}" ]] && echo "  [verbose] $*" || true; }

# ---------------------------------------------------------------------------
# 1. Remap non-root user UID/GID to host
# ---------------------------------------------------------------------------
remap_user() {
    local home="/home/${CONTAINER_USER}"

    if id "${CONTAINER_USER}" &>/dev/null; then
        local cur_uid
        cur_uid=$(id -u "${CONTAINER_USER}")
        if [ "${cur_uid}" != "${HOST_UID}" ]; then
            log "Remapping ${CONTAINER_USER} UID ${cur_uid} -> ${HOST_UID}"
            usermod  -u "${HOST_UID}" "${CONTAINER_USER}" 2>/dev/null || true
            groupmod -g "${HOST_GID}" "${CONTAINER_USER}" 2>/dev/null || true
        fi
    else
        log "Creating user ${CONTAINER_USER} (uid=${HOST_UID})"
        groupadd -g "${HOST_GID}" "${CONTAINER_USER}" 2>/dev/null || true
        useradd  -u "${HOST_UID}" -g "${HOST_GID}" -m -s /bin/bash "${CONTAINER_USER}"
    fi

    mkdir -p "${home}"
    chown "${HOST_UID}:${HOST_GID}" "${home}" 2>/dev/null || true

    if ! getent group render &>/dev/null; then
        groupadd -g "${RENDER_GID}" render 2>/dev/null || groupadd render 2>/dev/null || true
    fi
    usermod -aG video,render "${CONTAINER_USER}" 2>/dev/null || true
    usermod -aG video,render root              2>/dev/null || true

    chmod 666 /dev/kfd          2>/dev/null || true
    chmod 666 /dev/dri/render*  2>/dev/null || true
    chmod 666 /dev/dri/card*    2>/dev/null || true
}

# ---------------------------------------------------------------------------
# 2. Install the shared SSH keys for a user
# ---------------------------------------------------------------------------
install_ssh_keys() {
    local home="$1" user="$2"
    mkdir -p "${home}/.ssh"
    chmod 700 "${home}/.ssh"

    if [ -f "${SSH_KEY_SOURCE}/id_rsa" ]; then
        cp  "${SSH_KEY_SOURCE}/id_rsa"          "${home}/.ssh/id_rsa"
        cp  "${SSH_KEY_SOURCE}/id_rsa.pub"      "${home}/.ssh/id_rsa.pub"
        cp  "${SSH_KEY_SOURCE}/authorized_keys" "${home}/.ssh/authorized_keys"
        if [ -f "${SSH_KEY_SOURCE}/config" ]; then
            cp "${SSH_KEY_SOURCE}/config" "${home}/.ssh/config"
        else
            printf 'Host *\n    StrictHostKeyChecking no\n    UserKnownHostsFile /dev/null\n    LogLevel ERROR\n    Port %s\n' \
                "${SSH_PORT}" > "${home}/.ssh/config"
        fi
        log_verbose "Installed shared keys for ${user} from ${SSH_KEY_SOURCE}"
    else
        log "WARN: no shared SSH keys at ${SSH_KEY_SOURCE} -- container is single-node only"
        log "      run setup_ssh.sh on the host and bind-mount ~/.docker-ssh-keys -> ${SSH_KEY_SOURCE}"
        return
    fi

    chmod 600 "${home}/.ssh/id_rsa" "${home}/.ssh/authorized_keys" "${home}/.ssh/config"
    chmod 644 "${home}/.ssh/id_rsa.pub"
    id "${user}" &>/dev/null && chown -R "${user}:${user}" "${home}/.ssh" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "=== standalone entrypoint ==="

if [[ -n "${VERBOSE}" ]]; then
    log_verbose "SSH_PORT=${SSH_PORT}  SSH_KEY_SOURCE=${SSH_KEY_SOURCE}"
    log_verbose "CONTAINER_USER=${CONTAINER_USER}  HOST_UID=${HOST_UID}  HOST_GID=${HOST_GID}"
fi

remap_user
install_ssh_keys "/root" "root"
if id "${CONTAINER_USER}" &>/dev/null; then
    install_ssh_keys "/home/${CONTAINER_USER}" "${CONTAINER_USER}"
fi

log "Starting sshd on port ${SSH_PORT}"
/usr/sbin/sshd -p"${SSH_PORT}"

echo "=== Ready ==="

if [ $# -gt 0 ]; then
    log_verbose "exec: $*"
    exec "$@"
else
    log_verbose "no command -- idling with tail -f /dev/null"
    exec tail -f /dev/null
fi
