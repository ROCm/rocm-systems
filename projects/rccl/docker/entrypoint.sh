#!/bin/bash
#
# Container entrypoint - runtime-only setup that cannot be done at build time.
#
# 1. Remaps the non-root user's UID/GID to match the host (NFS access)
# 2. Copies shared SSH keys into each user's ~/.ssh
# 3. Starts sshd
# 4. Runs post-setup configuration hook (if provided)
# 5. Executes the given command, LAUNCH_SCRIPT, or idles
#
# Environment (all optional):
#   HOST_UID / HOST_GID  - target UID/GID for the non-root user  (default: 1000)
#   SSH_PORT             - sshd listen port                      (default: 2224)
#   SSH_KEY_SOURCE       - mounted dir with id_rsa/pub           (default: /opt/ssh-keys)
#   CONTAINER_USER       - non-root user name                    (default: ubuntu)
#   LAUNCH_SCRIPT        - script to exec after setup            (default: "")
#   LAUNCH_SCRIPT_ARGS   - args for the launch script            (default: "")
#   POST_SETUP_DIRS      - colon-separated post-setup dirs        (default: "")
#                          Each dir may contain setup.sh/env.sh.
#                          Dirs are processed in order; later env.sh
#                          exports override earlier ones.  mnctl mounts
#                          host dirs at /opt/post-setup.0, .1, ...
#   VERBOSE              - set to 1 for detailed debug logging   (default: "")
#

set -e

# ============================================================================
# Defaults for every environment variable consumed below.
#
# Mirrors the role of mnctl/config.py's Config.__init__: every knob has a
# documented default in ONE place so later code can read ${VAR} without
# repeating the fallback expression.  ``:=`` (assign) is intentional --
# values become bound in this shell so children inherit them via export
# below where applicable.
# ============================================================================
: "${SSH_PORT:=2224}"
: "${SSH_KEY_SOURCE:=/opt/ssh-keys}"
: "${CONTAINER_USER:=ubuntu}"
: "${NIC_TYPE:=mellanox}"
: "${VERBOSE:=}"
: "${POST_SETUP_DIRS:=}"
: "${HOST_UID:=1000}"
: "${HOST_GID:=1000}"
: "${RENDER_GID:=109}"
: "${LAUNCH_SCRIPT:=}"
: "${LAUNCH_SCRIPT_ARGS:=}"
: "${FORCE_POST_SETUP:=}"
: "${GPUS:=}"

log_verbose() {
    [[ -n "${VERBOSE}" ]] && echo "  [verbose] $*" || true
}

# ============================================================================
# Remap non-root user UID/GID to match host
# ============================================================================
setup_container_user() {
    local target_uid="${HOST_UID}"
    local target_gid="${HOST_GID}"
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
    if ! getent group render &>/dev/null; then
        if ! groupadd -g "${RENDER_GID}" render 2>/dev/null; then
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
        echo "  WARN: no shared SSH keys at ${SSH_KEY_SOURCE}; generating local-only keys"
        echo "  Hint: for multi-node SSH, pass --ssh [KEY_PATH] to mnctl"
        echo "        (auto-detected when running inside a SLURM allocation)"
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
# Post-setup configuration hook (setup.sh + env.sh)
#
# Each entry in POST_SETUP_DIRS (colon-separated) is processed by
# run_one_post_setup.  env.sh contents are CONCATENATED (in order) into
# /etc/profile.d/post-setup-env.sh; later dirs win on conflicting exports.
# setup.sh is per-dir, idempotent via SHA256 marker under /opt/builds.
# ============================================================================
run_one_post_setup() {
    local dir="$1"
    local idx="$2"

    if [[ ! -d "${dir}" ]] || [[ -z "$(ls -A "${dir}" 2>/dev/null)" ]]; then
        log_verbose "  Post-setup[${idx}]: ${dir} empty/missing, skipping"
        return
    fi

    echo "  Post-setup[${idx}]: ${dir}"

    if [[ -f "${dir}/env.sh" ]]; then
        local hash
        hash=$(sha256sum "${dir}/env.sh" 2>/dev/null | awk '{print $1}')
        log_verbose "    env.sh SHA256: ${hash}"
        # Append (rather than overwrite) so multiple dirs compose.  The
        # caller (run_post_setup) truncates the file once, before the loop.
        {
            echo ""
            echo "# --- post-setup[${idx}] from ${dir} ---"
            cat "${dir}/env.sh"
        } >> /etc/profile.d/post-setup-env.sh
        echo "    env.sh appended ($(grep -c '^export' "${dir}/env.sh" 2>/dev/null || echo 0) exports)"
    fi

    if [[ -f "${dir}/setup.sh" ]]; then
        local first_line
        first_line=$(head -1 "${dir}/setup.sh")
        if [[ "${first_line}" != "#!/bin/bash"* ]] && [[ "${first_line}" != "#!/usr/bin/env bash"* ]]; then
            echo "    WARN: setup.sh missing bash shebang, skipping for safety"
            return
        fi

        local hash
        hash=$(sha256sum "${dir}/setup.sh" 2>/dev/null | awk '{print $1}')
        echo "    setup.sh (SHA256: ${hash:0:16}...)"

        local marker="/opt/builds/.post-setup.${hash:0:16}.done"

        if [[ "${FORCE_POST_SETUP}" == "1" ]] && [[ -f "${marker}" ]]; then
            echo "    clearing stale marker (FORCE_POST_SETUP=1)"
            rm -f "${marker}"
        fi

        if [[ -f "${marker}" ]]; then
            echo "    already completed (cached)"
            log_verbose "    marker: ${marker}"
            return
        fi

        local work_dir
        work_dir=$(mktemp -d /tmp/post-setup.XXXXXX)
        cp -a "${dir}/." "${work_dir}/"
        chmod +x "${work_dir}/setup.sh"

        echo "    running setup.sh ..."
        local rc=0
        ( set -o pipefail; bash "${work_dir}/setup.sh" 2>&1 | sed 's/^/      [post-setup] /' ) || rc=$?

        if [[ "${rc}" -eq 0 ]]; then
            touch "${marker}" 2>/dev/null || true
            echo "    [OK] setup.sh completed"
        else
            echo "    [FAIL] setup.sh exited with code ${rc}"
        fi

        rm -rf "${work_dir}"
    fi
}

run_post_setup() {
    if [[ -z "${POST_SETUP_DIRS}" ]]; then
        log_verbose "No POST_SETUP_DIRS set, skipping post-setup"
        return
    fi

    # Truncate combined env file once; per-dir env.sh blocks are appended.
    : > /etc/profile.d/post-setup-env.sh
    chmod 644 /etc/profile.d/post-setup-env.sh

    local idx=0
    local IFS=':'
    for dir in ${POST_SETUP_DIRS}; do
        run_one_post_setup "${dir}" "${idx}"
        idx=$((idx + 1))
    done

    # Source the combined env, and ensure shells pick it up.
    if [[ -s /etc/profile.d/post-setup-env.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/post-setup-env.sh
        for rc in /root/.bashrc /home/${CONTAINER_USER}/.bashrc; do
            if [[ -f "$rc" ]] && ! grep -q 'post-setup-env.sh' "$rc" 2>/dev/null; then
                echo 'source /etc/profile.d/post-setup-env.sh' >> "$rc"
            fi
        done
    else
        rm -f /etc/profile.d/post-setup-env.sh
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
    log_verbose "  HOST_UID=${HOST_UID}  HOST_GID=${HOST_GID}"
    log_verbose "  LAUNCH_SCRIPT=${LAUNCH_SCRIPT}"
    log_verbose "  POST_SETUP_DIRS=${POST_SETUP_DIRS}"
    log_verbose "  NIC_TYPE=${NIC_TYPE}"
    log_verbose "  GPUS=${GPUS}"
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

echo "  NIC type: ${NIC_TYPE}"
log_verbose "NIC-specific setup (if any) is delivered as a post-setup dir"
log_verbose "(mnctl auto-prepends post-setup/<NIC_TYPE> unless"
log_verbose " --no-builtin-nic-setup is passed)"

run_post_setup

echo "  User: ${CONTAINER_USER} ($(id ${CONTAINER_USER} 2>/dev/null || echo 'n/a'))"
echo "=== Ready ==="

# Exec command, launch script, or idle
if [ $# -gt 0 ]; then
    log_verbose "Executing command: $*"
    exec "$@"
elif [ -n "${LAUNCH_SCRIPT}" ] && [ -f "${LAUNCH_SCRIPT}" ]; then
    log_verbose "Executing launch script: ${LAUNCH_SCRIPT} ${LAUNCH_SCRIPT_ARGS}"
    exec "${LAUNCH_SCRIPT}" ${LAUNCH_SCRIPT_ARGS}
else
    log_verbose "No command or launch script; idling with tail -f /dev/null"
    exec tail -f /dev/null
fi
