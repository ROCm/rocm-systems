#!/bin/bash
#
# standalone/setup_ssh.sh -- generate a shared SSH keypair for the
# container-side sshd (port 2224).  The same private key is used by every
# container so that mpirun can SSH between containers without prompts.
#
# Output: ~/.docker-ssh-keys/{id_rsa, id_rsa.pub, authorized_keys, config}
# Idempotent: re-running with existing keys is a no-op.
#
# After this script:
#   - On a SHARED filesystem (NFS / SLURM shared $HOME): you're done.
#     The directory is visible from every node.
#   - On UNMANAGED nodes: copy the directory to every other node, e.g.
#         for h in node-b node-c; do
#             rsync -a ~/.docker-ssh-keys/ "$h:.docker-ssh-keys/"
#         done

set -e

KEY_DIR="${KEY_DIR:-${HOME}/.docker-ssh-keys}"
SSH_PORT="${SSH_PORT:-2224}"

mkdir -p "${KEY_DIR}"
chmod 700 "${KEY_DIR}"

if [ -f "${KEY_DIR}/id_rsa" ] && [ -f "${KEY_DIR}/id_rsa.pub" ]; then
    echo "[ok] keys already present at ${KEY_DIR}"
else
    echo "[..] generating new RSA keypair at ${KEY_DIR}/id_rsa"
    ssh-keygen -t rsa -b 4096 -N "" -f "${KEY_DIR}/id_rsa" -C "rccl-mn-standalone" -q
fi

if ! grep -qxF "$(cat "${KEY_DIR}/id_rsa.pub")" "${KEY_DIR}/authorized_keys" 2>/dev/null; then
    cat "${KEY_DIR}/id_rsa.pub" >> "${KEY_DIR}/authorized_keys"
fi

cat > "${KEY_DIR}/config" <<EOF
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    Port ${SSH_PORT}
EOF

chmod 600 "${KEY_DIR}/id_rsa" "${KEY_DIR}/authorized_keys" "${KEY_DIR}/config"
chmod 644 "${KEY_DIR}/id_rsa.pub"

echo "[ok] keypair ready at ${KEY_DIR}"
ls -l "${KEY_DIR}/" | sed 's/^/      /'
