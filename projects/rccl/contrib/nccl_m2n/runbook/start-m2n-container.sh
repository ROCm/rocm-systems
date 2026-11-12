#!/bin/bash
# Start the m2n-multi container on the current node (run on every allocated node).
# Host network, GPUs + RDMA devices, $HOME bind-mounted, sshd on 2222 for mpirun,
# raised fd limit (2-node GIN/DevComm setup exceeds the default 1024 fds).
# Usage: bash start-m2n-container.sh

set -euo pipefail

IMAGE="${IMAGE:-m2n-ruby:primus-rdma61}"
NAME="${NAME:-m2n-multi}"
NOFILE="${NOFILE:-1048576}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_m2n}"
USER_NAME="$(whoami)"

if [[ -f "$SSH_KEY.pub" ]]; then
  AUTH_KEY="$(cat "$SSH_KEY.pub")"
else
  AUTH_KEY="$(ssh-keygen -y -f "$SSH_KEY")"
fi

sudo docker rm -f "$NAME" 2>/dev/null || true

PASSWD_LINE="$(getent passwd "$USER_NAME")"
GROUP_LINES="$(id -G | tr ' ' '\n' | while read -r gid; do getent group "$gid" 2>/dev/null; done)"

echo "[$(hostname -s)] Starting $NAME from $IMAGE (nofile=$NOFILE) ..."
sudo docker run -d \
  --name "$NAME" \
  --network host \
  --ipc host \
  --device /dev/kfd \
  --device /dev/dri \
  --device /dev/infiniband \
  --cap-add IPC_LOCK --cap-add SYS_NICE --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --ulimit "nofile=$NOFILE:$NOFILE" \
  -v "$HOME:$HOME" \
  -e "PASSWD_LINE=$PASSWD_LINE" \
  -e "GROUP_LINES=$GROUP_LINES" \
  -e "AUTH_KEY=$AUTH_KEY" \
  "$IMAGE" \
  bash -c '
    grep -q "^${PASSWD_LINE%%:*}:" /etc/passwd || echo "$PASSWD_LINE" >> /etc/passwd
    u=${PASSWD_LINE%%:*}
    # Keep the host GIDs (device nodes such as /dev/dri/renderD* are owned by them): reuse the group
    # that already has the GID, or add it (as host_<name> if the name is taken with another GID).
    echo "$GROUP_LINES" | grep -v "^$" | while IFS=: read -r gname _ gid _; do
      g=$(getent group "$gid" | cut -d: -f1)
      if [ -z "$g" ]; then
        g=$gname; getent group "$g" >/dev/null && g=host_$gname
        groupadd -g "$gid" "$g"
      fi
      usermod -aG "$g" "$u"
    done
    mkdir -p /etc/ssh/keys
    echo "$AUTH_KEY" > /etc/ssh/keys/$u
    chown "$u" /etc/ssh/keys/$u
    chmod 600 /etc/ssh/keys/$u
    exec /usr/sbin/sshd -D -e
  '

sleep 2
# The image has no clang/clang++ under $ROCM_PATH/bin, which the HIP build expects.
sudo docker exec "$NAME" bash -c '
  R=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
  for c in clang clang++; do
    rm -f $R/bin/$c
    printf "#!/bin/sh\nexec $R/llvm/bin/$c \"\$@\"\n" > $R/bin/$c
    chmod 755 $R/bin/$c
  done'
sudo docker exec -u "$(id -u)" "$NAME" bash -c 'echo "[$(hostname -s)] nofile soft=$(ulimit -Sn) hard=$(ulimit -Hn)"'
