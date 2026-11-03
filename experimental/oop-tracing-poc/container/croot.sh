#!/usr/bin/env bash
# Run a command in the POC container: host ROCm read-only at /opt/rocm, workspace at /w, GPU devices.
R=$(readlink -f /opt/rocm)
exec docker run --rm -i -u 0 --cap-add PERFMON -e HOME=/w/home -w "${CWD:-/w}" \
  -v "$HOME/dlog-poc:/w" -v "$R:/opt/rocm:ro" \
  --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 --shm-size=8g --ulimit nofile=1048576:1048576 \
  dlog-poc:24.04 bash -c "$*"
