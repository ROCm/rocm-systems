#!/bin/bash
# 1-node Crusoe AINIC probe: 8x gfx950, ionic_0..7 in sysfs and ibv_devices,
# GID-index-1 present on every ionic, and a CPU/NUMA/NIC-map fingerprint so
# the launcher can pair symmetric nodes.
# Prints one line: "OK host=... nics=... fp=..." or "BAD host=... reason=...".
#
# Do not hash RoCE GID values: the 4th hextet is unique per node (same trap
# as PCI BDFs) and would make every host a group of one.

set +e
export PATH="/opt/rocm/bin:/usr/bin:/usr/sbin:/bin:${PATH:-}"

export PATH="${PATH}:/usr/sbin:/sbin"
host=$(hostname -s)
want="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
gid_idx="${NCCL_IB_GID_INDEX:-1}"
# Management/OOB interface OpenMPI is pinned to (oob_tcp_if_include /
# btl_tcp_if_include in the ainic mpi_args). A node whose mgmt NIC is missing,
# down, or has no IPv4 cannot bootstrap mpirun daemons, so gate on it here --
# this is the cheap per-node half of the check; crusoe_mpi_smoke.sh proves the
# pair can actually route to each other over it before the pair is pinned.
oob_if="${PROBE_OOB_IF:-ens3}"

gpus=$(rocminfo 2>/dev/null | grep -c gfx950)
gpus=${gpus:-0}

# IPv4 on the OOB interface, only if the link is up (`ip ... up` filter).
oob_ipv4=$(ip -o -4 addr show dev "${oob_if}" up 2>/dev/null | awk '{print $4}' | head -1)

nics=$(ls -1 /sys/class/infiniband 2>/dev/null | grep -E '^ionic_[0-7]$' | sort | paste -sd, -)
ibv=$(ibv_devices 2>/dev/null | awk 'NF && $1 ~ /^ionic_[0-7]$/ {print $1}' | sort | paste -sd, -)

cpu=$(grep -E '^(cpu family|model|model name|stepping)' /proc/cpuinfo | sort -u | tr '\n' '|')
nnuma=$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)
ncpu=$(nproc 2>/dev/null || echo 0)
nic_numa=""
gid_missing=""
for i in 0 1 2 3 4 5 6 7; do
  nn=$(cat "/sys/class/infiniband/ionic_${i}/device/numa_node" 2>/dev/null || echo x)
  nic_numa="${nic_numa:+${nic_numa},}ionic_${i}:${nn}"
  g=$(tr -d '[:space:]' < "/sys/class/infiniband/ionic_${i}/ports/1/gids/${gid_idx}" 2>/dev/null || true)
  pfx=$(printf '%s\n' "${g}" | awk -F: '{print tolower($4)}' | sed 's/^0*//')
  if [ -z "${pfx}" ]; then
    gid_missing="${gid_missing:+${gid_missing},}ionic_${i}"
  fi
done

fp=$(printf 'cpu=%s|numa=%s|ncpu=%s|nics=%s|nic_numa=%s|gpus=%s' \
  "${cpu}" "${nnuma}" "${ncpu}" "${nics}" "${nic_numa}" "${gpus}" \
  | sha256sum | awk '{print $1}')

reason=""
ok=1
if [ "${gpus}" -lt 8 ]; then
  ok=0
  reason="gpus=${gpus}"
fi
if [ "${nics}" != "${want}" ]; then
  ok=0
  reason="${reason:+${reason},}sysfs_nics=${nics:-none}"
fi
if [ "${ibv}" != "${want}" ]; then
  ok=0
  reason="${reason:+${reason},}ibv_nics=${ibv:-none}"
fi
if [ -n "${gid_missing}" ]; then
  ok=0
  reason="${reason:+${reason},}gid_missing=${gid_missing}"
fi
if [ -z "${oob_ipv4}" ]; then
  ok=0
  reason="${reason:+${reason},}oob_${oob_if}=down_or_no_ipv4"
fi

if [ "${ok}" -eq 1 ]; then
  echo "OK host=${host} gpus=${gpus} nics=${nics} oob=${oob_if}/${oob_ipv4} fp=${fp}"
else
  echo "BAD host=${host} gpus=${gpus} nics=${nics:-none} ibv=${ibv:-none} oob=${oob_if}/${oob_ipv4:-none} reason=${reason} fp=${fp}"
fi
exit 0
