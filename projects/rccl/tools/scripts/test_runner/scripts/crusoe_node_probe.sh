#!/bin/bash
# 1-node Crusoe AINIC probe: 8x gfx950, ionic_0..7 in sysfs and ibv_devices,
# and a CPU/NUMA/NIC-map fingerprint so the launcher can pair symmetric nodes.
# Prints one line: "OK host=... nics=... fp=..." or "BAD host=... reason=...".

set +e
export PATH="/opt/rocm/bin:/usr/bin:/usr/sbin:/bin:${PATH:-}"

host=$(hostname -s)
want="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"

gpus=$(rocminfo 2>/dev/null | grep -c gfx950)
gpus=${gpus:-0}

nics=$(ls -1 /sys/class/infiniband 2>/dev/null | grep -E '^ionic_[0-7]$' | sort | paste -sd, -)
ibv=$(ibv_devices 2>/dev/null | awk 'NF && $1 ~ /^ionic_[0-7]$/ {print $1}' | sort | paste -sd, -)

cpu=$(grep -E '^(cpu family|model|model name|stepping)' /proc/cpuinfo | sort -u | tr '\n' '|')
nnuma=$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)
ncpu=$(nproc 2>/dev/null || echo 0)
nic_numa=""
for i in 0 1 2 3 4 5 6 7; do
  nn=$(cat "/sys/class/infiniband/ionic_${i}/device/numa_node" 2>/dev/null || echo x)
  nic_numa="${nic_numa:+${nic_numa},}ionic_${i}:${nn}"
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

if [ "${ok}" -eq 1 ]; then
  echo "OK host=${host} gpus=${gpus} nics=${nics} fp=${fp}"
else
  echo "BAD host=${host} gpus=${gpus} nics=${nics:-none} ibv=${ibv:-none} reason=${reason} fp=${fp}"
fi
exit 0
