#!/bin/bash
# ORTE/PRRTE plm_slurm -> SPUR adapter (Crusoe).
#
# SPUR's srun is not Slurm's: it rejects --ntasks-per-node / --kill-on-bad-exit
# and ignores a step's --nodelist. ORTE launches remote daemons with
# `srun --nodes=K --ntasks=K --nodelist=<hosts> orted`, so without this shim
# every daemon starts on the launcher node and all ranks pile onto one host.
#
# Install as `srun` first on PATH before `mpirun --mca plm slurm`. Idle tasks
# in the step park until PARK_SENTINEL disappears so the step outlives the
# first non-daemon exit.
#
# Env (defaults):
#   OMPI           /opt/openmpi
#   SRUN_REAL      /usr/local/bin/srun
#   ROCM_LIB       /opt/rocm/lib
#   PARK_SENTINEL  /dev/shm/crusoe-mpirun.active
#   PARK_MAX_SECS  10800

set -u

OMPI="${OMPI:-/opt/openmpi}"
SRUN_REAL="${SRUN_REAL:-/usr/local/bin/srun}"
ROCM_LIB="${ROCM_LIB:-/opt/rocm/lib}"
PARK_SENTINEL="${PARK_SENTINEL:-/dev/shm/crusoe-mpirun.active}"
PARK_MAX_SECS="${PARK_MAX_SECS:-10800}"

expand_hostlist() {
  python3 - "$1" <<'PY'
import re, sys
spec = (sys.argv[1] or "").strip()
hosts = []
for match in re.finditer(r"([^,\[]+)(?:\[([^\]]+)\])?", spec):
    prefix, inner = match.group(1).strip(" ,"), match.group(2)
    if not inner:
        if prefix:
            hosts.append(prefix)
        continue
    for token in inner.split(","):
        token = token.strip()
        if re.fullmatch(r"\d+-\d+", token):
            start, end = token.split("-", 1)
            width = len(start)
            for index in range(int(start), int(end) + 1):
                hosts.append(f"{prefix}{index:0{width}d}")
        elif token:
            hosts.append(f"{prefix}{token}")
print(",".join(hosts))
PY
}

PRE=(); ORTED=(); NL=""; seen_orted=0
while [ $# -gt 0 ]; do
  if [ "$seen_orted" = 1 ]; then ORTED+=("$1"); shift; continue; fi
  case "$1" in
    --ntasks-per-node) shift 2; continue ;;
    --ntasks-per-node=*|--kill-on-bad-exit|--kill-on-bad-exit=*) shift; continue ;;
    --export) shift 2; continue ;;
    --export=*|--exclusive|--exclusive=*) shift; continue ;;
    --nodes=*|--ntasks=*) shift; continue ;;
    --nodes|--ntasks|-N|-n) shift 2; continue ;;
    --nodelist=*) NL="${1#--nodelist=}"; shift ;;
    --nodelist|-w) NL="$2"; shift 2 ;;
    orted|prted)
      seen_orted=1
      cmd="$1"
      [ "$cmd" = "orted" ] && cmd="$OMPI/bin/orted"
      [ "$cmd" = "prted" ] && cmd="$OMPI/bin/prted"
      ORTED+=("$cmd"); shift ;;
    *) PRE+=("$1"); shift ;;
  esac
done

if [ "${#ORTED[@]}" = 0 ]; then exec "$SRUN_REAL" "${PRE[@]}"; fi

NODES="${SLURM_NNODES:-2}"
if [ -n "$NL" ]; then
  EXPANDED=$(expand_hostlist "$NL")
  [ -n "$EXPANDED" ] && NL="$EXPANDED"
fi
ORTED_Q=$(printf '%q ' "${ORTED[@]}")

TASK=$(cat <<EOF
H=\$(hostname -s); IDX=-1; i=0
for h in \$(echo "$NL" | tr ',' ' '); do [ "\$H" = "\$h" ] && IDX=\$i; i=\$((i + 1)); done
if [ \$IDX -ge 0 ]; then
  export SLURM_PROCID=\$IDX SLURM_NODEID=\$IDX
  export PATH=$OMPI/bin:/opt/rocm/bin:\${PATH:-/usr/bin:/bin}
  export LD_LIBRARY_PATH=$OMPI/lib:$ROCM_LIB:/opt/ucx/lib:\${LD_LIBRARY_PATH:-}
  export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
  export HSA_NO_SCRATCH_RECLAIM=1
  export TMPDIR=/dev/shm
  unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  exec $ORTED_Q
fi
END=\$(( \$(date +%s) + $PARK_MAX_SECS ))
while [ -f "$PARK_SENTINEL" ] && [ \$(date +%s) -lt \$END ]; do sleep 5; done
EOF
)

exec "$SRUN_REAL" "${PRE[@]}" --nodes="$NODES" --ntasks="$NODES" bash -c "$TASK"
