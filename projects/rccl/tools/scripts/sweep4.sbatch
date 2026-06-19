#!/usr/bin/env bash
#SBATCH -N4
#SBATCH --ntasks-per-node=1
#SBATCH -c4
#SBATCH -t10
#SBATCH -J bnp_sweep
#SBATCH -o /home/ilkosare/.bnp_runs/sweep4_%j.log
set -u
cd /home/ilkosare/development/rocm-systems/projects/rccl
OUT="$HOME/.bnp_runs/sweep4_job${SLURM_JOB_ID}"; mkdir -p "$OUT"
echo "OUT=$OUT nodes=$SLURM_JOB_NODELIST"
PROBE=./tools/scripts/bootstrap_net_probe
run(){ tag=$1; shift; d="$OUT/$tag"; rdv="$OUT/.rdv_$tag"; mkdir -p "$d" "$rdv"
  env "$@" PROBE_IFACE=eno8303 PROBE_ITERS=500 PROBE_WARMUP=5 PROBE_OUT="$d" PROBE_RDV="$rdv" PROBE_TAG="$tag" \
    srun -N4 --ntasks-per-node=1 $PROBE 2>"$d/stderr.txt"
  echo "$tag done rows=$(cat "$d"/rank_*.csv 2>/dev/null|grep -c ,iter,)"; }
run baseline  PROBE_NODELAY=1 PROBE_QUICKACK=0
run nagle_on  PROBE_NODELAY=0 PROBE_QUICKACK=0
run quickack  PROBE_NODELAY=1 PROBE_QUICKACK=1
run busypoll  PROBE_NODELAY=1 PROBE_QUICKACK=0 PROBE_BUSYPOLL=50
run bigbuf    PROBE_NODELAY=1 PROBE_QUICKACK=0 PROBE_SNDBUF=4194304 PROBE_RCVBUF=4194304
echo "ALL_DONE $OUT"
