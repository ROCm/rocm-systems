#!/usr/bin/env bash
# Local POC: per-dispatch counters, new design (runtime hook + selector) vs rocprofv3 --pmc.
# Runs inside the container as root with CAP_PERFMON. Output: /w/counters/results.csv
#   config,rounds,rep,gpu_ns,wall_ns   (one row per 2000-kernel segment, 1 stream)
set -u
O=/w/counters; rm -rf $O; mkdir -p $O
echo "config,rounds,rep,gpu_ns,wall_ns" > $O/results.csv
CTRS="SQ_WAVES GRBM_COUNT"
run() {  # config rounds rep
  local cfg=$1 rounds=$2 rep=$3; local d=$O/$cfg-r$rounds-$rep; mkdir -p $d
  export BENCH_SETTLE_MS=3000
  case $cfg in
    stock)  /w/bench multi "" 1 $rounds > $d/out 2> $d/err ;;
    probed) LD_LIBRARY_PATH=/w/clr-build/hipamd/lib /w/bench multi "" 1 $rounds > $d/out 2> $d/err ;;
    poc)    POC_NO_FINAL_HARVEST=1 POC_MODE=user_only POC_SELECTOR=/w/libpoc_selector.so POC_SELECTOR_OUT=$d/poc.csv \
            POC_COUNTERS=SQ_WAVES,GRBM_COUNT POC_POOL=4096 LD_LIBRARY_PATH=/w/clr-build/hipamd/lib \
            /w/bench multi "" 1 $rounds > $d/out 2> $d/err ;;
    rc|rc_ser) RCENV="LD_LIBRARY_PATH=/w/rpreg-build/lib:/w/clr-build/hipamd/lib ROCPROFILER_REGISTER_LIBRARY=/w/libpoc_selector_rc.so POC_MODE=user_only POC_POOL=4096 POC_SVC_EXECUTABLE=1 POC_SELECTOR_OUT=$d/poc.csv"
            [ $cfg = rc_ser ] && RCENV="$RCENV POC_SERIALIZE=1"
            env $RCENV /w/bench multi "" 1 $rounds > $d/out 2> $d/err ;;
    rocprofv3) ROCR_VISIBLE_DEVICES=0 /opt/rocm/bin/rocprofv3 --pmc $CTRS --output-format csv \
            -d $d/rp -o run -- /w/bench multi "" 1 $rounds > $d/out 2> $d/err ;;
  esac
  grep "^multi," $d/out | awk -F, -v c=$cfg -v r=$rounds -v p=$rep '{print c","r","p","$4","$5}' >> $O/results.csv
}
CFGS=(stock rocprofv3 rc rc_ser)
for rounds in 500 2000 10000; do
  for rep in 1 2 3 4 5; do
    order=("${CFGS[@]}")
    if [ $((rep % 2)) = 0 ]; then order=(); for ((i=${#CFGS[@]}-1; i>=0; i--)); do order+=("${CFGS[$i]}"); done; fi
    for c in "${order[@]}"; do run $c $rounds $rep; done
  done
done
chown -R "$(stat -c %u:%g /w)" $O
echo done
