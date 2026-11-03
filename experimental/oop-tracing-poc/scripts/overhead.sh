#!/usr/bin/env bash
# Local POC: overhead of probes, LTTng and the dispatch-log collector. Runs inside the container.
# Output: /w/overhead/results.csv  config,segment,rep,ns
set -u
O=/w/overhead; rm -rf $O; mkdir -p $O
echo "config,segment,rep,ns" > $O/results.csv
pgrep -x lttng-sessiond >/dev/null || { lttng-sessiond --daemonize >/dev/null 2>&1; sleep 1; }
PROBED=/w/clr-build/hipamd/lib

run() {  # config segment rep
  local cfg=$1 seg=$2 rep=$3 lib="" lt=0 dl=0
  case $cfg in
    stock) ;;
    probed_idle) lib=$PROBED ;;
    probed_lttng) lib=$PROBED; lt=1 ;;
    probed_lttng_dlog) lib=$PROBED; lt=1; dl=1 ;;
    stock_dlog) dl=1 ;;
  esac
  local d=$O/$cfg-$seg-$rep; mkdir -p $d
  if [ $lt = 1 ]; then
    lttng create b --output=$d/lttng >/dev/null
    lttng enable-channel -u ch --subbuf-size=4M --num-subbuf=8 >/dev/null
    lttng enable-event -u -c ch "rocm_hip:*" >/dev/null
    lttng add-context -u -c ch -t vpid -t vtid >/dev/null
    lttng start >/dev/null
  fi
  LD_LIBRARY_PATH=$lib /w/bench $seg $d/go > $d/out 2>&1 &
  local app=$!
  until grep -q waiting $d/out 2>/dev/null; do sleep 0.02; done
  local col=""
  if [ $dl = 1 ]; then
    /w/dlogc $app $d > $d/dlogc.out 2> $d/dlogc.err & col=$!
    until grep -q ready $d/dlogc.out 2>/dev/null || ! kill -0 $col 2>/dev/null; do sleep 0.02; done
  fi
  touch $d/go; wait $app
  [ -n "$col" ] && wait $col
  if [ $lt = 1 ]; then
    lttng stop >/dev/null
    local disc; disc=$(lttng list b | awk "/Discarded events:/{s+=\$3} END{print s+0}")
    lttng destroy >/dev/null; rm -rf $d/lttng
    echo "$cfg,$seg,$rep,discarded=$disc" >> $O/notes.txt
  fi
  grep -E "^$seg," $d/out | sed "s/^/$cfg,/; s/,$seg,/,$seg,$rep,/" | awk -F, -v c=$cfg -v s=$seg -v r=$rep '{print c","s","r","$3}' >> $O/results.csv
}

CFGS=(stock probed_idle probed_lttng probed_lttng_dlog stock_dlog)
for seg in calls launches; do
  for rep in 1 2 3 4 5; do
    order=("${CFGS[@]}")
    if [ $((rep % 2)) = 0 ]; then order=(); for ((i=${#CFGS[@]}-1; i>=0; i--)); do order+=("${CFGS[$i]}"); done; fi
    for c in "${order[@]}"; do run $c $seg $rep; done
  done
done
echo done
