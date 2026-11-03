#!/usr/bin/env bash
# Local POC: GPU-side cost of the CP dispatch log vs number of streams and kernel length.
# Runs inside the container. Output: /w/multi/results.csv  config,streams,rounds,rep,ns
set -u
O=/w/multi; rm -rf $O; mkdir -p $O
echo "config,streams,rounds,rep,ns" > $O/results.csv
run() {  # config streams rounds rep
  local cfg=$1 ns=$2 rounds=$3 rep=$4
  local d=$O/$cfg-s$ns-r$rounds-$rep; mkdir -p $d
  /w/bench multi $d/go $ns $rounds > $d/out 2>&1 &
  local app=$! col=""
  until grep -q waiting $d/out 2>/dev/null; do sleep 0.02; done
  if [ $cfg = dlog ]; then
    /w/dlogc $app $d > $d/dlogc.out 2> $d/dlogc.err & col=$!
    until grep -q ready $d/dlogc.out 2>/dev/null || ! kill -0 $col 2>/dev/null; do sleep 0.02; done
  fi
  touch $d/go; wait $app; [ -n "$col" ] && wait $col
  grep "^multi," $d/out | awk -F, -v c=$cfg -v r=$rep '{print c","$2","$3","r","$4}' >> $O/results.csv
}
for rounds in 500 2000 10000; do
  for ns in 1 2 4; do
    for rep in 1 2 3 4 5; do
      if [ $((rep % 2)) = 1 ]; then run stock $ns $rounds $rep; run dlog $ns $rounds $rep
      else run dlog $ns $rounds $rep; run stock $ns $rounds $rep; fi
    done
  done
done
echo done
