#!/bin/bash
# Is the 4 MB L2 figure a reporting bug or something deeper?
#
# rocminfo, amd-smi and hipDeviceProp all say 4 MB, but they are not independent
# witnesses - all three read the KFD topology the driver publishes in sysfs.
# So go to that source and see exactly what the driver is claiming: how many L2
# entries exist, what size each one is, and how many CUs each says it serves.
#
# Two very different possibilities:
#   (a) the driver publishes ONE 4 MB L2 -> it is simply wrong about the size
#   (b) the driver publishes N x 4 MB entries and the tools report only one of
#       them -> the driver is right and the aggregation in userspace is wrong
set -uo pipefail

echo "=== GPU nodes in KFD topology ==="
for n in /sys/class/kfd/kfd/topology/nodes/*/; do
  simd=$(awk '/^simd_count/{print $2}' "$n/properties" 2>/dev/null)
  [ -z "$simd" ] && continue
  [ "$simd" = "0" ] && continue
  echo "node: $n"
  awk '/^(simd_count|array_count|cu_per_simd_array|simd_per_cu|num_xcc|gfx_target_version|location_id)/' \
      "$n/properties"
  echo
done

echo "=== every cache entry the driver publishes for the GPU node ==="
printf '%-6s %-10s %-12s %-10s %-12s %s\n' "level" "size_KB" "cu_shared" "assoc" "lines" "type"
for n in /sys/class/kfd/kfd/topology/nodes/*/; do
  simd=$(awk '/^simd_count/{print $2}' "$n/properties" 2>/dev/null)
  [ -z "$simd" ] || [ "$simd" = "0" ] && continue
  for c in "$n"caches/*/; do
    [ -d "$c" ] || continue
    lvl=$(awk '/^level/{print $2}' "$c/properties")
    sz=$(awk '/^size/{print $2}'  "$c/properties")
    cu=$(awk '/^cu_cnt/{print $2}' "$c/properties")
    as=$(awk '/^association/{print $2}' "$c/properties")
    ln=$(awk '/^cache_line_size/{print $2}' "$c/properties")
    ty=$(awk '/^type/{print $2}' "$c/properties")
    printf '%-6s %-10s %-12s %-10s %-12s %s\n' "$lvl" "$sz" "$cu" "$as" "$ln" "$ty"
  done
done

echo
echo "=== count and total capacity per cache level ==="
for n in /sys/class/kfd/kfd/topology/nodes/*/; do
  simd=$(awk '/^simd_count/{print $2}' "$n/properties" 2>/dev/null)
  [ -z "$simd" ] || [ "$simd" = "0" ] && continue
  for c in "$n"caches/*/; do
    [ -d "$c" ] || continue
    awk '/^level/{l=$2} /^size/{s=$2} END{print l, s}' "$c/properties"
  done
done | sort -n | awk '
  { count[$1]++; total[$1] += $2 }
  END {
    printf "%-8s %-10s %-14s %s\n", "level", "entries", "each_KB(avg)", "aggregate"
    for (l in count) {
      avg = total[l] / count[l]
      if (total[l] >= 1024)
        printf "%-8s %-10d %-14.0f %.1f MB\n", l, count[l], avg, total[l]/1024
      else
        printf "%-8s %-10d %-14.0f %d KB\n", l, count[l], avg, total[l]
    }
  }'

echo
echo "=== what the L2 entry says verbatim ==="
for n in /sys/class/kfd/kfd/topology/nodes/*/; do
  for c in "$n"caches/*/; do
    [ -d "$c" ] || continue
    if awk '/^level 2$/{f=1} END{exit !f}' "$c/properties" 2>/dev/null; then
      echo "--- $c"
      cat "$c/properties"
      break 2
    fi
  done
done
echo "KFD_CACHE_DONE"
