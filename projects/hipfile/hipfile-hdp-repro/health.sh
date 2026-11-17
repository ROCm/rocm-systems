#!/bin/bash
# Read-only snapshot of this node's GPU and PCIe health, for before/after comparison around tests.
# Reads logs, sysfs AER counters, process states and rocm-smi. Changes nothing.
ROCM=${ROCM_PATH:-/opt/rocm}

echo "== $(date -Is) $(hostname) up $(uptime -p 2>/dev/null) load $(cut -d' ' -f1-3 /proc/loadavg)"

echo "== PCIe AER totals for each AMD GPU and every bridge above it (nonzero is bad)"
for dev in /sys/bus/pci/devices/*; do
    [ "$(cat "$dev/vendor" 2>/dev/null)" = 0x1002 ] || continue
    case $(cat "$dev/class" 2>/dev/null) in 0x0300* | 0x0302* | 0x0380*) ;; *) continue ;; esac
    path=$(readlink -f "$dev")
    while [ "$path" != / ] && [ -n "$path" ]; do
        bdf=${path##*/}
        if [[ $bdf =~ ^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$ ]]; then
            line="$bdf"
            for f in aer_dev_correctable aer_dev_nonfatal aer_dev_fatal; do
                [ -r "$path/$f" ] && line+=" ${f#aer_dev_}=$(awk '/TOTAL/{print $NF}' "$path/$f")"
            done
            echo "   $line"
        fi
        path=${path%/*}
    done
done | sort -u

echo "== processes in uninterruptible sleep (D state)"
ps -eo pid,stat,wchan:32,comm | awk 'NR > 1 && $2 ~ /^D/' | head -20
echo "   (end)"

echo "== GPU use and KFD processes"
timeout 20 "$ROCM/bin/rocm-smi" --showuse --showpids 2>&1 | grep -E 'GPU use|KFD|^ *[0-9]+ ' | head -20

echo "== kernel log since boot: counts of trouble markers"
if sudo -n true 2>/dev/null; then
    klog() { timeout 60 sudo -n journalctl -k -b 0 --no-pager -q 2>/dev/null; }
elif dmesg >/dev/null 2>&1; then
    klog() { dmesg 2>/dev/null; }
else
    echo "   kernel log not readable without root; skipped"
    exit 0
fi
log=$(klog)
for pattern in 'flip_done timed out' '\*ERROR\*' 'blocked for more than' 'soft lockup' 'Call Trace' \
               'AER: (Corrected|Uncorrect|Multiple)' 'GPU reset' 'ring .* timeout' 'amdgpu.*(fault|timeout)'; do
    printf '   %-40s %s\n' "$pattern" "$(grep -cE "$pattern" <<<"$log")"
done
echo "== last 5 amdgpu/kfd/drm/AER kernel messages"
grep -iE 'amdgpu|kfd|\[drm\]|AER|pcieport' <<<"$log" | grep -viE 'enabled with IRQ|_OSC|DPC: error containment' | tail -5
