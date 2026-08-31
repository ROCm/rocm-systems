#!/bin/bash
# Capture everything needed to make later benchmark numbers reproducible / comparable.
set -u

echo "=================== HOST ==================="
hostname
uname -r
date -u '+%Y-%m-%dT%H:%M:%SZ (UTC)'

echo "=================== ROCM ==================="
cat /opt/rocm/.info/version
/opt/rocm/llvm/bin/clang --version | head -2
/opt/rocm/bin/hipcc --version 2>/dev/null | head -3

echo "=================== GPU AGENTS ==================="
/opt/rocm/bin/rocminfo 2>/dev/null | awk '
  /^Agent /            { agent=$2 }
  /Marketing Name/     { mk=$0 }
  /^ *Name: *gfx/      { print "Agent " agent ": " $2 }
  /Compute Unit:/      { cu=$3 }
  /Max Waves Per CU/   { print "   CUs=" cu "  " $0 }
'
echo "--- gfx targets present ---"
/opt/rocm/bin/rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-z]+' | sort -u

echo "=================== CACHE HIERARCHY (GPU agent) ==================="
# rocminfo prints a Cache Info block per agent; show the GPU one.
/opt/rocm/bin/rocminfo 2>/dev/null | awk '/Agent /{a++} a>1' | grep -A12 'Cache Info' | head -40

echo "=================== MEMORY / CLOCKS ==================="
amd-smi static -g 0 2>/dev/null | grep -iE 'MARKET_NAME|VRAM_SIZE|SIZE|VENDOR|TARGET_GRAPHICS_VERSION|NUM_COMPUTE_UNITS' | head -20
echo "--- current clocks / power ---"
amd-smi metric -g 0 2>/dev/null | grep -iE 'GFX_[0-9]|SOCKET_POWER|HOTSPOT|MEM_[0-9]|THROTTLE|GFXCLK|MCLK' | head -25

echo "=================== TOPOLOGY ==================="
echo -n "KFD nodes: "; ls /sys/class/kfd/kfd/topology/nodes/ | wc -l
echo -n "HIP visible devices: "; /opt/rocm/bin/rocminfo 2>/dev/null | grep -c 'Device Type:.*GPU'

echo "=================== OTHER USERS ON GPU ==================="
amd-smi process 2>/dev/null | head -20 || echo "(amd-smi process unavailable)"
