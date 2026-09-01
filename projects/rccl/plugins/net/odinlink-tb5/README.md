# OdinLink Thunderbolt 5 / USB4 RCCL Net Plugin

RCCL/NCCL network plugin that carries collectives over Thunderbolt 5, Thunderbolt 4,
and USB4 links using the OdinLink host-pointer DMA path.

The plugin is a single shared library, `librccl-net-odl_tb5.so`. RCCL loads it when
`NCCL_NET_PLUGIN=odl_tb5` (or when the library is named `librccl-net.so` on the
dynamic-loader path).

Runtime I/O goes through `/dev/odl_tb5_N`, so the OdinLink kernel module
(`odl_tb5.ko`) must already be loaded. The driver is not part of RCCL; build and
install it from [OdinLink-Five](https://github.com/Geramy/OdinLink-Five).

## Layout

| Path | Role |
|------|------|
| `plugin.c` | NCCL net ABI (v6–v12) and per-comm worker threads |
| `lib/` | Userspace `libodl_tb5` sources, compiled into the plugin |
| `include/odl_tb5/` | Device ioctl / stream API headers |
| `nccl/` | Vendored net plugin ABI headers |

GPU buffers are staged to host memory by RCCL (`ptrSupport = NCCL_PTR_HOST`).
Each send/recv comm owns one TB5 stream; worker threads issue the blocking
stream ioctls so RCCL's proxy thread does not deadlock on bidirectional traffic.

## Build

Standalone:

```bash
cd plugins/net/odinlink-tb5
make -j$(nproc)
# produces librccl-net-odl_tb5.so
```

Or with CMake:

```bash
cmake -S plugins/net/odinlink-tb5 -B /tmp/odl-tb5-plugin
cmake --build /tmp/odl-tb5-plugin -j$(nproc)
```

In-tree (optional): configure RCCL with
`BUILD_TESTS=ON -DBUILD_PLUGIN_EXAMPLES=ON`.

## Use

Disable PCIe ASPM on every node before running. ASPM power-saving states add
microseconds of exit latency on the Thunderbolt / USB4 tunnel; `performance`
keeps the link fully active.

```bash
echo performance | sudo tee /sys/module/pcie_aspm/parameters/policy
cat /sys/module/pcie_aspm/parameters/policy
# expect: [performance] ...

# Kernel module from OdinLink-Five
sudo insmod /path/to/odl_tb5.ko

export NCCL_NET_PLUGIN=odl_tb5
export LD_LIBRARY_PATH=/path/to/this/plugin:$LD_LIBRARY_PATH
# Optional: pin the plugin by name
export NCCL_NET=OdinLink_TB5
```

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `ODL_TB5_VIRTUAL_SINGLE_DEV` | `1` | Present all TB5 ports as one virtual NIC and pick the route from the listen handle. 2-node (one `/dev/odl_tb5_0`) and 4-node mesh share this path; connect always handshakes so windowed ACKs can reverse to the sender stream. |
| `ODL_TB5_FORCE_DEV_INDEX` | unset | Force a kernel device index (`/dev/odl_tb5_N`). Optional; not required for 2-node. |
| `ODL_TB5_RECV_TIMEOUT_MS` | `30000` | Receive wait timeout |
| `ODL_TB5_TRACE` | off | Log plugin calls to stderr |
| `ODL_TB5_PROFILE` | off | Print periodic TX/RX timing counters |
| `ODL_TB5_PROFILE_INTERVAL` | `1024` | Ops between profile prints |

Shared-memory stats are published at `/run/odl_tb5/rccl_stats` for the OdinLink daemon.

## License

Userspace plugin and library (`plugin.c`, `lib/`, `include/odl_tb5/`): **MIT**,
from [OdinLink-Five](https://github.com/Geramy/OdinLink-Five). See [`LICENSE`](LICENSE).

NCCL ABI headers under `nccl/` keep their upstream NVIDIA copyright
(BSD / Apache-2.0 as marked in each file).

The kernel driver `odl_tb5.ko` is **GPL-2.0** and is **not** vendored in this
tree. Obtain it from OdinLink-Five.
