import argparse
import os
import struct
import sys

from .ioctl import (
    ALGO, DTYPE, ELEMOP, STAT_NAMES,
    sdxi_info, sdxi_stats, sdxi_copy,
    sdxi_compress, sdxi_decompress, sdxi_elemwise,
    sdxi_dmabuf_copy,
)


def _open_dev(path: str) -> int:
    try:
        return os.open(path, os.O_RDWR)
    except OSError as e:
        print(f"sdxi-cli: cannot open {path}: {e.strerror}", file=sys.stderr)
        sys.exit(1)


def _parse_scalar(s: str, dtype_name: str) -> int:
    if s.lower().startswith("0x"):
        return int(s, 16) & 0xFFFFFFFFFFFFFFFF
    if dtype_name == "fp32":
        return struct.unpack("=I", struct.pack("=f", float(s)))[0]
    return int(s) & 0xFFFFFFFFFFFFFFFF


def _cmd_info(args) -> None:
    fd = _open_dev(args.device)
    try:
        d = sdxi_info(fd)
    finally:
        os.close(fd)
    major = (d["version"] >> 8) & 0xFF
    minor = d["version"] & 0xFF
    print(f"version:   {major}.{minor}")
    print(f"vendor_id: 0x{d['vendor_id']:04x}")
    print(f"device_id: 0x{d['device_id']:04x}")
    print(f"max_cxt:   {d['max_cxt']}")
    print(f"cap0:      0x{d['cap0']:016x}")
    print(f"cap1:      0x{d['cap1']:016x}")


def _cmd_stats(args) -> None:
    fd = _open_dev(args.device)
    try:
        vals = sdxi_stats(fd, args.cid)
    finally:
        os.close(fd)
    width = max(len(n) for n in STAT_NAMES)
    for name, val in zip(STAT_NAMES, vals):
        print(f"{name:<{width}}  {val}")


def _cmd_copy(args) -> None:
    data = open(args.src, "rb").read()
    fd = _open_dev(args.device)
    try:
        out = sdxi_copy(fd, data)
    finally:
        os.close(fd)
    open(args.dst, "wb").write(out)
    if args.verbose:
        print(f"copied {len(data)} bytes")


def _cmd_compress(args) -> None:
    data = open(args.src, "rb").read()
    algo = ALGO[args.algo]
    fd = _open_dev(args.device)
    try:
        out = sdxi_compress(fd, data, algo=algo, level=args.level)
    finally:
        os.close(fd)
    open(args.dst, "wb").write(out)
    if args.verbose:
        print(f"compressed {len(data)} -> {len(out)} bytes")


def _cmd_decompress(args) -> None:
    data = open(args.src, "rb").read()
    algo = ALGO[args.algo]
    dst_cap = args.dst_cap if args.dst_cap else len(data) * 16
    fd = _open_dev(args.device)
    try:
        out = sdxi_decompress(fd, data, dst_cap=dst_cap, algo=algo)
    finally:
        os.close(fd)
    open(args.dst, "wb").write(out)
    if args.verbose:
        print(f"decompressed {len(data)} -> {len(out)} bytes")


def _cmd_dmabuf_copy(args) -> None:
    fd = _open_dev(args.device)
    try:
        sdxi_dmabuf_copy(fd, args.src_fd, args.dst_fd, args.len,
                         src_offset=args.src_offset, dst_offset=args.dst_offset)
    finally:
        os.close(fd)
    if args.verbose:
        print(f"dmabuf-copy: {args.len} bytes src_fd={args.src_fd} dst_fd={args.dst_fd}")


def _cmd_elemwise(args) -> None:
    data = open(args.src, "rb").read()
    dtype_val = DTYPE[args.dtype]
    op_val = ELEMOP[args.op]
    scalar = _parse_scalar(args.scalar, args.dtype)
    fd = _open_dev(args.device)
    try:
        out = sdxi_elemwise(fd, data, dtype=dtype_val, op=op_val, scalar=scalar)
    finally:
        os.close(fd)
    open(args.dst, "wb").write(out)
    if args.verbose:
        print(f"elemwise {args.op} {args.dtype}: {len(data)} -> {len(out)} bytes")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sdxi-cli",
        description="Command-line tool for the SDXI char device",
    )
    p.add_argument("-d", "--device", default="/dev/sdxi0", metavar="DEV",
                   help="SDXI device path (default: /dev/sdxi0)")
    p.add_argument("-v", "--verbose", action="store_true")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="print device version, PCI IDs, and capabilities")

    sp = sub.add_parser("stats", help="print all 14 performance counters for a context")
    sp.add_argument("cid", nargs="?", type=int, default=0, metavar="CID",
                    help="context id (default: 0)")

    sp = sub.add_parser("copy", help="DMA copy a file")
    sp.add_argument("src")
    sp.add_argument("dst")

    sp = sub.add_parser("compress", help="compress a file via SDXI")
    sp.add_argument("src")
    sp.add_argument("dst")
    sp.add_argument("-a", "--algo", choices=list(ALGO), default="zlib")
    sp.add_argument("-l", "--level", type=int, default=0, metavar="N",
                    help="compression level (0 = default)")

    sp = sub.add_parser("decompress", help="decompress a file via SDXI")
    sp.add_argument("src")
    sp.add_argument("dst")
    sp.add_argument("-a", "--algo", choices=list(ALGO), default="zlib")
    sp.add_argument("--dst-cap", type=int, default=0, metavar="N",
                    help="output buffer capacity in bytes (default: 16 * input size)")

    sp = sub.add_parser("dmabuf-copy",
                        help="DMA copy between two dma-buf file descriptors")
    sp.add_argument("src_fd", type=int, metavar="SRC_FD",
                    help="open dma-buf fd for source")
    sp.add_argument("dst_fd", type=int, metavar="DST_FD",
                    help="open dma-buf fd for destination")
    sp.add_argument("len", type=int, metavar="LEN",
                    help="bytes to copy")
    sp.add_argument("--src-offset", type=int, default=0, metavar="N",
                    help="byte offset into source dma-buf (default: 0)")
    sp.add_argument("--dst-offset", type=int, default=0, metavar="N",
                    help="byte offset into destination dma-buf (default: 0)")

    sp = sub.add_parser("elemwise", help="element-wise operation via SDXI")
    sp.add_argument("op", choices=list(ELEMOP))
    sp.add_argument("dtype", choices=list(DTYPE))
    sp.add_argument("scalar",
                    help="scalar value: decimal int/float, or 0x hex for raw bits")
    sp.add_argument("src")
    sp.add_argument("dst")

    return p


def main(argv=None) -> None:
    parser = _build_parser()
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass
    args = parser.parse_args(argv)

    dispatch = {
        "info":         _cmd_info,
        "stats":        _cmd_stats,
        "copy":         _cmd_copy,
        "compress":     _cmd_compress,
        "decompress":   _cmd_decompress,
        "elemwise":     _cmd_elemwise,
        "dmabuf-copy":  _cmd_dmabuf_copy,
    }
    try:
        dispatch[args.cmd](args)
    except OSError as e:
        print(f"sdxi-cli: {e.strerror}", file=sys.stderr)
        sys.exit(1)
