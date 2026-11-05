import ctypes
import fcntl
import struct

_IOC_WRITE = 1
_IOC_READ  = 2

def _IOC(dir_, type_, nr, size):
    return (dir_ << 30) | (size << 16) | (ord(type_) << 8) | nr

def _IOW(t, nr, size):  return _IOC(_IOC_WRITE,            t, nr, size)
def _IOR(t, nr, size):  return _IOC(_IOC_READ,             t, nr, size)
def _IOWR(t, nr, size): return _IOC(_IOC_READ | _IOC_WRITE, t, nr, size)

_M = 'X'

_SZ_MEMCPY     = struct.calcsize("=QQQ")
_SZ_COMPRESS   = struct.calcsize("=QQQQIIQ")
_SZ_DECOMPRESS = struct.calcsize("=QQQQIIQ")
_SZ_ELEMWISE   = struct.calcsize("=QQQIIQQ")
_SZ_INFO       = struct.calcsize("=IHHIIQQ")
_SZ_STATS      = struct.calcsize("=II14Q")

SDXI_IOC_MEMCPY     = _IOW(_M,  1, _SZ_MEMCPY)
SDXI_IOC_COMPRESS   = _IOWR(_M, 2, _SZ_COMPRESS)
SDXI_IOC_DECOMPRESS = _IOWR(_M, 3, _SZ_DECOMPRESS)
SDXI_IOC_ELEMWISE   = _IOWR(_M, 4, _SZ_ELEMWISE)
SDXI_IOC_INFO       = _IOR(_M,  5, _SZ_INFO)
SDXI_IOC_STATS      = _IOWR(_M, 6, _SZ_STATS)

ALGO = {"zlib": 0, "lz4": 1, "zstd": 2}

DTYPE = {
    "fp32": 0, "fp16": 1, "bf16": 2, "int8": 3,
    "int32": 4, "fp8e4m3": 5, "fp8e5m2": 6, "int4": 7,
}

ELEMOP = {
    "add": 0, "min": 1, "max": 2, "reduce_min": 3, "reduce_max": 4,
}

STAT_NAMES = [
    "bytes_moved",        # 0x00
    "num_copy",           # 0x01
    "num_errors",         # 0x02
    "num_intr",           # 0x03
    "num_cxt_start",      # 0x04
    "num_cxt_stop",       # 0x05
    "num_nop",            # 0x06
    "num_sync",           # 0x07
    "bytes_compressed",   # 0x08
    "bytes_decompressed", # 0x09
    "num_compress",       # 0x0A
    "num_decompress",     # 0x0B
    "num_elemwise",       # 0x0C
    "bytes_elemwise",     # 0x0D
]


def sdxi_info(fd: int) -> dict:
    buf = ctypes.create_string_buffer(_SZ_INFO)
    fcntl.ioctl(fd, SDXI_IOC_INFO, buf)
    version, vendor_id, device_id, max_cxt, _pad, cap0, cap1 = struct.unpack_from("=IHHIIQQ", buf)
    return {
        "version": version,
        "vendor_id": vendor_id,
        "device_id": device_id,
        "max_cxt": max_cxt,
        "cap0": cap0,
        "cap1": cap1,
    }


def sdxi_stats(fd: int, cid: int) -> list:
    buf = ctypes.create_string_buffer(_SZ_STATS)
    struct.pack_into("=II", buf, 0, cid, 0)
    fcntl.ioctl(fd, SDXI_IOC_STATS, buf)
    return list(struct.unpack_from("=14Q", buf, 8))


def sdxi_copy(fd: int, data: bytes) -> bytes:
    n = len(data)
    src_buf = ctypes.create_string_buffer(data, n)
    dst_buf = ctypes.create_string_buffer(n)
    ioc_buf = ctypes.create_string_buffer(_SZ_MEMCPY)
    struct.pack_into("=QQQ", ioc_buf, 0, ctypes.addressof(src_buf), ctypes.addressof(dst_buf), n)
    fcntl.ioctl(fd, SDXI_IOC_MEMCPY, ioc_buf)
    return bytes(dst_buf)


def sdxi_compress(fd: int, data: bytes, algo: int = 0, level: int = 0) -> bytes:
    n = len(data)
    dst_cap = n + n // 100 + 64
    src_buf = ctypes.create_string_buffer(data, n)
    dst_buf = ctypes.create_string_buffer(dst_cap)
    ioc_buf = ctypes.create_string_buffer(_SZ_COMPRESS)
    struct.pack_into("=QQQQIIQ", ioc_buf, 0,
                     ctypes.addressof(src_buf), n,
                     ctypes.addressof(dst_buf), dst_cap,
                     level, algo, 0)
    fcntl.ioctl(fd, SDXI_IOC_COMPRESS, ioc_buf)
    out_len = struct.unpack_from("=Q", ioc_buf, 40)[0]
    return bytes(dst_buf[:out_len])


def sdxi_decompress(fd: int, data: bytes, dst_cap: int, algo: int = 0) -> bytes:
    n = len(data)
    src_buf = ctypes.create_string_buffer(data, n)
    dst_buf = ctypes.create_string_buffer(dst_cap)
    ioc_buf = ctypes.create_string_buffer(_SZ_DECOMPRESS)
    struct.pack_into("=QQQQIIQ", ioc_buf, 0,
                     ctypes.addressof(src_buf), n,
                     ctypes.addressof(dst_buf), dst_cap,
                     algo, 0, 0)
    fcntl.ioctl(fd, SDXI_IOC_DECOMPRESS, ioc_buf)
    out_len = struct.unpack_from("=Q", ioc_buf, 40)[0]
    return bytes(dst_buf[:out_len])


def sdxi_elemwise(fd: int, data: bytes, dtype: int, op: int, scalar: int) -> bytes:
    n = len(data)
    src_buf = ctypes.create_string_buffer(data, n)
    dst_buf = ctypes.create_string_buffer(n)
    ioc_buf = ctypes.create_string_buffer(_SZ_ELEMWISE)
    struct.pack_into("=QQQIIQQ", ioc_buf, 0,
                     ctypes.addressof(src_buf), n,
                     ctypes.addressof(dst_buf),
                     dtype, op, scalar, 0)
    fcntl.ioctl(fd, SDXI_IOC_ELEMWISE, ioc_buf)
    out_len = struct.unpack_from("=Q", ioc_buf, 40)[0]
    return bytes(dst_buf[:out_len])
