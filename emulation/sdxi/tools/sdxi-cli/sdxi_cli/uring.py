import ctypes
import mmap
import os
import struct

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.syscall.restype = ctypes.c_long

_SYS_io_uring_setup  = 425
_SYS_io_uring_enter  = 426
_IORING_OP_URING_CMD = 46
_IORING_ENTER_GETEVENTS = 1
_IORING_OFF_SQ_RING  = 0
_IORING_OFF_CQ_RING  = 0x8000000
_IORING_OFF_SQES     = 0x10000000
_PARAMS_SIZE = 120
_SQE_SIZE    = 64
_CQE_SIZE    = 16


def _pageup(n: int) -> int:
    return (n + 4095) & ~4095


class SdxiRing:
    """Single-queue io_uring instance for SDXI uring_cmd submissions."""

    def __init__(self, entries: int = 64) -> None:
        params = ctypes.create_string_buffer(_PARAMS_SIZE)
        fd = int(_libc.syscall(
            ctypes.c_long(_SYS_io_uring_setup),
            ctypes.c_uint(entries),
            ctypes.c_ulong(ctypes.addressof(params)),
        ))
        if fd < 0:
            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
        self._fd = fd

        p = bytes(params)
        sq_entries  = struct.unpack_from("=I", p, 0)[0]
        cq_entries  = struct.unpack_from("=I", p, 4)[0]

        # sq_off is at byte 40 in io_uring_params; cq_off at byte 80
        sq_tail_off  = struct.unpack_from("=I", p, 44)[0]
        sq_mask_off  = struct.unpack_from("=I", p, 48)[0]
        sq_array_off = struct.unpack_from("=I", p, 64)[0]
        cq_head_off  = struct.unpack_from("=I", p, 80)[0]
        cq_mask_off  = struct.unpack_from("=I", p, 88)[0]
        cq_cqes_off  = struct.unpack_from("=I", p, 100)[0]

        sq_ring_sz = _pageup(sq_array_off + sq_entries * 4)
        sqes_sz    = _pageup(sq_entries * _SQE_SIZE)
        cq_ring_sz = _pageup(cq_cqes_off + cq_entries * _CQE_SIZE)

        self._sq_ring = mmap.mmap(fd, sq_ring_sz, offset=_IORING_OFF_SQ_RING)
        self._sqes    = mmap.mmap(fd, sqes_sz,    offset=_IORING_OFF_SQES)
        self._cq_ring = mmap.mmap(fd, cq_ring_sz, offset=_IORING_OFF_CQ_RING)

        self._sq_entries  = sq_entries
        self._sq_tail_off  = sq_tail_off
        self._sq_mask_off  = sq_mask_off
        self._sq_array_off = sq_array_off
        self._cq_head_off  = cq_head_off
        self._cq_mask_off  = cq_mask_off
        self._cq_cqes_off  = cq_cqes_off
        self._sq_tail = struct.unpack_from("=I", self._sq_ring, sq_tail_off)[0]

    def close(self) -> None:
        self._sq_ring.close()
        self._sqes.close()
        self._cq_ring.close()
        os.close(self._fd)

    def __enter__(self) -> "SdxiRing":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def submit_batch(self, dev_fd: int, ops: list) -> list:
        """Submit a batch of (cmd_op, ioc_buf) operations.

        Returns a list of kernel result codes (0 = success, negative = -errno).
        """
        n = len(ops)
        if n == 0:
            return []
        if n > self._sq_entries:
            raise ValueError(f"batch {n} > ring capacity {self._sq_entries}")

        mask = struct.unpack_from("=I", self._sq_ring, self._sq_mask_off)[0]
        tail = self._sq_tail
        _bufs = []  # keep ctypes buffers alive until io_uring_enter returns

        for i, (cmd_op, ioc_buf) in enumerate(ops):
            idx = tail & mask
            _bufs.append(ioc_buf)
            addr = ctypes.addressof(ioc_buf)

            sqe = bytearray(_SQE_SIZE)
            struct.pack_into("=BBHi", sqe,  0, _IORING_OP_URING_CMD, 0, 0, dev_fd)
            struct.pack_into("=II",   sqe,  8, cmd_op, 0)   # cmd_op at union{off,cmd_op}
            struct.pack_into("=Q",    sqe, 16, addr)          # addr → driver's sqe->addr
            struct.pack_into("=Q",    sqe, 32, i)             # user_data = slot index
            self._sqes[idx * _SQE_SIZE : (idx + 1) * _SQE_SIZE] = bytes(sqe)
            struct.pack_into("=I", self._sq_ring,
                             self._sq_array_off + idx * 4, idx)
            tail += 1

        struct.pack_into("=I", self._sq_ring, self._sq_tail_off, tail)
        self._sq_tail = tail

        ret = int(_libc.syscall(
            ctypes.c_long(_SYS_io_uring_enter),
            ctypes.c_uint(self._fd),
            ctypes.c_uint(n),
            ctypes.c_uint(n),
            ctypes.c_uint(_IORING_ENTER_GETEVENTS),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
        ))
        if ret < 0:
            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))

        cq_head = struct.unpack_from("=I", self._cq_ring, self._cq_head_off)[0]
        cq_mask = struct.unpack_from("=I", self._cq_ring, self._cq_mask_off)[0]
        results = [0] * n
        for _ in range(n):
            idx = cq_head & cq_mask
            cqe_off = self._cq_cqes_off + idx * _CQE_SIZE
            user_data, res, _ = struct.unpack_from("=QiI", self._cq_ring, cqe_off)
            results[int(user_data)] = res
            cq_head += 1
        struct.pack_into("=I", self._cq_ring, self._cq_head_off, cq_head)

        return results

    def submit(self, dev_fd: int, cmd_op: int, ioc_buf) -> int:
        """Submit one uring_cmd and return the kernel result."""
        return self.submit_batch(dev_fd, [(cmd_op, ioc_buf)])[0]
