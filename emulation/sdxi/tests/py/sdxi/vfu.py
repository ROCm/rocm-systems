"""A vfio-user client, and the guest memory it shares with the device.

This talks the vfio-user protocol to a real rocm-sdxi process over its
Unix socket, which is what a VMM such as QEMU does. Nothing here loads
libvfio-user: the protocol is a 16-byte header and a payload, with file
descriptors passed as ancillary data, so a client needs only the standard
library.

Guest memory is a memfd. The client mmaps it and passes the descriptor to
the server in a DMA_MAP, so the server maps the same pages and a guest
physical address is an offset from the region's base. That makes the real
DMA path -- vfu_addr_to_sgl() and vfu_sgl_read/write() in the device code --
run exactly as it does for a VM.

Adapted from rocm-ernic/tests/py/ernic/vfu.py for the SDXI emulator.

Copyright (C) Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
"""

import array
import errno
import json
import mmap
import os
import socket
import struct
import time

# enum vfio_user_command
VFIO_USER_VERSION = 1
VFIO_USER_DMA_MAP = 2
VFIO_USER_DMA_UNMAP = 3
VFIO_USER_DEVICE_GET_INFO = 4
VFIO_USER_DEVICE_GET_REGION_INFO = 5
VFIO_USER_DEVICE_GET_REGION_IO_FDS = 6
VFIO_USER_DEVICE_GET_IRQ_INFO = 7
VFIO_USER_DEVICE_SET_IRQS = 8
VFIO_USER_REGION_READ = 9
VFIO_USER_REGION_WRITE = 10
VFIO_USER_DMA_READ = 11
VFIO_USER_DMA_WRITE = 12
VFIO_USER_DEVICE_RESET = 13
VFIO_USER_REGION_WRITE_MULTI = 15
VFIO_USER_DEVICE_FEATURE = 16
VFIO_USER_MIG_DATA_READ = 17
VFIO_USER_MIG_DATA_WRITE = 18

VFIO_USER_F_TYPE_MASK = 0xF
VFIO_USER_F_TYPE_COMMAND = 0
VFIO_USER_F_TYPE_REPLY = 1
VFIO_USER_F_NO_REPLY = 0x10
VFIO_USER_F_ERROR = 0x20

HEADER_SIZE = 16
HEADER_FMT = "<HHIII"

LIBVFIO_USER_MAJOR = 0
LIBVFIO_USER_MINOR = 1

DEFAULT_MAX_DATA_XFER_SIZE = 1 << 20
MAX_MSG_SIZE = DEFAULT_MAX_DATA_XFER_SIZE + 32

# Region indices of a PCI device (enum vfu_pci_dev_region_idx)
VFU_PCI_DEV_BAR0_REGION_IDX = 0
VFU_PCI_DEV_BAR1_REGION_IDX = 1
VFU_PCI_DEV_BAR2_REGION_IDX = 2
VFU_PCI_DEV_CFG_REGION_IDX = 7
VFU_PCI_DEV_NUM_REGIONS = 9

# struct vfio_user_dma_map flags
VFIO_USER_F_DMA_REGION_READ = 1 << 0
VFIO_USER_F_DMA_REGION_WRITE = 1 << 1
VFIO_USER_F_DMA_REGION_MMAP = 1 << 2

# from linux/vfio.h
VFIO_REGION_INFO_FLAG_READ = 1 << 0
VFIO_REGION_INFO_FLAG_WRITE = 1 << 1
VFIO_REGION_INFO_FLAG_MMAP = 1 << 2
VFIO_REGION_INFO_FLAG_CAPS = 1 << 3

VFIO_IRQ_SET_DATA_NONE = 1 << 0
VFIO_IRQ_SET_DATA_BOOL = 1 << 1
VFIO_IRQ_SET_DATA_EVENTFD = 1 << 2
VFIO_IRQ_SET_ACTION_MASK = 1 << 3
VFIO_IRQ_SET_ACTION_UNMASK = 1 << 4
VFIO_IRQ_SET_ACTION_TRIGGER = 1 << 5

VFIO_DMA_UNMAP_FLAG_ALL = 1 << 1

# enum vfu_dev_irq_type
VFU_DEV_INTX_IRQ = 0
VFU_DEV_MSI_IRQ = 1
VFU_DEV_MSIX_IRQ = 2
VFU_DEV_ERR_IRQ = 3
VFU_DEV_REQ_IRQ = 4

PAGE_SIZE = 4096


class VfioUserError(Exception):
    """The server replied to a command with an error."""

    def __init__(self, cmd, err):
        self.cmd = cmd
        self.err = err
        name = errno.errorcode.get(err, err)
        super().__init__(f"vfio-user command {cmd} failed: {name}")


class Region:
    """One region of the device, as DEVICE_GET_REGION_INFO described it."""

    def __init__(self, index, flags, size, offset):
        self.index = index
        self.flags = flags
        self.size = size
        self.offset = offset

    def __repr__(self):
        return (
            f"Region(index={self.index}, size={self.size:#x}, "
            f"flags={self.flags:#x})"
        )


class GuestMemory:
    """Memory shared with the device, the way a VM's RAM is.

    Backed by a memfd so the server can map the same pages. Addresses
    handed to the device are guest physical addresses: ``base`` plus an
    offset into the memfd.
    """

    def __init__(self, size=64 << 20, base=0x1000_0000, name="sdxi-guest-mem"):
        if size % PAGE_SIZE:
            raise ValueError("guest memory size must be a multiple of the page size")
        self.size = size
        self.base = base
        self.fd = os.memfd_create(name, 0)
        os.ftruncate(self.fd, size)
        self.map = mmap.mmap(
            self.fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
        )
        self._next = 0

    def close(self):
        if self.map is not None:
            self.map.close()
            self.map = None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def alloc(self, size, align=PAGE_SIZE):
        """Reserve @size bytes and return their guest physical address."""
        start = (self._next + align - 1) & ~(align - 1)
        if start + size > self.size:
            raise MemoryError(
                f"guest memory exhausted: {self.size - start} bytes left, "
                f"{size} wanted"
            )
        self._next = start + size
        self.map[start : start + size] = b"\0" * size
        return self.base + start

    def _offset(self, gpa, length):
        off = gpa - self.base
        if off < 0 or length < 0 or off + length > self.size:
            raise ValueError(f"gpa {gpa:#x}+{length} is outside guest memory")
        return off

    def write(self, gpa, data):
        off = self._offset(gpa, len(data))
        self.map[off : off + len(data)] = data

    def read(self, gpa, length):
        off = self._offset(gpa, length)
        return bytes(self.map[off : off + length])

    def write_u32(self, gpa, value):
        self.write(gpa, struct.pack("<I", value))

    def read_u32(self, gpa):
        return struct.unpack("<I", self.read(gpa, 4))[0]

    def write_u64(self, gpa, value):
        self.write(gpa, struct.pack("<Q", value))

    def read_u64(self, gpa):
        return struct.unpack("<Q", self.read(gpa, 8))[0]


def wait_for_socket(path, timeout=10.0, proc=None):
    """Wait for @path to exist."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return True
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(0.02)
    return False


class VfioUserClient:
    """A vfio-user client attached to one server's socket."""

    def __init__(self, sock_path, max_data_xfer_size=DEFAULT_MAX_DATA_XFER_SIZE):
        self.sock_path = sock_path
        self.max_data_xfer_size = max_data_xfer_size
        self.sock = None
        self.next_msg_id = 1
        self.server_caps = {}
        self.regions = {}
        self.num_irqs = 0
        self._dma_maps = []

    def connect(self, timeout=10.0, capabilities=None):
        """Connect and negotiate the version, then learn the device's shape."""
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.sock.connect(self.sock_path)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise
                if exc.errno not in (
                    errno.EAGAIN,
                    errno.ECONNREFUSED,
                    errno.ENOENT,
                ):
                    raise
                time.sleep(0.02)
        self.negotiate(capabilities)
        self.device_info()
        for index in range(VFU_PCI_DEV_NUM_REGIONS):
            region = self.region_info(index)
            if region is not None:
                self.regions[index] = region
        return self

    def negotiate(self, capabilities=None):
        """Exchange VFIO_USER_VERSION and record what the server offers."""
        caps = {
            "capabilities": {
                "max_data_xfer_size": self.max_data_xfer_size,
                "max_msg_fds": 8,
            }
        }
        if capabilities:
            caps["capabilities"].update(capabilities)
        caps_json = json.dumps(caps).encode()
        payload = struct.pack("<HH", LIBVFIO_USER_MAJOR, LIBVFIO_USER_MINOR)
        payload += caps_json + b"\0"
        reply = self.request(VFIO_USER_VERSION, payload)
        major, minor = struct.unpack_from("<HH", reply)
        body = reply[4:].rstrip(b"\0")
        self.server_caps = json.loads(body.decode()) if body else {}
        return major, minor

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def send_command(self, cmd, payload=b"", fds=None, no_reply=False):
        msg_id = self.next_msg_id
        self.next_msg_id = (self.next_msg_id + 1) & 0xFFFF
        flags = VFIO_USER_F_TYPE_COMMAND
        if no_reply:
            flags |= VFIO_USER_F_NO_REPLY
        hdr = struct.pack(HEADER_FMT, msg_id, cmd, HEADER_SIZE + len(payload), flags, 0)
        if fds:
            packed = array.array("i", fds)
            self.sock.sendmsg(
                [hdr + payload],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, packed.tobytes())],
            )
        else:
            self.sock.sendall(hdr + payload)
        return msg_id

    def _recvall(self, length):
        buf = b""
        while len(buf) < length:
            chunk = self.sock.recv(length - len(buf))
            if not chunk:
                raise ConnectionError("server closed the connection")
            buf += chunk
        return buf

    def recv_reply(self):
        fds = array.array("i")
        data, ancillary, _flags, _addr = self.sock.recvmsg(
            HEADER_SIZE, socket.CMSG_SPACE(8 * fds.itemsize)
        )
        if len(data) < HEADER_SIZE:
            data += self._recvall(HEADER_SIZE - len(data))
        for level, ctype, cdata in ancillary:
            if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
                fds.frombytes(cdata[: len(cdata) - (len(cdata) % fds.itemsize)])
        msg_id, cmd, msg_size, flags, error_no = struct.unpack(HEADER_FMT, data)
        if msg_size < HEADER_SIZE or msg_size > MAX_MSG_SIZE:
            raise ValueError(f"reply has an implausible size: {msg_size}")
        payload = self._recvall(msg_size - HEADER_SIZE)
        if (flags & VFIO_USER_F_TYPE_MASK) != VFIO_USER_F_TYPE_REPLY:
            raise ValueError(f"expected a reply, got flags {flags:#x}")
        return msg_id, cmd, error_no, payload, list(fds)

    def request(self, cmd, payload=b"", fds=None, expect_errno=0):
        """Round-trip one command and return its reply payload."""
        self.send_command(cmd, payload, fds)
        _msg_id, _cmd, error_no, reply, _fds = self.recv_reply()
        if error_no != expect_errno:
            raise VfioUserError(cmd, error_no)
        return reply

    def device_info(self):
        payload = struct.pack("<IIII", 16, 0, 0, 0)
        reply = self.request(VFIO_USER_DEVICE_GET_INFO, payload)
        _argsz, flags, num_regions, num_irqs = struct.unpack("<IIII", reply[:16])
        self.num_irqs = num_irqs
        return flags, num_regions, num_irqs

    def region_info(self, index):
        payload = struct.pack("<IIIIQQ", 32, 0, index, 0, 0, 0)
        try:
            reply = self.request(VFIO_USER_DEVICE_GET_REGION_INFO, payload)
        except VfioUserError as exc:
            if exc.err == errno.EINVAL:
                return None
            raise
        _argsz, flags, idx, _cap, size, offset = struct.unpack("<IIIIQQ", reply[:32])
        if size == 0:
            return None
        return Region(idx, flags, size, offset)

    def reset(self):
        self.request(VFIO_USER_DEVICE_RESET)

    def read_region(self, index, offset, count):
        payload = struct.pack("<QII", offset, index, count)
        reply = self.request(VFIO_USER_REGION_READ, payload)
        return reply[16 : 16 + count]

    def write_region(self, index, offset, data):
        payload = struct.pack("<QII", offset, index, len(data)) + bytes(data)
        self.request(VFIO_USER_REGION_WRITE, payload)

    def read_config(self, offset, count):
        return self.read_region(VFU_PCI_DEV_CFG_REGION_IDX, offset, count)

    def write_config(self, offset, data):
        self.write_region(VFU_PCI_DEV_CFG_REGION_IDX, offset, data)

    def read_bar32(self, index, offset):
        return struct.unpack("<I", self.read_region(index, offset, 4))[0]

    def write_bar32(self, index, offset, value):
        self.write_region(index, offset, struct.pack("<I", value))

    def read_bar64(self, index, offset):
        return struct.unpack("<Q", self.read_region(index, offset, 8))[0]

    def write_bar64(self, index, offset, value):
        self.write_region(index, offset, struct.pack("<Q", value))

    def dma_map(self, mem, iova=None, size=None, offset=0, flags=None):
        if iova is None:
            iova = mem.base + offset
        if size is None:
            size = mem.size - offset
        if flags is None:
            flags = (
                VFIO_USER_F_DMA_REGION_READ
                | VFIO_USER_F_DMA_REGION_WRITE
                | VFIO_USER_F_DMA_REGION_MMAP
            )
        payload = struct.pack("<IIQQQ", 32, flags, offset, iova, size)
        self.request(VFIO_USER_DMA_MAP, payload, fds=[mem.fd])
        self._dma_maps.append((iova, size))

    def dma_unmap(self, iova, size, flags=0):
        payload = struct.pack("<IIQQ", 24, flags, iova, size)
        self.request(VFIO_USER_DMA_UNMAP, payload)
        self._dma_maps = [m for m in self._dma_maps if m != (iova, size)]

    def irq_info(self, index):
        payload = struct.pack("<IIII", 16, 0, index, 0)
        reply = self.request(VFIO_USER_DEVICE_GET_IRQ_INFO, payload)
        _argsz, flags, _index, count = struct.unpack("<IIII", reply[:16])
        return flags, count

    def set_irqs(self, index, start, count, fds):
        flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER
        argsz = 20 + 4 * len(fds)
        payload = struct.pack("<IIIII", argsz, flags, index, start, count)
        self.request(VFIO_USER_DEVICE_SET_IRQS, payload, fds=list(fds))

    def disable_irqs(self, index):
        flags = VFIO_IRQ_SET_DATA_NONE | VFIO_IRQ_SET_ACTION_TRIGGER
        payload = struct.pack("<IIIII", 20, flags, index, 0, 0)
        self.request(VFIO_USER_DEVICE_SET_IRQS, payload)
