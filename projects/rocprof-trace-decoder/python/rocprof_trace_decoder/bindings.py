from __future__ import annotations

import ctypes
import importlib
import importlib.util
import os
from pathlib import Path
from typing import Callable, Iterable, Protocol, Union

from ._version import __version__ as _PACKAGE_VERSION
from .records import (
    DecoderStatus,
    Dispatch,
    Event,
    EventPayload,
    Instruction,
    Occupancy,
    OtherSimdInstruction,
    Pc,
    PerfEvent,
    Realtime,
    RecordType,
    ShaderData,
    TraceRecords,
    Wave,
    WaveState,
)

BytesLike = Union[bytes, bytearray, memoryview]


class DecoderError(RuntimeError):
    def __init__(self, status: int, message: str | None = None):
        self.status = int(status)
        super().__init__(message or f"rocprof-trace-decoder returned status {status}")


class IsaProvider(Protocol):
    def isa_for_pc(self, pc: Pc) -> tuple[str, int] | None:
        """Return (instruction text, instruction byte size) for *pc*."""


class _PcInfo(ctypes.Structure):
    _fields_ = [
        ("address", ctypes.c_uint64),
        ("code_object_id", ctypes.c_uint64),
    ]


class _PerfEvent(ctypes.Structure):
    _fields_ = [
        ("time", ctypes.c_int64),
        ("events0", ctypes.c_uint16),
        ("events1", ctypes.c_uint16),
        ("events2", ctypes.c_uint16),
        ("events3", ctypes.c_uint16),
        ("CU", ctypes.c_uint8),
        ("bank", ctypes.c_uint8),
    ]


class _Occupancy(ctypes.Structure):
    _fields_ = [
        ("pc", _PcInfo),
        ("time", ctypes.c_uint64),
        ("reserved", ctypes.c_uint8),
        ("cu", ctypes.c_uint8),
        ("simd", ctypes.c_uint8),
        ("wave_id", ctypes.c_uint8),
        ("_bits", ctypes.c_uint32),
    ]


class _WaveState(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int32), ("duration", ctypes.c_int32)]


class _Instruction(ctypes.Structure):
    _fields_ = [
        ("_cat_stall", ctypes.c_uint32),
        ("duration", ctypes.c_int32),
        ("time", ctypes.c_int64),
        ("pc", _PcInfo),
    ]


class _Wave(ctypes.Structure):
    _fields_ = [
        ("cu", ctypes.c_uint8),
        ("simd", ctypes.c_uint8),
        ("wave_id", ctypes.c_uint8),
        ("contexts", ctypes.c_uint8),
        ("dispatcher", ctypes.c_uint8),
        ("workgroup_id", ctypes.c_uint8),
        ("cluster_id", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
        ("size", ctypes.c_uint64),
        ("begin_time", ctypes.c_int64),
        ("end_time", ctypes.c_int64),
        ("timeline_size", ctypes.c_uint64),
        ("instructions_size", ctypes.c_uint64),
        ("timeline_array", ctypes.POINTER(_WaveState)),
        ("instructions_array", ctypes.POINTER(_Instruction)),
    ]


class _Realtime(ctypes.Structure):
    _fields_ = [
        ("shader_clock", ctypes.c_int64),
        ("realtime_clock", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64),
    ]


class _ShaderData(ctypes.Structure):
    _fields_ = [
        ("time", ctypes.c_int64),
        ("value", ctypes.c_uint64),
        ("cu", ctypes.c_uint8),
        ("simd", ctypes.c_uint8),
        ("wave_id", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("reserved", ctypes.c_uint32),
    ]


class _OtherSimdInstruction(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("time", ctypes.c_int64),
        ("cycles", ctypes.c_uint16),
        ("wgp", ctypes.c_uint8),
        ("category", ctypes.c_uint8),
    ]


class _ClusterBarrier(ctypes.Structure):
    _fields_ = [("cluster_id", ctypes.c_int32), ("barrier_id", ctypes.c_int32)]


class _EventPayload(ctypes.Union):
    _fields_ = [
        ("raw", ctypes.c_uint64),
        ("code_object_id", ctypes.c_uint64),
        ("cluster_barrier", _ClusterBarrier),
    ]


class _Event(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("time", ctypes.c_int64),
        ("type", ctypes.c_int),
        ("me_id", ctypes.c_uint8),
        ("pipe_id", ctypes.c_uint8),
        ("flags", ctypes.c_uint16),
        ("payload", _EventPayload),
        ("byte_offset", ctypes.c_uint64),
    ]


class _Dispatch(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("time", ctypes.c_int64),
        ("me_id", ctypes.c_uint8),
        ("pipe_id", ctypes.c_uint8),
        ("user_sgprs", ctypes.c_uint16),
        ("flags", ctypes.c_int),
        ("vgprs", ctypes.c_uint32),
        ("sgprs", ctypes.c_uint32),
        ("lds_size", ctypes.c_uint32),
        ("thread_dim_x", ctypes.c_uint32),
        ("thread_dim_y", ctypes.c_uint32),
        ("thread_dim_z", ctypes.c_uint32),
        ("dispatch_pkt_addr", ctypes.c_uint64),
        ("byte_offset", ctypes.c_uint64),
        ("entry_point", _PcInfo),
    ]


class _Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


_TRACE_CB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_void_p,
)

_ISA_CB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    _PcInfo,
    ctypes.c_void_p,
)

_SE_DATA_CB = ctypes.CFUNCTYPE(
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_void_p,
)


def _pc(c_pc: _PcInfo) -> Pc:
    return Pc(address=int(c_pc.address), code_object_id=int(c_pc.code_object_id))


def _find_library(explicit: str | os.PathLike[str] | None = None) -> str:
    if explicit:
        return str(explicit)

    env = os.environ.get("ROCPROF_TRACE_DECODER_LIB")
    if env:
        return env

    names = _library_names()
    for root in _library_roots():
        for name in names:
            candidate = root / name
            if candidate.exists():
                return str(candidate)
        for candidate in sorted(root.glob("librocprof-trace-decoder.so.*"), reverse=True):
            if candidate.is_file():
                return str(candidate)

    # Let the platform loader try its normal search path.
    return "librocprof-trace-decoder.so"


def _library_names() -> list[str]:
    names = ["librocprof-trace-decoder.so", "librocprof-trace-decoder.dylib"]
    version_parts = _PACKAGE_VERSION.split(".")
    if len(version_parts) >= 2:
        names.append(f"librocprof-trace-decoder.so.{version_parts[0]}.{version_parts[1]}")
    return names


def _library_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if path not in seen:
            roots.append(path)
            seen.add(path)

    for raw in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if raw:
            add(Path(raw).expanduser())

    for var in ("ROCM_HOME", "ROCM_PATH"):
        value = os.environ.get(var)
        if value:
            add(Path(value).expanduser() / "lib")

    for root in _therock_package_roots():
        add(root / "lib")

    add(Path("/opt/rocm/lib"))
    return roots


def _therock_package_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    def add_package_root(package_name: str) -> None:
        spec = importlib.util.find_spec(package_name)
        if spec is None or spec.origin is None:
            return
        root = Path(spec.origin).resolve().parent
        if root not in seen:
            roots.append(root)
            seen.add(root)

    for dist_info_name in (
        "rocm_sdk_core._dist_info",
        "rocm_profiler._dist_info",
        "rocm_sdk._dist_info",
    ):
        try:
            dist_info = importlib.import_module(dist_info_name)
        except Exception:
            continue
        for logical_name in ("core", "devel", "profiler"):
            entry = getattr(dist_info, "ALL_PACKAGES", {}).get(logical_name)
            if entry is None or getattr(entry, "is_target_specific", False):
                continue
            try:
                add_package_root(entry.get_py_package_name())
            except Exception:
                continue

    for package_name in ("_rocm_sdk_core", "_rocm_sdk_devel", "_rocm_profiler"):
        add_package_root(package_name)

    return roots


class Decoder:
    def __init__(self, lib_path: str | os.PathLike[str] | None = None):
        self.lib_path = _find_library(lib_path)
        self._lib = ctypes.CDLL(self.lib_path)
        self._configure_library()
        self._handle = _Handle()
        self._closed = False
        self._isa_cb: _ISA_CB | None = None
        self._trace_cb: _TRACE_CB | None = None

        status = self._lib.rocprof_trace_decoder_create_handle(ctypes.byref(self._handle))
        self._check(status)

    def __enter__(self) -> "Decoder":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        status = self._lib.rocprof_trace_decoder_destroy_handle(self._handle)
        self._check(status)
        self._closed = True
        self._isa_cb = None
        self._trace_cb = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise DecoderError(DecoderStatus.ERROR_INVALID_ARGUMENT, "Decoder handle is closed")

    def _configure_library(self) -> None:
        lib = self._lib
        lib.rocprof_trace_decoder_create_handle.argtypes = [ctypes.POINTER(_Handle)]
        lib.rocprof_trace_decoder_create_handle.restype = ctypes.c_int

        lib.rocprof_trace_decoder_destroy_handle.argtypes = [_Handle]
        lib.rocprof_trace_decoder_destroy_handle.restype = ctypes.c_int

        lib.rocprof_trace_decoder_codeobj_load.argtypes = [
            _Handle,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        lib.rocprof_trace_decoder_codeobj_load.restype = ctypes.c_int

        lib.rocprof_trace_decoder_codeobj_unload.argtypes = [_Handle, ctypes.c_uint64]
        lib.rocprof_trace_decoder_codeobj_unload.restype = ctypes.c_int

        lib.rocprof_trace_decoder_set_isa_callback.argtypes = [_Handle, _ISA_CB, ctypes.c_void_p]
        lib.rocprof_trace_decoder_set_isa_callback.restype = ctypes.c_int

        lib.rocprof_trace_decoder_set_se_data_callback.argtypes = [
            _Handle,
            _SE_DATA_CB,
            ctypes.c_void_p,
        ]
        lib.rocprof_trace_decoder_set_se_data_callback.restype = ctypes.c_int

        lib.rocprof_trace_decoder_parse.argtypes = [
            _Handle,
            ctypes.c_void_p,
            ctypes.c_uint64,
            _TRACE_CB,
            ctypes.c_void_p,
        ]
        lib.rocprof_trace_decoder_parse.restype = ctypes.c_int

        lib.rocprof_trace_decoder_get_info_string.argtypes = [ctypes.c_int]
        lib.rocprof_trace_decoder_get_info_string.restype = ctypes.c_char_p

        lib.rocprof_trace_decoder_get_status_string.argtypes = [ctypes.c_int]
        lib.rocprof_trace_decoder_get_status_string.restype = ctypes.c_char_p

    def _check(self, status: int) -> None:
        if int(status) == int(DecoderStatus.SUCCESS):
            return
        message = None
        try:
            raw = self._lib.rocprof_trace_decoder_get_status_string(int(status))
            if raw:
                message = raw.decode("utf-8", errors="replace")
        except Exception:
            pass
        raise DecoderError(status, message)

    def info_string(self, info: int) -> str:
        raw = self._lib.rocprof_trace_decoder_get_info_string(int(info))
        return raw.decode("utf-8", errors="replace") if raw else ""

    def status_string(self, status: int) -> str:
        raw = self._lib.rocprof_trace_decoder_get_status_string(int(status))
        return raw.decode("utf-8", errors="replace") if raw else ""

    def load_code_object_data(
        self,
        load_id: int,
        load_addr: int,
        load_size: int,
        data: BytesLike,
    ) -> None:
        """Compatibility wrapper for the original data-loading argument order.

        *load_size* is the loaded memory range size, not the code-object byte
        length.
        """
        self._load_code_object_bytes(
            data,
            load_id=load_id,
            load_addr=load_addr,
            load_size=load_size,
        )

    def load_code_object(
        self,
        data: BytesLike | str | os.PathLike[str],
        load_id: int,
        load_addr: int,
        load_size: int,
    ) -> None:
        """Load a code object into this decoder handle.

        *data* may be a bytes-like object or a filesystem path. New code that
        wants to be explicit about paths can use `load_code_object_file()`.
        *load_size* is the size of the loaded memory range and is intentionally
        separate from the code-object byte length passed to the native API.
        """
        if _is_bytes_like(data):
            self._load_code_object_bytes(
                data,
                load_id=load_id,
                load_addr=load_addr,
                load_size=load_size,
            )
        else:
            self.load_code_object_file(
                data,
                load_id=load_id,
                load_addr=load_addr,
                load_size=load_size,
            )

    def load_code_object_file(
        self,
        path: str | os.PathLike[str],
        load_id: int,
        load_addr: int,
        load_size: int,
    ) -> None:
        """Load a code object from a file path.

        *load_size* is the loaded memory range size, not the file length.
        """
        self._load_code_object_bytes(
            Path(path).read_bytes(),
            load_id=load_id,
            load_addr=load_addr,
            load_size=load_size,
        )

    def _load_code_object_bytes(
        self,
        data: BytesLike,
        *,
        load_id: int,
        load_addr: int,
        load_size: int,
    ) -> None:
        self._ensure_open()
        blob = bytes(data)
        buf = ctypes.create_string_buffer(blob)
        status = self._lib.rocprof_trace_decoder_codeobj_load(
            self._handle, load_id, load_addr, load_size, buf, len(blob)
        )
        self._check(status)

    def unload_code_object(self, load_id: int) -> None:
        self._ensure_open()
        status = self._lib.rocprof_trace_decoder_codeobj_unload(self._handle, load_id)
        self._check(status)

    def parse_file(
        self,
        path: str | os.PathLike[str],
        isa: IsaProvider | None = None,
        on_batch: Callable[[RecordType, list[object] | int], None] | None = None,
    ) -> TraceRecords:
        return self.parse(Path(path).read_bytes(), isa=isa, on_batch=on_batch)

    def parse_chunks(
        self,
        chunks: Iterable[BytesLike],
        isa: IsaProvider | None = None,
        on_batch: Callable[[RecordType, list[object] | int], None] | None = None,
    ) -> TraceRecords:
        out = TraceRecords()
        for chunk in chunks:
            partial = self.parse(chunk, isa=isa, on_batch=on_batch)
            merge_records(out, partial)
        return out

    def parse(
        self,
        data: BytesLike,
        isa: IsaProvider | None = None,
        on_batch: Callable[[RecordType, list[object] | int], None] | None = None,
    ) -> TraceRecords:
        """Decode a binary ATT payload into Python records.

        *isa* is called by the native decoder when it needs instruction text
        for a PC. *on_batch*, when provided, receives each converted native
        record batch before this method returns the accumulated TraceRecords.
        """
        self._ensure_open()

        raw = bytes(data)
        records = TraceRecords()
        callback_error: list[BaseException] = []

        if isa is not None:
            self._isa_cb = self._make_isa_callback(isa)
        else:
            self._isa_cb = _ISA_CB()
        self._check(
            self._lib.rocprof_trace_decoder_set_isa_callback(self._handle, self._isa_cb, None)
        )

        def _trace(record_type: int, events: int | None, size: int, _userdata: int | None) -> int:
            try:
                batch_type = RecordType(record_type)
                batch = self._convert_batch(batch_type, events, int(size))
                _append_batch(records, batch_type, batch)
                if on_batch is not None:
                    on_batch(batch_type, batch)
                return int(DecoderStatus.SUCCESS)
            except BaseException as exc:
                callback_error.append(exc)
                return int(DecoderStatus.ERROR)

        trace_cb = _TRACE_CB(_trace)
        self._trace_cb = trace_cb

        if raw:
            data_buf = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
            data_ptr = ctypes.cast(data_buf, ctypes.c_void_p)
            data_size = len(raw)
        else:
            data_buf = None
            data_ptr = None
            data_size = 0

        status = self._lib.rocprof_trace_decoder_parse(
            self._handle, data_ptr, data_size, trace_cb, None
        )
        if callback_error:
            raise callback_error[0]
        self._check(status)
        return records

    def _make_isa_callback(self, isa: IsaProvider) -> _ISA_CB:
        def _isa(
            instr_buf: int,
            mem_size_p: ctypes.POINTER(ctypes.c_uint64),
            size_p: ctypes.POINTER(ctypes.c_uint64),
            c_pc: _PcInfo,
            _userdata: int | None,
        ) -> int:
            pc = _pc(c_pc)
            resolved = isa.isa_for_pc(pc)
            if resolved is None:
                return int(DecoderStatus.ERROR_INVALID_ARGUMENT)

            text, memory_size = resolved
            encoded = text.encode("utf-8")
            available = int(size_p[0])
            size_p[0] = len(encoded)
            mem_size_p[0] = max(int(memory_size), 4)
            if len(encoded) > available:
                return int(DecoderStatus.ERROR_OUT_OF_RESOURCES)
            if encoded:
                ctypes.memmove(instr_buf, encoded, len(encoded))
            return int(DecoderStatus.SUCCESS)

        return _ISA_CB(_isa)

    def _convert_batch(
        self, record_type: RecordType, events: int | None, size: int
    ) -> list[object] | int:
        if record_type == RecordType.GFXIP:
            return int(events or 0)
        if size <= 0:
            return []
        if not events:
            return []

        if record_type == RecordType.INFO:
            ptr = ctypes.cast(events, ctypes.POINTER(ctypes.c_int))
            return [int(ptr[i]) for i in range(size)]
        if record_type == RecordType.OCCUPANCY:
            ptr = ctypes.cast(events, ctypes.POINTER(_Occupancy))
            return [_convert_occupancy(ptr[i]) for i in range(size)]
        if record_type == RecordType.PERFEVENT:
            ptr = ctypes.cast(events, ctypes.POINTER(_PerfEvent))
            return [_convert_perf(ptr[i]) for i in range(size)]
        if record_type == RecordType.WAVE:
            ptr = ctypes.cast(events, ctypes.POINTER(_Wave))
            return [_convert_wave(ptr[i]) for i in range(size)]
        if record_type == RecordType.EVENT:
            ptr = ctypes.cast(events, ctypes.POINTER(_Event))
            return [_convert_event(ptr[i]) for i in range(size)]
        if record_type == RecordType.SHADERDATA:
            ptr = ctypes.cast(events, ctypes.POINTER(_ShaderData))
            return [_convert_shaderdata(ptr[i]) for i in range(size)]
        if record_type == RecordType.REALTIME:
            ptr = ctypes.cast(events, ctypes.POINTER(_Realtime))
            return [_convert_realtime(ptr[i]) for i in range(size)]
        if record_type == RecordType.RT_FREQUENCY:
            ptr = ctypes.cast(events, ctypes.POINTER(ctypes.c_uint64))
            return int(ptr[0])
        if record_type == RecordType.INST_OTHER_SIMD:
            ptr = ctypes.cast(events, ctypes.POINTER(_OtherSimdInstruction))
            return [_convert_other_simd(ptr[i]) for i in range(size)]
        if record_type == RecordType.DISPATCH:
            ptr = ctypes.cast(events, ctypes.POINTER(_Dispatch))
            return [_convert_dispatch(ptr[i]) for i in range(size)]
        return []


def _convert_perf(c: _PerfEvent) -> PerfEvent:
    return PerfEvent(
        time=int(c.time),
        events0=int(c.events0),
        events1=int(c.events1),
        events2=int(c.events2),
        events3=int(c.events3),
        cu=int(c.CU),
        bank=int(c.bank),
    )


def _convert_occupancy(c: _Occupancy) -> Occupancy:
    bits = int(c._bits)
    return Occupancy(
        pc=_pc(c.pc),
        time=int(c.time),
        reserved=int(c.reserved),
        cu=int(c.cu),
        simd=int(c.simd),
        wave_id=int(c.wave_id),
        start=bits & 0x1,
        me_id=(bits >> 1) & 0x7,
        pipe_id=(bits >> 4) & 0xF,
        is_ext=(bits >> 8) & 0x1,
        workgroup_id=(bits >> 9) & 0x7F,
        cluster_id=(bits >> 16) & 0x1F,
    )


def _convert_instruction(c: _Instruction) -> Instruction:
    raw = int(c._cat_stall)
    return Instruction(
        category=raw & 0xFF,
        stall=(raw >> 8) & 0xFFFFFF,
        duration=int(c.duration),
        time=int(c.time),
        pc=_pc(c.pc),
    )


def _convert_wave(c: _Wave) -> Wave:
    timeline = []
    if c.timeline_array:
        timeline = [
            WaveState(
                type=int(c.timeline_array[i].type),
                duration=int(c.timeline_array[i].duration),
            )
            for i in range(int(c.timeline_size))
        ]

    instructions = []
    if c.instructions_array:
        instructions = [
            _convert_instruction(c.instructions_array[i]) for i in range(int(c.instructions_size))
        ]

    return Wave(
        cu=int(c.cu),
        simd=int(c.simd),
        wave_id=int(c.wave_id),
        contexts=int(c.contexts),
        dispatcher=int(c.dispatcher),
        workgroup_id=int(c.workgroup_id),
        cluster_id=int(c.cluster_id),
        reserved=int(c.reserved),
        size=int(c.size),
        begin_time=int(c.begin_time),
        end_time=int(c.end_time),
        timeline=timeline,
        instructions=instructions,
    )


def _convert_realtime(c: _Realtime) -> Realtime:
    return Realtime(
        shader_clock=int(c.shader_clock),
        realtime_clock=int(c.realtime_clock),
        reserved=int(c.reserved),
    )


def _convert_shaderdata(c: _ShaderData) -> ShaderData:
    return ShaderData(
        time=int(c.time),
        value=int(c.value),
        cu=int(c.cu),
        simd=int(c.simd),
        wave_id=int(c.wave_id),
        flags=int(c.flags),
        reserved=int(c.reserved),
    )


def _convert_other_simd(c: _OtherSimdInstruction) -> OtherSimdInstruction:
    return OtherSimdInstruction(
        size=int(c.size),
        time=int(c.time),
        cycles=int(c.cycles),
        wgp=int(c.wgp),
        category=int(c.category),
    )


def _convert_event(c: _Event) -> Event:
    payload = EventPayload(
        raw=int(c.payload.raw),
        code_object_id=int(c.payload.code_object_id),
        cluster_id=int(c.payload.cluster_barrier.cluster_id),
        barrier_id=int(c.payload.cluster_barrier.barrier_id),
    )
    return Event(
        size=int(c.size),
        time=int(c.time),
        type=int(c.type),
        me_id=int(c.me_id),
        pipe_id=int(c.pipe_id),
        flags=int(c.flags),
        payload=payload,
        byte_offset=int(c.byte_offset),
    )


def _convert_dispatch(c: _Dispatch) -> Dispatch:
    return Dispatch(
        size=int(c.size),
        time=int(c.time),
        me_id=int(c.me_id),
        pipe_id=int(c.pipe_id),
        user_sgprs=int(c.user_sgprs),
        flags=int(c.flags),
        vgprs=int(c.vgprs),
        sgprs=int(c.sgprs),
        lds_size=int(c.lds_size),
        thread_dim_x=int(c.thread_dim_x),
        thread_dim_y=int(c.thread_dim_y),
        thread_dim_z=int(c.thread_dim_z),
        dispatch_pkt_addr=int(c.dispatch_pkt_addr),
        byte_offset=int(c.byte_offset),
        entry_point=_pc(c.entry_point),
    )


def _append_batch(
    records: TraceRecords,
    record_type: RecordType,
    batch: list[object] | int,
) -> None:
    records.batches.append((record_type, batch))
    if record_type == RecordType.GFXIP:
        records.gfxip = int(batch)
    elif record_type == RecordType.INFO:
        records.info.extend(batch)  # type: ignore[arg-type]
    elif record_type == RecordType.OCCUPANCY:
        records.occupancy.extend(batch)  # type: ignore[arg-type]
    elif record_type == RecordType.PERFEVENT:
        records.perf_events.extend(batch)  # type: ignore[arg-type]
    elif record_type == RecordType.WAVE:
        records.waves.extend(batch)  # type: ignore[arg-type]
    elif record_type == RecordType.EVENT:
        records.events.extend(batch)  # type: ignore[arg-type]
    elif record_type == RecordType.SHADERDATA:
        records.shaderdata.extend(batch)  # type: ignore[arg-type]
    elif record_type == RecordType.REALTIME:
        records.realtime.extend(batch)  # type: ignore[arg-type]
    elif record_type == RecordType.RT_FREQUENCY:
        records.realtime_frequency = int(batch)
    elif record_type == RecordType.INST_OTHER_SIMD:
        records.other_simd.extend(batch)  # type: ignore[arg-type]
    elif record_type == RecordType.DISPATCH:
        records.dispatches.extend(batch)  # type: ignore[arg-type]


def merge_records(dst: TraceRecords, src: TraceRecords) -> TraceRecords:
    if src.gfxip is not None:
        dst.gfxip = src.gfxip
    if src.realtime_frequency is not None:
        dst.realtime_frequency = src.realtime_frequency
    dst.info.extend(src.info)
    dst.occupancy.extend(src.occupancy)
    dst.perf_events.extend(src.perf_events)
    dst.waves.extend(src.waves)
    dst.events.extend(src.events)
    dst.shaderdata.extend(src.shaderdata)
    dst.realtime.extend(src.realtime)
    dst.other_simd.extend(src.other_simd)
    dst.dispatches.extend(src.dispatches)
    dst.batches.extend(src.batches)
    return dst


def _is_bytes_like(value: object) -> bool:
    return isinstance(value, (bytes, bytearray, memoryview))


__all__ = ["BytesLike", "Decoder", "DecoderError", "IsaProvider", "merge_records"]
