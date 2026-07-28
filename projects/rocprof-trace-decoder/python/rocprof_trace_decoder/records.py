from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

__all__ = [
    "DecoderInfo",
    "DecoderStatus",
    "Dispatch",
    "DispatchFlags",
    "Event",
    "EventFlags",
    "EventPayload",
    "EventType",
    "InstCategory",
    "Instruction",
    "Occupancy",
    "OtherSimdInstruction",
    "Pc",
    "PerfEvent",
    "Realtime",
    "RecordType",
    "ShaderData",
    "ShaderDataFlags",
    "TraceRecords",
    "Wave",
    "WaveState",
    "WaveStateType",
]


class DecoderStatus(IntEnum):
    SUCCESS = 0
    ERROR = 1
    ERROR_OUT_OF_RESOURCES = 2
    ERROR_INVALID_ARGUMENT = 3
    ERROR_INVALID_SHADER_DATA = 4
    ERROR_NOT_IMPLEMENTED = 5
    LAST = 6


class DecoderInfo(IntEnum):
    NONE = 0
    DATA_LOST = 1
    STITCH_INCOMPLETE = 2
    WAVE_INCOMPLETE = 3
    LAST = 4


class WaveStateType(IntEnum):
    EMPTY = 0
    IDLE = 1
    EXEC = 2
    WAIT = 3
    STALL = 4
    LAST = 5


class InstCategory(IntEnum):
    NONE = 0
    SMEM = 1
    SALU = 2
    VMEM = 3
    FLAT = 4
    LDS = 5
    VALU = 6
    JUMP = 7
    NEXT = 8
    IMMED = 9
    CONTEXT = 10
    MESSAGE = 11
    BVH = 12
    LAST = 13


class EventType(IntEnum):
    NONE = 0
    CS_PARTIAL_FLUSH = 1
    BOTTOM_OF_PIPE_TS = 2
    SAVE_CONTEXT = 3
    DISPATCH_END = 4
    CACHE_FLUSH = 5
    PACKET_LOSS = 6
    CODE_OBJECT_LOAD = 7
    CODE_OBJECT_UNLOAD = 8
    TT_STALL_BEGIN = 9
    TT_STALL_END = 10
    TT_FLUSH = 11
    DIDT_STALL_BEGIN = 12
    DIDT_STALL_END = 13
    CLUSTER_BARRIER = 14
    RESERVED = 15
    GC_RINSE = 16
    SPM_SAMPLE = 17
    LAST = 18


class EventFlags(IntEnum):
    NONE = 0
    PER_PIPE = 0x1
    BOP = 0x2
    LAST = 0x2


class DispatchFlags(IntEnum):
    NONE = 0
    SCALAR_CACHE_INVALIDATE = 0x1
    VECTOR_CACHE_INVALIDATE = 0x2
    IS_CTX_RESTORE = 0x4
    SCRATCH_ENABLED = 0x8
    REALTIME_TS = 0x10
    LAST = 0x10


class ShaderDataFlags(IntEnum):
    IMM = 0
    PRIV = 1


class RecordType(IntEnum):
    GFXIP = 0
    OCCUPANCY = 1
    PERFEVENT = 2
    WAVE = 3
    INFO = 4
    EVENT = 5
    SHADERDATA = 6
    REALTIME = 7
    RT_FREQUENCY = 8
    INST_OTHER_SIMD = 9
    DISPATCH = 10
    LAST = 11


@dataclass(frozen=True, order=True)
class Pc:
    address: int
    code_object_id: int


@dataclass
class PerfEvent:
    time: int
    events0: int
    events1: int
    events2: int
    events3: int
    cu: int
    bank: int


@dataclass
class Occupancy:
    pc: Pc
    time: int
    reserved: int
    cu: int
    simd: int
    wave_id: int
    start: int
    me_id: int
    pipe_id: int
    is_ext: int
    workgroup_id: int
    cluster_id: int


@dataclass
class WaveState:
    type: int
    duration: int


@dataclass
class Instruction:
    category: int
    stall: int
    duration: int
    time: int
    pc: Pc


@dataclass
class Wave:
    cu: int
    simd: int
    wave_id: int
    contexts: int
    dispatcher: int
    workgroup_id: int
    cluster_id: int
    reserved: int
    size: int
    begin_time: int
    end_time: int
    timeline: list[WaveState]
    instructions: list[Instruction]


@dataclass
class Realtime:
    shader_clock: int
    realtime_clock: int
    reserved: int


@dataclass
class ShaderData:
    time: int
    value: int
    cu: int
    simd: int
    wave_id: int
    flags: int
    reserved: int


@dataclass
class OtherSimdInstruction:
    size: int
    time: int
    cycles: int
    wgp: int
    category: int


@dataclass
class EventPayload:
    raw: int
    code_object_id: int
    cluster_id: int
    barrier_id: int


@dataclass
class Event:
    size: int
    time: int
    type: int
    me_id: int
    pipe_id: int
    flags: int
    payload: EventPayload
    byte_offset: int


@dataclass
class Dispatch:
    size: int
    time: int
    me_id: int
    pipe_id: int
    user_sgprs: int
    flags: int
    vgprs: int
    sgprs: int
    lds_size: int
    thread_dim_x: int
    thread_dim_y: int
    thread_dim_z: int
    dispatch_pkt_addr: int
    byte_offset: int
    entry_point: Pc


@dataclass
class TraceRecords:
    gfxip: int | None = None
    info: list[int] = field(default_factory=list)
    occupancy: list[Occupancy] = field(default_factory=list)
    perf_events: list[PerfEvent] = field(default_factory=list)
    waves: list[Wave] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    shaderdata: list[ShaderData] = field(default_factory=list)
    realtime: list[Realtime] = field(default_factory=list)
    realtime_frequency: int | None = None
    other_simd: list[OtherSimdInstruction] = field(default_factory=list)
    dispatches: list[Dispatch] = field(default_factory=list)
    batches: list[tuple[RecordType, list[Any] | int]] = field(default_factory=list)
