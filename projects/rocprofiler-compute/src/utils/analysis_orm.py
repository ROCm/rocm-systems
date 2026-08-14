# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""SQLAlchemy ORM models and SQLite backend for the analysis database.

After changing a table, column, foreign key, or view, regenerate the diagrams
under docs/data/analyze/ with tools/schema_visualizer.py and commit them.
"""

import csv
import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    case,
    cast,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.dialects import sqlite
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.sql import Select
from sqlalchemy.sql.expression import Subquery

from pc_sampling.source_snapshot_analysis import (
    SOURCE_FRAME_SEPARATOR,
    UNKNOWN_SOURCE_LINE_TOKEN,
)
from utils.logger import console_debug, console_error, console_warning

PREFIX = "compute_"
SCHEMA_VERSION = "2.2.0"


Base = declarative_base()


class Workload(Base):
    __tablename__ = f"{PREFIX}workload"
    # The pair names the workload's source export folder, so it must be unique.
    __table_args__ = (UniqueConstraint("name", "sub_name"),)

    workload_id = Column(Integer, primary_key=True)
    name = Column(String)
    sub_name = Column(String)
    sys_info_extdata = Column(JSON)
    roofline_bench_extdata = Column(JSON)
    profiling_config_extdata = Column(JSON)

    # Workload can have multiple kernels
    kernels = relationship("Kernel", back_populates="workload")
    # Workload can have multiple metric definitions
    metric_definitions = relationship("MetricDefinition", back_populates="workload")
    # Workload can have multiple workload-level metric values
    workload_metric_values = relationship(
        "WorkloadMetricValue", back_populates="workload"
    )
    # Workload can have multiple workload-level roofline data points
    workload_roofline_data_points = relationship(
        "WorkloadRooflineData", back_populates="workload"
    )
    # Workload can have multiple code objects
    code_object_stores = relationship("CodeObjectStore", back_populates="workload")
    # Workload can have multiple source files
    source_files = relationship("SourceFile", back_populates="workload")


class MetricDefinition(Base):
    __tablename__ = f"{PREFIX}metric_definition"
    # One definition per metric per workload.
    __table_args__ = (UniqueConstraint("workload_id", "metric_id"),)

    metric_uuid = Column(Integer, primary_key=True)
    workload_id = Column(
        Integer, ForeignKey(f"{PREFIX}workload.workload_id"), nullable=False
    )
    name = Column(String)  # e.g. Wavefronts Num
    metric_id = Column(String)  # e.g. 4.1.3
    description = Column(Text)  # e.g. Number of wavefronts
    table_name = Column(String)  # e.g. Wavefront
    sub_table_name = Column(String)  # e.g. Wavefront stats
    unit = Column(String)  # e.g. Gbps

    # Metric can have one workload
    workload = relationship("Workload", back_populates="metric_definitions")
    # Metric can have multiple kernel-level metric values
    kernel_metric_values = relationship("KernelMetricValue", back_populates="metric")
    # Metric can have multiple workload-level metric values
    workload_metric_values = relationship(
        "WorkloadMetricValue", back_populates="metric"
    )


class KernelRooflineData(Base):
    __tablename__ = f"{PREFIX}kernel_roofline_data"

    roofline_uuid = Column(Integer, primary_key=True)
    # One roofline data point per kernel.
    kernel_uuid = Column(
        Integer, ForeignKey(f"{PREFIX}kernel.kernel_uuid"), nullable=False, unique=True
    )
    total_flops = Column(Float)
    l0_cache_data = Column(Float)
    l1_cache_data = Column(Float)
    l2_cache_data = Column(Float)
    hbm_cache_data = Column(Float)
    lds_cache_data = Column(Float)

    # Roofline data point can have one kernel
    kernel = relationship("Kernel", back_populates="roofline_data_points")


class Dispatch(Base):
    __tablename__ = f"{PREFIX}dispatch"
    # dispatch_id is unique within a kernel.
    __table_args__ = (UniqueConstraint("kernel_uuid", "dispatch_id"),)

    dispatch_uuid = Column(Integer, primary_key=True)
    kernel_uuid = Column(
        Integer, ForeignKey(f"{PREFIX}kernel.kernel_uuid"), nullable=False
    )
    dispatch_id = Column(Integer)
    gpu_id = Column(Integer)
    start_timestamp = Column(Integer)
    end_timestamp = Column(Integer)

    # Dispatch can have one kernel
    kernel = relationship("Kernel", back_populates="dispatches")


class Kernel(Base):
    __tablename__ = f"{PREFIX}kernel"
    # One kernel row per name per workload.
    __table_args__ = (UniqueConstraint("workload_id", "kernel_name"),)

    kernel_uuid = Column(Integer, primary_key=True)
    workload_id = Column(
        Integer, ForeignKey(f"{PREFIX}workload.workload_id"), nullable=False
    )
    kernel_name = Column(String)

    # Kernel can have one workload
    workload = relationship("Workload", back_populates="kernels")
    # Kernel can have multiple dispatches
    dispatches = relationship("Dispatch", back_populates="kernel")
    # Kernel can have multiple metric values
    metric_values = relationship("KernelMetricValue", back_populates="kernel")
    # Kernel can have multiple roofline data points
    roofline_data_points = relationship("KernelRooflineData", back_populates="kernel")
    # Kernel is compiled into one symbol per code object that holds it
    kernel_symbols = relationship("KernelSymbol", back_populates="kernel")


class CodeObjectStore(Base):
    __tablename__ = f"{PREFIX}code_object_store"
    # code_object_id is process-local within a workload.
    __table_args__ = (UniqueConstraint("workload_id", "pid", "code_object_id"),)

    code_object_uuid = Column(Integer, primary_key=True)
    workload_id = Column(
        Integer, ForeignKey(f"{PREFIX}workload.workload_id"), nullable=False
    )
    pid = Column(Integer)
    code_object_id = Column(Integer)
    load_base = Column(Integer, nullable=True)

    # Code object belongs to one workload
    workload = relationship("Workload", back_populates="code_object_stores")
    # One code object owns many kernel symbols
    kernel_symbols = relationship("KernelSymbol", back_populates="code_object_store")


class KernelSymbol(Base):
    __tablename__ = f"{PREFIX}kernel_symbol"
    # One symbol per kernel per code object, and one kernel per offset: an
    # offset picks a single instruction, which lies in a single symbol.
    __table_args__ = (
        UniqueConstraint("code_object_uuid", "kernel_uuid"),
        UniqueConstraint("code_object_uuid", "code_object_offset"),
    )

    kernel_symbol_uuid = Column(Integer, primary_key=True)
    code_object_uuid = Column(
        Integer,
        ForeignKey(f"{PREFIX}code_object_store.code_object_uuid"),
        nullable=False,
    )
    kernel_uuid = Column(
        Integer, ForeignKey(f"{PREFIX}kernel.kernel_uuid"), nullable=False
    )
    # Null for a code object whose ISA was skipped for want of a load_base.
    code_object_offset = Column(Integer, nullable=True)

    # Symbol belongs to one code object
    code_object_store = relationship("CodeObjectStore", back_populates="kernel_symbols")
    # Symbol is one compilation of one kernel
    kernel = relationship("Kernel", back_populates="kernel_symbols")
    # One symbol owns many instruction lines
    instruction_lines = relationship("InstructionLine", back_populates="kernel_symbol")


class InstructionTypeLookup(Base):
    __tablename__ = f"{PREFIX}instruction_type_lookup"

    instruction_type_lookup_uuid = Column(Integer, primary_key=True)
    # Deduplicated: one row per distinct static instruction-type string.
    text = Column(String, unique=True)

    instruction_lines = relationship(
        "InstructionLine", back_populates="instruction_type_lookup"
    )


class InstructionLine(Base):
    __tablename__ = f"{PREFIX}instruction_line"
    # One row per sampled or disassembled offset within a symbol.
    __table_args__ = (UniqueConstraint("kernel_symbol_uuid", "code_object_offset"),)

    instruction_uuid = Column(Integer, primary_key=True)
    kernel_symbol_uuid = Column(
        Integer,
        ForeignKey(f"{PREFIX}kernel_symbol.kernel_symbol_uuid"),
        nullable=False,
    )
    # TODO: populate from the disassembled mnemonic. This is the static
    # compiler class of the instruction at this offset (Matrix / Vector /
    # Scalar / Branch), one value per line. Distinct from InstructionSample,
    # which counts the issue state seen at each sample.
    instruction_type_uuid = Column(
        Integer,
        ForeignKey(f"{PREFIX}instruction_type_lookup.instruction_type_lookup_uuid"),
        nullable=True,
    )
    code_object_offset = Column(Integer)
    instruction = Column(Text)

    # Instruction line belongs to one kernel symbol
    kernel_symbol = relationship("KernelSymbol", back_populates="instruction_lines")
    # Instruction line has at most one static instruction type
    instruction_type_lookup = relationship(
        "InstructionTypeLookup", back_populates="instruction_lines"
    )
    # An instruction line has at most one sampled state
    pc_sample_state = relationship(
        "PCSampleState", back_populates="instruction_line", uselist=False
    )
    # An instruction line has an inline stack of source lines, innermost first
    source_lines = relationship(
        "InstructionSourceLine",
        back_populates="instruction_line",
        order_by="InstructionSourceLine.frame_index",
    )


class SourceFile(Base):
    """One source file a workload's instructions came from.

    file_path is the absolute path on the capture host, which is also where the
    file sits under the workload's source snapshot.
    """

    __tablename__ = f"{PREFIX}source_file"
    # Two workloads can each hold a different a.cpp.
    __table_args__ = (UniqueConstraint("workload_id", "file_path"),)

    source_file_uuid = Column(Integer, primary_key=True)
    workload_id = Column(
        Integer, ForeignKey(f"{PREFIX}workload.workload_id"), nullable=False
    )
    file_path = Column(Text)
    # Null when the file is absent from the workload's source snapshot.
    md5_checksum = Column(String, nullable=True)

    # Source file belongs to one workload
    workload = relationship("Workload", back_populates="source_files")
    # Source file owns many source lines
    source_lines = relationship("SourceLine", back_populates="source_file")


class SourceLine(Base):
    """One line of a source file, whether or not an instruction points at it.

    A null line_number is a line the compiler could not attribute. SQLite
    treats nulls in a unique constraint as distinct, so analysis_db keeps that
    row unique, not the database.
    """

    __tablename__ = f"{PREFIX}source_line"
    __table_args__ = (UniqueConstraint("source_file_uuid", "line_number"),)

    source_line_uuid = Column(Integer, primary_key=True)
    source_file_uuid = Column(
        Integer, ForeignKey(f"{PREFIX}source_file.source_file_uuid"), nullable=False
    )
    line_number = Column(Integer, nullable=True)
    # Null when the line is unknown or past the end of the snapshot copy.
    content = Column(Text, nullable=True)

    # Line belongs to one source file
    source_file = relationship("SourceFile", back_populates="source_lines")
    # Line can be referenced by many instructions
    instruction_source_lines = relationship(
        "InstructionSourceLine", back_populates="source_line"
    )


class InstructionSourceLine(Base):
    """One frame of an instruction's inline stack.

    frame_index 0 is the line where the code was written; the highest index is
    the line in the kernel the compiler charged it to. The ordinal belongs here
    because one inlined line can have several call sites.
    """

    __tablename__ = f"{PREFIX}instruction_source_line"
    __table_args__ = (UniqueConstraint("instruction_uuid", "frame_index"),)

    instruction_source_line_uuid = Column(Integer, primary_key=True)
    instruction_uuid = Column(
        Integer,
        ForeignKey(f"{PREFIX}instruction_line.instruction_uuid"),
        nullable=False,
    )
    source_line_uuid = Column(
        Integer, ForeignKey(f"{PREFIX}source_line.source_line_uuid"), nullable=False
    )
    frame_index = Column(Integer)

    # Frame belongs to one instruction line
    instruction_line = relationship("InstructionLine", back_populates="source_lines")
    # Frame points at one source line
    source_line = relationship("SourceLine", back_populates="instruction_source_lines")


class PCSampleState(Base):
    __tablename__ = f"{PREFIX}pc_sample_state"

    pc_sample_state_uuid = Column(Integer, primary_key=True)
    instruction_uuid = Column(
        Integer,
        ForeignKey(f"{PREFIX}instruction_line.instruction_uuid"),
        nullable=False,
    )
    total_count = Column(Integer)
    issue_count = Column(Integer, nullable=True)
    stall_count = Column(Integer, nullable=True)
    # TODO: populate from the popcount of record.exec_mask over the wave size
    # (64 on CDNA, 32 or 64 on RDNA), averaged across the samples in this group.
    active_thread_percent = Column(Float, nullable=True)
    # TODO: populate from record.wave_cnt.
    wave_occupancy_percent = Column(Float, nullable=True)
    # TODO: populate from record.dispatch_id, resolved to its dispatch row.
    # Blocked: this table aggregates samples grouped by
    # (code_object_id, code_object_offset, kernel_id), a key that spans
    # dispatches, so no single dispatch describes a row. Meaningful only once
    # dispatch joins that group key, which multiplies row count by
    # dispatches-per-kernel. Until then this stays null and never resolves.
    dispatch_uuid = Column(
        Integer, ForeignKey(f"{PREFIX}dispatch.dispatch_uuid"), nullable=True
    )

    # State belongs to one instruction line
    instruction_line = relationship("InstructionLine", back_populates="pc_sample_state")
    # State has many stall-reason counts
    stall_reasons = relationship(
        "PCSampleStallReason", back_populates="pc_sample_state"
    )
    # State has many issue-state counts
    instruction_samples = relationship(
        "InstructionSample", back_populates="pc_sample_state"
    )


class PCSampleStallReasonLookup(Base):
    __tablename__ = f"{PREFIX}pc_sample_stall_reason_lookup"

    pc_sample_stall_reason_lookup_uuid = Column(Integer, primary_key=True)
    # Deduplicated: one row per distinct stall-reason string.
    text = Column(String, unique=True)

    stall_reasons = relationship(
        "PCSampleStallReason", back_populates="stall_reason_lookup"
    )


class PCSampleStallReason(Base):
    __tablename__ = f"{PREFIX}pc_sample_stall_reason"

    pc_sample_stall_reason_uuid = Column(Integer, primary_key=True)
    pc_sample_state_uuid = Column(
        Integer,
        ForeignKey(f"{PREFIX}pc_sample_state.pc_sample_state_uuid"),
        nullable=False,
    )
    pc_sample_stall_reason_lookup_uuid = Column(
        Integer,
        ForeignKey(
            f"{PREFIX}pc_sample_stall_reason_lookup.pc_sample_stall_reason_lookup_uuid"
        ),
        nullable=False,
    )
    count = Column(Integer)

    pc_sample_state = relationship("PCSampleState", back_populates="stall_reasons")
    stall_reason_lookup = relationship(
        "PCSampleStallReasonLookup", back_populates="stall_reasons"
    )


class InstructionSampleLookup(Base):
    __tablename__ = f"{PREFIX}instruction_sample_lookup"

    instruction_sample_lookup_uuid = Column(Integer, primary_key=True)
    # Deduplicated: one row per distinct issue-state string.
    text = Column(String, unique=True)

    instruction_samples = relationship(
        "InstructionSample", back_populates="instruction_sample_lookup"
    )


class InstructionSample(Base):
    """Per-sample issue state, counted across the samples at one offset.

    Holds the issue state seen each cycle a sample landed, so one offset
    accumulates several rows: NO_INST, BRANCH_TAKEN vs BRANCH_NOT_TAKEN, and
    DUAL_VALU vs VALU all vary sample to sample for one static instruction.
    For that static instruction's own class see
    InstructionLine.instruction_type_uuid.
    """

    __tablename__ = f"{PREFIX}instruction_sample"

    instruction_sample_uuid = Column(Integer, primary_key=True)
    pc_sample_state_uuid = Column(
        Integer,
        ForeignKey(f"{PREFIX}pc_sample_state.pc_sample_state_uuid"),
        nullable=False,
    )
    instruction_sample_lookup_uuid = Column(
        Integer,
        ForeignKey(f"{PREFIX}instruction_sample_lookup.instruction_sample_lookup_uuid"),
        nullable=False,
    )
    count = Column(Integer)

    pc_sample_state = relationship(
        "PCSampleState", back_populates="instruction_samples"
    )
    instruction_sample_lookup = relationship(
        "InstructionSampleLookup", back_populates="instruction_samples"
    )


class KernelMetricValue(Base):
    __tablename__ = f"{PREFIX}kernel_metric_value"
    # One value per (kernel, metric, value_name e.g. min/max/avg).
    __table_args__ = (UniqueConstraint("kernel_uuid", "metric_uuid", "value_name"),)

    value_uuid = Column(Integer, primary_key=True)
    metric_uuid = Column(
        Integer, ForeignKey(f"{PREFIX}metric_definition.metric_uuid"), nullable=False
    )
    kernel_uuid = Column(
        Integer, ForeignKey(f"{PREFIX}kernel.kernel_uuid"), nullable=False
    )
    value_name = Column(String)  # e.g. min, max, avg
    value = Column(Float)  # e.g. 123.45

    # Value can have one metric
    metric = relationship("MetricDefinition", back_populates="kernel_metric_values")
    # Value can have one kernel
    kernel = relationship("Kernel", back_populates="metric_values")


class WorkloadMetricValue(Base):
    __tablename__ = f"{PREFIX}workload_metric_value"
    # One value per (workload, metric, value_name e.g. min/max/avg).
    __table_args__ = (UniqueConstraint("workload_id", "metric_uuid", "value_name"),)

    value_uuid = Column(Integer, primary_key=True)
    metric_uuid = Column(
        Integer, ForeignKey(f"{PREFIX}metric_definition.metric_uuid"), nullable=False
    )
    workload_id = Column(
        Integer, ForeignKey(f"{PREFIX}workload.workload_id"), nullable=False
    )
    value_name = Column(String)  # e.g. min, max, avg
    value = Column(Float)

    # Relationships
    metric = relationship("MetricDefinition", back_populates="workload_metric_values")
    workload = relationship("Workload", back_populates="workload_metric_values")


class WorkloadRooflineData(Base):
    __tablename__ = f"{PREFIX}workload_roofline_data"

    roofline_uuid = Column(Integer, primary_key=True)
    # One roofline data point per workload.
    workload_id = Column(
        Integer,
        ForeignKey(f"{PREFIX}workload.workload_id"),
        nullable=False,
        unique=True,
    )
    total_flops = Column(Float)
    l0_cache_data = Column(Float)
    l1_cache_data = Column(Float)
    l2_cache_data = Column(Float)
    hbm_cache_data = Column(Float)
    lds_cache_data = Column(Float)

    # Relationships
    workload = relationship("Workload", back_populates="workload_roofline_data_points")


class Metadata(Base):
    __tablename__ = f"{PREFIX}metadata"

    id = Column(Integer, primary_key=True)
    compute_version = Column(String)
    git_version = Column(String)
    schema_version = Column(String)


class Database:
    _session: Optional[Session] = None
    _engine: Optional[Engine] = None
    _db_name: Optional[str] = None
    _view_sql_cache: Optional[dict[str, str]] = None
    _type_cache: Optional[dict[tuple[type[Base], str], Base]] = None

    @classmethod
    def init(cls, db_name: str) -> str:
        # StaticPool pins the engine to a single sqlite3 connection so the
        # session and the backup in write() share the same in-memory DB.
        cls._engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            json_serializer=lambda value: json.dumps(
                cls._json_sanitize(value), allow_nan=False
            ),
        )
        Base.metadata.create_all(cls._engine)
        cls._session = sessionmaker(bind=cls._engine)()
        cls._db_name = db_name
        cls._type_cache = {}
        # Compile views eagerly so a broken definition fails at init time.
        cls._view_sql_cache = cls._compile_view_sql()
        console_debug("SQLite database initialized in memory")
        return db_name

    @classmethod
    def get_session(cls) -> Optional[Session]:
        return cls._session

    @classmethod
    def get_or_create_type(cls, orm_class: type[Base], text: str) -> Base:
        """Return a de-duplicated lookup-table row for the text, creating it once.

        Deduplicates DB-wide across workloads. orm_class must be a lookup table
        with a unique text column.
        """
        key = (orm_class, text)
        if key not in cls._type_cache:
            cls._type_cache[key] = orm_class(text=text)
            cls._session.add(cls._type_cache[key])
        return cls._type_cache[key]

    @classmethod
    def commit(cls) -> None:
        """Seal pending session writes. Must be called before any export."""
        if cls._session is None:
            console_error("No active database session")
        try:
            cls._session.commit()
        except Exception as e:
            cls._session.rollback()
            console_error(f"Error committing analysis database: {e}")

    @classmethod
    def write(cls) -> None:
        """Back up the in-memory database to disk at the configured path."""
        if cls._session is None:
            console_error("No active database session")
        try:
            # Writing to disk is slow, so we built the database in memory.
            # Now copy the finished database to disk in one step.
            with closing(cls._engine.raw_connection()) as memory_conn:
                with closing(sqlite3.connect(cls._db_name)) as disk_conn:
                    memory_conn.backup(disk_conn)
            console_debug("Completed writing database")
            console_warning(f"Created file: {cls._db_name}")
        except Exception as e:
            console_error(f"Error writing analysis database: {e}")
        finally:
            cls._session.close()
            cls._session = None

    @classmethod
    def write_csv_dir(cls, csv_dir: Path) -> None:
        """Stream each view's rows directly into a CSV file in csv_dir.

        Uses the raw sqlite3 cursor and csv.writer so the full result set
        is never held in memory at once.
        """
        if cls._session is None:
            console_error("No active database session")
        try:
            csv_dir.mkdir(parents=True, exist_ok=True)
            # session.connection() is a SQLAlchemy Connection; its .connection
            # attribute is the underlying sqlite3.Connection.
            raw_conn = cls._session.connection().connection
            for view_name, sql in cls.get_view_sql().items():
                cursor = raw_conn.execute(sql)
                csv_path = csv_dir / f"{view_name}.csv"
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([column[0] for column in cursor.description])
                    writer.writerows(cursor)
                console_warning(f"Created file: {csv_path}")
        finally:
            cls._session.close()
            cls._session = None

    @classmethod
    def get_stall_reasons_by_workload(cls) -> dict[int, list[str]]:
        """Return the stall reasons each workload's samples actually carry.

        A host_trap workload records none, so it maps to an empty list and its
        exports carry no stall columns.
        """
        stall_reasons_by_workload: dict[int, list[str]] = {}
        for workload_id, reason in cls._session.execute(
            cls._stall_reasons_by_workload_statement()
        ):
            stall_reasons_by_workload.setdefault(workload_id, []).append(reason)
        return {
            workload_id: sorted(reasons)
            for workload_id, reasons in stall_reasons_by_workload.items()
        }

    @classmethod
    def stream_per_kernel_isa_rows(
        cls,
        workload_id: int,
        stall_reasons: list[str],
    ) -> Iterator[tuple[Any, ...]]:
        """Yield one workload's instruction lines, grouped file by file.

        Rows arrive in the order the exporter writes them, through the raw
        sqlite3 cursor so the full result set is never held in memory.
        """
        statement = cls._per_kernel_isa_statement(workload_id, stall_reasons)
        raw_conn = cls._session.connection().connection
        yield from raw_conn.execute(
            str(
                statement.compile(
                    dialect=sqlite.dialect(),
                    compile_kwargs={"literal_binds": True},
                )
            )
        )

    @classmethod
    def create_views(cls) -> None:
        """Materialize CREATE VIEW statements in the in-memory DB."""
        for name, sql in cls.get_view_sql().items():
            cls._session.execute(text(f"CREATE VIEW {PREFIX}{name}_view AS {sql}"))

    @classmethod
    def get_view_sql(cls) -> dict[str, str]:
        """Return {bare_view_name: compiled SELECT SQL} for analysis views.

        Returns a shallow copy of the cache populated in init() so callers
        can't poison it.
        """
        return dict(cls._view_sql_cache)

    @staticmethod
    def _json_sanitize(value: object) -> object:
        """Recursively replace non-finite floats (NaN, Inf) with None for valid JSON."""
        if isinstance(value, dict):
            return {key: Database._json_sanitize(v) for key, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Database._json_sanitize(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    @staticmethod
    def _source_chain_subquery() -> Subquery:
        """Join each instruction's frames back into one source string.

        SQLite gained group_concat's ORDER BY in 3.44, so the frames are
        joined by walking frame_index one step at a time instead.
        """
        frames = (
            select(
                InstructionSourceLine.instruction_uuid.label("instruction_uuid"),
                InstructionSourceLine.frame_index.label("frame_index"),
                (
                    SourceFile.file_path
                    + ":"
                    + func.coalesce(
                        cast(SourceLine.line_number, Text), UNKNOWN_SOURCE_LINE_TOKEN
                    )
                ).label("frame"),
            )
            .select_from(InstructionSourceLine)
            .join(
                SourceLine,
                InstructionSourceLine.source_line_uuid == SourceLine.source_line_uuid,
            )
            .join(
                SourceFile,
                SourceLine.source_file_uuid == SourceFile.source_file_uuid,
            )
        ).subquery("source_frames")

        chain = (
            select(
                frames.c.instruction_uuid,
                frames.c.frame_index,
                frames.c.frame.label("source"),
            )
            .where(frames.c.frame_index == 0)
            .cte("source_chain", recursive=True)
        )
        chain = chain.union_all(
            select(
                frames.c.instruction_uuid,
                frames.c.frame_index,
                chain.c.source + SOURCE_FRAME_SEPARATOR + frames.c.frame,
            ).where(
                (frames.c.instruction_uuid == chain.c.instruction_uuid)
                & (frames.c.frame_index == chain.c.frame_index + 1)
            )
        )

        # Every step of the walk stays in the result, so keep the step that
        # has no next frame.
        return (
            select(chain.c.instruction_uuid, chain.c.source).where(
                ~select(InstructionSourceLine.instruction_source_line_uuid)
                .where(
                    (InstructionSourceLine.instruction_uuid == chain.c.instruction_uuid)
                    & (InstructionSourceLine.frame_index == chain.c.frame_index + 1)
                )
                .exists()
            )
        ).subquery("source_chain")

    @staticmethod
    def _stall_reasons_by_workload_statement() -> Select[Any]:
        """Select the distinct (workload, stall reason) pairs that were stored."""
        return (
            select(
                CodeObjectStore.workload_id,
                PCSampleStallReasonLookup.text,
            )
            .select_from(PCSampleStallReason)
            .join(
                PCSampleStallReasonLookup,
                PCSampleStallReason.pc_sample_stall_reason_lookup_uuid
                == PCSampleStallReasonLookup.pc_sample_stall_reason_lookup_uuid,
            )
            .join(
                PCSampleState,
                PCSampleStallReason.pc_sample_state_uuid
                == PCSampleState.pc_sample_state_uuid,
            )
            .join(
                InstructionLine,
                PCSampleState.instruction_uuid == InstructionLine.instruction_uuid,
            )
            .join(
                KernelSymbol,
                InstructionLine.kernel_symbol_uuid == KernelSymbol.kernel_symbol_uuid,
            )
            .join(
                CodeObjectStore,
                KernelSymbol.code_object_uuid == CodeObjectStore.code_object_uuid,
            )
            .distinct()
        )

    @staticmethod
    def _per_kernel_isa_statement(
        workload_id: int,
        stall_reasons: list[str],
    ) -> Select[Any]:
        """Select one workload's instruction lines with their sample counts.

        The first columns name the file each row belongs in; the rest are the
        row as it is written, in CSV column order. Every disassembled line is
        returned, so an unsampled line arrives with empty counts.
        """
        source_chain_subquery = Database._source_chain_subquery()
        # The CASE is non-null on one row of each group; MAX skips the nulls.
        stall_reason_columns = [
            func.max(
                case((
                    PCSampleStallReasonLookup.text == stall_reason,
                    PCSampleStallReason.count,
                ))
            ).label(stall_reason)
            for stall_reason in stall_reasons
        ]

        return (
            select(
                Workload.name.label("workload_name"),
                Workload.sub_name.label("workload_sub_name"),
                Kernel.kernel_uuid.label("kernel_uuid"),
                CodeObjectStore.code_object_id.label("code_object_id"),
                CodeObjectStore.pid.label("pid"),
                InstructionLine.code_object_offset.label("offset"),
                InstructionLine.instruction,
                PCSampleState.total_count.label("count"),
                PCSampleState.issue_count.label("count_issue"),
                PCSampleState.stall_count.label("count_stall"),
                PCSampleState.wave_occupancy_percent,
                PCSampleState.active_thread_percent,
                *stall_reason_columns,
                source_chain_subquery.c.source.label("source"),
                CodeObjectStore.code_object_id.label("code_object_id_cell"),
                CodeObjectStore.pid.label("pid_cell"),
            )
            .select_from(InstructionLine)
            .join(
                KernelSymbol,
                InstructionLine.kernel_symbol_uuid == KernelSymbol.kernel_symbol_uuid,
            )
            .join(
                CodeObjectStore,
                KernelSymbol.code_object_uuid == CodeObjectStore.code_object_uuid,
            )
            .join(Kernel, KernelSymbol.kernel_uuid == Kernel.kernel_uuid)
            .join(Workload, CodeObjectStore.workload_id == Workload.workload_id)
            # A line the disassembly holds but no sample landed on has no state.
            .outerjoin(
                PCSampleState,
                InstructionLine.instruction_uuid == PCSampleState.instruction_uuid,
            )
            .outerjoin(
                PCSampleStallReason,
                PCSampleState.pc_sample_state_uuid
                == PCSampleStallReason.pc_sample_state_uuid,
            )
            .outerjoin(
                PCSampleStallReasonLookup,
                PCSampleStallReason.pc_sample_stall_reason_lookup_uuid
                == PCSampleStallReasonLookup.pc_sample_stall_reason_lookup_uuid,
            )
            .outerjoin(
                source_chain_subquery,
                InstructionLine.instruction_uuid
                == source_chain_subquery.c.instruction_uuid,
            )
            .where(CodeObjectStore.workload_id == workload_id)
            # One row per line: the stall-reason join multiplies them otherwise.
            .group_by(InstructionLine.instruction_uuid)
            .order_by(
                Kernel.kernel_uuid,
                CodeObjectStore.code_object_id,
                CodeObjectStore.pid,
                InstructionLine.code_object_offset,
            )
        )

    @staticmethod
    def _compile_view_sql() -> dict[str, str]:
        """Build and compile the analysis views to SQLite SQL strings."""
        median_sort_subquery = (
            select(
                Kernel.kernel_uuid,
                (Dispatch.end_timestamp - Dispatch.start_timestamp).label("duration"),
                func
                .row_number()
                .over(
                    partition_by=Kernel.kernel_uuid,
                    order_by=Dispatch.end_timestamp - Dispatch.start_timestamp,
                )
                .label("row_num"),
                func.count().over(partition_by=Kernel.kernel_uuid).label("total_count"),
            )
            .select_from(Dispatch)
            .join(Kernel, Dispatch.kernel_uuid == Kernel.kernel_uuid)
        ).subquery()

        median_calc_subquery = (
            select(
                median_sort_subquery.c.kernel_uuid,
                func.avg(median_sort_subquery.c.duration).label("duration_ns_median"),
            )
            .where(
                # For odd counts: get the middle row
                # For even counts: get the two middle rows and average them
                median_sort_subquery.c.row_num.in_([
                    func.cast((median_sort_subquery.c.total_count + 1) / 2, Integer),
                    func.cast((median_sort_subquery.c.total_count + 2) / 2, Integer),
                ])
            )
            .group_by(median_sort_subquery.c.kernel_uuid)
        ).subquery()

        stall_reason_json_subquery = (
            select(
                PCSampleStallReason.pc_sample_state_uuid,
                func.json_group_object(
                    PCSampleStallReasonLookup.text, PCSampleStallReason.count
                ).label("stall_reason"),
            )
            .select_from(PCSampleStallReason)
            .join(
                PCSampleStallReasonLookup,
                PCSampleStallReason.pc_sample_stall_reason_lookup_uuid
                == PCSampleStallReasonLookup.pc_sample_stall_reason_lookup_uuid,
            )
            .group_by(PCSampleStallReason.pc_sample_state_uuid)
        ).subquery()

        source_chain_subquery = Database._source_chain_subquery()

        definitions: dict[str, Select[Any]] = {
            "kernel": select(
                Kernel.kernel_uuid.label("kernel_uuid"),
                Kernel.workload_id.label("workload_id"),
                Workload.name.label("workload_name"),
                Kernel.kernel_name,
                func.count(Dispatch.dispatch_id).label("dispatch_count"),
                func.sum(Dispatch.end_timestamp - Dispatch.start_timestamp).label(
                    "duration_ns_sum"
                ),
                func.min(Dispatch.end_timestamp - Dispatch.start_timestamp).label(
                    "duration_ns_min"
                ),
                func.max(Dispatch.end_timestamp - Dispatch.start_timestamp).label(
                    "duration_ns_max"
                ),
                median_calc_subquery.c.duration_ns_median,
                func.avg(Dispatch.end_timestamp - Dispatch.start_timestamp).label(
                    "duration_ns_mean"
                ),
            )
            .select_from(Dispatch)
            .join(Kernel, Dispatch.kernel_uuid == Kernel.kernel_uuid)
            .join(Workload, Kernel.workload_id == Workload.workload_id)
            .join(
                median_calc_subquery,
                Kernel.kernel_uuid == median_calc_subquery.c.kernel_uuid,
            )
            .group_by(
                Kernel.kernel_uuid,
                Kernel.workload_id,
                Workload.name,
                Kernel.kernel_name,
            ),
            "kernel_metric": select(
                Workload.workload_id.label("workload_id"),
                Workload.name.label("workload_name"),
                Kernel.kernel_uuid.label("kernel_uuid"),
                Kernel.kernel_name,
                MetricDefinition.metric_uuid.label("metric_uuid"),
                MetricDefinition.name.label("metric_name"),
                MetricDefinition.metric_id,
                MetricDefinition.description,
                MetricDefinition.table_name,
                MetricDefinition.sub_table_name,
                MetricDefinition.unit,
                KernelMetricValue.value_uuid.label("value_uuid"),
                KernelMetricValue.value_name,
                KernelMetricValue.value,
            )
            .select_from(MetricDefinition)
            .join(Workload, MetricDefinition.workload_id == Workload.workload_id)
            .join(
                KernelMetricValue,
                MetricDefinition.metric_uuid == KernelMetricValue.metric_uuid,
            )
            .join(Kernel, KernelMetricValue.kernel_uuid == Kernel.kernel_uuid),
            "workload_metric": select(
                Workload.workload_id.label("workload_id"),
                Workload.name.label("workload_name"),
                MetricDefinition.metric_uuid.label("metric_uuid"),
                MetricDefinition.name.label("metric_name"),
                MetricDefinition.metric_id,
                MetricDefinition.description,
                MetricDefinition.table_name,
                MetricDefinition.sub_table_name,
                MetricDefinition.unit,
                WorkloadMetricValue.value_uuid.label("value_uuid"),
                WorkloadMetricValue.value_name,
                WorkloadMetricValue.value,
            )
            .select_from(MetricDefinition)
            .join(Workload, MetricDefinition.workload_id == Workload.workload_id)
            .join(
                WorkloadMetricValue,
                MetricDefinition.metric_uuid == WorkloadMetricValue.metric_uuid,
            ),
            # One row per sampled instruction line. Identity is
            # (pid, code_object_id, kernel, offset): the same offset in two
            # processes can be different code, so the rows stay separate.
            "pc_sampling_summary": select(
                CodeObjectStore.workload_id.label("workload_id"),
                CodeObjectStore.pid.label("pid"),
                CodeObjectStore.code_object_id.label("code_object_id"),
                Kernel.kernel_uuid.label("kernel_uuid"),
                Kernel.kernel_name,
                InstructionLine.code_object_offset.label("offset"),
                InstructionLine.instruction,
                source_chain_subquery.c.source.label("source"),
                PCSampleState.total_count.label("count"),
                PCSampleState.issue_count.label("count_issue"),
                PCSampleState.stall_count.label("count_stall"),
                stall_reason_json_subquery.c.stall_reason,
            )
            .select_from(PCSampleState)
            .join(
                InstructionLine,
                PCSampleState.instruction_uuid == InstructionLine.instruction_uuid,
            )
            .join(
                KernelSymbol,
                InstructionLine.kernel_symbol_uuid == KernelSymbol.kernel_symbol_uuid,
            )
            .join(
                CodeObjectStore,
                KernelSymbol.code_object_uuid == CodeObjectStore.code_object_uuid,
            )
            .join(Kernel, KernelSymbol.kernel_uuid == Kernel.kernel_uuid)
            # host_trap samples have no stall reasons, so the subquery is empty.
            .outerjoin(
                stall_reason_json_subquery,
                PCSampleState.pc_sample_state_uuid
                == stall_reason_json_subquery.c.pc_sample_state_uuid,
            )
            # An instruction line can have no source at all.
            .outerjoin(
                source_chain_subquery,
                InstructionLine.instruction_uuid
                == source_chain_subquery.c.instruction_uuid,
            ),
            "source_lines": select(
                SourceFile.workload_id.label("workload_id"),
                SourceFile.file_path,
                SourceFile.md5_checksum,
                SourceLine.line_number,
                SourceLine.content,
            )
            .select_from(SourceLine)
            .join(
                SourceFile,
                SourceLine.source_file_uuid == SourceFile.source_file_uuid,
            ),
        }

        dialect = sqlite.dialect()
        return {
            name: str(
                stmt.compile(
                    dialect=dialect,
                    compile_kwargs={"literal_binds": True},
                )
            )
            for name, stmt in definitions.items()
        }
