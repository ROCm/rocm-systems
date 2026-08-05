# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""SQLAlchemy ORM models and SQLite backend for the analysis database.

The schema is documented visually in:
    docs/data/analyze/analysis_data_dump_schema.png
generated from its Mermaid source:
    docs/data/analyze/analysis_data_dump_schema.mmd
When changing the schema, update the .mmd file to match,
then re-export the .png via draw.io.
"""

import csv
import json
import math
import sqlite3
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

from utils.logger import console_debug, console_error, console_warning

PREFIX = "compute_"
SCHEMA_VERSION = "2.0.0"


Base = declarative_base()


class Workload(Base):
    __tablename__ = f"{PREFIX}workload"

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
    # Kernel can have multiple sampled instruction lines
    instruction_lines = relationship("InstructionLine", back_populates="kernel")


class CodeObjectStore(Base):
    __tablename__ = f"{PREFIX}code_object_store"
    # code_object_id is only unique per process, so pid disambiguates it.
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
    # One code object owns many instruction lines
    instruction_lines = relationship(
        "InstructionLine", back_populates="code_object_store"
    )


class InstructionLine(Base):
    __tablename__ = f"{PREFIX}instruction_line"
    # A shared code object samples one offset under several kernels.
    __table_args__ = (
        UniqueConstraint("code_object_uuid", "code_object_offset", "kernel_uuid"),
    )

    instruction_uuid = Column(Integer, primary_key=True)
    code_object_uuid = Column(
        Integer,
        ForeignKey(f"{PREFIX}code_object_store.code_object_uuid"),
        nullable=False,
    )
    # Attributed per-sample to its dispatch's kernel via dispatch correlation.
    kernel_uuid = Column(
        Integer, ForeignKey(f"{PREFIX}kernel.kernel_uuid"), nullable=False
    )
    code_object_offset = Column(Integer)
    comment = Column(Text)
    instruction = Column(Text)

    # Instruction line belongs to one code object
    code_object_store = relationship(
        "CodeObjectStore", back_populates="instruction_lines"
    )
    # Instruction line is attributed to one kernel
    kernel = relationship("Kernel", back_populates="instruction_lines")
    # An instruction line has at most one sampled state
    pc_sample_state = relationship(
        "PCSampleState", back_populates="instruction_line", uselist=False
    )


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

    # State belongs to one instruction line
    instruction_line = relationship("InstructionLine", back_populates="pc_sample_state")
    # State has many stall-reason counts
    stall_reasons = relationship(
        "PCSampleStallReason", back_populates="pc_sample_state"
    )
    # State has many instruction-sample-type counts
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
    # Deduplicated: one row per distinct instruction-type string.
    text = Column(String, unique=True)

    instruction_samples = relationship(
        "InstructionSample", back_populates="instruction_sample_lookup"
    )


class InstructionSample(Base):
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
                with csv_path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([column[0] for column in cursor.description])
                    writer.writerows(cursor)
                console_warning(f"Created file: {csv_path}")
        finally:
            cls._session.close()
            cls._session = None

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
            "pc_sampling": select(
                CodeObjectStore.workload_id.label("workload_id"),
                Kernel.kernel_uuid.label("kernel_uuid"),
                Kernel.kernel_name,
                InstructionLine.code_object_offset.label("offset"),
                InstructionLine.instruction,
                InstructionLine.comment.label("source"),
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
                CodeObjectStore,
                InstructionLine.code_object_uuid == CodeObjectStore.code_object_uuid,
            )
            .join(Kernel, InstructionLine.kernel_uuid == Kernel.kernel_uuid)
            # host_trap samples have no stall reasons, so the subquery is empty.
            .outerjoin(
                stall_reason_json_subquery,
                PCSampleState.pc_sample_state_uuid
                == stall_reason_json_subquery.c.pc_sample_state_uuid,
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
