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
SCHEMA_VERSION = "1.4.0"


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


class MetricDefinition(Base):
    __tablename__ = f"{PREFIX}metric_definition"

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
    kernel_uuid = Column(
        Integer, ForeignKey(f"{PREFIX}kernel.kernel_uuid"), nullable=False
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
    # Kernel can have multiple pc_sampling values
    pc_sampling_values = relationship("PCsampling", back_populates="kernel")


class PCsampling(Base):
    __tablename__ = f"{PREFIX}pcsampling"

    pc_sampling_uuid = Column(Integer, primary_key=True)
    kernel_uuid = Column(
        Integer, ForeignKey(f"{PREFIX}kernel.kernel_uuid"), nullable=False
    )
    source = Column(String)
    instruction = Column(String)
    count = Column(Integer)
    offset = Column(Integer)
    count_issue = Column(Integer)
    count_stall = Column(Integer)
    stall_reason = Column(JSON)

    # PCsampling can have one kernel
    kernel = relationship("Kernel", back_populates="pc_sampling_values")


class KernelMetricValue(Base):
    __tablename__ = f"{PREFIX}kernel_metric_value"

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
    workload_id = Column(
        Integer, ForeignKey(f"{PREFIX}workload.workload_id"), nullable=False
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

    @classmethod
    def init(cls, db_name: str) -> str:
        # StaticPool pins the engine to a single sqlite3 connection so the
        # session and the backup in write() share the same in-memory DB.
        cls._engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(cls._engine)
        cls._session = sessionmaker(bind=cls._engine)()
        cls._db_name = db_name
        # Compile views eagerly so a broken definition fails at init time.
        cls._view_sql_cache = cls._compile_view_sql()
        console_debug("SQLite database initialized in memory")
        return db_name

    @classmethod
    def get_session(cls) -> Optional[Session]:
        return cls._session

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
