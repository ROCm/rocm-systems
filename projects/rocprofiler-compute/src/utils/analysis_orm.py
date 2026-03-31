# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    TextClause,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker
from sqlalchemy.sql import Select

from utils.logger import console_debug, console_error

PREFIX = "compute_"
SCHEMA_VERSION = "1.3.0"


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
    l1_cache_data = Column(Float)
    l2_cache_data = Column(Float)
    hbm_cache_data = Column(Float)

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
    l1_cache_data = Column(Float)
    l2_cache_data = Column(Float)
    hbm_cache_data = Column(Float)

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

    @classmethod
    def init(cls, db_name: str) -> str:
        cls._engine = create_engine(f"sqlite:///{db_name}")
        Base.metadata.create_all(cls._engine)
        cls._session = sessionmaker(bind=cls._engine)()
        console_debug(f"SQLite database initialized with name: {db_name}")
        return db_name

    @classmethod
    def get_session(cls) -> Optional[Session]:
        return cls._session

    @classmethod
    def write(cls) -> None:
        if cls._session is None:
            console_error("No active database session")

        try:
            cls._session.commit()
        except Exception as e:
            cls._session.rollback()
            console_error(f"Error writing analysis database: {e}")
        finally:
            cls._session.close()
            cls._session = None


def get_views() -> list[TextClause]:
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

    views: dict[str, Select[Any]] = {
        "kernel_view": select(
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
            Kernel.kernel_uuid, Kernel.workload_id, Workload.name, Kernel.kernel_name
        ),
        "kernel_metric_view": select(
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
        "workload_metric_view": select(
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

    return [
        text(
            f"CREATE VIEW {PREFIX}{view_name} AS "
            f"{stmt.compile(compile_kwargs={'literal_binds': True})}"
        )
        for view_name, stmt in views.items()
    ]
