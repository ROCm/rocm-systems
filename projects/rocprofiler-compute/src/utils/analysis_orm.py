##############################################################################bl
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

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
SCHEMA_VERSION = "1.2.0"


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
    # Metric can have multiple metric values
    metric_values = relationship("MetricValue", back_populates="metric")


class RooflineData(Base):
    __tablename__ = f"{PREFIX}roofline_data"

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
    metric_values = relationship("MetricValue", back_populates="kernel")
    # Kernel can have multiple roofline data points
    roofline_data_points = relationship("RooflineData", back_populates="kernel")
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


class MetricValue(Base):
    __tablename__ = f"{PREFIX}metric_value"

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
    metric = relationship("MetricDefinition", back_populates="metric_values")
    # Value can have one kernel
    kernel = relationship("Kernel", back_populates="metric_values")


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
    # Calculate median by finding middle value(s)
    median_subquery = (
        select(
            Kernel.kernel_name,
            (Dispatch.end_timestamp - Dispatch.start_timestamp).label("duration"),
            func
            .row_number()
            .over(
                partition_by=Kernel.kernel_name,
                order_by=Dispatch.end_timestamp - Dispatch.start_timestamp,
            )
            .label("row_num"),
            func.count().over(partition_by=Kernel.kernel_name).label("total_count"),
        )
        .select_from(Dispatch)
        .join(Kernel, Dispatch.kernel_uuid == Kernel.kernel_uuid)
    )

    median_calc = (
        select(
            median_subquery.c.kernel_name,
            func.avg(median_subquery.c.duration).label("duration_ns_median"),
        )
        .where(
            # For odd counts: get the middle row
            # For even counts: get the two middle rows and average them
            median_subquery.c.row_num.in_([
                func.cast((median_subquery.c.total_count + 1) / 2, Integer),
                func.cast((median_subquery.c.total_count + 2) / 2, Integer),
            ])
        )
        .group_by(median_subquery.c.kernel_name)
    )

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
            median_calc.c.duration_ns_median,
            func.avg(Dispatch.end_timestamp - Dispatch.start_timestamp).label(
                "duration_ns_mean"
            ),
        )
        .select_from(Dispatch)
        .join(Kernel, Dispatch.kernel_uuid == Kernel.kernel_uuid)
        .join(Workload, Kernel.workload_id == Workload.workload_id)
        .join(median_calc.subquery(), Kernel.kernel_name == median_calc.c.kernel_name)
        .group_by(
            Kernel.kernel_uuid, Kernel.workload_id, Workload.name, Kernel.kernel_name
        ),
        "metric_view": select(
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
            MetricValue.value_uuid.label("value_uuid"),
            MetricValue.value_name,
            MetricValue.value,
        )
        .select_from(MetricDefinition)
        .join(Workload, MetricDefinition.workload_id == Workload.workload_id)
        .join(MetricValue, MetricDefinition.metric_uuid == MetricValue.metric_uuid)
        .join(Kernel, MetricValue.kernel_uuid == Kernel.kernel_uuid),
    }

    return [
        text(
            f"CREATE VIEW {PREFIX}{view_name} AS "
            f"{stmt.compile(compile_kwargs={'literal_binds': True})}"
        )
        for view_name, stmt in views.items()
    ]
