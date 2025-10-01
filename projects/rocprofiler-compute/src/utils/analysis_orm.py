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
SCHEMA_VERSION = "1.0.0"


Base = declarative_base()


class Workload(Base):
    __tablename__ = f"{PREFIX}workload"

    workload_id = Column(Integer, primary_key=True)
    name = Column(String)
    sub_name = Column(String)
    sys_info_extdata = Column(JSON)
    roofline_bench_extdata = Column(JSON)
    profiling_config_extdata = Column(JSON)

    # Workload can have multiple dispatches
    dispatches = relationship("Dispatch", back_populates="workload")
    # Workload can have multiple metrics
    metrics = relationship("Metric", back_populates="workload")
    # Workload can have multiple roofline data points
    roofline_data_points = relationship("RooflineData", back_populates="workload")
    # Workload can have multiple pc_sampling values
    pc_sampling_values = relationship("PCsampling", back_populates="workload")


class Metric(Base):
    __tablename__ = f"{PREFIX}metric"

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
    workload = relationship("Workload", back_populates="metrics")
    # Metric can have multiple values
    values = relationship("Value", back_populates="metric")


class RooflineData(Base):
    __tablename__ = f"{PREFIX}roofline_data"

    roofline_uuid = Column(Integer, primary_key=True)
    workload_id = Column(
        Integer, ForeignKey(f"{PREFIX}workload.workload_id"), nullable=False
    )
    kernel_name = Column(String)
    total_flops = Column(Float)
    l1_cache_data = Column(Float)
    l2_cache_data = Column(Float)
    hbm_cache_data = Column(Float)

    # Roofline data point can have one workload
    workload = relationship("Workload", back_populates="roofline_data_points")


class Dispatch(Base):
    __tablename__ = f"{PREFIX}dispatch"

    dispatch_uuid = Column(Integer, primary_key=True)
    workload_id = Column(
        Integer, ForeignKey(f"{PREFIX}workload.workload_id"), nullable=False
    )
    dispatch_id = Column(Integer)
    kernel_name = Column(String)
    gpu_id = Column(Integer)
    duration = Column(Integer)

    # Dispatch can have one workload
    workload = relationship("Workload", back_populates="dispatches")


class PCsampling(Base):
    __tablename__ = f"{PREFIX}pcsampling"

    pc_sampling_uuid = Column(Integer, primary_key=True)
    workload_id = Column(
        Integer, ForeignKey(f"{PREFIX}workload.workload_id"), nullable=False
    )
    source = Column(String)
    instruction = Column(String)
    count = Column(Integer)
    kernel_name = Column(String)
    offset = Column(Integer)
    count_issue = Column(Integer)
    count_stall = Column(Integer)
    stall_reason = Column(JSON)

    # PCsampling can have one workload
    workload = relationship("Workload", back_populates="pc_sampling_values")


class Value(Base):
    __tablename__ = f"{PREFIX}value"

    value_uuid = Column(Integer, primary_key=True)
    metric_uuid = Column(
        Integer, ForeignKey(f"{PREFIX}metric.metric_uuid"), nullable=False
    )
    value_name = Column(String)  # e.g. min, max, avg
    value = Column(Float)  # e.g. 123.45

    # Value can have one metric
    metric = relationship("Metric", back_populates="values")


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
    views: dict[str, Select[Any]] = {
        "kernel_view": select(
            Dispatch.kernel_name,
            func.count(Dispatch.dispatch_id).label("dispatch_count"),
            func.sum(Dispatch.duration).label("duration_sum"),
            func.avg(Dispatch.duration).label("duration_mean"),
        ).group_by(Dispatch.kernel_name),
        "metric_view": select(
            Metric.workload_id,
            Metric.name,
            Metric.metric_id,
            Metric.description,
            Metric.table_name,
            Metric.sub_table_name,
            Metric.unit,
            Value.value_name,
            Value.value,
        ).join(Value, Metric.metric_uuid == Value.metric_uuid),
    }

    return [
        text(
            f"CREATE VIEW {PREFIX}{view_name} AS "
            f"{stmt.compile(compile_kwargs={'literal_binds': True})}"
        )
        for view_name, stmt in views.items()
    ]
