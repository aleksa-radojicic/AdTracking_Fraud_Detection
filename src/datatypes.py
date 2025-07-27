from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandera.polars as pa
import polars as pl


@dataclass
class Filepaths:
    train: Path = Path("~/Projects/AdTracking_Fraud_Detection/data/train.parquet")
    test: Path = Path("~/Projects/AdTracking_Fraud_Detection/data/test.parquet")
    sample_submission: Path = Path("~/Projects/AdTracking_Fraud_Detection/data/test.csv")
    train_unique: Path = Path("~/Projects/AdTracking_Fraud_Detection/data/train_unique.parquet")


class BaseSchema(pa.DataFrameModel):
    ip: pl.UInt32
    app: pl.UInt32
    device: pl.UInt16
    os: pl.UInt16
    channel: pl.UInt16
    click_time: pl.Datetime('us')


class TrainSchema(BaseSchema):
    attributed_time: pl.Datetime('us') = pa.Field(nullable=True)
    is_attributed: pl.Boolean

    @staticmethod
    def label() -> Literal["is_attributed"]:
        return TrainSchema.is_attributed


class TestSchema(BaseSchema):
    click_id: pl.UInt32


class ExtendedSchema(pa.DataFrameModel):
    click_timestamp: pl.UInt32
    previous_sessions: pl.UInt32 # Can be viewed as an ID of the current session
    total_sessions: pl.UInt32
    current_session_duration_till_now: pl.UInt32
    current_session_duration: pl.UInt32
    total_current_session_duration: pl.UInt32
    avg_previous_sessions_duration: pl.Float64


filepaths = Filepaths()
