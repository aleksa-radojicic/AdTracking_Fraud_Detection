from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandera.polars as pa
import polars as pl


@dataclass
class Filepaths:
    train: Path = Path("~/Projects/AdTracking_Fraud_Detection/data/train.parquet")
    test: Path = Path("~/Projects/AdTracking_Fraud_Detection/data/test.csv")
    sample_submission: Path = Path("~/Projects/AdTracking_Fraud_Detection/data/test.csv")
    train_unique: Path = Path("~/Projects/AdTracking_Fraud_Detection/data/train_unique.parquet")


class BaseSchema(pa.DataFrameModel):
    ip: pl.UInt32
    app: pl.UInt16
    device: pl.UInt16
    os: pl.UInt16
    channel: pl.UInt16
    click_time: pl.Datetime
    attributed_time: pl.Datetime = pa.Field(nullable=True)


class TrainSchema(pa.DataFrameModel):
    ip: pl.UInt32
    app: pl.UInt16
    device: pl.UInt16
    os: pl.UInt16
    channel: pl.UInt16
    click_time: pl.Datetime
    attributed_time: pl.Datetime = pa.Field(nullable=True)
    is_attributed: pl.Boolean

    @staticmethod
    def label() -> Literal["is_attributed"]:
        return TrainSchema.is_attributed


class TestSchema(BaseSchema):
    click_id: pl.UInt32
